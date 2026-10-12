"""The whole-file quantile noise print (`modules/noiseprint_quantile.py`): cathar's frames, the mute rule, the streamed
moving mean and its histogram quantile, the written print and every fallback to the window pick."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

import modules.noiseprint_quantile as nq

RATE = 16000
DEFAULT_BLOCK = nq.BLOCK_FRAMES
LOG = "log_msg"
HISS_WAV = "hiss.wav"
UTF8 = "utf-8"


def _hiss(seconds, sigma=0.01, seed=3, rate=RATE):
    return np.random.default_rng(seed).normal(0.0, sigma, int(seconds * rate))


def _write(path, left, right=None, rate=RATE):
    audio = left if right is None else np.stack([left, right], axis=1)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), rate, subtype="FLOAT")
    return path


def _cathar_frames(x):
    """cathar 0.8.0's print frames, built the plain way: symmetric Hann 2048 (N - 1), hop 512, whole frames only."""
    window = np.hanning(nq.FRAME)
    spans = [(start, start + nq.FRAME) for start in range(0, len(x) - nq.FRAME + 1, nq.HOP)]
    return np.abs(np.fft.rfft(np.asarray([x[start:stop] * window for start, stop in spans]), axis=1))


def _moving_means(mags):
    spans = [(first, first + nq.SMOOTH_FRAMES) for first in range(len(mags) - nq.SMOOTH_FRAMES + 1)]
    return np.asarray([mags[first:last].mean(axis=0) for first, last in spans])


def _learned(path, block=DEFAULT_BLOCK):
    with patch.object(nq, "BLOCK_FRAMES", block):
        return nq.learn(path)


def test_the_frame_window_is_cathars_symmetric_hann():
    assert np.allclose(nq.WINDOW, np.hanning(nq.FRAME), atol=1e-12)


def test_the_frames_are_cathars_whole_frames():
    x = _hiss(1.3)
    mags, live = nq.frame_magnitudes(x)
    expected = _cathar_frames(x)
    assert mags.shape == expected.shape
    assert np.allclose(mags, expected, rtol=1e-9)
    assert live.all()


def test_a_block_shorter_than_a_frame_has_no_frame():
    mags, live = nq.frame_magnitudes(np.zeros(nq.FRAME - 1))
    assert mags.shape == (0, nq.BINS)
    assert len(live) == 0


@pytest.mark.parametrize(
    ("block", "live"),
    [
        (np.zeros(nq.FRAME), False),
        (np.full(nq.FRAME, 0.25), False),
        (_hiss(nq.FRAME / RATE, sigma=10.0 ** (-95.0 / 20.0)), False),
        (_hiss(nq.FRAME / RATE, sigma=10.0 ** (-80.0 / 20.0)), True),
    ],
    ids=["zeros", "held DC", "under the digital floor", "analog floor"],
)
def test_a_frame_is_a_mute_by_the_mute_guards_rules(block, live):
    assert bool(nq.frame_magnitudes(block)[1][0]) is live


@pytest.mark.parametrize("seconds", [2.3, 4.0, 6.1])
@pytest.mark.parametrize("block", [5, 31, 32, 33, DEFAULT_BLOCK])
def test_the_streamed_windows_are_the_whole_files(tmp_path, seconds, block):
    """Every SMOOTH_FRAMES-frame window of the file is counted once, whatever the block size (seams included)."""
    x = _hiss(seconds)
    _magnitudes, acc = _learned(_write(tmp_path / HISS_WAV, x), block)
    frames = len(_cathar_frames(x.astype(np.float32).astype(np.float64)))
    assert (acc.frames, acc.windows) == (frames, frames - nq.SMOOTH_FRAMES + 1)


def test_the_streamed_print_does_not_depend_on_the_block_size(tmp_path):
    wav = _write(tmp_path / HISS_WAV, _hiss(3.0))
    small, _acc = _learned(wav, 7)
    large, _acc2 = _learned(wav)
    assert np.allclose(small, large, rtol=0.012)


def test_the_print_is_the_quantile_of_the_moving_means_plus_the_bias(tmp_path):
    """Against the plain computation: the K-frame means' quantile (the first at or over q of them, numpy's 'inverted_cdf'), times
    10^(BIAS_DB/20), within one 0.1 dB histogram step: the print interpolates inside the bin that holds that mean."""
    x = _hiss(4.0) * (1.0 + 0.5 * np.sin(2 * np.pi * 0.7 * np.arange(int(4.0 * RATE)) / RATE))
    magnitudes, _acc = _learned(_write(tmp_path / "wobble.wav", x))
    means = _moving_means(_cathar_frames(x.astype(np.float32).astype(np.float64)))
    expected = np.quantile(means, nq.QUANTILE, axis=0, method="inverted_cdf") * 10.0 ** (nq.BIAS_DB / 20.0)
    assert np.max(np.abs(20.0 * np.log10(magnitudes / expected))) <= nq.HIST_STEP_DB + 1e-6


def test_white_noise_reads_under_its_mean_by_the_estimators_own_spread(tmp_path):
    """Stationary white noise: the 5th percentile of 32-frame means sits about 1.9 dB under the mean, +0.62 dB back."""
    x = _hiss(20.0)
    magnitudes, _acc = _learned(_write(tmp_path / "white.wav", x))
    mean = _cathar_frames(x.astype(np.float32).astype(np.float64)).mean(axis=0)
    offset = float(np.median(20.0 * np.log10(magnitudes[10:-10] / mean[10:-10])))
    assert -1.8 < offset < -0.8


def test_a_digital_mute_is_left_out(tmp_path):
    """3 s of zeros in 10 s of hiss: its frames and every window over them are left out, and the print barely moves."""
    x = _hiss(10.0)
    muted = x.copy()
    first, last = 3 * RATE, 6 * RATE
    muted[first:last] = 0.0
    plain, plain_acc = _learned(_write(tmp_path / "plain.wav", x))
    print_, acc = _learned(_write(tmp_path / "muted.wav", muted))
    assert acc.muted >= (3 * RATE - nq.FRAME) // nq.HOP
    assert acc.windows == plain_acc.windows - acc.muted - (nq.SMOOTH_FRAMES - 1)
    assert abs(float(np.median(20.0 * np.log10(print_ / plain)))) < 0.3


def test_programme_in_most_frames_stays_out_of_the_print(tmp_path):
    """A tone in four seconds of every seven (no pause shorter than the 1.02 s mean at 16 kHz): its bin's print stays at the hiss."""
    t = np.arange(14 * RATE) / RATE
    tone = 0.3 * np.sin(2 * np.pi * 1000.0 * t) * ((t % 3.5) < 2.0)
    magnitudes, _acc = _learned(_write(tmp_path / "tone.wav", _hiss(14.0) + tone))
    hiss_only, _acc2 = _learned(_write(tmp_path / HISS_WAV, _hiss(14.0)))
    k = int(round(1000.0 * nq.FRAME / RATE))
    assert 20.0 * np.log10(magnitudes[k] / hiss_only[k]) < 6.0


def test_the_print_follows_the_left_channel_as_cathar_does(tmp_path):
    left, right = _hiss(4.0, sigma=0.001), _hiss(4.0, sigma=0.1, seed=9)
    stereo, _acc = _learned(_write(tmp_path / "stereo.wav", left, right))
    mono, _acc2 = _learned(_write(tmp_path / "left.wav", left))
    assert np.allclose(stereo, mono, rtol=1e-9)


def test_too_few_live_windows_leave_the_print_unread(tmp_path):
    magnitudes, acc = _learned(_write(tmp_path / "mute.wav", np.zeros(3 * RATE)))
    assert magnitudes is None
    assert acc.windows == 0


def test_the_histogram_quantile_interpolates_within_a_bin():
    """Twenty windows at one level and twenty 10 dB up: the 5th percentile lands in the low level's bin."""
    acc = nq.MovingQuantile()
    low = np.full((20, nq.BINS), 10.0 ** (-40.0 / 20.0))
    acc._count(np.vstack((low, low * 10.0**0.5)))
    level = 20.0 * np.log10(acc.quantile(0.05))
    assert np.all(np.abs(level + 40.0) <= nq.HIST_STEP_DB)


def test_levels_outside_the_histogram_are_clipped_to_its_edges():
    acc = nq.MovingQuantile()
    acc._count(np.vstack((np.zeros((30, nq.BINS)), np.full((30, nq.BINS), 1e9))))
    assert np.allclose(acc.quantile(0.05), 10.0 ** ((nq.HIST_LO_DB + 0.05 * 60 / 30 * nq.HIST_STEP_DB) / 20.0))
    assert np.allclose(acc.quantile(0.99), 10.0 ** (nq.HIST_HI_DB / 20.0), rtol=1e-3)


def test_the_print_name_carries_the_estimator():
    assert nq.print_name(Path("x/clip_audio.wav")) == "noise_clip_audio_quantile_k32_q0.05.np.json"


def test_the_written_print_is_cathars_json_list(tmp_path):
    wav = _write(tmp_path / HISS_WAV, _hiss(3.0))
    with patch.object(nq, LOG) as log:
        path = nq.quantile_print(wav, tmp_path)
    values = json.loads(path.read_text(encoding=UTF8))
    assert path == tmp_path / "noise_hiss_quantile_k32_q0.05.np.json"
    assert len(values) == nq.BINS
    assert all(isinstance(v, float) for v in values)
    assert log.call_args.args[0].startswith(f"{nq.LOG_PREFIX} per-bin 5% quantile of the 1.02 s moving mean over 59 windows of 90 frames")


def test_the_log_counts_the_mute_frames(tmp_path):
    x = _hiss(4.0)
    end = 2 * RATE
    x[RATE:end] = 0.0
    with patch.object(nq, LOG) as log:
        nq.quantile_print(_write(tmp_path / "muted.wav", x), tmp_path)
    assert "digital-mute frames left out" in log.call_args.args[0]
    assert "(0 digital-mute" not in log.call_args.args[0]


def test_an_existing_print_is_reused(tmp_path):
    wav = _write(tmp_path / HISS_WAV, _hiss(3.0))
    kept = tmp_path / nq.print_name(wav)
    kept.write_text(json.dumps([0.5] * nq.BINS), encoding=UTF8)
    with patch.object(nq, "learn", side_effect=AssertionError("learned again")):
        assert nq.quantile_print(wav, tmp_path) == kept


@pytest.mark.parametrize("content", ["not json", json.dumps([0.5] * 10), json.dumps({"a": 1})])
def test_a_broken_existing_print_is_learned_again(tmp_path, content):
    wav = _write(tmp_path / HISS_WAV, _hiss(3.0))
    (tmp_path / nq.print_name(wav)).write_text(content, encoding=UTF8)
    with patch.object(nq, LOG):
        path = nq.quantile_print(wav, tmp_path)
    assert len(json.loads(path.read_text(encoding=UTF8))) == nq.BINS


def test_a_mute_file_falls_back_to_the_window_pick_with_a_warning(tmp_path):
    log = MagicMock()
    with patch.object(nq, LOG, log):
        assert nq.quantile_print(_write(tmp_path / "mute.wav", np.zeros(3 * RATE)), tmp_path) is None
    assert log.call_args.args[0] == f"{nq.LOG_PREFIX} 0 live windows, fewer than 20; the window pick learns the print"
    assert log.call_args.kwargs == {"level": "WARNING"}
    assert not (tmp_path / "noise_mute_quantile_k32_q0.05.np.json").exists()


def test_an_unreadable_file_falls_back_to_the_window_pick_with_a_warning(tmp_path):
    log = MagicMock()
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with patch.object(nq, LOG, log):
        assert nq.quantile_print(bad, tmp_path) is None
    assert log.call_args.args[0].startswith(f"{nq.LOG_PREFIX} could not be learned (")
    assert log.call_args.kwargs == {"level": "WARNING"}


def test_a_print_that_cannot_be_written_falls_back_to_the_window_pick(tmp_path):
    wav = _write(tmp_path / HISS_WAV, _hiss(3.0))
    log = MagicMock()
    with patch.object(nq, LOG, log), patch.object(nq, "write_print", side_effect=OSError("disk full")):
        assert nq.quantile_print(wav, tmp_path) is None
    assert log.call_args.args[0] == f"{nq.LOG_PREFIX} could not be learned (disk full); the window pick learns the print"
