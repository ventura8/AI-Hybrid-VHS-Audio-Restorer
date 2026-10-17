"""The reference-SNR suppressor (`apl_reference_suppress`): its whole-file noise print, its gain and the stage around it."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from modules import hiss_floor, noiseprint_quantile, post_neural_suppress, spectral_suppress, utils
from modules.blend_weights import FRAME
from tests.unit.test_hiss_floor import RATE, _bursts, _hiss, _write

SETTINGS = {"noise_bias": 1.0, "gain_floor_db": -20.0, "dd_alpha": 0.96}
ALL = {"apply_air": True, "post_neural_suppress": True, "air_gate": True}


@pytest.fixture(autouse=True)
def _quiet_log(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "LOG_FILE", tmp_path / "session_log.txt")
    monkeypatch.delenv("AI_RESTORE_EVENT_LOG", raising=False)


def _band_db(path, lo_hz, hi_hz, frames=None):
    """Mean power in dB of a file's left channel over a band (whole file, or the boolean frame mask of 2048-sample frames)."""
    samples = sf.read(str(path), dtype="float64", always_2d=True)[0][:, 0]
    count = len(samples) // FRAME
    spectrum = np.abs(np.fft.rfft(samples[: count * FRAME].reshape(count, FRAME) * np.hanning(FRAME), axis=1)) ** 2
    freqs = np.fft.rfftfreq(FRAME, 1.0 / RATE)
    rows = spectrum if frames is None else spectrum[frames[:count]]
    return 10.0 * np.log10(np.mean(rows[:, (freqs >= lo_hz) & (freqs < hi_hz)]))


def _gaps(programme):
    """Per 2048-sample frame, whether the programme is silent all through it."""
    count = len(programme) // FRAME
    return np.max(np.abs(programme[: count * FRAME].reshape(count, FRAME)), axis=1) == 0.0


# ----------------------------------------------------------------------------- the noise print


def _interview(seconds):
    """`_bursts` with a 0.8 s pause every 3 s: never 4 s of quiet, like an interview, but 0.37 s windows of hiss alone."""
    t = np.arange(int(seconds * RATE)) / RATE
    return _bursts(seconds) * ((t % 3.0) < 2.2)


def test_the_quantile_print_reads_the_hiss_under_a_programme_that_never_pauses_for_four_seconds(tmp_path):
    seconds = 12.0
    source = _write(tmp_path / "source.wav", _interview(seconds) + _hiss(seconds))
    noise, depth = spectral_suppress.reference_noise(source)
    probe = spectral_suppress.noise_profile(source, 4.0)
    hiss_only = spectral_suppress.reference_noise(_write(tmp_path / "hiss.wav", _hiss(seconds)))[0]
    band = (np.fft.rfftfreq(FRAME, 1.0 / RATE) >= 7000) & (np.fft.rfftfreq(FRAME, 1.0 / RATE) < 12000)
    low = (np.fft.rfftfreq(FRAME, 1.0 / RATE) >= 300) & (np.fft.rfftfreq(FRAME, 1.0 / RATE) < 3000)
    # Within 1.5 dB of the print the hiss alone writes, where the probe's 4 s stretch carries the programme.
    assert abs(10 * np.log10(np.mean(noise[band]) / np.mean(hiss_only[band]))) < 1.5
    assert abs(10 * np.log10(np.mean(noise[low]) / np.mean(hiss_only[low]))) < 2.0
    assert 10 * np.log10(np.mean(probe[low]) / np.mean(noise[low])) > 10.0
    assert depth > spectral_suppress.MIN_DEPTH_DB


def test_the_quantile_print_on_hiss_alone_sits_near_its_mean_power(tmp_path):
    hiss = _write(tmp_path / "hiss.wav", _hiss(10.0))
    noise = spectral_suppress.reference_noise(hiss)[0]
    with sf.SoundFile(str(hiss)) as handle:
        span = handle.read(dtype="float32", always_2d=True)
    mean = spectral_suppress.frame_powers(span)[0].mean(axis=0)
    # The 5th percentile of a 16-frame mean sits 2.3 dB under the mean; the bias puts it back within a dB.
    assert abs(10 * np.log10(np.median(noise[50:-50] / mean[50:-50]))) < 1.0


@pytest.mark.parametrize("seconds", [0.3, 6.0])
def test_too_little_live_audio_learns_no_print(tmp_path, seconds):
    silent = _write(tmp_path / "silent.wav", np.zeros(int(seconds * RATE)))
    assert spectral_suppress.reference_noise(silent) == (None, None)


def test_the_quantile_histogram_reads_powers_at_scale_ten():
    acc = noiseprint_quantile.MovingQuantile(bins=3, smooth=4, scale=10.0)
    acc.add(np.full((40, 3), 1e-3) * np.array([1.0, 10.0, 100.0]), np.ones(40, dtype=bool))
    assert np.allclose(10 * np.log10(acc.quantile(0.05)), [-30.0, -20.0, -10.0], atol=0.11)


def test_the_held_level_never_tracks():
    assert spectral_suppress.HeldLevel().update(123.0) == 1.0


# ----------------------------------------------------------------------------- the gain


def _leaky_pair(tmp_path, seconds=8.0, residual=0.5):
    """A source (programme plus hiss) and a neural output that kept `residual` of the hiss under and between the bursts."""
    programme, hiss = _interview(seconds), _hiss(seconds)
    source = _write(tmp_path / "source.wav", programme + hiss)
    restored = _write(tmp_path / "restored.wav", programme + residual * hiss)
    return source, restored, _write(tmp_path / "programme.wav", programme), _gaps(programme)


def test_the_source_s_snr_takes_the_leaked_hiss_down_and_keeps_the_programme(tmp_path):
    source, restored, programme, gaps = _leaky_pair(tmp_path)
    output = spectral_suppress.suppress_by_reference(source, restored, tmp_path / "out.wav", **SETTINGS)
    assert sf.info(str(output)).frames == sf.info(str(restored)).frames
    # The hiss in the gaps (the source holds only hiss there) comes down by most of the 20 dB floor.
    assert _band_db(output, 3000, 16000, gaps) < _band_db(restored, 3000, 16000, gaps) - 12.0
    # The programme's partials keep their level within half a dB.
    assert abs(_band_db(output, 100, 3000, ~gaps) - _band_db(programme, 100, 3000, ~gaps)) < 0.5


def test_where_the_neural_stage_left_nothing_there_is_nothing_to_lower(tmp_path):
    source, restored, programme, gaps = _leaky_pair(tmp_path, residual=0.0)
    output = spectral_suppress.suppress_by_reference(source, restored, tmp_path / "out.wav", **SETTINGS)
    assert abs(_band_db(output, 100, 3000, ~gaps) - _band_db(programme, 100, 3000, ~gaps)) < 0.5
    assert _band_db(output, 3000, 16000, gaps) < _band_db(source, 3000, 16000, gaps) - 30.0


def test_the_restored_length_is_kept_when_the_reference_is_shorter(tmp_path):
    source = _write(tmp_path / "source.wav", (_interview(6.0) + _hiss(6.0))[: int(5.5 * RATE)])
    restored = _write(tmp_path / "restored.wav", _interview(6.0) + 0.5 * _hiss(6.0))
    output = spectral_suppress.suppress_by_reference(source, restored, tmp_path / "out.wav", **SETTINGS)
    assert sf.info(str(output)).frames == sf.info(str(restored)).frames


def test_mismatched_rates_and_a_silent_reference_hand_back_none(tmp_path):
    source, restored, _programme, _gaps_ = _leaky_pair(tmp_path, seconds=6.0)
    other = tmp_path / "other.wav"
    sf.write(str(other), sf.read(str(restored))[0], 48000, subtype="FLOAT")
    assert spectral_suppress.suppress_by_reference_or_none(source, other, tmp_path / "a.wav", **SETTINGS) is None
    assert "[Reference Suppress] Failed" in (tmp_path / "session_log.txt").read_text(encoding="utf-8")
    silent = _write(tmp_path / "silent.wav", np.zeros(int(6.0 * RATE)))
    assert spectral_suppress.suppress_by_reference_or_none(silent, restored, tmp_path / "b.wav", **SETTINGS) is None
    assert "Too little live audio" in (tmp_path / "session_log.txt").read_text(encoding="utf-8")


def test_a_print_that_learned_the_programme_runs_no_suppressor(tmp_path):
    gapless = _write(tmp_path / "gapless.wav", _bursts(8.0) + _hiss(8.0))
    noise, depth = spectral_suppress.reference_noise(gapless)
    assert noise is not None and depth < spectral_suppress.MIN_DEPTH_DB
    restored = _write(tmp_path / "restored.wav", _bursts(8.0) + 0.5 * _hiss(8.0))
    assert spectral_suppress.suppress_by_reference(gapless, restored, tmp_path / "out.wav", **SETTINGS) is None
    assert "it holds programme" in (tmp_path / "session_log.txt").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- the stage


@pytest.fixture(name="reference_on")
def reference_on_fixture(monkeypatch):
    monkeypatch.setattr(post_neural_suppress, "APL_REFERENCE_SUPPRESS", True)


def test_off_by_default_and_not_reached_without_the_opt_in(tmp_path, reference_on):
    del reference_on
    assert post_neural_suppress.APL_REFERENCE_SUPPRESS is True
    restored = tmp_path / "restored.wav"
    with patch.object(hiss_floor, "read", side_effect=AssertionError("read")):
        assert post_neural_suppress.apply_when_needed(tmp_path / "s.wav", restored, tmp_path, {"apply_air": True}) == (restored, True)


def test_the_shipped_default_is_off():
    from modules import config as cfg

    assert (cfg.CONFIG["apl_reference_suppress"], cfg.APL_REFERENCE_SUPPRESS) == (False, False)


def test_on_it_runs_ungated_after_the_gated_suppressor(tmp_path, reference_on, monkeypatch):
    del reference_on
    monkeypatch.setattr(post_neural_suppress, "APL_POST_NEURAL_SUPPRESS", True)
    shut = {"floor_gap_db": 4.0, "median_delta_db": -40.0, "presence_delta_db": -20.0}
    with (
        patch.object(hiss_floor, "read", return_value=shut),
        patch.object(post_neural_suppress, "suppress", side_effect=AssertionError("gated")),
        patch.object(post_neural_suppress, "suppress_by_reference", return_value=tmp_path / "ref.wav") as referenced,
    ):
        output, air = post_neural_suppress.apply_when_needed(tmp_path / "s.wav", tmp_path / "r.wav", tmp_path, ALL)
    assert (output, air) == (tmp_path / "ref.wav", True)
    referenced.assert_called_once_with(tmp_path / "s.wav", Path(tmp_path / "r.wav"), tmp_path)


def test_with_both_on_and_the_gate_open_the_reference_reads_the_gated_output(tmp_path, reference_on, monkeypatch):
    del reference_on
    monkeypatch.setattr(post_neural_suppress, "APL_POST_NEURAL_SUPPRESS", True)
    monkeypatch.setenv("AI_RESTORE_EVENT_LOG", str(tmp_path / "events"))
    opened = {"floor_gap_db": 2.0, "median_delta_db": -2.0, "presence_delta_db": -2.0}
    with (
        patch.object(hiss_floor, "read", return_value=opened),
        patch.object(post_neural_suppress, "suppress", return_value=tmp_path / "gated.wav"),
        patch.object(post_neural_suppress, "suppress_by_reference", return_value=tmp_path / "ref.wav") as referenced,
    ):
        output, _air = post_neural_suppress.apply_when_needed(tmp_path / "s.wav", tmp_path / "r.wav", tmp_path, ALL)
    assert output == tmp_path / "ref.wav"
    assert referenced.call_args.args[1] == tmp_path / "gated.wav"
    (record,) = (tmp_path / "events").glob("post_neural_suppress__*.json")
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert (payload["suppressed"], payload["referenced"]) == (True, True)


def test_on_real_files_the_stage_writes_and_reuses(tmp_path, reference_on):
    del reference_on
    source, restored, _programme, _gaps_ = _leaky_pair(tmp_path, seconds=6.0)
    output, _air = post_neural_suppress.apply_when_needed(source, restored, tmp_path, ALL)
    assert output.parent.name == post_neural_suppress.STAGE and output.name.startswith("referenced_")
    assert post_neural_suppress.suppress_by_reference(source, restored, tmp_path) == output
    assert "[Reference Suppress] Reusing" in (tmp_path / "session_log.txt").read_text(encoding="utf-8")


def test_a_reference_suppressor_that_cannot_run_hands_back_its_input(tmp_path):
    _source, restored, _programme, _gaps_ = _leaky_pair(tmp_path, seconds=6.0)
    silent = _write(tmp_path / "silent.wav", np.zeros(int(6.0 * RATE)))
    assert post_neural_suppress.suppress_by_reference(silent, restored, tmp_path / "other") == restored
    assert "[Reference Suppress] Skipped" in (tmp_path / "session_log.txt").read_text(encoding="utf-8")
