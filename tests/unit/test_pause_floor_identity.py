"""The pause floor keeper's defaults write HEAD bb5fa2a's samples, through both engines' callers.

`_shipped` is the stage as commit bb5fa2a shipped it (the 'cluster' target, the 'flat' fill),
frozen here from that commit's modules/pause_floor.py on whole arrays: the frame levels of the
channel-mean, the envelope lag, the quiet weight, the one target under the quiet frames'
median, the skip rules and the gain curve. The stage runs as the repository's config.yaml sets
it (no target or shape patched), once through auto_pure_linear's post-neural stages and once
through cathar's step, and must write exactly those samples, or skip exactly where it skipped.
The CPU replay of 2026-10-12 found the same on real audio: the default wrote the shipped
paused WAV sample for sample on all 13 diagnosis clips, on cathar's Tata excerpts and on the
235 hunt work directories.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

import modules.config as cfg
import modules.pause_floor as pause_floor
import modules.processing as processing
from modules.modes.cathar import CatharMode
from tests.unit.test_pause_floor import RATE, _gated, _on, _voice, _write

FILL_DB, PERCENTILE, MARGIN_DB, MAX_LAG, MIN_QUIET, MIN_DEFICIT, SHARE = 12.0, 15.0, 10.0, 10, 50, 0.5, 0.95


def _levels(audio, frame):
    """bb5fa2a's frame_levels on a whole float32 array: RMS of every whole frame of the channel-mean."""
    mono = audio.mean(axis=1).astype(np.float64)
    usable = (len(mono) // frame) * frame
    return np.sqrt(np.mean(mono[:usable].reshape(-1, frame) ** 2, axis=1))


def _score(a, b, lag, count):
    a_cut = slice(max(0, -lag), count - max(0, lag))
    b_cut = slice(max(0, lag), count - max(0, -lag))
    return float(np.dot(a[a_cut], b[b_cut]))


def _aligned(reference, restored):
    """bb5fa2a's envelope_lag and _aligned."""
    count = min(len(reference), len(restored))
    a, b = np.log10(reference[:count] + 1e-9), np.log10(restored[:count] + 1e-9)
    a, b = a - a.mean(), b - b.mean()
    lags = list(range(-MAX_LAG, MAX_LAG + 1))
    lag = lags[int(np.argmax([_score(a, b, lag, count) for lag in lags]))]
    ref_start, res_start = max(-lag, 0), max(lag, 0)
    count = min(len(reference) - ref_start, len(restored) - res_start)
    return reference[slice(ref_start, ref_start + count)], restored[slice(res_start, res_start + count)], lag


def _weight(reference):
    """bb5fa2a's quiet_weight."""
    quiet = (reference <= np.percentile(reference, PERCENTILE) * 10.0 ** (MARGIN_DB / 20.0)).astype(np.float64)
    return np.convolve(quiet, np.ones(3) / 3, mode="same")


def _gains(reference, restored, weight):
    """bb5fa2a's _nothing_to_fill and fill_gains: None where it skipped."""
    quiet = weight > 0.5
    if int(quiet.sum()) < MIN_QUIET:
        return None
    target = np.percentile(reference[quiet], 50.0) * 10.0 ** (-FILL_DB / 20.0)
    if float(np.mean(20.0 * np.log10((target + 1e-12) / (restored[quiet] + 1e-12)) < MIN_DEFICIT)) >= SHARE:
        return None
    deficit = np.sqrt(np.maximum(target**2 - restored**2, 0.0))
    return weight * np.clip(deficit / (reference + 1e-12), 0.0, 1.0)


def _curve(frame_gain, length, frame):
    """bb5fa2a's gain_curve."""
    gain = np.interp(np.arange(length), (np.arange(len(frame_gain)) + 0.5) * frame, frame_gain)
    ramp = max(int(0.02 * RATE), 1)
    kernel = np.hanning(ramp + 2)[1:-1]
    inner = slice(ramp, ramp + length)
    smoothed = np.convolve(np.pad(gain, ramp, mode="edge"), kernel / kernel.sum(), mode="same")[inner]
    smoothed[np.abs(smoothed) < 1e-9] = 0.0
    return smoothed


def _shipped(reference_wav, restored_wav):
    """The samples bb5fa2a's stage wrote for the pair, or None where it skipped."""
    reference, _rate = sf.read(str(reference_wav), dtype="float32", always_2d=True)
    restored, _rate = sf.read(str(restored_wav), dtype="float32", always_2d=True)
    frame = int(0.02 * RATE)
    ref_env, res_env, lag = _aligned(_levels(reference, frame), _levels(restored, frame))
    weight = _weight(ref_env)
    gains = _gains(ref_env, res_env, weight)
    if gains is None:
        return None
    curve = _curve(np.concatenate((np.zeros(max(lag, 0)), gains)), len(restored), frame)
    usable = min(len(restored), len(curve), len(reference))
    out = restored.astype(np.float64)
    out[:usable] += reference[:usable].astype(np.float64) * curve[:usable, None]
    return out.astype(np.float32)


def _apl(reference, restored, work):
    """auto_pure_linear's post-neural stages with the pause floor on, the polish returning `restored`."""
    with patch("modules.processing._polish_full_audio_step", return_value=restored):
        return processing._post_neural_stages(reference, reference, restored, work, 8.0, {"profile": {}}, {"pause_floor": True})


def _cathar(reference, restored, work):
    """cathar's step: its chain and its polish return `restored`, the pause floor runs as the mode calls it."""
    with (
        patch("modules.processing._resolve_preconditioned_audio", return_value=(reference, {})),
        patch("modules.processing._process_single_track_pipeline") as pipeline,
        patch("modules.cathar.filter_cathar_vhs_pipeline", return_value=restored),
        patch("modules.processing._polish_full_audio_step", return_value=restored),
    ):
        CatharMode().execute(work, reference, work / "in.mp4", work / "out.mp4", 8.0, strategy={})
        return pipeline.call_args[0][5](reference, work)


def _lagged(source):
    """The restored audio three frames late and 10 dB down, its pauses gated: the separator's level change and a lag."""
    late = np.concatenate((np.zeros((3 * int(0.02 * RATE), source.shape[1]), dtype=np.float32), _gated(source)))[: len(source)]
    return (late * 0.316).astype(np.float32)


CASES = {
    "gated": lambda source: _gated(source),
    "lagged_and_quieter": _lagged,
    "shorter": lambda source: _gated(source)[: len(source) - int(0.3 * RATE)],
    "floor_passed": lambda source: source,
}


def _pair(tmp_path, case):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = _voice()
    return _write(tmp_path / "ref.wav", source), _write(tmp_path / "res.wav", CASES[case](source))


def _wrote(produced, restored, expected):
    """Whether the stage wrote exactly `expected`, or returned the restored audio where bb5fa2a skipped."""
    if expected is None:
        return Path(produced) == Path(restored)
    audio, _rate = sf.read(str(produced), dtype="float32", always_2d=True)
    return Path(produced).name.startswith("paused_12db_p15_res") and np.array_equal(audio, expected)


def test_the_new_keys_default_to_todays_fill():
    """The resolved config, its module constants and the stage's copies hold the cluster target and the flat fill."""
    assert (cfg.CONFIG["pause_floor_target"], cfg.CONFIG["pause_floor_shape"]) == ("cluster", "flat")
    assert (cfg.PAUSE_FLOOR_TARGET, cfg.PAUSE_FLOOR_SHAPE) == ("cluster", "flat")
    assert (pause_floor.PAUSE_FLOOR_TARGET, pause_floor.PAUSE_FLOOR_SHAPE) == ("cluster", "flat")
    assert (pause_floor.PAUSE_FLOOR_SHELF_HZ, pause_floor.PAUSE_FLOOR_HF_CUT_DB) == (3000.0, 18.0)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_defaults_write_bb5fa2a_samples_in_both_engines(tmp_path, case):
    """Fill, lag, level change, a short tail and a skip: both callers write what bb5fa2a wrote, to the bit."""
    reference, restored = _pair(tmp_path, case)
    expected = _shipped(reference, restored)
    with _on()[0], _on()[1], _on()[2]:
        produced = [_apl(reference, restored, tmp_path / "apl"), _cathar(reference, restored, tmp_path / "cathar")]
    assert [_wrote(path, restored, expected) for path in produced] == [True, True]


def test_the_cases_cover_both_a_fill_and_a_skip(tmp_path):
    """The frozen stage fills the gated voice and skips the one whose floor the restoration passed."""
    filled, skipped = _pair(tmp_path / "a", "gated"), _pair(tmp_path / "b", "floor_passed")
    assert (_shipped(*filled) is None, _shipped(*skipped) is None) == (False, True)
