"""The harmonic hum canceller: it takes the hum, leaves the programme, and skips when there is none.

Every signal is synthetic and seeded. The hum is a series at a fundamental off its nominal
value, with the transport's wow on it, under tape-like hiss and a voice with pauses.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from modules import hum_cancel
from scripts.measure_hum import hum_excess_db

RATE = 44100
SECONDS = 8.0
TRUE_F0 = 50.3


def _time():
    return np.arange(int(RATE * SECONDS)) / RATE


def _hum(harmonics=8, drift_pct=0.2, phase_step=0.4, level=0.05):
    """A mains series with the transport's wow on it."""
    t = _time()
    instantaneous = TRUE_F0 * (1.0 + drift_pct / 100.0 * np.sin(2 * np.pi * 0.13 * t))
    phase = 2 * np.pi * np.cumsum(instantaneous) / RATE
    return sum((level / k**0.7) * np.cos(k * phase + phase_step * k) for k in range(1, harmonics + 1))


def _voice(fundamental=137.0, seed=1):
    """A voiced signal with a moving pitch, as speech has, in phrases with pauses."""
    t, rng = _time(), np.random.default_rng(seed)
    phase = 2 * np.pi * np.cumsum(fundamental * (1.0 + 0.06 * np.sin(2 * np.pi * 5.0 * t))) / RATE
    partials = sum((0.3 / k) * np.sin(k * phase + rng.uniform(0, 6)) for k in range(1, 12))
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t)) ** 2 * (np.sin(2 * np.pi * 0.4 * t) > -0.2)
    return partials * envelope


# An EMI buzz: a mains-rate series running well past the eight harmonics of hum, with two
# harmonics missing so the gate has something to leave alone.
BUZZ_HARMONICS = [k for k in range(1, 31) if k not in (12, 13)]


def _hiss(seed=2, level=0.01):
    return np.random.default_rng(seed).normal(0.0, level, int(RATE * SECONDS))


def _write(path, channels):
    data = np.stack(channels, axis=1).astype(np.float32)
    sf.write(str(path), data, RATE, subtype="FLOAT")
    return path


def _residual_db(before, after, hum):
    """Level of what is left of the hum after cancellation, relative to the hum."""
    left = (after - before) + hum
    return 20.0 * np.log10(np.sqrt(np.mean(left**2)) / np.sqrt(np.mean(hum**2)))


def _series(f0, harmonics, level=0.05, rolloff=0.0):
    """A steady harmonic series at an exact fundamental."""
    t = _time()
    return sum((level / k**rolloff) * np.cos(2 * np.pi * f0 * k * t) for k in harmonics)


def _harmonics_of(gated):
    return [harmonic for harmonic, _line, _floor in gated]


@pytest.fixture(name="humming")
def humming_fixture(tmp_path):
    hum = _hum()
    source = _write(tmp_path / "humming.wav", [hum + _voice() + _hiss(), hum + _voice(seed=3) + _hiss(seed=4)])
    return source, hum


def test_a_drifting_harmonic_hum_under_a_voice_is_cancelled(humming, tmp_path):
    """The harmonic excess falls to what the voice alone reads, and in the pauses the hum is 25 dB down.

    cathar's adaptive dehum on the same file leaves the pauses at -18 dB; during the voice
    both leave a residual within 10 dB of the hum, which is the voice's own energy on the
    lines and is 17 dB under the voice itself.
    """
    source, hum = humming
    mono = sf.read(str(source), dtype="float32")[0].mean(axis=1)
    refined, gated = hum_cancel.plan_harmonics(mono.astype(np.float64), RATE, 50.0)
    restored = sf.read(str(hum_cancel.cancel_mains(source, tmp_path / "out.wav", refined, gated)), dtype="float32")[0]
    floor = hum_excess_db(_voice() + _hiss(), RATE, 50.0)
    assert hum_excess_db(mono, RATE, 50.0) > floor + 10.0
    assert hum_excess_db(restored.mean(axis=1), RATE, 50.0) < floor + 1.0
    original = sf.read(str(source), dtype="float32")[0][:, 0]
    paused = np.abs(_voice()) < 1e-6
    assert _residual_db(original[paused], restored[paused, 0], hum[paused]) < -25.0
    assert _residual_db(original, restored[:, 0], hum) < -9.0


def test_the_series_is_gated_to_the_harmonics_that_are_there(humming):
    """Eight harmonics in, eight out: the voice's partials do not open lines of their own."""
    source, _hum = humming
    mono = sf.read(str(source), dtype="float32")[0].mean(axis=1).astype(np.float64)
    refined, gated = hum_cancel.plan_harmonics(mono, RATE, 50.0)
    assert _harmonics_of(gated) == list(range(1, 9))
    assert abs(refined - TRUE_F0) < 0.1


def test_content_outside_the_harmonics_is_unchanged(humming, tmp_path):
    """Energy above the series moves by nothing worth measuring; the pauses read as before."""
    source, _hum = humming
    original = sf.read(str(source), dtype="float32")[0]
    refined, gated = hum_cancel.plan_harmonics(original.mean(axis=1).astype(np.float64), RATE, 50.0)
    restored = sf.read(str(hum_cancel.cancel_mains(source, tmp_path / "out.wav", refined, gated)), dtype="float32")[0]
    spectrum_before = np.abs(np.fft.rfft(original[:, 0]))
    spectrum_after = np.abs(np.fft.rfft(restored[:, 0]))
    high = np.fft.rfftfreq(len(original), 1.0 / RATE) > 600.0
    assert np.abs(np.sum(spectrum_after[high] ** 2) / np.sum(spectrum_before[high] ** 2) - 1.0) < 0.01


def test_the_fundamental_is_refined():
    """The series' own offset is recovered to a few hundredths of a hertz."""
    mono = _hum(drift_pct=0.0) + _hiss()
    assert abs(hum_cancel.refine_f0(mono, RATE, 50.0, list(range(1, 9))) - TRUE_F0) < 0.03
    near = _series(50.45, range(1, 5)) + _hiss()
    assert abs(hum_cancel.refine_f0(near, RATE, 50.0, [1, 2, 3, 4]) - 50.45) < 0.03


def test_the_refinement_is_clamped_and_skipped_when_it_cannot_read():
    """An offset past the range is held to the edge; a snippet or an empty series keeps the nominal value."""
    far = _series(50.6, range(1, 5)) + _hiss()
    assert hum_cancel.refine_f0(far, RATE, 50.0, [1, 2, 3, 4]) == pytest.approx(50.5)
    mono = _hum(drift_pct=0.0) + _hiss()
    assert hum_cancel.refine_f0(mono[:1000], RATE, 50.0, [1]) == 50.0
    assert hum_cancel.refine_f0(mono, RATE, 50.0, []) == 50.0


def test_bandwidth_grows_with_harmonic_number():
    """Wow scales with the harmonic, so the envelope band does too, up to a cap."""
    assert hum_cancel.bandwidths_for([1, 2, 8, 40], base_hz=1.5) == [1.5, 2.0, 5.0, 5.0]


def test_a_harmonic_at_the_floor_is_left_alone():
    """A harmonic the pre-conditioning notch already flattened is neither gated nor invented."""
    hum = _hum(harmonics=8)
    t = _time()
    without_third = hum - (0.05 / 3**0.7) * np.cos(
        3 * 2 * np.pi * np.cumsum(TRUE_F0 * (1.0 + 0.002 * np.sin(2 * np.pi * 0.13 * t))) / RATE + 1.2
    )
    mono = without_third + _hiss()
    _refined, gated = hum_cancel.plan_harmonics(mono, RATE, 50.0)
    assert 3 not in _harmonics_of(gated)
    floors = np.array([1e-9, 1e-9])
    envelope = np.full((100, 1, 2), 1e-6 + 0j)
    shrunk = hum_cancel.shrink_to_floor(envelope, floors, [1.5, 2.0])
    assert np.all(shrunk == 0.0)


def test_an_extended_series_is_taken_only_where_it_stands_out():
    """A buzz running past the mains series is gated harmonic by harmonic, at the higher bar."""
    mono = _series(TRUE_F0, BUZZ_HARMONICS, level=0.02, rolloff=0.3) + _hiss(level=0.002)
    _refined, gated = hum_cancel.plan_harmonics(mono, RATE, 50.0, count=40)
    harmonics = _harmonics_of(gated)
    assert set(range(1, 9)) <= set(harmonics)
    assert not {12, 13} & set(harmonics)
    assert max(harmonics) > 20


def test_stereo_channels_are_tracked_independently(tmp_path):
    """Hum at another amplitude and phase in the right channel is cancelled there too."""
    left = _hum(level=0.05, phase_step=0.4)
    right = _hum(level=0.03, phase_step=1.1)
    source = _write(tmp_path / "stereo.wav", [left + _hiss(), right + _hiss(seed=5)])
    refined, gated = hum_cancel.plan_harmonics((left + right) / 2 + _hiss(seed=6), RATE, 50.0)
    restored = sf.read(str(hum_cancel.cancel_mains(source, tmp_path / "out.wav", refined, gated)), dtype="float32")[0]
    original = sf.read(str(source), dtype="float32")[0]
    assert _residual_db(original[:, 0], restored[:, 0], left) < -25.0
    assert _residual_db(original[:, 1], restored[:, 1], right) < -25.0


def test_block_processing_matches_whole_file_processing(humming, tmp_path, monkeypatch):
    """The analysis grid is absolute, so the block size cannot change the result beyond rounding."""
    source, _hum = humming
    mono = sf.read(str(source), dtype="float32")[0].mean(axis=1).astype(np.float64)
    refined, gated = hum_cancel.plan_harmonics(mono, RATE, 50.0)
    whole = sf.read(str(hum_cancel.cancel_mains(source, tmp_path / "whole.wav", refined, gated)), dtype="float32")[0]
    monkeypatch.setattr(hum_cancel, "BLOCK_SAMPLES", 50000)
    blocked = sf.read(str(hum_cancel.cancel_mains(source, tmp_path / "blocked.wav", refined, gated)), dtype="float32")[0]
    assert np.allclose(whole, blocked, atol=1e-6)


def test_the_output_is_deterministic(humming, tmp_path):
    source, _hum = humming
    mono = sf.read(str(source), dtype="float32")[0].mean(axis=1).astype(np.float64)
    refined, gated = hum_cancel.plan_harmonics(mono, RATE, 50.0)
    first = sf.read(str(hum_cancel.cancel_mains(source, tmp_path / "a.wav", refined, gated)), dtype="float32")[0]
    second = sf.read(str(hum_cancel.cancel_mains(source, tmp_path / "b.wav", refined, gated)), dtype="float32")[0]
    assert np.array_equal(first, second)


def test_the_stage_cancels_when_the_recording_carries_hum(humming, tmp_path):
    """End to end through the chain's entry point: a new file, in the stage's own directory."""
    source, _hum = humming
    with patch.object(hum_cancel, "APL_ENABLE_HUM_CANCEL", True), patch("modules.hum_cancel.log_msg") as log:
        produced = hum_cancel.apply_when_needed(source, tmp_path / "work")
    assert produced == tmp_path / "work" / "hum_cancel" / "humcancel_humming.wav"
    assert produced.is_file()
    assert "Cancelled 8 harmonics of 50.2" in log.call_args[0][0] or "Cancelled 8 harmonics of 50.3" in log.call_args[0][0]


def test_tonal_material_is_held_to_the_mains_series_proper(humming):
    """On tonal programme the extended series is not looked at: a note is as likely there as a buzz."""
    source, _hum = humming
    with patch.object(hum_cancel, "estimate_tonality", return_value=0.001):
        assert hum_cancel._series_length(source) == hum_cancel.MAINS_HARMONICS
    with patch.object(hum_cancel, "estimate_tonality", return_value=0.2):
        assert hum_cancel._series_length(source) == hum_cancel.APL_HUM_MAX_HARMONICS
    with patch.object(hum_cancel, "estimate_tonality", return_value=None):
        assert hum_cancel._series_length(source) == hum_cancel.APL_HUM_MAX_HARMONICS


def test_a_recording_without_hum_is_left_alone(tmp_path):
    """A voice over hiss comes back untouched, whichever gate stops it.

    The detector is the mode's, calibrated on real tape, and it fires on some hum-free
    material; the harmonic gate reads the quiet frames and is what stops the stage then.
    """
    source = _write(tmp_path / "clean.wav", [_voice() + _hiss(), _voice(seed=3) + _hiss(seed=4)])
    with patch.object(hum_cancel, "APL_ENABLE_HUM_CANCEL", True), patch("modules.hum_cancel.log_msg") as log:
        assert hum_cancel.apply_when_needed(source, tmp_path / "work") == source
    assert "Skipped: no" in log.call_args[0][0]
    with (
        patch.object(hum_cancel, "APL_ENABLE_HUM_CANCEL", True),
        patch.object(hum_cancel, "detect_mains_hz", return_value=0.0),
        patch("modules.hum_cancel.log_msg") as log,
    ):
        assert hum_cancel.apply_when_needed(source, tmp_path / "work") == source
    assert "no mains hum" in log.call_args[0][0]


def test_a_gated_but_empty_series_skips(humming, tmp_path):
    """Hum detected, yet no harmonic stands out as a line of its own: nothing to cancel."""
    source, _hum = humming
    with (
        patch.object(hum_cancel, "APL_ENABLE_HUM_CANCEL", True),
        patch.object(hum_cancel, "plan_harmonics", return_value=(50.3, [])),
        patch("modules.hum_cancel.log_msg") as log,
    ):
        assert hum_cancel.apply_when_needed(source, tmp_path / "work") == source
    assert "no harmonic stands above" in log.call_args[0][0]


@pytest.mark.parametrize("case", ["disabled", "unreadable", "too_short"])
def test_disabled_unreadable_or_too_short_yields_the_input(tmp_path, case):
    """The stage never fails a restoration: switched off, given rubbish, or given a snippet, the input comes back."""
    if case == "too_short":
        source = _write(tmp_path / "short.wav", [_hum()[:4000] + _hiss()[:4000]])
    elif case == "unreadable":
        source = tmp_path / "rubbish.wav"
        source.write_text("not audio")
    else:
        source = _write(tmp_path / "off.wav", [_hum() + _hiss()])
    enabled = case != "disabled"
    with patch.object(hum_cancel, "APL_ENABLE_HUM_CANCEL", enabled), patch("modules.hum_cancel.log_msg"):
        assert hum_cancel.apply_when_needed(source, tmp_path / "work") == source


def test_a_short_recording_with_hum_is_reported_as_too_short(tmp_path):
    """Hum found but under the frames the envelope filter needs: skipped with that reason."""
    source = _write(tmp_path / "brief.wav", [_hum()[: RATE * 1] + _hiss()[: RATE * 1]])
    with (
        patch.object(hum_cancel, "APL_ENABLE_HUM_CANCEL", True),
        patch.object(hum_cancel, "_plan", return_value=((50.3, [(1, 50.3, 1e-9)]), None)),
        patch("modules.hum_cancel.log_msg") as log,
    ):
        assert hum_cancel.apply_when_needed(source, tmp_path / "work") == source
    assert "too short" in log.call_args[0][0]


def test_a_failure_inside_the_stage_leaves_the_audio_usable(humming, tmp_path):
    source, _hum = humming
    with (
        patch.object(hum_cancel, "APL_ENABLE_HUM_CANCEL", True),
        patch.object(hum_cancel, "cancel_mains", side_effect=MemoryError("no room")),
        patch("modules.hum_cancel.log_msg") as log,
    ):
        assert hum_cancel.apply_when_needed(source, tmp_path / "work") == source
    assert "after failure" in log.call_args[0][0]


def test_the_cap_bounds_a_passing_partial(tmp_path):
    """A voiced harmonic parked on a line for a moment cannot lift the estimate past the cap."""
    frames = 400
    envelope = np.full((frames, 1, 1), 0.01 + 0j)
    envelope[180:200] = 0.5
    capped = hum_cancel.cap_envelope(envelope, 43.07)
    assert np.max(np.abs(capped[180:200])) <= 0.01 * 10 ** (hum_cancel.CAP_DB / 20.0) * 1.001
    assert np.allclose(np.abs(capped[:100]), 0.01)


def test_the_stage_is_wired_between_repair_and_the_noise_probe(tmp_path):
    """The chain runs the canceller after physical repair and ahead of tonal cleanup and subtraction."""
    from modules import apl_chain

    plan = apl_chain.stage_plan(tmp_path, None, None, physical_repair=True, spectral_denoise=True, hum_cancel=True)
    wanted = [name for name, wanted, _stage in plan if wanted]
    assert wanted == ["physical_repair", "hum_cancel", "tonal_cleanup", "spectral_denoise"]
    assert isinstance(Path(tmp_path), Path)


def test_the_scanner_report_names_the_preconditioned_harmonics_only_when_the_switch_is_on():
    """A scanner report of mains hum marks the first two harmonics as notched; no report, or the switch off, marks none."""
    with patch("modules.hum_cancel.APL_HUM_SKIP_NOTCHED", True):
        assert hum_cancel.notched_harmonics({"profile": {"notch_hz": 50.0}}) == (1, 2)
        assert hum_cancel.notched_harmonics({"precondition_filters": {"notch_hz": 0.0}}) == ()
        assert hum_cancel.notched_harmonics(None) == ()
    with patch("modules.hum_cancel.APL_HUM_SKIP_NOTCHED", False):
        assert hum_cancel.notched_harmonics({"profile": {"notch_hz": 50.0}}) == ()


def test_the_preconditioned_harmonics_are_left_out_of_the_plan(humming):
    """Skipped harmonics leave both the gated series and the refinement; without a skip they are gated as before."""
    source, _hum = humming
    mono, rate = hum_cancel._scannable_mono(source)
    _refined, gated = hum_cancel.plan_harmonics(mono, rate, 50.0, skip=(1, 2))
    _refined, whole = hum_cancel.plan_harmonics(mono, rate, 50.0)
    assert gated
    assert not {1, 2} & set(_harmonics_of(gated))
    assert {1, 2} <= set(_harmonics_of(whole))


def test_the_plan_reads_the_scanner_report_through_the_stage(humming, tmp_path):
    """The stage hands the scanner's report to the plan, and the plan cancels the harmonics that are left."""
    source, _hum = humming
    with patch("modules.hum_cancel.APL_HUM_SKIP_NOTCHED", True), patch("modules.hum_cancel.log_msg") as log:
        produced = hum_cancel.apply_when_needed(source, tmp_path, strategy={"profile": {"notch_hz": 50.0}})
    assert produced != source and produced.is_file()
    assert "Cancelled" in log.call_args[0][0]
