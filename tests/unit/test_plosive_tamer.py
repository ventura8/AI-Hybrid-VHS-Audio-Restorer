"""The plosive tamer: it finds the blasts, takes their low-band excess, and leaves everything else to the bit.

Bursts are drawn the way the calibrated fixture draws them -- band-limited under 150 Hz,
fast attack, decaying tail, +6 dB over the programme -- at known positions on a voice.
"""

from unittest.mock import patch

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

from modules import plosive_tamer

RATE = 44100
SECONDS = 8.0


def _time():
    return np.arange(int(RATE * SECONDS)) / RATE


def _phrases():
    """Phrase gating with onsets and offsets that take thirty milliseconds, as a voice's do."""
    t = _time()
    gate = (np.sin(2 * np.pi * 0.35 * t) > -0.3).astype(np.float64)
    ramp = int(0.03 * RATE)
    kernel = np.hanning(ramp) / np.hanning(ramp).sum()
    return np.convolve(gate, kernel, mode="same")


def _voice(fundamental=200.0, seed=1):
    """A voiced signal in phrases over a faint noise floor, its fundamental above the crossover unless asked otherwise."""
    t, rng = _time(), np.random.default_rng(seed)
    phase = 2 * np.pi * np.cumsum(fundamental * (1.0 + 0.05 * np.sin(2 * np.pi * 4.0 * t))) / RATE
    partials = sum((0.2 / k**0.7) * np.sin(k * phase + rng.uniform(0, 6)) for k in range(1, 14))
    return partials * (0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t)) ** 2 * _phrases() + rng.normal(0.0, 0.001, len(t))


def _burst(length, seed, band=(10.0, 150.0)):
    rng = np.random.default_rng(seed)
    b, a = scipy.signal.butter(2, [band[0] / (RATE / 2), band[1] / (RATE / 2)], "band")
    burst = scipy.signal.lfilter(b, a, rng.normal(0.0, 1.0, length))
    burst *= np.exp(-np.arange(length) / (0.3 * length))
    return burst / (np.sqrt(np.mean(burst**2)) + 1e-12)


def _with_bursts(programme, positions, length_ms=50.0, level_db=6.0, band=(10.0, 150.0)):
    """The programme with a burst at each position, at the fixture's level over the programme's RMS."""
    out = programme.astype(np.float64).copy()
    length = int(length_ms * RATE / 1000.0)
    level = float(np.sqrt(np.mean(programme**2))) * 10.0 ** (level_db / 20.0)
    for index, position in enumerate(positions):
        end = position + length
        out[position:end] += level * _burst(length, seed=10 + index, band=band)
    return out


def _write(path, channels):
    sf.write(str(path), np.stack(channels, axis=1).astype(np.float32), RATE, subtype="FLOAT")
    return path


POSITIONS = [int(p * RATE) for p in (0.9, 2.3, 4.1, 6.4)]


@pytest.fixture(name="blasted")
def blasted_fixture(tmp_path):
    voice = _voice()
    both = _with_bursts(voice, POSITIONS)
    return _write(tmp_path / "blasted.wav", [both, both]), voice


def test_injected_plosives_are_found_where_they_are(blasted):
    """Every burst has an event starting within ten milliseconds of it, and there are no others."""
    source, _voice_signal = blasted
    events, _low, _base, _rate = plosive_tamer.detect_events(source)
    assert len(events) == len(POSITIONS)
    for position, (start, _stop) in zip(POSITIONS, events):
        assert abs(start - position) < 0.010 * RATE


def test_a_voice_alone_raises_no_events(tmp_path):
    source = _write(tmp_path / "voice.wav", [_voice(), _voice(seed=3)])
    events, _low, _base, _rate = plosive_tamer.detect_events(source)
    assert events == []


def test_a_kick_pattern_is_left_alone(tmp_path):
    """Low bursts every half second are rhythm, not plosives: the density rule drops them all."""
    positions = [int(p * RATE) for p in np.arange(0.5, 7.5, 0.5)]
    source = _write(tmp_path / "kick.wav", [_with_bursts(_voice(), positions)] * 2)
    events, _low, _base, _rate = plosive_tamer.detect_events(source)
    assert events == []


def test_a_long_thump_is_not_a_plosive(tmp_path):
    """A burst three hundred milliseconds long is handling noise, outside a plosive's length."""
    source = _write(tmp_path / "thump.wav", [_with_bursts(_voice(), [int(3.0 * RATE)], length_ms=300.0)] * 2)
    events, _low, _base, _rate = plosive_tamer.detect_events(source)
    assert events == []


def test_the_repair_removes_the_burst_and_keeps_the_fundamental(tmp_path):
    """Inside the event the error against the clean voice falls by 8 dB; a 110 Hz fundamental keeps its level."""
    voice = _voice(fundamental=110.0)
    position = int(3.0 * RATE)
    blasted = _with_bursts(voice, [position])
    source = _write(tmp_path / "male.wav", [blasted, blasted])
    events, low_db, baseline, _rate = plosive_tamer.detect_events(source)
    assert len(events) == 1
    restored = sf.read(str(plosive_tamer.tame_file(source, tmp_path / "out.wav", events, low_db, baseline)), dtype="float32")[0][:, 0]
    start, stop = events[0]
    before = np.sqrt(np.mean((blasted[start:stop] - voice[start:stop]) ** 2))
    after = np.sqrt(np.mean((restored[start:stop] - voice[start:stop]) ** 2))
    assert 20.0 * np.log10(before / after) > 8.0
    sos = scipy.signal.butter(4, [100.0, 120.0], btype="band", fs=RATE, output="sos")
    fundamental_before = np.sqrt(np.mean(scipy.signal.sosfiltfilt(sos, voice)[start:stop] ** 2))
    fundamental_after = np.sqrt(np.mean(scipy.signal.sosfiltfilt(sos, restored.astype(np.float64))[start:stop] ** 2))
    assert abs(20.0 * np.log10(fundamental_after / fundamental_before)) < 3.0


def test_audio_outside_the_events_is_bit_identical(blasted, tmp_path):
    """Away from the events and their ramps the output is the input, sample for sample."""
    source, _voice_signal = blasted
    events, low_db, baseline, _rate = plosive_tamer.detect_events(source)
    restored = sf.read(str(plosive_tamer.tame_file(source, tmp_path / "out.wav", events, low_db, baseline)), dtype="float32")[0]
    original = sf.read(str(source), dtype="float32")[0]
    margin = int(0.02 * RATE)
    untouched = np.ones(len(original), dtype=bool)
    for start, stop in events:
        low, high = max(0, start - margin), stop + margin
        untouched[low:high] = False
    assert np.array_equal(restored[untouched], original[untouched])
    assert not np.array_equal(restored[~untouched], original[~untouched])


def test_tonal_material_skips_the_stage(blasted, tmp_path):
    """Bass is what a tonal recording's low band is made of, so the stage does not run there."""
    source, _voice_signal = blasted
    with (
        patch.object(plosive_tamer, "APL_ENABLE_PLOSIVE_TAMER", True),
        patch.object(plosive_tamer, "estimate_tonality", return_value=0.001),
        patch("modules.plosive_tamer.log_msg") as log,
    ):
        assert plosive_tamer.apply_when_needed(source, tmp_path / "work") == source
    assert "tonal material" in log.call_args[0][0]


def test_the_stage_tames_a_recording_with_plosives(blasted, tmp_path):
    """End to end through the chain's entry point: a new file in the stage's own directory, the count logged."""
    source, _voice_signal = blasted
    with (
        patch.object(plosive_tamer, "APL_ENABLE_PLOSIVE_TAMER", True),
        patch.object(plosive_tamer, "estimate_tonality", return_value=0.2),
        patch("modules.plosive_tamer.log_msg") as log,
    ):
        produced = plosive_tamer.apply_when_needed(source, tmp_path / "work")
    assert produced == tmp_path / "work" / "plosive_tamer" / "tamed_blasted.wav"
    assert "Tamed 4 plosives" in log.call_args[0][0]


def test_zero_threshold_switches_detection_off(blasted):
    source, _voice_signal = blasted
    events, _low, _base, _rate = plosive_tamer.detect_events(source, excess_db=0.0)
    assert events == []


@pytest.mark.parametrize("case", ["disabled", "unreadable", "none"])
def test_disabled_unreadable_or_clean_yields_the_input(tmp_path, case):
    """The stage never fails a restoration: off, given rubbish, or given a recording without plosives."""
    if case == "unreadable":
        source = tmp_path / "rubbish.wav"
        source.write_text("not audio")
    else:
        source = _write(tmp_path / "clean.wav", [_voice(), _voice(seed=3)])
    enabled = case != "disabled"
    with patch.object(plosive_tamer, "APL_ENABLE_PLOSIVE_TAMER", enabled), patch("modules.plosive_tamer.log_msg") as log:
        assert plosive_tamer.apply_when_needed(source, tmp_path / "work") == source
    if case != "disabled":
        assert "Skipped" in log.call_args[0][0]


def test_a_failure_inside_the_stage_leaves_the_audio_usable(blasted, tmp_path):
    source, _voice_signal = blasted
    with (
        patch.object(plosive_tamer, "APL_ENABLE_PLOSIVE_TAMER", True),
        patch.object(plosive_tamer, "estimate_tonality", return_value=0.2),
        patch.object(plosive_tamer, "detect_events", side_effect=MemoryError("no room")),
        patch("modules.plosive_tamer.log_msg") as log,
    ):
        assert plosive_tamer.apply_when_needed(source, tmp_path / "work") == source
    assert "after failure" in log.call_args[0][0]


def test_block_processing_matches_whole_file_processing(blasted, tmp_path, monkeypatch):
    """The band levels carry their filter state across blocks and the repair pads its blocks, so the size cannot show."""
    source, _voice_signal = blasted
    events, low_db, baseline, _rate = plosive_tamer.detect_events(source)
    whole = sf.read(str(plosive_tamer.tame_file(source, tmp_path / "whole.wav", events, low_db, baseline)), dtype="float32")[0]
    monkeypatch.setattr(plosive_tamer, "BLOCK_SAMPLES", 40000)
    blocked_events, blocked_low, blocked_base, _rate = plosive_tamer.detect_events(source)
    assert blocked_events == events
    assert np.allclose(blocked_low, low_db, atol=1e-6)
    blocked = sf.read(
        str(plosive_tamer.tame_file(source, tmp_path / "blocked.wav", blocked_events, blocked_low, blocked_base)), dtype="float32"
    )[0]
    assert np.allclose(whole, blocked, atol=1e-6)


def test_the_output_is_deterministic(blasted, tmp_path):
    source, _voice_signal = blasted
    events, low_db, baseline, _rate = plosive_tamer.detect_events(source)
    first = sf.read(str(plosive_tamer.tame_file(source, tmp_path / "a.wav", events, low_db, baseline)), dtype="float32")[0]
    second = sf.read(str(plosive_tamer.tame_file(source, tmp_path / "b.wav", events, low_db, baseline)), dtype="float32")[0]
    assert np.array_equal(first, second)


def test_the_stage_is_wired_after_the_hum_canceller(tmp_path):
    from modules import apl_chain

    plan = apl_chain.stage_plan(tmp_path, None, None, physical_repair=True, spectral_denoise=True, hum_cancel=True, plosive_tamer=True)
    names = [name for name, _wanted, _stage in plan]
    assert names.index("plosive_tamer") == names.index("hum_cancel") + 1
    assert names.index("plosive_tamer") < names.index("tonal_cleanup")
