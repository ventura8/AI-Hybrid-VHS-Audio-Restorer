"""The sibilant guard: it finds the fricatives on the reference, puts their high band back, and leaves everything else to the bit.

The fixture is a vowel in bursts with an 's' in each gap; the restored copy is the 's'
with its 1-4 kHz body gone and its noise 10 dB down, which is what the neural stage did.
"""

import functools
from unittest.mock import patch

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

from modules import sibilant_guard

RATE = 44100
SECONDS = 6.0
VOWEL_S, GAP_S, SIB_S, SIB_OFFSET_S = 0.3, 0.3, 0.12, 0.09
VOWELS = [(int(i * (VOWEL_S + GAP_S) * RATE), int((i * (VOWEL_S + GAP_S) + VOWEL_S) * RATE)) for i in range(10)]
SIBILANTS = [(stop + int(SIB_OFFSET_S * RATE), stop + int((SIB_OFFSET_S + SIB_S) * RATE)) for _start, stop in VOWELS]


def _gate(spans, length, ramp_ms=10.0):
    gate = np.zeros(length)
    for start, stop in spans:
        gate[start:stop] = 1.0
    ramp = int(ramp_ms * RATE / 1000.0)
    kernel = np.hanning(ramp) / np.hanning(ramp).sum()
    return np.convolve(gate, kernel, mode="same")


def _band_noise(band, seed, length):
    rng = np.random.default_rng(seed)
    sos = scipy.signal.butter(4, band, btype="band", fs=RATE, output="sos")
    return scipy.signal.sosfilt(sos, rng.normal(0.0, 1.0, length))


@functools.lru_cache(maxsize=None)
def _components():
    """Vowel, fricative body, fricative noise and floor, each on its own so the restored copy can thin the 's'. Cached, read-only."""
    length = int(RATE * SECONDS)
    t = np.arange(length) / RATE
    vowel = sum(np.sin(2 * np.pi * 160.0 * k * t) / k for k in range(1, 31))
    vowel *= 0.1 / np.sqrt(np.mean(vowel**2)) * _gate(VOWELS, length)
    sib_gate = _gate(SIBILANTS, length)
    body = 0.03 * _band_noise((1000.0, 4000.0), 2, length) * sib_gate
    noise = 0.05 * _band_noise((4000.0, 10000.0), 3, length) * sib_gate
    floor = np.random.default_rng(4).normal(0.0, 1e-4, length)
    return vowel, body, noise, floor


def _write(path, mono):
    sf.write(str(path), np.stack([mono, mono], axis=1).astype(np.float32), RATE, subtype="FLOAT")
    return path


@pytest.fixture(name="pair")
def pair_fixture(tmp_path):
    vowel, body, noise, floor = _components()
    reference = _write(tmp_path / "reference.wav", vowel + body + noise + floor)
    restored = _write(tmp_path / "restored.wav", vowel + noise * 10.0 ** (-10.0 / 20.0) + floor)
    return reference, restored


def _overlaps(span, spans):
    return any(span[0] < stop and start < span[1] for start, stop in spans)


def _high_energy_db(samples, start, stop):
    sos = scipy.signal.butter(4, 4000.0, btype="high", fs=RATE, output="sos")
    high = scipy.signal.sosfiltfilt(sos, samples.astype(np.float64))
    return 10.0 * np.log10(np.mean(high[start:stop] ** 2) + 1e-12)


def test_the_fricatives_are_found_and_the_vowels_are_not(pair):
    reference, _restored = pair
    events, rate = sibilant_guard.detect_events(reference)
    assert rate == RATE
    assert all(_overlaps(burst, events) for burst in SIBILANTS)
    assert not any(_overlaps(event, VOWELS) for event in events)


def test_frames_merge_across_a_short_dip_and_empty_readings_yield_nothing():
    """Two runs of hops closer than the merge gap are one fricative; no hops at all is no fricative."""
    loud = np.full(300, -80.0)
    loud[200:260] = -30.0
    loud[230:232] = -80.0
    events = sibilant_guard.detect_frames(loud, np.full(300, 0.8), np.full(300, 0.3), RATE)
    assert events == [(200 * sibilant_guard.HOP, 260 * sibilant_guard.HOP)]
    assert sibilant_guard.detect_frames(np.zeros(0), np.zeros(0), np.zeros(0), RATE) == []


def _guarded(pair, tmp_path):
    """The pair through the guard at full strength: (events, guarded samples, original left channel, thinned samples)."""
    reference, restored = pair
    events, _rate = sibilant_guard.detect_events(reference)
    produced = sibilant_guard.guard_file(reference, restored, tmp_path / "out.wav", events, 1.0, 4000)
    out = sf.read(str(produced), dtype="float32")[0]
    original = sf.read(str(reference), dtype="float32")[0][:, 0]
    thinned = sf.read(str(restored), dtype="float32")[0]
    return events, out, original, thinned


def test_the_guard_restores_the_high_band_inside_the_events(pair, tmp_path):
    events, out, original, thinned = _guarded(pair, tmp_path)
    margin = int(sibilant_guard.RAMP_MS * RATE / 1000.0)
    for start, stop in events:
        inner_start, inner_stop = start + margin, stop - margin
        assert abs(_high_energy_db(out[:, 0], inner_start, inner_stop) - _high_energy_db(original, inner_start, inner_stop)) < 1.0
        assert _high_energy_db(thinned[:, 0], inner_start, inner_stop) < _high_energy_db(original, inner_start, inner_stop) - 9.0


def test_the_guard_changes_nothing_outside_the_events_and_their_ramps(pair, tmp_path):
    events, out, _original, thinned = _guarded(pair, tmp_path)
    margin = int(sibilant_guard.RAMP_MS * RATE / 1000.0)
    untouched = np.ones(len(out), dtype=bool)
    for start, stop in events:
        low, high = max(0, start - margin), stop + margin
        untouched[low:high] = False
    assert np.array_equal(out[untouched], thinned[untouched])
    assert not np.array_equal(out[~untouched], thinned[~untouched])


def test_mismatched_channels_are_refused(pair, tmp_path):
    reference, restored = pair
    mono = tmp_path / "mono.wav"
    sf.write(str(mono), sf.read(str(restored), dtype="float32")[0][:, 0], RATE, subtype="FLOAT")
    with pytest.raises(ValueError):
        sibilant_guard.guard_file(reference, mono, tmp_path / "out.wav", [(0, 4096)], 1.0, 4000)


def test_switched_off_returns_the_restored_audio_and_writes_nothing(pair, tmp_path):
    reference, restored = pair
    with patch("modules.sibilant_guard.APL_ENABLE_SIBILANT_GUARD", False), patch("modules.sibilant_guard.log_msg") as log:
        assert sibilant_guard.apply_when_needed(reference, restored, tmp_path / "work") == restored
    assert not (tmp_path / "work").exists()
    log.assert_not_called()


def test_a_vowel_alone_has_nothing_to_guard(tmp_path):
    vowel, _body, _noise, floor = _components()
    reference = _write(tmp_path / "vowel.wav", vowel + floor)
    restored = _write(tmp_path / "vowel_restored.wav", vowel + floor)
    with (
        patch("modules.sibilant_guard.APL_ENABLE_SIBILANT_GUARD", True),
        patch("modules.sibilant_guard.log_msg") as log,
    ):
        assert sibilant_guard.apply_when_needed(reference, restored, tmp_path / "work") == restored
    assert "no sibilants" in log.call_args[0][0]


def test_the_stage_writes_a_guarded_file_once_and_reuses_it(pair, tmp_path):
    reference, restored = pair
    with (
        patch("modules.sibilant_guard.APL_ENABLE_SIBILANT_GUARD", True),
        patch("modules.sibilant_guard.log_msg") as log,
    ):
        produced = sibilant_guard.apply_when_needed(reference, restored, tmp_path / "work")
        assert produced == tmp_path / "work" / "sibilant_guard" / "guarded_restored.wav"
        assert "Guarded 10 sibilants" in log.call_args[0][0]
        first_mtime = produced.stat().st_mtime_ns
        assert sibilant_guard.apply_when_needed(reference, restored, tmp_path / "work") == produced
    assert produced.stat().st_mtime_ns == first_mtime


def test_block_processing_matches_whole_file_processing(pair, tmp_path, monkeypatch):
    """The hop readings carry their filter state across blocks and the blend pads its blocks, so the size cannot show."""
    reference, restored = pair
    events, _rate = sibilant_guard.detect_events(reference)
    whole = sf.read(str(sibilant_guard.guard_file(reference, restored, tmp_path / "whole.wav", events, 0.5, 4000)), dtype="float32")[0]
    monkeypatch.setattr(sibilant_guard, "BLOCK_SAMPLES", 8192)
    small_events, _rate = sibilant_guard.detect_events(reference)
    assert small_events == events
    blocked = sf.read(str(sibilant_guard.guard_file(reference, restored, tmp_path / "blocked.wav", events, 0.5, 4000)), dtype="float32")[0]
    assert np.max(np.abs(whole - blocked)) < 1e-6


def test_a_failure_inside_the_stage_leaves_the_audio_usable(pair, tmp_path):
    reference, restored = pair
    with (
        patch("modules.sibilant_guard.APL_ENABLE_SIBILANT_GUARD", True),
        patch("modules.sibilant_guard.guard_file", side_effect=RuntimeError("no room")),
        patch("modules.sibilant_guard.log_msg") as log,
    ):
        assert sibilant_guard.apply_when_needed(reference, restored, tmp_path / "work") == restored
    assert "after failure" in log.call_args[0][0]
