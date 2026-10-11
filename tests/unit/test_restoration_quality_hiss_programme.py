"""Ear v4, reading 1: the hiss left under the programme, its window guard, its validation and its guards (at 75 dB SPL)."""

import dataclasses
from functools import lru_cache

import numpy as np
import pytest
import scipy.signal

from scripts.restoration_quality import hiss_floor, hiss_programme, pause_metrics
from tests.unit.test_restoration_quality_hiss_floor import RATE, anchor_of, capture, context_for

WINDOW = slice(0, 15 * RATE)
MARGIN = "hiss_prog_margin_db"


def _lowpass(mono, hz, order=8):
    return scipy.signal.sosfiltfilt(scipy.signal.butter(order, hz, "low", fs=RATE, output="sos"), mono)


def read(output, source=None, anchor=None):
    """The programme readings of one window: the capture's first 15 s against `output`'s."""
    source = capture()[2] if source is None else source
    src, out = source[WINDOW], np.asarray(output, dtype=np.float64)[WINDOW]
    return hiss_programme.programme_readings(src, out, RATE, context_for(src, out, anchor))


def out_of(readings, name=MARGIN):
    """The output side of one reading."""
    return readings[name][1]


@lru_cache(maxsize=None)
def ladder():
    """The oracle ladder's output margins: identity, the hiss 6 / 12 / 20 / 30 dB down, the programme alone."""
    programme, noise, source = capture()
    outputs = [source] + [programme + noise * 10.0 ** (-gain / 20.0) for gain in (6.0, 12.0, 20.0, 30.0)] + [programme]
    return [out_of(read(output)) for output in outputs]


def _speech_gate(source):
    """1 where the source's DSP VAD calls speech, per sample."""
    frame = int(pause_metrics.FRAME_S * RATE)
    speech = np.repeat(pause_metrics.dsp_vad(source, RATE, frame), frame)
    return np.concatenate([speech, np.zeros(len(source) - len(speech), dtype=bool)])[: len(source)]


def _islands(noise, keep, seed=5):
    """The noise with only `keep` of its 256-point STFT cells left (rescaled): spectral-subtraction islands."""
    _freqs, _times, spectrum = scipy.signal.stft(noise, RATE, nperseg=256)
    cells = np.random.default_rng(seed).random(spectrum.shape) < keep
    _times, out = scipy.signal.istft(np.where(cells, spectrum / np.sqrt(keep), 0.0), RATE, nperseg=256)
    return out[: len(noise)]


def test_identity_reads_the_source_exactly():
    """An output equal to its source reads every margin as the source does, validated and audible here."""
    readings = read(capture()[2])
    names = ("hiss_prog_margin_db", "hiss_prog_excess_db", "hiss_prog_margin_all_db", "hiss_prog_margin_hf_db")
    assert [readings[name][0] for name in names] == [readings[name][1] for name in names]
    assert (out_of(readings) > 0.0, out_of(readings, "hiss_prog_validated")) == (True, 1.0)


def test_identity_reads_every_guard_at_the_identity():
    """Pump, kept highs and removal read 0 on an identity."""
    readings = read(capture()[2])
    guards = [out_of(readings, name) for name in ("hiss_pump_db", "hiss_prog_kept_hf_db", "hiss_prog_removed_db")]
    assert guards == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


@pytest.mark.parametrize("gain_db", [0.1, -0.1])
def test_a_tenth_of_a_db_moves_nothing(gain_db):
    """The programme gain match absorbs a +-0.1 dB gain: the margin moves under 0.01 dB."""
    source = capture()[2]
    assert out_of(read(source * 10.0 ** (gain_db / 20.0))) == pytest.approx(out_of(read(source)), abs=0.01)


def test_the_oracle_ladder_is_monotone_and_a_perfect_output_is_inaudible():
    """Each step of removal reads no more hiss than the one before (0.05 dB slack); the programme alone reads <= 0."""
    margins = ladder()
    assert all(later <= earlier + 0.05 for earlier, later in zip(margins, margins[1:]))
    assert margins[0] > margins[2] + 6.0 and margins[-1] <= 0.0


def test_the_removal_reads_the_oracle_step():
    """The physical removal above 3 kHz reads the oracle's 6 and 12 dB within 0.1 dB."""
    programme, noise, _source = capture()
    for gain in (6.0, 12.0):
        assert out_of(read(programme + noise * 10.0 ** (-gain / 20.0)), "hiss_prog_removed_hf_db") == pytest.approx(gain, abs=0.1)


@lru_cache(maxsize=None)
def _hf_capture():
    """`(programme, hiss, source, anchor)` with tape-like hiss heard above 4 kHz: the capture's hiss under 4 kHz 20 dB down."""
    programme, noise, _source = capture()
    low = _lowpass(noise, 4000.0)
    hf_noise = noise - 0.9 * low
    source = programme + hf_noise
    return programme, hf_noise, source, hiss_floor.source_anchor(source.astype(np.float32), RATE, 4500.0, None)


def test_taking_only_the_low_band_noise_earns_nothing():
    """Where the hiss heard sits above 4 kHz, taking the rest 20 dB further down reads within 0.2 dB of the identity."""
    programme, noise, source, anchor = _hf_capture()
    low = _lowpass(noise, 4000.0)
    left = read(programme + (noise - low) + 0.1 * low, source=source, anchor=anchor)
    assert out_of(left) == pytest.approx(out_of(read(source, source=source, anchor=anchor)), abs=0.2)


def test_hiss_gated_under_the_programme_is_caught_by_the_pump_not_the_margin():
    """Hiss kept where the VAD calls speech and 30 dB down in the pauses: the margin barely moves, the pump reads >= 15 dB."""
    programme, noise, source = capture()
    gated = programme + noise * _speech_gate(source) + noise * 10.0 ** (-30.0 / 20.0)
    readings = read(gated)
    assert out_of(readings) <= out_of(read(source)) + 0.5
    assert out_of(readings, "hiss_pump_db") >= 15.0


def test_islands_trip_the_musical_noise_guard():
    """The hiss broken into islands (15 % of the cells kept) reads island kurtosis >= 0.3 over the source."""
    programme, noise, _source = capture()
    assert out_of(read(programme + _islands(noise, 0.15)), "hiss_island_kurt") >= 0.3


def test_a_low_pass_trips_the_dulling_guard():
    """Removing hiss by a 4 kHz low pass of everything reads the kept programme highs at -1 dB or less."""
    assert out_of(read(_lowpass(capture()[2], 4000.0)), "hiss_prog_kept_hf_db") <= -1.0


def test_a_mismatched_window_reads_nothing():
    """Correlation under 0.5, a programme gain over 6 dB, a dead output or a NaN: every name reads None."""
    source = capture()[2]
    stranger = np.random.default_rng(11).standard_normal(len(source)) * 0.05
    broken = source.copy()
    broken[100] = np.nan
    for output in (stranger, source * 10.0 ** (8.0 / 20.0), np.zeros(len(source)), broken):
        assert all(pair == (None, None) for pair in read(output).values())


def test_an_unvalidated_anchor_abstains_on_the_margins_but_keeps_the_guards():
    """Without a validated anchor the margins, excesses and audible share read None; pump, kept HF and removal still read."""
    programme, noise, _source = capture()
    readings = read(programme + noise * 10.0 ** (-12.0 / 20.0), anchor=dataclasses.replace(anchor_of(), validated=False))
    names = ("hiss_prog_margin_db", "hiss_prog_excess_db", "hiss_prog_excess_all_db", "hiss_prog_audible_frac")
    assert [readings[name] for name in names] == [(None, None)] * 4
    assert None not in (out_of(readings, "hiss_pump_db"), out_of(readings, "hiss_prog_kept_hf_db"))
    assert out_of(readings, "hiss_prog_validated") == 0.0


def test_no_anchor_at_all_abstains_too():
    """A source with no floor (`no_floor`) leaves every window unvalidated."""
    programme, noise, _source = capture()
    empty = hiss_floor.Anchor(None, np.zeros(0), 0.0, hiss_floor.NO_FLOOR, False)
    assert out_of(read(programme + noise * 0.25, anchor=empty)) is None


def _chord(seconds=15.0, seed=3):
    """A held three-note chord with 5.5 Hz vibrato, decaying into a 0.4 s exponential reverb, over a white floor at -84 dBFS.

    The floor stands over the 16-bit dither plus 10 dB, so the bands read, and under the threshold in quiet at 75 dB SPL.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    tone = np.zeros(len(t))
    for f0 in (220.0, 277.2, 329.6):
        phase = 2.0 * np.pi * np.cumsum(f0 * (1.0 + 0.01 * np.sin(2.0 * np.pi * 5.5 * t))) / RATE
        tone += sum(np.sin(k * phase + rng.uniform(0.0, 6.28)) / k**1.5 for k in range(1, 20))
    tone *= np.where((t % 5.0) < 3.5, 1.0, 0.0)
    reverb = np.exp(-np.arange(int(0.4 * RATE)) / (0.4 * RATE / 6.9)) * rng.standard_normal(int(0.4 * RATE)) * 0.02
    wet = _lowpass(tone + scipy.signal.fftconvolve(tone, reverb)[: len(t)], 4500.0)
    return 0.1 * wet + 6e-5 * rng.standard_normal(len(t))


def test_a_held_chord_with_vibrato_and_reverb_is_not_heard_as_hiss():
    """A hiss-free chord on a validated window: its vibrato and reverb tails are not hiss (the measured margin reads <= 0).

    The all-band margin reads +2.0 here: it counts the floor estimate of the chord's own bands, which vibrato and reverb lift.
    """
    chord = _chord()
    readings = read(chord, source=chord)
    assert out_of(readings, "hiss_prog_bands") >= 2.0 and out_of(readings, "hiss_prog_validated") == 1.0
    assert out_of(readings) <= 0.0


def test_the_excess_follows_the_margin_and_a_short_window_has_no_programme():
    """The excess pair is max(0, margin); a 10-sample window reads nothing."""
    readings = read(capture()[2])
    short = np.ones(10)
    assert readings["hiss_prog_excess_db"][1] == max(0.0, readings[MARGIN][1])
    assert hiss_programme.programme_readings(short, short, RATE, context_for(short, short)) == (
        hiss_floor.none_pairs(hiss_programme.PROGRAMME_NAMES)
    )


def test_the_guards_edge_cases():
    """No readable high band reads no island kurtosis; no finite band no summary; a mismatched anchor is not applied."""
    assert hiss_programme.island_kurt((np.zeros(10), np.zeros(10)), RATE, np.ones(1, dtype=bool), None, 0.0) is None
    assert hiss_programme.band_summary(np.array([np.nan]), np.array([2000.0]), 1000.0) is None
    assert hiss_programme.capped(np.ones(3), np.ones(2)).tolist() == [1.0, 1.0, 1.0]
