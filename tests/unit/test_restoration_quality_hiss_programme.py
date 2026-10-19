"""Ear v4, reading 1: the hiss left under the programme, its window guard, its validation and its guards (at 75 dB SPL)."""

import dataclasses
import types
from functools import lru_cache

import numpy as np
import pytest
import scipy.signal

from scripts import same_percept
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
    assert margins[0] > margins[2] + 6.0
    assert margins[-1] <= 0.0


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


def _pause_gate(source):
    """Per sample, the source's true-pause runs (R4's VAD runs, 20 ms frames) of the window: where a pause gate acts."""
    frame = int(pause_metrics.FRAME_S * RATE)
    _mask, runs = pause_metrics.true_pause_mask(source[WINDOW], RATE)
    gate = np.zeros(len(source), dtype=bool)
    gate[: len(runs) * frame] = np.repeat(runs, frame)
    return gate


def test_a_pause_gate_reads_its_depth_on_the_pump():
    """The source with its true pauses gated 10 / 20 dB (an expander: hiss kept under speech, the pauses emptied) reads
    the pump about the gate's depth; the mean over the pause frames read Tele7abc's 20-50 dB gates at +8.2..+11.5."""
    source = capture()[2]
    gate = _pause_gate(source)
    pumps = [out_of(read(source * np.where(gate, 10.0 ** (-depth / 20.0), 1.0)), "hiss_pump_db") for depth in (10.0, 20.0)]
    assert pumps == pytest.approx([10.0, 20.0], abs=0.5)


def test_pauses_the_source_holds_as_digital_silence_carry_no_pump():
    """A source whose pauses are digital silence (a muted Hi-Fi break) has no hiss there to gate: the pump reads None,
    whatever the output puts in them."""
    _programme, noise, source = capture()
    gate = _pause_gate(source)
    muted = np.where(gate, 0.0, source)
    assert out_of(read(muted, source=muted), "hiss_pump_db") is None
    assert out_of(read(muted + noise * gate * 0.25, source=muted), "hiss_pump_db") is None


def test_the_paused_ratio_takes_medians_and_skips_a_silent_source_band():
    """Per band: output median over source median, a straddling frame left out; an output emptied to digital silence reads
    far under the dither, not at it; a source band under the dither plus 10 dB reads NaN."""
    dither = hiss_floor.dither_floor_per_bin()
    src = np.array([[1e-6, 1e-6, dither], [1e-6, 1e-6, dither], [5.0, 1e-6, dither]])
    out = np.array([[1e-7, 0.0, dither], [1e-7, 0.0, dither], [9.0, 0.0, dither]])
    bands = types.SimpleNamespace(nbins=np.ones(3), band_power=lambda power: power)
    ratio = hiss_programme.paused_ratio((src, out), np.ones(3, dtype=bool), bands)
    assert ratio[0] == pytest.approx(0.1)
    assert ratio[1] < 1e-12 < dither / 1e-6
    assert np.isnan(ratio[2])


def test_islands_trip_the_musical_noise_guard():
    """The hiss broken into islands (15 % of the cells kept) reads island kurtosis >= 0.3 over the source."""
    programme, noise, _source = capture()
    assert out_of(read(programme + _islands(noise, 0.15)), "hiss_island_kurt") >= 0.3


def test_an_inaudible_residual_reads_no_islands_rather_than_none():
    """Where the output's margin sits under -10 dB the island guard reads 0.0: no island can be heard, and a None would
    trip the guard as a lost reading against the render that removed the hiss best."""
    programme, noise, _source = capture()
    margin = out_of(read(programme + 0.001 * noise), "hiss_prog_margin_db")
    assert margin is not None and margin < hiss_programme.ISLAND_MIN_MARGIN_DB
    assert out_of(read(programme + 0.001 * noise), "hiss_island_kurt") == 0.0


def test_a_low_pass_trips_the_dulling_guard():
    """Removing hiss by a 4 kHz low pass of everything reads the kept programme highs at -1 dB or less."""
    assert out_of(read(_lowpass(capture()[2], 4000.0)), "hiss_prog_kept_hf_db") <= -1.0


def test_a_mismatched_window_reads_nothing():
    """An unrelated programme envelope, a programme gain over 24 dB, a dead output or a NaN: every name reads None."""
    source = capture()[2]
    stranger = np.random.default_rng(11).standard_normal(len(source)) * 0.05
    broken = source.copy()
    broken[100] = np.nan
    for output in (stranger, source * 10.0 ** (30.0 / 20.0), np.zeros(len(source)), broken):
        assert all(pair == (None, None) for pair in read(output).values())


def test_a_polarity_inverted_late_programme_still_reads():
    """The guard reads power envelopes: the source inverted, 2 ms late and 8 dB down reads like the source itself."""
    source = capture()[2]
    late = -np.concatenate([np.zeros(88), source[:-88]]) * 10.0 ** (-8.0 / 20.0)
    assert out_of(read(late), "hiss_prog_validated") == 1.0
    assert out_of(read(late)) == pytest.approx(out_of(read(source)), abs=1.0)


def test_a_fresh_realisation_of_the_hiss_reads_like_the_hiss():
    """The programme with another draw of the same noise reads the identity's margin within 0.5 dB (no own-cell selection)."""
    programme, noise, source = capture()
    fresh = np.random.default_rng(23).permutation(noise.reshape(-1, RATE // 10)).reshape(-1)
    assert out_of(read(programme + fresh)) == pytest.approx(out_of(read(source)), abs=0.5)


def test_hiss_left_only_in_the_gaps_between_words_is_read_there():
    """The hiss kept (12 dB down) only where the VAD calls no speech reads more hiss than the hiss gone everywhere."""
    programme, noise, source = capture()
    gaps = programme + noise * 10.0 ** (-12.0 / 20.0) * (~_speech_gate(source)) + noise * 10.0 ** (-40.0 / 20.0)
    assert out_of(read(gaps)) > out_of(read(programme + noise * 10.0 ** (-40.0 / 20.0))) + 3.0


def test_a_cut_of_the_programme_band_edge_trips_the_dulling_guard():
    """A -6 dB cut of 3-4.5 kHz alone (no low pass) reads the kept programme highs at -1 dB or less."""
    source = capture()[2]
    band = scipy.signal.sosfiltfilt(scipy.signal.butter(4, [3000.0, 4500.0], "bandpass", fs=RATE, output="sos"), source)
    assert out_of(read(source + band * (10.0 ** (-6.0 / 20.0) - 1.0)), "hiss_prog_kept_hf_db") <= -1.0


def test_the_anchor_is_filled_across_its_unread_bands():
    """Unread anchor bands take the log-interpolated level of their neighbours; fewer than two read, or none unread, stay as they are."""
    filled = hiss_programme.filled_anchor(np.array([1.0, np.nan, 100.0, 0.0, 100.0]))
    assert filled.tolist() == pytest.approx([1.0, 10.0, 100.0, 100.0, 100.0])
    assert np.isnan(hiss_programme.filled_anchor(np.array([np.nan, 2.0, np.nan]))).sum() == 2
    assert hiss_programme.capped(np.full((1, 3), 50.0), np.array([1.0, np.nan, 100.0])).tolist() == [[1.0, 10.0, 50.0]]


def test_the_spectral_cap_holds_a_band_within_6_db_of_the_other_bands():
    """A band reading 20 dB over its pool is held at the pool plus 6 dB; a thin pool or a NaN leaves it as it is."""
    ratio = np.array([[1.0, 1.0, np.nan]])
    pool = np.array([[[0.01, 0.01, 0.01]], [[0.001, 0.0001, 0.001]], [[100.0, 10.0, 100.0]]])
    capped = hiss_programme.spectral_cap(ratio, pool)
    assert capped[0, 0] == pytest.approx(10.0**0.6 * 0.1, rel=1e-6)
    assert capped[0, 1] == 1.0
    assert np.isnan(capped[0, 2])


def test_the_pool_leaves_out_the_band_and_its_neighbours():
    """Band 4 of 9 pools the free cells of bands 0-1 and 7-8 only (two bands either side out), frames +-1 summed."""
    layout = types_layout(np.full(9, 2000.0))
    src = np.zeros((3, 9))
    src[1] = np.arange(1.0, 10.0)
    pool = hiss_programme.pool_sums((src, src * 0.5, np.ones((3, 9))), layout)
    assert pool[0][1, 4] == pytest.approx(1 + 2 + 8 + 9)
    assert pool[1][1, 4] == pytest.approx(0.5 * (1 + 2 + 8 + 9))


def types_layout(centres):
    """A stand-in layout holding only `centres` (what `pool_sums` reads)."""
    return type("Stub", (), {"centres": np.asarray(centres, dtype=np.float64)})()


def test_a_band_is_free_when_two_frames_away_it_holds_only_the_floor():
    """A frame is free when the band holds only its floor two frames before and after: frames 0 and 4 see the loud frame 2."""
    layout = type("Stub", (), {"nbins": np.array([1.0]), "band_power": staticmethod(lambda power: power)})()
    power = np.array([[1.0], [1.0], [5.0], [1.0], [1.0], [1.0]])
    free = hiss_programme.free_band_frames(power, np.ones((6, 1)), layout)[:, 0]
    assert free.tolist() == [False, True, True, True, False, True]


def test_the_window_ratio_falls_back_with_too_few_cells():
    """Under 40 cells the class ratio is the fallback; with enough it is sum O / sum S."""
    sums = (np.full((2, 2), 10.0), np.full((2, 2), 5.0), np.array([[30.0, 1.0], [30.0, 1.0]]))
    assert hiss_programme.window_ratio(sums, np.array([0.7, 0.7])).tolist() == pytest.approx([0.5, 0.7])


def test_the_output_plays_at_its_own_loud_frames():
    """An output 3 dB quieter at its loud frames plays 3 dB louder; the programme gain moves it back."""
    context = hiss_floor.HissContext(None, None, 10.0, -20.0, -23.0)
    assert hiss_programme.output_spl_db(context, 0.0) == pytest.approx(hiss_floor.L_CRIT_DB + 23.0)
    assert hiss_programme.output_spl_db(context, -3.0) == pytest.approx(hiss_floor.L_CRIT_DB + 20.0)


def test_a_constant_envelope_correlates_nothing():
    """Either side holding one level in every frame reads correlation 0 (no envelope to match)."""
    freqs = np.array([500.0, 1000.0])
    assert hiss_programme.envelope_correlation(freqs, np.ones((4, 2)), np.arange(8.0).reshape(4, 2)) == 0.0


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
    assert out_of(readings, "hiss_prog_bands") >= 2.0
    assert out_of(readings, "hiss_prog_validated") == 1.0
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


def test_the_window_keeps_whole_frames_and_leaves_its_edges_out():
    """The half-padded first and last STFT frames are dropped; two frames at either end are neither active nor pause."""
    assert hiss_programme.whole_frames(np.arange(6.0)).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert hiss_programme.edge_frames(6).tolist() == [True, True, False, False, True, True]
    assert hiss_programme.edge_frames(1).tolist() == [True]


def test_a_fraction_of_a_sample_moves_nothing():
    """The output 0.37 samples late and a sample shorter reads every margin within 0.02 dB of the output as it was."""
    programme, noise, _source = capture()
    output = programme + noise * 0.25
    late = same_percept.fractional_delay(output, 0.37)[:-1]
    names = ("hiss_prog_margin_db", "hiss_prog_margin_all_db")
    plain, shifted = read(output), read(np.concatenate([late, [0.0]]))
    assert [out_of(shifted, name) for name in names] == pytest.approx([out_of(plain, name) for name in names], abs=0.02)


def test_a_window_under_four_frames_reads_nothing():
    """Under four STFT frames of samples there is no whole frame to read."""
    short = capture()[2][: 3 * hiss_floor.FRAME]
    assert hiss_programme.programme_readings(short, short, RATE, context_for(short, short)) == hiss_floor.none_pairs(
        hiss_programme.PROGRAMME_NAMES
    )
