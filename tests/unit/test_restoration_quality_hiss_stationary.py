"""Ear v4's second hiss anchor (`hiss_stationary`): the stationary floor of continuous programme, its band edge, its
validation, and the programme reading that caps with it when the pause anchor is not validated."""

from functools import lru_cache

import numpy as np
import pytest
import scipy.signal

from scripts.restoration_quality import hiss_floor, hiss_programme, hiss_stationary
from tests.unit.test_restoration_quality_hiss_floor import RATE, capture, hiss

LONG_S = 30.0
LINEAR_HZ = 4500.0
MARGIN = "hiss_prog_margin_db"
BLOCK = hiss_stationary.frames_per(hiss_stationary.BLOCK_S, RATE)
SECOND = hiss_stationary.frames_per(hiss_stationary.EDGE_BLOCK_S, RATE)
CENTRES = np.array([500.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0, 11000.0, 13000.0])


def lowpass(mono, cutoff_hz=LINEAR_HZ):
    """`mono` through an 8th-order Butterworth low pass, zero phase."""
    return scipy.signal.sosfiltfilt(scipy.signal.butter(8, cutoff_hz, "low", fs=RATE, output="sos"), mono)


@lru_cache(maxsize=None)
def notes(seconds=LONG_S, linear=True):
    """Harmonic notes of 0.12 s every 0.3 s: the gaps (0.18 s) are too short for R4's true pauses, long enough for the hiss."""
    rng = np.random.default_rng(3)
    out = np.zeros(int(seconds * RATE))
    for start in np.arange(0.05, seconds - 0.12, 0.3):
        first, last = int(start * RATE), int((start + 0.12) * RATE)
        t = np.arange(last - first) / RATE
        f0 = rng.uniform(200.0, 400.0)
        partials = [np.sin(2 * np.pi * k * f0 * t + rng.uniform(0.0, 2.0 * np.pi)) / k for k in range(1, 40) if k * f0 < RATE / 2.2]
        out[first:last] += np.sum(partials, axis=0) * np.hanning(last - first) * 0.2
    return lowpass(out) if linear else out


@lru_cache(maxsize=None)
def dense(seconds=15.0):
    """A held chord over a band-limited bed, both swelling +-3 dB at 0.4 Hz: no gap anywhere for the hiss to show."""
    rng = np.random.default_rng(4)
    t = np.arange(int(seconds * RATE)) / RATE
    chord = sum(np.sin(2 * np.pi * f * k * t + rng.uniform(0.0, 2.0 * np.pi)) / k for f in (220.0, 277.2, 329.6) for k in range(1, 18))
    swell = 10.0 ** (3.0 * np.sin(2 * np.pi * 0.4 * t) / 20.0)
    return lowpass(chord * 0.05 * swell) + lowpass(rng.standard_normal(len(t))) * 0.05 * swell


def source_of(programme):
    """`(source, hiss)`: the programme over the shared tape-like hiss."""
    noise = hiss(len(programme))
    return programme + noise, noise


@lru_cache(maxsize=None)
def gapped_anchor():
    """The runner's anchor on the gapped, linear programme."""
    return hiss_stationary.source_anchor(source_of(notes())[0], RATE, LINEAR_HZ, None)


def true_band_means(noise):
    """The noise's true per-bin band means and the band centres."""
    freqs, power = hiss_floor.stft_power(noise, RATE)
    index, centres = hiss_floor.band_layout(freqs, RATE)
    return hiss_floor.band_means(power, index, len(centres)), centres


def segment(ok, level, present, hiss_only):
    """A `Segment` holding a floor in every band."""
    ok = np.asarray(ok, dtype=bool)
    return hiss_stationary.Segment(
        ok, np.asarray(level, dtype=np.float64), np.asarray(present), np.asarray(hiss_only), np.ones(len(ok), bool)
    )


def test_frames_and_whole_blocks():
    """A 3 s block is 129 hops at 44.1 kHz, never fewer than one; the tail that fills no block is dropped."""
    assert (hiss_stationary.frames_per(3.0, RATE), hiss_stationary.frames_per(0.0, RATE)) == (129, 1)
    assert hiss_stationary.blocks_of(np.arange(10.0).reshape(5, 2), 2).shape == (2, 2, 2)


def test_block_floors_scale_the_p10_by_c_b_and_read_nan_as_zero():
    """Per block, the P10 of a constant valley times C_b; a band never read floors at 0."""
    valley = np.column_stack([np.full(3 * BLOCK, 2.0), np.full(3 * BLOCK, np.nan)])
    floors = hiss_stationary.block_floors(valley, np.array([0.5, 1.0]), BLOCK)
    np.testing.assert_allclose(floors, [[1.0, 0.0]] * 3)


def test_a_spread_over_a_zero_p10_is_infinite():
    """A steady column spreads 0 dB; one whose p10 is 0 cannot be read as steady."""
    spread = hiss_stationary.spread_db(np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))
    assert spread[0] == pytest.approx(0.0, abs=1e-9)
    assert np.isinf(spread[1])


def test_the_stationary_bands_need_three_blocks_over_the_dither():
    """Two blocks read nothing; with three, a steady band over the dither is accepted at its blocks' P25, one under it is not."""
    level = hiss_floor.dither_limit() * 100.0
    valley = np.column_stack([np.full(3 * BLOCK, level), np.full(3 * BLOCK, level * 1e-4)])
    ok, low = hiss_stationary.stationary_bands(valley, np.ones(2), RATE)
    assert hiss_stationary.stationary_bands(valley[: 2 * BLOCK], np.ones(2), RATE) is None
    assert ok.tolist() == [True, False]
    assert low[0] == pytest.approx(level)


def test_a_wandering_band_is_not_stationary():
    """Block floors 3 dB apart span more than the 1 dB a stationary floor may."""
    level = hiss_floor.dither_limit() * 100.0
    steps = np.repeat([1.0, 2.0, 1.0, 2.0], BLOCK)[:, np.newaxis] * level
    assert hiss_stationary.stationary_bands(steps, np.ones(1), RATE)[0].tolist() == [False]


def test_the_envelope_correlation_finds_programme_and_spares_a_constant():
    """A band that follows the envelope correlates 1; a constant band reads 0, not NaN."""
    envelope = np.random.default_rng(1).random(400) + 0.5
    band_power = np.column_stack([3.0 * envelope + 1.0, np.ones(400)])
    correlation = hiss_stationary.envelope_correlation(band_power, envelope)
    assert correlation[0] == pytest.approx(1.0)
    assert correlation[1] == pytest.approx(0.0, abs=1e-12)


def test_the_programme_flags_of_one_segment():
    """Bands: following the envelope, steady hiss at the floor, 20 dB over the floor, and nothing over the dither."""
    count, level = 10 * SECOND, hiss_floor.dither_limit() * 1e3
    envelope = np.random.default_rng(2).random(count) + 0.5
    valley = np.column_stack([np.full(count, level)] * 3 + [np.full(count, level * 1e-6)])
    band_power = np.column_stack([level * envelope, np.full(count, level), np.full(count, level * 100.0), np.full(count, level * 1e-6)])
    present, hiss_only, held = hiss_stationary.programme_flags((valley, band_power, envelope), np.ones(4), RATE)
    assert present.tolist() == [True, False, True, False]
    assert hiss_only.tolist() == [False, True, False, False]
    assert held.tolist() == [True, True, True, False]


def test_majority_and_the_accepted_level():
    """A band counts where half the segments say so; its level is the median over the segments that accept it."""
    segments = [segment([True, True], [1.0, 9.0], [0, 0], [0, 0]), segment([True, False], [3.0, 5.0], [0, 0], [0, 0])]
    accepted = hiss_stationary.majority([[True, False], [True, True], [False, False]])
    assert accepted.tolist() == [True, False]
    level = hiss_stationary.accepted_level(segments, np.array([True, False]))
    assert level[0] == pytest.approx(2.0)
    assert np.isnan(level[1])


def test_the_band_edge_starts_half_an_octave_over_the_programme():
    """Programme up to 4 kHz: the hiss-alone bands from 5.66 kHz to under 12 kHz; no programme from 1 kHz up: 1 kHz."""
    present = np.array([True, True, False, True, False, False, False, False])
    hiss_only = ~present
    assert hiss_stationary.programme_top(CENTRES, present) == pytest.approx(4000.0)
    np.testing.assert_allclose(CENTRES[hiss_stationary.edge_bands(CENTRES, present, hiss_only)], [6000.0, 8000.0, 11000.0])
    assert hiss_stationary.programme_top(CENTRES, np.array([True] + [False] * 7)) == pytest.approx(hiss_stationary.LO_HZ)


def test_coverage_allows_four_misses_and_wants_six_bands():
    """Seven bands of 1 kHz and up hold a floor: six accepted cover them, five do not; ten held want six of them."""
    held = np.ones(len(CENTRES), dtype=bool)
    six = np.array([False, True, True, True, True, True, True, False])
    five = six & (CENTRES != 2000.0)
    wide = np.linspace(1000.0, 10000.0, 10)
    assert (hiss_stationary.covered(CENTRES, six, held), hiss_stationary.covered(CENTRES, five, held)) == (True, False)
    assert hiss_stationary.covered(wide, np.arange(10) < 6, np.ones(10, dtype=bool))


def test_a_gapped_linear_programme_validates_the_stationary_anchor():
    """Notes low-passed at 4.5 kHz with gaps under R4's 200 ms: no pause anchor, a stationary one within 2 dB of the hiss."""
    anchor = gapped_anchor()
    truth, centres = true_band_means(source_of(notes())[1])
    second = anchor.programme
    read = np.isfinite(second.level) & (centres >= hiss_stationary.LO_HZ)
    assert (anchor.validated, second.how, second.validated, anchor.programme_how()) == (
        False,
        hiss_floor.STATIONARY,
        True,
        hiss_floor.STATIONARY,
    )
    assert int(read.sum()) >= 20
    assert np.max(np.abs(10.0 * np.log10(second.level[read] / truth[read]))) <= 2.0


def test_a_full_band_programme_has_no_band_edge():
    """The same notes not low-passed reach 15 kHz: the hiss shows through every band, but no band is hiss alone."""
    second = hiss_stationary.stationary_anchor(source_of(notes(linear=False))[0], RATE, None)
    assert second.validated is False


def test_a_dense_programme_hides_the_hiss():
    """A swelling chord over a bed leaves no band's floor stationary enough: no stationary anchor."""
    second = hiss_stationary.source_anchor(source_of(dense())[0], RATE, LINEAR_HZ, None).programme
    assert second.validated is False


def test_a_validated_pause_anchor_is_returned_as_it_is():
    """Gapped speech with long pauses: the pause anchor, its level unchanged, and no second path tried."""
    source = capture()[2].astype(np.float32)
    anchor = hiss_stationary.source_anchor(source, RATE, LINEAR_HZ, None)
    np.testing.assert_array_equal(anchor.level, hiss_floor.source_anchor(source, RATE, LINEAR_HZ, None).level)
    assert (anchor.programme, anchor.programme_how()) == (None, hiss_floor.PAUSES)


def test_a_source_too_short_for_three_blocks_has_no_second_anchor():
    """8 s of notes hold two 3 s blocks: nothing to accept; under four frames no segment is read at all."""
    assert hiss_stationary.source_anchor(source_of(notes(8.0))[0], RATE, LINEAR_HZ, None).programme is None
    assert hiss_stationary.read_segment(np.zeros(100), RATE, hiss_floor.layout_top(RATE)) is None


def _window_readings(output, anchor):
    source = source_of(notes())[0][: int(15.0 * RATE)]
    output = output[: len(source)]
    context = hiss_floor.HissContext(anchor, None, hiss_floor.spl_offset_db(source), hiss_floor.loud_db(source), hiss_floor.loud_db(output))
    return hiss_programme.programme_readings(source, output, RATE, context)


def test_the_programme_reading_validates_on_the_stationary_anchor():
    """The perfect output reads inaudible, the raw tape audible, both validated through the second anchor."""
    source, _noise = source_of(notes())
    perfect, raw = _window_readings(notes(), gapped_anchor()), _window_readings(source, gapped_anchor())
    assert perfect[MARGIN][1] <= 0.0
    assert raw[MARGIN][1] > 0.0
    assert raw["hiss_prog_validated"][1] > 0.5


def test_an_unvalidated_second_anchor_reads_no_margin():
    """When neither anchor validated, the margins read None as before."""
    anchor = gapped_anchor()
    unvalidated = hiss_floor.Anchor(
        anchor.level, anchor.centres, 0.0, anchor.how, False, hiss_floor.Anchor(None, np.zeros(0), 0.0, "x", False)
    )
    assert _window_readings(source_of(notes())[0], unvalidated)[MARGIN] == (None, None)
