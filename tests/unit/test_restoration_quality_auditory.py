"""The auditory building blocks: ERB scale, programme cells, the programme gain match, the pieces of the audibility check."""

import numpy as np
import pytest

from scripts.restoration_quality import auditory

RATE = 44100


def test_erb_number_round_trips_and_matches_glasberg_moore():
    """Glasberg & Moore 1990: 1 kHz sits at 15.62 Cams, and the inverse returns the frequency."""
    assert abs(float(auditory.erb_number(1000.0)) - 15.62) < 0.01
    freqs = np.array([100.0, 1000.0, 8000.0])
    assert np.allclose(auditory.erb_to_hz(auditory.erb_number(freqs)), freqs)


def test_band_edges_are_one_erb_apart_and_span_the_range():
    """The bands tile 100 Hz-16 kHz in steps of one ERB."""
    edges = auditory.erb_band_edges(100.0, 16000.0)
    assert abs(edges[0] - 100.0) < 1e-6
    assert abs(edges[-1] - 16000.0) < 1e-6
    steps = np.diff(auditory.erb_number(edges))
    assert np.all((steps > 0.9) & (steps < 1.1))


def test_a_degenerate_range_is_one_band():
    """An empty range is one band from the low edge to the high one."""
    assert list(auditory.erb_band_edges(500.0, 500.0)) == [500.0, 500.0]


def test_band_index_marks_bins_outside_the_edges():
    """Bins below the first edge or at and above the last belong to no band."""
    edges = np.array([100.0, 200.0, 400.0])
    index = auditory.band_index(np.array([50.0, 150.0, 250.0, 400.0]), edges)
    assert list(index) == [-1, 0, 1, -1]


def _power(frames=200, bins=64, seed=0):
    rng = np.random.default_rng(seed)
    floor = np.full((bins, frames), 1e-6)
    loud = rng.random(frames) > 0.5
    floor[:, loud] += 1e-2
    return floor * (1.0 + 0.1 * rng.random((bins, frames))), loud


def test_programme_cells_are_loud_frames_above_the_floor():
    """Only loud frames carry programme cells, and they do carry some."""
    power, loud = _power()
    cells = auditory.programme_cells(power)
    assert cells[:, ~loud].sum() == 0
    assert cells.any()


def test_the_programme_gain_match_reads_a_broadband_gain_and_ignores_a_shelf_above_3_khz():
    """The match is taken in 300-3000 Hz, so a shelf above 4 kHz does not leak into the +3 dB it reads."""
    power, _loud = _power(bins=128)
    freqs = np.linspace(0.0, 12000.0, 128)
    cells = auditory.programme_cells(power)
    shelf = power * 2.0
    shelf[freqs >= 4000.0] *= 4.0
    assert abs(auditory.programme_gain_db(power, shelf, freqs, cells) - 3.01) < 0.05


def test_too_few_programme_cells_give_no_gain():
    """With no programme cell in 300-3000 Hz there is no gain to read."""
    power, _loud = _power(bins=8)
    freqs = np.linspace(5000.0, 9000.0, 8)
    assert auditory.programme_gain_db(power, power, freqs, auditory.programme_cells(power)) is None


# --- the audibility check's building blocks ---------------------------------------------


def test_the_lag_follows_the_reference_scorer_convention():
    """A positive lag is a late candidate, as in `score_reference._align`; an inverted early one reads (-2, -1)."""
    noise = (0.1 * np.random.default_rng(3).standard_normal(2 * RATE)).astype(np.float32)
    late = np.concatenate([np.zeros(3, dtype=np.float32), noise[:-3]])
    assert auditory.estimate_lag(noise, late, RATE) == (3, 1)
    assert auditory.estimate_lag(noise, -noise[2:], RATE) == (-2, -1)


def test_null_aligned_trims_the_early_side_and_applies_the_polarity():
    """The late side loses its head, both are cut to the common length, the candidate takes the sign."""
    ramp = np.arange(10.0, dtype=np.float32)
    incumbent, candidate = auditory.null_aligned(ramp, ramp + 100.0, 3, -1)
    assert (len(incumbent), len(candidate), float(candidate[0])) == (7, 7, -103.0)
    incumbent, candidate = auditory.null_aligned(ramp, ramp, -2, 1)
    assert (float(incumbent[0]), len(candidate)) == (2.0, 8)


def test_window_slices_cover_every_sample_and_fold_a_short_tail():
    """A tail under half a window joins the window before; a longer one stands alone."""
    assert auditory.window_slices(100, 1, 30.0) == [slice(0, 30), slice(30, 60), slice(60, 100)]
    assert auditory.window_slices(100, 1, 40.0) == [slice(0, 40), slice(40, 80), slice(80, 100)]
    assert auditory.window_slices(0, RATE) == []


def test_the_threshold_in_quiet_follows_terhardt():
    """Terhardt 1979: 3.4 dB SPL at 1 kHz, about -5 dB at 3.3 kHz, about 60 dB at the 15.6 kHz line."""
    levels = auditory.threshold_in_quiet_db(np.array([1000.0, 3300.0, 15625.0]))
    assert abs(levels[0] - 3.37) < 0.05
    assert levels[1] < -4.9
    assert levels[2] > 50.0


def test_masking_spreads_further_upwards_and_is_floored_in_quiet():
    """A band masks its upper neighbour more than its lower one; with no masker the threshold is the one in quiet."""
    spread = auditory.spreading_matrix(3)
    assert np.allclose(np.diag(spread), 1.0)
    assert spread[1, 0] > spread[0, 1]
    _matrix, centres = auditory.nmr_bands(RATE)
    quiet = 10.0 ** (auditory.threshold_in_quiet_db(centres) / 10.0)
    assert np.allclose(auditory.masked_threshold(np.zeros((len(centres), 4)), centres), quiet[:, np.newaxis])


def test_every_nmr_band_holds_a_bin_and_a_low_rate_stops_at_nyquist():
    """1-ERB bands from 50 Hz on 21.5 Hz bins are never empty; at 16 kHz the bands end below 8 kHz."""
    matrix, centres = auditory.nmr_bands(RATE)
    assert matrix.sum(axis=1).min() >= 1.0
    assert centres[-1] < 16000.0
    assert auditory.nmr_bands(16000)[1][-1] < 8000.0


def test_frame_power_sums_to_the_mean_square_level():
    """The STFT scaling: a 0.5 amplitude sine sums to 0.125 over the bins of a frame."""
    t = np.arange(RATE) / RATE
    power = auditory.frame_power(0.5 * np.sin(2 * np.pi * 1000.0 * t))
    assert abs(float(np.median(power.sum(axis=0))) - 0.125) < 0.002


def test_loud_frame_power_reads_short_and_silent_signals():
    """Under one frame the level is the plain mean square; silence reads the epsilon, not zero."""
    assert auditory.loud_frame_power(np.full((10, 1), 0.5, dtype=np.float32)) == pytest.approx(0.25)
    assert auditory.loud_frame_power(np.zeros((4096, 2), dtype=np.float32)) == pytest.approx(auditory.EPS)


def test_floored_db_keeps_nan_and_floors_no_difference():
    """An undefined ratio stays NaN, never "no difference"; zero reads the floor and 10 reads +10 dB."""
    readings = auditory.floored_db(np.array([np.nan, 0.0, 10.0]))
    assert np.isnan(readings[0])
    assert readings[1:].tolist() == pytest.approx([auditory.DB_FLOOR, 10.0])


def test_the_verdict_takes_a_spread_change_or_an_event():
    """5% audible frames or 2 event frames make a pair audible; one event frame alone under +12 dB does not."""
    assert auditory.is_audible(0.04, 1) is False
    assert auditory.is_audible(0.05, 0) is True
    assert auditory.is_audible(0.0, auditory.EVENT_MIN_FRAMES) is True


def test_one_frame_at_12_db_is_an_event():
    """A click on a hop boundary lands in one frame only: that frame at +12 dB is an event, 0.1 dB less is not."""
    readings = [auditory.is_audible(0.0, 1, nmr_max) for nmr_max in (auditory.SINGLE_EVENT_NMR_DB, auditory.SINGLE_EVENT_NMR_DB - 0.1)]
    assert readings == [True, False]


def test_the_verdict_takes_a_pair_it_cannot_vouch_for():
    """A channel mismatch, a non-finite sample or more than 50 ms left uncompared once aligned (head and tail together)
    each make a pair audible."""
    unvouched = [{"channel_mismatch": True}, {"nonfinite": 1}, {"length_mismatch_s": auditory.LENGTH_TOLERANCE_S + 0.001}]
    assert [auditory.is_audible(0.0, 0, flags=flags) for flags in unvouched] == [True, True, True]
    assert auditory.is_audible(0.0, 0, flags={"length_mismatch_s": auditory.LENGTH_TOLERANCE_S}) is False


def test_the_unmatched_span_is_the_head_the_lag_drops_plus_the_tail_of_the_longer_side():
    """Equal lengths at lag 0 leave nothing; a lag of one leaves a sample at each end; a head cut is counted in full."""
    cases = [(100, 100, 0), (100, 100, 1), (100, 100, -1), (100, 98, -2), (100, 40, 0), (40, 100, 0), (100, 60, -40), (60, 100, 40)]
    assert [auditory.unmatched_samples(*case) for case in cases] == [0, 2, 2, 2, 60, 60, 40, 40]


def test_a_mono_side_is_repeated_on_every_channel_of_the_other():
    """Mono against stereo becomes stereo against stereo, in either order, with no mismatch flagged."""
    mono, stereo = np.arange(4.0, dtype=np.float32), np.ones((4, 2), dtype=np.float32)
    incumbent, candidate, mismatch = auditory.matched_channels(mono, stereo)
    assert (incumbent.shape, candidate.shape, mismatch) == ((4, 2), (4, 2), False)
    assert np.array_equal(incumbent[:, 1], mono)
    assert auditory.matched_channels(stereo, mono)[1].shape == (4, 2)


def test_two_multichannel_counts_are_downmixed_and_flagged():
    """Stereo against three channels: both become their mono downmix and the mismatch is reported."""
    incumbent, candidate, mismatch = auditory.matched_channels(np.ones((4, 2), dtype=np.float32), np.ones((4, 3), dtype=np.float32))
    assert (incumbent.shape, candidate.shape, mismatch) == ((4, 1), (4, 1), True)


def test_finite_part_zeroes_and_counts_nan_and_inf():
    """NaN and both infinities become 0 and are counted; finite samples pass unchanged."""
    cleaned, count = auditory.finite_part(np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32))
    assert count == 3
    assert list(cleaned[:, 0]) == [1.0, 0.0, 0.0, 0.0]
