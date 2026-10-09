"""R7 reads the output along the sync's lag track: a speed error the sync missed is no ride, a real ride still is."""

import functools

import numpy as np
import pytest

from scripts import degradations_v3 as v3
from scripts.restoration_quality import file_metrics
from tests.unit.test_restoration_quality_file_metrics import RATE, _long, _rider, _speech


@functools.lru_cache(maxsize=None)
def _long200():
    return _speech(200.0)


def _drifting(source, percent):
    """The source running `percent` slow, cut to the source's length as the mux keeps it beside the picture."""
    return v3.speed_drift(source, percent)[: len(source)]


def _ride(entries):
    return entries["file.gain_ride_lu"]["output"]


# The ride under a drift ------------------------------------------------------------------


@pytest.mark.parametrize(("percent", "bound"), [(0.2, 0.05), (0.5, 0.1)])
def test_a_drifting_output_is_no_ride(percent, bound):
    """0.2 % slow over 200 s drifts about 380 ms: the median lag alone reads it as a ride, the lag track reads none."""
    source = _long200()
    output = _drifting(source, percent)
    entries = file_metrics.file_entries(source, output, RATE)
    sync = file_metrics.sync_drift(source, output, RATE)
    median_only = file_metrics.gain_ride(source, output, RATE, file_metrics.median_lag_samples(sync, RATE))["p95_abs_lu"]
    assert abs(entries["file.sync_drift_ms"]["output"] - percent * 1920.0) < 0.03 * percent * 1920.0
    assert _ride(entries) < bound
    assert median_only > 3.0 * bound


def test_a_real_ride_still_reads_through_the_drift():
    """A +-3 dB rider at 0.2 Hz, then 0.2 % slow: the track takes out the drift, not the ride."""
    entries = file_metrics.file_entries(_long(), _drifting(_rider(_long()), 0.2), RATE)
    assert _ride(entries) > 1.5


def test_a_track_that_moves_no_window_leaves_the_block_grid_reading():
    """Anchors a hair apart (a benign file's 0.01 ms) read exactly what the median lag alone reads."""
    source, output = _long(), _rider(_long())
    track = [(4.0, 0.0), (50.0, 0.01), (96.0, -0.01)]
    assert file_metrics.gain_ride(source, output, RATE, 0, track) == file_metrics.gain_ride(source, output, RATE, 0)


# The track and the grid ------------------------------------------------------------------


def test_the_lag_track_is_zero_constant_or_linear_and_extends_past_the_end_anchors():
    """No anchor reads 0, one a constant, three a line through them carried on past the first and the last."""
    times = np.array([0.0, 10.0, 55.0, 100.0, 110.0])
    assert np.array_equal(file_metrics.lag_track([], times), np.zeros(5))
    assert np.allclose(file_metrics.lag_track([(50.0, 25.0)], times), 0.025)
    track = [(100.0, 100.0), (10.0, 10.0), (55.0, 55.0)]
    assert np.allclose(file_metrics.lag_track(track, times), times / 1000.0)


def test_two_anchors_at_one_time_carry_no_slope():
    """Anchors at one centre give the extension no slope to follow: the lag holds past them."""
    assert np.allclose(file_metrics.lag_track([(5.0, 2.0), (5.0, 4.0)], [0.0, 9.0]), [0.002, 0.004])


@pytest.mark.parametrize(("rate", "step"), [(16000, 16), (22050, 21), (44100, 42), (48000, 48)])
def test_the_track_step_divides_the_block_within_a_millisecond(rate, step):
    """44.1 kHz cannot step 44 samples: 4410 is no multiple of it, so a window the track does not move would land off the grid."""
    block = int(round(file_metrics.BLOCK_S * rate))
    assert file_metrics.track_step(rate) == step
    assert block % step == 0
    assert step <= file_metrics.TRACK_STEP_S * rate


def test_window_levels_on_the_grid_are_the_short_term_levels_and_nan_off_either_end():
    """Read at the block grid's own starts the levels are the block path's; a window off either end is NaN."""
    mono = _speech(12.0)
    levels = file_metrics.short_term_loudness(mono, RATE)
    starts = file_metrics.window_starts(len(levels), RATE)
    assert np.allclose(file_metrics.window_levels(mono, RATE, starts), levels, atol=1e-9)
    assert np.isnan(file_metrics.window_levels(mono, RATE, np.array([-RATE, 10.5 * RATE]))).all()


def test_offsets_are_zero_without_a_track_and_follow_an_early_output_on_the_raw_timeline():
    """No anchor leaves the shift alone; an early output cut the source's head, so its centres sit later on the raw clock."""
    assert np.array_equal(file_metrics.window_offsets([], 160, RATE, 4), np.zeros(4))
    track = [(0.0, 0.0), (100.0, 100.0)]
    late = file_metrics.window_offsets(track, 0, RATE, 2)
    early = file_metrics.window_offsets(track, -160, RATE, 2)
    assert np.allclose(late, [1.5 * RATE / 1000.0, 2.5 * RATE / 1000.0])
    assert np.allclose(early, late + 160 + 0.16)
