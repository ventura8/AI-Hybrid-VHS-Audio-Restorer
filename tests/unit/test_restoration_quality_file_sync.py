"""The sync reading times the output against the source near its start, centre and end, and a fault fails it, never skips it."""

import functools

import numpy as np
import pytest
import scipy.signal

from scripts.restoration_quality import file_metrics
from tests.unit.test_restoration_quality_file_metrics import PAL_FRAME_MS, RATE, _delayed, _long, _speech

SECONDS = (0, 8, 16, 24)
# The local normalisation leaves the parabola a hair off centre on an exact match: under 0.01 ms.
EXACT_MS = 0.05


@functools.lru_cache(maxsize=None)
def _long200():
    return _speech(200.0)


def _slower(source, percent):
    """The source played `percent` slower: every second of it lasts 1 + percent/100 s."""
    return scipy.signal.resample_poly(source, int(round(1000 * (1 + percent / 100.0))), 1000).astype(np.float32)


def _lags(sync):
    return [sync[f"lag_{name}_ms"] for name in file_metrics.SEGMENT_NAMES]


def _silenced(seconds):
    source = _long().copy()
    source[: int(seconds * RATE)] = 0.0
    return source


# Anchors ---------------------------------------------------------------------------------


def test_anchors_start_at_the_edges_and_the_centre_and_step_inward():
    """8 s anchors over a 100 s file: forward from 0, outward from 46 s, back from 92 s, four tries a third."""
    length, (head, middle, tail) = file_metrics.anchor_candidates(100 * RATE, RATE)
    assert length == 8 * RATE
    assert np.array_equal(np.array(head) / RATE, SECONDS)
    assert np.array_equal(np.array(middle) / RATE, (46, 38, 54, 30, 62))
    assert np.array_equal(92 - np.array(tail) / RATE, SECONDS)


def test_a_short_file_reads_thirds_a_long_one_caps_its_tries_and_a_tiny_one_reads_nothing():
    """Under 24 s each anchor is a third of the file; a three-hour tape tries 16 a third; under 6 s there are none."""
    length = int(10 * RATE / 3)
    assert file_metrics.anchor_candidates(10 * RATE, RATE) == (length, ([0], [(10 * RATE - length) // 2], [10 * RATE - length]))
    assert len(file_metrics.anchor_candidates(3 * 3600 * RATE, RATE)[1][0]) == file_metrics.MAX_ANCHOR_TRIES
    assert file_metrics.anchor_candidates(5 * RATE, RATE) == (0, ((), (), ()))


def test_the_search_widens_along_the_file():
    """2 s at the start, 2.5 % of the end's time further in, 30 s at most."""
    frame = 16
    assert file_metrics.search_frames(slice(0, 8 * RATE), RATE, frame) == 2 * RATE // frame
    assert file_metrics.search_frames(slice(192 * RATE, 200 * RATE), RATE, frame) == 5 * RATE // frame
    assert file_metrics.search_frames(slice(0, 3 * 3600 * RATE), RATE, frame) == 30 * RATE // frame


def test_a_silent_start_moves_the_start_anchor_inward():
    """A 16 s silent leader: the start anchor is the first 8 s that carries programme, and it still matches."""
    source = _silenced(16.0)
    length, (head, _middle, _tail) = file_metrics.anchor_candidates(len(source), RATE)
    anchor = file_metrics.pick_anchor(source, RATE, file_metrics.source_floor(source, RATE), head, length)
    assert anchor[0].start == 16 * RATE
    assert file_metrics.sync_drift(source, source, RATE)["segments"] == 3


def test_a_smooth_burst_has_no_timing_and_speech_has():
    """An envelope that barely moves within one PAL frame cannot time the output to one; speech can."""
    count = 3 * RATE
    burst = np.hanning(count) * np.sin(2 * np.pi * 1000.0 * np.arange(count) / RATE)
    assert not file_metrics.has_timing(file_metrics.envelope(burst, RATE, 16), RATE)
    assert file_metrics.has_timing(file_metrics.envelope(_speech(8.0), RATE, 16), RATE)
    assert not file_metrics.has_timing(np.zeros(1000), RATE)
    assert not file_metrics.has_timing(np.ones(2), RATE)


# Lags, drift and offset ------------------------------------------------------------------


def test_identity_reads_no_lag_and_no_drift():
    """The source against itself lags by 0 ms on every anchor."""
    sync = file_metrics.sync_drift(_long(), _long(), RATE)
    assert (sync["segments"], sync["unmatched"]) == (3, 0)
    assert np.max(np.abs(_lags(sync))) < EXACT_MS
    assert sync["drift_ms"] < EXACT_MS


def test_a_constant_delay_reads_the_same_lag_everywhere_and_no_drift():
    """A fixed 25 ms delay is a lag on each anchor, the offset, and no drift."""
    sync = file_metrics.sync_drift(_long(), _delayed(_long(), 25.0), RATE)
    assert np.allclose(_lags(sync), 25.0, atol=0.5)
    assert sync["drift_ms"] < 1.0
    assert abs(sync["offset_ms"] - 25.0) < 0.5


def test_an_output_ahead_of_the_source_reads_a_negative_lag():
    """An output that leads its source reads a negative lag, and the offset is its size."""
    sync = file_metrics.sync_drift(_long(), _delayed(_long(), -10.0), RATE)
    assert abs(sync["lag_middle_ms"] + 10.0) < 0.5
    assert abs(sync["offset_ms"] - 10.0) < 0.5


def test_a_progressively_late_output_reads_drift_between_the_anchor_centres():
    """0.1 % slower falls 1 ms behind per second: 4 ms at the start anchor's centre, 92 ms of drift to the end's."""
    sync = file_metrics.sync_drift(_long(), _slower(_long(), 0.1), RATE)
    assert sync["lag_start_ms"] < sync["lag_middle_ms"] < sync["lag_end_ms"]
    assert abs(sync["lag_start_ms"] - 4.0) < 1.5
    assert abs(sync["drift_ms"] - 92.0) < 5.0
    assert sync["drift_ms"] > PAL_FRAME_MS


@pytest.mark.parametrize("percent", [1.0, 2.0])
def test_a_one_or_two_percent_slow_output_reads_its_drift(percent):
    """A rate error the old +-1 s search lost: 1-2 % over 192 s between the anchors is 1.9-3.8 s of drift."""
    sync = file_metrics.sync_drift(_long200(), _slower(_long200(), percent), RATE)
    assert sync["unmatched"] == 0
    assert abs(sync["drift_ms"] - percent * 1920.0) < 0.02 * percent * 1920.0


def test_a_five_percent_slow_output_is_unmatched_not_skipped():
    """5 % slow outruns the search: every anchor reads unmatched, which fails where a None drift would skip."""
    sync = file_metrics.sync_drift(_long200(), _slower(_long200(), 5.0), RATE)
    assert sync["unmatched"] == 3
    assert sync["drift_ms"] is None


def test_a_constant_shift_of_one_and_a_half_seconds_is_an_offset():
    """1.5 s late everywhere is no drift, but the offset says the lips and the picture are 1.5 s apart."""
    sync = file_metrics.sync_drift(_long(), _delayed(_long(), 1500.0), RATE)
    assert sync["unmatched"] == 0
    assert sync["drift_ms"] < 1.0
    assert abs(sync["offset_ms"] - 1500.0) < 1.0


def test_a_shift_past_the_search_is_unmatched_where_it_cannot_be_found():
    """3 s late: the start and the centre search 2-2.6 s and read unmatched; the end searches 5 s and finds it."""
    sync = file_metrics.sync_drift(_long200(), _delayed(_long200(), 3000.0), RATE)
    assert sync["unmatched"] == 2
    assert abs(sync["lag_end_ms"] - 3000.0) < 1.0


# Unread and unmatched --------------------------------------------------------------------


def test_an_unrelated_or_silent_output_is_unmatched():
    """Noise does not follow the speech envelope, and a silent output has no envelope: every anchor is unmatched."""
    noise = (0.05 * np.random.default_rng(1).standard_normal(len(_long()))).astype(np.float32)
    assert file_metrics.sync_drift(_long(), noise, RATE)["unmatched"] == 3
    silent = file_metrics.sync_drift(_long(), np.zeros_like(_long()), RATE)
    assert silent["unmatched"] == 3
    assert silent["segments"] == 0


def test_a_third_without_programme_is_unread_and_the_others_still_count():
    """A silent first third reads None at the start, counts as nothing, and the middle and the end give the drift."""
    sync = file_metrics.sync_drift(_silenced(34.0), _silenced(34.0), RATE)
    assert sync["lag_start_ms"] is None
    assert (sync["segments"], sync["unmatched"]) == (2, 0)
    assert sync["drift_ms"] < EXACT_MS


def test_a_lone_lag_is_not_a_drift():
    """One lag found is no change to report, but it is still an offset."""
    sync = file_metrics.sync_drift(_silenced(70.0), _silenced(70.0), RATE)
    assert sync["segments"] == 1
    assert sync["drift_ms"] is None
    assert sync["offset_ms"] < EXACT_MS


def test_a_nan_in_the_source_reads_no_false_lag():
    """A NaN in the source's first anchor makes that anchor unreadable: the next 8 s is read, at 0 ms."""
    source = _long().copy()
    source[1000] = np.nan
    sync = file_metrics.sync_drift(source, _long(), RATE)
    assert sync["unmatched"] == 0
    assert abs(sync["lag_start_ms"]) < EXACT_MS


def test_a_nan_in_the_output_is_unmatched():
    """A broken stage that leaves a NaN in the output fails the anchor it lands in."""
    output = _long().copy()
    output[1000] = np.nan
    sync = file_metrics.sync_drift(_long(), output, RATE)
    assert sync["lag_start_ms"] is None
    assert (sync["segments"], sync["unmatched"]) == (2, 1)


def test_an_output_ending_one_frame_into_the_end_search_is_unmatched_not_a_crash():
    """At 16 kHz one 1 ms frame is 16 samples, inside the band filter's 27-sample edge padding: the end anchor's
    search holds that one frame of output and reads unmatched, where the filter used to raise."""
    source, frame = _long(), int(round(file_metrics.ENVELOPE_FRAME_S * RATE))
    length, (_head, _middle, tail) = file_metrics.anchor_candidates(len(source), RATE)
    span, _template, reach = file_metrics.pick_anchor(source, RATE, file_metrics.source_floor(source, RATE), tail, length)
    sync = file_metrics.sync_drift(source, source[: (span.start // frame - reach + 1) * frame], RATE)
    assert (sync["segments"], sync["unmatched"]) == (2, 1)
    assert sync["lag_end_ms"] is None


def test_a_slice_inside_the_filters_edge_padding_reads_a_flat_envelope():
    """27 samples or fewer read the envelope's mean, one zero per whole frame; 32 samples are filtered."""
    sos = scipy.signal.butter(4, file_metrics.SYNC_BAND_HZ, btype="bandpass", fs=RATE, output="sos")
    assert file_metrics.edge_padding(sos) == 27
    speech = _speech(1.0)
    assert np.array_equal(file_metrics.envelope(speech[:27], RATE, 16), np.zeros(1))
    assert len(file_metrics.envelope(speech[:32], RATE, 16)) == 2


def test_the_parabola_finds_the_true_peak_and_leaves_the_edges():
    """The parabola recovers a sub-frame peak and returns 0 at an edge or on a flat top."""
    values = -((np.arange(5) - 2.3) ** 2)
    assert abs(file_metrics.parabolic_offset(values, 2) - 0.3) < 1e-12
    assert file_metrics.parabolic_offset(values, 0) == 0.0
    assert file_metrics.parabolic_offset(values, 4) == 0.0
    assert file_metrics.parabolic_offset(np.array([1.0, 1.0, 1.0]), 1) == 0.0


def test_the_envelope_follows_the_band_at_a_low_rate():
    """Below 6.7 kHz the band's top follows Nyquist; the envelope is one value per frame, mean removed."""
    env = file_metrics.envelope(_speech(3.0, rate=4000), 4000, 4)
    assert len(env) == 3000
    assert abs(float(env.mean())) < 1e-9
