"""The harness's aligner: the loudest spans voting by GCC-PHAT, abstention, polarity, and the old head search kept for stored scores."""

import numpy as np
import pytest
import scipy.signal

from scripts import pair_alignment as pa

RATE = 44100
MUX = 219


def _speechlike(seconds, seed=3, floor=1e-3):
    """Band-limited noise gated by a random 100 ms syllable pattern over a faint hiss floor."""
    rng = np.random.default_rng(seed)
    count = int(seconds * RATE)
    sos = scipy.signal.butter(4, [200.0, 4000.0], btype="band", fs=RATE, output="sos")
    voiced = scipy.signal.sosfilt(sos, rng.standard_normal(count))
    gates = np.repeat(rng.uniform(0.0, 1.0, count // 4410 + 1) > 0.4, 4410).astype(float)
    envelope = scipy.signal.lfilter([0.01], [1.0, -0.99], gates[:count])
    return 0.3 * voiced * envelope + floor * rng.standard_normal(count)


def _late(signal, lag):
    """`signal` delayed by `lag` samples, cut to its own length."""
    return np.concatenate((np.zeros(lag), signal))[: len(signal)]


def _old_align_lag(reference, estimate):
    """The aligner the harness used until 2026-10-11, copied verbatim from `score_reference._align`."""
    span = min(len(reference), len(estimate), 10 * 1024 * 8)
    correlation = scipy.signal.correlate(estimate[:span], reference[:span], mode="full")
    lag = int(np.argmax(np.abs(correlation))) - (span - 1)
    return max(-4096, min(4096, lag))


def test_a_lag_behind_a_silent_lead_in_is_found_where_the_head_search_fails():
    """6 s of hiss the restoration removed, then programme: the old search reads the empty head, the vote reads the programme."""
    source = _speechlike(20.0)
    source[: 6 * RATE] = 1e-3 * np.random.default_rng(9).standard_normal(6 * RATE)
    output = _late(source, MUX) * 0.5
    output[: 6 * RATE + MUX] = 0.0
    alignment = pa.gcc_alignment(source, output, RATE)
    assert (alignment.lag, alignment.status, alignment.polarity) == (MUX, pa.STATUS_OK, 1)
    assert alignment.vote.agree >= 4
    assert pa.legacy_alignment(source, output).lag != MUX


def test_an_inverted_output_reads_its_polarity():
    source = _speechlike(12.0)
    alignment = pa.gcc_alignment(source, -0.7 * _late(source, MUX), RATE)
    assert (alignment.lag, alignment.polarity) == (MUX, -1)
    assert alignment.vote.coherence > 0.5
    assert pa.legacy_alignment(source, -_late(source, MUX)).polarity == -1


def test_a_gain_ridden_output_like_dynamic_loudnorm_keeps_its_lag():
    """+-9 dB at 0.3 Hz on top of a 12 dB ramp: the phase alone decides the peak."""
    source = _speechlike(30.0, seed=11)
    times = np.arange(len(source)) / RATE
    gain = 10.0 ** ((9.0 * np.sin(2.0 * np.pi * 0.3 * times) + 12.0 * times / times[-1]) / 20.0)
    alignment = pa.gcc_alignment(source, _late(source * gain, MUX), RATE)
    assert (alignment.lag, alignment.status) == (MUX, pa.STATUS_OK)


def test_an_unrelated_output_makes_the_aligner_abstain():
    source = _speechlike(15.0, seed=1)
    stranger = _speechlike(15.0, seed=2)
    alignment = pa.gcc_alignment(source, stranger, RATE)
    assert (alignment.lag, alignment.applied_lag, alignment.status, alignment.reason) == (None, 0, pa.STATUS_ABSTAINED, "weak correlation")
    assert alignment.polarity is None
    assert pa.gcc_alignment(source, stranger, RATE, prior_lag=MUX).applied_lag == MUX


def test_spans_that_disagree_make_the_aligner_abstain():
    """Every 3 s carries its own lag (a warp a constant shift cannot undo): no lag wins half the spans."""
    source = _speechlike(21.0, seed=4)
    piece = 3 * RATE
    lags = np.repeat(100 + 600 * np.arange(7), piece)
    output = source[np.maximum(np.arange(7 * piece) - lags, 0)]
    alignment = pa.gcc_alignment(source, output, RATE)
    assert (alignment.status, alignment.reason, alignment.lag) == (pa.STATUS_ABSTAINED, "spans disagree", None)
    assert alignment.vote.spans == 7


def test_silence_and_a_too_short_pair_abstain_with_their_reason():
    silent = pa.gcc_alignment(np.zeros(5 * RATE), np.zeros(5 * RATE), RATE)
    short = pa.gcc_alignment(np.ones(1000), np.ones(1000), RATE)
    assert (silent.reason, silent.vote.spans) == ("no programme", 0)
    assert (short.reason, short.applied_lag) == ("too short", 0)


def test_a_pair_shorter_than_one_span_is_read_on_that_span():
    source = _speechlike(1.0, seed=6)
    alignment = pa.gcc_alignment(source, _late(source, 37), RATE)
    assert (alignment.lag, alignment.vote.spans, alignment.vote.agree) == (37, 1, 1)


def test_the_prior_breaks_a_tie_between_equal_peaks():
    """Two equal echoes at 100 and 300 samples are two equal peaks: the prior picks one, the smaller shift wins without it."""
    source = _speechlike(8.0, seed=13)
    output = 0.5 * (_late(source, 100) + _late(source, 300))
    assert pa.gcc_alignment(source, output, RATE).lag == 100
    assert pa.gcc_alignment(source, output, RATE, prior_lag=290).lag == 300


def test_the_vote_breaks_a_tie_between_equally_supported_lags():
    peaks = [pa.SpanPeak(10, 0.5), pa.SpanPeak(11, 0.5), pa.SpanPeak(500, 0.5), pa.SpanPeak(501, 0.5)]
    assert [peak.lag for peak in pa.agreeing_peaks(peaks, None)] == [10, 11]
    assert [peak.lag for peak in pa.agreeing_peaks(peaks, 480)] == [500, 501]


def test_the_head_search_is_the_stored_one_to_the_sample():
    rng = np.random.default_rng(5)
    source = _speechlike(4.0, seed=8)
    cases = [(source, _late(source, 219)), (source, rng.standard_normal(len(source))), (np.zeros(5000), np.zeros(5000))]
    for reference, output in cases:
        assert pa.legacy_alignment(reference, output).lag == _old_align_lag(reference, output)
    assert pa.legacy_alignment(np.zeros(5000), np.zeros(5000)).lag == -4096


def test_the_method_comes_from_the_argument_then_the_environment(monkeypatch):
    monkeypatch.delenv(pa.METHOD_ENV, raising=False)
    assert pa.resolve_method() == pa.ALIGN_GCC == pa.DEFAULT_METHOD
    monkeypatch.setenv(pa.METHOD_ENV, pa.ALIGN_LEGACY)
    assert pa.resolve_method() == pa.ALIGN_LEGACY
    assert pa.resolve_method(pa.ALIGN_GCC) == pa.ALIGN_GCC
    with pytest.raises(ValueError, match="unknown alignment method"):
        pa.resolve_method("argmax")


def test_align_trims_both_sides_and_names_its_method(monkeypatch):
    monkeypatch.delenv(pa.METHOD_ENV, raising=False)
    source = _speechlike(5.0, seed=12)
    output = _late(source, MUX)
    reference, trimmed, alignment = pa.align(source, output, RATE)
    assert alignment.method == pa.ALIGN_GCC
    assert len(reference) == len(trimmed) == len(source) - MUX
    np.testing.assert_allclose(trimmed, reference)
    _ref, _out, legacy = pa.align(source, output, RATE, pa.ALIGN_LEGACY)
    assert (legacy.method, legacy.status, legacy.lag) == (pa.ALIGN_LEGACY, pa.STATUS_LEGACY, MUX)


def test_trim_drops_the_head_of_whichever_side_is_late():
    reference, output = pa.trim(np.arange(10), np.arange(8), -3)
    assert reference.tolist() == [3, 4, 5, 6, 7, 8, 9]
    assert output.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert pa.trim(np.arange(4), np.arange(4), 0)[1].tolist() == [0, 1, 2, 3]


def test_the_record_a_document_carries():
    record = pa.Alignment(pa.ALIGN_GCC, 219, 219, 1, pa.STATUS_OK, "", pa.Vote(7, 6, 0.9)).as_dict()
    assert record == {
        "method": pa.ALIGN_GCC,
        "lag_samples": 219,
        "applied_lag_samples": 219,
        "polarity": 1,
        "status": pa.STATUS_OK,
        "reason": "",
        "spans": 7,
        "agree": 6,
        "coherence": 0.9,
    }


def test_span_starts_take_the_loudest_spans_apart_and_skip_silence():
    signal = np.zeros(100)
    signal[60:70] = 3.0
    signal[10:20] = 2.0
    signal[30:40] = 1.0
    assert pa.span_starts(signal, 10, count=2) == [10, 60]
    assert pa.span_starts(signal, 10) == [10, 30, 60]
    assert pa.span_starts(signal, 200) == []
    assert pa.span_starts(np.zeros(100), 10) == []


def test_a_span_without_live_bins_reads_nothing():
    assert pa.span_peak(np.zeros(4096), np.zeros(4096), RATE) is None
    assert pa.span_length(48000) == 72000
    assert pa.MUX_DELAY_S * RATE == pytest.approx(MUX)
