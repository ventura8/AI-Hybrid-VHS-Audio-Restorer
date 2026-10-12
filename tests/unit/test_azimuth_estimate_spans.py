"""azimuth_estimate_method: the pre-conditioning's azimuth delay read from the head ('head') or the loudest spans ('spans').

'head' (the default) cross-correlates the first 32768 samples and takes the integer peak, with a sign the
graph does not correct. The readings and graphs pinned below were recorded on these fixtures with HEAD
57bfdb8's modules/filters.py (the PAL graph and a full scan of the phasey-intro pair): the default key must
rebuild them exactly, so every output keeps its bytes. 'spans' votes GCC-PHAT peaks on the loudest spans
(modules/azimuth_spans.py), reads a fraction of a sample, abstains when the spans disagree or cohere weakly,
keeps the correlation gate and the inverted-pair rule, and returns the sign the graph corrects.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

import modules.config as cfg
from modules import auto_scanner, azimuth_spans, filters
from scripts import tune_restoration as tune

RATE = 44100
PAL = {"highpass_hz": 80, "notch_hz": 50.0, "crt_notch_hz": 15625.0}
PAL_TAIL = (
    "highpass=f=80,adeclick,bandreject=f=50.0:width_type=q:w=15,bandreject=f=100.0:width_type=q:w=15,bandreject=f=15625.0:width_type=q:w=30"
)
# HEAD 57bfdb8: the head reading of each fixture and the PAL pre-conditioning graph it builds.
HEAD_READINGS = {
    "phasey_intro": (-0.05, "adelay=0|2S," + PAL_TAIL),
    "skew_0p3ms": (-0.29, "adelay=0|13S," + PAL_TAIL),
    "skew_left_late": (0.29, "adelay=13S|0," + PAL_TAIL),
    "inverted_skew": (0.0, PAL_TAIL),
    "in_phase": (0.0, PAL_TAIL),
}
# HEAD 57bfdb8: the full scan of the phasey-intro pair (scan_and_decide_restoration_strategy).
HEAD_SCAN_GRAPH = "adelay=0|2S,bandreject=f=15734.0:width_type=q:w=30"
TRUE_SKEW_MS = 13.23 / RATE * 1000.0


def _band(low, high, count, rng):
    """Gaussian noise band-passed to low..high Hz."""
    sos = scipy.signal.butter(4, (low, high), btype="bandpass", fs=RATE, output="sos")
    return scipy.signal.sosfilt(sos, rng.standard_normal(count))


def _programme(seconds, seed, level=0.3):
    """Speech-like programme: a strong 80-500 Hz body and a 500-6000 Hz presence 20 dB under it, at a 3 Hz syllable rate."""
    rng = np.random.default_rng(seed)
    count = int(RATE * seconds)
    body = _band(80.0, 500.0, count, rng)
    presence = _band(500.0, 6000.0, count, rng)
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 3.0 * np.arange(count) / RATE)
    return level * (body + 0.1 * presence) * envelope


def _delayed(signal, samples):
    """`signal` delayed by a whole or fractional number of samples (a linear phase over its spectrum)."""
    freqs = np.fft.rfftfreq(len(signal))
    return np.fft.irfft(np.fft.rfft(signal) * np.exp(-2j * np.pi * freqs * samples), len(signal))


def _hiss(count, seed, level=0.001):
    """Independent tape hiss for one channel."""
    return level * np.random.default_rng(seed).standard_normal(count)


def _pair(left, right):
    """Two channels as the float32 stereo array the scan reads, each with its own hiss."""
    count = len(left)
    return np.stack([left + _hiss(count, 101), right + _hiss(count, 202)], axis=1).astype(np.float32)


def _phasey_intro_pair():
    """A lag-0 programme whose first 0.8 s, a quieter jingle, runs 2 samples late on the right."""
    intro = _programme(0.8, 1, level=0.1)
    body = _programme(9.2, 2)
    return _pair(np.concatenate([intro, body]), np.concatenate([_delayed(intro, 2.0), body]))


def _skewed_pair(samples=13.23, polarity=1.0):
    """A right channel `samples` late (13.23 is 0.3 ms at 44.1 kHz), inverted when `polarity` is -1."""
    left = _programme(10.0, 3)
    return _pair(left, polarity * _delayed(left, samples))


def _in_phase_pair():
    """The same programme on both channels."""
    left = _programme(10.0, 4)
    return _pair(left, left)


FIXTURES = {
    "phasey_intro": _phasey_intro_pair,
    "skew_0p3ms": _skewed_pair,
    "skew_left_late": lambda: _skewed_pair(-13.23),
    "inverted_skew": lambda: _skewed_pair(13.23, -1.0),
    "in_phase": _in_phase_pair,
}


def _graph(delay_ms):
    """The PAL pre-conditioning graph with this azimuth delay."""
    return filters._precondition_filter_from_config({**PAL, "azimuth_delay_ms": delay_ms})


@pytest.fixture(name="spans_log")
def _spans_log():
    """The 'spans' reading, the inverted-pair switch off, and the scan's log captured."""
    with (
        patch("modules.filters.AZIMUTH_ESTIMATE_METHOD", "spans"),
        patch("modules.filters.AZIMUTH_REJECT_INVERTED_PAIR", False),
        patch("modules.filters.log_msg") as log,
    ):
        yield log


def _logged(log):
    """Every line the scan logged, as one string."""
    return " | ".join(str(call.args[0]) for call in log.call_args_list)


def test_the_key_defaults_to_head():
    """The resolved config, its module constant and the scan's copy all hold every release's reading."""
    assert cfg.CONFIG["azimuth_estimate_method"] == "head"
    assert cfg.AZIMUTH_ESTIMATE_METHOD == "head"
    assert filters.AZIMUTH_ESTIMATE_METHOD == "head"


def _loaded(user):
    """The configuration load_config resolves from a config.yaml holding `user`, and the warning printer."""
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = user
    with (
        patch("modules.config.Path.exists", return_value=True),
        patch.object(cfg, "yaml", mock_yaml),
        patch("builtins.open", MagicMock()),
        patch("builtins.print") as warn,
    ):
        conf, _source = cfg.load_config()
    return conf, warn


@pytest.mark.parametrize(
    "user, expected, warned",
    [({}, "head", False), ({"azimuth_estimate_method": " Spans "}, "spans", False), ({"azimuth_estimate_method": "gcc"}, "head", True)],
)
def test_the_key_is_a_choice(user, expected, warned):
    """An older config.yaml gets 'head' quietly; case and spaces are forgiven; an unknown value warns and keeps 'head'."""
    conf, warn = _loaded({"process_mode": "auto_pure_linear", **user})
    assert conf["azimuth_estimate_method"] == expected
    assert warn.called is warned


def test_the_repository_config_and_the_tuning_driver_know_the_key():
    """config.yaml ships 'head'; a tuning grid may set the key, and an unknown value is a problem there."""
    text = (Path(cfg.__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
    assert '\nazimuth_estimate_method: "head"\n' in text
    fields = tune.known_config_fields()
    assert fields["azimuth_estimate_method"][0] == {"head", "spans"}
    assert tune._problem("azimuth_estimate_method", "gcc", fields["azimuth_estimate_method"])


@pytest.mark.parametrize("name", sorted(HEAD_READINGS))
def test_the_default_reading_and_graph_are_heads(name):
    """The default key reads each fixture as HEAD 57bfdb8 did and builds the same graph, byte for byte."""
    expected_reading, expected_graph = HEAD_READINGS[name]
    with patch("modules.filters.log_msg"):
        reading = filters._detect_stereo_azimuth_skew(FIXTURES[name](), RATE)
    assert reading == pytest.approx(expected_reading, abs=1e-12)
    assert _graph(reading) == expected_graph


def test_a_scan_with_the_default_key_builds_heads_graph(tmp_path):
    """The whole scan of the phasey-intro pair builds HEAD 57bfdb8's pre-conditioning graph, its head delay included."""
    wav = tmp_path / "phasey.wav"
    sf.write(str(wav), _phasey_intro_pair(), RATE, subtype="FLOAT")
    with patch("modules.auto_scanner.log_msg"), patch("modules.filters.log_msg"):
        strategy = auto_scanner.scan_and_decide_restoration_strategy(wav)
    assert filters._precondition_filter_from_config(strategy["precondition_filters"]) == HEAD_SCAN_GRAPH


def test_a_lag0_pair_with_a_phasey_intro_reads_no_delay_with_spans(spans_log):
    """The head reads the intro's 2 samples (-0.05 ms, pinned above); the loudest spans read 0, so the graph delays nothing."""
    reading = filters._detect_stereo_azimuth_skew(_phasey_intro_pair(), RATE)
    assert reading == 0.0
    assert "adelay" not in _graph(reading)
    assert "no delay" in _logged(spans_log)


@pytest.mark.parametrize("samples, token", [(13.23, "adelay=13S|0"), (-13.23, "adelay=0|13S")])
def test_a_real_skew_is_read_to_a_fraction_of_a_sample_and_corrected(spans_log, samples, token):
    """A 0.3 ms skew reads 0.3 ms with the sign the graph corrects: a late right channel delays the left, and back."""
    reading = filters._detect_stereo_azimuth_skew(_skewed_pair(samples), RATE)
    assert reading == pytest.approx(np.sign(samples) * TRUE_SKEW_MS, abs=0.002)
    assert _graph(reading).startswith(token + ",")
    assert "13.2" in _logged(spans_log)


def test_the_delay_puts_the_channels_back_on_one_timeline(spans_log):
    """Delaying the left channel by the samples the reading gives leaves the pair within half a sample of aligned."""
    stereo = _skewed_pair()
    delay = round(filters._detect_stereo_azimuth_skew(stereo, RATE) * filters.AZIMUTH_SAMPLE_RATE / 1000)
    left = np.concatenate([np.zeros(delay), stereo[:-delay, 0]])
    residual = azimuth_spans.read_skew(left, stereo[:, 1].astype(np.float64), RATE, 44)
    assert abs(residual.lag) < 0.5
    spans_log.assert_called()


def test_an_inverted_pair_is_flagged_and_read_by_default(spans_log):
    """Switch off: the pair is logged as inverted, as the head does, and its true skew is read through the magnitude of the peak."""
    reading = filters._detect_stereo_azimuth_skew(_skewed_pair(13.23, -1.0), RATE)
    assert reading == pytest.approx(TRUE_SKEW_MS, abs=0.002)
    assert "polarity-inverted (r=-0.71)" in _logged(spans_log)
    assert "unreliable" in _logged(spans_log)


def test_an_inverted_pair_is_refused_behind_the_switch(spans_log):
    """Switch on: no delay, the refusal logged, and the vote never runs."""
    with patch("modules.filters.AZIMUTH_REJECT_INVERTED_PAIR", True), patch("modules.azimuth_spans.read_skew") as vote:
        reading = filters._detect_stereo_azimuth_skew(_skewed_pair(13.23, -1.0), RATE)
    assert reading == 0.0
    assert "no azimuth delay read from it" in _logged(spans_log)
    vote.assert_not_called()


@pytest.mark.parametrize("right", [np.zeros(RATE * 4), _hiss(RATE * 4, 7, level=0.1)])
def test_dead_or_unrelated_channels_fail_the_gate_with_spans(spans_log, right):
    """A dead channel or an unrelated one fails the |r| gate on the whole scanned audio: no delay, no vote."""
    stereo = np.stack([_programme(4.0, 5), right], axis=1)
    with patch("modules.azimuth_spans.read_skew") as vote:
        reading = filters._detect_stereo_azimuth_skew(stereo, RATE)
    assert reading == 0.0
    vote.assert_not_called()
    spans_log.assert_not_called()


def _disagreeing_pair():
    """Seven 1.5 s bursts between hiss, each with the right channel late by a different 4-sample step."""
    gap = np.zeros(int(1.5 * RATE))
    left, right = [], []
    for index in range(7):
        burst = _programme(1.5, 10 + index)
        left += [burst, gap]
        right += [_delayed(burst, 4.0 * index), gap]
    return _pair(np.concatenate(left), np.concatenate(right))


def test_spans_that_disagree_abstain(spans_log):
    """No lag holds half the counted spans: the reading abstains and the graph delays nothing."""
    reading = filters._detect_stereo_azimuth_skew(_disagreeing_pair(), RATE)
    assert reading == 0.0
    assert "spans disagree" in _logged(spans_log)


def test_weak_coherence_abstains():
    """Channels sharing only their bass pass the |r| gate, but their 150-7000 Hz phase is unrelated: no span counts."""
    rng = np.random.default_rng(8)
    count = RATE * 6
    bass = scipy.signal.sosfilt(scipy.signal.butter(6, 100.0, fs=RATE, output="sos"), rng.standard_normal(count))
    left, right = bass + 0.02 * rng.standard_normal(count), bass + 0.02 * rng.standard_normal(count)
    reading = azimuth_spans.read_skew(left, right, RATE, 44)
    assert reading.lag is None
    assert reading.reason == "weak coherence"


@pytest.mark.parametrize(
    "left, reason",
    [
        (np.ones(1000), "too short"),
        (np.concatenate([np.full(5000, np.nan), np.ones(5000)]), "non-finite samples"),
        (np.zeros(RATE * 3), "no programme"),
    ],
)
def test_audio_the_vote_cannot_read_abstains(left, reason):
    """Too short for a span, holding a NaN, or silent: no lag, and the reason says which."""
    reading = azimuth_spans.read_skew(left, left.copy(), RATE, 44)
    assert reading.lag is None
    assert reading.reason == reason


@pytest.mark.parametrize("lag, expected", [(None, 0.0), (0.49, 0.0), (44.3, 0.0), (-45.0, 0.0), (13.23, 0.3), (-0.6, -0.014)])
def test_only_a_lag_the_graph_can_apply_becomes_a_delay(lag, expected):
    """Abstained, under half a sample or outside the 44-sample window: no delay; otherwise the lag in ms to 0.001 ms."""
    assert azimuth_spans.delay_ms(lag, RATE, 44) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "reading, delay_ms, expected",
    [
        (azimuth_spans.SkewReading(None, 7, 2, reason="spans disagree"), 0.0, "no delay (spans disagree; 2 of 7 spans agree)"),
        (azimuth_spans.SkewReading(13.22, 6, 6, 0.98), 0.3, "delay +0.300 ms (6 of 6 spans agree on +13.22 samples, coherence 0.98)"),
        (azimuth_spans.SkewReading(0.2, 6, 5, 0.9), 0.0, "no delay (5 of 6 spans agree on +0.20 samples, coherence 0.90)"),
    ],
)
def test_the_log_line_says_what_the_vote_gave(reading, delay_ms, expected):
    assert azimuth_spans.describe(reading, delay_ms) == expected


@pytest.mark.parametrize(
    "before, peak, after, expected",
    [(0.5, 1.0, 0.5, 0.0), (1.0, 1.0, 1.0, 0.0), (0.0, 1.0, 1.0, 0.5), (0.9, 1.0, 0.1, -0.4), (1.0, 1.0, 0.0, -0.5)],
)
def test_the_parabola_refines_within_half_a_sample(before, peak, after, expected):
    """A symmetric peak stays put, a flat or valley-shaped one is not refined, and the offset never passes half a sample."""
    assert azimuth_spans.parabolic_offset(before, peak, after) == pytest.approx(expected)


def test_a_peak_on_the_window_edge_stays_whole():
    lags = np.arange(-3, 4)
    heights = np.array([1.0, 0.5, 0.2, 0.1, 0.2, 0.5, 0.9])
    assert azimuth_spans.refined_lag(lags, heights, 0) == -3.0
    assert azimuth_spans.refined_lag(lags, heights, 6) == 3.0


def test_a_rate_too_low_for_the_band_reads_no_bins():
    """At 300 Hz the band (150 Hz up to 0.45 of the rate) is empty: the span is not read."""
    span = np.random.default_rng(9).standard_normal(4096)
    assert azimuth_spans.span_peak(span, span, 300, 10) is None


@pytest.mark.parametrize("count, spans", [(7, 0), (3, 0), (8, 1), (13, 2)])
def test_the_span_grid_steps_by_half_a_span(count, spans):
    """Spans of 8 samples start every 4: audio shorter than two hops holds none."""
    hop, energies = azimuth_spans.span_energies(np.ones(count), 8)
    assert hop == 4
    assert energies.size == spans


@pytest.mark.parametrize("counted, spans, needed", [(1, 1, 1), (1, 7, 2), (3, 7, 2), (7, 7, 4)])
def test_the_vote_needs_half_the_counted_spans_and_two(counted, spans, needed):
    assert azimuth_spans.needed_agreement(counted, spans) == needed


def test_a_tied_vote_goes_to_the_smaller_shift():
    """Two lags with equal support: the one nearer zero wins."""
    peaks = [azimuth_spans.SpanPeak(lag, 0.9) for lag in (3.0, 3.2, -1.0, -1.1)]
    assert [peak.lag for peak in azimuth_spans.agreeing_peaks(peaks)] == [-1.0, -1.1]
