"""The transient readings hear a drum hit's edge, its timing, and how much of the hit is left."""

import numpy as np
import scipy.signal

from scripts.restoration_quality import transient_metrics

RATE = 44100
HIT_S = 0.03
PERIOD_S = 0.4
FIRST_HIT_S = 0.2


def _fixture(seconds=8.0, seed=5, hiss=3e-3):
    """`(bed, hits, spans)`: a 220 Hz tone over tape-like hiss (-50 dBFS) and a train of bandpassed bursts with instant attacks."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    bed = 0.05 * sum(np.sin(2 * np.pi * 220.0 * k * t) / k for k in range(1, 6)) + hiss * rng.standard_normal(len(t))
    sos = scipy.signal.butter(4, [1000.0, 6000.0], btype="bandpass", fs=RATE, output="sos")
    burst = int(HIT_S * RATE)
    decay = np.exp(-np.arange(burst) / (0.008 * RATE))
    hits, spans = np.zeros(len(t)), []
    for start in np.arange(FIRST_HIT_S, seconds - HIT_S, PERIOD_S):
        lo, hi, settled = int(start * RATE), int(start * RATE) + burst, 2 * burst
        hits[lo:hi] = 0.3 * scipy.signal.sosfilt(sos, rng.standard_normal(3 * burst))[settled:] * decay
        spans.append((lo, hi))
    return bed, hits, spans


def _mix(bed, hits, spans=(), shape=None):
    """`bed + hits` as float32, each span of the hits multiplied by `shape(length)` first."""
    shaped = np.array(hits)
    for lo, hi in spans:
        shaped[lo:hi] *= shape(hi - lo)
    return (bed + shaped).astype(np.float32)


def _ramp(length):
    """The first `length` samples of a 60 ms raised-cosine rise from 0 to 1."""
    span = int(0.06 * RATE)
    return (0.5 * (1.0 - np.cos(np.pi * np.arange(span) / span)))[:length]


def test_identity_reads_the_same_on_both_sides_and_a_steep_attack():
    bed, hits, _spans = _fixture()
    source = _mix(bed, hits)
    readings = transient_metrics.transient_readings(source, source, RATE)
    for name in transient_metrics.NAMES:
        src, out = readings[name]
        assert name == "onset_corr" or abs(src - out) < 1e-9
    assert abs(readings["onset_corr"][1] - 1.0) < 1e-6
    assert readings["attack_db"][0] > 10.0


def test_onset_samples_land_on_the_hits():
    bed, hits, spans = _fixture()
    onsets = transient_metrics.onset_samples(_mix(bed, hits), RATE)
    assert 16 <= len(onsets) <= 24
    tolerance = int(0.015 * RATE)
    near = [np.min(np.abs(onsets - lo)) <= tolerance for lo, _hi in spans]
    assert np.mean(near) >= 0.8


def test_softened_attacks_read_lower_and_less_correlated():
    bed, hits, spans = _fixture()
    source = _mix(bed, hits)
    readings = transient_metrics.transient_readings(source, _mix(bed, hits, spans, _ramp), RATE)
    attack_src, attack_out = readings["attack_db"]
    assert attack_out < attack_src - 6.0
    assert readings["onset_corr"][1] < 0.95


def test_too_few_onsets_or_too_little_signal_reads_none():
    nones = {name: (None, None) for name in transient_metrics.NAMES}
    bed, hits, _spans = _fixture(hiss=0.0)
    tone = _mix(bed, np.zeros_like(hits))
    assert transient_metrics.transient_readings(tone, tone, RATE) == nones
    short = _mix(bed, hits)[: int(0.1 * RATE)]
    assert transient_metrics.transient_readings(short, short, RATE) == nones


def test_hpss_parts_add_back_up_to_the_input():
    bed, hits, _spans = _fixture()
    source = _mix(bed, hits)
    harmonic, percussive = transient_metrics.hpss(source, RATE)
    assert len(harmonic) == len(source)
    assert len(percussive) == len(source)
    error = harmonic + percussive - source.astype(np.float64)
    assert np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(source.astype(np.float64) ** 2)) < 0.2


def test_attenuated_hits_read_a_smaller_percussive_share():
    bed, hits, spans = _fixture()
    source = _mix(bed, hits)
    quieter = _mix(bed, hits, spans, lambda length: np.full(length, 0.25))
    share_src, share_out = transient_metrics.transient_readings(source, quieter, RATE)["percussive_share_db"]
    assert share_out < share_src
