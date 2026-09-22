"""The sibilance readings find the 's' bursts and read a thinned or dulled 's' the right way."""

import numpy as np
import scipy.signal

from scripts.restoration_quality import sibilance

RATE = 44100


def _bandpass(noise, low, high):
    sos = scipy.signal.butter(4, [low, high], btype="bandpass", fs=RATE, output="sos")
    return scipy.signal.sosfiltfilt(sos, noise)


def _voice_with_esses(seconds=6.0, seed=7):
    """Vowel bursts (harmonic buzz, 0.3 s) alternating with 120 ms 's' bursts (1-12 kHz noise) over a faint hiss."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    phases = rng.uniform(0.0, 2 * np.pi, 30)
    vowel = sum(np.sin(2 * np.pi * 160.0 * k * t + phases[k]) / k for k in range(1, 30))
    vowel_gate = ((t % 0.6) < 0.3).astype(np.float64)
    ess_gate = ((t % 0.6) >= 0.36) & ((t % 0.6) < 0.48)
    ramp = np.hanning(int(0.01 * RATE))
    vowel_gate = np.convolve(vowel_gate, ramp / ramp.sum(), mode="same")
    ess_gate = np.convolve(ess_gate.astype(np.float64), ramp / ramp.sum(), mode="same")
    ess = 0.5 * _bandpass(rng.standard_normal(len(t)), 4000.0, 10000.0) + 0.35 * _bandpass(rng.standard_normal(len(t)), 1000.0, 4000.0)
    return (0.1 * vowel * vowel_gate + 0.1 * ess * ess_gate + 1e-4 * rng.standard_normal(len(t))).astype(np.float32), ess_gate > 0.5


def _thinned(source, ess_gate, depth_db=12.0):
    """The 1-4 kHz body under the 's' removed inside the bursts only."""
    body = _bandpass(np.asarray(source, dtype=np.float64), 1000.0, 4000.0)
    return (source - (1.0 - 10 ** (-depth_db / 20.0)) * body * ess_gate).astype(np.float32)


def _dulled(source, ess_gate, depth_db=12.0):
    """The 6-12 kHz top of the 's' attenuated inside the bursts only."""
    top = _bandpass(np.asarray(source, dtype=np.float64), 6000.0, 12000.0)
    return (source - (1.0 - 10 ** (-depth_db / 20.0)) * top * ess_gate).astype(np.float32)


def test_the_detector_finds_the_ess_bursts_and_not_the_vowels():
    source, ess_gate = _voice_with_esses()
    mask = sibilance.fricative_mask(source, RATE)
    hit = mask & ess_gate
    assert hit.sum() > 0.6 * mask.sum()
    assert mask.sum() > 0.4 * ess_gate.sum()
    assert sibilance.fricative_mask(np.zeros(100, dtype=np.float32), RATE).sum() == 0


def test_identity_reads_the_same_shape_on_both_sides():
    source, _gate = _voice_with_esses()
    readings = sibilance.sib_readings(source, source, RATE)
    for name in sibilance.NAMES:
        src, out = readings[name]
        assert abs(src - out) < 1e-6
    assert readings["sib_centroid_hz"][0] > sibilance.CENTROID_MIN_HZ


def test_a_thinned_ess_moves_the_centroid_up_and_the_body_ratio_up():
    source, ess_gate = _voice_with_esses()
    readings = sibilance.sib_readings(source, _thinned(source, ess_gate), RATE)
    centroid_src, centroid_out = readings["sib_centroid_hz"]
    body_src, body_out = readings["sib_body_db"]
    assert centroid_out - centroid_src > 300.0
    assert body_out - body_src > 3.0


def test_a_dulled_ess_moves_the_centroid_down():
    source, ess_gate = _voice_with_esses()
    readings = sibilance.sib_readings(source, _dulled(source, ess_gate), RATE)
    centroid_src, centroid_out = readings["sib_centroid_hz"]
    body_src, body_out = readings["sib_body_db"]
    assert centroid_out - centroid_src < -300.0
    assert body_out - body_src < -3.0


def test_too_few_fricatives_reads_none():
    short, _gate = _voice_with_esses(seconds=0.3)
    assert sibilance.sib_readings(short, short, RATE) == {name: (None, None) for name in sibilance.NAMES}
    t = np.arange(4 * RATE) / RATE
    tone = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    assert sibilance.sib_readings(tone, tone, RATE)["sib_centroid_hz"] == (None, None)
