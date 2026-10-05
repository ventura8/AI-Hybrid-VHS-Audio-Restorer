"""Tonal persistence and syllabic modulation: the readings that tell music from speech."""

import numpy as np

import modules.auto_scanner
from modules import tonal_persistence as tp

RATE = 44100


def _chord(seconds=15.0):
    t = np.arange(int(seconds * RATE)) / RATE
    # A harmonic series: a held tone whose envelope does not beat (a chord's beating reads as syllables).
    return sum(0.2 / k * np.sin(2 * np.pi * 110.0 * k * t) for k in (1, 2, 3, 4, 5)).astype(np.float32)


def _syllables(seconds=15.0):
    rng = np.random.default_rng(5)
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.3 * rng.standard_normal(len(t)) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 4.0 * t)))).astype(np.float32)


def test_tonal_persistence_is_high_on_a_held_chord_and_nil_on_syllabic_noise():
    assert tp.tonal_persistence(_chord(), RATE) > 0.05
    assert tp.tonal_persistence(_syllables(), RATE) < 0.004


def test_tonal_persistence_needs_more_frames_than_the_hold():
    assert tp.tonal_persistence(np.zeros(2048, dtype=np.float32), RATE) == 0.0
    assert tp.tonal_persistence(np.zeros(RATE, dtype=np.float32), RATE) == 0.0


def test_held_share_counts_only_peaks_present_in_every_following_frame():
    peaks = np.zeros((3, 12), dtype=bool)
    peaks[0, :] = True
    peaks[1, :5] = True
    assert tp._held_share(peaks, 8) == 4 / 8
    assert tp._held_share(peaks[:, :8], 8) == 0.0


def test_syllabic_modulation_reads_the_four_hertz_bursts_and_nothing_on_a_steady_envelope():
    assert tp.syllabic_modulation(_syllables(), RATE) > 0.2
    assert tp.syllabic_modulation(_chord(), RATE) < 0.2
    assert tp.syllabic_modulation(np.zeros(RATE // 2, dtype=np.float32), RATE) == 0.0


def test_median_persistence_reads_the_programme_windows_and_leaves_silence_out():
    music = np.concatenate([np.zeros(15 * RATE, dtype=np.float32), _chord(), _chord()])
    assert tp.median_persistence(music, RATE) > 0.05
    assert tp.median_persistence(np.zeros(20 * RATE, dtype=np.float32), RATE) == 0.0
    assert tp.median_persistence(np.zeros(100, dtype=np.float32), RATE) == 0.0


def test_the_scanner_profile_carries_the_reading():
    profile = modules.auto_scanner._extract_profile_from_signal(_chord(4.0), RATE)
    assert profile["tonal_persistence"] == modules.auto_scanner._estimate_tonal_persistence(_chord(4.0), RATE)
    assert profile["tonal_persistence"] > 0.05
