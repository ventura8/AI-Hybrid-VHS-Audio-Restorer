"""The audibility check on a speech-like signal: what it must call audible, what a tie, and why."""

import numpy as np
import pytest
import scipy.signal

from scripts.restoration_quality import auditory

RATE = 44100


def _bandpass(noise, low, high):
    sos = scipy.signal.butter(4, [low, high], btype="bandpass", fs=RATE, output="sos")
    return scipy.signal.sosfiltfilt(sos, noise)


def _speech(seconds=6.0, seed=7):
    """Speech-like: 0.3 s vowels (160 Hz buzz) and 120 ms 's' bursts (1-10 kHz noise) every 0.6 s over a faint hiss."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    phases = rng.uniform(0.0, 2 * np.pi, 30)
    vowel = sum(np.sin(2 * np.pi * 160.0 * k * t + phases[k]) / k for k in range(1, 30))
    ramp = np.hanning(int(0.01 * RATE))
    vowel_gate = np.convolve(((t % 0.6) < 0.3).astype(np.float64), ramp / ramp.sum(), mode="same")
    ess_gate = np.convolve((((t % 0.6) >= 0.36) & ((t % 0.6) < 0.48)).astype(np.float64), ramp / ramp.sum(), mode="same")
    ess = 0.5 * _bandpass(rng.standard_normal(len(t)), 4000.0, 10000.0) + 0.35 * _bandpass(rng.standard_normal(len(t)), 1000.0, 4000.0)
    return (0.1 * vowel * vowel_gate + 0.1 * ess * ess_gate + 1e-4 * rng.standard_normal(len(t))).astype(np.float32)


def _shelf(mono, gain_db, corner_hz=4000.0):
    """`mono` with everything above `corner_hz` raised by `gain_db` (zero phase)."""
    sos = scipy.signal.butter(4, corner_hz, btype="highpass", fs=RATE, output="sos")
    return (mono + (10.0 ** (gain_db / 20.0) - 1.0) * scipy.signal.sosfiltfilt(sos, np.asarray(mono, dtype=np.float64))).astype(np.float32)


def _with_noise(mono, level_dbfs, seed=1):
    noise = 10.0 ** (level_dbfs / 20.0) * np.random.default_rng(seed).standard_normal(len(mono))
    return (mono + noise).astype(np.float32)


def test_identical_signals_read_no_difference():
    """Nothing moved: the difference sits on the -200 dB floor and no frame is audible."""
    voice = _speech()
    result = auditory.audibility(voice, voice, RATE)
    assert result["diff_db"] == auditory.DB_FLOOR
    assert result["changed_share"] == 0.0
    assert result["nmr_max"] == pytest.approx(auditory.DB_FLOOR)
    assert result["audible"] is False


def test_a_one_sample_shift_is_aligned_away_and_inaudible():
    """The benign anchor of plan 1.2: the lag takes the shift out, so nothing is left to hear."""
    voice = _speech()
    shifted = np.concatenate([np.zeros(1, dtype=np.float32), voice[:-1]])
    result = auditory.audibility(voice, shifted, RATE)
    assert (result["lag"], result["polarity"]) == (1, 1)
    assert result["audible_frac"] == 0.0
    assert result["audible"] is False


def test_a_2_db_shelf_above_4_khz_on_speech_is_audible():
    """The air-shelf complaint in miniature: +2 dB above 4 kHz moves every 's' burst past the threshold."""
    voice = _speech()
    result = auditory.audibility(voice, _shelf(voice, 2.0), RATE)
    assert result["audible"] is True
    assert result["nmr_max"] > 6.0
    assert result["audible_frac"] > 0.5


def test_a_difference_at_minus_90_dbfs_is_inaudible_though_every_sample_changed():
    """The null test sees every sample move; the NMR keeps it under the threshold in quiet, even in the pauses."""
    voice = _speech()
    result = auditory.audibility(voice, _with_noise(voice, -90.0), RATE)
    assert result["changed_share"] > 0.9
    assert result["nmr_max"] < -6.0
    assert result["audible"] is False


def test_noise_at_minus_60_dbfs_is_audible_in_the_pauses():
    """30 dB more noise clears the threshold in quiet where the incumbent is silent."""
    voice = _speech()
    result = auditory.audibility(voice, _with_noise(voice, -60.0), RATE)
    assert result["audible"] is True
    assert result["diff_db"] < -30.0


def test_an_inverted_candidate_is_aligned_by_its_polarity():
    """A polarity flip alone is not a difference a listener hears on one channel."""
    voice = _speech()
    result = auditory.audibility(voice, -voice, RATE)
    assert result["polarity"] == -1
    assert result["audible"] is False


def test_the_offset_puts_a_half_db_level_change_at_the_threshold():
    """At the 24 dB offset a broadband +1 dB reads audible and +0.4 dB a tie."""
    voice = _speech()
    assert auditory.audibility(voice, voice * np.float32(10.0 ** (1.0 / 20.0)), RATE)["audible"] is True
    assert auditory.audibility(voice, voice * np.float32(10.0 ** (0.4 / 20.0)), RATE)["audible"] is False


def test_a_smaller_masking_offset_calls_a_1_db_shelf_a_tie():
    """The offset is the knob that moves the verdict: at 12 dB a 1 dB shelf hides under the masker."""
    voice = _speech()
    shelved = _shelf(voice, 1.0)
    assert auditory.audibility(voice, shelved, RATE)["audible"] is True
    assert auditory.audibility(voice, shelved, RATE, offset_db=12.0)["audible"] is False


def test_a_change_in_one_channel_reads_audible_and_a_dual_mono_copy_reads_nothing():
    """The worst channel decides; a stereo copy of the mono voice is the voice on every channel."""
    voice = _speech()
    stereo = np.stack([voice, voice], axis=1)
    changed = stereo.copy()
    changed[:, 1] = _shelf(voice, 2.0)
    assert auditory.audibility(stereo, changed, RATE)["audible"] is True
    dual_mono = auditory.audibility(stereo, voice, RATE)
    assert (dual_mono["audible_frac"], dual_mono["audible"], dual_mono["channel_mismatch"]) == (0.0, False, False)


def test_a_wide_stereo_pair_against_its_mono_downmix_reads_audible():
    """Collapsing two different channels to their mean is an image change, not a tie: the mono side is compared on each."""
    wide = np.stack([_speech(seed=7), _speech(seed=11)], axis=1)
    result = auditory.audibility(wide, wide.mean(axis=1), RATE)
    assert result["audible"] is True
    assert result["nmr_max"] > 20.0


def test_two_multichannel_counts_read_audible_and_flagged():
    """Stereo against three channels cannot be vouched for: the downmixes match, yet the pair is no tie."""
    voice = _speech()
    result = auditory.audibility(np.stack([voice] * 2, axis=1), np.stack([voice] * 3, axis=1), RATE)
    assert result["channel_mismatch"] is True
    assert result["audible_frac"] == 0.0
    assert result["audible"] is True


def test_a_change_in_the_second_window_shows_in_that_window_only():
    """Per-window shares tell where the change is; 31 s folds into two windows (the 1 s tail joins the second)."""
    voice = _speech(seconds=31.0)
    changed, start = voice.copy(), 20 * RATE
    changed[start:] = _shelf(voice[start:], 2.0)
    fractions = auditory.audibility(voice, changed, RATE)["window_audible_frac"]
    assert len(fractions) == 2
    assert fractions[0] < auditory.TIE_AUDIBLE_FRAC <= fractions[1]


def test_silence_and_empty_input_do_not_break_the_check():
    """Silence aligns at lag 0; noise against silence is audible; empty input reads the floors."""
    silence = np.zeros(RATE, dtype=np.float32)
    assert auditory.estimate_lag(silence, silence, RATE) == (0, 1)
    assert auditory.audibility(silence, _with_noise(silence, -60.0), RATE)["audible"] is True
    empty = auditory.audibility(np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), RATE)
    assert (empty["diff_db"], empty["window_audible_frac"]) == (auditory.DB_FLOOR, [])
