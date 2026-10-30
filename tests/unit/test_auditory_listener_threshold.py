"""A calibrated listener's threshold shape moves the audibility check's floor in quiet, frequency by frequency (2026-10-23)."""

import numpy as np
import pytest

from scripts.restoration_quality import auditory, playback_calibration

RATE = 48000


def _record(offsets, full_scale=100.0):
    """A calibration record body: `offsets` maps Hz to `spl_offset_db`."""
    thresholds = [{"hz": hz, "spl_offset_db": offset, "censored": False} for hz, offset in offsets.items()]
    return {"spl_full_scale_db": full_scale, "thresholds": thresholds}


def test_the_shift_is_the_placed_level_minus_each_frequency_s_offset_sorted_by_frequency():
    shift = playback_calibration.threshold_shift(_record({8000.0: 90.0, 1000.0: 100.0, 4000.0: 101.0}))
    assert shift == ((1000.0, 4000.0, 8000.0), (0.0, -1.0, 10.0))


@pytest.mark.parametrize("record", [None, {"spl_full_scale_db": None, "thresholds": []}])
def test_without_a_record_or_a_level_there_is_no_shift(record):
    assert playback_calibration.threshold_shift(record) == ((), ())


def _shifted(monkeypatch, hz=(1000.0, 8000.0), db=(0.0, 20.0)):
    monkeypatch.setattr(auditory, "LISTENER_SHIFT_HZ", hz)
    monkeypatch.setattr(auditory, "LISTENER_SHIFT_DB", db)


def test_without_a_calibration_the_listener_s_threshold_is_terhardt_s():
    freqs = np.array([500.0, 4000.0, 12000.0])
    assert np.array_equal(auditory.listener_threshold_db(freqs), auditory.threshold_in_quiet_db(freqs))


def test_the_shift_is_read_over_log_frequency_and_held_past_the_ends(monkeypatch):
    _shifted(monkeypatch)
    freqs = np.array([200.0, 1000.0, np.sqrt(1000.0 * 8000.0), 8000.0, 14000.0])
    moved = auditory.listener_threshold_db(freqs) - auditory.threshold_in_quiet_db(freqs)
    assert np.allclose(moved, [0.0, 0.0, 10.0, 20.0, 20.0])


def _quiet_air(seconds=2.0):
    """A soft programme and the same with a little 9-11 kHz noise added: an air-band change near the threshold in quiet."""
    rng = np.random.default_rng(3)
    voice = rng.standard_normal(int(seconds * RATE)) * 0.05
    spectrum = np.fft.rfft(rng.standard_normal(len(voice)))
    freqs = np.fft.rfftfreq(len(voice), 1.0 / RATE)
    air = np.fft.irfft(np.where((freqs > 9000.0) & (freqs < 11000.0), spectrum, 0.0), n=len(voice))
    return voice, voice + air * 0.0004


def test_a_listener_who_hears_less_above_8_khz_reads_a_quiet_air_change_lower(monkeypatch):
    voice, aired = _quiet_air()
    plain = auditory.audibility(voice, aired, RATE)["nmr_max"]
    _shifted(monkeypatch, db=(0.0, 30.0))
    assert auditory.audibility(voice, aired, RATE)["nmr_max"] < plain


def test_the_hiss_readings_curve_stays_terhardt_s_under_a_calibration(monkeypatch):
    _shifted(monkeypatch)
    terhardt = 3.64 * 10.0**-0.8 - 6.5 * np.exp(-0.6 * (10.0 - 3.3) ** 2) + 1e-3 * 10.0**4
    assert float(auditory.threshold_in_quiet_db(10000.0)) == pytest.approx(terhardt)
