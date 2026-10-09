"""R4 leaves out the pause bins whose floor sits under 16-bit requantisation: benign requantising moves no reading."""

import numpy as np
import pytest

from scripts import quality_degradations as deg
from scripts.restoration_quality import pause_metrics
from scripts.restoration_quality.dsp_metrics import framed_psd
from tests.unit.test_restoration_quality_pause_residual import RATE, _parts, _read

NOTCH_HZ = (10500.0, 12500.0)
NOTCH_DB = 60.0


def _notched(noise, band=NOTCH_HZ, depth_db=NOTCH_DB):
    """`noise` with `band` taken `depth_db` down: Vaccin's 11 kHz notch, its floor sunk under the 16-bit dither."""
    freqs = np.fft.rfftfreq(len(noise), 1.0 / RATE)
    gain = np.where((freqs >= band[0]) & (freqs < band[1]), 10.0 ** (-depth_db / 20.0), 1.0)
    return np.fft.irfft(np.fft.rfft(noise) * gain, n=len(noise))


def _requantised_pair():
    """Bursts over hiss notched 60 dB at 10.5-12.5 kHz, and the same signal requantised to 16 bits with TPDF dither."""
    voice, noise = _parts()
    source = (voice + _notched(noise)).astype(np.float32)
    return source, deg.benign_requantise(source, np.random.default_rng(4))


def test_requantising_a_notched_floor_moves_no_shape_reading():
    """The dither lifts the notch's top ERB band by about 14 dB; with those bins out, slope and LSD stay at 0."""
    out = _read(*_requantised_pair())
    assert abs(out["gap_slope_db_oct"]) < 0.02
    assert out["gap_lsd_db"] < 0.05


def test_without_the_dither_floor_the_requantised_notch_reads_as_hiss(monkeypatch):
    """Every bin kept, the same pair reads the shape change the calibration's Vaccin requantise case read (slope +0.39, LSD +2.5)."""
    monkeypatch.setattr(pause_metrics, "DITHER_MARGIN_DB", -1000.0)
    out = _read(*_requantised_pair())
    assert out["gap_slope_db_oct"] > 0.2
    assert out["gap_lsd_db"] > 1.0


@pytest.mark.parametrize("rate", [44100, 48000])
def test_the_dither_floor_is_the_power_of_requantised_silence(rate):
    """The floor matches the mean per-bin power of 20 s of requantised zeros within 0.5 dB."""
    frame = int(pause_metrics.FRAME_S * rate)
    zeros = deg.benign_requantise(np.zeros(20 * rate, dtype=np.float32), np.random.default_rng(1))
    _freqs, power, _level = framed_psd(zeros, rate, frame)
    assert abs(10.0 * np.log10(power[:, 1:-1].mean() / pause_metrics.dither_floor(frame))) < 0.5


def test_above_dither_keeps_hiss_and_drops_a_bin_held_near_silence():
    """3e-3 white hiss stands about 46 dB over the floor; a bin held at 1e-7 power sits under it plus 10 dB."""
    frame = int(pause_metrics.FRAME_S * RATE)
    hiss = 3e-3 * np.random.default_rng(2).standard_normal(40 * frame)
    _freqs, power, _level = framed_psd(hiss, RATE, frame)
    assert pause_metrics.above_dither(power, frame).all()
    power[:, 100] = 1e-7
    kept = pause_metrics.above_dither(power, frame)
    assert not kept[100]
    assert kept.sum() == power.shape[1] - 1


def test_the_frame_defaults_to_the_even_frame_of_the_bins():
    """Without a frame the bins say it: 442 rfft bins are an 882-sample frame."""
    frame = int(pause_metrics.FRAME_S * RATE)
    freqs = np.fft.rfftfreq(frame, 1.0 / RATE)
    power = np.full((30, len(freqs)), 1e-3) * np.random.default_rng(5).exponential(size=(30, len(freqs)))
    top = pause_metrics.top_hz(RATE)
    assert np.array_equal(pause_metrics.residual_bins(freqs, power, top), pause_metrics.residual_bins(freqs, power, top, frame))
