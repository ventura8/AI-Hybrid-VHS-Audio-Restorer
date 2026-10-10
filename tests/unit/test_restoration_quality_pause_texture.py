"""The residual's texture and the tones left out of it: islands against pumping, hum, a notch, the rate bands."""

from functools import lru_cache

import numpy as np
import pytest
import scipy.signal

from scripts.restoration_quality import pause_metrics

RATE = 44100
SCALE = 10.0 ** (-12.0 / 20.0)


@lru_cache(maxsize=None)
def _scene(rate=RATE, mains_hz=0.0):
    """`(voice, hiss, hum)`: harmonic bursts 0.5 s of every 1 s, white hiss, and 7 mains harmonics at 1e-2/k (none at 0 Hz)."""
    rng = np.random.default_rng(3)
    t = np.arange(8 * rate) / rate
    voice = sum(np.sin(2 * np.pi * 150.0 * k * t + rng.uniform(0.0, 2 * np.pi)) / np.sqrt(k) for k in range(1, 60))
    ramp = np.hanning(int(0.05 * rate))
    gate = np.convolve(((t % 1.0) < 0.5).astype(np.float64), ramp / ramp.sum(), mode="same")
    hum = sum(1e-2 / k * np.sin(2 * np.pi * mains_hz * k * t) for k in range(1, 8))
    return 0.05 * voice * gate, 3e-3 * rng.standard_normal(len(t)), hum


def _read(source, output, rate=RATE):
    """The output side of every residual reading."""
    readings = pause_metrics.gap_residual_readings(np.asarray(source, dtype=np.float32), np.asarray(output, dtype=np.float32), rate)
    return {name: value[1] for name, value in readings.items()}


def _with_residual(residual):
    """The scene's source and an output that keeps the voice and replaces the hiss by `residual(hiss)`."""
    voice, noise, _hum = _scene()
    return voice + noise, voice + residual(noise)


def _islands(noise, keep=0.05, seed=5):
    """Spectral-subtraction islands: 5% of 256-point STFT cells kept, the rest zeroed, power held."""
    _freqs, _times, cells = scipy.signal.stft(noise, fs=RATE, nperseg=256)
    kept = np.random.default_rng(seed).random(cells.shape) < keep
    _times, rebuilt = scipy.signal.istft(cells * kept / np.sqrt(keep), fs=RATE, nperseg=256)
    return rebuilt[: len(noise)]


def _gated(noise, on_s=0.15):
    """The source's own hiss switched on and off every `on_s` seconds: pumping, with no island anywhere."""
    t = np.arange(len(noise)) / RATE
    return noise * ((t % (2 * on_s)) < on_s)


def _band_stopped(noise, band_hz=(1800.0, 2200.0)):
    """A notch in the residual: its shape departs from the source's floor without any tilt."""
    return scipy.signal.sosfiltfilt(scipy.signal.butter(4, band_hz, "bandstop", fs=RATE, output="sos"), noise)


def test_musical_noise_islands_raise_both_texture_readings():
    """Islands flicker from frame to frame and leave each frame sparse: modulation and island kurtosis both rise."""
    out = _read(*_with_residual(lambda noise: SCALE * _islands(noise)))
    assert out["gap_mod_dist_db"] > 8.0
    assert out["gap_island_kurt"] > 0.5
    assert abs(out["gap_atten_db"] - 12.0) < 4.0


def test_a_gated_residual_reads_as_pumping_not_islands():
    """150 ms on, 150 ms off: the modulation rises as for islands, the island kurtosis stays at the source's."""
    out = _read(*_with_residual(lambda noise: SCALE * _gated(noise)))
    assert out["gap_mod_dist_db"] > 5.0
    assert abs(out["gap_island_kurt"]) < 0.2
    assert max(abs(out["gap_slope_db_oct"]), abs(out["gap_hf_excess_db"])) < 0.5


@pytest.mark.parametrize(("rate", "mains_hz"), [(48000, 50.0), (44100, 60.0)])
def test_removing_mains_hum_is_not_a_shaped_residual(rate, mains_hz):
    """Hum in the source, only the scaled hiss in the output: the tones leave both sides, the shape reads flat."""
    voice, noise, hum = _scene(rate, mains_hz)
    out = _read(voice + noise + hum, voice + SCALE * noise, rate)
    assert max(abs(out["gap_hf_excess_db"]), abs(out["gap_slope_db_oct"]), out["gap_lsd_db"]) < 0.2
    assert abs(out["gap_atten_db"] - 12.0) < 0.1
    assert out["gap_mod_dist_db"] < 0.2


def test_a_notched_residual_reads_spread_without_slope():
    """A 1.8-2.2 kHz band stop: the spread and the LSD see the notch, the slope and the median level do not."""
    out = _read(*_with_residual(lambda noise: SCALE * _band_stopped(noise)))
    assert out["gap_spread_db"] > 3.0
    assert out["gap_lsd_db"] > 3.0
    assert abs(out["gap_slope_db_oct"]) < 0.5
    assert abs(out["gap_atten_db"] - 12.0) < 0.5


def test_a_non_finite_sample_reads_none():
    """One NaN in the output's pauses: every reading None, never a NaN for the scorecard to compare."""
    source, output = _with_residual(lambda noise: SCALE * noise)
    output = output.copy()
    output[int(0.75 * RATE)] = np.nan
    assert set(_read(source, output).values()) == {None}


def test_tonal_bins_take_a_steady_tone_and_its_neighbours_but_not_noise():
    """A 1 kHz tone 41 dB over noise on 20 ms frames: bins 19-21 (its Hann main lobe) and two more either side, no noise bin."""
    rng = np.random.default_rng(1)
    t = np.arange(40 * 882) / RATE
    frames = (1e-3 * rng.standard_normal(len(t)) + 1e-2 * np.sin(2 * np.pi * 1000.0 * t)).reshape(40, 882)
    power = np.abs(np.fft.rfft(frames * np.hanning(882), axis=1)) ** 2
    assert list(np.flatnonzero(pause_metrics.tonal_bins(power))) == list(range(17, 24))


def test_a_rate_band_reads_only_spans_two_periods_long():
    """At a 100 Hz envelope rate 0.2 s reaches 16-32 Hz only, 0.5 s down to 4 Hz, 1 s every band."""
    weights = pause_metrics.rate_weights([20, 50, 100], 100.0)
    assert list(weights[0] > 0) == [False, False, False, True]
    assert list(weights[1] > 0) == [False, True, True, True]
    assert list(weights[2] > 0) == [True, True, True, True]


def test_an_unreached_rate_band_reads_nan_and_drops_out_of_the_distance():
    """Two short envelopes: the 2-8 Hz bands read NaN; the distance stands on the bands they reach."""
    rng = np.random.default_rng(2)
    envelopes = [rng.exponential(size=(25, 2)), rng.exponential(size=(30, 2))]
    spectrum = pause_metrics.modulation_spectrum_db(envelopes, 100.0)
    assert np.isnan(spectrum[:, :2]).all()
    assert np.isfinite(spectrum[:, 2:]).all()


def test_the_modulation_reads_none_when_every_bin_is_left_out():
    """No bin kept (all tones, or out of range): no envelope to read."""
    voice, noise, _hum = _scene()
    freqs = np.fft.rfftfreq(882, 1.0 / RATE)
    spans = [(int(0.6 * RATE), int(0.9 * RATE))]
    bins = (freqs, np.zeros(len(freqs), dtype=bool), 12000.0)
    assert pause_metrics.modulation_distance_db((voice + noise, voice + noise), RATE, spans, bins) is None
