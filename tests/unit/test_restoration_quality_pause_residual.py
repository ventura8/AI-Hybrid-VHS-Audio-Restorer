"""The residual left in the true pauses (R4): its level, shape and texture against the source's own floor."""

from functools import lru_cache

import numpy as np
import scipy.signal

from scripts.restoration_quality import pause_metrics

RATE = 44100
FRAME = int(pause_metrics.FRAME_S * RATE)
SCALE = 10.0 ** (-12.0 / 20.0)
SHAPE = ("gap_slope_db_oct", "gap_spread_db", "gap_hf_excess_db", "gap_lsd_db")


@lru_cache(maxsize=None)
def _parts(seconds=8.0, seed=3, period=1.0, on=0.5):
    """`(voice, hiss)`: bright harmonic bursts (`on` s every `period` s) and the white hiss under them."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    phases = rng.uniform(0.0, 2 * np.pi, 60)
    voice = sum(np.sin(2 * np.pi * 150.0 * k * t + phases[k]) / np.sqrt(k) for k in range(1, 60))
    ramp = np.hanning(int(0.05 * RATE))
    gate = np.convolve(((t % period) < on).astype(np.float64), ramp / ramp.sum(), mode="same")
    return 0.05 * voice * gate, 3e-3 * rng.standard_normal(len(t))


@lru_cache(maxsize=None)
def _tailed(tau=0.08, period=1.3, on=0.5, seconds=12.0):
    """`(voice, hiss)` whose bursts decay into the pauses with time constant `tau`: word tails in the gap class."""
    rng = np.random.default_rng(3)
    t = np.arange(int(seconds * RATE)) / RATE
    voice = sum(np.sin(2 * np.pi * 150.0 * k * t + rng.uniform(0.0, 2 * np.pi)) / np.sqrt(k) for k in range(1, 60))
    gate = ((t % period) < on).astype(np.float64)
    tail = np.exp(-np.where(gate > 0, 0.0, (t % period) - on) / tau) * (1.0 - gate)
    ramp = np.hanning(int(0.01 * RATE))
    return 0.05 * voice * np.convolve(gate + tail, ramp / ramp.sum(), mode="same"), 3e-3 * rng.standard_normal(len(t))


def _tilted(noise, corner_hz=2000.0):
    """+6 dB/oct above `corner_hz`: a residual leaning to the highs (cathar's hiss)."""
    freqs = np.fft.rfftfreq(len(noise), 1.0 / RATE)
    return np.fft.irfft(np.fft.rfft(noise) * np.maximum(freqs / corner_hz, 1.0), n=len(noise))


def _low_passed(noise, cut_hz=1000.0):
    """Only the lows left: a residual with no air (dull, dead)."""
    return scipy.signal.sosfiltfilt(scipy.signal.butter(8, cut_hz, fs=RATE, output="sos"), noise)


def _read(source, output, **kwargs):
    """The output side of every residual reading."""
    readings = pause_metrics.gap_residual_readings(
        np.asarray(source, dtype=np.float32), np.asarray(output, dtype=np.float32), RATE, **kwargs
    )
    return {name: value[1] for name, value in readings.items()}


def _with_residual(residual):
    """The fixture's source and an output that keeps the voice and replaces the hiss by `residual(hiss)`."""
    voice, noise = _parts()
    return voice + noise, voice + residual(noise)


def _never_speech(mono, _rate, frame):
    """A VAD that hears no speech at all: the mask falls back to the bare level class."""
    return np.zeros(len(mono) // frame, dtype=bool)


def test_the_identity_reads_zero_and_reports_its_coverage():
    """Every shape and texture reading is 0 on the source itself; the mask covers the fixture's pauses."""
    source, _output = _with_residual(lambda noise: noise)
    readings = pause_metrics.gap_residual_readings(source, source, RATE)
    assert all(readings[name] == (0.0, 0.0) for name in pause_metrics.RESIDUAL_NAMES[:-1])
    assert readings["gap_pause_s"][0] == readings["gap_pause_s"][1] > 0.5


def test_scaled_source_noise_reads_uniform():
    """The source's own hiss 12 dB down: 12 dB of attenuation and a shape identical to the source's floor."""
    out = _read(*_with_residual(lambda noise: SCALE * noise))
    assert abs(out["gap_atten_db"] - 12.0) < 0.1
    assert max(abs(out[name]) for name in SHAPE) < 0.05
    assert abs(out["gap_mod_dist_db"]) < 0.05
    assert abs(out["gap_island_kurt"]) < 0.05


def test_an_independent_copy_of_the_floor_reads_close_to_uniform():
    """A fresh realisation of the same hiss: estimation noise only, far under what a tilt or islands read."""
    fresh = np.random.default_rng(11).standard_normal(len(_parts()[1])) * 3e-3
    out = _read(*_with_residual(lambda _noise: SCALE * fresh))
    assert abs(out["gap_atten_db"] - 12.0) < 1.0
    assert abs(out["gap_slope_db_oct"]) < 0.5
    assert out["gap_lsd_db"] < 1.0
    assert out["gap_mod_dist_db"] < 4.0


def test_an_hf_tilted_residual_reads_positive_hf_excess_and_slope():
    """+6 dB/oct above 2 kHz: the residual leans to the highs relative to the source's floor."""
    out = _read(*_with_residual(lambda noise: SCALE * _tilted(noise)))
    assert out["gap_hf_excess_db"] > 6.0
    assert out["gap_slope_db_oct"] > 1.5
    assert out["gap_lsd_db"] > 3.0


def test_an_lf_only_residual_reads_negative_hf_excess_and_slope():
    """Only the lows left: the HF excess and the slope turn negative, the empty bands read the 60 dB floor."""
    out = _read(*_with_residual(lambda noise: SCALE * _low_passed(noise)))
    assert out["gap_hf_excess_db"] < -20.0
    assert out["gap_slope_db_oct"] < -5.0
    assert out["gap_atten_db"] > 12.0


def test_a_pause_emptied_to_silence_reads_attenuation_only():
    """Digital silence reads the 60 dB floor as attenuation and the source's shape and texture, not a blown-up distance."""
    out = _read(*_with_residual(np.zeros_like))
    assert abs(out["gap_atten_db"] + pause_metrics.RESIDUAL_FLOOR_DB) < 1e-6
    assert max(abs(out[name]) for name in (*SHAPE, "gap_mod_dist_db", "gap_island_kurt")) < 1e-6


def test_gaps_shorter_than_a_true_pause_read_none():
    """120 ms gaps between bursts are syllabic dips, not pauses: no run reaches 200 ms after the VAD's hangover."""
    voice, noise = _parts(period=0.42, on=0.3)
    assert set(_read(voice + noise, voice + SCALE * noise).values()) == {None}


def test_a_window_without_pauses_or_too_short_reads_none():
    """A held tone over hiss has no level contrast; half a second has too few frames."""
    voice, noise = _parts(period=1.0, on=1.01)
    short = voice[: RATE // 2] + noise[: RATE // 2]
    assert set(_read(voice + noise, voice + SCALE * noise).values()) == {None}
    assert set(_read(short, short).values()) == {None}


def test_word_tails_stay_out_of_the_mask():
    """Bursts decaying into the pauses: the bare level class reads the kept tails, the VAD keeps them out."""
    voice, noise = _tailed()
    source, output = voice + noise, voice + SCALE * noise
    with_vad, without = _read(source, output), _read(source, output, vad=_never_speech)
    assert abs(with_vad["gap_atten_db"] - 12.0) < 1.0
    assert without["gap_atten_db"] < with_vad["gap_atten_db"] - 2.0


def test_the_mask_holds_the_deep_frames_of_a_true_pause_and_never_a_loud_one():
    """The true pauses sit at the source's deepest levels: the mask keeps them, inside runs of at least 200 ms."""
    voice, noise = _parts()
    source = voice + noise
    mask, runs = pause_metrics.true_pause_mask(source, RATE)
    _freqs, _power, level = pause_metrics.framed_psd(source, RATE, FRAME)
    deep, _gap, loud = pause_metrics.level_classes(20.0 * np.log10(level + 1e-9))
    assert (mask & deep).sum() > 0.5 * deep.sum()
    assert not (mask & loud).any()
    assert min(stop - start for start, stop in pause_metrics.run_spans(runs)) >= int(pause_metrics.MIN_RUN_S / pause_metrics.FRAME_S)


def test_the_capture_bandwidth_limits_the_bands():
    """A 3 kHz capture has no 4-10 kHz band to compare; a 150 Hz one has too few bands to read at all."""
    source, output = _with_residual(lambda noise: SCALE * noise)
    narrow = _read(source, output, bandwidth_hz=3000.0)
    assert narrow["gap_hf_excess_db"] is None
    assert abs(narrow["gap_atten_db"] - 12.0) < 0.1
    assert set(_read(source, output, bandwidth_hz=150.0).values()) == {None}


def test_top_hz_takes_the_lowest_limit():
    """The bands stop at the capture's bandwidth, 12 kHz or Nyquist."""
    assert pause_metrics.top_hz(44100) == 12000.0
    assert pause_metrics.top_hz(16000) == 8000.0
    assert pause_metrics.top_hz(44100, 7000.0) == 7000.0


def _speech_but_200_ms_of_each_pause(mono, _rate, frame):
    """A VAD that calls everything speech but frames 35-44 of each second: one 200 ms run per pause."""
    index = np.arange(len(mono) // frame) % 50
    return (index < 35) | (index >= 45)


def _all_speech(mono, _rate, frame):
    return np.ones(len(mono) // frame, dtype=bool)


def test_the_readings_take_any_vad_callable():
    """The mask runs on the VAD the caller passes: 200 ms per pause holds less than the DSP VAD's 1.98 s."""
    out = _read(*_with_residual(lambda noise: SCALE * noise), vad=_speech_but_200_ms_of_each_pause)
    assert abs(out["gap_atten_db"] - 12.0) < 0.1
    assert 0.5 <= out["gap_pause_s"] <= 8 * pause_metrics.MIN_RUN_S


def test_a_vad_that_hears_only_speech_leaves_nothing_to_read():
    """No non-speech run at all: every reading is None, whatever the level classes say."""
    assert set(_read(*_with_residual(lambda noise: SCALE * noise), vad=_all_speech).values()) == {None}
