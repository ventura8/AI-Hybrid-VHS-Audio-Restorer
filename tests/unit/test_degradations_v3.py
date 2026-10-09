"""The ear v3 degradation generators: each changes the signal the way it claims, and only where it claims."""

import functools

import numpy as np
import pytest
import scipy.signal
import scipy.stats

from scripts import degradations_v3 as v3

RATE = 48000
# FFmpeg 8.0.1 `treble=g=2:f=7500` on a 0.5 impulse at 48 kHz, divided by 0.5: its first six taps.
FFMPEG_TREBLE_TAPS = (1.168119, -0.136427, -0.052591, -0.000256, 0.012725, 0.008214)


@functools.lru_cache(maxsize=None)
def _noise(seconds=2.0, seed=3):
    return (0.1 * np.random.default_rng(seed).standard_normal(int(seconds * RATE))).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _bursts(seconds=4.0):
    """Noise bursts (0.4 s on, 0.6 s off) over a faint floor: speech-like frames and long pauses."""
    t = np.arange(int(seconds * RATE)) / RATE
    gate = (t % 1.0) < 0.4
    rng = np.random.default_rng(5)
    return (0.1 * rng.standard_normal(len(t)) * gate + 1e-3 * rng.standard_normal(len(t))).astype(np.float32), gate


def _band_db(mono, low, high):
    freqs, psd = scipy.signal.welch(mono, RATE, nperseg=4096)
    return float(10.0 * np.log10(psd[(freqs >= low) & (freqs < high)].mean() + 1e-30))


def _gain_db(out, ref, low, high):
    return _band_db(out, low, high) - _band_db(ref, low, high)


def test_the_air_shelf_is_ffmpegs_treble_filter():
    """The coefficients reproduce FFmpeg's own impulse response (measured) and the shelf's half gain at 7.5 kHz."""
    impulse = np.zeros(64)
    impulse[0] = 1.0
    b, a = v3.treble_coefficients(2.0, RATE)
    assert np.allclose(scipy.signal.lfilter(b, a, impulse)[:6], FFMPEG_TREBLE_TAPS, atol=2e-6)
    _w, response = scipy.signal.freqz(b, a, worN=[100.0, 7500.0, 20000.0], fs=RATE)
    assert np.allclose(20.0 * np.log10(np.abs(response)), [0.0, 1.0, 1.99], atol=0.02)


def test_the_air_shelf_lifts_and_cuts_the_top_only():
    """The air shelf lifts and cuts the top only."""
    noise = _noise()
    lifted, cut = v3.air_shelf(noise, RATE, 2.0), v3.air_shelf(noise, RATE, -2.0)
    assert abs(_gain_db(lifted, noise, 200.0, 1000.0)) < 0.05
    assert 1.8 < _gain_db(lifted, noise, 14000.0, 20000.0) < 2.1
    assert -2.1 < _gain_db(cut, noise, 14000.0, 20000.0) < -1.8


def test_the_tilt_is_its_slope_per_octave_above_one_kilohertz():
    """The tilt is its slope per octave above one kilohertz."""
    noise = _noise()
    tilted = v3.spectral_tilt(noise, RATE, 1.5)
    assert abs(_gain_db(tilted, noise, 300.0, 900.0)) < 0.05
    assert abs(_gain_db(tilted, noise, 3900.0, 4100.0) - 3.0) < 0.2
    assert abs(_gain_db(tilted, noise, 7800.0, 8200.0) - 4.5) < 0.2


def test_the_tilt_gain_is_flat_below_the_pivot():
    """The tilt gain is flat below the pivot."""
    assert np.allclose(v3.tilt_gain_db([100.0, 1000.0, 2000.0, 8000.0], 2.0), [0.0, 0.0, 2.0, 6.0])


def test_the_pause_weight_is_zero_on_speech_and_near_it_and_one_deep_in_the_pauses():
    """The pause weight is zero on speech and near it and one deep in the pauses."""
    speech = np.zeros(RATE, dtype=bool)
    speech[: RATE // 4] = True
    weight = v3.pause_weight(speech, RATE)
    closed = RATE // 4 + int(v3.PAUSE_HANGOVER_S * RATE)
    opened = closed + int(v3.PAUSE_RAMP_S * RATE)
    assert np.all(weight[:closed] == 0.0)
    assert np.all(weight[opened:] == 1.0)
    assert np.all(np.diff(weight) >= 0.0)


def test_a_signal_without_speech_is_all_pause():
    """A signal without speech is all pause."""
    assert np.all(v3.pause_weight(np.zeros(100, dtype=bool), RATE) == 1.0)


def test_the_residual_lands_in_the_pauses_only():
    """Where the weight is 0 the output is the input bit for bit; deep in a pause it is the residual."""
    mono, gate = _bursts()
    weight = v3.pause_weight(gate, RATE)
    out = v3.in_pauses(mono, weight, v3.scaled_residual(mono, -12.0))
    assert np.array_equal(out[weight == 0.0], mono[weight == 0.0])
    assert np.allclose(out[weight == 1.0], mono[weight == 1.0] * 10.0 ** (-12.0 / 20.0), atol=1e-7)


def test_the_hiss_residual_leans_to_the_highs():
    """The hiss residual leans to the highs."""
    noise = _noise()
    hissy = v3.hiss_residual(noise, RATE, 6.0)
    assert abs(_gain_db(hissy, noise, 300.0, 1500.0) + 12.0) < 0.1
    assert abs(_gain_db(hissy, noise, 7800.0, 8200.0) - 0.0) < 0.3


def test_the_dull_residual_loses_its_highs():
    """The dull residual loses its highs."""
    noise = _noise()
    dull = v3.dull_residual(noise, RATE, 2000.0)
    assert abs(_gain_db(dull, noise, 200.0, 1000.0) + 12.0) < 0.2
    assert _gain_db(dull, noise, 6000.0, 10000.0) < -60.0


def test_the_island_residual_nearly_holds_the_power_and_breaks_the_floor():
    """Kept cells carry the dropped cells' power; the overlap-add of incoherent frames still loses a little of it."""
    noise = _noise()
    islands = v3.island_residual(noise, RATE, 0.1, np.random.default_rng(2))
    assert len(islands) == len(noise)
    assert -15.0 < _ratio_db(islands, noise) < -12.0
    assert scipy.stats.kurtosis(islands) > scipy.stats.kurtosis(noise) + 0.2


def test_the_follower_rises_with_its_attack_and_falls_with_its_release():
    """The follower rises with its attack and falls with its release."""
    step = np.concatenate([np.zeros(100), np.ones(1000), np.zeros(1000)])
    out = v3.follow(step, 1000.0, 0.01, 0.1)
    assert out[109] == pytest.approx(1.0 - np.exp(-1.0), rel=0.05)
    assert out[1199] == pytest.approx(out[1099] * np.exp(-1.0), rel=0.05)


def test_the_mistracking_expander_lowers_the_quiet_stretches_by_up_to_its_error():
    """Bursts keep their level; the pauses drop by up to the error (the 70 ms release lifts their start)."""
    mono, gate = _bursts()
    out = v3.compander_mistrack(mono, RATE, 6.0)
    loud, quiet = gate.copy(), ~gate
    loud[: RATE // 10], quiet[: RATE // 10] = False, False
    assert abs(_ratio_db(out[loud], mono[loud])) < 1.0
    assert -6.0 <= _ratio_db(out[quiet], mono[quiet]) < -2.5


def _ratio_db(out, ref):
    return float(10.0 * np.log10(np.mean(np.asarray(out, dtype=np.float64) ** 2) / np.mean(np.asarray(ref, dtype=np.float64) ** 2)))


def test_the_sidechain_hears_the_highs_twenty_decibels_louder():
    """The sidechain hears the highs twenty decibels louder."""
    low = np.sin(2 * np.pi * 200.0 * np.arange(RATE) / RATE)
    high = np.sin(2 * np.pi * 15000.0 * np.arange(RATE) / RATE)
    _centres, low_db = v3.sidechain_level_db(low, RATE)
    _centres, high_db = v3.sidechain_level_db(high, RATE)
    assert 17.0 < np.median(high_db) - np.median(low_db) < 21.0


def _steps():
    """A 12 s signal in four 3 s steps at -30, -18, -30 and -18 dBFS."""
    levels = np.repeat(10.0 ** (np.array([-30.0, -18.0, -30.0, -18.0]) / 20.0), 3 * RATE)
    return (levels * _noise(12.0, 9) * 10.0).astype(np.float32)


def test_the_rider_at_depth_zero_is_linear_mode():
    """The rider at depth zero is linear mode."""
    steps = _steps()
    assert np.allclose(v3.loudnorm_ride(steps, RATE, 0.0), steps)


def test_the_rider_narrows_the_level_swing_with_its_depth():
    """The rider narrows the level swing with its depth."""
    steps = _steps()
    swing = [_swing(v3.loudnorm_ride(steps, RATE, depth)) for depth in (0.0, 0.5, 0.9)]
    assert swing[0] > swing[1] > swing[2]
    assert swing[0] == pytest.approx(12.0, abs=0.5)


def _swing(mono):
    loud_start = 4 * RATE
    quiet, loud = mono[RATE:][:RATE], mono[loud_start:][:RATE]
    return _ratio_db(loud, quiet)


def test_the_rider_holds_its_gain_through_silence():
    """The rider holds its gain through silence."""
    steps = _steps()
    silent_start, silent_length = 6 * RATE, 3 * RATE
    steps[silent_start:][:silent_length] = 0.0
    centres, level = v3.short_term_level_db(steps, RATE)
    gain = v3.ride_gain_db(level, 1.0)
    assert len(gain) == len(centres)
    assert np.all(np.abs(gain) <= v3.RIDE_MAX_DB)


def test_phase_resynthesis_keeps_the_length_and_moves_the_waveform_with_the_amount():
    """Phase resynthesis keeps the length and moves the waveform with the amount."""
    noise = _noise()
    differences = [np.abs(v3.phasey_resynth(noise, RATE, amount, np.random.default_rng(1)) - noise).mean() for amount in (0.0, 0.3, 1.0)]
    assert len(v3.phasey_resynth(noise, RATE, 0.3, np.random.default_rng(1))) == len(noise)
    assert differences[0] < 1e-6
    assert differences[0] < differences[1] < differences[2]


def test_the_band_limit_cuts_above_its_corner():
    """The band limit cuts above its corner."""
    noise = _noise()
    limited = v3.band_limited(noise, RATE, 5000.0)
    assert abs(_gain_db(limited, noise, 500.0, 3000.0)) < 0.1
    assert _gain_db(limited, noise, 8000.0, 12000.0) < -60.0


def test_wallace_spacing_loss_at_pal_sp_linear_speed():
    """Wallace spacing loss at pal sp linear speed."""
    loss = v3.spacing_loss_db([1000.0, 8000.0], 1.0)
    assert loss == pytest.approx([-2.33, -18.67], abs=0.02)


def test_dropout_events_are_fifty_to_three_hundred_milliseconds_long():
    """Dropout events are fifty to three hundred milliseconds long."""
    weight = v3.dropout_weight(10 * RATE, RATE, np.random.default_rng(4))
    runs = np.flatnonzero(np.diff(np.concatenate([[0], (weight >= 0.5).astype(int), [0]])))
    lengths = (runs[1::2] - runs[::2]) / RATE
    assert 1 <= len(lengths) <= round(v3.DROPOUT_RATE_PER_S * 10)
    assert np.all((lengths >= 0.045) & (lengths <= 0.6))


def test_treble_dropouts_take_the_treble_inside_the_events_only():
    """Treble dropouts take the treble inside the events only."""
    noise = _noise(4.0)
    rng_state = np.random.default_rng(6)
    out = v3.treble_dropouts(noise, RATE, 2.0, rng_state)
    weight = v3.dropout_weight(len(noise), RATE, np.random.default_rng(6))
    assert np.array_equal(out[weight == 0.0], noise[weight == 0.0])
    inside = out[weight == 1.0]
    assert _band_db(inside, 8000.0, 12000.0) < _band_db(noise[weight == 1.0], 8000.0, 12000.0) - 30.0


@pytest.mark.parametrize("length", [1023, 1024, 4097])
def test_the_stft_generators_return_the_input_length(length):
    """The stft generators return the input length."""
    noise = _noise()[:length]
    assert len(v3.island_residual(noise, RATE, 0.5, np.random.default_rng(1))) == length
    assert len(v3.phasey_resynth(noise, RATE, 0.5, np.random.default_rng(1))) == length


def test_sibilant_islands_change_the_marked_frames_only():
    """Away from the marked 's' (one 256-point frame of overlap aside) the signal is kept; inside, the top is broken."""
    noise = _noise()
    half = RATE // 2
    fricative = np.zeros(len(noise))
    fricative[half:][:half] = 1.0
    out = v3.sibilant_islands(noise, RATE, 0.5, fricative, np.random.default_rng(2))
    kept = half - 256
    assert np.allclose(out[:kept], noise[:kept], atol=1e-6)
    marked, before = out[half:][:half], noise[half:][:half]
    assert not np.allclose(marked, before, atol=1e-3)
    assert abs(_gain_db(marked, before, 4000.0, 12000.0)) < 3.0 and abs(_gain_db(marked, before, 300.0, 3000.0)) < 0.1


def test_speed_drift_makes_the_output_longer_by_its_percentage():
    """0.2 % slow on 2 s at 48 kHz: 192 samples more."""
    noise = _noise()
    assert len(v3.speed_drift(noise, 0.2)) - len(noise) == 192
    assert len(v3.speed_drift(noise, 0.0)) == len(noise)
