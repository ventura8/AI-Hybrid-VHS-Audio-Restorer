"""Ear v4's same-percept transforms: the gain, the fractional delay, the soxr round trip, the fresh floor and the alias."""

import argparse

import numpy as np
import pytest
import scipy.signal

from scripts import reward_noise_floor as nf
from scripts import same_percept

RATE = 44100
LOW_RATE = 8000


def _programme_over_noise(seconds=4.0, rate=RATE, seed=2):
    """`(bursts, noise)`: a 1 kHz sine at 0.1 sounding 0.5 s in every second, over white noise at 1e-3.

    The programme must come and go: a steady tone's envelope sits at its own per-bin P20, so it reads noise-led and is
    redrawn as noise like the floor (hum included).
    """
    t = np.arange(int(seconds * rate)) / rate
    bursts = 0.1 * np.sin(2 * np.pi * 1000.0 * t) * ((t % 1.0) < 0.5)
    return bursts, 1e-3 * np.random.default_rng(seed).standard_normal(len(t))


def _spectrum(mono, rate=RATE):
    return scipy.signal.stft(mono, rate, nperseg=same_percept.STFT_FRAME, noverlap=same_percept.STFT_OVERLAP)


def test_the_gain_transforms_are_exact():
    """+-0.1 dB scales every sample by exactly 10^(+-0.1/20), channel by channel."""
    audio = np.stack(_programme_over_noise(1.0), axis=1)
    up, down = nf.gain_up_0_1(audio, RATE, None), nf.gain_down_0_1(audio, RATE, None)
    assert up == pytest.approx(audio * 10.0 ** (0.1 / 20.0))
    assert down == pytest.approx(audio * 10.0 ** (-0.1 / 20.0))


def test_the_fractional_delay_is_a_phase_ramp():
    """A 0.37-sample delay of a sine is the sine 0.37 samples later, away from the edges."""
    t = np.arange(RATE) / RATE
    delayed = nf.delay_0_37(np.sin(2 * np.pi * 440.0 * t)[:, np.newaxis], RATE, None)[:, 0]
    expected = np.sin(2 * np.pi * 440.0 * (t - 0.37 / RATE))
    assert np.max(np.abs(delayed[1000:-1000] - expected[1000:-1000])) < 1e-5


def test_the_soxr_round_trip_keeps_the_length_and_the_programme():
    """The round trip returns as many samples, the 1 kHz bursts within -60 dB."""
    sine, _noise = _programme_over_noise(1.0)
    trip = nf.resample_soxr(sine[:, np.newaxis], RATE, None)[:, 0]
    assert len(trip) == len(sine)
    assert 10.0 * np.log10(np.mean((trip[2000:-2000] - sine[2000:-2000]) ** 2) / np.mean(sine**2)) < -60.0


def test_a_short_round_trip_is_padded_back_to_its_length():
    """A round trip that comes back a sample short is padded with zeros."""
    assert len(same_percept.soxr_round_trip(np.ones(101), RATE, 48000)) == 101


def test_the_fresh_floor_redraws_the_noise_and_keeps_the_programme():
    """The bursts' cells keep their level (within 0.1 dB); the noise-led cells are redrawn (correlation under 0.5)."""
    sine, noise = _programme_over_noise()
    fresh = nf.fresh_floor_a((sine + noise)[:, np.newaxis], RATE, None)[:, 0]
    freqs, _times, before = _spectrum(sine + noise)
    _freqs, _times, after = _spectrum(fresh)
    tone, far = np.argmin(np.abs(freqs - 1000.0)), freqs > 5000.0
    level = 10.0 * np.log10(np.sum(np.abs(after[tone]) ** 2) / np.sum(np.abs(before[tone]) ** 2))
    correlation = np.corrcoef(np.abs(before[far]).ravel(), np.abs(after[far]).ravel())[0, 1]
    assert (abs(level) < 0.1, correlation < 0.5) == (True, True)


def test_the_fresh_floor_keeps_a_held_tone():
    """A steady tone sits at its own P20 like the floor, but stands over the bins around it: its cells are kept (0.1 dB)."""
    t = np.arange(int(4.0 * RATE)) / RATE
    tone = 0.01 * np.sin(2 * np.pi * 3000.0 * t)
    noise = 1e-3 * np.random.default_rng(4).standard_normal(len(t))
    fresh = nf.fresh_floor_b((tone + noise)[:, np.newaxis], RATE, None)[:, 0]
    freqs, _times, before = _spectrum(tone + noise)
    _freqs, _times, after = _spectrum(fresh)
    bin_ = np.argmin(np.abs(freqs - 3000.0))
    assert abs(10.0 * np.log10(np.sum(np.abs(after[bin_]) ** 2) / np.sum(np.abs(before[bin_]) ** 2))) < 0.1


def test_the_noise_led_test_leaves_out_a_spectral_peak():
    """A cell 10x over the bins around it is not noise-led even at its own P20; the flat cells are."""
    power = np.ones((41, 6))
    power[20] = 10.0
    led = same_percept.noise_led(power)
    assert (bool(led[20].any()), bool(led[0].all())) == (False, True)


def test_the_fresh_floor_is_deterministic_per_seed():
    """One seed gives one realisation; the three named transforms give three."""
    audio = np.stack(_programme_over_noise(2.0), axis=1).sum(axis=1)[:, np.newaxis]
    first, again = nf.fresh_floor_b(audio, RATE, None), nf.fresh_floor_b(audio, RATE, None)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, nf.fresh_floor_c(audio, RATE, None))


def test_a_long_floor_is_redrawn_block_by_block_with_crossfades():
    """65 s at 8 kHz takes three 30 s blocks; the length is kept and the level stays within 1 dB."""
    noise = 0.01 * np.random.default_rng(4).standard_normal(65 * LOW_RATE)
    fresh = same_percept.fresh_floor(noise, LOW_RATE, 9)
    assert len(fresh) == len(noise)
    assert abs(10.0 * np.log10(np.mean(fresh**2) / np.mean(noise**2))) < 1.0
    assert same_percept.block_starts(65 * LOW_RATE, 30 * LOW_RATE, LOW_RATE) == [0, 29 * LOW_RATE, 58 * LOW_RATE]


def _level_db(after, before):
    return 10.0 * np.log10(np.mean(after**2) / np.mean(before**2))


def test_the_fresh_floor_keeps_the_power_of_a_white_noise():
    """A stationary white noise comes back within 0.1 dB (the first build lost 0.32 dB) and as a new draw."""
    noise = 1e-3 * np.random.default_rng(1).standard_normal(4 * RATE)
    fresh = same_percept.fresh_floor(noise, RATE, 9)
    assert abs(_level_db(fresh, noise)) < 0.1
    assert abs(np.corrcoef(noise, fresh)[0, 1]) < 0.1


def test_the_fresh_floor_keeps_a_gated_floor_at_both_levels():
    """A noise 20 dB down every other half second keeps each level within 0.1 dB (44.1 kHz: 23 ms frames)."""
    noise = 1e-3 * np.random.default_rng(3).standard_normal(6 * RATE)
    gain = np.where((np.arange(len(noise)) % RATE) < RATE // 2, 0.1, 1.0)
    fresh = same_percept.fresh_floor(noise * gain, RATE, 5)
    quiet = gain < 0.5
    levels = (_level_db(fresh[quiet], (noise * gain)[quiet]), _level_db(fresh[~quiet], (noise * gain)[~quiet]))
    assert max(abs(level) for level in levels) < 0.1


def test_the_blocks_share_one_noise_through_the_crossfade():
    """The second the 30 s blocks overlap in keeps the floor within 0.2 dB (independent draws lost up to 3 dB there)."""
    noise = 0.01 * np.random.default_rng(2).standard_normal(65 * LOW_RATE)
    fresh = same_percept.fresh_floor(noise, LOW_RATE, 9)
    overlap = slice(29 * LOW_RATE, 30 * LOW_RATE)
    assert abs(_level_db(fresh[overlap], noise[overlap])) < 0.2


def test_the_redraw_share_fades_in_from_kept_cells():
    """A kept cell takes no fresh noise, a noise-led cell inside the floor all of it, one beside kept cells a part."""
    led = np.ones((9, 11), dtype=bool)
    led[3:6, 3:6] = False
    share = same_percept.redraw_share(led)
    assert (share[4, 4], share[0, 10]) == (0.0, 1.0)
    assert share[4, 6] == pytest.approx((19 / 25 - 0.5) / 0.4)


def test_the_crossfade_weights_sum_to_one_over_an_overlap():
    """A block's falling ramp and the next one's rising ramp add to 1; the file's own ends are not faded."""
    first, last = same_percept.crossfade_weight(10, 4, True, False), same_percept.crossfade_weight(10, 4, False, True)
    assert first[-4:] + last[:4] == pytest.approx(np.ones(4))
    assert (first[0], last[-1]) == (1.0, 1.0)


def test_fresh_like_redraws_every_cell_of_a_noise_track():
    """A noise track on its own keeps its level (within 1 dB) and loses its sample-by-sample draw."""
    noise = 0.01 * np.random.default_rng(6).standard_normal(3 * RATE)
    fresh = same_percept.fresh_like(noise, RATE, 7)
    assert abs(10.0 * np.log10(np.mean(fresh**2) / np.mean(noise**2))) < 1.0
    assert abs(np.corrcoef(noise, fresh)[0, 1]) < 0.1


def test_the_hiss_alias_names_the_same_percept_set():
    """`--transforms hiss` reads the eight transforms; each is registered; resample_roundtrip stays out."""
    assert nf.transforms_arg("hiss") == nf.HISS_TRANSFORMS
    assert set(nf.HISS_TRANSFORMS) <= set(nf.TRANSFORMS)
    assert "resample_roundtrip" not in nf.HISS_TRANSFORMS
    with pytest.raises(argparse.ArgumentTypeError):
        nf.transforms_arg("hiss,shift_1")
