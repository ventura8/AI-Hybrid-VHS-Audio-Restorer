"""The split-band crossover: complementary kernels, an identity on a file recombined with itself, and the bands it selects."""

import numpy as np
import pytest
import soundfile as sf

from modules import split_band

RATE = 44100
SECONDS = 2.0


def _noise(channels, seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.25, (int(RATE * SECONDS), channels)).astype(np.float32)


def _tone(hz):
    t = np.arange(int(RATE * SECONDS)) / RATE
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _write(path, samples):
    sf.write(str(path), samples, RATE, subtype="FLOAT")
    return path


def _tone_db(samples, hz):
    """The level of a tone in dB, read from an rfft over the middle second, where both tones sit on exact bins."""
    start, stop = RATE, 2 * RATE
    spectrum = np.abs(np.fft.rfft(samples[start:stop]))
    return 20.0 * np.log10(spectrum[int(hz)] + 1e-12)


def test_crossover_kernels_sum_to_a_delayed_impulse():
    h_lo, h_hi = split_band.crossover(RATE, 6000.0)
    delta = np.zeros(split_band.TAPS)
    delta[(split_band.TAPS - 1) // 2] = 1.0
    assert np.max(np.abs(h_lo + h_hi - delta)) < 1e-12


@pytest.mark.parametrize("hz", [0.0, -100.0, RATE / 2.0, RATE])
def test_crossover_rejects_a_frequency_outside_the_open_band(hz):
    with pytest.raises(ValueError):
        split_band.crossover(RATE, hz)


@pytest.mark.parametrize("channels", [1, 2])
def test_recombining_a_file_with_itself_is_the_identity(tmp_path, channels):
    source = _noise(channels)
    original = _write(tmp_path / "x.wav", source)
    whole = split_band.recombine(original, original, tmp_path / "whole.wav", 6000.0)
    got, _rate = sf.read(str(whole), dtype="float32", always_2d=True)
    assert whole == tmp_path / "whole.wav"
    assert got.shape == source.shape
    assert np.max(np.abs(got - source)) < 1e-6


@pytest.mark.parametrize("channels", [1, 2])
def test_a_small_block_size_does_not_change_the_recombination(tmp_path, monkeypatch, channels):
    original = _write(tmp_path / "x.wav", _noise(channels))
    whole = split_band.recombine(original, original, tmp_path / "whole.wav", 6000.0)
    got, _rate = sf.read(str(whole), dtype="float32", always_2d=True)
    monkeypatch.setattr(split_band, "BLOCK_SAMPLES", 4096)
    streamed = split_band.recombine(original, original, tmp_path / "streamed.wav", 6000.0)
    got_streamed, _rate = sf.read(str(streamed), dtype="float32", always_2d=True)
    assert got_streamed.shape == got.shape
    assert np.max(np.abs(got_streamed - got)) < 1e-9


def test_each_rendering_contributes_only_its_own_band(tmp_path):
    low_tone, high_tone = _tone(1000.0), _tone(10000.0)
    low = _write(tmp_path / "low.wav", low_tone)
    high = _write(tmp_path / "high.wav", high_tone)
    kept, _rate = sf.read(str(split_band.recombine(low, high, tmp_path / "kept.wav", 6000.0)))
    swapped, _rate = sf.read(str(split_band.recombine(high, low, tmp_path / "swapped.wav", 6000.0)))
    low_ref, high_ref = _tone_db(low_tone, 1000.0), _tone_db(high_tone, 10000.0)
    assert abs(_tone_db(kept, 1000.0) - low_ref) < 0.5
    assert abs(_tone_db(kept, 10000.0) - high_ref) < 0.5
    assert low_ref - _tone_db(swapped, 1000.0) > 40.0
    assert high_ref - _tone_db(swapped, 10000.0) > 40.0


def test_renderings_of_different_lengths_recombine_to_the_shorter(tmp_path):
    source = _noise(1)
    low = _write(tmp_path / "low.wav", source)
    high = _write(tmp_path / "high.wav", source[: len(source) - 3])
    got, _rate = sf.read(str(split_band.recombine(low, high, tmp_path / "out.wav", 6000.0)), always_2d=True)
    assert len(got) == len(source) - 3


def test_renderings_that_do_not_match_are_refused(tmp_path):
    low = _write(tmp_path / "low.wav", _noise(1))
    high = _write(tmp_path / "high.wav", _noise(2))
    with pytest.raises(ValueError):
        split_band.recombine(low, high, tmp_path / "out.wav", 6000.0)


def test_an_existing_valid_target_is_returned_without_recomputing(tmp_path):
    target = _write(tmp_path / "out.wav", _noise(1))
    before = target.stat().st_mtime_ns
    got = split_band.recombine(tmp_path / "missing_low.wav", tmp_path / "missing_high.wav", target, 6000.0)
    assert got == target
    assert target.stat().st_mtime_ns == before
