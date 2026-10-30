"""The scanner's spectra by blocks of frames: scipy's values bit for bit, a fraction of its memory (issue #14).

The routing thresholds were calibrated on the numbers `scipy.signal.stft` and `scipy.signal.welch`
produce, so every comparison here is exact. A 7-frame block makes short signals cross many
blocks, odd frame lengths take the odd-length branches, and the peaks are measured the way the
issue measured them: tracemalloc on the 240 s the analysis read hands the scanner at most.
"""

import tracemalloc
from unittest.mock import patch

import numpy as np
import pytest
import scipy.signal

import modules.auto_scanner as scanner
import modules.lean_spectra as lean_spectra
from modules.spectral_denoise import tonality_of_signal

BAND_HZ = (100.0, 5000.0)


def _programme(seconds, rate=44100, dtype=np.float32, seed=0):
    """Hiss under a gated tone with a slow swell: frames of every level, a held partial."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    gate = (np.sin(2 * np.pi * 0.35 * t) > 0.2) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t))
    return (0.02 * rng.standard_normal(t.size) + 0.15 * gate * np.sin(2 * np.pi * 330.0 * t)).astype(dtype)


def _whole_tonality(signal, rate):
    """The tonality reading before issue #14: the whole transform, then the band."""
    freqs, _times, spectrum = scipy.signal.stft(signal, rate, nperseg=4096)
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    power = np.abs(spectrum[band]) ** 2 + 1e-20
    return float(np.median(np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0)))


def _whole_loudest(signal, frames):
    """The loud-frame selection before issue #14: every frame squared at once."""
    blocks = signal[: frames * scanner.PROBE_FRAME].reshape(frames, scanner.PROBE_FRAME)
    levels = 20.0 * np.log10(np.sqrt(np.mean(blocks**2, axis=1)) + 1e-9)
    return blocks[levels >= np.percentile(levels, scanner.PROBE_LOUD_PERCENTILE)].reshape(-1)


@pytest.mark.parametrize("length", [4096 * 2, 100_001, 441_000, 524_288, 524_289])
@pytest.mark.parametrize("nperseg", [4096, 255])
def test_the_frame_count_is_scipy_s(length, nperseg):
    assert lean_spectra.stft_frame_count(length, nperseg) == scipy.signal.stft(np.zeros(length), nperseg=nperseg)[2].shape[1]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("block", [7, 64])
def test_the_band_rows_equal_scipy_s_stft_across_blocks(dtype, block):
    signal = _programme(9.3, dtype=dtype)
    freqs, _times, spectrum = scipy.signal.stft(signal, 44100, nperseg=4096)
    expected = spectrum[(freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])]
    with patch.object(lean_spectra, "BLOCK_FRAMES", block):
        produced = lean_spectra.stft_band(signal, 44100, 4096, BAND_HZ)
    assert produced.dtype == expected.dtype
    assert np.array_equal(produced, expected)


def test_an_odd_frame_takes_the_same_rows():
    signal = _programme(1.5, rate=8000, seed=4)
    freqs, _times, spectrum = scipy.signal.stft(signal, 8000, nperseg=255)
    with patch.object(lean_spectra, "BLOCK_FRAMES", 5):
        produced = lean_spectra.stft_band(signal, 8000, 255, BAND_HZ)
    assert np.array_equal(produced, spectrum[(freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])])


def test_the_band_array_is_laid_out_like_scipy_s_boolean_index():
    """The flatness means sum along the bins; the same C layout makes them sum in the same order."""
    signal = _programme(9.3)
    freqs, _times, spectrum = scipy.signal.stft(signal, 44100, nperseg=4096)
    with patch.object(lean_spectra, "BLOCK_FRAMES", 7):
        produced = lean_spectra.stft_band(signal, 44100, 4096, BAND_HZ)
    assert spectrum[(freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])].flags.c_contiguous
    assert produced.flags.c_contiguous


def test_a_short_signal_goes_to_scipy_itself():
    signal = _programme(2.0)
    with patch("modules.lean_spectra._LegacyStft", side_effect=AssertionError("no blocks for one block's frames")):
        produced = lean_spectra.stft_band(signal, 44100, 4096, BAND_HZ)
    assert produced.shape[1] == lean_spectra.stft_frame_count(len(signal), 4096)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(("nperseg", "block"), [(4096, 7), (4096, 256), (255, 5)])
def test_welch_equals_scipy_s_across_blocks(dtype, nperseg, block):
    signal = _programme(13.1, dtype=dtype, seed=2)
    freqs, expected = scipy.signal.welch(signal, 44100, nperseg=nperseg)
    with patch.object(lean_spectra, "BLOCK_FRAMES", block):
        produced_freqs, produced = lean_spectra.welch(signal, 44100, nperseg)
    assert np.array_equal(produced_freqs, freqs)
    assert produced.dtype == expected.dtype
    assert np.array_equal(produced, expected)


@pytest.mark.parametrize("block", [7, 256])
def test_the_tonality_equals_the_whole_transform_s(block):
    signal = _programme(31.0, seed=3)
    with patch.object(lean_spectra, "BLOCK_FRAMES", block):
        assert tonality_of_signal(signal, 44100) == _whole_tonality(signal, 44100)


@pytest.mark.parametrize("rows", [37, 2048])
def test_the_loud_frames_equal_the_whole_signal_s(rows):
    signal = _programme(20.0, seed=5)
    frames = len(signal) // scanner.PROBE_FRAME
    with patch.object(scanner, "PROBE_LEVEL_ROWS", rows):
        assert np.array_equal(scanner._loudest_frames(signal, frames), _whole_loudest(signal, frames))


def test_the_speech_band_shape_equals_scipy_s_welch():
    signal = _programme(20.0, seed=6)
    freqs, psd = scipy.signal.welch(signal, 44100, nperseg=4096)
    spectrum = 10.0 * np.log10(psd[(freqs >= scanner.PROBE_BAND_HZ[0]) & (freqs <= scanner.PROBE_BAND_HZ[1])] + 1e-20)
    with patch.object(lean_spectra, "BLOCK_FRAMES", 7):
        assert np.array_equal(scanner._speech_band_shape(signal, 44100), spectrum - spectrum.mean())


def _peak_mb(function, *args):
    tracemalloc.start()
    try:
        function(*args)
        return tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()


@pytest.fixture(scope="module")
def longest_analysis_read():
    """240 s at 44.1 kHz in float32: four 60 s blocks, the most the analysis read returns."""
    return _programme(240.0, seed=7)


def test_the_tonality_peak_stays_under_150_mb_on_the_longest_read(longest_analysis_read):
    assert _peak_mb(scanner._estimate_tonality, longest_analysis_read, 44100) < 150.0


def test_the_probe_similarity_peak_stays_under_100_mb_on_the_longest_read(longest_analysis_read):
    assert _peak_mb(scanner._probe_programme_similarity, longest_analysis_read, 44100) < 100.0
