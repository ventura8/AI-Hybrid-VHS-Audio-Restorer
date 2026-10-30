"""scipy.signal's stft and welch, value for value, without holding the whole transform.

The scanner's profile extraction reads up to 240 s of audio (`filters._read_analysis_audio`
reads four 60 s blocks of a long file). On that input `scipy.signal.stft` peaked at 593 MB in
the tonality reading and `scipy.signal.welch` at 127 MB in the probe-similarity reading
(issue #14). The legacy stft pads the end of a float32 signal with float64 zeros and windows it
with a complex64 window, so it holds a complex128 copy of every overlapping segment; the welch
holds every frame's complex spectrum beside its periodogram.

The routing thresholds (`auto_cathar_flatness_max`, `auto_cathar_probe_similarity`) and the
tonal gate (`apl_tonal_flatness_max`) were calibrated on the numbers those calls produce, so
nothing here approximates them. The same arithmetic runs on a block of frames at a time, and
the reductions get an array laid out the way scipy's is, so a mean sums in the same order and
every value equals scipy's bit for bit (`tests/unit/test_lean_spectra.py` compares them on
signals that cross many blocks). An input with no more frames than one block goes to scipy
itself. Both follow scipy 1.18's code paths; a scipy release that changes them fails those
tests rather than moving a routing number unseen.
"""

import numpy as np
import scipy.fft
import scipy.signal

# Frames per block: 256 frames of 4096 samples hold 17 MB of windowed segments. A multiple of
# any FFT vector width, so every frame takes the SIMD lane it takes in one whole-signal call.
BLOCK_FRAMES = 256
WINDOW = "hann_periodic"


def stft_frame_count(length, nperseg):
    """The frames `scipy.signal.stft` (zero boundary, padded) cuts from `length` samples."""
    hop = nperseg - nperseg // 2
    extended = length + 2 * (nperseg // 2)
    padded = extended + (-(extended - nperseg) % hop) % nperseg
    return (padded - nperseg) // hop + 1


def _blocks(frames):
    """The (first, last) frame ranges of the blocks, in order."""
    return [(first, min(first + BLOCK_FRAMES, frames)) for first in range(0, frames, BLOCK_FRAMES)]


def stft_band(signal, sample_rate, nperseg, band_hz):
    """`Zxx[band]` of `scipy.signal.stft(signal, sample_rate, nperseg=nperseg)`, band_hz inclusive.

    Only the band's rows are kept: 455 of 2049 at 100-5000 Hz and a 4096-sample frame, the
    array scipy's boolean index returns (C order, bins by frames).
    """
    signal = np.asarray(signal)
    freqs = scipy.fft.rfftfreq(nperseg, 1 / sample_rate)
    band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    frames = stft_frame_count(len(signal), nperseg)
    if frames <= BLOCK_FRAMES:
        return scipy.signal.stft(signal, sample_rate, nperseg=nperseg)[2][band]
    transform = _LegacyStft(signal, nperseg)
    out = np.empty((int(np.count_nonzero(band)), frames), dtype=transform.outdtype)
    for first, last in _blocks(frames):
        out[:, first:last] = transform.rows(first, last)[:, band].T
    return out


class _LegacyStft:
    """`scipy.signal._spectral_py._spectral_helper` in 'stft' mode, for a range of frames.

    Its dtype path is kept as it is: the end padding scipy concatenates is float64, so a
    float32 signal is windowed in float64; the window is cast to the complex output type
    first, and the rows are rounded to it last.
    """

    def __init__(self, signal, nperseg):
        self.signal = signal
        self.nperseg = nperseg
        self.hop = nperseg - nperseg // 2
        self.outdtype = np.result_type(signal, np.complex64)
        self.work_dtype = np.result_type(signal, np.float64)
        win = scipy.signal.get_window(WINDOW, nperseg)
        if np.result_type(win, np.complex64) != self.outdtype:
            win = win.astype(self.outdtype)
        self.win = win
        self.scale = np.sqrt(1.0 / win.sum() ** 2)

    def rows(self, first, last):
        """Frames first..last-1, one row each, as scipy computes them before moving the axes."""
        start = first * self.hop - self.nperseg // 2
        piece = np.zeros((last - first - 1) * self.hop + self.nperseg, dtype=self.work_dtype)
        low, high = max(start, 0), min(start + len(piece), len(self.signal))
        head, tail = low - start, high - start
        piece[head:tail] = self.signal[low:high]
        segments = np.lib.stride_tricks.sliding_window_view(piece, self.nperseg)[:: self.hop]
        result = scipy.fft.rfft((self.win * segments).real, n=self.nperseg)
        result *= self.scale
        return result.astype(self.outdtype)


def welch(segment, sample_rate, nperseg):
    """`scipy.signal.welch(segment, sample_rate, nperseg=nperseg)`, its periodogram filled by blocks.

    scipy's csd builds the periodogram from `ShortTimeFFT.spectrogram`, which transforms one
    frame at a time, then averages it over the frame axis. The blocks fill the same
    periodogram (float64, C order, bins by frames), and the mean is scipy's own call on it.
    """
    segment = np.asarray(segment)
    hop = nperseg - nperseg // 2
    frames = (len(segment) - nperseg // 2) // hop
    if frames <= BLOCK_FRAMES:
        return scipy.signal.welch(segment, sample_rate, nperseg=nperseg)
    sft = scipy.signal.ShortTimeFFT(
        scipy.signal.get_window(WINDOW, nperseg), hop, sample_rate, fft_mode="onesided", mfft=nperseg, scale_to="psd", phase_shift=None
    )
    periodogram = np.empty((sft.f_pts, frames))
    for first, last in _blocks(frames):
        periodogram[:, first:last] = sft.spectrogram(segment, detr="constant", p0=first, p1=last, k_offset=nperseg // 2)
    # Every bin but DC (and Nyquist, for an even frame) counts twice in a one-sided density.
    paired = slice(1, -1 if nperseg % 2 == 0 else None)
    periodogram[paired] *= 2
    return sft.f, periodogram.mean(axis=-1).astype(np.result_type(segment, np.complex64)).real
