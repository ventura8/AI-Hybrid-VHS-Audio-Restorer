"""Noise suppression of the mode's own, in the slot cathar's subtraction holds.

The subtraction stage produces nearly all of `auto_pure_linear`'s noise removal, and it is
an opaque CLI: power subtraction at one global factor over the whole spectrum, with a
spectral floor. One factor cannot fit both hiss and programme -- the most tonal third of
the corpus deviates 0.49 dB against cathar's 0.32 at the factor real tape wants, and nine
of the eleven captures cathar wins outright are tonal -- and the corpus-wide factor is held
short of what removal alone would take because past it the deviation lands beyond cathar's.

This is the estimator the literature settled on for exactly that trade: a per-bin,
per-frame gain from the bin's own signal-to-noise ratio (the MMSE log-spectral amplitude
estimator of Ephraim and Malah), with the a priori ratio smoothed by the decision-directed
rule so the gain is steady where the noise is steady and lets a sustained partial twenty
decibels over the floor through where a global factor shaves it. The noise's spectral shape
comes from the same quietest stretch the subtraction learns from, chosen by the same rule;
its level is then tracked through the recording by minimum statistics on one whitened
number per frame, so a compander's slow ride is followed without letting per-bin minima
climb into a held note. One gain, from the channel-power mean, serves both channels; the
phase is kept.

The frame grid is the learned blend's, so the blend reads exactly the bins this changed.
Analysis and synthesis both use the square root of a Hann window: the plain window adds up
to one only when the gain is constant, and this gain is not.
"""

from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from .blend_weights import FRAME, HOP, _frame_count, _read_frames
from .hygiene import atomic_target
from .utils import log_msg

BLOCK_FRAMES = 256
# The a priori ratio's floor, in power. The textbook -15 dB leaves a noise-only bin at about
# -14 dB of gain, short of the -20 dB floor cathar's subtraction reaches in the same bin;
# at -25 dB the estimator's own gain lands under the floor there and the floor decides.
XI_MIN = 10.0 ** (-25.0 / 10.0)
# Minimum statistics on the whitened frame energy: eight sub-windows of sixteen frames span
# three seconds at the pipeline's hop, long enough to hold a pause of speech; the bias
# corrects a minimum's downward lean; and the level is held inside a band around the probe,
# so a stretch of digital silence cannot switch the suppression off for the next three
# seconds and three seconds of unbroken programme cannot lift the estimate into it. The
# frame energy is a mean over a thousand bins already and needs no smoothing of its own --
# smoothed, it could not fall to the floor inside a one-second pause.
MS_SUBWINDOWS = 8
MS_SUBWINDOW_FRAMES = 16
MS_BIAS = 1.5
LEVEL_FLOOR = 10.0 ** (-6.0 / 10.0)
LEVEL_CEILING = 10.0 ** (6.0 / 10.0)
# The noise probe: the quietest of up to eight one-minute windows across the recording,
# and inside it the quietest of forty probe positions -- cathar's rule, so the two engines
# learn from the same stretch. The level is read across the channels' power rather than
# their downmix, the one departure: anti-phase content cancels in a downmix and would make
# the loudest stretch read as the quietest.
PROBE_WINDOW_S = 60.0
PROBE_WINDOWS = 8
PROBE_POSITIONS = 40
# E1(0) is infinite and E1 past this is zero to double precision; both ends are guarded.
NU_MIN = 1e-10
EPS = 1e-20
STAGE_FAILURES = (OSError, RuntimeError, ValueError, MemoryError)


def _window():
    """The square root of a periodic Hann window: its square overlap-adds to one at half a frame."""
    return np.sqrt(scipy.signal.get_window("hann", FRAME, fftbins=True))


def _quietest_of(windows):
    """The (samples, start) pair with the lowest RMS."""
    return min(windows, key=lambda item: float(np.sqrt(np.mean(item[0] ** 2) + 1e-12)))


def _probe_position(mono, count):
    """The start of the quietest probe inside a window, on cathar's grid of positions, skipping digital silence."""
    latest = len(mono) - count
    if latest <= 0:
        return 0
    best, lowest = 0, float("inf")
    for position in np.linspace(0, latest, min(PROBE_POSITIONS, latest + 1), dtype=int):
        start, stop = int(position), int(position) + count
        rms = float(np.sqrt(np.mean(mono[start:stop] ** 2) + 1e-12))
        if 1e-6 < rms < lowest:
            best, lowest = start, rms
    return best


def quietest_probe(source_wav, probe_s):
    """The start sample and length of the quietest stretch of the recording, or None when it is too short."""
    with sf.SoundFile(str(source_wav)) as handle:
        count = int(probe_s * handle.samplerate)
        if handle.frames <= count or count < FRAME:
            return None
        window = min(handle.frames, int(PROBE_WINDOW_S * handle.samplerate))
        latest = max(0, handle.frames - window)
        starts = np.linspace(0, latest, num=min(PROBE_WINDOWS, max(1, handle.frames // window)), dtype=int)
        windows = []
        for start in starts:
            handle.seek(int(start))
            windows.append((_channel_power(handle.read(window, dtype="float32", always_2d=True)), int(start)))
        level, start = _quietest_of(windows)
    return start + _probe_position(level, count), count


def _channel_power(samples):
    """A per-sample level read across the channels' power, not their sum: anti-phase content does not cancel in it."""
    return np.sqrt(np.mean(samples.astype(np.float64) ** 2, axis=1))


def _spectra(span):
    """The STFT of a span read on the blend's grid: (frames x channels x bins), one frame per hop."""
    frames = np.lib.stride_tricks.sliding_window_view(span, FRAME, axis=0)[::HOP]
    return np.fft.rfft(frames * _window(), axis=-1)


def noise_profile(source_wav, probe_s):
    """The noise's power per bin, averaged over the quietest stretch and both channels, or None."""
    probe = quietest_probe(source_wav, probe_s)
    if probe is None:
        return None
    start, count = probe
    with sf.SoundFile(str(source_wav)) as handle:
        handle.seek(start)
        samples = handle.read(count, dtype="float32", always_2d=True)
    power = np.abs(_spectra(samples)) ** 2
    return power.mean(axis=(0, 1)) + EPS


# Abramowitz and Stegun 5.1.53 and 5.1.56: the exponential integral E1 to better than 2e-7
# below one and 2e-8 above it, in plain arithmetic. scipy's own is a compiled function
# the static checks cannot see, and this is what an amplitude gain needs of it.
_E1_SMALL = (-0.57721566, 0.99999193, -0.24991055, 0.05519968, -0.00976004, 0.00107857)
_E1_LARGE_NUMERATOR = (1.0, 8.5733287401, 18.0590169730, 8.6347608925, 0.2677737343)
_E1_LARGE_DENOMINATOR = (1.0, 9.5733223454, 25.6329561486, 21.0996530827, 3.9584969228)


def exponential_integral(x):
    """E1(x) for positive x, elementwise."""
    x = np.asarray(x, dtype=np.float64)
    small = np.minimum(x, 1.0)
    below = np.polyval(_E1_SMALL[::-1], small) - np.log(small)
    large = np.maximum(x, 1.0)
    ratio = np.polyval(_E1_LARGE_NUMERATOR, large) / np.polyval(_E1_LARGE_DENOMINATOR, large)
    above = np.exp(-large) / large * ratio
    return np.where(x <= 1.0, below, above)


def lsa_gain(gamma, xi):
    """The MMSE log-spectral amplitude gain for a posteriori and a priori ratios, unclipped."""
    nu = np.maximum(xi * gamma / (1.0 + xi), NU_MIN)
    return xi / (1.0 + xi) * np.exp(0.5 * exponential_integral(nu))


class LevelTracker:
    """The noise level relative to the probe, tracked by minimum statistics on the whitened frame energy.

    A frame's energy divided by the probe's shape reads near one on probe-like frames and
    far above it on programme; the minimum over the last second and a half of a smoothed
    version of that number is the noise's level now, whatever the programme is doing.
    """

    def __init__(self):
        self.current = float("inf")
        self.count = 0
        self.ring = [1.0] * MS_SUBWINDOWS

    def update(self, energy):
        """Feeds one frame's whitened energy and returns the noise level to use for it."""
        self.current = min(self.current, energy)
        self.count += 1
        if self.count == MS_SUBWINDOW_FRAMES:
            self.ring = self.ring[1:] + [self.current]
            self.current, self.count = float("inf"), 0
        level = MS_BIAS * min(min(self.ring), self.current)
        return float(np.clip(level, LEVEL_FLOOR, LEVEL_CEILING))


class Suppressor:
    """The per-frame estimator with its state: the previous frame's gain and ratio, and the level tracker."""

    def __init__(self, noise, noise_bias, gain_floor_db, dd_alpha):
        self.noise = noise
        self.bias = noise_bias
        self.floor = 10.0 ** (gain_floor_db / 20.0)
        self.alpha = dd_alpha
        self.tracker = LevelTracker()
        self.gain = np.ones_like(noise)
        self.gamma = np.ones_like(noise)

    def _gain(self, power):
        """One frame's gain from its channel-power mean, advancing the state."""
        level = self.tracker.update(float(np.mean(power / self.noise)))
        gamma = power / (self.bias * self.noise * level)
        estimate = self.alpha * self.gain**2 * self.gamma + (1.0 - self.alpha) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(estimate, XI_MIN)
        self.gain = np.clip(lsa_gain(gamma, xi), self.floor, 1.0)
        self.gamma = gamma
        return self.gain

    def apply(self, spectrum):
        """Gains every frame of a block in order, one gain across the channels; the phase is kept."""
        out = np.empty_like(spectrum)
        for index, frame in enumerate(spectrum):
            out[index] = frame * self._gain(np.mean(np.abs(frame) ** 2, axis=0))[None, :]
        return out


def _resynthesise(spectrum):
    """Frames back to windowed samples: (frames x channels x FRAME)."""
    return np.fft.irfft(spectrum, n=FRAME, axis=-1) * _window()


class _Emitter:
    """Writes reconstructed samples, dropping the half frame before the file starts and stopping at its end."""

    def __init__(self, out, length):
        self.out, self.length, self.position = out, length, -(FRAME // 2)

    def push(self, samples):
        start, stop = self.position, self.position + len(samples)
        self.position = stop
        first, last = max(start, 0) - start, min(stop, self.length) - start
        if last > first:
            self.out.write(samples[first:last].astype(np.float32))


def _overlap_add(frames, carry):
    """Overlap-adds a block's frames onto the tail carried from the previous block; returns (ready, new tail)."""
    count, channels = len(frames), frames.shape[1]
    ready, tail = count * HOP, FRAME - HOP
    buffer = np.zeros((ready + tail, channels), dtype=np.float64)
    buffer[:tail] += carry
    for index in range(count):
        offset, end = index * HOP, index * HOP + FRAME
        buffer[offset:end] += frames[index].T
    return buffer[:ready], buffer[ready:]


def suppress_file(source_wav, target_wav, noise_bias, gain_floor_db, dd_alpha, probe_s):
    """Writes the suppressed recording a block at a time; returns the target, or None when there is no probe to learn from."""
    noise = noise_profile(source_wav, probe_s)
    if noise is None:
        return None
    with (
        atomic_target(target_wav) as partial,
        sf.SoundFile(str(source_wav)) as source,
        sf.SoundFile(str(partial), "w", samplerate=source.samplerate, channels=source.channels, subtype="FLOAT") as out,
    ):
        length, channels = source.frames, source.channels
        suppressor, emitter = Suppressor(noise, noise_bias, gain_floor_db, dd_alpha), _Emitter(out, length)
        carry = np.zeros((FRAME - HOP, channels))
        for first in range(0, _frame_count(length), BLOCK_FRAMES):
            last = min(first + BLOCK_FRAMES, _frame_count(length))
            frames = _resynthesise(suppressor.apply(_spectra(_read_frames(source, first, last, length))))
            ready, carry = _overlap_add(frames, carry)
            emitter.push(ready)
        emitter.push(carry)
    return Path(target_wav)


def suppress_or_none(source_wav, target_wav, **settings):
    """The stage's guarded entry: a failure inside the estimator is logged and returns None, so the caller can fall back."""
    try:
        return suppress_file(source_wav, target_wav, **settings)
    except STAGE_FAILURES as exc:
        log_msg(f"    [Spectral Denoise] Native suppressor failed ({exc}); falling back.")
        return None
