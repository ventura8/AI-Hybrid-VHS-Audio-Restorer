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

from . import noiseprint_quantile
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
# The reference-SNR suppressor's noise (`quantile_noise_profile`): per bin, the 5th percentile
# over the whole recording of the 16-frame (0.37 s) moving mean of the channel-mean power, the
# quantile print's estimator (`noiseprint_quantile`) read in power on this grid. The quietest
# stretch above holds speech on a speech tape: on the four Tata tapes it read 13-25 dB over the
# floor of their true pauses below 8 kHz (0-3 dB over it at 12-16 kHz), and a gain read against
# it took the loud programme over 3 kHz down 4 dB. The quantile read that floor within -2.7 to
# +2.2 dB per band before its bias; a 16-frame mean's 5th percentile sits 2.3 dB under the mean
# on white noise, and +2.0 dB centres it.
QUANTILE_SMOOTH_FRAMES = 16
QUANTILE = 0.05
QUANTILE_BIAS_DB = 2.0
# With no 0.37 s of hiss alone in 5 % of the windows (dense music) the print learns the programme,
# and a gain read against it takes the programme down: a fixture of 0.3 s bursts and 0.2 s gaps
# lost 15 dB of its partials. Such a print sits close under the loud frames: over 300-3000 Hz the
# loud frames' P90 stood 24.7-31.9 dB over the print on the four Tata tapes and 32.5 on bursts with
# a 0.8 s pause every 3 s, 3.8 on the gapless bursts and 1.7-58.6 on the twelve music clips. Under
# MIN_DEPTH_DB the suppressor does not run. On the IA hunt's 15 s clips (CPU replay, 2026-10-17) a
# print 18-23 dB down still held programme often enough to matter: on 37 such clips the loud HF fell
# more than 1 dB on 11 and a v4 guard newly tripped on 4, against 9 and 5 of the 63 clips at 23 dB and
# over, for the same share of clear hiss wins (8 and 14, 22 % each). 23 dB keeps every Tata tape.
DEPTH_BAND_HZ = (300.0, 3000.0)
DEPTH_LOUD_PERCENTILE = 90.0
MIN_DEPTH_DB = 23.0
# The programme release (`release_db`, off at 0). The masks multiply, so a cell the neural stage kept
# loses the source-SNR gain too: soft programme 0-10 dB over the source's hiss lost 2-9 dB, and on 5
# of the 63 IA hunt clips past MIN_DEPTH_DB a v4 guard tripped (the top band past -1 dB, the loud HF
# past -6 dB). Hiss the restoring stage left sits at or under the print (its mask is at most one), so
# where the restored frame stands `release_db` over the print the gain eases toward 1 in dB, fully
# released RELEASE_SPAN_DB higher; the frame's power is a three-bin mean, so a hiss cell's periodogram
# seldom crosses. CPU replay (2026-10-20, 19 IA clips): at 3 dB the top band came back on the three
# clips that crossed -1 dB, soft programme 1.5-2 dB, the hiss excess moved under 0.45 dB.
RELEASE_SPAN_DB = 6.0
RELEASE_SMOOTH = np.ones(3) / 3.0
# The release's reference, the pause print (2026-10-21): per bin, the median of the PAUSE_SMOOTH_FRAMES-frame moving
# mean over the source's pause frames (frame power at or under its PAUSE_PERCENTILE frame plus PAUSE_MARGIN_DB), never
# under the print. Vaccin's MP3 coder drops 10-16 kHz in its speech frames, so the whole-file print's 5th percentile
# sat 11.7 dB under the hiss its pauses hold, the residual hiss stood over it and the release gave it back (the APL
# v9 loop's round 4: hiss under Vaccin's speech +2.18 dB over its masked threshold). The coder keeps the hiss in the
# pauses. Under PAUSE_MIN_WINDOWS pause windows the release reads against the print.
PAUSE_PERCENTILE = 15.0
PAUSE_MARGIN_DB = 3.0
PAUSE_SMOOTH_FRAMES = 4
PAUSE_QUANTILE = 0.5
PAUSE_MIN_WINDOWS = 20


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


def frame_powers(span):
    """`(channel-mean power per frame and bin, live mask)` of a span read on the blend's grid; live as the quantile print's."""
    frames = np.lib.stride_tricks.sliding_window_view(span, FRAME, axis=0)[::HOP]
    power = np.mean(np.abs(np.fft.rfft(frames * _window(), axis=-1)) ** 2, axis=1)
    return power, noiseprint_quantile.live_frames(np.mean(frames, axis=1))


def reference_noise(source_wav):
    """`(noise power per bin, depth in dB)` from the whole recording: the quantile print (QUANTILE_* above) and how far
    the live frames' DEPTH_LOUD_PERCENTILE stands over it in DEPTH_BAND_HZ; `(None, None)` under the print's minimum of
    live windows."""
    acc = noiseprint_quantile.MovingQuantile(bins=FRAME // 2 + 1, smooth=QUANTILE_SMOOTH_FRAMES, scale=10.0)
    levels = []
    with sf.SoundFile(str(source_wav)) as source:
        length, freqs = source.frames, np.fft.rfftfreq(FRAME, 1.0 / source.samplerate)
        band = (freqs >= DEPTH_BAND_HZ[0]) & (freqs < DEPTH_BAND_HZ[1])
        for first in range(0, _frame_count(length), BLOCK_FRAMES):
            power, live = frame_powers(_read_frames(source, first, min(first + BLOCK_FRAMES, _frame_count(length)), length))
            acc.add(power, live)
            levels.append(power[live][:, band].mean(axis=1))
    if acc.windows < noiseprint_quantile.MIN_WINDOWS:
        return None, None
    noise = acc.quantile(QUANTILE) * 10.0 ** (QUANTILE_BIAS_DB / 10.0) + EPS
    loud = np.percentile(np.concatenate(levels), DEPTH_LOUD_PERCENTILE)
    return noise, float(10.0 * np.log10(loud / np.mean(noise[band])))


def _frame_level_pass(source_wav):
    """`(channel-mean power summed over the bins per frame, live mask)` over the whole recording, on the suppressor's grid."""
    levels, lives = [], []
    with sf.SoundFile(str(source_wav)) as source:
        count = _frame_count(source.frames)
        for first in range(0, count, BLOCK_FRAMES):
            power, live = frame_powers(_read_frames(source, first, min(first + BLOCK_FRAMES, count), source.frames))
            levels.append(power.sum(axis=1))
            lives.append(live)
    return np.concatenate(levels), np.concatenate(lives)


def pause_print(source_wav, noise):
    """The programme release's reference: per bin the larger of the print and the pause frames' moving-mean median
    (PAUSE_* above); the print itself under PAUSE_MIN_WINDOWS windows of pause."""
    levels, live = _frame_level_pass(source_wav)
    if not live.any():
        return noise
    pause = live & (levels <= np.percentile(levels[live], PAUSE_PERCENTILE) * 10.0 ** (PAUSE_MARGIN_DB / 10.0))
    acc = noiseprint_quantile.MovingQuantile(bins=FRAME // 2 + 1, smooth=PAUSE_SMOOTH_FRAMES, scale=10.0)
    with sf.SoundFile(str(source_wav)) as source:
        count = _frame_count(source.frames)
        for first in range(0, count, BLOCK_FRAMES):
            last = min(first + BLOCK_FRAMES, count)
            acc.add(frame_powers(_read_frames(source, first, last, source.frames))[0], pause[first:last])
    if acc.windows < PAUSE_MIN_WINDOWS:
        return noise
    return np.maximum(noise, acc.quantile(PAUSE_QUANTILE))


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


class HeldLevel:
    """No level tracking: tape hiss is stationary, and the whole-file print is its level already."""

    def update(self, energy):
        """The level for a frame: always the print's own."""
        del energy
        return 1.0


class ReferenceSuppressor(Suppressor):
    """The estimator's gain read on the reference's frame and applied to the restored audio's frame.

    Model the neural stage as a mask M = |restored| / |reference|: the hiss it leaves is M^2 N,
    so the restored audio's a posteriori ratio against that hiss is the reference's own,
    |reference|^2 / N. The gain read on the reference therefore applies to the restored audio:
    the two masks multiply, a cell where the reference held only hiss loses up to the gain floor
    and a cell where the programme stood over the hiss keeps its level.
    """

    def __init__(self, noise, noise_bias, gain_floor_db, dd_alpha, release=(0.0, None)):
        super().__init__(noise, noise_bias, gain_floor_db, dd_alpha)
        self.tracker = HeldLevel()
        self.release_db, release_noise = release
        self.release_noise = noise if release_noise is None else release_noise

    def apply_pair(self, reference, restored):
        """Gains every restored frame of a block by the gain its reference frame reads, in order."""
        out = np.empty_like(restored)
        for index, (ref_frame, frame) in enumerate(zip(reference, restored)):
            gain = self._gain(np.mean(np.abs(ref_frame) ** 2, axis=0))
            out[index] = frame * self._released(gain, frame)[None, :]
        return out

    def _released(self, gain, frame):
        """The gain eased toward 1 in dB where the restored frame stands `release_db` to `release_db` +
        RELEASE_SPAN_DB over the release's reference, the pause print (programme the restoring stage kept); the gain
        itself with the release off."""
        if self.release_db <= 0.0:
            return gain
        power = np.convolve(np.mean(np.abs(frame) ** 2, axis=0), RELEASE_SMOOTH, mode="same")
        over_db = 10.0 * np.log10(np.maximum(power, EPS) / self.release_noise)
        weight = np.clip((over_db - self.release_db) / RELEASE_SPAN_DB, 0.0, 1.0)
        return gain ** (1.0 - weight)


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


def _stream(out, length, channels, gained):
    """Writes the overlap-added resynthesis of `gained(first, last)`, the gained spectra of each block of frames, in order."""
    emitter = _Emitter(out, length)
    carry = np.zeros((FRAME - HOP, channels))
    for first in range(0, _frame_count(length), BLOCK_FRAMES):
        last = min(first + BLOCK_FRAMES, _frame_count(length))
        ready, carry = _overlap_add(_resynthesise(gained(first, last)), carry)
        emitter.push(ready)
    emitter.push(carry)


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
        length, suppressor = source.frames, Suppressor(noise, noise_bias, gain_floor_db, dd_alpha)
        _stream(out, length, source.channels, lambda first, last: suppressor.apply(_spectra(_read_frames(source, first, last, length))))
    return Path(target_wav)


def suppress_by_reference(reference_wav, restored_wav, target_wav, noise_bias, gain_floor_db, dd_alpha, release_db=0.0):
    """Writes the restored recording gained by the reference's SNR (`ReferenceSuppressor`, its programme release at
    `release_db` against the pause print, 0 off); returns the target, or None when the reference holds too few live
    windows to learn its noise from or its print holds programme (MIN_DEPTH_DB). Raises when the two rates differ."""
    noise, depth = reference_noise(reference_wav)
    if noise is None:
        log_msg("    [Reference Suppress] Too little live audio in the source to learn its noise from.")
        return None
    if depth < MIN_DEPTH_DB:
        log_msg(
            f"    [Reference Suppress] The source's noise print sits {depth:.1f} dB under its loud frames "
            f"({DEPTH_BAND_HZ[0]:.0f}-{DEPTH_BAND_HZ[1]:.0f} Hz), under {MIN_DEPTH_DB:.0f}: it holds programme."
        )
        return None
    log_msg(f"    [Reference Suppress] The source's noise print sits {depth:.1f} dB under its loud frames.")
    release = (release_db, pause_print(reference_wav, noise) if release_db > 0.0 else None)
    with (
        atomic_target(target_wav) as partial,
        sf.SoundFile(str(reference_wav)) as reference,
        sf.SoundFile(str(restored_wav)) as restored,
        sf.SoundFile(str(partial), "w", samplerate=restored.samplerate, channels=restored.channels, subtype="FLOAT") as out,
    ):
        if reference.samplerate != restored.samplerate:
            raise ValueError(f"reference {reference.samplerate} Hz and restored {restored.samplerate} Hz do not match")
        suppressor = ReferenceSuppressor(noise, noise_bias, gain_floor_db, dd_alpha, release)

        def gained(first, last):
            """The restored frames [first, last) gained by their reference frames."""
            ref = _spectra(_read_frames(reference, first, last, reference.frames))
            return suppressor.apply_pair(ref, _spectra(_read_frames(restored, first, last, restored.frames)))

        _stream(out, restored.frames, restored.channels, gained)
    return Path(target_wav)


def suppress_by_reference_or_none(reference_wav, restored_wav, target_wav, **settings):
    """The reference suppressor's guarded entry: a failure is logged and returns None, so the caller keeps its input."""
    try:
        return suppress_by_reference(reference_wav, restored_wav, target_wav, **settings)
    except STAGE_FAILURES as exc:
        log_msg(f"    [Reference Suppress] Failed ({exc}); keeping the restored audio.", level="WARNING")
        return None


def suppress_or_none(source_wav, target_wav, **settings):
    """The stage's guarded entry: a failure inside the estimator is logged and returns None, so the caller can fall back."""
    try:
        return suppress_file(source_wav, target_wav, **settings)
    except STAGE_FAILURES as exc:
        log_msg(f"    [Spectral Denoise] Native suppressor failed ({exc}); falling back.", level="WARNING")
        return None
