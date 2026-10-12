"""cathar's whole-file quantile noise print (`cathar_noiseprint_method: quantile`): the floor under the programme, per bin.

cathar learns its print from the quietest 0.75 s window of a clip, or from six stitched 0.75 s
pauses on 90 s of material or more (`cathar._probe_duration_s`). A 15 s clip with no pause has
no quiet window, so the print is programme and the subtraction strips the background bed; a
digital mute teaches it nothing. This print is read from the whole material instead, streamed
block by block: cathar 0.8.0's own frames (2048-sample symmetric Hann, hop 512, whole frames
only, the left channel; a print written this way is the JSON list of 1025 mean |rfft| that
`cathar noiseprint` writes, checked to 4.5e-7 against cathar's on a probe), each bin's
magnitude averaged over SMOOTH_FRAMES frames (0.37 s at 44.1 kHz), and per bin the QUANTILE of
those moving means over the windows whose every frame is live, times BIAS_DB. A frame is a
digital mute (left out, with every window that holds it) under NOISEPRINT_DIGITAL_FLOOR_DBFS or
when half its samples repeat the one before (`noiseprint_guard`'s rules). The quantiles come
from a per-bin histogram of the moving means in dB (HIST_STEP_DB bins, linear within a bin), so
memory does not grow with the material.

Calibrated on artifacts/realistic-v2 (2026-10-12): the exact tape noise of each fixture is
vhs - clean (the hiss is injected after the recorder's gain control), and each estimator is
read by the log-spectral distance of its print to that noise's print over the bins within 40 dB
of the noise's loudest (100 Hz-16 kHz). The bank's tape noise is no Rayleigh magnitude (its
5 kHz bin's mean sits 3.2-4.7 times its p10, class medians over the 200 tape-noise-only
fixtures, against 2.73, and 2.6-3.8 in the Tata tapes' true pauses; codec holes empty bins
above 10 kHz), so a raw low quantile with the Rayleigh correction misses by 5-20 dB; a moving mean first
narrows each bin to its local mean, holes included. Over 1-64 frames and quantiles 0.05-0.5,
each with its correction fitted as the median per-bin error over the 300 tape-noise-only
fixtures, the 5th percentile of the 32-frame mean reads closest: a median distance of 4.22 dB,
against 4.93 for today's 0.75 s window and 8.59 for the stitched pick, at +0.62 dB (on white
noise the estimator reads 1.93 dB under the mean; the programme lifts most of that back). On
fixtures with pauses (speech, music beds) the window is closer (3.4-4.1 dB against 3.7-4.6) but
reads the 3-16 kHz noise +1.3 to +2.4 dB high, this print -0.4 to +1.1; on music-led and
music-only fixtures, with no pause, the window learns programme (+5.0 to +5.8 dB at 3-16 kHz,
45-48 % of the bins over +6 dB) and this print reads -1.4 to -0.3 dB (11-15 %). Both learn the
room tone with the hiss (+1.5 to +2.8 dB under 3 kHz). The streamed print reads within 0.06 dB
(median) of the in-memory estimator it was calibrated as.

Measured on CPU (cathar 0.8.0, this code on copies; docs/configuration.md, "cathar Quantile
Noise Print"): at the denoise stage of the IA hunt's 56 denoise-class stems.octave_programme
failures, 25 stop failing (53 read failing there, 28 after), the print sitting a median 13.5 dB
under the loud frames' programme band against 8.6 for the window; 3 of the 96 passing clips
start failing. It removes less: the hiss floor under the programme comes down 10.2 dB against
14.1 (failing clips) and 8.9 against 10.7 (passing). On the four Tata tapes, whole, it sits
1.2-3.5 dB under the stitched print above 6 kHz and 2-13 dB under it below, where the stitched
pauses carry speech: in the output the floor above 7 kHz comes down 1.5-3.1 dB less, the 24-32
dB cut at 3-6 kHz on soti and tele7abc (that speech) becomes 12-13 dB, and musical noise and
the 's' lift fall.

On real tape, against the noise of the true pauses, this print reads 1.0-1.1 dB under it at
6-10 kHz (the Tata tapes, 52 IA hunt clips with pauses) where it reads +0.3 dB over it on the
realistic-v2 fixtures with pauses, so BIAS_DB does not transfer whole. The 5th percentile needs
noise-only material in 5 % of the windows (about 1.1 s of a 15 s clip): with less it learns the
programme's floor (itv-continuity, 12-18 dB over its pause noise). A near-silent stretch over
the digital floor whose bins a codec emptied is learned as zero (gogs-vhs-collection), a 2 s
fade to silence at each end pulls the print 4.2 dB down, and a partial held through 95 % of the
windows is learned in full (docs/configuration.md, "Real tape and limits").

cathar mode alone reads the key (`cathar._learned_print`), on the speech profile (the music
profile learns no print). The window path's mute and depth guards do not apply: this print has
no window. Material with fewer than MIN_WINDOWS live windows gets the window pick, with a
WARNING, and so does a print that cannot be read or written.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .hygiene import atomic_target
from .noiseprint_guard import NOISEPRINT_DIGITAL_FLOOR_DBFS, NOISEPRINT_MUTE_HELD_SHARE
from .utils import log_msg

# cathar 0.8.0's print frame: 2048 samples, symmetric Hann (denominator N - 1), hop 512.
FRAME = 2048
HOP = 512
BINS = FRAME // 2 + 1
WINDOW = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(FRAME) / (FRAME - 1))
# The estimator the realistic-v2 calibration chose (module docstring).
SMOOTH_FRAMES = 32
QUANTILE = 0.05
BIAS_DB = 0.62
# Fewer live windows than this leaves the 5th percentile unread (one twentieth of 20 windows is one).
MIN_WINDOWS = 20
# The histogram of the moving means, in dB of the print's own scale (a full-scale sine reads about +54 dB).
HIST_LO_DB = -160.0
HIST_HI_DB = 80.0
HIST_STEP_DB = 0.1
HIST_BINS = int(round((HIST_HI_DB - HIST_LO_DB) / HIST_STEP_DB))
# Frames per streamed read.
BLOCK_FRAMES = 1024
PRINT_SUFFIX = ".np.json"
LOG_PREFIX = "    [Cathar] Quantile noise print:"
FALLBACK = "the window pick learns the print"


def print_name(wav_path):
    """The print's file name: named by its method and estimator, so a preserved work directory never hands on another print."""
    return f"noise_{Path(wav_path).stem}_quantile_k{SMOOTH_FRAMES}_q{QUANTILE:g}{PRINT_SUFFIX}"


def frame_magnitudes(samples):
    """`(|rfft| per whole frame, live mask)` of a mono block; a frame is live unless it is a digital mute."""
    count = (len(samples) - FRAME) // HOP + 1
    if count <= 0:
        return np.zeros((0, BINS)), np.zeros(0, dtype=bool)
    frames = np.lib.stride_tricks.sliding_window_view(samples, FRAME)[::HOP][:count]
    rms = np.sqrt(np.mean(frames**2, axis=1))
    held = np.mean(np.diff(frames, axis=1) == 0, axis=1)
    live = (20.0 * np.log10(np.maximum(rms, 1e-12)) > NOISEPRINT_DIGITAL_FLOOR_DBFS) & (held < NOISEPRINT_MUTE_HELD_SHARE)
    return np.abs(np.fft.rfft(frames * WINDOW, axis=1)), live


class MovingQuantile:
    """Per-bin histogram of the SMOOTH_FRAMES-frame moving mean of the magnitudes, fed block by block in order."""

    def __init__(self):
        self.counts = np.zeros(BINS * HIST_BINS, dtype=np.int64)
        self.tail = np.zeros((0, BINS))
        self.tail_live = np.zeros(0, dtype=bool)
        self.windows = 0
        self.frames = 0
        self.muted = 0

    def add(self, magnitudes, live):
        """Counts every window that ends in this block and holds no mute frame; keeps the last frames for the next block."""
        self.frames += len(live)
        self.muted += int(np.count_nonzero(~live))
        mags = np.concatenate((self.tail, magnitudes))
        alive = np.concatenate((self.tail_live, live))
        start = max(len(mags) - (SMOOTH_FRAMES - 1), 0)
        self.tail, self.tail_live = mags[start:], alive[start:]
        if len(mags) < SMOOTH_FRAMES:
            return
        summed = np.cumsum(np.vstack((np.zeros((1, BINS)), mags)), axis=0)
        means = (summed[SMOOTH_FRAMES:] - summed[:-SMOOTH_FRAMES]) / SMOOTH_FRAMES
        dead = np.cumsum(np.concatenate(([0], (~alive).astype(np.int64))))
        clean = (dead[SMOOTH_FRAMES:] - dead[:-SMOOTH_FRAMES]) == 0
        self._count(means[clean])

    def _count(self, means):
        self.windows += len(means)
        if not len(means):
            return
        level = 20.0 * np.log10(np.maximum(means, 1e-12))
        index = np.clip(np.floor((level - HIST_LO_DB) / HIST_STEP_DB), 0, HIST_BINS - 1).astype(np.int64)
        self.counts += np.bincount((index + np.arange(BINS) * HIST_BINS).ravel(), minlength=BINS * HIST_BINS)

    def quantile(self, q=QUANTILE):
        """Per-bin magnitude at quantile q of the counted moving means, linear in dB within a histogram bin."""
        counts = self.counts.reshape(BINS, HIST_BINS)
        cumulative = np.cumsum(counts, axis=1)
        target = q * self.windows
        index = np.argmax(cumulative >= target, axis=1)
        rows = np.arange(BINS)
        below = np.where(index > 0, cumulative[rows, np.maximum(index - 1, 0)], 0)
        fraction = (target - below) / counts[rows, index]
        return 10.0 ** ((HIST_LO_DB + (index + fraction) * HIST_STEP_DB) / 20.0)


def learn(wav_path):
    """`(magnitudes or None, the accumulator)`: the print over the whole file, None under MIN_WINDOWS live windows."""
    acc = MovingQuantile()
    with sf.SoundFile(str(wav_path)) as handle:
        for block in handle.blocks(blocksize=FRAME - HOP + BLOCK_FRAMES * HOP, overlap=FRAME - HOP, dtype="float64", always_2d=True):
            acc.add(*frame_magnitudes(block[:, 0]))
    if acc.windows < MIN_WINDOWS:
        return None, acc
    return acc.quantile() * 10.0 ** (BIAS_DB / 20.0), acc


def write_print(magnitudes, out_path):
    """Writes the print as cathar's JSON list of 1025 magnitudes, through a partial file."""
    with atomic_target(out_path) as partial:
        Path(partial).write_text(json.dumps([float(value) for value in magnitudes]), encoding="utf-8")
    return Path(out_path)


def existing_print(out_path):
    """A print already in the work directory (a resumed run), when it reads as 1025 numbers; None otherwise."""
    try:
        values = json.loads(Path(out_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return Path(out_path) if isinstance(values, list) and len(values) == BINS else None


def quantile_print(wav_path, output_dir, fallback=FALLBACK):
    """The quantile print of wav_path written into output_dir, or None (logged, with what the caller does instead: `fallback`)."""
    out_path = Path(output_dir) / print_name(wav_path)
    if existing_print(out_path) is not None:
        return out_path
    try:
        magnitudes, acc = learn(wav_path)
        if magnitudes is None:
            log_msg(f"{LOG_PREFIX} {acc.windows} live windows, fewer than {MIN_WINDOWS}; {fallback}", level="WARNING")
            return None
        write_print(magnitudes, out_path)
    except (OSError, RuntimeError, ValueError) as exc:
        log_msg(f"{LOG_PREFIX} could not be learned ({exc}); {fallback}", level="WARNING")
        return None
    seconds = SMOOTH_FRAMES * HOP / sf.info(str(wav_path)).samplerate
    log_msg(
        f"{LOG_PREFIX} per-bin {QUANTILE:.0%} quantile of the {seconds:.2f} s moving mean over {acc.windows} windows of "
        f"{acc.frames} frames ({acc.muted} digital-mute frames left out), {BIAS_DB:+.2f} dB, into {out_path.name}"
    )
    return out_path
