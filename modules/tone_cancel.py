"""Cancellation of persistent lines that are not the mains series: a recorded whine, a buzz, a whistle off the notch.

A line is a defect when it is there whether or not the programme is: it stands over its
neighbourhood in the long-term spectrum, and it stands there in the quietest fifth of the
frames too, where a note of the programme does not. Each such line is cancelled by the hum
canceller's tracker at the frequency it sits at, with a bandwidth that widens above the
speech range, where a whine under the transport's flutter wanders and there is no
programme on a linear track to protect.

What this leaves alone, by design: the mains series (the hum canceller's, found by its own
detector), the CRT line whistle inside the pre-conditioning notch's reach, the sticky-shed
squeal (three per cent of wander at three hertz is a hundred hertz of excursion at two
kilohertz, past any narrow tracker, and a wide one would take the formants under it), and
the enclosure resonance (a peaking filter on the programme, not a line added to it) -- the
detector reads no persistent line in any of them and the stage skips.
"""

from pathlib import Path

import numpy as np
import scipy.ndimage
import scipy.signal

from . import hum_cancel
from .config import APL_ENABLE_TONE_CANCEL, APL_HUM_MAX_HARMONICS
from .spectral_denoise import MAINS_DETECT_MIN_SAMPLES, _scannable_mono, detect_mains_hz
from .utils import log_msg

WINDOW = 16384
HOP = 8192
# A line stands this far over its smoothed neighbourhood, in this fraction of the frames
# and of the quietest frames alike.
LINE_EXCESS_DB = 8.0
PERSISTENCE = 0.5
QUIET_FRACTION = 0.2
FLOOR_BINS = 61
# Lines closer than this merge into one; at most this many are cancelled.
MERGE_BINS = 5
MAX_LINES = 16
# Bandwidths: narrow below the speech range's top, wide above it where a recorded whine
# wanders under flutter and no programme needs protecting on a linear track.
LOW_BANDWIDTH_HZ = 2.0
HIGH_BANDWIDTH_HZ = 40.0
HIGH_FROM_HZ = 8000.0
CENTROID_BINS = 10
# The tracker's grid: a quarter of the hum canceller's window and hop, so the frame rate of
# 172 Hz carries the wide band and a line that moves thirty hertz -- a line whistle under
# the transport's wow -- still reads as one line inside a 23 ms frame.
TRACK_WINDOW = 1024
TRACK_HOP = 256
# Left to other stages: the mains series within this many bins of a harmonic, and the
# CRT line rates within this of their nominal frequencies.
MAINS_GUARD_BINS = 2
CRT_LINE_RATES_HZ = (15625.0, 15734.0)
CRT_GUARD_HZ = 40.0
STAGE_FAILURES = (OSError, RuntimeError, ValueError, MemoryError)


def _spectrogram(mono_signal, sample_rate):
    """The power spectral density per frame, in V^2/Hz: (bins x frames), and the bin frequencies."""
    freqs, _times, spectrum = scipy.signal.stft(mono_signal, sample_rate, nperseg=WINDOW, noverlap=WINDOW - HOP, scaling="psd")
    return freqs, np.abs(spectrum) ** 2


def _line_excess_db(power):
    """Each bin's long-term median level over its smoothed neighbourhood, in dB."""
    median = np.median(power, axis=1)
    floor = scipy.ndimage.median_filter(median, size=FLOOR_BINS, mode="nearest") + 1e-30
    return 10.0 * np.log10((median + 1e-30) / floor), floor


def _persistence(power, floor):
    """The fraction of all frames, and of the quietest frames, in which each bin stands clear of its floor."""
    clear = power > 4.0 * floor[:, None]
    level = power.sum(axis=0)
    quiet = level <= np.quantile(level, QUIET_FRACTION)
    return clear.mean(axis=1), clear[:, quiet].mean(axis=1)


def _guarded(freqs, mains_hz):
    """The bins another stage owns: the mains series and the CRT line rates."""
    guarded = np.zeros(len(freqs), dtype=bool)
    bin_hz = freqs[1] - freqs[0]
    for harmonic in range(1, APL_HUM_MAX_HARMONICS + 1) if mains_hz else ():
        guarded |= np.abs(freqs - harmonic * mains_hz) <= MAINS_GUARD_BINS * bin_hz
    for line_rate in CRT_LINE_RATES_HZ:
        guarded |= np.abs(freqs - line_rate) <= CRT_GUARD_HZ
    return guarded


def _merged_peaks(candidates, excess):
    """The strongest bin of each run of candidate bins closer than the merge span."""
    indices = np.flatnonzero(candidates)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) >= MERGE_BINS) + 1)
    return [int(group[np.argmax(excess[group])]) for group in groups]


def _refined_hz(freqs, spectrum, index):
    """The line's frequency between bins: the peak's curvature below the wide band, its centroid above it.

    A line under flutter leaves a plateau in the long-term spectrum rather than a peak, and
    the plateau's centre is what a wide tracker should sit on.
    """
    if index in (0, len(spectrum) - 1):
        return float(freqs[index])
    if freqs[index] >= HIGH_FROM_HZ:
        low, high = max(index - CENTROID_BINS, 0), index + CENTROID_BINS + 1
        weights = spectrum[low:high]
        return float(np.sum(freqs[low:high] * weights) / (np.sum(weights) + 1e-30))
    before, after = index - 1, index + 2
    left, centre, right = np.log(spectrum[before:after] + 1e-30)
    denominator = left - 2.0 * centre + right
    shift = 0.0 if denominator >= 0.0 else 0.5 * (left - right) / denominator
    return float(freqs[index] + shift * (freqs[1] - freqs[0]))


def detect_lines(mono_signal, sample_rate, mains_hz=0.0):
    """The persistent lines of a recording as (frequency, floor) pairs, strongest first, at most MAX_LINES."""
    if len(mono_signal) < 4 * WINDOW:
        return []
    freqs, power = _spectrogram(mono_signal, sample_rate)
    excess, floor = _line_excess_db(power)
    overall, quiet = _persistence(power, floor)
    candidates = (excess >= LINE_EXCESS_DB) & (overall >= PERSISTENCE) & (quiet >= PERSISTENCE)
    # Another stage's line is dropped by its peak, after merging: a guard punched into the
    # candidates would leave the line's own skirts standing as lines of their own.
    guarded = _guarded(freqs, mains_hz)
    peaks = [index for index in _merged_peaks(candidates, excess) if not guarded[index]]
    ranked = sorted(peaks, key=lambda index: -excess[index])[:MAX_LINES]
    median = np.median(power, axis=1)
    return [(_refined_hz(freqs, median, index), float(floor[index])) for index in ranked]


def bandwidths_for(lines_hz):
    """Narrow inside the speech range's reach, wide above it."""
    return [LOW_BANDWIDTH_HZ if hz < HIGH_FROM_HZ else HIGH_BANDWIDTH_HZ for hz in lines_hz]


def cancel_tones(source_wav, target_wav, lines):
    """Cancels the given lines; returns the target, or None when the recording is too short to track."""
    freqs = [hz for hz, _floor in lines]
    floors = [floor for _hz, floor in lines]
    return hum_cancel.cancel_lines(source_wav, target_wav, freqs, bandwidths_for(freqs), floors, hop=TRACK_HOP, window=TRACK_WINDOW)


def _plan(source_wav):
    """The lines to cancel, or a reason to skip."""
    mono_signal, sample_rate = _scannable_mono(source_wav)
    if sample_rate is None or len(mono_signal) < MAINS_DETECT_MIN_SAMPLES:
        return None, "unreadable"
    mains_hz = detect_mains_hz(source_wav) or 0.0
    lines = detect_lines(mono_signal, sample_rate, mains_hz)
    if not lines:
        return None, "no persistent line"
    return lines, None


def _cancelled(source_wav, audio_dir, lines):
    """The cancelled file, or None with the reason logged."""
    output_dir = Path(audio_dir) / "tone_cancel"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        produced = cancel_tones(source_wav, output_dir / f"tonecancel_{Path(source_wav).name}", lines)
    except STAGE_FAILURES as exc:
        log_msg(f"    [Tone Cancel] Skipped after failure: {exc}")
        return None
    if produced is None:
        log_msg("    [Tone Cancel] Skipped: too short to track.")
    return produced


def apply_when_needed(source_wav, audio_dir, strategy=None):
    """Cancels the persistent non-mains lines a recording carries; returns the new path, or the input untouched."""
    del strategy
    if not APL_ENABLE_TONE_CANCEL:
        return source_wav
    lines, reason = _plan(source_wav)
    if lines is None:
        log_msg(f"    [Tone Cancel] Skipped: {reason}.")
        return source_wav
    produced = _cancelled(source_wav, audio_dir, lines)
    if produced is None:
        return source_wav
    log_msg(f"    [Tone Cancel] Cancelled {len(lines)} lines: " + ", ".join(f"{hz:.0f} Hz" for hz, _floor in lines) + ".")
    return produced
