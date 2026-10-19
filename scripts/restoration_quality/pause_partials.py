"""Ear v4, the pause reading's sung runs: true pauses whose source holds partials held across the run.

On a choir tape R4's true pauses (`pause_metrics.true_pause_mask`: the quiet frames inside
DSP-VAD non-speech runs of 200 ms or more) hold soft singing and choir reverb as well as hiss,
and the 3 dB test against the stationary hiss (`hiss_pause.silent_rows`) cannot tell them
apart: on Gaudeamus5 the Mel-RoFormer kept the source's 500-3000 Hz power within 4 dB in runs
holding 927 of the 2,369 true-pause frames (singing, 36.8-92.4 s) and emptied the others by
8 dB or more, and the pause reading read that kept singing as hiss (APL frame fill 21.67 at
the tail, with the reference suppressor 6.43: a render that deleted the singing won). Neither
one 200 ms block's whitened peaks (4.6 dB p95 / median on the singing, speech's pauses
4.6-5.7) nor the output's in-band valleys told them apart (docs/ear_v4_hiss.md).

What does: sung partials hold their pitch across the run, hiss does not. Per VAD non-speech
run of the window that holds true-pause frames, on the source:

1. the non-overlapping 2048-point Hann frames over the run (46 ms, 21.5 Hz bins at
   44.1 kHz: harmonics of a 110 Hz voice stand five bins apart, a choir's spread of pitch
   stays within one to three bins up to 3 kHz);
2. the mean power of the odd frames and of the even frames: disjoint samples, so hiss draws
   independently in each half, while both halves span the whole run, so a note change or a
   glide is the same in both;
3. each half in dB less its running median over 7 bins (150 Hz), 250-4000 Hz (under 250 Hz
   the mains family, over 4 kHz few resolved partials), less the window's tonal bins on 20 ms
   frames (`pause_metrics.tonal_bins`, the reading's own, widened 25 Hz);
4. the file's own fine structure regressed out of each half: the anchor's pause spectrum
   (`Anchor.pause_spectrum`, every pause frame of the file), whitened the same way, fitted to
   each half by least squares and subtracted. A stationary comb (mains hum, buzz, a machine's
   lines) is the same in every run and in the file's pauses, sung partials come and go. The
   test's hum (a 50 Hz comb 20-35 dB over the hiss in these bins, invisible to `tonal_bins`
   on 20 ms frames, where the Hann leakage of its in-phase 1 / k harmonics cancels) read r up
   to 0.77 without this step, 8 of 9 hiss-only runs over 0.3, and at most 0.25 with it;
   excluding the file's tone bins instead (a 1 dB bump over the 7-bin median, or bins steadier
   than noise) left 6-7 flagged. Subtracting the pattern unfitted gave every hiss-only run of a
   choir tape the same negative copy of its recurring pitches (56 % of Gaudeamus5's emptied
   frames flagged); a fitted regression gains nothing on hiss;
5. r, the Pearson correlation of the two residuals. On hiss r scatters around 0 with sd
   0.094-0.099 whatever the run's length (synthetic tape hiss, 4-40 frames, p99.9 0.27-0.28,
   with a hiss or a hummy file's control). A run of 4 frames or more with r >= 0.3 (3 sd)
   leaves the mask.

Short runs hold too few frames to lift weak partials over the noise of each half: on
Gaudeamus5 r read 0.15-0.5 on the sung runs of 4-6 frames, 0.44-0.60 on 13-17 and 0.72-0.84
on 33-37. A window whose sung runs held half its true-pause frames or more is a sung passage,
and its runs too short to be told apart are sung too: it keeps no true pause and the pause
reading reads None there.

Measured against what the neural stage kept (the share of true-pause frames left out, 15 s
windows, hop 7.5): of Gaudeamus5's runs kept within 4 dB (programme) the run test leaves 65.4 %
and the vote 79.3 % in all (735 of the 773 frames of 36-93 s), of those emptied by 8 dB or
more 0 % and 2.5 % (hiss runs inside sung windows); Tele7abc and SOTI 0 % of either; Vaccin
0.8 % of its emptied runs. Rejected on the same proxy: adjacent 200 ms blocks on 20 ms frames (Vaccin's per-scene room colour
correlated 52.6 % of its emptied frames), DPSS tapers (they found Vaccin's weak lines: 67-80 %),
rank correlation (no help with the hum, singing 49 %), a 500 Hz low edge (the hum still read
0.65), and a pooled test of a window's remaining runs (seven Vaccin windows over 0.3).
"""

import numpy as np
import scipy.ndimage

from scripts.restoration_quality import dsp_metrics, pause_metrics

FRAME = 2048
BAND_HZ = (250.0, 4000.0)
WHITEN_BINS = 7
MIN_FRAMES = 4
MIN_BINS = 4
HELD_R = 0.3
SUNG_SHARE = 0.5
TONE_REACH_HZ = 25.0
EPS = 1e-30


def whitened(power):
    """`power` (per bin) in dB less its running median over 7 bins."""
    level = 10.0 * np.log10(np.asarray(power, dtype=np.float64) + EPS)
    return level - scipy.ndimage.median_filter(level, size=WHITEN_BINS, mode="reflect")


def fine_power(segment):
    """`(frames, bins)`: Hann power of the non-overlapping 2048-point frames of `segment`."""
    count = len(segment) // FRAME
    frames = np.asarray(segment[: count * FRAME], dtype=np.float64).reshape(count, FRAME)
    return np.abs(np.fft.rfft(frames * np.hanning(FRAME), axis=1)) ** 2


def near(freqs, tone_hz, reach):
    """The bins within `reach` Hz of any frequency in `tone_hz`."""
    tones = np.asarray(tone_hz, dtype=np.float64)
    if tones.size == 0:
        return np.zeros(len(freqs), dtype=bool)
    return np.min(np.abs(freqs[:, np.newaxis] - tones[np.newaxis, :]), axis=1) <= reach


def residual(values, control):
    """`values` less their mean and their least-squares fit on `control` (a constant control fits nothing)."""
    values, control = values - values.mean(), control - control.mean()
    scale = np.sum(control**2)
    return values - control * (np.sum(values * control) / scale) if scale > 0.0 else values


def correlation(first, second):
    """Pearson r of two vectors; 0 when either is constant."""
    first, second = first - first.mean(), second - second.mean()
    scale = np.sqrt(np.sum(first**2) * np.sum(second**2))
    return float(np.sum(first * second) / scale) if scale > 0.0 else 0.0


def split_correlation(power, keep, control):
    """r of the odd and the even frames' whitened mean spectra over `keep`, `control` regressed out of both.

    None under 4 frames or 4 bins.
    """
    if len(power) < MIN_FRAMES or keep.sum() < MIN_BINS:
        return None
    odd, even = (residual(whitened(power[first::2].mean(axis=0))[keep], control[keep]) for first in (0, 1))
    return correlation(odd, even)


def file_control(spectrum, bins):
    """The file's pause spectrum whitened (`bins` long), or zeros (nothing to regress) without one of that length."""
    if spectrum is None or len(spectrum) != bins:
        return np.zeros(bins)
    return whitened(spectrum)


def judged_bins(source, rate, mask):
    """The fine bins the run test reads: 250-4000 Hz, less the window's tonal bins (20 ms frames, +-25 Hz)."""
    frame = max(1, int(pause_metrics.FRAME_S * rate))
    freqs20, power20, _level = dsp_metrics.framed_psd(np.asarray(source, dtype=np.float64), rate, frame)
    tones20 = freqs20[pause_metrics.tonal_bins(power20[: len(mask)][mask])]
    freqs = np.fft.rfftfreq(FRAME, 1.0 / rate)
    return (freqs >= BAND_HZ[0]) & (freqs < BAND_HZ[1]) & ~near(freqs, tones20, TONE_REACH_HZ)


def run_correlations(source, rate, pauses, spectrum=None):
    """`[(start, stop, r)]` of every VAD run holding true-pause frames (20 ms frame spans; r None under 4 frames).

    `pauses` is `(mask, runs)` from `pause_metrics.true_pause_mask` on `source`; `spectrum` the anchor's pause spectrum.
    """
    mask, runs = pauses
    frame = max(1, int(pause_metrics.FRAME_S * rate))
    keep = judged_bins(source, rate, mask)
    control = file_control(spectrum, len(keep))
    spans = [(start, stop) for start, stop in pause_metrics.run_spans(runs) if mask[start:stop].any()]
    return [(start, stop, split_correlation(fine_power(source[start * frame : stop * frame]), keep, control)) for start, stop in spans]


def held(found):
    """Whether one run's r says it holds partials: read, and 0.3 or more."""
    return found is not None and found >= HELD_R


def unsung(source, rate, pauses, spectrum=None):
    """R4's mask without its sung runs; empty when they held half its frames or more (a sung passage).

    A mask under 25 frames comes back as it is: it reads nothing anyway.
    """
    mask = np.asarray(pauses[0], dtype=bool)
    if mask.sum() < pause_metrics.MIN_PAUSE_FRAMES:
        return mask
    out = mask.copy()
    for start, stop, found in run_correlations(source, rate, (mask, pauses[1]), spectrum):
        out[start:stop] &= not held(found)
    return out if mask.sum() - out.sum() < SUNG_SHARE * mask.sum() else np.zeros_like(mask)
