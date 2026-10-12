"""The scan's CRT line search on the whole file's averaged spectrum (`crt_detect_mode: welch`).

The default search (`head`, `filters._head_crt_line`) takes one FFT of the first 32768 samples
(0.74 s at 44.1 kHz) and calls a line when the strongest bin of 15450-15900 Hz stands 1.7x
(magnitude) over the band's median. On noise that ratio is already 2.9x, the maximum of 335
Rayleigh-distributed bins against their median, so it fires on any hiss in the band (white
noise 200 times in 200) and names PAL or NTSC by the half of the band the noise peak fell in.
That verdict picks the mains family the hum notch may take
(`filters.MAINS_CANDIDATES_BY_LINE_RATE`): a hissy PAL tape read as NTSC loses its 50 Hz notch.
And it sees only the head. On the IA hunt (`tune_ia_hunt_v3`, 235 sources) it missed two lines
R0 tracks (et-el-extraterrestre, a 23 dB PAL line; smokymountainsundayvhs, a 21 dB NTSC one),
named the wrong standard for two (thevistagroup's NTSC line, 2003-commercial-for-toshiba's PAL
line), and fired on 59 sources R0 finds no line on; the IA corpus clip vid-20220730-091707,
whose PAL line wanders down from about 15700 Hz in its first seconds, reads NTSC.

This search averages the Hann-windowed power spectra of up to `MAX_SEGMENTS` segments of 32768
samples spread evenly over the file (half-overlapping when it is short; a 3 h tape costs 256
FFTs), so a noise bin's spread shrinks with the segments averaged: on white noise the 99.9th
percentile of the band's strongest bin over the band's median, its sides included, is 12.5 dB
on one segment, 7.1 on four, 4.6 on 12 and 2.7 on 39 (15 s), about 14 dB over the square root
of the segments. It calls a line when the band's strongest bin stands `MIN_PROMINENCE_DB` over
the band's median and `MIN_SIDE_DB` over the median of the bins 80-160 away on each side
(108-215 Hz at 44.1 kHz), each at least `NOISE_SCALE_DB` over the root of the segments (1.5
times the noise's 99.9th percentile), and the band holds fewer than `COMB_TEETH` other local
peaks (local maxima over 7 bins), 27 Hz or more from it, within 3 dB of it and over that noise
floor. A codec cut inside the band leaves one side as loud as the peak; a line wandering +-30 Hz,
or a recorded whine spread +-55 Hz by flutter (realistic-v2's transport classes; with the sides
40-120 bins away only 5 of the 20 planted lines were found), keeps both sides clear; a comb is
no line: the BBC iPlayer captures carry a 15703 Hz tone 83 dB out among four more like it, two
renders a 43 Hz comb (44100 / 1024) at 21 dB, and with a tooth near 15734 Hz either reads as
NTSC. The verdict is the nearer nominal rate, as the head search returns.

On the hunt's sources it named R0's standard on 68 of R0's 71 lines (the head 67, and the wrong
standard on two), missed three weak ones (7.0-9.9 dB) and fired on 15 sources R0 finds no line
on, narrow tones such as a 15656 Hz line on five UK tapes and pingu's 15734 Hz line. Read with
the head's mains-family rule, the hum notch moved on 14 sources: toward R0's mains on 8
(terrytoons' 50 Hz hum, 23 dB of evidence, read as NTSC by the head, is notched again; teddy-bear's
60 Hz notch on a 50 Hz tape becomes 50 Hz) and elsewhere on 6, where the mains detector's own
guess, on four clips with 3.2 dB of R0 evidence or less, is no longer held to the family the
head's noise verdict picked. A weak line at a rate (4-7 dB on the averaged spectrum) is not
called: a lower threshold there caught it, and caught 50 Hz buzz harmonics near 15734 Hz on PAL
tapes, which took the 50 Hz notch away from tempo-pre-schools (55.7 dB of R0 evidence). On
realistic-v2 it finds 19 of the 20 planted lines (the head 20) and fires on 51 of the 440
captures without one (the head on 284), 34 of which carry a line R0 tracks from their noise bed
(50-170 Hz off the rates `realistic_defects._without_tones` cuts). White noise of 0.8-60 s never
makes it fire. In cathar's CPU renders of 91 hunt clips it takes out et-el's and toshiba's lines
and notches terrytoons' hum; 9-october-2008-comedy-channel-promo-break-8's 4.1 dB PAL line, not
called (the head notched it by chance), comes out of the denoise 4.6 dB over the source: a line
that stands out of the hiss only after the denoise wants a search after the denoise.
"""

import math
from typing import NamedTuple

import numpy as np
import scipy.ndimage

SEGMENT = 32768
MAX_SEGMENTS = 256
SEARCH_HZ = (15450.0, 15900.0)
MIN_PROMINENCE_DB = 10.0
MIN_SIDE_DB = 6.0
NOISE_SCALE_DB = 21.0
SIDE_BINS = (80, 160)
COMB_TEETH = 5
COMB_APART_BINS = 20
COMB_WITHIN_DB = 3.0
LOCAL_BINS = 7
EPS = 1e-30


class Search(NamedTuple):
    """What the search read: the segments averaged; the band's strongest bin, its prominence over the band's median and
    over its weaker side (dB); and the comb's other teeth."""

    segments: int
    hz: float
    prominence_db: float
    side_db: float
    teeth: int


def averaged_power(mono):
    """`(mean Hann-windowed power spectrum, segments)` of up to `MAX_SEGMENTS` 32768-sample segments spread evenly over `mono`."""
    count = (len(mono) - SEGMENT) // (SEGMENT // 2) + 1
    starts = np.linspace(0, len(mono) - SEGMENT, min(count, MAX_SEGMENTS)).astype(int)
    window = np.hanning(SEGMENT)
    power = np.zeros(SEGMENT // 2 + 1)
    for start in starts:
        stop = start + SEGMENT
        power += np.abs(np.fft.rfft(np.asarray(mono[start:stop], dtype=np.float64) * window)) ** 2
    return power / len(starts), len(starts)


def noise_db(segments):
    """The least any threshold may be on this many averaged segments: `NOISE_SCALE_DB` over their square root."""
    return NOISE_SCALE_DB / math.sqrt(segments)


def _side_db(level, peak, sign):
    """The peak over the median of the bins 80-160 away on one side (`sign` -1 below, +1 above), dB; -inf off the spectrum."""
    near, far = SIDE_BINS
    first, last = (max(index, 0) for index in sorted((peak + sign * near, peak + sign * far)))
    side = level[first:last]
    return float(level[peak] - np.median(side)) if len(side) else -math.inf


def comb_teeth(level, band, peak, floor_db):
    """The band's other local peaks, `COMB_APART_BINS` or more from `peak`, within `COMB_WITHIN_DB` of it and over `floor_db`."""
    local = scipy.ndimage.maximum_filter1d(level, size=LOCAL_BINS) == level
    cut = max(level[peak] - COMB_WITHIN_DB, floor_db)
    others = band[np.abs(band - peak) >= COMB_APART_BINS]
    return int(np.count_nonzero(local[others] & (level[others] >= cut)))


def search(mono, rate):
    """The `Search` of `mono`, or None when it is shorter than a segment or the band lies over Nyquist."""
    if len(mono) < SEGMENT:
        return None
    freqs = np.fft.rfftfreq(SEGMENT, 1.0 / rate)
    band = np.flatnonzero((freqs >= SEARCH_HZ[0]) & (freqs <= SEARCH_HZ[1]))
    if len(band) == 0:
        return None
    power, segments = averaged_power(mono)
    level = 10.0 * np.log10((power + EPS) / (np.median(power[band]) + EPS))
    peak = int(band[np.argmax(level[band])])
    sides = min(_side_db(level, peak, -1), _side_db(level, peak, 1))
    return Search(segments, float(freqs[peak]), float(level[peak]), sides, comb_teeth(level, band, peak, noise_db(segments)))


def is_line(found):
    """Whether the strongest bin is a line: out of the band and of both sides, over the noise floor, and no comb's tooth."""
    if found is None:
        return False
    floor = noise_db(found.segments)
    stands = found.prominence_db >= max(MIN_PROMINENCE_DB, floor) and found.side_db >= max(MIN_SIDE_DB, floor)
    return stands and found.teeth < COMB_TEETH


def verdict(found, line_rates):
    """The nominal rate (of `line_rates`) nearest a line, else 0.0."""
    if not is_line(found):
        return 0.0
    return min(line_rates, key=lambda nominal: abs(found.hz - nominal))


def describe(found, rate_hz):
    """The scan's log line for the search: the verdict and what it rests on."""
    if found is None:
        return "no line (shorter than one 32768-sample segment, or the band over Nyquist)"
    label = f"{rate_hz:.0f} Hz line" if rate_hz else "no line"
    return (
        f"{label} over {found.segments} segments (strongest bin {found.hz:.1f} Hz, {found.prominence_db:.1f} dB over the band, "
        f"{found.side_db:.1f} dB over its sides, {found.teeth} comb teeth)"
    )
