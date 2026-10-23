"""File-level readings no window sees: the level riding under the speech, and the sync drifting.

Every paired reading runs on a lag-aligned, gain-matched pair (`audio_io.align_pair`), so two
faults a listener hears on a whole tape are invisible to them. Both readings take the RAW mono
pair at one rate (the runner's `Pair` resamples the output to the source's rate first: the
Tele7abc source is 48 kHz, its outputs 44.1 kHz). `file_entries` reads the sync first and
hands the ride its median lag and its lag track (the matched anchors' lags at their centres),
so a sync fault does not read as a ride as well: neither a constant offset nor a drift.

Gain riding (`file.gain_ride_lu`, R7 of ear v3)
-----------------------------------------------
ffmpeg's loudnorm leaves linear mode for its dynamic mode whenever the true-peak rule or the
loudness-range target forbids one static gain, and the dynamic mode rides the level with a
3 s window; a least-squares gain match per 15 s window absorbs most of that ride. The reading
is the EBU short-term loudness (BS.1770 K-weighting with pyloudnorm's filters on 100 ms
blocks, 3 s windows every 1 s, ungated mean square) of the source and of the output, the
output shifted by the lag (timing only: still raw, not gain-matched), differenced per window
over the windows where the source carries programme, minus the median difference (the
static gain, the mastering stage's job). It reports the spread of what is left: `std_lu`,
`p95_abs_lu` (the headline) and the p10-p90 range of its 1 s steps (`deriv_p10_p90`, LU/s),
with the number of windows read; None under five windows.

A window carries programme when it is above -70 LUFS, within 10 LU of the power mean of
those, and 10 LU over the source's noise floor (the 10th percentile of its 100 ms blocks, the
level between words, held at -70 LUFS or above). The programme gate alone leaves out a pause
only when the floor sits more than 10 LU under the speech: on 96 s of synthetic speech in 8 s
turns with 8 s pauses under a 1e-2 hiss (short-term -37.4..-26.2 LUFS) it kept 94/94
windows, and removing the hiss read 31.5 LU p95, a -12 dB pause expander 11.5 LU, which is
expander depth, not riding. With the floor gate 8 windows remain and read 0.06 and 0.02 LU.
A window 10 LU over the floor loses at most 0.46 LU when its whole floor is removed
(10 log10(1 - 0.1)), so no denoiser reads as more than that. Where nothing stands 10 LU over
the floor the reading is None rather than a denoiser's depth: dense music whose quietest
tenth is still music (3 of the 11 15 s clips of the v2 set; the old gate read 5.2 LU on
APL's output of one of them).

Measured on synthetic speech (random syllables, 16 and 44.1 kHz): identity and a static
-6 dB read 0, a denoiser that removes a 3e-3 hiss reads 0.03 LU p95, a +-3 dB rider at
0.2 Hz reads 2.2 LU p95 (std 1.2, steps 3.9 LU/s; the 3 s window halves a 5 s period's
swing). A late copy read without the lag rides 0.02 / 0.19 / 0.43 / 0.93 / 2.03 LU p95 at
5 / 40 / 100 / 300 / 800 ms, and 0 with it.

A speed error the sync stage missed is no constant lag: once the median lag is out the output
is still early by up to half the drift near the start and late by up to half near the end,
and each 3 s window compares the source with a shifted output. Calibration v3 (2026-10-09)
read `sync_drift` 0.05 / 0.1 / 0.2 % slow on the 300 s Tata cuts (146 / 293 / 587 ms of
drift) as a ride of 0.41 / 0.80 / 1.34 LU p95 (Vaccin; SOTI 0.24 / 0.42 / 1.06, Tele7abc's
134 s 0.12 / 0.19 / 0.37), over the 0.1 LU the table allows a speed error. So each output
window is read along the lag track (`lag_track`: linear between the matched anchors and past
the end ones on the end segments' slopes, a constant with one anchor), on a grid of the
largest divisor of the 100 ms block within 1 ms (`track_step`), so a window the track does
not move starts where the block grid starts it. Those cases then read 0.008 / 0.014 / 0.031
(Vaccin), 0.008 / 0.015 / 0.032 (SOTI), 0.009 / 0.017 / 0.021 (Tele7abc) and 0.006 / 0.009 /
0.010 LU (en); 200 s of synthetic speech 0.2 / 0.5 % slow (382 / 958 ms) 0.62 / 1.11 LU with
the median lag alone, 0.021 / 0.049 along the track, and the +-3 dB rider 0.2 % slow still
reads 2.27 (2.25 without the drift). A track that moves no window by half a step (0.48 ms at
44.1 kHz, 0.5 ms at 48 kHz; the v2 finals drift 0.02-0.26 ms, the round-0 files read for this
0.07-0.44 ms) leaves the block grid's reading exactly as it was; the calibration's
`loudnorm_ride` and benign cases read the same to the fourth decimal.

Measured on the v2 finals the user accepted
(round 2, "pauses natural on both"), p95 APL / cathar: Tele7abc 1.13 / 1.55, SOTI
0.96 / 0.82, Vaccin 1.80 / 1.66, Gaudeamus5 (music) 0.22 / 0.54; the floor gate drops 0-5
of their 131-458 windows. The design's 1.5 LU flag [S] would fire on three accepted files,
so it stays display-only until the `loudnorm_ride` degradation calibrates it. On Vaccin the
deviation falls 0.09-0.14 LU per LU of source level: the quiet windows come up, which is
compression (hiss removal would push them down).

Sync (`file.sync_drift_ms`, `file.sync_offset_ms`, `file.sync_unmatched`)
-------------------------------------------------------------------------
The output's lag behind the source is the peak of the Pearson correlation of an 8 s source
anchor's 300-3000 Hz amplitude envelope (1 ms frames, 10 ms smoothing, parabolic peak) slid
along the output's envelope. The anchors sit at the file's start, centre and end (a third of
the file each under 24 s); the search widens with the anchor's place in the file, 2.5 % of
its end time held to 2-30 s, so a 2 % speed error is measured at least 20 min into the
file and a 1.5 s shift anywhere. Inside an 8 s anchor a speed error moves the lag too: the peak reads
0.96-0.97 at 0.5 % slow, 0.89-0.92 at 1 %, 0.76-0.81 at 2 %, and 0.09-0.11 for noise,
0.28-0.41 for other synthetic speech, so a match needs 0.5.

An anchor is read only when it carries programme with timing in it, else the next 8 s
inward is tried (up to 16 per third, so a blank leader moves the start anchor in):

- its K-weighted level 10 LU over the floor. Of the 748 anchors (stepped 4 s) of the 32 v2
  pairs, the 716 over it all peaked at 0.90-1.00 (two at a wrong lag: the burst below); of
  the 32 under it, 3 peaked under 0.5 and 4 at a wrong lag (a denoiser had removed the
  low-level content they held);
- its envelope changes within one PAL frame: self-similarity 40 ms apart under 0.9. Speech
  anchors on the Tata tapes read at most 0.75; a lone sustained burst heading an otherwise
  silent 15 s clip read 0.95 and matched an output that trails by 5 ms at 111 ms (its
  envelope correlating 0.93-0.98 from 0 to 150 ms), a false sync fault on an accepted file;
- a finite source envelope.

The readings: the matched lags (`lag_start_ms` ...), `drift_ms`, the largest change between
them (a sync stage that shifts once by the start's lag, or a DTW that warps the wrong way,
leaves a lag that grows along the tape; one PAL video frame, 40 ms, of change is where the
picture and the lips part), `offset_ms`, the largest |lag| (a constant shift is no drift but
still parts lips from picture), and `unmatched`, the anchors that carried programme with
timing whose output did not match (peak under 0.5 or not finite): a broken sync stage, a
rate mislabel, an output with NaN. A gate reads a None drift as skipped, so `unmatched` is
the reading that fails those. `drift_ms` spans the anchors' centres, the duration less 8 s
(two thirds of it under 24 s), so it is a lower bound of the start-to-end drift: 92 ms for
0.1 % slow over 100 s. On 200 s of synthetic speech 0.1 / 0.2 / 0.5 / 1 / 2 % slow read
191 / 383 / 958 / 1919 / 3830 ms of drift, 5 % slow reads 3 unmatched, a constant 1.5 s
shift reads a 1500 ms offset and no drift, 3 s reads 3000 ms at the end and 2 unmatched.
On the v2 finals every matched lag is 4.7-5.4 ms on all 32 pairs, drift 0.02-0.26 ms, none
unmatched; dense music clips read no anchor.

The K-weighting runs in 60 s chunks carrying the filter state, so a three-hour capture never
needs a float64 copy of itself; the sync reading touches only its anchors and their search.
Both readings together take 1.0 s on the 7.7 min Vaccin pair.
"""

import numpy as np
import pyloudnorm
import scipy.ndimage
import scipy.signal

BLOCK_S = 0.1
BLOCKS_PER_HOP = 10
SHORT_TERM_HOPS = 3
CHUNK_S = 60.0
LUFS_OFFSET = -0.691
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
FLOOR_PERCENTILE = 10.0
FLOOR_MARGIN_LU = 10.0
MIN_RIDE_WINDOWS = 5
# The output is read along the sync's lag track on a grid of at most 1 ms (`track_step`).
TRACK_STEP_S = 0.001
RIDE_PERCENTILE = 95.0
STEP_PERCENTILES = (10.0, 90.0)
RIDE_NAMES = ("std_lu", "p95_abs_lu", "deriv_p10_p90")

SYNC_WINDOW_S = 8.0
MIN_SYNC_WINDOW_S = 2.0
MAX_ANCHOR_TRIES = 16
TIMING_LAG_S = 0.04
MAX_SELF_SIMILARITY = 0.9
SYNC_SEARCH_S = (2.0, 30.0)
SYNC_SEARCH_RATE = 0.025
SYNC_BAND_HZ = (300.0, 3000.0)
ENVELOPE_FRAME_S = 0.001
ENVELOPE_SMOOTH_FRAMES = 10
MIN_ENVELOPE_CORR = 0.5
SEGMENT_NAMES = ("start", "middle", "end")
MATCHED, UNREAD, UNMATCHED = "matched", "unread", "unmatched"
EPS = 1e-20


# Short-term loudness ---------------------------------------------------------------------


def _k_stages(rate):
    """BS.1770's K-weighting as pyloudnorm builds it: a +4 dB shelf at 1.5 kHz, then a 38 Hz high-pass."""
    return (
        pyloudnorm.IIRfilter(4.0, 1.0 / np.sqrt(2.0), 1500.0, rate, "high_shelf"),
        pyloudnorm.IIRfilter(0.0, 0.5, 38.0, rate, "high_pass"),
    )


def hop_energies(mono, rate, hop):
    """The K-weighted energy of each whole `hop` of `mono`, filtered in 60 s chunks that carry the filter state."""
    stages = _k_stages(rate)
    states = [np.zeros(max(len(stage.a), len(stage.b)) - 1) for stage in stages]
    count = len(mono) // hop
    step = max(1, int(round(CHUNK_S * rate / hop)))
    energies = np.zeros(count)
    for first in range(0, count, step):
        last = min(count, first + step)
        data = np.asarray(mono[slice(first * hop, last * hop)], dtype=np.float64)
        for index, stage in enumerate(stages):
            data, states[index] = scipy.signal.lfilter(stage.b, stage.a, data, zi=states[index])
            data = data * stage.passband_gain
        energies[first:last] = np.sum(data.reshape(last - first, hop) ** 2, axis=1)
    return energies


def block_energies(mono, rate):
    """The K-weighted energy of each 100 ms block and the block length in samples."""
    block = int(round(BLOCK_S * rate))
    return hop_energies(mono, rate, block), block


def lufs(mean_square):
    """BS.1770 loudness of a K-weighted mean square."""
    return LUFS_OFFSET + 10.0 * np.log10(np.maximum(mean_square, EPS))


def short_term_levels(energies, block):
    """EBU short-term loudness from 100 ms block energies: 3 s windows every 1 s, ungated; empty under one window."""
    hops = len(energies) // BLOCKS_PER_HOP
    if hops < SHORT_TERM_HOPS:
        return np.zeros(0)
    hop_energy = energies[: hops * BLOCKS_PER_HOP].reshape(hops, BLOCKS_PER_HOP).sum(axis=1)
    window = np.convolve(hop_energy, np.ones(SHORT_TERM_HOPS), mode="valid")
    return lufs(window / (SHORT_TERM_HOPS * BLOCKS_PER_HOP * block))


def short_term_loudness(mono, rate):
    """EBU short-term loudness in LUFS of `mono`: 3 s windows every 1 s, ungated."""
    return short_term_levels(*block_energies(mono, rate))


def noise_floor(energies, block):
    """The 10th percentile of the finite 100 ms block loudness (the floor between words), held at -70 LUFS or above."""
    levels = lufs(np.asarray(energies) / block)
    levels = levels[np.isfinite(levels)]
    if levels.size == 0:
        return ABSOLUTE_GATE_LUFS
    return float(max(np.percentile(levels, FLOOR_PERCENTILE), ABSOLUTE_GATE_LUFS))


def source_floor(source, rate):
    """The source's noise floor in LUFS (`noise_floor` of its 100 ms blocks)."""
    return noise_floor(*block_energies(source, rate))


def speech_active(levels, floor=-np.inf):
    """Windows carrying programme: above -70 LUFS, within 10 LU of the power mean of those and 10 LU over the floor."""
    above = levels > ABSOLUTE_GATE_LUFS
    if not above.any():
        return above
    relative = 10.0 * np.log10(np.mean(10.0 ** (levels[above] / 10.0))) + RELATIVE_GATE_LU
    return above & (levels >= max(relative, floor + FLOOR_MARGIN_LU))


# Gain ride -------------------------------------------------------------------------------


def shifted(source, output, lag_samples):
    """The pair on one timeline without a copy: a late output (positive lag) loses its head, an early one the source's."""
    lag = int(lag_samples)
    if lag >= 0:
        return source, output[lag:]
    return source[-lag:], output


def track_step(rate):
    """The output's read grid along a lag track: the largest divisor of the 100 ms block within 1 ms of samples.

    It must divide the block, so a window the track does not move starts exactly where the
    block grid starts it (44.1 kHz: 42 samples, 48 kHz: 48, 22.05 kHz: 21, 16 kHz: 16).
    """
    block = int(round(BLOCK_S * rate))
    limit = max(1, int(TRACK_STEP_S * rate))
    return max(size for size in range(1, limit + 1) if block % size == 0)


def window_starts(count, rate):
    """The sample where each of `count` short-term windows starts, on the 1 s hop of 100 ms blocks."""
    return np.arange(count) * BLOCKS_PER_HOP * int(round(BLOCK_S * rate))


def window_levels(mono, rate, starts):
    """The 3 s K-weighted short-term loudness of `mono` from each of `starts` (samples, rounded to `track_step`); NaN off either end.

    One energy per step and a cumulative sum: a three-hour capture holds one float64 per
    millisecond (86 MB at 44.1 kHz), never a float64 copy of itself.
    """
    step = track_step(rate)
    energies = hop_energies(mono, rate, step)
    span = SHORT_TERM_HOPS * BLOCKS_PER_HOP * int(round(BLOCK_S * rate)) // step
    cumulative = np.concatenate(([0.0], np.cumsum(energies)))
    first = np.rint(np.asarray(starts, dtype=np.float64) / step).astype(np.int64)
    inside = (first >= 0) & (first + span <= len(energies))
    first = np.where(inside, first, 0)
    total = cumulative[np.minimum(first + span, len(energies))] - cumulative[first]
    return np.where(inside, lufs(total / (span * step)), np.nan)


def lag_track(track, times_s):
    """The output's lag in seconds at each of `times_s` (source time) from the matched anchors `[(centre_s, lag_ms), ...]`.

    Linear between the anchors and past the end ones along the end segments' slopes (a speed
    error keeps drifting beyond them); one anchor is a constant lag, none is 0.
    """
    times = np.asarray(times_s, dtype=np.float64)
    if not track:
        return np.zeros(len(times))
    centres, lags = (np.array(column, dtype=np.float64) for column in zip(*sorted(track)))
    lags = lags / 1000.0
    if len(centres) == 1:
        return np.full(len(times), lags[0])
    return _extended(centres, lags, times)


def _extended(centres, lags, times):
    """`np.interp` inside the anchors, the first and last segments' slopes carried past them."""
    head = lags[0] + (times - centres[0]) * _slope(centres[:2], lags[:2])
    tail = lags[-1] + (times - centres[-1]) * _slope(centres[-2:], lags[-2:])
    return np.where(times < centres[0], head, np.where(times > centres[-1], tail, np.interp(times, centres, lags)))


def _slope(centres, lags):
    """The lag's change per second between two anchors; 0 for two anchors at one time."""
    span = float(centres[1] - centres[0])
    return float(lags[1] - lags[0]) / span if span > 0.0 else 0.0


def window_offsets(track, lag_samples, rate, count):
    """Per short-term window, how far (samples) the lag track puts the output from where the median lag shift left it.

    The track's times are the raw source's; an early output (negative lag) cut the source's
    head, so the shifted source's window centres sit that much later on the raw timeline. With
    no matched anchor the shift is all there is: every offset is 0.
    """
    if not track:
        return np.zeros(count)
    block = int(round(BLOCK_S * rate))
    centres = window_starts(count, rate) + SHORT_TERM_HOPS * BLOCKS_PER_HOP * block / 2.0 + max(0, -int(lag_samples))
    return lag_track(track, centres / rate) * rate - int(lag_samples)


def output_levels(output, rate, offsets):
    """The output's short-term levels against each source window: on the block grid, or along `offsets` once one moves a step.

    A track that moves no window by half a step or more (every v2 final and judged file:
    drift under 0.5 ms) leaves the block grid's reading exactly as it was.
    """
    if not np.any(np.abs(offsets) >= 0.5 * track_step(rate)):
        return short_term_loudness(output, rate)
    return window_levels(output, rate, window_starts(len(offsets), rate) + offsets)


def gain_ride(source, output, rate, lag_samples=0, track=()):
    """How far the output's short-term loudness wanders from the source's once the static gain is taken out.

    `lag_samples` is the output's lag behind the source (`sync_drift`'s median lag), taken out by
    timing only; `track` is the sync's matched anchors (`sync_drift`'s `"track"`), along which each
    output window is read where a speed error moved it (module docstring).
    `{"std_lu", "p95_abs_lu", "deriv_p10_p90", "offset_lu", "windows"}`; the readings are None when
    fewer than five windows carry programme. A window whose output runs off either end is not read.
    """
    src, out, active = paired_levels(source, output, rate, (lag_samples, track))
    windows = int(active.sum())
    if windows < MIN_RIDE_WINDOWS:
        return {**dict.fromkeys(RIDE_NAMES), "offset_lu": None, "windows": windows}
    return {**ride_readings(np.maximum(out, ABSOLUTE_GATE_LUFS) - src, active), "windows": windows}


def paired_levels(source, output, rate, sync):
    """`(source levels, output levels, active)` over the windows both sides carry; `sync = (lag_samples, track)`.

    A window is active when the source carries programme there (`speech_active`) and the
    output's window, read along the track, lies inside the output.
    """
    lag_samples, track = sync
    source, output = shifted(source, output, lag_samples)
    energies, block = block_energies(source, rate)
    src = short_term_levels(energies, block)
    out = output_levels(output, rate, window_offsets(track, lag_samples, rate, len(src)))
    count = min(len(src), len(out))
    return src[:count], out[:count], speech_active(src[:count], noise_floor(energies, block)) & np.isfinite(out[:count])


def ride_readings(deviation, active):
    """The ride's spread once the static gain (the median deviation over the active windows) is out."""
    offset = float(np.median(deviation[active]))
    ride = np.where(active, deviation - offset, np.nan)
    kept = ride[active]
    return {
        "std_lu": float(np.std(kept)),
        "p95_abs_lu": float(np.percentile(np.abs(kept), RIDE_PERCENTILE)),
        "deriv_p10_p90": step_spread(ride),
        "offset_lu": offset,
    }


def step_spread(ride):
    """p90 - p10 of the 1 s steps between consecutive active windows, or None with fewer than two steps."""
    steps = np.diff(ride)
    steps = steps[np.isfinite(steps)]
    if len(steps) < 2:
        return None
    low, high = np.percentile(steps, STEP_PERCENTILES)
    return float(high - low)


# Sync ------------------------------------------------------------------------------------


def envelope(mono, rate, frame):
    """The 300-3000 Hz amplitude envelope, one value per `frame` samples, smoothed over 10 frames, mean removed.

    A slice no longer than the zero-phase filter's edge padding (`edge_padding`: 27 samples
    here, more than one 1 ms frame below 28 kHz) cannot be filtered: it reads zero, the
    envelope's mean, the value `_padded_envelope` pads with off the file's end. An output
    that ends one frame into an anchor's search then reads that anchor unmatched; at 16 kHz
    it raised and cost the whole dsp family.
    """
    band = (SYNC_BAND_HZ[0], min(SYNC_BAND_HZ[1], 0.45 * rate))
    sos = scipy.signal.butter(4, band, btype="bandpass", fs=rate, output="sos")
    if len(mono) <= edge_padding(sos):
        return np.zeros(len(mono) // frame)
    filtered = scipy.signal.sosfiltfilt(sos, np.asarray(mono, dtype=np.float64))
    count = len(filtered) // frame
    power = np.mean(filtered[: count * frame].reshape(count, frame) ** 2, axis=1)
    # The running mean can dip a hair under zero on float rounding; the envelope cannot.
    smooth = np.sqrt(np.maximum(scipy.ndimage.uniform_filter1d(power, ENVELOPE_SMOOTH_FRAMES, mode="nearest"), 0.0))
    return smooth - smooth.mean()


def edge_padding(sos):
    """`scipy.signal.sosfiltfilt`'s default edge padding for `sos`: an input must be longer than this."""
    taps = 2 * len(sos) + 1 - min(int(np.sum(sos[:, 2] == 0)), int(np.sum(sos[:, 5] == 0)))
    return 3 * taps


def sliding_correlation(template, region):
    """Pearson correlation of `template` with every same-length stretch of `region` (0 where the stretch is flat)."""
    centred = template - template.mean()
    count = len(centred)
    dots = scipy.signal.correlate(region, centred, mode="valid", method="fft")
    sums = np.concatenate(([0.0], np.cumsum(region)))
    squares = np.concatenate(([0.0], np.cumsum(region**2)))
    local = sums[count:] - sums[:-count]
    variance = np.maximum(squares[count:] - squares[:-count] - local**2 / count, 0.0)
    scale = np.sum(centred**2)
    flat = variance <= 1e-9 * max(float(np.max(variance)), EPS)
    return np.where(flat, 0.0, dots / np.sqrt(scale * np.where(flat, 1.0, variance) + EPS))


def parabolic_offset(values, peak):
    """Sub-frame offset of the peak at `peak` from a parabola through it and its neighbours."""
    if peak <= 0 or peak >= len(values) - 1:
        return 0.0
    curvature = values[peak - 1] - 2.0 * values[peak] + values[peak + 1]
    if curvature >= 0.0:
        return 0.0
    return float(0.5 * (values[peak - 1] - values[peak + 1]) / curvature)


def anchor_candidates(count, rate):
    """The anchor length and its candidate starts: forward from the file's start, outward from its centre, back from its end.

    Each group covers about a third of the file in whole anchors, at most 16 (2 min); a file whose
    third is under 2 s has none.
    """
    seconds = min(SYNC_WINDOW_S, count / float(rate) / 3.0)
    if seconds < MIN_SYNC_WINDOW_S:
        return 0, ((), (), ())
    length = int(seconds * rate)
    tries = min(MAX_ANCHOR_TRIES, max(1, (count // 3) // length))
    centre, half = (count - length) // 2, tries // 2
    head = [k * length for k in range(tries)]
    middle = [centre + k * length for k in sorted(range(-half, half + 1), key=abs)]
    tail = [count - length - k * length for k in range(tries)]
    return length, (head, middle, tail)


def search_frames(span, rate, frame):
    """The lag searched at an anchor, in frames: 2.5 % of its end's time into the file, held to 2-30 s."""
    seconds = float(np.clip(SYNC_SEARCH_RATE * span.stop / rate, *SYNC_SEARCH_S))
    return int(seconds * rate / frame)


def _frame(rate):
    return max(1, int(round(ENVELOPE_FRAME_S * rate)))


def _padded_envelope(mono, first, last, rate, frame):
    """The envelope of frames `first`..`last` of `mono`, zero (its mean) where they run off either end."""
    start, stop = max(0, first), min(len(mono) // frame, last)
    env = envelope(mono[slice(start * frame, stop * frame)], rate, frame) if stop > start else np.zeros(0)
    return np.pad(env, (start - first, max(0, last - start - len(env))))


def _region(mono, span, reach, rate):
    """The envelope over an anchor's frames plus `reach` frames either side."""
    frame = _frame(rate)
    first, length = span.start // frame, (span.stop - span.start) // frame
    return _padded_envelope(mono, first - reach, first + length + reach, rate, frame)


def carries_programme(source, span, rate, floor):
    """Whether an anchor of the source stands 10 LU over the source's noise floor (a blank leader does not)."""
    energies, block = block_energies(source[span], rate)
    if energies.size == 0 or not np.all(np.isfinite(energies)):
        return False
    return bool(lufs(np.mean(energies) / block) >= floor + FLOOR_MARGIN_LU)


def self_similarity(template, shift):
    """Pearson correlation of an envelope with itself `shift` frames later; 1 for a flat one."""
    early, late = template[:-shift] - template[:-shift].mean(), template[shift:] - template[shift:].mean()
    norm = np.sqrt(np.sum(early**2) * np.sum(late**2))
    return float(np.sum(early * late) / norm) if norm > EPS else 1.0


def has_timing(template, rate):
    """A finite envelope that changes within one PAL frame (self-similarity 40 ms apart under 0.9)."""
    shift = max(1, int(round(TIMING_LAG_S * rate / _frame(rate))))
    if len(template) <= shift or not np.all(np.isfinite(template)):
        return False
    return self_similarity(template, shift) < MAX_SELF_SIMILARITY


def pick_anchor(source, rate, floor, starts, length):
    """The first candidate that carries programme with timing in it: `(span, template, reach)`, or None."""
    for start in starts[:MAX_ANCHOR_TRIES]:
        span = slice(start, start + length)
        if not carries_programme(source, span, rate, floor):
            continue
        reach = search_frames(span, rate, _frame(rate))
        region = _region(source, span, reach, rate)
        template = region[slice(reach, len(region) - reach)]
        if has_timing(template, rate):
            return span, template, reach
    return None


def anchor_lag(output, anchor, rate):
    """`(status, lag_ms)` of one anchor: matched with the output's lag, or unread / unmatched with None."""
    if anchor is None:
        return UNREAD, None
    span, template, reach = anchor
    region = _region(output, span, reach, rate)
    if not np.all(np.isfinite(region)):
        return UNMATCHED, None
    return _peak_lag(sliding_correlation(template, region), reach, _frame(rate) / rate * 1000.0)


def _peak_lag(correlation, reach, frame_ms):
    """Matched with the lag of the correlation's peak, or unmatched when the peak is under 0.5."""
    peak = int(np.argmax(correlation))
    if correlation[peak] < MIN_ENVELOPE_CORR:
        return UNMATCHED, None
    return MATCHED, float((peak - reach + parabolic_offset(correlation, peak)) * frame_ms)


def sync_drift(source, output, rate, floor=None):
    """The output's lag behind the source near its start, middle and end, and what they say about the sync.

    `{"lag_start_ms", "lag_middle_ms", "lag_end_ms", "drift_ms", "offset_ms", "segments", "unmatched", "track"}`:
    the matched lags (None when unread or unmatched), their largest change, the largest |lag|, how
    many anchors matched, how many carried programme the output did not follow within the search,
    and the matched anchors' `(centre_s, lag_ms)` for the ride to read along (`matched_track`).
    """
    floor = source_floor(source, rate) if floor is None else floor
    length, groups = anchor_candidates(len(source), rate)
    anchors = [pick_anchor(source, rate, floor, starts, length) for starts in groups]
    readings = [anchor_lag(output, anchor, rate) for anchor in anchors]
    lags = {f"lag_{name}_ms": lag for name, (_status, lag) in zip(SEGMENT_NAMES, readings)}
    return {**lags, **_summary(readings), "track": matched_track(anchors, readings, rate)}


def matched_track(anchors, readings, rate):
    """`[(centre_s, lag_ms), ...]` of the matched anchors: each anchor's centre in source seconds and its lag."""
    return [((anchor[0].start + anchor[0].stop) / 2.0 / rate, lag) for anchor, (status, lag) in zip(anchors, readings) if status == MATCHED]


def _summary(readings):
    """Drift, offset, and the matched and unmatched counts of the anchors' `(status, lag)` readings."""
    found = [lag for status, lag in readings if status == MATCHED]
    drift, offset = _drift_and_offset(found)
    return {"drift_ms": drift, "offset_ms": offset, "segments": len(found), "unmatched": sum(status == UNMATCHED for status, _ in readings)}


def _drift_and_offset(found):
    """The largest change between the matched lags (None under two) and the largest |lag| (None with none)."""
    if not found:
        return None, None
    drift = float(max(found) - min(found)) if len(found) >= 2 else None
    return drift, float(np.max(np.abs(found)))


def median_lag_samples(sync, rate):
    """The median matched lag of a `sync_drift` reading in samples, 0 with none: the shift `gain_ride` takes out."""
    found = [sync[f"lag_{name}_ms"] for name in SEGMENT_NAMES if sync[f"lag_{name}_ms"] is not None]
    return int(round(np.median(found) * rate / 1000.0)) if found else 0


def file_entries(source, output, rate):
    """The card's file-level entries (`{"source": 0, "output": value, "delta": value}`) from the raw mono pair."""
    sync = sync_drift(source, output, rate)
    ride = gain_ride(source, output, rate, median_lag_samples(sync, rate), sync["track"])
    values = {
        "file.gain_ride_lu": ride["p95_abs_lu"],
        "file.gain_ride_std_lu": ride["std_lu"],
        "file.gain_ride_deriv_lu": ride["deriv_p10_p90"],
        "file.sync_drift_ms": sync["drift_ms"],
        "file.sync_offset_ms": sync["offset_ms"],
        "file.sync_unmatched": float(sync["unmatched"]),
        **{f"file.sync_lag_{name}_ms": sync[f"lag_{name}_ms"] for name in SEGMENT_NAMES},
    }
    return {name: {"source": 0.0, "output": value, "delta": value} for name, value in values.items()}
