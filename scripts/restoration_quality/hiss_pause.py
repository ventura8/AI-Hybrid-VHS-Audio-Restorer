"""Ear v4, reading 2: the tape hiss left in the pauses, the hiss-weighted pause floor over the threshold in quiet.

Per window, on R4's true pauses (`pause_metrics.true_pause_mask` on the source: its quiet
frames inside DSP-VAD non-speech runs of 200 ms or more) less the runs whose source holds
partials held across the run (`pause_partials.unsung`: the odd and the even 2048-point frames
of the run agree on their fine spectrum, r >= 0.3; a window whose such runs held half its
true-pause frames keeps none), so the decision rests on the source and is the same for every
candidate. On Gaudeamus5 the sung runs left 606 of 2,369 true-pause frames and the vote 163
more, 8 windows of 30 reading None; the speech tapes left 0 (Tele7abc, SOTI) and 23 of 3,036
(Vaccin, no window); APL's frame fill read 13.14 at the tail, not 21.67, and the reference
suppressor 0.00, not 6.43 (docs/ear_v4_hiss.md, "Sung Pauses"):

1. 20 ms mean-square frames of both sides (`hiss_floor.ms_frames`) on the mask; the bins a
   tone holds on either side leave both (`pause_metrics.tonal_bins`: hum, which both engines
   remove on purpose, is not hiss). An output bin under the 16-bit dither carries nothing and
   is no tone: a pause emptied to digital silence would otherwise read every bin "steady" and
   take the source's floor out with it.
2. 1-ERB bands from 50 Hz to min(16 kHz, Nyquist, R0's brickwall), centres sqrt(lo hi); per
   frame a band's power is the mean over its kept bins times its full bin count.
3. The stationary floor per band: the median over consecutive 200 ms blocks (10 frames) inside
   each contiguous mask run, a tail of 5 frames or more kept, divided by the Wilson-Hilferty
   median/mean ratio of a Gamma with k = max(kept bins / 1.5, 0.5) x 10 degrees (z = 0):
   max(1 - 1 / (9k), 0.05)^3. The median, not the mean, so a word tail or a breath in one
   block does not lift the floor, nor soft programme the source's hiss hid in its quiet frames
   and a clean output reveals (with the mean a perfect output's pauses read up to +43 dB on the
   fixtures); 200 ms is the ear's integration time for a noise. Only the source's silent blocks
   count, band by band, on both sides: a block whose source band of 1 kHz and up stands 3 dB
   or more over the source's stationary hiss (the anchor, `anchor_band_power`) holds programme
   there, not hiss. On a choir tape R4's quiet frames are soft singing: Gaudeamus5's source
   pause blocks stood +4.1 dB (median) over its anchor at 2-8 kHz, against +0.2..+1.2 per
   window on Tele7abc (SOTI -1.0..+2.0, Vaccin -1.2..+2.9). At 6 dB most singing blocks passed
   the test, so the singing set the floor: the pause tail of the APL keeper ladder read 36.25 /
   33.00 / 32.67 / 32.67 (fill 8 / 12 / 18 / keeper off, Gaudeamus5), and a render that took
   the soft singing out would have won the family. At 3 dB it reads 34.66 / 31.21 / 25.09 /
   20.50; the speech tapes move under 0.6 dB (adversarial review, docs/ear_v4_hiss.md). A
   window whose every such band held programme reads None.
4. Each side's floor relative to its own whole aligned file's loud frames (the listener sets
   the volume once), at `hiss_floor.L_CRIT_DB`: rel_b = 10 log10 floor_b - loud, weighted by
   the hiss weight (step 5) and then, for the margin, each band takes the power mean of itself
   and its two neighbours either side (tape hiss is smooth across bands, and the max over
   single bands picked their upward draws: three bands halved the realisation scatter, sd 0.25
   -> 0.13-0.18 dB on the fixtures; five, with the silent blocks, took the fresh-realisation
   p95 of the fixtures' pause margin from 0.51-0.54 to 0.18-0.21 dB). The weight comes first:
   smoothed before it, a room floor under 2 kHz leaked into the 2.4-2.7 kHz bands at their
   full weight.
5. The hiss weight w(f) = 1 / (1 + exp(-log2(f / 3150) / 0.15)): 0.01 at 2 kHz, 0.10 at
   2.5 kHz, 0.38 at 3 kHz, 0.5 at 3.15 kHz, 0.73 at 3.5 kHz, 0.91 at 4 kHz. Tape hiss is the
   broadband, HF-heavy stationary floor, and the ear is most sensitive to it at 3-4 kHz, where
   tape hiss set the margin on every fixture (3.5-3.9 kHz); room tone sits mostly under about
   2-3 kHz and the user allows it (they rejected dead pauses: "silent in pauses"). The first
   weight (0.5 at 2.5 kHz, a quarter octave per e-fold, smoothed before weighting) read a
   hiss-free pink room floor low-passed at 2 kHz, at the source's own 100-1000 Hz pause level,
   at +9.7 dB median (9 of 10 speech fixtures flagged); this one reads it at -4.2 (none), and
   tape hiss 0.5 dB lower than before.
6. `hiss_pause_margin_db` = max over bands of 2 kHz and up holding a floor of
   L_CRIT + smoothed (rel_b + 10 log10 w) - (Tq - 3) (Terhardt's threshold less 3 dB), never
   under L_CRIT - 140 = -65. With no band holding anything it reads -65, never None: an empty
   output is inaudible and loses nothing for being unread (a pause band-filtered to 1e-90 read
   -1843 before this floor held it). The ranked form is the excess, max(0, margin).
   `hiss_pause_onset_db` = L_CRIT - margin is the playback level (loud frames in dB SPL) at
   which the hiss becomes audible, level-free (display).
7. Guards and display: `hiss_room_onset_db`, min over 100-2000 Hz of Tq - 3 - rel_b,
   unweighted (where room tone becomes audible); `hiss_room_delta_db`, the 100-1500 Hz floor
   relative to the loud frames, output minus source (dead air reads far under 0); and
   `hiss_pause_sone`, Zwicker's specific loudness at L_crit, x the Traunmuller Bark width and
   w(f), summed over bands of 2 kHz and up.

R4's HF excess ordered round one's hiss pair (+3.11 dB "has hiss" against +0.49 clean,
Tele7abc) where the pause level alone did not (the accepted v2 outputs read gap air -8..-12 dB,
the "hissy" cathar alpha 2 -15..-18.5): shape and audibility, not raw level, so this reading
weighs the floor's high bands at their audibility rather than its level.
"""

import numpy as np
import scipy.ndimage

from scripts.restoration_quality import auditory, hiss_floor, pause_metrics, pause_partials

FRAME_S = pause_metrics.FRAME_S
SMOOTH_BANDS = 5
PROGRAMME_DB = 3.0
SILENT_FROM_HZ = 1000.0
BLOCK_FRAMES = 10
MIN_BLOCK_FRAMES = 5
TQ_MARGIN_DB = 3.0
HISS_CENTRE_HZ = 3150.0
HISS_WIDTH_OCT = 0.15
HISS_FROM_HZ = 2000.0
ROOM_DELTA_BAND_HZ = (100.0, 1500.0)
ROOM_ONSET_BAND_HZ = (100.0, 2000.0)
LO_HZ = 50.0
TOP_HZ = 16000.0
ONSET_CAP = 140.0
CORRELATED_BINS = 1.5
MIN_SHAPE = 0.5
WH_FLOOR = 0.05
EPS = hiss_floor.EPS
PAUSE_NAMES = (
    "hiss_pause_margin_db",
    "hiss_pause_excess_db",
    "hiss_pause_onset_db",
    "hiss_pause_sone",
    "hiss_room_onset_db",
    "hiss_room_delta_db",
    "hiss_pause_s",
)


def hiss_weight(freq_hz):
    """w(f): the logistic hiss weight, 0.5 at 3.15 kHz, 0.15 octave per e-fold."""
    octaves = np.log2(np.maximum(np.asarray(freq_hz, dtype=np.float64), 1.0) / HISS_CENTRE_HZ) / HISS_WIDTH_OCT
    return 1.0 / (1.0 + np.exp(-octaves))


def bark(freq_hz):
    """Traunmuller's critical-band rate in Bark."""
    freq = np.asarray(freq_hz, dtype=np.float64)
    return 26.81 * freq / (1960.0 + freq) - 0.53


def quiet_threshold(centres):
    """Terhardt's threshold in quiet less the 3 dB margin, per band."""
    return auditory.threshold_in_quiet_db(centres) - TQ_MARGIN_DB


def specific_loudness(level_db, quiet_db):
    """Zwicker and Fastl: N' = 0.08 (E_TQ / E0)^0.23 [(0.5 + 0.5 E / E_TQ)^0.23 - 1] sone per Bark, clamped at 0."""
    ratio = 10.0 ** ((np.asarray(level_db, dtype=np.float64) - quiet_db) / 10.0)
    return np.maximum(0.08 * 10.0 ** (0.023 * quiet_db) * ((0.5 + 0.5 * ratio) ** 0.23 - 1.0), 0.0)


def wilson_hilferty_median_ratio(shape):
    """The median over the mean of a Gamma(shape), Wilson-Hilferty with z = 0: max(1 - 1 / (9 shape), 0.05)^3."""
    return np.maximum(1.0 - 1.0 / (9.0 * np.asarray(shape, dtype=np.float64)), WH_FLOOR) ** 3


def band_table(freqs, rate, top_hz):
    """`(edges, index per bin, centres)`: 1-ERB bands from 50 Hz to min(16 kHz, Nyquist, `top_hz`), geometric centres."""
    edges = auditory.erb_band_edges(LO_HZ, float(min(TOP_HZ, 0.5 * rate, top_hz or np.inf)))
    return edges, auditory.band_index(freqs, edges), np.sqrt(edges[:-1] * edges[1:])


def band_frame_power(power, keep, index, nb):
    """`((frames, bands) power over each band's kept bins x its full bin count, kept bins per band)`; NaN with none kept."""
    out = np.full((power.shape[0], nb), np.nan)
    kept = np.zeros(nb)
    for band in range(nb):
        inside = index == band
        chosen = inside & keep
        kept[band] = chosen.sum()
        if kept[band]:
            out[:, band] = power[:, chosen].mean(axis=1) * inside.sum()
    return out, kept


def masked_spans(mask):
    """`(start, stop)` of each contiguous run of `mask`, as indices into the masked frames."""
    spans, position = [], 0
    for start, stop in pause_metrics.run_spans(mask):
        spans.append((position, position + stop - start))
        position += stop - start
    return spans


def block_means(frames_power, spans):
    """`(blocks, bands)`: band power over consecutive 10-frame blocks of each run, a tail of 5 frames or more kept; None with none."""
    rows = [block.mean(axis=0) for start, stop in spans for block in run_blocks(frames_power[start:stop]) if len(block) >= MIN_BLOCK_FRAMES]
    return np.array(rows) if rows else None


def run_blocks(run):
    """The consecutive 10-frame blocks of one run (the last one shorter)."""
    return [run[first:][:BLOCK_FRAMES] for first in range(0, len(run), BLOCK_FRAMES)]


def block_rows(frames_power, spans):
    """`(rows, frames per row)`: the 200 ms block means of the runs, or the frames themselves when no block fits."""
    blocks = block_means(frames_power, spans)
    return (frames_power, 1.0) if blocks is None else (blocks, float(BLOCK_FRAMES))


def masked_median(values, keep):
    """Per column, the median of the finite values `keep` selects; NaN for a column with none."""
    return hiss_floor.column_stat(np.where(keep, values, np.nan), np.median)


def stationary_floor(rows, kept, per_row, silent=None):
    """Per band, the median over the silent rows (`block_rows`), bias-corrected to the mean (module docstring)."""
    degrees = np.maximum(kept / CORRELATED_BINS, MIN_SHAPE) * per_row
    keep = np.ones(rows.shape, dtype=bool) if silent is None else silent
    return masked_median(rows, keep) / wilson_hilferty_median_ratio(degrees)


def anchor_band_power(anchor, centres, counts, frame):
    """The anchor's stationary hiss as band power on these bands, or None without an anchor read in two bands or more.

    The anchor's per-bin level (`hiss_floor.stft_power`'s 2048-point bins) is interpolated in log level over log
    frequency onto the band centres, rescaled to `frame`-point bins (a bin's power scales as 1 / its length) and summed
    over each band's `counts` bins, the scaling `band_frame_power` gives a band.
    """
    if anchor is None or anchor.level is None:
        return None
    level = np.asarray(anchor.level, dtype=np.float64)
    read = np.isfinite(level) & (level > 0.0)
    if read.sum() < 2:
        return None
    per_bin = 10.0 ** np.interp(np.log(centres), np.log(np.asarray(anchor.centres)[read]), np.log10(level[read]))
    return per_bin * (hiss_floor.FRAME / frame) * counts


def silent_rows(src_rows, expected, centres):
    """Per row and band: True unless the source's band of 1 kHz and up stands 3 dB or more over the anchor's hiss there.

    Every row is silent without an anchor. A row's NaN band (no kept bin) stays silent; the median skips it anyway.
    """
    silent = np.ones(src_rows.shape, dtype=bool)
    if expected is None:
        return silent
    judged = centres >= SILENT_FROM_HZ
    silent[:, judged] = ~(src_rows[:, judged] >= expected[judged] * 10.0 ** (PROGRAMME_DB / 10.0))
    return silent


def kept_bins(src_power, out_power, frame):
    """The bins no tone holds on either side; an output bin under the 16-bit dither is empty, not a tone."""
    dither = 2.0 * (pause_metrics.QUANT_LSB**2 / 4.0) / frame
    out_tones = pause_metrics.tonal_bins(out_power) & (out_power.mean(axis=0) >= dither)
    return ~(pause_metrics.tonal_bins(src_power) | out_tones)


def masked_frames(pair, rate, mask):
    """`(freqs, source power, output power)` of the 20 ms frames on the mask."""
    frame = max(1, int(FRAME_S * rate))
    freqs, src_power = hiss_floor.ms_frames(pair[0], rate, frame)
    out_power = hiss_floor.ms_frames(pair[1], rate, frame)[1]
    return freqs, src_power[: len(mask)][mask], out_power[: len(mask)][mask]


def pause_bands(pair, rate, mask, anchor=None, top_hz=None):
    """`(edges, centres, source floor, output floor)` per band in the true pauses, tones out, over the source's silent blocks.

    A band whose every block holds programme on the source (`silent_rows`) is NaN on both sides.
    """
    freqs, src_power, out_power = masked_frames(pair, rate, mask)
    edges, index, centres = band_table(freqs, rate, top_hz)
    frame = max(1, int(FRAME_S * rate))
    rows = band_rows((src_power, out_power), (kept_bins(src_power, out_power, frame), index, len(centres)), masked_spans(mask))
    silent = silent_rows(rows[0], anchor_band_power(anchor, centres, band_counts(index, len(centres)), frame), centres)
    return edges, centres, stationary_floor(rows[0], rows[2], rows[3], silent), stationary_floor(rows[1], rows[2], rows[3], silent)


def band_counts(index, nb):
    """The bins of each of `nb` bands."""
    return np.array([np.sum(index == band) for band in range(nb)], dtype=np.float64)


def band_rows(powers, bands, spans):
    """`(source rows, output rows, kept bins per band, frames per row)` of the two sides (`side_rows`)."""
    src_rows, kept, per_row = side_rows(powers[0], bands, spans)
    return src_rows, side_rows(powers[1], bands, spans)[0], kept, per_row


def side_rows(power, bands, spans):
    """`(rows, kept bins per band, frames per row)` of one side; `bands` is `(kept bins, bin -> band, band count)`."""
    frames, kept = band_frame_power(power, *bands)
    rows, per_row = block_rows(frames, spans)
    return rows, kept, per_row


def hiss_read(bands):
    """Whether the source's pause floor holds a band of 2 kHz and up: not every such band's blocks held programme."""
    _edges, centres, src_floor, _out_floor = bands
    return bool(np.any((centres >= HISS_FROM_HZ) & np.isfinite(src_floor)))


def relative_levels(floor, loud_db):
    """Per band, the floor in dB relative to the side's loud frames; NaN where the band holds nothing."""
    floor = np.asarray(floor, dtype=np.float64)
    held = np.isfinite(floor) & (floor > 0.0)
    return np.where(held, 10.0 * np.log10(np.where(held, floor, 1.0)) - loud_db, np.nan)


def smoothed(rel):
    """`rel` (dB per band) as the power mean of each band and its two neighbours either side; NaN bands neither give nor take."""
    held = np.isfinite(rel)
    power = np.where(held, 10.0 ** (np.where(held, rel, 0.0) / 10.0), 0.0)
    total = scipy.ndimage.uniform_filter1d(power, SMOOTH_BANDS, mode="nearest")
    count = scipy.ndimage.uniform_filter1d(held.astype(np.float64), SMOOTH_BANDS, mode="nearest")
    return np.where(held, 10.0 * np.log10(np.maximum(total / np.maximum(count, EPS), EPS)), np.nan)


def weighted(rel, centres):
    """`rel` with each band's hiss weight applied, 10 log10 w(f) dB (before the smoothing: module docstring, step 4)."""
    return np.asarray(rel, dtype=np.float64) + 10.0 * np.log10(hiss_weight(centres))


def hiss_margin(rel, centres):
    """max over bands of 2 kHz and up of L_crit + the smoothed weighted rel - (Tq - 3), never under L_crit - 140."""
    level = smoothed(weighted(rel, centres))
    bands = (centres >= HISS_FROM_HZ) & np.isfinite(level)
    if not bands.any():
        return hiss_floor.L_CRIT_DB - ONSET_CAP
    above = hiss_floor.L_CRIT_DB + level[bands] - quiet_threshold(centres[bands])
    return float(max(np.max(above), hiss_floor.L_CRIT_DB - ONSET_CAP))


def room_onset(rel, centres):
    """min over the 100-2000 Hz bands of Tq - 3 - rel (unweighted): the playback level where room tone turns audible; 140 empty."""
    bands = (centres >= ROOM_ONSET_BAND_HZ[0]) & (centres < ROOM_ONSET_BAND_HZ[1]) & np.isfinite(rel)
    return float(np.min(quiet_threshold(centres[bands]) - rel[bands])) if bands.any() else ONSET_CAP


def hiss_sone(rel, centres, edges):
    """Zwicker loudness at L_crit of the hiss-weighted floor, sone, over bands of 2 kHz and up."""
    bands = (centres >= HISS_FROM_HZ) & np.isfinite(rel)
    width = bark(edges[1:]) - bark(edges[:-1])
    loudness = specific_loudness(rel[bands] + hiss_floor.L_CRIT_DB, quiet_threshold(centres[bands]))
    return float(np.sum(loudness * width[bands] * hiss_weight(centres[bands])))


def room_level(floor, centres, loud_db):
    """The 100-1500 Hz floor relative to the side's loud frames, dB; None without a band there."""
    bands = (centres >= ROOM_DELTA_BAND_HZ[0]) & (centres < ROOM_DELTA_BAND_HZ[1])
    if not bands.any():
        return None
    return float(10.0 * np.log10(np.nansum(floor[bands]) + EPS) - loud_db)


def side_readings(floor, edges, centres, loud_db):
    """`{margin, onset, sone, room_onset}` of one side's pause floor."""
    rel = relative_levels(floor, loud_db)
    margin = hiss_margin(rel, centres)
    return {
        "margin": margin,
        "onset": hiss_floor.L_CRIT_DB - margin,
        "sone": hiss_sone(rel, centres, edges),
        "room_onset": room_onset(rel, centres),
    }


def room_delta(bands, context):
    """Output minus source of `room_level`, None without a room band."""
    _edges, centres, src_floor, out_floor = bands
    src, out = room_level(src_floor, centres, context.src_loud_db), room_level(out_floor, centres, context.out_loud_db)
    return None if src is None else (0.0, out - src)


def paired(bands, context, pause_s):
    """The `PAUSE_NAMES` pairs of one window's pause floors."""
    edges, centres, src_floor, out_floor = bands
    src, out = side_readings(src_floor, edges, centres, context.src_loud_db), side_readings(out_floor, edges, centres, context.out_loud_db)
    delta = room_delta(bands, context)
    return {
        "hiss_pause_margin_db": (src["margin"], out["margin"]),
        "hiss_pause_excess_db": (hiss_floor.excess(src["margin"]), hiss_floor.excess(out["margin"])),
        "hiss_pause_onset_db": (src["onset"], out["onset"]),
        "hiss_pause_sone": (src["sone"], out["sone"]),
        "hiss_room_onset_db": (src["room_onset"], out["room_onset"]),
        "hiss_room_delta_db": (None, None) if delta is None else delta,
        "hiss_pause_s": (pause_s, pause_s),
    }


def pause_readings(src, out, rate, context):
    """`{name: (source, output)}` for `PAUSE_NAMES` on one aligned window; every name None with under 25 true-pause frames.

    A window whose source pauses held programme in every band of 2 kHz and up (`hiss_read`) reads None too, and so does
    one whose true pauses were sung (`unsung_mask`).
    """
    prepared = window_pair(src, out, rate)
    mask = None if prepared is None else unsung_mask(prepared[0][0], rate, context.anchor)
    if mask is None or mask.sum() < pause_metrics.MIN_PAUSE_FRAMES:
        return hiss_floor.none_pairs(PAUSE_NAMES)
    bands = pause_bands(prepared[0], rate, mask, context.anchor, context.top_hz)
    if not hiss_read(bands):
        return hiss_floor.none_pairs(PAUSE_NAMES)
    return paired(bands, context, float(mask.sum() * prepared[1] / rate))


def unsung_mask(source, rate, anchor):
    """R4's true-pause mask of the window's source without its sung runs (`pause_partials.unsung`, the anchor's pause spectrum)."""
    return pause_partials.unsung(source, rate, pause_metrics.true_pause_mask(source, rate), getattr(anchor, "pause_spectrum", None))


def window_pair(src, out, rate):
    """`(pair, frame)`: both sides cut to whole 20 ms frames, or None with under 25 frames or a non-finite sample."""
    frame = max(1, int(FRAME_S * rate))
    count = min(len(src), len(out)) // frame
    pair = (np.asarray(src[: count * frame], dtype=np.float64), np.asarray(out[: count * frame], dtype=np.float64))
    usable = count >= pause_metrics.MIN_PAUSE_FRAMES and bool(np.isfinite(pair[0]).all()) and bool(np.isfinite(pair[1]).all())
    return (pair, frame) if usable else None
