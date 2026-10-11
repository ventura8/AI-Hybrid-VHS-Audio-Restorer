"""Ear v4, reading 2: the tape hiss left in the pauses, the hiss-weighted pause floor over the threshold in quiet.

Per window, on R4's true pauses (`pause_metrics.true_pause_mask` on the source: its quiet
frames inside DSP-VAD non-speech runs of 200 ms or more), so the decision rests on the source
and is the same for every candidate:

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
   block does not lift the floor; 200 ms is the ear's integration time for a noise.
4. Each side's floor relative to its own whole aligned file's loud frames (the listener sets
   the volume once), at `hiss_floor.L_CRIT_DB`: rel_b = 10 log10 floor_b - loud.
5. The hiss weight w(f) = 1 / (1 + exp(-log2(f / 2500) / 0.25)): 0.05 at 1.5 kHz, 0.22 at
   2 kHz, 0.5 at 2.5 kHz, 0.94 at 4 kHz. Tape hiss is the broadband, HF-heavy stationary
   floor; room tone sits mostly under 2-3 kHz and the user allows it (they rejected dead
   pauses: "silent in pauses").
6. `hiss_pause_margin_db` = max over bands of 1 kHz and up holding a floor of
   L_CRIT + rel_b + 10 log10 w - (Tq - 3) (Terhardt's threshold less 3 dB). With no band
   holding anything it reads L_CRIT - 140 = -65, never None: an empty output is inaudible and
   loses nothing for being unread. The ranked form is the excess, max(0, margin).
   `hiss_pause_onset_db` = L_CRIT - margin is the playback level (loud frames in dB SPL) at
   which the hiss becomes audible, level-free (display).
7. Guards and display: `hiss_room_onset_db`, min over 100-2000 Hz of Tq - 3 - rel_b,
   unweighted (where room tone becomes audible); `hiss_room_delta_db`, the 100-1500 Hz floor
   relative to the loud frames, output minus source (dead air reads far under 0); and
   `hiss_pause_sone`, Zwicker's specific loudness at L_crit, x the Traunmuller Bark width and
   w(f), summed over bands of 1 kHz and up.

R4's HF excess ordered round one's hiss pair (+3.11 dB "has hiss" against +0.49 clean,
Tele7abc) where the pause level alone did not (the accepted v2 outputs read gap air -8..-12 dB,
the "hissy" cathar alpha 2 -15..-18.5): shape and audibility, not raw level, so this reading
weighs the floor's high bands at their audibility rather than its level.
"""

import numpy as np

from scripts.restoration_quality import auditory, hiss_floor, pause_metrics

FRAME_S = pause_metrics.FRAME_S
BLOCK_FRAMES = 10
MIN_BLOCK_FRAMES = 5
TQ_MARGIN_DB = 3.0
HISS_CENTRE_HZ = 2500.0
HISS_WIDTH_OCT = 0.25
HISS_FROM_HZ = 1000.0
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
    """w(f): the logistic hiss weight, 0.5 at 2.5 kHz, a quarter octave per e-fold."""
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


def stationary_floor(frames_power, kept, spans):
    """Per band, the median over 200 ms blocks (over frames when no block fits), bias-corrected to the mean (module docstring)."""
    blocks = block_means(frames_power, spans)
    degrees = np.maximum(kept / CORRELATED_BINS, MIN_SHAPE) * (1.0 if blocks is None else BLOCK_FRAMES)
    values = frames_power if blocks is None else blocks
    return hiss_floor.column_stat(values, np.median) / wilson_hilferty_median_ratio(degrees)


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


def pause_bands(pair, rate, mask, top_hz):
    """`(edges, centres, source floor, output floor)` per band in the true pauses, tones out."""
    freqs, src_power, out_power = masked_frames(pair, rate, mask)
    edges, index, centres = band_table(freqs, rate, top_hz)
    keep = kept_bins(src_power, out_power, max(1, int(FRAME_S * rate)))
    src_floor, out_floor = (side_floor(power, (keep, index, len(centres)), masked_spans(mask)) for power in (src_power, out_power))
    return edges, centres, src_floor, out_floor


def side_floor(power, bands, spans):
    """One side's stationary floor per band; `bands` is `(kept bins, bin -> band, band count)`."""
    frames, kept = band_frame_power(power, *bands)
    return stationary_floor(frames, kept, spans)


def relative_levels(floor, loud_db):
    """Per band, the floor in dB relative to the side's loud frames; NaN where the band holds nothing."""
    floor = np.asarray(floor, dtype=np.float64)
    held = np.isfinite(floor) & (floor > 0.0)
    return np.where(held, 10.0 * np.log10(np.where(held, floor, 1.0)) - loud_db, np.nan)


def hiss_margin(rel, centres):
    """max over bands of 1 kHz and up of L_crit + rel + 10 log10 w - (Tq - 3); L_crit - 140 when no band holds anything."""
    bands = (centres >= HISS_FROM_HZ) & np.isfinite(rel)
    if not bands.any():
        return hiss_floor.L_CRIT_DB - ONSET_CAP
    level = hiss_floor.L_CRIT_DB + rel[bands] + 10.0 * np.log10(hiss_weight(centres[bands])) - quiet_threshold(centres[bands])
    return float(np.max(level))


def room_onset(rel, centres):
    """min over the 100-2000 Hz bands of Tq - 3 - rel (unweighted): the playback level where room tone turns audible; 140 empty."""
    bands = (centres >= ROOM_ONSET_BAND_HZ[0]) & (centres < ROOM_ONSET_BAND_HZ[1]) & np.isfinite(rel)
    return float(np.min(quiet_threshold(centres[bands]) - rel[bands])) if bands.any() else ONSET_CAP


def hiss_sone(rel, centres, edges):
    """Zwicker loudness at L_crit of the hiss-weighted floor, sone, over bands of 1 kHz and up."""
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
    """`{name: (source, output)}` for `PAUSE_NAMES` on one aligned window; every name None with under 25 true-pause frames."""
    frame = max(1, int(FRAME_S * rate))
    count = min(len(src), len(out)) // frame
    pair = (np.asarray(src[: count * frame], dtype=np.float64), np.asarray(out[: count * frame], dtype=np.float64))
    if count < pause_metrics.MIN_PAUSE_FRAMES or not (np.isfinite(pair[0]).all() and np.isfinite(pair[1]).all()):
        return hiss_floor.none_pairs(PAUSE_NAMES)
    mask, _runs = pause_metrics.true_pause_mask(pair[0], rate)
    if mask.sum() < pause_metrics.MIN_PAUSE_FRAMES:
        return hiss_floor.none_pairs(PAUSE_NAMES)
    return paired(pause_bands(pair, rate, mask, context.top_hz), context, float(mask.sum() * frame / rate))
