"""Ear v4's second hiss anchor: the stationary floor of continuous programme (music, singing, dense speech).

The programme reading (`hiss_programme.py`) reports its margins only on a validated anchor. The first
path (`hiss_floor.source_anchor`) reads the hiss in R4's true pauses; music and singing rarely hold
1 s of them, so their anchor came from every live frame and was never validated: the reading read 26
of 235 IA hunt clips and 2 of 12 music clips (docs/ear_v4_hiss.md, "Continuous Programme"). This
module is the second path, tried only when the first did not validate; the pause reading never sees
it, and a source whose pause anchor validates reads exactly as before.

Per 15 s segment of the raw source mono, on `hiss_floor`'s STFT, bands and valley levels (the tones
of the quietest frames left out), over the live frames:

1. Stationary floor (minimum statistics): each band's floor per 3 s block is C_b x P10 of its valley
   (`hiss_floor.white_calibration`). A band is accepted where its block floors span at most 1 dB from
   p10 to p90 (tape hiss under the programme holds its level; a programme's own valleys follow the
   music) and every block holds over the 16-bit dither plus 10 dB; its level is the P25 of its block
   floors. The median sat up to 0.9 dB over the exact hiss of a music fixture in the bands its notes
   touch and read that perfect output +0.22 dB; the lowest block read every perfect output under
   -3.4 dB but biased a fresh realisation of the hiss +0.44 dB (p95), the P25 +0.34 with every perfect
   output under -1.7. Three blocks at least, so a segment under 9 s of live frames is not read.
2. Band edge: a band carries programme where its mean power stands more than 1 dB over the segment's
   floor (C_b x P10 of the valley) or its frame power follows the 300-4000 Hz envelope (correlation
   over 0.15); it is hiss alone where neither holds, its 1 s block means span at most 1.5 dB and its
   floor holds over the dither. The programme's top is the highest band of 1 kHz and up that carries
   programme; the edge bands are the hiss-alone bands from half an octave over that top to 12 kHz.

Over the segments a band counts where at least half of them say so; an accepted band's level is the
median of its accepting segments' levels, every other band NaN (the programme reading interpolates
them, `hiss_programme.filled_anchor`). The anchor is validated when (a) every band of 1 kHz and up
that holds anything over the dither is accepted but at most four (and six at least), and (b) at least
two edge bands lie under 12 kHz. (a) says the programme leaves the hiss exposed in every band within
every 3 s; (b) says it stops under the top bands, so the reading's spectral cap has bands where no
programme can be revealed.

Why so strict (docs/ear_v4_hiss.md, "Continuous Programme"): the programme reading cannot tell
programme a hiss-free output reveals from residual hiss in a noise-dominated cell. On realistic-v2's
music fixtures even the exact noise as the anchor read the perfect output (the target) at or under 0
on 0 of 36 music-bed and 1 of 40 music-led fixtures (m09-m24); only music-only programme, which holds
real silences, read under 0 (23 of 40). Each dB the anchor stands over the true hiss reads about 3 dB
on a perfect output. A floor extrapolated from the bands above the programme (the other candidate)
stood over the hiss where the tape's hiss falls toward the low bands and read perfect outputs up to
+23.7 dB; holding the reading to those bands alone read a render that kept the hiss under the
programme (+13..+30 dB over its masked threshold with the exact anchor) at -21.9..+20.8. What
validates here: 38 of the 360 stationary-hiss music variants (music-only programme low-passed at
4.5 kHz), every perfect output at or under 0 (median -19.6, max -1.70), the raw tape +24.0..+37.3,
the ladder monotone, a render that keeps the hiss under the programme +19.6..+32.6; 3 more of the 40
gapped hiss-shape fixtures of `check_hiss_fixtures`. No realistic-v2 corpus fixture (a median of 3 of
23 bands accepted), no music clip and no IA hunt source: real continuous programme exposes its hiss in
pauses, not in a stationary floor (even the Tata speech tapes accept 0-8 of 20-24 bands).
"""

import dataclasses

import numpy as np

from scripts.restoration_quality import hiss_floor

BLOCK_S = 3.0
EDGE_BLOCK_S = 1.0
MIN_BLOCKS = 3
SPREAD_DB = 1.0
LEVEL_PCT = 25.0
LO_HZ = 1000.0
ALLOWED_MISS = 4
MIN_ACCEPTED = 6
EDGE_EXCESS_DB = 1.0
EDGE_SPREAD_DB = 1.5
EDGE_COMOD = 0.15
EDGE_GUARD = float(np.sqrt(2.0))
EDGE_TOP_HZ = hiss_floor.HISS_REF_TOP_HZ
MIN_EDGE_BANDS = 2
ENV_BAND_HZ = (300.0, 4000.0)
MAJORITY = 0.5


@dataclasses.dataclass(frozen=True)
class Segment:
    """One segment's verdict per band: accepted (stationary) with its level, carries programme, hiss alone, holds a floor."""

    ok: np.ndarray
    level: np.ndarray
    present: np.ndarray
    hiss_only: np.ndarray
    held: np.ndarray


def frames_per(seconds, rate):
    """The STFT frames in `seconds` (at least one)."""
    return max(1, int(round(seconds * rate / hiss_floor.HOP)))


def segment_rows(segment, rate, top_hz):
    """`(centres, valley, band power, envelope)` over the live frames of one segment; band power is per kept bin."""
    freqs, power = hiss_floor.stft_power(segment, rate)
    index, centres = hiss_floor.band_layout(freqs, rate, top_hz)
    kindex = hiss_floor.tonal_exclusion(power, index)
    valley = hiss_floor.valley_levels(power, kindex, len(centres), hiss_floor.band_shape(power))
    onehot = (kindex[:, np.newaxis] == np.arange(len(centres))[np.newaxis, :]).astype(np.float64)
    band_power = (power @ onehot) / np.maximum(onehot.sum(axis=0), 1.0)
    envelope = power[:, (freqs >= ENV_BAND_HZ[0]) & (freqs < ENV_BAND_HZ[1])].sum(axis=1)
    level = power.sum(axis=1)
    live = level > level.max() * 10.0 ** (-hiss_floor.LIVE_RANGE_DB / 10.0)
    return centres, valley[live], band_power[live], envelope[live]


def blocks_of(rows, size):
    """`(blocks, size, columns)`: the whole `size`-row blocks of `rows` (the tail dropped)."""
    count = len(rows) // size
    return np.asarray(rows[: count * size]).reshape(count, size, rows.shape[1])


def block_floors(valley, calib, size):
    """`(blocks, bands)`: C_b x P10 of the valley in each whole block, NaN read as 0."""
    blocks = blocks_of(np.where(np.isfinite(valley), valley, 0.0), size)
    return np.percentile(blocks, hiss_floor.FLOOR_PCT, axis=1) * calib[np.newaxis, :]


def spread_db(values):
    """Per column, p90 over p10 in dB; infinite where the p10 is not positive."""
    high, low = np.percentile(values, 90.0, axis=0), np.percentile(values, 10.0, axis=0)
    ratio = np.divide(high, low, out=np.full(len(low), np.inf), where=low > 0.0)
    return 10.0 * np.log10(np.maximum(ratio, hiss_floor.EPS))


def stationary_bands(valley, calib, rate):
    """`(accepted, level)` of one segment (step 1, the level the P25 of its block floors), or None under three 3 s blocks."""
    floors = block_floors(valley, calib, frames_per(BLOCK_S, rate))
    if len(floors) < MIN_BLOCKS:
        return None
    held = floors.min(axis=0) >= hiss_floor.dither_limit()
    return (spread_db(floors) <= SPREAD_DB) & held, np.percentile(floors, LEVEL_PCT, axis=0)


def envelope_correlation(band_power, envelope):
    """Per band, the correlation of its frame power with the envelope (0 where either side is constant)."""
    centred = band_power - band_power.mean(axis=0)
    env = envelope - envelope.mean()
    scale = np.sqrt((centred**2).sum(axis=0) * float((env**2).sum()))
    return np.divide(centred.T @ env, scale, out=np.zeros(band_power.shape[1]), where=scale > 0.0)


def programme_flags(rows, calib, rate):
    """`(carries programme, hiss alone, holds a floor)` per band of one segment (step 2)."""
    valley, band_power, envelope = rows
    floor = hiss_floor.percentile_floor(valley) * calib
    held = floor >= hiss_floor.dither_limit()
    excess = 10.0 * np.log10(np.maximum(band_power.mean(axis=0), hiss_floor.EPS) / np.maximum(floor, hiss_floor.EPS))
    comod = envelope_correlation(band_power, envelope)
    present = (excess > EDGE_EXCESS_DB) | (comod > EDGE_COMOD)
    steady = spread_db(blocks_of(band_power, frames_per(EDGE_BLOCK_S, rate)).mean(axis=1)) <= EDGE_SPREAD_DB
    return present, ~present & steady & (np.abs(comod) <= EDGE_COMOD) & held, held


def read_segment(segment, rate, top_hz):
    """`(centres, Segment)` of one segment, or None when it is too short or holds under three 3 s blocks."""
    if len(segment) < 4 * hiss_floor.FRAME:
        return None
    centres, valley, band_power, envelope = segment_rows(segment, rate, top_hz)
    calib = hiss_floor.white_calibration(rate, top_hz)
    stationary = stationary_bands(valley, calib, rate)
    if stationary is None:
        return None
    return centres, Segment(*stationary, *programme_flags((valley, band_power, envelope), calib, rate))


def read_segments(mono, rate, top_hz):
    """`(centres, [Segment])` over every 15 s segment; centres None when no segment was read."""
    size = int(hiss_floor.ANCHOR_SEGMENT_S * rate)
    centres, segments = None, []
    for start in range(0, len(mono), size):
        found = read_segment(np.asarray(mono[start:][:size], dtype=np.float64), rate, top_hz)
        if found is not None:
            centres = found[0]
            segments.append(found[1])
    return centres, segments


def majority(masks):
    """Per band, whether at least half the masks hold."""
    return np.mean(np.asarray(masks, dtype=np.float64), axis=0) >= MAJORITY


def accepted_level(segments, accepted):
    """Per band, the median level of the segments that accept it where the band is accepted, else NaN."""
    levels = np.array([np.where(segment.ok, segment.level, np.nan) for segment in segments])
    return np.where(accepted, hiss_floor.column_stat(levels, np.median), np.nan)


def programme_top(centres, present):
    """The highest band of 1 kHz and up that carries programme, 1 kHz when none does."""
    found = centres[present & (centres >= LO_HZ)]
    return float(found.max()) if found.size else LO_HZ


def edge_bands(centres, present, hiss_only):
    """The hiss-alone bands from half an octave over the programme's top up to 12 kHz."""
    return hiss_only & (centres >= programme_top(centres, present) * EDGE_GUARD) & (centres < EDGE_TOP_HZ)


def covered(centres, accepted, held):
    """Validation (a): every band of 1 kHz and up that holds a floor is accepted but at most four, and six at least."""
    considered = (centres >= LO_HZ) & held
    return int((accepted & considered).sum()) >= max(MIN_ACCEPTED, int(considered.sum()) - ALLOWED_MISS)


def segment_majority(segments, field):
    """Per band, whether `field` holds in at least half the segments."""
    return majority([getattr(segment, field) for segment in segments])


def validated_on(centres, segments, accepted):
    """Validation (a) and (b) over the segments (module docstring)."""
    present, hiss_only = segment_majority(segments, "present"), segment_majority(segments, "hiss_only")
    held = np.any([segment.held for segment in segments], axis=0)
    return covered(centres, accepted, held) and int(edge_bands(centres, present, hiss_only).sum()) >= MIN_EDGE_BANDS


def stationary_anchor(mono, rate, brickwall_hz=None):
    """The continuous-programme `Anchor` (module docstring), or None when no segment holds three 3 s blocks."""
    centres, segments = read_segments(mono, rate, hiss_floor.layout_top(rate, brickwall_hz))
    if not segments:
        return None
    accepted = segment_majority(segments, "ok")
    validated = validated_on(centres, segments, accepted)
    return hiss_floor.Anchor(accepted_level(segments, accepted), centres, 0.0, hiss_floor.STATIONARY, bool(validated))


def source_anchor(mono, rate, prog_bandwidth_hz=None, brickwall_hz=None):
    """`hiss_floor.source_anchor`, carrying the stationary anchor for the programme reading when the pause anchor is not validated."""
    anchor = hiss_floor.source_anchor(mono, rate, prog_bandwidth_hz, brickwall_hz)
    if anchor.validated:
        return anchor
    return dataclasses.replace(anchor, programme=stationary_anchor(mono, rate, brickwall_hz))
