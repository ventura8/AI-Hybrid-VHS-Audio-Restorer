"""Ear v4's shared hiss floor: the band layout, the valley level and the source's stationary floor (the anchor).

The user's rule (2026-10-11, in capitals): the AI human ear must be very critical of VHS hiss
under speech or singing and in silence and pauses. Two readings implement it,
`hiss_programme.py` (the hiss left under the programme over its masked threshold) and
`hiss_pause.py` (the hiss-weighted pause floor over the threshold in quiet). Both are read at
one declared critical playback level, `L_CRIT_DB`: each file's loud frames (>= p70 of
1024-sample frame power) at 75 dB SPL, 10 dB over the audibility check's 65 dB. One listener
sets one volume, so both places use one level. A reconciliation probe re-ran the programme
prototype at 65 and 75 dB SPL: at 65 it reproduced none of round one's hiss labels; at 75 it
separates them on Tele7abc (cathar alpha 2 on 0.7.3 +2.39, its beta 0.005 +2.16, against the
cleared 0.7.5 alpha 2 -0.70, alpha 2.25 -0.98 and baseline -3.01) and orders single4s ("light
hiss", +0.01) over stitched (-3.01). The fixture check still holds at 75: the worst validated
perfect output reads -8.79, the fresh-realisation p95 0.617.

This module holds what both readings share:

- `stft_power`: Hann 2048 / hop 1024 (the NMR grid of `auditory`), centred frames, scaled so
  a band's sum is its mean-square level;
- `band_layout`: 1-ERB bands from 100 Hz to min(16 kHz, Nyquist, R0's brickwall), the CRT
  line bins (+-250 Hz) out;
- the valley level (`valley_levels`): per frame and band, the 25th-percentile bin after the
  bins are whitened by a smooth shape (per-bin P20 over frames, 31-bin median in dB), divided
  by -ln(0.75) (the exponential's q25 over its mean) and multiplied back by the band's mean
  shape; a band under 8 bins takes its mean. Hiss fills a band's valleys; speech partials do
  not, so the valley tracks the floor under the programme;
- `block_floor`: per 1 s block, C_b x P10 of the valley over the active frames within
  +-1.5 s, where C_b (`white_calibration`) maps the estimator onto the true mean of seeded
  white noise;
- `source_anchor`: the source's stationary floor per band, read once per file from R4's true
  pauses (`pause_metrics.true_pause_mask`, per 60 s chunk to bound memory on 22 min tapes),
  C_b x P10 of the valley over at least 1 s of pause frames; fewer pauses, or pauses that are
  digital silence (a muted break), read every live frame instead (`all_frames`). Bands under
  the 16-bit dither plus 10 dB read NaN. Under R0's programme bandwidth a pause can carry
  reverb tails and room tone 20-48 dB over the hiss, so those bands are held at the hiss level
  above the band plus 3 dB (`hiss_cap`). The anchor is `validated` only when it came from the
  pauses, the programme band reaches 2 kHz and at least two reference bands set the cap: an
  unvalidated window abstains from the ranked hiss readings rather than guess.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.ndimage
from numpy.lib.stride_tricks import sliding_window_view

from scripts.restoration_quality import auditory, balance_metrics, dsp_metrics, pause_metrics

# Each file's loud frames (>= p70) play at this level: the critical side of one listener's volume.
L_CRIT_DB = auditory.PLAYBACK_SPL_LOUD_FRAMES + 10.0
FRAME, HOP = auditory.NMR_FRAME, auditory.NMR_HOP
LO_HZ, TOP_HZ = 100.0, 16000.0
VALLEY_Q = 0.25
VALLEY_NORM = float(-np.log1p(-VALLEY_Q))
MIN_VALLEY_BINS = 8
SHAPE_PCT = 20.0
SHAPE_MEDIAN_BINS = 31
FLOOR_PCT = 10.0
BLOCK_S, SPAN_S, MIN_SPAN_S = 1.0, 1.5, 0.45
LIVE_RANGE_DB = balance_metrics.DYNAMIC_RANGE_DB
DITHER_MARGIN_DB = pause_metrics.DITHER_MARGIN_DB
MIN_ANCHOR_S = 1.0
ANCHOR_SEGMENT_S = 15.0
MASK_CHUNK_S = 60.0
HISS_REF_TOP_HZ = 12000.0
HISS_SLACK_DB = 3.0
MIN_REF_BANDS = 2
VALIDATED_MIN_BANDWIDTH_HZ = 2000.0
WHITE_SEED = 12345
WHITE_SECONDS = 15.0
WHITE_LEVEL = 1e-3
MIN_TONAL_FRAMES = 12
QUIET_PCT = 20.0
EPS = 1e-20
PAUSES, ALL_FRAMES, NO_FLOOR = "pauses", "all_frames", "no_floor"


@dataclass(frozen=True)
class Anchor:
    """The source's stationary floor per band (per-bin mean power, NaN where unread), and how it was read."""

    level: np.ndarray | None
    centres: np.ndarray
    pause_s: float
    how: str
    validated: bool

    def to_dict(self):
        """A JSON-ready view; NaN is stored as null."""
        return {
            "level": nan_to_none(self.level),
            "centres": [float(c) for c in self.centres],
            "pause_s": float(self.pause_s),
            "how": self.how,
            "validated": bool(self.validated),
        }

    @classmethod
    def from_dict(cls, data):
        """The anchor `to_dict` wrote (null back to NaN)."""
        centres = np.asarray(data.get("centres", []), dtype=np.float64)
        return cls(none_to_nan(data.get("level")), centres, float(data["pause_s"]), str(data["how"]), bool(data["validated"]))


def nan_to_none(values):
    """A list of floats with NaN as None, or None for no values."""
    if values is None:
        return None
    return [float(value) if np.isfinite(value) else None for value in values]


def none_to_nan(values):
    """`nan_to_none` reversed: an array with None as NaN, or None."""
    if values is None:
        return None
    return np.array([np.nan if value is None else float(value) for value in values])


@dataclass(frozen=True)
class HissContext:
    """What a window's hiss readings need from the whole pair: the anchor, R0's brickwall and the file-level SPL anchors.

    `src_spl_db` turns the source's mean-square dBFS into dB SPL at `L_CRIT_DB`; `src_loud_db` /
    `out_loud_db` are each whole aligned file's loud-frame level (the pause reading's anchors).
    """

    anchor: Anchor | None
    top_hz: float | None
    src_spl_db: float
    src_loud_db: float
    out_loud_db: float


def none_pairs(names):
    """`{name: (None, None)}`: a reading the window cannot give."""
    return {name: (None, None) for name in names}


def excess(value):
    """max(0, value): the one-sided form of a margin (target 0, nothing under the threshold counts), None stays None."""
    return None if value is None else max(0.0, float(value))


def column_stat(values, function):
    """Per column, `function` of the finite values; NaN for a column with none (no warning)."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape[1], np.nan)
    for col in np.flatnonzero(np.isfinite(values).any(axis=0)):
        column = values[:, col]
        out[col] = function(column[np.isfinite(column)])
    return out


def stft_power(mono, rate):
    """`(freqs, power (frames, bins))`: Hann 2048 / hop 1024, centred frames, scaled so a band's sum is its mean square."""
    padded = np.pad(np.asarray(mono, dtype=np.float64), (FRAME // 2, FRAME // 2))
    count = 1 + (len(padded) - FRAME) // HOP
    window = np.hanning(FRAME)
    spectrum = np.fft.rfft(sliding_window_view(padded, FRAME)[::HOP][:count] * window, axis=1)
    return np.fft.rfftfreq(FRAME, 1.0 / rate), (2.0 / (FRAME * np.sum(window**2))) * (spectrum.real**2 + spectrum.imag**2)


def layout_top(rate, top_hz=None):
    """Where the bands end: min(16 kHz, Nyquist, `top_hz`)."""
    return float(min(TOP_HZ, 0.5 * rate, top_hz or np.inf))


def band_layout(freqs, rate, top_hz=None):
    """`(index per bin or -1, centres)`: 1-ERB bands from 100 Hz to `layout_top`, the CRT line bins out; ERB-midpoint centres."""
    edges = auditory.erb_band_edges(LO_HZ, layout_top(rate, top_hz))
    index = auditory.band_index(freqs, edges)
    index[~dsp_metrics.band_bins(freqs, (0.0, np.inf))] = -1
    centres = auditory.erb_to_hz((auditory.erb_number(edges[:-1]) + auditory.erb_number(edges[1:])) / 2.0)
    return index, centres


def band_shape(power):
    """Per bin: per-bin P20 over frames, median-filtered over 31 bins in dB (partials out, slopes kept)."""
    level = 10.0 * np.log10(np.percentile(power, SHAPE_PCT, axis=0) + EPS)
    return 10.0 ** (scipy.ndimage.median_filter(level, size=SHAPE_MEDIAN_BINS, mode="nearest") / 10.0)


def low_quantile(rows, q):
    """Per row, the k-th smallest value, k = floor(q * (n - 1))."""
    k = int(q * (rows.shape[1] - 1))
    return np.partition(rows, k, axis=1)[:, k]


def _band_valley(power, sel, shape):
    """One band's valley level per frame (module docstring)."""
    weights = shape[sel]
    rows = power[:, sel] / weights
    level = low_quantile(rows, VALLEY_Q) / VALLEY_NORM if int(sel.sum()) >= MIN_VALLEY_BINS else rows.mean(axis=1)
    return level * weights.mean()


def valley_levels(power, index, nb, shape):
    """`(frames, bands)` valley level; NaN for a band with no bin."""
    out = np.full((power.shape[0], nb), np.nan)
    for band in range(nb):
        sel = index == band
        if sel.any():
            out[:, band] = _band_valley(power, sel, shape)
    return out


def percentile_floor(valley, pct=FLOOR_PCT):
    """Per band, the `pct` percentile of the valley over frames, NaN read as 0."""
    return np.percentile(np.where(np.isfinite(valley), valley, 0.0), pct, axis=0)


def _block_level(valley, active, geometry, fallback):
    """P10 of the active frames around one block, or `fallback` with too few of them."""
    start, block, span, need = geometry
    lo, hi = max(0, start - span), min(len(valley), start + block + span)
    sel = active[lo:hi]
    return percentile_floor(valley[lo:hi][sel]) if sel.sum() >= need else fallback


def block_floor(valley, active, rate, calib):
    """`(frames, bands)`: C_b x P10 of `valley` over the active frames within +-1.5 s of each 1 s block.

    A block with fewer than 0.45 s of active frames takes the window's P10; bands that are all NaN stay NaN.
    """
    fps = rate / HOP
    block, span, need = max(1, int(round(BLOCK_S * fps))), int(round(SPAN_S * fps)), max(3, int(round(MIN_SPAN_S * fps)))
    fallback = percentile_floor(valley[active] if active.sum() >= need else valley)
    floor = np.empty_like(valley)
    for start in range(0, len(valley), block):
        stop = start + block
        floor[start:stop] = _block_level(valley, active, (start, block, span, need), fallback)
    floor[:, ~np.isfinite(valley).any(axis=0)] = np.nan
    return floor * calib[np.newaxis, :]


def band_means(power, index, nb):
    """Per band, the mean per-bin power over every frame; NaN for a band with no bin."""
    return np.array([power[:, index == band].mean() if (index == band).any() else np.nan for band in range(nb)])


@lru_cache(maxsize=32)
def _white_calibration(rate, top):
    rng = np.random.default_rng(WHITE_SEED)
    freqs, power = stft_power(rng.standard_normal(int(WHITE_SECONDS * rate)) * WHITE_LEVEL, rate)
    index, centres = band_layout(freqs, rate, top)
    nb = len(centres)
    estimate = block_floor(valley_levels(power, index, nb, band_shape(power)), np.ones(len(power), dtype=bool), rate, np.ones(nb))
    calib = band_means(power, index, nb) / column_stat(estimate, np.median)
    calib.flags.writeable = False
    return calib


def white_calibration(rate, top_hz=None):
    """C_b per band: the true mean over the median `block_floor` estimate on 15 s of seeded white noise (every frame active)."""
    return _white_calibration(int(rate), layout_top(rate, top_hz))


def tonal_exclusion(power, index):
    """`index` with the bins a stationary tone holds in the quietest 20 % of frames set to -1 (with at least 12 such frames)."""
    level = power[:, index >= 0].sum(axis=1)
    quiet = level <= np.percentile(level, QUIET_PCT)
    if quiet.sum() < MIN_TONAL_FRAMES:
        return index
    return np.where(pause_metrics.tonal_bins(power[quiet]), -1, index)


def frame_map(rate, n_frames, mask20, start=0):
    """A per-20-ms-frame mask mapped onto the STFT frame centres of a segment starting at sample `start`; past its end False."""
    frame20 = max(1, int(pause_metrics.FRAME_S * rate))
    index = (int(start) + np.arange(n_frames) * HOP) // frame20
    mask20 = np.asarray(mask20, dtype=bool)
    inside = index < len(mask20)
    out = np.zeros(n_frames, dtype=bool)
    out[inside] = mask20[index[inside]]
    return out


def dither_floor_per_bin():
    """16-bit TPDF requantisation noise (variance LSB^2 / 4) per bin in this module's scaling (2 sigma^2 / FRAME): -126.4 dB."""
    return 2.0 * ((1.0 / 32768.0) ** 2 / 4.0) / FRAME


def dither_limit():
    """The lowest per-bin floor a reading trusts: the 16-bit dither plus 10 dB."""
    return dither_floor_per_bin() * 10.0 ** (DITHER_MARGIN_DB / 10.0)


def reference_bands(anchor, centres, prog_bandwidth_hz, top_hz):
    """The bands that set the hiss level: finite, centred from R0's programme bandwidth up to min(12 kHz, top)."""
    if anchor is None or not prog_bandwidth_hz:
        return np.zeros(len(centres), dtype=bool)
    top = min(HISS_REF_TOP_HZ, top_hz or np.inf)
    return (centres >= prog_bandwidth_hz) & (centres < top) & np.isfinite(anchor)


def hiss_cap(anchor, centres, prog_bandwidth_hz, top_hz):
    """The anchor with every band held at the hiss level above the programme band plus 3 dB; NaN kept.

    The level is the median anchor over `reference_bands`; with fewer than two of them the anchor is returned as it is.
    """
    ref = reference_bands(anchor, centres, prog_bandwidth_hz, top_hz)
    if ref.sum() < MIN_REF_BANDS:
        return anchor
    level = float(np.median(anchor[ref]))
    return np.where(np.isfinite(anchor), np.minimum(anchor, level * 10.0 ** (HISS_SLACK_DB / 10.0)), np.nan)


def chunk_bounds(n_samples, step):
    """`[(start, stop)]` of `step`-sample chunks; a tail under half a chunk joins the one before."""
    starts = list(range(0, n_samples, step))
    if len(starts) > 1 and n_samples - starts[-1] < step // 2:
        starts.pop()
    return list(zip(starts, starts[1:] + [n_samples]))


def chunked_pause_mask(mono, rate):
    """R4's true-pause mask on 20 ms frames, read per 60 s chunk (a whole-tape mask held about 10 MB per chunk at once)."""
    frame = max(1, int(pause_metrics.FRAME_S * rate))
    if len(mono) < 4 * FRAME:
        return np.zeros(len(mono) // frame, dtype=bool)
    step = int(round(MASK_CHUNK_S / pause_metrics.FRAME_S)) * frame
    masks = [pause_metrics.true_pause_mask(np.asarray(mono[a:b], dtype=np.float64), rate)[0] for a, b in chunk_bounds(len(mono), step)]
    return np.concatenate(masks)


def _segment_rows(segment, rate, mask20, start, top_hz):
    """`(pause rows, live rows, centres)` of one 15 s segment's valley levels, or None when it is too short."""
    if len(segment) < 4 * FRAME:
        return None
    freqs, power = stft_power(segment, rate)
    index, centres = band_layout(freqs, rate, top_hz)
    valley = valley_levels(power, tonal_exclusion(power, index), len(centres), band_shape(power))
    level = power.sum(axis=1)
    live = level > level.max() * 10.0 ** (-LIVE_RANGE_DB / 10.0)
    return valley[frame_map(rate, len(power), mask20, start) & live], valley[live], centres


def anchor_rows(mono, rate, mask20, top_hz):
    """`(pause rows, live rows, centres)` over every 15 s segment; centres None when no segment was long enough."""
    size = int(ANCHOR_SEGMENT_S * rate)
    pause, live, centres = [], [], None
    for start in range(0, len(mono), size):
        found = _segment_rows(mono[start:][:size], rate, mask20, start, top_hz)
        if found is not None:
            pause.append(found[0])
            live.append(found[1])
            centres = found[2]
    return pause, live, centres


def _level_of(rows, calib):
    """C_b x P10 over the stacked rows, bands under the dither limit NaN; None with no row."""
    stacked = np.concatenate(rows) if rows else np.zeros((0, len(calib)))
    if len(stacked) == 0:
        return None
    level = column_stat(stacked, lambda column: np.percentile(column, FLOOR_PCT)) * calib
    return np.where(level >= dither_limit(), level, np.nan)


def _half_finite(level):
    """Whether at least half the bands of `level` were read."""
    return level is not None and np.isfinite(level).sum() >= 0.5 * len(level)


def _anchor_level(pause, live, calib, pause_s):
    """`(level, how)`: from the pauses when at least 1 s of them leaves half the bands finite, else every live frame."""
    level = _level_of(pause, calib) if pause_s >= MIN_ANCHOR_S else None
    if _half_finite(level):
        return level, PAUSES
    level = _level_of(live, calib)
    if level is None or not np.isfinite(level).any():
        return None, NO_FLOOR
    return level, ALL_FRAMES


def anchor_validated(how, prog_bandwidth_hz, reference_count):
    """An anchor read from the pauses, with a programme band of 2 kHz or more and at least two reference bands."""
    wide = (prog_bandwidth_hz or 0.0) >= VALIDATED_MIN_BANDWIDTH_HZ
    return bool(how == PAUSES and wide and reference_count >= MIN_REF_BANDS)


def source_anchor(mono, rate, prog_bandwidth_hz=None, brickwall_hz=None):
    """The source's `Anchor` (module docstring), read once per file on its raw mono."""
    pause, live, centres = anchor_rows(mono, rate, chunked_pause_mask(mono, rate), brickwall_hz)
    if centres is None:
        return Anchor(None, np.zeros(0), 0.0, NO_FLOOR, False)
    pause_s = sum(len(rows) for rows in pause) * HOP / rate
    top = layout_top(rate, brickwall_hz)
    level, how = _anchor_level(pause, live, white_calibration(rate, top), pause_s)
    if level is None:
        return Anchor(None, centres, pause_s, how, False)
    validated = anchor_validated(how, prog_bandwidth_hz, int(reference_bands(level, centres, prog_bandwidth_hz, top).sum()))
    return Anchor(hiss_cap(level, centres, prog_bandwidth_hz, top), centres, pause_s, how, validated)


def loud_db(mono):
    """The mean-square level of `mono`'s loud frames (>= p70 of 1024-sample frame power), dBFS; a non-finite sample reads 0."""
    return float(10.0 * np.log10(auditory.loud_frame_power(np.nan_to_num(np.asarray(mono, dtype=np.float32))[:, np.newaxis])))


def spl_offset_db(mono):
    """dB to add to a mean-square dBFS so `mono`'s loud frames play at `L_CRIT_DB`."""
    return L_CRIT_DB - loud_db(mono)


def ms_frames(mono, rate, frame):
    """`(freqs, power (frames, bins))` on non-overlapping `frame`-sample Hann frames, scaled so a band's sum is its mean square."""
    freqs, power, _level = dsp_metrics.framed_psd(np.asarray(mono, dtype=np.float64), rate, frame)
    return freqs, power * (2.0 / (frame * float(np.sum(np.hanning(frame) ** 2))))
