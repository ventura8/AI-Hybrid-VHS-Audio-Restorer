"""Ear v4, reading 1: the tape hiss left under speech or singing, over the programme's masked threshold.

Per window (15 s, hop 7.5, the runner's aligned and gain-matched pair), source and output on
one timeline:

1. Window guard: a non-finite or still sample, a programme-band power envelope (300-4000 Hz,
   on the STFT frames below) correlating under 0.5 between the sides, or a programme gain
   over 24 dB (R1's match, `auditory.programme_gain_db`) reads every name None. The
   envelope, not the waveform: a render whose programme band is polarity-inverted and 2 ms
   off the runner's broadband lag (Gaudeamus1's renders: waveform correlation 0.31-0.34,
   envelope 0.96-1.00) is the same programme to a power reading, and the least-squares gain
   match then shrinks it in every window (Gaudeamus1 8.2-8.3 dB, Gaudeamus5 14.6-14.9 dB, at
   envelope correlations of 0.995-0.998), which R1's gain undoes; the gain bound is only a
   sanity limit once the envelopes match.
2. The STFT frames wholly inside the window (`whole_frames`: the centred first and last are half
   padding), the two at either end neither active nor pause (`edge_frames`: their tests two
   frames away read past the window); validation round 2 found a 0.37-sample shift of the output
   moving a window 0.26-2.4 dB through those frames. `hiss_floor` layout and valley levels on the
   source; the floor under the programme is C_b x P10 of the valley over the active frames of
   each 1 s block (+-1.5 s), held no higher than the source's anchor, whose unread bands take the anchor interpolated across them (a
   held note in the quiet frames left a band of a music fixture unread and its floor
   uncapped: the note, 15 dB over the hiss). A true pause (a DSP-VAD non-speech run of 200 ms
   or more whose kept-band power sits within 6 dB of the floor) is not active.
3. Noise-dominated cells (NDC): an active frame whose band valley sits within 1.5 dB of the
   floor two frames before and after (never the frame itself), a ring mean (bins f+-2..4) at
   or under the floor and no partial passing through the same bin two frames away (that bin
   10x over its own ring). The cell's own power never selects it: a cap on it (10x its ring)
   kept the low draws and read a fresh realisation of the same noise 3.7 % (0.16 dB per
   band) over the source. A band is readable with at least 40 such cells, 2 % of its active
   cells, a source NDC mean within -2..+1 dB of the floor and a floor 10 dB over the 16-bit
   dither.
4. The removal per band and frame is sum O / sum S over NDC cells (the output in the
   source's gain), locally over +-0.5 s where 40 cells allow, read apart in two classes of
   band-frame: free, where the band's mean power two frames before and after stays within
   1.3x the floor (no programme reaches the band there: gaps between words, the bands above
   the programme), and busy, the rest. Each frame takes its own class's ratio, so hiss the
   pause floor keeper adds to the gaps between words is read in those gaps.
5. Spectral cap: a band's ratio may stand at most 6 dB over the ratio the free cells of the
   other bands of 1 kHz and up (outside +-2 bands, +-1 frame) show at the same instant. Under
   the hiss the source hides programme it does not show: a hiss-free output reveals it, and
   in the band-edge bands of a 4.5 kHz programme it sits in the noise-dominated cells at
   9-16 dB under the floor (the top tenth of those cells carrying 74 % of it), where the
   ratio read it as hiss and a perfect output read +9..+22 dB (fixtures). Revealed programme
   stays in the bands the programme reaches; residual hiss is broadband, and a residual
   with more than 6 dB more hiss in one region than everywhere else reads 6 dB over the
   rest there. Unreadable bands take the readable ones' ratio interpolated in dB over log2
   f, the pooled ratio outside their range, capped at 0 dB and bounded by each side's own
   valley (`valley_bound`). Hiss band power H = floor x bins x ratio, and on every band no
   more than the side's own band power over three frames.
6. Masking, Johnston 1988: offset alpha (14.5 + z) + (1 - alpha) 5.5 dB, alpha the
   programme's own tonality read on the source with its floor removed. One masker for both
   sides, the source's band power less its hiss: read on the output's own band power, the
   threshold carried the output's own draw of the residual, and the source-chosen frames
   leaned the source's masker up against a fresh realisation of the same noise, which read
   +0.4..+0.9 dB more hiss (fixtures, 8 seeds). Each side plays at its own loud frames at
   `hiss_floor.L_CRIT_DB` (the source at `context.src_spl_db`, the output, in the source's
   gain, at L_crit less its own loud level plus that gain), the pause reading's convention:
   one listener sets the volume once per file. Spread 27 / 12 dB per ERB, forward-spread over
   two frames (PEAQ: tau = 8 ms + (100 / fc)(30 - 8) ms), floored at Terhardt's threshold.
7. NMR per frame and band, integrated as 10 log10 sqrt(sum NMR^2) over (a) the readable bands
   of 1 kHz and up, the measured margin; (b) every band of 1 kHz and up, the all-band margin
   (singing leaves few readable bands); (c) the readable bands of 3 kHz and up. Each margin
   is the P90 over the active frames; `audible_frac` the share of active frames over 0 dB.
   The ranked form is the excess, max(0, margin): inaudible is the target and nothing under it
   counts. Under two readable bands of 1 kHz and up the margins and guards read None; on a
   window whose anchor is not validated (`hiss_floor.source_anchor`) the margins, excesses and
   the audible share read None while the guards and the removal still read.
8. Guards, paired as (0, value): `hiss_pump_db` (the hiss kept under the programme against the
   pauses' own removal, median over bands of 1 kHz and up: positive = gated out of the pauses
   and left under speech), `hiss_prog_kept_hf_db` (per band of 3 kHz and up, the median
   output over source on active cells standing 18 dB over the floor, both held at the 16-bit
   dither, averaged over the bands holding 20 such cells: dulling inside the bands the margin
   rewards, a cut of a few bands as visible as a low pass) and `hiss_island_kurt` (R4's frame
   kurtosis on 20 ms frames of the active frames, 3 kHz to 1.06 x the top readable band, the
   output in the source's gain; None where the output's margin is under -10 dB, where the
   texture is inaudible).

Checked on `artifacts/realistic-v2` m12 tape-noise fixtures with two spliced 1.2 s gaps at the
tape's noise relation (`scripts/check_hiss_fixtures.py`, docs/ear_v4_hiss.md): see the
acceptance record for the perfect output, the oracle ladder and the same-percept rows. A
full-band programme (not low-passed) reveals programme in every band, which the cap cannot tell
from hiss: there a perfect output still reads above 0.
"""

import numpy as np
import scipy.ndimage

from scripts.restoration_quality import auditory, balance_metrics, dsp_metrics, hiss_floor, pause_metrics

RING = (2, 3, 4)
PEAK_CAP = 10.0
NOISE_LED_DB = 1.5
LED_OFFSET = 2
LEAK_DB = 1.0
LEAK_LOW_DB = -2.0
MIN_CELLS = 40
MIN_NDC_SHARE = 0.02
LOCAL_HALF_S = 0.5
PAUSE_SNR_DB = 6.0
MIN_PAUSE_S = 0.5
FREE_FACTOR = 1.3
SPECTRAL_SLACK_DB = 6.0
POOL_REACH = 2
POOL_FRAMES = 3
POWER_BOUND_FRAMES = 3
TAU_100 = 0.030
TAU_MIN = 0.008
TONAL_FULL_DB = 15.0
TONAL_CONTEXT_BINS = 24
SUMMARY_LO_HZ = 1000.0
HF_LO_HZ = 3000.0
PROG_MARGIN_DB = 18.0
ISLAND_MIN_MARGIN_DB = -10.0
AUDIBLE_DB = 0.0
BOUND_FRAMES = 5
MIN_KEPT_BAND_CELLS = 20
ISLAND_MIN_FRAMES = 25
ISLAND_MIN_BINS = 16
ISLAND_TOP_FACTOR = 1.06
MIN_READABLE_BANDS = 2
MIN_WINDOW_FRAMES = 4
MIN_ENV_CORR = 0.5
ENV_BAND_HZ = (300.0, 4000.0)
MAX_GAIN_DB = 24.0
TONE_MASKING_DB = 14.5
NOISE_MASKING_DB = 5.5
EPS = hiss_floor.EPS
PROGRAMME_NAMES = (
    "hiss_prog_margin_db",
    "hiss_prog_excess_db",
    "hiss_prog_margin_all_db",
    "hiss_prog_excess_all_db",
    "hiss_prog_margin_hf_db",
    "hiss_prog_audible_frac",
    "hiss_prog_removed_db",
    "hiss_prog_removed_hf_db",
    "hiss_pump_db",
    "hiss_island_kurt",
    "hiss_prog_kept_hf_db",
    "hiss_prog_active_s",
    "hiss_prog_bands",
    "hiss_prog_validated",
)
# The fields an unvalidated anchor nulls (step 7): the margins, their excesses and the audible share.
MARGIN_FIELDS = ("margin", "margin_all", "margin_hf", "audible_frac")


class Layout:
    """The window's band layout: bin -> band with the source's tones out, centres, kept bins per band, the shape, C_b."""

    def __init__(self, freqs, power, rate, top_hz):
        self.rate = rate
        index, self.centres = hiss_floor.band_layout(freqs, rate, top_hz)
        self.kindex = hiss_floor.tonal_exclusion(power, index)
        self.nbins = np.bincount(self.kindex[self.kindex >= 0], minlength=self.nb).astype(np.float64)
        self.shape = hiss_floor.band_shape(power)
        self.calib = hiss_floor.white_calibration(rate, top_hz)
        self.onehot = (self.kindex[:, np.newaxis] == np.arange(self.nb)[np.newaxis, :]).astype(np.float64)

    @property
    def nb(self):
        """The number of bands."""
        return len(self.centres)

    def band_power(self, power):
        """`(frames, bands)` power summed over each band's kept bins."""
        return power @ self.onehot

    def bins_of(self, band_mask):
        """Per bin, whether its band is in `band_mask` (False for the bins out of every band)."""
        out = np.zeros(len(self.kindex), dtype=bool)
        kept = self.kindex >= 0
        out[kept] = band_mask[self.kindex[kept]]
        return out

    def cells_of(self, flags):
        """`(frames, bins)` from per-band frame flags `(frames, bands)`; False on the bins out of every band."""
        out = np.zeros((flags.shape[0], len(self.kindex)), dtype=bool)
        kept = self.kindex >= 0
        out[:, kept] = flags[:, self.kindex[kept]]
        return out


def _moving(side):
    """At least two finite samples that are not all equal."""
    return len(side) > 1 and bool(np.isfinite(side).all()) and float(np.std(side)) > 0.0


def window_usable(src, out):
    """Step 1's sample half: at least four STFT frames of finite samples, both sides moving."""
    return min(len(src), len(out)) >= MIN_WINDOW_FRAMES * hiss_floor.FRAME and _moving(src) and _moving(out)


def envelope_correlation(freqs, src_power, out_power):
    """The correlation of the two sides' 300-4000 Hz frame power (0 when either side holds a constant envelope)."""
    band = (freqs >= ENV_BAND_HZ[0]) & (freqs < ENV_BAND_HZ[1])
    src_env, out_env = src_power[:, band].sum(axis=1), out_power[:, band].sum(axis=1)
    if float(np.std(src_env)) == 0.0 or float(np.std(out_env)) == 0.0:
        return 0.0
    return float(np.corrcoef(src_env, out_env)[0, 1])


def programme_gain(src_power, out_power, freqs):
    """R1's programme gain match in dB (0 with too few programme cells), or None past +-24 dB (a mismatched window)."""
    gain = auditory.programme_gain_db(src_power.T, out_power.T, freqs, balance_metrics.guarded_cells(src_power.T))
    gain = 0.0 if gain is None else float(gain)
    return None if abs(gain) > MAX_GAIN_DB else gain


def anchor_level(context):
    """The anchor's per-band floor, or None without one."""
    anchor = context.anchor
    return None if anchor is None else anchor.level


def filled_anchor(anchor):
    """The anchor with its unread bands (NaN or non-positive) interpolated in log level over the band index.

    The anchor is returned as it is with fewer than two bands read or none unread; past the read ones the nearest holds.
    """
    read = np.isfinite(anchor) & (anchor > 0.0)
    if read.sum() < 2 or read.all():
        return anchor
    index = np.arange(len(anchor))
    return 10.0 ** np.interp(index, index[read], np.log10(anchor[read]))


def capped(floor, anchor):
    """`floor` held no higher than the anchor, band by band, the anchor's unread bands interpolated across them."""
    if anchor is None or len(anchor) != floor.shape[-1]:
        return floor
    return np.fmin(floor, filled_anchor(np.asarray(anchor, dtype=np.float64)))


def output_spl_db(context, gain_db):
    """dB to add to the output's mean square, in the source's gain, so that its own loud frames play at `L_CRIT_DB`."""
    return hiss_floor.L_CRIT_DB - context.out_loud_db + gain_db


def whole_frames(power):
    """The STFT frames that lie wholly inside the window: the centred first and last frames are half padding.

    A half-padded frame reads the masker 3 dB low under the full hiss estimate, and it sits among the quietest frames
    that set the shape, the tonal bins and the P10 floor: one sample more or less at the end of a window (the runner's
    alignment, under a 0.37-sample shift of the output) moved a band of Gaudeamus5 across the leak test and the window's
    margin 2.4 dB.
    """
    return power[1:-1]


def edge_frames(count):
    """Per STFT frame, whether it lies within two frames of either end of the window (step 2).

    The tests two frames away read past the window there; on Tele7abc a 0.37-sample shift of the output moved one
    window's margin 0.26 dB through its first two frames alone.
    """
    edge = np.zeros(count, dtype=bool)
    tail = max(0, count - LED_OFFSET)
    edge[:LED_OFFSET] = True
    edge[tail:] = True
    return edge


def true_pauses(src, rate, power, layout, floor0):
    """Per STFT frame: inside a DSP-VAD non-speech run of 200 ms or more, with kept-band power within 6 dB of the floor."""
    frame20 = max(1, int(pause_metrics.FRAME_S * rate))
    speech = pause_metrics.dsp_vad(src, rate, frame20)
    runs = pause_metrics.long_runs(~speech, max(1, int(round(pause_metrics.MIN_RUN_S * rate / frame20))))
    near = power[:, layout.kindex >= 0].sum(axis=1) <= np.nansum(floor0 * layout.nbins) * 10.0 ** (PAUSE_SNR_DB / 10.0)
    return hiss_floor.frame_map(rate, len(power), runs, start=hiss_floor.HOP) & near


def ring_mean(power):
    """Per cell, the mean power of bins f+-2, f+-3, f+-4 in the same frame (edges reflect)."""
    kernel = np.zeros(9)
    for reach in RING:
        kernel[4 - reach] = kernel[4 + reach] = 1.0
    return scipy.ndimage.correlate1d(power, kernel / kernel.sum(), axis=1, mode="reflect")


def two_frames_away(ok):
    """Per frame (first axis), whether `ok` holds two frames before and two after; past the window's ends it holds."""
    pad = np.ones((LED_OFFSET,) + ok.shape[1:], dtype=bool)
    return np.concatenate([pad, ok[:-LED_OFFSET]]) & np.concatenate([ok[LED_OFFSET:], pad])


def noise_led(valley, floor):
    """Per frame: the band's valley within 1.5 dB of the floor in the frames two before and two after (never the frame itself)."""
    return two_frames_away(valley <= floor * 10.0 ** (NOISE_LED_DB / 10.0))


def peaky_neighbours(src_power, ring):
    """Per cell, whether the same bin two frames before or after stands over 10x its own ring (a partial passing through)."""
    return ~two_frames_away(src_power <= PEAK_CAP * ring)


def ndc_mask(src_power, floor, valley, layout, active):
    """`(frames, bins)` noise-dominated cells (module docstring, step 3)."""
    cell_floor = np.full(src_power.shape, -1.0)
    led = np.zeros(src_power.shape, dtype=bool)
    for band in range(layout.nb):
        sel = layout.kindex == band
        band_floor = np.nan_to_num(floor[:, band], nan=-1.0)
        cell_floor[:, sel] = band_floor[:, np.newaxis]
        led[:, sel] = (noise_led(valley[:, band], band_floor) & active)[:, np.newaxis]
    ring = ring_mean(src_power)
    return led & (ring <= cell_floor) & ~peaky_neighbours(src_power, ring) & (layout.kindex >= 0)[np.newaxis, :]


def ndc_sums(src_power, out_power, cells, floor, layout):
    """`(source, output, count, floor)` sums per frame and band over the NDC cells."""
    weights = cells.astype(np.float64)
    count = layout.band_power(weights)
    return layout.band_power(src_power * weights), layout.band_power(out_power * weights), count, count * np.nan_to_num(floor, nan=0.0)


def readable_bands(sums, floor, layout, active):
    """The bands the removal can be read in (module docstring, step 3)."""
    src_sum, _out_sum, count, floor_sum = (part.sum(axis=0) for part in sums)
    leak = 10.0 * np.log10((src_sum + EPS) / (floor_sum + EPS))
    floor_ok = np.median(np.nan_to_num(floor[active], nan=0.0), axis=0) >= hiss_floor.dither_limit()
    enough = (count >= MIN_CELLS) & (count >= MIN_NDC_SHARE * np.maximum(active.sum() * layout.nbins, 1.0))
    return enough & (leak <= LEAK_DB) & (leak >= LEAK_LOW_DB) & floor_ok


def local_ratio(sums, fps, ratio_w, readable):
    """Per frame and band, output over source on the NDC cells within +-0.5 s; the window's ratio with fewer than 40 cells.

    The running sums are held at 0 or above: the filter's rounding left them slightly negative beside empty cells, and a
    negative ratio put NaN into the filled bands (a fixture probe's warning in `fill_bands`).
    """
    src_sum, out_sum, count, _floor = sums
    size = 2 * max(1, int(round(LOCAL_HALF_S * fps))) + 1
    src_local, out_local, cells = (
        np.maximum(scipy.ndimage.uniform_filter1d(part, size, axis=0, mode="nearest") * size, 0.0) for part in (src_sum, out_sum, count)
    )
    ratio = np.where(cells >= MIN_CELLS, (out_local + EPS) / (src_local + EPS), ratio_w[np.newaxis, :])
    return np.where(readable[np.newaxis, :], ratio, np.nan)


def window_ratio(sums, fallback):
    """Per band, output over source over every frame of `sums`; `fallback` where they hold fewer than 40 cells."""
    count = sums[2].sum(axis=0)
    return np.where(count >= MIN_CELLS, (sums[1].sum(axis=0) + EPS) / (sums[0].sum(axis=0) + EPS), fallback)


def free_band_frames(src_power, floor, layout):
    """`(frames, bands)`: the band's mean power two frames before and after within 1.3x the floor (no programme reaches it)."""
    mean = layout.band_power(src_power) / np.maximum(layout.nbins, 1.0)[np.newaxis, :]
    return two_frames_away(mean <= FREE_FACTOR * np.nan_to_num(floor, nan=-1.0))


def pool_sums(free_sums, layout):
    """`(source, output, count)` per frame and band over the free cells of the other bands of 1 kHz and up (step 5)."""
    high = (layout.centres >= SUMMARY_LO_HZ)[np.newaxis, :]
    parts = np.stack(
        [scipy.ndimage.uniform_filter1d(part * high, POOL_FRAMES, axis=0, mode="nearest") * POOL_FRAMES for part in free_sums[:3]]
    )
    span = 2 * POOL_REACH + 1
    near = scipy.ndimage.uniform_filter1d(parts, span, axis=2, mode="constant") * span
    return np.maximum(parts.sum(axis=2, keepdims=True) - near, 0.0)


def spectral_cap(ratio, pool):
    """`ratio` held within 6 dB over the pool's own ratio, where the pool holds at least 40 cells (module docstring, step 5)."""
    limit = 10.0 ** (SPECTRAL_SLACK_DB / 10.0) * (pool[1] + EPS) / (pool[0] + EPS)
    return np.where((pool[2] >= MIN_CELLS) & np.isfinite(ratio), np.minimum(ratio, limit), ratio)


def fill_bands(ratio_bt, centres, pooled):
    """Unreadable bands interpolated in dB over log2 f between readable ones, `pooled` outside; filled ratios capped at 0 dB."""
    ok = np.isfinite(ratio_bt[0])
    if ok.all():
        return ratio_bt
    octave = np.log2(centres)
    level = np.minimum(10.0 * np.log10(np.where(np.isfinite(ratio_bt), ratio_bt, 1.0)), 0.0)
    edge = 10.0 * np.log10(pooled)
    out = np.array([10.0 ** (np.interp(octave, octave[ok], row[ok], left=edge, right=edge) / 10.0) for row in level])
    out[:, ok] = ratio_bt[:, ok]
    return out


def valley_bound(valley, calib, nbins):
    """Per frame and band, the most hiss a side can hold: its valley (median over 5 frames) x C_b x bins."""
    smooth = scipy.ndimage.median_filter(np.nan_to_num(valley, nan=0.0), size=(BOUND_FRAMES, 1), mode="nearest")
    return smooth * calib[np.newaxis, :] * nbins[np.newaxis, :]


def power_bound(power, layout):
    """Per frame and band, the side's own band power averaged over three frames: no more hiss than the side holds there."""
    return scipy.ndimage.uniform_filter1d(layout.band_power(power), POWER_BOUND_FRAMES, axis=0, mode="nearest")


def zbark(freq_hz):
    """Zwicker's critical-band rate in Bark."""
    freq = np.asarray(freq_hz, dtype=np.float64)
    return 13.0 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq / 7500.0) ** 2)


def forward_spread(excitation, centres, rate):
    """PEAQ time spreading over two frames: max(E(t), a E(t-1), a^2 E(t-2)), tau = 8 ms + (100 / fc)(30 - 8) ms."""
    tau = TAU_MIN + (100.0 / np.maximum(centres, 1.0)) * (TAU_100 - TAU_MIN)
    decay = np.exp(-hiss_floor.HOP / (tau * rate))[np.newaxis, :]
    out = excitation.copy()
    out[1:] = np.maximum(out[1:], excitation[:-1] * decay)
    out[2:] = np.maximum(out[2:], excitation[:-2] * decay * decay)
    return out


def _context_columns(layout, band):
    """The bin range of a band's tonality context: its own bins grown to at least 24."""
    cols = np.flatnonzero(layout.kindex == band)
    lo, hi = cols[0], cols[-1] + 1
    grow = max(0, TONAL_CONTEXT_BINS - (hi - lo))
    return max(0, lo - grow // 2), min(len(layout.kindex), hi + grow - grow // 2), cols


def _band_tonality(white, layout, band, floor):
    """alpha of one band per frame: the programme's peak over its valley in the context, floor out, over 15 dB (0..1)."""
    first, last, cols = _context_columns(layout, band)
    context = white[:, first:last]
    noise = np.nan_to_num(floor[:, band], nan=0.0) / layout.shape[cols].mean()
    peak = context.mean(axis=1) - noise
    valley = np.maximum(hiss_floor.low_quantile(context, hiss_floor.VALLEY_Q) / hiss_floor.VALLEY_NORM - noise, 0.1 * np.maximum(peak, EPS))
    level = 10.0 * np.log10(np.maximum(peak, EPS) / (valley + EPS))
    return np.where(peak >= noise, np.clip(level / TONAL_FULL_DB, 0.0, 1.0), 0.0)


def programme_tonality(src_power, layout, floor):
    """alpha per frame and band of the programme alone, read on the source with its floor taken out (both sides share it)."""
    white = src_power / layout.shape[np.newaxis, :]
    alpha = np.zeros((src_power.shape[0], layout.nb))
    for band in range(layout.nb):
        if layout.nbins[band] > 0 and np.isfinite(floor[:, band]).any():
            alpha[:, band] = _band_tonality(white, layout, band, floor)
    return alpha


def nmr(masker, hiss, alpha, layout, spl_db):
    """Per frame and band, the hiss's power over the masked threshold of `masker`, both at `spl_db` (module docstring, step 6)."""
    gain = 10.0 ** (spl_db / 10.0)
    offset = alpha * (TONE_MASKING_DB + zbark(layout.centres)[np.newaxis, :]) + (1.0 - alpha) * NOISE_MASKING_DB
    spread = forward_spread((masker * gain) @ auditory.spreading_matrix(layout.nb).T, layout.centres, layout.rate)
    quiet = 10.0 ** (auditory.threshold_in_quiet_db(layout.centres) / 10.0)
    return (hiss * gain + EPS) / np.maximum(spread * 10.0 ** (-offset / 10.0), quiet[np.newaxis, :])


def integrated_db(ratio, bands):
    """Per frame, 10 log10 sqrt(sum NMR^2) over `bands` (Green's energy summation)."""
    return 10.0 * np.log10(np.sqrt(np.sum(ratio[:, bands] ** 2, axis=1)) + EPS)


def _p90(values):
    return float(np.percentile(values, 90.0))


def side_margins(ratio, layout, readable, active):
    """`{margin, margin_all, margin_hf, audible_frac}` of one side's NMR over the active frames."""
    measured = (layout.centres >= SUMMARY_LO_HZ) & readable
    high = (layout.centres >= HF_LO_HZ) & readable
    level = integrated_db(ratio, measured)[active]
    return {
        "margin": _p90(level),
        "margin_all": _p90(integrated_db(ratio, layout.centres >= SUMMARY_LO_HZ)[active]),
        "margin_hf": _p90(integrated_db(ratio, high)[active]) if high.any() else None,
        "audible_frac": float(np.mean(level > AUDIBLE_DB)),
    }


def band_summary(values, centres, lo_hz):
    """The median over the finite bands centred at `lo_hz` and up, None with none."""
    values = np.asarray(values, dtype=np.float64)
    sel = (centres >= lo_hz) & np.isfinite(values)
    return float(np.median(values[sel])) if sel.any() else None


def pump_db(powers, pause, ratio_w, layout, rate):
    """The hiss kept under the programme over the pauses' own output/source ratio, median over bands of 1 kHz and up.

    None with under 0.5 s of true pause. Positive: the hiss rides on the programme while the pauses are emptied.
    """
    src_power, out_power = powers
    if pause.sum() < MIN_PAUSE_S * rate / hiss_floor.HOP:
        return None
    paused = (layout.band_power(out_power[pause]).mean(axis=0) + EPS) / (layout.band_power(src_power[pause]).mean(axis=0) + EPS)
    return band_summary(10.0 * np.log10(ratio_w / paused), layout.centres, SUMMARY_LO_HZ)


def programme_cells(src_power, floor, layout, active):
    """Active cells of 3 kHz and up standing 18 dB over the floor: where the programme, not the hiss, holds a bin."""
    kept = layout.kindex >= 0
    cell_floor = np.full(src_power.shape, np.inf)
    cell_floor[:, kept] = np.nan_to_num(floor[:, layout.kindex[kept]], nan=np.inf)
    cells = active[:, np.newaxis] & layout.bins_of(layout.centres >= HF_LO_HZ)[np.newaxis, :]
    return cells & (src_power >= cell_floor * 10.0 ** (PROG_MARGIN_DB / 10.0))


def band_medians(level, cells, layout):
    """Per band holding at least 20 of `cells`, the median of `level` over them."""
    medians = []
    for band in range(layout.nb):
        sel = cells[:, layout.kindex == band]
        if int(sel.sum()) >= MIN_KEPT_BAND_CELLS:
            medians.append(float(np.median(level[:, layout.kindex == band][sel])))
    return medians


def kept_hf_db(powers, floor, layout, active):
    """The mean over bands of 3 kHz and up of each band's median output over source dB on its programme cells.

    Both sides are held at the 16-bit dither per bin (requantising a render whose top sits near the 16-bit floor moved
    the guard 2-3 dB); a band counts with at least 20 programme cells; None with no such band.
    """
    src_power, out_power = powers
    dither = hiss_floor.dither_floor_per_bin()
    level = 10.0 * np.log10(np.maximum(out_power, dither) / np.maximum(src_power, dither))
    medians = band_medians(level, programme_cells(src_power, floor, layout, active), layout)
    return float(np.mean(medians)) if medians else None


def island_top_hz(layout, readable):
    """1.06 x the top readable band of 3 kHz and up, None with none."""
    high = layout.centres[readable & (layout.centres >= HF_LO_HZ)]
    return None if high.size == 0 else float(high.max() * ISLAND_TOP_FACTOR)


def active_20ms(active, count, frame, rate):
    """The STFT frames' activity mapped onto `count` 20 ms frames of `frame` samples (by each one's centre).

    The window keeps its STFT frames from the second on (`whole_frames`), so frame j is centred at (j + 1) x HOP.
    """
    centres = (np.arange(count) * frame + frame / 2.0) / rate * rate / hiss_floor.HOP
    return active[np.clip(centres.astype(int) - 1, 0, len(active) - 1)]


def island_kurt(pair, rate, active, top_hz, gain_db):
    """R4's frame kurtosis (ln output over source) on 20 ms frames of the active frames, 3 kHz to `top_hz`; None when unread."""
    if top_hz is None:
        return None
    frame = max(1, int(pause_metrics.FRAME_S * rate))
    freqs, src_power, _level = dsp_metrics.framed_psd(pair[0], rate, frame)
    out_power = dsp_metrics.framed_psd(pair[1], rate, frame)[1] / 10.0 ** (gain_db / 10.0)
    act = active_20ms(active, len(src_power), frame, rate)
    bins = (freqs >= HF_LO_HZ) & (freqs < top_hz)
    if act.sum() < ISLAND_MIN_FRAMES or bins.sum() < ISLAND_MIN_BINS:
        return None
    src_cells = src_power[act][:, bins]
    return float(pause_metrics.island_kurtosis(src_cells, pause_metrics.floored(out_power[act][:, bins], src_cells)))


class Window:
    """One window's analysis (module docstring), built step by step; `readings()` returns the internal values or None."""

    def __init__(self, src, out, rate, context):
        self.pair, self.context = (src, out), context
        freqs, src_power = hiss_floor.stft_power(src, rate)
        out_power = hiss_floor.stft_power(out, rate)[1]
        count = min(len(src_power), len(out_power))
        self.freqs, self.src_power, self.out_power = freqs, whole_frames(src_power[:count]), whole_frames(out_power[:count])
        self.layout = Layout(freqs, self.src_power, rate, context.top_hz)
        self.valley = hiss_floor.valley_levels(self.src_power, self.layout.kindex, self.layout.nb, self.layout.shape)

    @property
    def rate(self):
        """The window's sample rate."""
        return self.layout.rate

    def floors(self):
        """`(true pauses, active frames, floor under the programme)`, or None with under 0.45 s of active frames."""
        anchor = anchor_level(self.context)
        floor0 = capped(hiss_floor.percentile_floor(self.valley) * self.layout.calib, anchor)
        edge = edge_frames(len(self.src_power))
        pause = true_pauses(self.pair[0], self.rate, self.src_power, self.layout, floor0) & ~edge
        active = ~pause & ~edge
        if active.sum() < max(3.0, hiss_floor.MIN_SPAN_S * self.rate / hiss_floor.HOP):
            return None
        return pause, active, capped(hiss_floor.block_floor(self.valley, active, self.rate, self.layout.calib), anchor)

    def matched_gain(self):
        """R1's programme gain, or None when the programme envelopes do not match or the gain is past 24 dB (step 1)."""
        if envelope_correlation(self.freqs, self.src_power, self.out_power) < MIN_ENV_CORR:
            return None
        return programme_gain(self.src_power, self.out_power, self.freqs)

    def readings(self):
        """The internal readings of the window, or None when it has too little programme or a mismatched programme."""
        gain = self.matched_gain()
        found = self.floors() if gain is not None else None
        if found is None:
            return None
        pause, active, floor = found
        self.out_power = self.out_power / 10.0 ** (gain / 10.0)
        return self._measured(pause, active, floor, gain)

    def _measured(self, pause, active, floor, gain):
        """The removal per band, then the margins and guards where at least two bands of 1 kHz and up are readable."""
        cells = ndc_mask(self.src_power, floor, self.valley, self.layout, active)
        sums = ndc_sums(self.src_power, self.out_power, cells, floor, self.layout)
        readable = readable_bands(sums, floor, self.layout, active)
        ratio_w = np.where(readable, (sums[1].sum(axis=0) + EPS) / (sums[0].sum(axis=0) + EPS), np.nan)
        removed = -10.0 * np.log10(ratio_w)
        bands = int(np.sum(readable & (self.layout.centres >= SUMMARY_LO_HZ)))
        found = {
            "removed_db": band_summary(removed, self.layout.centres, SUMMARY_LO_HZ),
            "removed_hf_db": band_summary(removed, self.layout.centres, HF_LO_HZ),
            "active_s": float(active.sum() * hiss_floor.HOP / self.rate),
            "bands": float(bands),
        }
        if bands >= MIN_READABLE_BANDS:
            pooled = min(1.0, (sums[1].sum(axis=0)[readable].sum() + EPS) / (sums[0].sum(axis=0)[readable].sum() + EPS))
            found.update(self._audibility(cells, (readable, ratio_w, pooled), (pause, active), floor, gain))
        return found

    def _class_ratio(self, cells, floor, reading):
        """`(sums, per frame and band ratio)` of one class of cells: its local ratio, its window ratio, the plain one."""
        readable, ratio_w, _pooled = reading
        sums = ndc_sums(self.src_power, self.out_power, cells, floor, self.layout)
        fallback = window_ratio(sums, np.where(np.isfinite(ratio_w), ratio_w, 1.0))
        return sums, local_ratio(sums, self.rate / hiss_floor.HOP, fallback, readable)

    def _ratio(self, cells, floor, reading):
        """Per frame and band, the output's hiss over the source's (module docstring, steps 4 and 5), filled where unread."""
        free = free_band_frames(self.src_power, floor, self.layout)
        free_cells = self.layout.cells_of(free)
        free_sums, free_ratio = self._class_ratio(cells & free_cells, floor, reading)
        busy_ratio = self._class_ratio(cells & ~free_cells, floor, reading)[1]
        ratio = spectral_cap(np.where(free, free_ratio, busy_ratio), pool_sums(free_sums, self.layout))
        return fill_bands(ratio, self.layout.centres, reading[2])

    def _hiss(self, cells, floor, reading):
        """`(source hiss, output hiss)` band power per frame (module docstring, step 5)."""
        layout = self.layout
        floor_power = np.nan_to_num(floor, nan=0.0) * layout.nbins[np.newaxis, :]
        out_power = floor_power * self._ratio(cells, floor, reading)
        out_valley = hiss_floor.valley_levels(self.out_power, layout.kindex, layout.nb, layout.shape)
        measured = reading[0][np.newaxis, :]
        src_hiss = np.where(measured, floor_power, np.fmin(floor_power, valley_bound(self.valley, layout.calib, layout.nbins)))
        out_hiss = np.where(measured, out_power, np.fmin(out_power, valley_bound(out_valley, layout.calib, layout.nbins)))
        return np.fmin(src_hiss, power_bound(self.src_power, layout)), np.fmin(out_hiss, power_bound(self.out_power, layout))

    def _margins(self, hiss, alpha, frames, gain):
        """`(source, output)` margins of the two sides' hiss over the source programme's masked threshold (step 6)."""
        readable, active = frames
        masker = np.maximum(self.layout.band_power(self.src_power) - hiss[0], 0.0)
        levels = (self.context.src_spl_db, output_spl_db(self.context, gain))
        return tuple(
            side_margins(nmr(masker, side, alpha, self.layout, level), self.layout, readable, active) for side, level in zip(hiss, levels)
        )

    def _audibility(self, cells, reading, frames, floor, gain):
        """Both sides' margins and the three guards."""
        readable, ratio_w = reading[:2]
        pause, active = frames
        alpha = programme_tonality(self.src_power, self.layout, floor)
        src_side, out_side = self._margins(self._hiss(cells, floor, reading), alpha, (readable, active), gain)
        top = island_top_hz(self.layout, readable) if out_side["margin"] >= ISLAND_MIN_MARGIN_DB else None
        return {
            "src": src_side,
            "out": out_side,
            "pump_db": pump_db((self.src_power, self.out_power), pause, ratio_w, self.layout, self.rate),
            "kept_hf_db": kept_hf_db((self.src_power, self.out_power), floor, self.layout, active),
            "island_kurt": island_kurt(self.pair, self.rate, active, top, gain),
        }


def validated(context, found):
    """Whether the window's margins may be reported: a validated anchor and at least two readable bands of 1 kHz and up."""
    anchor = context.anchor
    return bool(anchor is not None and anchor.validated and found.get("bands", 0.0) >= MIN_READABLE_BANDS)


def _side(found, key, ok):
    """One side's margins, None where they were not read or the window is not validated."""
    side = found.get(key)
    return dict.fromkeys(MARGIN_FIELDS) if side is None or not ok else side


def _guard(found, key):
    value = found.get(key)
    return (None, None) if value is None else (0.0, value)


def paired(found, ok):
    """The `PROGRAMME_NAMES` pairs of one window's internal readings; `ok` says whether the margins may be reported."""
    src, out = _side(found, "src", ok), _side(found, "out", ok)
    margins = {
        "hiss_prog_margin_db": (src["margin"], out["margin"]),
        "hiss_prog_excess_db": (hiss_floor.excess(src["margin"]), hiss_floor.excess(out["margin"])),
        "hiss_prog_margin_all_db": (src["margin_all"], out["margin_all"]),
        "hiss_prog_excess_all_db": (hiss_floor.excess(src["margin_all"]), hiss_floor.excess(out["margin_all"])),
        "hiss_prog_margin_hf_db": (src["margin_hf"], out["margin_hf"]),
        "hiss_prog_audible_frac": (src["audible_frac"], out["audible_frac"]),
    }
    guards = {f"hiss_{name}": _guard(found, name) for name in ("pump_db", "island_kurt")}
    guards["hiss_prog_kept_hf_db"] = _guard(found, "kept_hf_db")
    guards["hiss_prog_removed_db"], guards["hiss_prog_removed_hf_db"] = _guard(found, "removed_db"), _guard(found, "removed_hf_db")
    coverage = {f"hiss_prog_{name}": (found[name], found[name]) for name in ("active_s", "bands")}
    coverage["hiss_prog_validated"] = (float(ok), float(ok))
    return {**margins, **guards, **coverage}


def programme_readings(src, out, rate, context):
    """`{name: (source, output)}` for `PROGRAMME_NAMES` on one aligned, gain-matched window (module docstring)."""
    src, out = np.asarray(src, dtype=np.float64), np.asarray(out, dtype=np.float64)
    if not window_usable(src, out):
        return hiss_floor.none_pairs(PROGRAMME_NAMES)
    found = Window(src, out, rate, context).readings()
    if found is None:
        return hiss_floor.none_pairs(PROGRAMME_NAMES)
    return paired(found, validated(context, found))
