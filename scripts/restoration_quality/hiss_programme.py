"""Ear v4, reading 1: the tape hiss left under speech or singing, over the programme's masked threshold.

Per window (15 s, hop 7.5, the runner's aligned and gain-matched pair), source and output on
one timeline:

1. Window guard: a non-finite sample, a correlation under 0.5 or a programme gain over 6 dB
   (R1's match, `auditory.programme_gain_db`) reads every name None. Without it the
   prototype's p90 over windows read +19..+28 on SOTI and Vaccin round-one renders, above the
   sources' own median: misaligned or mismatched windows, not hiss.
2. `hiss_floor` layout and valley levels on the source; the floor under the programme is
   C_b x P10 of the valley over the active frames of each 1 s block (+-1.5 s), held no higher
   than the source's anchor. A true pause (a DSP-VAD non-speech run of 200 ms or more whose
   kept-band power sits within 6 dB of the floor) is not active.
3. Noise-dominated cells (NDC): an active frame whose band valley sits within 1.5 dB of the
   floor two frames before and after (never the frame itself: selecting on its own noise draw
   read a fresh realisation 0.35 dB less removed), a ring mean (bins f+-2..4) at or under the
   floor and the cell no more than 10x its ring. A band is readable with at least 40 such
   cells, 2 % of its active cells, a source NDC mean within -2..+1 dB of the floor (no
   programme leaking in, no over-estimated floor) and a floor 10 dB over the 16-bit dither.
4. The removal per band is sum O / sum S over its NDC cells (the output in the source's gain),
   locally over +-0.5 s where 40 cells allow. Unreadable bands take the readable ones'
   ratio interpolated in dB over log2 f, the pooled ratio outside their range, capped at 0 dB
   and bounded by each side's own valley (`valley_bound`: a band whose valleys sit low cannot
   hold much hiss). Hiss band power H = floor x bins x ratio.
5. Masking, Johnston 1988: offset alpha (14.5 + z) + (1 - alpha) 5.5 dB, alpha the
   programme's own tonality read on the source with its floor removed (one estimate for both
   sides: read per side, a denoised output's clean valleys made it tonal and lenient). The
   masker is the band power less H at `hiss_floor.L_CRIT_DB` (the source's loud frames at
   75 dB SPL, the output in the source's gain), spread 27 / 12 dB per ERB, forward-spread over
   two frames (PEAQ: tau = 8 ms + (100 / fc)(30 - 8) ms), floored at Terhardt's threshold.
6. NMR per frame and band, integrated as 10 log10 sqrt(sum NMR^2) over (a) the readable bands
   of 1 kHz and up, the measured margin; (b) every band of 1 kHz and up, the all-band margin
   (singing leaves few readable bands); (c) the readable bands of 3 kHz and up. Each margin
   is the P90 over the active frames; `audible_frac` the share of active frames over 0 dB.
   The ranked form is the excess, max(0, margin): inaudible is the target and nothing under it
   counts. Under two readable bands of 1 kHz and up the margins and guards read None; on a
   window whose anchor is not validated (`hiss_floor.source_anchor`) the margins, excesses and
   the audible share read None while the guards and the removal still read.
7. Guards, paired as (0, value): `hiss_pump_db` (the hiss kept under the programme against the
   pauses' own removal, median over bands of 1 kHz and up: positive = gated out of the pauses
   and left under speech), `hiss_prog_kept_hf_db` (the output over the source on active cells
   of 3 kHz and up standing 18 dB over the floor: dulling inside the bands the margin
   rewards) and `hiss_island_kurt` (R4's frame kurtosis on 20 ms frames of the active frames,
   3 kHz to 1.06 x the top readable band, the output in the source's gain; None where the
   output's margin is under -10 dB, where the texture is inaudible).

Validated on `artifacts/realistic-v2` m12 tape-noise fixtures with two spliced 1.2 s gaps (the
prototype at 75 dB): every perfect output reads -8.79 or less, the oracle ladder is monotone,
gain +-0.1 dB and a 0.37-sample delay move nothing, a fresh realisation of the noise p95 0.617.
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
MIN_KEPT_CELLS = 50
ISLAND_MIN_FRAMES = 25
ISLAND_MIN_BINS = 16
ISLAND_TOP_FACTOR = 1.06
MIN_READABLE_BANDS = 2
MIN_CORR = 0.5
MAX_GAIN_DB = 6.0
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
# The fields an unvalidated anchor nulls (step 6): the margins, their excesses and the audible share.
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


def _moving(side):
    """At least two finite samples that are not all equal."""
    return len(side) > 1 and bool(np.isfinite(side).all()) and float(np.std(side)) > 0.0


def window_usable(src, out):
    """Step 1's alignment half: finite samples, both sides moving, correlated at least 0.5."""
    if not (_moving(src) and _moving(out)):
        return False
    return float(np.corrcoef(src, out)[0, 1]) >= MIN_CORR


def programme_gain(src_power, out_power, freqs):
    """R1's programme gain match in dB (0 with too few programme cells), or None past +-6 dB (a mismatched window)."""
    gain = auditory.programme_gain_db(src_power.T, out_power.T, freqs, balance_metrics.guarded_cells(src_power.T))
    gain = 0.0 if gain is None else float(gain)
    return None if abs(gain) > MAX_GAIN_DB else gain


def anchor_level(context):
    """The anchor's per-band floor, or None without one."""
    anchor = context.anchor
    return None if anchor is None else anchor.level


def capped(floor, anchor):
    """`floor` held no higher than the anchor, band by band (NaN anchor bands leave it as it is)."""
    if anchor is None or len(anchor) != floor.shape[-1]:
        return floor
    return np.fmin(floor, anchor)


def true_pauses(src, rate, power, layout, floor0):
    """Per STFT frame: inside a DSP-VAD non-speech run of 200 ms or more, with kept-band power within 6 dB of the floor."""
    frame20 = max(1, int(pause_metrics.FRAME_S * rate))
    speech = pause_metrics.dsp_vad(src, rate, frame20)
    runs = pause_metrics.long_runs(~speech, max(1, int(round(pause_metrics.MIN_RUN_S * rate / frame20))))
    near = power[:, layout.kindex >= 0].sum(axis=1) <= np.nansum(floor0 * layout.nbins) * 10.0 ** (PAUSE_SNR_DB / 10.0)
    return hiss_floor.frame_map(rate, len(power), runs) & near


def ring_mean(power):
    """Per cell, the mean power of bins f+-2, f+-3, f+-4 in the same frame (edges reflect)."""
    kernel = np.zeros(9)
    for reach in RING:
        kernel[4 - reach] = kernel[4 + reach] = 1.0
    return scipy.ndimage.correlate1d(power, kernel / kernel.sum(), axis=1, mode="reflect")


def noise_led(valley, floor):
    """Per frame: the band's valley within 1.5 dB of the floor in the frames two before and two after (never the frame itself)."""
    ok = valley <= floor * 10.0 ** (NOISE_LED_DB / 10.0)
    pad = np.ones(LED_OFFSET, dtype=bool)
    return np.concatenate([pad, ok[:-LED_OFFSET]]) & np.concatenate([ok[LED_OFFSET:], pad])


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
    return led & (ring <= cell_floor) & (src_power <= PEAK_CAP * ring) & (layout.kindex >= 0)[np.newaxis, :]


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
    """Per frame and band, output over source on the NDC cells within +-0.5 s; the window's ratio with fewer than 40 cells."""
    src_sum, out_sum, count, _floor = sums
    size = 2 * max(1, int(round(LOCAL_HALF_S * fps))) + 1
    src_local, out_local, cells = (
        scipy.ndimage.uniform_filter1d(part, size, axis=0, mode="nearest") * size for part in (src_sum, out_sum, count)
    )
    ratio = np.where(cells >= MIN_CELLS, (out_local + EPS) / (src_local + EPS), ratio_w[np.newaxis, :])
    return np.where(readable[np.newaxis, :], ratio, np.nan)


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


def nmr(band_power, hiss, alpha, layout, spl_db):
    """Per frame and band, the hiss's power over the programme's masked threshold at `spl_db` (module docstring, step 5)."""
    gain = 10.0 ** (spl_db / 10.0)
    masker = np.maximum(band_power - hiss, 0.0) * gain
    offset = alpha * (TONE_MASKING_DB + zbark(layout.centres)[np.newaxis, :]) + (1.0 - alpha) * NOISE_MASKING_DB
    spread = forward_spread(masker @ auditory.spreading_matrix(layout.nb).T, layout.centres, layout.rate)
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


def kept_hf_db(powers, floor, layout, active):
    """The median output over source dB on active cells of 3 kHz and up standing 18 dB over the floor; None under 50 cells."""
    src_power, out_power = powers
    kept = layout.kindex >= 0
    cell_floor = np.full(src_power.shape, np.inf)
    cell_floor[:, kept] = np.nan_to_num(floor[:, layout.kindex[kept]], nan=np.inf)
    cells = active[:, np.newaxis] & layout.bins_of(layout.centres >= HF_LO_HZ)[np.newaxis, :]
    cells &= src_power >= cell_floor * 10.0 ** (PROG_MARGIN_DB / 10.0)
    if cells.sum() < MIN_KEPT_CELLS:
        return None
    return float(np.median(10.0 * np.log10((out_power[cells] + EPS) / (src_power[cells] + EPS))))


def island_top_hz(layout, readable):
    """1.06 x the top readable band of 3 kHz and up, None with none."""
    high = layout.centres[readable & (layout.centres >= HF_LO_HZ)]
    return None if high.size == 0 else float(high.max() * ISLAND_TOP_FACTOR)


def active_20ms(active, count, frame, rate):
    """The STFT frames' activity mapped onto `count` 20 ms frames of `frame` samples (by each one's centre)."""
    centres = (np.arange(count) * frame + frame / 2.0) / rate * rate / hiss_floor.HOP
    return active[np.minimum(centres.astype(int), len(active) - 1)]


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
        self.freqs, self.src_power, self.out_power = freqs, src_power[:count], out_power[:count]
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
        pause = true_pauses(self.pair[0], self.rate, self.src_power, self.layout, floor0)
        active = ~pause
        if active.sum() < max(3.0, hiss_floor.MIN_SPAN_S * self.rate / hiss_floor.HOP):
            return None
        return pause, active, capped(hiss_floor.block_floor(self.valley, active, self.rate, self.layout.calib), anchor)

    def readings(self):
        """The internal readings of the window, or None when it has too little programme or a mismatched gain."""
        found = self.floors()
        gain = programme_gain(self.src_power, self.out_power, self.freqs)
        if found is None or gain is None:
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
            found.update(self._audibility(sums, (readable, ratio_w), (pause, active), floor, gain))
        return found

    def _ratio(self, sums, readable, ratio_w):
        """Per frame and band, the output's hiss over the source's: measured where readable, filled elsewhere."""
        pooled = min(1.0, (sums[1].sum(axis=0)[readable].sum() + EPS) / (sums[0].sum(axis=0)[readable].sum() + EPS))
        return fill_bands(local_ratio(sums, self.rate / hiss_floor.HOP, ratio_w, readable), self.layout.centres, pooled)

    def _hiss(self, sums, readable, ratio_w, floor):
        """`(source hiss, output hiss)` band power per frame (module docstring, step 4)."""
        layout = self.layout
        floor_power = np.nan_to_num(floor, nan=0.0) * layout.nbins[np.newaxis, :]
        out_power = floor_power * self._ratio(sums, readable, ratio_w)
        out_valley = hiss_floor.valley_levels(self.out_power, layout.kindex, layout.nb, layout.shape)
        measured = readable[np.newaxis, :]
        src_hiss = np.where(measured, floor_power, np.fmin(floor_power, valley_bound(self.valley, layout.calib, layout.nbins)))
        out_hiss = np.where(measured, out_power, np.fmin(out_power, valley_bound(out_valley, layout.calib, layout.nbins)))
        return src_hiss, out_hiss

    def _margins(self, hiss, alpha, readable, active):
        """`(source, output)` margins of the two sides' hiss over the shared masked threshold."""
        powers = (self.src_power, self.out_power)
        ratios = [
            nmr(self.layout.band_power(power), side, alpha, self.layout, self.context.src_spl_db) for power, side in zip(powers, hiss)
        ]
        return tuple(side_margins(ratio, self.layout, readable, active) for ratio in ratios)

    def _audibility(self, sums, reading, frames, floor, gain):
        """Both sides' margins and the three guards."""
        readable, ratio_w = reading
        pause, active = frames
        alpha = programme_tonality(self.src_power, self.layout, floor)
        src_side, out_side = self._margins(self._hiss(sums, readable, ratio_w, floor), alpha, readable, active)
        powers = (self.src_power, self.out_power)
        top = island_top_hz(self.layout, readable) if out_side["margin"] >= ISLAND_MIN_MARGIN_DB else None
        return {
            "src": src_side,
            "out": out_side,
            "pump_db": pump_db(powers, pause, ratio_w, self.layout, self.rate),
            "kept_hf_db": kept_hf_db(powers, floor, self.layout, active),
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
