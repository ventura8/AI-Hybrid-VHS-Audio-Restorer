"""Pause readings that hear what the listener heard: dead air, hiss left, pauses collapsing, the residual's shape.

The Tata listening set showed the median level of the source's quiet frames cannot tell a
dead pause from a hissy one: every restoration drives the deepest pauses to -96..-104 dBFS,
so a mean over them is noise. The ear reacts to two other things, both read on 20 ms
frames classed by the SOURCE's level:

- the air left in the inter-word gaps (source frames between p15 and p40): the 3-10 kHz
  power there relative to the loud frames (>= p70). Source -6..-4.5 dB on the three speech
  tapes; APL baseline -27 / -33 / -33 ("silent in pauses"), cathar alpha 2 -18.5 / -17 / -15
  ("has hiss"), the cathar plateau -16.5 / -11.5 / -10, the APL plateau -11 / -13 / -7.
- the pause depth: the loud frames' median level minus the deep frames' (<= p15), output
  minus source. The roformer variant reads +41 / +42 / +54 (the long pauses collapse),
  APL baseline +36 / +39 / +45, cathar baseline +32 / +33 / +40, the cathar plateau
  +23 / +23 / +30, the APL plateau +9 / +4 / +17.

Read relative to the loud frames on each side, both survive the gain match (cathar
attenuates the speech itself by ~4 dB).

The residual in the true pauses (ear v3, reading R4, `gap_residual_readings`). The hiss
flag on `gap_air_db` never moved, and gap level alone cannot be why: `listener.hiss` passes
at an output gap_air of -19.95 dB or less, the round-2 finals the user heard as natural
pauses read -8..-12 (cathar -10.4 / -9.0 / -8.3, APL -10.7 / -12.1 / -9.5), while cathar
alpha 2 ("has hiss", round 1) read -15..-18.5, less air than the accepted outputs. Two
hypotheses are left to test, and these readings make them measurable:

- the gap frames carry word tails, breaths and fricatives, so a "gap" reading partly reads
  speech. Measured on the cached 133 s Tele7abc cut (`experiments/tata_listen/cache`): the
  DSP VAD below calls 89% of the source's p15-p40 frames speech (97% with its hangover);
  the non-speech runs of at least 200 ms hold 338 frames, 302 of them at or under p15. A
  p15-p40 mask would keep 36 frames (0.7 s) of 133 s, so the true-pause mask is every
  source frame at or under p40 inside such a run. The 200 ms floor is longer than a
  syllabic dip (speech modulates at 3-8 Hz, `modules.tonal_persistence.syllabic_modulation`).
- the listener hears the residual's shape and texture, not its level: natural = a roughly
  uniformly scaled copy of the source's own floor (the comfort-noise principle, a
  hypothesis the `pause_residual` degradations test, not a literature fact); hiss = a
  residual tilted to the highs relative to the source's floor, or one broken into
  spectral-subtraction islands.

The DSP VAD (`dsp_vad`, the default): a 20 ms frame is speech when any of four coarse
bands (100-500, 500-2000, 2000-5000, 5000-12000 Hz), averaged over 60 ms, stands more than
6 dB over that band's own 10th percentile; speech is widened by 80 ms either side. Per band,
because a fricative or a breath lifts the top band while hum holds the broadband level
still. The 60 ms average is needed: on 60 s of white noise the 100-500 Hz band (eight 50 Hz
bins) crossed its threshold on 5.4% of single frames, 39% of frames after the hangover; on
the average it crosses on 0.02% (pink noise 0.01%). The plan names a DSP VAD "on
`syllabic_modulation`"; that function returns one 3-8 Hz modulation ratio per clip and the
mask needs a decision per 20 ms frame, so it only justifies the 200 ms floor here. A VAD
callable that returns too few frames is padded as speech (`vad_frames`), the side that
keeps those frames out of the mask. silero-vad is approved but not installed (the critique
asks to confirm it does not pull plain `onnxruntime` first); `resolve_vad` runs it only
when `AI_RESTORE_PAUSE_VAD=silero` asks for it and the package imports, so a reading never
changes because a package appeared. Asked for and missing, it warns and the DSP VAD runs;
`resolved_vad_name()` says which one ran, for the runner to store beside `gap_pause_s`.

Tones leave first (`tonal_bins`, on the source's mask frames): mains hum, which both engines
remove on purpose, is not residual. A bin is a tone when its dB level moves less than 3 dB
(std) across the frames, or its mean stands 6 dB over the median of the 61 bins (3 kHz)
around it; each tone takes the two bins either side with it (the Hann main lobe). With 50 Hz
hum and 7 harmonics (1e-2/k) over the fixture's 3e-3 white hiss and an output keeping only
the scaled hiss, the readings were HF excess +1.19 dB, slope +0.49 dB/oct, spread 1.70,
LSD 2.03 and modulation 3.1 with every bin in; they read under 0.01 now, at 44.1 and 48 kHz,
50 and 60 Hz. Each part is needed: the steadiness test alone misses 60 Hz harmonics (two to
a 50 Hz bin, they beat from frame to frame: slope +3.4, HF excess +9.3); one bin of reach
leaves 50 Hz at +0.2-0.3 (the Hann neighbours of an in-phase 1/k comb cancel a bin's own
harmonic to within 1 dB of the floor). R0's `mains_hz` cannot replace the detection: on
20 ms frames the bins are 50 Hz apart, so one bin either side of every 50 Hz harmonic is
every bin, and how far a hum leaks depends on its level, which only the detection sees; a
median over near neighbours reads a 50 Hz comb (every bin below its last harmonic) as the
floor. On coloured noise the test calls 1.1% of the bins tones at 25 mask frames (the
fewest a window reads), 0.08% at 40, none at 100. On Tele7abc it takes 100-400 or 100-600
Hz on every window, plus 750-950 Hz on one and 5.25-5.45 kHz on two.

Per 1-ERB band (Glasberg-Moore, `auditory.erb_band_edges`) from 100 Hz to the caller's band
cap (the runner passes R0's brickwall, not its programme bandwidth: the pauses hold the
residual heard above the voiced band), 12 kHz or Nyquist, whichever is lowest, the
output over source pause power (time mean over the mask, median over the band's bins left
after the tones; the output held no lower than 60 dB under the source, cell by cell). The
final names (the design's `gap_shape_slope_db_oct` and `gap_shape_spread_db` became
`gap_slope_db_oct` and `gap_spread_db`; its `gap_kurt_pi` is not built, see below):

- `gap_atten_db`: the median over bands, as attenuation (positive = quieter pauses; a
  pause emptied to digital silence reads the 60 dB floor and the source's own shape);
- `gap_slope_db_oct`: the least-squares slope against log2 frequency (positive = the
  residual leans to the highs relative to the source's floor, negative = to the lows);
- `gap_spread_db`: the standard deviation left after removing that slope;
- `gap_hf_excess_db`: the mean of the 4-10 kHz bands minus the 200-1000 Hz bands;
- `gap_lsd_db`: the RMS dB distance of the two band spectra each normalised to unit power;
- `gap_mod_dist_db`: the RMS dB distance between the 2-4, 4-8, 8-16 and 16-32 Hz
  modulation power of the mean-normalised envelopes (20 ms frames every 10 ms, the tones of
  that grid out too) of four ERB-wide band groups, over the whole non-speech runs. It reads
  pumping as well as islands (shared with R6): a residual gated 150 ms on, 150 ms off
  reads 9.8 (and 3 dB more attenuation, half its frames being empty), islands 12.4-12.5.
  A rate band counts a span only when the span holds two periods of the band's lower edge
  (2-4 Hz needs 1 s, 4-8 Hz 0.5 s, 8-16 Hz 0.25 s): a Hann window over 0.2-0.5 s has a
  main lobe of +-4-10 Hz, so a shorter span reads leakage there. Tele7abc's runs are
  0.2-1.0 s, so the reading rests mostly on 8-32 Hz;
- `gap_island_kurt`: ln(output / source) of the median per-frame spectral kurtosis of the
  mask power, each bin whitened by its time mean and each frame divided by its own mean.
  Islands are sparse in frequency within a frame; a gated or pumped floor is not, so this
  reads islands and not pumping: islands 0.99-1.00, the gated residual under 0.03 (the
  per-bin kurtosis across frames it replaces read 0.43 there, against 1.15 for islands).
  It stands in for R10's delta-kurtPI (Torcoli 2019: A-weighted dB power, a P-20 floor,
  three sub-bands), which is deferred to P1, and is not the same measure;
- `gap_pause_s`: the seconds of mask the readings stand on (coverage before content).

The source side of each shape reading is the identity (0); a window with a non-finite
sample, or a reading that comes out non-finite, reads None. On the synthetic fixtures
(`tests/unit/test_restoration_quality_pause_residual.py` and `..._pause_texture.py`, bursts
over white hiss): the source noise scaled by -12 dB reads 12.0 dB attenuation with every
shape and texture reading 0; a fresh realisation of the same hiss reads slope 0.15, LSD
0.44, HF excess 0.67, modulation 1.2, island kurtosis -0.03; a +6 dB/oct tilt above 2 kHz
reads slope +2.4 and HF excess +9.8, a 1 kHz low pass -11.3 and -47.7; a 1.8-2.2 kHz band
stop spread 7.4 and LSD 7.5 with slope -0.1; islands (5% of 256-point STFT cells kept) read
modulation 12.5 and island kurtosis 1.00; bursts decaying into the pauses (80 ms) read
11.7 dB attenuation with the VAD and 8.1 without it. On the Tele7abc cut (13 of 17 windows
read, 0.8 s of mask each, medians): cathar alpha 2 on 0.7.6 (heard: hiss) HF excess
+3.1 dB, slope +1.9 dB/oct, island kurtosis 1.75; the same setting on 0.7.5 (clean, only
the binary differs) +0.5, +1.1, 1.73; cathar baseline (clean) 0.0, +0.2, 0.96; APL
baseline (dead air) -13.7, -4.4, 0.18; the roformer variant (pauses gated, dead) empties
them to digital silence (the 60 dB floor). The attenuation does not order hiss and clean
(42.3 against 42.8 dB), nor do the modulation (9.5 against 9.9) and the island kurtosis:
both alpha 2 renders carry islands, and what the 0.7.6 one adds is highs. Before the tones
left, APL baseline read -8.5 and -1.2 (hum in the 200-1000 Hz reference). No threshold is
set: Round 0 decides which reading separates the verdicts and tests it on SOTI and Vaccin.
"""

import importlib
import importlib.util
import math
import os
import warnings
from functools import lru_cache

import numpy as np
import scipy.ndimage
import scipy.signal
from numpy.lib.stride_tricks import sliding_window_view

from scripts.measure_tradeoff import LOUD_PERCENTILE
from scripts.restoration_quality import auditory
from scripts.restoration_quality.dsp_metrics import framed_psd

FRAME_S = 0.02
DEEP_PERCENTILE = 15.0
GAP_PERCENTILES = (15.0, 40.0)
AIR_BAND_HZ = (3000.0, 10000.0)
MIN_CLASS_FRAMES = 20
NAMES = ("gap_air_db", "pause_depth_db")

# The residual in the true pauses (R4).
RESIDUAL_LO_HZ = 100.0
RESIDUAL_TOP_HZ = 12000.0
VAD_BANDS_HZ = ((100.0, 500.0), (500.0, 2000.0), (2000.0, 5000.0), (5000.0, 12000.0))
VAD_FLOOR_PERCENTILE = 10.0
VAD_MARGIN_DB = 6.0
VAD_SMOOTH_S = 0.06
VAD_HANGOVER_S = 0.08
VAD_ENV = "AI_RESTORE_PAUSE_VAD"
SILERO_MODULE = "silero_vad"
SILERO_RATE = 16000
MIN_RUN_S = 0.2
MIN_CONTRAST_DB = 6.0
MIN_PAUSE_FRAMES = 25
MIN_BANDS = 4
RESIDUAL_FLOOR_DB = -60.0
HF_EXCESS_HZ = (4000.0, 10000.0)
HF_EXCESS_REFERENCE_HZ = (200.0, 1000.0)
TONAL_STD_DB = 3.0
TONAL_PEAK_DB = 6.0
TONAL_WINDOW_BINS = 61
TONAL_REACH_BINS = 2
MOD_HOP_S = 0.01
MOD_BAND_COUNT = 4
MOD_RATES_HZ = ((2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0))
MOD_MIN_PERIODS = 2.0
MOD_FFT = 128
MOD_FLOOR = 1e-6
KURT_EPS = 1e-12
EPS = 1e-20
RESIDUAL_NAMES = (
    "gap_atten_db",
    "gap_slope_db_oct",
    "gap_spread_db",
    "gap_hf_excess_db",
    "gap_lsd_db",
    "gap_mod_dist_db",
    "gap_island_kurt",
    "gap_pause_s",
)


def level_classes(level_db):
    """`(deep, gap, loud)` masks over the source's frame levels in dB."""
    deep_cut = np.percentile(level_db, DEEP_PERCENTILE)
    gap_cut = np.percentile(level_db, GAP_PERCENTILES[1])
    loud_cut = np.percentile(level_db, LOUD_PERCENTILE)
    return level_db <= deep_cut, (level_db > deep_cut) & (level_db <= gap_cut), level_db >= loud_cut


def gap_air_db(power, freqs, gap, loud):
    """Median 3-10 kHz power on the gap frames over the loud frames, in dB (one side)."""
    band = power[:, (freqs >= AIR_BAND_HZ[0]) & (freqs < AIR_BAND_HZ[1])].sum(axis=1)
    return float(10.0 * np.log10((np.median(band[gap]) + 1e-20) / (np.median(band[loud]) + 1e-20)))


def pause_depth_db(level_db, deep, loud):
    """How far under the loud frames the deep pauses sit, in dB (one side)."""
    return float(np.median(level_db[loud]) - np.median(level_db[deep]))


def pause_readings(source, output, rate):
    """`{"gap_air_db": (source, output), "pause_depth_db": (source, output)}`, or Nones when a class is too thin."""
    frame = max(1, int(FRAME_S * rate))
    count = min(len(source), len(output)) // frame
    if count < 3 * MIN_CLASS_FRAMES:
        return {name: (None, None) for name in NAMES}
    freqs, src_power, src_level = framed_psd(source[: count * frame], rate, frame)
    _freqs, out_power, out_level = framed_psd(output[: count * frame], rate, frame)
    src_db, out_db = _db(src_level), _db(out_level)
    deep, gap, loud = level_classes(src_db)
    if min(deep.sum(), gap.sum(), loud.sum()) < MIN_CLASS_FRAMES:
        return {name: (None, None) for name in NAMES}
    return {
        "gap_air_db": (gap_air_db(src_power, freqs, gap, loud), gap_air_db(out_power, freqs, gap, loud)),
        "pause_depth_db": (pause_depth_db(src_db, deep, loud), pause_depth_db(out_db, deep, loud)),
    }


def _db(level):
    return 20.0 * np.log10(np.asarray(level, dtype=np.float64) + 1e-9)


# --- R4: the residual left in the true pauses ------------------------------------------------


def gap_residual_readings(source, output, rate, bandwidth_hz=None, vad=None):
    """`{name: (source, output)}` for `RESIDUAL_NAMES` on the true pauses, or Nones when the source has too few.

    The source side of each shape reading is the identity (0.0); `gap_pause_s` reads the
    mask's length on both sides. `bandwidth_hz` is the band cap when known (the runner
    passes R0's brickwall). `vad` is a callable `(mono, rate, frame) -> speech per frame`; None runs
    `resolve_vad()` (`resolved_vad_name()` says which one, for the ledger). A window with a
    non-finite sample reads None.
    """
    frame = max(1, int(FRAME_S * rate))
    count = min(len(source), len(output)) // frame
    pair = np.stack([np.asarray(side[: count * frame], dtype=np.float64) for side in (source, output)])
    if count < 3 * MIN_CLASS_FRAMES or not np.isfinite(pair).all():
        return _residual_none()
    mask, runs = true_pause_mask(pair[0], rate, vad)
    if mask.sum() < MIN_PAUSE_FRAMES:
        return _residual_none()
    return _paired_readings(_residual(pair[0], pair[1], rate, (mask, runs), top_hz(rate, bandwidth_hz)), float(mask.sum() * frame / rate))


def true_pause_mask(source, rate, vad=None):
    """`(mask, runs)` on 20 ms frames: the source's quiet frames (<= p40) inside non-speech runs of at least 200 ms.

    `runs` is every frame of those non-speech runs (the texture readings need them
    contiguous). A source whose loud frames stand less than 6 dB over its quiet frames (a
    held note, a stretch of nothing but hiss) has no pauses to read and returns empty masks.
    """
    frame = max(1, int(FRAME_S * rate))
    _freqs, _power, level = framed_psd(source, rate, frame)
    level_db = _db(level)
    deep, gap, loud = level_classes(level_db)
    quiet = deep | gap
    if np.median(level_db[loud]) - np.median(level_db[quiet]) < MIN_CONTRAST_DB:
        return np.zeros_like(quiet), np.zeros_like(quiet)
    speech = vad_frames(vad or resolve_vad(), (source, rate, frame), len(quiet))
    runs = long_runs(~speech, max(1, int(round(MIN_RUN_S * rate / frame))))
    return quiet & runs, runs


def vad_frames(vad, call, count):
    """The VAD's speech per frame for `call = (mono, rate, frame)`, cut to `count` frames or padded as speech.

    A callable that comes back a frame or two short (a timestamp rounded down) must not crash
    the reading; the missing frames count as speech, the side that keeps them out of the mask.
    """
    speech = np.asarray(vad(*call), dtype=bool)[:count]
    return np.concatenate([speech, np.ones(count - len(speech), dtype=bool)])


def top_hz(rate, bandwidth_hz=None):
    """Where the residual bands stop: the band cap when known (R0's brickwall), 12 kHz or Nyquist, whichever is lowest."""
    return float(min(RESIDUAL_TOP_HZ, 0.5 * rate, bandwidth_hz or np.inf))


def _residual(src, out, rate, pauses, top):
    """The shape and texture readings on the mask, or None when the range holds too few ERB bands."""
    mask, runs = pauses
    frame = max(1, int(FRAME_S * rate))
    freqs, src_power = _masked_power(src, rate, frame, mask)
    _freqs, out_power = _masked_power(out, rate, frame, mask)
    out_power = floored(out_power, src_power)
    keep = residual_bins(freqs, src_power, top)
    readings = residual_shape(freqs[keep], src_power[:, keep], out_power[:, keep], top)
    if readings is None:
        return None
    spans = [(start * frame, stop * frame) for start, stop in run_spans(runs)]
    readings["gap_mod_dist_db"] = modulation_distance_db((src, out), rate, spans, (freqs, keep, top))
    readings["gap_island_kurt"] = island_kurtosis(src_power[:, keep], out_power[:, keep])
    return readings


def floored(out_power, src_power):
    """`out_power` held no lower than 60 dB under `src_power`, cell by cell.

    Below that the residual sits far under the source's own floor and only the attenuation
    should say so: a pause emptied to digital silence reads 60 dB of attenuation and the
    source's own shape and texture, not a distance blown up by a division by zero.
    """
    return np.maximum(out_power, src_power * 10.0 ** (RESIDUAL_FLOOR_DB / 10.0))


def _masked_power(mono, rate, frame, mask):
    """`(freqs, power (mask frames, bins))` of `mono` on `frame`-sample frames."""
    freqs, power, _level = framed_psd(mono, rate, frame)
    return freqs, power[mask]


def residual_bins(freqs, src_power, top):
    """The bins every residual reading uses: 100 Hz to `top`, less the bins a tone holds in the source's pauses."""
    return (freqs >= RESIDUAL_LO_HZ) & (freqs < top) & ~tonal_bins(src_power)


def tonal_bins(power):
    """Bins a tone holds in the source's `power` (`(frames, bins)` on 20 ms frames), widened one bin either side.

    Two tests, either one enough. Steady: noise in one bin is exponential from frame to frame
    (its dB level moves by 5.6 dB, std), a tone 10 dB over it by about 2 dB; under 3 dB is a
    tone. Peaked: the bin's mean stands 6 dB over the median of the 61 bins (3 kHz) around
    it. The second catches what the first misses: 60 Hz harmonics fall two to a 50 Hz bin
    and beat from frame to frame. The window is wide because a 50 Hz comb fills every bin
    below its last harmonic, and a median over fewer bins reads the comb itself as the floor;
    on a monotonic floor (a roll-off) the median is the centre bin, so a slope never reads as
    a peak. A bin the source holds at digital silence reads as steady and leaves too (there is
    no floor there to compare against).
    """
    level = 10.0 * np.log10(power + EPS)
    mean = 10.0 * np.log10(power.mean(axis=0) + EPS)
    steady = np.std(level, axis=0) < TONAL_STD_DB
    peaked = mean > scipy.ndimage.median_filter(mean, size=TONAL_WINDOW_BINS, mode="reflect") + TONAL_PEAK_DB
    return scipy.ndimage.binary_dilation(steady | peaked, iterations=TONAL_REACH_BINS)


def residual_shape(freqs, src_power, out_power, top):
    """The five shape readings from per-ERB-band output/source pause power, or None with fewer than four bands.

    `src_power` and `out_power` are `(mask frames, bins)` over the bins at `freqs`.
    """
    centres, src_band, out_band = erb_band_powers(freqs, src_power.mean(axis=0), out_power.mean(axis=0), top)
    if len(centres) < MIN_BANDS:
        return None
    ratio = 10.0 * np.log10((out_band + EPS) / (src_band + EPS))
    octaves = np.log2(centres / 1000.0)
    slope, intercept = np.polyfit(octaves, ratio, 1)
    return {
        "gap_atten_db": float(-np.median(ratio)),
        "gap_slope_db_oct": float(slope),
        "gap_spread_db": float(np.std(ratio - (slope * octaves + intercept))),
        "gap_hf_excess_db": hf_excess_db(centres, ratio),
        "gap_lsd_db": normalised_lsd_db(src_band, out_band),
    }


def erb_band_powers(freqs, src_psd, out_psd, top):
    """`(centres_hz, source, output)` per 1-ERB band from 100 Hz to `top` that holds a bin, each the median over its bins."""
    edges = auditory.erb_band_edges(RESIDUAL_LO_HZ, top)
    index = auditory.band_index(freqs, edges)
    bands = np.unique(index[index >= 0])
    centres = np.sqrt(edges[bands] * edges[bands + 1])
    return centres, _band_medians(src_psd, index, bands), _band_medians(out_psd, index, bands)


def _band_medians(psd, index, bands):
    return np.array([np.median(psd[index == band]) for band in bands])


def hf_excess_db(centres, ratio):
    """Mean output/source dB over the 4-10 kHz bands minus the 200-1000 Hz bands; None when either range is empty."""
    high = ratio[(centres >= HF_EXCESS_HZ[0]) & (centres < HF_EXCESS_HZ[1])]
    reference = ratio[(centres >= HF_EXCESS_REFERENCE_HZ[0]) & (centres < HF_EXCESS_REFERENCE_HZ[1])]
    if len(high) == 0 or len(reference) == 0:
        return None
    return float(high.mean() - reference.mean())


def normalised_lsd_db(src_band, out_band):
    """RMS dB distance between the two band spectra, each normalised to unit power (blind to the overall level)."""
    src_shape = src_band / (src_band.sum() + EPS)
    out_shape = out_band / (out_band.sum() + EPS)
    return float(np.sqrt(np.mean((10.0 * np.log10((out_shape + EPS) / (src_shape + EPS))) ** 2)))


def modulation_distance_db(pair, rate, spans, bins):
    """RMS dB distance between the modulation spectra of four band-group envelopes of `pair = (source, output)`.

    The envelopes run over the sample `spans` (the whole non-speech runs) on 20 ms frames
    every 10 ms. `bins = (freqs, keep, top)`: the 20 ms frame's bins and the ones the shape
    readings kept; the envelopes also leave out the bins a tone holds on their own grid, where
    a hum harmonic's leakage moves with the half-frame hop. A rate band no span is long enough
    for is left out of the distance; with no group left the reading is None.
    """
    timing = (max(1, int(FRAME_S * rate)), max(1, int(MOD_HOP_S * rate)))
    src_powers, out_powers = (_hop_powers(side, spans, timing) for side in pair)
    groups = mod_groups(bins, np.concatenate(src_powers))
    if groups.max() < 0:
        return None
    src_env, out_env = _group_envelopes(src_powers, out_powers, groups)
    envelope_rate = rate / timing[1]
    return float(
        np.sqrt(np.nanmean((modulation_spectrum_db(out_env, envelope_rate) - modulation_spectrum_db(src_env, envelope_rate)) ** 2))
    )


def mod_groups(bins, src_power):
    """Each bin's group among four ERB-wide groups from 100 Hz to `top`; -1 for a bin outside them, not kept, or tonal here."""
    freqs, keep, top = bins
    edges = auditory.erb_to_hz(np.linspace(auditory.erb_number(RESIDUAL_LO_HZ), auditory.erb_number(top), MOD_BAND_COUNT + 1))
    return np.where(keep & ~tonal_bins(src_power), auditory.band_index(freqs, edges), -1)


def _hop_powers(mono, spans, timing):
    """`hop_power` of each sample span of `mono`."""
    return [hop_power(mono[slice(*span)], timing) for span in spans]


def hop_power(segment, timing):
    """`(frames, bins)` Hann-windowed rfft power of `timing = (frame, hop)` frames: 20 ms long every 10 ms.

    As long as the mask's frames, so the bins are the same 50 Hz grid; the 10 ms hop keeps the
    16-32 Hz modulation band under the envelope's Nyquist.
    """
    frame, hop = timing
    frames = sliding_window_view(np.asarray(segment, dtype=np.float64), frame)[::hop]
    return np.abs(np.fft.rfft(frames * np.hanning(frame), axis=1)) ** 2


def _group_envelopes(src_powers, out_powers, groups):
    """`(source, output)` lists of `(frames, groups)` envelopes; each output held no lower than 60 dB under the source."""
    src_env = [group_envelopes(power, groups) for power in src_powers]
    return src_env, [floored(group_envelopes(power, groups), env) for power, env in zip(out_powers, src_env)]


def group_envelopes(power, groups):
    """`(frames, groups)`: the power in each group of `groups` that holds a bin."""
    return np.stack([power[:, groups == group].sum(axis=1) for group in np.unique(groups[groups >= 0])], axis=1)


def modulation_spectrum_db(envelopes, envelope_rate):
    """`(groups, rates)` modulation power in dB of a list of `(frames, groups)` envelopes, weighted by their length.

    Normalising each envelope by its mean makes the reading blind to level: a scaled copy of
    the source's floor reads the source's modulation exactly, islands read more. A group that
    never carries power reads the -60 dB floor; a rate band that no envelope holds two periods
    of reads NaN.
    """
    power = np.stack([modulation_power(envelope, envelope_rate) for envelope in envelopes])
    weights = rate_weights([len(envelope) for envelope in envelopes], envelope_rate)
    total = weights.sum(axis=0)
    mean = (power * weights[:, np.newaxis, :]).sum(axis=0) / np.where(total > 0, total, np.nan)
    return 10.0 * np.log10(np.maximum(mean, MOD_FLOOR))


def rate_weights(lengths, envelope_rate):
    """`(envelopes, rates)`: each envelope's length in frames where it spans two periods of the band's lower edge, else 0.

    A Hann window over 0.2-0.5 s cannot resolve 2-4 or 4-8 Hz (its main lobe is +-4-10 Hz
    wide); a band read on a shorter span would read window leakage from the bands above.
    """
    frames = np.asarray(lengths, dtype=np.float64)[:, np.newaxis]
    lowest = np.array([lo for lo, _hi in MOD_RATES_HZ])[np.newaxis, :]
    return frames * (frames >= MOD_MIN_PERIODS * envelope_rate / lowest)


def modulation_power(envelopes, envelope_rate):
    """`(groups, rates)`: the power of each mean-normalised envelope in the 2-4, 4-8, 8-16 and 16-32 Hz bands.

    One-sided, Hann-windowed and normalised so the bands sum to the envelope's variance
    whatever the span's length (zero-padded to at least 128 frames).
    """
    mean = envelopes.mean(axis=0)
    depth = (envelopes - mean) / (mean + EPS)
    window = np.hanning(len(depth))[:, np.newaxis]
    size = max(MOD_FFT, len(depth))
    spectrum = 2.0 * np.abs(np.fft.rfft(depth * window, n=size, axis=0)) ** 2 / (size * float(np.sum(window**2)) + EPS)
    rates = np.fft.rfftfreq(size, 1.0 / envelope_rate)
    return np.stack([spectrum[(rates >= lo) & (rates < hi)].sum(axis=0) for lo, hi in MOD_RATES_HZ], axis=1)


def island_kurtosis(src_power, out_power):
    """ln(output / source) of the median per-frame spectral kurtosis on the whitened mask power (`(frames, bins)`).

    Each bin is first divided by its own time mean (whitening, so the floor's colour and any
    tilt drop out); each frame's kurtosis across bins then reads how sparse the frame is.
    Islands leave a few bins lit and the rest empty in every frame; a residual gated or pumped
    as a whole keeps every frame as broadband as the source's. Each frame is divided by its
    own mean before the kurtosis, so a frame gated 60 dB down reads its shape, not the
    epsilon. So this reads islands and not pumping, where `gap_mod_dist_db` reads both.
    """
    return float(np.log(frame_kurtosis(out_power)) - np.log(frame_kurtosis(src_power)))


def frame_kurtosis(power):
    """Median over frames of the Pearson kurtosis across bins of each frame, each bin whitened by its time mean."""
    white = power / (power.mean(axis=0) + EPS)
    white = white / (white.mean(axis=1, keepdims=True) + EPS)
    centred = white - white.mean(axis=1, keepdims=True)
    kurtosis = (np.mean(centred**4, axis=1) + KURT_EPS) / (np.mean(centred**2, axis=1) ** 2 + KURT_EPS)
    return float(np.median(kurtosis))


def _paired_readings(readings, pause_s):
    """Shape readings as `(identity, output)`, a None or non-finite one as `(None, None)`, the coverage on both sides.

    No readings at all (too few bands) is every name None.
    """
    if readings is None:
        return _residual_none()
    paired = {name: _identity_pair(readings[name]) for name in RESIDUAL_NAMES[:-1]}
    paired["gap_pause_s"] = (pause_s, pause_s)
    return paired


def _identity_pair(value):
    return (None, None) if value is None or not np.isfinite(value) else (0.0, value)


def _residual_none():
    return {name: (None, None) for name in RESIDUAL_NAMES}


# --- the true-pause VAD -----------------------------------------------------------------------


def resolve_vad(name=None):
    """The VAD the true-pause mask runs: `dsp_vad`, or `silero_vad` when asked for (`AI_RESTORE_PAUSE_VAD`) and installed."""
    return silero_vad if resolved_vad_name(name) == "silero" else dsp_vad


def resolved_vad_name(name=None):
    """`"silero"` or `"dsp"`: the VAD the mask actually runs, for the runner to store beside `gap_pause_s`.

    Asking for silero-vad without the package installed warns (`RuntimeWarning`, once per
    process under Python's default filter) and runs the DSP VAD, so a run believed to use
    silero cannot carry DSP readings without a trace.
    """
    if _vad_name(name) != "silero":
        return "dsp"
    if importlib.util.find_spec(SILERO_MODULE) is None:
        warnings.warn("the pause VAD asks for silero-vad, which is not installed: the DSP VAD runs", RuntimeWarning, stacklevel=2)
        return "dsp"
    return "silero"


def _vad_name(name):
    return (name or os.environ.get(VAD_ENV) or "dsp").strip().lower()


def dsp_vad(mono, rate, frame):
    """Speech per `frame`-sample frame: a coarse band 6 dB over its own p10 on a 60 ms average, widened 80 ms either side."""
    freqs, power, _level = framed_psd(np.asarray(mono, dtype=np.float64), rate, frame)
    smooth = max(1, int(round(VAD_SMOOTH_S * rate / frame)))
    active = np.zeros(power.shape[0], dtype=bool)
    for band in VAD_BANDS_HZ:
        active |= _band_active(power, freqs, band, smooth)
    reach = max(0, int(round(VAD_HANGOVER_S * rate / frame)))
    return scipy.ndimage.maximum_filter1d(active.astype(np.uint8), size=2 * reach + 1) > 0


def _band_active(power, freqs, band, smooth):
    """Frames whose `band` power, averaged over `smooth` frames, stands more than 6 dB over its p10 (none past Nyquist)."""
    bins = (freqs >= band[0]) & (freqs < band[1])
    if not bins.any():
        return np.zeros(power.shape[0], dtype=bool)
    averaged = scipy.ndimage.uniform_filter1d(power[:, bins].sum(axis=1), size=smooth, mode="nearest")
    level = 10.0 * np.log10(averaged + EPS)
    return level > np.percentile(level, VAD_FLOOR_PERCENTILE) + VAD_MARGIN_DB


def silero_vad(mono, rate, frame):
    """Speech per `frame`-sample frame from silero-vad at 16 kHz (`resolve_vad` checks the package is installed)."""
    module = importlib.import_module(SILERO_MODULE)
    audio = _resample(np.asarray(mono, dtype=np.float32), rate, SILERO_RATE)
    stamps = module.get_speech_timestamps(audio, silero_model(), sampling_rate=SILERO_RATE)
    return stamps_to_frames(stamps, SILERO_RATE, len(mono) // frame, frame / rate)


@lru_cache(maxsize=1)
def silero_model():
    """The silero-vad model, loaded once per process."""
    return importlib.import_module(SILERO_MODULE).load_silero_vad()


def stamps_to_frames(stamps, stamp_rate, count, frame_s):
    """Speech per frame from silero's `[{"start": sample, "end": sample}, ...]` at `stamp_rate`."""
    speech = np.zeros(count, dtype=bool)
    for stamp in stamps:
        first, last = int(stamp["start"] / stamp_rate / frame_s), int(math.ceil(stamp["end"] / stamp_rate / frame_s))
        speech[first:last] = True
    return speech


def run_spans(mask):
    """`(start, stop)` index pairs of each run of True in `mask`."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def long_runs(mask, min_length):
    """The frames of `mask` that belong to a run of at least `min_length` True frames."""
    keep = np.zeros(len(mask), dtype=bool)
    for start, stop in run_spans(mask):
        keep[start:stop] = stop - start >= min_length
    return keep


def _resample(mono, rate, target):
    factor = math.gcd(int(rate), int(target))
    return scipy.signal.resample_poly(mono, int(target) // factor, int(rate) // factor).astype(np.float32)
