"""R0 `src.capture_profile`: what the capture itself carries, read once per source.

Ear v3 clips every reading to what the source can hold. Above a PAL SP linear track's
programme band (about 80 Hz-8 kHz) an "air" reading measures hiss or synthesis, not
programme; an Internet Archive AAC clip has nothing above its ~16 kHz cut at all; and hum and
whistle readings assumed 50 Hz and 15625 Hz on every tape. `capture_profile(audio, rate)` is
a pure function of the source; the caller caches it per source. Its fields:

- `prog_bandwidth_hz`: the centre of the highest 1/6-octave band (250 Hz up, CRT line +-250 Hz
  left out through `dsp_metrics.band_bins`) where the loud frames (>= p70) stand >= 3 dB over
  the inter-word gap frames (p15-p40, `pause_metrics.level_classes`) on 40 ms frames
  (`dsp_metrics.framed_psd`) over the whole file. "Sustained" twice over: each class is read
  as its median frame (a few loud transients cannot lift a band) and the band below must
  qualify too (one stray band cannot). Hiss sits in both classes and cancels; programme sits
  only in the loud one. A band counts only below the brickwall and within 80 dB of the
  loudest band's density: above a cut both classes sit at the digital floor, where a
  relative excess means nothing (the Tata sources read 12-19 dB of "excess" at 18-20 kHz,
  110-123 dB under the loudest band). A band-limited programme reads within 1/6 octave of
  its cut.
- `brickwall_hz`: the top of the highest 100 Hz band (2 kHz up to 0.9 x Nyquist, CRT line out)
  that stands > 30 dB over every band from 500 Hz higher to the top, on the mean spectrum of
  the gap frames: a codec low-pass takes the hiss with it (AAC ~16 kHz, DV ~15 kHz), while a
  programme band edge leaves the hiss above it, and so does a tape roll-off, which is gentle
  besides. Both read None. Hann leakage puts the reading up to ~200 Hz above the true cut.
- `mains_hz` / `mains_evidence_db` / `mains_excess_db`: the mains whose harmonic series
  stands out on the 1 s frames. Harmonic k (1..8) of either series is searched within
  +-max(2 bins, 1 % of k x 60 Hz) of its bin, a width set by k alone: both series search the
  same number of bins and noise reads the same on each. (A search of 1 % of each harmonic's
  own frequency gave the 60 Hz series the wider windows; on 300 noise-only 15 s clips, white
  and pink at 48 kHz, 86 of the 129 with any evidence named 60 Hz. With equal windows it is
  120 against 114.) 1 % of k x 60 Hz still covers a 1.2 % speed error on the 50 Hz series.
  300 Hz is the 6th harmonic of 50 Hz and the 5th of 60 Hz, so each series reads the other's
  hum there. Per harmonic the median prominence over a +-20 Hz neighbourhood, summed above
  6 dB, is `mains_evidence_db` (a search with no harmonic in it reads 0.8-5.7 dB on the Tata
  tapes). The frames name the mains only with >= 3 dB of evidence on >= 10 frames: noise
  alone reads at most 2.7 dB on 10 s, 2.2 dB on 15 s and 0.6 dB on 60 s (the same 300 clips
  per length), but up to 7.1 dB on 5 s. Otherwise `measure_hum._mains_by_evidence` names it
  (`summed_rule_mains`). That rule sums peaks and floors over the series on one long-term
  spectrum, so the loudest neighbourhoods decide, and those are programme: on SOTI the
  150-250 Hz speech peaks sit 21-29 dB over the 50 Hz fundamental, both series read
  1.2-1.7 dB of excess and 60 Hz wins by 0.5 dB, while on the frames the fundamental's
  median prominence is 14.3 dB and no 60 Hz-series harmonic reaches 5 dB. A hum holds its
  bins on every frame and so holds the frames' medians; speech moves its pitch from frame
  to frame and sits on one harmonic's bins on few of them. `mains_excess_db` is
  `measure_hum.hum_excess_db` at the chosen mains (>= 6 dB "carries hum"), read on up to
  eight evenly spaced 15 s excerpts.
  `measure_hum` is already imported by `dsp_metrics`, so this adds no import side effect.
- `line_hz` / `line_ppm` / `line_sd_ppm` / `line_hum_corr` / `line_origin`: the CRT line
  tracked on 1 s frames: the strongest bin in 15375-15984 Hz standing >= 12 dB over the
  band's median and above -125 dBFS as a sine (the Tata lines peak at -94..-100 dBFS, a band
  with no line at the 16-bit floor, -136..-149), its frequency refined by Grandke's two-bin
  interpolation for the Hann window (exact for one tone). A frame counts when it sits
  within 10 Hz of the running median of five tracked frames: on SOTI 7 of 434 frames jump
  16-100 Hz to a sideband and alone lift the spread from 13 to 430 ppm, while a band with no
  line (Vaccin) hops ~100 Hz from frame to frame and only 24 % of its frames count. A line
  needs >= 10 such frames (the corpus clips are 15 s) covering half the file, its median
  within 1 % of the nearer standard rate (15625 Hz PAL, 15734 Hz NTSC), else every line
  field is None. `line_ppm` is the mean offset from that rate, `line_sd_ppm` the spread,
  `line_hum_corr` the Pearson r of the line's ppm track with the hum's on >= 20 shared
  frames (the chosen mains' harmonic with the strongest median prominence, frames where
  it stands >= 10 dB out). Tape speed moves
  everything recorded on the tape by the same ppm, so a line that wanders with the hum
  (r >= 0.6) is `speed_locked`: it follows the transport. That does not say the line was
  recorded: a deck without a time-base corrector takes its video timing off the tape, so a
  whistle from video crosstalk in its own playback chain follows the transport just as
  recorded hum does. A steady line (spread <= 200 ppm) that does not wander with the hum is
  `playback_chain`, the class the Tata measurements back (on a linear track the line
  cannot be on the tape: at PAL SP its wavelength is ~1.5 um, far beyond the ~8 kHz
  response); anything else is `unresolved`. The 200 ppm and r 0.6 cuts are declared: no
  speed-locked line has been measured yet.
- `channel_state`: `channels`, the SIGNED Pearson r of left and right on 100-1000 Hz
  (`correlation`; the app's `filters._channel_correlation` returns abs(r), so its azimuth gate
  passes an inverted pair), the left-over-right energy ratio (`energy_ratio_db`, positive =
  left louder), `dual_mono` (the channels agree sample for sample within 1e-6), the
  `dead_channel` ("left" / "right" beyond `filters.DEAD_CHANNEL_DB`, 25 dB: a real 21-tape
  library clusters at 0 dB or 30-53 dB) and `inverted` (r <= -0.5 on a pair with no dead
  side: real pairs that share content read |r| 0.98-1.00, an L/-R pair -1.0). The other
  fields are read on the live channel of a dead pair and on (L - R) / 2 of an inverted one, so a
  plain downmix that cancels the programme does not empty the profile.

Measured on the three cached Tata sources (`experiments/tata_listen/cache`, 44.1 kHz,
134-460 s, 0.6-1.5 s each):

| source   | band    | brickwall | mains, evidence, excess | line (playback_chain) | channels          |
|----------|---------|-----------|-------------------------|-----------------------|-------------------|
| Tele7abc | 4490 Hz | 18.9 kHz  | 50 Hz, 7.3, 2.0 dB      | 15624.97 Hz, 7 ppm    | right dead, 43 dB |
| SOTI     | 5040 Hz | 15.4 kHz  | 50 Hz, 8.3, 1.2 dB      | 15624.99 Hz, 13 ppm   | right dead, 51 dB |
| Vaccin   | 7127 Hz | None      | 50 Hz, 21.5, 4.3 dB     | None                  | r 1.00, 0 dB      |

The two interviews' 4-5 kHz bands look like PAL LP linear tracks and Vaccin's 7 kHz like SP
(an inference, not checked against the cassettes). On 24 Internet Archive corpus clips (12
evenly spaced per region folder of `experiments/ia_corpus_1000`, 15 s, decoded at 48 kHz):
a brickwall on 20 (12.0-21.0 kHz, 13 of them 13.6-16.2 kHz), a line on 6: four NTSC at
15731.9-15735.4 Hz (all "america"), two PAL at 15624.0-15624.2 Hz ("europe"), five steady
(0.9-43.5 ppm, playback_chain), one at 223 ppm (unresolved); 4 dual mono, none inverted.
The frames named the mains on 17 (3.3-35.5 dB of evidence); the other 7 read 0-1.4 dB
and went to the summed rule.

Every pass is blocked (256 profile frames or 16 one-second frames at a time), so a whole tape
never holds more than one block of spectra beside its samples.
"""

import numpy as np
import scipy.ndimage

from modules.filters import DEAD_CHANNEL_DB
from scripts.measure_hum import HARMONICS, MAINS_CANDIDATES_HZ, _mains_by_evidence, hum_excess_db
from scripts.restoration_quality.dsp_metrics import CRT_LINE_HALF_WIDTH_HZ, CRT_LINE_HZ, band_bins, frame_levels, framed_psd
from scripts.restoration_quality.pause_metrics import MIN_CLASS_FRAMES, level_classes

EPS = 1e-20
PPM = 1e6
# 40 ms profile frames (2048 samples at 44.1/48 kHz, 23 Hz bins): short enough that the
# inter-word gaps form their own class, fine enough for 1/6-octave bands from 250 Hz.
PROFILE_FRAME_S = 0.04
BLOCK_FRAMES = 256
BAND_REFERENCE_HZ = 1000.0
BANDS_PER_OCTAVE = 6
BANDWIDTH_MIN_HZ = 250.0
EXCESS_MIN_DB = 3.0
PROGRAMME_RANGE_DB = 80.0
BRICKWALL_MIN_HZ = 2000.0
BRICKWALL_BAND_HZ = 100.0
BRICKWALL_SPAN_HZ = 500.0
BRICKWALL_FALL_DB = 30.0
# The converter's own anti-alias filter falls just as steeply near Nyquist; it is not a cut.
BRICKWALL_TOP_SHARE = 0.9
MAINS_BLOCK_S = 15.0
MAINS_BLOCKS = 8
TRACK_FRAME_S = 1.0
TRACK_BLOCK_FRAMES = 16
# The Internet Archive corpus clips are 15 s long; a line on ten 1 s frames is a line. A
# correlation needs twenty: two unrelated noise tracks reach r >= 0.6 with p ~0.003 there.
MIN_TRACK_FRAMES = 10
MIN_CORR_FRAMES = 20
LINE_BAND_HZ = (min(CRT_LINE_HZ) - CRT_LINE_HALF_WIDTH_HZ, max(CRT_LINE_HZ) + CRT_LINE_HALF_WIDTH_HZ)
LINE_PRESENT_DB = 12.0
# The Tata tapes' lines peak at -94..-100 dBFS as a sine; a band with no line (Vaccin, a
# 16-bit floor) peaks at -136..-149, where any rounding residue stands 15+ dB "out".
LINE_MIN_DBFS = -125.0
LINE_SMOOTH_FRAMES = 5
LINE_MAX_JUMP_HZ = 10.0
LINE_MIN_SHARE = 0.5
# A line sits within 1 % of its standard rate (a tape-speed error moves a recorded one by
# less); a peak further out is something else, e.g. the edge of a 15.4 kHz codec cut.
LINE_NOMINAL_SHARE = 0.01
LINE_STEADY_SD_PPM = 200.0
LINE_LOCK_CORR = 0.6
# Harmonic k of either series is searched within +-max(2 bins, 1 % of k x 60 Hz): one width
# per index, so noise reads the same on both series.
HUM_SEARCH_SHARE = 0.01
HUM_SEARCH_MIN_BINS = 2
HUM_FLOOR_HZ = 20.0
HUM_EVIDENCE_BASE_DB = 6.0
# Noise alone reads at most 2.2 dB of evidence on 15 s and 2.7 dB on 10 s (300 white and pink
# clips each at 48 kHz), so 3 dB on >= 10 frames is hum; below it the summed rule decides.
MAINS_MIN_EVIDENCE_DB = 3.0
HUM_TRACK_DB = 10.0
# A track that moves less than this is constant for a correlation (1 ppm is 1/64 of a bin
# at 15.6 kHz on 1 s frames, below what the interpolation resolves in programme noise).
MIN_SPREAD_PPM = 1.0
CHANNEL_BAND_HZ = (100.0, 1000.0)
ENERGY_MIN_HZ = 20.0
INVERTED_MAX_CORR = -0.5
DUAL_MONO_TOLERANCE = 1e-6
DUAL_MONO_CHUNK = 1 << 20
NO_MAINS = {"mains_hz": None, "mains_evidence_db": None, "mains_excess_db": None}
NO_LINE = {"line_hz": None, "line_ppm": None, "line_sd_ppm": None, "line_hum_corr": None, "line_origin": None}
MONO_STATE = {"channels": 1, "correlation": None, "energy_ratio_db": None, "dual_mono": False, "dead_channel": None, "inverted": False}


def capture_profile(audio, rate):
    """The source's capture profile (module docstring) from `audio` `(frames,)` or `(frames, channels)` at `rate`."""
    audio = _as_columns(audio)
    state = channel_state(audio, rate)
    mono = _profile_mono(audio, state)
    if len(mono) < int(TRACK_FRAME_S * rate):
        return {"prog_bandwidth_hz": None, "brickwall_hz": None, **NO_MAINS, **NO_LINE, "channel_state": state}
    spectra = long_term_spectra(mono, rate)
    tracks = track_tones(mono, rate)
    mains = mains_profile(mono, rate, tracks["hum"])
    brickwall = brickwall_hz(spectra, rate)
    return {
        "prog_bandwidth_hz": programme_bandwidth_hz(spectra, brickwall),
        "brickwall_hz": brickwall,
        **mains,
        **line_profile(tracks, hum_series(*tracks["hum"][mains["mains_hz"]])),
        "channel_state": state,
    }


def _as_columns(audio):
    """`audio` as float32 `(frames, channels)`, a non-finite sample (a float WAV can carry NaN or inf) read as 0."""
    audio = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio[:, np.newaxis] if audio.ndim == 1 else audio


def _profile_mono(audio, state):
    """The live channel of a dead pair, (L - R) / 2 of an inverted one, the channel mean otherwise."""
    if audio.shape[1] < 2:
        return audio[:, 0]
    left, right = audio[:, 0], audio[:, 1]
    if state["dead_channel"] is not None:
        return right if state["dead_channel"] == "left" else left
    if state["inverted"]:
        return 0.5 * (left - right)
    return 0.5 * (left + right)


def profile_frame(rate):
    """The power of two nearest 40 ms at `rate`."""
    return int(2 ** int(round(np.log2(PROFILE_FRAME_S * rate))))


def _blocks(mono, frame, block_frames):
    """Consecutive slices of whole frames, `block_frames` frames at a time (a trailing partial frame is dropped)."""
    end = (len(mono) // frame) * frame
    step = frame * block_frames
    for start in range(0, end, step):
        stop = min(start + step, end)
        yield mono[start:stop]


def sixth_octave_bands(freqs, rate):
    """Centres and the `(bins, bands)` membership matrix of the 1/6-octave bands from 250 Hz to Nyquist.

    The CRT line and its skirt are left out (`dsp_metrics.band_bins`); a band left with no bin
    at all is dropped.
    """
    first = int(np.ceil(BANDS_PER_OCTAVE * np.log2(BANDWIDTH_MIN_HZ / BAND_REFERENCE_HZ)))
    last = int(np.floor(BANDS_PER_OCTAVE * np.log2(rate / 2.0 / BAND_REFERENCE_HZ) - 0.5))
    centres = BAND_REFERENCE_HZ * 2.0 ** (np.arange(first, last + 1) / BANDS_PER_OCTAVE)
    half = 2.0 ** (0.5 / BANDS_PER_OCTAVE)
    members = np.stack([band_bins(freqs, (centre / half, centre * half)) for centre in centres], axis=1).astype(np.float64)
    keep = members.sum(axis=0) > 0
    return centres[keep], members[:, keep]


def long_term_spectra(mono, rate):
    """The profile frames' gap and loud classes, 1/6-octave band power per frame and the gap frames' mean power per bin.

    Two blocked passes: the frame levels first (the classes need the whole file's
    percentiles), then the spectra.
    """
    frame = profile_frame(rate)
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    centres, members = sixth_octave_bands(freqs, rate)
    gap, loud = _classes(_frame_levels_db(mono, frame))
    band_power, gap_total = _band_and_gap_power(mono, rate, frame, members, gap)
    return {
        "freqs": freqs,
        "gap_power": gap_total / max(1, int(gap.sum())),
        "gap": gap,
        "loud": loud,
        "band_power": band_power,
        "band_bins": members.sum(axis=0),
        "centres": centres,
    }


def _band_and_gap_power(mono, rate, frame, members, gap):
    """Per profile frame the band power `(frames, bands)`, and the power per bin summed over the `gap` frames."""
    bands, gap_total = [np.zeros((0, members.shape[1]))], np.zeros(len(members))
    for index, block in enumerate(_blocks(mono, frame, BLOCK_FRAMES)):
        _freqs, power, _level = framed_psd(block, rate, frame)
        bands.append(power @ members)
        first = index * BLOCK_FRAMES
        last = first + len(power)
        gap_total += power[gap[first:last]].sum(axis=0)
    return np.concatenate(bands), gap_total


def _frame_levels_db(mono, frame):
    """RMS level in dB of each whole frame, read block by block."""
    levels = [np.zeros(0)] + [frame_levels(block, frame) for block in _blocks(mono, frame, BLOCK_FRAMES)]
    return 20.0 * np.log10(np.concatenate(levels) + 1e-9)


def _classes(level_db):
    """`(gap, loud)` frame masks (`pause_metrics.level_classes`), both empty when the file is too short to class."""
    if len(level_db) < 3 * MIN_CLASS_FRAMES:
        return np.zeros(len(level_db), dtype=bool), np.zeros(len(level_db), dtype=bool)
    _deep, gap, loud = level_classes(level_db)
    return gap, loud


def programme_bandwidth_hz(spectra, top_hz=None):
    """The centre of the highest 1/6-octave band with a sustained >= 3 dB loud-over-gap excess, or None.

    Bands above `top_hz` (the brickwall) or more than 80 dB under the loudest band's density
    are left out.
    """
    gap, loud, band_power = spectra["gap"], spectra["loud"], spectra["band_power"]
    if min(int(gap.sum()), int(loud.sum())) < MIN_CLASS_FRAMES:
        return None
    loud_power, gap_power = np.median(band_power[loud], axis=0), np.median(band_power[gap], axis=0)
    density_db = 10.0 * np.log10(loud_power / spectra["band_bins"] + EPS)
    present = (density_db >= density_db.max() - PROGRAMME_RANGE_DB) & (spectra["centres"] <= (top_hz or np.inf))
    excess = np.where(present, 10.0 * np.log10((loud_power + EPS) / (gap_power + EPS)), -np.inf)
    return _highest_sustained(spectra["centres"], excess)


def _highest_sustained(centres, excess_db):
    """The highest centre whose band and the band below both clear `EXCESS_MIN_DB`."""
    qualifies = excess_db >= EXCESS_MIN_DB
    sustained = qualifies & np.concatenate(([False], qualifies[:-1]))
    hits = np.flatnonzero(sustained)
    return round(float(centres[hits[-1]]), 1) if len(hits) else None


def brickwall_hz(spectra, rate):
    """The top edge of a > 30 dB cliff on the gap frames' mean spectrum, or None (module docstring)."""
    if int(spectra["gap"].sum()) < MIN_CLASS_FRAMES:
        return None
    starts = np.arange(BRICKWALL_MIN_HZ, BRICKWALL_TOP_SHARE * rate / 2.0 - BRICKWALL_BAND_HZ, BRICKWALL_BAND_HZ)
    level = np.array([_band_level_db(spectra["freqs"], spectra["gap_power"], start) for start in starts])
    hits = np.flatnonzero(level - _max_beyond(starts, level) >= BRICKWALL_FALL_DB)
    return round(float(starts[hits[-1]] + BRICKWALL_BAND_HZ), 1) if len(hits) else None


def _band_level_db(freqs, power, start):
    """Mean power per bin of the 100 Hz band from `start`, in dB, or NaN when the CRT exclusion empties it."""
    bins = band_bins(freqs, (start, start + BRICKWALL_BAND_HZ))
    return float(10.0 * np.log10(np.mean(power[bins]) + EPS)) if bins.any() else np.nan


def _max_beyond(starts, level):
    """For each band, the loudest band starting 500 Hz or more above it (+inf when there is none)."""
    filled = np.where(np.isfinite(level), level, -np.inf)
    suffix = np.maximum.accumulate(filled[::-1])[::-1]
    suffix = np.append(np.where(np.isfinite(suffix), suffix, np.inf), np.inf)
    return suffix[np.searchsorted(starts, starts + BRICKWALL_SPAN_HZ - 1e-6)]


def track_tones(mono, rate):
    """Per 1 s frame: `{"line_hz", "line_prominence", "hum": {mains: (ppm, prominence) per harmonic}}`."""
    frame = int(round(TRACK_FRAME_S * rate))
    rows = [_frame_tracks(block, rate, frame) for block in _blocks(mono, frame, TRACK_BLOCK_FRAMES)]
    empty = [(np.zeros(0),)] * 2 + [(np.zeros((0, HARMONICS)),)] * (2 * len(MAINS_CANDIDATES_HZ))
    line_hz, line_prominence, *hum = (np.concatenate(column) for column in (list(zip(*rows)) or empty))
    pairs = zip(hum[0::2], hum[1::2])
    return {"line_hz": line_hz, "line_prominence": line_prominence, "hum": dict(zip(MAINS_CANDIDATES_HZ, pairs))}


def _frame_tracks(block, rate, frame):
    """The line track and every mains candidate's harmonic tracks of one block of 1 s frames."""
    freqs, power, _level = framed_psd(block, rate, frame)
    hum = []
    for mains in MAINS_CANDIDATES_HZ:
        hum.extend(_hum_tracks(power, freqs, mains))
    return (*_line_track(power, freqs), *hum)


def _line_track(power, freqs):
    """Per frame: the line band's strongest peak in Hz and its prominence.

    The prominence is -inf on a frame whose peak sits under `LINE_MIN_DBFS` as a sine
    (Hann-windowed peak power P on N samples is a sine of amplitude sqrt(16 P) / N), and on
    every frame when the band lies above Nyquist.
    """
    if LINE_BAND_HZ[1] + 2.0 >= freqs[-1]:
        return np.full(len(power), np.nan), np.full(len(power), -np.inf)
    first, stop = _bin_range(freqs, LINE_BAND_HZ)
    peak_bin, prominence = _peak_track(power, (first, stop), (first, stop))
    frame = 2 * (len(freqs) - 1)
    level_dbfs = 10.0 * np.log10(16.0 * power[:, first:stop].max(axis=1) / frame**2 + EPS)
    return peak_bin * float(freqs[1]), np.where(level_dbfs >= LINE_MIN_DBFS, prominence, -np.inf)


def _bin_range(freqs, band_hz):
    """`(first, stop)` bin indices of `band_hz`."""
    return int(np.searchsorted(freqs, band_hz[0])), int(np.searchsorted(freqs, band_hz[1]))


def _peak_track(power, search, floor):
    """Per frame: the interpolated bin of the strongest peak in `search` and its prominence over the median of `floor`, dB."""
    (search_lo, search_hi), (floor_lo, floor_hi) = search, floor
    band = power[:, search_lo:search_hi]
    rows = np.arange(len(band))
    peak = np.argmax(band, axis=1)
    reference = np.median(power[:, floor_lo:floor_hi], axis=1)
    prominence = 10.0 * np.log10((band[rows, peak] + EPS) / (reference + EPS))
    return search_lo + peak + grandke_offset(band, peak), prominence


def grandke_offset(band, peak):
    """Fractional bin offset of each frame's peak: Grandke's Hann-window interpolation, (2a - 1) / (a + 1).

    `a` is the larger neighbour's magnitude over the peak's; for a single tone under a Hann
    window it gives the exact offset. A peak on the edge of `band` keeps offset 0.
    """
    rows = np.arange(len(band))
    inner = (peak > 0) & (peak < band.shape[1] - 1)
    safe = np.clip(peak, 1, band.shape[1] - 2)
    magnitude = np.sqrt(band)
    left, centre, right = magnitude[rows, safe - 1], magnitude[rows, safe], magnitude[rows, safe + 1]
    alpha = np.maximum(left, right) / (centre + EPS)
    offset = np.clip((2.0 * alpha - 1.0) / (alpha + 1.0), 0.0, 0.5)
    return np.where(inner, np.where(right >= left, offset, -offset), 0.0)


def _hum_tracks(power, freqs, mains_hz):
    """Per frame and harmonic (1..8): the peak's offset from k x mains in ppm and its prominence in dB."""
    step = float(freqs[1])
    ppm, prominence = [], []
    for harmonic in range(1, HARMONICS + 1):
        target = harmonic * mains_hz
        search, floor = hum_windows(target, harmonic, step)
        peak_bin, peak_prominence = _peak_track(power, search, floor)
        ppm.append((peak_bin * step / target - 1.0) * PPM)
        prominence.append(peak_prominence)
    return np.stack(ppm, axis=1), np.stack(prominence, axis=1)


def hum_windows(target_hz, harmonic, step_hz):
    """`(search, floor)` bin ranges of one harmonic: +-max(2 bins, 1 % of k x 60 Hz) and +-20 Hz around its bin.

    The search half-width depends on the harmonic's index, not on its frequency, so both
    series search the same number of bins at each index and noise reads the same on each.
    """
    centre = int(round(target_hz / step_hz))
    half = max(HUM_SEARCH_MIN_BINS, int(np.ceil(HUM_SEARCH_SHARE * harmonic * max(MAINS_CANDIDATES_HZ) / step_hz)))
    floor = int(round(HUM_FLOOR_HZ / step_hz))
    return (centre - half, centre + half + 1), (centre - floor, centre + floor + 1)


def mains_profile(mono, rate, hum_tracks):
    """`{"mains_hz", "mains_evidence_db", "mains_excess_db"}` from `track_tones`'s hum tracks (module docstring)."""
    evidence = {mains: harmonic_evidence_db(prominence) for mains, (_ppm, prominence) in hum_tracks.items()}
    excerpt = mains_excerpt(mono, rate)
    mains = max(evidence, key=evidence.get)
    if not frames_decide(evidence[mains], len(hum_tracks[mains][1])):
        mains = summed_rule_mains(excerpt, rate)
    return {
        "mains_hz": mains,
        "mains_evidence_db": round(evidence[mains], 2),
        "mains_excess_db": round(float(hum_excess_db(excerpt, rate, mains)), 2),
    }


def frames_decide(evidence_db, frames):
    """Whether the 1 s frames name the mains: >= `MAINS_MIN_EVIDENCE_DB` of evidence on >= `MIN_TRACK_FRAMES` frames."""
    return frames >= MIN_TRACK_FRAMES and evidence_db >= MAINS_MIN_EVIDENCE_DB


def summed_rule_mains(excerpt, rate):
    """The mains `measure_hum._mains_by_evidence` names: the series whose summed peaks stand furthest over their summed floors."""
    return float(_mains_by_evidence(excerpt, rate))


def harmonic_evidence_db(prominence):
    """Sum over harmonics of the median prominence above `HUM_EVIDENCE_BASE_DB` (0 without frames)."""
    if len(prominence) == 0:
        return 0.0
    return float(np.sum(np.maximum(0.0, np.median(prominence, axis=0) - HUM_EVIDENCE_BASE_DB)))


def mains_excerpt(mono, rate):
    """The whole signal up to 2 minutes, eight evenly spaced 15 s blocks of a longer one."""
    block = int(MAINS_BLOCK_S * rate)
    if len(mono) <= block * MAINS_BLOCKS:
        return np.asarray(mono, dtype=np.float64)
    starts = np.linspace(0, len(mono) - block, MAINS_BLOCKS).astype(int)
    return np.concatenate([np.asarray(mono[start:stop], dtype=np.float64) for start, stop in zip(starts, starts + block)])


def hum_series(ppm, prominence):
    """The ppm track of the harmonic with the strongest median prominence, NaN where it stands < 10 dB out."""
    if len(prominence) == 0:
        return np.zeros(0)
    best = int(np.argmax(np.median(prominence, axis=0)))
    return np.where(prominence[:, best] >= HUM_TRACK_DB, ppm[:, best], np.nan)


def line_profile(tracks, hum_ppm):
    """The line fields of the profile (module docstring) from `track_tones` and the hum's ppm track, all None without a line."""
    tracked = tracks["line_prominence"] >= LINE_PRESENT_DB
    line_hz, hum_ppm = tracks["line_hz"][tracked], hum_ppm[tracked]
    coherent = _coherent(line_hz)
    if coherent.sum() < max(MIN_TRACK_FRAMES, LINE_MIN_SHARE * len(tracked)):
        return dict(NO_LINE)
    nominal = _nominal_line_hz(line_hz[coherent])
    if abs(np.median(line_hz[coherent]) / nominal - 1.0) > LINE_NOMINAL_SHARE:
        return dict(NO_LINE)
    return _line_fields(line_hz[coherent], hum_ppm[coherent], nominal)


def _coherent(line_hz):
    """Frames within 10 Hz of the running median of five: a line moves smoothly, a stray peak does not."""
    if len(line_hz) == 0:
        return np.zeros(0, dtype=bool)
    smooth = scipy.ndimage.median_filter(line_hz, size=LINE_SMOOTH_FRAMES, mode="nearest")
    return np.abs(line_hz - smooth) <= LINE_MAX_JUMP_HZ


def _nominal_line_hz(line_hz):
    """The standard line rate (`dsp_metrics.CRT_LINE_HZ`) nearest the track's median."""
    lines = np.asarray(CRT_LINE_HZ)
    return float(lines[np.argmin(np.abs(lines - np.median(line_hz)))])


def _line_fields(line_hz, hum_ppm, nominal):
    """The line fields from the coherent frames' line frequencies, the hum's ppm on the same frames and the nominal rate."""
    ppm = (line_hz / nominal - 1.0) * PPM
    spread = float(np.std(ppm))
    corr = _correlation(ppm, hum_ppm)
    return {
        "line_hz": round(float(np.mean(line_hz)), 3),
        "line_ppm": round(float(np.mean(ppm)), 1),
        "line_sd_ppm": round(spread, 1),
        "line_hum_corr": corr,
        "line_origin": line_origin(spread, corr),
    }


def _correlation(line_ppm, hum_ppm):
    """Pearson r of the two tracks on the frames both hold, or None with too few frames or a constant track."""
    both = np.isfinite(line_ppm) & np.isfinite(hum_ppm)
    if both.sum() < MIN_CORR_FRAMES:
        return None
    line_ppm, hum_ppm = line_ppm[both], hum_ppm[both]
    if min(float(np.std(line_ppm)), float(np.std(hum_ppm))) < MIN_SPREAD_PPM:
        return None
    return round(float(np.corrcoef(line_ppm, hum_ppm)[0, 1]), 3)


def line_origin(spread_ppm, corr):
    """`speed_locked` when the line moves with the hum, `playback_chain` when it holds still, `unresolved` otherwise."""
    if corr is not None and corr >= LINE_LOCK_CORR:
        return "speed_locked"
    if spread_ppm <= LINE_STEADY_SD_PPM:
        return "playback_chain"
    return "unresolved"


def channel_state(audio, rate):
    """`{"channels", "correlation", "energy_ratio_db", "dual_mono", "dead_channel", "inverted"}` (module docstring)."""
    audio = _as_columns(audio)
    if audio.shape[1] < 2:
        return dict(MONO_STATE)
    left, right = audio[:, 0], audio[:, 1]
    cross, left_band, right_band, left_all, right_all = _channel_sums(left, right, rate)
    ratio = _energy_ratio_db(left_all, right_all)
    corr = _signed_correlation(cross, left_band, right_band)
    dead = _dead_channel(ratio)
    return {
        "channels": int(audio.shape[1]),
        "correlation": corr,
        "energy_ratio_db": ratio,
        "dual_mono": _dual_mono(left, right),
        "dead_channel": dead,
        "inverted": bool(dead is None and corr is not None and corr <= INVERTED_MAX_CORR),
    }


def _channel_sums(left, right, rate):
    """Blocked sums over Hann frames: the 100-1000 Hz cross power and each side's power there, then each side's power >= 20 Hz."""
    frame = profile_frame(rate)
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    band = (freqs >= CHANNEL_BAND_HZ[0]) & (freqs < CHANNEL_BAND_HZ[1])
    audible = freqs >= ENERGY_MIN_HZ
    totals = np.zeros(5)
    for left_block, right_block in zip(_blocks(left, frame, BLOCK_FRAMES), _blocks(right, frame, BLOCK_FRAMES)):
        totals += _block_sums(_spectrum(left_block, frame), _spectrum(right_block, frame), band, audible)
    return tuple(float(value) for value in totals)


def _spectrum(block, frame):
    """Hann-windowed rfft of each whole frame of `block`, `(frames, bins)`."""
    frames = np.asarray(block, dtype=np.float64).reshape(-1, frame)
    return np.fft.rfft(frames * np.hanning(frame), axis=1)


def _block_sums(left, right, band, audible):
    """The five sums `_channel_sums` accumulates, for one block of spectra."""
    left_power, right_power = np.abs(left) ** 2, np.abs(right) ** 2
    return np.array(
        [
            np.sum(np.real(left * np.conj(right))[:, band]),
            np.sum(left_power[:, band]),
            np.sum(right_power[:, band]),
            np.sum(left_power[:, audible]),
            np.sum(right_power[:, audible]),
        ]
    )


def _energy_ratio_db(left_power, right_power):
    """Left over right power in dB, or None when both sides are silent."""
    if left_power + right_power <= 0.0:
        return None
    return round(float(10.0 * np.log10((left_power + EPS) / (right_power + EPS))), 2)


def _signed_correlation(cross, left_power, right_power):
    """The signed band correlation from the cross and own powers, or None when either side is silent there."""
    if left_power * right_power <= 0.0:
        return None
    return round(float(cross / np.sqrt(left_power * right_power)), 4)


def _dead_channel(ratio_db):
    """The side ("left" / "right") sitting more than `DEAD_CHANNEL_DB` under the other, else None."""
    if ratio_db is None or abs(ratio_db) <= DEAD_CHANNEL_DB:
        return None
    return "right" if ratio_db > 0.0 else "left"


def _dual_mono(left, right):
    """Whether the two channels agree sample for sample (within float rounding), read in chunks."""
    for start in range(0, len(left), DUAL_MONO_CHUNK):
        stop = start + DUAL_MONO_CHUNK
        if not np.all(np.abs(left[start:stop] - right[start:stop]) <= DUAL_MONO_TOLERANCE):
            return False
    return bool(len(left))
