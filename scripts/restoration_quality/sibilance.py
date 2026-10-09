"""Sibilance integrity: whether the 's' sounds kept their shape.

The listener heard APL "distort the spoken 's'"; nothing in the harness read it, and nothing
published detects it. Measured on the Tata listening set with 10 ms frames, fricatives are
the source frames at least 12 dB above the pause floor (the p15 level; an 's' is quieter
than the vowels around it, so a vowel-relative cut would miss it) whose spectral centroid
sits above 3.5 kHz with more than 40 % of their power in 4-12 kHz (about 1-2 % of
Tele7abc's frames, ~19 per 15 s window in the planning prototype). On those frames, the
output's centroid minus the source's, net of the same shift on the non-fricative loud frames:
APL baseline +629 Hz, no_air +490, roformer +631, the APL plateau +269; cathar alpha 2 -5,
cathar 0.7.5 -27, cathar baseline -2. APL's neural stage
removes the 1-4 kHz body under the 's' and leaves a thin one; cathar's de-esser lowers the
whole 's' (level jitter 1.6 dB against 0.3) without changing its shape, and the listener
did not complain about it.

Ear v3 (R2) adds two readings beside the three net ones, on the same source fricative frames:

- `sib_abs_level_db`: the 4 kHz-min(12 kHz, bandwidth) level change of the 's', output minus
  source (median over the fricative frames of the per-frame dB change), minus R1's programme
  gain match (`balance_metrics.band_profile`'s: the median out/src dB over the 300-3000 Hz
  programme cells of the live frames, a cell standing 18 dB over its bin's bias-corrected mean
  noise; None wherever R1 reads none), reproduced step for step from R1's public constants
  and pinned to R1's own gain by a unit test. With R1's 90 dB range term alone, hiss cells
  joined the match: on the test voice low-passed at 8 kHz with -45 dBFS white hiss removed by
  an oracle it read -0.054 / -0.067 dB (seeds 7 / 8) against R1's -0.017 / -0.011, a
  spurious 0.04-0.06 dB lift in this reading (on the clean voice the two gains differ by
  under 0.002 dB, so the figures below stand). It is NOT net of the plain frames: the v2
  level reading subtracts the plain frames' own 4-12 kHz change, so a shelf that lifts every
  frame cancels in it: the user heard a thinner 's' at +2 dB air than at +1 dB while no
  reading moved. (Round 1 also called `apl__no_air` thin; if this reading calls that file
  clean, its "thin" was a distortion, the texture reading's business, not a level change.)
  Measured on the synthetic voice of the unit tests (vowels to 4.6 kHz, 's' bursts of
  4-10 kHz noise over a 1-4 kHz body; four seeds): a flat +2 dB shelf above 3.5 kHz reads
  +2.00 with the net level reading 0.00; any broadband gain reads 0; the 12 dB dulled 's'
  (6-12 kHz top lowered) reads -4.65..-4.76; the 12 dB thinned 's' (1-4 kHz body lowered)
  reads -0.18..-0.20, since its top is untouched and only the body filter's skirt above
  4 kHz leaves (the net body ratio reads that one, +10.4 dB). An RBJ +2 dB shelf at 7.5 kHz
  reads +0.84..+0.86 (its mean lift over the band) and moves the net level reading by +0.72
  on this voice, because the vowels' 4-12 kHz energy sits at 4-4.6 kHz where that shelf has
  not risen yet. Random STFT phase on the 's' reads -2.7..-2.9: the overlap-add of
  incoherent frames loses power.
- `sib_texture_db`: the roughness of the 's' spectrum, output minus source. Per 4-12 kHz
  bin, the spread (std over the fricative frames) of the cell's level relative to its own
  frame's median, floored 30 dB down; the reading is the median over bins of output minus
  source, in dB. A static filter shifts each bin's deviations by a constant and a frame's
  broadband gain shifts all of its bins alike, so shelves, gains and the thinned/dulled
  fixtures read ~0 (|x| < 0.05 dB over four seeds); level jitter from frame to frame is
  R9's flicker reading, not this one (a 40 Hz gargle on the 's' reads -0.02..+0.05). What
  moves it is a fluctuating time-frequency mask on the 's' (the round-1 "distorted s" of APL
  baseline and roformer is the hypothesis it stands for, untested on that verdict yet):
  random spectral islands in the 's' (a 256-point STFT with 15/30/50 % of its 4-12 kHz cells
  zeroed and the rest raised to keep the power) read +0.13..+0.34 / +0.46..+0.61 /
  +0.72..+1.00, random STFT phase on the 's' +0.25..+0.42, 2 ms sign flips of its top
  +0.15..+0.29. Kurtosis of the band power, the LKR form, was tried first: normalised per
  frame by its sum it read the 12 dB dull 's' at +0.16..+0.57 dB (by its median -0.21..+0.54),
  and un-normalised it read the 40 Hz gargle at +2.43.

The benign floor (plan principle 9: hiss removal must not move a reading), measured with an
oracle denoiser: the test voice low-passed at 8 kHz (8th-order Butterworth, forward and
back), white hiss added to the source only, the clean voice as the output; seeds 7-10. Read
with `bandwidth_hz=8000`, at -50 / -55 dBFS hiss the level moves -0.020..-0.036 /
-0.007..-0.016 dB and the texture -0.037..+0.029 / -0.008..+0.037. At -45 dBFS, where the 's'
stands 13.5 dB over the hiss (the detector wants 12), the level moves -0.05..-0.12 and the
texture -0.03..+0.19, as far as the 15 % islands. Read without a cap, the 8-12 kHz bins carry
hiss alone and clearing it moves the texture -0.11..-0.21 at -50 dBFS (-0.05..-0.43 over
-45..-55 dBFS): the size of the 15 % islands, with the opposite sign. The level moves
-0.04..-0.05 there (-0.01..-0.18), a true band change (the hiss sat in the 's'), so its
benign floor is its dead zone. The runner's `_listener_window` caps both readings at R0's
`brickwall_hz`, not at `prog_bandwidth_hz`: the programme bandwidth is read on the loud,
voiced frames, and no 's' frame is one. A brickwall at or above 12 kHz, or none (the Tata
tapes: Tele7abc 18.9 kHz, SOTI 15.4 kHz, Vaccin none), leaves the band at 12 kHz, so the
uncapped floor above applies. The texture is scored all the same: two-sided around 0 in
the `tata_v3` grid, where that floor sits inside the 1.75 dB dead zone, and "lower is
better" in the scorecard, whose tail and listening picks (`listening.DEFAULT_PICK_METRICS`)
rank a denoiser that clears hiss above the programme band 0.1-0.2 dB better than one that
leaves it.

`sib_peak_hz`, `sib_rel_amp_db` and the Romanian s-comma class stay out until they reproduce a
verdict (the plan's R2 note; listeners judging /s/ by peak place was a refuted claim).
"""

import numpy as np
import scipy.signal

from scripts.restoration_quality import auditory, balance_metrics
from scripts.restoration_quality.dsp_metrics import framed_psd

FRAME_S = 0.01
# A fricative is quieter than the vowels around it, so it is told from the pauses, not from
# the vowels: at least this far above the floor (the p15 level, the deep pauses).
FLOOR_PERCENTILE = 15.0
LEVEL_ABOVE_FLOOR_DB = 12.0
GAP_PERCENTILE = 40.0
SIB_ABOVE_HISS_DB = 10.0
LOUD_PERCENTILE = 70.0
CENTROID_MIN_HZ = 3500.0
HF_SHARE_MIN = 0.4
SIB_BAND_HZ = (4000.0, 12000.0)
BODY_BAND_HZ = (1000.0, 4000.0)
MIN_FRAMES = 10
NAMES = ("sib_centroid_hz", "sib_body_db", "sib_level_db")
# R2 (ear v3): the absolute twin of the level reading and the texture of the 's'.
ABS_NAMES = ("sib_abs_level_db", "sib_texture_db")
READINGS = NAMES + ABS_NAMES
# The programme gain match is R1's (`balance_metrics.band_profile`), reproduced step for step
# from its public constants so the two readings agree on a file's gain; a unit test pins them.
GAIN_NOISE_FACTOR = balance_metrics.NOISE_MEAN_OVER_FLOOR * 10.0 ** (balance_metrics.NOISE_MARGIN_DB / 10.0)
GAIN_RANGE = 10.0 ** (-balance_metrics.DYNAMIC_RANGE_DB / 10.0)
TEXTURE_FLOOR_DB = -30.0
EPS = 1e-20


def centroid_hz(power, freqs):
    """Spectral centroid per frame, in Hz."""
    return (power * freqs).sum(axis=1) / (power.sum(axis=1) + 1e-20)


def _band(power, freqs, band):
    return power[:, (freqs >= band[0]) & (freqs < band[1])].sum(axis=1)


def fricative_frames(power, freqs, level_db):
    """Frames the source calls fricative: above the pause floor, centroid above 3.5 kHz, most of the power in 4-12 kHz.

    Tape hiss has a high centroid too, so a frame must also carry 4-12 kHz power well above
    the hiss in the gaps (the frames between the p15 and p40 levels); without that rule the
    hiss frames counted as fricatives and a denoiser that removed hiss read as dulling the 's'.
    """
    floor = np.percentile(level_db, FLOOR_PERCENTILE)
    above_floor = level_db >= floor + LEVEL_ABOVE_FLOOR_DB
    sib = _band(power, freqs, SIB_BAND_HZ)
    gaps = (level_db > floor) & (level_db <= np.percentile(level_db, GAP_PERCENTILE))
    hiss = np.median(sib[gaps]) if gaps.any() else 0.0
    share = sib / (power.sum(axis=1) + 1e-20)
    return (
        above_floor
        & (sib > hiss * 10 ** (SIB_ABOVE_HISS_DB / 10.0))
        & (centroid_hz(power, freqs) > CENTROID_MIN_HZ)
        & (share > HF_SHARE_MIN)
    )


def fricative_mask(mono, rate):
    """Per-sample boolean mask of the fricative frames of `mono` (for the calibration's degradations)."""
    frame = max(1, int(FRAME_S * rate))
    freqs, power, level = framed_psd(mono, rate, frame)
    if len(level) == 0:
        return np.zeros(len(mono), dtype=bool)
    frames = fricative_frames(power, freqs, 20.0 * np.log10(level + 1e-9))
    mask = np.repeat(frames, frame)
    return np.concatenate([mask, np.zeros(len(mono) - len(mask), dtype=bool)])


def _shape(power, freqs, fricative, plain):
    """`(centroid on fricatives, body ratio on fricatives net of plain frames, level net of plain frames)` for one side."""
    centroid = float(np.median(centroid_hz(power, freqs)[fricative]))
    ratio = 10.0 * np.log10((_band(power, freqs, SIB_BAND_HZ) + 1e-20) / (_band(power, freqs, BODY_BAND_HZ) + 1e-20))
    body = float(np.median(ratio[fricative]) - np.median(ratio[plain]))
    sib_level = 10.0 * np.log10(_band(power, freqs, SIB_BAND_HZ) + 1e-20)
    level = float(np.median(sib_level[fricative]) - np.median(sib_level[plain]))
    return centroid, body, level


def _classes(power, level_db):
    """`(fricative, plain)` frame masks of the source, or None when either class is too thin."""
    fricative = fricative_frames(power[0], power[1], level_db)
    plain = (level_db >= np.percentile(level_db, LOUD_PERCENTILE)) & ~fricative
    if fricative.sum() < MIN_FRAMES or plain.sum() < MIN_FRAMES:
        return None
    return fricative, plain


def sib_readings(source, output, rate, bandwidth_hz=None):
    """`{name: (source, output)}` for every name in `READINGS`; all (None, None) when the window has too few fricatives.

    The output centroid is net of the centroid shift on non-fricative loud frames. The R2
    readings are paired: 0.0 on the source side, the change on the output side.
    `bandwidth_hz` caps their band (the runner passes R0's brickwall, module docstring); None
    reads to 12 kHz, and a cap at or under 4 kHz (0 and NaN included) leaves both unread.
    `source` and `output` are one mono window each.
    """
    frame = max(1, int(FRAME_S * rate))
    count = min(len(source), len(output)) // frame
    if count < 4 * MIN_FRAMES:
        return _unread(READINGS)
    freqs, src_power, src_level = framed_psd(source[: count * frame], rate, frame)
    _freqs, out_power, _level = framed_psd(output[: count * frame], rate, frame)
    classes = _classes((src_power, freqs), 20.0 * np.log10(src_level + 1e-9))
    if classes is None:
        return _unread(READINGS)
    powers = (src_power, out_power, freqs)
    return {**_net_readings(powers, classes), **_abs_readings(_common(source, output, rate), powers, classes[0], bandwidth_hz)}


def _common(source, output, rate):
    """`(source, output, rate)` over the pair's common length: the window R1 matches the gain on, not trimmed to 10 ms frames."""
    length = min(len(source), len(output))
    return source[:length], output[:length], rate


def _unread(names):
    return {name: (None, None) for name in names}


def _paired(value):
    """A paired reading: 0.0 on the source side and `value` on the output side, or (None, None) when unread."""
    return (None, None) if value is None else (0.0, value)


def _net_readings(powers, classes):
    """The v2 readings: the centroid net of the plain frames' shift, the body ratio and the level net of the plain frames."""
    src_power, out_power, freqs = powers
    fricative, plain = classes
    src = _shape(src_power, freqs, fricative, plain)
    out = _shape(out_power, freqs, fricative, plain)
    plain_shift = float(np.median(centroid_hz(out_power, freqs)[plain]) - np.median(centroid_hz(src_power, freqs)[plain]))
    return {"sib_centroid_hz": (src[0], out[0] - plain_shift), "sib_body_db": (src[1], out[1]), "sib_level_db": (src[2], out[2])}


def _abs_readings(signals, powers, fricative, bandwidth_hz):
    """The R2 readings on the fricative frames' 4 kHz-min(12 kHz, bandwidth) bins; unread when no bin is left.

    Only None means "no cap": a bandwidth of 0 (or NaN, which `np.minimum` carries) leaves no bin.
    """
    src_power, out_power, freqs = powers
    cap = SIB_BAND_HZ[1] if bandwidth_hz is None else np.minimum(SIB_BAND_HZ[1], float(bandwidth_hz))
    bins = (freqs >= SIB_BAND_HZ[0]) & (freqs < cap)
    if not bins.any():
        return _unread(ABS_NAMES)
    src_band, out_band = src_power[fricative][:, bins], out_power[fricative][:, bins]
    level = abs_level_db(src_band, out_band, _programme_gain_db(*signals))
    return {"sib_abs_level_db": _paired(level), "sib_texture_db": _paired(texture_db(src_band, out_band))}


def abs_level_db(src_band, out_band, gain_db):
    """Median per-frame dB change of the band power (`(frames, bins)` per side) minus the programme gain; None without a gain."""
    if gain_db is None:
        return None
    change = 10.0 * np.log10((out_band.sum(axis=1) + EPS) / (src_band.sum(axis=1) + EPS))
    return float(np.median(change) - gain_db)


def texture_db(src_band, out_band):
    """Median over bins of the output's spectral roughness minus the source's, in dB; None when the output band is silent."""
    if not np.any(out_band):
        return None
    return float(np.median(_roughness_db(out_band) - _roughness_db(src_band)))


def _roughness_db(band):
    """Per bin: the std over frames of the cell's level relative to its frame's median level, floored 30 dB down."""
    level = 10.0 * np.log10(band + EPS)
    deviation = np.maximum(level - np.median(level, axis=1, keepdims=True), TEXTURE_FLOOR_DB)
    return deviation.std(axis=0)


def _programme_gain_db(source, output, rate):
    """R1's programme gain match on this mono window, or None wherever R1 reads none (a gain under -60 dB included)."""
    spectra = _gain_spectra(source, output, rate)
    gain = None if spectra is None else auditory.programme_gain_db(*spectra, _gain_cells(spectra[0]))
    return None if gain is None or gain < balance_metrics.MIN_PROGRAMME_GAIN_DB else gain


def _gain_spectra(source, output, rate):
    """`(src_power, out_power, freqs)` on R1's live frames (within 90 dB of the loudest), or None where R1 refuses the window."""
    if not (_gain_readable(source) and _gain_readable(output)):
        return None
    freqs, src_power = _stft_power(source, rate)
    level = src_power.sum(axis=0)
    live = level > level.max() * GAIN_RANGE
    if int(live.sum()) < balance_metrics.MIN_LIVE_FRAMES:
        return None
    return src_power[:, live], _stft_power(output, rate)[1][:, live], freqs


def _gain_readable(mono):
    """R1's window guard: at or above -70 dBFS RMS.

    R1's other half, one STFT frame of length, never binds here: a window with fricatives runs
    over 7 kHz (a 3.5 kHz centroid) for at least 40 frames of 10 ms, over 2800 samples.
    """
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    return 20.0 * np.log10(rms + EPS) >= balance_metrics.MIN_LEVEL_DBFS


def _gain_cells(src_power):
    """R1's programme cells: loud frames over the p10 floor + 12 dB, 18 dB over the bias-corrected mean noise, within 90 dB."""
    floor = np.percentile(src_power, auditory.PROGRAMME_FLOOR_PERCENTILE, axis=1, keepdims=True)
    over_noise = src_power >= floor * GAIN_NOISE_FACTOR
    return auditory.programme_cells(src_power) & over_noise & (src_power > src_power.max() * GAIN_RANGE)


def _stft_power(mono, rate):
    """STFT power `(bins, frames)` on R1's grid (Hann, 2048/512) and its bin frequencies."""
    frame = balance_metrics.STFT_FRAME
    freqs, _times, spec = scipy.signal.stft(
        np.asarray(mono, dtype=np.float64), fs=rate, window="hann", nperseg=frame, noverlap=frame - balance_metrics.STFT_HOP
    )
    return freqs, np.abs(spec) ** 2
