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
"""

import numpy as np

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


def sib_readings(source, output, rate):
    """`{name: (source, output)}`; the output centroid is net of the centroid shift on non-fricative loud frames."""
    frame = max(1, int(FRAME_S * rate))
    count = min(len(source), len(output)) // frame
    if count < 4 * MIN_FRAMES:
        return {name: (None, None) for name in NAMES}
    freqs, src_power, src_level = framed_psd(source[: count * frame], rate, frame)
    _freqs, out_power, _level = framed_psd(output[: count * frame], rate, frame)
    classes = _classes((src_power, freqs), 20.0 * np.log10(src_level + 1e-9))
    if classes is None:
        return {name: (None, None) for name in NAMES}
    fricative, plain = classes
    src = _shape(src_power, freqs, fricative, plain)
    out = _shape(out_power, freqs, fricative, plain)
    plain_shift = float(np.median(centroid_hz(out_power, freqs)[plain]) - np.median(centroid_hz(src_power, freqs)[plain]))
    return {"sib_centroid_hz": (src[0], out[0] - plain_shift), "sib_body_db": (src[1], out[1]), "sib_level_db": (src[2], out[2])}
