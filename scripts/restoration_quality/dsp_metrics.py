"""Pure numpy/scipy guardrails at the native rate: the things a listener hears that no MOS model reads.

Each function takes float32 mono arrays that are already aligned and gain-matched
(`audio_io.align_pair`), except the loudness readings, which are level-sensitive by design.
"""

import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.stats

from scripts.measure_hum import hum_excess_db as _hum_excess_db
from scripts.measure_tradeoff import LOUD_PERCENTILE, QUIET_PERCENTILE, measure

FRAME = 1024
STFT_FRAME = 1024
STFT_HOP = 256
LKR_BAND_HZ = (1000.0, 8000.0)
LKR_MIN_FRAMES = 20
HF_REFERENCE_HZ = (300.0, 4000.0)
HF_BANDS_HZ = {"hf_4k8k": (4000.0, 8000.0), "hf_8k16k": (8000.0, 16000.0)}
# The CRT line whistle (15625 Hz PAL, 15734 Hz NTSC) can carry most of a VHS capture's
# 8-16 kHz energy; removing it is restoration, not air lost, so every band reading leaves
# these bins out on both sides (the stem octave reading shares the constants).
CRT_LINE_HZ = (15625.0, 15734.0)
CRT_LINE_HALF_WIDTH_HZ = 250.0
CLICK_HIGHPASS_HZ = 4000.0
CLICK_FLOOR_FRAME_S = 0.01
CLICK_FLOOR_PERCENTILE = 90.0
CLICK_RATIO = 12.0
CLICK_REFRACTORY_S = 0.005
# An impulse must also clear this level above 4 kHz: a denoiser that gates a pause to
# -85 dBFS leaves dither-scale residue that stands 60x above its own floor and is not a
# click anyone hears (measured on the APL output of the Tata tapes, -87..-90 dBFS).
CLICK_MIN_LEVEL_DBFS = -60.0
# Oxide dropouts last 5-50 ms (docs/vhs_audio_defects_research.md 2.7); 10 ms frames see them,
# and a run of two keeps a single-frame flicker of a denoiser from counting.
DROPOUT_FRAME_S = 0.01
DROPOUT_DROP_DB = -10.0
DROPOUT_MIN_FRAMES = 2
# A frame carries programme when it sits within 6 dB of the source's loud level (p70).
DROPOUT_PROGRAMME_SHARE = 0.5
WHISTLE_HZ = 15625.0
WHISTLE_CORE_HZ = 30.0
WHISTLE_NEIGHBOURHOOD_HZ = 500.0
LRA_WINDOW_S = 3.0
LRA_HOP_S = 1.0
LRA_ABSOLUTE_GATE = -70.0
LRA_RELATIVE_GATE = -20.0
# Pauses, read on 20 ms frames the source calls quiet: how much the floor jumps about
# (a denoiser's mask opening and closing, heard as the room switching off between words)
# and whether what is left is low rumble or hiss. Measured on the Tata tapes: a source
# pause moves 3.8 dB (sd) and every denoiser 11-23 dB; APL left 100-500 Hz untouched
# while taking 13 dB off 2-8 kHz (heard as dead pauses), cathar the reverse (heard as hiss).
PAUSE_FRAME_S = 0.02
PAUSE_QUIET_PERCENTILE = 15.0
PAUSE_BANDS_HZ = {"lf": (100.0, 500.0), "hf": (2000.0, 8000.0)}


def frame_levels(mono, frame=FRAME):
    """RMS of non-overlapping frames, or an empty array when the signal is shorter than one frame."""
    count = len(mono) // frame
    if count == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(np.asarray(mono[: count * frame], dtype=np.float64).reshape(count, frame) ** 2, axis=1))


def quiet_loud_masks(source, frame=FRAME):
    """Frames the source calls quiet (<= p20) and loud (>= p70): noise is read on the first, programme on the second."""
    level = frame_levels(source, frame)
    if len(level) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
    return level <= np.percentile(level, QUIET_PERCENTILE), level >= np.percentile(level, LOUD_PERCENTILE)


def pause_dynamics(source, output, rate):
    """`{"pause_pumping_db": (source, output), "pause_tilt_db": (0, output)}` on the source's quiet 20 ms frames.

    Pumping is the standard deviation of the pause frames' level in dB. Tilt is the low band's
    residual minus the high band's (each the output over the source in dB): positive leaves
    rumble and takes the air, negative leaves hiss; a uniform reduction reads 0.
    """
    frame = max(1, int(PAUSE_FRAME_S * rate))
    src_level = frame_levels(source, frame)
    out_level = frame_levels(output, frame)[: len(src_level)]
    if len(out_level) < 10:
        return {"pause_pumping_db": (None, None), "pause_tilt_db": (0.0, None)}
    quiet = src_level[: len(out_level)] <= np.percentile(src_level[: len(out_level)], PAUSE_QUIET_PERCENTILE)
    pumping = (_level_spread_db(src_level[: len(out_level)][quiet]), _level_spread_db(out_level[quiet]))
    residual = {}
    for name, (lo, hi) in PAUSE_BANDS_HZ.items():
        residual[name] = _quiet_band_residual_db(source, output, rate, lo, hi, frame, quiet)
    return {"pause_pumping_db": pumping, "pause_tilt_db": (0.0, residual["lf"] - residual["hf"])}


def _level_spread_db(levels):
    return float(np.std(20.0 * np.log10(np.asarray(levels, dtype=np.float64) + 1e-9)))


def _quiet_band_residual_db(source, output, rate, lo, hi, frame, quiet):
    sos = scipy.signal.butter(4, [lo, hi], btype="bandpass", fs=rate, output="sos")
    src = frame_levels(scipy.signal.sosfiltfilt(sos, np.asarray(source, dtype=np.float64)), frame)[: len(quiet)][quiet]
    out = frame_levels(scipy.signal.sosfiltfilt(sos, np.asarray(output, dtype=np.float64)), frame)[: len(quiet)][quiet]
    return float(10.0 * np.log10((np.mean(out**2) + 1e-14) / (np.mean(src**2) + 1e-14)))


def residual_noise_db(source, output):
    """Level of the output in the source's quiet frames minus the source's own, in dB (lower is better)."""
    quiet, _loud = quiet_loud_masks(source)
    if not quiet.any():
        return None
    src = frame_levels(source)[quiet]
    out = frame_levels(output)[: len(quiet)][quiet[: len(frame_levels(output))]]
    if len(out) == 0:
        return None
    return float(20.0 * np.log10((np.mean(out) + 1e-12) / (np.mean(src) + 1e-12)))


def _power_stft(mono, rate):
    """STFT power `(bins, frames)` and its bin frequencies."""
    freqs, _times, spec = scipy.signal.stft(np.asarray(mono, dtype=np.float64), fs=rate, nperseg=STFT_FRAME, noverlap=STFT_FRAME - STFT_HOP)
    return freqs, np.abs(spec) ** 2


def lkr_musical_noise(source, output, rate):
    """Log-kurtosis ratio of the output over the source, per bin, in the source's quiet frames.

    Spectral subtraction leaves isolated tonal islands where the noise was; along time
    each bin then has a heavy-tailed, high-kurtosis distribution. The ratio of the two
    kurtoses is the classic musical-noise measure (Uemura et al.; the LKR-PI variant
    correlates at |rho| 0.95 with listening tests). Positive = more musical noise.
    """
    freqs, src_power = _power_stft(source, rate)
    _freqs, out_power = _power_stft(output, rate)
    frames = min(src_power.shape[1], out_power.shape[1])
    quiet = _quiet_stft_frames(src_power[:, :frames])
    if quiet.sum() < LKR_MIN_FRAMES:
        return None
    band = (freqs >= LKR_BAND_HZ[0]) & (freqs < LKR_BAND_HZ[1])
    src_k = scipy.stats.kurtosis(src_power[band][:, quiet], axis=1, fisher=False)
    out_k = scipy.stats.kurtosis(out_power[band][:, quiet], axis=1, fisher=False)
    return float(np.median(np.log(out_k + 1e-9) - np.log(src_k + 1e-9)))


def _quiet_stft_frames(power):
    """Frames whose total power sits in the quietest fifth."""
    level = power.sum(axis=0)
    return level <= np.percentile(level, QUIET_PERCENTILE)


def hf_ratio_db(mono, rate, band_hz):
    """Energy in `band_hz` relative to the 300-4000 Hz speech body, in dB; a drop means muffled."""
    freqs, psd = scipy.signal.welch(np.asarray(mono, dtype=np.float64), rate, nperseg=4096)
    return _band_ratio_db(freqs, psd, band_hz)


def _band_ratio_db(freqs, psd, band_hz):
    band = psd[band_bins(freqs, band_hz)].sum()
    body = psd[(freqs >= HF_REFERENCE_HZ[0]) & (freqs < HF_REFERENCE_HZ[1])].sum()
    return float(10.0 * np.log10((band + 1e-20) / (body + 1e-20)))


def band_bins(freqs, band_hz):
    """Bins inside `band_hz` with the CRT line and its +-250 Hz skirt left out."""
    bins = (freqs >= band_hz[0]) & (freqs < band_hz[1])
    for line in CRT_LINE_HZ:
        bins &= np.abs(freqs - line) > CRT_LINE_HALF_WIDTH_HZ
    return bins


def loud_band_ratios_db(source, output, rate, frame=4096):
    """`{band: (source_ratio, output_ratio)}` read on the frames where the source is loud.

    Pauses would confound the reading: a denoiser that empties the pauses lowers the
    high band there without touching the voice. Only the frames carrying programme say
    whether the speech itself lost its highs.
    """
    count = min(len(source), len(output)) // frame
    if count == 0:
        return {name: (None, None) for name in HF_BANDS_HZ}
    freqs, src_psd, level = framed_psd(source[: count * frame], rate, frame)
    _freqs, out_psd, _level = framed_psd(output[: count * frame], rate, frame)
    loud = level >= np.percentile(level, LOUD_PERCENTILE)
    src_mean, out_mean = np.mean(src_psd[loud], axis=0), np.mean(out_psd[loud], axis=0)
    return {name: (_band_ratio_db(freqs, src_mean, band), _band_ratio_db(freqs, out_mean, band)) for name, band in HF_BANDS_HZ.items()}


def framed_psd(mono, rate, frame):
    """Per non-overlapping frame: `(freqs, power (frames, bins), rms level (frames,))`, Hann-windowed rfft power."""
    count = len(mono) // frame
    frames = np.asarray(mono[: count * frame], dtype=np.float64).reshape(count, frame)
    level = np.sqrt(np.mean(frames**2, axis=1))
    power = np.abs(np.fft.rfft(frames * np.hanning(frame), axis=1)) ** 2
    return np.fft.rfftfreq(frame, 1.0 / rate), power, level


def click_density(mono, rate):
    """Impulses per second: the second difference standing far above its own 50 ms rolling median.

    A click is a discontinuity, so it dominates the second difference (a crude high-order
    derivative) while voiced speech, however bright, stays smooth from sample to sample.
    The floor is relative to the surrounding 50 ms and absolute (`CLICK_MIN_LEVEL_DBFS`),
    so residue in a gated pause does not count.
    """
    sos = scipy.signal.butter(4, CLICK_HIGHPASS_HZ, btype="highpass", fs=rate, output="sos")
    curvature = np.abs(np.diff(scipy.signal.sosfiltfilt(sos, np.asarray(mono, dtype=np.float64)), 2))
    hits = (curvature > CLICK_RATIO * (_local_floor(curvature, rate) + 1e-9)) & (curvature > 10 ** (CLICK_MIN_LEVEL_DBFS / 20.0))
    events = _count_events(hits, int(CLICK_REFRACTORY_S * rate))
    return float(events / (len(mono) / float(rate)))


def _local_floor(curvature, rate):
    """The 90th percentile of each 10 ms frame, held at the maximum of its +-2 neighbours, spread back to samples.

    A percentile rather than a median so a speech onset next to a pause is judged against
    the speech it starts, not the silence it ends.
    """
    frame = max(1, int(CLICK_FLOOR_FRAME_S * rate))
    count = max(1, int(np.ceil(len(curvature) / frame)))
    padded = np.concatenate([curvature, np.zeros(count * frame - len(curvature))])
    per_frame = np.percentile(padded.reshape(count, frame), CLICK_FLOOR_PERCENTILE, axis=1)
    held = scipy.ndimage.maximum_filter1d(per_frame, size=5, mode="nearest")
    return np.repeat(held, frame)[: len(curvature)]


def _count_events(hits, refractory):
    """Distinct events among `hits`, merging any that fall inside the refractory span."""
    positions = np.flatnonzero(hits)
    if len(positions) == 0:
        return 0
    gaps = np.diff(positions)
    return int(1 + np.count_nonzero(gaps > refractory))


def dropout_count(source, output, rate):
    """New holes: runs of >= 2 frames (100 ms) where the output fell > 10 dB under a source frame that carried programme."""
    frame = int(DROPOUT_FRAME_S * rate)
    src = frame_levels(source, frame)
    out = frame_levels(output, frame)
    count = min(len(src), len(out))
    if count == 0:
        return 0
    src, out = src[:count], out[:count]
    drop = 20.0 * np.log10((out + 1e-9) / (src + 1e-9))
    programme = src >= DROPOUT_PROGRAMME_SHARE * np.percentile(src, LOUD_PERCENTILE)
    return _count_runs((drop < DROPOUT_DROP_DB) & programme, DROPOUT_MIN_FRAMES)


def _count_runs(mask, min_length):
    """Runs of consecutive True values at least `min_length` long."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    lengths = edges[1::2] - edges[0::2]
    return int(np.count_nonzero(lengths >= min_length))


def whistle_line_db(mono, rate, line_hz=WHISTLE_HZ):
    """The CRT line whistle: peak within +-30 Hz of `line_hz` over the median of its +-500 Hz neighbourhood, in dB."""
    if len(mono) < 32768 or line_hz >= rate / 2.0:
        return None
    freqs, psd = scipy.signal.welch(np.asarray(mono, dtype=np.float64), rate, nperseg=32768)
    core = np.abs(freqs - line_hz) <= WHISTLE_CORE_HZ
    around = (np.abs(freqs - line_hz) <= WHISTLE_NEIGHBOURHOOD_HZ) & ~core
    return float(10.0 * np.log10((psd[core].max() + 1e-20) / (np.median(psd[around]) + 1e-20)))


def hum_excess_db(mono, rate, mains_hz=50.0):
    """Mains harmonics above their own neighbourhood, from `scripts.measure_hum`."""
    return float(_hum_excess_db(np.asarray(mono, dtype=np.float64), rate, mains_hz))


def loudness(mono, rate):
    """Integrated loudness (LUFS) and loudness range (LU, EBU Tech 3342) of the raw, un-matched signal."""
    import pyloudnorm

    meter = pyloudnorm.Meter(rate)
    data = np.asarray(mono, dtype=np.float64)
    integrated = float(meter.integrated_loudness(data))
    return integrated, _loudness_range(meter, data, rate)


def _loudness_range(meter, data, rate):
    """LRA: p95 - p10 of gated 3 s short-term loudness, or 0 when the file is too short."""
    window, hop = int(LRA_WINDOW_S * rate), int(LRA_HOP_S * rate)
    if len(data) < window:
        return 0.0
    starts = range(0, len(data) - window + 1, hop)
    short_term = np.array([meter.integrated_loudness(data[start:][:window]) for start in starts])
    short_term = short_term[np.isfinite(short_term) & (short_term > LRA_ABSOLUTE_GATE)]
    if len(short_term) == 0:
        return 0.0
    gated = short_term[short_term > _power_mean_db(short_term) + LRA_RELATIVE_GATE]
    return float(np.percentile(gated, 95) - np.percentile(gated, 10)) if len(gated) else 0.0


def _power_mean_db(levels_db):
    """Mean of levels taken in the power domain, back in dB."""
    return float(10.0 * np.log10(np.mean(10.0 ** (levels_db / 10.0))))


def trade(source_wav, output_wav):
    """The branch's arbiter, unchanged: noise removed and programme deviation (`scripts.measure_tradeoff.measure`)."""
    return measure(str(source_wav), str(output_wav))
