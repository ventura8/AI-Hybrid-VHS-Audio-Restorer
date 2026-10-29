"""The hiss floor a restored file keeps under the programme, read against the audio it was restored from.

The diagnosis of 2026-10-11 (77-91 hiss-dominated clips of the IA hunt, the Tata excerpts)
found the Mel-RoFormer denoiser strong wherever a tape's 8-16 kHz band is hiss-dominated and
nearly an identity on programme-dominated material (music, adverts): -27.7 dB under the
programme on the hiss-dominated clips, -1.2 dB on the others. Even where it is strong, in
energy terms it removes much less of the 3-4.8 kHz hiss than cathar does (-4.4 and -7.7 dB
against -8.0 and -9.2 in the two bands that hold about half the floor's energy); a median over
bands hides that, so the floor is read per band and the 3-4.8 kHz bands are summed as energy.

This module reads it the way the diagnosis did (its `measure.py`), on the channel mean:

- 20 ms Hann frames, the energy per ERB band from 3 kHz to 16 kHz (15 bands at 44.1 kHz);
- the programme frames from the source, by a coarse four-band activity detector (a band 6 dB
  over its own 10th percentile on a 60 ms average, widened 80 ms either side; the harness's
  `dsp_vad`);
- the floor under the programme per band, a minimum statistic: for each programme frame, the
  10th percentile of the band over the programme frames within 0.75 s either side (where at
  least half that window is programme), then the median of that over those frames;
- the restored file matched to the source's programme level by the least-squares projection of
  its 100-4000 Hz spectrum onto the source's on the source's loud frames (70th percentile and
  up), which the noise the source carries and the output lost does not bias;
- the source's 8-16 kHz stationarity: per band, the median minus the 10th percentile of the
  frame energy in dB. A band that hiss holds reads 2-4 dB; one programme moves reads more.

Nothing here writes audio or reads the configuration: the stages that act on a reading
(`modules/post_neural_suppress.py`, `modules/neural_level.py`) decide what it means.
"""

import numpy as np
import scipy.ndimage
import soundfile as sf

FRAME_S = 0.02
BLOCK_SAMPLES = 1 << 20
HISS_LO_HZ = 3000.0
HISS_HI_HZ = 16000.0
# The top band stops this far under Nyquist, as the diagnosis's frames did.
NYQUIST_MARGIN_HZ = 100.0
# Glasberg and Moore's ERB-number scale: one band per ERB.
ERB_SCALE = 21.4
ERB_CORNER = 4.37 / 1000.0
# The presence bands: the ERB bands that end at or under 4.8 kHz (3.0-4.7 kHz at 44.1 kHz).
PRESENCE_HI_HZ = 4800.0
# The stationarity reading's bands start here.
STATIONARY_LO_HZ = 8000.0
PROGRAMME_BAND_HZ = (100.0, 4000.0)
LOUD_PERCENTILE = 70.0
VAD_BANDS_HZ = ((100.0, 500.0), (500.0, 2000.0), (2000.0, 5000.0), (5000.0, 12000.0))
VAD_FLOOR_PERCENTILE = 10.0
VAD_MARGIN_DB = 6.0
VAD_SMOOTH_S = 0.06
VAD_HANGOVER_S = 0.08
MINSTAT_HALF_S = 0.75
MINSTAT_PERCENTILE = 10.0
MIN_FLOOR_FRAMES = 10
# A long tape's floor is read on at most this many programme frames, evenly spread: the
# median over them is the same reading, and a two-hour tape stays a few seconds of work.
MAX_FLOOR_POINTS = 20000
# A band is read only where the source stands this far over 16-bit TPDF dither (variance
# LSB^2 / 4, times the Hann window's sum of squares per bin): its floor under the programme for
# the deltas, its 10th percentile for the stationarity. Under it the band carries nothing a
# capture could hold, such as the empty top of a source resampled from a lower rate (one hunt
# clip's 14.3-16 kHz band reads -96 dB on the extracted source and -67 dB after
# pre-conditioning; counted, its 17 dB spread moved that tape's 8-16 kHz reading from 2.5 to 4.6).
QUANT_LSB = 1.0 / 32768.0
DITHER_MARGIN_DB = 10.0
# Under this correlation on the loud frames the restored programme does not follow the
# source's (a misaligned or unrelated file), and no level match is read from it.
MIN_CORRELATION = 0.5
EPS = 1e-20


def erb_band_edges(f_lo, f_hi):
    """Band edges in Hz, one ERB apart, from `f_lo` to `f_hi` (the last band ends at `f_hi`)."""
    lo = ERB_SCALE * np.log10(ERB_CORNER * f_lo + 1.0)
    hi = ERB_SCALE * np.log10(ERB_CORNER * f_hi + 1.0)
    count = max(1, int(np.floor(hi - lo)))
    return (10.0 ** (np.linspace(lo, hi, count + 1) / ERB_SCALE) - 1.0) / ERB_CORNER


def db(power):
    """Power in dB, floored so silence reads as a number."""
    return 10.0 * np.log10(np.asarray(power, dtype=np.float64) + EPS)


def _layout(rate, frame):
    """The bin-to-band matrices for one frame size: the ERB hiss bands, the detector's bands, the programme band."""
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    edges = erb_band_edges(HISS_LO_HZ, min(HISS_HI_HZ, rate / 2.0 - NYQUIST_MARGIN_HZ))
    index = np.searchsorted(edges, freqs, side="right") - 1
    index[(freqs < edges[0]) | (freqs >= edges[-1])] = -1
    hiss = (index[:, None] == np.arange(len(edges) - 1)[None, :]).astype(np.float64)
    vad = np.stack([(freqs >= lo) & (freqs < hi) for lo, hi in VAD_BANDS_HZ], axis=1).astype(np.float64)
    programme = (freqs >= PROGRAMME_BAND_HZ[0]) & (freqs < PROGRAMME_BAND_HZ[1])
    return {"edges": edges, "hiss": hiss, "vad": vad, "programme": programme, "window": np.hanning(frame)}


def _mono_frames(handle, frame):
    """The file's channel mean as blocks of whole `frame`-sample frames, streamed."""
    carry = np.zeros(0)
    for block in handle.blocks(blocksize=BLOCK_SAMPLES, dtype="float32", always_2d=True):
        mono = np.concatenate((carry, block.mean(axis=1).astype(np.float64)))
        usable = (len(mono) // frame) * frame
        yield mono[:usable].reshape(-1, frame)
        carry = mono[usable:]


def _frame_stats(source, restored, layout):
    """One block's per-frame readings: both files' band energies, the source's detector bands and level, the projection terms."""
    count = min(len(source), len(restored))
    src = np.fft.rfft(source[:count] * layout["window"], axis=1)
    out = np.fft.rfft(restored[:count] * layout["window"], axis=1)
    src_power, out_power = np.abs(src) ** 2, np.abs(out) ** 2
    band = layout["programme"]
    return {
        "src_bands": src_power @ layout["hiss"],
        "out_bands": out_power @ layout["hiss"],
        "vad": src_power @ layout["vad"],
        "level": np.sqrt(np.mean(source[:count] ** 2, axis=1)),
        "cross": np.sum((out[:, band] * np.conj(src[:, band])).real, axis=1),
        "out_power": out_power[:, band].sum(axis=1),
        "src_power": src_power[:, band].sum(axis=1),
    }


def _joined(parts):
    """The blocks' readings as one array per reading."""
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def frame_pair(source_wav, restored_wav):
    """Per-frame readings of the two files over their common length, with the layout; or a string saying why there are none."""
    with sf.SoundFile(str(source_wav)) as source, sf.SoundFile(str(restored_wav)) as restored:
        if source.samplerate != restored.samplerate:
            return "the restored audio's rate differs from the source's"
        rate = source.samplerate
        frame = max(1, int(FRAME_S * rate))
        layout = _layout(rate, frame)
        pairs = zip(_mono_frames(source, frame), _mono_frames(restored, frame))
        parts = [_frame_stats(src, out, layout) for src, out in pairs]
    stats = _joined(parts) if parts else {"level": np.zeros(0)}
    if len(stats["level"]) < MIN_FLOOR_FRAMES:
        return "too short to read"
    return {**stats, "rate": rate, "frame": frame, "layout": layout}


def programme_match(stats):
    """`(gain, correlation)`: the amplitude gain that puts the restored programme at the source's level, and their match.

    Least squares on the source's loud frames in 100-4000 Hz: the source is `gain` times the
    restored programme plus what the restoration took out, which is uncorrelated with it.
    """
    loud = stats["level"] >= np.percentile(stats["level"], LOUD_PERCENTILE)
    cross = float(np.sum(stats["cross"][loud]))
    out_power = float(np.sum(stats["out_power"][loud]))
    src_power = float(np.sum(stats["src_power"][loud]))
    gain = cross / out_power if out_power > EPS else 0.0
    correlation = cross / np.sqrt(out_power * src_power) if out_power * src_power > EPS else 0.0
    return gain, float(correlation)


def programme_frames(vad_power, rate, frame):
    """The frames the source carries programme in: any detector band 6 dB over its own p10 on a 60 ms average, widened."""
    smooth = max(1, int(round(VAD_SMOOTH_S * rate / frame)))
    averaged = scipy.ndimage.uniform_filter1d(vad_power, size=smooth, axis=0, mode="nearest")
    level = db(averaged)
    active = np.any(level > np.percentile(level, VAD_FLOOR_PERCENTILE, axis=0) + VAD_MARGIN_DB, axis=1)
    reach = max(0, int(round(VAD_HANGOVER_S * rate / frame)))
    return scipy.ndimage.maximum_filter1d(active.astype(np.uint8), size=2 * reach + 1) > 0


def _window_floor(bands, speech, cumulative, frame_index, half):
    """The p10 of every band over the programme frames around one frame, or None where under half the window is programme."""
    lo, hi = max(0, frame_index - half), min(len(speech), frame_index + half + 1)
    if cumulative[hi] - cumulative[lo] < 0.5 * (hi - lo):
        return None
    return np.percentile(bands[lo:hi][speech[lo:hi]], MINSTAT_PERCENTILE, axis=0)


def programme_floor(bands, speech):
    """The minimum-statistic floor of every column of `bands` under the programme, or None with too few frames to read."""
    half = int(round(MINSTAT_HALF_S / FRAME_S))
    cumulative = np.concatenate(([0], np.cumsum(speech.astype(np.int64))))
    points = np.flatnonzero(speech)
    points = points[:: max(1, len(points) // MAX_FLOOR_POINTS)]
    floors = [floor for floor in (_window_floor(bands, speech, cumulative, int(t), half) for t in points) if floor is not None]
    if len(floors) < MIN_FLOOR_FRAMES:
        return None
    return np.median(np.stack(floors), axis=0)


def readable_db(layout):
    """Per band, the level the source must stand over to be read: 16-bit dither in the band plus DITHER_MARGIN_DB."""
    dither = (QUANT_LSB**2 / 4.0) * float(np.sum(layout["window"] ** 2))
    return db(dither * layout["hiss"].sum(axis=0)) + DITHER_MARGIN_DB


def floor_gap_db(src_bands, layout):
    """The source's 8-16 kHz stationarity: the median over its readable bands of (median minus p10) of the frame energy, in dB."""
    levels = db(src_bands)
    low = np.percentile(levels, 10.0, axis=0)
    gap = np.median(levels, axis=0) - low
    keep = (layout["edges"][:-1] >= STATIONARY_LO_HZ) & (low > readable_db(layout))
    return float(np.median(gap[keep])) if keep.any() else None


def _energy_delta(src_db, out_db):
    """The output's energy over the source's across a set of bands, in dB; None for an empty set."""
    if len(src_db) == 0:
        return None
    return float(db(np.sum(10.0 ** (out_db / 10.0))) - db(np.sum(10.0 ** (src_db / 10.0))))


def _band_deltas(floor, stats):
    """The restored floor over the source's per band, the median over the readable bands and the presence bands' energy."""
    layout = stats["layout"]
    count = layout["hiss"].shape[1]
    src_db, out_db = db(floor[:count]), db(floor[count:])
    valid = src_db > readable_db(layout)
    presence = valid & (layout["edges"][1:] <= PRESENCE_HI_HZ)
    delta = out_db - src_db
    return {
        "band_delta_db": [float(value) if ok else None for value, ok in zip(delta, valid)],
        "median_delta_db": float(np.median(delta[valid])) if valid.any() else None,
        "presence_delta_db": _energy_delta(src_db[presence], out_db[presence]),
    }


def _matched(stats):
    """The level match, or the reason there is none."""
    gain, correlation = programme_match(stats)
    if correlation < MIN_CORRELATION or gain <= 0.0:
        return None, f"the restored programme does not follow the source's (r {correlation:.2f})"
    return (gain, correlation), None


def read(source_wav, restored_wav):
    """The hiss floor the restored file keeps under the programme against the source's, as a dict of readings.

    `floor_gap_db` is the source's 8-16 kHz stationarity; `band_delta_db` the restored floor
    over the source's per ERB band (None where the source's floor is too near dither to
    read); `median_delta_db` their median; `presence_delta_db` the 3-4.8 kHz bands' energy
    delta; `programme_gain_db` and `programme_correlation` the level match. Without a reading
    the dict holds `reason` alone.
    """
    stats = frame_pair(source_wav, restored_wav)
    if isinstance(stats, str):
        return {"reason": stats}
    match, reason = _matched(stats)
    if match is None:
        return {"reason": reason}
    speech = programme_frames(stats["vad"], stats["rate"], stats["frame"])
    floor = programme_floor(np.hstack((stats["src_bands"], stats["out_bands"] * match[0] ** 2)), speech)
    if floor is None:
        return {"reason": "too little programme to read the floor under"}
    return {
        "floor_gap_db": floor_gap_db(stats["src_bands"], stats["layout"]),
        "band_edges_hz": [float(edge) for edge in stats["layout"]["edges"]],
        **_band_deltas(floor, stats),
        "programme_gain_db": float(20.0 * np.log10(match[0])),
        "programme_correlation": match[1],
        "programme_share": float(np.mean(speech)),
    }
