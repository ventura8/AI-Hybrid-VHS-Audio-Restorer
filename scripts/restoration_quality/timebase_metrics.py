"""R12, the time base: whether the output plays the source frame for frame (backlog B1).

The app's DTW sync (`modules/sync.py`, taken on every clip the scanner reads `has_drift`)
interpolates the warp linearly between whole chroma frames (25 per second, `dtw_resolution`)
and smooths it over 51 samples, so every off-diagonal step of its path plays a 40 ms frame at
0x (a hold: the output sits at one sample's value) or at 2x (80 ms of programme in 40 ms).
Both engines' chains preserve length and share the source's time base, so a shift sync leaves
the output sample-synchronous and any warp is error. Nothing in the ear read it: the sync
reading (`file_metrics.sync_drift`) reads three 8 s anchors, and `dsp_metrics.dropout_count`
misses most holds (a frozen sample keeps its level).

Every reading is one value per file, read on the pair at the runner's alignment lag
(`runner.timebase_entry`: the raw pair, not the gain-matched one, see `peak_readings`), and
none depends on the gain, so a pure lag or a gain change reads nothing:

Holds (`file.timebase_holds`, `file.timebase_holds_per_min`)
    The output's 300 Hz high-passed level on 5 ms frames against the source's, both read in
    60 s chunks (`dsp_metrics.chunk_levels`). A frame dips when the output-minus-source level
    falls 25 dB under its own 205 ms running median (a gain or a denoiser's steady depth is no
    dip), where the source stands within 50 dB of its p99 frame and does not dip 20 dB under
    its own 205 ms running median. A hold is a run of 20 ms or more of such frames over which
    the output sat at one value: its mean 10 dB over its spread (`frozen`), the value within
    50 dB of the output's p99 frame. A muted hole holds no value (an engine's dropout stays
    with `file.dropouts`), and digital silence holds only the loader's DC residue (about
    -99 dBFS): neither is a hold.
Lag track (`file.timebase_plateaus`, `file.timebase_rate_jumps`, `file.timebase_tracked_share`)
    Both sides band-passed to 200-4000 Hz and decimated to about 11 kHz; every 10 ms, a 60 ms
    source window's Pearson correlation with the output over +-100 ms (batched FFTs), its peak
    placed by a parabola. A frame is confident when the source window stands within 50 dB of
    the p99 window, the peak reads 0.85 or more, beats the best peak 2 ms or further from it by
    0.1 (harmonic music correlates almost as well one pitch period away), and is off the
    search's edge. `tracked_share` is the confident share of the programme frames: the coverage
    the two counts stand on. A plateau is two or more confident frames on one whole DTW frame
    (40 or 80 ms, within 2 ms) off the local lag, unread stretches of up to 0.5 s inside it
    allowed; the local lag is the running median of 0.5 s block medians over nine blocks, so a
    slow drift is no plateau, and a step that never comes back is a jump, not a plateau. A rate
    jump is a pair of confident frames at most 200 ms apart whose lags, through a five-frame
    median, differ by one or two whole DTW frames; consecutive pairs make one (`count_jumps`).
Speed (`file.timebase_speed_dev_pct`, shown, not gated)
    The 0.5 s block medians of the confident lags, differenced over 1 s: the local speed error
    in percent, p95 of its magnitude. A constant error reads itself (0.3 % slow: 0.3); a
    sinusoidal one at f Hz its depth times about sin(pi f) / (pi f) (0.2 Hz: 0.935); a 300 ms
    plateau reads 0 or 4 % as it fills half a block or not, and a programme repeating 96 ms
    apart read 9.6 % on one shift clip.

The track readings are None when the pair is shorter than one search span (260 ms) or no
frame is confident (a steady periodic tone; an output of the other polarity); the holds are
always read. The gates (`gates.py`, `timebase.holds`, `timebase.rate_jumps`) start at none.

Measured (2026-10-12; the hunt's stored outputs, `experiments/tune_ia_hunt_v3`, cathar and
`auto_pure_linear` baselines, the 30 clips the scanner sent to DTW and all 205 it sent to
shift; read through the harness's own loader and alignment):

- DTW: cathar 177 holds on 24 clips (median 7 a clip where any), 64 plateaus on 17, 82 rate
  jumps on 16; `auto_pure_linear` 49 holds on 15, 35 plateaus on 13, 54 jumps on 12; the gates
  fail 24 and 16 of the 30. Four clips per engine track no frame (two BBC iPlayer captures,
  `bbcnewsatten07062019172904`, `ein-halleluja-fur-spirito-sant`), cathar's tracked share has
  a median of 0.35 against APL's 0.81 (cathar's dewind and subtraction decorrelate the
  waveform). The holds stand 42.5 dB (median; p10 25.7) of raw level over their high-passed
  level, 94 % of them 20 dB or more: frozen samples, not mutes.
- shift, 205 per engine: no hold, plateau or rate jump; the speed reading's p95 is 0.007 %
  (cathar) and 0.0009 % (APL) in the median.
- the 32 accepted v2 finals (`D:/Tata/New folder/variants/v2`): nothing, but one hold on the APL
  final of `x_videoplayback-20260504-2342`, the one music clip the scanner sent to DTW; round
  two heard all twelve music clips as fine, unblinded.
- the en / fr calibration targets: the benign cases (identity, requantise, resample round
  trip, shifts of 5 / 30 / 60 ms) read nothing; `sync_holds` (1 / 3 / 6 glitches) reads holds
  1 / 3 / 6 and plateaus 1 / 3 / 6, jumps 1 / 5 / 10 (fr 2 / 6 / 10); `sync_drift` (0.05 /
  0.1 / 0.2 % slow) reads the speed 0.056 / 0.108 / 0.217 % (fr 0.058 / 0.116 / 0.239) and no
  event.
- the gain-matched pair would have blinded it: the least-squares gain of the 0.1 % slow en
  fixture came out at -0.016 (no confident frame), and of the hunt's APL output of
  `frequently-asked-questions-about-time-travel` negative too (no track; 10 holds, 9 plateaus
  and 13 jumps at its own polarity).
- cost: about 0.25 s per 15 s pair and 9 s on the 7.7 min Vaccin pair, the lag track nearly
  all of it (the FFTs run on every core, `FFT_WORKERS`).
"""

import warnings

import numpy as np
import scipy.fft
import scipy.ndimage
import scipy.signal

from scripts.restoration_quality import dsp_metrics

# Holds.
HOLD_FRAME_S = 0.005
HOLD_HIGHPASS_HZ = 300.0
HOLD_DROP_DB = 25.0
HOLD_MIN_S = 0.020
HOLD_CONTEXT_FRAMES = 41
HOLD_SOURCE_RANGE_DB = 50.0
HOLD_SOURCE_DIP_DB = 20.0
HOLD_FROZEN_DB = 10.0
HOLD_VALUE_RANGE_DB = 50.0
LOUD_PERCENTILE = 99.0
LEVEL_EPS = 1e-12

# The lag track.
TRACK_RATE = 11025
TRACK_BAND_HZ = (200.0, 4000.0)
TRACK_WINDOW_S = 0.06
TRACK_HOP_S = 0.01
TRACK_REACH_S = 0.10
TRACK_CHUNK_FRAMES = 6000
TRACK_BATCH = 512
TRACK_CONTEXT_S = 0.5
TRACK_RANGE_DB = 50.0
MIN_CORR = 0.85
MIN_MARGIN = 0.1
PEAK_EXCLUSION_S = 0.002
# Every core for the batched FFTs (scipy.fft's pocketfft threads).
FFT_WORKERS = -1

# Plateaus, rate jumps and speed.
STEP_MS = 40.0
MAX_STEPS = 2
STEP_TOLERANCE_MS = 2.0
PLATEAU_MIN_FRAMES = 2
RUN_GAP_FRAMES = 50
JUMP_GAP_S = 0.2
JUMP_MEDIAN_FRAMES = 5
BLOCK_S = 0.5
BLOCK_MIN_FRAMES = 10
BASELINE_BLOCKS = 9
SPEED_SPAN_BLOCKS = 2
SPEED_PERCENTILE = 95.0

READINGS = ("holds", "holds_per_min", "plateaus", "rate_jumps", "speed_dev_pct", "tracked_share")
PREFIX = "file.timebase_"


# ----------------------------------------------------------------------------- holds


def _db(rms):
    return 20.0 * np.log10(np.asarray(rms, dtype=np.float64) + LEVEL_EPS)


def hold_frame(rate):
    """The hold detector's 5 ms frame in samples."""
    return max(1, int(round(HOLD_FRAME_S * rate)))


def hold_levels(source, output, rate):
    """`(source, output)` 300 Hz high-passed 5 ms frame levels in dB, read 60 s at a time."""
    frame = hold_frame(rate)
    sos = scipy.signal.butter(4, HOLD_HIGHPASS_HZ, btype="highpass", fs=rate, output="sos")
    count = min(len(source), len(output)) // frame * frame
    return _db(dsp_metrics.chunk_levels(source[:count], rate, frame, sos)), _db(dsp_metrics.chunk_levels(output[:count], rate, frame, sos))


def dipped_frames(src_db, out_db):
    """Frames where the output fell 25 dB under its local level against a source that carries programme and does not dip."""
    if src_db.size == 0:
        return np.zeros(0, dtype=bool)
    difference = out_db - src_db
    local = scipy.ndimage.median_filter(difference, HOLD_CONTEXT_FRAMES, mode="nearest")
    source_local = scipy.ndimage.median_filter(src_db, HOLD_CONTEXT_FRAMES, mode="nearest")
    carried = src_db > np.percentile(src_db, LOUD_PERCENTILE) - HOLD_SOURCE_RANGE_DB
    return (difference < local - HOLD_DROP_DB) & carried & (src_db > source_local - HOLD_SOURCE_DIP_DB)


def runs(mask, min_length):
    """`(start, stop)` of each run of True at least `min_length` long."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    starts, stops = edges[0::2], edges[1::2]
    keep = stops - starts >= min_length
    return list(zip(starts[keep].tolist(), stops[keep].tolist()))


def frozen(segment, loud_db):
    """Whether the output sat at one value over `segment`: its mean 10 dB over its spread, within 50 dB of the loud level.

    A muted hole holds no value and digital silence holds only the residue of the loader's DC
    removal (about -99 dBFS on the hunt's outputs), so neither is a hold.
    """
    samples = np.asarray(segment, dtype=np.float64)
    held, spread = abs(float(np.mean(samples))), float(np.std(samples))
    return bool(held >= spread * 10.0 ** (HOLD_FROZEN_DB / 20.0) and _db(held) >= loud_db - HOLD_VALUE_RANGE_DB)


def holds(source, output, rate):
    """The output's frozen holds the source does not have: `[(start_s, length_s), ...]` on the pair at its lag."""
    src_db, out_db = hold_levels(source, output, rate)
    if out_db.size == 0:
        return []
    spans = runs(dipped_frames(src_db, out_db), int(round(HOLD_MIN_S / HOLD_FRAME_S)))
    held = frozen_spans(output, spans, hold_frame(rate), float(np.percentile(out_db, LOUD_PERCENTILE)))
    return [(lo / float(rate), (hi - lo) / float(rate)) for lo, hi in held]


def frozen_spans(output, spans, frame, loud_db):
    """The dip runs `spans` (in `frame`-sample frames) over which the output sat at one value, as sample spans."""
    samples = [(start * frame, stop * frame) for start, stop in spans]
    return [(lo, hi) for lo, hi in samples if frozen(output[lo:hi], loud_db)]


# ----------------------------------------------------------------------------- the lag track


def decimation(rate):
    """The integer factor that brings `rate` nearest 11 kHz from above (1 at or under it)."""
    return max(1, int(rate // TRACK_RATE))


def track_sizes(track_rate):
    """`(window, hop, reach)` of the lag track in samples at `track_rate`."""
    return tuple(max(1, int(round(seconds * track_rate))) for seconds in (TRACK_WINDOW_S, TRACK_HOP_S, TRACK_REACH_S))


def band_slice(mono, rate, factor, span):
    """`mono[span]` band-passed zero-phase (0.5 s of context either side) and kept every `factor`-th sample.

    `span` starts on a multiple of `factor`, so the slice sits on the whole file's decimated grid.
    """
    first, last = span
    pad = factor * int(np.ceil(TRACK_CONTEXT_S * rate / factor))
    lo, hi = max(0, first - pad), min(len(mono), last + pad)
    band = (TRACK_BAND_HZ[0], min(TRACK_BAND_HZ[1], 0.45 * rate))
    sos = scipy.signal.butter(4, band, btype="bandpass", fs=rate, output="sos")
    data = scipy.signal.sosfiltfilt(sos, np.asarray(mono[lo:hi], dtype=np.float64))
    begin, end = first - lo, last - lo
    return data[begin:end:factor]


def window_moments(signal, window):
    """The sum and the energy of every `window`-long stretch of `signal`, one per start (`len - window + 1` each)."""
    data = np.asarray(signal, dtype=np.float64)
    sums = np.concatenate(([0.0], np.cumsum(data)))
    squares = np.concatenate(([0.0], np.cumsum(data * data)))
    return sums[window:] - sums[:-window], squares[window:] - squares[:-window]


def normalised_correlation(src_frames, out_segments, out_moments):
    """Per row, the Pearson correlation of a source window with each same-length stretch of its output segment.

    `out_moments` holds each stretch's sum and energy (`window_moments` of the output, gathered
    per row), so the output's local mean comes out of the product without a centred copy.
    """
    window = src_frames.shape[1]
    local, energy = out_moments
    centred = src_frames - src_frames.mean(axis=1, keepdims=True)
    size = scipy.fft.next_fast_len(out_segments.shape[1] + window, real=True)
    spectrum = scipy.fft.rfft(out_segments, size, axis=1, workers=FFT_WORKERS) * np.conj(
        scipy.fft.rfft(centred, size, axis=1, workers=FFT_WORKERS)
    )
    dots = np.asarray(scipy.fft.irfft(spectrum, size, axis=1, workers=FFT_WORKERS))[:, : local.shape[1]]
    variance = np.maximum(energy - local**2 / window, 0.0)
    return dots / np.sqrt(np.sum(centred**2, axis=1, keepdims=True) * variance + LEVEL_EPS)


def peak_readings(correlation, reach, exclusion):
    """Per row: the peak's lag in samples (parabola), the peak, the best peak `exclusion` or more away, and whether it sits on an edge.

    Signed: read as a magnitude, a periodic stretch also peaks half a period away (one 15 s hunt
    clip then read a 9.6 % speed error and a rate jump), so an output of the other polarity reads
    no confident frame. The runner passes the raw pair at its lag for that reason
    (`runner.timebase_entry`).
    """
    rows = np.arange(len(correlation))
    best = np.argmax(correlation, axis=1)
    peak = correlation[rows, best]
    far = np.abs(np.arange(correlation.shape[1])[None, :] - best[:, None]) > exclusion
    second = np.max(np.where(far, correlation, -np.inf), axis=1)
    edge = (best == 0) | (best == correlation.shape[1] - 1)
    inner = np.clip(best, 1, correlation.shape[1] - 2)
    before, centre, after = (correlation[rows, inner + step] for step in (-1, 0, 1))
    curvature = before - 2.0 * centre + after
    offset = np.where(~edge & (curvature < 0.0), 0.5 * (before - after) / np.where(curvature < 0.0, curvature, -1.0), 0.0)
    return best - reach + offset, peak, second, edge


def _batch_track(src, out, starts, sizes):
    """The track's columns for frames at `starts` (decimated samples, relative to the slices): lag, peak, second, edge, energy.

    `out` is the output slice and its `window_moments`: `(samples, (sums, energies))`.
    """
    window, _hop, reach = sizes
    samples, moments = out
    src_frames = np.lib.stride_tricks.sliding_window_view(src, window)[starts]
    views = (
        np.lib.stride_tricks.sliding_window_view(samples, window + 2 * reach)[starts - reach],
        tuple(np.lib.stride_tricks.sliding_window_view(moment, 2 * reach + 1)[starts - reach] for moment in moments),
    )
    readings = peak_readings(normalised_correlation(src_frames, *views), reach, _exclusion(sizes))
    energy = np.mean((src_frames - src_frames.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return np.column_stack([*readings[:3], readings[3].astype(np.float64), energy])


def _exclusion(sizes):
    """The 2 ms exclusion round a peak in track samples (the reach is 100 ms of them)."""
    return max(1, int(round(sizes[2] * PEAK_EXCLUSION_S / TRACK_REACH_S)))


def _chunk_track(source, output, rate, starts, sizes):
    """The track over frames at `starts` (decimated samples on the whole file's grid), sliced and filtered once."""
    factor = decimation(rate)
    window, _hop, reach = sizes
    first, last = int(starts[0]) - reach, int(starts[-1]) + window + reach
    src = band_slice(source, rate, factor, (first * factor, last * factor))
    out = band_slice(output, rate, factor, (first * factor, last * factor))
    out = (out, window_moments(out, window))
    local = np.asarray(starts) - first
    pieces = [_batch_track(src, out, batch, sizes) for batch in np.split(local, np.arange(TRACK_BATCH, len(local), TRACK_BATCH))]
    return np.concatenate(pieces)


def lag_track(source, output, rate):
    """The dense lag track: `{"times_s", "lag_ms", "peak", "second", "edge", "energy"}`, one entry per 10 ms frame.

    `lag_ms` is how late the output plays a source window (positive: late); `times_s` the window
    centre on the source's timeline. Empty arrays when the pair is shorter than one search span.
    """
    factor = decimation(rate)
    track_rate = rate / float(factor)
    sizes = track_sizes(track_rate)
    window, hop, reach = sizes
    count = min(len(source), len(output)) // factor
    starts = np.arange(reach, count - window - reach + 1, hop)
    chunks = np.split(starts, np.arange(TRACK_CHUNK_FRAMES, len(starts), TRACK_CHUNK_FRAMES)) if starts.size else []
    columns = [_chunk_track(source, output, rate, chunk, sizes) for chunk in chunks]
    table = np.concatenate(columns) if columns else np.zeros((0, 5))
    return {
        "times_s": (starts + window / 2.0) / track_rate,
        "lag_ms": table[:, 0] / track_rate * 1000.0,
        "peak": table[:, 1],
        "second": table[:, 2],
        "edge": table[:, 3] > 0.5,
        "energy": table[:, 4],
    }


def programme(track):
    """The frames whose source window stands within 50 dB of the p99 window: where a lag can be read at all."""
    energy = track["energy"]
    if energy.size == 0:
        return np.zeros(0, dtype=bool)
    loud = np.percentile(energy, LOUD_PERCENTILE)
    return energy > loud * 10.0 ** (-TRACK_RANGE_DB / 10.0)


def confident(track):
    """Programme frames with a peak of 0.85 or more, 0.1 over the best peak 2 ms away, off the search's edge."""
    clear = (track["peak"] >= MIN_CORR) & (track["peak"] - track["second"] >= MIN_MARGIN) & ~track["edge"]
    return programme(track) & clear


# ----------------------------------------------------------------------------- plateaus, rate jumps, speed


def block_medians(times, values, block_s):
    """`(centres, medians)` per `block_s` block of `times`; NaN for a block holding under ten values."""
    blocks = np.floor(np.asarray(times) / block_s).astype(np.int64)
    count = int(blocks.max()) + 1 if blocks.size else 0
    medians = np.full(count, np.nan)
    order = np.argsort(blocks, kind="stable")
    keys, first = np.unique(blocks[order], return_index=True)
    for key, group in zip(keys, np.split(np.asarray(values)[order], first[1:])):
        if len(group) >= BLOCK_MIN_FRAMES:
            medians[key] = np.median(group)
    return (np.arange(count) + 0.5) * block_s, medians


def rolling_nanmedian(values, width):
    """The centred running median over `width` values, ignoring NaN (NaN where a whole stretch is)."""
    half = width // 2
    padded = np.pad(np.asarray(values, dtype=np.float64), half, constant_values=np.nan)
    stretches = np.lib.stride_tricks.sliding_window_view(padded, width)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stretches, axis=1)


def local_lag(times, lags):
    """The local lag at each of `times`: the running median of the 0.5 s block medians over nine blocks; None with no block read."""
    centres, medians = block_medians(times, lags, BLOCK_S)
    smooth = rolling_nanmedian(medians, BASELINE_BLOCKS)
    valid = np.isfinite(smooth)
    if not valid.any():
        return None
    return np.interp(times, centres[valid], smooth[valid])


def step_index(deviation):
    """Per frame, the whole DTW frames (1 or 2, signed) the deviation sits on within 2 ms, else 0."""
    values = np.asarray(deviation, dtype=np.float64)
    steps = np.rint(values / STEP_MS)
    on_step = (np.abs(values - steps * STEP_MS) <= STEP_TOLERANCE_MS) & (np.abs(steps) >= 1) & (np.abs(steps) <= MAX_STEPS)
    return np.where(on_step, steps, 0.0).astype(np.int64)


def count_plateaus(steps, frames):
    """Runs of one nonzero step held over two or more confident frames (track frame numbers `frames`).

    Unconfident frames between two confident ones on the same step do not break a run while they
    span 0.5 s or less: a pause inside a 300 ms plateau left two runs of one glitch before.
    """
    steps, frames = np.asarray(steps), np.asarray(frames)
    if steps.size == 0:
        return 0
    joined = (np.diff(steps) == 0) & (np.diff(frames) <= RUN_GAP_FRAMES)
    starts = np.flatnonzero(np.concatenate(([True], ~joined)))
    lengths = np.diff(np.append(starts, steps.size))
    return int(np.count_nonzero((steps[starts] != 0) & (lengths >= PLATEAU_MIN_FRAMES)))


def count_jumps(times, lags):
    """Rate jumps: confident frames at most 200 ms apart whose lags differ by one or two whole DTW frames, consecutive pairs as one.

    The lags first pass a five-frame median, so a stray frame or two at a wrong lag (a peak one
    pitch period or one repeat of the programme away) is no jump in and out; and a jump must be
    40 or 80 ms within 2 ms (`step_index`): of the 137 jumps of 5 ms or more on the hunt's DTW
    outputs 136 were, and the one shift clip that read any (96 ms, a sound repeating at the
    start of `potsworth-nick-super-switch-the-nightmirror-pal`, both engines) was not.
    """
    steady = scipy.ndimage.median_filter(np.asarray(lags, dtype=np.float64), JUMP_MEDIAN_FRAMES, mode="nearest")
    close = np.diff(times) <= JUMP_GAP_S + 1e-9
    return len(runs(close & (step_index(np.diff(steady)) != 0), 1))


def speed_dev_pct(times, lags):
    """p95 of the local speed error in percent: the 0.5 s block medians differenced over 1 s; None with no pair of blocks."""
    _centres, medians = block_medians(times, lags, BLOCK_S)
    span = SPEED_SPAN_BLOCKS * BLOCK_S
    speeds = (medians[SPEED_SPAN_BLOCKS:] - medians[:-SPEED_SPAN_BLOCKS]) / span / 10.0
    speeds = speeds[np.isfinite(speeds)]
    return float(np.percentile(np.abs(speeds), SPEED_PERCENTILE)) if speeds.size else None


def track_readings(track):
    """Plateaus, rate jumps, speed error and coverage from a lag track; every one None when no frame is confident."""
    sure = confident(track)
    if not sure.any():
        return dict.fromkeys(("plateaus", "rate_jumps", "speed_dev_pct", "tracked_share"))
    frames = np.flatnonzero(sure)
    times, lags = track["times_s"][sure], track["lag_ms"][sure]
    baseline = local_lag(times, lags)
    plateaus = 0 if baseline is None else count_plateaus(step_index(lags - baseline), frames)
    return {
        "plateaus": float(plateaus),
        "rate_jumps": float(count_jumps(times, lags)),
        "speed_dev_pct": speed_dev_pct(times, lags),
        "tracked_share": float(sure.sum() / max(1, programme(track).sum())),
    }


def timebase_readings(source, output, rate):
    """Every R12 reading of an aligned pair: `{"holds", "holds_per_min", "plateaus", "rate_jumps", "speed_dev_pct", "tracked_share"}`."""
    found = holds(source, output, rate)
    minutes = min(len(source), len(output)) / float(rate) / 60.0
    per_minute = len(found) / minutes if minutes > 0.0 else None
    return {"holds": float(len(found)), "holds_per_min": per_minute, **track_readings(lag_track(source, output, rate))}


def timebase_entries(source, output, rate):
    """The card's `file.timebase_*` entries (`{"source": 0, "output": value, "delta": value}`) from the aligned pair."""
    readings = timebase_readings(source, output, rate)
    return {f"{PREFIX}{name}": {"source": 0.0, "output": readings[name], "delta": readings[name]} for name in READINGS}
