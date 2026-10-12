"""Pop removal for the modes that opt in: detection on a prediction residual, autoregressive fill.

The calibrated fixture set showed the impulse tools in the chain doing almost nothing to pops
the size a tape carries, in the noise a tape carries them in. Measured inside the damaged
samples of crackle drawn at eight pops a second, 6-16 dB under the programme, with real tape
noise 15 dB down: cathar's `decrackle` recovers +1.7 dB at any sensitivity, its `declick`
+0.3 to +1.5 before its threshold starts costing undamaged speech 7 dB, and FFmpeg's
`adeclick` takes minutes per fixture at the settings that would do more.

This stage does the classic thing, in numpy. A short autoregressive model is fitted to each
block of the signal; a pop is where the model's prediction residual stands far above the
block's own robust scale, since a pop is exactly what a short predictor cannot foresee and
a speech onset is not. Each detected span, with a few guard samples either side, is refilled
by least-squares autoregressive interpolation on a model fitted to the audio around it --
Janssen's method, one pass. Measured at the true pop positions of the calibrated crackle
class it repairs +5.5 dB against `decrackle`'s +1.7, detects 98% of the pops, and on
undamaged speech introduces a change 40 dB under the programme. It runs before `decrackle`,
which then handles the fine residue it was built for, and it is gated on the same click
detection.

On real tape the threshold alone fires far more often than on the fixtures' synthetic voice.
The shipped renders of the four local interviews (2026-10-09) logged 3.9 to 24.2 refilled
spans a second over both channels, against the 4 a second the calibrated class allowed, on
sources the harness reads as free of clicks. Re-rendered on CPU (2026-10-11), SOTI's and
Vaccin's spans are single-sample residual outliers in loud frames (median at the 80th and
71st percentile frame level), 98% of them no more than 1 dB over the waveform's own peak
within 10 ms and 79-80% within 15 ms of another span: the heavy-tailed residual of speech,
not clicks.
They lift no quiet frame and change the audio 31.5-32.7 dB under the programme. The
prominence guard (`apl_depop_min_prominence_db`, off by default) refills a span only when its
peak residual stands that far above every other residual within 5 ms either side; at 6 dB it
keeps +4.35 of the +4.67 dB of repair inside the crackle fixtures' pops and refills 0.50
spans a second per channel on SOTI and 1.38 on Vaccin (6.32 and 12.11 without it).
"""

from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.signal

try:
    import soundfile as sf
except ImportError:
    sf = None

from .hygiene import atomic_target
from .utils import log_msg

# Model order, detection block, and the span limits. A tape's pops are a few samples to a
# couple of milliseconds; anything longer than MAX_SPAN_SAMPLES is not a pop and is left to
# the dropout stage. Spans closer than MERGE_SAMPLES are one pop with a tail.
AR_ORDER = 24
BLOCK_SAMPLES = 8192
CONTEXT_SAMPLES = 4096
GUARD_SAMPLES = 3
MERGE_SAMPLES = 8
MAX_SPAN_SAMPLES = 160
# Residual outliers this many robust scales out are pops. 7 was chosen on the calibrated
# crackle class: 5 repairs +4.6 dB and touches 312 spans of undamaged speech per 15 s,
# 7 repairs +5.5 dB and touches 61, 8 starts missing pops.
DEFAULT_THRESHOLD = 7.0
# A pop stands alone. A drum hit's onset is followed by more outliers as it rings and by the
# next hit a beat later, and without this rule the stage refilled the onsets of the
# calibrated music-led classes, changing them 17 dB under the programme; with it, 34 dB
# under on music with speech and untouched on music alone, and the crackle repair is intact.
ISOLATION_WINDOW_SAMPLES = 1323
ISOLATION_MAX_DENSITY = 0.01
# The prominence guard's window (`apl_depop_min_prominence_db`, off at 0): 5 ms either side
# of the span. A click is one impulse and its residual stands alone; the residual of speech
# that trips the threshold comes in clusters. A 10 ms window refused more of both: at 6 dB
# the crackle fixtures kept +3.88 dB of repair against +4.35 at 5 ms, for 0.19 spans/s per
# channel on SOTI against 0.50.
PROMINENCE_WINDOW_SAMPLES = 220


def _ar_coefficients(samples, order):
    """Prediction coefficients [1, a1..ap] by the autocorrelation method (Levinson-Durbin)."""
    # Only the lags the model needs, each a dot product: the full correlation is quadratic
    # in the block and computes thousands of lags that are thrown away.
    count = len(samples)
    autocorrelation = np.array([np.dot(samples[: count - lag], samples[lag:]) for lag in range(order + 1)])
    autocorrelation[0] += 1e-9 * autocorrelation[0] + 1e-12
    tail = scipy.linalg.solve_toeplitz(autocorrelation[:order], -autocorrelation[1:])
    return np.concatenate(([1.0], tail))


def _residual(samples):
    """The prediction residual of a mono signal, block by block, and each sample's block scale.

    A block too short or too quiet to model keeps a zero residual and an infinite scale, so
    nothing in it reads as an outlier.
    """
    residual = np.zeros(len(samples))
    scale = np.full(len(samples), np.inf)
    for start in range(0, len(samples), BLOCK_SAMPLES):
        stop = min(start + BLOCK_SAMPLES, len(samples))
        low = max(0, start - AR_ORDER)
        block = samples[low:stop].astype(np.float64)
        if len(block) < 4 * AR_ORDER or float(np.max(np.abs(block))) < 1e-6:
            continue
        skip = stop - low - (stop - start)
        residual[start:stop] = scipy.signal.lfilter(_ar_coefficients(block, AR_ORDER), [1.0], block)[skip:]
        block_residual = residual[start:stop]
        scale[start:stop] = 1.4826 * float(np.median(np.abs(block_residual - np.median(block_residual)))) + 1e-9
    return residual, scale


def detect_pops(samples, threshold=DEFAULT_THRESHOLD):
    """Sample mask of the pops in a mono signal: prediction residual outliers, block by block."""
    residual, scale = _residual(samples)
    return np.abs(residual) > threshold * scale


def _spans(mask):
    """Contiguous true runs of a mask as (start, stop) pairs."""
    if mask.size == 0:
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    bounds = np.concatenate(([0], edges + 1, [len(mask)]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if mask[a]]


def _merged(spans, gap):
    """Joins spans closer than `gap` samples: a pop and its tail are one repair."""
    merged = []
    for start, stop in spans:
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return merged


def _interpolate(samples, start, stop, coefficients):
    """Refills samples[start:stop] by least-squares autoregressive interpolation, in place."""
    order = len(coefficients) - 1
    low, high = start - order, stop + order
    if low < 0 or high > len(samples):
        return False
    segment = samples[low:high].astype(np.float64)
    length, missing = high - low, stop - start
    rows = length - order
    equations = np.zeros((rows, length))
    for row in range(rows):
        row_end = row + order + 1
        equations[row, row:row_end] = coefficients[::-1]
    unknown = np.arange(start - low, stop - low)
    known = np.setdiff1d(np.arange(length), unknown)
    on_unknown, on_known = equations[:, unknown], equations[:, known]
    normal = on_unknown.T @ on_unknown + 1e-9 * np.eye(missing)
    segment[unknown] = np.linalg.solve(normal, -on_unknown.T @ (on_known @ segment[known]))
    samples[start:stop] = segment[unknown]
    return True


def _repair_span(samples, start, stop):
    """Fits a model to the audio around one span and refills it; False when it cannot."""
    low, high = max(0, start - CONTEXT_SAMPLES), min(len(samples), stop + CONTEXT_SAMPLES)
    context = np.concatenate([samples[low:start], samples[stop:high]])
    if len(context) < 4 * AR_ORDER:
        return False
    try:
        return _interpolate(samples, start, stop, _ar_coefficients(context, AR_ORDER))
    except (np.linalg.LinAlgError, ValueError):
        return False


def _isolated(mask, start, stop):
    """Whether a span stands alone: few other outliers within the window either side of it."""
    low, high = max(0, start - ISOLATION_WINDOW_SAMPLES), min(len(mask), stop + ISOLATION_WINDOW_SAMPLES)
    around = np.concatenate([mask[low:start], mask[stop:high]])
    return around.size == 0 or float(around.mean()) <= ISOLATION_MAX_DENSITY


def _prominent(residual, start, stop, min_prominence_db):
    """Whether a span's peak residual stands `min_prominence_db` over every other residual within the window.

    Always true with the guard off (0 dB or less), so the stage then refills what it always did.
    """
    if min_prominence_db <= 0.0:
        return True
    low, high = max(0, start - PROMINENCE_WINDOW_SAMPLES), min(len(residual), stop + PROMINENCE_WINDOW_SAMPLES)
    around = np.abs(np.concatenate([residual[low:start], residual[stop:high]]))
    peak = float(np.max(np.abs(residual[start:stop])))
    return around.size == 0 or peak >= float(np.max(around)) * 10.0 ** (min_prominence_db / 20.0)


def _refillable(residual, start, stop, min_prominence_db):
    """A guarded span short enough to be a pop and, with the guard on, prominent enough to be one."""
    return stop - start <= MAX_SPAN_SAMPLES and _prominent(residual, start, stop, min_prominence_db)


def remove_pops(samples, threshold=DEFAULT_THRESHOLD, min_prominence_db=0.0):
    """Returns (repaired mono signal, number of spans repaired)."""
    repaired = samples.astype(np.float64).copy()
    residual, scale = _residual(repaired)
    mask = np.abs(residual) > threshold * scale
    count = 0
    for start, stop in _merged(_spans(mask), MERGE_SAMPLES):
        if not _isolated(mask, start, stop):
            continue
        start, stop = max(0, start - GUARD_SAMPLES), min(len(repaired), stop + GUARD_SAMPLES)
        if _refillable(residual, start, stop, min_prominence_db) and _repair_span(repaired, start, stop):
            count += 1
    return repaired.astype(np.float32), count


def _repaired_channels(audio, threshold, min_prominence_db=0.0):
    """Every channel of a capture with its pops refilled, and how many spans that took."""
    repaired = np.empty_like(audio)
    total = 0
    for channel in range(audio.shape[1]):
        repaired[:, channel], count = remove_pops(audio[:, channel], threshold, min_prominence_db)
        total += count
    return repaired, total


def _guard_note(min_prominence_db):
    """The log's note on the prominence guard: nothing when it is off, so the line reads as it always did."""
    return f" (prominence guard {min_prominence_db:g} dB)" if min_prominence_db > 0.0 else ""


def depop(input_wav, output_dir, threshold=DEFAULT_THRESHOLD, total_duration=None, min_prominence_db=0.0):
    """Removes pops from a WAV, channel by channel; returns the output path or None.

    `total_duration` is accepted for the repair runner's uniform stage signature and unused.
    `min_prominence_db` is the prominence guard (`apl_depop_min_prominence_db`); 0 is off.
    """
    del total_duration
    if sf is None or threshold <= 0.0:
        return None
    try:
        audio, rate = sf.read(str(input_wav), dtype="float32", always_2d=True)
        repaired, total = _repaired_channels(audio, threshold, min_prominence_db)
    except (OSError, RuntimeError, ValueError) as exc:
        log_msg(f"    [Repair] depop skipped after failure: {exc}", level="WARNING")
        return None
    except MemoryError:
        # A capture the host cannot hold twice over is left unrepaired rather than ending
        # the restoration.
        log_msg("    [Repair] depop skipped: not enough memory for this capture.", level="WARNING")
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"depopped_{Path(input_wav).name}"
    with atomic_target(target) as partial:
        sf.write(str(partial), repaired, rate, subtype="FLOAT")
    log_msg(f"    [Repair] Refilled {total} pop spans{_guard_note(min_prominence_db)}.")
    return target
