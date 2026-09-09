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
"""

from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.signal

try:
    import soundfile as sf
except ImportError:
    sf = None

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


def _ar_coefficients(samples, order):
    """Prediction coefficients [1, a1..ap] by the autocorrelation method (Levinson-Durbin)."""
    zero_lag, last_lag = len(samples) - 1, len(samples) + order
    autocorrelation = np.correlate(samples, samples, mode="full")[zero_lag:last_lag]
    autocorrelation[0] += 1e-9 * autocorrelation[0] + 1e-12
    tail = scipy.linalg.solve_toeplitz(autocorrelation[:order], -autocorrelation[1:])
    return np.concatenate(([1.0], tail))


def detect_pops(samples, threshold=DEFAULT_THRESHOLD):
    """Sample mask of the pops in a mono signal: prediction residual outliers, block by block."""
    mask = np.zeros(len(samples), dtype=bool)
    for start in range(0, len(samples), BLOCK_SAMPLES):
        stop = min(start + BLOCK_SAMPLES, len(samples))
        low = max(0, start - AR_ORDER)
        block = samples[low:stop].astype(np.float64)
        if len(block) < 4 * AR_ORDER or float(np.max(np.abs(block))) < 1e-6:
            continue
        skip = stop - low - (stop - start)
        residual = scipy.signal.lfilter(_ar_coefficients(block, AR_ORDER), [1.0], block)[skip:]
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) + 1e-9
        mask[start:stop] = np.abs(residual) > threshold * scale
    return mask


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


def remove_pops(samples, threshold=DEFAULT_THRESHOLD):
    """Returns (repaired mono signal, number of spans repaired)."""
    repaired = samples.astype(np.float64).copy()
    mask = detect_pops(repaired, threshold)
    count = 0
    for start, stop in _merged(_spans(mask), MERGE_SAMPLES):
        if not _isolated(mask, start, stop):
            continue
        start, stop = max(0, start - GUARD_SAMPLES), min(len(repaired), stop + GUARD_SAMPLES)
        if stop - start <= MAX_SPAN_SAMPLES and _repair_span(repaired, start, stop):
            count += 1
    return repaired.astype(np.float32), count


def depop(input_wav, output_dir, threshold=DEFAULT_THRESHOLD, total_duration=None):
    """Removes pops from a WAV, channel by channel; returns the output path or None.

    `total_duration` is accepted for the repair runner's uniform stage signature and unused.
    """
    del total_duration
    if sf is None or threshold <= 0.0:
        return None
    try:
        audio, rate = sf.read(str(input_wav), dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError):
        return None
    repaired = np.empty_like(audio)
    total = 0
    for channel in range(audio.shape[1]):
        repaired[:, channel], count = remove_pops(audio[:, channel], threshold)
        total += count
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"depopped_{Path(input_wav).name}"
    sf.write(str(target), repaired, rate, subtype="FLOAT")
    log_msg(f"    [Repair] Refilled {total} pop spans.")
    return target
