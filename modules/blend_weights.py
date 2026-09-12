"""Per-bin blending between a signal and its spectrally subtracted version.

Spectral subtraction removes noise and takes some programme with it. Measured against a
clean reference, its band energy lands 2.1-3.7 dB below the truth on hum-dominated material,
and that over-subtraction has been the recurring failure of every restoration stage measured
on this branch: strong enough to clear the noise is too strong for the content.

Blending the subtracted result back toward the original per frequency bin escapes the
compromise, because the choice stops being global. An oracle that picks the ideal weight
from a clean reference is worth 1.7-1.9 dB of log-spectral distance over the unblended
chain, so the structure is right; two hand-designed weightings lost anyway, which made the
weighting a thing to learn rather than guess.

The shipped weights were fitted across six languages and held out by language. On fixtures
built from an independent seed the blend improves log-spectral distance by 0.307 dB on
average and wins five of seven defect families, and it does so by repairing the
over-subtraction rather than trading against it: the defect band moves from -2.21 dB to
-0.71 on hum and from -1.51 to +0.90 on the combined case.

Inference is numpy only. The model is two small dense layers, so loading torch to evaluate
it would cost far more than the arithmetic.

The capture is processed in blocks of frames, never whole: the eleven features of a two-hour
tape would be 28 GB. Two of the features are statistics over the whole recording, the
per-bin noise floor and the per-bin temporal spread, so a first pass gathers every frame's
magnitude into a memory-mapped scratch file and reduces it bin by bin, and a second pass
blends each block with enough context either side that its interior comes out identical to
whole-file processing.
"""

import zipfile
from pathlib import Path

import numpy as np

try:
    import scipy.signal
except ImportError:
    scipy = None

try:
    import soundfile as sf
except ImportError:
    sf = None

from .utils import log_msg

# The analysis window the weights were fitted with. Changing it invalidates the model,
# because every feature is expressed in terms of these bins.
FRAME = 2048
HOP = FRAME // 2
EPS = 1e-10
# Frames per processing block, and the context either side of one. The context covers the
# temporal smoothing (two frames each way) and the analysis window's reach past a block's
# first and last hop, so a block's interior is what whole-file processing would produce. A
# 256-frame block is six seconds at 44.1 kHz and under 200 MB of features and activations.
BLOCK_FRAMES = 256
CONTEXT_FRAMES = 8
# The arrays a weights file has to carry, and the shapes they have to agree on, for the
# network below to run at all. A file missing any of them, or fitted to another width,
# skips the blend rather than raising out of the arithmetic.
REQUIRED_KEYS = (
    "feature_mean",
    "feature_std",
    "stack.0.weight",
    "stack.0.bias",
    "stack.2.weight",
    "stack.2.bias",
    "stack.4.weight",
    "stack.4.bias",
)

DEFAULT_WEIGHTS_PATH = Path(__file__).parent.parent / "assets" / "blend_weights.npz"

FEATURE_NAMES = (
    "log_original",
    "log_denoised",
    "log_removed",
    "removal_ratio",
    "bin_position",
    "frame_level",
    "bin_excess_over_floor",
    # Context. The seven above are strictly per-bin and per-frame, and a four-times wider
    # model on that set bought 1% of the available headroom, so capacity was not what the
    # weighting was short of. What none of them can say is what the neighbouring bins and
    # the surrounding frames looked like -- which is exactly what separates stationary tape
    # noise from programme.
    "bin_temporal_std",
    "spectral_context",
    "temporal_context",
    "removal_ratio_context",
)

_CACHE = {}


def _smooth(values, size, axis):
    """Box-averages along one axis, holding the edges rather than padding with zeros.

    Zero padding would invent a cliff at the first and last frame, and the model would learn
    to read that cliff as an onset.
    """
    if size <= 1:
        return values
    pad = size // 2
    padding = [(0, 0), (0, 0)]
    padding[axis] = (pad, pad)
    padded = np.pad(values, padding, mode="edge")
    kernel = np.ones(size, dtype=values.dtype) / size
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis, padded)


def feature_stack(magnitude_original, magnitude_denoised, floor=None, temporal_spread=None):
    """Builds the per-bin features the weight predictor consumes.

    This is the single definition, imported by the dataset builder as well. Training and
    inference computing features differently is the classic way a learned stage quietly
    stops working, and one shared function is the only reliable defence.

    Every feature is observable without a clean reference, which is what lets the model run
    at inference at all. `floor` and `temporal_spread` are the two per-bin statistics of the
    whole recording; when the magnitudes given are the whole recording they are computed
    here, and when they are one block of it the caller passes the recording's.
    """
    removed = np.maximum(magnitude_original - magnitude_denoised, 0.0)
    log_original = 20.0 * np.log10(magnitude_original + EPS)
    log_denoised = 20.0 * np.log10(magnitude_denoised + EPS)
    log_removed = 20.0 * np.log10(removed + EPS)
    removal_ratio = removed / (magnitude_original + EPS)

    bins, frames = magnitude_original.shape
    bin_position = np.tile(np.linspace(0.0, 1.0, bins).reshape(-1, 1), (1, frames))
    frame_level = np.tile(20.0 * np.log10(np.mean(magnitude_original, axis=0, keepdims=True) + EPS), (bins, 1))
    if floor is None:
        floor = np.percentile(magnitude_original, 10.0, axis=1, keepdims=True)
    bin_excess = 20.0 * np.log10((magnitude_original + EPS) / (floor + EPS))

    # Stationarity. A bin carrying only tape noise barely moves across the recording, while
    # one carrying speech swings by tens of dB. On a synthetic check the two populations
    # separate by 4.49 dB, which is the strongest single cue the per-bin features lacked.
    if temporal_spread is None:
        temporal_spread = np.std(log_original, axis=1, keepdims=True)
    bin_temporal_std = np.tile(temporal_spread, (1, frames))

    # Neighbourhood. Subtraction damage is isolated where real content is locally coherent
    # in both directions, so the surrounding bins and frames say whether a removal belongs.
    spectral_context = _smooth(log_original, 5, axis=0) - log_original
    temporal_context = _smooth(log_original, 5, axis=1) - log_original
    removal_ratio_context = _smooth(_smooth(removal_ratio, 5, axis=0), 5, axis=1)

    return np.stack(
        [
            log_original,
            log_denoised,
            log_removed,
            removal_ratio,
            bin_position,
            frame_level,
            bin_excess,
            bin_temporal_std,
            spectral_context,
            temporal_context,
            removal_ratio_context,
        ],
        axis=-1,
    )


def _read_weights(path):
    """Reads one weights file, or None when it is absent or unreadable."""
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            return {name: data[name] for name in data.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return None


def _normaliser_fits(value, width):
    """Whether a feature mean or spread broadcasts over rows of `width` features."""
    return value.ndim in (1, 2) and value.shape[-1] == width and (value.ndim == 1 or value.shape[0] == 1)


def _layers_of(model):
    """The (weight, bias) pairs of the network, in order."""
    return [(model[f"stack.{index}.weight"], model[f"stack.{index}.bias"]) for index in (0, 2, 4)]


def _layers_agree(layers, _model):
    """Whether every weight is a matrix with a bias per output row."""
    return all(weight.ndim == 2 and bias.shape == (weight.shape[0],) for weight, bias in layers)


def _layers_chain(layers, _model):
    """Whether each layer's input width is the previous layer's output width, ending in one output."""
    widths_meet = all(layers[index][0].shape[1] == layers[index - 1][0].shape[0] for index in (1, 2))
    return widths_meet and layers[2][0].shape[0] == 1


def _normalisation_fits(layers, model):
    """Whether the feature mean and spread fit the first layer's input width."""
    width = layers[0][0].shape[1]
    return _normaliser_fits(model["feature_mean"], width) and _normaliser_fits(model["feature_std"], width)


def _values_usable(_layers, model):
    """Whether every array is finite and every feature spread is positive.

    A NaN or infinity anywhere in the network turns every weight into NaN, and a zero
    spread divides the features by zero; either would write silence or garbage rather
    than a blend, and neither is a mismatch the shape rules can see.
    """
    finite = all(np.isfinite(model[key]).all() for key in REQUIRED_KEYS)
    return finite and bool(np.all(model["feature_std"] > 0.0))


# Each rule a weights file has to satisfy, with what to say when it does not.
def _features_named(model):
    """A file that carries its feature names was fitted on this feature set, by name and not only by width.

    Older artifacts carry no names and are held to the width rule alone.
    """
    if "feature_names" not in model:
        return True
    return tuple(str(name) for name in model["feature_names"]) == tuple(FEATURE_NAMES)


_SCHEMA_RULES = (
    (lambda _layers, model: _features_named(model), "the weights were fitted on a different feature set"),
    (_layers_agree, "a layer's weight and bias disagree"),
    (_layers_chain, "the layers do not chain to one output"),
    (_normalisation_fits, "the feature normalisation does not fit the first layer"),
    (_values_usable, "the weights carry non-finite values or a zero feature spread"),
)


def schema_error(model):
    """Why a weights file cannot drive the network, or None when it can.

    A partial or truncated file, or one fitted to another layout, used to raise KeyError or
    ValueError out of the arithmetic and take the restoration down with it; here it is a
    reason to skip the refinement.
    """
    missing = [key for key in REQUIRED_KEYS if key not in model]
    if missing:
        return f"missing {', '.join(missing)}"
    return _shape_error(_layers_of(model), model)


def _shape_error(layers, model):
    """The first shape rule a complete weights file breaks, or None.

    The rules run in order and stop at the first failure: a later rule reads shapes an
    earlier one has already found wrong, and would raise rather than report.
    """
    for holds, problem in _SCHEMA_RULES:
        if not holds(layers, model):
            return problem
    return None


def load_model(model_path=None):
    """Loads and caches the fitted weights, or returns None when they are unavailable or malformed."""
    path = Path(model_path or DEFAULT_WEIGHTS_PATH)
    key = str(path)
    if key not in _CACHE:
        model = _read_weights(path)
        problem = None if model is None else schema_error(model)
        if problem is not None:
            log_msg(f"    [Blend] Skipped, weights file unusable: {problem}.")
            model = None
        _CACHE[key] = model
    return _CACHE[key]


def predict_weights(features, model):
    """Runs the two-layer network; returns a weight in [0, 1] per row.

    Returns None when the weights file is malformed or was fitted against a different
    feature set. The feature definition and the weights file have to move together, and a
    mismatch used to raise straight through apply_blend, so a stale file would have taken a
    whole restoration down rather than skipping a refinement.
    """
    problem = schema_error(model)
    if problem is None and features.shape[-1] != model["stack.0.weight"].shape[1]:
        problem = "weights do not match the current feature set"
    if problem is not None:
        log_msg(f"    [Blend] Skipped, {problem}.")
        return None
    scaled = (features - model["feature_mean"]) / model["feature_std"]
    hidden = np.maximum(scaled @ model["stack.0.weight"].T + model["stack.0.bias"], 0.0)
    hidden = np.maximum(hidden @ model["stack.2.weight"].T + model["stack.2.bias"], 0.0)
    logits = hidden @ model["stack.4.weight"].T + model["stack.4.bias"]
    return 1.0 / (1.0 + np.exp(-logits[..., 0]))


def _frame_count(length):
    """Frames the pipeline's STFT has for `length` samples: one per hop, plus the boundary frame."""
    return -(-length // HOP) + 1


def _read_frames(source, first, last, length):
    """The samples under frames [first, last) of a file, zero beyond the file's `length`.

    Frame k is the analysis window centred on sample k * HOP, so the span starts half a
    window before the first frame and ends half a window after the last; taken this way, a
    block's STFT is frame for frame the whole file's.
    """
    start, stop = first * HOP - FRAME // 2, (last - 1) * HOP + FRAME // 2
    span = np.zeros((stop - start, source.channels), dtype=np.float32)
    low, high = max(0, start), min(length, stop)
    if high > low:
        source.seek(low)
        into = slice(low - start, high - start)
        span[into] = source.read(high - low, dtype="float32", always_2d=True)
    return span


def _stft(samples, rate):
    """The STFT of a span read by _read_frames: exactly one frame per hop, no padding of its own."""
    _f, _t, spectrum = scipy.signal.stft(samples, fs=rate, nperseg=FRAME, boundary=None, padded=False)
    return spectrum


def _reduce_statistics(magnitudes):
    """Per-bin floor and temporal spread of one channel's magnitudes, taken a few bins at a time."""
    bins = magnitudes.shape[0]
    floor = np.zeros((bins, 1))
    spread = np.zeros((bins, 1))
    for low in range(0, bins, 64):
        high = min(low + 64, bins)
        rows = np.asarray(magnitudes[low:high])
        floor[low:high] = np.percentile(rows, 10.0, axis=1, keepdims=True)
        spread[low:high] = np.std(20.0 * np.log10(rows + EPS), axis=1, keepdims=True)
    return floor, spread


def _recording_statistics(source, rate, length, channels, scratch):
    """The per-bin floor and temporal spread of every channel, from one pass over the file.

    The 10th-percentile floor needs every frame's magnitude at once, so the magnitudes go to
    a memory-mapped scratch file rather than memory and are reduced bin by bin afterwards.
    """
    total = _frame_count(length)
    # float32 halves the scratch file (a two-hour stereo tape's is 2.5 GB rather than 5);
    # the floor and spread it yields differ from float64's by parts in ten million, which
    # the equivalence test bounds at the sample level.
    magnitudes = np.memmap(scratch, dtype=np.float32, mode="w+", shape=(channels, FRAME // 2 + 1, total))
    try:
        for first in range(0, total, BLOCK_FRAMES):
            last = min(first + BLOCK_FRAMES, total)
            span = _read_frames(source, first, last, length)
            for channel in range(channels):
                magnitudes[channel, :, first:last] = np.abs(_stft(span[:, channel], rate))
        return [_reduce_statistics(magnitudes[channel]) for channel in range(channels)]
    finally:
        del magnitudes
        Path(scratch).unlink(missing_ok=True)


def _blend_span(original, denoised, rate, model, statistics):
    """Blends one channel over one span of frames; returns its waveform, or None."""
    spec_original = _stft(original, rate)
    magnitude_original = np.abs(spec_original)
    magnitude_denoised = np.abs(_stft(denoised, rate))
    floor, spread = statistics
    stack = feature_stack(magnitude_original, magnitude_denoised, floor=floor, temporal_spread=spread)
    predicted = predict_weights(stack.reshape(-1, stack.shape[-1]), model)
    if predicted is None:
        return None
    weight = predicted.reshape(magnitude_original.shape)
    magnitude = weight * magnitude_denoised + (1.0 - weight) * magnitude_original

    # The original's phase is kept throughout. A magnitude mask has no business moving
    # phase, and subtraction already did.
    spectrum = magnitude * np.exp(1j * np.angle(spec_original))
    # `boundary` here only strips half a window at each end of the span: sample 0 of the
    # result is the first frame's centre, and the frames either side of the span's ends are
    # the context the caller read for exactly this purpose.
    _t, wave = scipy.signal.istft(spectrum, fs=rate, nperseg=FRAME, boundary=True)
    return wave


def _blend_block(sources, rate, length, channels, model, statistics, first, last):
    """Blends every channel of the samples under frames [first, last); None when the model refuses."""
    count = min(last * HOP, length) - first * HOP
    block = np.zeros((max(count, 0), channels), dtype=np.float32)
    if count <= 0:
        return block
    low, high = max(0, first - CONTEXT_FRAMES), min(_frame_count(length), last + CONTEXT_FRAMES)
    spans = [_read_frames(source, low, high, length) for source in sources]
    offset = (first - low) * HOP
    stop = offset + count
    for channel in range(channels):
        wave = _blend_span(spans[0][:, channel], spans[1][:, channel], rate, model, statistics[channel])
        if wave is None:
            return None
        block[:, channel] = wave[offset:stop]
    return block


def _write_blend(sources, rate, length, channels, model, target):
    """Blends the capture block by block into `target`; None when the model refuses the features."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    statistics = _recording_statistics(sources[0], rate, length, channels, target.with_suffix(".magnitudes"))
    with sf.SoundFile(str(target), "w", samplerate=rate, channels=channels, subtype="FLOAT") as out:
        for first in range(0, _frame_count(length), BLOCK_FRAMES):
            last = min(first + BLOCK_FRAMES, _frame_count(length))
            block = _blend_block(sources, rate, length, channels, model, statistics, first, last)
            if block is None:
                break
            out.write(block)
        else:
            return target
    target.unlink(missing_ok=True)
    return None


def _unavailable(model):
    """Returns whether the blend can run at all on this host."""
    return model is None or sf is None or scipy is None


def _blend_open(original, denoised, target, model):
    """Blends two open files; None when they differ in format or length, or there is too little audio.

    The two are the same capture before and after subtraction, so a rate, channel count or
    frame count that differs is a wrong file, not something to reconcile by resampling,
    dropping a channel or truncating to the shorter -- a truncated blend would replace the
    complete subtraction with a prefix of it.
    """
    if not _same_capture(original, denoised):
        log_msg("    [Blend] Skipped, the original and the denoised audio differ in rate, channels or length.")
        return None
    if original.frames < FRAME or original.channels == 0:
        return None
    return _write_blend((original, denoised), original.samplerate, original.frames, original.channels, model, target)


def _same_capture(original, denoised):
    """Whether two open files agree on rate, channel count and frame count."""
    return (original.samplerate, original.channels, original.frames) == (denoised.samplerate, denoised.channels, denoised.frames)


def _blend_files(original_wav, denoised_wav, target, model):
    """Opens both signals and blends them; the denoised path when there is nothing to blend."""
    try:
        with sf.SoundFile(str(original_wav)) as original, sf.SoundFile(str(denoised_wav)) as denoised:
            return _blend_open(original, denoised, target, model) or denoised_wav
    except (OSError, RuntimeError) as exc:
        log_msg(f"    [Blend] Skipped, audio unreadable: {exc}")
    except MemoryError:
        log_msg("    [Blend] Skipped, out of memory.")
    return denoised_wav


def apply_blend(original_wav, denoised_wav, target, model_path=None):
    """Blends a subtracted signal back toward its original, per bin.

    Returns the denoised input untouched when the weights are missing or malformed, the
    audio cannot be read, or the host runs out of memory, so a restoration never fails
    because this refinement is unavailable.
    """
    model = load_model(model_path)
    if _unavailable(model):
        return denoised_wav
    return _blend_files(original_wav, denoised_wav, target, model)
