"""What a denoise model declares about its output: the stem that is clean, and the grid its inference windows fall on.

The separator names every file it writes after a stem (`<input>_(<stem>)_<model>.wav`,
common_separator.get_stem_output_path in audio-separator 0.48.0). The VR denoisers
(UVR-DeNoise, UVR-DeNoise-Lite) write '(No Noise)' and '(Noise)'; an MDXC model writes its
training target and the residual, which for the shipped Mel-RoFormer denoiser
(`apl_neural_model`) is '(dry)' and '(other)': the `training.target_instrument` of the
`_config.yaml` that sits beside the checkpoint in the model store. The clean stem is
resolved by one rule wherever a denoised file is picked (the whole-file pass, the chunked
pass, a chunk or a whole output a previous run left): '(No Noise)' first, then the MDXC
target, then the first file in name order that is not a residual stem -- never '(Noise)',
'(other)' or another instrument the yaml lists beside its target. The whole-file pass
already picked '(dry)' for the RoFormer, by name order, so its choice does not move.
models-scores.json's `target_stem` is never read: it names 'noise' for the UVR models,
the stem this must never pick.

An MDXC model denoises in fixed windows of `hop x (dim_t - 1)` samples, overlap-added
every `window // num_overlap` samples (mdxc_separator.demix: the RoFormer branch takes the
model's `stft_hop_length`, the other branch the audio section's `hop_length`; `num_overlap`
comes from the yaml because the app passes the separator only a batch size). The shipped
model's yaml gives 441 x 800 = 352,800 samples, 8 s at 44.1 kHz, every 88,200 samples
(2 s). Each window is denoised on its own, so a chunk whose windows start where the whole
file's do, with a full window of true audio around its own range, sees exactly what the
whole file's windows saw (`denoise_chunking`).
"""

import functools
import os
import re
from pathlib import Path

import yaml

from . import utils

# The VR denoisers' clean stem, preferred for every model as it always was.
VR_CLEAN_TAG = "(No Noise)"
# Stems that are what the denoiser removed, never the denoised signal.
RESIDUAL_NAMES = ("noise", "other")
# The stem the separator names a file after is the last parenthesised group: the input's own
# name may carry an earlier one (a background stem from a vocal split is '..._(other)_...').
STEM_GROUP = re.compile(r"\(([^()]*)\)")
# audio-separator loads every model and its yaml from this folder when it is set, whatever
# folder the app passes (separator.py in 0.48.0); the stage cache key reads it the same way.
MODEL_DIR_ENV = "AUDIO_SEPARATOR_MODEL_DIR"
# Beside the checkpoint: `<stem>_config.yaml` (the denoise RoFormers), `<stem>.yaml` (the BS-RoFormer).
CONFIG_SUFFIXES = ("_config.yaml", ".yaml")
# The MDXC configs write one list as a Python tuple, which a safe loader refuses; it is
# read as the plain list it is.
PYTHON_TUPLE_TAG = "!!python/tuple"
# mdxc_separator's overlap when neither the app nor the yaml sets one.
DEFAULT_MDXC_OVERLAP = 8
# Keys only a RoFormer's model section carries (the library's own structural markers).
ROFORMER_MARKERS = ("num_bands", "freqs_per_bands")
# audio-separator's MDXC checkpoints (the RoFormers, MDX23C) end in '.ckpt'; its VR models end
# in '.pth' and its MDX models in '.onnx'. The yaml of an MDXC model reaches the store only
# when the separator first loads the model, so a fresh store has none.
MDXC_CHECKPOINT_SUFFIX = ".ckpt"


def model_store(environ=None):
    """The folder the separator loads models from: `AUDIO_SEPARATOR_MODEL_DIR` when set, else the app's model store."""
    environ = os.environ if environ is None else environ
    override = environ.get(MODEL_DIR_ENV)
    return Path(override) if override else Path(utils.MODELS_DIR)


def config_path(model_name, store=None):
    """The yaml beside an MDXC checkpoint, or None (a VR model has none)."""
    store = model_store() if store is None else Path(store)
    stem = Path(str(model_name or "")).stem
    candidates = [store / f"{stem}{suffix}" for suffix in CONFIG_SUFFIXES]
    return next(filter(Path.is_file, candidates), None) if stem else None


def is_mdxc_checkpoint(model_name):
    """Whether the model is named as an MDXC checkpoint ('.ckpt'), whose chunk grid comes from its yaml."""
    return str(model_name or "").lower().endswith(MDXC_CHECKPOINT_SUFFIX)


def config_missing(model_name, store=None):
    """Whether an MDXC checkpoint has no yaml in the store yet: the separator fetches it on the model's first load."""
    return is_mdxc_checkpoint(model_name) and config_path(model_name, store) is None


@functools.lru_cache(maxsize=32)
def _parsed_config(path, _stamp):
    """The yaml at `path` as a mapping, or None when it cannot be read; cached per file revision."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8").replace(PYTHON_TUPLE_TAG, ""))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        utils.log_msg(f"    [Warning] Could not read the model config {Path(path).name} ({exc}).", level="WARNING")
        return None
    return data if isinstance(data, dict) else None


def _config_at(path):
    """The yaml at `path` (None for no file) as a mapping, or None."""
    if path is None:
        return None
    stat = path.stat()
    return _parsed_config(str(path), (stat.st_size, stat.st_mtime_ns))


def model_config(model_name, store=None):
    """The model's MDXC yaml as a mapping, or None for a VR model, a missing yaml or an unreadable one."""
    return _config_at(config_path(model_name, store))


def _section(data, name):
    """One section of a model config, empty when absent or not a mapping."""
    section = (data or {}).get(name)
    return section if isinstance(section, dict) else {}


def _target_name(training):
    """The training target when it names a clean signal, else None (absent, or itself a residual)."""
    target = training.get("target_instrument")
    if not isinstance(target, str) or not target or target.lower() in RESIDUAL_NAMES:
        return None
    return target


def target_tag(model_name, store=None):
    """'(<training.target_instrument>)' of an MDXC model, or None."""
    target = _target_name(_section(model_config(model_name, store), "training"))
    return None if target is None else f"({target})"


def residual_names(model_name, store=None):
    """Lower-case stem names never taken as clean: noise, other, and every instrument the yaml lists beside a clean target.

    A model without a clean target (none, or one named noise) adds nothing: its clean
    stem is then the instrument that is not noise.
    """
    training = _section(model_config(model_name, store), "training")
    target = _target_name(training)
    if target is None:
        return RESIDUAL_NAMES
    listed = training.get("instruments") or []
    return RESIDUAL_NAMES + tuple(str(name).lower() for name in listed if str(name) != target)


def _is_residual(path, residuals):
    """Whether the stem the file is named after (its last parenthesised group) is a residual."""
    groups = STEM_GROUP.findall(Path(path).name.lower())
    return bool(groups) and groups[-1] in residuals


def _first_tagged(ordered, tags):
    """The first file carrying the first tag (None tags skipped) any file carries, or None."""
    for tag in filter(None, tags):
        tagged = [path for path in ordered if tag in Path(path).name]
        if tagged:
            return tagged[0]
    return None


def _first_plain(ordered, residuals):
    """The first file not named after a residual stem, or None."""
    return next((path for path in ordered if not _is_residual(path, residuals)), None)


def clean_stem(paths, model_name=None, store=None):
    """The clean stem among a separator's outputs, or None when every file is a residual.

    '(No Noise)' first (the VR models), then the MDXC model's training target ('(dry)' for
    the shipped RoFormer), then the first file in name order that is not a residual stem;
    callers that need valid audio filter before calling.
    """
    ordered = sorted(paths, key=lambda path: Path(path).name.lower())
    chosen = _first_tagged(ordered, (VR_CLEAN_TAG, target_tag(model_name, store)))
    return chosen if chosen is not None else _first_plain(ordered, residual_names(model_name, store))


def _is_roformer(path, data):
    """Whether the separator runs this config through its RoFormer branch (yaml name or structural marker)."""
    model = _section(data, "model")
    return "roformer" in Path(path).name.lower() or any(marker in model for marker in ROFORMER_MARKERS)


def _hop_length(path, data):
    """The hop the separator sizes its window from: the RoFormer's STFT hop, else the audio section's."""
    model, audio = _section(data, "model"), _section(data, "audio")
    if _is_roformer(path, data) and model.get("stft_hop_length") is not None:
        return int(model["stft_hop_length"])
    return int(audio["hop_length"])


def _window_from(path, data):
    """(window, step) from a parsed config; raises on a config without the inference keys."""
    inference = _section(data, "inference")
    window = _hop_length(path, data) * (int(inference["dim_t"]) - 1)
    overlap = inference.get("num_overlap")
    overlap = DEFAULT_MDXC_OVERLAP if overlap is None else int(overlap)
    return window, window // overlap


def mdxc_window(model_name, store=None):
    """(window, step) in samples of an MDXC model's overlap-add, or None for a VR model or an unusable yaml."""
    path = config_path(model_name, store)
    data = _config_at(path)
    if data is None:
        return None
    try:
        window, step = _window_from(path, data)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return (window, step) if 0 < step <= window else None
