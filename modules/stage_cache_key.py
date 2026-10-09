"""The key of a stage-cache entry: everything the cached stages read, so any change that could move their output misses.

The cached point is the neural stage of `auto_pure_linear` and `denoise_only`
(`processing._neural_output`: the surgical notch, the deterministic chain and the neural
model). The key is the SHA-256 of one canonical JSON document holding:

- `input`: the WAV the cached stages read, by its decoded samples (a soundfile FLOAT WAV
  carries a PEAK chunk with a write timestamp, so two writes of the same samples differ in
  their bytes), with its name, format, rate, channels and length. That one file stands for
  the source, the extraction and the pre-conditioning together.
- `call`: every argument of the cached stages, the scanner's whole strategy and the seven
  stage switches included, plus the pipeline rate and precision.
- `config`: the resolved `config.CONFIG` minus `POST_NEURAL_CONFIG_KEYS`. Deny by default:
  a key is left out only when every read of it is after the cached point (the read sites are
  frozen in `tests/unit/test_stage_cache_allowlist.py`), and a key nobody knows is kept.
- `code`: the sources of every module but the four that only run after the cached point, and
  what importing those four runs (their import-time code, with the bodies of the functions
  import never calls left out). Read when the process starts, with the cache on (`prime`):
  the process runs the code it loaded then, and `drift` refuses to store a render when a
  source or a package changed after that.
- `binaries`, `models`, `assets`, `runtime`: ffmpeg and cathar by content, the model files the
  stage can load from the folder the separator really uses, the learned blend weights, and
  the interpreter, packages, torch build, device, GPU driver, CPU, thread counts, chunk
  length and every environment variable with a prefix that steers them.

In-process patches of a module constant are not seen (the key reads `config.CONFIG`); a test
or an experiment that patches one patches `CONFIG` too, or leaves the cache off.
"""

import ast
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None

from . import blend_weights, config, denoise_chunking, hardware, spectral_denoise, utils

KEY_SCHEMA = 2
STAGE_FULL_MIX_NEURAL = "full_mix_neural"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENTRY_SCRIPT = "restore_audio_hybrid.py"
HASH_BLOCK_FRAMES = 1 << 20
FILE_BLOCK_BYTES = 1 << 22
VERSION_TIMEOUT_S = 30
CACHE_ENV_VAR = "AI_RESTORE_STAGE_CACHE"
# audio-separator loads every model, its `_config.yaml` and its parameter tables from this
# folder when it is set, whatever folder the stage passes (separator.py in 0.47.0).
MODEL_DIR_ENV = "AUDIO_SEPARATOR_MODEL_DIR"

# Read only after the cached point; each read site is checked by the allowlist test.
POST_NEURAL_CONFIG_KEYS = frozenset(
    {
        # The polish (processing._polish_full_audio_step -> filters.build_full_audio_polish_filter).
        "enable_linear_air",
        "linear_air_gain_db",
        "enable_dynamic_expander",
        "expander_depth_db",
        "expander_knee_offset_db",
        # Passed by the mode as a post-neural stage setting, never to the cached stages.
        "apl_expander_depth_db",
        # The sibilant guard and the pause floor keeper (their whole modules run after it).
        "apl_enable_sibilant_guard",
        "apl_sibilant_mix",
        "apl_sibilant_guard_hz",
        "apl_sibilant_hf_share_min",
        "enable_pause_floor",
        "pause_floor_fill_db",
        "pause_floor_quiet_percentile",
        # The mux (modules/mastering.py, processing's mux commands) and the sync (modules/sync.py).
        "enable_loudnorm",
        "loudnorm_target_lra",
        "loudnorm_linear_fallback",
        "preserve_original_audio_track",
        "dtw_resolution",
        # The stem path's background pass, which never enters the cached stages.
        "apl_music_bg_floor_db",
    }
)

# Modules whose functions run only after the cached point. Their function bodies stay out of
# the code key; what importing them runs stays in it, since processing imports two of them at
# start-up and the other two are imported in the middle of a process that may cache again.
POST_CACHE_MODULES = frozenset({"modules/sibilant_guard.py", "modules/pause_floor.py", "modules/mastering.py", "modules/sync.py"})

# The environment, keyed by prefix and so deny by default as the configuration is: every
# variable that steers the app, audio-separator, the GPU stack, the thread pools, numpy's
# instruction-set dispatch or cathar's Rust runtime, whatever the loop's `env:` knobs set.
# TEMP and TMP, which point into each work folder, match none of them.
ENV_PREFIXES = (
    "AI_RESTORE_",
    "AUDIO_SEPARATOR_",
    "CUDA",
    "CUBLAS",
    "CUDNN",
    "NVIDIA_",
    "TORCH",
    "PYTORCH_",
    "OMP_",
    "KMP_",
    "MKL_",
    "OPENBLAS_",
    "GOTO",
    "BLIS_",
    "VECLIB_",
    "NUMEXPR_",
    "NUMBA_",
    "NPY_",
    "RAYON_",
    "ORT_",
    "ONNXRUNTIME",
)
ENV_NAMES = frozenset({"PYTHONHASHSEED"})
# Matched by a prefix, but naming only where this run keeps its cache and its event records
# (the event log bypasses the cache).
ENV_UNKEYED = frozenset({CACHE_ENV_VAR, "AI_RESTORE_STAGE_CACHE_MAX_GB", "AI_RESTORE_EVENT_LOG"})

# audio-separator's model parameter tables, read beside every model file.
MODEL_DATA_FILES = frozenset({"vr_model_data.json", "mdx_model_data.json", "download_checks.json"})

FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
DEFINITIONS = (*FUNCTIONS, ast.ClassDef)


def jsonable(value):
    """What `json` cannot write itself: numpy values as Python ones, sets sorted, anything else by repr."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    return repr(value)


def canonical_json(document):
    """One spelling per document: sorted keys, no spaces, NaN allowed, numpy values as Python ones."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=True, default=jsonable)


def key_of(document):
    """The SHA-256 of a key document's canonical JSON."""
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def _sample_dtype(subtype):
    """Float files read as stored, every integer PCM as int32, so no two sample values collapse into one."""
    return {"FLOAT": "float32", "DOUBLE": "float64"}.get(subtype, "int32")


def input_identity(wav_path, block_frames=HASH_BLOCK_FRAMES):
    """The input WAV by its name, layout and decoded samples (the PEAK chunk's timestamp cannot reach it).

    The sample hash follows `scripts/autotune_restoration.exact_audio_sha256`: rate, channels
    and sample type, then the samples as stored, read in blocks.
    """
    if sf is None:
        raise RuntimeError("soundfile is not installed")
    digest = hashlib.sha256()
    with sf.SoundFile(str(wav_path)) as audio:
        dtype = _sample_dtype(audio.subtype)
        identity = {
            "name": Path(wav_path).name,
            "format": audio.format,
            "subtype": audio.subtype,
            "rate": audio.samplerate,
            "channels": audio.channels,
            "frames": audio.frames,
        }
        digest.update(f"{audio.samplerate}|{audio.channels}|{dtype}|".encode("ascii"))
        for block in audio.blocks(blocksize=block_frames, dtype=dtype, always_2d=True):
            digest.update(block.tobytes())
    identity["pcm_sha256"] = digest.hexdigest()
    return identity


def config_identity(resolved=None):
    """The resolved configuration minus the keys read only after the cached point."""
    resolved = config.CONFIG if resolved is None else resolved
    return {key: value for key, value in resolved.items() if key not in POST_NEURAL_CONFIG_KEYS}


# ----------------------------------------------------------------------------- the code


def _signature_nodes(function):
    """What a `def` runs when the module is imported: its decorators, default values and annotations."""
    for part in (*function.decorator_list, function.args, function.returns):
        if part is not None:
            yield from ast.walk(part)


def _import_time_nodes(node):
    """Every node importing a module runs: all of it but the bodies of its functions and methods (class bodies run)."""
    for child in ast.iter_child_nodes(node):
        yield child
        yield from _signature_nodes(child) if isinstance(child, FUNCTIONS) else _import_time_nodes(child)


def _names(nodes):
    return {node.id for node in nodes if isinstance(node, ast.Name)}


def _definitions(tree):
    """The module's own top-level functions and classes, by name."""
    return {node.name: node for node in tree.body if isinstance(node, DEFINITIONS)}


def _named_at_import(tree):
    """The top-level functions and classes import may run: named by import-time code, then by what those name, in turn."""
    definitions = _definitions(tree)
    named, todo = set(), sorted(_names(_import_time_nodes(tree)))
    while todo:
        name = todo.pop()
        if name in definitions and name not in named:
            named.add(name)
            todo.extend(sorted(_names(ast.walk(definitions[name]))))
    return named


def _strip_bodies(node):
    """Drops a function's body, or every method body of a class; a class's own statements run at import and stay."""
    if isinstance(node, FUNCTIONS):
        node.body = [ast.Pass()]
        return
    for child in node.body:
        if isinstance(child, DEFINITIONS):
            _strip_bodies(child)


def _parsed(source):
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"a post-cache module does not parse: {exc}") from exc


def import_time_code(source):
    """What importing a module runs, as an AST dump without line numbers.

    The bodies of the top-level functions and methods nothing at import can reach are left
    out; their signatures, decorators and defaults, every class body, every statement at the
    top level and the whole of any function import-time code names (followed through the
    module) stay, since importing the module runs them.
    """
    tree = _parsed(source)
    named = _named_at_import(tree)
    for name, node in _definitions(tree).items():
        if name not in named:
            _strip_bodies(node)
    return ast.dump(tree)


def _module_sources(root):
    """Relative posix name -> path of every module source but the post-cache four."""
    named = ((path.relative_to(root).as_posix(), path) for path in (root / "modules").rglob("*.py"))
    return {name: path for name, path in named if name not in POST_CACHE_MODULES}


def _post_cache_parts(root):
    """What importing each post-cache module runs, by name."""
    paths = [(name, root / name) for name in sorted(POST_CACHE_MODULES)]
    return {f"{name}#import": import_time_code(path.read_text(encoding="utf-8")).encode("utf-8") for name, path in paths if path.is_file()}


def code_parts(root):
    """Name -> bytes of everything the code key holds: the pre-cache sources, the post-cache import-time code, the entry script.

    A tree without module sources (a frozen build) raises ValueError, which bypasses the cache.
    """
    root = Path(root)
    parts = {name: path.read_bytes() for name, path in _module_sources(root).items()}
    if not parts:
        raise ValueError(f"no module sources under {root} to fingerprint")
    parts.update(_post_cache_parts(root))
    entry = root / ENTRY_SCRIPT
    if entry.is_file():
        parts[ENTRY_SCRIPT] = entry.read_bytes()
    return parts


def _parts_sha256(parts):
    """SHA-256 over the parts in name order, each framed by its name and length."""
    digest = hashlib.sha256()
    for name in sorted(parts):
        data = parts[name]
        digest.update(f"{name}\0{len(data)}\0".encode("utf-8"))
        digest.update(data)
    return digest.hexdigest()


@functools.cache
def code_fingerprint(root):
    """SHA-256 of `code_parts`; cached per process, and read at start-up when the cache is on (`prime`)."""
    return _parts_sha256(code_parts(root))


# ----------------------------------------------------------------------------- what runs outside Python


@functools.cache
def _content_sha256(path, size, mtime_ns):
    """A file's SHA-256, cached per path, size and modification time."""
    del size, mtime_ns
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(FILE_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path):
    """A file's SHA-256, read once per process while its size and modification time hold."""
    stat = Path(path).stat()
    return _content_sha256(str(path), stat.st_size, stat.st_mtime_ns)


@functools.cache
def _version_sha256(executable, size, mtime_ns):
    """The SHA-256 of `<executable> -version`; a shared-library build changes it without changing the executable."""
    del size, mtime_ns
    try:
        result = subprocess.run([executable, "-version"], capture_output=True, timeout=VERSION_TIMEOUT_S, check=False)
    except subprocess.SubprocessError as exc:
        raise RuntimeError(f"{Path(executable).name} -version did not finish: {exc}") from exc
    return hashlib.sha256(result.stdout).hexdigest()


def _resolved_executable(name_or_path):
    """The executable a configured binary names, or None when there is none."""
    path = Path(name_or_path)
    if path.is_file():
        return path
    found = shutil.which(str(name_or_path))
    return Path(found) if found else None


def binary_identity(name_or_path, with_version=False):
    """A binary by content (and, when asked, by its `-version` output); `{"absent": True}` when it is not there."""
    executable = _resolved_executable(name_or_path)
    if executable is None:
        return {"absent": True}
    identity = {"name": executable.name, "sha256": file_sha256(executable)}
    if with_version:
        stat = executable.stat()
        identity["version_sha256"] = _version_sha256(str(executable), stat.st_size, stat.st_mtime_ns)
    return identity


def binaries():
    """ffmpeg (the surgical notch) and cathar (physical repair, the subtraction), resolved as the stages resolve them."""
    return {"ffmpeg": binary_identity(utils.FFMPEG_BIN, with_version=True), "cathar": binary_identity(utils.CATHAR_BIN)}


def model_candidates(call):
    """Every model file the cached stages can load on this call, by name."""
    strategy = call.get("strategy") or {}
    names = (
        config.APL_NEURAL_MODEL,
        config.APL_MUSIC_NEURAL_MODEL,
        config.DENOISE_MODEL,
        config.DEFAULT_DENOISE_MODEL,
        spectral_denoise.DEEP_DENOISE_MODEL,
        call.get("denoise_model"),
        strategy.get("denoise_model"),
    )
    return sorted({str(name) for name in names if name})


def _model_file_wanted(path, stems):
    """A model-store file the stage may read: a parameter table, or a file a candidate's stem starts (its config beside it)."""
    return path.is_file() and (path.name in MODEL_DATA_FILES or path.name.startswith(stems))


def _model_entry(path):
    stat = path.stat()
    return [path.name, stat.st_size, stat.st_mtime_ns]


def model_identity(models_dir, names):
    """(name, size, mtime_ns) of every model-store file the named models can read, in name order.

    A re-download changes the modification time, so it misses rather than replays.
    """
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    stems = tuple(Path(name).stem for name in names)
    return [_model_entry(path) for path in _wanted_model_files(models_dir, stems)]


def _wanted_model_files(models_dir, stems):
    """The model-store files `_model_file_wanted` keeps, in name order."""
    return sorted((path for path in models_dir.iterdir() if _model_file_wanted(path, stems)), key=lambda path: path.name)


def model_store(environ=None):
    """The folder the separator loads models from: `AUDIO_SEPARATOR_MODEL_DIR` when set, else the one the stage passes."""
    environ = os.environ if environ is None else environ
    override = environ.get(MODEL_DIR_ENV)
    return Path(override) if override else Path(utils.MODELS_DIR)


def models(call):
    """The model store by its absolute path and the files in it the call can load."""
    store = model_store()
    return {"dir": os.path.abspath(store), "files": model_identity(store, model_candidates(call))}


def assets():
    """The learned per-bin blend weights, the one asset a module reads before the cached point."""
    path = Path(blend_weights.DEFAULT_WEIGHTS_PATH)
    return {"blend_weights": file_sha256(path) if path.is_file() else None}


# ----------------------------------------------------------------------------- the runtime


@functools.cache
def _distributions_sha256():
    """SHA-256 of every installed distribution's name and version, cached per process."""
    pins = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions())
    return hashlib.sha256("\n".join(pins).encode("utf-8")).hexdigest()


def torch_identity(module):
    """The torch build and the device it runs on, or None without torch."""
    if module is None:
        return None
    cuda = bool(module.cuda.is_available())
    return {
        "version": str(module.__version__),
        "cuda_available": cuda,
        "device": module.cuda.get_device_name(0) if cuda else None,
        "cuda": module.version.cuda,
        "cudnn": module.backends.cudnn.version(),
    }


@functools.cache
def gpu_driver():
    """The NVIDIA driver `nvidia-smi` reports, or None without one; a driver update can move the GPU's numerics.

    Asked once per process; a query that fails reads as None, which can only make a miss.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    command = [executable, "--query-gpu=driver_version", "--format=csv,noheader"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=VERSION_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


@functools.cache
def cpu_name():
    """The processor's name, asked once per process: numpy, torch and cathar pick their SIMD kernels by the CPU at run time."""
    return hardware.get_cpu_name()


def _keyed_env_name(name):
    upper = name.upper()
    return upper not in ENV_UNKEYED and (upper in ENV_NAMES or upper.startswith(ENV_PREFIXES))


def keyed_environment(environ=None):
    """Every variable with a prefix in ENV_PREFIXES (or named in ENV_NAMES), by name, but the cache's and the event log's own."""
    environ = os.environ if environ is None else environ
    return {name: environ[name] for name in sorted(environ) if _keyed_env_name(name)}


def runtime():
    """The interpreter, the packages, torch, the device and its driver, the CPU, the thread counts, the chunk length, the environment."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "distributions": _distributions_sha256(),
        "torch": torch_identity(torch),
        "gpu_driver": gpu_driver(),
        "hardware": {
            "cpu_threads": hardware.CPU_THREADS,
            "gpu_batch_size": hardware.GPU_BATCH_SIZE,
            "cuda_device": hardware.CUDA_DEVICE,
            "profile": hardware.PROFILE_NAME,
        },
        # Chunked and whole-file denoising differ, and the length follows host memory when the setting is 0.
        "chunk_seconds": denoise_chunking.resolved_chunk_seconds(),
        "env": keyed_environment(),
    }


def neural_call(total_duration, denoise_model, strategy, flags, pipeline_rate):
    """Every argument of the cached neural stage besides its input file: the `call` part of its key.

    `flags` are the seven stage switches (`processing.NEURAL_FLAGS`); the post-neural settings
    the step also takes are never among them.
    """
    return {
        "stage": STAGE_FULL_MIX_NEURAL,
        "total_duration": total_duration,
        "denoise_model": denoise_model,
        "strategy": strategy,
        "flags": {name: bool(value) for name, value in flags.items()},
        "pipeline_rate": pipeline_rate,
        "pcm": "pcm_f32le",
    }


def key_document(input_wav, call, root=None):
    """The full key document for one call of the cached stages."""
    return {
        "schema": KEY_SCHEMA,
        "stage": call["stage"],
        "input": input_identity(input_wav),
        "call": call,
        "config": config_identity(),
        "code": code_fingerprint(Path(root) if root else PROJECT_ROOT),
        "binaries": binaries(),
        "models": models(call),
        "assets": assets(),
        "runtime": runtime(),
    }


# ----------------------------------------------------------------------------- what this process loaded


# When processing imported this module, at start-up: the latest this process can have started.
IMPORTED_NS = time.time_ns()


def process_started_ns(now_ns=None):
    """When this process started, in ns since the epoch: psutil's reading when it is installed, else `now_ns`."""
    now_ns = time.time_ns() if now_ns is None else now_ns
    if psutil is None:
        return now_ns
    try:
        return min(now_ns, int(psutil.Process().create_time() * 1e9))
    except (psutil.Error, OSError, ValueError):
        return now_ns


def started_ns():
    """When this process started: psutil's reading, else when this module was imported."""
    return process_started_ns(IMPORTED_NS)


def prime(environ=None, root=PROJECT_ROOT):
    """With the cache on, fingerprints the code and the packages now, as this process loads them; the key reads these.

    Read at the first cached stage instead, minutes into a run, they would name whatever was on
    disk then, and an edit or an install in between would store the old code's output under
    the new code's key. Returns whether it read them; one that fails leaves them to the key,
    which bypasses the cache when it cannot read them either, so an import never fails here.
    """
    environ = os.environ if environ is None else environ
    if not environ.get(CACHE_ENV_VAR, "").strip():
        return False
    try:
        code_fingerprint(Path(root))
        _distributions_sha256()
    except (OSError, ValueError, RuntimeError, TypeError, LookupError, AttributeError):
        return False
    return True


def _mtimes_ns(root):
    """The modification time of every module source and of the entry script."""
    paths = [*Path(root, "modules").rglob("*.py"), Path(root, ENTRY_SCRIPT)]
    return [path.stat().st_mtime_ns for path in paths if path.is_file()]


def _code_drift(document, root, now_ns):
    """Why the code this process runs may not be the code the key names, or None.

    A source written after the process started may differ from what it loaded (a time past
    now is a skewed clock, not an edit); sources that hash otherwise than the key changed
    while it ran, which also covers a module imported only inside the render.
    """
    started = started_ns()
    if any(started < mtime <= now_ns for mtime in _mtimes_ns(root)):
        return "a module source was written after this process started, so it may run other code than its key names"
    if code_fingerprint.__wrapped__(root) != document["code"]:
        return "the module sources changed while it rendered"
    return None


def _package_drift(document):
    if _distributions_sha256.__wrapped__() != document["runtime"]["distributions"]:
        return "the installed packages changed while it rendered"
    return None


def drift(document, root=None, now_ns=None):
    """Why a render keyed by `document` must not be stored because the code or the packages moved under it, or None.

    Read just before storing: the key's code and package fingerprints are from start-up.
    """
    root = PROJECT_ROOT if root is None else Path(root)
    return _code_drift(document, root, time.time_ns() if now_ns is None else now_ns) or _package_drift(document)


prime()
