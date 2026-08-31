import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

VALID_PROCESS_MODES = {
    "auto",
    "multipass_auto",
    "multipass",
    "auto_pure",
    "pure",
    "hybrid",
    "denoise_only",
    "ffmpeg_native",
    "auto_ffmpeg_native",
    "vhs_native",
    "auto_vhs_native",
    "arnndn_speech",
}
DEFAULT_PROCESS_MODE = "auto_pure"
DEFAULT_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".mpg", ".mpeg", ".ts", ".m2ts"]

# Single source of truth for mode-specific output naming. Both the processing
# pipeline (suffix selection) and the UI file scanner (output exclusion) key off
# this map.
OUTPUT_SUFFIX_BY_MODE = {
    "auto": "_Auto_Cleaned",
    "multipass_auto": "_MultiPass_Cleaned",
    "multipass": "_MultiPass_Cleaned",
    "auto_pure": "_Pure_Cleaned",
    "pure": "_Pure_Cleaned",
    "hybrid": "_Hybrid_Cleaned",
    "denoise_only": "_Denoised_Cleaned",
    "ffmpeg_native": "_FFmpeg_Cleaned",
    "auto_ffmpeg_native": "_AutoFFmpeg_Cleaned",
    "vhs_native": "_FFmpeg_Cleaned",
    "auto_vhs_native": "_AutoFFmpeg_Cleaned",
    "arnndn_speech": "_Speech_Cleaned",
}

# Legacy VHS suffixes from earlier releases, retained so the scanner keeps
# excluding those outputs even though no current mode emits them.
_LEGACY_CLEANED_OUTPUT_SUFFIXES = ("_VHS_Cleaned", "_AutoVHS_Cleaned")
CLEANED_OUTPUT_SUFFIXES = tuple(dict.fromkeys(OUTPUT_SUFFIX_BY_MODE.values())) + _LEGACY_CLEANED_OUTPUT_SUFFIXES


def _config_paths():
    """Returns launch-directory and bundled configuration paths in priority order."""
    launch_config = Path.cwd() / "config.yaml"
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    bundled_config = bundle_root / "config.yaml"
    return (launch_config, bundled_config)


def _normalize_process_mode(raw_value):
    if not isinstance(raw_value, str):
        print(f"[Warning] Invalid process_mode={raw_value!r}; falling back to '{DEFAULT_PROCESS_MODE}'.")
        return DEFAULT_PROCESS_MODE

    normalized = raw_value.strip().lower()
    if normalized in VALID_PROCESS_MODES:
        return normalized

    print(f"[Warning] Invalid process_mode={raw_value!r} (normalized={normalized!r}); falling back to '{DEFAULT_PROCESS_MODE}'.")
    return DEFAULT_PROCESS_MODE


def _parse_mix_float(raw_value):
    try:
        val = float(raw_value)
        if math.isfinite(val) and 0.0 <= val <= 10.0:
            return round(val, 4)
    except (TypeError, ValueError):
        pass
    return None


def _normalize_mix_volume(raw_value, param_name="mix_volume", default=1.0):
    val = _parse_mix_float(raw_value)
    if val is not None:
        return val
    print(f"[Warning] Invalid {param_name}={raw_value!r}; falling back to {default}.")
    return default


_NUMERIC_CONFIG_FIELDS = (
    ("enhance_nfe", int, 128),
    ("enhance_tau", float, 0.3),
    ("dtw_resolution", int, 40),
    ("afftdn_nr", float, 12.0),
    ("afftdn_nf", float, -45.0),
    ("highpass_freq", int, 60),
    ("notch_freq", float, 0.0),
    ("arnndn_highpass_freq", int, 60),
)
_BOOL_CONFIG_FIELDS = (
    ("afftdn_tn", True),
    ("enable_adeclick", True),
    ("arnndn_enable_adeclick", True),
    ("enable_multipass", True),
    ("enable_deesser", True),
    ("enable_loudnorm", True),
    ("enable_dynamic_expander", True),
    ("preserve_original_audio_track", False),
    ("debug_logging", False),
)
_BOOL_STRINGS = {
    "1": True,
    "true": True,
    "yes": True,
    "on": True,
    "y": True,
    "t": True,
    "0": False,
    "false": False,
    "no": False,
    "off": False,
    "n": False,
    "f": False,
    "": False,
}


def _reject_config_value(param_name, raw_value, default):
    """Logs a normalisation warning and returns the field's default."""
    print(f"[Warning] Invalid {param_name}={raw_value!r}; falling back to {default}.")
    return default


def _is_bad_number(val):
    """True for a float that came back as NaN or infinity."""
    return isinstance(val, float) and not math.isfinite(val)


def _coerce_number(raw_value, caster, param_name, default):
    """Casts a config value to int/float, rejecting bools, None, and non-finite results."""
    if raw_value is None or isinstance(raw_value, bool):
        return _reject_config_value(param_name, raw_value, default)
    try:
        val = caster(raw_value)
    except (TypeError, ValueError):
        return _reject_config_value(param_name, raw_value, default)
    if _is_bad_number(val):
        return _reject_config_value(param_name, raw_value, default)
    return val


def _coerce_bool(raw_value, param_name, default):
    """Parses a config Boolean by content, so quoted 'false'/'0' resolve to False."""
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    if isinstance(raw_value, str):
        token = raw_value.strip().lower()
        if token in _BOOL_STRINGS:
            return _BOOL_STRINGS[token]
    return _reject_config_value(param_name, raw_value, default)


def _normalize_typed_config_fields(defaults):
    """Normalizes every typed numeric/Boolean field before module-level conversion."""
    for name, caster, default in _NUMERIC_CONFIG_FIELDS:
        defaults[name] = _coerce_number(defaults.get(name, default), caster, name, default)
    for name, default in _BOOL_CONFIG_FIELDS:
        defaults[name] = _coerce_bool(defaults.get(name, default), name, default)


def _find_config_path():
    """Returns the first configuration file that exists, or None when there is none."""
    for path in _config_paths():
        if path.exists():
            return path
    return None


def _read_user_config(config_path):
    """Parses a configuration file, returning None when it cannot be read."""
    try:
        with open(config_path, "r") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        print(f"[Warning] Failed to load config.yaml: {exc}")
        return None


def _apply_user_config(defaults, user_config):
    """Merges parsed configuration over the defaults, sanitising every value."""
    if not isinstance(user_config, Mapping):
        return False
    defaults.update(user_config)
    extensions = defaults.get("extensions")
    if isinstance(extensions, (str, bytes)) or not isinstance(extensions, (list, tuple, set)):
        print(f"[Warning] Invalid extensions={extensions!r}; falling back to default extensions.")
        defaults["extensions"] = list(DEFAULT_EXTENSIONS)
    else:
        defaults["extensions"] = list(extensions)
    defaults["process_mode"] = _normalize_process_mode(defaults.get("process_mode"))
    defaults["vocal_mix_volume"] = _normalize_mix_volume(defaults.get("vocal_mix_volume"), "vocal_mix_volume")
    defaults["background_mix_volume"] = _normalize_mix_volume(defaults.get("background_mix_volume"), "background_mix_volume")
    _normalize_typed_config_fields(defaults)
    return True


def load_config():
    defaults = {
        "vocal_mix_volume": 1.0,
        "background_mix_volume": 1.0,
        "extensions": list(DEFAULT_EXTENSIONS),
        "vocals_model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "background_model": "UVR-MDX-NET-Inst_HQ_3.onnx",
        "denoise_model": "UVR-DeNoise-Lite.pth",
        "enhance_nfe": 128,
        "enhance_tau": 0.3,
        "sync_method": "shift",  # 'shift' or 'dtw'
        "dtw_resolution": 40,  # Analysis resolution in Hz (40Hz = 25ms, Sufficient for Lipsync)
        "process_mode": DEFAULT_PROCESS_MODE,  # includes aliases: 'multipass', 'pure', 'ffmpeg_native', 'auto_vhs_native'
        "enable_multipass": True,
        "afftdn_nr": 12.0,
        "afftdn_nf": -45.0,
        "afftdn_tn": True,
        "highpass_freq": 60,
        "enable_adeclick": True,
        "notch_freq": 0.0,
        "arnndn_model": "cb.rnnn",
        "arnndn_highpass_freq": 60,
        "arnndn_enable_adeclick": True,
        "enable_deesser": True,
        "enable_loudnorm": True,
        "enable_dynamic_expander": True,
        "preserve_original_audio_track": False,
        "debug_logging": False,
    }
    config_path = _find_config_path()
    if config_path is None:
        return defaults, "Defaults"

    if yaml is None:
        print("[Warning] config.yaml exists but PyYAML is not installed; using defaults.")
        return defaults, "Defaults (PyYAML missing)"

    user_config = _read_user_config(config_path)
    if not user_config:
        return defaults, "Defaults"

    if not _apply_user_config(defaults, user_config):
        return defaults, "Defaults (invalid config.yaml)"
    return defaults, "config.yaml"


CONFIG, CONFIG_SOURCE = load_config()

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
LOG_FILE = Path("session_log.txt")

EXTS = set(CONFIG["extensions"])
KEEP_INPUT_FILES = os.environ.get("AI_RESTORE_TEST_MODE") == "1"

# Audio mix levels
VOCAL_MIX_VOL = float(CONFIG["vocal_mix_volume"])
BACKGROUND_MIX_VOL = float(CONFIG["background_mix_volume"])

# AI Configs
VOCALS_MODEL = CONFIG["vocals_model"]
BACKGROUND_MODEL = CONFIG["background_model"]
DENOISE_MODEL = CONFIG["denoise_model"]
ENHANCE_NFE = str(CONFIG["enhance_nfe"])
ENHANCE_TAU = str(CONFIG["enhance_tau"])
SYNC_METHOD = CONFIG["sync_method"]
DTW_RESOLUTION = int(CONFIG["dtw_resolution"])
PROCESS_MODE = CONFIG["process_mode"]
ENABLE_MULTIPASS = bool(CONFIG.get("enable_multipass", True))
DEBUG_LOGGING = CONFIG.get("debug_logging", False)

# Native VHS filter configs
AFFTDN_NR = float(CONFIG.get("afftdn_nr", 12.0))
AFFTDN_NF = float(CONFIG.get("afftdn_nf", -45.0))
AFFTDN_TN = bool(CONFIG.get("afftdn_tn", True))
HIGHPASS_FREQ = int(CONFIG.get("highpass_freq", 60))
ENABLE_ADECLICK = bool(CONFIG.get("enable_adeclick", True))
NOTCH_FREQ = float(CONFIG.get("notch_freq", 0.0))

# ARNNDN Speech configs
ARNNDN_MODEL = str(CONFIG.get("arnndn_model", "cb.rnnn"))
ARNNDN_HIGHPASS_FREQ = int(CONFIG.get("arnndn_highpass_freq", 60))
ARNNDN_ENABLE_ADECLICK = bool(CONFIG.get("arnndn_enable_adeclick", True))

# Advanced Audio Polish & Archival Configs
ENABLE_DEESSER = bool(CONFIG.get("enable_deesser", True))
ENABLE_LOUDNORM = bool(CONFIG.get("enable_loudnorm", True))
ENABLE_DYNAMIC_EXPANDER = bool(CONFIG.get("enable_dynamic_expander", True))
PRESERVE_ORIGINAL_AUDIO_TRACK = bool(CONFIG.get("preserve_original_audio_track", False))
