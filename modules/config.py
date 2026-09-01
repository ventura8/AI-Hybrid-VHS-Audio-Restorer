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
    "auto_pure_linear",
    "pure",
    "hybrid",
    "denoise_only",
    "ffmpeg_native",
    "auto_ffmpeg_native",
    "vhs_native",
    "auto_vhs_native",
    "arnndn_speech",
    "cathar",
    "cathar_vhs",
}
DEFAULT_PROCESS_MODE = "auto_pure_linear"
DEFAULT_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".mpg", ".mpeg", ".ts", ".m2ts"]

# Single source of truth for mode-specific output naming. Both the processing
# pipeline (suffix selection) and the UI file scanner (output exclusion) key off
# this map.
OUTPUT_SUFFIX_BY_MODE = {
    "auto": "_Auto_Cleaned",
    "multipass_auto": "_MultiPass_Cleaned",
    "multipass": "_MultiPass_Cleaned",
    "auto_pure": "_Pure_Cleaned",
    "auto_pure_linear": "_PureLinear_Cleaned",
    "pure": "_Pure_Cleaned",
    "hybrid": "_Hybrid_Cleaned",
    "denoise_only": "_Denoised_Cleaned",
    "ffmpeg_native": "_FFmpeg_Cleaned",
    "auto_ffmpeg_native": "_AutoFFmpeg_Cleaned",
    "vhs_native": "_FFmpeg_Cleaned",
    "auto_vhs_native": "_AutoFFmpeg_Cleaned",
    "arnndn_speech": "_Speech_Cleaned",
    "cathar": "_Cathar_Cleaned",
    "cathar_vhs": "_Cathar_Cleaned",
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


DEFAULT_DENOISE_MODEL = "UVR-DeNoise-Lite.pth"
MAX_ENHANCE_NFE = 128


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
    ("enhance_nfe", int, MAX_ENHANCE_NFE, 1),
    ("enhance_tau", float, 0.3, 0.0),
    ("dtw_resolution", int, 40, 1),
    ("afftdn_nr", float, 10.0, 0.0),
    ("afftdn_nf", float, -55.0, None),
    ("highpass_freq", int, 80, 0),
    ("notch_freq", float, 50.0, 0.0),
    ("arnndn_highpass_freq", int, 80, 0),
    ("cathar_alpha", float, 2.5, 0.0),
    ("cathar_beta", float, 0.01, 0.0),
    ("cathar_dewind_cutoff", int, 80, 0),
    ("cathar_declick_threshold", float, 8.0, 0.0),
    ("cathar_decrackle_sensitivity", int, 6, 0),
    ("cathar_declip_threshold", float, 0.95, 0.0),
    ("cathar_azimuth_max_ms", float, 5.0, 0.0),
    ("cathar_repair_strength", int, 4, 0),
    ("cathar_inpaint_max_gap_ms", int, 50, 0),
    ("cathar_inpaint_iterations", int, 3, 0),
    ("cathar_noiseprint_duration_s", float, 0.75, 0.0),
    ("cathar_dehum_harmonics", int, 8, 0),
    ("cathar_mono_below_hz", int, 100, 0),
    ("cathar_deplosive_strength", int, 4, 0),
    ("cathar_deesser_bands", int, 3, 0),
    ("cathar_deesser_freq", int, 4000, 0),
    ("cathar_deesser_threshold", float, -24.0, None),
    ("cathar_dereverb_strength", float, 2.0, 0.0),
    ("linear_air_gain_db", float, 2.0, None),
    ("adaptive_denoise_threshold_db", float, -50.0, None),
)
_BOOL_CONFIG_FIELDS = (
    ("afftdn_tn", True),
    ("enable_adeclick", True),
    ("arnndn_enable_adeclick", True),
    ("enable_multipass", True),
    ("enable_deesser", True),
    ("enable_loudnorm", True),
    ("enable_dynamic_expander", True),
    ("enable_linear_air", True),
    ("preserve_original_audio_track", False),
    ("debug_logging", False),
    ("cathar_enable_coherent", True),
    ("cathar_enable_dewind", True),
    ("cathar_enable_azimuth", True),
    ("cathar_enable_declick", True),
    ("cathar_enable_decrackle", True),
    ("cathar_enable_inpaint", True),
    ("cathar_enable_declip", True),
    ("cathar_enable_dehum", True),
    ("cathar_dehum_adaptive", True),
    ("cathar_enable_repair", True),
    ("cathar_enable_dewow", False),
    ("cathar_enable_enhance", True),
    ("cathar_enable_noiseprint", True),
    ("cathar_enable_mono_below", True),
    ("cathar_enable_deplosive", True),
    ("cathar_enable_deesser", True),
    ("cathar_enable_dereverb", False),
    ("cathar_dereverb_wpe", True),
)


def _typed_config_defaults():
    """Returns the canonical numeric and Boolean defaults for configuration loading."""
    numeric = {name: default for name, _, default, _ in _NUMERIC_CONFIG_FIELDS}
    boolean = {name: default for name, default in _BOOL_CONFIG_FIELDS}
    return {**numeric, **boolean}


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
VALID_CATHAR_DENOISE_METHODS = {"spectral", "wiener"}
VALID_CATHAR_AZIMUTH_METHODS = {"correlation", "gcc-phat"}
VALID_CATHAR_ENHANCE_METHODS = {"replicate", "interpolate"}


def _reject_config_value(param_name, raw_value, default):
    """Logs a normalisation warning and returns the field's default."""
    print(f"[Warning] Invalid {param_name}={raw_value!r}; falling back to {default}.")
    return default


def _normalize_choice(raw_value, allowlist, field_name, fallback):
    """Validates string choice against an allowlist, warning and returning fallback on mismatch."""
    if isinstance(raw_value, str) and raw_value.strip().lower() in allowlist:
        return raw_value.strip().lower()
    return _reject_config_value(field_name, raw_value, fallback)


def _normalize_cathar_denoise_method(raw_value):
    return _normalize_choice(raw_value, VALID_CATHAR_DENOISE_METHODS, "cathar_denoise_method", "spectral")


def _normalize_cathar_azimuth_method(raw_value):
    return _normalize_choice(raw_value, VALID_CATHAR_AZIMUTH_METHODS, "cathar_azimuth_method", "gcc-phat")


def _normalize_cathar_enhance_method(raw_value):
    return _normalize_choice(raw_value, VALID_CATHAR_ENHANCE_METHODS, "cathar_enhance_method", "replicate")


def _is_bad_number(val):
    """True for a float that came back as NaN or infinity."""
    return isinstance(val, float) and not math.isfinite(val)


def _is_invalid_number(val, min_val):
    """True for non-finite floats or values below the minimum bound."""
    if _is_bad_number(val):
        return True
    return min_val is not None and val < min_val


def _coerce_number(raw_value, caster, param_name, default, min_val=None):
    """Casts a config value to int/float, rejecting bools, None, non-finite, and bounds violations."""
    if raw_value is None or isinstance(raw_value, bool):
        return _reject_config_value(param_name, raw_value, default)
    try:
        val = caster(raw_value)
    except (TypeError, ValueError):
        return _reject_config_value(param_name, raw_value, default)
    if _is_invalid_number(val, min_val):
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
    for name, caster, default, min_val in _NUMERIC_CONFIG_FIELDS:
        defaults[name] = _coerce_number(defaults.get(name, default), caster, name, default, min_val=min_val)
    defaults["enhance_nfe"] = min(defaults["enhance_nfe"], MAX_ENHANCE_NFE)
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
    defaults["cathar_denoise_method"] = _normalize_cathar_denoise_method(defaults.get("cathar_denoise_method"))
    defaults["cathar_azimuth_method"] = _normalize_cathar_azimuth_method(defaults.get("cathar_azimuth_method"))
    defaults["cathar_enhance_method"] = _normalize_cathar_enhance_method(defaults.get("cathar_enhance_method"))
    defaults["vocal_mix_volume"] = _normalize_mix_volume(defaults.get("vocal_mix_volume"), "vocal_mix_volume")
    defaults["background_mix_volume"] = _normalize_mix_volume(defaults.get("background_mix_volume"), "background_mix_volume")
    _normalize_typed_config_fields(defaults)
    return True


def load_config():
    """Load bundled defaults and apply the optional user configuration."""
    defaults = {
        "vocal_mix_volume": 1.0,
        "background_mix_volume": 1.0,
        "extensions": list(DEFAULT_EXTENSIONS),
        "vocals_model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "background_model": "UVR-MDX-NET-Inst_HQ_3.onnx",
        "denoise_model": DEFAULT_DENOISE_MODEL,
        "sync_method": "shift",  # 'shift' or 'dtw'
        "process_mode": DEFAULT_PROCESS_MODE,  # includes aliases: 'multipass', 'pure', 'ffmpeg_native', 'auto_vhs_native'
        "arnndn_model": "cb.rnnn",
        "cathar_denoise_method": "spectral",
        "cathar_azimuth_method": "gcc-phat",
        "cathar_enhance_method": "replicate",
    }
    defaults.update(_typed_config_defaults())
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
ADAPTIVE_DENOISE_THRESHOLD_DB = float(CONFIG.get("adaptive_denoise_threshold_db", -50.0))
ENHANCE_NFE = str(CONFIG["enhance_nfe"])
ENHANCE_TAU = str(CONFIG["enhance_tau"])
SYNC_METHOD = CONFIG["sync_method"]
DTW_RESOLUTION = int(CONFIG["dtw_resolution"])
PROCESS_MODE = CONFIG["process_mode"]
ENABLE_MULTIPASS = bool(CONFIG.get("enable_multipass", True))
DEBUG_LOGGING = CONFIG.get("debug_logging", False)

# Native VHS filter configs
AFFTDN_NR = float(CONFIG.get("afftdn_nr", 10.0))
AFFTDN_NF = float(CONFIG.get("afftdn_nf", -55.0))
AFFTDN_TN = bool(CONFIG.get("afftdn_tn", True))
HIGHPASS_FREQ = int(CONFIG.get("highpass_freq", 80))
ENABLE_ADECLICK = bool(CONFIG.get("enable_adeclick", True))
NOTCH_FREQ = float(CONFIG.get("notch_freq", 50.0))

# ARNNDN Speech configs
ARNNDN_MODEL = str(CONFIG.get("arnndn_model", "cb.rnnn"))
ARNNDN_HIGHPASS_FREQ = int(CONFIG.get("arnndn_highpass_freq", 80))
ARNNDN_ENABLE_ADECLICK = bool(CONFIG.get("arnndn_enable_adeclick", True))

# Cathar Restoration Settings
CATHAR_DENOISE_METHOD = str(CONFIG.get("cathar_denoise_method", "spectral"))
CATHAR_ALPHA = float(CONFIG.get("cathar_alpha", 2.5))
CATHAR_BETA = float(CONFIG.get("cathar_beta", 0.01))
CATHAR_ENABLE_COHERENT = bool(CONFIG.get("cathar_enable_coherent", True))
CATHAR_ENABLE_DEWIND = bool(CONFIG.get("cathar_enable_dewind", True))
CATHAR_DEWIND_CUTOFF = int(CONFIG.get("cathar_dewind_cutoff", 80))
CATHAR_ENABLE_AZIMUTH = bool(CONFIG.get("cathar_enable_azimuth", True))
CATHAR_AZIMUTH_METHOD = str(CONFIG.get("cathar_azimuth_method", "gcc-phat"))
CATHAR_AZIMUTH_MAX_MS = float(CONFIG.get("cathar_azimuth_max_ms", 5.0))
CATHAR_ENABLE_DECLICK = bool(CONFIG.get("cathar_enable_declick", True))
CATHAR_DECLICK_THRESHOLD = float(CONFIG.get("cathar_declick_threshold", 8.0))
CATHAR_ENABLE_DECRACKLE = bool(CONFIG.get("cathar_enable_decrackle", True))
CATHAR_DECRACKLE_SENSITIVITY = int(CONFIG.get("cathar_decrackle_sensitivity", 6))
CATHAR_ENABLE_INPAINT = bool(CONFIG.get("cathar_enable_inpaint", True))
CATHAR_INPAINT_MAX_GAP_MS = int(CONFIG.get("cathar_inpaint_max_gap_ms", 50))
CATHAR_INPAINT_ITERATIONS = int(CONFIG.get("cathar_inpaint_iterations", 3))
CATHAR_ENABLE_DECLIP = bool(CONFIG.get("cathar_enable_declip", True))
CATHAR_DECLIP_THRESHOLD = float(CONFIG.get("cathar_declip_threshold", 0.95))
CATHAR_ENABLE_DEHUM = bool(CONFIG.get("cathar_enable_dehum", True))
CATHAR_DEHUM_ADAPTIVE = bool(CONFIG.get("cathar_dehum_adaptive", True))
CATHAR_DEHUM_HARMONICS = int(CONFIG.get("cathar_dehum_harmonics", 8))
CATHAR_ENABLE_REPAIR = bool(CONFIG.get("cathar_enable_repair", True))
CATHAR_REPAIR_STRENGTH = int(CONFIG.get("cathar_repair_strength", 4))
CATHAR_ENABLE_DEWOW = bool(CONFIG.get("cathar_enable_dewow", False))
CATHAR_ENABLE_ENHANCE = bool(CONFIG.get("cathar_enable_enhance", True))
CATHAR_ENHANCE_METHOD = str(CONFIG.get("cathar_enhance_method", "replicate"))
CATHAR_ENABLE_NOISEPRINT = bool(CONFIG.get("cathar_enable_noiseprint", True))
CATHAR_NOISEPRINT_DURATION_S = float(CONFIG.get("cathar_noiseprint_duration_s", 0.75))
CATHAR_ENABLE_MONO_BELOW = bool(CONFIG.get("cathar_enable_mono_below", True))
CATHAR_MONO_BELOW_HZ = int(CONFIG.get("cathar_mono_below_hz", 100))
CATHAR_ENABLE_DEPLOSIVE = bool(CONFIG.get("cathar_enable_deplosive", True))
CATHAR_DEPLOSIVE_STRENGTH = int(CONFIG.get("cathar_deplosive_strength", 4))
CATHAR_ENABLE_DEESSER = bool(CONFIG.get("cathar_enable_deesser", True))
CATHAR_DEESSER_BANDS = int(CONFIG.get("cathar_deesser_bands", 3))
CATHAR_DEESSER_FREQ = int(CONFIG.get("cathar_deesser_freq", 4000))
CATHAR_DEESSER_THRESHOLD = float(CONFIG.get("cathar_deesser_threshold", -24.0))
CATHAR_ENABLE_DEREVERB = bool(CONFIG.get("cathar_enable_dereverb", False))
CATHAR_DEREVERB_WPE = bool(CONFIG.get("cathar_dereverb_wpe", True))
CATHAR_DEREVERB_STRENGTH = float(CONFIG.get("cathar_dereverb_strength", 2.0))

# Advanced Audio Polish & Archival Configs
ENABLE_DEESSER = bool(CONFIG.get("enable_deesser", True))
ENABLE_LOUDNORM = bool(CONFIG.get("enable_loudnorm", True))
ENABLE_DYNAMIC_EXPANDER = bool(CONFIG.get("enable_dynamic_expander", True))
ENABLE_LINEAR_AIR = bool(CONFIG.get("enable_linear_air", True))
LINEAR_AIR_GAIN_DB = float(CONFIG.get("linear_air_gain_db", 2.0))
PRESERVE_ORIGINAL_AUDIO_TRACK = bool(CONFIG.get("preserve_original_audio_track", False))
