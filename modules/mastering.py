"""FFmpeg loudness normalization and container audio mastering utilities.

Provides two-pass EBU R128 loudness measurement, true-peak limiting,
and container-specific codec resolution for lossless video remuxing.
"""

import json
import math
import subprocess

from . import config
from .config import _parse_mix_float
from .utils import FFMPEG_BIN, log_msg

ENABLE_LOUDNORM = config.ENABLE_LOUDNORM
VOCAL_MIX_VOL = config.VOCAL_MIX_VOL
BACKGROUND_MIX_VOL = config.BACKGROUND_MIX_VOL


def _get_config_val(name):
    """Retrieves the current configuration value."""
    return getattr(config, name)


PIPELINE_SAMPLE_RATE = 44100
LOUDNORM_TARGET_I = -16.0
LOUDNORM_TARGET_TP = -1.0
LOUDNORM_TARGET_LRA = 11.0
LOUDNORM_ANALYSIS_TIMEOUT = 900

LOUDNORM_TRUE_PEAK_LIMITER = "alimiter=limit=0.891:level=disabled"
LOUDNORM_MEASURE_KEYS = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")

AUDIO_CODEC_ARGS_BY_EXT = {
    ".mp4": ["-c:a", "aac", "-b:a", "320k"],
    ".m4v": ["-c:a", "aac", "-b:a", "320k"],
    ".mpg": ["-c:a", "mp2", "-b:a", "384k"],
    ".mpeg": ["-c:a", "mp2", "-b:a", "384k"],
    ".ts": ["-c:a", "aac", "-b:a", "320k"],
    ".m2ts": ["-c:a", "aac", "-b:a", "320k"],
    ".avi": ["-c:a", "pcm_s16le"],
}


def _resolve_override(override, fallback):
    """Returns an explicit pipeline override when provided, else the configured fallback."""
    return fallback if override is None else override


def _get_audio_encoding_args(video_suffix):
    """Returns codec arguments for transparent remuxing.

    Args:
        video_suffix (str): Container extension (e.g. '.mp4', '.mkv', '.avi').

    Returns:
        list: FFmpeg audio codec arguments.
    """
    return AUDIO_CODEC_ARGS_BY_EXT.get(video_suffix.lower(), ["-c:a", "pcm_f32le"])


def _scope_audio_arg(arg, prefix):
    """Appends stream prefix to audio codec or bitrate flags."""
    if arg in ("-c:a", "-b:a"):
        return f"{arg}{prefix}"
    return arg


def _scope_audio_args_for_stream(audio_args, stream_index=0):
    """Scopes audio flags like -c:a and -b:a to a specific audio output stream index."""
    prefix = f":{stream_index}"
    return [_scope_audio_arg(arg, prefix) for arg in audio_args]


def _preserved_audio_args(video_suffix, audio_args):
    """Returns a compatible codec configuration for the preserved source stream."""
    if video_suffix.lower() in {".avi", ".mpg", ".mpeg"}:
        return _scope_audio_args_for_stream(audio_args, 1)
    return ["-c:a:1", "copy"]


def _sanitize_mix_level(vol_val):
    """Normalizes a configured mix volume, falling back to unity gain."""
    val = _parse_mix_float(vol_val)
    return 1.0 if val is None else val


def _loudnorm_analysis_timeout(total_duration):
    """Scales the analysis-pass timeout with media duration, never below the floor."""
    if not total_duration or total_duration <= 0:
        return LOUDNORM_ANALYSIS_TIMEOUT
    return max(LOUDNORM_ANALYSIS_TIMEOUT, int(total_duration * 8))


def _loudnorm_target_args():
    """Loudness target shared by the measurement pass and the applied pass."""
    return f"I={LOUDNORM_TARGET_I}:TP={LOUDNORM_TARGET_TP}:LRA={LOUDNORM_TARGET_LRA}"


def _is_valid_loudnorm_number(value):
    """Returns True if value can be parsed as a finite float (rejecting nan, inf, -inf)."""
    try:
        val = float(value)
        return math.isfinite(val)
    except (ValueError, TypeError):
        return False


def _has_loudnorm_measurements(measurements):
    """Confirms an analysis block carries every required finite value the applied pass needs."""
    return all(key in measurements and _is_valid_loudnorm_number(measurements[key]) for key in LOUDNORM_MEASURE_KEYS)


def _extract_valid_loudnorm_object(text, decoder):
    """Attempts to decode a valid loudnorm measurement object from text."""
    try:
        measurements, _ = decoder.raw_decode(text)
        return measurements if isinstance(measurements, dict) and _has_loudnorm_measurements(measurements) else None
    except ValueError:
        return None


def _parse_loudnorm_json(stderr_text):
    """Extracts the measurement block that loudnorm's analysis pass prints."""
    pos = 0
    decoder = json.JSONDecoder()
    while (start := stderr_text.find("{", pos)) >= 0:
        if (measurements := _extract_valid_loudnorm_object(stderr_text[start:], decoder)) is not None:
            return measurements
        pos = start + 1
    return None


def _measured_loudnorm_args(measurements):
    """Builds second-pass loudnorm arguments from the measured programme values."""
    return (
        f"{_loudnorm_target_args()}"
        f":measured_I={measurements['input_i']}"
        f":measured_TP={measurements['input_tp']}"
        f":measured_LRA={measurements['input_lra']}"
        f":measured_thresh={measurements['input_thresh']}"
        f":offset={measurements['target_offset']}"
        ":linear=true"
    )


def _build_mix_base_expression(vocal_mix_vol, bg_mix_vol):
    """Volume-scaled two-stem amix, with no loudness stage attached."""
    vocal_vol = _sanitize_mix_level(_resolve_override(vocal_mix_vol, _get_config_val("VOCAL_MIX_VOL")))
    bg_vol = _sanitize_mix_level(_resolve_override(bg_mix_vol, _get_config_val("BACKGROUND_MIX_VOL")))
    return f"[1:a]volume={vocal_vol}[v];[2:a]volume={bg_vol}[b];[v][b]amix=inputs=2:duration=first:dropout_transition=0:normalize=0"


def _mastering_chain(loudnorm_args, label):
    """Applied loudness chain shared by the mix and single-track paths."""
    applied = _resolve_override(loudnorm_args, _loudnorm_target_args())
    return f"loudnorm={applied},aresample={PIPELINE_SAMPLE_RATE},{LOUDNORM_TRUE_PEAK_LIMITER}[{label}]"


def _build_mix_filter_expression(vocal_mix_vol=None, bg_mix_vol=None, loudnorm_args=None):
    """Builds amix expression optionally with EBU R128 loudness normalization."""
    base_filter = _build_mix_base_expression(vocal_mix_vol, bg_mix_vol)
    if not _get_config_val("ENABLE_LOUDNORM"):
        return f"{base_filter}[mixed]"
    return f"{base_filter},{_mastering_chain(loudnorm_args, 'mixed')}"


def _run_loudness_analysis(input_paths, expression, total_duration=None):
    """Runs loudnorm's analysis pass over a built filter graph."""
    cmd = [FFMPEG_BIN, "-hide_banner", "-nostdin"]
    for path in input_paths:
        cmd.extend(["-i", str(path)])
    cmd.extend(["-filter_complex", expression, "-map", "[mastered]", "-f", "null", "-"])
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=_loudnorm_analysis_timeout(total_duration))
    except Exception:
        return None
    return _parse_loudnorm_json(completed.stderr or "")


def _resolve_measured_loudnorm(input_paths, analysis_expression, total_duration=None):
    """Measures programme loudness so normalisation can be applied accurately."""
    if not _get_config_val("ENABLE_LOUDNORM"):
        return None
    log_msg("    [Mastering] Measuring programme loudness (pass 1 of 2)...")
    measurements = _run_loudness_analysis(input_paths, analysis_expression, total_duration=total_duration)
    if measurements is None:
        log_msg("    [Warning] Loudness measurement unavailable; using single-pass normalisation.", is_error=True)
        return None
    return _measured_loudnorm_args(measurements)


def _build_single_audio_filter_expression(loudnorm_args=None):
    """Mastering graph for single-track modes, matching the two-stem mix path."""
    if not _get_config_val("ENABLE_LOUDNORM"):
        return None
    return f"[1:a]{_mastering_chain(loudnorm_args, 'mastered')}"


def _resolve_loudnorm_args(video_path, aligned_vocals, aligned_background, vocal_mix_vol, bg_mix_vol, total_duration=None):
    """Measures the finished two-stem mix before normalisation is applied."""
    base_filter = _build_mix_base_expression(vocal_mix_vol, bg_mix_vol)
    expression = f"{base_filter},loudnorm={_loudnorm_target_args()}:print_format=json[mastered]"
    return _resolve_measured_loudnorm((video_path, aligned_vocals, aligned_background), expression, total_duration=total_duration)


def _resolve_single_track_loudnorm_args(video_path, processed_audio_wav, total_duration=None):
    """Measures a single processed track before normalisation is applied."""
    expression = f"[1:a]loudnorm={_loudnorm_target_args()}:print_format=json[mastered]"
    return _resolve_measured_loudnorm((video_path, processed_audio_wav), expression, total_duration=total_duration)
