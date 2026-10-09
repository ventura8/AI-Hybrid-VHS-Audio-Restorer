"""FFmpeg loudness normalization and container audio mastering utilities.

Provides two-pass EBU R128 loudness measurement, peak limiting,
and container-specific codec resolution for lossless video remuxing.

Linear or dynamic. loudnorm's applied pass keeps its linear mode (one gain for the whole
programme) only while ffmpeg's own rule holds: the measured true peak plus the gain to the
target stays at or under the true-peak target, TP + (I_target - I) <= TP_target, the
measured range stays at or under the range target, and the measurement is set (I and LRA
not 0, TP not 99, threshold not -70). Otherwise it rides the gain dynamically, lifting the
pause floors after every engine has finished. One case overrides all of that: a programme
shorter than loudnorm's 3 s frame buffer runs linear whatever was measured (ffmpeg forces
it; such a programme also measures LRA 0, the unset value). The run log used to check the
range half only: of the 616 distinct decisions it logged as "linear" in the 1020 run logs
under the repository on 2026-10-09, 229 (37%) broke the true-peak half and ran dynamic.
The count moves with the logs on disk; `linear_mode_blockers` over the logged I, TP and
LRA re-derives it. That function is the rule with its short-programme case, which applies
when the caller passes the duration (the mux passes the one it knows); on synthetic
programmes it reproduced the `normalization_type` ffmpeg 8.0.1 reported on all nine cases
checked (true peak over and under the target, range over and under it, a steady tone
measuring LRA 0), and a tone given measurements that rule out linear ran linear at 2.9 s
and dynamic at 3.0 s on ffmpeg 8.0.1 and 9.0.1.

`loudnorm_linear_fallback: gain_limiter` (default `ffmpeg`, which keeps every output's
bytes) answers the true-peak-only case with `volume=(I_target - I)dB` and the mux's
existing limiter after it, in place of the dynamic ride; a range over its target still
goes to loudnorm. That limiter (`LOUDNORM_TRUE_PEAK_LIMITER`, a name older than this
fallback) is alimiter at 0.891, -1 dBFS on the samples at 44.1 kHz, not a true-peak
limiter, so the gain can leave intersample overs. On three synthetic programmes with
clicks, given +7.0, +28.9 and +11.4 dB, the fallback's output read +0.24, +0.24 and -0.41
dBTP against the -1 dBTP target, where loudnorm's dynamic mode through the same tail held
-1.8 to -1.9 dBTP and the same limiter run at 176.4 kHz held -1.6 to -2.0. On a synthetic
programme with one spike the gain-plus-limiter render read -16.5 LUFS, LRA 0.1 LU, against
-16.4 LUFS, LRA 0.5 LU for loudnorm's dynamic mode.

With `AI_RESTORE_EVENT_LOG=<dir>` set, the mux also measures what its loudness stage
renders, in one more null-sink pass with `print_format=json`. For loudnorm that is the
applied pass with the graph and measured arguments the mux renders with, so ffmpeg reports
its own `normalization_type` and output readings. For the gain_limiter fallback it is the
fallback's whole chain (gain, resample, limiter) read by loudnorm's analysis pass, whose
input readings, true peak included, are the render's output; a true peak over the target
is logged as a warning. The record is written as
`loudnorm__<recording>__<mode>__<track>.json` (`modules/event_log.py`) into the event-log
folder, which outlives the run, and under the same name beside the track the mux masters
(the work directory is removed after a successful run unless `AI_RESTORE_TEST_MODE=1`
keeps it). Unset, no extra pass runs and nothing is written.
"""

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import config, event_log
from .config import _parse_mix_float
from .utils import FFMPEG_BIN, log_msg

ENABLE_LOUDNORM = config.ENABLE_LOUDNORM
VOCAL_MIX_VOL = config.VOCAL_MIX_VOL
BACKGROUND_MIX_VOL = config.BACKGROUND_MIX_VOL


def _get_config_val(name):
    """Retrieves the current configuration value for current and legacy callers."""
    return getattr(config, name)


PIPELINE_SAMPLE_RATE = 44100
LOUDNORM_TARGET_I = -16.0
LOUDNORM_TARGET_TP = -1.0
LOUDNORM_TARGET_LRA = config.LOUDNORM_TARGET_LRA
LOUDNORM_ANALYSIS_TIMEOUT = 900

LOUDNORM_TRUE_PEAK_LIMITER = "alimiter=limit=0.891:level=disabled"
LOUDNORM_MEASURE_KEYS = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
# loudnorm's own "not measured" values: an applied pass given any of them never runs linear.
LOUDNORM_UNSET_VALUES = (("input_i", 0.0), ("input_tp", 99.0), ("input_lra", 0.0), ("input_thresh", -70.0))
# loudnorm's frame buffer: a programme shorter than this runs linear whatever was measured.
LOUDNORM_SHORT_PROGRAMME_S = 3.0
BLOCKER_UNREADABLE = "unreadable"
BLOCKER_LRA = "lra"
BLOCKER_TRUE_PEAK = "true_peak"
BLOCKER_TEXT = {
    BLOCKER_UNREADABLE: "measurement unreadable",
    BLOCKER_LRA: "measured LRA above the target",
    BLOCKER_TRUE_PEAK: "true peak after the gain above the target",
}
GAIN_LIMITER = "gain_limiter"
LOUDNORM_STAGE = "loudnorm"
APPLIED_READINGS = ("normalization_type", "output_i", "output_tp", "output_lra", "output_thresh")
# The fallback's render read by loudnorm's analysis pass: its input readings are the render's output.
FALLBACK_READINGS = (("output_i", "input_i"), ("output_tp", "input_tp"), ("output_lra", "input_lra"), ("output_thresh", "input_thresh"))

AUDIO_CODEC_ARGS_BY_EXT = {
    ".mp4": ["-c:a", "aac", "-b:a", "320k"],
    ".m4v": ["-c:a", "aac", "-b:a", "320k"],
    ".mpg": ["-c:a", "mp2", "-b:a", "384k"],
    ".mpeg": ["-c:a", "mp2", "-b:a", "384k"],
    ".ts": ["-c:a", "aac", "-b:a", "320k"],
    ".m2ts": ["-c:a", "aac", "-b:a", "320k"],
    ".avi": ["-c:a", "pcm_s16le"],
}


@dataclass(frozen=True)
class LinearGain:
    """The gain_limiter fallback's loudness stage: one gain to the target, with the mux's sample-peak limiter after it."""

    gain_db: float

    def filter(self):
        """The stage as an FFmpeg filter."""
        return f"volume={self.gain_db:.2f}dB"


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
    """Loudness target shared by the measurement pass and the applied pass.

    The LRA target comes from the config (`loudnorm_target_lra`): loudnorm holds its linear
    mode only while the measured range is at or under it, and a denoised interview with
    silent pauses measures well above the broadcast 11 LU, so the mux would otherwise ride
    the gain between words.
    """
    return f"I={LOUDNORM_TARGET_I}:TP={LOUDNORM_TARGET_TP}:LRA={float(_get_config_val('LOUDNORM_TARGET_LRA'))}"


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


def _loudness_stage(applied):
    """The chain's loudness filter: loudnorm with its arguments, or the gain_limiter fallback's single gain."""
    if isinstance(applied, LinearGain):
        return applied.filter()
    return f"loudnorm={applied}"


def _mastering_filters(applied):
    """The applied loudness chain without its output label: the loudness stage, the resample, the limiter."""
    return f"{_loudness_stage(applied)},aresample={PIPELINE_SAMPLE_RATE},{LOUDNORM_TRUE_PEAK_LIMITER}"


def _mastering_chain(loudnorm_args, label):
    """Applied loudness chain shared by the mix and single-track paths."""
    applied = _resolve_override(loudnorm_args, _loudnorm_target_args())
    return f"{_mastering_filters(applied)}[{label}]"


def _build_mix_filter_expression(vocal_mix_vol=None, bg_mix_vol=None, loudnorm_args=None):
    """Builds amix expression optionally with EBU R128 loudness normalization."""
    base_filter = _build_mix_base_expression(vocal_mix_vol, bg_mix_vol)
    if not _get_config_val("ENABLE_LOUDNORM"):
        return f"{base_filter}[mixed]"
    return f"{base_filter},{_mastering_chain(loudnorm_args, 'mixed')}"


def _run_loudness_analysis(input_paths, expression, total_duration=None):
    """Runs loudnorm's analysis pass over a built filter graph."""
    cmd = [FFMPEG_BIN, "-hide_banner", "-nostdin", "-nostats"]
    for path in input_paths:
        cmd.extend(["-i", str(path)])
    cmd.extend(["-filter_complex", expression, "-map", "[mastered]", "-f", "null", "-"])
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=_loudnorm_analysis_timeout(total_duration))
    except Exception:
        return None
    return _parse_loudnorm_json(completed.stderr or "")


def _loudnorm_graph(graph_prefix, loudnorm_args):
    """A measuring graph: the programme through loudnorm with these arguments, its readings printed as JSON."""
    return f"{graph_prefix}loudnorm={loudnorm_args}:print_format=json[mastered]"


def _resolve_measured_loudnorm(input_paths, graph_prefix, total_duration=None):
    """Measures programme loudness so normalisation can be applied accurately.

    `graph_prefix` is the graph up to the loudness stage (the mix, or the single track's
    label). Returns the applied pass's loudness stage: loudnorm's measured arguments, or the
    gain_limiter fallback's LinearGain; None when loudness is off or cannot be measured.
    """
    if not _get_config_val("ENABLE_LOUDNORM"):
        return None
    log_msg("    [Mastering] Measuring programme loudness (pass 1 of 2)...")
    analysis = _loudnorm_graph(graph_prefix, _loudnorm_target_args())
    measurements = _run_loudness_analysis(input_paths, analysis, total_duration=total_duration)
    if measurements is None:
        log_msg("    [Warning] Loudness measurement unavailable; using single-pass normalisation.", is_error=True)
        return None
    applied = _applied_loudness(measurements, _log_loudness_range(measurements, total_duration))
    _record_loudnorm(input_paths, graph_prefix, measurements, applied, total_duration)
    return applied


def _rule_value(measurements, key, unset):
    """One measured value as a float, or None when it is missing, not finite, or loudnorm's own 'not measured' value."""
    raw = measurements.get(key)
    if not _is_valid_loudnorm_number(raw) or float(raw) == unset:
        return None
    return float(raw)


def _short_programme(duration):
    """True when a known programme duration (seconds) is under loudnorm's frame buffer; None or <= 0 is unknown."""
    return duration is not None and 0 < duration < LOUDNORM_SHORT_PROGRAMME_S


def _measured_blockers(measurements, target_lra):
    """The rule's measured halves: an unset value, the range over its target, the true peak after the gain over its target.

    The arithmetic is ffmpeg's, in its order (offset = I_target - I, then TP + offset), on the
    same decimal strings it is given, so the boundary falls where ffmpeg's does.
    """
    values = {key: _rule_value(measurements, key, unset) for key, unset in LOUDNORM_UNSET_VALUES}
    if None in values.values():
        return (BLOCKER_UNREADABLE,)
    offset = LOUDNORM_TARGET_I - values["input_i"]
    checks = ((BLOCKER_LRA, values["input_lra"] > target_lra), (BLOCKER_TRUE_PEAK, values["input_tp"] + offset > LOUDNORM_TARGET_TP))
    return tuple(code for code, blocked in checks if blocked)


def linear_mode_blockers(measurements, target_lra, duration=None):
    """Why loudnorm's applied pass would leave its linear mode, by ffmpeg's own rule; empty when it holds.

    A programme `duration` (seconds) under 3 s has no blockers, since ffmpeg runs it linear
    whatever was measured; without a duration that case is not checked.
    """
    return () if _short_programme(duration) else _measured_blockers(measurements, target_lra)


def _mode_text(blockers, duration):
    """The log's name for the mode the rule predicts, with why it is not linear, or why a short programme is."""
    if blockers:
        return f"dynamic ({'; '.join(BLOCKER_TEXT[code] for code in blockers)})"
    return f"linear (programme under {LOUDNORM_SHORT_PROGRAMME_S:g} s)" if _short_programme(duration) else "linear"


def _log_loudness_range(measurements, duration=None):
    """Says which mode loudnorm's applied pass will run in, by ffmpeg's rule; returns why it leaves linear mode."""
    target = float(_get_config_val("LOUDNORM_TARGET_LRA"))
    blockers = linear_mode_blockers(measurements, target, duration)
    mode = _mode_text(blockers, duration)
    readings = f"I={measurements.get('input_i')} LRA={measurements.get('input_lra')} TP={measurements.get('input_tp')}"
    log_msg(f"    [Mastering] Measured {readings}; target LRA {target:g} -> {mode}")
    return blockers


def _applied_loudness(measurements, blockers):
    """The applied pass's loudness stage: loudnorm with the measured values, or, configured, the gain_limiter fallback.

    The fallback answers only the case where the true-peak rule alone takes loudnorm out of
    linear mode; a range over its target, or an unreadable measurement, stays with loudnorm.
    """
    if blockers != (BLOCKER_TRUE_PEAK,) or _get_config_val("LOUDNORM_LINEAR_FALLBACK") != GAIN_LIMITER:
        return _measured_loudnorm_args(measurements)
    gain = LinearGain(round(LOUDNORM_TARGET_I - float(measurements["input_i"]), 2))
    log_msg(f"    [Mastering] Linear fallback: {gain.gain_db:+.2f} dB to the target, the sample-peak limiter (-1 dBFS) on the peaks")
    return gain


def _check_true_peak(output_tp):
    """Logs a gain_limiter render whose true peak sits over the target; an unreadable reading is not checked."""
    if not _is_valid_loudnorm_number(output_tp) or float(output_tp) <= LOUDNORM_TARGET_TP:
        return
    log_msg(
        f"    [Warning] The gain_limiter render peaks at {float(output_tp):.2f} dBTP, over the {LOUDNORM_TARGET_TP:g} dBTP target: "
        "its limiter holds sample peaks only.",
        is_error=True,
    )


def _fallback_readings(input_paths, graph_prefix, applied, total_duration=None):
    """What the gain_limiter fallback renders: its whole chain read by loudnorm's analysis pass, true peak included.

    A render that cannot be read records its gain and no readings.
    """
    rendered = _loudnorm_graph(f"{graph_prefix}{_mastering_filters(applied)},", _loudnorm_target_args())
    report = _run_loudness_analysis(input_paths, rendered, total_duration=total_duration) or {}
    readings = {"normalization_type": GAIN_LIMITER, "gain_db": applied.gain_db}
    readings.update({output: report.get(measured) for output, measured in FALLBACK_READINGS})
    _check_true_peak(readings["output_tp"])
    return readings


def _applied_readings(input_paths, graph_prefix, applied, total_duration=None):
    """What the applied loudness stage does: ffmpeg's own report from a null-sink run of the applied loudnorm.

    The gain_limiter fallback runs no loudnorm; its record is its render's measured output.
    A run that cannot be read records no type.
    """
    if isinstance(applied, LinearGain):
        return _fallback_readings(input_paths, graph_prefix, applied, total_duration)
    report = _run_loudness_analysis(input_paths, _loudnorm_graph(graph_prefix, applied), total_duration=total_duration)
    if report is None:
        return {"normalization_type": None}
    return {key: report.get(key) for key in APPLIED_READINGS}


def _check_rule(predicted, actual):
    """Logs a decision ffmpeg made differently from the rule here; a missing or fallback record is not one."""
    if actual in (None, GAIN_LIMITER, predicted):
        return
    log_msg(f"    [Warning] loudnorm ran {actual} where its rule predicts {predicted}; the rule needs re-deriving.", is_error=True)


def _record_loudnorm(input_paths, graph_prefix, measurements, applied, total_duration=None):
    """With the event log on, records what the applied loudness stage does, beside the track it masters and in the log.

    Returns the record, or None when the event log is off (then no extra pass runs).
    """
    if not event_log.enabled():
        return None
    target_lra = float(_get_config_val("LOUDNORM_TARGET_LRA"))
    blockers = linear_mode_blockers(measurements, target_lra, total_duration)
    predicted = "dynamic" if blockers else "linear"
    record = {
        "measured": {key: measurements.get(key) for key in LOUDNORM_MEASURE_KEYS},
        "target": {"I": LOUDNORM_TARGET_I, "TP": LOUDNORM_TARGET_TP, "LRA": target_lra},
        "blockers": list(blockers),
        "predicted": predicted,
        "fallback": _get_config_val("LOUDNORM_LINEAR_FALLBACK"),
        **_applied_readings(input_paths, graph_prefix, applied, total_duration),
    }
    _check_rule(predicted, record["normalization_type"])
    track = Path(input_paths[1])
    event_log.write_json(_sidecar_path(track), event_log.document(LOUDNORM_STAGE, track, record))
    event_log.write(LOUDNORM_STAGE, track, record)
    return record


def _sidecar_path(track):
    """The loudness record beside the track the mux masters, named as in the event log: each mode's run keeps its own."""
    return Path(track).parent / event_log.record_name(LOUDNORM_STAGE, track)


def _build_single_audio_filter_expression(loudnorm_args=None):
    """Mastering graph for single-track modes, matching the two-stem mix path."""
    if not _get_config_val("ENABLE_LOUDNORM"):
        return None
    return f"[1:a]{_mastering_chain(loudnorm_args, 'mastered')}"


def _resolve_loudnorm_args(video_path, aligned_vocals, aligned_background, vocal_mix_vol, bg_mix_vol, total_duration=None):
    """Measures the finished two-stem mix before normalisation is applied."""
    base_filter = _build_mix_base_expression(vocal_mix_vol, bg_mix_vol)
    return _resolve_measured_loudnorm((video_path, aligned_vocals, aligned_background), f"{base_filter},", total_duration=total_duration)


def _resolve_single_track_loudnorm_args(video_path, processed_audio_wav, total_duration=None):
    """Measures a single processed track before normalisation is applied."""
    return _resolve_measured_loudnorm((video_path, processed_audio_wav), "[1:a]", total_duration=total_duration)
