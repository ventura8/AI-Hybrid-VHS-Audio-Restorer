"""Pure-Rust Cathar DSP audio restoration module tailored for VHS captures.

Orchestrates multi-stage deterministic DSP filtering using the Cathar engine:
- Pre-conditioning: dewind, GCC-PHAT azimuth alignment, sub-bass mono-maker,
  AR declick, surface decrackle, AR dropout inpaint, deplosive.
- Analog Repair: SPADE declip, adaptive I/Q tracking dehum, transient repair,
  dewow, and WPE dereverberation.
- Denoising: empirical noiseprint learning and phase-coherent spectral subtraction.
- Polish: multiband adaptive sibilance de-esser and SBR harmonic synthesis.
"""

import hashlib
import json
import shutil
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    import soundfile as sf
except ImportError:
    sf = None

from .config import (
    CATHAR_ALPHA,
    CATHAR_AZIMUTH_MAX_MS,
    CATHAR_AZIMUTH_METHOD,
    CATHAR_BETA,
    CATHAR_DECLICK_THRESHOLD,
    CATHAR_DECLIP_THRESHOLD,
    CATHAR_DECRACKLE_SENSITIVITY,
    CATHAR_DEESSER_BANDS,
    CATHAR_DEESSER_FREQ,
    CATHAR_DEESSER_THRESHOLD,
    CATHAR_DEHUM_ADAPTIVE,
    CATHAR_DEHUM_HARMONICS,
    CATHAR_DENOISE_METHOD,
    CATHAR_DEPLOSIVE_STRENGTH,
    CATHAR_DEREVERB_STRENGTH,
    CATHAR_DEREVERB_WPE,
    CATHAR_DEWIND_CUTOFF,
    CATHAR_ENABLE_AZIMUTH,
    CATHAR_ENABLE_COHERENT,
    CATHAR_ENABLE_DECLICK,
    CATHAR_ENABLE_DECLIP,
    CATHAR_ENABLE_DECRACKLE,
    CATHAR_ENABLE_DEESSER,
    CATHAR_ENABLE_DEHUM,
    CATHAR_ENABLE_DEPLOSIVE,
    CATHAR_ENABLE_DEREVERB,
    CATHAR_ENABLE_DEWIND,
    CATHAR_ENABLE_DEWOW,
    CATHAR_ENABLE_ENHANCE,
    CATHAR_ENABLE_INPAINT,
    CATHAR_ENABLE_MONO_BELOW,
    CATHAR_ENABLE_NOISEPRINT,
    CATHAR_ENABLE_REPAIR,
    CATHAR_ENHANCE_METHOD,
    CATHAR_INPAINT_ITERATIONS,
    CATHAR_INPAINT_MAX_GAP_MS,
    CATHAR_MONO_BELOW_HZ,
    CATHAR_NOISEPRINT_DURATION_S,
    CATHAR_REPAIR_STRENGTH,
    NOTCH_FREQ,
)
from .utils import CATHAR_BIN, FFMPEG_BIN, is_valid_audio, log_msg, run_command_with_progress


def _promote_cathar_tmp(tmp_wav, output_wav, step_label):
    """Safely promotes valid temporary audio file to final output destination."""
    if is_valid_audio(tmp_wav):
        if output_wav.exists():
            output_wav.unlink()
        tmp_wav.rename(output_wav)
        return output_wav
    if tmp_wav.exists():
        tmp_wav.unlink()
    raise RuntimeError(f"Cathar {step_label} failed: Output audio is invalid or empty.")


def _build_cathar_cache_key(cmd_args, input_wav):
    """Computes identity hash and command cache payload for a Cathar step."""
    digest = hashlib.sha256()
    with input_wav.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "command": list(cmd_args),
        "input_sha256": digest.hexdigest(),
    }


def _is_cathar_cached(cmd_args, input_wav, output_wav, cache_path):
    """Checks if a valid output artifact and matching sidecar cache exists."""
    if not (is_valid_audio(output_wav) and cache_path.is_file()):
        return False
    cache_key = _build_cathar_cache_key(cmd_args, input_wav)
    return _cathar_cache_matches(cache_path, cache_key)


def _run_cathar_step(cmd_args, input_wav, output_wav, step_label, task_desc, total_duration=None):
    """Executes a Cathar CLI restoration command with atomic tmp handling."""
    cache_path = output_wav.with_suffix(".json")
    if _is_cathar_cached(cmd_args, input_wav, output_wav, cache_path):
        log_msg(f"    [Cathar] Skipping {step_label} (exists: {output_wav.name})")
        return output_wav
    log_msg(f"    [Cathar] Applying {step_label}...")
    tmp_wav = output_wav.with_suffix(".tmp.wav")
    cmd = [CATHAR_BIN] + list(cmd_args) + [str(input_wav), "-o", str(tmp_wav), "--no-banner"]
    try:
        run_command_with_progress(cmd, description=task_desc, total_duration=total_duration)
    except Exception as exc:
        if tmp_wav.exists():
            tmp_wav.unlink()
        raise RuntimeError(f"Cathar {step_label} failed: {exc}") from exc
    result = _promote_cathar_tmp(tmp_wav, output_wav, step_label)
    cache_path.write_text(json.dumps(_build_cathar_cache_key(cmd_args, input_wav)), encoding="utf-8")
    return result


def _cathar_cache_matches(cache_path, cache_key):
    """Returns whether a Cathar artifact sidecar matches this exact command/input."""
    try:
        return cache_path.is_file() and json.loads(cache_path.read_text(encoding="utf-8")) == cache_key
    except (OSError, json.JSONDecodeError):
        return False


def _cathar_dewind_step(input_wav, output_dir, cutoff=CATHAR_DEWIND_CUTOFF, total_duration=None):
    output_wav = output_dir / f"dewinded_{input_wav.name}"
    cmd = ["dewind", "--cutoff", str(cutoff)]
    return _run_cathar_step(cmd, input_wav, output_wav, "Low-Frequency Motor Dewind", "Cathar Dewind", total_duration)


def _cathar_azimuth_step(
    input_wav,
    output_dir,
    max_ms=CATHAR_AZIMUTH_MAX_MS,
    method=CATHAR_AZIMUTH_METHOD,
    total_duration=None,
):
    output_wav = output_dir / f"azimuth_{input_wav.name}"
    cmd = ["azimuth", "--max-ms", str(max_ms), "--method", str(method)]
    return _run_cathar_step(cmd, input_wav, output_wav, "Stereo Azimuth Phase Alignment", "Cathar Azimuth", total_duration)


def _is_stereo_audio(wav_path):
    """Returns True if audio file has at least 2 channels."""
    if sf is None:
        return True
    try:
        info = sf.info(str(wav_path))
        return info.channels >= 2
    except Exception:
        return True


def _cathar_mono_below_step(input_wav, output_dir, cutoff_hz=CATHAR_MONO_BELOW_HZ, total_duration=None):
    if not _is_stereo_audio(input_wav):
        log_msg("    [Cathar] Skipping Sub-Bass Mono Collapse (mono audio input)")
        return input_wav
    output_wav = output_dir / f"monobelow_{input_wav.name}"
    cmd = ["stereo", "--mono-below", str(cutoff_hz)]
    return _run_cathar_step(cmd, input_wav, output_wav, f"Sub-Bass Mono Collapse ({cutoff_hz} Hz)", "Cathar Mono Below", total_duration)


def _cathar_declick_step(input_wav, output_dir, threshold=CATHAR_DECLICK_THRESHOLD, total_duration=None):
    output_wav = output_dir / f"declicked_{input_wav.name}"
    cmd = ["declick", "--method", "ar", "--threshold", str(threshold)]
    return _run_cathar_step(cmd, input_wav, output_wav, "Impulse Pop/Click Declick", "Cathar Declick", total_duration)


def _cathar_decrackle_step(input_wav, output_dir, sensitivity=CATHAR_DECRACKLE_SENSITIVITY, total_duration=None):
    output_wav = output_dir / f"decrackled_{input_wav.name}"
    cmd = ["decrackle", "--sensitivity", str(sensitivity)]
    return _run_cathar_step(cmd, input_wav, output_wav, "Surface Noise Decrackle", "Cathar Decrackle", total_duration)


def _cathar_inpaint_step(
    input_wav,
    output_dir,
    max_gap_ms=CATHAR_INPAINT_MAX_GAP_MS,
    iterations=CATHAR_INPAINT_ITERATIONS,
    total_duration=None,
):
    output_wav = output_dir / f"inpainted_{input_wav.name}"
    cmd = ["inpaint", "--max-gap-ms", str(max_gap_ms), "--iterations", str(iterations)]
    return _run_cathar_step(cmd, input_wav, output_wav, "Dropout AR Inpainting", "Cathar Inpaint", total_duration)


def _cathar_deplosive_step(input_wav, output_dir, strength=CATHAR_DEPLOSIVE_STRENGTH, total_duration=None):
    output_wav = output_dir / f"deplosived_{input_wav.name}"
    cmd = ["deplosive", "-s", str(strength)]
    return _run_cathar_step(cmd, input_wav, output_wav, "Speech Plosive/Pop Taming", "Cathar Deplosive", total_duration)


def _cathar_enhance_step(
    input_wav,
    output_dir,
    method=CATHAR_ENHANCE_METHOD,
    rate=48000,
    total_duration=None,
):
    output_wav = output_dir / f"enhanced_{input_wav.name}"
    cmd = ["enhance", "--method", str(method), "-r", str(rate)]
    return _run_cathar_step(cmd, input_wav, output_wav, "High-Frequency Harmonic SBR", "Cathar Enhance", total_duration)


def _cathar_declip_step(input_wav, output_dir, threshold=CATHAR_DECLIP_THRESHOLD, total_duration=None):
    output_wav = output_dir / f"declipped_{input_wav.name}"
    cmd = ["declip", "--method", "spade", "--threshold", str(threshold)]
    return _run_cathar_step(cmd, input_wav, output_wav, "Peak Saturation Declip", "Cathar Declip", total_duration)


MAINS_FUNDAMENTAL_BY_HARMONIC = {100.0: 50.0, 120.0: 60.0}


def _cathar_dehum_step(
    input_wav,
    output_dir,
    freq=60.0,
    adaptive=CATHAR_DEHUM_ADAPTIVE,
    harmonics=CATHAR_DEHUM_HARMONICS,
    total_duration=None,
):
    if freq is None or freq <= 0:
        return input_wav
    freq = MAINS_FUNDAMENTAL_BY_HARMONIC.get(float(freq), float(freq))
    output_wav = output_dir / f"dehummed_{input_wav.name}"
    base_freq = 50.0 if abs(freq - 50.0) < abs(freq - 60.0) else 60.0
    cmd = ["dehum", "--freq", str(int(base_freq)), "--harmonics", str(harmonics)]
    if adaptive:
        cmd.append("--adaptive")
    return _run_cathar_step(cmd, input_wav, output_wav, "Adaptive Mains Buzz Dehum", "Cathar Dehum", total_duration)


def _cathar_repair_step(input_wav, output_dir, strength=CATHAR_REPAIR_STRENGTH, total_duration=None):
    output_wav = output_dir / f"repaired_{input_wav.name}"
    cmd = ["repair", "-s", str(strength)]
    return _run_cathar_step(cmd, input_wav, output_wav, "Transient Spectral Spike Repair", "Cathar Repair", total_duration)


def _cathar_dereverb_step(
    input_wav,
    output_dir,
    wpe=CATHAR_DEREVERB_WPE,
    strength=CATHAR_DEREVERB_STRENGTH,
    total_duration=None,
):
    output_wav = output_dir / f"dereverbed_{input_wav.name}"
    cmd = ["dereverb", "--wpe"] if wpe else ["dereverb", "-s", str(strength)]
    return _run_cathar_step(cmd, input_wav, output_wav, "Acoustic Reflection Dereverb", "Cathar Dereverb", total_duration)


def _cathar_deesser_step(
    input_wav,
    output_dir,
    bands=CATHAR_DEESSER_BANDS,
    freq=CATHAR_DEESSER_FREQ,
    threshold=CATHAR_DEESSER_THRESHOLD,
    total_duration=None,
):
    output_wav = output_dir / f"deessed_{input_wav.name}"
    cmd = ["deesser", "--bands", str(bands), "-f", str(freq), f"--threshold={threshold}"]
    return _run_cathar_step(cmd, input_wav, output_wav, "Multiband Sibilance Control", "Cathar De-Esser", total_duration)


def _cathar_dewow_step(input_wav, output_dir, total_duration=None):
    output_wav = output_dir / f"dewowed_{input_wav.name}"
    cmd = ["dewow"]
    try:
        return _run_cathar_step(cmd, input_wav, output_wav, "Analog Transport Dewow", "Cathar Dewow", total_duration)
    except RuntimeError as exc:
        log_msg(f"    [Cathar] Dewow stage bypassed due to upstream error: {exc}", is_error=True)
        return input_wav


def _evaluate_quiet_probes(mono, win):
    max_start = len(mono) - win
    if max_start <= 0:
        return 0
    positions = np.linspace(0, max_start, min(40, max_start + 1), dtype=int)
    best_pos, min_rms = 0, float("inf")
    for pos in positions:
        pos = int(pos)
        end_pos = pos + win
        chunk = mono[pos:end_pos]
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
        if 1e-6 < rms < min_rms:
            best_pos, min_rms = pos, rms
    return best_pos


def _mono_window(wav_path, start, frames):
    """Read one bounded float32 window and collapse it to mono."""
    data, sr = sf.read(str(wav_path), dtype="float32", start=start, frames=frames)
    mono = np.mean(data, axis=1) if data.ndim > 1 else data
    return mono.astype(np.float32, copy=False), sr


def _read_mono_samples(wav_path):
    """Return the quietest bounded window distributed across the recording."""
    info = sf.info(str(wav_path))
    window_frames = min(info.frames, info.samplerate * 60)
    latest_start = max(0, info.frames - window_frames)
    starts = np.linspace(0, latest_start, num=min(8, max(1, info.frames // max(window_frames, 1))), dtype=int)
    windows = [(*_mono_window(wav_path, int(start), window_frames), int(start)) for start in starts]
    return min(windows, key=lambda item: float(np.sqrt(np.mean(item[0] ** 2) + 1e-12)))


def _find_quiet_window(wav_path, duration_s=0.75):
    """Locates timestamp (seconds) of lowest RMS energy window in wav_path."""
    if sf is None or np is None:
        return 0.0
    try:
        mono, sr, source_start = _read_mono_samples(wav_path)
        win = int(duration_s * sr)
        if len(mono) <= win:
            return 0.0
        return (source_start + _evaluate_quiet_probes(mono, win)) / sr
    except Exception:
        return 0.0


def _extract_noiseprint_slice(input_wav, slice_wav, duration_s):
    start_s = _find_quiet_window(input_wav, duration_s)
    cmd_slice = [
        FFMPEG_BIN,
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(input_wav),
        "-t",
        f"{duration_s:.3f}",
        "-c:a",
        "pcm_f32le",
        str(slice_wav),
    ]
    run_command_with_progress(cmd_slice, description="Cathar Noise Probe Extraction")
    return is_valid_audio(slice_wav)


def _execute_noiseprint(slice_wav, output_json):
    cmd_np = [CATHAR_BIN, "noiseprint", str(slice_wav), "-o", str(output_json), "--no-banner"]
    run_command_with_progress(cmd_np, description="Cathar Learn Noiseprint")
    return output_json if output_json.exists() else None


def _validate_existing_noiseprint(output_json):
    """Validates existing noiseprint JSON, unlinking if invalid. Returns True if valid."""
    if not output_json.exists():
        return False
    try:
        with open(output_json, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        try:
            output_json.unlink()
        except OSError:
            pass
        return False


def _cathar_noiseprint_step(input_wav, output_dir, duration_s=CATHAR_NOISEPRINT_DURATION_S):
    """Learns an empirical noise print JSON from the quietest section of input_wav."""
    output_json = output_dir / f"noise_{input_wav.stem}.np.json"
    if _validate_existing_noiseprint(output_json):
        return output_json
    slice_wav = output_dir / f"silence_probe_{input_wav.stem}.wav"
    try:
        if not _extract_noiseprint_slice(input_wav, slice_wav, duration_s):
            return None
        return _execute_noiseprint(slice_wav, output_json)

    except Exception as exc:
        log_msg(f"    [Cathar] Noiseprint extraction bypassed: {exc}")
        return None
    finally:
        if slice_wav.exists():
            slice_wav.unlink()


def _build_cathar_denoise_cmd(method, alpha, beta, coherent, noiseprint_path=None):
    coherent_flags = ["--coherent"] if coherent else []
    np_flags = ["--noiseprint", str(noiseprint_path)] if noiseprint_path and Path(noiseprint_path).exists() else []
    if method == "wiener":
        return ["denoise", "--wiener"] + coherent_flags + np_flags
    return ["denoise", "--alpha", str(alpha), "--beta", str(beta)] + coherent_flags + np_flags


def _cathar_denoise_step(
    input_wav,
    output_dir,
    method=CATHAR_DENOISE_METHOD,
    alpha=CATHAR_ALPHA,
    beta=CATHAR_BETA,
    coherent=CATHAR_ENABLE_COHERENT,
    noiseprint_path=None,
    total_duration=None,
):
    output_wav = output_dir / f"denoised_{input_wav.name}"
    cmd = _build_cathar_denoise_cmd(method, alpha, beta, coherent, noiseprint_path=noiseprint_path)
    return _run_cathar_step(cmd, input_wav, output_wav, f"Phase-Coherent Denoise ({method})", "Cathar Denoise", total_duration)


def _cathar_clean_transients(current_wav, work_dir, total_duration=None):
    """Suppresses impulse clicks, surface crackle, and dropout gaps."""
    if CATHAR_ENABLE_DECLICK:
        current_wav = _cathar_declick_step(current_wav, work_dir, total_duration=total_duration)
    if CATHAR_ENABLE_DECRACKLE:
        current_wav = _cathar_decrackle_step(current_wav, work_dir, total_duration=total_duration)
    if CATHAR_ENABLE_INPAINT:
        current_wav = _cathar_inpaint_step(current_wav, work_dir, total_duration=total_duration)
    return current_wav


def _cathar_precondition_pass(current_wav, work_dir, total_duration=None):
    """Executes initial sub-audible, azimuth, and impulse noise suppression stages."""
    if CATHAR_ENABLE_DEWIND:
        current_wav = _cathar_dewind_step(current_wav, work_dir, total_duration=total_duration)
    if CATHAR_ENABLE_AZIMUTH:
        current_wav = _cathar_azimuth_step(current_wav, work_dir, total_duration=total_duration)
    if CATHAR_ENABLE_MONO_BELOW:
        current_wav = _cathar_mono_below_step(current_wav, work_dir, total_duration=total_duration)
    current_wav = _cathar_clean_transients(current_wav, work_dir, total_duration=total_duration)
    if CATHAR_ENABLE_DEPLOSIVE:
        current_wav = _cathar_deplosive_step(current_wav, work_dir, total_duration=total_duration)
    return current_wav


def _cathar_analog_repair_pass(current_wav, work_dir, notch_freq=60.0, total_duration=None):
    """Executes analog clipping reconstruction, mains dehum, and spectral glitch repair."""
    if CATHAR_ENABLE_DECLIP:
        current_wav = _cathar_declip_step(current_wav, work_dir, total_duration=total_duration)
    if CATHAR_ENABLE_DEHUM:
        current_wav = _cathar_dehum_step(current_wav, work_dir, freq=notch_freq, total_duration=total_duration)
    if CATHAR_ENABLE_REPAIR:
        current_wav = _cathar_repair_step(current_wav, work_dir, total_duration=total_duration)
    return current_wav


def _cathar_repair_pass(current_wav, work_dir, notch_freq=60.0, total_duration=None):
    """Executes clipping repair, mains dehum, glitch repair, dewow, and dereverb."""
    current = _cathar_analog_repair_pass(current_wav, work_dir, notch_freq=notch_freq, total_duration=total_duration)
    if CATHAR_ENABLE_DEWOW:
        current = _cathar_dewow_step(current, work_dir, total_duration=total_duration)
    if CATHAR_ENABLE_DEREVERB:
        current = _cathar_dereverb_step(current, work_dir, total_duration=total_duration)
    return current


def _cathar_polish_pass(current_wav, work_dir, total_duration=None):
    """Applies sibilance de-essing and high-frequency harmonic synthesis."""
    if CATHAR_ENABLE_DEESSER:
        current_wav = _cathar_deesser_step(current_wav, work_dir, total_duration=total_duration)
    if CATHAR_ENABLE_ENHANCE:
        current_wav = _cathar_enhance_step(current_wav, work_dir, total_duration=total_duration)
    return current_wav


def _require_cathar_binary():
    """Raises when the Cathar CLI is missing, since every stage below shells out to it."""
    if not CATHAR_BIN or (shutil.which(CATHAR_BIN) is None and not Path(CATHAR_BIN).exists()):
        raise FileNotFoundError(f"Cathar binary not found or unusable: {CATHAR_BIN}")


def _resolve_notch_freq(strategy):
    """Returns the mains hum frequency to notch, falling back to the configured mains frequency.

    The fallback used to be a hardcoded 60 Hz, which silently ignored a PAL user's configured
    notch_freq whenever the scanner did not supply one -- notching 60 Hz on 50 Hz hum.
    """
    notch_hz = (strategy or {}).get("precondition_filters", {}).get("notch_hz")
    return float(NOTCH_FREQ) if notch_hz is None else float(notch_hz)


def filter_cathar_vhs_pipeline(original_wav, work_dir, total_duration=None, strategy=None):
    """Orchestrates the full Cathar VHS audio restoration pipeline."""
    _require_cathar_binary()
    notch_freq = _resolve_notch_freq(strategy)
    current = _cathar_precondition_pass(original_wav, work_dir, total_duration=total_duration)
    current = _cathar_repair_pass(current, work_dir, notch_freq=notch_freq, total_duration=total_duration)
    np_path = _cathar_noiseprint_step(current, work_dir) if CATHAR_ENABLE_NOISEPRINT else None
    current = _cathar_denoise_step(current, work_dir, noiseprint_path=np_path, total_duration=total_duration)
    return _cathar_polish_pass(current, work_dir, total_duration=total_duration)
