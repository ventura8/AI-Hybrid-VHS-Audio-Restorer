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
    if int(bands) > 1 and float(threshold) <= 0:
        # Multiband threshold is dB above each band's running average; a non-positive
        # value engages the de-esser on every frame and strips everything above `freq`.
        log_msg(
            f"    [Cathar] De-esser threshold {threshold} dB with {bands} bands keeps the stage "
            f"engaged on every frame and mutes speech above {freq} Hz; using cathar's suggested 6 dB.",
            is_error=True,
        )
        threshold = 6.0
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


# A pause on a dialogue tape is short: a single 4 s window rarely fits inside one, so it
# carries speech and the print learns sibilance (measured on a 134 s tape: the quietest 4 s
# sat 12 dB above the quietest 0.75 s, correlated 0.86 with the loud frames in the speech
# band, and the subtraction took 6 dB more off speech at 8-12 kHz). A single 0.75 s window
# is a pause but too short to average the hiss. So on a tape the print is stitched from
# several 0.75 s pauses spread across the quietest fifth of the windows: the shape of a
# pause, the level of the typical floor, the length to average it.
NOISEPRINT_WINDOW_S = 0.75
NOISEPRINT_HOP_S = 0.25
NOISEPRINT_LEVEL_SHARE = 0.2
NOISEPRINT_CROSSFADE_S = 0.01


def _window_levels(wav_path, hop_frames, window_hops):
    """RMS of every window (window_hops hops long) at hop spacing, streamed from the file."""
    energies = []
    with sf.SoundFile(str(wav_path)) as handle:
        for block in handle.blocks(blocksize=hop_frames * 240, dtype="float32", always_2d=True):
            mono = block.mean(axis=1)
            frames = len(mono) // hop_frames
            energies.extend(np.mean(mono[: frames * hop_frames].reshape(frames, hop_frames) ** 2, axis=1).tolist())
    energies = np.asarray(energies, dtype=np.float64)
    if len(energies) < window_hops:
        return np.zeros(0), np.zeros(0, dtype=int)
    summed = np.convolve(energies, np.ones(window_hops), mode="valid")
    return np.sqrt(summed / window_hops), np.arange(len(summed)) * hop_frames


def _pick_pause_windows(levels, starts, window_frames, count, level_share=NOISEPRINT_LEVEL_SHARE):
    """Starts of `count` non-overlapping windows spread evenly by level over the quietest share."""
    if len(levels) == 0:
        return []
    picked = _quietest_non_overlapping(levels, starts, window_frames, level_share)
    if not picked:
        return []
    by_level = sorted(picked, key=lambda s: levels[int(np.searchsorted(starts, s))])
    chosen = np.linspace(0, len(by_level) - 1, min(count, len(by_level))).astype(int)
    return sorted(by_level[k] for k in chosen)


def _quietest_non_overlapping(levels, starts, window_frames, level_share):
    """Starts of the windows within the quietest share, quietest first, each at least a window apart."""
    cut = np.percentile(levels, level_share * 100.0)
    order = [int(k) for k in np.argsort(levels, kind="stable") if levels[k] <= cut]
    picked = []
    for k in order:
        if _clear_of(int(starts[k]), picked, window_frames):
            picked.append(int(starts[k]))
    return picked


def _clear_of(start, picked, window_frames):
    return all(abs(start - j) >= window_frames for j in picked)


def _stitch_windows(wav_path, starts, window_frames, crossfade_frames):
    """Concatenates the windows with short crossfades so the seams add no click to the print."""
    ramp = np.linspace(0.0, 1.0, crossfade_frames, dtype=np.float32)[:, None]
    out = None
    for start in starts:
        seg, _rate = sf.read(str(wav_path), dtype="float32", start=start, frames=window_frames, always_2d=True)
        if out is None:
            out = seg.copy()
            continue
        out[-crossfade_frames:] = out[-crossfade_frames:] * (1.0 - ramp) + seg[:crossfade_frames] * ramp
        out = np.concatenate([out, seg[crossfade_frames:]])
    return out


def _extract_stitched_noiseprint(input_wav, slice_wav, duration_s):
    """Writes a probe stitched from the quiet pauses of input_wav, about duration_s long in total."""
    info = sf.info(str(input_wav))
    rate = info.samplerate
    hop = int(NOISEPRINT_HOP_S * rate)
    window_hops = max(1, round(NOISEPRINT_WINDOW_S / NOISEPRINT_HOP_S))
    window = hop * window_hops
    count = max(1, round(duration_s / NOISEPRINT_WINDOW_S))
    levels, starts = _window_levels(input_wav, hop, window_hops)
    picked = _pick_pause_windows(levels, starts, window, count)
    if not picked:
        return False
    stitched = _stitch_windows(input_wav, picked, window, int(NOISEPRINT_CROSSFADE_S * rate))
    sf.write(str(slice_wav), stitched, rate, subtype="FLOAT")
    log_msg(f"    [Cathar] Noise print stitched from {len(picked)} pauses at {', '.join(f'{s / rate:.1f}' for s in picked)} s")
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


def _cathar_noiseprint_step(input_wav, output_dir, duration_s=CATHAR_NOISEPRINT_DURATION_S, stitched=False):
    """Learns an empirical noise print JSON from the quietest section of input_wav.

    `stitched` learns it from several quiet pauses instead of one window (see
    _extract_stitched_noiseprint); the single-window path is untouched, so every caller
    that does not ask for it keeps its output.
    """
    output_json = output_dir / _noiseprint_name(input_wav, duration_s, stitched)
    if _validate_existing_noiseprint(output_json):
        return output_json
    slice_wav = output_dir / f"silence_probe_{input_wav.stem}.wav"
    extract = _extract_stitched_noiseprint if stitched else _extract_noiseprint_slice
    try:
        return _learn_noiseprint(extract, input_wav, slice_wav, duration_s, output_json)
    except Exception as exc:
        log_msg(f"    [Cathar] Noiseprint extraction bypassed: {exc}")
        return None
    finally:
        if slice_wav.exists():
            slice_wav.unlink()


def _noiseprint_name(input_wav, duration_s, stitched):
    """The print is named by how it was learned: a preserved work directory must not hand a
    0.75 s single-window print to a later stitched 6 s request, or the reverse."""
    mode = "stitched" if stitched else "single"
    return f"noise_{input_wav.stem}_{mode}_{float(duration_s):g}s.np.json"


def _learn_noiseprint(extract, input_wav, slice_wav, duration_s, output_json):
    """The probe slice through cathar's noiseprint, or None when there was nothing to learn from."""
    if not extract(input_wav, slice_wav, duration_s):
        return None
    return _execute_noiseprint(slice_wav, output_json)


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


# The noise print is learned from the quietest stretch of the capture, and the stretch has to
# be a pause: on a 15 s clip a 4 s window is a quarter of the material and carries programme,
# on a two-hour tape it is almost certainly pure noise. So the full window is used only once
# the material is at least this many times longer than it (6 s from 120 s of tape); anything
# shorter keeps the 0.75 s cathar shipped with, exactly, so every corpus-length clip and
# 60 s excerpt stays bit-identical while a real tape gets the full window.
NOISEPRINT_SHORT_S = 0.75
NOISEPRINT_MIN_MATERIAL_RATIO = 20.0


def _probe_duration_s(total_duration, wav_path, cap_s=CATHAR_NOISEPRINT_DURATION_S):
    """Seconds of quiet tape to learn the noise print from: the cap on a tape, 0.75 s on a clip.

    Measured on a 134 s dialogue tape, 0.75 s removed 3.40 dB of noise for 0.12 dB of
    programme deviation and 4 s removed 11.58 for 0.30 -- the 0.75 s print is too short to
    average the hiss, so it under-subtracts in most bins and over-subtracts in a few.
    """
    cap = float(cap_s)
    if cap <= NOISEPRINT_SHORT_S:
        return cap
    duration = _material_duration_s(total_duration, wav_path)
    return cap if duration >= cap * NOISEPRINT_MIN_MATERIAL_RATIO else NOISEPRINT_SHORT_S


def _material_duration_s(total_duration, wav_path):
    """The known duration, else the WAV header's, else 0."""
    if total_duration and total_duration > 0:
        return total_duration
    try:
        return sf.info(str(wav_path)).duration
    except Exception:
        return 0.0


def filter_cathar_vhs_pipeline(original_wav, work_dir, total_duration=None, strategy=None):
    """Orchestrates the full Cathar VHS audio restoration pipeline."""
    _require_cathar_binary()
    notch_freq = _resolve_notch_freq(strategy)
    current = _cathar_precondition_pass(original_wav, work_dir, total_duration=total_duration)
    current = _cathar_repair_pass(current, work_dir, notch_freq=notch_freq, total_duration=total_duration)
    np_path = None
    if CATHAR_ENABLE_NOISEPRINT:
        probe_s = _probe_duration_s(total_duration, current)
        np_path = _cathar_noiseprint_step(current, work_dir, duration_s=probe_s, stitched=probe_s > NOISEPRINT_SHORT_S)
    current = _cathar_denoise_step(current, work_dir, noiseprint_path=np_path, total_duration=total_duration)
    return _cathar_polish_pass(current, work_dir, total_duration=total_duration)
