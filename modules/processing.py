import shutil
import subprocess
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

from . import utils as _utils
from .config import (
    BACKGROUND_MIX_VOL,
    DENOISE_MODEL,
    ENHANCE_NFE,
    ENHANCE_TAU,
    KEEP_INPUT_FILES,
    PROCESS_MODE,
    VOCAL_MIX_VOL,
    VOCALS_MODEL,
)
from .hardware import CPU_THREADS, CUDA_ENV, CUDA_VISIBLE_DEVICE, GPU_BATCH_SIZE
from .sync import _align_stems
from .utils import (
    FFMPEG_BIN,
    attempt_cpu_run_with_retry,
    format_time,
    is_valid_audio,
    is_valid_video,
    log_msg,
    run_command_with_progress,
)

# Expose the GPU retry helper for test patching without aliasing to CPU retry.
attempt_run_with_retry = _utils.attempt_run_with_retry

OUTPUT_SUFFIX_BY_MODE = {
    "hybrid": "_Hybrid_Cleaned",
    "denoise_only": "_Denoised_Cleaned",
}


def _get_output_suffix(process_mode):
    return OUTPUT_SUFFIX_BY_MODE.get(process_mode, OUTPUT_SUFFIX_BY_MODE["hybrid"])


def _get_audio_separator_class():
    try:
        from audio_separator.separator import Separator

        return Separator
    except ImportError as exc:
        raise ImportError("audio-separator is required for stem separation/denoising.") from exc


def get_audio_duration_sec(wav_path):
    if sf is None:
        return None

    try:
        with sf.SoundFile(str(wav_path)) as f:
            return f.frames / f.samplerate
    except Exception:
        return None


def get_video_duration_sec(video_path):  # pragma: no cover
    """Gets video duration using ffprobe."""
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
        val = subprocess.check_output(cmd).decode().strip()
        return float(val)
    except Exception:  # pragma: no cover
        return None


def _extract_audio_step(video_path, original_wav, total_duration=None):
    """Step 1: Extract High-Res Audio."""
    if is_valid_audio(original_wav):
        log_msg("  [System] Skipping Extraction (valid exists)")
        return

    if original_wav.exists():
        original_wav.unlink()

    log_msg("  [System] Extracting Audio Stream...")

    # Extract to .tmp first
    tmp_wav = original_wav.with_suffix(".tmp.wav")

    def build_extract_cmd(threads):
        return [
            FFMPEG_BIN,
            "-stats",
            "-hide_banner",
            "-threads",
            str(threads),
            "-i",
            str(video_path),
            "-acodec",
            "pcm_f32le",
            "-ar",
            "44100",
            "-y",
            str(tmp_wav),
        ]

    attempt_cpu_run_with_retry(build_extract_cmd, CPU_THREADS, description="Extracting Audio", total_duration=total_duration)

    if is_valid_audio(tmp_wav):
        tmp_wav.rename(original_wav)
    else:
        if tmp_wav.exists():
            tmp_wav.unlink()
        raise Exception("Extraction failed: Output audio is invalid, empty, or too small.")  # pragma: no cover


def _collect_stem_candidates(separation_out_dir):
    vocals = list(separation_out_dir.glob("*(Vocals)*.wav"))
    background = list(separation_out_dir.glob("*(Instrumental)*.wav"))
    background += list(separation_out_dir.glob("*(Background)*.wav"))
    background += list(separation_out_dir.glob("*(No Vocals)*.wav"))
    all_wavs = list(separation_out_dir.glob("*.wav"))
    return vocals, background, all_wavs


def _needs_background_fallback(v_files, b_files, all_wavs):
    return not b_files and len(all_wavs) == 2 and bool(v_files)


def _fallback_background_candidates(v_files, b_files, all_wavs):
    if not _needs_background_fallback(v_files, b_files, all_wavs):
        return b_files
    return [f for f in all_wavs if f != v_files[0]]


def _sorted_valid_audio(paths):
    sorted_paths = sorted(paths, key=lambda path: path.stat().st_size, reverse=True)
    return [path for path in sorted_paths if is_valid_audio(path)]


def _select_valid_stem(v_files, b_files):
    v_valid = _sorted_valid_audio(v_files)
    b_valid = _sorted_valid_audio(b_files)
    if v_valid and b_valid:
        return v_valid[0], b_valid[0], v_valid, b_valid
    return None, None, v_valid, b_valid


def _normalize_background_name(separation_out_dir, vocals, background):
    if "(Background)" in background.name:
        return background

    new_name = vocals.name.replace("(Vocals)", "(Background)")
    new_path = separation_out_dir / new_name
    if new_path.exists():
        return background

    try:
        background.rename(new_path)
        return new_path
    except OSError:  # pragma: no cover
        return background


def _log_separation_failure(separation_out_dir, v_valid, b_valid):
    log_msg("    [Debug] Separation output mismatch.", level="DEBUG")
    log_msg(f"    [Debug] Found Vocals: {[f.name for f in v_valid]}", level="DEBUG")
    log_msg(f"    [Debug] Found Background: {[f.name for f in b_valid]}", level="DEBUG")
    all_any = list(separation_out_dir.glob("*"))
    log_msg(f"    [Debug] All files in dir: {[f.name for f in all_any]}", level="DEBUG")


def _verify_separation_output(separation_out_dir, original_wav):
    """Verifies that separation produced both stems with extreme robustness."""
    del original_wav
    v_files, b_files, all_wavs = _collect_stem_candidates(separation_out_dir)
    b_files = _fallback_background_candidates(v_files, b_files, all_wavs)
    vocals, background, v_valid, b_valid = _select_valid_stem(v_files, b_files)

    if vocals and background:
        return vocals, _normalize_background_name(separation_out_dir, vocals, background)

    _log_separation_failure(separation_out_dir, v_valid, b_valid)
    return None, None


def _build_separator(output_dir):
    Separator = _get_audio_separator_class()
    model_dir = Path("models").resolve()
    model_dir.mkdir(exist_ok=True)
    return Separator(
        output_dir=str(output_dir),
        model_file_dir=str(model_dir),
        output_format="wav",
        use_soundfile=True,
        normalization_threshold=0.9,
        vr_params={"batch_size": GPU_BATCH_SIZE, "window_size": 320},
        mdxc_params={"batch_size": GPU_BATCH_SIZE},
        mdx_params={"batch_size": GPU_BATCH_SIZE},
    )


def _resolve_separation_result(separation_out_dir, original_wav, output_files):
    vocals_wav, background_wav = _verify_separation_output(separation_out_dir, original_wav)
    if vocals_wav and background_wav:
        return vocals_wav, background_wav

    if hasattr(output_files, "__len__") and len(output_files) >= 2:
        log_msg(f"    [Debug] Separator returned: {output_files}", level="DEBUG")
    raise Exception("Separation completed but output stems were not identified.")


def _separate_stems_step(original_wav, separation_out_dir, total_duration=None):
    """
    Step 2: Separate Stems (BS-Roformer) via Python API.
    Returns path to (vocals_wav, background_wav).
    """
    # 1. Check Existing
    existing_v, existing_b = _verify_separation_output(separation_out_dir, original_wav)
    if existing_v and existing_b:
        log_msg("  [Step 1/5] Skipping Separation (exists & valid)")
        return existing_v, existing_b

    log_msg("  [Step 1/5] Separating Stems (BS-Roformer - AI Engine)...")

    try:
        separator = _build_separator(separation_out_dir)

        log_msg(f"    [AI] Loading Model: {VOCALS_MODEL}")
        separator.load_model(model_filename=VOCALS_MODEL)

        log_msg("    [AI] Starting Inference (GPU Accelerated)...")
        # BS-Roformer settings from original CLI call
        # No simple way to get progress updates into our custom bar via the Python API
        # in 0.41.1 without complex logging hooks, so we provide status updates.
        output_files = separator.separate(str(original_wav))

        if output_files is None:
            raise Exception("Separation completed but output stems were not identified.")

        return _resolve_separation_result(separation_out_dir, original_wav, output_files)

    except Exception as e:  # pragma: no cover
        log_msg(f"    [Error] AI Separation failed: {e}", is_error=True)
        raise e


def _clear_cuda_retry_state():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()


def _run_enhance_retry(cmd_enhance, total_duration, env=None):
    """Retries the enhancement command."""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            run_command_with_progress(cmd_enhance, env=env, description="Enhancing Vocals", total_duration=total_duration)
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                log_msg(f"    [Warning] Enhancement failed (Attempt {attempt + 1}). Retrying...", is_error=True)
                _clear_cuda_retry_state()
                time.sleep(2)
            else:  # pragma: no cover
                raise e


def _prepare_clean_directory(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_result_to_final_dir(result, output_dir):
    final_output = output_dir / result.name
    shutil.copy(result, final_output)
    return final_output


def _cleanup_directory(path):
    try:
        shutil.rmtree(path)
    except Exception:
        pass


def _handle_enhance_output(enhanced_vocals_dir, vocals_wav):
    """Checks output and verifies validity."""
    candidates_enhanced = list(enhanced_vocals_dir.glob("*.wav"))
    if not candidates_enhanced:
        log_msg("    [Warning] Resemble-Enhance did not produce output. Using raw vocals.", is_error=True)
        fb_path = enhanced_vocals_dir / f"fallback_{vocals_wav.name}"
        shutil.copy(vocals_wav, fb_path)
        return fb_path

    return candidates_enhanced[0]


def _enhance_vocals_step(vocals_wav, enhanced_vocals_dir, work_dir, total_duration=None):
    """Step 3: Enhance Vocals (Resemble-Enhance)."""
    enhanced_vocals_dir.mkdir(parents=True, exist_ok=True)
    candidates_enhanced = list(enhanced_vocals_dir.glob("*.wav"))
    valid_enhanced = [f for f in candidates_enhanced if is_valid_audio(f)]

    if valid_enhanced:
        log_msg("  [Step 2/5] Skipping Vocal Enhancement (exists)")
        return valid_enhanced[0]

    log_msg("  [Step 2/5] Enhancing Vocals (Resemble-Enhance)...")

    enhance_input_dir = work_dir / "enhance_input"
    _prepare_clean_directory(enhance_input_dir)

    shutil.copy(vocals_wav, enhance_input_dir / vocals_wav.name)

    enhanced_vocals_tmp_dir = work_dir / "enhanced_vocals_tmp"
    _prepare_clean_directory(enhanced_vocals_tmp_dir)

    cmd_enhance = [
        "resemble-enhance",
        str(enhance_input_dir),
        str(enhanced_vocals_tmp_dir),
        "--denoise_only",
        "--nfe",
        str(ENHANCE_NFE),
        "--solver",
        "rk4",
        "--tau",
        str(ENHANCE_TAU),
        "--device",
        CUDA_VISIBLE_DEVICE,
    ]

    _run_enhance_retry(cmd_enhance, total_duration, env=CUDA_ENV)

    _cleanup_directory(enhance_input_dir)

    result = _handle_enhance_output(enhanced_vocals_tmp_dir, vocals_wav)
    final_output = _copy_result_to_final_dir(result, enhanced_vocals_dir)

    _cleanup_directory(enhanced_vocals_tmp_dir)

    return final_output


def _has_denoised_candidates(candidates_denoised):
    return bool(candidates_denoised)


def _handle_missing_denoised_candidate(warning_message, input_wav, fallback_on_failure):
    if fallback_on_failure:
        log_msg(warning_message, is_error=True)
        return input_wav
    raise RuntimeError(warning_message.strip())


def _select_denoised_candidate(candidates_denoised, warning_message, input_wav, fallback_on_failure):
    clean_candidates = [f for f in candidates_denoised if "(No Noise)" in f.name]
    if clean_candidates:
        return clean_candidates[0]
    if _has_denoised_candidates(candidates_denoised):
        return candidates_denoised[0]
    return _handle_missing_denoised_candidate(warning_message, input_wav, fallback_on_failure)


def _run_denoise_separator(input_wav, denoised_output_dir, selected_label, warning_message, error_message, fallback_on_failure=True):
    """Runs UVR-DeNoise-Lite through audio-separator and selects best output."""
    try:
        separator = _build_separator(denoised_output_dir)

        log_msg(f"    [AI] Loading Model: {DENOISE_MODEL}")
        separator.load_model(model_filename=DENOISE_MODEL)

        log_msg("    [AI] Starting Inference (GPU Accelerated)...")
        separator.separate(str(input_wav))

        candidates_denoised = sorted(denoised_output_dir.glob("*.wav"), key=lambda path: path.name.lower())
        result = _select_denoised_candidate(candidates_denoised, warning_message, input_wav, fallback_on_failure)
        log_msg(f"    {selected_label}: {result.name}")
        return result

    except Exception as e:
        log_msg(f"    [Error] {error_message}: {e}", is_error=True)
        if fallback_on_failure:
            return input_wav
        raise


def _select_preferred_denoised_output(valid_denoised):
    """Pick a deterministic denoised output, preferring '(No Noise)' when present."""
    sorted_valid_denoised = sorted(valid_denoised, key=lambda path: path.name.lower())
    clean_candidates = [f for f in sorted_valid_denoised if "(No Noise)" in f.name]

    if clean_candidates:
        return clean_candidates[0]

    return sorted_valid_denoised[0]


def _denoise_background_step(background_wav, denoised_background_dir, total_duration=None):
    """Step 4: Denoise Background (UVR-DeNoise-Lite) via Python API."""
    candidates_denoised = list(denoised_background_dir.glob("*.wav"))
    valid_denoised = [f for f in candidates_denoised if is_valid_audio(f)]

    if valid_denoised:
        log_msg("  [Step 3/5] Skipping Background Denoising (exists)")
        return _select_preferred_denoised_output(valid_denoised)

    log_msg("  [Step 3/5] Denoising Background (UVR-DeNoise-Lite - AI Engine)...")

    return _run_denoise_separator(
        input_wav=background_wav,
        denoised_output_dir=denoised_background_dir,
        selected_label="Selected Denoised Stem",
        warning_message="    [Warning] UVR-DeNoise failed. Using raw background.",
        error_message="Background Denoising failed",
        fallback_on_failure=True,
    )


def _denoise_full_audio_step(original_wav, denoised_audio_dir, total_duration=None):
    """Denoises the full extracted audio track for denoise_only mode."""
    candidates_denoised = list(denoised_audio_dir.glob("*.wav"))
    valid_denoised = [f for f in candidates_denoised if is_valid_audio(f)]

    if valid_denoised:
        log_msg("  [Step 2/4] Skipping Full-Audio Denoise (exists)")
        return _select_preferred_denoised_output(valid_denoised)

    log_msg("  [Step 2/4] Denoising Full Audio (UVR-DeNoise-Lite - AI Engine)...")

    return _run_denoise_separator(
        input_wav=original_wav,
        denoised_output_dir=denoised_audio_dir,
        selected_label="Selected Denoised Track",
        warning_message="    [Warning] UVR-DeNoise failed for full-audio mode.",
        error_message="Full-Audio Denoising failed",
        fallback_on_failure=False,
    )


def _ensure_audio_inputs_exist(*paths):
    for label, path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")


def _promote_valid_output(tmp_output, final_output_video, success_message):
    if not is_valid_video(tmp_output):
        if tmp_output.exists():
            tmp_output.unlink()
        raise Exception(success_message)

    if final_output_video.exists():
        final_output_video.unlink()
    tmp_output.rename(final_output_video)


def _build_mix_command(video_path, enhanced_vocals_wav, denoised_background_wav, tmp_output, threads):
    return [
        FFMPEG_BIN,
        "-stats",
        "-hide_banner",
        "-threads",
        str(threads),
        "-i",
        str(video_path),
        "-i",
        str(enhanced_vocals_wav),
        "-i",
        str(denoised_background_wav),
        "-map",
        "0:v",
        "-filter_complex",
        f"[1:a]volume={VOCAL_MIX_VOL}[v];[2:a]volume={BACKGROUND_MIX_VOL}[m];[v][m]amix=inputs=2:duration=longest:normalize=0[out]",
        "-map",
        "[out]",
        "-c:v",
        "copy",
        "-c:a",
        "pcm_f32le",
        "-shortest",
        "-y",
        str(tmp_output),
    ]


def _final_mix_step(video_path, enhanced_vocals_wav, denoised_background_wav, final_output_video, total_duration=None):
    """Step 5: Mix with FFmpeg."""
    if is_valid_video(final_output_video):
        log_msg(f"  [Step 5/5] Skipping Final Mix (exists: {final_output_video.name})")
        return

    log_msg("  [Step 5/5] Final Mix (32-bit Float)...")

    duration = total_duration or get_audio_duration_sec(enhanced_vocals_wav)
    _ensure_audio_inputs_exist(("Vocals", enhanced_vocals_wav), ("Background", denoised_background_wav))

    tmp_output = final_output_video.with_suffix(f".tmp{final_output_video.suffix}")

    def build_mix_cmd(threads):
        return _build_mix_command(video_path, enhanced_vocals_wav, denoised_background_wav, tmp_output, threads)

    attempt_cpu_run_with_retry(build_mix_cmd, CPU_THREADS, description="Final Mixing", total_duration=duration)
    _promote_valid_output(tmp_output, final_output_video, "Final Mix Failed: Output video invalid/empty.")
    log_msg(f"  [System] Success! Saved to: {final_output_video.name}")


def _build_single_audio_mux_command(video_path, processed_audio_wav, tmp_output, audio_codec, threads):
    return [
        FFMPEG_BIN,
        "-stats",
        "-hide_banner",
        "-threads",
        str(threads),
        "-i",
        str(video_path),
        "-i",
        str(processed_audio_wav),
        "-map",
        "0",
        "-map",
        "1:a",
        "-map",
        "-0:a",
        "-c:v",
        "copy",
        "-c:s",
        "copy",
        "-c:d",
        "copy",
        "-c:t",
        "copy",
        "-c:a",
        audio_codec,
        "-shortest",
        "-y",
        str(tmp_output),
    ]


def _final_mux_single_audio_step(video_path, processed_audio_wav, final_output_video, total_duration=None):
    """Final remux step for denoise_only mode (single processed track)."""
    if is_valid_video(final_output_video):
        log_msg(f"  [Step 4/4] Skipping Final Remux (exists: {final_output_video.name})")
        return True

    log_msg("  [Step 4/4] Final Remux (32-bit Float)...")

    duration = total_duration or get_audio_duration_sec(processed_audio_wav)
    _ensure_audio_inputs_exist(("Processed Audio", processed_audio_wav))

    tmp_output = final_output_video.with_suffix(f".tmp{final_output_video.suffix}")
    audio_codec = "aac" if final_output_video.suffix.lower() == ".mp4" else "pcm_f32le"

    def build_mux_cmd(threads):
        return _build_single_audio_mux_command(video_path, processed_audio_wav, tmp_output, audio_codec, threads)

    attempt_cpu_run_with_retry(build_mux_cmd, CPU_THREADS, description="Final Remux", total_duration=duration)
    _promote_valid_output(tmp_output, final_output_video, "Final Remux Failed: Output video invalid/empty.")
    log_msg(f"  [System] Success! Saved to: {final_output_video.name}")
    return True


def _log_video_duration(video_path):
    video_dur = get_video_duration_sec(video_path)
    if video_dur:
        log_msg(f"  [Info] Duration: {format_time(video_dur)}")
    return video_dur


def _process_denoise_only_mode(work_dir, original_wav, video_path, final_output_video, video_dur):
    denoised_audio_dir = work_dir / "denoised_full_audio"
    denoised_audio_dir.mkdir(exist_ok=True)

    denoised_full_audio_wav = _denoise_full_audio_step(original_wav, denoised_audio_dir, total_duration=video_dur)

    log_msg("  [Step 3/4] Smart Audio Sync (Full-Audio)...")
    aligned_full_audio = work_dir / f"aligned_{denoised_full_audio_wav.name}"
    _align_stems(original_wav, denoised_full_audio_wav, aligned_full_audio)

    _final_mux_single_audio_step(video_path, aligned_full_audio, final_output_video, total_duration=video_dur)


def _process_hybrid_mode(work_dir, original_wav, video_path, final_output_video, video_dur):
    separation_out_dir = work_dir / "separation"
    separation_out_dir.mkdir(exist_ok=True)

    vocals_wav, background_wav = _separate_stems_step(original_wav, separation_out_dir, total_duration=video_dur)
    enhanced_vocals_dir = work_dir / "enhanced_vocals"
    enhanced_vocals_wav = _enhance_vocals_step(vocals_wav, enhanced_vocals_dir, work_dir, total_duration=video_dur)

    denoised_background_dir = work_dir / "denoised_background"
    denoised_background_wav = _denoise_background_step(background_wav, denoised_background_dir, total_duration=video_dur)

    log_msg("  [Step 4/5] Smart Audio Sync (Sequential for clean output)...")
    log_msg("    [Info] Syncing Stems (Sequential for clean output)...")

    aligned_vocals = work_dir / f"aligned_{enhanced_vocals_wav.name}"
    _align_stems(original_wav, enhanced_vocals_wav, aligned_vocals)

    aligned_background = work_dir / f"aligned_{denoised_background_wav.name}"
    _align_stems(original_wav, denoised_background_wav, aligned_background)

    _final_mix_step(video_path, aligned_vocals, aligned_background, final_output_video, total_duration=video_dur)


def _cleanup_work_dir(work_dir, final_output_video):
    if not work_dir.exists() or KEEP_INPUT_FILES:
        return
    if is_valid_video(final_output_video):
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
        return
    log_msg(f"  [System] Preservation: Keeping {work_dir.name} for inspection on failure.", level="DEBUG")


def _resolve_output_dir(video_path, target_output_dir):
    if target_output_dir:
        return target_output_dir
    return video_path.parent


def _run_processing_mode(work_dir, original_wav, video_path, final_output_video, video_dur):
    if PROCESS_MODE == "denoise_only":
        _process_denoise_only_mode(work_dir, original_wav, video_path, final_output_video, video_dur)
        return
    _process_hybrid_mode(work_dir, original_wav, video_path, final_output_video, video_dur)


def process_hybrid_audio(video_path, gpu_name, target_output_dir=None):
    """Main Orchestrator."""
    log_msg(f"\n[System] Processing Task: {video_path.name}")
    del gpu_name

    if not video_path.exists():
        log_msg(f"  [Error] File not found: {video_path}", is_error=True)
        return False

    # Create safe working directory pattern
    # Use a hidden temp dir in the same location to ensure atomic moves work
    work_dir = video_path.parent / f".temp_work_{video_path.stem}"
    output_dir = _resolve_output_dir(video_path, target_output_dir)

    output_suffix = _get_output_suffix(PROCESS_MODE)
    final_output_video = output_dir / f"{video_path.stem}{output_suffix}{video_path.suffix}"

    # Resume checks: If final exists, skip
    if is_valid_video(final_output_video):
        log_msg("  [System] Output already exists. Skipping.")
        return True

    try:
        work_dir.mkdir(exist_ok=True)
        original_wav = work_dir / "original.wav"
        video_dur = _log_video_duration(video_path)
        _extract_audio_step(video_path, original_wav, total_duration=video_dur)
        _run_processing_mode(work_dir, original_wav, video_path, final_output_video, video_dur)

        log_msg(f"  [System] Task Completed: {video_path.name}")
        return True

    except Exception as e:
        log_msg(f"  [Error] Processing failed: {e}", is_error=True)
        return False

    finally:
        _cleanup_work_dir(work_dir, final_output_video)


# Need format_time for `process_hybrid_audio` duration logging
