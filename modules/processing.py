"""Audio processing, stem separation, AI enhancement, and native DSP filtering module.

Orchestrates multi-stage audio restoration pipelines across four modes:
- hybrid: AI Separation (BS-Roformer) + Neural Speech Enhancement (Resemble-Enhance) +
  Dynamic Vocal De-Essing + Background Denoise (UVR-DeNoise) + Downward Noise Expander +
  Sub-Sample Audio Synchronization + 32-bit Float amix with Optional EBU R128 Normalization.
- denoise_only: Full-audio neural denoise (UVR-DeNoise-Lite) + Smart Sync + Video Remux.
- vhs_native: Multi-threaded FFmpeg DSP filter chain (highpass + adeclick + afftdn + notch) + Sync + Remux.
- arnndn_speech: FFmpeg RNNoise recurrent neural network speech denoiser + Sync + Remux.
"""

import re
import shutil
import subprocess
import time

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import torch
except ImportError:
    torch = None

from . import mastering as _mastering
from . import utils as _utils
from .config import (
    ADAPTIVE_DENOISE_THRESHOLD_DB,
    DEFAULT_DENOISE_MODEL,
    DENOISE_MODEL,
    ENABLE_DEESSER,
    ENABLE_DYNAMIC_EXPANDER,
    ENHANCE_NFE,
    ENHANCE_TAU,
    KEEP_INPUT_FILES,
    OUTPUT_SUFFIX_BY_MODE,
    PRESERVE_ORIGINAL_AUDIO_TRACK,
    PROCESS_MODE,
    VOCALS_MODEL,
)
from .filters import (
    _filter_arnndn_step,
    _filter_auto_vhs_native_step,
    _filter_precondition_step,
    _filter_vhs_native_step,
    build_full_audio_polish_filter,
    build_post_denoise_cleanup_filter,
    build_pre_denoise_surgical_filter,
)
from .hardware import CPU_THREADS, CUDA_ENV, CUDA_VISIBLE_DEVICE, GPU_BATCH_SIZE
from .sync import _align_stems
from .utils import (
    FFMPEG_BIN,
    MODELS_DIR,
    attempt_cpu_run_with_retry,
    format_time,
    is_valid_audio,
    is_valid_video,
    log_msg,
    run_command_with_progress,
)

# Expose the GPU retry helper for test patching without aliasing to CPU retry.
attempt_run_with_retry = _utils.attempt_run_with_retry

# Every stage runs at this rate: extraction, intermediate DSP, and the final mix.
# ffmpeg's loudnorm upsamples internally for true-peak detection and does not come
# back down, so without an explicit resample the encoder inherits 96 kHz and the
# output file roughly doubles in size for no quality gain.
PIPELINE_SAMPLE_RATE = 44100


def _resolve_override(override, fallback):
    """Returns an explicit pipeline override when provided, else the configured fallback."""
    return fallback if override is None else override


def _strategy_value(strategy, key, default):
    """Reads a strategy-selected pipeline parameter, falling back to the configured default."""
    return _resolve_override((strategy or {}).get(key), default)


def _get_output_suffix(process_mode):
    """Maps process mode identifier to corresponding output filename suffix."""
    return OUTPUT_SUFFIX_BY_MODE.get(process_mode, OUTPUT_SUFFIX_BY_MODE["hybrid"])


def _get_audio_separator_class():
    """Dynamically imports and returns Separator class from audio_separator library."""
    try:
        from audio_separator.separator import Separator

        return Separator
    except ImportError as exc:
        raise ImportError("audio-separator is required for stem separation/denoising.") from exc


def get_audio_duration_sec(wav_path):
    """Calculates total duration in seconds of a WAV file using soundfile metadata."""
    if sf is None:
        return None

    try:
        with sf.SoundFile(str(wav_path)) as f:
            return f.frames / f.samplerate
    except Exception:
        return None


def get_video_duration_sec(video_path):  # pragma: no cover
    """Probes container metadata using ffprobe to obtain video duration in seconds."""
    try:
        cmd = [
            _utils.FFPROBE_BIN,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        val = subprocess.check_output(cmd).decode().strip()
        return float(val)
    except Exception:  # pragma: no cover
        return None


def _extract_audio_step(video_path, original_wav, total_duration=None):
    """Extracts raw input audio stream as 32-bit float stereo PCM (44.1 kHz)."""
    if is_valid_audio(original_wav):
        log_msg("  [System] Skipping Extraction (valid exists)")
        return

    if original_wav.exists():
        original_wav.unlink()

    log_msg("  [System] Extracting Audio Stream...")

    # Extract to .tmp first to maintain atomic guarantees
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
            str(PIPELINE_SAMPLE_RATE),
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


# audio-separator labels stems per model, not to one convention: BS-Roformer emits
# "(Vocals)"/"(Instrumental)", modern MelBand Roformers "(vocals)"/"(other)", and
# the crowd model - which the auto-scanner can select - "(crowd)"/"(other)". Matching
# only the first spelling made every other model fail stem detection outright.
FOREGROUND_STEM_TOKENS = ("vocals", "crowd", "speech", "dry")
BACKGROUND_STEM_TOKENS = ("instrumental", "background", "no vocals", "other")


def _matches_stem_token(path, tokens):
    """Checks a stem filename for any of the parenthesised labels a model may use."""
    name = path.name.lower()
    return any(f"({token})" in name for token in tokens)


def _collect_stem_candidates(separation_out_dir):
    """Scans directory for potential vocal and background stem outputs."""
    all_wavs = list(separation_out_dir.glob("*.wav"))
    vocals = [path for path in all_wavs if _matches_stem_token(path, FOREGROUND_STEM_TOKENS)]
    background = [path for path in all_wavs if _matches_stem_token(path, BACKGROUND_STEM_TOKENS)]
    return vocals, background, all_wavs


def _is_missing_background_pair(v_files, b_files, all_wavs):
    """Determines whether background stem fallback should be invoked."""
    return not b_files and len(all_wavs) == 2 and bool(v_files)


def _fallback_background_candidates(v_files, b_files, all_wavs):
    """Provides fallback stem list when background filename naming diverges."""
    if _is_missing_background_pair(v_files, b_files, all_wavs):
        return [f for f in all_wavs if f != v_files[0]]
    return b_files


def _sorted_valid_audio(paths):
    """Sorts audio paths by filesize descending and filters to valid files."""
    sorted_paths = sorted(paths, key=lambda path: path.stat().st_size, reverse=True)
    return [path for path in sorted_paths if is_valid_audio(path)]


def _select_valid_stem(v_files, b_files):
    """Selects highest-ranked valid vocals and background audio stems."""
    v_valid = _sorted_valid_audio(v_files)
    b_valid = _sorted_valid_audio(b_files)
    if v_valid and b_valid:
        return v_valid[0], b_valid[0], v_valid, b_valid
    return None, None, v_valid, b_valid


def _normalized_background_name(vocals):
    """Replaces a supported foreground label in a stem filename."""
    token_pattern = "|".join(re.escape(token) for token in FOREGROUND_STEM_TOKENS)
    new_name = re.sub(rf"\(({token_pattern})\)", "(Background)", vocals.name, count=1, flags=re.IGNORECASE)
    return None if new_name == vocals.name else new_name


def _rename_background_stem(background, new_path):
    """Renames a background stem while retaining the original on filesystem failure."""
    try:
        background.rename(new_path)
        return new_path
    except OSError:  # pragma: no cover
        return background


def _normalize_background_name(separation_out_dir, vocals, background):
    """Standardizes background stem filename to match vocal pairing."""
    if "(background)" in background.name.lower():
        return background

    new_name = _normalized_background_name(vocals)
    if new_name is None:
        return background
    new_path = separation_out_dir / new_name
    return background if new_path.exists() else _rename_background_stem(background, new_path)


def _log_separation_failure(separation_out_dir, v_valid, b_valid):
    """Logs debugging diagnostics when separation fails to produce expected stems."""
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
    """Configures audio-separator instance with dynamic batching parameters."""
    Separator = _get_audio_separator_class()
    model_dir = MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
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
    """Verifies that vocal and background stem outputs exist, raising on mismatch."""
    vocals_wav, background_wav = _verify_separation_output(separation_out_dir, original_wav)
    if vocals_wav and background_wav:
        return _ensure_float_pcm(vocals_wav), _ensure_float_pcm(background_wav)

    if hasattr(output_files, "__len__") and len(output_files) >= 2:
        log_msg(f"    [Debug] Separator returned: {output_files}", level="DEBUG")
    raise Exception("Separation completed but output stems were not identified.")


def _purge_corrupted_model_file(model_name):
    """Deletes a downloaded model file from the model store if it became corrupt."""
    candidate = MODELS_DIR / model_name
    if candidate.is_file():
        try:
            log_msg(f"    [Warning] Corrupt model '{model_name}' detected. Deleting to re-download...", is_error=True)
            candidate.unlink()
        except OSError:
            pass


def _load_separator_model(separator, model_name):
    """Loads model into separator, automatically purging and re-downloading if corrupt."""
    log_msg(f"    [AI] Loading Model: {model_name}")
    try:
        separator.load_model(model_filename=model_name)
    except Exception as exc:
        log_msg(f"    [Warning] Initial model load failed ({exc}). Purging local file and retrying...", is_error=True)
        _purge_corrupted_model_file(model_name)
        separator.load_model(model_filename=model_name)


def _separate_stems_step(original_wav, separation_out_dir, total_duration=None, vocals_model=None):
    """Step 2: Separate Stems (BS-Roformer) via Python API.

    Extracts isolated vocals and subtractive background tracks.
    """
    # 1. Check Existing
    existing_v, existing_b = _verify_separation_output(separation_out_dir, original_wav)
    if existing_v and existing_b:
        log_msg("  [Step 1/5] Skipping Separation (exists & valid)")
        return existing_v, existing_b

    log_msg("  [Step 1/5] Separating Stems (BS-Roformer - AI Engine)...")

    try:
        separator = _build_separator(separation_out_dir)

        model_name = _resolve_override(vocals_model, VOCALS_MODEL)
        _load_separator_model(separator, model_name)

        log_msg("    [AI] Starting Inference (GPU Accelerated)...")
        output_files = separator.separate(str(original_wav))

        if output_files is None:
            raise Exception("Separation completed but output stems were not identified.")

        return _resolve_separation_result(separation_out_dir, original_wav, output_files)

    except Exception as e:  # pragma: no cover
        log_msg(f"    [Error] AI Separation failed: {e}", is_error=True)
        raise e


def _clear_cuda_retry_state():
    """Flushes CUDA device cache and initiates Python garbage collection before retrying."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()


def _run_enhance_retry(cmd_enhance, total_duration, env=None):
    """Executes enhancement command with automatic GPU cache clearing and retry on transient errors."""
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
    """Ensures empty, clean working directory exists."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_result_to_final_dir(result, output_dir):
    """Copies temporary processing result to final output destination."""
    final_output = output_dir / result.name
    shutil.copy(result, final_output)
    return final_output


def _cleanup_directory(path):
    """Safely removes temporary working directory ignoring missing paths."""
    try:
        shutil.rmtree(path)
    except Exception:
        pass


def _handle_enhance_output(enhanced_vocals_dir, vocals_wav):
    """Verifies that Resemble-Enhance generated output, providing fallback if empty."""
    candidates_enhanced = list(enhanced_vocals_dir.glob("*.wav"))
    if not candidates_enhanced:
        log_msg("    [Warning] Resemble-Enhance did not produce output. Using raw vocals.", is_error=True)
        fb_path = enhanced_vocals_dir / f"fallback_{vocals_wav.name}"
        shutil.copy(vocals_wav, fb_path)
        return fb_path

    return candidates_enhanced[0]


def _enhance_vocals_step(vocals_wav, enhanced_vocals_dir, work_dir, total_duration=None, enhance_nfe=None, enhance_tau=None):
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
        "--nfe",
        str(_resolve_override(enhance_nfe, ENHANCE_NFE)),
        "--solver",
        "rk4",
        "--tau",
        str(_resolve_override(enhance_tau, ENHANCE_TAU)),
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
    """Checks whether any candidate files exist in denoised directory."""
    return bool(candidates_denoised)


def _handle_missing_denoised_candidate(warning_message, input_wav, fallback_on_failure):
    """Handles failure when separator produces no candidates."""
    if fallback_on_failure:
        log_msg(warning_message, is_error=True)
        return input_wav
    raise RuntimeError(warning_message.strip())


def _select_denoised_candidate(candidates_denoised, warning_message, input_wav, fallback_on_failure):
    """Selects primary denoised output file from generated candidates."""
    clean_candidates = [f for f in candidates_denoised if "(No Noise)" in f.name]
    if clean_candidates:
        return clean_candidates[0]
    if _has_denoised_candidates(candidates_denoised):
        return candidates_denoised[0]
    return _handle_missing_denoised_candidate(warning_message, input_wav, fallback_on_failure)


def _run_denoise_separator(
    input_wav, denoised_output_dir, selected_label, warning_message, error_message, fallback_on_failure=True, denoise_model=None
):
    """Runs UVR-DeNoise-Lite through audio-separator and selects best output."""
    try:
        separator = _build_separator(denoised_output_dir)

        model_name = _resolve_override(denoise_model, DENOISE_MODEL)
        _load_separator_model(separator, model_name)

        log_msg("    [AI] Starting Inference (GPU Accelerated)...")
        separator.separate(str(input_wav))

        candidates_denoised = sorted(denoised_output_dir.glob("*.wav"), key=lambda path: path.name.lower())
        result = _select_denoised_candidate(candidates_denoised, warning_message, input_wav, fallback_on_failure)
        log_msg(f"    {selected_label}: {result.name}")
        return _ensure_float_pcm(result)

    except Exception as e:
        log_msg(f"    [Error] {error_message}: {e}", is_error=True)
        if fallback_on_failure:
            return input_wav
        raise


def _write_float_blocks_atomic(wav_path, info, temp_path):
    """Streams audio blocks to a temporary float32 WAV and replaces target."""
    try:
        with sf.SoundFile(str(temp_path), mode="w", samplerate=info.samplerate, channels=info.channels, subtype="FLOAT") as out_f:
            for block in sf.blocks(str(wav_path), blocksize=65536, dtype="float32", always_2d=True):
                out_f.write(block)
        temp_path.replace(wav_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _ensure_float_pcm(wav_path):
    """Rewrites a stem as 32-bit float when a separator emitted fixed-point WAV.

    The pipeline's contract is 32-bit float end to end, but audio-separator falls
    back to PCM_16 whenever it cannot read the input subtype, silently dropping the
    intermediate precision every later stage assumes.

    Args:
        wav_path (pathlib.Path): Stem produced by a separator run.

    Returns:
        pathlib.Path: The same path, converted in place when it was fixed-point.
    """
    if sf is None:
        return wav_path

    try:
        info = sf.info(str(wav_path))
        if info.subtype != "FLOAT":
            temp_path = wav_path.with_name(f"{wav_path.stem}.float.tmp{wav_path.suffix}")
            _write_float_blocks_atomic(wav_path, info, temp_path)
    except Exception as exc:
        log_msg(f"    [Debug] Left {wav_path.name} at its original bit depth: {exc}", level="DEBUG")
    return wav_path


def _select_preferred_denoised_output(valid_denoised):
    """Pick a deterministic denoised output, preferring '(No Noise)' when present."""
    sorted_valid_denoised = sorted(valid_denoised, key=lambda path: path.name.lower())
    clean_candidates = [f for f in sorted_valid_denoised if "(No Noise)" in f.name]

    if clean_candidates:
        return clean_candidates[0]

    return sorted_valid_denoised[0]


def _denoise_vocals_step(vocals_wav, denoised_vocals_dir, total_duration=None, denoise_model=None):
    """Step 2b: Denoise Speech Stem (UVR-DeNoise-Lite) via Python API."""
    del total_duration
    candidates_denoised = list(denoised_vocals_dir.glob("*.wav"))
    valid_denoised = [f for f in candidates_denoised if is_valid_audio(f)]

    if valid_denoised:
        log_msg("  [Step 2/5] Skipping Speech Denoising (exists)")
        return _select_preferred_denoised_output(valid_denoised)

    log_msg("  [Step 2/5] Denoising Speech Stem (UVR-DeNoise-Lite - AI Engine)...")

    return _run_denoise_separator(
        input_wav=vocals_wav,
        denoised_output_dir=denoised_vocals_dir,
        selected_label="Selected Denoised Speech",
        warning_message="    [Warning] UVR-DeNoise failed for speech stem. Using raw vocals.",
        error_message="Speech Denoising failed",
        fallback_on_failure=True,
        denoise_model=denoise_model,
    )


def _denoise_background_step(background_wav, denoised_background_dir, total_duration=None, denoise_model=None):
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
        denoise_model=denoise_model,
    )


def _denoise_full_audio_step(original_wav, denoised_audio_dir, total_duration=None, denoise_model=None):
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
        denoise_model=denoise_model,
    )


def _final_mix_output_command(video_path, aligned_vocals, aligned_background, tmp_output_video, audio_args, threads, filter_expr=None):
    """Generates FFmpeg arguments for final 2-stem mix.

    Args:
        video_path (pathlib.Path): Source video file path.
        aligned_vocals (pathlib.Path): Aligned vocal WAV file path.
        aligned_background (pathlib.Path): Aligned background WAV file path.
        tmp_output_video (pathlib.Path): Destination temporary video file path.
        audio_args (list): Audio encoding parameters.
        threads (int): CPU worker threads count.
        filter_expr (str, optional): Prebuilt amix expression; built from config when omitted.

    Returns:
        list: FFmpeg CLI arguments.
    """
    filter_expr = _resolve_override(filter_expr, _build_mix_filter_expression())
    cmd = [
        FFMPEG_BIN,
        "-stats",
        "-hide_banner",
        "-threads",
        str(threads),
        "-i",
        str(video_path),
        "-i",
        str(aligned_vocals),
        "-i",
        str(aligned_background),
        "-filter_complex",
        filter_expr,
        "-map",
        "0:v:0",
        "-map",
        "[mixed]",
        "-c:v",
        "copy",
    ]
    if PRESERVE_ORIGINAL_AUDIO_TRACK:
        cmd.extend(["-map", "0:a:0?", *_preserved_audio_args(video_path.suffix, audio_args), *_scope_audio_args_for_stream(audio_args, 0)])
    else:
        cmd.extend(audio_args)
    cmd.extend(["-y", str(tmp_output_video)])
    return cmd


def _ensure_audio_inputs_exist(*named_audio_paths):
    """Validates that all input audio files exist before muxing.

    Args:
        *named_audio_paths (tuple): Pairs of (label, path) to check.

    Raises:
        FileNotFoundError: If any input audio file is missing.
    """
    for label, path in named_audio_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} input for mux: {path}")


def _promote_valid_output(tmp_path, final_path, error_message):
    """Atomically promotes valid temporary video file to final destination.

    Args:
        tmp_path (pathlib.Path): Path to candidate output file.
        final_path (pathlib.Path): Path to final destination file.
        error_message (str): Exception message if output is invalid.
    """
    if is_valid_video(tmp_path):
        if final_path.exists():
            final_path.unlink()
        tmp_path.rename(final_path)
        return

    if tmp_path.exists():
        tmp_path.unlink()
    raise Exception(error_message)


def _final_mix_step(
    video_path, aligned_vocals, aligned_background, final_output_video, total_duration=None, vocal_mix_vol=None, bg_mix_vol=None
):
    """Step 5: Final Mix (amix) - Lossless PCM Output."""
    if is_valid_video(final_output_video):
        log_msg(f"  [Step 5/5] Skipping Final Mix (exists: {final_output_video.name})")
        return True

    log_msg("  [Step 5/5] Final Audio Mix (32-bit Float)...")

    duration = total_duration or get_audio_duration_sec(aligned_vocals)
    _ensure_audio_inputs_exist(("Vocals", aligned_vocals), ("Background", aligned_background))

    tmp_output = final_output_video.with_suffix(f".tmp{final_output_video.suffix}")
    audio_args = _get_audio_encoding_args(final_output_video.suffix)
    loudnorm_args = _resolve_loudnorm_args(
        video_path, aligned_vocals, aligned_background, vocal_mix_vol, bg_mix_vol, total_duration=duration
    )
    filter_expr = _build_mix_filter_expression(vocal_mix_vol, bg_mix_vol, loudnorm_args)

    def build_mix_cmd(threads):
        return _final_mix_output_command(video_path, aligned_vocals, aligned_background, tmp_output, audio_args, threads, filter_expr)

    attempt_cpu_run_with_retry(build_mix_cmd, CPU_THREADS, description="Final Mixing", total_duration=duration)
    _promote_valid_output(tmp_output, final_output_video, "Final Mix Failed: Output video invalid/empty.")
    log_msg(f"  [System] Success! Saved to: {final_output_video.name}")
    return True


def _single_audio_stream_args(filter_expr):
    """Maps the processed track, through the mastering graph when one is built.

    A filter_complex label is used rather than -af because the preserved archival
    track is a second audio output; -af would normalise that one too.
    """
    if filter_expr is None:
        return ["-map", "1:a:0"]
    return ["-filter_complex", filter_expr, "-map", "[mastered]"]


def _build_single_audio_mux_command(video_path, processed_audio_wav, tmp_output_video, audio_args, threads, filter_expr=None):
    """Generates FFmpeg arguments for remuxing a single audio track.

    Args:
        video_path (pathlib.Path): Source video file path.
        processed_audio_wav (pathlib.Path): Processed single audio WAV track.
        tmp_output_video (pathlib.Path): Temporary video output destination.
        audio_args (list): Audio encoding parameters.
        threads (int): CPU worker threads count.
        filter_expr (str, optional): Mastering graph; the track is mapped directly
            when omitted.

    Returns:
        list: FFmpeg CLI arguments.
    """
    cmd = [
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
        "0:v:0",
        *_single_audio_stream_args(filter_expr),
        "-c:v",
        "copy",
    ]
    if PRESERVE_ORIGINAL_AUDIO_TRACK:
        cmd.extend(["-map", "0:a:0?", *_preserved_audio_args(video_path.suffix, audio_args), *_scope_audio_args_for_stream(audio_args, 0)])
    else:
        cmd.extend(audio_args)
    cmd.extend(["-y", str(tmp_output_video)])
    return cmd


def _final_mux_single_audio_step(video_path, processed_audio_wav, final_output_video, total_duration=None):
    """Final remux step for single processed track modes.

    Args:
        video_path (pathlib.Path): Source video file path.
        processed_audio_wav (pathlib.Path): Processed audio WAV track.
        final_output_video (pathlib.Path): Final destination video path.
        total_duration (float, optional): Total duration in seconds for progress bar.

    Returns:
        bool: True upon successful remux.
    """
    if is_valid_video(final_output_video):
        log_msg(f"  [Step 4/4] Skipping Final Remux (exists: {final_output_video.name})")
        return True

    log_msg("  [Step 4/4] Final Remux (32-bit Float)...")

    duration = total_duration or get_audio_duration_sec(processed_audio_wav)
    _ensure_audio_inputs_exist(("Processed Audio", processed_audio_wav))

    tmp_output = final_output_video.with_suffix(f".tmp{final_output_video.suffix}")
    audio_args = _get_audio_encoding_args(final_output_video.suffix)
    loudnorm_args = _resolve_single_track_loudnorm_args(video_path, processed_audio_wav, total_duration=duration)
    filter_expr = _build_single_audio_filter_expression(loudnorm_args)

    def build_mux_cmd(threads):
        return _build_single_audio_mux_command(video_path, processed_audio_wav, tmp_output, audio_args, threads, filter_expr)

    attempt_cpu_run_with_retry(build_mux_cmd, CPU_THREADS, description="Final Remux", total_duration=duration)
    _promote_valid_output(tmp_output, final_output_video, "Final Remux Failed: Output video invalid/empty.")
    log_msg(f"  [System] Success! Saved to: {final_output_video.name}")
    return True


def _log_video_duration(video_path):
    """Logs the detected video duration in human-readable format.

    Args:
        video_path (pathlib.Path): Path to video container file.

    Returns:
        float: Video duration in seconds.
    """
    video_dur = get_video_duration_sec(video_path)
    if video_dur:
        log_msg(f"  [Info] Duration: {format_time(video_dur)}")
    return video_dur


def _bind_step_model(step_func, model_kwarg, model_name):
    """Binds a strategy-selected model override onto a single-track pipeline step."""
    if model_name is None:
        return step_func

    def _bound_step(input_wav, out_dir, total_duration=None):
        return step_func(input_wav, out_dir, total_duration=total_duration, **{model_kwarg: model_name})

    return _bound_step


def _process_single_track_pipeline(
    work_dir,
    original_wav,
    video_path,
    final_output_video,
    video_dur,
    step_func,
    dir_name,
    sync_label,
    sync_method=None,
    ref_wav=None,
):
    """Executes single-track pipeline pattern: filter -> sync -> remux.

    Args:
        work_dir (pathlib.Path): Working directory path.
        original_wav (pathlib.Path): Audio WAV file to process through step_func.
        video_path (pathlib.Path): Original video container file.
        final_output_video (pathlib.Path): Final destination video file.
        video_dur (float): Video duration in seconds.
        step_func (callable): Filter step executor function.
        dir_name (str): Intermediate folder name.
        sync_label (str): Label for progress logging.
        sync_method (str, optional): Strategy-selected alignment method override.
        ref_wav (pathlib.Path, optional): Original reference audio for smart sync.
    """
    audio_dir = work_dir / dir_name
    audio_dir.mkdir(exist_ok=True)

    filtered_audio_wav = step_func(original_wav, audio_dir, total_duration=video_dur)

    log_msg(f"  [Step 3/4] Smart Audio Sync ({sync_label})...")
    aligned_audio = work_dir / f"aligned_{filtered_audio_wav.name}"
    sync_reference = ref_wav or original_wav
    _align_stems(sync_reference, filtered_audio_wav, aligned_audio, sync_method=sync_method)

    _final_mux_single_audio_step(video_path, aligned_audio, final_output_video, total_duration=video_dur)


def _resolve_preconditioned_audio(work_dir, original_wav, video_dur, mode, strategy):
    """Ensures strategy is available and applies analog pre-conditioning filters."""
    if strategy is None:
        from .auto_scanner import scan_and_decide_restoration_strategy

        strategy = scan_and_decide_restoration_strategy(original_wav, executed_mode=mode)
    precond_wav = work_dir / "preconditioned_audio.wav"
    precond_cfg = strategy.get("precondition_filters", {})
    clean_wav = _filter_precondition_step(original_wav, precond_wav, precond_cfg, total_duration=video_dur)
    return clean_wav, strategy


def _process_denoise_only_mode(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=None):
    """Full audio track denoising with UVR-DeNoise and preconditioning."""
    from .modes import DenoiseOnlyMode

    DenoiseOnlyMode().execute(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=strategy)


def _process_ffmpeg_native_mode(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=None):
    """FFmpeg native DSP filter chain restoration."""
    _process_single_track_pipeline(
        work_dir,
        original_wav,
        video_path,
        final_output_video,
        video_dur,
        _filter_vhs_native_step,
        "ffmpeg_native_audio",
        "FFmpeg Native",
        sync_method=_strategy_value(strategy, "sync_method", None),
    )


def _process_auto_ffmpeg_native_mode(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=None):
    """Intelligent adaptive FFmpeg native DSP restoration with auto-tuned acoustic scan."""
    _process_single_track_pipeline(
        work_dir,
        original_wav,
        video_path,
        final_output_video,
        video_dur,
        _filter_auto_vhs_native_step,
        "auto_ffmpeg_native_audio",
        "Auto-Tuned FFmpeg Native",
        sync_method=_strategy_value(strategy, "sync_method", None),
    )


def _process_arnndn_speech_mode(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=None):
    """FFmpeg RNNoise recurrent neural network speech restoration."""
    clean_wav, strategy = _resolve_preconditioned_audio(work_dir, original_wav, video_dur, "arnndn_speech", strategy)
    step_func = _bind_step_model(_filter_arnndn_step, "model_name", _strategy_value(strategy, "arnndn_model", None))
    _process_single_track_pipeline(
        work_dir,
        clean_wav,
        video_path,
        final_output_video,
        video_dur,
        step_func,
        "arnndn_speech_audio",
        "ARNNDN Speech",
        sync_method=_strategy_value(strategy, "sync_method", None),
        ref_wav=original_wav,
    )


def _run_dsp_filter_file(input_wav, output_wav, filter_expr, desc, total_duration):
    """Executes FFmpeg audio filter on a WAV file with robust fallback."""
    cmd = [
        FFMPEG_BIN,
        "-threads",
        str(CPU_THREADS),
        "-i",
        str(input_wav),
        "-af",
        filter_expr,
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(PIPELINE_SAMPLE_RATE),
        "-y",
        str(output_wav),
    ]
    try:
        run_command_with_progress(cmd, description=desc, total_duration=total_duration)
    except Exception as e:
        log_msg(f"    [Warning] {desc} failed: {e}", is_error=True)
    return output_wav if is_valid_audio(output_wav) else input_wav


def _deess_vocals_step(vocals_wav, enhanced_vocals_dir, total_duration=None):
    """Applies dynamic vocal de-essing to smooth harsh high-frequency consonants.

    Args:
        vocals_wav (pathlib.Path): Enhanced vocal WAV track.
        enhanced_vocals_dir (pathlib.Path): Working destination folder.
        total_duration (float, optional): Total duration in seconds for progress bar.

    Returns:
        pathlib.Path: De-essed vocal WAV track.
    """
    if not ENABLE_DEESSER or not is_valid_audio(vocals_wav):
        return vocals_wav
    output_wav = enhanced_vocals_dir / f"deessed_{vocals_wav.name}"
    if is_valid_audio(output_wav):
        return output_wav
    log_msg("    [AI Vocal Polish] Applying Dynamic Sibilance De-Esser...")
    return _run_dsp_filter_file(vocals_wav, output_wav, "deesser=i=0.5:m=0.5:f=0.5:s=o", "De-Essing Vocals", total_duration)


def _expand_background_step(background_wav, denoised_bg_dir, total_duration=None):
    """Applies smooth downward dynamic expansion on background audio below -45 dB.

    Args:
        background_wav (pathlib.Path): Denoised background WAV track.
        denoised_bg_dir (pathlib.Path): Working destination folder.
        total_duration (float, optional): Total duration in seconds for progress bar.

    Returns:
        pathlib.Path: Expanded background WAV track.
    """
    if not ENABLE_DYNAMIC_EXPANDER or not is_valid_audio(background_wav):
        return background_wav
    output_wav = denoised_bg_dir / f"expanded_{background_wav.name}"
    if is_valid_audio(output_wav):
        return output_wav
    log_msg("    [AI Background Polish] Applying Downward Dynamic Noise Expander...")
    exp_filter = "compand=attacks=0.1:decays=0.3:points=-80/-90|-45/-45|0/0"
    return _run_dsp_filter_file(background_wav, output_wav, exp_filter, "Expanding Background", total_duration)


def _polish_full_audio_step(denoised_wav, polish_dir, total_duration=None, strategy=None, apply_air=False):
    """Applies optional high-frequency air shelf and adaptive downward dynamic expansion."""
    if not is_valid_audio(denoised_wav):
        return denoised_wav
    polish_filter = build_full_audio_polish_filter(strategy=strategy, apply_air=apply_air)
    if not polish_filter:
        return denoised_wav
    output_wav = polish_dir / f"polished_{denoised_wav.name}"
    if is_valid_audio(output_wav):
        return output_wav
    log_msg("    [AI Full-Audio Polish] Applying Air Polish and Downward Dynamic Expander...")
    return _run_dsp_filter_file(denoised_wav, output_wav, polish_filter, "Polishing Full Audio", total_duration)


def _pre_denoise_surgical_step(precond_wav, audio_dir, total_duration=None, strategy=None):
    """Pass 2.5: Pre-denoise surgical DSP notching (mains harmonics, CRT whistle, rumble)."""
    if not is_valid_audio(precond_wav):
        return precond_wav
    surgical_filter = build_pre_denoise_surgical_filter(strategy=strategy)
    if not surgical_filter:
        return precond_wav
    output_wav = audio_dir / f"surgical_{precond_wav.name}"
    if is_valid_audio(output_wav):
        return output_wav
    log_msg("    [Surgical Pre-Denoise DSP] Applying Pre-Denoising Tonal Notches & Rumble Filter...")
    return _run_dsp_filter_file(precond_wav, output_wav, surgical_filter, "Pre-Denoise Surgical DSP", total_duration)


def _post_denoise_cleanup_step(denoised_wav, audio_dir, total_duration=None, strategy=None):
    """Pass 3.5: Post-denoise high-Q surgical cleanup for lingering tonal residuals."""
    if not is_valid_audio(denoised_wav):
        return denoised_wav
    cleanup_filter = build_post_denoise_cleanup_filter(strategy=strategy)
    if not cleanup_filter:
        return denoised_wav
    output_wav = audio_dir / f"cleaned_{denoised_wav.name}"
    if is_valid_audio(output_wav):
        return output_wav
    log_msg("    [Surgical Post-Denoise DSP] Notching Residual Tones Post-Denoise...")
    return _run_dsp_filter_file(denoised_wav, output_wav, cleanup_filter, "Post-Denoise Residual Cleanup", total_duration)


def _resolve_adaptive_denoise_model(strategy, default_model):
    """Picks lighter UVR-DeNoise model on clean recordings to prevent over-processing."""
    profile = (strategy or {}).get("profile", {})
    raw_nf = profile.get("noise_floor_db")
    if raw_nf is not None:
        try:
            nf_val = float(raw_nf)
            if nf_val < ADAPTIVE_DENOISE_THRESHOLD_DB:
                log_msg(
                    f"    [Adaptive Denoise] Quiet source ({nf_val:.1f} dB); overriding {default_model} with {DEFAULT_DENOISE_MODEL} to preserve transients."
                )
                return DEFAULT_DENOISE_MODEL
        except (ValueError, TypeError):
            pass
    return default_model


def _denoise_and_polish_full_audio_step(original_wav, audio_dir, total_duration=None, denoise_model=None, strategy=None, apply_air=False):
    """Cascades pre-denoise surgical DSP, neural denoising, post-cleanup, and adaptive polish."""
    model_to_use = _resolve_adaptive_denoise_model(strategy, denoise_model)
    surgical_wav = _pre_denoise_surgical_step(original_wav, audio_dir, total_duration=total_duration, strategy=strategy)
    denoise_sub_dir = audio_dir / "neural_denoised"
    denoise_sub_dir.mkdir(exist_ok=True)
    denoised_wav = _denoise_full_audio_step(surgical_wav, denoise_sub_dir, total_duration=total_duration, denoise_model=model_to_use)
    cleaned_wav = _post_denoise_cleanup_step(denoised_wav, audio_dir, total_duration=total_duration, strategy=strategy)
    return _polish_full_audio_step(cleaned_wav, audio_dir, total_duration=total_duration, strategy=strategy, apply_air=apply_air)


def _align_and_mix_stems(work_dir, original_wav, vocals_wav, background_wav, video_path, final_output_video, video_dur, strategy=None):
    """Aligns polished stems to the source timing and renders the final 2-stem mix.

    Args:
        work_dir (pathlib.Path): Working temporary directory path.
        original_wav (pathlib.Path): Original reference WAV file for sync.
        vocals_wav (pathlib.Path): Polished speech stem.
        background_wav (pathlib.Path): Polished background stem.
        video_path (pathlib.Path): Source video file.
        final_output_video (pathlib.Path): Output video file.
        video_dur (float): Duration in seconds.
        strategy (dict, optional): Pre-computed acoustic restoration strategy.
    """
    sync_method = _strategy_value(strategy, "sync_method", None)

    aligned_vocals = work_dir / f"aligned_{vocals_wav.name}"
    _align_stems(original_wav, vocals_wav, aligned_vocals, sync_method=sync_method)

    aligned_background = work_dir / f"aligned_{background_wav.name}"
    _align_stems(original_wav, background_wav, aligned_background, sync_method=sync_method)

    _final_mix_step(
        video_path,
        aligned_vocals,
        aligned_background,
        final_output_video,
        total_duration=video_dur,
        vocal_mix_vol=_strategy_value(strategy, "vocal_mix_vol", None),
        bg_mix_vol=_strategy_value(strategy, "bg_mix_vol", None),
    )


def _execute_hybrid_restoration(work_dir, input_wav, original_wav, video_path, final_output_video, video_dur, strategy=None):
    """Executes 2-stem AI separation, speech enhancement, background denoising, sync, and mixing.

    Args:
        work_dir (pathlib.Path): Working temporary directory path.
        input_wav (pathlib.Path): Pre-conditioned or raw input WAV file.
        original_wav (pathlib.Path): Original reference WAV file for sync.
        video_path (pathlib.Path): Source video file.
        final_output_video (pathlib.Path): Output video file.
        video_dur (float): Duration in seconds.
        strategy (dict, optional): Pre-computed acoustic restoration strategy.
    """
    separation_out_dir = work_dir / "separation"
    separation_out_dir.mkdir(exist_ok=True)

    vocals_wav, background_wav = _separate_stems_step(
        input_wav, separation_out_dir, total_duration=video_dur, vocals_model=_strategy_value(strategy, "vocals_model", None)
    )
    enhanced_vocals_dir = work_dir / "enhanced_vocals"
    enhanced_vocals_wav = _enhance_vocals_step(
        vocals_wav,
        enhanced_vocals_dir,
        work_dir,
        total_duration=video_dur,
        enhance_nfe=_strategy_value(strategy, "enhance_nfe", None),
        enhance_tau=_strategy_value(strategy, "enhance_tau", None),
    )

    polished_vocals_dir = work_dir / "polished_vocals"
    polished_vocals_dir.mkdir(exist_ok=True)
    polished_vocals_wav = _deess_vocals_step(enhanced_vocals_wav, polished_vocals_dir, total_duration=video_dur)

    denoised_background_dir = work_dir / "denoised_background"
    denoised_background_wav = _denoise_background_step(
        background_wav, denoised_background_dir, total_duration=video_dur, denoise_model=_strategy_value(strategy, "denoise_model", None)
    )
    polished_background_wav = _expand_background_step(denoised_background_wav, denoised_background_dir, total_duration=video_dur)

    _align_and_mix_stems(
        work_dir, original_wav, polished_vocals_wav, polished_background_wav, video_path, final_output_video, video_dur, strategy
    )


def _process_hybrid_mode(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=None):
    clean_wav, strategy = _resolve_preconditioned_audio(work_dir, original_wav, video_dur, "hybrid", strategy)
    _execute_hybrid_restoration(work_dir, clean_wav, original_wav, video_path, final_output_video, video_dur, strategy=strategy)


def _execute_pure_restoration(work_dir, input_wav, original_wav, video_path, final_output_video, video_dur, strategy=None):
    """Executes 2-stem pure restoration: stem separation + speech/ambient denoising without vocoder.

    Args:
        work_dir (pathlib.Path): Working temporary directory path.
        input_wav (pathlib.Path): Pre-conditioned or raw input WAV file.
        original_wav (pathlib.Path): Original reference WAV file for sync.
        video_path (pathlib.Path): Source video file.
        final_output_video (pathlib.Path): Output video file.
        video_dur (float): Duration in seconds.
        strategy (dict, optional): Pre-computed acoustic restoration strategy.
    """
    separation_out_dir = work_dir / "separation"
    separation_out_dir.mkdir(exist_ok=True)
    denoise_model = _strategy_value(strategy, "denoise_model", None)

    vocals_wav, background_wav = _separate_stems_step(
        input_wav, separation_out_dir, total_duration=video_dur, vocals_model=_strategy_value(strategy, "vocals_model", None)
    )

    denoised_vocals_dir = work_dir / "denoised_vocals"
    denoised_vocals_wav = _denoise_vocals_step(vocals_wav, denoised_vocals_dir, total_duration=video_dur, denoise_model=denoise_model)

    polished_vocals_dir = work_dir / "polished_vocals"
    polished_vocals_dir.mkdir(exist_ok=True)
    polished_vocals_wav = _deess_vocals_step(denoised_vocals_wav, polished_vocals_dir, total_duration=video_dur)

    denoised_background_dir = work_dir / "denoised_background"
    denoised_background_wav = _denoise_background_step(
        background_wav, denoised_background_dir, total_duration=video_dur, denoise_model=denoise_model
    )
    polished_background_wav = _expand_background_step(denoised_background_wav, denoised_background_dir, total_duration=video_dur)

    _align_and_mix_stems(
        work_dir, original_wav, polished_vocals_wav, polished_background_wav, video_path, final_output_video, video_dur, strategy
    )


def _process_auto_pure_mode(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=None):
    """4-Pass Cascaded Pure Restoration: Pre-Scan -> Pre-Conditioning -> Separation & Denoise -> Mix."""
    from .modes import AutoPureMode

    AutoPureMode().execute(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=strategy)


def _process_auto_pure_linear_mode(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=None):
    """Runs pure full-mix restoration without stem separation."""
    from .modes import AutoPureLinearMode

    AutoPureLinearMode().execute(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=strategy)


def _process_cathar_mode(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=None):
    """Executes pure-Rust Cathar DSP restoration pipeline tailored for VHS captures."""
    from .modes import CatharMode

    CatharMode().execute(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=strategy)


def _process_multipass_mode(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=None):
    """4-Pass Cascaded Restoration: Pre-Scan -> Pre-Conditioning -> AI Separation -> Polish & Sync."""
    from .modes import MultiPassMode

    MultiPassMode().execute(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=strategy)


def _get_mode_pipeline_handler(target_mode):
    """Maps restoration mode identifier to its execution pipeline callable.

    Args:
        target_mode (str): Restoration mode identifier string.

    Returns:
        callable: Function implementing the target restoration pipeline.
    """
    from .modes.registry import get_mode_instance

    return get_mode_instance(target_mode).execute


def _dispatch_mode_pipeline(target_mode, work_dir, original_wav, video_path, final_output_video, video_dur, strategy=None):
    """Dispatches execution to designated restoration mode pipeline.

    Args:
        target_mode (str): Destination restoration pipeline mode.
        work_dir (pathlib.Path): Working directory path.
        original_wav (pathlib.Path): Original extracted audio WAV file.
        video_path (pathlib.Path): Source video file.
        final_output_video (pathlib.Path): Destination output video file.
        video_dur (float): Video duration in seconds.
        strategy (dict, optional): Pre-computed acoustic restoration strategy.
    """
    handler = _get_mode_pipeline_handler(target_mode)
    handler(work_dir, original_wav, video_path, final_output_video, video_dur, strategy=strategy)


def _process_auto_mode(work_dir, original_wav, video_path, final_output_video, video_dur):
    """AI auto-detects scene acoustic characteristics and dispatches best restoration pipeline.

    Args:
        work_dir (pathlib.Path): Working directory path.
        original_wav (pathlib.Path): Original extracted audio WAV file.
        video_path (pathlib.Path): Source video file.
        final_output_video (pathlib.Path): Destination output video file.
        video_dur (float): Video duration in seconds.
    """
    from .auto_scanner import scan_and_decide_restoration_strategy
    from .config import ENABLE_MULTIPASS

    strategy = scan_and_decide_restoration_strategy(original_wav)
    target_mode = strategy["mode"]
    if ENABLE_MULTIPASS and target_mode == "hybrid":
        _dispatch_mode_pipeline("multipass_auto", work_dir, original_wav, video_path, final_output_video, video_dur, strategy=strategy)
        return
    _dispatch_mode_pipeline(target_mode, work_dir, original_wav, video_path, final_output_video, video_dur, strategy=strategy)


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
    if PROCESS_MODE == "auto":
        _process_auto_mode(work_dir, original_wav, video_path, final_output_video, video_dur)
        return
    _dispatch_mode_pipeline(PROCESS_MODE, work_dir, original_wav, video_path, final_output_video, video_dur)


def process_hybrid_audio(video_path, gpu_name, target_output_dir=None):
    """Main Orchestrator."""
    log_msg(f"\n[System] Processing Task: {video_path.name}")
    del gpu_name

    if not video_path.exists():
        log_msg(f"  [Error] File not found: {video_path}", is_error=True)
        return False

    work_dir = video_path.parent / f".temp_work_{video_path.stem.strip()}"
    output_dir = _resolve_output_dir(video_path, target_output_dir)
    output_suffix = _get_output_suffix(PROCESS_MODE)
    final_output_video = output_dir / f"{video_path.stem}{output_suffix}{video_path.suffix}"

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


# Re-export mastering symbols for backward compatibility with existing tests and imports
AUDIO_CODEC_ARGS_BY_EXT = _mastering.AUDIO_CODEC_ARGS_BY_EXT
LOUDNORM_ANALYSIS_TIMEOUT = _mastering.LOUDNORM_ANALYSIS_TIMEOUT
LOUDNORM_MEASURE_KEYS = _mastering.LOUDNORM_MEASURE_KEYS
LOUDNORM_TARGET_I = _mastering.LOUDNORM_TARGET_I
LOUDNORM_TARGET_LRA = _mastering.LOUDNORM_TARGET_LRA
LOUDNORM_TARGET_TP = _mastering.LOUDNORM_TARGET_TP
LOUDNORM_TRUE_PEAK_LIMITER = _mastering.LOUDNORM_TRUE_PEAK_LIMITER
_build_mix_base_expression = _mastering._build_mix_base_expression
_build_mix_filter_expression = _mastering._build_mix_filter_expression
_build_single_audio_filter_expression = _mastering._build_single_audio_filter_expression
_extract_valid_loudnorm_object = _mastering._extract_valid_loudnorm_object
_get_audio_encoding_args = _mastering._get_audio_encoding_args
_has_loudnorm_measurements = _mastering._has_loudnorm_measurements
_loudnorm_analysis_timeout = _mastering._loudnorm_analysis_timeout
_loudnorm_target_args = _mastering._loudnorm_target_args
_mastering_chain = _mastering._mastering_chain
_measured_loudnorm_args = _mastering._measured_loudnorm_args
_parse_loudnorm_json = _mastering._parse_loudnorm_json
_preserved_audio_args = _mastering._preserved_audio_args
_resolve_loudnorm_args = _mastering._resolve_loudnorm_args
_resolve_measured_loudnorm = _mastering._resolve_measured_loudnorm
_resolve_single_track_loudnorm_args = _mastering._resolve_single_track_loudnorm_args
_run_loudness_analysis = _mastering._run_loudness_analysis
_sanitize_mix_level = _mastering._sanitize_mix_level
_scope_audio_arg = _mastering._scope_audio_arg
_scope_audio_args_for_stream = _mastering._scope_audio_args_for_stream

# Re-export config symbols for test compatibility
ENABLE_LOUDNORM = _mastering.ENABLE_LOUDNORM
VOCAL_MIX_VOL = _mastering.VOCAL_MIX_VOL
BACKGROUND_MIX_VOL = _mastering.BACKGROUND_MIX_VOL
