"""Resemble-Enhance's denoiser as an optional neural stage for the full-mix chain.

The package ships two models: a denoiser, a masking UNet on the complex spectrogram trained
at 44.1 kHz, and an enhancer, a flow-matching resynthesis. Only the denoiser is a candidate
here -- a masking model cannot put on a tape what was never there, which is the line this
mode holds -- and it is invoked with `--denoise_only`. The `hybrid` and `auto` modes have
used the package for years; this is the first time its denoiser alone is measured against
UVR-DeNoise in `auto_pure_linear`'s chain. As with DeepFilterNet, the stage is opt-in and
any failure falls back to the UVR path.
"""

import shutil
import subprocess
from pathlib import Path

from . import enhance_chunking as _chunking
from .hardware import CUDA_ENV, CUDA_VISIBLE_DEVICE
from .utils import is_valid_audio, log_msg, run_command_with_progress

STAGE_FAILURES = (OSError, RuntimeError, ValueError, subprocess.SubprocessError)


def _command(input_dir, output_dir):
    """The denoise-only invocation of the package's CLI over a directory."""
    return ["resemble-enhance", str(input_dir), str(output_dir), "--denoise_only", "--device", CUDA_VISIBLE_DEVICE]


def _fresh(directory):
    """An empty directory at the path, whatever was there."""
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def denoise(input_wav, output_dir, total_duration=None):
    """Runs the denoiser over the file, chunked when the card cannot hold it; returns the output path, or None.

    The package processes every file in a directory, so the input is staged in one of its
    own and the chunks, when the clip was split, are rejoined with the crossfade the
    hybrid mode uses.
    """
    input_wav = Path(input_wav)
    staging = _fresh(Path(output_dir) / "input")
    produced = _fresh(Path(output_dir) / "output")
    chunk_paths = _chunking.prepare_enhance_input(input_wav, staging, total_duration)
    run_command_with_progress(_command(staging, produced), env=CUDA_ENV, description="Resemble Denoise", total_duration=total_duration)
    result = _chunking.collect_enhance_result(produced, input_wav, chunk_paths)
    shutil.rmtree(staging, ignore_errors=True)
    if result is None or result.name.startswith("fallback_") or not is_valid_audio(result):
        return None
    target = Path(output_dir) / f"resemble_{input_wav.name}"
    shutil.move(str(result), str(target))
    return target


def denoise_or(input_wav, output_dir, wanted, fallback, total_duration=None):
    """Runs the Resemble denoiser when asked for and working, else whatever the caller falls back to."""
    if wanted:
        try:
            produced = denoise(input_wav, output_dir, total_duration)
        except STAGE_FAILURES as exc:
            log_msg(f"    [Resemble Denoise] Failed ({type(exc).__name__}: {str(exc)[:80]}); falling back.")
            produced = None
        if produced is not None:
            log_msg("    [Resemble Denoise] Denoised the full mix.")
            return produced
    return fallback()
