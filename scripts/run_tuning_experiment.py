#!/usr/bin/env python3
"""Run tuning experiments on VHS captures across different restoration modes."""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure modules package can be resolved
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.hardware import get_gpu_name
from modules.processing import process_hybrid_audio
from modules.utils import FFMPEG_BIN, is_valid_video


def _extract_sample_clip(input_video, sample_output, start_sec, duration_sec):
    """Extracts a test video clip with pcm_f32le audio for fast restoration evaluation."""
    import subprocess

    sample_output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(input_video),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-c:a",
        "pcm_f32le",
        "-ar",
        "44100",
        str(sample_output),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return is_valid_video(sample_output)


def _setup_experiment_target(tape_path, output_dir, duration_sec):
    """Prepares destination directory and sample video path."""
    project_root = Path(__file__).resolve().parent.parent
    if output_dir:
        target_out = Path(output_dir)
    else:
        target_out = project_root / "experiments" / "restoration_runs"
    target_out.mkdir(parents=True, exist_ok=True)
    sample_video = target_out / f"{tape_path.stem}_sample_{duration_sec}s.mp4"
    return target_out, sample_video


def run_experiment(tape_path, mode, duration_sec=90, start_sec=30, output_dir=None, device=None):
    """Extracts a segment and runs the restoration mode to evaluate quality."""
    tape_path = Path(tape_path)

    if not tape_path.exists():
        print(f"File not found: {tape_path}", file=sys.stderr)
        return False

    target_out, sample_video = _setup_experiment_target(tape_path, output_dir, duration_sec)
    print(f"Extracting {duration_sec}s sample from {tape_path.name}...")
    if not _extract_sample_clip(tape_path, sample_video, start_sec, duration_sec):
        print("Failed to create sample video", file=sys.stderr)
        return False

    gpu_name = device if device else get_gpu_name()
    print(f"Running mode: '{mode}' on {sample_video.name} using {gpu_name}...")
    from scripts.ia_benchmark_common import apply_mode_override

    apply_mode_override(mode)

    success = process_hybrid_audio(sample_video, gpu_name, target_output_dir=target_out)
    print(f"Mode '{mode}' finished with success: {success}")
    return success


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True, help="Path to input video/audio file")
    parser.add_argument("--mode", type=str, default="cathar", help="Restoration mode to run")
    parser.add_argument("--duration", type=int, default=90, help="Clip duration in seconds")
    parser.add_argument("--start", type=int, default=30, help="Clip start offset in seconds")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output destination directory")
    parser.add_argument("--device", type=str, default=None, help="Processing device (GPU/CPU)")
    return parser.parse_args()


def main():
    args = _parse_args()
    success = run_experiment(
        args.tape,
        args.mode,
        duration_sec=args.duration,
        start_sec=args.start,
        output_dir=args.output_dir,
        device=args.device,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
