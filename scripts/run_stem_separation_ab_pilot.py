#!/usr/bin/env python3
"""Run a reversible all-recordings stem-separation fidelity comparison."""

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path

# Ensure modules package can be resolved
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import config, processing
from modules.config import EXTS
from modules.hardware import get_gpu_name
from modules.processing import _get_output_suffix, get_video_duration_sec, process_hybrid_audio
from modules.ui import _is_cleaned_output
from modules.utils import FFMPEG_BIN, is_valid_video

SEPARATED_MODE = "auto_pure"
NO_STEM_MODE = "auto_pure_linear"
CLIP_DURATION_SECONDS = 30.0
CLIP_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="Directory containing original VHS captures.")
    parser.add_argument("output_dir", type=Path, help="New directory for A/B outputs, clips, and review sheets.")
    parser.add_argument("--skip-clips", action="store_true", help="Do not make randomized listening clips.")
    return parser.parse_args()


def _source_videos(source_dir):
    return [path for path in sorted(source_dir.iterdir()) if path.suffix.lower() in EXTS and not _is_cleaned_output(path.name)]


def _run_mode(video_path, output_dir, mode, gpu_name):
    """Runs one candidate mode while keeping output in its dedicated directory."""
    processing.PROCESS_MODE = mode
    config.PROCESS_MODE = mode
    config.CONFIG["process_mode"] = mode
    output_dir.mkdir(parents=True, exist_ok=True)
    success = process_hybrid_audio(video_path, gpu_name, target_output_dir=output_dir)
    output_path = output_dir / f"{video_path.stem}{_get_output_suffix(mode)}{video_path.suffix}"
    return success and is_valid_video(output_path), output_path


def _run_comparison(videos, output_dir):
    """Processes each source with the separated and full-mix candidates."""
    gpu_name = get_gpu_name()
    results = []
    for video_path in videos:
        separated_ok, separated = _run_mode(video_path, output_dir / "separated", SEPARATED_MODE, gpu_name)
        no_stem_ok, no_stem = _run_mode(video_path, output_dir / "no_stem", NO_STEM_MODE, gpu_name)
        results.append((video_path, separated, separated_ok, no_stem, no_stem_ok))
    return results


def _clip_starts(duration):
    """Returns deterministic in-range start times for matched listening clips."""
    latest_start = max(0.0, duration - CLIP_DURATION_SECONDS)
    return sorted({round(latest_start * fraction, 3) for fraction in CLIP_FRACTIONS})


def _extract_clip(source, destination, start):
    """Extracts one float-PCM audio clip without altering the source media."""
    command = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start),
        "-i",
        str(source),
        "-t",
        str(CLIP_DURATION_SECONDS),
        "-map",
        "0:a:0",
        "-c:a",
        "pcm_f32le",
        "-y",
        str(destination),
    ]
    try:
        return subprocess.run(command, check=False, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _write_csv(path, headings, rows):
    """Writes a UTF-8 CSV file with a supplied header and rows."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headings)
        writer.writerows(rows)


def _extract_pair_clips(clips_dir, trial, labels, start):
    """Extracts candidate clips for trial, returning True if both succeed."""
    for label, (_, source_path) in zip(("A", "B"), labels, strict=True):
        if not _extract_clip(source_path, clips_dir / f"{trial}_{label}.wav", start):
            return False
    return True


def _write_blind_trial(clips_dir, comparison, trial_data, randomizer):
    """Writes one randomized A/B pair and returns its review/key rows or None on failure."""
    source, separated, no_stem = comparison
    trial, start = trial_data
    labels = [("separated", separated), ("no_stem", no_stem)]
    randomizer.shuffle(labels)
    if not _extract_pair_clips(clips_dir, trial, labels, start):
        return None
    return [trial, "", "", "", "", ""], [trial, source.name, start, labels[0][0], labels[1][0]]


def _successful_trial_inputs(results):
    """Yields trial identifiers, starts, and valid paired candidate outputs."""
    trial_number = 1
    for source, separated, separated_ok, no_stem, no_stem_ok in results:
        if separated_ok and no_stem_ok:
            comparison = (source, separated, no_stem)
            for start in _clip_starts(get_video_duration_sec(source) or 0.0):
                yield comparison, (f"trial_{trial_number:04d}", start)
                trial_number += 1


def _write_blind_sheets(output_dir, review_rows, key_rows):
    """Writes the listener-facing scoresheet and the private randomization key."""
    _write_csv(
        output_dir / "blind_review.csv",
        [
            "trial",
            "natural_fidelity_1_to_5",
            "speech_distortion_1_to_5",
            "background_intrusion_1_to_5",
            "overall_quality_1_to_5",
            "intelligibility_1_to_5",
        ],
        review_rows,
    )
    _write_csv(output_dir / "blind_key.csv", ["trial", "source", "clip_start_seconds", "A_mode", "B_mode"], key_rows)


def _build_blind_materials(results, output_dir):
    """Creates randomized clips plus separated listener and key spreadsheets."""
    clips_dir = output_dir / "blind_clips"
    clips_dir.mkdir(exist_ok=True)
    review_rows, key_rows = [], []
    randomizer = random.Random(20260901)
    for comparison, trial_data in _successful_trial_inputs(results):
        trial_res = _write_blind_trial(clips_dir, comparison, trial_data, randomizer)
        if trial_res is not None:
            review_row, key_row = trial_res
            review_rows.append(review_row)
            key_rows.append(key_row)
    _write_blind_sheets(output_dir, review_rows, key_rows)


def _write_run_summary(results, output_dir):
    """Writes per-recording outcome paths and success states for the pilot."""
    rows = []
    for source, separated, separated_ok, no_stem, no_stem_ok in results:
        rows.append([source.name, separated_ok, separated, no_stem_ok, no_stem])
    _write_csv(
        output_dir / "processing_summary.csv",
        ["source", "separated_success", "separated_output", "no_stem_success", "no_stem_output"],
        rows,
    )


def main():
    """Runs the selected all-recordings pilot and reports incomplete comparisons."""
    args = _parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")
    videos = _source_videos(source_dir)
    if not videos:
        raise SystemExit(f"No supported source videos found: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = _run_comparison(videos, output_dir)
    _write_run_summary(results, output_dir)
    if not args.skip_clips:
        _build_blind_materials(results, output_dir)
    failures = [source.name for source, _, separated_ok, _, no_stem_ok in results if not separated_ok or not no_stem_ok]
    if failures:
        raise SystemExit(f"A/B pilot completed with failures: {', '.join(failures)}")


if __name__ == "__main__":
    main()
