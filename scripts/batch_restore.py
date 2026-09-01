#!/usr/bin/env python3
"""Batch audio restoration runner for multi-folder processing.

Processes all video files across designated directories using the configured
PROCESS_MODE restoration pipeline until every video has been cleanly restored.
"""

import sys
import time
from pathlib import Path

# Ensure modules package can be resolved
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import EXTS, PROCESS_MODE
from modules.hardware import get_gpu_name
from modules.processing import _get_output_suffix, process_hybrid_audio
from modules.ui import _is_cleaned_output, _show_banner
from modules.utils import check_dependencies, format_time, is_verified_video, log_msg

DEFAULT_TARGET_FOLDERS = [
    ".",
]


def _parse_target_directories(cli_args):
    """Resolves target directories from command line arguments or defaults."""
    if len(cli_args) > 1:
        return [Path(arg).resolve() for arg in cli_args[1:]]
    return [Path(folder).resolve() for folder in DEFAULT_TARGET_FOLDERS]


def _scan_folder_videos(folder_path):
    """Scans a single folder for valid, unprocessed video files."""
    if not folder_path.exists() or not folder_path.is_dir():
        log_msg(f">> [Warning] Directory not found or inaccessible: {folder_path}", is_error=True)
        return []

    found = []
    try:
        items = sorted(folder_path.iterdir(), key=lambda p: p.name.lower())
    except (PermissionError, OSError) as exc:
        log_msg(f">> [Warning] Directory not found or inaccessible: {folder_path} ({exc})", is_error=True)
        return []

    for item in items:
        if item.is_file() and item.suffix.lower() in EXTS and not _is_cleaned_output(item.name):
            found.append(item)
    return found


def _collect_video_queue(target_dirs):
    """Aggregates all candidate video files across all target directories."""
    queue = []
    for folder in target_dirs:
        videos = _scan_folder_videos(folder)
        log_msg(f"  [Queue] Discovered {len(videos)} video(s) in: {folder}")
        queue.extend(videos)
    return queue


def _is_already_restored(video_path):
    """Checks if a fully valid cleaned output video already exists for this input."""
    output_suffix = _get_output_suffix(PROCESS_MODE)
    output_path = video_path.parent / f"{video_path.stem}{output_suffix}{video_path.suffix}"
    return is_verified_video(output_path)


def _process_video_item(video_path, gpu_name, idx, total):
    """Processes a single video item, reporting timing and completion status."""
    log_msg(f"\n[{idx}/{total}] Processing: {video_path.name}")
    log_msg(f"       Folder: {video_path.parent}")

    if _is_already_restored(video_path):
        log_msg("       Status: Already restored and verified. Skipping.")
        return "SKIPPED"

    start_time = time.time()
    try:
        success = process_hybrid_audio(video_path, gpu_name, target_output_dir=video_path.parent)
    except Exception as exc:
        log_msg(f"       Exception during restoration: {exc}", is_error=True)
        success = False
    elapsed = time.time() - start_time

    if success:
        log_msg(f"       Status: Successfully Restored ({format_time(elapsed)})")
        return "SUCCESS"

    log_msg(f"       Status: Failed ({format_time(elapsed)})", is_error=True)
    return "FAILED"


def _print_batch_summary(results, total_time):
    """Prints the final summary report of the batch restoration process."""
    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    skipped_count = sum(1 for status in results.values() if status == "SKIPPED")
    fail_count = sum(1 for status in results.values() if status == "FAILED")

    print("\n" + "=" * 65)
    print("   BATCH RESTORATION SUMMARY REPORT")
    print("=" * 65)
    print(f"   Total Videos In Queue : {len(results)}")
    print(f"   Successfully Restored : {success_count}")
    print(f"   Skipped Videos        : {skipped_count}")
    print(f"   Failed Videos         : {fail_count}")
    print(f"   Total Elapsed Time    : {format_time(total_time)}")
    print("=" * 65)

    for path, status in results.items():
        print(f"   - [{status:<7}] {path.name} ({path.parent.name})")
    print("=" * 65 + "\n")


def run_batch_restoration(target_dirs=None):
    """Runs the unattended multi-folder batch restoration workflow."""
    if not check_dependencies():
        log_msg("[Init] Critical Error: Core dependencies missing.", is_error=True)
        return False

    gpu_name = get_gpu_name()
    _show_banner()

    dirs = target_dirs or _parse_target_directories(sys.argv)
    log_msg(f"\n[Batch] Scanning {len(dirs)} Target Folder(s)...")

    queue = _collect_video_queue(dirs)
    if not queue:
        log_msg("[Batch] No video files found to process.")
        return True

    log_msg(f"\n[Batch] Total queue: {len(queue)} video files. Mode: '{PROCESS_MODE}'")
    log_msg("[Batch] Starting batch pipeline...")

    results = {}
    batch_start = time.time()

    for idx, video_path in enumerate(queue, start=1):
        status = _process_video_item(video_path, gpu_name, idx, len(queue))
        results[video_path] = status

    total_time = time.time() - batch_start
    _print_batch_summary(results, total_time)
    return all(status in ("SUCCESS", "SKIPPED") for status in results.values())


def main():
    """CLI entry point for batch restoration."""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    success = run_batch_restoration()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
