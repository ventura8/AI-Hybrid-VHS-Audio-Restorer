#!/usr/bin/env python3
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from modules.config import BATCH_JOBS, PROCESS_MODE
from modules.processing import process_hybrid_audio
from modules.ui import _get_input_files, _show_banner, run_init_sequence
from modules.utils import check_dependencies

# Ensure local modules can be imported
sys.path.append(str(Path(__file__).parent))

HELP_TEXT = """\
AI Hybrid VHS Audio Restorer

Usage:
  start.sh  [PATH ...]                  (Linux / macOS)
  start.bat [PATH ...]                  (Windows)
  python restore_audio_hybrid.py [PATH ...]

Arguments:
  PATH         One or more video files or folders to restore. The cleaned file
               is written next to each source with a mode-specific
               '*_Cleaned' suffix.

Options:
  -h, --help   Show this help message and exit.

With no arguments the app starts in interactive mode: drag & drop a file, or
press Enter to scan the 'input/' folder. The restoration engine is chosen by
'process_mode' in config.yaml (current: {mode})."""


def _wants_help(argv):
    """Returns True when the CLI arguments request the help screen."""
    return any(arg in ("-h", "--help") for arg in argv)


def _print_help():
    """Prints CLI usage, including the currently configured process mode."""
    print(HELP_TEXT.format(mode=PROCESS_MODE))


if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    if _wants_help(sys.argv[1:]):
        _print_help()
        return
    _run_restoration()


def _run_restoration():
    # 1. Initialization Sequence
    cpu_name, gpu_name = run_init_sequence()

    if not check_dependencies():
        print("\n[Init] Critical Error: Dependencies Missing.")
        return

    # 2. Show Banner
    # We retrieve names again or pass them. _show_banner gets them internally
    # but run_init_sequence returned them.
    # Current UI implementation of _show_banner calls get_cpu_name() again.
    # It prints the banner.
    _show_banner()

    print("-" * 60)
    print(" [HOW TO USE]")
    print(" 1. Drag and Drop a video file (or folder) here.")
    print(" 2. Or paste the file path below.")

    # 3. Get Inputs
    files, use_source_as_output = _get_input_files()

    if not files:
        print(">> No valid video files found.")
        return

    print(f"\n[System] Found {len(files)} files in queue.")
    print("[System] Starting Batch Processing...")

    # 4. Processing Loop
    _process_files(files, gpu_name)

    print("\n" + "=" * 60)
    print("   BATCH PROCESSING COMPLETE")
    print("=" * 60)
    try:
        input("Press Enter to exit...")
    except (EOFError, KeyboardInterrupt):
        print(">> Exiting gracefully.")


def _process_files(files, gpu_name, jobs=None):
    """The batch: `batch_jobs` files at a time in child interpreters, or one after another in this one."""
    jobs = BATCH_JOBS if jobs is None else jobs
    if jobs > 1 and len(files) > 1:
        _run_parallel_batch(files, jobs)
        return
    for video_path in files:
        # User requested output to ALWAYS be in the input folder (from original logic)
        process_hybrid_audio(video_path, gpu_name, target_output_dir=video_path.parent)


def _run_parallel_batch(files, jobs):
    """Restores `files` `jobs` at a time, each in its own interpreter with its own log under logs/.

    A child is this same entry point on one file, so every file keeps the exact output of a
    solo run (its own work directory, its own scanner profile); the parent only reports.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    print(f"[System] {len(files)} files, {jobs} at a time; per-file logs under {log_dir}/")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda path: _restore_in_child(path, log_dir), files))
    failed = [path.name for path, code in zip(files, results) if code != 0]
    if failed:
        print(f"[System] {len(failed)} of {len(files)} files failed: {', '.join(failed)}")


def _restore_in_child(video_path, log_dir):
    """One file through a child interpreter; returns its exit code."""
    log_path = log_dir / f"{video_path.stem}.log"
    started = time.time()
    print(f"[Batch] start  {video_path.name}")
    with open(log_path, "w", encoding="utf-8") as log:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(video_path)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            check=False,
        )
    state = "done  " if completed.returncode == 0 else f"FAILED ({completed.returncode})"
    print(f"[Batch] {state} {video_path.name} in {time.time() - started:.0f} s (log: {log_path})")
    return completed.returncode


if __name__ == "__main__":
    main()
