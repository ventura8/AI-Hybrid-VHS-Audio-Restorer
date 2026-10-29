#!/usr/bin/env python3
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from modules.config import APP_DIR, BATCH_JOBS, PROCESS_MODE
from modules.processing import process_hybrid_audio
from modules.ui import _get_input_files, _show_banner, run_init_sequence, stdin_is_console
from modules.utils import check_dependencies, one_line, open_private_log

# The batch's per-file logs sit beside the application, like session_log.txt, never in the launch directory: a
# folder of tapes shipping logs/<stem>.log as a symlink to ~/.bashrc or ~/.ssh/authorized_keys would otherwise have
# each child's stdout, which prints the file's name, written through it (security review, 2026-10-19).
BATCH_LOG_DIR = APP_DIR / "logs"

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
press Enter to scan the 'input/' folder. Without a terminal on stdin it asks
nothing: pass the paths as arguments. The restoration engine is chosen by
'process_mode' in config.yaml (current: {mode}).

Exit status: 0 when every file was restored (or already was), 1 when any file
failed or was skipped, 2 when nothing ran (dependencies missing, no files)."""

# The process exit status a batch reports (scripts launching the app read it).
EXIT_OK = 0
EXIT_TASKS_FAILED = 1
EXIT_NOTHING_RAN = 2


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
    """Runs the app; returns the process exit status (EXIT_OK, EXIT_TASKS_FAILED, EXIT_NOTHING_RAN)."""
    if _wants_help(sys.argv[1:]):
        _print_help()
        return EXIT_OK
    return _run_restoration()


def _wait_for_console_user():
    """Keeps a console window open until Enter; a batch without a terminal on stdin never waits."""
    if not stdin_is_console():
        return
    try:
        input("Press Enter to exit...")
    except (EOFError, KeyboardInterrupt):
        print(">> Exiting gracefully.")


def _run_restoration():
    # 1. Initialization Sequence
    _, gpu_name = run_init_sequence()

    if not check_dependencies():
        print("\n[Init] Critical Error: Dependencies Missing.")
        return EXIT_NOTHING_RAN

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
    files, _ = _get_input_files()

    if not files:
        print(">> No valid video files found.")
        return EXIT_NOTHING_RAN

    print(f"\n[System] Found {len(files)} files in queue.")
    print("[System] Starting Batch Processing...")

    # 4. Processing Loop
    failed = _process_files(files, gpu_name)

    print("\n" + "=" * 60)
    print("   BATCH PROCESSING COMPLETE")
    print("=" * 60)
    _wait_for_console_user()
    return EXIT_TASKS_FAILED if failed else EXIT_OK


def _report_failures(failed, total):
    """Names the files that failed or were skipped; returns them."""
    if failed:
        print(f"[System] {len(failed)} of {total} files failed: {', '.join(one_line(name) for name in failed)}")
    return failed


def _process_files(files, gpu_name, jobs=None):
    """The batch: `batch_jobs` files at a time in child interpreters, or one after another in this one.

    Returns the names of the files that failed or were skipped.
    """
    jobs = BATCH_JOBS if jobs is None else jobs
    if jobs > 1 and len(files) > 1:
        return _run_parallel_batch(files, jobs)
    return _report_failures(_restore_in_this_process(files, gpu_name), len(files))


def _restore_in_this_process(files, gpu_name):
    """One file after another in this interpreter, each output beside its source; returns the names that failed."""
    return [path.name for path in files if not process_hybrid_audio(path, gpu_name, target_output_dir=path.parent)]


def _batch_log_dir():
    """The per-file log folder beside the application, created; raises OSError when it is a symlink."""
    if BATCH_LOG_DIR.is_symlink():
        raise OSError(f"refusing to follow a symlinked batch log folder: {BATCH_LOG_DIR}")
    BATCH_LOG_DIR.mkdir(exist_ok=True)
    return BATCH_LOG_DIR


# Characters no Windows file name may hold; a stem from Linux or macOS may carry them.
_UNSAFE_NAME = str.maketrans(dict.fromkeys('<>:"/\\|?*', "_"))


def batch_log_path(log_dir, video_path):
    """`<log_dir>/<stem>.log`, the stem on one line and valid on every platform: a line break in a file's name never
    reaches a log's name, and no character in it can name a folder."""
    return Path(log_dir) / f"{one_line(video_path.stem).translate(_UNSAFE_NAME)}.log"


def _run_parallel_batch(files, jobs):
    """Restores `files` `jobs` at a time, each in its own interpreter with its own log beside the application.

    A child is this same entry point on one file, so every file keeps the exact output of a
    solo run (its own work directory, its own scanner profile); the parent only reports. A
    symlinked log folder stops the batch before anything runs.
    """
    try:
        log_dir = _batch_log_dir()
    except OSError as exc:
        print(f"[System] {exc}; nothing was restored.")
        return _report_failures([path.name for path in files], len(files))
    print(f"[System] {len(files)} files, {jobs} at a time; per-file logs under {log_dir}")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda path: _restore_in_child(path, log_dir), files))
    failed = [path.name for path, code in zip(files, results) if code != 0]
    return _report_failures(failed, len(files))


def _restore_in_child(video_path, log_dir):
    """One file through a child interpreter, its stdout in a log opened refusing a symlink; returns its exit code."""
    log_path, name = batch_log_path(log_dir, video_path), one_line(video_path.name)
    started = time.time()
    print(f"[Batch] start  {name}")
    with open_private_log(log_path, truncate=True) as log:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(video_path)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            check=False,
        )
    state = "done  " if completed.returncode == 0 else f"FAILED ({completed.returncode})"
    print(f"[Batch] {state} {name} in {time.time() - started:.0f} s (log: {log_path})")
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
