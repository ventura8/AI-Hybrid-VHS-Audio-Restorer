import atexit
import collections
import datetime
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Set

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import torch
except ImportError:
    torch = None

# Import config constants
from .config import DEBUG_LOGGING, LOG_FILE
from .hardware import get_nvidia_paths

# === AUTO-CONFIGURE PATH ===
project_dir = Path(__file__).parent.parent.resolve()  # modules/..


def _get_scripts_dirs(base_dir):
    candidates = [
        base_dir / ".venv" / "Scripts",
        base_dir / "venv" / "Scripts",
    ]
    return [p for p in candidates if p.exists()]


scripts_dirs = _get_scripts_dirs(project_dir)
primary_scripts_dir = scripts_dirs[0] if scripts_dirs else (project_dir / ".venv" / "Scripts")

# 1. Base Binary Paths
FFMPEG_BIN = "ffmpeg"
for scripts_dir in scripts_dirs:
    ffmpeg_candidate = scripts_dir / "ffmpeg.exe"
    if ffmpeg_candidate.exists():
        FFMPEG_BIN = str(ffmpeg_candidate)
        break

# 2. NVIDIA / CUDA Library Injection (Critical for Hybrid GPUs)
extra_paths = [str(p) for p in scripts_dirs]
extra_paths.extend(get_nvidia_paths())

current_path = os.environ.get("PATH", "")
path_list = current_path.split(os.pathsep)

added_any = False
for p in extra_paths:
    if p and p not in path_list:
        path_list.insert(0, p)
        added_any = True

if added_any:
    os.environ["PATH"] = os.pathsep.join(path_list)

_venv_scripts_missing = not scripts_dirs


def _resolve_log_level(is_error, level):
    if is_error:
        return "ERROR"
    return level.upper()


def _should_print_log(console, effective_level):
    if effective_level == "DEBUG" and not DEBUG_LOGGING:
        return False
    return console


def _print_log_message(message):
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()
    print(f"   {message}" if not message.startswith("   ") else message)


def _append_log_file(effective_level, clean_msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{effective_level:5}] {clean_msg}\n")


def log_msg(message, is_error=False, console=True, level="INFO"):
    """
    Logs messages to console and log file.

    Args:
        message: The message to log.
        is_error: If True, marks as ERROR level (overrides 'level').
        console: If True, prints to console (unless level is DEBUG).
        level: Log level - 'INFO', 'DEBUG', or 'ERROR'. DEBUG never prints to console.
    """
    effective_level = _resolve_log_level(is_error, level)
    should_print = _should_print_log(console, effective_level)

    if should_print:
        _print_log_message(message)

    clean_msg = message.strip()

    try:
        _append_log_file(effective_level, clean_msg)
    except Exception:
        pass


if _venv_scripts_missing:
    log_msg(f"Venv Scripts not found at: {primary_scripts_dir}", level="DEBUG")


# === SUBPROCESS MANAGEMENT ===
_active_processes: Set[subprocess.Popen] = set()


def _terminate_processes(processes, terminate_fn):
    for process in processes:
        try:
            if process.poll() is None:
                terminate_fn(process)
        except Exception:
            pass


def cleanup_subprocesses():
    """Terminates all registered active subprocesses."""
    if not _active_processes:
        return

    log_msg(f"\n[System] Cleaning up {_len_active()} processes...", level="DEBUG")
    active_processes = list(_active_processes)
    _terminate_processes(active_processes, lambda process: process.terminate())

    time.sleep(0.5)

    _terminate_processes(active_processes, lambda process: process.kill())
    _active_processes.clear()


def _len_active():
    return len(_active_processes)


def signal_handler(sig, frame):
    """Handles termination signals."""
    log_msg("\n[System] Termination signal received. Stopping...", is_error=True)
    cleanup_subprocesses()
    sys.exit(1)


# Register handlers
atexit.register(cleanup_subprocesses)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if sys.platform == "Windows":
    sig_break = getattr(signal, "SIGBREAK", None)
    if sig_break is not None:
        signal.signal(sig_break, signal_handler)


def is_valid_audio(file_path):
    """
    Checks if a file exists, has content, and has a valid audio header.
    Returns False if corrupted, empty, or extremely short (<0.1s).
    """
    path = Path(file_path)
    if not _is_audio_candidate(path):
        return False

    if sf is None:
        return False

    duration = _read_audio_duration(path)
    return duration is not None and duration > 0.1


def _is_audio_candidate(path):
    try:
        if not path.exists():
            return False
        return path.stat().st_size >= 1024
    except OSError:
        return False


def _read_audio_duration(path):
    try:
        with sf.SoundFile(str(path)) as f:
            if f.frames <= 0 or f.samplerate <= 0:
                return None
            return f.frames / f.samplerate
    except Exception:
        return None


def is_valid_video(file_path):
    """
    Checks if a video file exists and has reasonable size.
    Does not do full header check to avoid overhead, but filters empty files.
    """
    path = Path(file_path)
    if not path.exists():
        return False
    # 10KB minimum to be considered a successful video write
    # (Reduced from 1MB to support very short or low-bitrate SD clips)
    if path.stat().st_size < 10240:
        return False
    return True


def format_time(seconds):
    """Formats seconds as HH:MM:SS.mm"""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def parse_ffmpeg_time(line):
    """Extracts time=HH:MM:SS.mm from FFmpeg output."""
    match = re.search(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})", line)
    if match:
        h, m, s, ms = map(int, match.groups())
        return h * 3600 + m * 60 + s + ms / 100.0
    return None


# Import drawing from UI - WAIT, UI depends on Utils for time/etc.
# Circular dependency risk if UI needs format_time and Utils needs draw_progress_bar.
# run_command_with_progress calls draw_progress_bar.
# So utils needs ui?
# Let's put UI components (draw_progress_bar) in utils OR
# move run_command_with_progress to ui?
# run_command_with_progress is logic, but has UI.
# Maybe I should put draw_progress_bar in utils as well?
# Or make a separate `ui_base`?
# I will put draw_progress_bar and format_time ALL in utils.py for now to avoid circular deps.
# The plan said `modules/ui.py` has `draw_progress_bar`.
# If I stick to the plan, `utils` needs to import `ui`. `ui` needs `utils` (for format_time?).
# Logic: `utils` is low level. `ui` is high level.
# `run_command_with_progress` is high level util?
# I'll put `draw_progress_bar` and `run_command_with_progress` ALL in `utils.py` for simplicity/robustness against circular deps.
# And `modules/ui.py` will just have the CLI/Banner stuff (`_show_banner`, `_get_input_files`).
# That seems safer.


# Global Thread Locks and State for UI
_print_lock = threading.Lock()
_last_bar_time = 0
_last_bar_pc = -1.0


def _time_progress_part(media_sec, elapsed_sec, total_duration):
    if media_sec is not None:
        current_str = format_time(media_sec)
        if total_duration:
            return f"{current_str} / {format_time(total_duration)}"
        return current_str
    if elapsed_sec is not None:
        return format_time(elapsed_sec)
    return None


def _eta_progress_part(percent, elapsed_sec):
    if percent <= 0 or percent >= 100 or elapsed_sec is None:
        return None
    total_est = (elapsed_sec / percent) * 100
    eta_sec = total_est - elapsed_sec
    return f"ETA {format_time(eta_sec).split(',')[0]}"


def _speed_progress_part(media_sec, elapsed_sec):
    if media_sec is None or elapsed_sec is None or elapsed_sec <= 0.1:
        return None
    speed = media_sec / elapsed_sec
    return f"{speed:5.2f}x"


def _build_progress_info(percent, elapsed_sec, media_sec, total_duration=None):
    """
    Builds the info string: 74.3% | 00:20:43,400 / 00:27:52,777 | ETA 00:00:39 | 10.76x
    """
    parts = [f"{percent:5.1f}%"]
    for extra_part in [
        _time_progress_part(media_sec, elapsed_sec, total_duration),
        _eta_progress_part(percent, elapsed_sec),
        _speed_progress_part(media_sec, elapsed_sec),
    ]:
        if extra_part:
            parts.append(extra_part)

    return " | ".join(parts)


def _get_terminal_columns(default=79):
    try:
        return shutil.get_terminal_size((80, 20)).columns - 1
    except Exception:
        return default


def _truncate_label(clean_label, label_len, excess):
    if label_len <= 20:
        return clean_label, label_len, 0
    label_can_give = max(0, label_len - 15)
    shrink_label = min(excess, label_can_give)
    label_len -= shrink_label
    clean_label = clean_label[:label_len] + "..."
    return clean_label, label_len, shrink_label


def _shrink_bar(width, excess):
    can_shrink = max(0, width - 5)
    return min(excess, can_shrink)


def _hard_truncate_label(clean_label, excess):
    label_available = max(0, len(clean_label) - excess - 3)
    if label_available < 5:
        return ""
    return clean_label[:label_available] + "..."


def _truncate_bar_label(clean_label, label_len, total_len, columns):
    if total_len <= columns:
        return clean_label, label_len, total_len
    excess = total_len - columns
    clean_label, label_len, shrink_label = _truncate_label(clean_label, label_len, excess)
    if shrink_label:
        total_len -= shrink_label - 3
    return clean_label, label_len, total_len


def _shrink_bar_width(width, total_len, columns):
    if total_len <= columns:
        return width, total_len
    shrink_amt = _shrink_bar(width, total_len - columns)
    return width - shrink_amt, total_len - shrink_amt


def _apply_hard_label_truncation(clean_label, total_len, columns):
    if total_len <= columns:
        return clean_label
    return _hard_truncate_label(clean_label, total_len - columns)


def _adjust_bar_layout(width, info_str, label, columns):
    """Adjusts bar width and truncates label to fit terminal."""
    overhead = 8  # Indent(4) + [] + Space + Safety(1)
    clean_label = re.sub(r"[\r\n]", "", label).strip()

    label_len = len(clean_label) if clean_label else 0
    total_len = overhead + width + len(info_str) + label_len + 3  # +3 for '   ' padding

    clean_label, label_len, total_len = _truncate_bar_label(clean_label, label_len, total_len, columns)
    width, total_len = _shrink_bar_width(width, total_len, columns)
    clean_label = _apply_hard_label_truncation(clean_label, total_len, columns)

    return width, info_str, clean_label


def _draw_bar_line(width, filled_length, info_str, label=""):
    """Draws the final bar line with explicit clearing and no-wrap safety."""
    bar = "█" * filled_length + "░" * (width - filled_length)

    # Standardized 3-space padding
    if label:
        line_content = f"   {label}[{bar}] {info_str}"
    else:
        line_content = f"   [{bar}] {info_str}"

    # Terminal width safety
    cols = _get_terminal_columns()
    if len(line_content) >= cols:
        line_content = line_content[: cols - 1]

    with _print_lock:
        # \r to start, \033[K to clear, then content. NO trailing newline.
        sys.stdout.write(f"\r\033[K{line_content}")
        sys.stdout.flush()


def draw_progress_bar(percent, label="", width=20, elapsed_sec=None, media_sec=None, total_duration=None):
    """
    Draws a modern visual progress bar with rate-limiting.
    """
    global _last_bar_time, _last_bar_pc

    percent = max(0.0, min(100.0, float(percent)))
    now = time.time()

    # Rate limit: Max 20 FPS, but always allow 0%, 100%, or major jumps
    if now - _last_bar_time < 0.05 and abs(percent - _last_bar_pc) < 1.0 and 0 < percent < 100:
        return

    _last_bar_time = now
    _last_bar_pc = percent

    columns = _get_terminal_columns()

    # Build Info String
    info_str = _build_progress_info(percent, elapsed_sec, media_sec, total_duration)

    # Clean label and ensure it has spacing if present
    clean_label = re.sub(r"[\r\n]", "", label).strip()

    # Layout Adjustment
    width, info_str, final_label = _adjust_bar_layout(width, info_str, clean_label, columns)

    if width < 2:
        width = 2

    filled_length = int(width * percent // 100)

    _draw_bar_line(width, filled_length, info_str, final_label)


def _parse_tqdm_progress(line, tqdm_re):
    """Parses TQDM progress lines, returning only the percentage for our own bar."""
    match = tqdm_re.search(line)
    if not match:
        return None, None

    return float(match.group(1)), ""


def _update_ffmpeg_progress(line, duration, description, start_time):
    current_time = parse_ffmpeg_time(line)
    if current_time is None or not duration:
        return False
    percent = (current_time / duration) * 100
    elapsed = time.time() - start_time
    draw_progress_bar(percent, description, elapsed_sec=elapsed, media_sec=current_time, total_duration=duration)
    return True


def _should_skip_output_line(line):
    return "muxing overhead" in line.lower()


def _should_emit_tqdm_progress(process, percent):
    if not hasattr(process, "_last_pc"):
        return True
    return abs(percent - process._last_pc) >= 0.1 or percent in [0, 100]


def _update_tqdm_progress(process, percent, tqdm_info, description, duration, start_time):
    elapsed = time.time() - start_time
    media_sec = (percent / 100) * duration if duration else None
    if not _should_emit_tqdm_progress(process, percent):
        return
    process._last_pc = percent
    label = f"{description} {tqdm_info}" if tqdm_info else description
    draw_progress_bar(percent, label, elapsed_sec=elapsed, media_sec=media_sec)


def _handle_progress_output_line(process, line, start_time, duration, description, tqdm_re):
    if _update_ffmpeg_progress(line, duration, description, start_time):
        return
    if _should_skip_output_line(line):
        return

    percent, tqdm_info = _parse_tqdm_progress(line, tqdm_re)
    if percent is not None:
        _update_tqdm_progress(process, percent, tqdm_info, description, duration, start_time)


def _monitor_process_output(process, start_time, duration, description, tqdm_re):
    """Monitors process stdout for progress updates."""
    output_buffer = collections.deque(maxlen=20)

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break

        if line:
            output_buffer.append(line)
            _handle_progress_output_line(process, line, start_time, duration, description, tqdm_re)
    return output_buffer


def _cleanup_after_monitor_error(process):
    """Best-effort child cleanup used after monitor failures."""
    if process.poll() is not None:
        return True

    try:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    except Exception:
        return False

    return process.poll() is not None


def _prepare_process_environment(env):
    prepared_env = os.environ.copy()
    if env is not None:
        prepared_env.update(env)
    prepared_env["PYTHONIOENCODING"] = "utf-8"
    return prepared_env


def _drain_process_output(process, start_time, duration, description, tqdm_re):
    try:
        return _monitor_process_output(process, start_time, duration, description, tqdm_re)
    except Exception as monitor_exc:
        if _cleanup_after_monitor_error(process):
            _active_processes.discard(process)
        raise monitor_exc


def _handle_command_failure(cmd, output_buffer):
    sys.stdout.write("\n")
    log_msg(f"\n[Error] Command {cmd[0]} failed. Last output:", is_error=True)
    for err_line in output_buffer:
        log_msg(f"  > {err_line.strip()}", is_error=True)
    raise subprocess.CalledProcessError(1, cmd)


def _finish_command(process, cmd, description, start_time, duration, output_buffer):
    process.wait()
    _active_processes.discard(process)

    if process.returncode == 0:
        elapsed = time.time() - start_time
        draw_progress_bar(100.0, description, elapsed_sec=elapsed, media_sec=duration)
        sys.stdout.write("\n")
        return

    sys.stdout.write("\n")
    log_msg(f"\n[Error] Command {cmd[0]} failed. Last output:", is_error=True)
    for err_line in output_buffer:
        log_msg(f"  > {err_line.strip()}", is_error=True)

    raise subprocess.CalledProcessError(process.returncode, cmd)


def run_command_with_progress(cmd, env=None, description="Running...", total_duration=None):
    """
    Runs a subprocess and parses progress from its output.
    Supports FFmpeg and TQDM-style output.
    """
    start_time = time.time()
    sys.stdout.write("\n")  # Ensure we start on a new line
    sys.stdout.flush()
    env = _prepare_process_environment(env)

    process = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, encoding="utf-8", errors="replace"
    )
    _active_processes.add(process)

    duration = max(0.1, total_duration) if total_duration else None
    tqdm_re = re.compile(r"(\d+)%\s*[|:]")
    output_buffer = _drain_process_output(process, start_time, duration, description, tqdm_re)
    _finish_command(process, cmd, description, start_time, duration, output_buffer)


def _clear_cuda_retry_state():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()


def attempt_run_with_retry(command_builder_func, initial_batch_size, description="Running...", total_duration=None):
    """
    Retries a command with reduced GPU batch sizes on OOM.
    command_builder_func: Accepts 'batch_size' (int), returns [cmd, args...].
    """
    current_batch_size = initial_batch_size

    while True:
        try:
            cmd = command_builder_func(current_batch_size)
            # Use run_command_with_progress to get %, ETA, and Speed
            run_command_with_progress(cmd, description=f"{description} (BS:{current_batch_size})", total_duration=total_duration)
            return True  # Success

        except subprocess.CalledProcessError as e:
            if current_batch_size > 1:
                log_msg("    [Warning] GPU failed. Retrying with reduced batch...", is_error=True)
                _clear_cuda_retry_state()
                current_batch_size = max(1, current_batch_size // 2)
            else:
                raise e


def _remove_if_exists(path):
    if path.exists():
        path.unlink()


def _replace_with_valid_audio(temp_path, final_path):
    if not is_valid_audio(temp_path):
        log_msg(f"[Error] Atomic Save Failed: {temp_path} is invalid.", is_error=True)
        _remove_if_exists(temp_path)
        return False
    os.replace(str(temp_path), str(final_path))
    return True


def attempt_cpu_run_with_retry(command_builder_func, initial_threads, description="Running...", total_duration=None):
    """
    Retries a CPU-bound command with reduced threads on RAM OOM.
    command_builder_func: Accepts 'threads' (int), returns [cmd, args...].
    """
    current_threads = initial_threads

    while True:
        try:
            cmd = command_builder_func(current_threads)
            # If we don't have duration, use standard print
            if total_duration is None:
                print(f"      {description} (Threads: {current_threads})")

            run_command_with_progress(cmd, description=description, total_duration=total_duration)

            return True  # Success

        except subprocess.CalledProcessError as e:
            if current_threads > 1:
                log_msg("    [Warning] CPU failed. Retrying with fewer threads...", is_error=True)
                current_threads = max(1, current_threads // 2)

                # Cleanup RAM
                import gc

                gc.collect()
            else:
                raise e


def _save_audio_atomic(file_path, data, sample_rate, subtype="FLOAT"):
    """
    Saves audio to a temporary file, then renames it to the final path.
    This prevents corrupted partial files if the process is interrupted.
    """
    path = Path(file_path)
    # Use .tmp.wav to ensure soundfile and is_valid_audio recognize the format
    temp_path = path.with_suffix(f".tmp{path.suffix}")

    if sf is None:
        log_msg(f"[Error] Failed to save audio {path}: soundfile is not installed.", is_error=True)
        return False

    try:
        sf.write(str(temp_path), data, sample_rate, subtype=subtype)
        return _replace_with_valid_audio(temp_path, path)

    except Exception as e:
        log_msg(f"[Error] Failed to save audio {path}: {e}", is_error=True)
        _remove_if_exists(temp_path)
        return False


# Need to import torch at end or lazily?
# attempt_run_with_retry uses torch.cuda.is_available.
# I need 'import torch' at top.


def check_dependencies():
    missing = []

    def _record_missing(name, command):
        try:
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=10)
            if result.returncode != 0:
                missing.append(name)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError, subprocess.SubprocessError):
            missing.append(name)

    _record_missing("FFmpeg", [FFMPEG_BIN, "-version"])
    _record_missing("Audio-Separator", ["audio-separator", "--help"])
    _record_missing("Resemble-Enhance", [sys.executable, "-c", "import resemble_enhance"])
    _record_missing("OmegaConf", [sys.executable, "-c", "from omegaconf import OmegaConf"])

    if missing:
        log_msg(f"CRITICAL: Missing: {', '.join(missing)}", is_error=True)
        log_msg(f"Search Path: {os.environ['PATH']}", is_error=True)
        return False
    return True
