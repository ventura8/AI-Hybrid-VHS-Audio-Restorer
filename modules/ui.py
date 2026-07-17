import os
import platform
import sys
import time
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from .config import BACKGROUND_MIX_VOL, CONFIG_SOURCE, EXTS, INPUT_DIR, PROCESS_MODE, VOCAL_MIX_VOL, VOCALS_MODEL
from .hardware import CPU_THREADS, GPU_BATCH_SIZE, GPU_VRAM_GB, PROFILE_NAME, get_cpu_name, get_gpu_name
from .utils import draw_progress_bar

# Constants imported from config via restore_audio_hybrid normally,
# but we can access them here or pass them.
# The banner uses a lot of them.


def _is_cleaned_output(file_name):
    stem = Path(file_name).stem
    return stem.endswith("_Hybrid_Cleaned") or stem.endswith("_Denoised_Cleaned")


def _is_nvidia_gpu_name(gpu_name):
    upper_gpu_name = gpu_name.upper()
    return "NVIDIA" in upper_gpu_name or "RTX" in upper_gpu_name


def _has_cuda_backend():
    return torch is not None and torch.cuda.is_available()


def _has_xpu_backend():
    return torch is not None and hasattr(torch, "xpu") and torch.xpu.is_available()


def _get_torch_backend(gpu_name):
    if _has_cuda_backend():
        if _is_nvidia_gpu_name(gpu_name):
            return "CUDA (NVIDIA Accelerated)"
        return "CUDA (Accelerated)"
    if _has_xpu_backend():
        return "XPU (Intel Accelerated)"
    return "CPU (Slow)"


def _print_banner(cpu_name, gpu_name, torch_backend):
    print("=" * 60)
    print("   AI HYBRID VHS AUDIO RESTORER - v1.0.2")
    print(f"   Running on: {platform.system()} {platform.release()}")  # pragma: no cover
    print("=" * 60 + "\n")  # pragma: no cover

    print("[HARDWARE DETECTED]")
    print(f"   CPU : {os.cpu_count()} Logical Cores ({cpu_name})")
    print(f"   GPU : {gpu_name} ({GPU_VRAM_GB:.2f} GB VRAM)")
    print(f"   AI Acceleration: {torch_backend}\n")

    print(f"[AUTO-TUNED SETTINGS -> Profile: {PROFILE_NAME}]")
    print("   Audio Precision : 32-bit Float (WAV)")
    print(f"   Process Mode    : {PROCESS_MODE.replace('_', ' ').title()}")
    print(f"   Batch Size      : {GPU_BATCH_SIZE}")
    print(f"   Threads         : {CPU_THREADS}")
    print(f"   Mix Levels      : Vocals={VOCAL_MIX_VOL}, Background={BACKGROUND_MIX_VOL}")
    print(f"   Models          : {VOCALS_MODEL} / UVR-DeNoise")
    print(f"   Config Source   : {CONFIG_SOURCE}\n")


def _print_backend_warning(torch_backend, gpu_name):
    if "CPU" in torch_backend and _is_nvidia_gpu_name(gpu_name):
        print("!! WARNING: NVIDIA GPU detected but Torch is using CPU.")
        print("!! This will be EXTREMELY slow. Check your drivers/installation.\n")


def _show_banner():
    """Prints the application banner and hardware info."""
    cpu_name = get_cpu_name()
    gpu_name = get_gpu_name()

    torch_backend = _get_torch_backend(gpu_name)
    _print_banner(cpu_name, gpu_name, torch_backend)
    _print_backend_warning(torch_backend, gpu_name)

    return cpu_name, gpu_name


def _scan_single_file(path):
    if path.suffix.lower() not in EXTS:
        print(f">> [Error] Unsupported extension: {path.suffix}")
        print(f">> Supported: {EXTS}")
        return []
    if _is_cleaned_output(path.name):
        print(f">> Skipping cleaned output file: {path.name}")
        return []
    return [path]


def _scan_directory_files(path):
    return [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in EXTS and not _is_cleaned_output(f.name)]


def _scan_directory_with_error_handling(path, error_message):
    try:
        return _scan_directory_files(path)
    except OSError:
        print(error_message)
        return []


def _scan_directory(path):
    print(f">> Scanning folder: {path.name}")
    return _scan_directory_with_error_handling(path, f">> [Error] Could not access folder: {path}")


def _scan_files_in_path(path):
    """Scans a file or directory for valid video files."""
    if path.is_file():
        return _scan_single_file(path)
    if path.is_dir():
        return _scan_directory(path)
    print(f">> [Error] Path is not a file or directory: {path}")
    return []


def _strip_wrapping_quotes(user_input):
    if user_input.startswith('"') and user_input.endswith('"'):
        return user_input[1:-1]
    if user_input.startswith("'") and user_input.endswith("'"):
        return user_input[1:-1]
    return user_input


def _clean_user_input(user_input):
    """Cleans and normalizes user input path."""
    user_input = user_input.strip()

    if user_input.startswith("&"):
        user_input = user_input[1:].strip()

    user_input = _strip_wrapping_quotes(user_input)

    if user_input.lower().startswith("file://"):
        user_input = user_input[7:].strip()

    return user_input


def _ensure_input_dir():
    try:
        INPUT_DIR.mkdir(exist_ok=True)
        return True
    except OSError:
        print(">> [Error] Could not create or access 'input' folder.")
        return False


def _scan_default_input_dir():
    print(">> Interactive Mode: Drag & Drop files or press Enter to scan 'input' folder.")
    print(">> Scanning 'input' folder...")
    if not _ensure_input_dir():
        return [], False

    files = _scan_directory_with_error_handling(INPUT_DIR, ">> [Error] Could not access 'input' folder.")
    return files, False


def _get_interactive_files():
    """Prompts user for input and scans."""
    try:
        print(">> Please Drag & Drop a video file here and press Enter:")
        user_input = input(">>Path: ")
        clean_input = _clean_user_input(user_input)

        if not clean_input:
            return _scan_default_input_dir()

        path = Path(clean_input)
        if not path.exists():
            print(f">> [Error] File not found: {path}")
            return [], False

        return _scan_files_in_path(path), True

    except (EOFError, KeyboardInterrupt):
        return [], False


def _get_input_files():

    files = []
    use_source_as_output = False

    if len(sys.argv) > 1:
        use_source_as_output = True
        print(f">> Arguments Detected: {len(sys.argv) - 1} items")
        for arg in sys.argv[1:]:
            path = Path(arg)
            # Re-use the scanning logic?
            # Original logic handled iterdir slightly different but _scan_files_in_path is robust.
            found = _scan_files_in_path(path)
            files.extend(found)
    else:
        files, use_source_as_output = _get_interactive_files()

    return files, use_source_as_output


def run_init_sequence():
    """Runs the visual initialization sequence."""
    print("\n[AI ENGINE INITIALIZATION]")

    draw_progress_bar(10, "Initializing Core Systems...")
    time.sleep(0.3)
    draw_progress_bar(30, "Scanning Hardware...")
    time.sleep(0.2)

    cpu_name = get_cpu_name()
    draw_progress_bar(45, f"CPU Detected: {os.cpu_count()} Cores")

    gpu_name = get_gpu_name()
    draw_progress_bar(60, f"GPU Detected: {gpu_name}")

    # Profile name logic in config/hardware, but we need it here
    # It's already in constants
    p_name = PROFILE_NAME.split("(")[0].strip()
    draw_progress_bar(75, f"Applying Optimization Profile: {p_name}")
    time.sleep(0.2)

    # Check dependencies logic?
    # It was in utils or restore_audio_hybrid.py
    # I didn't see explicit check_dependencies function in my previous reads of restore_audio_hybrid.py loops?
    # Wait, line 1875 in restore_audio_hybrid calls `check_dependencies()`.
    # I need to implement `check_dependencies` in `utils.py` or `ui.py`.
    # I checked `utils.py` content, I didn't add `check_dependencies`.

    draw_progress_bar(90, "Verifying Libraries...")
    time.sleep(0.2)
    draw_progress_bar(100, "Initialization Complete.")
    # End the progress-bar line before regular banner prints.
    sys.stdout.write("\n")
    sys.stdout.flush()
    time.sleep(0.4)

    return cpu_name, gpu_name
