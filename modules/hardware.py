import os
import platform
import subprocess
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

try:
    import winreg
except ImportError:
    winreg = None


NVIDIA_NAME_TOKENS = ("NVIDIA", "RTX", "GTX", "QUADRO")


def _uses_nvidia_name(name):
    upper_name = name.upper()
    return any(token in upper_name for token in NVIDIA_NAME_TOKENS)


def _set_cpu_only_defaults(settings):
    settings["device_index"] = 0
    settings["is_nvidia"] = False
    settings["is_mps"] = False
    settings["cuda_device"] = None
    settings["cuda_child_device"] = None
    settings["cuda_env"] = {}
    settings["gpu_vram_gb"] = 0
    settings["cpu_only_fallback"] = True


def _select_cuda_device_index():
    for i in range(torch.cuda.device_count()):
        if _uses_nvidia_name(torch.cuda.get_device_name(i)):
            return i, True
    return 0, False


def _apply_vram_profile(settings, vram_gb):
    settings["gpu_vram_gb"] = vram_gb
    if vram_gb >= 24:
        settings["gpu_batch_size"] = 32
        settings["profile_name"] = "EXTREME (RTX 3090/4090/5090)"
        return
    if vram_gb >= 15:
        settings["gpu_batch_size"] = 8
        settings["profile_name"] = "HIGH (RTX 3080/4080/5080)"
        return
    if vram_gb >= 10:
        settings["gpu_batch_size"] = 4
        settings["profile_name"] = "MID (RTX 3070/4070)"
        return

    settings["gpu_batch_size"] = 1
    settings["profile_name"] = "LOW (Entry Config)"


def _populate_cuda_settings(settings, device_id, is_nvidia_device):
    settings["device_index"] = device_id
    settings["is_nvidia"] = settings["is_nvidia"] or is_nvidia_device
    settings["cuda_device"] = f"cuda:{device_id}"
    settings["cuda_child_device"] = "cuda:0" if settings["is_nvidia"] else settings["cuda_device"]
    settings["cuda_env"] = {
        "CUDA_VISIBLE_DEVICES": str(settings["device_index"]),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "ORT_TENSORRT_FP16_ENABLE": "1",
    }
    gpu_props = torch.cuda.get_device_properties(device_id)
    _apply_vram_profile(settings, gpu_props.total_memory / (1024**3))


def _detect_nvidia_smi(settings):
    """Detects NVIDIA GPU via nvidia-smi."""
    try:
        output = subprocess.check_output("nvidia-smi -L", shell=True, stderr=subprocess.DEVNULL).decode()
        if "NVIDIA" in output:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            settings["is_nvidia"] = True
    except Exception:
        pass


def _get_macos_total_memory_gb():
    """Extracts unified physical memory size in GB on macOS."""
    try:
        output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL, timeout=5)
        mem_bytes = int(output.decode("utf-8").strip())
        return mem_bytes / (1024**3)
    except Exception:
        return 8.0


def _detect_mps_backend(settings):
    """Detects Apple Silicon MPS support."""
    if torch is None:
        return False
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return False
    settings["cuda_device"] = "mps"
    settings["cuda_child_device"] = "mps"
    settings["cpu_only_fallback"] = False
    settings["is_mps"] = True
    vram_gb = _get_macos_total_memory_gb()
    _apply_vram_profile(settings, vram_gb)
    settings["profile_name"] = f"APPLE SILICON ({vram_gb:.0f} GB Unified Memory)"
    return True


def _detect_pytorch_device(settings):
    """Detects GPU via PyTorch CUDA or MPS and sets VRAM/Profile."""
    try:
        if torch is not None and torch.cuda.is_available():
            device_id, is_nvidia_device = _select_cuda_device_index()
            _populate_cuda_settings(settings, device_id, is_nvidia_device)
            return

        if _detect_mps_backend(settings):
            return

        _set_cpu_only_defaults(settings)
    except Exception:
        _set_cpu_only_defaults(settings)


_detect_pytorch_cuda = _detect_pytorch_device


def _apply_cuda_env(settings):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings["device_index"])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"
    settings.setdefault("cuda_device", "cuda:0")
    settings["cuda_child_device"] = settings.get("cuda_child_device") or "cuda:0"
    settings["cuda_env"] = settings.get("cuda_env") or {
        "CUDA_VISIBLE_DEVICES": str(settings["device_index"]),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "ORT_TENSORRT_FP16_ENABLE": "1",
    }


def _clear_cuda_env_vars():
    """Removes CUDA masking variables from the current process environment."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_DEVICE_ORDER", None)
    os.environ.pop("ORT_TENSORRT_FP16_ENABLE", None)


def _clear_cuda_env(settings):
    settings["is_nvidia"] = False
    settings["cuda_device"] = None
    settings["cuda_child_device"] = None
    settings["cuda_env"] = {}
    _clear_cuda_env_vars()


def _apply_env_vars(settings):
    """Applies environment variables based on settings."""
    if settings.get("cpu_only_fallback"):
        _clear_cuda_env(settings)
        return

    if settings.get("is_nvidia"):
        _apply_cuda_env(settings)
    elif settings.get("is_mps"):
        _clear_cuda_env_vars()
        settings["cuda_child_device"] = "mps"
        settings["cuda_env"] = {}
    else:
        settings["cuda_child_device"] = settings.get("cuda_device")
        settings["cuda_env"] = {}


def get_optimal_settings():
    """Auto-detect hardware and return optimal settings, prioritizing NVIDIA / MPS."""
    settings = {
        "cpu_threads": os.cpu_count() or 16,
        "gpu_batch_size": 1,
        "cuda_device": "cuda:0",
        "gpu_vram_gb": 0,
        "profile_name": "Low (Entry Config)",
        "is_nvidia": False,
        "is_mps": False,
        "device_index": 0,
        "cpu_only_fallback": False,
    }

    _detect_nvidia_smi(settings)
    _detect_pytorch_device(settings)
    _apply_env_vars(settings)

    return settings


# Auto-configure on import
_hw_settings = get_optimal_settings()

CPU_THREADS = _hw_settings["cpu_threads"]
GPU_BATCH_SIZE = _hw_settings["gpu_batch_size"]
CUDA_DEVICE = _hw_settings["cuda_device"]
CUDA_VISIBLE_DEVICE = _hw_settings.get("cuda_child_device")
CUDA_ENV = _hw_settings.get("cuda_env", {})
GPU_VRAM_GB = _hw_settings["gpu_vram_gb"]
PROFILE_NAME = _hw_settings["profile_name"]
IS_NVIDIA = _hw_settings["is_nvidia"]
IS_MPS = _hw_settings.get("is_mps", False)
DEVICE_INDEX = _hw_settings["device_index"]


def _close_registry_key(key):
    if key is None or winreg is None:
        return
    try:
        winreg.CloseKey(key)
    except Exception:
        pass


def _open_windows_cpu_key():
    if winreg is None:
        return None
    key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    return winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)


def _query_windows_cpu_name(key):
    processor_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    return processor_name.strip()


def _read_windows_cpu_name():
    try:
        key = _open_windows_cpu_key()
    except Exception:
        return None

    if key is None:
        return None

    try:
        return _query_windows_cpu_name(key)
    except Exception:
        return None
    finally:
        _close_registry_key(key)


def _get_linux_cpu_name():
    """Extract CPU model name from Linux /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _get_macos_cpu_name():
    """Extract CPU brand string on macOS."""
    try:
        output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL, timeout=5)
        return output.decode("utf-8").strip()
    except Exception:
        return None


def _get_unix_cpu_name():
    """Extract CPU name on Linux or macOS."""
    if sys.platform.startswith("linux"):
        return _get_linux_cpu_name()
    if sys.platform == "darwin":
        return _get_macos_cpu_name()
    return None


def get_cpu_name():
    """Returns the processor name across Windows, Linux, and macOS."""
    if sys.platform == "win32":
        cpu_name = _read_windows_cpu_name()
        if cpu_name:
            return cpu_name
    return _get_unix_cpu_name() or platform.processor() or "Unknown CPU"


def _get_torch_gpu_name():
    try:
        idx = int(str(CUDA_DEVICE).split(":")[-1])
        return torch.cuda.get_device_name(idx)
    except Exception:
        return None


def _get_nvidia_smi_gpu_name():
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, timeout=5).decode()
        if "NVIDIA" in output:
            return output.split(":")[1].split("(")[0].strip()
    except Exception:
        return None
    return None


def _get_macos_gpu_name():
    if sys.platform == "darwin":
        cpu_name = _get_macos_cpu_name()
        if cpu_name and "Apple" in cpu_name:
            return f"{cpu_name} (Integrated GPU)"
        if IS_MPS:
            return "Apple Silicon GPU (MPS)"
    return None


def _get_cuda_gpu_name():
    if torch is not None and torch.cuda.is_available():
        return _get_torch_gpu_name()
    return None


def get_gpu_name():
    """Returns GPU identifier across CUDA, MPS, and host environments."""
    return _get_cuda_gpu_name() or _get_nvidia_smi_gpu_name() or _get_macos_gpu_name() or "Generic / Not Detected"


def get_nvidia_paths():
    """Returns a list of paths containing CUDNN/CUBLAS DLLs or shared libraries."""

    def _append_existing(path_list, candidate):
        if candidate and os.path.exists(candidate):
            path_list.append(candidate)

    def _module_root(module):
        if hasattr(module, "__path__") and module.__path__:
            return module.__path__[0]
        return os.path.dirname(module.__file__)

    nvidia_paths = []

    try:
        import torch

        _append_existing(nvidia_paths, str(Path(torch.__file__).parent / "lib"))
    except ImportError:
        pass

    try:
        import nvidia.cublas
        import nvidia.cudnn

        for lib in [nvidia.cudnn, nvidia.cublas]:
            lib_root = _module_root(lib)
            _append_existing(nvidia_paths, os.path.join(lib_root, "bin"))
            _append_existing(nvidia_paths, os.path.join(lib_root, "lib"))
    except ImportError:
        pass

    return nvidia_paths


def _update_dynamic_linker_env(path):
    """Adds a runtime path to child-process linker environments.

    The environment is inherited by subprocesses such as FFmpeg and FFprobe;
    it does not change the current Python interpreter's loader search path.
    """
    env_vars = ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"]
    if sys.platform == "win32":
        # Windows resolves child-process DLLs via PATH rather than *_LIBRARY_PATH.
        env_vars.append("PATH")
    for env_var in env_vars:
        current_paths = os.environ.get(env_var, "").split(os.pathsep)
        if path not in current_paths:
            os.environ[env_var] = path + os.pathsep + os.environ[env_var] if os.environ.get(env_var) else path


def prepare_runtime_library_paths():
    """Injects library paths into the dynamic linker environment.

    Updates the ``LD_LIBRARY_PATH`` and ``DYLD_LIBRARY_PATH`` variables (and
    ``PATH`` on Windows) so the in-project CUDNN/CUBLAS shared libraries resolve
    at runtime.
    """
    for path in get_nvidia_paths():
        _update_dynamic_linker_env(path)
