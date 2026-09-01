"""Produce a portable, JSON hardware capability report for validation runs."""

import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
hardware = importlib.import_module("modules.hardware")


def _nvidia_smi():
    """Return NVIDIA's concise GPU query output when available."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return []
    command = [binary, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _onnx_providers():
    """Return installed ONNX Runtime providers without making it required."""
    try:
        import onnxruntime

        return onnxruntime.get_available_providers()
    except ImportError:
        return []


def build_report():
    """Collect only local, non-invasive accelerator readiness information."""
    settings = hardware.get_optimal_settings()
    return {
        "platform": platform.platform(),
        "cpu": hardware.get_cpu_name(),
        "nvidia_smi": _nvidia_smi(),
        "pytorch_device": settings.get("cuda_device") or "cpu",
        "pytorch_cuda_ready": not settings["cpu_only_fallback"],
        "vram_gb": round(settings["gpu_vram_gb"], 2),
        "recommended_batch_size": settings["gpu_batch_size"],
        "profile": settings["profile_name"],
        "onnx_execution_providers": _onnx_providers(),
        "directml_note": "Install onnxruntime-directml to validate DirectML providers on Windows.",
    }


def main():
    """Print the report as stable JSON for humans and CI artifacts."""
    print(json.dumps(build_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
