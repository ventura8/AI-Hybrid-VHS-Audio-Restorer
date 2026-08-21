import importlib
import os
import subprocess
import sys
import types
from unittest.mock import MagicMock, mock_open, patch

import pytest

import modules.hardware


@pytest.fixture(autouse=True)
def mock_hardware_torch(monkeypatch):
    mock_torch = MagicMock()
    mock_torch.__file__ = "C:/mock/torch/__init__.py"
    mock_torch.cuda = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setattr(modules.hardware, "torch", mock_torch)
    return mock_torch


# ---------------------------------------------------------
# HW & Config
# ---------------------------------------------------------


def test_hw_optimal_settings():
    """Test hardware optimal settings detection."""
    # Test with high VRAM GPU
    with patch.object(modules.hardware.torch.cuda, "is_available", return_value=True):
        mock_props = MagicMock()
        mock_props.total_memory = 32 * 1024**3
        with patch.object(modules.hardware.torch.cuda, "get_device_properties", return_value=mock_props):
            settings = modules.hardware.get_optimal_settings()
            assert "EXTREME" in settings["profile_name"]

    # Test with medium VRAM GPU
    with patch.object(modules.hardware.torch.cuda, "is_available", return_value=True):
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024**3
        with patch.object(modules.hardware.torch.cuda, "get_device_properties", return_value=mock_props):
            settings = modules.hardware.get_optimal_settings()
            assert "LOW" in settings["profile_name"]


def test_get_nvidia_paths_coverage():
    """Test NVIDIA paths detection - verifies function returns a list without error."""
    # The function may or may not find paths depending on installed packages
    # We just verify it returns a list and handles imports gracefully
    paths = modules.hardware.get_nvidia_paths()
    assert isinstance(paths, list)


@patch("modules.hardware.subprocess.check_output")
def test_get_gpu_name_robust(mock_out):
    """Test GPU name detection."""
    mock_out.return_value = b"GPU 0: NVIDIA RTX 5000 (UUID: abc-123)"
    with patch.object(modules.hardware.torch.cuda, "is_available", return_value=False):
        result = modules.hardware.get_gpu_name()
        assert "5000" in result or "NVIDIA" in result

    # Test failure case
    mock_out.side_effect = Exception("nvidia-smi not found")
    # Ensure torch and MPS don't interfere
    with (
        patch("sys.platform", "linux"),
        patch.object(modules.hardware.torch.cuda, "is_available", return_value=False),
        patch.object(modules.hardware, "IS_MPS", False),
    ):
        result = modules.hardware.get_gpu_name()
        assert ("Not Detected" in result) or ("Generic" in result)


def test_get_cpu_name_windows():
    """Test CPU name detection on Windows."""
    # Mock winreg module structure
    mock_winreg = MagicMock()
    mock_key = MagicMock()
    mock_winreg.OpenKey.return_value = mock_key
    # QueryValueEx returns (value, type)
    mock_winreg.QueryValueEx.return_value = ("Intel Core i9 Windows", 1)

    with patch("sys.platform", "win32"), patch.object(modules.hardware, "winreg", mock_winreg):
        result = modules.hardware.get_cpu_name()
        assert "Intel" in result
        mock_winreg.CloseKey.assert_called_once_with(mock_key)


def test_get_cpu_name_linux():
    """Test CPU name detection on Linux."""
    with patch("sys.platform", "linux"), patch("modules.hardware._get_unix_cpu_name", return_value="AMD Ryzen Linux"):
        result = modules.hardware.get_cpu_name()
        assert "AMD" in result


def test_get_cpu_name_fallback():
    """Test CPU name fallback to platform.processor."""
    with (
        patch("sys.platform", "linux"),
        patch("modules.hardware._get_unix_cpu_name", return_value=None),
        patch("modules.hardware.platform.processor", return_value="Fallback CPU"),
    ):
        result = modules.hardware.get_cpu_name()
        assert result == "Fallback CPU"


def test_optimal_settings_all_profiles():
    """Test all GPU profile detection branches."""
    profiles = [
        (32, "EXTREME"),  # 32GB -> EXTREME
        (16, "HIGH"),  # 16GB -> HIGH
        (15, "HIGH"),  # 15GB threshold
        (12, "MID"),  # 12GB -> MID
        (10, "MID"),  # 10GB threshold
        (6, "LOW"),  # 6GB -> LOW
    ]

    for vram_gb, expected_profile in profiles:
        with patch.object(modules.hardware.torch.cuda, "is_available", return_value=True):
            mock_props = MagicMock()
            mock_props.total_memory = vram_gb * 1024**3
            with patch.object(modules.hardware.torch.cuda, "get_device_properties", return_value=mock_props):
                settings = modules.hardware.get_optimal_settings()
                assert (
                    expected_profile in settings["profile_name"]
                ), f"Expected {expected_profile} for {vram_gb}GB, got {settings['profile_name']}"


def test_optimal_settings_no_cuda():
    """Test optimal settings when CUDA not available."""
    with patch.object(modules.hardware.torch.cuda, "is_available", return_value=False):
        settings = modules.hardware.get_optimal_settings()
        assert settings["gpu_vram_gb"] == 0
        assert "LOW" in settings["profile_name"] or "Entry" in settings["profile_name"]


def test_optimal_settings_cuda_exception():
    """Test optimal settings when CUDA throws exception."""
    with patch.object(modules.hardware.torch.cuda, "is_available", return_value=True):
        with patch.object(modules.hardware.torch.cuda, "get_device_properties", side_effect=Exception("CUDA error")):
            # Should handle exception gracefully and return defaults
            settings = modules.hardware.get_optimal_settings()
            assert settings is not None


def test_module_path_logic(mock_hardware_torch):
    """Test module-level FFMPEG path resolution logic."""
    try:
        # Mocking venv_scripts.exists() to True
        with patch.dict(sys.modules, {"torch": mock_hardware_torch}):
            with patch("modules.hardware.Path.exists", return_value=True):
                importlib.reload(modules.hardware)

        # Mocking to False
        with patch.dict(sys.modules, {"torch": mock_hardware_torch}):
            with patch("modules.hardware.Path.exists", return_value=False):
                importlib.reload(modules.hardware)
    finally:
        # Restore normal module-level configuration for subsequent tests.
        with patch.dict(sys.modules, {"torch": mock_hardware_torch}):
            importlib.reload(modules.hardware)


def test_nvidia_paths_branches():
    """Test get_nvidia_paths branches."""
    # Test torch branch
    m_torch = MagicMock()
    m_torch.__file__ = "/p/torch/__init__.py"
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            return m_torch
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=fake_import):
        with patch("modules.hardware.Path.exists", return_value=True):
            paths = modules.hardware.get_nvidia_paths()
            assert isinstance(paths, list)

    # Test nvidia.* import failure
    with patch("builtins.__import__", side_effect=ImportError):
        paths = modules.hardware.get_nvidia_paths()
        assert isinstance(paths, list)


@patch("modules.hardware.subprocess.check_output")
def test_detect_nvidia_smi_success(mock_out):
    """Test _detect_nvidia_smi success path."""
    mock_out.return_value = b"GPU 0: NVIDIA RTX 4090"
    settings = {"is_nvidia": False}
    with patch.dict(os.environ, {}, clear=True):
        modules.hardware._detect_nvidia_smi(settings)
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert settings["is_nvidia"] is True


@patch("modules.hardware.subprocess.check_output")
def test_detect_nvidia_smi_failure(mock_out):
    """Test _detect_nvidia_smi failure path."""
    mock_out.side_effect = Exception("No SMI")
    settings = {"is_nvidia": False}
    modules.hardware._detect_nvidia_smi(settings)
    assert settings["is_nvidia"] is False


@patch("modules.hardware.subprocess.check_output")
def test_detect_nvidia_smi_non_nvidia_output(mock_out):
    """Test _detect_nvidia_smi ignores non-NVIDIA output."""
    mock_out.return_value = b"GPU 0: Intel Arc"
    settings = {"is_nvidia": False}
    with patch.dict(os.environ, {}, clear=True):
        modules.hardware._detect_nvidia_smi(settings)
        assert "CUDA_DEVICE_ORDER" not in os.environ
    assert settings["is_nvidia"] is False


def test_detect_pytorch_cuda_logic():
    """Test _detect_pytorch_cuda loop and settings application."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 2

    # Mock get_device_name to return Intel first (skip) then NVIDIA (pick)
    mock_cuda.get_device_name.side_effect = ["Intel Graphics", "NVIDIA RTX 4090"]

    # Mock properties for VRAM calculation (RTX 4090 ~ 24GB)
    mock_props = MagicMock()
    mock_props.total_memory = 24 * 1024**3
    mock_cuda.get_device_properties.return_value = mock_props

    settings = {"device_index": 0, "is_nvidia": False}
    with patch.object(modules.hardware, "torch", MagicMock(cuda=mock_cuda)):
        modules.hardware._detect_pytorch_cuda(settings)

    assert settings["device_index"] == 1
    assert settings["is_nvidia"] is True
    assert settings["profile_name"].startswith("EXTREME")
    assert settings["gpu_batch_size"] == 32


def test_detect_pytorch_cuda_no_cuda():
    """Test _detect_pytorch_cuda early return when CUDA not available."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = False
    mock_torch = MagicMock(cuda=mock_cuda)
    mock_torch.backends.mps.is_available.return_value = False
    settings = {"device_index": -1, "is_nvidia": True, "cuda_device": "cuda:3"}  # default test val
    with patch.object(modules.hardware, "torch", mock_torch):
        modules.hardware._detect_pytorch_cuda(settings)
    assert settings["device_index"] == 0
    assert settings["is_nvidia"] is False
    assert settings["cuda_device"] is None
    assert settings["cpu_only_fallback"] is True


def test_apply_env_vars_sets_cuda_environment():
    """CUDA-enabled settings should populate the expected child-process environment."""
    settings = {"is_nvidia": True, "device_index": 2}
    with patch.dict(os.environ, {}, clear=True):
        modules.hardware._apply_env_vars(settings)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert os.environ["ORT_TENSORRT_FP16_ENABLE"] == "1"


def test_apply_env_vars_keeps_current_process_device_label():
    """Direct calls should still expose the canonical current-process CUDA label."""
    settings = {"is_nvidia": True, "device_index": 2}
    with patch.dict(os.environ, {}, clear=True):
        modules.hardware._apply_env_vars(settings)
        assert settings["cuda_device"] == "cuda:0"


def test_apply_env_vars_cpu_only_fallback_does_not_set_cuda_env():
    """CPU-only fallback must not populate CUDA-related environment variables."""
    settings = {"is_nvidia": True, "device_index": 2, "cuda_device": "cuda:2", "cpu_only_fallback": True}
    with patch.dict(os.environ, {}, clear=True):
        modules.hardware._apply_env_vars(settings)
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
        assert "CUDA_DEVICE_ORDER" not in os.environ
        assert "ORT_TENSORRT_FP16_ENABLE" not in os.environ


def test_apply_env_vars_cpu_only_fallback_resets_cuda_state():
    """CPU-only fallback should reset the in-memory CUDA state as well."""
    settings = {"is_nvidia": True, "device_index": 2, "cuda_device": "cuda:2", "cpu_only_fallback": True}
    with patch.dict(os.environ, {}, clear=True):
        modules.hardware._apply_env_vars(settings)
    assert settings["is_nvidia"] is False
    assert settings["cuda_device"] is None


def test_get_cpu_name_hard_exception():
    """Test get_cpu_name handles winreg exception robustly (simulated on non-windows or locked registry)."""
    with (
        patch("sys.platform", "win32"),
        patch("modules.hardware.platform.processor", return_value="Fallback"),
        patch.object(modules.hardware, "winreg", None),
    ):
        # Fallback to platform.processor when winreg import fails
        assert modules.hardware.get_cpu_name() == "Fallback"


def test_get_cpu_name_closes_key_on_query_error():
    """Test get_cpu_name closes registry key and falls back when QueryValueEx fails."""
    mock_winreg = MagicMock()
    mock_key = MagicMock()
    mock_winreg.OpenKey.return_value = mock_key
    mock_winreg.QueryValueEx.side_effect = Exception("Registry read error")

    with (
        patch("sys.platform", "win32"),
        patch("modules.hardware.platform.processor", return_value="Fallback CPU"),
        patch.object(modules.hardware, "winreg", mock_winreg),
    ):
        assert modules.hardware.get_cpu_name() == "Fallback CPU"
        mock_winreg.CloseKey.assert_called_once_with(mock_key)


def test_get_gpu_name_non_nvidia_smi_output_returns_generic():
    """Test get_gpu_name returns generic when nvidia-smi output has no NVIDIA marker."""
    with (
        patch.object(modules.hardware, "torch", None),
        patch("modules.hardware.subprocess.check_output", return_value=b"GPU 0: Intel Arc"),
    ):
        assert modules.hardware.get_gpu_name() == "Generic / Not Detected"


@patch("modules.hardware.subprocess.check_output")
def test_get_nvidia_smi_gpu_name_uses_safe_subprocess_call(mock_out):
    """nvidia-smi lookup should avoid shell=True and use a finite timeout."""
    mock_out.return_value = b"GPU 0: NVIDIA RTX 5000 (UUID: abc-123)"

    result = modules.hardware._get_nvidia_smi_gpu_name()

    assert result == "NVIDIA RTX 5000"
    mock_out.assert_called_once_with(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, timeout=5)


@patch("modules.hardware.subprocess.check_output")
def test_get_nvidia_smi_gpu_name_timeout_returns_none(mock_out):
    """Timeouts from nvidia-smi should be handled as no GPU name."""
    mock_out.side_effect = subprocess.TimeoutExpired(cmd=["nvidia-smi", "-L"], timeout=5)

    assert modules.hardware._get_nvidia_smi_gpu_name() is None


def test_module_import_without_optional_deps():
    """Test module import guards when optional torch/winreg are unavailable."""
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"torch", "winreg"}:
            raise ImportError(f"Missing optional dependency: {name}")
        return original_import(name, globals, locals, fromlist, level)

    try:
        with patch("builtins.__import__", side_effect=fake_import):
            importlib.reload(modules.hardware)
            assert modules.hardware.torch is None
            assert modules.hardware.winreg is None
    finally:
        importlib.reload(modules.hardware)


def test_get_nvidia_paths_torch_lib_missing_and_module_file_fallback():
    """Test get_nvidia_paths when torch lib path missing and nvidia modules use __file__ fallback."""
    torch_module = types.ModuleType("torch")
    torch_module.__file__ = "/fake/torch/__init__.py"

    nvidia_pkg = types.ModuleType("nvidia")
    cudnn_module = types.ModuleType("nvidia.cudnn")
    cublas_module = types.ModuleType("nvidia.cublas")
    cudnn_module.__file__ = "/fake/cudnn/__init__.py"
    cublas_module.__file__ = "/fake/cublas/__init__.py"
    nvidia_pkg.cudnn = cudnn_module
    nvidia_pkg.cublas = cublas_module

    sys_modules_patch = {
        "torch": torch_module,
        "nvidia": nvidia_pkg,
        "nvidia.cudnn": cudnn_module,
        "nvidia.cublas": cublas_module,
    }

    with (
        patch.dict("sys.modules", sys_modules_patch),
        patch("modules.hardware.Path.exists", return_value=False),
        patch("modules.hardware.os.path.exists", side_effect=lambda p: p.endswith("bin")),
    ):
        paths = modules.hardware.get_nvidia_paths()

    assert any(path.replace("\\", "/").endswith("cudnn/bin") for path in paths)
    assert any(path.replace("\\", "/").endswith("cublas/bin") for path in paths)


def test_get_linux_cpu_name_success():
    """Test _get_linux_cpu_name parses /proc/cpuinfo correctly."""
    cpuinfo_content = "processor\t: 0\nmodel name\t: AMD Ryzen 9 5950X 16-Core Processor\nflags\t: fpu\n"
    with patch("builtins.open", mock_open(read_data=cpuinfo_content)):
        result = modules.hardware._get_linux_cpu_name()
        assert result == "AMD Ryzen 9 5950X 16-Core Processor"


def test_get_linux_cpu_name_oserror():
    """Test _get_linux_cpu_name returns None on OSError."""
    with patch("modules.hardware.open", side_effect=OSError("Read error")):
        assert modules.hardware._get_linux_cpu_name() is None


@patch("modules.hardware.subprocess.check_output")
def test_get_macos_cpu_name_success(mock_out):
    """Test _get_macos_cpu_name parses sysctl brand string."""
    mock_out.return_value = b"Apple M3 Max\n"
    assert modules.hardware._get_macos_cpu_name() == "Apple M3 Max"
    mock_out.assert_called_once_with(["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL, timeout=5)


@patch("modules.hardware.subprocess.check_output")
def test_get_macos_cpu_name_error(mock_out):
    """Test _get_macos_cpu_name returns None on subprocess error."""
    mock_out.side_effect = subprocess.CalledProcessError(1, ["sysctl"])
    assert modules.hardware._get_macos_cpu_name() is None


def test_get_unix_cpu_name_dispatch():
    """Test _get_unix_cpu_name dispatches based on sys.platform."""
    with patch("sys.platform", "linux"), patch("modules.hardware._get_linux_cpu_name", return_value="Linux CPU"):
        assert modules.hardware._get_unix_cpu_name() == "Linux CPU"

    with patch("sys.platform", "darwin"), patch("modules.hardware._get_macos_cpu_name", return_value="Mac CPU"):
        assert modules.hardware._get_unix_cpu_name() == "Mac CPU"

    with patch("sys.platform", "win32"):
        assert modules.hardware._get_unix_cpu_name() is None


def _detect_mps_with_stale_fallback(mem_gb=12.0):
    """Runs MPS detection on a settings dict left over from a CPU-only probe."""
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = True
    settings = {"cpu_only_fallback": True}
    with patch.object(modules.hardware, "torch", mock_torch):
        with patch("modules.hardware._get_macos_total_memory_gb", return_value=mem_gb):
            detected = modules.hardware._detect_mps_backend(settings)
    return detected, settings


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("cuda_device", "mps"),
        ("cuda_child_device", "mps"),
        ("is_mps", True),
        ("gpu_batch_size", 4),
        ("gpu_vram_gb", 12.0),
        # Left unset, _apply_env_vars would take the CPU-only branch and wipe
        # cuda_device back to None, silently disabling MPS.
        ("cpu_only_fallback", False),
    ],
)
def test_detect_mps_backend_sets_every_flag_it_owns(key, expected):
    """MPS detection must leave no flag it owns carrying a stale value."""
    detected, settings = _detect_mps_with_stale_fallback(mem_gb=12.0)
    assert detected is True
    assert settings[key] == expected


def test_detect_mps_backend_available():
    """Test _detect_mps_backend when Apple Silicon MPS is available."""
    detected, settings = _detect_mps_with_stale_fallback(mem_gb=32.0)
    assert detected is True
    assert settings["profile_name"] == "APPLE SILICON (32 GB Unified Memory)"
    assert settings["gpu_batch_size"] == 32
    assert settings["gpu_vram_gb"] == 32.0


def test_mps_survives_apply_env_vars_after_a_cpu_only_probe():
    """The end-to-end guarantee the cpu_only_fallback reset exists to provide."""
    _, settings = _detect_mps_with_stale_fallback()
    with patch.dict(os.environ, {}, clear=True):
        modules.hardware._apply_env_vars(settings)
    assert settings["cuda_device"] == "mps"
    assert settings["cuda_child_device"] == "mps"


def test_detect_mps_backend_unavailable():
    """Test _detect_mps_backend when MPS is not available."""
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = False
    settings = {}
    with patch.object(modules.hardware, "torch", mock_torch):
        assert modules.hardware._detect_mps_backend(settings) is False

    with patch.object(modules.hardware, "torch", None):
        assert modules.hardware._detect_mps_backend(settings) is False


def test_detect_pytorch_device_mps_fallback():
    """Test _detect_pytorch_device falls back to MPS when CUDA is not available."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    settings = {}
    with patch.object(modules.hardware, "torch", mock_torch):
        modules.hardware._detect_pytorch_device(settings)
        assert settings["cuda_device"] == "mps"
        assert settings["is_mps"] is True


def test_apply_env_vars_mps():
    """Test _apply_env_vars configuration when is_mps is True."""
    settings = {"is_nvidia": False, "is_mps": True, "cuda_device": "mps"}
    initial_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "ORT_TENSORRT_FP16_ENABLE": "1",
    }
    with patch.dict(os.environ, initial_env, clear=True):
        modules.hardware._apply_env_vars(settings)
        assert settings["cuda_child_device"] == "mps"
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
        assert "CUDA_DEVICE_ORDER" not in os.environ
        assert "ORT_TENSORRT_FP16_ENABLE" not in os.environ


def test_get_macos_gpu_name_integrated():
    """Test _get_macos_gpu_name returns Apple Silicon GPU on Darwin."""
    with patch("sys.platform", "darwin"), patch("modules.hardware._get_macos_cpu_name", return_value="Apple M2"):
        assert "Apple M2 (Integrated GPU)" in modules.hardware._get_macos_gpu_name()

    with patch("sys.platform", "darwin"), patch("modules.hardware._get_macos_cpu_name", return_value=None):
        with patch.object(modules.hardware, "IS_MPS", True):
            assert "MPS" in modules.hardware._get_macos_gpu_name()

    with patch("sys.platform", "linux"):
        assert modules.hardware._get_macos_gpu_name() is None


def test_get_gpu_name_macos_dispatch():
    """Test get_gpu_name dispatches to macOS GPU name when no CUDA or SMI."""
    with (
        patch.object(modules.hardware, "torch", None),
        patch("modules.hardware.subprocess.check_output", side_effect=Exception("No SMI")),
        patch("modules.hardware._get_macos_gpu_name", return_value="Apple M1 (Integrated GPU)"),
    ):
        assert modules.hardware.get_gpu_name() == "Apple M1 (Integrated GPU)"


def test_dynamic_linker_env_updates():
    """Test _update_dynamic_linker_env and prepare_runtime_library_paths."""
    fake_path = "/opt/nvidia/cuda/lib64"
    with patch.dict(os.environ, {}, clear=True):
        modules.hardware._update_dynamic_linker_env(fake_path)
        assert fake_path in os.environ.get("LD_LIBRARY_PATH", "")
        assert fake_path in os.environ.get("DYLD_LIBRARY_PATH", "")

        with patch("modules.hardware.get_nvidia_paths", return_value=[fake_path]):
            modules.hardware.prepare_runtime_library_paths()
            assert fake_path in os.environ["LD_LIBRARY_PATH"]
