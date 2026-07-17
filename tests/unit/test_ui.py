from unittest.mock import patch

import modules.ui
import modules.utils


def test_show_banner(capsys):
    """Test _show_banner function displays banner."""
    modules.ui._show_banner()
    captured = capsys.readouterr()
    assert "AI HYBRID VHS AUDIO RESTORER" in captured.out
    assert "HARDWARE DETECTED" in captured.out


@patch("modules.ui.input")
def test_get_input_files_cli_valid(mock_input, tmp_path):
    """Test _get_input_files with valid CLI argument."""
    video = tmp_path / "test.mp4"
    video.write_text("video")

    with patch("sys.argv", ["script.py", str(video)]):
        files, use_source = modules.ui._get_input_files()
        assert len(files) == 1
        assert use_source is True


@patch("modules.ui.input")
def test_get_input_files_interactive_empty(mock_input, tmp_path):
    """Test _get_input_files returns empty when no valid input."""
    # Simulate user pressing Enter without input, then folder scan finds nothing
    mock_input.side_effect = ["", ""]  # Empty triggers folder scan

    with patch("sys.argv", ["script.py"]):
        empty_dir = tmp_path / "empty_input"
        empty_dir.mkdir()
        with patch.object(modules.ui, "INPUT_DIR", empty_dir):
            files, use_source = modules.ui._get_input_files()
            # Empty folder returns empty list
            assert files == []
            assert use_source is False


@patch("modules.ui.input")
def test_get_input_files_keyboard_interrupt(mock_input):
    """Test _get_input_files handles keyboard interrupt."""
    mock_input.side_effect = KeyboardInterrupt()

    with patch("sys.argv", ["script.py", "/bad.mp4"]):
        files, use_source = modules.ui._get_input_files()
        assert files == []


@patch("modules.ui.input")
def test_get_input_files_powershell_style(mock_input, tmp_path):
    """Test _get_input_files handles PowerShell & 'path' style."""
    video = tmp_path / "video.mp4"
    video.write_text("content")

    # Input mimics: & 'C:\path\to\video.mp4'
    ps_input = f"& '{str(video)}'"
    mock_input.return_value = ps_input

    with patch("sys.argv", ["script.py"]):  # No args, trigger input
        files, use_source = modules.ui._get_input_files()
        assert len(files) == 1
        assert files[0] == video


def test_get_input_files_recursive_glob(tmp_path):
    """Test coverage for glob branches in _get_input_files."""
    d = tmp_path / "glob_input"
    d.mkdir()
    (d / "v1.mp4").write_text("v")

    with patch("modules.ui.INPUT_DIR", d):
        with patch("sys.argv", ["script.py"]):
            with patch("modules.ui.input", side_effect=["", ""]):
                files, skip = modules.ui._get_input_files()
                assert len(files) == 1


def test_scan_files_excludes_cleaned_outputs(tmp_path, capsys):
    """Ensure scanners do not requeue generated cleaned outputs."""
    source = tmp_path / "source.mp4"
    source.write_text("v")
    hybrid = tmp_path / "source_Hybrid_Cleaned.mp4"
    hybrid.write_text("v")
    denoised = tmp_path / "source_Denoised_Cleaned.mp4"
    denoised.write_text("v")

    files = modules.ui._scan_files_in_path(tmp_path)
    assert files == [source]

    hybrid_direct = modules.ui._scan_files_in_path(hybrid)
    denoised_direct = modules.ui._scan_files_in_path(denoised)
    assert hybrid_direct == []
    assert denoised_direct == []
    captured = capsys.readouterr()
    assert "Skipping cleaned output file" in captured.out


def test_scan_files_unsupported(tmp_path, capsys):
    """Test _scan_files_in_path with unsupported file extension."""
    f = tmp_path / "test.txt"
    f.touch()
    files = modules.ui._scan_files_in_path(f)
    assert len(files) == 0
    captured = capsys.readouterr()
    assert "[Error] Unsupported extension" in captured.out


def test_clean_user_input_variations():
    """Test various input cleanup scenarios."""
    # file:// prefix
    assert modules.ui._clean_user_input("file://C:/test.mp4") == "C:/test.mp4"
    # Quotes
    assert modules.ui._clean_user_input('"C:/test.mp4"') == "C:/test.mp4"
    assert modules.ui._clean_user_input("'C:/test.mp4'") == "C:/test.mp4"
    # Combined
    assert modules.ui._clean_user_input("& 'file://C:/test.mp4'") == "C:/test.mp4"


@patch("modules.ui.input")
def test_get_interactive_files_not_found(mock_input, capsys):
    """Test _get_interactive_files when file does not exist."""
    mock_input.return_value = "nonexistent.mp4"
    files, use_source = modules.ui._get_interactive_files()
    assert files == []
    assert use_source is False
    captured = capsys.readouterr()
    assert "[Error] File not found" in captured.out


@patch("modules.ui.input", side_effect=EOFError)
def test_get_interactive_files_eof(mock_input):
    """Test _get_interactive_files handles EOFError."""
    files, use_source = modules.ui._get_interactive_files()
    assert files == []


@patch("modules.ui.draw_progress_bar")
@patch("modules.ui.time.sleep")
@patch("modules.ui.get_cpu_name", return_value="TestCPU")
@patch("modules.ui.get_gpu_name", return_value="TestGPU")
def test_run_init_sequence(mock_gpu, mock_cpu, mock_sleep, mock_bar):
    """Test run_init_sequence coverage."""
    cpu, gpu = modules.ui.run_init_sequence()
    assert cpu == "TestCPU"
    assert gpu == "TestGPU"
    assert mock_bar.call_count >= 5


def test_get_torch_backend_prefers_cuda_capability_over_gpu_name():
    """CUDA availability should still report a CUDA backend even for non-NVIDIA names."""
    with patch("modules.ui._has_cuda_backend", return_value=True):
        assert modules.ui._get_torch_backend("Unknown GPU") == "CUDA (Accelerated)"


def test_get_torch_backend_uses_nvidia_label_for_nvidia_names():
    """NVIDIA-like GPU names should keep the NVIDIA label when CUDA is available."""
    with patch("modules.ui._has_cuda_backend", return_value=True):
        assert modules.ui._get_torch_backend("NVIDIA RTX 4090") == "CUDA (NVIDIA Accelerated)"


def test_get_torch_backend_xpu_and_cpu_fallbacks():
    """Backend selection should preserve XPU and CPU fallback behavior."""
    with patch("modules.ui._has_cuda_backend", return_value=False), patch("modules.ui._has_xpu_backend", return_value=True):
        assert modules.ui._get_torch_backend("Any") == "XPU (Intel Accelerated)"

    with patch("modules.ui._has_cuda_backend", return_value=False), patch("modules.ui._has_xpu_backend", return_value=False):
        assert modules.ui._get_torch_backend("Any") == "CPU (Slow)"
