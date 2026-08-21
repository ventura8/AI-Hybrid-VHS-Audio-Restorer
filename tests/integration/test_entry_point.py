import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

import restore_audio_hybrid

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def restored_entry_module():
    """Reloads the entry module under the real platform after a patched reload."""
    yield
    importlib.reload(restore_audio_hybrid)


def test_main_success():
    """Test main execution path with valid inputs."""
    mock_input_files = ([Path("test.mp4")], False)

    with (
        patch("restore_audio_hybrid.run_init_sequence", return_value=("CPU", "GPU")),
        patch("restore_audio_hybrid.check_dependencies", return_value=True),
        patch("restore_audio_hybrid._show_banner"),
        patch("restore_audio_hybrid.OUTPUT_DIR") as mock_output_dir,
        patch("restore_audio_hybrid._get_input_files", return_value=mock_input_files),
        patch("restore_audio_hybrid.process_hybrid_audio") as mock_process,
        patch("builtins.input"),
    ):
        restore_audio_hybrid.main()

        mock_output_dir.mkdir.assert_called_once()
        mock_process.assert_called_once()
        args, kwargs = mock_process.call_args
        assert args[0] == Path("test.mp4")
        assert args[1] == "GPU"


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_main_prints_help_and_exits_early(flag, capsys):
    """`-h` / `--help` print usage and never touch the init sequence."""
    with (
        patch.object(restore_audio_hybrid.sys, "argv", ["restore_audio_hybrid.py", flag]),
        patch("restore_audio_hybrid.run_init_sequence") as mock_init,
        patch("restore_audio_hybrid.check_dependencies") as mock_deps,
    ):
        restore_audio_hybrid.main()

    mock_init.assert_not_called()
    mock_deps.assert_not_called()
    assert "Usage:" in capsys.readouterr().out


def test_wants_help_false_without_flag():
    """Regular target arguments do not trigger the help screen."""
    assert restore_audio_hybrid._wants_help(["video.mp4"]) is False


def test_main_no_dependencies():
    """Test main exits if dependencies missing."""
    with (
        patch("restore_audio_hybrid.run_init_sequence", return_value=("CPU", "GPU")),
        patch("restore_audio_hybrid.check_dependencies", return_value=False),
        patch("builtins.print") as mock_print,
    ):
        restore_audio_hybrid.main()

        # Should print error and return
        assert any("Critical Error" in str(c) for c in mock_print.call_args_list)


def test_main_no_files():
    """Test main exits if no files selected."""
    with (
        patch("restore_audio_hybrid.run_init_sequence", return_value=("CPU", "GPU")),
        patch("restore_audio_hybrid.check_dependencies", return_value=True),
        patch("restore_audio_hybrid._show_banner"),
        patch("restore_audio_hybrid.OUTPUT_DIR"),
        patch("restore_audio_hybrid._get_input_files", return_value=([], False)),
        patch("builtins.print") as mock_print,
    ):
        restore_audio_hybrid.main()
        assert any("No valid video files found" in str(c) for c in mock_print.call_args_list)


def test_main_keyboard_interrupt():
    """Test main handles interruptions graciously."""
    mock_input_files = ([Path("test.mp4")], False)

    with (
        patch("restore_audio_hybrid.run_init_sequence", return_value=("CPU", "GPU")),
        patch("restore_audio_hybrid.check_dependencies", return_value=True),
        patch("restore_audio_hybrid._show_banner"),
        patch("restore_audio_hybrid.OUTPUT_DIR"),
        patch("restore_audio_hybrid._get_input_files", return_value=mock_input_files),
        patch("restore_audio_hybrid.process_hybrid_audio"),
        patch("builtins.input", side_effect=KeyboardInterrupt),
        patch("builtins.print") as mock_print,
    ):
        restore_audio_hybrid.main()
        assert any("Exiting gracefully" in str(call) for call in mock_print.call_args_list)


def test_windows_reconfigure_stdout_success(restored_entry_module):
    """Test Windows stdout/stderr reconfiguration."""
    with (
        patch("sys.platform", "win32"),
        patch.object(restore_audio_hybrid.sys.stdout, "reconfigure", create=True) as mock_out_rec,
        patch.object(restore_audio_hybrid.sys.stderr, "reconfigure", create=True) as mock_err_rec,
    ):
        importlib.reload(restore_audio_hybrid)
        mock_out_rec.assert_called_with(encoding="utf-8", errors="replace")
        mock_err_rec.assert_called_with(encoding="utf-8", errors="replace")


def test_windows_reconfigure_stdout_exception(restored_entry_module):
    """Test Windows stdout/stderr reconfiguration handles exceptions."""
    with (
        patch("sys.platform", "win32"),
        patch.object(restore_audio_hybrid.sys.stdout, "reconfigure", create=True, side_effect=RuntimeError("reconfigure failed")),
    ):
        importlib.reload(restore_audio_hybrid)


def test_entry_point_name_main():
    """Test __main__ block execution via runpy."""
    import runpy

    with (
        patch("modules.ui.run_init_sequence", return_value=("CPU", "GPU")),
        patch("modules.utils.check_dependencies", return_value=False),
        patch("builtins.print") as mock_print,
    ):
        runpy.run_path(str(REPO_ROOT / "restore_audio_hybrid.py"), run_name="__main__")
        assert any("Critical Error" in str(c) for c in mock_print.call_args_list)
