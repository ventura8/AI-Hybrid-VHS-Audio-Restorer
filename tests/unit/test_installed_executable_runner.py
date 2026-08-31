"""Unit tests for test_installed_executable test runner script."""

from unittest.mock import MagicMock, patch

import pytest

import scripts.test_installed_executable as tie


def test_run_cmd_success():
    """Test successful command execution."""
    with patch("scripts.test_installed_executable.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")
        res = tie._run_cmd(["echo", "hello"])
        assert res.returncode == 0
        assert res.stdout == "Success"
        mock_run.assert_called_once_with(
            ["echo", "hello"],
            cwd=None,
            input=None,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )


def test_run_cmd_failure():
    """Test failed command execution raising RuntimeError."""
    with patch("scripts.test_installed_executable.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error")
        with pytest.raises(RuntimeError, match="Command failed with exit code 1"):
            tie._run_cmd(["false"])


def test_create_test_fixture(tmp_path):
    """Test test fixture synthesis function."""
    out_file = tmp_path / "fixture.mp4"
    with patch("scripts.test_installed_executable._run_cmd") as mock_cmd:
        out_file.write_text("dummy")
        tie._create_test_fixture(out_file)
        mock_cmd.assert_called_once()


def test_create_test_fixture_failure(tmp_path):
    """Test fixture failure when output file is not generated."""
    out_file = tmp_path / "fixture.mp4"
    with patch("scripts.test_installed_executable._run_cmd"):
        with pytest.raises(RuntimeError, match="Failed to generate test fixture"):
            tie._create_test_fixture(out_file)


def test_verify_output(tmp_path):
    """Test output verification with existing non-empty file."""
    out_file = tmp_path / "output.mp4"
    out_file.write_text("media")
    with patch("scripts.test_installed_executable._run_cmd") as mock_cmd:
        tie._verify_output(out_file)
        mock_cmd.assert_called_once()


def test_verify_output_missing(tmp_path):
    """Test output verification failing on missing file."""
    out_file = tmp_path / "missing.mp4"
    with pytest.raises(AssertionError, match="does not exist or is empty"):
        tie._verify_output(out_file)


def test_test_help_flag():
    """Test help flag verification scenario."""
    with patch("scripts.test_installed_executable._run_cmd") as mock_cmd:
        mock_cmd.return_value = MagicMock(stdout="AI Hybrid VHS Audio Restorer Usage...")
        tie._test_help_flag(["launcher"])
        assert mock_cmd.call_count == 2


def test_test_help_flag_missing_text():
    """Test help flag assertion error on missing text."""
    with patch("scripts.test_installed_executable._run_cmd") as mock_cmd:
        mock_cmd.return_value = MagicMock(stdout="Different text")
        with pytest.raises(AssertionError, match="Help text missing"):
            tie._test_help_flag(["launcher"])


def test_test_single_file(tmp_path):
    """Test single file restoration testing scenario."""
    cleaned = tmp_path / "test_single_FFmpeg_Cleaned.mp4"
    cleaned.write_text("clean")
    with (
        patch("scripts.test_installed_executable._create_test_fixture"),
        patch("scripts.test_installed_executable._run_cmd"),
        patch("scripts.test_installed_executable._verify_output") as mock_verify,
    ):
        tie._test_single_file(["launcher"], tmp_path, extension=".mp4")
        mock_verify.assert_called_once()


def test_test_multi_files(tmp_path):
    """Test multiple files restoration testing scenario."""
    with (
        patch("scripts.test_installed_executable._create_test_fixture"),
        patch("scripts.test_installed_executable._run_cmd"),
        patch("scripts.test_installed_executable._verify_output") as mock_verify,
    ):
        tie._test_multi_files(["launcher"], tmp_path)
        assert mock_verify.call_count == 3


def test_test_directory_input(tmp_path):
    """Test folder directory input testing scenario."""
    with (
        patch("scripts.test_installed_executable._create_test_fixture"),
        patch("scripts.test_installed_executable._run_cmd"),
        patch("scripts.test_installed_executable._verify_output") as mock_verify,
    ):
        tie._test_directory_input(["launcher"], tmp_path)
        assert mock_verify.call_count == 2


def test_main_cli_dispatch(tmp_path):
    """Test CLI entry point with mock functions."""
    launcher = tmp_path / "mock_app"
    launcher.write_text("dummy")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("process_mode: 'auto'")

    with (
        patch.object(tie.sys, "argv", ["test_installed_executable.py", str(launcher), "--config-dir", str(config_dir)]),
        patch("scripts.test_installed_executable._test_help_flag"),
        patch("scripts.test_installed_executable._test_single_file"),
        patch("scripts.test_installed_executable._test_multi_files"),
        patch("scripts.test_installed_executable._test_directory_input"),
    ):
        tie.main()
        cfg_text = (config_dir / "config.yaml").read_text(encoding="utf-8")
        assert 'process_mode: "vhs_native"' in cfg_text
