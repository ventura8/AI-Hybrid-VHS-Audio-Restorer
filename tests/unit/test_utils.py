import os
import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import modules.utils


@pytest.fixture
def mock_utils_sf(monkeypatch):
    mock_sf = MagicMock()
    monkeypatch.setattr(modules.utils, "sf", mock_sf)
    return mock_sf


# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------


def test_is_valid_audio_robust(tmp_path, mock_utils_sf):
    """Test audio validation."""
    p = tmp_path / "v.wav"
    p.write_text("x" * 1500)
    mock_ctx = mock_utils_sf.SoundFile.return_value.__enter__.return_value
    mock_ctx.frames = 50000  # > 0.1s
    mock_ctx.samplerate = 44100
    assert modules.utils.is_valid_audio(p) is True
    # Test internal check for non-existence logic is covered by is_valid_audio implementation
    assert modules.utils.is_valid_audio(tmp_path / "no") is False


def test_is_valid_audio_small_file(tmp_path):
    """Test is_valid_audio with small files."""
    # File smaller than 1KB
    small_file = tmp_path / "small.wav"
    small_file.write_text("x" * 500)
    assert modules.utils.is_valid_audio(small_file) is False


def test_is_valid_audio_zero_frames(tmp_path, mock_utils_sf):
    """Test is_valid_audio with zero frames."""
    p = tmp_path / "empty.wav"
    p.write_text("x" * 1500)
    mock_utils_sf.SoundFile.return_value.__enter__.return_value.frames = 0
    assert modules.utils.is_valid_audio(p) is False


def test_is_valid_audio_exception(tmp_path, mock_utils_sf):
    """Test is_valid_audio handles SoundFile exceptions."""
    p = tmp_path / "corrupt.wav"
    p.write_text("x" * 2000)
    mock_utils_sf.SoundFile.side_effect = Exception("Corrupt file")
    assert modules.utils.is_valid_audio(p) is False


def test_is_valid_audio_returns_false_when_soundfile_missing(tmp_path, monkeypatch):
    p = tmp_path / "missing_sf.wav"
    p.write_text("x" * 2000)
    monkeypatch.setattr(modules.utils, "sf", None)
    assert modules.utils.is_valid_audio(p) is False


def test_is_valid_audio_filesystem_oserror_fallback(tmp_path):
    """is_valid_audio should return False when file metadata checks raise OSError."""
    p = tmp_path / "oserror.wav"
    p.write_text("x" * 2000)

    with patch.object(Path, "exists", side_effect=OSError("exists failed")):
        assert modules.utils.is_valid_audio(p) is False

    with patch.object(Path, "exists", return_value=True), patch.object(Path, "stat", side_effect=OSError("stat failed")):
        assert modules.utils.is_valid_audio(p) is False


def test_is_valid_video_small(tmp_path):
    """Test is_valid_video rejects files smaller than 1MB."""
    p = tmp_path / "small.mp4"
    p.write_text("x" * 500)  # 500 bytes
    assert modules.utils.is_valid_video(p) is False


def test_retry_loop():
    """Test retry loop with batch size reduction."""
    # First call fails, second succeeds
    mock_proc_fail = MagicMock()
    mock_proc_fail.wait.return_value = None
    mock_proc_fail.returncode = 1  # Failure

    mock_proc_success = MagicMock()
    mock_proc_success.wait.return_value = None
    mock_proc_success.returncode = 0  # Success

    # Fix: Mock stdout.readline to avoid regex TypeError
    mock_proc_fail.stdout.readline.return_value = ""
    mock_proc_success.stdout.readline.return_value = ""

    with patch("modules.utils.subprocess.Popen", side_effect=[mock_proc_fail, mock_proc_success]):
        # Test with batch size > 1 so it can retry
        result = modules.utils.attempt_run_with_retry(lambda b: ["echo", str(b)], 2)
        assert result is True


@patch("modules.utils.subprocess.run")
def test_deps_fail(mock_run):
    """Test dependency check failure."""
    # Use FileNotFoundError since that's what check_dependencies catches
    mock_run.side_effect = FileNotFoundError("ffmpeg not found")
    assert modules.utils.check_dependencies() is False
    assert mock_run.call_count == 4
    for call in mock_run.call_args_list:
        assert call.kwargs["timeout"] == 10


@patch("modules.utils.subprocess.run")
def test_deps_fail_timeout(mock_run):
    """Test dependency check failure when probe commands time out."""
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg", "-version"], timeout=10)
    assert modules.utils.check_dependencies() is False
    assert mock_run.call_count == 4
    for call in mock_run.call_args_list:
        assert call.kwargs["timeout"] == 10


def test_draw_progress_bar_renders_progress(capsys):
    """Progress bar rendering should include the percentage and label."""
    modules.utils._last_bar_time = 0
    modules.utils.draw_progress_bar(50, "Testing...")
    captured = capsys.readouterr()
    assert "50.0%" in captured.out
    assert "Testing..." in captured.out


def test_draw_progress_bar_clamps_low_values(capsys):
    """Progress bar rendering should clamp negative values to zero."""
    modules.utils._last_bar_time = 0
    modules.utils.draw_progress_bar(-10, "Under")
    captured = capsys.readouterr()
    assert "0.0%" in captured.out


def test_draw_progress_bar_clamps_high_values(capsys):
    """Progress bar rendering should clamp values above 100 percent."""
    modules.utils._last_bar_time = 0
    modules.utils.draw_progress_bar(150, "Over")
    captured = capsys.readouterr()
    assert "100.0%" in captured.out


def test_log_msg_writes_console_and_file(tmp_path, capsys):
    """log_msg should print messages and persist them to the log file."""
    original_log = modules.utils.LOG_FILE
    modules.utils.LOG_FILE = tmp_path / "test_log.txt"

    try:
        modules.utils.log_msg("Test message", console=True)
        captured = capsys.readouterr()
        assert "Test message" in captured.out

        modules.utils.log_msg("Error!", is_error=True)
        captured = capsys.readouterr()
        assert "Error!" in captured.out

        log_content = modules.utils.LOG_FILE.read_text()
        assert "Test message" in log_content
        assert "ERROR" in log_content
    finally:
        modules.utils.LOG_FILE = original_log


def test_log_msg_suppresses_debug_and_silent_output(tmp_path, capsys):
    """Debug and silent log messages should not reach the console."""
    original_log = modules.utils.LOG_FILE
    modules.utils.LOG_FILE = tmp_path / "test_log.txt"

    try:
        modules.utils.log_msg("Debug info", level="DEBUG")
        captured = capsys.readouterr()
        assert "Debug info" not in captured.out

        modules.utils.log_msg("Silent", console=False)
        captured = capsys.readouterr()
        assert "Silent" not in captured.out
    finally:
        modules.utils.LOG_FILE = original_log


def test_log_msg_file_error(tmp_path, capsys, monkeypatch):
    """Test log_msg handles file write errors gracefully."""
    # Set LOG_FILE to an existing directory so file write fails deterministically.
    original = modules.utils.LOG_FILE
    modules.utils.LOG_FILE = tmp_path

    try:
        # Should not raise, just silently fail file write
        modules.utils.log_msg("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out
    finally:
        modules.utils.LOG_FILE = original


def test_parse_ffmpeg_time():
    """Test FFmpeg time parsing."""
    # Standard format
    result = modules.utils.parse_ffmpeg_time("time=01:23:45.67")
    assert result == pytest.approx(1 * 3600 + 23 * 60 + 45 + 0.67, rel=0.01)

    # Zero time
    result = modules.utils.parse_ffmpeg_time("time=00:00:00.00")
    assert result == 0.0

    # No match
    result = modules.utils.parse_ffmpeg_time("some random text")
    assert result is None


def test_format_time_negative():
    """Test format_time with negative input."""
    assert modules.utils.format_time(-10) == "00:00:00,000"


@patch("modules.utils.subprocess.Popen")
def test_run_command_with_progress_passthrough(mock_popen, capsys):
    """Test run_command_with_progress without duration (passthrough mode)."""
    mock_proc = MagicMock()
    mock_proc.wait.return_value = None
    mock_proc.returncode = 0
    # Fix: Mock stdout.readline
    mock_proc.stdout.readline.return_value = ""
    mock_popen.return_value = mock_proc

    # Run without total_duration - passthrough mode
    modules.utils.run_command_with_progress(["echo", "test"], description="Testing passthrough")
    captured = capsys.readouterr()
    assert "Testing passthrough" in captured.out


@patch("modules.utils.subprocess.Popen")
def test_run_command_with_progress_passthrough_fail(mock_popen):
    """Test run_command_with_progress failure in passthrough mode."""
    mock_proc = MagicMock()
    mock_proc.wait.return_value = None
    mock_proc.returncode = 1  # Failure
    # Fix: Mock stdout.readline
    mock_proc.stdout.readline.return_value = ""
    mock_popen.return_value = mock_proc

    with pytest.raises(subprocess.CalledProcessError):
        modules.utils.run_command_with_progress(["fail_cmd"])


@patch("modules.utils.subprocess.Popen")
def test_run_command_with_progress_ffmpeg(mock_popen):
    """Test run_command_with_progress with FFmpeg progress parsing."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    # Fix: Mock stdout.readline to avoid regex TypeError
    mock_proc.stdout.readline.return_value = ""

    # Create a list of lines to return, then empty forever
    lines = [
        "frame=100 time=00:00:05.00 bitrate=1000kbits/s",
        "frame=200 time=00:00:10.00 bitrate=1000kbits/s",
        "",  # Empty signals end
    ]
    line_iter = iter(lines)

    def readline_mock():
        try:
            return next(line_iter)
        except StopIteration:
            return ""

    mock_proc.stderr.readline = readline_mock

    # poll returns None while there's output, then 0 when done
    poll_values = [None, None, 0]
    poll_iter = iter(poll_values)

    def poll_mock():
        try:
            return next(poll_iter)
        except StopIteration:
            return 0

    mock_proc.poll = poll_mock
    mock_popen.return_value = mock_proc

    modules.utils.run_command_with_progress(["ffmpeg", "-i", "input.mp4", "output.mp4"], total_duration=20.0, description="Encoding")
    mock_proc.wait.assert_called()


@patch("modules.utils.subprocess.Popen")
def test_run_command_with_progress_ffmpeg_fail(mock_popen):
    """Test run_command_with_progress FFmpeg failure."""
    mock_proc = MagicMock()
    mock_proc.returncode = 1  # Failure
    mock_proc.poll.return_value = 1
    # Fix: Mock stdout.readline to avoid regex TypeError
    mock_proc.stdout.readline.return_value = ""
    mock_proc.stderr.readline.return_value = ""
    mock_popen.return_value = mock_proc

    with pytest.raises(subprocess.CalledProcessError):
        modules.utils.run_command_with_progress(["ffmpeg"], total_duration=10.0)


@patch("modules.utils.run_command_with_progress")
def test_attempt_cpu_run_with_retry_success(mock_run):
    """Test CPU retry on first success."""
    mock_run.return_value = None  # Success
    result = modules.utils.attempt_cpu_run_with_retry(lambda t: ["cmd", str(t)], initial_threads=8, description="CPU Task")
    assert result is True
    mock_run.assert_called_once()


@patch("modules.utils.run_command_with_progress")
def test_attempt_cpu_run_with_retry_fallback(mock_run):
    """Test CPU retry with thread reduction."""
    # First call fails, second succeeds
    mock_run.side_effect = [
        subprocess.CalledProcessError(1, "cmd"),
        None,  # Success
    ]

    result = modules.utils.attempt_cpu_run_with_retry(lambda t: ["cmd", str(t)], initial_threads=4, description="CPU Fallback")
    assert result is True
    assert mock_run.call_count == 2


@patch("modules.utils.run_command_with_progress")
def test_attempt_cpu_run_with_retry_exhausted(mock_run):
    """Test CPU retry when all threads exhausted."""
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")

    with pytest.raises(subprocess.CalledProcessError):
        modules.utils.attempt_cpu_run_with_retry(lambda t: ["cmd", str(t)], initial_threads=1)


def test_attempt_run_with_retry_error_path():
    """Test attempt_run_with_retry error logging branch."""
    with patch("modules.utils.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = ""
        mock_proc.wait.return_value = None
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        # Should trigger warning log on first failure with batch > 1
        with patch("modules.utils.log_msg") as mock_log:
            with pytest.raises(subprocess.CalledProcessError):
                modules.utils.attempt_run_with_retry(lambda b: ["cmd"], 1)

            # Reset and try with batch > 1
            mock_popen.side_effect = [mock_proc, mock_proc]  # Fail twice
            with pytest.raises(subprocess.CalledProcessError):
                modules.utils.attempt_run_with_retry(lambda b: ["cmd"], 2)
            assert mock_log.called


def test_cleanup_subprocesses_none():
    """Test cleanup when no processes are active."""
    modules.utils._active_processes.clear()
    modules.utils.cleanup_subprocesses()
    # Should just return without error


def test_cleanup_subprocesses_active():
    """Test cleanup with active processes."""
    mock_p1 = MagicMock(spec=subprocess.Popen)
    mock_p1.poll.return_value = None

    mock_p2 = MagicMock(spec=subprocess.Popen)
    mock_p2.poll.return_value = 0  # Already done

    modules.utils._active_processes.add(mock_p1)
    modules.utils._active_processes.add(mock_p2)

    # Mock time.sleep to speed up test
    with patch("time.sleep"):
        modules.utils.cleanup_subprocesses()

    mock_p1.terminate.assert_called()
    mock_p1.kill.assert_called()
    assert len(modules.utils._active_processes) == 0


def test_len_active():
    """Test _len_active utility."""
    modules.utils._active_processes.clear()
    assert modules.utils._len_active() == 0
    proc = MagicMock()
    modules.utils._active_processes.clear()
    modules.utils._active_processes.add(proc)
    assert modules.utils._len_active() == 1
    assert proc in modules.utils._active_processes
    modules.utils._active_processes.clear()


@patch("modules.utils.draw_progress_bar")
@patch("modules.utils.subprocess.Popen")
def test_run_command_with_progress_adds_to_active(mock_popen, mock_bar):
    """Verify process is added to _active_processes during execution."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    # Return empty string immediately to finish loop
    mock_proc.stdout.readline.return_value = ""
    mock_popen.return_value = mock_proc

    # Assert membership while the process output loop is running.
    def stdout_side_effect():
        assert mock_proc in modules.utils._active_processes
        return ""

    mock_proc.stdout.readline.side_effect = stdout_side_effect
    mock_proc.wait.return_value = 0

    modules.utils.run_command_with_progress(["cmd"])

    # After it finishes, it should be removed (checked by other tests probably)
    # But inside wait(), it was active.
    assert mock_proc.wait.called


@patch("modules.utils.subprocess.Popen")
@patch("modules.utils._monitor_process_output")
def test_run_command_with_progress_monitor_exception_cleanup(mock_monitor, mock_popen):
    """Ensure monitor exceptions trigger terminate/kill cleanup paths."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.poll.side_effect = [None, 0]
    mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="cmd", timeout=5), 0]
    mock_popen.return_value = mock_proc
    mock_monitor.side_effect = RuntimeError("monitor boom")

    with pytest.raises(RuntimeError, match="monitor boom"):
        modules.utils.run_command_with_progress(["cmd"])

    assert mock_proc.terminate.called
    assert mock_proc.kill.called
    assert mock_proc not in modules.utils._active_processes


@patch("modules.utils.subprocess.Popen")
@patch("modules.utils._monitor_process_output")
def test_run_command_with_progress_monitor_exception_retains_active_process_on_cleanup_error(mock_monitor, mock_popen):
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.poll.return_value = None
    mock_proc.terminate.side_effect = RuntimeError("terminate failed")
    mock_popen.return_value = mock_proc
    mock_monitor.side_effect = RuntimeError("monitor boom")

    with pytest.raises(RuntimeError, match="monitor boom"):
        modules.utils.run_command_with_progress(["cmd"])

    try:
        assert mock_proc in modules.utils._active_processes
    finally:
        modules.utils._active_processes.discard(mock_proc)


def test_check_dependencies_success():
    """Test check_dependencies returns True when all found."""
    with patch("modules.utils.subprocess.run", return_value=MagicMock(returncode=0)):
        assert modules.utils.check_dependencies() is True
        assert modules.utils.subprocess.run.call_count == 4
        for call in modules.utils.subprocess.run.call_args_list:
            assert call.kwargs["timeout"] == 10


def test_signal_handler_exits_cleanly_monkeypatch(monkeypatch):
    """Signal handler should cleanup subprocesses and exit with code 1."""
    cleanup_called = {"value": False}

    def _fake_cleanup():
        cleanup_called["value"] = True

    monkeypatch.setattr(modules.utils, "cleanup_subprocesses", _fake_cleanup)

    with pytest.raises(SystemExit) as exc:
        modules.utils.signal_handler(None, None)

    assert cleanup_called["value"] is True
    assert exc.value.code == 1


def test_parse_tqdm_progress_match_and_miss():
    """Exercise both tqdm parsing branches."""
    tqdm_re = re.compile(r"(\d+)%\s*[|:]")
    percent, info = modules.utils._parse_tqdm_progress(" 52%|#####", tqdm_re)
    assert percent == 52.0
    assert info == ""

    percent, info = modules.utils._parse_tqdm_progress("no progress", tqdm_re)
    assert percent is None
    assert info is None


def test_cleanup_subprocesses_exception():
    """Test cleanup handling of strict exceptions."""
    p1 = MagicMock()
    p1.poll.return_value = None
    p1.terminate.side_effect = Exception("Fail")
    p1.kill.side_effect = Exception("Fail")

    with patch("modules.utils._active_processes", {p1}):
        modules.utils.cleanup_subprocesses()
        # Should not raise
        assert p1.terminate.called


def test_adjust_layout_extreme_truncate():
    """Test layout strategy 3: truncation."""
    # Force very narrow columns
    width, info, label = modules.utils._adjust_bar_layout(width=20, info_str="INFO", label="VeryLongLabelThatNeedsTruncation", columns=30)
    assert "..." in label
    assert len(label) < 20


@patch("modules.utils.is_valid_audio", return_value=False)
def test_save_atomic_fail_cleanup(mock_valid, tmp_path, mock_utils_sf):
    """Test atomic save cleans up invalid temp file."""
    f = tmp_path / "test.wav"
    f.with_suffix(".tmp.wav").write_text("temp")
    mock_utils_sf.write.return_value = None
    modules.utils._save_audio_atomic(f, [], 44100)
    # Temp file should be unlinked
    assert not (f.with_suffix(".tmp.wav")).exists()


@patch("modules.utils.draw_progress_bar")
@patch("modules.utils.subprocess.Popen")
def test_monitor_progress_tqdm(mock_popen, mock_draw):
    """Test TQDM progress parsing in run_command_with_progress."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    # Ensure has no _last_pc initially (or set it to -1)
    # properly mocking hasattr is hard, so let's set it to valid float start
    mock_proc._last_pc = -1.0

    mock_proc.poll.side_effect = [None, None, 0]  # Run loop twice then exit

    # TQDM output simulation
    lines = ["45%|xxxx| 10/20 [00:10<00:10, 1.00it/s]", "100%|xxxx| 20/20 [00:20<00:00, 1.00it/s]", ""]
    # Robust iteration for readline
    mock_proc.stdout.readline.side_effect = lines + [""] * 5
    mock_popen.return_value = mock_proc

    modules.utils.run_command_with_progress(["cmd"], description="TQDM Test")

    # Verify 45% was drawn
    # Note: draw_progress_bar args: (percent, label, ...)
    calls = mock_draw.call_args_list
    found_45 = any(c[0][0] == 45.0 for c in calls)
    assert found_45, "Did not find 45% progress update"


@patch("modules.utils.draw_progress_bar")
@patch("modules.utils.subprocess.Popen")
def test_monitor_progress_ffmpeg_time(mock_popen, mock_draw):
    """Test FFmpeg 'time=' parsing."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    # Provide enough poll side_effects
    mock_proc.poll.side_effect = [None, None, 0]

    # FFmpeg time output (15s)
    lines = ["frame=100 time=00:00:15.00 bitrate=...", ""]
    # Robust iteration
    mock_proc.stdout.readline.side_effect = lines + [""] * 5
    mock_popen.return_value = mock_proc

    # Total duration 30s -> 15s should be 50%
    modules.utils.run_command_with_progress(["ffmpeg"], total_duration=30.0, description="FFmpeg Test")

    calls = mock_draw.call_args_list
    # Look for approx 50%
    found_50 = any(abs(c[0][0] - 50.0) < 0.1 for c in calls)
    assert found_50, "Did not find expected ~50% progress update"


def test_log_msg_debug_level(tmp_path, capsys, monkeypatch):
    """Test log_msg with DEBUG level and DEBUG_LOGGING disabled."""
    monkeypatch.setattr("modules.utils.DEBUG_LOGGING", False)
    monkeypatch.setattr("modules.utils.LOG_FILE", str(tmp_path / "test.log"))

    modules.utils.log_msg("debug message", level="DEBUG", console=True)

    # Should not print to console because DEBUG_LOGGING is False
    captured = capsys.readouterr()
    assert "debug message" not in captured.out


def test_log_msg_with_error_flag(tmp_path, capsys, monkeypatch):
    """Test log_msg with is_error=True."""
    log_file = tmp_path / "test.log"
    monkeypatch.setattr("modules.utils.LOG_FILE", str(log_file))

    modules.utils.log_msg("error message", is_error=True)

    # Should print ERROR level
    content = log_file.read_text()
    assert "[ERROR]" in content
    assert "error message" in content


@patch("modules.utils.is_valid_audio")
def test_save_atomic_success(mock_valid, tmp_path, mock_utils_sf):
    """Test save_atomic_audio successful write."""
    # Create actual temp file to simulate successful write
    temp_file = tmp_path / "output.tmp.wav"
    temp_file.write_text("test data")

    # First call to is_valid_audio checks if temp is valid (True)
    # Second call checks if we should proceed
    mock_valid.return_value = True
    mock_utils_sf.write.return_value = None  # sf.write returns None on success

    output_file = tmp_path / "output.wav"

    # Call the function
    result = modules.utils._save_audio_atomic(str(output_file), b"test data", 44100)

    assert result is True
    assert output_file.exists()

    # Mock write was called
    assert mock_utils_sf.write.called


def test_parse_ffmpeg_time_variations():
    """Test parse_ffmpeg_time with various time formats."""
    # Test different valid formats
    assert modules.utils.parse_ffmpeg_time("time=00:00:05.50") == 5.5
    assert modules.utils.parse_ffmpeg_time("time=00:01:00.00") == 60.0
    assert modules.utils.parse_ffmpeg_time("time=01:00:00.00") == 3600.0


def test_format_time_variations():
    """Test format_time with various durations."""
    assert modules.utils.format_time(0) == "00:00:00,000"
    assert modules.utils.format_time(0.5) == "00:00:00,500"
    assert modules.utils.format_time(3661) == "01:01:01,000"


@patch("modules.utils.subprocess.run")
def test_check_dependencies_all_present(mock_run):
    """Test check_dependencies when all are present."""
    mock_run.return_value = MagicMock(returncode=0)
    assert modules.utils.check_dependencies() is True
    assert mock_run.call_count == 4
    for call in mock_run.call_args_list:
        assert call.kwargs["timeout"] == 10


@patch("modules.utils.is_valid_audio")
def test_save_atomic_invalid_output(mock_valid, tmp_path, mock_utils_sf):
    """Test save_atomic_file when output validation fails."""
    mock_valid.return_value = False

    output_file = tmp_path / "output.wav"

    # Should return False when invalid
    result = modules.utils._save_audio_atomic(str(output_file), b"test", 44100)
    assert result is False


def test_save_audio_atomic_returns_false_when_soundfile_missing(tmp_path, monkeypatch):
    dst = tmp_path / "missing_sf.wav"
    monkeypatch.setattr(modules.utils, "sf", None)
    assert modules.utils._save_audio_atomic(dst, b"data", 44100) is False


def test_signal_handler_exits_cleanly():
    """Test signal handler performs cleanup and exits."""
    with patch("modules.utils.cleanup_subprocesses") as mock_cleanup:
        with patch("modules.utils.log_msg") as mock_log:
            with patch("modules.utils.sys.exit", side_effect=SystemExit(1)):
                with pytest.raises(SystemExit):
                    modules.utils.signal_handler(None, None)

    mock_cleanup.assert_called_once()
    mock_log.assert_called_once()


def test_is_valid_video_large_real(tmp_path):
    """Test is_valid_video accepts files above the minimum size threshold."""
    p = tmp_path / "video.mp4"
    p.write_bytes(b"x" * 11000)
    assert modules.utils.is_valid_video(p) is True


def test_terminal_columns_and_parse_tqdm_fallbacks():
    """Test helper fallbacks for terminal size and tqdm parsing."""
    with patch("modules.utils.shutil.get_terminal_size", side_effect=RuntimeError("boom")):
        assert modules.utils._get_terminal_columns(default=77) == 77

    percent, info = modules.utils._parse_tqdm_progress("no percent in this line", re.compile(r"(\\d+)%\\s*[|:]"))
    assert percent is None
    assert info is None


def test_build_progress_info_total_and_speed():
    """Test build progress info includes total and speed sections."""
    info = modules.utils._build_progress_info(percent=50.0, elapsed_sec=10.0, media_sec=30.0, total_duration=60.0)
    assert " / " in info
    assert "x" in info


@patch("modules.utils.sys.stdout")
def test_draw_bar_line_truncates_when_terminal_small(mock_stdout):
    """Test _draw_bar_line truncates content to terminal width."""
    with patch("modules.utils._get_terminal_columns", return_value=20):
        modules.utils._draw_bar_line(width=10, filled_length=5, info_str="info", label="very-long-label")
    assert mock_stdout.write.called


@patch("modules.utils._draw_bar_line")
@patch("modules.utils._get_terminal_columns", return_value=80)
@patch("modules.utils.time.time", side_effect=[1.0, 2.0])
def test_draw_progress_bar_min_width(mock_time, mock_cols, mock_draw):
    """Test draw_progress_bar enforces minimum width of 2."""
    modules.utils._last_bar_time = 0
    modules.utils._last_bar_pc = -1.0
    modules.utils.draw_progress_bar(percent=10.0, width=1)
    called_width, called_filled = mock_draw.call_args[0][0], mock_draw.call_args[0][1]
    assert called_width == 2
    assert called_filled == 0

    modules.utils._last_bar_time = 0
    modules.utils._last_bar_pc = -1.0
    modules.utils.draw_progress_bar(percent=100.0, width=1)
    called_width, called_filled = mock_draw.call_args[0][0], mock_draw.call_args[0][1]
    assert called_width == 2
    assert called_filled == 2


@patch("modules.utils.log_msg")
@patch("modules.utils.subprocess.Popen")
def test_run_command_with_progress_copies_explicit_env(mock_popen, mock_log):
    """run_command_with_progress should merge the provided environment."""
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.wait.return_value = None

    lines = ["first line", "second line", ""]
    mock_proc.stdout.readline.side_effect = lines + [""] * 10
    mock_proc.poll.side_effect = [None, None, 1]
    mock_popen.return_value = mock_proc

    base_env = {"A": "1"}
    with pytest.raises(subprocess.CalledProcessError):
        modules.utils.run_command_with_progress(["badcmd"], env=base_env)

    called_env = mock_popen.call_args.kwargs["env"]
    assert called_env["A"] == "1"
    assert called_env["PYTHONIOENCODING"] == "utf-8"
    assert called_env["PATH"] == os.environ["PATH"]


@patch("modules.utils.log_msg")
@patch("modules.utils.subprocess.Popen")
def test_run_command_with_progress_logs_error_buffer(mock_popen, mock_log):
    """run_command_with_progress should flush buffered output when the command fails."""
    _configure_failed_command_mock(mock_popen)

    with pytest.raises(subprocess.CalledProcessError):
        modules.utils.run_command_with_progress(["badcmd"], env={"A": "1"})

    logged_text = "\n".join(str(call.args[0]) for call in mock_log.call_args_list)
    assert "first line" in logged_text and "second line" in logged_text


def _configure_failed_command_mock(mock_popen):
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.wait.return_value = None

    lines = ["first line", "second line", ""]
    mock_proc.stdout.readline.side_effect = lines + [""] * 10
    mock_proc.poll.side_effect = [None, None, 1]
    mock_popen.return_value = mock_proc


@patch("modules.utils.run_command_with_progress")
@patch("modules.utils.print")
def test_attempt_cpu_run_with_retry_no_plain_print_when_duration(mock_print, mock_run):
    """Test CPU retry path skips plain print when total_duration is provided."""
    mock_run.return_value = None
    ok = modules.utils.attempt_cpu_run_with_retry(
        lambda t: ["cmd", str(t)],
        initial_threads=2,
        total_duration=1.0,
    )
    assert ok is True
    mock_print.assert_not_called()


@patch("modules.utils.log_msg")
@patch("modules.utils.is_valid_audio", return_value=True)
def test_save_audio_atomic_replaces_existing_file(mock_valid, mock_log, tmp_path, mock_utils_sf):
    """Test atomic save removes existing destination before rename."""
    dst = tmp_path / "out.wav"
    dst.write_bytes(b"old")

    def _write_side_effect(path, data, sample_rate, subtype="FLOAT"):
        Path(path).write_bytes(b"new")

    mock_utils_sf.write.side_effect = _write_side_effect
    assert modules.utils._save_audio_atomic(dst, b"data", 44100) is True
    assert dst.read_bytes() == b"new"
    mock_log.assert_not_called()


@patch("modules.utils.log_msg")
def test_save_audio_atomic_exception_cleanup(mock_log, tmp_path, mock_utils_sf):
    """Test atomic save cleans up temp file on write exception."""
    dst = tmp_path / "broken.wav"
    temp = tmp_path / "broken.tmp.wav"
    temp.write_bytes(b"temp")

    mock_utils_sf.write.side_effect = RuntimeError("disk error")
    assert modules.utils._save_audio_atomic(dst, b"data", 44100) is False
    assert not temp.exists()
    assert mock_log.called
