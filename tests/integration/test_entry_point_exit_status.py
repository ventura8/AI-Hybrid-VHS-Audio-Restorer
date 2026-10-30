"""The entry point's exit status and prompts: a failed batch exits non-zero, and nothing waits on a stdin that is not a terminal.

Reproduced 2026-10-11 (D:/ai_restore_work/a1_seam/head_run.log): a restoration that failed
("[Error] Processing failed: UVR-DeNoise produced no clean stem ...") still printed "Press
Enter to exit..." and the process exited 0, so a launching script could not tell. The last
test runs the real entry point on a file ffmpeg cannot read, with stdin detached.
"""

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import modules.ui as ui
import restore_audio_hybrid as entry
from modules import utils

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Stdin:
    def __init__(self, tty=False, error=None):
        self.tty, self.error = tty, error

    def isatty(self):
        if self.error:
            raise self.error
        return self.tty

    def fileno(self):
        return 0


@pytest.mark.parametrize(
    ("stdin", "expected"), [(_Stdin(tty=True), True), (_Stdin(), False), (_Stdin(error=ValueError("closed")), False), (None, False)]
)
def test_only_an_open_terminal_is_a_console(monkeypatch, stdin, expected):
    monkeypatch.setattr(ui.sys, "stdin", stdin)
    monkeypatch.setattr(ui, "_windows_console", lambda fd: True)
    assert ui.stdin_is_console() is expected


@pytest.mark.parametrize(("platform", "console", "expected"), [("win32", False, False), ("win32", True, True), ("linux", False, True)])
def test_windows_asks_the_console_itself(monkeypatch, platform, console, expected):
    """On Windows the NUL device of a detached stdin reports a tty; only a handle GetConsoleMode accepts is a console."""
    monkeypatch.setattr(ui.sys, "platform", platform)
    monkeypatch.setattr(ui, "_windows_console", lambda fd: console)
    assert ui._is_terminal(_Stdin(tty=True)) is expected


@pytest.mark.parametrize(("mode_ok", "expected"), [(1, True), (0, False)])
def test_the_console_check_reads_getconsolemode(monkeypatch, mode_ok, expected):
    import ctypes
    import types

    handles = []
    monkeypatch.setitem(sys.modules, "msvcrt", types.SimpleNamespace(get_osfhandle=lambda fd: handles.append(fd) or 77))
    kernel32 = types.SimpleNamespace(GetConsoleMode=lambda handle, mode: mode_ok if handle == 77 else 1)
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=kernel32), raising=False)
    assert ui._windows_console(5) is expected
    assert handles == [5]


@pytest.mark.skipif(sys.platform != "win32", reason="the NUL device quirk is Windows'")
def test_the_nul_device_reports_a_tty_but_is_no_console():
    with open(os.devnull, "r", encoding="utf-8") as nul:
        assert (nul.isatty(), ui._windows_console(nul.fileno())) == (True, False)


def test_without_paths_or_a_console_nothing_is_asked(capsys):
    with (
        patch.object(ui.sys, "argv", ["restore_audio_hybrid.py"]),
        patch.object(ui, "stdin_is_console", return_value=False),
        patch("builtins.input") as ask,
    ):
        assert ui._get_input_files() == ([], False)
    ask.assert_not_called()
    assert "stdin is not a terminal" in capsys.readouterr().out


def test_a_batch_never_waits_for_enter():
    with patch.object(entry, "stdin_is_console", return_value=False), patch("builtins.input") as ask:
        entry._wait_for_console_user()
    ask.assert_not_called()


def test_a_console_user_still_gets_the_prompt():
    with patch.object(entry, "stdin_is_console", return_value=True), patch("builtins.input", return_value="") as ask:
        entry._wait_for_console_user()
    ask.assert_called_once_with("Press Enter to exit...")


def test_a_closed_console_exits_gracefully(capsys):
    with patch.object(entry, "stdin_is_console", return_value=True), patch("builtins.input", side_effect=EOFError):
        entry._wait_for_console_user()
    assert "Exiting gracefully" in capsys.readouterr().out


def _run_batch(results):
    """The restoration with every step but the per-file results mocked; returns (exit status, printed text)."""
    files = [Path(name) for name in results]
    out = io.StringIO()
    with (
        patch.object(entry, "run_init_sequence", return_value=("CPU", "GPU")),
        patch.object(entry, "check_dependencies", return_value=True),
        patch.object(entry, "_show_banner"),
        patch.object(entry, "_get_input_files", return_value=(files, True)),
        patch.object(entry, "process_hybrid_audio", side_effect=lambda path, *_a, **_k: results[path.name]),
        patch.object(entry, "BATCH_JOBS", 1),
        patch.object(entry, "stdin_is_console", return_value=False),
        patch("sys.stdout", out),
    ):
        code = entry._run_restoration()
    return code, out.getvalue()


def test_a_batch_with_a_failed_file_exits_one_and_names_it():
    code, printed = _run_batch({"a.mp4": True, "b.mp4": False, "c.mp4": True})
    assert code == entry.EXIT_TASKS_FAILED
    assert "1 of 3 files failed: b.mp4" in printed
    assert "Press Enter" not in printed


def test_a_clean_batch_exits_zero():
    code, printed = _run_batch({"a.mp4": True, "b.mp4": True})
    assert code == entry.EXIT_OK
    assert "failed" not in printed


def test_the_parallel_batch_returns_its_failures(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with patch.object(entry, "_restore_in_child", side_effect=lambda path, log_dir: 0 if path.name == "a.mp4" else 1):
        assert entry._process_files([Path("a.mp4"), Path("b.mp4")], "GPU", jobs=2) == ["b.mp4"]
    capsys.readouterr()


def test_help_exits_zero(capsys):
    with patch.object(entry.sys, "argv", ["restore_audio_hybrid.py", "--help"]):
        assert entry.main() == entry.EXIT_OK
    assert "Exit status" in capsys.readouterr().out


def _available(binary):
    return shutil.which(binary) is not None or Path(binary).is_file()


@pytest.mark.skipif(not (_available(utils.FFMPEG_BIN) and _available(utils.FFPROBE_BIN)), reason="needs the ffmpeg and ffprobe binaries")
def test_the_real_entry_point_exits_one_on_a_failed_file_without_waiting(tmp_path):
    """vhs_native needs only ffmpeg; the GPU is hidden so the child never opens a CUDA context."""
    (tmp_path / "config.yaml").write_text('process_mode: "vhs_native"\n', encoding="utf-8")
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video" * 200)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "restore_audio_hybrid.py"), str(broken)],
        cwd=str(tmp_path),
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    assert completed.returncode == entry.EXIT_TASKS_FAILED, completed.stdout[-2000:]
    assert "Processing failed" in completed.stdout
    assert "Press Enter" not in completed.stdout
