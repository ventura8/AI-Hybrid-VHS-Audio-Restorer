"""The parallel batch of the entry point: files `batch_jobs` at a time, each in a child interpreter."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import restore_audio_hybrid


def test_process_files_runs_sequentially_when_jobs_is_one():
    files = [Path("a.mp4"), Path("b.mp4")]
    with patch("restore_audio_hybrid.process_hybrid_audio") as process, patch("restore_audio_hybrid._run_parallel_batch") as parallel:
        restore_audio_hybrid._process_files(files, "GPU", jobs=1)
    assert process.call_count == 2
    parallel.assert_not_called()


def test_process_files_uses_children_for_several_files_and_several_jobs():
    files = [Path("a.mp4"), Path("b.mp4")]
    with patch("restore_audio_hybrid.process_hybrid_audio") as process, patch("restore_audio_hybrid._run_parallel_batch") as parallel:
        restore_audio_hybrid._process_files(files, "GPU", jobs=3)
        restore_audio_hybrid._process_files(files[:1], "GPU", jobs=3)
    parallel.assert_called_once_with(files, 3)
    assert process.call_count == 1


def test_run_parallel_batch_reports_the_failed_files(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")]
    codes = {"a.mp4": 0, "b.mp4": 1, "c.mp4": 0}
    with patch("restore_audio_hybrid._restore_in_child", side_effect=lambda path, log_dir: codes[path.name]):
        restore_audio_hybrid._run_parallel_batch(files, 2)
    out = capsys.readouterr().out
    assert "3 files, 2 at a time" in out
    assert "1 of 3 files failed: b.mp4" in out
    assert (tmp_path / "logs").is_dir()


def _restore_one_in_child(tmp_path):
    completed = MagicMock(returncode=0)
    with patch("restore_audio_hybrid.subprocess.run", return_value=completed) as run:
        code = restore_audio_hybrid._restore_in_child(Path("tape one.mov"), tmp_path)
    return code, run.call_args


def test_restore_in_child_runs_this_entry_point_on_one_file(tmp_path):
    code, call = _restore_one_in_child(tmp_path)
    cmd = call[0][0]
    assert code == 0
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("restore_audio_hybrid.py")
    assert cmd[2] == "tape one.mov"


def test_restore_in_child_detaches_stdin_and_keeps_a_log(tmp_path, capsys):
    _code, call = _restore_one_in_child(tmp_path)
    assert call.kwargs["stdin"] is restore_audio_hybrid.subprocess.DEVNULL
    assert (tmp_path / "tape one.log").exists()
    assert "done" in capsys.readouterr().out


def test_restore_in_child_reports_a_failure_code(tmp_path, capsys):
    with patch("restore_audio_hybrid.subprocess.run", return_value=MagicMock(returncode=3)):
        assert restore_audio_hybrid._restore_in_child(Path("x.mp4"), tmp_path) == 3
    assert "FAILED (3)" in capsys.readouterr().out
