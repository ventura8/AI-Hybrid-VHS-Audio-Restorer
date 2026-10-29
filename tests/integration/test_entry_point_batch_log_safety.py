"""The batch's per-file logs cannot be redirected by the launch directory, a planted link, or a crafted file name.

The attack this closes (security review, 2026-10-19): a folder of tapes used as the working directory ships
logs/<stem>.log as a symlink to ~/.bashrc or ~/.ssh/authorized_keys and a video whose name embeds a line break and a
payload; each batch child's stdout, which prints the file's name, was written through the link.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import modules.config as config
import modules.utils as utils
import restore_audio_hybrid

PAYLOAD = Path("tape\npython3 -c \"import os;os.system('id')\"\rssh-ed25519 AAAA_ATTACKER.mp4")


def _symlink_or_skip(link, target, directory=False):
    try:
        os.symlink(target, link, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available to this user")


@pytest.fixture(name="app_logs")
def app_logs_fixture(tmp_path, monkeypatch):
    """The batch log folder moved into a temporary application directory; the launch directory elsewhere."""
    (tmp_path / "app").mkdir()
    logs = tmp_path / "app" / "logs"
    monkeypatch.setattr(restore_audio_hybrid, "BATCH_LOG_DIR", logs)
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    return logs


def _run_child(video, log_dir, code=0):
    with patch("restore_audio_hybrid.subprocess.run", return_value=MagicMock(returncode=code)) as run:
        return restore_audio_hybrid._restore_in_child(video, log_dir), run


def test_the_batch_logs_live_beside_the_application():
    assert restore_audio_hybrid.BATCH_LOG_DIR == config.APP_DIR / "logs"
    assert restore_audio_hybrid.BATCH_LOG_DIR.is_absolute()


def test_a_planted_launch_directory_link_is_never_written(app_logs, tmp_path):
    target = tmp_path / "bashrc"
    target.write_text("original\n", encoding="utf-8")
    (tmp_path / "cwd" / "logs").mkdir()
    _symlink_or_skip(tmp_path / "cwd" / "logs" / "a.log", target)
    with patch("restore_audio_hybrid.subprocess.run", return_value=MagicMock(returncode=0)):
        restore_audio_hybrid._run_parallel_batch([Path("a.mp4"), Path("b.mp4")], 2)
    assert target.read_text(encoding="utf-8") == "original\n"
    assert ((app_logs / "a.log").is_file(), (app_logs / "a.log").is_symlink()) == (True, False)


def test_a_symlinked_per_file_log_is_refused_and_its_target_untouched(app_logs, tmp_path):
    app_logs.mkdir()
    target = tmp_path / "authorized_keys"
    target.write_text("ssh-ed25519 OWNER\n", encoding="utf-8")
    _symlink_or_skip(app_logs / "a.log", target)
    with pytest.raises(OSError):
        _run_child(Path("a.mp4"), app_logs)
    assert target.read_text(encoding="utf-8") == "ssh-ed25519 OWNER\n"


def test_a_symlinked_log_folder_stops_the_batch_before_anything_runs(app_logs, tmp_path, capsys):
    (tmp_path / "elsewhere").mkdir()
    _symlink_or_skip(app_logs, tmp_path / "elsewhere", directory=True)
    with patch("restore_audio_hybrid._restore_in_child") as child:
        failed = restore_audio_hybrid._run_parallel_batch([Path("a.mp4"), Path("b.mp4")], 2)
    child.assert_not_called()
    assert failed == ["a.mp4", "b.mp4"]
    assert "refusing to follow a symlinked batch log folder" in capsys.readouterr().out


def test_a_line_break_in_a_file_name_never_reaches_the_log_name(app_logs):
    app_logs.mkdir()
    code, _run = _run_child(PAYLOAD, app_logs, code=3)
    (written,) = list(app_logs.iterdir())
    assert (code, "\n" in written.name, "\r" in written.name) == (3, False, False)


def test_a_line_break_in_a_file_name_prints_no_line_of_its_own(app_logs, capsys):
    app_logs.mkdir()
    _run_child(PAYLOAD, app_logs, code=3)
    restore_audio_hybrid._report_failures([PAYLOAD.name], 2)
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert [line.split()[0] for line in lines] == ["[Batch]", "[Batch]", "[System]"]


def test_a_reused_log_is_truncated_not_appended(app_logs):
    app_logs.mkdir()
    (app_logs / "a.log").write_text("an earlier run\n", encoding="utf-8")
    _run_child(Path("a.mp4"), app_logs)
    assert (app_logs / "a.log").read_text(encoding="utf-8") == ""


def test_the_console_keeps_a_leading_blank_line_and_collapses_the_rest():
    assert utils._console_text("\n[System] Processing Task: tape\nid\rkeys") == "\n   [System] Processing Task: tape id keys"
    assert utils._console_text("    [Stage] already indented") == "    [Stage] already indented"


def test_a_crafted_name_logged_by_a_child_prints_as_one_line(capsys):
    utils.log_msg(f"\n[System] Processing Task: {PAYLOAD.name}", console=True)
    printed = [line for line in capsys.readouterr().out.replace("\r\033[K", "").splitlines() if line.strip()]
    assert (len(printed), "ssh-ed25519 AAAA_ATTACKER" in printed[0]) == (1, True)


def test_a_log_name_holds_no_character_a_folder_or_windows_would_take():
    class Crafted:
        stem = 'a<b>:c"d/e\\f|g?h*i\nj'

    name = restore_audio_hybrid.batch_log_path(Path("logs"), Crafted()).name
    assert name == "a_b__c_d_e_f_g_h_i j.log"
