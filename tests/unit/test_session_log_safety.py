"""The session log cannot be redirected by the launch directory or split by a crafted message."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

import modules.config as config
import modules.utils as utils


def test_the_log_lives_beside_the_application_not_in_the_launch_directory():
    assert config.LOG_FILE.is_absolute()
    assert config.LOG_FILE.parent == Path(config.__file__).resolve().parent.parent


def test_a_newline_in_a_message_cannot_add_a_line(tmp_path):
    log = tmp_path / "session_log.txt"
    with patch("modules.utils.LOG_FILE", log):
        utils._append_log_file("INFO", "Processing: tape\npython3 -c 'import os'\rssh-ed25519 AAAA.mp4")
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "tape python3 -c 'import os' ssh-ed25519 AAAA.mp4" in lines[0]


def _symlink_or_skip(link, target):
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available to this user")


def test_a_symlinked_log_is_refused_and_its_target_untouched(tmp_path):
    target = tmp_path / "bashrc"
    target.write_text("original\n", encoding="utf-8")
    log = tmp_path / "session_log.txt"
    _symlink_or_skip(log, target)
    with patch("modules.utils.LOG_FILE", log):
        utils.log_msg("Processing: x\necho owned", console=False)
        with pytest.raises(OSError):
            utils._append_log_file("INFO", "direct")
        assert utils._rotate_log_file(log, limit=0) is False
    assert target.read_text(encoding="utf-8") == "original\n"


def test_one_line_keeps_tabs_and_ordinary_text():
    assert utils._one_line("a\tb c") == "a\tb c"
    assert utils._one_line("a\u2028b\x85c\x1bd") == "a b c d"
