"""Where the app finds its binaries: the interpreter's own directory first, then ~/.cargo/bin, then an environment override."""

from pathlib import Path
from unittest.mock import patch

import modules.utils


def test_get_scripts_dirs_posix_and_windows(tmp_path):
    """Test _get_scripts_dirs finds existing bin and Scripts directories."""
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    scripts_dir = tmp_path / ".venv" / "Scripts"
    scripts_dir.mkdir(parents=True)
    dirs = modules.utils._get_scripts_dirs(tmp_path)
    assert bin_dir in dirs
    assert scripts_dir in dirs


def test_cathar_binary_honours_the_environment_override_only_when_set():
    assert modules.utils._cathar_binary({"AI_RESTORE_CATHAR_BIN": " D:/builds/cathar.exe "}) == "D:/builds/cathar.exe"
    assert modules.utils._cathar_binary({"AI_RESTORE_CATHAR_BIN": ""}) == modules.utils._cathar_binary({})
    assert Path(modules.utils._cathar_binary({})).stem == "cathar"


def test_resolve_binary_with_extension(tmp_path):
    """Test _resolve_binary finds binary with or without extension."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ffmpeg_file = bin_dir / "ffmpeg"
    ffmpeg_file.write_text("dummy", encoding="utf-8")

    with patch("sys.platform", "linux"):
        resolved = modules.utils._resolve_binary("ffmpeg", [bin_dir])
        assert resolved == str(ffmpeg_file)

    # On win32 the .exe variant is preferred over the extensionless executable.
    ffmpeg_exe = bin_dir / "ffmpeg.exe"
    ffmpeg_exe.write_text("dummy", encoding="utf-8")

    with patch("sys.platform", "win32"):
        resolved_win = modules.utils._resolve_binary("ffmpeg", [bin_dir])
        assert resolved_win == str(ffmpeg_exe)
