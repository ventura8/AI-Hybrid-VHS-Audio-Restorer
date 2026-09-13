"""Where the project's binaries are looked for: beside the interpreter that runs it, then the project venv."""

import sys
from pathlib import Path

import modules.utils


def test_get_scripts_dirs_puts_the_running_interpreters_directory_first(tmp_path):
    """A checkout without a venv beside it -- a git worktree -- still resolves binaries beside the interpreter."""
    dirs = modules.utils._get_scripts_dirs(tmp_path)
    assert dirs[0] == Path(sys.executable).resolve().parent
    assert len(dirs) == len(set(dirs))


def test_get_scripts_dirs_still_lists_the_project_venv(tmp_path):
    """The project venv's own directories follow, each once, whichever platform layout exists."""
    scripts_dir = tmp_path / ".venv" / "Scripts"
    scripts_dir.mkdir(parents=True)
    dirs = modules.utils._get_scripts_dirs(tmp_path)
    assert scripts_dir in dirs
    assert dirs.index(scripts_dir) > 0
