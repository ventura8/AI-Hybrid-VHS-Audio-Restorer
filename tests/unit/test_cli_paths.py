"""scripts/cli_paths.py: command-line paths refused when shaped like options or outside the allowed roots."""

import pytest

from scripts import cli_paths


def test_a_path_inside_the_repository_is_accepted_and_resolved():
    assert cli_paths.confined_path("scripts", "--x", must_exist=True) == cli_paths.REPO_ROOT / "scripts"


def test_an_option_shaped_or_missing_path_is_refused():
    with pytest.raises(SystemExit, match="looks like an option"):
        cli_paths.confined_path("-rf", "--x")
    with pytest.raises(SystemExit, match="does not exist"):
        cli_paths.confined_path("no_such_dir_here", "--x", must_exist=True)


def test_a_path_outside_every_root_is_refused_until_its_root_is_listed(tmp_path, monkeypatch):
    outside = cli_paths.REPO_ROOT.anchor + "definitely_not_allowed_root_xyz"
    monkeypatch.delenv(cli_paths.DATA_ROOTS_ENV, raising=False)
    with pytest.raises(SystemExit, match="must lie inside"):
        cli_paths.confined_path(outside, "--x")
    monkeypatch.setenv(cli_paths.DATA_ROOTS_ENV, outside)
    assert cli_paths.confined_path(outside, "--x").name == "definitely_not_allowed_root_xyz"


def test_the_temp_directory_is_always_allowed(tmp_path, monkeypatch):
    monkeypatch.delenv(cli_paths.DATA_ROOTS_ENV, raising=False)
    assert cli_paths.confined_path(tmp_path, "--x", must_exist=True) == tmp_path.resolve()
