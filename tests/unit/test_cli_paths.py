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


def test_numbers_are_rebuilt_and_range_checked():
    assert cli_paths.checked_number("30", "--start") == 30.0
    assert cli_paths.checked_number(3, "--limit", 1, kind=int) == 3
    with pytest.raises(SystemExit, match="not a number"):
        cli_paths.checked_number("-ss", "--start")
    with pytest.raises(SystemExit, match="must lie between"):
        cli_paths.checked_number(0, "--limit", 1, kind=int)


def test_tokens_and_urls_refuse_option_shaped_values():
    assert cli_paths.checked_token("bandlimited_m15", "--variants") == "bandlimited_m15"
    assert cli_paths.checked_url("https://archive.org/x.mp4", "url") == "https://archive.org/x.mp4"
    with pytest.raises(SystemExit):
        cli_paths.checked_token("-rf", "--variants")
    with pytest.raises(SystemExit):
        cli_paths.checked_url("-i /etc/passwd", "url")
