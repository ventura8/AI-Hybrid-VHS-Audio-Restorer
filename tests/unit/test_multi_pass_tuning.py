"""Tests for duration-window selection in the multi-pass tuning runner."""

import pytest

from scripts import run_multi_pass_tuning


def test_probe_duration_reads_ffprobe_container_metadata(monkeypatch, tmp_path):
    """Use metadata probing instead of decoding a tape to obtain its duration."""
    tape = tmp_path / "medium.mp4"
    expected = [
        run_multi_pass_tuning.FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(tape),
    ]
    captured = []

    def check_output(command, text, timeout=None):
        captured.extend((command, text, timeout))
        return "3720.0\n"

    monkeypatch.setattr(run_multi_pass_tuning.subprocess, "check_output", check_output)

    assert run_multi_pass_tuning._probe_duration(tape) == 62.0
    assert captured == [expected, True, 10]


def test_collect_eligible_tapes_includes_window_boundaries(tmp_path, monkeypatch):
    """Keep tapes exactly at either inclusive duration bound."""
    first = tmp_path / "first.mp4"
    middle = tmp_path / "middle.mkv"
    last = tmp_path / "last.avi"
    outside = tmp_path / "outside.mpg"
    for tape in (first, middle, last, outside):
        tape.touch()
    durations = {first: 60.0, middle: 88.2, last: 120.0, outside: 120.1}
    monkeypatch.setattr(run_multi_pass_tuning, "_probe_duration", durations.get)

    tapes = run_multi_pass_tuning._collect_eligible_tapes(tmp_path, 60.0, 120.0)

    assert tapes == [(first, 60.0), (last, 120.0), (middle, 88.2)]


def test_collect_eligible_tapes_excludes_unknown_duration(tmp_path, monkeypatch):
    """Skip videos when probing cannot establish a duration."""
    tape = tmp_path / "unknown.mp4"
    tape.touch()
    monkeypatch.setattr(run_multi_pass_tuning, "_probe_duration", lambda _: None)

    assert run_multi_pass_tuning._collect_eligible_tapes(tmp_path, 60.0, 120.0) == []


def test_parse_args_rejects_an_invalid_duration_window(monkeypatch):
    """Require non-negative, ascending duration bounds."""
    monkeypatch.setattr(
        "sys.argv",
        ["run_multi_pass_tuning.py", "--input-dir", "tapes", "--min-duration", "120", "--max-duration", "60"],
    )

    with pytest.raises(SystemExit) as error:
        run_multi_pass_tuning._parse_args()
    assert error.value.code == 2


def test_parse_args_uses_a_zero_to_sixty_minute_window_by_default(monkeypatch):
    """Keep short-tape tuning as the default selection window."""
    monkeypatch.setattr("sys.argv", ["run_multi_pass_tuning.py", "--input-dir", "tapes"])

    args = run_multi_pass_tuning._parse_args()

    assert args.min_duration == 0.0
    assert args.max_duration == 60.0
