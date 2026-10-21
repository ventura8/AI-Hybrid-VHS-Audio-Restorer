"""A failed render is set aside with its reason, a full disk stops the loop before it renders (2026-10-20)."""

from collections import namedtuple

import pytest

from scripts import autotune_renders as renders

Usage = namedtuple("Usage", "total used free")

RUN_LOG = """\
\x1b[K   [System] Processing Task: alpha.mov
\x1b[K   [Error] Failed to save audio D:\\work\\aligned_long.wav: Error opening 'aligned_long.tmp.wav': System error.
\x1b[K     [Error] Processing failed: Missing Processed Audio input for mux
   [System] Processing Task: beta.mov
\x1b[K     [System] Success! Saved to: beta_PureLinear_Cleaned.mov
   [System] Processing Task: gamma.mov
\x1b[K   [Warning] something odd
"""


def test_render_failed_names_every_tape_and_its_reason():
    error = renders.RenderFailed("abc123", {"t1": "disk full", "t2": "no output"})
    assert (isinstance(error, SystemExit), error.reasons) == (False, {"t1": "disk full", "t2": "no output"})
    assert str(error) == "abc123: no output for t1: disk full; t2: no output"


def test_the_reason_is_the_first_error_logged_for_that_tape(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(RUN_LOG, encoding="utf-8")
    reason = renders.failure_reason(log, tmp_path / "tapes" / "alpha.mov")
    assert (reason.startswith("[Error] Failed to save audio"), "System error" in reason) == (True, True)


@pytest.mark.parametrize(
    "tape, expected",
    [
        ("beta.mov", "no output (the run log names no error)"),
        ("gamma.mov", "no output (the run log names no error)"),
        ("delta.mov", "no output (the run log names no error)"),
    ],
)
def test_a_tape_whose_block_holds_no_error_gets_a_plain_note(tmp_path, tape, expected):
    log = tmp_path / "run.log"
    log.write_text(RUN_LOG, encoding="utf-8")
    assert renders.failure_reason(log, tape) == expected


def test_without_a_run_log_the_reason_says_so(tmp_path):
    assert renders.failure_reason(tmp_path / "missing.log", "alpha.mov") == "no output (no run log)"


def test_a_long_error_line_is_cut(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("Processing Task: alpha.mov\n[Error] " + "x" * 1000 + "\n", encoding="utf-8")
    assert len(renders.failure_reason(log, "alpha.mov")) == renders.REASON_CHARS


def test_free_space_is_read_on_the_nearest_existing_folder(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(renders.shutil, "disk_usage", lambda path: seen.append(path) or Usage(0, 0, 50 * renders.GB))
    assert renders.free_gb(tmp_path / "not" / "yet") == 50.0
    assert seen == [tmp_path]


def test_a_drive_under_the_floor_stops_the_loop_before_the_render(tmp_path, monkeypatch):
    monkeypatch.setattr(renders.shutil, "disk_usage", lambda _path: Usage(0, 0, 12 * renders.GB))
    with pytest.raises(SystemExit, match="abc123: 12.0 GB free .* under the 40 GB a render needs"):
        renders.check_free_space("abc123", [tmp_path])


def test_enough_space_lets_the_render_go(tmp_path, monkeypatch):
    monkeypatch.setattr(renders.shutil, "disk_usage", lambda _path: Usage(0, 0, 41 * renders.GB))
    assert renders.check_free_space("abc123", [tmp_path, tmp_path]) is None
    assert renders.check_free_space("abc123", [tmp_path], min_gb=0.0) is None


def test_failures_in_a_row_stop_the_loop_and_a_pass_resets_the_count():
    streak = renders.FailureStreak(limit=2)
    streak.failed("one")
    streak.passed()
    streak.failed("two")
    with pytest.raises(SystemExit, match="2 renders in a row failed.*two \\| three"):
        streak.failed("three")
