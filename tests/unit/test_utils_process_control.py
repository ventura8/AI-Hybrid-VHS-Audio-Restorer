"""Tests for run_command_with_progress bookkeeping: redraw throttling and child cleanup.

These live apart from test_utils.py because that file sits at a maintainability index of
19.19 against a gate that requires above 19. Adding to it directly pushed it to grade B.
"""

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import modules.utils


def test_progress_is_throttled_between_meaningful_steps():
    """Progress redraws only when the number moved, or at the two ends.

    Without the throttle a long encode redraws on every ffmpeg status line, which is a
    steady stream of terminal writes for a bar that has not changed.
    """
    fresh = SimpleNamespace()
    assert modules.utils._should_emit_tqdm_progress(fresh, 0) is True

    seen = SimpleNamespace(_last_pc=40.0)
    assert modules.utils._should_emit_tqdm_progress(seen, 40.05) is False
    assert modules.utils._should_emit_tqdm_progress(seen, 40.2) is True


def test_progress_always_redraws_the_final_frame():
    """100% must be drawn even when it is a hair above the last redraw.

    Suppressing it would leave a finished job showing 99.9% for the rest of the session.
    """
    nearly_done = SimpleNamespace(_last_pc=99.95)
    assert modules.utils._should_emit_tqdm_progress(nearly_done, 99.99) is False
    assert modules.utils._should_emit_tqdm_progress(nearly_done, 100) is True


def test_cleanup_reports_success_for_an_already_finished_child():
    """A process that exited on its own needs no termination.

    The monitor calls this after a failure, when the child has often already died, and
    terminating a finished process would raise on some platforms.
    """
    finished = MagicMock()
    finished.poll.return_value = 0
    assert modules.utils._cleanup_after_monitor_error(finished) is True
    finished.terminate.assert_not_called()


def test_cleanup_kills_a_child_that_ignores_termination():
    """A child that will not exit politely is killed rather than left running."""
    stubborn = MagicMock()
    stubborn.poll.side_effect = [None, 0]
    stubborn.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), None]
    assert modules.utils._cleanup_after_monitor_error(stubborn) is True
    stubborn.kill.assert_called_once()


def test_cleanup_reports_failure_when_the_child_cannot_be_stopped():
    """A cleanup that itself fails is reported, not swallowed as success."""
    unkillable = MagicMock()
    unkillable.poll.return_value = None
    unkillable.terminate.side_effect = OSError("no such process")
    assert modules.utils._cleanup_after_monitor_error(unkillable) is False
