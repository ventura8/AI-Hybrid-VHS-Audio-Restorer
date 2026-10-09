"""The session hooks in tests/conftest.py read only the coverage reports their own session wrote."""

import os
from types import SimpleNamespace

import pytest

import tests.conftest as session_hooks
from tests.tooling import badge_report

OLD_NS = 1_000_000_000_000_000_000
NEW_NS = 1_000_000_001_000_000_000


def _session(has_cov=True, no_cov=False):
    """A pytest session stand-in: pytest-cov registered or not, --no-cov on or off."""
    options = {"no_cov": no_cov}
    config = SimpleNamespace(
        pluginmanager=SimpleNamespace(hasplugin=lambda name: has_cov and name == "_cov"),
        stash=pytest.Stash(),
        getoption=lambda name, default=None: options.get(name, default),
    )
    return SimpleNamespace(config=config, exitstatus=0)


def _write_report(name, mtime_ns):
    with open(name, "w", encoding="utf-8") as handle:
        handle.write("<coverage/>")
    os.utime(name, ns=(mtime_ns, mtime_ns))


@pytest.fixture(name="calls")
def fixture_calls(monkeypatch, tmp_path):
    """Run each test in an empty directory and record the badge and per-file gate calls instead of running them."""
    monkeypatch.chdir(tmp_path)
    recorded = {"badge": [], "gate": [], "gate_exit": 0}

    def fake_gate(argv):
        """Record the per-file gate's arguments and return the exit code the test chose."""
        recorded["gate"].append(argv)
        return recorded["gate_exit"]

    monkeypatch.setattr(badge_report, "transform_coverage", recorded["badge"].append)
    monkeypatch.setattr(session_hooks, "quality_gate_main", fake_gate)
    monkeypatch.setattr(session_hooks, "get_coverage_threshold", lambda: 90.0)
    return recorded


def _run_session(session, written_ns=NEW_NS):
    session_hooks.pytest_sessionstart(session)
    for name in session_hooks.COVERAGE_REPORTS:
        _write_report(name, written_ns)
    session_hooks.pytest_sessionfinish(session, 0)


def test_reports_written_by_a_measured_session_update_the_badge_and_gate_per_file(calls):
    """A session measured by pytest-cov that writes both reports runs the badge and the per-file gate on them."""
    session = _session()
    _run_session(session)
    assert calls["badge"] == ["coverage.xml"]
    assert calls["gate"] == [["coverage.json", "--threshold", "90.00"]]
    assert session.exitstatus == 0


def test_rewritten_reports_count_as_this_sessions(calls):
    """Reports that existed before the session and were rewritten during it are this session's."""
    for name in session_hooks.COVERAGE_REPORTS:
        _write_report(name, OLD_NS)
    _run_session(_session(), NEW_NS)
    assert (len(calls["badge"]), len(calls["gate"])) == (1, 1)


@pytest.mark.parametrize("has_cov, no_cov", [(True, True), (False, False)], ids=["no-cov", "no-cov-plugin"])
def test_a_session_without_coverage_leaves_badge_and_gate_alone(calls, capsys, has_cov, no_cov):
    """--no-cov, or no pytest-cov plugin, returns early even when coverage files appear in the directory."""
    session = _session(has_cov, no_cov)
    _run_session(session)
    assert (calls["badge"], calls["gate"], session.exitstatus) == ([], [], 0)
    assert "Coverage was not collected in this session" in capsys.readouterr().out


def test_stale_reports_from_an_earlier_run_are_not_read(calls, capsys):
    """A measured session that writes no report leaves the stale coverage.xml and coverage.json unread."""
    for name in session_hooks.COVERAGE_REPORTS:
        _write_report(name, OLD_NS)
    session = _session()
    session_hooks.pytest_sessionstart(session)
    session_hooks.pytest_sessionfinish(session, 0)
    out = capsys.readouterr().out
    assert (calls["badge"], calls["gate"]) == ([], [])
    assert "No coverage.xml written by this session" in out and "No coverage.json written by this session" in out


def test_missing_reports_and_a_session_never_started_are_skipped(calls):
    """No report at all, or a finish hook without the start snapshot, reads nothing."""
    session = _session()
    session_hooks.pytest_sessionstart(session)
    session_hooks.pytest_sessionfinish(session, 0)
    unstarted = _session()
    for name in session_hooks.COVERAGE_REPORTS:
        _write_report(name, NEW_NS)
    session_hooks.pytest_sessionfinish(unstarted, 0)
    assert (calls["badge"], calls["gate"]) == ([], [])


def test_a_failing_per_file_gate_fails_the_session(calls):
    """A per-file coverage failure sets the session's exit status to 1."""
    calls["gate_exit"] = 1
    session = _session()
    _run_session(session)
    assert session.exitstatus == 1


def test_badge_and_gate_errors_are_reported(calls, monkeypatch, capsys):
    """A badge error is printed and the gate still runs; a gate error is printed and fails the session."""

    def broken(*_args):
        """Fail the way a broken badge or gate tool would."""
        raise RuntimeError("boom")

    monkeypatch.setattr(badge_report, "transform_coverage", broken)
    monkeypatch.setattr(session_hooks, "quality_gate_main", broken)
    session = _session()
    _run_session(session)
    out = capsys.readouterr().out
    assert "Failed to update coverage badge: boom" in out
    assert "Warning: Could not verify per-file coverage: boom" in out
    assert (calls["badge"], session.exitstatus) == ([], 1)


def test_the_badge_is_reported_updated(calls, capsys):
    """A badge rewrite says so."""
    _run_session(_session())
    assert calls["badge"] == ["coverage.xml"]
    assert "Coverage badge updated successfully." in capsys.readouterr().out
