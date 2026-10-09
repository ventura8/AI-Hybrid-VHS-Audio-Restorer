"""Session hooks: the coverage badge and the per-file coverage gate, from the reports this session wrote."""

import os

import pytest

from tests.tooling.quality_gate import main as quality_gate_main
from tests.tooling.threshold_policy import get_coverage_threshold

COVERAGE_XML = "coverage.xml"
COVERAGE_JSON = "coverage.json"
COVERAGE_REPORTS = (COVERAGE_XML, COVERAGE_JSON)

# The modification time (ns) of each coverage report when the session started,
# None for a report that did not exist yet.
REPORT_MTIMES_KEY = pytest.StashKey[dict]()


def _report_mtime_ns(path):
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _coverage_collected(config):
    """True when pytest-cov measured this session: its plugin is registered and --no-cov is off."""
    if not config.pluginmanager.hasplugin("_cov"):
        return False
    return not config.getoption("no_cov", default=False)


def _written_this_session(config, path):
    """True when ``path`` exists and was written after the session started (a report this run produced)."""
    before = config.stash.get(REPORT_MTIMES_KEY, {})
    now = _report_mtime_ns(path)
    return path in before and now is not None and now != before[path]


def _update_coverage_badge(config, xml_file):
    if not _written_this_session(config, xml_file):
        print(f"\nNo {xml_file} written by this session, skipping badge update.")
        return

    print("\nUpdating coverage badge...")
    try:
        from tests.tooling.badge_report import transform_coverage

        transform_coverage(xml_file)
        print("Coverage badge updated successfully.")
    except Exception as exc:
        print(f"Failed to update coverage badge: {exc}")


def _enforce_per_file_coverage(session, coverage_json):
    if not _written_this_session(session.config, coverage_json):
        print(f"No {coverage_json} written by this session, skipping per-file coverage verification.")
        return

    try:
        min_coverage = get_coverage_threshold()
        exit_code = quality_gate_main([coverage_json, "--threshold", f"{min_coverage:.2f}"])
        if exit_code != 0:
            session.exitstatus = 1
    except Exception as exc:
        print(f"Warning: Could not verify per-file coverage: {exc}")
        session.exitstatus = 1


@pytest.fixture(autouse=True)
def _stage_cache_off(monkeypatch):
    """A developer's AI_RESTORE_STAGE_CACHE never reaches a test, so no test writes into a real cache.

    Set first so the original state is recorded: whatever a test then sets is undone after it.
    """
    for name in ("AI_RESTORE_STAGE_CACHE", "AI_RESTORE_STAGE_CACHE_MAX_GB"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def pytest_sessionstart(session):
    """Record the coverage reports' modification times, so the finish hook reads only reports this run writes."""
    session.config.stash[REPORT_MTIMES_KEY] = {path: _report_mtime_ns(path) for path in COVERAGE_REPORTS}


def pytest_sessionfinish(session, exitstatus):
    """
    Hook to run after the entire test session is finished.
    Checks per-file coverage minimums and updates the coverage badge, from the
    coverage reports this session wrote only: a --no-cov run, a run without
    pytest-cov, and a stale coverage.xml or coverage.json left by an earlier
    run leave the badge and the per-file gate alone.
    """
    del exitstatus
    if not _coverage_collected(session.config):
        print("\nCoverage was not collected in this session, skipping badge update and per-file coverage verification.")
        return
    _update_coverage_badge(session.config, COVERAGE_XML)
    _enforce_per_file_coverage(session, COVERAGE_JSON)
