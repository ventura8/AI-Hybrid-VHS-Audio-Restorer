import os

from tests.tooling.quality_gate import main as quality_gate_main
from tests.tooling.threshold_policy import get_coverage_threshold


def _update_coverage_badge(xml_file):
    if not os.path.exists(xml_file):
        print("\nNo coverage.xml found, skipping badge update.")
        return

    print("\nUpdating coverage badge...")
    try:
        from tests.tooling.badge_report import transform_coverage

        transform_coverage(xml_file)
        print("Coverage badge updated successfully.")
    except Exception as exc:
        print(f"Failed to update coverage badge: {exc}")


def _enforce_per_file_coverage(session, coverage_json):
    if not os.path.exists(coverage_json):
        print("No coverage.json found, skipping per-file coverage verification.")
        return

    try:
        min_coverage = get_coverage_threshold()
        exit_code = quality_gate_main([coverage_json, "--threshold", f"{min_coverage:.2f}"])
        if exit_code != 0:
            session.exitstatus = 1
    except Exception as exc:
        print(f"Warning: Could not verify per-file coverage: {exc}")
        session.exitstatus = 1


def pytest_sessionfinish(session, exitstatus):
    """
    Hook to run after the entire test session is finished.
    Checks per-file coverage minimums and updates the coverage badge.
    """
    del exitstatus
    _update_coverage_badge("coverage.xml")
    _enforce_per_file_coverage(session, "coverage.json")
