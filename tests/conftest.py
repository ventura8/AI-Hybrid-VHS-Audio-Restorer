import os

from tests.tooling.quality_gate import main as quality_gate_main
from tests.tooling.threshold_policy import get_coverage_threshold


def pytest_sessionfinish(session, exitstatus):
    """
    Hook to run after the entire test session is finished.
    Checks per-file coverage minimums and updates the coverage badge.
    """
    xml_file = "coverage.xml"

    # Update badge first
    if os.path.exists(xml_file):
        print("\nUpdating coverage badge...")
        try:
            # Import here to avoid E402 and mypy attribute errors
            from tests.tooling.badge_report import transform_coverage

            transform_coverage(xml_file)
            print("Coverage badge updated successfully.")
        except Exception as e:
            print(f"Failed to update coverage badge: {e}")
    else:
        print("\nNo coverage.xml found, skipping badge update.")

    # Check per-file coverage using the same combined metric as the CLI quality gate.
    coverage_json = "coverage.json"
    if os.path.exists(coverage_json):
        try:
            min_coverage = get_coverage_threshold()
            exit_code = quality_gate_main([coverage_json, "--threshold", f"{min_coverage:.2f}"])
            if exit_code != 0:
                session.exitstatus = 1
        except Exception as e:
            print(f"Warning: Could not verify per-file coverage: {e}")
            session.exitstatus = 1
    else:
        print("No coverage.json found, skipping per-file coverage verification.")
