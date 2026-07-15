# Tests Layout

- tests/unit: fast unit-level tests for modules and helpers.
- tests/integration: broader flow and entry-point behavior tests.
- tests/tooling: validation/reporting helpers used by local pipeline and CI.

Tooling scripts:

- tests/tooling/quality_gate.py: strict per-file coverage gate.
- tests/tooling/badge_report.py: badge generation and summary transform.
- tests/tooling/threshold_policy.py: shared threshold policy and env override.
- tests/tooling/radon_cc_gate.py: strict Radon CC rank gate.
- tests/tooling/radon_mi_gate.py: strict Radon MI rank gate.
