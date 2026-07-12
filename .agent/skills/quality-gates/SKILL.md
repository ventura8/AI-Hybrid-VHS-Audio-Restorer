# SKILL: Quality Gates

Use this skill when modifying code, tests, scripts, CI, or dependency configuration.

## Objective

Keep the repository in a releasable state by passing all local and CI quality gates.

## Required Commands

```powershell
./run_pipeline_locally.ps1
```

## Validation Sequence

1. PowerShell lint for project scripts.
2. Ruff checks on `modules`, `tests`, `restore_audio_hybrid.py`, `scripts/apply_patches.py`.
3. Flake8 checks on the same target set.
4. Pylint gate on `modules`, `tests`, `restore_audio_hybrid.py`, and `scripts/apply_patches.py`.
5. Pytest with coverage threshold >= 90%.
6. Coverage badge overwrite via `tests/tooling/badge_report.py`.

## Rules

- Max line length is 140.
- Tests must be linted both locally and in CI.
- Do not add inline suppression comments (`noqa`, disable pragmas).
- Keep behavior unchanged unless the task explicitly requires behavior changes.
