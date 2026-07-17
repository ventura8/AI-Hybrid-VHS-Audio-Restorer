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
1. Black format check on Python targets.
1. isort import-order check on Python targets.
1. Ruff checks on `modules`, `tests`, `restore_audio_hybrid.py`, `scripts/apply_patches.py`.
1. Flake8 checks on the same target set.
1. Taplo TOML formatting check on tracked `.toml` files.
1. Pylint gate on `modules`, `tests`, `restore_audio_hybrid.py`, and `scripts/apply_patches.py`.
1. Bandit security gate (`-ll -ii`) on Python production/entry scripts.
1. pip-audit vulnerability scan.
1. Radon CC/MI/RAW/Halstead reports and CC/MI pass gates.
1. Markdown auto-delint via `mdformat` on docs and agent guidance files.
1. Markdown lint via `pymarkdown scan` on the same Markdown targets.
1. Pytest with coverage threshold >= 90%.
1. Strict per-file coverage gate from `coverage.json`.
1. Coverage badge overwrite via `tests/tooling/badge_report.py`.

## Rules

- Max line length is 140.
- Tests must be linted both locally and in CI.
- Do not add inline suppression comments (`noqa`, disable pragmas).
- Keep behavior unchanged unless the task explicitly requires behavior changes.
