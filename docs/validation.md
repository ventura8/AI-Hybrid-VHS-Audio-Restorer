# Validation

## Canonical Command

Run full local validation with:

```powershell
./run_pipeline_locally.ps1
```

## What Validation Covers

- PowerShell script linting.
- Python linting for modules and tests (Ruff and Flake8).
- Static analysis gate (Pylint).
- Markdown auto-delint via `mdformat`.
- Markdown linting via `pymarkdownlnt`.
- Test execution with total coverage threshold enforcement.
- Strict per-file coverage enforcement from `coverage.json`.
- Coverage badge regeneration.

## Expected Outcome

- Lint passes without suppressions.
- Tests pass.
- Total coverage remains >= 90%.
- Every measured source file remains >= 90% coverage.
- assets/coverage.svg is updated.

## Fast Spot Checks

```powershell
.\.venv\Scripts\python.exe -m poetry run ruff check modules tests `
    restore_audio_hybrid.py scripts/apply_patches.py
.\.venv\Scripts\python.exe -m poetry run flake8 modules tests `
    restore_audio_hybrid.py scripts/apply_patches.py
.\.venv\Scripts\python.exe -m poetry run pytest -o addopts= `
    --cov=restore_audio_hybrid --cov=modules --cov-branch `
    --cov-report=xml --cov-report=json --cov-report=term `
    --cov-fail-under=90 tests/
.\.venv\Scripts\python.exe -m poetry run python `
    tests/tooling/quality_gate.py coverage.json --threshold 90.0
```

## CI Parity

CI workflow mirrors local validation ordering and tooling to avoid environment drift.
