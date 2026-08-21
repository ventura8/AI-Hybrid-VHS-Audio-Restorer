# Validation

## Canonical Command

Run full local validation with:

```bash
./run_pipeline_locally.sh
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_pipeline_locally.ps1
```

## What Validation Covers

- PowerShell script linting.
- Python linting for modules and tests (Black, isort, Ruff, and Flake8).
- TOML formatting check via `taplo`.
- Static analysis gate (Pylint).
- Security checks via Bandit and pip-audit.
- Markdown formatting verified via `mdformat --check`.
- Markdown linting via `pymarkdown --config .pymarkdown.json scan`.
- Test execution with total coverage threshold enforcement.
- Radon complexity, maintainability, raw, and Halstead checks for modules and
  the test suite.
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
.\.venv\Scripts\python.exe -m poetry run black --check modules tests `
    restore_audio_hybrid.py scripts/apply_patches.py
.\.venv\Scripts\python.exe -m poetry run isort --check-only --diff `
    modules tests `
    restore_audio_hybrid.py scripts/apply_patches.py
.\.venv\Scripts\python.exe -m poetry run flake8 modules tests `
    restore_audio_hybrid.py scripts/apply_patches.py
$tomlFiles = @(git ls-files "*.toml")
if ($tomlFiles.Count -gt 0) {
    .\.venv\Scripts\python.exe -m poetry run taplo fmt --check $tomlFiles
}
.\.venv\Scripts\python.exe -m poetry run bandit -ll -ii -r modules `
    restore_audio_hybrid.py scripts/apply_patches.py
.\.venv\Scripts\python.exe -m poetry run pip-audit
.\.venv\Scripts\python.exe -m poetry run radon cc `
    modules `
    tests\conftest.py tests\unit tests\integration restore_audio_hybrid.py `
    scripts/apply_patches.py -s
.\.venv\Scripts\python.exe -m poetry run pytest -o addopts= `
    --cov=restore_audio_hybrid --cov=modules --cov-branch `
    --cov-report=xml --cov-report=json --cov-report=term `
    --cov-fail-under=90 tests/
.\.venv\Scripts\python.exe -m poetry run python `
    tests/tooling/quality_gate.py coverage.json `
    --threshold 90.0
```

## CI Parity

CI workflow mirrors local validation ordering and tooling to avoid environment
drift.
