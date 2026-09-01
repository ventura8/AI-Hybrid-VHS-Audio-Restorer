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
.\.poetry-venv\Scripts\poetry.exe run ruff check modules tests `
    restore_audio_hybrid.py scripts/apply_patches.py
.\.poetry-venv\Scripts\poetry.exe run black --check modules tests `
    restore_audio_hybrid.py scripts/apply_patches.py
.\.poetry-venv\Scripts\poetry.exe run isort --check-only --diff `
    modules tests `
    restore_audio_hybrid.py scripts/apply_patches.py
.\.poetry-venv\Scripts\poetry.exe run flake8 modules tests `
    restore_audio_hybrid.py scripts/apply_patches.py
$tomlFiles = @(git ls-files "*.toml")
if ($tomlFiles.Count -gt 0) {
    .\.poetry-venv\Scripts\poetry.exe run taplo fmt --check $tomlFiles
}
.\.poetry-venv\Scripts\poetry.exe run bandit -ll -ii -r modules `
    restore_audio_hybrid.py scripts/apply_patches.py
.\.poetry-venv\Scripts\poetry.exe run pip-audit
.\.poetry-venv\Scripts\poetry.exe run radon cc `
    modules `
    tests\conftest.py tests\unit tests\integration restore_audio_hybrid.py `
    scripts/apply_patches.py -s
.\.poetry-venv\Scripts\poetry.exe run pytest -o addopts= `
    --cov=restore_audio_hybrid --cov=modules --cov-branch `
    --cov-report=xml --cov-report=json --cov-report=term `
    --cov-fail-under=90 tests/
.\.poetry-venv\Scripts\poetry.exe run python `
    tests/tooling/quality_gate.py coverage.json `
    --threshold 90.0
```

## Optional Hardware Matrix

Hardware validation is opt-in and supplements the canonical quality gate. The
Piper catalog covers 50 language-native voices with checksum-pinned downloads.
Piper is installed in `tools/piper-tts/.venv`, separate from the application's
CUDA/TensorRT environment, so its CPU ONNX Runtime cannot change GPU provider
selection.

```powershell
.\.poetry-venv\Scripts\poetry.exe run python scripts/audit_hardware.py
.\.poetry-venv\Scripts\poetry.exe run python `
    scripts/generate_audio_matrix.py core --language all
$env:AI_RESTORE_HARDWARE_TESTS = "1"
.\.poetry-venv\Scripts\poetry.exe run pytest tests/hardware -v
```

Use `scripts/run_hardware_validation.py --execute` only on a prepared machine;
it drives selected modes through temporary video fixtures and writes timing and
peak-VRAM data beneath `artifacts/`.

## CI Parity

CI workflow mirrors local validation ordering and tooling to avoid environment
drift.
