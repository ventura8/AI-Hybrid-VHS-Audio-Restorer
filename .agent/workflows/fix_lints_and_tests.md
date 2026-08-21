---
description: >-
  Fix lint and test issues for a file in a single pass, ensuring 90% coverage
  and cross-platform compatibility.
---

# Workflow: Fix Lints and Tests

// turbo-all

This workflow ensures that code quality and testing standards are met in a
single, efficient pass.

## 1. Run Lints

First, identify all linting issues for the target file.

```powershell
black --check modules tests restore_audio_hybrid.py scripts/apply_patches.py
isort --check-only --diff modules tests restore_audio_hybrid.py scripts/apply_patches.py
ruff check modules tests restore_audio_hybrid.py scripts/apply_patches.py
flake8 modules tests restore_audio_hybrid.py scripts/apply_patches.py
pylint --errors-only modules tests restore_audio_hybrid.py scripts/apply_patches.py
```

## 2. Fix Lints

Apply fixes for all reported linting errors. Prioritize fixing lints before
moving to tests.

## 3. Run Tests & Coverage

Once lints pass, run tests and check coverage.

```bash
./run_pipeline_locally.sh
```

```powershell
./run_pipeline_locally.ps1
```

## 4. Fix Test Failures

Resolve any failing tests.

- Ensure mocks are **Windows/Linux compatible** (use `autospec=True` or
  `create=True` for platform-specific attributes).
- Use `pathlib` for any path-related fixes.

## 5. Generate Badge & Verify

The local pipeline already generates the coverage badge and enforces threshold
checks, then re-run the canonical gate for the platform in use.

> [!IMPORTANT]
> Always manually check that the coverage is at least **90%**.
