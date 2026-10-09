---
name: test-runner
description: >-
  Run unit, integration, and end-to-end test suites with pytest, manage mock
  fixtures, and enforce strict per-file coverage >= 90%.
---

# Test Runner Skill

Use this skill to run unit tests, integration tests, and coverage validation
workflows for AI Hybrid VHS Audio Restorer.

## Test Directory Topology

- `tests/unit/`: Fast, isolated unit tests covering all internal modules.
  - `test_config.py`: Configuration loading, validation, and mode parsing
    (supporting 11 identifiers including aliases).
  - `test_filters_auto_vhs.py` / `test_processing_native.py`: Native FFmpeg DSP
    filtering.
  - `test_filters_arnndn.py`: ARNNDN neural speech filtering and model
    resolution.
  - `test_hardware.py`: CUDA, XPU, MPS, and CPU device detection and fallback.
  - `test_processing.py` / `test_processing_resilience.py`: AI separation,
    vocal enhancement, and error recovery.
  - `test_sync.py` / `test_dtw_sync.py`: Audio sync, cross-correlation, and
    GPU/CPU DTW alignment.
  - `test_ui.py`: User interaction, file scanner, and terminal UI rendering.
  - `test_utils.py`: Subprocess runners, progress parsers, and atomic audio I/O.
- `tests/integration/`: High-level workflow tests.
  - `test_end_to_end.py`: Parametrized smoke tests across eight mode identifiers
    (`auto`, `multipass_auto`, `auto_pure`, `denoise_only`, `hybrid`,
    `ffmpeg_native`, `auto_ffmpeg_native`, `arnndn_speech`), normalized into six
    pipeline categories (pure, multipass, native, RNNoise, automatic, and
    denoise-only).
  - `test_entry_point.py`: Main CLI entry point execution tests.
- `tests/tooling/`: Test tooling and coverage threshold verification.
  - `quality_gate.py`, `radon_cc_gate.py`, `radon_mi_gate.py`,
    `threshold_policy.py`.

## Test Execution Commands

### 1. Run Full Test Suite with Coverage

```powershell
poetry run pytest --cov=restore_audio_hybrid --cov=modules --cov-branch `
    --cov-report=json --cov-report=term --cov-fail-under=90 tests/
poetry run python tests/tooling/quality_gate.py coverage.json --threshold 90.0
```

The `--verbose` behavior from `addopts` is preserved; the per-file 90%
threshold is enforced by `quality_gate.py`.

### 2. Run Targeted Unit Test Module

```powershell
poetry run pytest tests/unit/test_processing_native.py -v
```

### 3. Run Integration Smoke Tests

```powershell
poetry run pytest tests/integration/test_end_to_end.py -v
```

### 4. Check Per-File Coverage Gate

```powershell
poetry run python tests/tooling/quality_gate.py
```

## Tool Quirks That Cost Time

- **The conftest updates the badge only from this session's reports.**
  `tests/conftest.py`'s `pytest_sessionfinish` returns early under `--no-cov` or
  without `--cov` (pytest-cov's `_cov` plugin not registered). With `--cov` it
  rebuilds `assets/coverage.svg` and runs the per-file gate only from a
  `coverage.xml` / `coverage.json` this session wrote (`pytest_sessionstart`
  records their modification times and the finish compares them), so a stale
  report from an earlier run is never read
  (`tests/unit/test_conftest_coverage_hook.py`). Until 2026-10-09 every session
  rewrote the badge from the `coverage.xml` in its working directory, `--no-cov`
  included, and a targeted run from the repository root moved it (95.68% to
  95.87%). A targeted `--cov ... --cov-report=xml` run from the root still
  writes partial coverage there and moves the badge, so take the committed badge
  from the full quality gate run (whose runners also run `quality_gate.py` and
  `badge_report.py` as their own steps) and run targeted coverage from an empty
  directory.
- **`--cov` takes a path, not a dotted module.**
  `pytest --cov=scripts.restoration_quality.auditory` fails at collection
  with numpy's "cannot load module more than once per process"; pass the
  package directory instead (`--cov=scripts` or
  `--cov=scripts/restoration_quality`). With other agents in the checkout,
  give coverage a private data file (`COVERAGE_FILE` under the temp
  directory) and pass `-p no:cacheprovider`.
- **Never run a tool with a directory of stray `.py` files as the working
  directory.** Python puts the working directory (or a script's own
  folder) first on the import path, so a stray file there shadows a real
  module. On 2026-10-09 radon, run from a session scratchpad, imported an
  old probe named `six.py`, which rewrote `scripts/cli_paths.py` before it
  failed; the file was restored from `HEAD`. Run pytest, radon and the
  linters from the repository root or from an empty directory, and keep
  probe scripts in a folder of their own, run by path.

## Mocking Conventions and Invariants

1. **Subprocess Isolation**: Never invoke real external binaries (FFmpeg,
   resemble-enhance, etc.) during unit tests. Use `unittest.mock.patch` to mock
   `run_command_with_progress`, `attempt_cpu_run_with_retry`, or `subprocess.run`.
1. **Audio File Validation**: Use `@patch("modules.utils.is_valid_audio")` or
   temporary mock `.wav` files created via `tmp_path`.
1. **Atomic Writes**: Ensure tests clean up temporary working artifacts and
   verify atomic file renaming (`.tmp.wav` $\\rightarrow$ `.wav`).
1. **Radon CC in Tests**: Keep test function cyclomatic complexity $\\le 5$
   (Grade A). Use `@pytest.mark.parametrize` instead of consecutive assertions.
1. **Coverage Floor**: Strict $\\ge 90.00%$ coverage required on every single
   source module.
