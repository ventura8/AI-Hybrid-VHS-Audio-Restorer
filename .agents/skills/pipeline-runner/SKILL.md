---
name: pipeline-runner
description: >-
  Execute and diagnose the canonical local pipeline runner
  (run_pipeline_locally.sh on Linux/macOS, run_pipeline_locally.ps1 on
  Windows), validate quality gates, and regenerate coverage badges.
---

# Local Pipeline Runner Skill

Use this skill to execute the project's quality pipeline locally, diagnose gate
failures, and verify that code changes satisfy all CI standards before
committing.

## Canonical Command

Always execute the pipeline from the repository root using the canonical runner
for the current platform.

On Linux and macOS:

```bash
./run_pipeline_locally.sh
```

On Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_pipeline_locally.ps1
```

## Pipeline Execution Sequence

The local pipeline executes the following stages sequentially:

1. **Environment Setup & Bootstrap**:
   - Verifies Poetry installation and validates the committed lockfile.
   - Syncs runtime and development dependencies.
   - Bootstraps and checks developer tooling.
1. **Code Quality & Linting Gates**:
   - `PSScriptAnalyzer`: PowerShell script linting.
   - `Ruff`: Fast Python linting.
   - `Black`: Python formatting check.
   - `isort`: Python import sorting check.
   - `Taplo`: TOML format checking.
   - `Flake8`: PEP 8 compliance.
   - `Pylint`: In-depth static analysis and code smell detection.
1. **Security Audits**:
   - `Bandit`: High/medium confidence and severity Python security analysis
     (`-ll -ii`).
   - `pip-audit`: Dependency vulnerability scan.
1. **Complexity & Maintainability (Radon)**:
   - `radon cc`: Enforces Grade A ($\\le 5$) on all functions/methods.
   - `radon mi`: Enforces Grade A ($\\ge 20$) on all tracked files.
   - Generates JSON reports for CC, MI, Raw metrics, and Halstead complexity.
1. **Markdown Quality**:
   - `mdformat`: Automatic formatting and delinting.
   - `pymarkdown scan`: Strict Markdown linting (MD013 line wraps $\\le 80$
     chars).
1. **Testing & Coverage**:
   - `pytest`: Executes unit, integration, and tooling test suites.
   - Evaluates coverage against the dynamic threshold policy (strict $\\ge
     90.00%$).
   - Strict per-file coverage enforcement via `coverage.json`.
   - Regenerates the visual coverage badge at `assets/coverage.svg`.
1. **SonarQube Cloud (CI only, not in the local runner)**:
   - `SonarSource/sonarqube-scan-action` uploads `coverage.xml` and
     `junit.xml` per `sonar-project.properties` and waits for the quality
     gate, which blocks the CI check; findings are read on sonarcloud.io.

## Troubleshooting & Remediation

- **Radon MI Failure (Rank B < 20)**:
  - High LOC and low comment density reduce the Maintainability Index.
  - Add comprehensive docstrings and explanatory commentary to raise the
    comment ratio above 20%.
  - Extract self-contained helper functions into dedicated domain submodules.
- **Radon CC Failure (Rank B > 5)**:
  - Refactor complex functions, nested branches, or repetitive assertion
    chains.
  - In unit tests, replace sequential asserts with `@pytest.mark.parametrize`.
- **Markdown Lint (MD013) Failure**:
  - Break prose sentences into lines $\\le 80$ characters.
  - For long URLs or file links, format them as reference-style links or concise
    relative links.
- **Coverage Gate Failure (< 90%)**:
  - Inspect `coverage.json` to identify uncovered lines in newly added or
    modified modules.
  - Add parameterized unit tests in `tests/unit/` targeting all branch
    conditions.
- **Running the gate in a git worktree (Windows)**:
  - A fresh worktree checks every tracked file out with CRLF
    (`core.autocrlf=true`), so the gate stops at Taplo and then at the
    Markdown check whatever the content. Normalise first:
    `git ls-files '*.md' '*.toml' | xargs sed -i 's/\r$//'`; `git add -u`
    afterwards clears the stat-only "M" entries (2026-10-11: two gate runs
    lost on this). The RNNoise models are binary in `.gitattributes`
    (2026-10-17): a worktree made before that rule holds them with CRLF,
    and `tests/unit/test_arnndn_model_pins.py` fails until they are deleted
    and checked out again (`git checkout -- models/arnndn`).
  - Give the worktree a `.venv` junction to the main venv
    (`mklink /J <worktree>\.venv <repo>\.venv`); the gate creates its own
    `.poetry-venv` there (ignored) and its `poetry install` against the
    junction reports "No dependencies to install" when the lock is current.
  - A worktree has none of the ignored assets (`experiments/`, `models/`,
    `experiments/quality_calibration/gates.json`,
    `experiments/reward/noise_floor_benign.json`): tests pass there, but
    never run a tuning loop or a scorer from a worktree, where those
    defaults resolve to nothing.
