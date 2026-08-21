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
