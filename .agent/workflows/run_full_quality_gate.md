# Workflow: Run Full Local Quality Gate

Use this workflow to validate the entire repository against all CI and local
quality gates prior to proposing work completion.

## Step 1: Execute Canonical Pipeline Runner

Run the canonical runner for the current platform from the repository root.

On Linux and macOS:

```bash
./run_pipeline_locally.sh
```

On Windows:

```powershell
./run_pipeline_locally.ps1
```

## Step 2: Verify Individual Gate Outcomes

Confirm that each of the following stages reports `PASS` or completes with exit
code `0`:

1. **PowerShell Script Analyzer**: No errors or warnings on
   `install_dependencies.ps1`, `run_pipeline_locally.ps1`,
   `scripts/install_pr_review_tooling.ps1`, and
   `.github/scripts/Invoke-PowerShellLint.ps1`.
1. **Ruff / Black / isort**: Python code formatted and delinted.
1. **Taplo**: TOML configuration files formatted cleanly.
1. **Flake8 / Pylint**: Static analysis checks pass with 0 errors.
1. **Bandit / pip-audit**: 0 security issues and 0 known CVEs.
1. **Radon Complexity & Maintainability**:
   - Cyclomatic Complexity: All blocks Grade A ($\\le 5$).
   - Maintainability Index: All files Grade A ($\\ge 20$).
1. **Markdown Quality**:
   - `mdformat` runs in check mode and passes.
   - `pymarkdown scan` passes MD013 ($\\le 80$ char wraps) and syntax rules.
1. **Pytest**: All unit, integration, and tooling tests pass.
1. **Coverage Floor**: Strict $\\ge 90.00%$ per-file coverage met on all source
   files.
1. **Coverage Badge**: Regenerated at `assets/coverage.svg`.

## Step 3: Handle Failures

- **Lint / Format Failures**: Run auto-fixers (`black`, `isort`, `mdformat`)
  first, then resolve remaining syntax issues manually.
- **Coverage Drops**: Add parameterized unit tests in `tests/unit/` targeting
  uncovered lines in `coverage.json`.
- **Complexity Violations**: Refactor branching logic and split large functions.
