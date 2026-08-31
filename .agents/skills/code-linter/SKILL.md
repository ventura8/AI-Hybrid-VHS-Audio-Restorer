---
name: code-linter
description: >-
  Run Black, isort, Ruff, Flake8, Pylint, Taplo, PSScriptAnalyzer, Bandit,
  pip-audit, and Radon CC/MI checks over repository files without suppressions.
---

# Code Linter Skill

Use this skill to lint all Python modules, tests, tooling scripts, PowerShell
scripts, TOML configs, and Markdown documents without using suppression
comments or inline ignores.

Root `AGENTS.md` strictly enforces **No Suppressions Allowed**: never add
`# noqa`, `# pylint: disable`, `# type: ignore`, `# bandit: disable`, or
equivalent ignore pragmas.

## New and Modified Files

When adding or modifying files as part of a change set, always lint and format
them before completing your work:

1. Identify the file type and run the matching linter(s).
1. Auto-fix formatting and trivial issues first before manual edits.
1. For Python files, run `black`, `isort`, `ruff check`, `flake8`, and `pylint`.
1. For TOML files (`pyproject.toml`, `poetry.toml`), run `taplo fmt --check`.
1. For PowerShell scripts (`*.ps1`), run `PSScriptAnalyzer`.
1. For Markdown docs, run `mdformat` and `pymarkdown scan`.
1. Verify complexity and maintainability with Radon (all blocks CC $\\le 5$ Grade
   A, all files MI $\\ge 20$ Grade A).

## Standard Lint Commands

### 1. Python Auto-Formatting and Import Sorting

```powershell
poetry run black modules tests restore_audio_hybrid.py scripts/apply_patches.py scripts/batch_restore.py
poetry run isort modules tests restore_audio_hybrid.py scripts/apply_patches.py scripts/batch_restore.py
```

### 2. Static Analysis and PEP 8 Linters

```powershell
poetry run ruff check modules tests restore_audio_hybrid.py scripts/apply_patches.py scripts/batch_restore.py
poetry run flake8 modules tests restore_audio_hybrid.py scripts/apply_patches.py scripts/batch_restore.py
poetry run pylint modules tests restore_audio_hybrid.py scripts/apply_patches.py scripts/batch_restore.py
```

### 3. TOML Configuration Formatting

```powershell
poetry run taplo fmt --check pyproject.toml poetry.toml
```

### 4. PowerShell Linting

```powershell
Invoke-ScriptAnalyzer -Path .\run_pipeline_locally.ps1 -Severity Warning,Error
Invoke-ScriptAnalyzer -Path .\install_dependencies.ps1 -Severity Warning,Error
Invoke-ScriptAnalyzer -Path .\scripts\install_pr_review_tooling.ps1 -Severity Warning,Error
Invoke-ScriptAnalyzer -Path .\.github\scripts\Invoke-PowerShellLint.ps1 -Severity Warning,Error
```

### 5. Security and Vulnerability Audits

```powershell
poetry run bandit -ll -ii -r modules restore_audio_hybrid.py scripts/apply_patches.py scripts/batch_restore.py
poetry run pip-audit
```

### 6. Radon Maintainability and Cyclomatic Complexity Gates

```powershell
poetry run python tests/tooling/radon_cc_gate.py
poetry run python tests/tooling/radon_mi_gate.py
```

## Hard Invariants

- **Line Length**: Python code max line length is 140.
- **Zero Suppressions**: No inline ignores or bypass pragmas allowed anywhere in
  the repository (product code, tooling, or tests).
- **Tests Are Not Exempt**: Tests must pass the exact same linters (`black`,
  `isort`, `ruff`, `flake8`, `pylint`, `radon`) as production modules.
- **Radon Grades**: Every single block must be Cyclomatic Complexity Grade A
  (CC $\\le 5$), and every file must be Maintainability Index Grade A (MI $\\ge
  20$).
