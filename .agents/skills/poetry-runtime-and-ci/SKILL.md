---
name: poetry-runtime-and-ci
description: >-
  Manage Poetry dependencies, maintain pyproject.toml and poetry.lock, and
  enforce runtime vs development dependency separation for CI and installer.
---

# Poetry Runtime and CI Skill

Use this skill when modifying project dependencies, updating the lockfile, or
configuring the execution environment for local pipelines, CI workflows, and
the end-user installer.

## Core Dependency Rules

1. **Poetry as Single Source of Truth**:
   - Never use `requirements.txt` or `test-requirements.txt`.
   - All runtime dependencies belong in `[tool.poetry.dependencies]`.
   - All test and development tools belong in `[tool.poetry.group.dev.dependencies]`.
1. **Runtime vs Development Isolation**:
   - The end-user installer (`install_dependencies.ps1`) installs only runtime
     dependencies (`poetry install --only main --verbose`).
   - CI and local validation pipelines install both runtime and dev tools
     (`poetry install --with dev`).
1. **PyTorch & CUDA Runtime Constraints**:
   - Preserve CUDA 13.2 runtime compatibility.
   - Pinned wheel links and PyTorch extra index configurations must remain
     functional on Windows.

## Dependency Commands

### 1. Update and Lock Dependencies

```powershell
poetry lock
```

Use `poetry lock --regenerate` only for a full dependency refresh.

### 2. Install Runtime-Only Dependencies (Installer Mode)

```powershell
poetry install --only main --verbose
```

### 3. Install Full Runtime + Dev Dependencies (CI/Local Dev Mode)

```powershell
poetry install --with dev --verbose
```

### 4. Audit Dependencies for Vulnerabilities

```powershell
poetry run pip-audit
```

### 5. Check TOML Configuration Syntax

```powershell
poetry run taplo fmt --check pyproject.toml
```
