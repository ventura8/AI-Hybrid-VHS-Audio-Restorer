---
name: installer-tester
description: >-
  Test and validate Windows installer scripts (install_dependencies.ps1,
  start.bat), PowerShell environment detection, CUDA runtime provisioning, and
  venv bootstrapping.
---

# Installer Tester Skill

Use this skill to test, validate, and troubleshoot the end-user Windows
installation workflow (`install_dependencies.ps1` and `start.bat`).

## Installation Architecture

1. **PowerShell Execution & Elevation**:
   - `install_dependencies.ps1` detects administrative rights and prompts for
     elevation if necessary.
   - Sets PowerShell execution policy appropriately for script execution.
1. **Environment & Dependency Checks**:
   - Verifies Python 3.12.x installation (`>=3.12,<3.13`).
   - Requires NVIDIA CUDA 13.2 runtime stack compatibility.
   - Installs and bootstraps Poetry if absent.
1. **Virtual Environment Provisioning**:
   - Creates a dedicated `.venv` in the repository root.
   - Runs `poetry install --only main --verbose` to install runtime-only
     packages without bloat from test/dev tools.
1. **Pre-Trained Model Downloads**:
   - Ensures `models/` directory structure exists.
   - Verifies network connectivity and model integrity.

## Testing Commands

### 1. Dry-Run PowerShell Script Analysis

```powershell
Invoke-ScriptAnalyzer -Path .\install_dependencies.ps1 -Severity Warning,Error
```

### 2. Verify Runtime Dependency Set

```powershell
poetry install --only main --dry-run --verbose
```

### 3. Verify Batch File Launcher

```powershell
.\start.bat --help
```
