# Setup

## Prerequisites

- Windows 10/11
- Python 3.12.x in PATH
- Internet access for first dependency install

## Recommended One-Time Setup

Run installer:

```powershell
./install_dependencies.ps1
```

Installer responsibilities:

- Create local virtual environment.
- Install local FFmpeg binaries.
- Install Poetry.
- Install runtime dependencies with verbose Poetry install.
- Apply runtime patch script.

## Local Quality Environment

To provision runtime and development tooling and run full checks:

```powershell
./run_pipeline_locally.ps1
```

This command enforces total coverage (>= 90%), strict per-file
coverage (at or above 90%), and regenerates `assets/coverage.svg`.

## Manual Poetry Commands

Use these only when debugging setup issues.

```powershell
python -m poetry check --lock
python -m poetry install -v --with dev --no-root
```

## Common Setup Checks

- Verify lock file exists: poetry.lock
- Verify interpreter: Python 3.12.x
- Verify local environment path: .venv
- Verify FFmpeg binaries under virtual environment scripts folder

## Notes

- This repository no longer uses requirements.txt files.
- Dependency source of truth is pyproject.toml plus poetry.lock.
