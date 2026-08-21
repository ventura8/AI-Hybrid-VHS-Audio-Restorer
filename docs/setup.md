# Setup

## Prerequisites

- **Linux**: Ubuntu 22.04+, Debian 12+, Fedora, Arch Linux (with `ffmpeg`).
- **macOS**: macOS 13+ on Apple Silicon (M1/M2/M3/M4) or Intel (with
  `brew install ffmpeg`).
- **Windows**: Windows 10/11 (64-bit).
- **Python**: Python 3.12.x in PATH.
- Internet access for first dependency install.

## Recommended One-Time Setup

### Linux / macOS

```bash
chmod +x install_dependencies.sh start.sh run_pipeline_locally.sh
./install_dependencies.sh
```

### Apple Silicon release asset

For M1, M2, M3, or M4 Macs, download the `macos-arm64` archive from the GitHub
release. Extract the archive, then run the installer from a native arm64
terminal:

```bash
uname -m  # must print arm64
./install_dependencies.sh
```

Avoid opening Terminal with Rosetta, since an x86_64 Python process cannot use
the native MPS backend.

Intel Macs should download the separate `macos-intel` archive and run the same
installer from their standard Terminal session.

### Windows

```powershell
./install_dependencies.ps1
```

Installer responsibilities:

- Create local `.venv` virtual environment.
- Verify / provision FFmpeg and FFprobe binaries.
- Install Poetry.
- Install runtime dependencies
  (`poetry install -v --with ml --without dev --no-root --no-interaction`).
- Apply runtime patch scripts (`scripts/apply_patches.py`).

## Local Quality Environment

To provision runtime and development tooling and run full checks:

```bash
# On Linux / macOS:
./run_pipeline_locally.sh

# On Windows:
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_pipeline_locally.ps1
```

This command enforces total coverage (>= 90%), strict per-file coverage (at or
above 90%), and regenerates `assets/coverage.svg`.

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
