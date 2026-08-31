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

### Linux (.deb / .rpm)

Download the `.deb` (Debian/Ubuntu) or `.rpm` (Fedora/RHEL) package from the
GitHub Release:

```bash
# Debian / Ubuntu:
sudo dpkg -i AI-Hybrid-VHS-Audio-Restorer-v*.deb

# Fedora / RHEL:
sudo rpm -i AI-Hybrid-VHS-Audio-Restorer-v*.rpm
```

### macOS (.pkg & .app Bundle)

Download the native `.pkg` installer from the GitHub Release (`arm64` for
Apple Silicon M1/M2/M3/M4 or `intel` for x86_64), double-click to install or run:

```bash
# Apple Silicon (M1/M2/M3/M4):
sudo installer -pkg AI-Hybrid-VHS-Audio-Restorer-v*-macos-arm64.pkg -target /

# Intel (x86_64):
sudo installer -pkg AI-Hybrid-VHS-Audio-Restorer-v*-macos-x86_64.pkg -target /
```

This installs `/Applications/AI-Hybrid-VHS-Audio-Restorer.app` and creates the
`ai-hybrid-vhs-audio-restorer` command in `/usr/local/bin`.

On Intel x86_64 Macs, the installer skips the AI (`ml`) dependency group, so
only the native DSP modes (`vhs_native`, `auto_ffmpeg_native`, and
`arnndn_speech`) are supported.

### Windows (.exe)

Download the `AI-Hybrid-VHS-Audio-Restorer-v*-windows.exe` standalone launcher
from GitHub Releases, and run it directly or from PowerShell (it auto-installs
and sets up the environment on first run, then restores your audio files):

```powershell
$launcher = Get-ChildItem -Filter "AI-Hybrid-VHS-Audio-Restorer-*-windows.exe" |
    Select-Object -First 1
& $launcher.FullName
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
