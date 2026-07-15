# Architecture

## Overview

The project is a modular Windows-focused audio restoration pipeline for VHS-era material.

Core goals:

- Preserve non-vocal ambience.
- Improve vocal clarity.
- Keep pipeline resilient and resumable.
- Run consistently in local and CI quality gates.

## High-Level Flow

1. Input discovery and validation.
1. Audio extraction to high-quality intermediate WAV.
1. Branch on `process_mode`:
1. `hybrid`: stem separation (vocals/background) -> vocal enhancement ->
   background denoise -> per-stem synchronization (shift or DTW) ->
   final mix and mux.
1. `denoise_only`: full-audio denoise -> full-track synchronization
   (shift or DTW) -> final remux.

## Modules

- modules/config.py: loads runtime configuration and constants.
- modules/hardware.py: detects CPU/GPU and computes runtime profile.
- modules/processing.py: orchestrates extraction, separation,
  enhancement, denoise, sync, and final mix.
- modules/sync.py: shift and DTW synchronization logic.
- modules/ui.py: interactive/file-input behavior and startup banner.
- modules/utils.py: logging, progress output, process handling, and
  media validation helpers.
- restore_audio_hybrid.py: entry point.

## Runtime Composition

- Python 3.12.x.
- Poetry-managed dependencies.
- CUDA 13.2-compatible wheel strategy for GPU acceleration.
- Local FFmpeg binaries installed in virtual environment scripts path.

## Reliability Design

- Atomic writes for generated media where applicable.
- Resume checks rely on validity checks, not existence-only checks.
- Temporary work directories are isolated per input.
- Cleanup preserves failed work artifacts for troubleshooting when needed.

## Quality Gates

Local canonical gate:

```powershell
./run_pipeline_locally.ps1
```

Gate stages:

- PowerShell lint
- Black
- isort
- Ruff
- Flake8
- Taplo
- Pylint
- Bandit
- pip-audit
- Radon reports + CC/MI pass gates
- Markdown auto-delint/lint
- Pytest with total coverage threshold
- Strict per-file coverage check from coverage.json
- Coverage badge regeneration
