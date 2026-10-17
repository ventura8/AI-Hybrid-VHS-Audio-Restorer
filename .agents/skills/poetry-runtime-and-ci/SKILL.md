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

Lock with the repository's Poetry (2.5.1, the version CI and the installers
pin) through the venv, never a global `poetry`: an older one locks too, but
writes a different header and spellings that the next 2.5.1 lock rewrites.
On Linux and macOS the interpreter is `.venv/bin/python`.

```powershell
.venv\Scripts\python -m poetry lock
.venv\Scripts\python -m poetry update --lock accelerate tqdm
.venv\Scripts\python -m poetry check --lock
```

A plain `poetry lock` picks up a `pyproject.toml` edit and keeps every other
package at its locked version (about 10 s). `poetry lock --regenerate`, or
`poetry update --lock` without names, moves every package to its newest
allowed release (about 20 s; on 2026-10-09 that was 20 packages, one a major
`huggingface-hub` bump), so it belongs to a release's dependency step and the
full gate. Write over the existing lock rather than deleting it: on Windows a
lock created from nothing comes out with CRLF line endings, while Poetry keeps
the LF of the committed one.

### 2. Markers That Keep the Lock Fast

Until v1.4.0 `poetry lock` never finished here: over six hours in CI, and from
Poetry 2.4.3 even a targeted `poetry update` stalled at round 79 of the
solver's override loop (`Duplicate dependencies for torch`). The cause was two
marker defects in `pyproject.toml`, not Poetry:

- Poetry re-solves the whole graph once per branch of a dependency listed
  twice. The torch, torchvision and torchaudio pairs ended in
  `and python_version == '3.12' and implementation_name == 'cpython'`, so
  their two markers left a gap, and Poetry added a third "torch not required"
  branch that resolved PyPI torch 2.14 and a second CUDA stack (the old lock
  carried them as entries no platform installs).
- Poetry reads `sys_platform` and `platform_system` as unrelated variables.
  Our `platform_system` markers beside the `sys_platform` that
  audio-separator, cuda-toolkit and pandas write let it explore platforms no
  machine reports (`platform_system != "Darwin"` with
  `sys_platform == "darwin"`), where more dependencies split. Poetry 2.5
  intersects the combined marker with every dependency at each step, so one
  such sub-resolution took over five minutes.

The rules that keep the solve at seconds:

- Every platform-split pair (torch, torchvision, torchaudio, audio-separator,
  onnxruntime) uses `sys_platform`, and its two markers are exact
  complements: `sys_platform == 'linux' or sys_platform == 'win32'` for the
  cu130 and GPU builds, `sys_platform != 'linux' and sys_platform != 'win32'`
  for the PyPI (macOS) builds. A new pair takes the same form.
- `nvidia-cublas` and `nvidia-cuda-nvrtc` are pinned in the `ml` group to the
  versions torch's cuda-toolkit requires (13.1.1.3 and 13.0.88 for
  cuda-toolkit 13.0.3). cuda-toolkit pins them only behind its extras, and
  nvidia-cudnn-cu13 and nvidia-cublas ask for them unpinned, so without the
  pins the lock carries two of each on Linux. Their marker is torch's own
  `platform_system == 'Linux'`; a `sys_platform` marker leaves phantom
  entries. They change only when torch's cuda-toolkit version does (torch
  2.14.1+cu130 still requires 13.0.3), and a stale pin fails `poetry lock` in
  about 2 s with a message naming the version cuda-toolkit wants.
- cuda-toolkit's own duplicate requirements still make 19 of the 22 override
  rounds. That is upstream metadata and costs a few seconds.

Measured on 2026-10-09 with Poetry 2.5.1, one solve at a time: `poetry lock`
8-12 s, `poetry lock --regenerate` or a lock from no lock file 21 s, a torch
2.14.1+cu130 trio bump 13 s, `poetry update --lock accelerate tqdm` 9 s. The
old `pyproject.toml` was killed at a 15-minute cap, still inside its second
torch branch. Poetry 2.4.1 and 2.4.3 lock the fixed one in about 5 s. A solve
that runs past a minute means a rule above was broken: find the pair that was
added or changed instead of waiting it out.

### 3. Install Runtime-Only Dependencies (Installer Mode)

```powershell
poetry install --only main --verbose
```

### 4. Install Full Runtime + Dev Dependencies (CI/Local Dev Mode)

```powershell
poetry install --with dev --verbose
```

### 5. Audit Dependencies for Vulnerabilities

```powershell
poetry run pip-audit
```

### 6. Check TOML Configuration Syntax

```powershell
poetry run taplo fmt --check pyproject.toml
```
