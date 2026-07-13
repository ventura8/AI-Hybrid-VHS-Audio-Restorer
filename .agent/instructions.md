# AI Instructions: AI-Hybrid VHS Audio Restorer

This document provides technical guidance for AI agents and developers
working on this project.

## Core Directives

### 1. Smart Fixes: Lint & Test Pass

When fixing issues in a file, follow this order of operations in a single pass:

- **Lint First**: Run
  `ruff check modules tests restore_audio_hybrid.py scripts/apply_patches.py`
  and
  `flake8 modules tests restore_audio_hybrid.py scripts/apply_patches.py`.
- **Static Analysis Gate**: Run
  `pylint --errors-only modules tests restore_audio_hybrid.py scripts/apply_patches.py`
  before `pytest`.
- **Markdown Quality**: Run `mdformat` (auto-delint) and `pymarkdown scan`
  for Markdown docs and agent guidance.
- **Auto-Fix**: Use `ruff format ...` and `ruff check --fix ...` for
  safe automatic fixes.
- **Manual Fix**: If lints remain after Ruff fixes, resolve them manually
  (without `noqa` or suppression pragmas).
- **Tests Second**: Once lints pass, run tests (`pytest`).
  - **Test Linting**: Ensure test files pass both Ruff and Flake8 checks.
- **Single Pass**: Aim to resolve both lint and test issues in the same
  iteration whenever possible.

### 2. Coverage & Badges

- **Minimum Coverage**: Maintain at least **90%** code coverage.
- **Badge Generation**: Always generate/update the coverage badge locally
  after running tests.
- **Verification**: Check the generated badge to ensure it reflects
  $\\ge 90%$. If coverage drops below 90%, add tests or optimize code to
  meet the requirement.
- **Canonical Local Gate**: Run `./run_pipeline_locally.ps1`; it mirrors
  CI checks and regenerates the coverage badge.

### 3. Cross-Platform Mocking

- **Windows/Linux Compatibility**: Always use mocks and tests that are
  compatible with both Windows and Linux.
- **Dynamic Attributes**: When mocking platform-specific operations
  (e.g., `os.add_dll_directory` or `ctypes.windll`), use `autospec=True`
  or `create=True` in `unittest.mock.patch` to avoid `AttributeError` on
  systems where those attributes do not exist.
- **Path Handling**: Use `pathlib` for all file path operations to
  ensure cross-platform compatibility.

### 4. Robust Resume & File Validity

- **Check Validity, Not Just Existence**: When checking if a step is
  done, never rely solely on `path.exists()`. Always assume a file might
  be a 0-byte corruption from a crash.
  - Use helpers like `is_valid_audio(path)` or `is_valid_video(path)`.
- **Skip Logic**: Ensure every potentially expensive step has a
  "Skip if Exists & Valid" check at the very top.

### 5. Pipeline Architecture (Lossless Background)

- **Process Modes**:
  - `hybrid` (default/fallback): separation + vocal enhancement +
    background denoise + sync + final mix.
  - `denoise_only`: full-audio denoise + sync + final remux, with no
    separation or vocal enhancement.
- **Separation Strategy (`hybrid` only)**: The project uses a
  **Subtractive/Lossless Background** approach.
  - **Do NOT** use a dedicated "Music" model (like MDX-NET) because it
    filters out ambient sounds (birds, nature).
  - **DO** use `BS-Roformer` to extract "Vocals". The "Background"
    (Instrumental) stem from this process is used as the backing track.
    This guarantees 100% retention of non-vocal audio.
- **Parallelization**:
  - **Stability Priority**: While the engine supports parallel threads,
    the *high-level* pipeline defaults to **Sequential** execution for
    steps 2-4.
  - **Reason**: Running heavy GPU/CPU tasks in parallel on Windows
    causes `stdout` contention (interleaved progress bars) and possible
    race conditions.
  - **Note**: Development should favor UI stability (clean, readable
    logs) over raw theoretical speed if they conflict.

### 6. Synchronization Logic

- **Methods**: The project supports two sync methods:
  - `shift` (Default): Global delay correction (Cross-Correlation). Fast and artifact-free.
  - `dtw`: Dynamic Time Warping. Corrects variable drift (wow/flutter).
    - **Hybrid Engine**: Uses `torch.cdist` on GPU for distance
      calculation + CPU for pathfinding.
    - **Precision**: 40Hz - 100Hz.
- **Implementation**: Sync logic is in `_align_stems` / `_align_stems_dtw`.
  - **UI Standards**: Every task with a progress bar MUST explicitly
    call `draw_progress_bar(100, ...)` before completion.
  - **Sequential Execution**: In `hybrid`, syncing Vocals and
    Background runs sequentially to maintain clean grouped log output.
  - **Mode Behavior**: In `denoise_only`, only one full-audio sync pass
    is executed.
  - **Dynamic Radius**: `radius` for DTW must scale with resolution
    (e.g., `0.3 * Resolution`).
  - **Recommended Resolution**: 100Hz for high precision (lipsync),
    40Hz for general speed correction.

### 7. Hardware & GPU Acceleration

- **Hybrid GPU Support**:
  - **Masking**: The system uses `CUDA_VISIBLE_DEVICES` in
    `hardware.py` to hide the integrated GPU (Intel) and force the app
    to use the primary NVIDIA GPU.
  - **DLL Injection**: `utils.py` uses `get_nvidia_paths` from
    `hardware.py` to inject CUDA libraries (CUDNN/CUBLAS wheel
    packages) into process `PATH`. This is critical for portable
    Windows installations so GPU acceleration works without global
    driver modifications.
- **Python API Usage**:
  - Always use the `audio_separator.separator.Separator` Python class
    for separation and denoising. Avoid the CLI companion because it
    offers less control over DLL loading and device selection in hybrid
    environments.

### 8. Output Naming

- `hybrid` outputs must use `*_Hybrid_Cleaned.<ext>`.
- `denoise_only` outputs must use `*_Denoised_Cleaned.<ext>`.
- File scanning must exclude both cleaned output patterns to avoid
  reprocessing generated files.

### 9. Execution Modes

- **Interactive**: Running without arguments or double-clicking
  `start.bat` triggers an interactive prompt for drag-and-drop or
  scanning the `input` folder.
- **CLI**: Passing a file or directory as a command-line argument
  processes those targets directly and outputs to the same folder as
  the input.

### 10. PR Comment Resolution Discipline

- For pull request review comments (CodeRabbit and human), use both
  GitHub CLI (`gh`) and available MCP PR-comment tooling.
- Never resolve/close a comment thread before posting a detailed
  response that explains:
  - what changed (or why no change was needed),
  - where it changed,
  - what validation was run,
  - and any remaining risks/follow-ups.
- If blocked, post a detailed blocker update and keep the thread open
  until actionable.

## Documentation Index

- [Project Overview & Directory Structure](../docs/project_overview.md)
- [Key Logic & Pipeline](../docs/pipeline_logic.md)
- [Hardware Optimization](../docs/hardware_optimization.md)
- [Configuration](../docs/configuration.md)
- [Development & Standards](../docs/development_standards.md)
- [Architecture](../docs/architecture.md)
- [Setup](../docs/setup.md)
- [Validation](../docs/validation.md)
- [Instructions](../docs/instructions.md)
