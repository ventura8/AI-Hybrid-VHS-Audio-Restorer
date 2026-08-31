# Environment & Dependency Management

- **Installation**: `install_dependencies.sh` (Linux/macOS) and
  `install_dependencies.ps1` (Windows) set up a portable `.venv` environment,
  local FFmpeg, Poetry, and runtime-only dependencies.
- **Dependency Manager**: Use Poetry (`pyproject.toml` + `poetry.lock`) for
  dependency resolution and reproducible installs.
- **CUDA Runtime**: Use final CUDA 13.2 wheel stack (PyTorch source index + CUDA
  runtime packages).
- **Patching**: `scripts/apply_patches.py` removes DeepSpeed dependencies from
  Resemble-Enhance and fixes Torchaudio compatibility issues.

## Coding Standards

- **Audio Quality**: Always use 32-bit float (`pcm_f32le`) for intermediate
  files to maintain maximum dynamic range. Final audio encoding is
  container-dependent: AAC for `.mp4`/`.m4v` outputs, MP2 for `.mpg`/`.mpeg`
  outputs, and PCM only for configured PCM-capable containers.
- **Intermediate Precision**: Every intermediate stage stays 32-bit float. A
  separator that emits fixed-point WAV is converted back before the next stage
  consumes it.
- **Model Store**: Resolve separator models through the package-anchored
  `MODELS_DIR`, never a path relative to the working directory.
- **Stem Naming**: Match separator output by parenthesised token,
  case-insensitively. Models emit `(Vocals)`, `(vocals)`, `(crowd)`, `(other)`,
  and `(Instrumental)` depending on the architecture.
- **Pathing**: Use `pathlib.Path` for cross-platform compatibility.
- **Resiliency**: Always implement `is_valid_audio()` checks before skipping
  steps to ensure checkpoints are not corrupted.
- **Naming**: Temporary files should include the source stem (e.g.,
  `video_name.wav`).
- **Cleanup**: `_cleanup_work_dir()` purges `temp_work` only after a valid output
  is produced and `KEEP_INPUT_FILES` is false. `process_hybrid_audio()` may
  bypass cleanup when a valid output already exists.
- **Testing & Coverage**: Maintain **>= 90% total coverage** and **>= 90%
  per-file coverage** for every measured file. The per-file gate is enforced via
  `tests/tooling/quality_gate.py` over `coverage.json` in both local and CI
  validation.
- **Radon Coverage**: Run Radon complexity, maintainability, raw, and Halstead
  checks against the production code and the test suite (`tests/conftest.py`,
  `tests/unit/`, and `tests/integration/`) in both local and CI validation.
- **Badge Mandatory**: Always ensure the `assets/coverage.svg` badge is updated
  after making code changes. This is handled automatically by local test runs,
  but must be verified before pushing.
- **Code Quality**: Enforce `black`, `isort`, `ruff`, `flake8`, and `pylint`
  with max line length 140. Use `mypy` for type checking.
- **Security Quality**: Enforce `bandit -ll -ii` and `pip-audit` in local and CI
  validation.
- **Complexity Quality**: Enforce Radon CC/MI pass gates via
  `tests/tooling/radon_cc_gate.py` and `tests/tooling/radon_mi_gate.py`.
- **PowerShell Quality**: Lint project PowerShell scripts with
  `PSScriptAnalyzer`.
- **Pipeline Command**: Run `./run_pipeline_locally.sh` (Linux/macOS) or
  `.\run_pipeline_locally.ps1` (Windows) for local quality gate parity with CI.
- **Badge Automation**: Local pipeline always regenerates `assets/coverage.svg`
  after tests.
- **Documentation**:
  - Always update `.agent/instructions.md`, `AGENTS.md`, `.agents/skills/`,
    `.agent/workflows/`, `README.md`, and relevant `docs/` files when making
    changes.
