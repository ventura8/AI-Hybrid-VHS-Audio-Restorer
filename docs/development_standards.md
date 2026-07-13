# Environment & Dependency Management

- **Installation**: `install_dependencies.ps1` sets up a portable
  `.venv` environment, local FFmpeg, Poetry, and runtime-only
  dependencies.
- **Dependency Manager**: Use Poetry (`pyproject.toml` +
  `poetry.lock`) for dependency resolution and reproducible installs.
- **CUDA Runtime**: Use final CUDA 13.2 wheel stack (PyTorch source
  index + CUDA runtime packages).
- **Patching**: `scripts/apply_patches.py` removes DeepSpeed
  dependencies from Resemble-Enhance and fixes Torchaudio
  compatibility issues.

## Coding Standards

- **Audio Quality**: Always use 32-bit float (`pcm_f32le`) for
  intermediate and final audio to maintain maximum dynamic range.
- **Pathing**: Use `pathlib.Path` for cross-platform compatibility.
- **Resiliency**: Always implement `is_valid_audio()` checks before
  skipping steps to ensure checkpoints are not corrupted.
- **Naming**: Temporary files should include the source stem
  (e.g., `video_name.wav`).
- **Cleanup**: The `temp_work` directory is automatically purged upon
  successful task completion.
- **Testing & Coverage**: Maintain **>= 90% total coverage** and
  **>= 90% per-file coverage** for every measured file. The per-file
  gate is enforced via `tests/tooling/quality_gate.py` over
  `coverage.json` in both local and CI validation.
- **Badge Mandatory**: Always ensure the `assets/coverage.svg` badge is
  updated after making code changes. This is handled automatically by
  local test runs, but must be verified before pushing.
- **Code Quality**: Enforce `ruff`, `flake8`, and `pylint` with max
  line length 140. Use `mypy` for type checking. Complexity is
  monitored using `radon`.
- **PowerShell Quality**: Lint project PowerShell scripts with
  `PSScriptAnalyzer`.
- **Pipeline Command**: Run `./run_pipeline_locally.ps1` for local
  quality gate parity with CI.
- **Badge Automation**: Local pipeline always regenerates
  `assets/coverage.svg` after tests.
- **Documentation**:
  - Always update `.agent/instructions.md`, `AGENTS.md`,
    `.agent/skills/`, `.agent/workflows/`, `README.md`, and relevant
    `docs/` files when making changes.
