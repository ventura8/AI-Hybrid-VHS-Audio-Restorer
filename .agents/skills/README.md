# Antigravity Workspace Skills

This directory contains modular, executable skills discovered automatically by
the Antigravity agent system.

## Available Skills

- [code-linter/SKILL.md](code-linter/SKILL.md): Linting, static analysis, TOML
  formatting, PowerShell analysis, security audits, and Radon metrics.
- [pipeline-runner/SKILL.md](pipeline-runner/SKILL.md): Local pipeline execution
  (`./run_pipeline_locally.sh` on Linux/macOS, `./run_pipeline_locally.ps1` on
  Windows) and failure diagnosis.
- [test-runner/SKILL.md](test-runner/SKILL.md): Pytest orchestration, mock
  fixtures, and strict $\\ge 90%$ per-file coverage.
- [audio-restoration-engine/SKILL.md](audio-restoration-engine/SKILL.md): DSP
  filter graphs, ARNNDN speech denoisers, stem separation, and DTW audio
  synchronization.
- [markdown-quality/SKILL.md](markdown-quality/SKILL.md): Read-only Markdown
  formatting validation with `mdformat --check` and linting with `pymarkdown`
  `scan` (MD013).
- [poetry-runtime-and-ci/SKILL.md](poetry-runtime-and-ci/SKILL.md): Dependency
  management, lockfile maintenance, and runtime vs dev isolation.
- [resolve-pr-comments/SKILL.md](resolve-pr-comments/SKILL.md): GitHub CLI
  workflow for PR comments and review threads.
- [prepare-release/SKILL.md](prepare-release/SKILL.md): Version bumping,
  changelog curation, release documentation, and commit message preparation.
- [installer-tester/SKILL.md](installer-tester/SKILL.md): End-user Windows
  installation scripts and CUDA runtime provisioning.
