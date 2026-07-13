# SKILL: Markdown Quality Gates

Use this skill when editing Markdown docs, instructions, release notes, or workflow guidance files.

## Objective

Keep Markdown content consistently formatted and lint-clean in both local and CI flows.

## Required Commands

```powershell
# Auto-delint and normalize formatting locally
poetry run mdformat README.md AGENTS.md .agent docs tests/README.md

# Lint Markdown content after formatting
poetry run pymarkdown scan README.md AGENTS.md .agent docs tests/README.md
```

## CI and Pipeline Parity

- Local quality gate runs `mdformat` (auto-delint) before Markdown linting.
- CI runs `mdformat --check` and `pymarkdown scan` on the same target set.
- Keep the Markdown target paths aligned between `run_pipeline_locally.ps1` and `.github/workflows/ci.yml`.

## Rules

- Prefer automatic formatting first, then lint.
- Avoid adding rule suppressions unless explicitly approved.
- Keep docs readable and consistent with project instructions in `.agent/instructions.md` and `AGENTS.md`.
