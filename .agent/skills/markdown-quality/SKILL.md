# SKILL: Markdown Quality Gates

Use this skill when editing Markdown docs, instructions, release notes, or
workflow guidance files.

## Objective

Keep Markdown content consistently formatted and lint-clean in both local and CI
flows.

## Required Commands

```powershell
# Verify Markdown formatting locally
poetry run mdformat --check $(git ls-files '*.md')

# Lint Markdown content after formatting
poetry run pymarkdown --config .pymarkdown.json scan $(git ls-files '*.md')
```

## CI and Pipeline Parity

- Local quality gate runs `mdformat --check` before Markdown linting.
- CI runs `mdformat --check` and `pymarkdown scan` on the same target set.
- Keep the Markdown target paths aligned between `run_pipeline_locally.ps1` and `.github/workflows/ci.yml`.

## Rules

- Use `mdformat --check` to validate formatting; do not modify files in the
  quality gate.
- Avoid adding rule suppressions unless explicitly approved.
- Keep docs readable and consistent with project instructions in
  `.agent/instructions.md` and `AGENTS.md`.
