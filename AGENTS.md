# AGENTS

This repository defines agent-specific guidance and reusable skills for
development workflows.

## Primary Agent Instructions

- Global project instructions: `.agent/instructions.md`
- Workflow playbooks: `.agent/workflows/`
- Skills: `.agent/skills/`

## Required Local Quality Gate

Run this command before proposing completion:

```powershell
./run_pipeline_locally.ps1
```

This command runs PowerShell lint, Ruff, Flake8, Pylint, tests with
coverage, and regenerates `assets/coverage.svg`.

It also runs Markdown quality gates:

- `mdformat` (automatic delint/format) on repository docs and agent guidance files.
- `pymarkdown scan` lint checks on the same Markdown target set.

## Dependency Management Rules

- Use Poetry only.
- Runtime installation must be verbose and runtime-only in installer flow.
- CI/local quality environments must install runtime plus dev dependencies.
- Do not use `requirements.txt` or `test-requirements.txt`.

## Core Project Constraints

- Line length: 140.
- No `# noqa`, suppressions, or ignore-based bypasses in linting.
- Keep tests linted locally and in CI.
- Preserve CUDA 13.2 runtime stack.
