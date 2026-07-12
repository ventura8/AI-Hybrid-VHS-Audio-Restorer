# Instructions

## Purpose

This document provides concise contributor instructions for day-to-day changes.

## Before Coding

1. Read project context in docs/project_overview.md and docs/architecture.md.
2. Ensure environment is healthy via lock check and local pipeline.

## During Changes

- Keep line length at 140.
- Do not add lint suppressions.
- Keep tests linted and updated with behavior changes.
- Preserve audio quality and resume semantics.

## After Changes

Run:

```powershell
./run_pipeline_locally.ps1
```

Confirm:
- Lint and tests pass.
- Total coverage threshold is met (>= 90%).
- Per-file coverage threshold is met (>= 90% for each measured file).
- Coverage badge has been regenerated.

## Dependency Rules

- Use Poetry only.
- Keep runtime dependencies in main set.
- Keep testing/tooling dependencies in dev set.
- Keep lock file updated when dependencies change.

## CI/CD Rules

- Keep local and CI commands aligned.
- Keep action versions current and pinned.
- Keep test linting enabled in CI.
