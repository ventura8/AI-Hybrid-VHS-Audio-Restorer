# SKILL: Poetry Runtime and CI Parity

Use this skill when changing dependencies, installer logic, or CI workflows.

## Objective

Enforce Poetry-first dependency management with deterministic lockfile installs and local/CI parity.

## Dependency Policy

- Runtime dependencies belong in Poetry main dependencies.
- Test and tooling dependencies belong in Poetry dev dependencies.
- Keep `poetry.lock` committed and up to date.
- All Poetry installs in scripts/workflows must include verbose mode (`-v`).

## Installer Policy

- Installer sets up `.venv` and installs runtime dependencies only.
- Runtime-only exceptions that cannot be solved via Poetry resolver can be installed explicitly after main sync.

## CI Policy

- CI setup action installs runtime + dev dependencies for lint/test jobs.
- Local script and CI workflow command order must remain aligned.
- Keep GitHub Actions versions current and pinned to stable release tags.

## CUDA Policy

- Preserve final CUDA 13.2 wheel strategy and corresponding source index for torch packages.
