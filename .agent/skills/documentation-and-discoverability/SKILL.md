# SKILL: Documentation and Discoverability

Use this skill when adding features, changing behavior, or introducing new project guidance.

## Objective

Keep documentation accurate, discoverable, and synchronized with implementation.

## Scope

- Project docs in `docs/`.
- Top-level navigation in `README.md`.
- Agent-facing navigation in `.agent/instructions.md` and `.agent/skills/README.md`.

## Required Updates

1. Update or create relevant docs for behavior/config changes.
1. Add discoverability links in `README.md` when new docs are introduced.
1. Keep `.agent/instructions.md` documentation index aligned with available docs.
1. Ensure guidance does not conflict across docs.

## Rules

- Prefer concise, task-oriented sections.
- Keep technical constraints identical across all docs.
- Do not leave orphan docs that are not linked from an index.

## Validation

- Check all new or updated docs are linked from at least one index file.
- Run local quality gate when code/scripts changed:

```powershell
./run_pipeline_locally.ps1
```
