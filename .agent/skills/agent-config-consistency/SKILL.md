# SKILL: Agent Configuration Consistency

Use this skill when editing agent instructions, skills, workflows, or Copilot guidance files.

## Objective

Keep all agent control files consistent so AI behavior remains predictable.

## Scope

- `AGENTS.md`
- `.github/copilot-instructions.md`
- `.agent/instructions.md`
- `.agent/skills/**/SKILL.md`
- `.agent/workflows/*.md`

## Consistency Requirements

1. Commands referenced in agent docs must exist and be current.
2. Quality-gate command must be `./run_pipeline_locally.ps1`.
3. Dependency policy must stay Poetry-only and align with runtime/dev split.
4. Lint policy must keep max line length 140 and prohibit suppression bypasses.
5. CUDA runtime guidance must remain CUDA 13.2 aligned.

## Workflow Expectations

- Update indexes when adding/removing skills/workflows.
- Avoid contradictory requirements between files.
- Keep examples aligned with current CI/local pipeline behavior.

## Validation

- Run targeted lint/tests if code/scripts changed.
- Run full local quality gate before completion when implementation changed:

```powershell
./run_pipeline_locally.ps1
```
