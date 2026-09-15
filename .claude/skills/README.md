# Claude Code Skill Catalog

Each directory here holds a one-paragraph `SKILL.md` that points at the
matching skill in `.agents/skills/`, which is the single source of truth
shared by every agent that works in this repository. Edit the skill there;
never copy its content into this directory.

To add a skill: create it under `.agents/skills/<name>/SKILL.md`, then add a
pointer directory here with the same name so Claude Code can invoke it as
`/<name>`.
