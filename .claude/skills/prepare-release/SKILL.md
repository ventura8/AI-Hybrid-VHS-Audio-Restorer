---
name: prepare-release
description: >-
  Prepare an AI Hybrid VHS Audio Restorer release on the current
  feature/X.Y.Z branch: resolve the version from the branch, bump
  pyproject.toml, review every change against main, write
  docs/releases/vX.Y.Z.md and vX.Y.Z_github_description.md, refresh the docs
  the change touched, bring every product and CI dependency to its latest
  final release, run the full local quality gate, and amend the release
  commit with a vX.Y.Z title and a detailed body covering every file. Use
  for "prepare the release", "cut vX.Y.Z", "write release notes", or a
  version bump, push, open the PR against main, and watch it for CI and
  review findings until they are resolved.
---

# prepare-release

This skill lives in the workspace skill set shared with every agent, at:

- `.agents/skills/prepare-release/SKILL.md`

Read that file and follow it. It is the single source of truth; this one only
makes it invocable as `/prepare-release` in Claude Code.
