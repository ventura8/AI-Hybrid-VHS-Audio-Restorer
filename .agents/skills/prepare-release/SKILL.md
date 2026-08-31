---
name: prepare-release
description: >-
  Prepare AI Hybrid VHS Audio Restorer release documentation, synchronize
  version pins across pyproject.toml and docs, review all changes vs merge base,
  generate docs/releases/vX.Y.Z.md and
  docs/releases/vX.Y.Z_github_description.md, and amend the commit title and
  description with a structured summary.
---

# Prepare Release Skill

Use this skill when preparing a new version release, updating release
documentation, synchronizing version pins, and drafting or amending release
commit messages.

## Agent Mandate

1. **Version from the Current Branch Only**:
   - Parse `git branch --show-current` (e.g. `feature/v1.1.0` or
     `release/v1.1.0` $\\rightarrow$ `1.1.0`).
   - Sync the resolved version to `version = "X.Y.Z"` in `pyproject.toml`.
1. **Review ALL Changes vs Merge Base**:
   - Inspect all committed, staged, and unstaged changes against the merge base
     (`HEAD` vs `main`/`master`).
1. **Generate Release Documentation**:
   - Write `docs/releases/vX.Y.Z.md` (comprehensive release document).
   - Write `docs/releases/vX.Y.Z_github_description.md` (GitHub Release body).
1. **Update Documentation Indexes**:
   - Update `README.md`, `AGENTS.md`, and any user-facing configuration docs.
1. **Amend Commit Message (When User Confirms)**:
   - Request user confirmation before amending git commits.

## Version Extraction

```powershell
$branch = git branch --show-current
$version = $branch -replace '^.*/', '' -replace '^v', ''
if ($version -notmatch '^\d+\.\d+\.\d+$') {
    throw "Invalid semantic version '$version' extracted from branch '$branch'."
}
```

## Diff & Change Gathering

```powershell
$base = git merge-base HEAD main
git log --oneline "$base..HEAD"
git diff --stat "$base...HEAD"
git diff "$base...HEAD"
git diff --staged
git diff
git status --porcelain
# Enumerate and inspect untracked files
Get-ChildItem -Recurse -File -Force |
  Where-Object { $_.FullName -notmatch '[\\/]\.git[\\/]' } |
  Where-Object { (git status --porcelain $_.FullName) -match '^\?\?' }
```

## Output Release Files

1. `docs/releases/vX.Y.Z.md`: Full release documentation, mode matrix,
   architectural enhancements, testing results, and validation.
1. `docs/releases/vX.Y.Z_github_description.md`: GitHub Release markdown
   starting with `# AI Hybrid VHS Audio Restorer vX.Y.Z - <Theme Title>`.

## Commit Message Format

- **Title Pattern**:
  `vX.Y.Z: <primary theme and feature highlights>`
- **Body**: Structured thematic bullet points detailing:
  - Added features and restoration modes.
  - Architecture and modularization.
  - Quality gates, testing, and complexity scores.
  - Documentation and agent guidance updates.
