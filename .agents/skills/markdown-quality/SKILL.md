---
name: markdown-quality
description: >-
  Validate Markdown documentation files with mdformat and pymarkdown scan,
  enforcing MD013 line wrapping and structural rules.
---

# Markdown Quality Skill

Use this skill to validate all Markdown documentation files,
release notes, and agent guidance files across the repository.

## Formatting and Linting Commands

### 1. Validate Markdown Formatting

```powershell
poetry run mdformat --check $(git ls-files '*.md')
```

### 2. Scan and Delint Markdown Rules

```powershell
poetry run pymarkdown --config .pymarkdown.json scan $(git ls-files '*.md')
```

## Markdown Linting Standards

- **Line Length (MD013)**: All prose text in Markdown files must be hard-wrapped
  at $\\le 80$ characters per line.
  - Exceptions: Headings, long code block commands, and wide table rows may be
    up to 200 characters when splitting would reduce readability or copyability.
- **Headings (MD001, MD003, MD025)**:
  - Use ATX-style headings (`# Title`, `## Section`).
  - Keep heading hierarchies strictly incrementing by one level (e.g. `#`
    $\\rightarrow$ `##` $\\rightarrow$ `###`).
  - Exactly one top-level `#` title per document.
- **Lists (MD004, MD007, MD029)**:
  - Unordered lists must use hyphens (`-`).
  - Ordered lists must use `1.` prefixes for all items (or sequential numbers
    consistently).
- **Blank Lines (MD012, MD031, MD032)**:
  - No consecutive blank lines.
  - Fenced code blocks and lists must be surrounded by blank lines.
- **Links & Images**:
  - Always verify that relative file links point to existing repository files.
