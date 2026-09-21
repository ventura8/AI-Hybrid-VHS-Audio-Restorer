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

## Gate Quirks on a Windows Checkout

- `core.autocrlf=true` gives CRLF working copies and `mdformat --check` fails
  on them whatever their content; `git checkout -- file` restores CRLF too.
  Normalise the touched files to LF (`sed -i 's/\r$//'`) before the check;
  git stores LF either way. `Path.write_text` on Windows writes CRLF unless
  it is given `newline="\n"`. (A Bash heredoc turned both escapes into
  spaces the first time this was written: put backslashes in files with the
  editor, not through a heredoc.)
- `mdformat` runs with `wrap = keep`: it joins a code span that was broken
  across lines but never reflows a paragraph, so a sentence inserted into
  a wrapped paragraph must be wrapped by hand to 80 columns, and a code span
  must stay on one line (move it to the start of a line if it is long).
- The CI Markdown job scans every tracked `.md`, including generated ones
  such as `models/*/README.md`; MD034 rejects a bare URL there, so a
  generator writes `<https://...>`.
