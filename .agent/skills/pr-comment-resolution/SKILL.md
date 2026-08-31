# SKILL: PR Comment Resolution (GitHub CLI + MCP)

Use this skill when processing pull request review comments from both CodeRabbit
and human reviewers.

## Objective

Resolve PR discussions with full traceability by combining GitHub CLI (`gh`) and
MCP tools.

## Core Rule

Never resolve or close any PR comment thread until a detailed reply has been
posted to that thread.

Detailed reply minimum:

1. What changed (or why no code change was needed).
1. Exact files/areas affected.
1. Validation performed (lint/tests/manual verification).
1. Any follow-up limitations or risks.

## Required Tooling

- GitHub CLI for direct PR/review inspection and branch-aware workflows.
- MCP Git tools for structured PR comment retrieval and review actions.

Preferred MCP tools:

- `mcp_gitkraken_cli_pull_request_get_comments`
- `mcp_gitkraken_cli_pull_request_create_review`

## Standard Procedure

1. Identify the PR and gather all unresolved comment threads (CodeRabbit + humans).
1. Reproduce/verify each reported issue locally before changing code whenever possible.
1. Implement fixes in small, reviewable commits.
1. Run the local quality gate:

```powershell
./run_pipeline_locally.ps1
```

1. Post a detailed reply per comment thread including:
   - Root cause.
   - Concrete fix.
   - Validation evidence.
   - Any intentional non-fix rationale.
1. Only after the detailed reply is posted, resolve/close the thread.
1. Repeat until no unresolved review comments remain.

## Operational Notes

- Prioritize CodeRabbit findings that indicate correctness, data loss, security,
  or regression risk.
- Do not batch-resolve comments without per-thread replies.
- If a comment is outdated or invalid, still reply in detail before resolving.
- If blocked (missing context, flaky reproduction), reply with attempted steps
  and blocker details, then leave unresolved until actionable.

## Example Command Snippets

```powershell
# Inspect PR details/comments (example)
gh pr view <pr-number> --comments

# Checkout PR branch
gh pr checkout <pr-number>
```

Use MCP equivalents when provider-backed APIs are available, and keep
thread-level accountability identical.
