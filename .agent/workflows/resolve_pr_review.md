# Workflow: Resolve Pull Request Reviews

Use this workflow to systematically inspect, evaluate, and resolve pull request
review threads and automated reviewer comments.

## Step 1: Query Active Review Comments

Retrieve unresolved review threads using GitHub GraphQL API:

```powershell
gh api graphql --paginate -F owner="{owner}" -F repo="{repo}" -F pr={pr_number} -f query='
query($owner: String!, $repo: String!, $pr: Int!, $endCursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $pr) {
      reviewThreads(first: 50, after: $endCursor) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
          isResolved
          comments(first: 10) {
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              body
              path
              line
            }
          }
        }
      }
    }
  }
}'
```

`--paginate` follows `pageInfo.endCursor` until `hasNextPage` is `false`, so
every review thread is processed rather than only the first page of 50.

If a thread reports `comments.pageInfo.hasNextPage`, fetch its remaining comments
with a follow-up query scoped to that single thread id (each thread has its own
independent comment cursor).

Filter and process only threads where `isResolved` is `false`.

## Step 2: Categorize and Implement Fixes

For each review comment:

1. **Assess Validity**: Determine whether the suggestion improves code quality,
   fixes a bug, or breaks existing architecture/invariants.

1. **Apply Changes**: Edit code cleanly without adding bypass suppressions
   (`noqa`, `# pylint: disable`).

1. **Validate**: Run targeted tests:

   ```powershell
   poetry run pytest
   ```

## Step 3: Reply and Mark as Resolved

Post a clear response to the thread and resolve it via GraphQL API:

```powershell
# Reply to thread
gh api graphql -F threadId="{thread_id}" -F body="Fixed in commit <hash>: <description of fix>." -f query='
mutation($threadId: ID!, $body: String!) {
  addPullRequestReviewThreadReply(input: {pullRequestReviewThreadId: $threadId, body: $body}) {
    comment { id }
  }
}'

# Mark thread resolved
gh api graphql -F threadId="{thread_id}" -f query='
mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread { isResolved }
  }
}'
```

## Step 4: Final Pipeline Verification

Execute the full quality gate to ensure no regressions:

```bash
# On Linux / macOS:
./run_pipeline_locally.sh

# On Windows:
./run_pipeline_locally.ps1
```
