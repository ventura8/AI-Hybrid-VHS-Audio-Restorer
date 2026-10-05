---
name: prepare-release
description: >-
  Prepare an AI Hybrid VHS Audio Restorer release on the current
  feature/X.Y.Z branch: resolve the version from the branch, bump
  pyproject.toml, review every change against main, write
  docs/releases/vX.Y.Z.md and vX.Y.Z_github_description.md, refresh the docs
  the change touched, bring every product and CI dependency to its latest
  final release, run the full local quality gate, and amend the release
  commit with a vX.Y.Z title and a detailed body covering every file. Use for
  "prepare the release", "cut vX.Y.Z", "write release notes", or a version
  bump, push, open the PR against main, and watch it for CI and review
  findings until they are resolved.
---

# Prepare Release

A release here is one squash-style commit on `feature/X.Y.Z` titled
`vX.Y.Z: <theme>`, merged to `main` by PR, then tagged `vX.Y.Z`. The tag push
triggers `.github/workflows/release.yml`, which re-runs the quality gate and
builds the `.deb`/`.rpm`/`.pkg`/Windows installers. This skill pushes the
branch and opens the PR; it never merges or tags, which stay with the user.

## 0. Preflight: read the repository's own instructions

Before touching anything, read in full, in this order, and hold every rule
they state for the rest of the release:

- `AGENTS.md` -- the invariants (quality gates, documentation-in-the-same-pass,
  dependency rules, hardware validation triggers) and the skills index.
- `.agents/skills/` -- every `SKILL.md`, not only this one: `code-linter`,
  `test-runner`, `markdown-quality`, `poetry-runtime-and-ci`,
  `hardware-validation`, `coderabbit-review-wave` and `resolve-pr-comments`
  each own a step this release passes through.
- `.agent/workflows/` -- the playbooks, `run_full_quality_gate.md` above all.

If a step below disagrees with one of those files, the file wins and this
skill is the one to correct.

## 1. Resolve the version from the branch

```bash
branch="$(git branch --show-current)"
[[ "$branch" =~ ^feature/v?([0-9]+\.[0-9]+\.[0-9]+)$ ]] || { echo "not a release branch: '$branch' (want feature/X.Y.Z)"; exit 1; }
version="${BASH_REMATCH[1]}"
prev="$(git describe --tags --abbrev=0 main)"
echo "releasing $version, previous tag $prev"
```

Both `feature/1.3.1` and `feature/v1.3.1` are in use historically; accept
either, and nothing else: a `hotfix/` or `main` checkout must stop here,
before any file is touched. Never take the version from `pyproject.toml` or
from the user's message when the branch disagrees: the branch is the source
of truth, and a mismatch is something to raise, not silently resolve.

## 2. Sync the version pin

`pyproject.toml` `version = "X.Y.Z"` is the only code-side pin. Grep the
previous version to catch anything else that hard-codes it:

```bash
.venv/bin/python - "$version" <<'PY'
import re, sys
p = "pyproject.toml"; s = open(p).read()
s, n = re.subn(r'^version = "[^"]*"', f'version = "{sys.argv[1]}"', s, count=1, flags=re.M)
assert n == 1; open(p, "w").write(s)
PY
grep -rn --include='*.py' --include='*.toml' --include='*.yaml' --include='*.sh' --include='*.ps1' "${prev#v}" . | grep -v '\.venv\|docs/\|coverage'
```

Docs that quote an older version as history (benchmarks, research notes) are
correct as they stand; do not rewrite them.

## 3. Review everything since main

```bash
base="$(git merge-base HEAD main)"
git log --format='%h %s' "$base..HEAD"
git diff --stat "$base...HEAD"
git diff "$base...HEAD"           # committed on the branch
git diff HEAD                     # staged and unstaged, tracked files
git status --porcelain            # paths; untracked files show as ??
git ls-files --others --exclude-standard   # every untracked file: open each
```

Read the full diff, not just the stat, and the working tree too: the
committed range misses staged and unstaged edits, and `status` names paths
without their contents, so read `git diff HEAD` and open every untracked file
before writing a line of the release notes. Sort what you find into release
work and unrelated work now; the amend in step 8 stages only the former.

The release notes must describe what the code does now and why it changed;
they are written from the diff and the commit bodies, never from the branch
name or a guess.

## 4. Write the release documents

Two files, both mandatory, in `docs/releases/`. Read the previous release's
pair first and match its voice: concrete, measured, no marketing.

- `vX.Y.Z.md` is the full record. Sections used by every prior release:
  `## Highlights`, one `##` per theme (what changed, why, what was measured),
  `## Quality gates and testing` (test count, coverage, gates that ran),
  `## Upgrade Notes` (behaviour or config a user could notice), and a closing
  `**Full Changelog**` compare link
  `https://github.com/ventura8/AI-Hybrid-VHS-Audio-Restorer/compare/vPREV...vX.Y.Z`.
- `vX.Y.Z_github_description.md` is the GitHub Release body. Its heading is
  the bare version, `# vX.Y.Z`, and so is the heading of `vX.Y.Z.md`: the
  release workflow titles the GitHub Release with the bare tag, and the body
  must not restate a longer name under it. Then `## Highlights`,
  `## Upgrade Notes`, the compare link, and (when a release
  builds on `cathar`) the standing `## Thanks` to
  [vbasky/cathar](https://github.com/vbasky/cathar). Under a screen.

A patch release that fixes one thing gets a short pair; do not pad. Security
fixes state the sink, the trigger, and what is now refused, so a reader can
judge whether they were exposed.

## 5. Refresh the documentation the change touched

AGENTS.md's rule: every code or config change updates its docs in the same
pass. Check, at least:

- `docs/configuration.md` for any setting whose accepted values changed.
- `README.md` and `AGENTS.md` where the mode matrix, defaults, or gates moved.
- `docs/architecture.md` / `docs/pipeline_logic.md` when a stage was added.

Prose wraps at 80 columns (MD013); tables and code may reach 200.

## 6. Bring dependencies to their latest final release

A release ships on current dependencies: every product and dev pin in
`pyproject.toml`, and every action and tool pin in `.github/workflows/`.

```bash
# Latest stable on PyPI for every pin (pre-releases, yanked files, and releases
# that do not support this project's Python excluded)
.venv/bin/python - <<'PY'
import json, re, tomllib, urllib.request
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version
t = tomllib.load(open("pyproject.toml", "rb"))
py = Version("3.12.0")   # requires-python is >=3.12,<3.13: a 3.13-only release is not an update
groups = {"main": t["tool"]["poetry"]["dependencies"]}
groups |= {g: v["dependencies"] for g, v in t["tool"]["poetry"]["group"].items()}
for g, deps in groups.items():
    for name, spec in deps.items():
        if name == "python": continue
        rel = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json"))["releases"]
        # yanking is per file: a release stays eligible while any file is installable;
        # and only while some installable file's requires_python admits our interpreter
        def admits(spec):
            try: return py in SpecifierSet(spec or "")
            except InvalidSpecifier: return True   # malformed metadata: do not exclude on it
        def ok(files):
            return any(admits(x.get("requires_python")) for x in files if not x.get("yanked"))
        vs = [Version(v) for v, f in rel.items() if ok(f)]
        latest = max(v for v in vs if not v.is_prerelease)
        # a list-valued constraint (torch and friends) is one entry per platform: check every one
        entries = spec if isinstance(spec, list) else [spec]
        for e in entries:
            e = {"version": e} if isinstance(e, str) else e
            cur = re.sub(r"[^0-9.]", "", e["version"].split("+")[0])
            where = " ".join(x for x in (e.get("source", ""), e.get("markers", "")) if x)
            flag = "" if str(latest) == cur else "  <-- UPDATE"
            print(f"{g:4} {name:20} {e['version']:16} latest {latest}{flag}  {where}")
PY
```

Rules that the pins encode, and that a bump must respect:

- `torch`, `torchvision` and `torchaudio` are one matched set from the
  `pytorch-cu130` index (the comment above the pins says why). Bump them
  together to the newest trio the index carries; per package, the newest
  cp312 Linux wheel is:

  ```bash
  curl -s https://download.pytorch.org/whl/cu130/torch/ | .venv/bin/python -c '
  import re, sys
  from packaging.version import Version
  print(max({Version(v) for v in re.findall(r"torch-([0-9.]+)\+cu130-cp312", sys.stdin.read())}))'
  ```

  `torchaudio` no longer declares a torch pin, so its newest release is
  correct even when its number trails torch's.

- `onnxruntime-gpu` is `~` (tilde) pinned; keep the operator, bump the number.

- A `pip-audit` finding with no fix version is still a finding: look the
  advisory up (`https://api.osv.dev/v1/vulns/<ID>`) for `last_affected`, and
  bump past it. Never add `--ignore-vuln` to the gate.

Then relock, install, and let the gate prove the result. Never run a bare
`poetry lock`: re-resolving from scratch across the torch
multiple-constraint entries and the cu130 index has exceeded six hours (see
the note above "Verify Poetry Lockfile" in `.github/workflows/ci.yml`). As
of Poetry 2.4.3 a targeted `poetry update <pkg>` stalls at the same point
-- round 79 of its override loop, `Duplicate dependencies for torch` --
even for one pure-Python package, so try it with a cap and fall back
(the cap is Python's, so it behaves the same on macOS, which ships no
`timeout`):

```bash
# package names go after the "-" as ordinary arguments; sys.argv[1:] receives them
POETRY_REQUESTS_TIMEOUT=900 .venv/bin/python - "accelerate" "tqdm" <<'PY'
import subprocess, sys
try:
    subprocess.run([sys.executable, "-m", "poetry", "update", "--lock", *sys.argv[1:]], timeout=600, check=True)
except subprocess.TimeoutExpired:
    sys.exit("poetry update did not finish in 600 s: use the lock-patch fallback")
PY
```

Every command in this skill is portable across the Linux and macOS shells
the runner supports: Python or `git` does the editing, never GNU-only
`sed -i`, `sort -V` or `timeout`.

Fallback, for a bump whose dependency metadata is unchanged (compare
`requires_dist` and `requires_python` between the two versions on
`https://pypi.org/pypi/<pkg>/<ver>/json`; an extras-only change is applied
by hand to `[package.extras]`): rewrite that package's `version` and
`files` (filename + sha256 for every non-yanked file) in `poetry.lock`,
then recompute the content-hash with Poetry's own API --
`Factory().create_poetry(".").locker._get_content_hash()` -- into
`[metadata]`. Verify with `poetry check --lock` and a hash-checked
`poetry install --with dev,ml`. A bump that changes the dependency graph
(torch, onnxruntime, transformers) is not eligible: defer it, keep its pin
at the locked version, and say so in the release notes with the reason.

```bash
.venv/bin/python -m poetry check --lock
.venv/bin/python -m poetry install --with dev,ml
```

`scripts/refresh_lock.py` does the fallback mechanically for every package
(the safe, metadata-identical pass by default; `--graph name==version` rebuilds
a block whose dependency metadata changed and adds what it needs;
`--add name==version:group` for a new direct dependency), recomputes the
content-hash and writes `experiments/lock_refresh_report.json`; run it without
`--apply` first and read what it deferred. The cu130 torch trio moves only as a set,
and only when the cu130 index carries matching torch, torchvision and
torchaudio wheels.

CI pins: for each `uses:` in `.github/workflows/*.yml`, the line above it
names the release; update both the SHA and the comment to the latest release
(`gh api repos/<owner>/<repo>/releases/latest --jq .tag_name`, then resolve
the tag to its commit SHA; annotated tags need one more hop through
`git/tags/<sha>`). `POETRY_VERSION` in both workflows and the
`poetry==X.Y.Z` in `install_dependencies.sh` / `.ps1` move together.

Record every bump in the release notes: the security-motivated ones with
their advisory ID, the rest as one list.

## 7. Run the full quality gate

On Linux and macOS:

```bash
./run_pipeline_locally.sh
```

On Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_pipeline_locally.ps1
```

This is CI parity: ruff, black, isort, taplo, flake8, pylint, bandit,
pip-audit, radon (CC grade A, MI grade A), `mdformat --check`, `pymarkdown`,
pytest with the per-file >= 90% floor, and the coverage badge at
`assets/coverage.svg`. Fix and rerun until it passes; a release commit never
ships a red gate. `mdformat` is check-only in the gate, so run
`.venv/bin/python -m mdformat docs/releases/vX.Y.Z*.md` on the new files if
it complains. Commit the regenerated badge if it changed.

## 8. Amend the release commit

The branch carries one release commit. Stage every file the release touched
-- code, tests, docs, release notes, `pyproject.toml`, `poetry.lock`, the
coverage badge, skill and workflow files -- by explicit path, from the list
made in step 3, and amend that commit with the release title and a detailed
description. Never `git add -A`: a working tree that still holds changes
sorted as unrelated in step 3 is a stop, not a prompt -- leave them
unstaged, tell the user what they are, and amend only the release paths.
When the user asked for the release to be prepared, the amend itself is
part of the job; confirm first only if the branch has more than one commit
(squash or keep?).

- Title: `vX.Y.Z: <primary theme in one line>` -- detailed, never the bare
  version; the PR takes the same title (AGENTS.md, "Pull Request
  Conventions").
- Body: prose or bullets grouped by theme, in the same register as the release
  notes. For a fix: what was wrong, how it is fixed, how it is tested. Then a
  dependency section listing every bump, a docs section, and the gate result
  (test count, coverage). A reader of `git log` alone should learn everything
  the release notes say, in less space.
- Trailer: a `Co-Authored-By:` line only for the agent that actually did the
  work, in its own identity (Claude Code adds
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`; another agent
  adds its own; a human-prepared release adds none). Never a line for a tool
  that was not involved.

```bash
# The exact files sorted as release work in step 3 -- every one of them, and
# nothing else. This list is an example; build yours from that review.
release_paths=(
  modules/config.py modules/processing.py
  tests/unit/test_config.py tests/unit/test_processing.py
  docs/configuration.md docs/releases/vX.Y.Z.md docs/releases/vX.Y.Z_github_description.md
  pyproject.toml poetry.lock assets/coverage.svg
  .agents/skills/prepare-release/SKILL.md
)
msgfile="$(mktemp)"                # write the title, body and trailer here
git add -- "${release_paths[@]}"
git status --porcelain             # anything not "M  <release path>" stays unstaged and is reported
git commit --amend -F "$msgfile"
git show --stat HEAD
```

## 9. Push, open the PR, watch it

The PR body is the GitHub release description with its `# ...` title line
removed, followed by the attribution footer of the agent that did the work
(Claude Code's is `🤖 Generated with [Claude Code](https://claude.com/claude-code)`;
another agent uses its own; a human omits the footer):

```bash
prbody="$(mktemp)"
tail -n +2 "docs/releases/v${version}_github_description.md" > "$prbody"
printf '\n%s\n' "$attribution_footer" >> "$prbody"    # set by the agent, or empty
git push -u origin "$(git branch --show-current)" --force-with-lease
gh pr create --base main --title "v${version}: <theme>" --body-file "$prbody"
```

Then watch it until it is green and the review wave is answered:

- CI: `gh pr checks --watch`; a red check is fixed on the branch, amended
  into the release commit, and force-pushed with lease.
- Reviews: CodeRabbit and other reviewers comment within minutes of the
  push. Handle them with the `coderabbit-review-wave` and
  `resolve-pr-comments` skills: verify every finding, fix or refute, reply
  on the thread, resolve it. Re-run the gate after fixes.

## 10. Hand-off

Report: the version, the PR URL, the gate result, what the reviewers raised
and how each thread was closed, and the two steps left to the user: merge,
then `git tag vX.Y.Z && git push origin vX.Y.Z` to trigger the release build.
