---
name: sonarqube-quality-gate
description: >-
  SonarQube Cloud on this repository: how the CI scan is wired, how to read
  and fix its findings without suppressions, how to change the project's
  settings on sonarcloud.io, and what to do when the check fails.
---

# SonarQube Quality Gate

Use this skill whenever the CI check "Static Quality Verification + Automated
Pipeline Validation" fails at the step "SonarQube Cloud Scan", when a
SonarCloud finding (bug, vulnerability, security hotspot, code smell,
duplication, coverage on new code) has to be fixed, or when the project's
analysis settings need a change.

## How it is wired

- Project: `ventura8_AI-Hybrid-VHS-Audio-Restorer` in the organization
  `ventura8` on sonarcloud.io (free plan: the quality gate is the built-in
  "Sonar way"; the new-code definition is "previous version", and CI passes
  the app version from `poetry version -s` as `sonar.projectVersion`).
- `sonar-project.properties` (repository root) is the single configuration:
  sources `modules`, `scripts`, `restore_audio_hybrid.py`; tests `tests`;
  exclusions for local artefacts, weights, corpora and generated reports;
  coverage from `coverage.xml`, tests from `junit.xml` (both written by the
  CI test step); `sonar.qualitygate.wait=true` so the gate blocks the check.
- CI (`.github/workflows/ci.yml`, job `validation`): `actions/checkout` with
  `fetch-depth: 0` (blame and new-code detection), then after the per-file
  coverage gate the pinned `SonarSource/sonarqube-scan-action` with
  `SONAR_TOKEN` from the repository secrets. Pull requests from forks skip
  the step (they cannot read the secret). Automatic Analysis is OFF on
  sonarcloud.io: it would collide with the CI-based analysis and carry no
  coverage.
- There is no local scanner in `run_pipeline_locally.*`; findings are read
  on sonarcloud.io (project → Issues / Security Hotspots / Measures) or in
  the PR decoration once the GitHub app posts it.

## Reading a failure

```bash
gh pr checks <n> --repo ventura8/AI-Hybrid-VHS-Audio-Restorer
gh run view <run-id> --repo ventura8/AI-Hybrid-VHS-Audio-Restorer \
    --log-failed | grep "SonarQube Cloud Scan"
```

- `Not authorized or project not found`: the `SONAR_TOKEN` secret is missing
  or revoked. Only the user creates a token (sonarcloud.io → My Account →
  Security, or the project's "With GitHub Actions" setup page, which shows
  one) and stores it; an agent never handles the token value. The one-command
  path is `.\scripts\set_sonar_token.ps1`: it takes the token at a masked
  prompt, stores the secret through `gh secret set` and re-runs the failed CI
  job of the current branch.
- `QUALITY GATE STATUS: FAILED` with conditions listed: open the analysis
  link in the log, read the failed conditions on new code, fix the code.
- A scanner error about `coverage.xml` / `junit.xml`: the test step did not
  run or its report paths moved; keep `sonar-project.properties` and the
  pytest flags in `ci.yml` in step.

## Fixing findings

- Fix at the source. `# NOSONAR`, `# noqa`-style markers and issue
  "won't fix" / "accept" resolutions on sonarcloud.io are suppressions and
  are forbidden by the Zero Suppression Policy (AGENTS.md, section 4), the
  same as for every local linter.
- A security hotspot must be reviewed with a reason written in the review;
  treat it as a finding until it is either fixed or reviewed as safe with the
  evidence stated.
- Coverage on new code follows the local rule (>= 90 % per file); if Sonar
  reports less on new code, the missing tests are the fix.
- Duplication: extract the shared piece into a helper under `modules/` or
  `scripts/restoration_quality/`; the duplication threshold is on new code.
- An exclusion (a generated file, a vendored file) goes into
  `sonar.exclusions` in `sonar-project.properties` with a comment saying
  why, never into a Sonar UI setting that the repository does not record.

## Changing the project on sonarcloud.io

Settings that live outside the repository (new-code definition, analysis
method, quality gate assignment, GitHub app permissions) are changed on
sonarcloud.io by a logged-in user; an agent may drive the browser for the
user but never enters credentials or tokens. After such a change, record it
here and in `docs/validation.md` ("CI Parity") in the same change.

## Rules learned on this repository

- The import from GitHub generated the project key `<org>_<repo>`; the
  organization key (`ventura8`) is the one in the organization's URLs, not
  the display name.
- Turning Automatic Analysis off must happen before the first CI scan;
  otherwise SonarCloud rejects the CI analysis as a conflicting method.
- The scan reads `sonar-project.properties` from the checkout root; the
  `args` input of the action only adds `-D` properties on top.
- The issue list of a pull request is readable without the UI: a logged-in
  browser session can fetch
  `https://sonarcloud.io/api/issues/search?componentKeys=<project>&pullRequest=<n>&resolved=false&ps=500`
  (the branch and main listings read 0 while only the PR has been analysed).
- "LLM-supplied CLI argument" vulnerabilities (`pythonsecurity:S8705/S8707`)
  point at `argparse` values reaching `subprocess` or a file path unchecked;
  the fix is a validator on the way in (a regex for a code, a membership test
  against the known set, an existing path that does not start with `-`, a
  resolved path required to sit under the repository or the temp dir), not a
  suppression. The first pass fixed exactly these and the gate passed.
- Sonar's "split this composite assertion" (`python:S9073`) and radon's
  cyclomatic gate pull against each other: radon counts every `assert` as a
  branch, so a test may hold at most four assertions after the split. Split
  the test over a shared builder instead (`_board()`, `_merged_variant()`
  style), never merge the asserts back.
- Sonar's `S5857` suggestion `[^"]*` for `refresh_lock.block_meta` is wrong:
  `poetry.lock` markers hold escaped quotes (`python_version >= \"3.9\"`),
  so the negated class stops early and the marker reads as absent. A lock
  value ends at its line, so the greedy `.*` with `$` under `re.M` is the
  equivalent form that Sonar accepts.
- `S1172` (unused parameter) on a function that is one of a table of
  same-shaped callables (the degradations, the model loaders): drop the
  parameter and adapt the call site or the lambda in the registry; a leading
  underscore is not exempt.
- Pull-request scans judge new code only; main's first analysis (the v1.3.3
  merge, 2026-10-05) judged the whole repository as new code and failed the
  gate on 196 issues the PR scans never showed (33 "LLM-supplied CLI
  argument" vulnerabilities in `scripts/`, 48 composite assertions in tests,
  27 `[` tests in shell scripts). Read main's list with
  `api/issues/search?componentKeys=<project>&branch=main&resolved=false&ps=500`
  before declaring the project green, not only the PR's.
- Script arguments go through `scripts/cli_paths.py`. The argparse types
  (`existing_path_arg`, `path_arg`, `language_arg`) refuse bad input early, but
  Sonar's taint analysis starts at `parse_args()` and does not see them: main's
  re-analysis on 2026-10-06 still reported 29 S8705/S8707 vulnerabilities on
  arguments validated that way. What closes them is `confined_path()` applied
  to the value at the use site, right before
  it reaches a file or a subprocess: it refuses option-shaped values, resolves,
  and requires the path to lie inside the repository, the temp directory or a
  root listed in `AI_RESTORE_DATA_ROOTS` (`os.pathsep`-separated; set it to run
  a developer script on tapes or a corpus outside the checkout, for example
  `AI_RESTORE_DATA_ROOTS=D:\Tata`). `scripts/test_installed_executable.py` runs
  standalone in the release workflow, so it carries the same check locally and
  `release.yml` lists the install directory in `AI_RESTORE_DATA_ROOTS`. A new
  script's path argument gets a `cli_paths` type and a `confined_path` at its
  sink.
- Numbers and names reach a subprocess the same way: a start offset, a
  duration or a clip limit goes through `checked_number()` at the call
  (rebuilt with `kind`, range-checked), a variant name through
  `checked_token`, a stream URL through `checked_url`. The last six
  S8705 findings on main (2026-10-06) were exactly these: an `int` argparse
  type does not count, the conversion has to happen where the command is built.
  Use the script's own type (`kind=int` for integer seconds), or the command
  line changes (`30` becomes `30.0`).
- Shell scripts with a bash shebang use `[[ ... ]]` (`shelldre:S7688`) and a
  `*)` branch in every `case` (`S131`).
- Parallel fix agents share one checkout: an agent that runs `git stash` (or
  any index-changing git command) takes every other agent's edits with it.
  Brief fix agents with "read-only git only", and recover with
  `git checkout --` on line-ending-only files, then `git stash pop`.
- `S107` (over 13 parameters) on `_denoise_and_polish_full_audio_step`: the
  post-neural switches arrive as `**stages`, validated against
  `POST_NEURAL_STAGES` and normalised to booleans, so a misspelt switch is
  still a `TypeError` and every caller keeps its keywords.
