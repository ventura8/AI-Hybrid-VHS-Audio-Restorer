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
