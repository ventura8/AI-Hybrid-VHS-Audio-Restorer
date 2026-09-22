# Workflow: Triage a SonarQube Cloud Failure

Use this workflow when the CI check "Static Quality Verification + Automated
Pipeline Validation" fails at the "SonarQube Cloud Scan" step, or when the
user points at a SonarCloud finding. The skill
`.agents/skills/sonarqube-quality-gate/SKILL.md` carries the rules.

## Step 1: Read why it failed

1. `gh pr checks <n> --repo ventura8/AI-Hybrid-VHS-Audio-Restorer`, then
   `gh run view <run-id> --log-failed | grep "SonarQube Cloud Scan"`.
1. `Not authorized or project not found`: the `SONAR_TOKEN` secret is
   missing or revoked. Ask the user to create a token on sonarcloud.io and
   store it (`gh secret set SONAR_TOKEN --repo ...`); never handle the token
   value. Re-run the failed job afterwards.
1. `QUALITY GATE STATUS: FAILED`: open the analysis link the log prints and
   list the failed conditions on new code.

## Step 2: Fix at the source

1. Bugs, vulnerabilities, code smells: fix the code; no `# NOSONAR`, no
   "won't fix" / "accept" on sonarcloud.io (the Zero Suppression Policy
   applies to Sonar like to every local linter).
1. Security hotspots: fix, or review as safe with the evidence written in
   the review.
1. Coverage on new code under the floor: write the tests (the local rule is
   > = 90 % per file, the same as Sonar's condition on new code).
1. Duplication on new code: extract the shared piece into a helper.
1. A generated or vendored file: add it to `sonar.exclusions` in
   `sonar-project.properties` with a reason, never as a UI-only setting.

## Step 3: Verify and hand over

1. Run the full local gate (`run_pipeline_locally.*`), commit, push; the CI
   scan re-runs and its quality gate must pass.
1. A settings change on sonarcloud.io (new-code definition, analysis method,
   quality gate) is recorded in the skill and in `docs/validation.md`
   ("CI Parity") in the same change.
