# Antigravity Workspace Skills

This directory contains modular, executable skills discovered automatically by
the Antigravity agent system.

## Available Skills

- [code-linter/SKILL.md](code-linter/SKILL.md): Linting, static analysis, TOML
  formatting, PowerShell analysis, security audits, and Radon metrics.
- [pipeline-runner/SKILL.md](pipeline-runner/SKILL.md): Local pipeline execution
  (`./run_pipeline_locally.sh` on Linux/macOS, `./run_pipeline_locally.ps1` on
  Windows) and failure diagnosis.
- [test-runner/SKILL.md](test-runner/SKILL.md): Pytest orchestration, mock
  fixtures, and strict $\\ge 90%$ per-file coverage; the quirks of running
  pytest beside other agents (the badge only a coverage run's own reports
  update, `--cov` by path, never a directory of stray `.py` files as the
  working directory).
- [audio-restoration-engine/SKILL.md](audio-restoration-engine/SKILL.md): DSP
  filter graphs, ARNNDN speech denoisers, stem separation, and DTW audio
  synchronization; the loudnorm linear rule and its fallback, the
  polarity-inverted pair check, the engine's event log
  (`modules/event_log.py`, `AI_RESTORE_EVENT_LOG`), and the neural-stage
  cache and its deny-by-default key (`modules/stage_cache.py`,
  `AI_RESTORE_STAGE_CACHE`).
- [markdown-quality/SKILL.md](markdown-quality/SKILL.md): Read-only Markdown
  formatting validation with `mdformat --check` and linting with `pymarkdown`
  `scan` (MD013).
- [poetry-runtime-and-ci/SKILL.md](poetry-runtime-and-ci/SKILL.md): Dependency
  management, lockfile maintenance, and runtime vs dev isolation.
- [resolve-pr-comments/SKILL.md](resolve-pr-comments/SKILL.md): GitHub CLI
  workflow for PR comments and review threads.
- [prepare-release/SKILL.md](prepare-release/SKILL.md): Version bumping,
  changelog curation, release documentation, and commit message preparation.
- [installer-tester/SKILL.md](installer-tester/SKILL.md): End-user Windows
  installation scripts and CUDA runtime provisioning.
- [hardware-validation/SKILL.md](hardware-validation/SKILL.md): Deterministic
  Piper fixture generation, accelerator audit, and opt-in hardware validation.
- [output-quality-harness/SKILL.md](output-quality-harness/SKILL.md): The "AI
  human ear": listener-like scoring, calibration, real-tape tuning and the
  self-driving loop that asks the user only at its plateau. Ear v3 adds the
  capture profile and the balance, absolute-sibilance, pause-residual,
  gain-ride, sync and time-base readings (R12,
  `restoration_quality/timebase_metrics.py`: DTW holds, 40 ms plateaus and
  rate jumps, hard-gated, uncalibrated), and these scripts:
  - `scripts/audibility_check.py`: is a candidate audibly different from
    the incumbent (hash, null test, noise-to-mask ratio; "inaudible" is a
    tie, "audible" proves nothing);
  - `scripts/listen_ab.py` (`python -m scripts.listen_ab`): ABX, pairwise
    and blend listening on a localhost-only page, answers written to the
    verdict ledger `assets/quality_calibration/verdicts.jsonl`;
  - `scripts/reward_noise_floor.py` (`python -m scripts.reward_noise_floor`):
    the test-retest noise floor of every reading, which the group reward
    (`restoration_quality/reward.py`) uses as its tie band and the loop as
    its hiss readings' floor and the vetoes' move threshold
    (`--jitter-floors`). A v3 round needs
    its report on the shipped outputs first
    (`experiments/reward/noise_floor.json`, every family the grid's vetoes
    name, `--repeats 2`): the loop refuses to start when a veto has no
    floor (its family missing from `--families`) or a floor of 0 (no
    jitter measured, as CER at `--repeats 1`).
    `--benign-pair SOURCE OUTPUT_A OUTPUT_B [LEDGER_ID]` measures the
    vetoes' floors on pairs a blind listener could not tell apart, in a
    report of their own (`experiments/reward/noise_floor_benign.json`;
    points pool only with `--pool`);
  - `scripts/variation_precheck.py` (`python -m scripts.variation_precheck`):
    plan 3.2a's pre-check on finished loop rounds, the per-tape oracle
    against the single best candidate, each pick through the audibility
    check and the listener's unheard level (`--heard-db`);
  - `scripts/autotune_scoring.py` (a module of the self-driving loop, not a
    script): the families a grid reads (`--families auto`), the scorer
    processes, and the two-stage scoring (learned vetoes on contenders only);
  - `scripts/autotune_material.py` (a module of the self-driving loop): the
    music-only knobs left out when the incumbent's render log reads every
    tape as speech (held partials under the music floor);
  - `scripts/autotune_guards.py` (a module of the self-driving loop, not a
    script): the v3 grids' guards, the verdict-reversal refusal, the
    audibility tie (stored verdicts keyed on `auditory.VERDICT_RULE`) and
    the learned median veto;
  - `scripts/pair_alignment.py` (a module under every paired reading, not a
    script): the lag by GCC-PHAT on the loudest spans voting, abstention
    and polarity; `AI_RESTORE_ALIGN_METHOD=xcorr_head` (or
    `--align-method` on `validate_restoration.py` and
    `measure_tradeoff.py`) reproduces scores stored before 2026-10-11.
  - `scripts/autotune_judging.py` (a module of the self-driving loop): the
    judging resolution, the tie band and the dead zones' floor from the
    Session 0 benign pairs, the hiss exemption (hiss readings take only
    the scorer's jitter floor, `--jitter-floors`), the per-reading distance
    table and the round's label in `log.md`, and the family-wise learned
    veto (sign test across tapes, Holm across readings);
  - `scripts/rejudge_autotune.py` (`python -m scripts.rejudge_autotune`):
    the stored rounds of finished loops re-judged offline from their score
    files, old rule against new, with the vetoes' null simulation
    (`--null-draws`); CPU only, decides nothing by itself.
- [sonarqube-quality-gate/SKILL.md](sonarqube-quality-gate/SKILL.md): SonarQube
  Cloud on this repository: the CI scan, reading and fixing findings without
  suppressions, project settings on sonarcloud.io, failure triage.

Every session that learns a rule, a measured fact or a tool quirk writes it
into the skill that owns the topic in the same change (`AGENTS.md`, section
9).
