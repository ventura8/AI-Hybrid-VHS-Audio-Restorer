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
  gain-ride and sync readings, and these scripts:
  - `scripts/audibility_check.py`: is a candidate audibly different from
    the incumbent (hash, null test, noise-to-mask ratio; "inaudible" is a
    tie, "audible" proves nothing);
  - `scripts/listen_ab.py` (`python -m scripts.listen_ab`): ABX, pairwise
    and blend listening on a localhost-only page, answers written to the
    verdict ledger `assets/quality_calibration/verdicts.jsonl`;
  - `scripts/reward_noise_floor.py` (`python -m scripts.reward_noise_floor`):
    the test-retest noise floor of every reading, which the group reward
    (`restoration_quality/reward.py`) uses as its tie band. A v3 round needs
    its report on the shipped outputs first
    (`experiments/reward/noise_floor.json`, every family the grid's vetoes
    name, `--repeats 2`): the loop refuses to start when a veto has no
    floor (its family missing from `--families`) or a floor of 0 (no
    jitter measured, as CER at `--repeats 1`).
    `--benign-pair SOURCE OUTPUT_A OUTPUT_B [LEDGER_ID]` measures the
    vetoes' floors on pairs a blind listener could not tell apart, in a
    report of their own (`experiments/reward/noise_floor_benign.json`;
    points pool only with `--pool`). `--transforms hiss` scores ear v4's
    same-percept transforms (`scripts/same_percept.py`: +-0.1 dB, a
    0.37-sample delay, a soxr round trip, 16-bit requantisation, fresh
    realisations of the floor), the tie bands of the hiss readings
    (`experiments/reward/noise_floor_hiss.json`; each tie the reading's
    `tie_floor`, the widest transform's p95), never a benign pair;
  - `scripts/check_hiss_fixtures.py` (`python -m scripts.check_hiss_fixtures`):
    ear v4's acceptance probe on the realistic-v2 fixtures, gapped linear
    programmes with corpus, tape-hiss, flat or tape-shaped noise
    (`--shape-from` a capture), scored through both hiss readings, the
    acceptance table's fixture rows printed;
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
    the learned median veto, and the v4 grids' `constraints:` (hard guards
    on dsp readings: no new hiss, no dead air, no gated hiss, no dulling,
    no musical noise);
  - ear v4 (2026-10-12, the user's rule: be very critical of VHS hiss under
    the programme and in the pauses): the hiss readings
    (`restoration_quality/hiss_floor.py`, `hiss_programme.py`,
    `hiss_pause.py`), the grids `scripts/tune_grids/tata_v4.yaml` and
    `music_v4.yaml` (the loop's and the sweep's default) with their `tie`
    bands and `constraints:`, and the hiss degradations
    (`scripts/degradations_hiss.py`).
- [sonarqube-quality-gate/SKILL.md](sonarqube-quality-gate/SKILL.md): SonarQube
  Cloud on this repository: the CI scan, reading and fixing findings without
  suppressions, project settings on sonarcloud.io, failure triage.

Every session that learns a rule, a measured fact or a tool quirk writes it
into the skill that owns the topic in the same change (`AGENTS.md`, section
9).
