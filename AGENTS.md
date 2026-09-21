# Project Agent Rules & Development Guidelines

This repository defines the agent-specific development guidelines, operational
invariants, and reusable skills for **AI Hybrid VHS Audio Restorer**.

## 1. Project Overview & Core Architecture

`AI-Hybrid-VHS-Audio-Restorer` is a high-performance audio restoration engine
specifically designed for analog VHS tape captures suffering from tape hiss,
mechanical motor rumble, impulsive clicks/pops, Hi-Fi head-switching buzz, and
analog wow/flutter speed drift, measured against the tape's own recorded video
line whine rather than against programme pitch.

The pipeline combines state-of-the-art AI stem separation, speech enhancement,
background denoising, native FFmpeg DSP filtering, and sub-sample audio
alignment:

- **AI Vocal Separation**: BS-Roformer / MelBand-Roformer via `audio-separator`.
- **Speech Enhancement & Polish**: Resemble-Enhance (dynamic NFE/Tau) and neural
  speech sibilance de-esser (`deesser`).
- **AI Background Denoising**: UVR-DeNoise and downward dynamic noise expander
  (`compand`).
- **Analog Hardware Pre-Conditioning**: DC offset blocker (2 Hz highpass),
  stereo balance handling (`pan`) that levels a modest imbalance by attenuating
  the louder side and mirrors the live channel when the other one is dead,
  correlation-gated sub-ms azimuth delay (`adelay`), and peak de-clipping
  (`adeclip`).
- **Native VHS DSP Filtering**: Multi-threaded FFmpeg filter graphs (`afftdn`,
  `adeclick`, `highpass`, `bandreject`).
- **Surgical Tone Notching**: mains hum constrained to the family the detected
  video line rate allows and always notched at its fundamental, 15.625k/15.734k
  CRT flyback whistle located by band search, and enclosure acoustic resonance
  gated so speech formants are never notched.
- **Cathar DSP Engine**: Pure-Rust multi-stage restoration (dewind, azimuth,
  declick, decrackle, AR inpaint, deplosive, declip, adaptive dehum, dewow,
  noiseprint learning, phase-coherent denoise, de-esser, and SBR enhancement).
- **ARNNDN Neural Denoising**: FFmpeg RNNoise recurrent neural network denoiser
  (`.rnnn` models, with `cb.rnnn` as the canonical default).
- **Sub-Sample Audio Synchronization**: Cross-correlation lag estimation and
  Dynamic Time Warping (DTW) with GPU PyTorch and CPU fallback.
- **Lossless Mastering & Container Remux**: 32-bit float PCM (`pcm_f32le`)
  intermediates end-to-end, two-pass EBU R128 loudness normalization followed by
  a true-peak limiter in every mode, container-dependent final encoding,
  stream-copied video
  (`-c:v copy`), and optional dual-track archival audio preservation.

______________________________________________________________________

## 2. Restoration Modes Matrix

The engine supports 10 execution modes configured in `config.yaml`:

- **`auto_pure_linear`** (`*_PureLinear_Cleaned.<ext>`):
  - Stages: Dual-resolution scan $\\rightarrow$ analog pre-conditioning
    $\\rightarrow$ pre-denoise surgical bandreject $\\rightarrow$ gated physical
    damage repair $\\rightarrow$ tracked hum cancellation $\\rightarrow$
    event-gated plosive control $\\rightarrow$ noise-profile subtraction
    $\\rightarrow$ learned per-bin blend $\\rightarrow$ UVR-DeNoise full-mix
    $\\rightarrow$ post-cleanup $\\rightarrow$ linear air polish
    $\\rightarrow$ shift/DTW sync $\\rightarrow$ remux. The stages between
    pre-conditioning and the neural denoiser run through
    `modules/apl_chain.py`, each behind its own `apl_enable_*` switch.
  - Use case: Clean dialogue and high-throughput restoration without stem
    separation. Removes more tape noise than `cathar` while disturbing the
    programme less, and removes mains hum where neither mode used to; see
    `docs/cathar_vs_auto_pure_linear_1000_benchmark.md`.
- **`auto`** (`*_Auto_Cleaned.<ext>`, the default):
  - Stages: AI acoustic profiling (speech, music, rhythm, tonality, noise
    floor, hum) $\\rightarrow$ engine & model selection $\\rightarrow$ the
    selected engine's chain $\\rightarrow$ shift/DTW sync (DTW on drift, shift
    otherwise, falling back to `SYNC_METHOD` on analysis failure)
    $\\rightarrow$ remux.
  - Engine choice: `auto_pure_linear` on every acoustic class, because it
    leads `cathar` on each one measured on real tape and the engines the
    scanner used to pick for music and tape noise (`denoise_only`,
    `auto_ffmpeg_native`) leave the noise on the tape; `cathar` on sustained
    tonal programme with no silence for `auto_pure_linear`'s 4 s noise
    probe (flatness under `auto_cathar_flatness_max`, the quietest 4 s
    shaped like the loud frames above `auto_cathar_probe_similarity`, no
    beat), the one condition under which it deviates less on most clips of
    both the corpus and the local tapes, always for 2-3 dB less removal
    (`auto_cathar_tonal`); and when the neural denoiser is not installed;
    `auto_ffmpeg_native` when neither engine is.
    The numbers sit beside the rule in `modules/auto_scanner.py` and in
    `docs/cathar_vs_auto_pure_linear_1000_benchmark.md`.
  - Use case: Intelligent single-click end-to-end restoration.
- **`multipass_auto`** (`*_MultiPass_Cleaned.<ext>`):
  - Stages: Dual-resolution acoustic scan $\\rightarrow$ analog pre-conditioning
    $\\rightarrow$ AI stem separation & Resemble-Enhance $\\rightarrow$ residual
    polish $\\rightarrow$ DTW Sync $\\rightarrow$ mix.
  - Use case: Maximum quality 4-pass cascaded restoration.
- **`auto_pure`** (`*_Pure_Cleaned.<ext>`):
  - Stages: Dual-resolution scan $\\rightarrow$ analog pre-conditioning
    $\\rightarrow$ AI stem separation $\\rightarrow$ pure speech/ambient
    UVR-DeNoise (bypassing vocoder synthesis) $\\rightarrow$ DTW Sync
    $\\rightarrow$ mix.
  - Use case: Pure speech/ambient denoising without artificial synthesis.
- **`cathar`** (`*_Cathar_Cleaned.<ext>`):
  - Stages: Multi-stage Rust DSP pipeline: dewind $\\rightarrow$ azimuth
    $\\rightarrow$ mono-below $\\rightarrow$ declick $\\rightarrow$ decrackle
    $\\rightarrow$ inpaint $\\rightarrow$ deplosive $\\rightarrow$ declip
    $\\rightarrow$ dehum $\\rightarrow$ repair $\\rightarrow$ noiseprint denoise
    $\\rightarrow$ de-esser $\\rightarrow$ SBR enhance $\\rightarrow$ sync.
  - Use case: Spectral spikes, sustained tonal programme, zero AI
    hallucination for music and ambient archives. (`cathar_vhs` is an alias).
- **`hybrid`** (`*_Hybrid_Cleaned.<ext>`):
  - Stages: BS-Roformer $\\rightarrow$ Resemble-Enhance $\\rightarrow$
    UVR-DeNoise $\\rightarrow$ DTW Sync $\\rightarrow$ amix.
  - Use case: Full 2-stem vocal/background separation and enhancement.
- **`denoise_only`** (`*_Denoised_Cleaned.<ext>`):
  - Stages: UVR-DeNoise-Lite on full track $\\rightarrow$ DTW Sync
    $\\rightarrow$ remux.
  - Use case: Fast AI broadband denoising without stem separation.
- **`auto_ffmpeg_native`** (`*_AutoFFmpeg_Cleaned.<ext>`):
  - Stages: Dynamic acoustic profile tuning $\\rightarrow$ auto-parameterized
    FFmpeg DSP chain $\\rightarrow$ DTW Sync $\\rightarrow$ remux.
  - Use case: Fast, adaptive native DSP filtering without AI neural models.
- **`vhs_native`** (`*_FFmpeg_Cleaned.<ext>`):
  - Stages: `highpass` + `adeclick` + `afftdn` + optional `bandreject`
    $\\rightarrow$ DTW Sync $\\rightarrow$ remux.
  - Use case: Native FFmpeg multi-threaded DSP filter chain.
- **`arnndn_speech`** (`*_Speech_Cleaned.<ext>`):
  - Stages: `highpass` + `adeclick` + `arnndn` (RNNoise) $\\rightarrow$ DTW Sync
    $\\rightarrow$ remux.
  - Use case: Deep-learning RNNoise speech denoiser for dialogue.

______________________________________________________________________

## 3. Required Local Quality Gate & Hardware Validation

### Canonical Local Quality Gate

Before declaring any task or review wave complete, run the canonical local
quality gate command:

```bash
# On Linux / macOS:
./run_pipeline_locally.sh

# On Windows:
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_pipeline_locally.ps1
```

This runner orchestrates the complete local/CI quality gate set:

1. **PowerShell Script Lint**: `PSScriptAnalyzer` on project scripts (via pwsh).
1. **Python Code Formatting**: `black` and `isort` on all source targets.
1. **Static Analysis & Linting**: `ruff`, `flake8`, and `pylint`.
1. **TOML Configuration Formatting**: `taplo fmt --check`.
1. **Security & Vulnerability Gates**: `bandit -ll -ii` and `pip-audit`.
1. **Complexity & Maintainability**: Radon CC (Grade A $\\le 5$) and Radon MI
   (Grade A $\\ge 20$).
1. **Markdown Quality**: Read-only `mdformat --check` validation and
   `pymarkdown scan` linting.
1. **Test Suite & Coverage**: `pytest` with strict per-file $\\ge 90.00%$
   coverage gate.
1. **Coverage Badge Regeneration**: Updates `assets/coverage.svg`.

### Mandatory Hardware Validation on Code Changes & Review Waves

Whenever audio restoration, filtering, DSP, sync, or hardware routines are
modified (including during every CodeRabbit review wave):

1. **Host Hardware Audit & Dry-Run Checks**: Run
   `poetry run python scripts/audit_hardware.py` and
   `poetry run python scripts/run_hardware_validation.py core` to verify device
   capabilities, generate `artifacts/hardware-validation.json`, and ensure
   reproducible dry-run planning without requiring physical inference.
1. **Fixture Execution (Opt-In)**: `--execute` runs the generated Piper
   fixtures through the selected modes. `AI_RESTORE_HARDWARE_TESTS=1` enables
   the separate hardware test suite; neither option selects real captures or
   Internet Archive material.
1. **Real Tape and Corpus Validation (Separate Opt-In)**: When local analog
   captures or `experiments/ia_corpus/` are explicitly provisioned, run the
   relevant tuning or IA benchmark command and record its corpus selection,
   report path, and acoustic metrics.
1. **Calibrated Fixtures Before Tuning (Opt-In)**: A restoration setting is
   chosen on `artifacts/realistic-v2`, never on the hardware fixtures. Build
   the set with `scripts/make_realistic_fixtures_v2.py` and run
   `scripts/validate_fixture_realism.py`; a change that ranks one way there
   and the other way on real tape is a fixture defect to fix before the
   setting is judged. See `docs/validation.md`, "Fixtures That Predict Real
   Tape". `scripts/expand_ia_corpus.py` widens the corpus the set is
   calibrated against. The validator and `scripts/sweep_denoise_settings.py`
   patch `modules/config.py` in the checkout they run in and restore it
   afterwards: never edit or commit that file while either runs, and run
   them from a git worktree when the main checkout is being worked on.
1. **Acoustic Metrics Verification (Opt-In Execution)**: When physical
   validation is provisioned, run `analyze_audio_quality()` via
   `scripts/compare_restoration_quality.py` to confirm actual noise floor
   reduction, peak-to-noise ratio, CRT whistle elimination, and rumble
   suppression on physical audio data before finishing.

______________________________________________________________________

## 4. Strict Linting & Coding Standards

### Zero Suppression Policy

- **No Suppressions Allowed**: Never use `# noqa`, `# pylint: disable`,
  `# type: ignore`, `# bandit: disable`, or equivalent bypass pragmas.
- **Tests Are Not Exempt**: Test modules in `tests/` must adhere to the exact
  same formatting, linting, and complexity standards as production code.

### Python Rules

- **Line Length**: 140 characters maximum for Python files.
- **Auto-Fix First**: Always run automatic formatters (`black`, `isort`) before
  making manual edits.
- **Radon Metrics**: Every function/method must be Cyclomatic Complexity Grade A
  ($\\le 5$). Every file must be Maintainability Index Grade A ($\\ge 20$).

### Markdown Rules

- **Continuous Documentation Updates**: Every time code, configuration,
  architecture, models, or workflows are modified, all relevant Markdown
  documentation files (`README.md`, `docs/`, `AGENTS.md`, `.agent/`, etc.) must
  be reviewed and updated in the same pass.
- **Line Length (MD013)**: All Markdown prose lines must be wrapped to $\\le 80$
  characters per line.
- **MD013 Exceptions**: Headings, code blocks, and wide table rows may be up
  to 200 characters when splitting would reduce readability or copyability.
- **Headings & Structure**: Top-level `#` title only once per document; headings
  must strictly increment by one level.

______________________________________________________________________

## 5. Dependency Management Rules

- **Poetry as Single Source of Truth**: Use Poetry only. Do not use
  `requirements.txt` or `test-requirements.txt`.
- **Installer Mode vs CI/Dev Mode**:
  - The end-user installer (`install_dependencies.ps1`) must run verbose
    runtime-only installation (`poetry install --only main --verbose`).
  - CI and local dev environments install runtime plus development dependencies
    (`poetry install --with dev`).
- **CUDA Runtime Stack**: Preserve NVIDIA CUDA 13.2 runtime stack compatibility.
- **Agents Provision What a Task Needs**: A missing dependency, runtime, model,
  or fixture set is never a reason to skip or scale down a step. The agent
  installs it, the way the installer does, before reporting: the `ml` group
  with `poetry install --with dev,ml` when a mode needs it, the isolated Piper
  runtime with
  `poetry --directory tools/piper-tts install --only main --no-root` (its own
  `.venv`; never into the main one, whose CUDA ONNX Runtime it would replace),
  Piper voices and the fixture matrix with
  `scripts/generate_audio_matrix.py core --language all`, DNSMOS with
  `scripts/download_dnsmos.py`. Everything installs into the repository's
  virtual environments and is declared in Poetry; nothing goes to the system
  interpreter. Report what was provisioned and what it cost (time, disk).

______________________________________________________________________

## 6. Workspace Skills Index

The repository defines the following modular skills in `.agents/skills/`.
`.claude/skills/` holds a pointer catalog so Claude Code can invoke each one as
`/<name>`; the content lives only in `.agents/skills/`.

- [code-linter](.agents/skills/code-linter/SKILL.md): Comprehensive multi-linter
  rules and commands without suppressions.
- [pipeline-runner](.agents/skills/pipeline-runner/SKILL.md): Execution and
  diagnosis of `./run_pipeline_locally.sh` (Linux/macOS) or
  `powershell -NoProfile -ExecutionPolicy Bypass -File .\run_pipeline_locally.ps1`
  (Windows).
- [test-runner](.agents/skills/test-runner/SKILL.md): Unit and integration test
  runners with coverage floor verification.
- [audio-restoration-engine](.agents/skills/audio-restoration-engine/SKILL.md):
  Deep restoration domain knowledge, DSP graphs, and DTW sync.
- [hardware-validation](.agents/skills/hardware-validation/SKILL.md): Deterministic
  fixtures, host hardware auditing, and restoration validation.
- [markdown-quality](.agents/skills/markdown-quality/SKILL.md): Read-only
  Markdown validation via `mdformat --check` and `pymarkdown`.
- [poetry-runtime-and-ci](.agents/skills/poetry-runtime-and-ci/SKILL.md): Poetry
  dependency management and lockfile maintenance.
- [resolve-pr-comments](.agents/skills/resolve-pr-comments/SKILL.md): GitHub CLI
  PR comment resolution and reply workflows.
- [prepare-release](.agents/skills/prepare-release/SKILL.md): Release
  preparation, semver bumping, and changelog curation.
- [installer-tester](.agents/skills/installer-tester/SKILL.md): Windows
  installer validation and CUDA runtime provisioning.
- [output-quality-harness](.agents/skills/output-quality-harness/SKILL.md):
  The "AI human ear": scoring an output like a listener, calibration, tuning
  on real tapes, and the self-driving loop that asks the user only at its
  plateau.

______________________________________________________________________

## 7. Pull Request Conventions

- The PR title is detailed, never the bare version. A release branch takes
  the release commit's subject, `vX.Y.Z: theme in one line`; any other
  branch takes `component: what changed and why`. The PR list then reads as
  a changelog. Only the GitHub Release is titled with the bare tag, which
  the release workflow does from the description file.
- The PR body is the GitHub release description without its heading line,
  followed by the agent's attribution footer. Release PRs open as drafts
  and are pushed only: no tag, no merge, no un-draft unless asked.
- Retitle with `gh pr edit NUMBER --title "..."` when the scope grows after
  the PR is opened, so the title still says what the branch does.

## 8. Workflows Index

Targeted workflow playbooks are maintained under `.agent/workflows/`:

- [run_full_quality_gate.md](.agent/workflows/run_full_quality_gate.md): Running
  and troubleshooting the full quality pipeline.
- [run_hardware_validation.md](.agent/workflows/run_hardware_validation.md): Host
  hardware audit and validation playbook.
- [resolve_pr_review.md](.agent/workflows/resolve_pr_review.md): Step-by-step PR
  review resolution.
- [add_restoration_feature.md](.agent/workflows/add_restoration_feature.md):
  Adding new restoration modes, filters, or models.
- [fix_lints_and_tests.md](.agent/workflows/fix_lints_and_tests.md): Rapid
  delinting and test fixing playbook.
- [tune_on_real_tapes.md](.agent/workflows/tune_on_real_tapes.md): Measuring a
  listening complaint, letting the autotune loop refine, confirming on full
  tapes and feeding winners back.

______________________________________________________________________

## 9. Keeping the Agent Files Current

The agent files are the project's memory across sessions and agents. Every
change that teaches something the code does not say is written into them in
the same change, never left in a chat, a scratchpad or a personal memory:

- A rule the user states ("never two restorations on one source path",
  "ask me to listen only at the plateau") goes into `AGENTS.md` or the skill
  that owns the topic, with the reason.
- A measured fact that decided a default (a metric reading, a by-ear verdict,
  a corpus number) goes into the skill that owns the engine or the harness,
  and into the configuration comment beside the value.
- A tool quirk that cost time (a lockfile solve that never finishes, an
  encoding that breaks a linter, a checksum served as an LFS pointer) goes
  into the skill that runs the tool.
- A new script, grid, runner or workflow gets its entry in the skill that
  uses it, in `.agents/skills/README.md`, in section 6 or 8 above, and a
  `.claude/skills/<name>/SKILL.md` pointer when it is a skill.

Before declaring a task complete, re-read the touched skills and this file
and ask: would the next agent, with only these files, repeat today's work or
build on it? Then run the Markdown gates on everything touched.
