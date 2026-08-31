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

The engine supports 8 execution modes configured in `config.yaml`:

- **`auto`** (`*_Auto_Cleaned.<ext>`):
  - Stages: AI acoustic profiling $\\rightarrow$ dynamic engine & model
    selection $\\rightarrow$ shift/DTW sync (DTW on drift, shift otherwise,
    falling back to `SYNC_METHOD` on analysis failure) $\\rightarrow$ remux.
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

## 3. Required Local Quality Gate

Before declaring any task or feature complete, run the canonical local quality
gate command:

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
- **Prefer Installed Dependencies Over Building From Source**: Never rebuild a
  dependency that is already installed at the pinned revision. Before any
  source or VCS install (`pip install git+...`, source archives, portable
  binary downloads), check what is already present and skip the build when it
  matches.
  - Verify identity, not just presence. For VCS installs compare the recorded
    commit in the distribution's `direct_url.json` against the pinned ref; a
    version string alone can stay unchanged across revisions.
  - Rebuild only when the check fails, when the pin changes, or when an
    explicit force flag is set.
  - Rationale: re-cloning and rebuilding on every install wastes network and
    build time, and needlessly reverts files that
    `scripts/apply_patches.py` has already patched.

______________________________________________________________________

## 6. Workspace Skills Index

The repository defines the following modular skills in `.agents/skills/`:

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

______________________________________________________________________

## 7. Workflows Index

Targeted workflow playbooks are maintained under `.agent/workflows/`:

- [run_full_quality_gate.md](.agent/workflows/run_full_quality_gate.md): Running
  and troubleshooting the full quality pipeline.
- [resolve_pr_review.md](.agent/workflows/resolve_pr_review.md): Step-by-step PR
  review resolution.
- [add_restoration_feature.md](.agent/workflows/add_restoration_feature.md):
  Adding new restoration modes, filters, or models.
- [fix_lints_and_tests.md](.agent/workflows/fix_lints_and_tests.md): Rapid
  delinting and test fixing playbook.
