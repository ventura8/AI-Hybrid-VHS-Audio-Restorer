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
  correlation-gated sub-ms azimuth delay (`adelay`; a polarity-inverted pair,
  r at or under -0.3, is logged, and `azimuth_reject_inverted_pair` reads no
  skew from it; `azimuth_estimate_method` reads the skew from the first
  0.74 s, `head`, the default, whose sign doubles a real skew, or from the
  loudest spans' GCC-PHAT vote, `spans`, the one `auto_pure_linear` gains
  from), and peak de-clipping (`adeclip`).
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
  Dynamic Time Warping (DTW) with GPU PyTorch and CPU fallback. DTW plays
  whole 40 ms frames at 0x or 2x: the ear's time-base reading (R12) found
  frozen holds on 24 of 30 cathar and 15 of 30 APL hunt outputs sent to
  DTW, none on 205 per engine sent to shift; `sync_dtw_enable: false` sends
  the length-preserving chains to shift (default true until a blind block),
  and the CPU DTW path falls back to shift today (`docs/configuration.md`).
- **Lossless Mastering & Container Remux**: 32-bit float PCM (`pcm_f32le`)
  intermediates end-to-end, two-pass loudness normalization (ffmpeg
  `loudnorm`, EBU R128 measurement, -16 LUFS and -1 dBTP) followed by a
  sample-peak limiter (`alimiter` at -1 dBFS) in every mode,
  container-dependent final encoding, stream-copied video (`-c:v copy`), and
  optional dual-track archival audio preservation. loudnorm holds its linear
  mode only while the range and the true-peak rule allow it, and always on
  a programme under 3 s; 37% of the logged "linear" decisions broke the
  true-peak half and rode the gain (`loudnorm_linear_fallback`,
  `docs/configuration.md`). The restored track starts where the source's
  audio started against the video and MPEG-PS keeps its source's rate
  (`modules/mux_format.py`); the limiter still delays every output 219
  samples (4.97 ms at 44.1 kHz) until `mux_limiter_latency_compensation`
  is signed off. A failed or skipped file makes the app exit 1, and no
  prompt waits on a stdin that is not a console.

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
  - Listener-round stages (`modules/processing.py::_post_neural_stages`),
    on since the listener round of 2026-09-25 unless noted: the sibilant
    guard (`modules/sibilant_guard.py`, `apl_enable_sibilant_guard`: the 's'
    keeps the body the neural stage empties), the pause floor keeper
    (`modules/pause_floor.py`, `enable_pause_floor`, after the expander;
    `pause_floor_target` and `pause_floor_shape` keep `cluster` and `flat`,
    today's bytes, while `frame` with `shelf` took the replayed pause hiss
    from -7.6 to -30.3 dB on 2026-10-12, not yet rendered or heard), and
    on music the stem path (`modules/apl_stems.py`, `apl_music_stem_path`:
    the chain on the vocal stem, the music through the notches and a bounded
    suppressor; not accepted, off). The same round switched the subtraction
    stage off (`apl_enable_spectral_denoise`) and named the Mel-RoFormer
    denoiser on speech (`apl_neural_model`; `apl_music_neural_model`, empty,
    is the model on music). The polish expander's depth, knee and timing
    (`expander_depth_db`, `expander_knee_offset_db`, `expander_attack_s`,
    `expander_decay_s`), the mux's
    `loudnorm_target_lra` (loudnorm turns dynamic above it, and past its
    true-peak rule; `loudnorm_linear_fallback`) and `crt_notch_q` are keys
    both engines share; this mode takes its own expander depth
    (`apl_expander_depth_db`, 12 dB). With `AI_RESTORE_EVENT_LOG=<dir>` the
    sibilant guard, the plosive tamer, the pause floor and the mux record
    what they found (`modules/event_log.py`). With
    `AI_RESTORE_STAGE_CACHE=<absolute dir>` (the loop's `--stage-cache`)
    the neural stage's output is stored and replayed while only
    post-neural settings change (`modules/stage_cache.py`, deny-by-default
    key; off by default, never used by `cathar`). Post-neural hiss stages
    (2026-10-12, each off and opt-in, every output keeping its bytes):
    `apl_post_neural_suppress` runs the mode's MMSE-LSA suppressor after the
    sibilant guard on a tape whose 8-16 kHz source band is hiss-dominated
    and whose floor the neural stage left within
    `apl_post_neural_suppress_min_residual_db` on the band median or the
    3-4.8 kHz energy (`modules/post_neural_suppress.py`,
    `modules/hiss_floor.py`); `apl_neural_level_restore` undoes the
    separator's peak normalisation (`modules/neural_level.py`);
    `linear_air_gate` `hiss_dominated` skips the air shelf where that gate
    opens. On CPU replay the gate opens on 47 of 235 hunt clips and on no
    Tata or music clip (`docs/configuration.md`). The suppressor is never
    switched on with today's pause floor, which refills what it lowers
    with source hiss (pauses up 29-81 dB on 7 of 27 gated clips); it is
    judged with the per-frame pause fill. `apl_burst_ceiling` (`none`;
    `gaps` or `edges`, `modules/burst_ceiling.py`, after the polish) holds
    the hiss the RoFormer leaves next to speech to
    `apl_burst_ceiling_depth_db` (30) under the source's band floor in the
    cells where the source holds only hiss, and keeps the programme where it
    carries some, a word's decaying tail included: on replay with the
    shelved fill `edges` took the word-edge hiss from -20.5 to -22.9 dB
    (Tele7abc) and -24.6 to -25.9 (SOTI), guards unchanged. Most of the
    pause "bursts" are programme (a word's tail), which it keeps; as first
    committed it read tails under 6 dB as hiss and ate them (word-edge cells
    3-6 dB over the floor 3.7-6.1 dB down). Open: pumping where the hiss
    under the programme stays near the source's level
    (`docs/configuration.md`).
  - Two guards wait for a listening session (2026-10-11; both off, so
    every output keeps its bytes). SOTI's lifted pauses (332 of its 3,740
    quietest frames up 10 dB or more) are the Q-30 surgical notch ringing
    into gaps after speech, not the pop repair, which lifts none:
    `apl_surgical_ring_guard` caps the notch's output at 1 dB over its
    input, frame by frame. The pop repair refills clustered speech
    residual, 6-12 spans a second per channel on clean tape:
    `apl_depop_min_prominence_db` refuses a span that does not stand out
    from the residual around it (`docs/configuration.md`).
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
  - Music profile: when the scanner's tonal persistence
    (`modules/tonal_persistence.py`, median share of held spectral peaks;
    music 0.064-0.225, speech over a bed under 0.033) reaches
    `cathar_music_persistence_min`, the denoise runs at `cathar_music_alpha`
    with the `cathar_music_enable_*` switches; the band ratios read every
    music clip as dialogue, so they cannot make this call. Since round C3
    (2026-10-09) the profile also takes `cathar_music_beta` (the floor),
    `cathar_music_enable_repair` and `cathar_music_dewind_cutoff` in place of
    `cathar_beta`, `cathar_enable_repair` and `cathar_dewind_cutoff`, which
    no longer reach music; the defaults (0.02, on, 80 Hz) keep music's
    command lines, and so its output, unchanged
    (`tests/unit/test_cathar_music_round_c3.py`).
  - Round C4 (2026-10-10): `cathar_declip_mode` runs declip on every tape
    (`always`) or only on the scanner's clipping verdict counted over every
    channel (`detected`; a scan with no verdict declips with a WARNING), and
    `cathar_dehum_freq` gives dehum the scanner's mains family, skipping a
    tape it read as hum-free (`scanner`), or lets cathar detect it on every
    tape (`0`, `dehum --freq 0`). The defaults keep HEAD 549705a's command
    lines (`tests/unit/test_cathar_round_c4.py`); the other values go to the
    C3 listening session. Round C4 on the speech tapes (2026-10-11): plateau
    at the defaults; `detected` rendered the same audio as `always`, and
    `dehum --freq 0` won no tape.
  - Noise-print mute guard (2026-10-12, `cathar_noiseprint_mute_guard`,
    off): keeps digital mutes out of the print's pick
    (`modules/noiseprint_guard.py`). On the 7 IA clips that picked one it
    added a median 0.42 dB of removal and cost lion-king-dodge 3.4 dB at
    8-16 kHz. cathar's own estimate without a print also reads the mute, so
    dropping the print alone changes nothing. Off keeps HEAD bb5fa2a's
    command lines (`tests/unit/test_cathar_mute_guard.py`); it waits for a
    blind listening session.
  - Noise-print depth guard (2026-10-12, `cathar_noiseprint_min_depth_db`,
    0 = off, and `cathar_noiseprint_shallow_action`, `scale`, `drop` or
    `quantile`): a print whose windows sit less than the key under the loud
    frames is programme, and is scaled to the key, dropped, or replaced by
    the whole-file quantile print (scaled, with a WARNING, when that cannot
    be learned) (`modules/noiseprint_depth.py`). At 20 on the IA hunt's
    speech clips it leaves 25 (`scale`) or 17 (`drop`) of cathar's 62
    failing clips failing, none new, but it also fires on all four Tata
    tapes, whose stitched pauses sit 13.7-16.9 dB down; 13 leaves them byte
    for byte. At 13, `quantile` cures what `scale` cures (23 of the 53
    denoise-class failures) with more hiss removal and one new failure;
    it is the listening session's candidate. Off keeps HEAD 57bfdb8's
    command lines (`tests/unit/test_cathar_print_depth_guard.py`,
    `tests/unit/test_cathar_shallow_quantile.py`); it waits for a blind
    listening session.
  - Quantile noise print (2026-10-12, `cathar_noiseprint_method`, `window`
    or `quantile`): `quantile` writes the print from the whole material, per
    bin the 5th percentile of the 0.37 s moving mean plus 0.62 dB, chosen
    against realistic-v2's exact tape noise (median distance 4.22 dB, the
    window 4.93) (`modules/noiseprint_quantile.py`). At the IA hunt's denoise
    stage 25 of the 56 denoise-class octave failures stop failing and 3 of
    96 passing clips start, but it removes less hiss (the floor under the
    programme down 10.2 dB, not 14.1; on the Tata tapes 1.5-3.1 dB more hiss
    above 7 kHz, while the stitched print's 3-6 kHz cut was speech);
    `cathar_alpha` 1.25 brings the removal back. `window` keeps HEAD
    053ba30's command lines (`tests/unit/test_cathar_quantile_print.py`); it
    waits for a blind listening session.
  - Listener-round stages: split-band subtraction (`cathar_split_band_hz`,
    a second denoise pass at `cathar_alpha_high` above the crossover,
    recombined through `modules/split_band.py`; not accepted, off) and the
    pause floor keeper (`modules/pause_floor.py`, `enable_pause_floor`, after
    the expander; on since the listener round of 2026-09-25; the
    `pause_floor_target` and `pause_floor_shape` keys reach it too). On music
    the profile also switches the de-esser off (`cathar_music_enable_deesser`),
    sets the polish expander's depth (`cathar_music_expander_depth_db`) and
    narrows the CRT notch (`cathar_music_crt_notch_q`, carried into the
    pre-conditioning graph by `processing._precondition_config_for`). The
    installers provision cathar 0.8.0 (round C0, 2026-10-09: bit-identical
    to 0.7.6, the build the round tuned on, on every stage this app calls).
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
1. **SonarQube Cloud (CI only)**: after the tests, CI uploads `coverage.xml`
   and `junit.xml` to sonarcloud.io (`sonar-project.properties`: project
   `ventura8_AI-Hybrid-VHS-Audio-Restorer`, sources `modules`, `scripts`,
   `restore_audio_hybrid.py`, tests `tests`) and waits for the quality gate,
   which blocks the check. Its new-code conditions (bugs, vulnerabilities,
   hotspots reviewed, duplications, coverage on new code) apply on top of the
   local gates; treat a Sonar finding like a lint error: fix it, never
   suppress it. A fork PR skips the scan (no access to `SONAR_TOKEN`).

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

### Judging Outputs: the Harness and the User's Ear

The output-quality harness (the "AI human ear", output-quality-harness
skill) judges every restoration setting; the user's ear decides. Rules the
user set, with their reasons:

- Ask the user to listen only at a plateau, and only when the plateau is
  audibly different from what they last heard: run
  `scripts/audibility_check.py` first (the v3 sibilance loop's plateau
  differed by -70..-100 dBFS and would have wasted a session).
- Ask in short blind A/B batches (10-20 picks) through
  `python -m scripts.listen_ab`, which serves the page on 127.0.0.1 only:
  the family tapes never leave the machine, and no cloud or audio-LLM judge
  is used. Every answer goes to the verdict ledger
  (`assets/quality_calibration/verdicts.jsonl`); flags come only from the
  user's own words. Every block records `--device` and `--volume`
  (required since 2026-10-12) and plays at one speech level (-23 dBFS);
  a one-button block reads "no signal", never "disengaged" (Session 0:
  "I chose B every time because they all sounded the same"), and only a
  missed catch trial reads "disengaged".
- No default ships without a listening session and one
  `artifacts/realistic-v2` plus IA-corpus confirmation.
- Readings are ranked two-sided, as distance from what the user accepted;
  learned judges only veto. A reading that cannot see a percept (a
  whole-spectrum tilt cancels in every reading net of plain or loud frames)
  gets an absolute twin before a loop tunes against it.
- Every paired reading sits on the aligner `scripts/pair_alignment.py`
  names in the scored JSON (2026-10-11). Scores stored before then used the
  head search, which read 68 of 205 cathar hunt outputs more than 8 samples
  off their true lag: re-score them before they decide anything, and leave
  out a clip whose alignment abstained.
- The ear must be very critical of VHS hiss, under speech or singing and in
  the pauses (2026-10-11). A reading of hiss or residual noise never takes a
  tie band, a dead-zone floor or a veto limit from the Session 0 benign
  pairs: those pairs differed grossly there (residual noise 16.4 dB, pause
  depth 5.8 dB) and would blind the ear to hiss. Its floors come only from
  scorer repeat jitter and same-percept transforms
  (`scripts/autotune_judging.py`, `--jitter-floors`).
- Learned training goes only where it fits (2026-10-08): GRPO for a
  per-tape settings policy, later, and only if it beats the single best
  preset; an anchored fine-tune for the Mel-RoFormer denoiser; a
  Bradley-Terry preference head as the judge. No generative engines.

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
- **Locking**: lock with the repository's Poetry
  (`.venv/bin/python -m poetry lock`, 2.5.1 like CI and the installers); it
  takes seconds. The torch, torchvision, torchaudio, audio-separator and
  onnxruntime pairs keep exactly complementary `sys_platform` markers, and the
  `nvidia-cublas` / `nvidia-cuda-nvrtc` pins follow torch's cuda-toolkit: a
  gap between two markers, or `platform_system` beside upstream's
  `sys_platform`, made the solve run for hours until v1.4.0 (the
  poetry-runtime-and-ci skill, "Markers That Keep the Lock Fast").
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
- **Approved for ear v3 and the training plan (the user, 2026-10-08)**: the
  cathar 0.8.0 Windows archive (provisioned 2026-10-09 into
  `experiments/cathar-0.8.0/`, checksum verified; it passed the identity
  check, 5/5 on cathar and 3/3 on `auto_pure_linear`, and the installers
  now provision it); not provisioned yet: silero-vad (needs only torch),
  zimtohrli (needs only numpy), optuna (dev group) and an isolated
  `tools/msst/` venv for the denoiser fine-tune, kept apart like
  `tools/piper-tts` so the CUDA torch set and `onnxruntime-gpu` stay
  untouched. Check each one's dependency tree for plain `onnxruntime`
  before it enters the main venv. MERT's CC-BY-NC-4.0 weights stay
  (research use).

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
  plateau; ear v3's readings, the audibility check
  (`scripts/audibility_check.py`), the verdict ledger and the listening tool
  (`scripts/listen_ab.py`), the reward pieces
  (`scripts/reward_noise_floor.py`), the loop's judging resolution and its
  offline re-judge (`scripts/rejudge_autotune.py`), and the variation
  pre-check that gates the settings policy (`scripts/variation_precheck.py`).
- [sonarqube-quality-gate](.agents/skills/sonarqube-quality-gate/SKILL.md):
  SonarQube Cloud on this repository: the CI scan, reading and fixing its
  findings without suppressions, project settings on sonarcloud.io, and the
  triage of a failed check.

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
  listening complaint, letting the autotune loop refine, checking the
  plateau is audible, asking through the listening tool and the verdict
  ledger, confirming on full tapes and feeding winners back.
- [sonarqube_findings.md](.agent/workflows/sonarqube_findings.md): Triage of a
  failed SonarQube Cloud check and fixing its findings at the source.

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
