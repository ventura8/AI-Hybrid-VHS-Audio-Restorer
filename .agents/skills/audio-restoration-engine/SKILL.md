---
name: audio-restoration-engine
description: >-
  Domain knowledge and implementation patterns for VHS audio restoration
  modes, FFmpeg DSP filter graphs, ARNNDN speech denoisers, stem separation,
  and DTW alignment.
---

# Audio Restoration Engine Skill

Use this skill when developing, refactoring, or debugging audio restoration
algorithms, AI inference pipelines, DSP filters, Dynamic Time Warping (DTW)
audio alignment, or FFmpeg multiplexing.

## Restoration Modes Matrix

- **`auto`** -> `*_Auto_Cleaned.<ext>`
  - Acoustic profiling $\\rightarrow$ dynamic mode selection $\\rightarrow$
    shift/DTW (DTW when drift is detected, shift otherwise, falling back to
    configured `SYNC_METHOD` on analysis failure) $\\rightarrow$ remux.
  - Scans the capture and dispatches the best restoration pipeline:
    `auto_pure_linear` on every acoustic class, `cathar` on sustained tonal
    programme with no silence for the noise probe (AGENTS.md section 2).
- **`auto_pure_linear`** -> `*_PureLinear_Cleaned.<ext>`
  - Scan $\\rightarrow$ pre-conditioning $\\rightarrow$ `modules/apl_chain.py`
    (surgical notches, gated repair, hum canceller, plosive tamer, optional
    subtraction) $\\rightarrow$ neural denoiser $\\rightarrow$ sibilant
    guard, polish (expander, air shelf), pause floor $\\rightarrow$
    shift/DTW $\\rightarrow$ remux.
  - The engine the harness's loops tune beside `cathar`.
- **`cathar`** / **`cathar_vhs`** -> `*_Cathar_Cleaned.<ext>`
  - The Rust `cathar.exe` stage chain (dewind to SBR enhance), the music
    profile, polish and pause floor $\\rightarrow$ shift/DTW $\\rightarrow$
    remux; bit-identity on its five reference clips guards every change.
- **`multipass_auto`** -> `*_MultiPass_Cleaned.<ext>`
  - Pre-scan $\\rightarrow$ pre-conditioning $\\rightarrow$ BS-Roformer $\\rightarrow$
    Resemble-Enhance $\\rightarrow$ shift/DTW $\\rightarrow$ amix.
  - 4-pass cascaded restoration with analog pre-conditioning.
- **`auto_pure`** / **`pure`** -> `*_Pure_Cleaned.<ext>`
  - Pre-scan $\\rightarrow$ pre-conditioning $\\rightarrow$ BS-Roformer
    $\\rightarrow$ UVR-DeNoise
    $\\rightarrow$ de-esser $\\rightarrow$ shift/DTW $\\rightarrow$ amix.
  - Pure speech and ambient denoise, no vocoder synthesis.
- **`hybrid`** -> `*_Hybrid_Cleaned.<ext>`
  - BS-Roformer $\\rightarrow$ Resemble-Enhance $\\rightarrow$ UVR-DeNoise
    $\\rightarrow$ shift/DTW
    $\\rightarrow$ amix.
  - Full 2-stem vocal/background separation and enhancement.
- **`denoise_only`** -> `*_Denoised_Cleaned.<ext>`
  - UVR-DeNoise-Lite on the full track $\\rightarrow$ shift/DTW $\\rightarrow$ remux.
  - AI broadband denoising without stem separation.
- **`auto_ffmpeg_native`** / **`auto_vhs_native`** -> `*_AutoFFmpeg_Cleaned.<ext>`
  - Auto-tuned `highpass` + `adeclick` + `afftdn` + optional `bandreject`
    $\\rightarrow$ shift/DTW $\\rightarrow$ remux.
  - Native FFmpeg DSP chain tuned to the measured noise profile.
- **`vhs_native`** / **`ffmpeg_native`** -> `*_FFmpeg_Cleaned.<ext>`
  - `highpass` + `adeclick` + `afftdn` + optional `bandreject` $\\rightarrow$ DTW
    $\\rightarrow$ remux.
  - Fixed native FFmpeg multi-threaded DSP filter chain.
- **`arnndn_speech`** -> `*_Speech_Cleaned.<ext>`
  - `highpass` + `adeclick` + `arnndn` (RNNoise) $\\rightarrow$ shift/DTW
    $\\rightarrow$ remux.
  - Recurrent neural network denoiser for dialogue-heavy captures.

## Core Restoration Modules

1. **`modules/filters.py`**:
   - `_build_vhs_native_filter_string`: Constructs composite DSP filter graphs
     incorporating highpass, adeclick, notch bandreject, and adaptive FFT
     denoise (`afftdn`).
   - `_append_balance_correction`: Levels a modest stereo imbalance by
     attenuating the louder channel, and mirrors the live channel to both sides
     when the gap exceeds `DEAD_CHANNEL_DB`.
   - `_detect_stereo_azimuth_skew`: Cross-correlation lag, returned only when
     the channels clear `AZIMUTH_MIN_CORRELATION` (0.3) as |r|.
     `signed_channel_correlation` keeps the sign (`_channel_correlation`
     returned only |r|, so the gate passed a polarity-inverted pair). For L
     and -L the lag the correlation's maximum gives is half a period of the
     strongest partial, not a skew: a 1 kHz tone reads -0.50 ms, a 2 kHz tone
     -0.25 ms and noise low-passed at 3 kHz -0.23 ms, each of which the
     pre-conditioning would apply as a delay. A pair at r \<= -0.3 is always
     logged as inverted; `azimuth_reject_inverted_pair` (default false, so
     every output keeps its bytes) reads no skew from it. A NaN correlation
     passes the gate as before and is not flagged.
   - `_detect_crt_flyback_notch`: Band search across 15450-15900 Hz, classified
     by nearest line rate so an off-speed tape is still identified.
   - `_detect_mains_buzz_notch`: Hum vote constrained to the family the detected
     line rate allows; `_append_mains_notches` always covers the fundamental.
   - `_detect_enclosure_resonance_notch`: Requires a high-contrast bump of
     moderate width in the averaged spectral envelope, so speech formants and
     harmonic spikes are rejected.
   - `_resolve_arnndn_model_path`: Scans `models/arnndn/`, `models/`,
     `MODELS_DIR/arnndn/`, and `MODELS_DIR/` for RNNoise `.rnnn` models.
   - `_escape_ffmpeg_filter_path`: Escapes colons in Windows drive paths (e.g.
     `C\:/...`) for FFmpeg filter expressions.
   - `_run_ffmpeg_filter_step`: Atomic execution with CPU thread fallback.
1. **`modules/processing.py`**:
   - `_separate_stems_step`: Invokes `audio-separator` with BS-Roformer /
     MelBand-Roformer.
   - `_enhance_vocals_step`: Runs Resemble-Enhance with dynamic NFE and Tau
     parameters and GPU retry with CUDA cache flushing.
   - `_denoise_background_step` / `_denoise_full_audio_step`: UVR-DeNoise-Lite
     inference.
   - `_ensure_float_pcm`: Restores 32-bit float on any stem a separator emitted
     as fixed-point.
   - `_collect_stem_candidates`: Token-based, case-insensitive stem matching
     across every naming convention `audio-separator` emits.
   - `_final_mix_step`: Stem mix via FFmpeg `amix` with container-dependent
     audio encoding, resample to `PIPELINE_SAMPLE_RATE`, and the mastering
     tail of `modules/mastering.py`.
   - `_final_mux_single_audio_step`: Direct stream video copy with
     container-dependent audio mux.
1. **`modules/mastering.py`**:
   - `_resolve_loudnorm_args` / `_resolve_single_track_loudnorm_args`:
     two-pass loudnorm to -16 LUFS, -1 dBTP and `loudnorm_target_lra`,
     falling back to single-pass if measurement fails. The tail after the
     loudness stage is a resample and `LOUDNORM_TRUE_PEAK_LIMITER`, which
     despite its name is `alimiter` at 0.891 (-1 dBFS) on the samples at
     44.1 kHz, a sample-peak limiter.
   - `linear_mode_blockers`: ffmpeg's rule for loudnorm's linear mode,
     `TP + (I_target - I) <= TP_target`, `LRA <= target`, and a set
     measurement (I and LRA not 0, TP not 99, threshold not -70), with one
     override: given a duration under 3 s (`LOUDNORM_SHORT_PROGRAMME_S`,
     loudnorm's frame buffer) it returns no blockers, because ffmpeg runs
     such a programme linear whatever was measured (it also measures LRA
     0, the unset value). The mux passes the duration it knows; without one
     the short case is not checked. On synthetic programmes it reproduced
     the `normalization_type` ffmpeg 8.0.1 reported in all nine cases
     checked, and the boundary (linear at 2.9 s, dynamic at 3.0 s) held on
     8.0.1 and 9.0.1. `_log_loudness_range` logs the mode by it.
   - `loudnorm_linear_fallback` (`ffmpeg` default, or `gain_limiter`): what
     the mux does when the true-peak rule alone takes loudnorm out of linear
     mode (`LinearGain`: one `volume` gain and the existing limiter).
   - With `AI_RESTORE_EVENT_LOG` set, one more null-sink pass records what
     the loudness stage rendered (`normalization_type`, output I, TP, LRA,
     threshold, the blockers, the predicted mode, the fallback) as
     `loudnorm__<recording>__<mode>__<track>.json`, beside the track the mux
     masters and in the event-log folder. A decision ffmpeg made against the
     rule is logged as a warning.
1. **`modules/event_log.py`**: the engine's event log
   (`AI_RESTORE_EVENT_LOG=<dir>`, unset by default, and then nothing is
   written or computed). The sibilant guard, the plosive tamer and the pause
   floor keeper each write
   `<stage>__<recording>__<mode>__<track>.json`: the stage, the recording
   (the `.temp_work_<stem>` folder name, else the track's stem), the
   configured `process_mode`, the
   track's path and rate, the stage's thresholds, and the spans it found in
   seconds (`events_s`). The sibilant guard's and the plosive tamer's spans
   sit on the pre-neural reference / input timeline, the pause floor's on
   the restored file's (`events_s`, the pauses found, and `filled_s`, the
   frames the fill lifts; the gain reaches at most 20 ms past each `filled_s`
   run). A stage that does not look writes `skipped` with its reason
   (switched off, refused material) or `failed: <error>`; a failure after
   the detection replaces the events, since the output then carries no
   repair. Each stage writes the same samples with the log on or off
   (tested per stage). Two runs in one mode share record names, so give
   every tuning candidate its own event-log folder.
1. **`modules/stage_cache.py`, `modules/stage_cache_key.py`**: the neural-stage
   cache (`AI_RESTORE_STAGE_CACHE=<absolute dir>`, off by default;
   `docs/configuration.md`, "Stage Cache"). `processing._neural_output` is
   everything `auto_pure_linear` and `denoise_only` run up to and including
   the neural model (surgical notch, `apl_chain`, model choice, model);
   `_denoise_and_polish_full_audio_step` hands it to `stage_cache.through`,
   which stores or replays its `(surgical, denoised)` pair. Invariants:
   - Off, `through` is one environment read and `produce()`: the same calls
     in the same order as before, so every default output keeps its bytes.
     cathar never calls it (`test_stage_cache_reach.py` proves its mode
     cannot reach it).
   - Deny by default: a `config.yaml` key stays out of the key only when it
     is in `POST_NEURAL_CONFIG_KEYS` and every read of it is after the
     cached point. A new post-neural key goes into that set, into
     `READ_SITES` in `tests/unit/test_stage_cache_allowlist.py` (where it is
     read) and is checked by `tests/unit/test_stage_cache_reach.py` (a graph
     of every reference across `modules/` from the producer: nothing reached
     may read an allowlisted key). A key not on the list is always keyed.
     The `cathar_*` keys are read on this path (the repair and the
     subtraction run cathar's stages), so they stay keyed.
   - soundfile FLOAT WAVs are not byte-stable: two writes of the same
     samples differ at byte 60, the PEAK chunk's timestamp (soundfile
     0.14.0, libsndfile 1.2.2). The input is keyed by its decoded samples,
     and "a hit equals a fresh render" is checked on decoded samples
     (`autotune_restoration.exact_audio_sha256`'s rule), never file bytes.
   - A stage that falls back after a failure must log at WARNING (or
     ERROR): a render whose WARNING/ERROR count (`stage_cache.problem_count`:
     `utils.problem_count` plus audio-separator's) moved is never stored.
     The chain's fallbacks (repair, hum, tone, plosive, subtraction, tonal
     cleanup, native suppressor, blend, depop, cathar's noise print) were
     promoted for this, and on 2026-10-09 the shared analysis read behind
     every stage's "unreadable" (`filters._read_stereo_audio_for_analysis`),
     the surgical step's unreadable input, `_run_dsp_filter_file`'s invalid
     output, `_ensure_float_pcm`'s failure and cathar's quiet-window search
     (`tests/unit/test_stage_cache_fallbacks.py`). A new fallback in a
     pre-neural stage needs the same. A deterministic skip (a recording too
     short to scan) stays at INFO.
   - audio-separator logs through Python logging, not `log_msg`:
     `stage_cache.watch_separator_log` counts its WARNING and ERROR records
     with a log-record factory (a handler would stop the separator adding
     its own console handler, which it adds only when none is reachable).
     Its routine lines (`SEPARATOR_ROUTINE`) do not count: every one of the
     5,252 WARNING lines in 826 APL candidate logs was
     `Using soundfile for writing.`, logged on every render (2026-10-09).
   - The code and package fingerprints are read at start-up
     (`stage_cache_key.prime`, with the cache on), when the process loads
     its code; `stage_cache_key.drift` refuses a store when a module source
     was written after the process started or the sources or packages no
     longer hash as then. The four post-cache modules' import-time code
     (`import_time_code`) is in the key, since processing imports `sync`
     and `mastering` at start-up; only their function bodies are left out.
   - The environment is keyed by prefix (`ENV_PREFIXES`), deny by default:
     a numerics variable with another prefix goes into the list. The model
     files are keyed in the folder audio-separator really loads from
     (`AUDIO_SEPARATOR_MODEL_DIR` replaces the one the stage passes), and
     the CPU by name.
   - Tests that store an entry hold the host still
     (`test_stage_cache.hold_the_host_still`): a GitHub runner has 14 GB
     free, under `MIN_FREE_GB`, and another agent editing the checkout
     would trip the drift check mid-test.
   - Nothing is stored from a stage folder that already held files (a
     resumed work folder), nor with DeepFilterNet or Resemble on, nor with
     `AI_RESTORE_EVENT_LOG` set (the plosive tamer records before the cached
     point); a key that cannot be computed bypasses with a warning.
   - In-process patches of a module constant are not seen (the key reads
     `config.CONFIG`): a test or experiment that patches one patches
     `CONFIG` too, or leaves the cache off.
1. **`modules/auto_scanner.py`**:
   - `_detect_flutter_or_pitch_drift`: Tracks the recorded video line whine as a
     fixed-frequency speed reference, so programme pitch cannot read as drift.
     The harness's capture profile (R0, 2026-10-09) found the line on the
     Tata tapes steady (7 and 13 ppm) and not wandering with the hum, the
     `playback_chain` class: on a PAL SP linear track the line's wavelength
     (about 1.5 um) is far beyond the ~8 kHz response, so it cannot be on
     the tape. F3, its own change, revisits the DTW trigger and this
     wording, guarded by the harness's `file.sync_drift_ms`.
   - `_best_fitting_reference`: Picks PAL or NTSC by which nominal the tracked
     mean sits nearest, since the search bands overlap.
   - `_estimate_onset_periodicity`: Autocorrelation peak of the spectral-flux
     envelope, i.e. whether the audio carries a beat. This is what separates
     music from conversation; the spectral tonal-peak ratio does not, because
     sung vocals sit in the speech band.
   - `evaluate_restoration_strategy`: Emits the models, parameters, sync method,
     and mix gains that the pipeline then actually applies.
1. **`modules/sync.py`**:
   - `_align_stems`: Sub-sample audio synchronization.
   - Cross-correlation lag estimation for linear delay.
   - Dynamic Time Warping (DTW) with GPU PyTorch tensor acceleration and CPU
     `fastdtw` fallback to correct analog VHS tape speed drift.
1. **`modules/hardware.py`**:
   - Hardware detection for NVIDIA CUDA 13.2, Intel XPU, Apple MPS, and CPU.
   - Dynamic thread and batch allocation (`GPU_BATCH_SIZE`, `CPU_THREADS`).

## Key Engineering Rules

- **Audio Bit Depth**: Intermediate audio processing must remain 32-bit float
  PCM (`pcm_f32le`) at 44.1 kHz to prevent clipping or quantization noise. Stems
  a separator emits as fixed-point are converted back before the next stage.
  Output remuxing uses container-dependent codecs (AAC for `.mp4`/`.m4v`, MP2
  for `.mpg`/`.mpeg`, and `pcm_f32le` only for configured PCM-capable
  containers).
- **Atomic Operations**: Always output to `.tmp.wav` / `.tmp.mp4` first, verify
  validity with `is_valid_audio` / `is_valid_video`, and rename to final
  destination upon success.
- **Resume Protection**: Check for pre-existing valid stage outputs before
  launching compute-heavy inference steps.
- **Video Stream Copy**: Never re-encode the video stream (`-c:v copy`) during
  audio extraction or remuxing.
- **cathar bit-identity**: after any change under `modules/`, restore the five
  reference clips (`experiments/cathar_ab.py <tag>`) and require 5/5 decoded
  PCM hashes equal to `experiments/cathar_ab_head.json`. A deliberate default
  change (the user's call, made by ear) is followed by a re-base: copy the
  run into `experiments/cathar_ab_head`, rewrite the JSON, keep the previous
  reference as `cathar_ab_head_before_<tag>`. `--old-deesser` reinstates the
  pre-fix de-esser so the rest of the chain can be checked alone.
- **cathar de-esser semantics**: `deesser --threshold` changes meaning with
  `--bands`: single-band is an HF/broadband ratio (default -24), multiband is
  dB above each band's running average (use 6). A negative multiband
  threshold engages the stage on every frame and removes everything above
  the crossover ("under water" speech). `_cathar_deesser_step` guards it.
- **cathar noise print**: one contiguous window on dialogue lands on speech and
  the print learns sibilance; the shipped print is stitched from eight 0.75 s
  windows spread by level over the quietest 20 % (10 ms crossfades), only
  from 20 x `cathar_noiseprint_duration_s` of material.
- **cathar alpha 2.0 -> 1.0**: 2.0 was chosen by ear and by the harness on
  five real tapes (colouration +0.125 against +0.062, discontinuity tail
  -0.34 against -0.42, CER 0.058 against 0.093) for 1.6 dB less removal;
  3.5 and Wiener lose every listener-side reading. The listener round
  (2026-09-25, `docs/validation.md`, "The listener round's plateaus") took
  it to 1.0 with beta 0.02, repair strength 2, the coherent path and the
  enhance off, the 4.5 s stitched print and de-esser threshold 12, on cathar
  0.7.6; the identity reference was re-based on those defaults.
- **cathar music profile**: the speech settings shave music (a print learned
  from music is programme; subtracting it at the speech factor takes 8-16 kHz
  down 14 dB and removes no noise), so `filter_cathar_vhs_pipeline` reads the
  material from the strategy: `profile.tonal_persistence` at or above
  `cathar_music_persistence_min` (0.05) switches the denoise to
  `cathar_music_alpha` and the `cathar_music_enable_*` switches
  (`_material_settings`). The scanner's band ratios cannot make the call
  (speech ratio 1.0 on every archive music clip); the persistence reading
  lives in `modules/tonal_persistence.py`, shared with the harness router, and
  is a whole-file median over 15 s windows above -50 dBFS. Measured: music
  clips 0.064-0.225, speech over a bed 0.007-0.033 (three identity clips
  included), dry dialogue under 0.003; no identity clip crosses the floor, so
  the profile keeps 5/5 bit-identity. Values come from the music autotune
  (`experiments/autotune_music`); re-read them from its `final.json` before
  touching the defaults.
- **Listener-round stages (2026-09-23)**: every new behaviour is a
  config-gated stage the tuning loop switches, shipped off until a loop
  accepts it (the pause floor and the sibilant guard are on since
  2026-09-25; the stem path and the split band stay off). A shared key the
  two materials disagree on gets a music-profile key rather than a
  compromise (`cathar_music_enable_deesser`, `cathar_music_expander_depth_db`,
  `cathar_music_crt_notch_q`, `apl_music_neural_model`; `apl_expander_depth_db`
  where the engines disagree). What the listener heard and where the lever
  is:
  "silent in pauses" is the polish expander (`_build_full_audio_expander_filter`
  pushes what sits under its knee a further 7-10 dB down and maps -90 dBFS
  to -100; `expander_depth_db`, `expander_knee_offset_db`) plus the mask
  denoiser leaving near-silence, so `modules/pause_floor.py` puts the
  source's own pause texture back (`pause_floor_fill_db` under the source's
  pause level, quiet = within 10 dB of the p15 level, only the deficit,
  never above the source) after the expander in both engines; the mux then
  runs `loudnorm ... linear=true`, which ffmpeg silently turns dynamic when
  the measured LRA exceeds the target (`loudnorm_target_lra`) or the
  true-peak rule fails (see "Loudness mode" below; the run log states which
  mode held). "distortion of spoken 's'" is the neural stage
  emptying the 1-4 kHz body under fricatives (`dsp.sib_centroid_hz`
  +370..+620 Hz): `modules/sibilant_guard.py` puts a share of the pre-neural
  high band back inside the fricative events only. The guard has no tonal
  skip, unlike the plosive tamer: the Tata tapes, where the 's' was heard
  distorted, read tonal (flatness 0.022); its detector's own rules (most of
  the hop's energy above the guard frequency, a zero-crossing rate no voiced
  sound reaches, 20-400 ms) keep cymbals and held notes out. At
  `apl_sibilant_hf_share_min` 0.5 it caught 3 events in five minutes of
  Vaccin where the harness found 30 (12.5% of its fricative frames), so the
  loops' mix and crossover moves changed the audio by -70..-100 dBFS only;
  measure a detector-bound stage's coverage (the event log) before tuning
  it. Ear v3 found the thin 's' was the air shelf's whole-spectrum tilt
  (+2 dB at 7.5 kHz lifts every frame about 1.6 dB above 4 kHz); round one's
  "distortion" reads as texture, not level. cathar's hiss and its
  shaved highs get one factor each side of a crossover
  (`modules/split_band.py`, `cathar_split_band_hz`, `cathar_alpha_high`).
  Music loses its stem under a full-mix chain: `modules/apl_stems.py` runs
  the chain on the vocal stem and passes the background through the notches
  and a bounded MMSE floor (`apl_music_bg_floor_db`), keyed like cathar's
  profile on `profile.tonal_persistence`. New stages copy
  `modules/plosive_tamer.py`'s shape (streamed blocks, `atomic_target`,
  gain curves snapped to exactly 0/1 so untouched samples stay bit-exact,
  `STAGE_FAILURES` -> log and return the input).
- **Knob table hygiene**: a key the app overrides per material must be in
  the loop's `KNOBS` or its rounds are inert there (the music loop's rounds
  4-5 moved `cathar_alpha` while the music profile overrode it: three
  candidates rendered byte-identical audio). The loop now hashes every
  candidate's audio per tape and logs a byte-identical one as inert, never
  scored (output-quality-harness skill). A knob is dead only when every
  code path that reads it is: `apl_tonal_flatness_max` stays live with the
  subtraction stage off, because the plosive tamer (`_skip_reason`) and the
  hum canceller's series length (`_series_length`) read it. The tuned APL
  finals carried 0.01, the shipped default is 0.035 and the Tata tapes read
  0.022, so tuned APL ran the tamer and the 40-harmonic hum series where
  shipped APL does not (round A0 decides). `apl_music_persistence_min`
  picks the tapes that take `apl_music_neural_model`, and
  `apl_noiseprint_tonal_s` also feeds the stem path's background suppressor
  (`apl_stems._background_pass`, while `apl_music_bg_floor_db` is below 0).
- **Loudness mode**: loudnorm's applied pass holds linear mode (one gain for
  the programme) only while `TP + (I_target - I) <= TP_target`, the
  measured LRA is within `loudnorm_target_lra` and the measurement is set;
  otherwise it rides the gain and lifts the pause floors after every engine
  has finished. A programme under 3 s runs linear whatever was measured
  (the fixtures and the harness windows are all longer). The run log used
  to check the range half only: of the 616 distinct decisions it logged as
  "linear" in the 1020 run logs under the repository on 2026-10-09, 229
  (37%) broke the true-peak half and ran dynamic (the count moves with the
  logs on disk; `linear_mode_blockers` over the logged I, TP and LRA
  re-derives it). `loudnorm_linear_fallback`
  set to `gain_limiter` answers the true-peak-only case with one gain and the
  existing limiter; the limiter holds sample peaks only, so on three
  synthetic programmes with clicks the render peaked at up to +0.24 dBTP
  where loudnorm's dynamic mode held -1.8 to -1.9. The default stays
  `ffmpeg` (every output keeps its bytes) until the pause-texture round
  judges it by ear; with the event log on the record shows which mode ran.
- **A second cathar build**: `AI_RESTORE_CATHAR_BIN` names another binary
  (kept under `experiments/cathar-<version>/`, hash verified) so an upgrade
  is measured before it replaces `.venv/Scripts/cathar.exe`; 0.7.6 replaced
  0.7.3 on 2026-09-26 after the listener round tuned both engines on it
  (both installers pin the version and the archive checksums). Stages our
  chain calls were bit-identical between 0.7.5 and 0.7.6; the upstream
  `cathar vhs` chain is not a candidate (single quietest-4 s probe, alpha 3:
  colouration -0.43, discontinuity tail -1.61 on Tele7abc; vbasky/cathar#26).
  cathar 0.8.0 is approved for download (the user, 2026-10-08) into
  `experiments/cathar-0.8.0/` with its checksum, not fetched yet: round C0
  runs it behind `AI_RESTORE_CATHAR_BIN`, first through the identity check
  against `experiments/cathar_ab_head.json` (identical means adopt; else the
  ear and a user ABX). The installers stay at 0.7.6 until it is accepted.
- **No generative engines** (the user, 2026-10-08): the enhancement rounds
  and the learned training add no generative stage (no flow, vocoder or
  GRPO-trained enhancer, nothing like Resemble-Enhance's enhancer) to
  `auto_pure_linear` or cathar. The learned changes planned are
  an anchored fine-tune of the Mel-RoFormer denoiser (isolated
  `tools/msst/` venv, a new `apl_neural_model` value, default off until the
  loop and the ear accept it) and, later, a per-tape settings policy that
  offsets post-neural knobs from the shipped defaults (switch off by
  default).
- **Two engines at once**: the work directory is `.temp_work_<stem>` beside the
  source, so two runs on the same file collide. Run engines in parallel only
  on different paths (NTFS hardlinks of the tapes for the second engine).
- **cathar speed**: every cathar.exe stage is single-threaded (CPU time equals
  wall time; `RAYON_NUM_THREADS` changes nothing; measured 2026-09-21 on a
  134 s tape: SBR enhance 12 s, spike repair 3 s, dehum 2 s, denoise 0.3 s)
  and a file's stages run in sequence; on four full tapes the enhance stage
  was 39 % of the chain, spike repair 25 %, ffmpeg's two-pass loudness 17 %.
  Nothing in a stage can be split without changing bits, so the lever is
  `batch_jobs`: files at once, each in a child interpreter, outputs unchanged.
