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
    subtraction) $\\rightarrow$ neural denoiser (level restore, off)
    $\\rightarrow$ sibilant guard, post-neural suppressor (gated, off),
    polish (expander, air shelf, its gate off), pause floor $\\rightarrow$
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
   - `azimuth_estimate_method` (2026-10-12): `head`, the default, reads the
     first 32768 samples' integer cross-correlation peak with the sign the
     graph does not correct (`_head_azimuth_skew`: a right channel 5
     samples late reads -0.11 ms and ffmpeg's `adelay=0|5S` makes it 10).
     `spans` (`_spans_azimuth_skew`, `modules/azimuth_spans.py`) votes
     GCC-PHAT peaks of up to seven loudest 1.5 s spans (150-7000 Hz, phase
     coherence at least 0.3, agreement within half a sample, a parabola for
     the fraction), abstains on disagreement (no strict majority of the
     counted spans: a 2-2 tie from a print stitched from two parts skewed
     opposite ways would double one part's skew), weak coherence or under
     half a sample, keeps the |r| gate and the inverted-pair rule over the whole
     scanned audio, and returns the sign the graph corrects. IA hunt, 232
     stereo clips, the app's scan and pre-conditioning step on CPU, against
     the same graph without the delay: `head` delayed 43 and lowered the
     mono 4-8 kHz on 40 (median -1.43 dB, skew +1.00 sample); `spans`
     delayed 28 and raised it on 27 (median +0.45 dB), lowering it 0.36 dB
     on capture-a-3102. A full-band whole-file GCC-PHAT reads lag 0 on the
     clips the octave investigation blamed on this stage because a lag-0
     component above the programme band holds most of its bins; read
     azimuth in the programme band. cathar's own `dewind` plus `azimuth`
     left the same residual from either pre-conditioning on ten clips, so
     `auto_pure_linear` is the engine the key changes. The Tata tapes read
     no delay either way (four dead right channels, one identical pair;
     both Gaudeamus captures are dead on the right too). The residual skew
     the docs quote was measured with the estimator's own GCC-PHAT; judge a
     change to the vote with an estimator it does not share. The review
     re-run did (coherence-weighted Welch phase fit, mono 4-8 kHz by Welch:
     same counts, gains within 0.03 dB, residual median 0.26 sample in
     300-4000 Hz). Not judged by ear yet. Scratch evidence:
     `D:\ai_restore_work\scratch_azspan\` (`validate_precond.py`,
     `validate_none.py`, `val_hunt.json`, `cathar_follow.json`) and
     `D:\ai_restore_work\scratch_azspan_review\` (`identity.py`: the default
     scan's readings and every mode's graph equal 57bfdb8's on all 238 hunt
     clips; `review_val.py`, `rv.json`; `edge_probe.py`).
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
     inference, or the Mel-RoFormer that `apl_neural_model` names on APL
     speech. The output, and one a previous run left, is picked by
     `denoise_model.clean_stem` (`_select_denoised_candidate`,
     `_existing_denoised_output`); a directory holding only a residual is
     denoised again.
   - `_ensure_float_pcm`: Restores 32-bit float on any stem a separator emitted
     as fixed-point.
   - `_collect_stem_candidates`: Token-based, case-insensitive stem matching
     across every naming convention `audio-separator` emits.
   - `_final_mix_step`: Stem mix via FFmpeg `amix` with container-dependent
     audio encoding, resample to `PIPELINE_SAMPLE_RATE`, and the mastering
     tail of `modules/mastering.py`.
   - `_final_mux_single_audio_step`: Direct stream video copy with
     container-dependent audio mux.
   - Both mux steps take a `mux_format.MuxPlan` (one ffprobe of the source):
     `-copyts` and `-itsoffset <source audio start>` on each restored input,
     so the track starts where the source's audio did (on bb5fa2a it started
     with the video: a click landed up to 295 ms early on a source whose
     audio began 300 ms late), and the rate the container keeps (MPEG-PS its
     source's, 48 kHz for a rate MP2 cannot take; `mux_sample_rate: source`
     for every container). Without a readable probe the command is today's.
     Never `-copyts` on MPEG-PS or MPEG-TS: their demuxer corrects a
     timestamp jump only without it (two halves joined byte-wise: the
     second half's clicks moved -254/-620 ms in MPEG-PS, +84/+83 ms in
     MPEG-TS). There ffmpeg starts the input at the earliest stream the
     command maps (the video, plus the first audio stream when the
     archival track is kept), and `-itsoffset` is taken against that.
   - `_extract_audio_step`: a source ffprobe reads without an audio stream,
     and an extraction with a header and no samples (an edit list hiding
     every packet: hunt v3's Pooh, Powerpuff, Rocky), raise
     `source_checks.SourceSkipped`; `process_hybrid_audio` logs `[Skip]`
     and returns False. `utils.attempt_cpu_run_with_retry` raises at once on
     a failure `ffmpeg_failures.is_deterministic_failure` reads from the
     output's tail (it used to run Everest's extraction six times). Never
     key it on the bare "Invalid data found when processing input": ffmpeg
     9.0.1 prints it for every broken MP2 frame of a damaged capture on a
     run that exits 0; an unreadable input is its "Error opening input"
     line, and a memory message always keeps the retry.
1. **`modules/denoise_model.py`** and **`modules/denoise_chunking.py`**:
   - `clean_stem`: the one rule for the denoised file, whole or chunked:
     `(No Noise)` (the VR models), then `(<training.target_instrument>)` from
     the `_config.yaml` beside an MDXC checkpoint (`(dry)` for the shipped
     RoFormer), then the first file in name order that is not a residual;
     never `(Noise)` or `(other)`. It reads the yaml from the folder the
     separator loads from (`AUDIO_SEPARATOR_MODEL_DIR`, else `models/`).
   - `mdxc_window` / `grid_for`: an MDXC model's overlap-add window and step
     (`hop x (dim_t - 1)`, `// num_overlap`: 352,800 and 88,200 samples for
     the RoFormer). Its chunks start on the step, carry a window of context
     either side, take no anchor, and are written at the whole file's input
     gain (`whole_file_input_gain`). The VR models keep `PATCH_FRAMES`, the
     anchor and the one-stride context.
   - `_grid_after_fetch` / `processing._model_fetcher`: the yaml reaches the
     store only with the model's first load (the installers do not provision
     the RoFormer), so a `.ckpt` model without one is fetched
     (`download_model_files`, no load) before its chunks are cut; still
     without a usable yaml it runs on the VR grid with a WARNING.
1. **`modules/mastering.py`**:
   - `_resolve_loudnorm_args` / `_resolve_single_track_loudnorm_args`:
     two-pass loudnorm to -16 LUFS, -1 dBTP and `loudnorm_target_lra`,
     falling back to single-pass if measurement fails. The tail after the
     loudness stage is a resample to the mux's rate and
     `LOUDNORM_TRUE_PEAK_LIMITER`, which despite its name is `alimiter` at
     0.891 (-1 dBFS) on the samples, a sample-peak limiter. Its lookahead
     delays the audio 219 samples at 44.1 kHz (4.97 ms; an impulse at
     10000 came out at 10219, loudnorm's dynamic mode adds none), so every
     output sits that late against its video;
     `mux_limiter_latency_compensation: true` adds `latency=1` (impulse kept
     at 10000). Off by default: the cathar identity check compares bytes.
   - The analysis pass decodes ffmpeg's stderr as UTF-8 with replacement: a
     Cyrillic title (IA 1994-6-ntsc20241207, byte 0x8f) used to raise a
     cp1252 UnicodeDecodeError in the reader thread and drop the render to
     single-pass loudnorm.
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
   floor keeper each write (and APL's post-neural hiss gate, when its key is
   on, writes its decision and reading as `post_neural_suppress__...json`)
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
     longer hash as then. The seven post-cache modules' import-time code
     (`import_time_code`) is in the key, since processing imports `sync`,
     `mastering`, `post_neural_suppress` and `neural_level` at start-up;
     only their function bodies are left out.
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
1. **`modules/impulse_repair.py`, `modules/notch_ring_guard.py`**: the pop
   repair and the surgical notch's ring guard (2026-10-11). Facts that
   redirect a diagnosis:
   - A quiet-frame rise on an `auto_pure_linear` output is not
     necessarily the repair. SOTI's 332 lifted quiet frames (of 3,740, 10 dB
     or more at 100-1000 Hz) come from the Q-30 surgical notch at 250 Hz
     ringing into 46 ms gaps after speech; the pop repair and `decrackle`
     lift none. Reproduce before blaming a stage: the pre-neural chain
     re-renders bit for bit on CPU from the stage cache's `key.json`
     (`call.strategy.precondition_filters` through
     `filters._precondition_filter_from_config`, then
     `build_pre_denoise_surgical_filter(hum_cancel=True)`, then the chain),
     checked against `manifest.json`'s `input.pcm_sha256` and the
     `surgical.wav` samples.
   - The render log's "Refilled N pop spans" counts every channel: SOTI's
     pre-conditioning mirrors one channel, so its 5,486 are 2,743 spans
     twice. On clean speech the spans are the residual of speech
     (single-sample, clustered, no louder than the waveform), at 6-12 a
     second per channel on SOTI and Vaccin.
   - `apl_depop_min_prominence_db` (off) refuses a span whose peak residual
     does not stand that far over every residual within 5 ms;
     `apl_surgical_ring_guard` (off) caps each 512-sample frame of the
     notch's output at 1 dB over its input. Both keep today's bytes off and
     wait for a listening session; the numbers are in
     `docs/configuration.md`.
   - Judge the ring guard in the notch's own band too. The 100-1000 Hz
     reading (332 lifted frames down to 3) hides what the broadband cap
     leaves: at 200-300 Hz against the notch's input, 127 of SOTI's quiet
     frames still rise 10 dB or more (491 without it). A band-limited cap
     is the next step if the listening session still hears the ring.
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
   - The DTW warp is linear between whole 40 ms chroma frames
     (`dtw_resolution` 25) and smoothed over 51 samples, so every
     off-diagonal step of the path plays a frame at 0x (the output holds one
     sample's value) or at 2x. Every chain that syncs against `original_wav`
     and keeps its length (`auto_pure_linear`, cathar, the single-track modes)
     shares the source's time base: a warp can only add error. Measured
     2026-10-12 with the harness's time-base reading (R12) on the IA hunt:
     24 of 30 cathar and 15 of 30 APL outputs sent to DTW carry frozen holds
     (177 and 49 in all), none of 205 per engine sent to shift. On five of
     those clips rendered with cathar on the CPU, the DTW warp replayed on the
     kept pre-sync WAV (`AI_RESTORE_TEST_MODE=1`) reads 71 holds (the hunt's
     GPU renders of those clips read 72), the shift output 0.
     `sync_dtw_enable: false` sends `_process_single_track_pipeline` and the
     APL stem path (`length_preserving=True`) to shift; the stem modes keep
     DTW. Default true, outputs unchanged, until a blind block.
   - The CPU DTW path never runs: `_collect_dtw_results` hands every chunk a
     `multiprocessing.Event`, which a `ProcessPoolExecutor` cannot pickle
     ("Condition objects should only be shared between processes through
     inheritance"), so each chunk fails and the clip falls back to shift.
     Only a machine whose CUDA GPU has 4 GB or more (`_should_use_gpu_dtw`)
     warps today. Fixing it would bring the holds to CPU machines.
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
- **Denoised stem and long tapes**: pick a separator output only through
  `denoise_model.clean_stem`, never by `models-scores.json`'s `target_stem`
  (it names `noise` for UVR-DeNoise and UVR-DeNoise-Lite) and never by a
  hard-coded `(No Noise)`: that rule made the chunked path raise on the
  RoFormer's `(dry)` / `(other)`, so from v1.3.3 to the 2026-10-11 fix every
  APL speech tape over the chunk length (57 min on this 62 GB host, 14 min
  on 16 GB) failed; music tapes run UVR and were spared. The whole-file pass
  kept its choice (318 denoise directories on this machine: same file). The
  RoFormer's chunk grid (`grid_for`) reproduced the whole-file render on
  every body sample and at -158 dB at the seams in a CPU replica of its
  overlap-add with a stand-in mask (7.2-minute tape, two 300 s chunks; the
  VR grid drifted to -47 dB), and again with audio-separator 0.47.0's own
  `MDXCSeparator.separate` / `demix` and a stub network driven through
  `_run_denoise_separator` (synthetic 150 s, three 60 s chunks: body equal,
  seams -161 dB; VR grid -23 to -33 dB). On the GPU with the real network
  (2026-10-12, a 374 s SOTI cut, four 120 s chunks against the whole file):
  99.67 % of the `(dry)` samples equal, the rest within -150.5 dBFS around
  the seams; both renders peaked at 2.3-2.4 GB of process memory, so the
  chunk length, still set by UVR's 32 GB/h, is conservative for the
  RoFormer.
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
  touching the defaults. Round C3 (2026-10-09) split three speech keys music
  used to share: `cathar_music_beta` (the denoise floor, was `cathar_beta`
  through `_cathar_denoise_step`'s default), `cathar_music_enable_repair`
  (was `cathar_enable_repair`) and `cathar_music_dewind_cutoff` (was
  `cathar_dewind_cutoff`), defaulting to 0.02, true and 80 so music keeps its
  bytes (`tests/unit/test_cathar_music_round_c3.py` pins HEAD 54048a9's
  command lines). The "double high-pass": every cathar tape goes through the
  shared pre-conditioning `highpass=f=<highpass_hz>` (the scanner's rumble
  pick, 0/45/60/75 Hz, 80 on a failed scan; part of the cached file's
  fingerprint) and then `dewind --cutoff 80` (cathar 0.8.0: "high-pass", 80
  by default). The music cutoff moves the dewind only; the scanned high-pass
  stays, so 40 or 60 Hz gives bass back only down to the scanner's cut. APL
  never reads the music keys: `spectral_denoise` calls the dewind and
  denoise steps with their speech defaults.
- **cathar declip and dehum (round C4, 2026-10-10)**: until then the chain
  ran `declip --method spade --threshold 0.95` on every tape and gave
  `dehum --freq` the scanner's mains family: `_resolve_notch_freq` reads
  `precondition_filters.notch_hz` (`config.notch_freq`, 50, only when the
  strategy carries no `notch_hz`), `_mains_family` maps 100 to 50 and 120
  to 60 and rounds to the nearer of 50 / 60 (55 gives 60), and a `notch_hz`
  of 0 (the scan found no hum) runs no dehum at all. The scanner's hum vote
  (`filters._detect_mains_buzz_notch`) reads only the first 32768 samples of
  the analysed audio, about 0.7 s. `cathar_declip_mode: detected` gates the
  declip on `profile.has_clipping_any_channel`, which
  `auto_scanner._detect_clipping_any_channel` sets by running
  `filters._detect_analog_clipping` on the multichannel array: 20 or more
  samples at `|x| >= 0.985` counted over every channel of the scanned
  minutes (the first 60 s of a tape under 4 minutes, else four spread 60 s
  blocks), the count vhs_native's own scan takes. The mono-mix reading
  (`has_clipping`) could not serve: on a tape with one dead channel (half
  the 21-tape library) the mix halves the live side, so it never reaches
  0.985, while the pre-conditioning mirrors that side to full scale before
  cathar runs (a 10 s file with 95,592 clipped samples on its live side
  scanned `has_clipping` False). Wherever the mix reaches the threshold one
  channel does too, so the count never misses a tape the mix flags.
  `has_clipping` is left alone: it still switches the pre-conditioning
  `adeclip` and `physical_repair._stage_plan`'s declip. `_clipping_verdict`
  is three-way: the bool, or None when there is no strategy, profile or
  bool (the scanner's fallback strategy on unreadable audio carries no
  reading); `_declip_on_verdict` declips on True, logs the skip at INFO on
  False, and on None declips as on `always` with a WARNING, so a failed
  detection never silently drops the stage. The remaining blind spots:
  clipping outside the scanned minutes, and peaks between cathar's 0.95 and
  the detector's 0.985. `cathar_dehum_freq: 0`
  passes `dehum --freq 0` on every tape (0.8.0: "`0` auto-detects from the
  recording"); `_cathar_dehum_step(autodetect=True)` ignores `freq`, and
  without it a missing or non-positive `freq` still skips the stage, which
  `spectral_denoise._dehum` relies on. Probed on synthetic files with 0.8.0:
  declip gave a file peaking at 0.5 back sample for sample and reshaped every
  sample over 0.95 on unclipped files peaking at 0.97 and 1.0 (peaks to 0.96
  and 0.99; a clipped file's peaks went to 1.6); `dehum --adaptive` gave
  identical output at `--freq` 0, 50 and 60 on 50 Hz hum and returned a
  hum-free file unchanged, while without `--adaptive` `--freq 0` notched the
  60 Hz series on the hum-free file. So with the shipped `--adaptive` the
  dehum key shows only on tapes the scanner read as hum-free. Both keys
  default to the old behaviour (`tests/unit/test_cathar_round_c4.py` pins
  HEAD 549705a's command lines for ten strategies and APL's two calls); APL
  never reads them: its physical repair keeps its own `has_clipping` plan
  and its tonal cleanup calls the dehum step without `autodetect`.
- **Pause floor target and shape (2026-10-12)**: the hiss diagnosis of
  2026-10-11 found the pause floor is where `auto_pure_linear` puts the tape
  hiss back. The Mel-RoFormer takes the true pauses to about -93 dB; the
  flat fill brings them to -10.9 dB (SOTI) and -4.1 dB (Tele7abc), -7.1 dB
  median over the hunt, and lifts the floor under the programme 4-7 dB; on
  cathar's Tele7abc excerpt it sets the pauses too. Its one target misses
  three ways: the quiet cluster's median sits 9-15 dB over the true pauses
  on dense speech or music; where the neural stage passed the floor the
  fill sums on top of it (dvd-205405-0, +5.6 dB over the source); and
  `denoise_chunking.WHOLE_FILE_PEAK` (0.9) drops the programme up to
  10.6 dB across the separator on hot or declipped sources, which puts
  every absolute threshold after it off (the fill target, the expander
  knee). `pause_floor_target: frame` aims each quiet frame at `fill_db`
  under the source frame times `programme_gain` (restored over source
  energy on the source's p70+ frames) and gives a frame already at its
  target nothing. Its skip rule is "no quiet frame 0.5 dB short": the
  cluster's 95 % share skipped abc-primetime-1991 and left its 0.5 s of
  true pause at -77.6 dB. `pause_floor_shape: shelf | lowpass` fills from a
  zero-phase low-passed copy (`shaper_for`; `shaped_block` streams it with
  16384 samples of context and matches a whole-file `sosfiltfilt` to
  1e-9), and reads the deficit through the same shaping. That is the body
  guard: cathar leaves hiss bursts in its pauses, which read broadband as a
  target met, so the flat-read deficit kept Tele7abc's pause low band at
  -17.8 dB where the guard keeps -14.5. The CPU replay of 2026-10-12 (the
  local scratch scripts `D:\ai_restore_work\scratch_pfloor\replay.py` and
  `compare.py`, measured with the diagnosis's own `measure.py`): the
  defaults wrote the shipped paused WAV sample for sample on all 235 hunt,
  13 diagnosis and 2 cathar work directories; `frame` gave the prototype
  P1a's readings exactly (pause hiss -13.4 dB, hunt median), `frame` plus
  `shelf` P1b's within 0.3 dB but on three clips the guard moved 1.0-1.8 dB
  (-30.3 dB, pause low band -7.6 dB as with no fill). A non-default fill
  names its output (`paused_12db_p15_frame_shelf3000c18_<input>`), so no
  render reuses another's. Open: a GPU render and a listening session; the
  shelf trips the -25 dB dead-air flag on 16 of 234 hunt clips (11 today,
  24 with no fill); the programme gain reads low on a tape whose loud
  frames are noise; and no fill touches the RoFormer's residual hiss bursts
  near speech (2-11 % of pause frames, most of the pause hiss energy) or
  its weak 3-4.8 kHz removal under the programme.
- **APL post-neural hiss stages (2026-10-12)**: three opt-in keys, each off
  (`docs/configuration.md`, "auto_pure_linear Hiss After the Neural
  Stage"). `modules/hiss_floor.py` reads the restored floor under the
  programme the way the diagnosis's `measure.py` did (20 ms Hann frames, ERB
  bands 3-16 kHz, a four-band activity detector on the source, the p10 over
  +-0.75 s of programme frames, then the median; the restored file matched
  by the loud-frame projection gain in 100-4000 Hz) against the
  pre-conditioned audio, the reference the pause floor uses. That reference
  is the right one: on a dead-channel tape the extracted source's downmix
  carries the dead side's hiss and read the neural stage at -9.9 dB in
  3-4.8 kHz where its own input says -3.7. A band is read only 10 dB over
  16-bit dither: an absolute -90 dB cut let one tape's empty 14.3-16 kHz
  band (-96 dB on the source, -67 dB after pre-conditioning) move its
  8-16 kHz stationarity from 2.5 to 4.6. The pair is assumed sample-aligned
  (the separator's output is); a lag only lowers the projection, so both the
  gate and the level restore err towards doing less, and under a
  correlation of 0.5 they refuse (6 of 235 hunt clips, one a 31-sample
  shift). `modules/post_neural_suppress.py` opens the suppressor and the air
  gate on one reading; its output is bit-identical to the prototype P2's
  `p2_suppress.wav` on all 13 diagnosis clips, so the replay's 61-clip match
  of P2's numbers is exact, not approximate (the gate opens on 9 of P2's
  10; commercials-qtq reads a 3.3 dB gap). Two interactions found on
  replay: the shipped pause floor refills the pauses the suppressor empties
  (pause hiss +2.0 dB at the median, up on 19 of 27 clips with pauses and
  by 29-81 dB on 7, where its deficit rule switched on; the floor under
  the programme up on 6, by up to 19 dB), so never switch the suppressor on
  with today's pause floor: judge it with the per-frame fill. With the
  level restored the floor's quiet-cluster rule can find nothing to fill
  (fourstarrevue's true pauses stay at -55 dB) and the expander's knee
  lifts musical noise on two hot clips (chuckle-toons lkr 0.84 to 2.10).
  The adversarial review (2026-10-12) re-ran the replay through
  `processing._denoise_and_polish_full_audio_step` itself, only the model
  and the cleanup stood in: the defaults rendered the shipped bytes on all
  248 work folders, and every number above came back exact. The
  suppressor reuses `apl_suppress_*` and `apl_noiseprint_duration_s`, which
  the subtraction stage reads before the cached point, so a loop tuning
  them misses the stage cache; twins would replay it. On an 8 s synthetic
  fixture the probe holds programme and the suppressor takes 2 dB, against
  8.9 dB on tape: test its effect on real stage WAVs, not on fixtures. The
  air gate reads nothing when `enable_linear_air` is off or
  `linear_air_gain_db` is 0: there is no shelf to skip.
- **cathar noise-print mute guard (2026-10-12)**: the print's pick took the
  quietest window whatever it held (the single-window pick's only guard,
  `1e-6 < rms` with `rms = sqrt(ms + 1e-12)`, rejects an all-zero window
  and nothing else; the stitched pick has none and always keeps the
  quietest window, exact zeros included), so on an edited upload it learns
  a digital mute and removes nothing; 7 of 136 IA clips picked a window
  under -70 dBFS. `cathar_noiseprint_mute_guard`
  (off; `modules/noiseprint_guard.py`) leaves out a window with half or more
  of its samples held, or under -70 dBFS and 30 dB under the 23 ms frame
  p20 (frames under -90 dBFS out of that p20), in both picks, and drops the
  print when nothing is left. Measured on CPU on the 7 clips plus
  bbc-two-continuity (whose -82.7 dBFS window is an analog floor and stayed
  sample for sample): removal +0.42 dB median (0.01 to 2.60); lion-king-dodge
  lost 3.4 dB at 8-16 kHz because its new window straddles the mute and the
  programme before it. Two facts decided the design: cathar's own estimate
  (`denoise` without `--noiseprint`) also reads the mute (0.48 dB of hiss
  removed net of gain on a synthetic file without one, 0.0 dB with a 1 s
  mute anywhere in it, against 9.7 dB with a print learned from 0.75 s of
  its hiss), so dropping the print on a mute pick changes nothing (within
  0.03 dB on all 7); and rejecting straddling windows as well moves the pick
  onto programme on dense clips (lion-king-dodge: +6.7 dB removed, 9.6 dB
  less gap air). On the opt-in digital-mute fixtures
  (`make_realistic_fixtures_v2.py --digital-mute`) the key took the
  learned window's distance from the fixture's own noise from 56.0, 63.2
  and 14.3 dB LSD to 3.9, 2.3 and 5.1 on the three voices whose pick was a
  mute. On the seven distinct local full tapes (stitched path, raw decode)
  the level rule rejected nothing (no window under -69 dBFS); the held rule
  moved the stitched pick on two, a 1 s digital lead-in and a digital tail,
  and the single-window pick on none. Numbers and the table:
  `docs/configuration.md`, "cathar Noise-Print Mute Guard". Pending a blind
  listening session.
- **cathar noise-print depth guard (2026-10-12)**: on a 15 s clip with no
  pause the quietest 0.75 s is programme, and the learned print strips the
  bed (56 of cathar's 64 `stems.octave_programme` failures on the IA hunt).
  `cathar_noiseprint_min_depth_db` (0, off; `modules/noiseprint_depth.py`)
  re-picks the print's windows with the extractor's own code
  (`cathar._probe_windows`; the stitched pick is now `_stitched_picks`,
  shared with the extractor) and measures their RMS against the p90 of the
  mono mix's 1024-sample frame RMS; a shallower print is scaled to the key
  (`cathar_noiseprint_shallow_action: scale`, a sibling
  `<print>_depth<D>db.np.json`) or dropped (`drop`). Verified on cathar
  0.8.0: the print JSON is a list of 1025 mean |rfft| (symmetric Hann 2048,
  hop 512) of the probe's left channel, linear in level, so scaling the
  file equals learning from a quieter probe. Through the module at the
  denoise stage on the 159 speech-profile hunt clips (CPU): at 20, 25 of
  62 failing clips still fail on `scale`, 17 on `drop`, none new (the
  investigation's emulation exactly; deep prints wrote the shipped bytes);
  at 13, 38 and 22. The fact that decides the default: the four Tata tapes'
  stitched pauses sit only 13.7-16.9 dB under their loud frames, so 20
  fires on all four (removal at the denoise stage 12.6-17.0 dB down to
  6.6-13.0 on `scale`, to 0 on `drop`), while 13 leaves all four byte for
  byte. Depth alone cannot tell a dialogue tape's pauses from a dense
  clip's programme (8.3 dB median on failing clips, 13.0 on passing); a
  single-window-only guard would. Scripts and data:
  `D:\ai_restore_work\scratch_npdepth` (`validate_guard.py` with
  `GUARD_D`, `analyze_validate.py`, `run_tata.py`, `tata_compare.py`).
  Pending a blind listening session.
- **cathar quantile noise print (2026-10-12)**:
  `cathar_noiseprint_method: quantile` (default `window`;
  `modules/noiseprint_quantile.py`) writes cathar's print from the whole
  stage input instead of the quietest window: per bin, the 5th percentile
  of the 32-frame (0.37 s) moving mean of cathar's own frames (left
  channel, symmetric Hann 2048, hop 512, whole frames; verified to 4.5e-7
  against `cathar noiseprint`), +0.62 dB, over windows with no digital-mute
  frame (under -90 dBFS or half its samples held), from a streamed 0.1 dB
  histogram. Under 20 live windows, or on a read or write error, it goes
  back to the window pick; the window path's guards do not apply. How the
  estimator was chosen: realistic-v2's vhs minus clean is the exact tape
  noise, and the bank's noise is no Rayleigh magnitude (mean/p10 4.7-5.7 at
  5 kHz against 2.73, codec holes above 10 kHz), so the textbook raw
  quantile with the Rayleigh correction misses by 5-20 dB; over 1-64 frames
  and q 0.05-0.5 with the correction fitted as the median per-bin error,
  K32 q0.05 reads closest (median LSD 4.22 dB over 300 tape-noise-only
  fixtures, window 4.93, stitched 8.59). The window is closer on fixtures
  with pauses but reads the hiss band 1.3-2.4 dB high; on music with no
  pause it learns programme (+5-6 dB, 45-48 % of bins over +6 dB). On the
  IA hunt (denoise stage, CPU) 25 of the 56 denoise-class octave failures
  stop failing, 3 of 96 passing clips start (a near-silent window and a
  digital mute the window had learned); the hiss floor under the programme
  comes down 10.2 dB instead of 14.1. On the Tata tapes it leaves 1.5-3.1
  dB more hiss above 7 kHz and stops the stitched print's 24-32 dB cut of
  speech at 3-6 kHz (soti, tele7abc): the stitched pauses carry speech.
  `cathar_alpha` 1.25 with it brings the hiss removal back to the window's
  (37 of 56 still failing, 7 passing clips start). The fact to remember: at
  alpha 1.0 a correct print removes less than the over-reading one the
  listener round tuned against, so judge `quantile` with alpha, not alone.
  Scripts and data: `D:\ai_restore_work\scratch_qprint` (`tools/calibrate3.py`,
  `cal3_table.py`, `hunt_validate.py`, `hunt_alpha.py`, `run_tata.py`,
  `tata_measure.py`, `tata_alpha.py`). Pending a blind listening session.
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
  to -100; `expander_depth_db`, `expander_knee_offset_db`, and its timing
  `expander_attack_s` / `expander_decay_s`) plus the mask
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
- **Air shelf corner and expander timing**: `_build_linear_air_filter`
  reads `linear_air_freq_hz` (7500 Hz, the shelf's `treble=f`) and
  `_build_full_audio_expander_filter` reads `expander_attack_s` and
  `expander_decay_s` (0.04 / 0.18 s, the `compand` timing), all in
  `modules/filters.py`. They were hard-coded until 2026-10-09 and their
  defaults build the same strings: `tests/unit/test_polish_knobs.py` pins
  the default graphs, and a probe of 2010 argument and switch combinations
  matched HEAD 8d57030's builders byte for byte. Only APL runs the shelf.
  The expander is one stage that `cathar` (its mode calls
  `processing._polish_full_audio_step` with `apply_air=False`),
  `auto_pure_linear` and `denoise_only` share, so its timing moves both
  engines. All three keys are read after the neural stage (stage-cache
  allowlist) and are loop knobs for rounds A1 and A3; none is judged by ear.
  Round 3 heard every air shelf at 7500 Hz, so the loop judges a corner move
  with the gain against its +2 dB boundary (+1.5 dB at 6000 Hz lifts 4-8 kHz
  like +2.57 dB at 7500 Hz: `autotune_guards.ledger_air_gain`).
- **A second cathar build**: `AI_RESTORE_CATHAR_BIN` names another binary
  (kept under `experiments/cathar-<version>/`, hash verified) so an upgrade
  is measured before it replaces `.venv/Scripts/cathar.exe`; 0.7.6 replaced
  0.7.3 on 2026-09-26 after the listener round tuned both engines on it
  (both installers pin the version and the archive checksums). Stages our
  chain calls were bit-identical between 0.7.5 and 0.7.6; the upstream
  `cathar vhs` chain is not a candidate (single quietest-4 s probe, alpha 3:
  colouration -0.43, discontinuity tail -1.61 on Tele7abc; vbasky/cathar#26).
  cathar 0.8.0 (round C0, 2026-10-09): the Windows archive, checksum
  `6c85b985...`, sits in `experiments/cathar-0.8.0/bin/`. Behind
  `AI_RESTORE_CATHAR_BIN` it gave identical audio on the five
  `cathar_ab_head.json` clips (`experiments/cathar_ab_cathar080_c0`) and on
  three of them in `auto_pure_linear` (`experiments/apl_cathar_identity`,
  the hashes the stage-cache proof read too), so it was adopted without a
  listening session: both installers pin 0.8.0 and its four archive
  checksums, and `.venv/Scripts/cathar.exe` is 0.8.0 (0.7.6 kept in
  `experiments/cathar-0.7.6/`). The stage cache keys the binary's content,
  so entries made under 0.7.6 miss once and are rebuilt.
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
