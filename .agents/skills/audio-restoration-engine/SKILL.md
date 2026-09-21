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
  - Scans the capture and dispatches the best restoration pipeline.
- **`multipass_auto`** -> `*_MultiPass_Cleaned.<ext>`
  - Pre-scan $\\rightarrow$ pre-conditioning $\\rightarrow$ BS-Roformer $\\rightarrow$
    Resemble-Enhance $\\rightarrow$ shift/DTW $\\rightarrow$ amix.
  - 4-pass cascaded restoration with analog pre-conditioning.
- **`auto_pure`** / **`pure`** -> `*_Pure_Cleaned.<ext>`
  - Pre-scan $\\rightarrow$ pre-conditioning $\\rightarrow$ BS-Roformer
    $\\rightarrow$ UVR-DeNoise
    $\\rightarrow$ de-esser $\\rightarrow$ shift/DTW $\\rightarrow$ amix.
  - Default mode: pure speech and ambient denoise, no vocoder synthesis.
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
     the channels clear `AZIMUTH_MIN_CORRELATION`.
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
   - `_resolve_loudnorm_args` / `_measure_mix_loudness`: Two-pass EBU R128
     normalization, falling back to single-pass if measurement fails.
   - `_final_mix_step`: Stem mix via FFmpeg `amix` with container-dependent
     audio encoding, resample to `PIPELINE_SAMPLE_RATE`, and a true-peak
     limiter.
   - `_final_mux_single_audio_step`: Direct stream video copy with
     container-dependent audio mux.
1. **`modules/auto_scanner.py`**:
   - `_detect_flutter_or_pitch_drift`: Tracks the recorded video line whine as a
     fixed-frequency speed reference, so programme pitch cannot read as drift.
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
- **cathar alpha 2.0**: chosen by ear and by the harness on five real tapes
  (colouration +0.125 against +0.062, discontinuity tail -0.34 against
  -0.42, CER 0.058 against 0.093) for 1.6 dB less removal; 3.5 and Wiener
  lose every listener-side reading.
- **A second cathar build**: `AI_RESTORE_CATHAR_BIN` names another binary
  (kept under `experiments/cathar-<version>/`, hash verified) so an upgrade
  is measured before it replaces `.venv/Scripts/cathar.exe`. Stages our
  chain calls were bit-identical between 0.7.5 and 0.7.6; the upstream
  `cathar vhs` chain is not a candidate (single quietest-4 s probe, alpha 3:
  colouration -0.43, discontinuity tail -1.61 on Tele7abc; vbasky/cathar#26).
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
