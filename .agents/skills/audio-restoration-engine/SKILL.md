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
