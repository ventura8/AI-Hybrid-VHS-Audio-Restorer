# Configuration

- **config.yaml**: Manages global settings like audio mix levels
  (`vocal_mix_volume`, `background_mix_volume`), sync behavior, process mode,
  native filter parameters, and file extensions.
- **Defaults**: If `config.yaml` is missing, the script defaults to neutral mix
  levels (1.0), `process_mode: auto_pure`, and standard video extensions
  (`.mp4`, `.mkv`, `.avi`, `.mov`, `.mpg`, `.mpeg`, `.ts`, `.m2ts`).

## Process Modes

- `auto`:
  - AI Auto-Detection & Restoration Engine.
  - Automatically scans audio characteristics across speech formants,
    environmental textures (birds, cars, ambiance), musical harmonics, and
    analog tape defects to dynamically select the optimal AI/DSP pipeline and
    best AI models (`BS-Roformer`, `UVR-DeNoise`). The selected target is
    `hybrid`, `denoise_only`, or `auto_ffmpeg_native` (or `multipass_auto`
    when `enable_multipass` is enabled); `arnndn_speech` is an explicit
    process mode.
  - Output suffix: `*_Auto_Cleaned`.
- `multipass_auto` / `multipass`:
  - Explicit 4-Pass Cascaded Restoration Engine.
  - Pass 1 (Dual-Resolution Scan) -> Pass 2 (Analog Pre-Conditioning DSP) ->
    Pass 3 (AI Neural Separation & Speech Reconstruction) -> Pass 4 (Residual
    Background Polish & DTW Sync).
  - `multipass` is an alias for `multipass_auto`.
  - Output suffix: `*_MultiPass_Cleaned`.
- `auto_pure` (default) / `pure`:
  - Pure speech & ambient restoration without generative vocoder synthesis.
  - Dual-resolution acoustic scan -> analog pre-conditioning -> AI stem
    separation -> dedicated speech/background UVR-DeNoise + de-esser ->
    DTW/shift sync -> 32-bit float mix with EBU R128 loudness normalization.
  - `pure` is an alias for `auto_pure`.
  - Output suffix: `*_Pure_Cleaned`.
- `hybrid`:
  - Separation + vocal enhancement + background denoise + sync + final mix.
  - Output suffix: `*_Hybrid_Cleaned`.
- `denoise_only`:
  - Full-audio denoise + sync + final remux.
  - No separation or vocal enhancement.
  - Output suffix: `*_Denoised_Cleaned`.
- `auto_ffmpeg_native` / `auto_vhs_native`:
  - Intelligent adaptive FFmpeg DSP restoration with acoustic profile scanning.
  - Automatically analyzes tape hiss noise floor, mains hum / head-switching
    buzz frequencies, motor rumble power, and impulsive click density to
    auto-tune FFmpeg native DSP parameters.
  - `auto_vhs_native` is an alias for `auto_ffmpeg_native`.
  - Output suffix: `*_AutoFFmpeg_Cleaned`.
- `ffmpeg_native` / `vhs_native`:
  - Ultra-fast native FFmpeg DSP restoration (`highpass` + `adeclick` + `afftdn`
    \+ optional `bandreject` notch).
  - Best for continuous tape hiss, mechanical rumble, and impulsive electrical
    clicks without GPU.
  - `vhs_native` is an alias for `ffmpeg_native`.
  - Parameters: `afftdn_nr` (dB reduction, default 12.0), `afftdn_nf` (dB noise
    floor, default -45.0), `afftdn_tn` (adaptive noise tracking),
    `highpass_freq` (rumble cutoff, default 60 Hz), `enable_adeclick` (click
    removal), `notch_freq` (head switching buzz).
  - Output suffix: `*_FFmpeg_Cleaned`.
- `arnndn_speech`:
  - FFmpeg Recurrent Neural Network (RNNoise) speech/dialogue denoiser.
  - Best for dialogue-heavy VHS recordings.
  - Parameters: `arnndn_model` (default `"cb.rnnn"` in `models/arnndn/`),
    `arnndn_highpass_freq`, `arnndn_enable_adeclick`.
  - Output suffix: `*_Speech_Cleaned`.
