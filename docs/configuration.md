# Configuration

- **config.yaml**: Manages global settings like audio mix levels
  (`vocal_mix_volume`, `background_mix_volume`), sync behavior, process mode,
  native filter parameters, and file extensions.
- **Defaults**: If `config.yaml` is missing, the script defaults to neutral mix
  levels (1.0), `process_mode: auto_pure_linear`, and standard video extensions
  (`.mp4`, `.mkv`, `.avi`, `.mov`, `.mpg`, `.mpeg`, `.ts`, `.m2ts`).

## Process Modes

- `auto_pure_linear` (default):
  - Full-mix pure-denoising mode for natural archival fidelity.
  - Uses dual-resolution analysis and analog pre-conditioning, then denoises
    once with UVR-DeNoise without stem separation, speech synthesis, or mixing.
  - Output suffix: `*_PureLinear_Cleaned`.
- `cathar` / `cathar_vhs`:
  - Deterministic high-fidelity DSP restoration engine.
  - Applies 8-harmonic adaptive de-hum, surgical CRT whistle notch filter,
    spectral noise print subtraction, de-click/de-crackle, and azimuth phase alignment.
  - `cathar_vhs` is an alias for `cathar`.
  - Output suffix: `*_Cathar_Cleaned`.
- `auto`:
  - Intelligent acoustic profile scan dynamically selects the optimal restoration
    engine and model parameters based on measured noise, clicks, and hum.
  - Suffix: `*_Auto_Cleaned`.
- `multipass_auto` / `multipass`:
  - Maximum-quality 4-pass cascaded restoration.
  - Dual-resolution acoustic scan -> analog pre-conditioning -> stem separation
    & Resemble-Enhance -> residual polish -> DTW Sync -> final master mix.
  - `multipass` is an alias for `multipass_auto`.
  - Suffix: `*_MultiPass_Cleaned`.
- `auto_pure` / `pure`:
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
  - Parameters: `afftdn_nr` (dB reduction, default 10.0), `afftdn_nf` (dB noise
    floor, default -55.0), `afftdn_tn` (adaptive noise tracking),
    `highpass_freq` (rumble cutoff, default 80 Hz), `enable_adeclick` (click
    removal), `notch_freq` (head switching buzz, default 50.0 Hz).
  - Output suffix: `*_FFmpeg_Cleaned`.
- `arnndn_speech`:
  - FFmpeg Recurrent Neural Network (RNNoise) speech/dialogue denoiser.
  - Best for dialogue-heavy VHS recordings.
  - Parameters: `arnndn_model` (default `"cb.rnnn"` in `models/arnndn/`),
    `arnndn_highpass_freq`, `arnndn_enable_adeclick`.
  - Output suffix: `*_Speech_Cleaned`.
