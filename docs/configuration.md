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
  - Uses dual-resolution analysis and analog pre-conditioning, then subtracts a
    learned noise profile, blends the result back toward the original per
    frequency bin, and denoises once with UVR-DeNoise. No stem separation,
    speech synthesis, or mixing.
  - Removes more tape noise than `cathar` while disturbing the programme less,
    measured on 136 real clips in both regions: 174 were run and 38 excluded as
    degenerate sources on which the metric reads the same nonsense for both
    modes. See `docs/cathar_vs_auto_pure_linear_1000_benchmark.md`.
  - Parameters, all specific to this mode so that `cathar` cannot be affected by
    tuning them:
    - `apl_noiseprint_duration_s` (seconds of the quietest stretch used to learn
      the noise profile, default 4.0). The shared `cathar_noiseprint_duration_s`
      stays at 0.75 and is not used here. This is the single most consequential
      setting in the mode: at 0.75 s it removes 3.39 dB less noise than at 2.5,
      and on the full corpus 4 s removes 9.99 dB against 2.5 s's 8.73 at the
      same 0.32 dB of median deviation, with the cost in the deviation tail
      (upper quartile 0.76 to 0.91 dB) and on NTSC (0.33 to 0.45, against
      `cathar`'s 0.57). 6 s removes 10.64 at 0.35 but loses both halves of the
      trade to `cathar` on 13 clips rather than 10. On a long tape the quietest
      4 s is likelier to be pure noise than inside a 15 s corpus clip.
    - `apl_spectral_alpha` (over-subtraction factor, default 3.0). Higher removes
      more noise and takes more programme with it; 4.0 measures worse overall.
    - `apl_spectral_margin_db` (skip subtraction above this
      programme-above-noise margin, default 60.0). Effectively off for real
      tape, and deliberately so -- it guards only already-pristine audio.
    - `apl_enable_spectral_denoise` (default true), `apl_enable_learned_blend`
      (default true).
    - `apl_enable_physical_repair` (default true). Repairs crackle, dropouts,
      saturation and azimuth skew, each gated on its own defect being detected.
      Four cathar stages earned a place against paired fixtures and three were
      rejected: `repair` and `deplosive` make undamaged material measurably
      worse, and `declick` is dominated by `decrackle`. Gating is not an
      optimisation -- applied blanket-fashion, `decrackle` scores -11.09 dB on
      material whose only defect is azimuth skew. Leaving it on is close to free
      across 174 real captures: noise removal 9.66 to 9.74 dB, deviation 0.48 to
      0.50.
    - `apl_depop_threshold` (default 7.0; 0 switches it off). Pop removal ahead
      of `decrackle`, gated on the same click detection: prediction-residual
      outliers this many robust scales out are refilled by autoregressive
      interpolation, and a span is left alone when other outliers crowd it, so
      a drum hit is not a pop. On the calibrated crackle class it repairs
      +5.5 dB inside the pops where `decrackle` recovers +1.7 at any
      sensitivity; on undamaged speech it changes 40 dB under the programme,
      and across 50 real captures it moves neither noise removal nor deviation
      (9.91 to 9.97 dB, 0.22 to 0.22).
    - `apl_enable_hum_cancel` (default **true**), `apl_hum_max_harmonics`
      (default 40) and `apl_hum_bandwidth_hz` (default 1.5), with
      `apl_hum_min_excess_db` (default 6.0) as the detection threshold. The
      mode's own hum canceller: each mains harmonic that stands out as a line
      of its own in the quietest frames is tracked by complex demodulation at
      the frequency it actually sits at -- real hum harmonics sit one to eight
      hertz off the exact series -- and subtracted per channel, ahead of the
      noise probe, so the profile the subtraction learns is hiss and not hum.
      The frequency is the one the recording's harmonics support, 50 or 60 Hz,
      not the region's nominal one (on 10 of 48 hum-carrying tapes those
      differ). In the chain on the 48 corpus tapes that carry hum it takes hum
      removed from a median 0.19 dB to 2.49 (upper quartile 2.64 to 5.87), 26
      tapes past the 2.07 dB cathar's dehum manages run alone at the right
      frequency, better on 34 of 48 and worse by more than a decibel on none;
      the low band moves 1.41 to 1.48 dB across the chain and the broadband
      trade on those tapes 12.91/0.33 to 12.90/0.35. Tonal material is held to
      the eight harmonics of the mains series; on everything else the series
      is read to `apl_hum_max_harmonics`, which is how an EMI buzz is the same
      stage. Four tapes whose lines wander more than five hertz are not helped.
    - `apl_enable_dehum` (default **false**). cathar's adaptive `dehum` inside
      this mode, never requested while the native canceller is on. Off because
      the stage's isolated gain (2.07 dB of hum) collapsed in the chain to
      +0.66 dB on a coin flip: a notch cuts a band whether or not hum is in it.
    - `apl_enable_plosive_tamer` (default **true**) and `apl_plosive_excess_db`
      (default 12.0; 0 switches detection off). Event-gated plosive control: a
      burst under 150 Hz that stands this far over the low band's running level
      and leads the mid band's own rise -- a voice onset lifts both bands, a
      blast lifts the low band alone -- is taken down to the level the band
      held just before it, as a downward expander on the low band, and nothing
      else is touched. Tonal material skips the stage. On the calibrated
      plosive class it recovers 0.89 dB inside the blasts at 0.03 dB of
      collateral where cathar's `deplosive` recovers 1.55 at 0.34 and reads
      -6.7 dB on music-led programme; on 50 real captures it is free, 10.02/0.23
      to 10.02/0.23. A 9 dB threshold costs 0.02 dB of deviation for nothing.
    - `apl_enable_tone_cancel` (default **false**). A canceller for persistent
      lines above 4 kHz that are not the mains series or the CRT line whistle:
      a recorded whine, a buzz. Its first setting looked at the whole spectrum
      and on 50 real captures read sustained notes and missed mains lines as
      persistent lines, costing 0.06 dB of deviation and one capture 4.2 dB.
      Restricted to lines above 4 kHz it finds lines on 19 of the 50 -- mostly
      the field-rate sidebands the surgical notch leaves either side of the
      CRT line -- and moves the medians not at all (10.02/0.23 to 10.02/0.23)
      while one capture loses 10.45 dB of noise removal to it: a line that
      holds still is already in the noise profile and the subtraction
      removes it outright, where the tracker's smoothed envelope leaves a
      residual, and a line that wanders defeats both. Off; selectable for a
      whine the ear finds and the probe missed.
    - `apl_use_native_suppress` (default **false**), `apl_suppress_noise_bias`
      (1.0), `apl_suppress_gain_floor_db` (-20.0) and `apl_suppress_dd_alpha`
      (0.96). The mode's own noise suppressor in cathar's subtraction slot: a
      per-bin MMSE log-spectral gain with a tracked noise level. Off on
      real-tape evidence, the DeepFilterNet lesson again: on the calibrated
      fixtures it lands at 5.9-6.2 dB of log-spectral distance where the
      subtraction lands at 10-12.9, and on 50 real captures in the chain it
      removes 5.83 dB at 0.19 of deviation against the subtraction's 10.02 at
      0.22 (on the tonal 45, 3.06/0.22 against 7.96/0.28). A per-bin estimator
      keeps the low-level programme the quiet frames hold, which the trade
      metric reads as noise left behind. Selectable, with the cathar path as
      its fallback.
    - `apl_neural_model` (default empty) names the UVR model to run after
      subtraction outright; empty follows the chain's own choice. The
      Mel-Roformer denoiser (`denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt`)
      measured 9.39/0.22 against UVR-DeNoise's 10.02/0.23 on 50 captures and is
      not the default. `apl_use_resemble_denoise` (default **false**) puts
      Resemble-Enhance's denoiser -- a masking model; its enhancer is generative
      and is not a candidate -- in UVR-DeNoise's place, falling back to UVR on
      absence or failure.
    - `apl_spectral_alpha_tonal` (default 2.0) and `apl_tonal_flatness_max`
      (default 0.035). On tonal material -- median spectral flatness in 100-5000
      Hz below the threshold, the most tonal third of the corpus -- subtraction
      runs at the gentler factor. At 3.0 that material deviated 0.49 dB against
      cathar's 0.32; at 2.0 it is 0.33 with noise removal still ahead. The gate
      fires on none of the noisier material.
    - `apl_use_deepfilternet` (default **false**). DeepFilterNet3 in place of
      UVR-DeNoise as the neural stage, where it is installed. It wins on synthetic
      fixtures and loses on real tape -- programme deviation 0.22 to 0.66 dB across
      50 captures -- so it is opt-in. It is a from-source dependency (Rust core
      built under MSVC, installed with `--no-deps`); absence falls back to
      UVR-DeNoise.
    - `apl_mute_silence_db` (default -45.0), `apl_mute_min_ms` (default 15.0)
      and `apl_mute_max_ms` (default 200.0) define what counts as a dropout
      rather than a pause. Depth alone is not enough: synthesised speech carries
      genuine digital silence between phrases, so a span also has to be short and
      bounded by programme on both sides.
    - `apl_enable_tonal_cleanup` (default **false**). Rumble removal (`dewind`),
      now separate from dehum. Off because it did not measure as a gain on real
      tape.
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
    removal), `notch_freq` (mains hum, default 50.0 Hz; set 60.0 Hz for NTSC).
  - Output suffix: `*_FFmpeg_Cleaned`.
- `arnndn_speech`:
  - FFmpeg Recurrent Neural Network (RNNoise) speech/dialogue denoiser.
  - Best for dialogue-heavy VHS recordings.
  - Parameters: `arnndn_model` (default `"cb.rnnn"` in `models/arnndn/`),
    `arnndn_highpass_freq`, `arnndn_enable_adeclick`.
  - Output suffix: `*_Speech_Cleaned`.
