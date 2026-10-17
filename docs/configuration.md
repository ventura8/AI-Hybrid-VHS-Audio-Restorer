# Configuration

- **config.yaml**: Manages global settings like audio mix levels
  (`vocal_mix_volume`, `background_mix_volume`), sync behavior, process mode,
  native filter parameters, and file extensions.
- **Defaults**: If `config.yaml` is missing, the script defaults to neutral mix
  levels (1.0), `process_mode: auto`, and standard video extensions
  (`.mp4`, `.mkv`, `.avi`, `.mov`, `.mpg`, `.mpeg`, `.ts`, `.m2ts`).

## Process Modes

- `auto_pure_linear`: - Full-mix pure-denoising mode for natural
  archival fidelity. - Uses dual-resolution analysis and analog
  pre-conditioning, then subtracts a learned noise profile, blends the
  result back toward the original per frequency bin, and denoises once with
  UVR-DeNoise. No stem separation, speech synthesis, or mixing. - Removes
  more tape noise than `cathar` while disturbing the programme less,
  measured on 136 real clips in both regions: 174 were run and 38 excluded
  as degenerate sources on which the metric reads the same nonsense for both
  modes. See `docs/cathar_vs_auto_pure_linear_1000_benchmark.md`. -
  Parameters, all specific to this mode so that `cathar` cannot be affected
  by tuning them: - `apl_noiseprint_duration_s` (seconds of the quietest
  stretch used to learn the noise profile, default 4.0). `cathar` has its own
  `cathar_noiseprint_duration_s`, 6 s stitched from eight 0.75 s pauses on
  a tape of 120 s or more (twenty times the print) and one 0.75 s window on
  anything shorter (so
  corpus clips are unchanged); it is not used here. This is
  the single most consequential setting in the mode: at 0.75 s it removes
  3.39 dB less noise than at 2.5, and on the full corpus 4 s removes 9.99 dB
  against 2.5 s's 8.73 at the same 0.32 dB of median deviation, with the
  cost in the deviation tail (upper quartile 0.76 to 0.91 dB) and on NTSC
  (0.33 to 0.45, against `cathar`'s 0.57). 6 s removes 10.64 at 0.35 but
  loses both halves of the trade to `cathar` on 13 clips rather than 10. On
  a long tape the quietest 4 s is likelier to be pure noise than inside a 15
  s corpus clip. - `apl_spectral_alpha` (over-subtraction factor, default
  3.0). Higher removes more noise and takes more programme with it; 4.0
  measures worse overall. - `apl_spectral_margin_db` (skip subtraction above
  this programme-above-noise margin, default 60.0). Effectively off for real
  tape, and deliberately so -- it guards only already-pristine audio. -
  `apl_enable_spectral_denoise` (default **false** since the listener round of
  2026-09-25: on the four Tata tapes the self-driving loop judged the chain
  better without the subtraction stage on the pause and sibilance readings,
  and the music loop kept it off; with the stage off, the subtraction's own
  settings on this page do nothing until it is switched back on),
  `apl_enable_learned_blend` (default true). - `apl_enable_physical_repair`
  (default true). Repairs
  crackle, dropouts, saturation and azimuth skew, each gated on its own
  defect being detected. Four cathar stages earned a place against paired
  fixtures and three were rejected: `repair` and `deplosive` make undamaged
  material measurably worse, and `declick` is dominated by `decrackle`.
  Gating is not an optimisation -- applied blanket-fashion, `decrackle`
  scores -11.09 dB on material whose only defect is azimuth skew. Leaving it
  on is close to free across 174 real captures: noise removal 9.66 to 9.74
  dB, deviation 0.48 to 0.50. - `apl_depop_threshold` (default 7.0; 0
  switches it off). Pop removal ahead of `decrackle`, gated on the same
  click detection: prediction-residual outliers this many robust scales out
  are refilled by autoregressive interpolation, and a span is left alone
  when other outliers crowd it, so a drum hit is not a pop. On the
  calibrated crackle class it repairs +5.5 dB inside the pops where
  `decrackle` recovers +1.7 at any sensitivity; on undamaged speech it
  changes 40 dB under the programme, and across 50 real captures it moves
  neither noise removal nor deviation (9.91 to 9.97 dB, 0.22 to 0.22). -
  `apl_depop_min_prominence_db` (default 0.0, off; every output keeps its
  bytes). The pop repair's prominence guard: a span is refilled only when
  its peak prediction residual stands this many dB over every other residual
  within 5 ms either side. On real tape the threshold alone refills far more
  than the 4 spans a second the calibrated class allowed: the shipped renders
  of the four local interviews logged 3.9 to 24.2 a second over both
  channels, on sources the harness reads as free of clicks. On SOTI and
  Vaccin those spans are single-sample outliers in loud speech, 79-80% of
  them within 15 ms of another and 98% no more than 1 dB over the waveform
  around them; they lift no quiet frame. Per channel at 0 / 6 / 9 dB: SOTI
  6.32 / 0.50 / 0.05 spans a second, Vaccin 12.11 / 1.38 / 0.27, the
  calibrated speech classes 2.75 / 0.17 / 0.04 (15 dB margin) and 5.20 /
  0.41 / 0.16 (9 dB); inside the pops of the five crackle fixtures +4.67 /
  +4.35 / +4.25 dB of repair, 80 / 75 / 74% of the pops refilled. 6 dB waits
  for a listening session. A value over 120 is refused with a warning and
  leaves the guard off (past about 6,165 dB its amplitude ratio would overflow
  and end the render). -
  `apl_enable_hum_cancel` (default **true**), `apl_hum_max_harmonics`
  (default 40) and `apl_hum_bandwidth_hz` (default 1.5), with
  `apl_hum_min_excess_db` (default 6.0) as the detection threshold. The
  mode's own hum canceller: each mains harmonic that stands out as a line of
  its own in the quietest frames is tracked by complex demodulation at the
  frequency it actually sits at -- real hum harmonics sit one to eight hertz
  off the exact series -- and subtracted per channel, ahead of the noise
  probe, so the profile the subtraction learns is hiss and not hum. The
  frequency is the one the recording's harmonics support, 50 or 60 Hz, not
  the region's nominal one (on 10 of 48 hum-carrying tapes those differ). In
  the chain on the 48 corpus tapes that carry hum it takes hum removed from
  a median 0.19 dB to 2.49 (upper quartile 2.64 to 5.87), 26 tapes past the
  2.07 dB cathar's dehum manages run alone at the right frequency, better on
  34 of 48 and worse by more than a decibel on none; the low band moves 1.41
  to 1.48 dB across the chain and the broadband trade on those tapes
  12.91/0.33 to 12.90/0.35. Tonal material is held to the eight harmonics of
  the mains series; on everything else the series is read to
  `apl_hum_max_harmonics`, which is how an EMI buzz is the same stage. Four
  tapes whose lines wander more than five hertz are not helped. Three gates
  keep programme out of the series: a series whose lines place the
  fundamental outside the half-hertz window is refused; a gated line further
  off an exact multiple of the fundamental than half the band the canceller
  tracks it in (0.75 Hz at the fundamental, a quarter more per harmonic) is
  a partial, not a harmonic; and a series with nothing at the fundamental is
  a voice or a note pitched at a multiple of the mains frequency, and is
  refused unless the shared scanner reported the hum and the
  pre-conditioning notched the fundamental ahead of the stage. The
  calibrated music-only class had the canceller taking partials at 98, 147
  and 196 Hz for hum, and on two tonal tapes the same reading was hiding as
  hum removal (the low band moved 0.91 and 1.50 dB before, 0.11 and 0.01
  after); with the gates the class reads free. On the 33 readable hum tapes the
  chain then removes a median 1.19 dB of excess and takes the lines
  themselves down 3.04 dB, against `cathar`'s -0.74 and 0.75, ahead of it on
  24 and 26 of the 33, with the low band moved 0.48 dB against `cathar`'s
  0.55 -- less hum removed than the 2.43 dB the ungated canceller read, and
  less programme taken with it. -
  `apl_surgical_mains_notch` (default **true**) and `apl_hum_skip_notched`
  (default **false**). Two switches for the notches that precede the
  canceller: the mode's own bandrejects at the third to fifth mains
  harmonics, and the shared pre-conditioning's at the fundamental and its
  second. Leaving the higher harmonics to the canceller (the first off)
  loses hum removal on the 33 readable hum tapes, 2.43 to 1.71 dB of excess
  at the median and worse by more than a decibel on 8 tapes against better
  on 1; leaving the two pre-conditioned harmonics out of the canceller's
  plan and refinement (the second on) changes nothing measurable. The notch
  and the canceller do better together. - `apl_surgical_ring_guard`
  (default **false**; every output keeps its bytes). Caps each 512-sample
  frame of those third-to-fifth harmonic notches' output at 1 dB over the
  frame's input energy (`modules/notch_ring_guard.py`). A Q-30 notch rings at
  its own frequency when the audio feeding it stops, so speech holding a
  partial near a notch and stopping dead hands the gap that follows a tone
  burst. On SOTI the notch lifted 332 of the 3,740 quietest 23 ms frames by
  10 dB or more at 100-1000 Hz, up to +28 dB (46 ms gaps inside speech at
  111.15, 100.98 and 48.81 s filled with 237-258 Hz ringing at -27 dB), and
  the finished output kept them (323 of its 331 lifted frames are the
  notch's) where `cathar`'s lifts 4. With the
  guard the notch's output lifts 3, its quiet frames sit 0.80 dB over the
  source's on average instead of 5.83, and 77% of its samples pass to the
  bit. The cap is broadband, so where a gap's floor is hiss the ringing can
  still stand over its own band: at 200-300 Hz, against the notch's input,
  127 quiet frames still rise 10 dB or more (491 without the guard). Vaccin,
  whose gaps the notch does not fill, moves on no reading, and
  on the calibrated hum fixtures the lines stay as far under the clean
  reference as the notch put them. Off until a listening session judges it;
  like `apl_surgical_mains_notch` it reaches `denoise_only`'s notch too. -
  `apl_enable_dehum` (default
  **false**). cathar's adaptive `dehum` inside this mode, never requested
  while the native canceller is on. Off because the stage's isolated gain
  (2.07 dB of hum) collapsed in the chain to +0.66 dB on a coin flip: a
  notch cuts a band whether or not hum is in it. -
  `apl_enable_plosive_tamer` (default **true**) and `apl_plosive_excess_db`
  (default 12.0; 0 switches detection off). Event-gated plosive control: a
  burst under 150 Hz that stands this far over the low band's running level
  and leads the mid band's own rise -- a voice onset lifts both bands, a
  blast lifts the low band alone -- is taken down to the level the band held
  just before it, as a downward expander on the low band, and nothing else
  is touched. Tonal material skips the stage. On the calibrated plosive
  class it recovers 0.89 dB inside the blasts at 0.03 dB of collateral where
  cathar's `deplosive` recovers 1.55 at 0.34 and reads -6.7 dB on music-led
  programme; on 50 real captures it is free, 10.02/0.23 to 10.02/0.23. A 9
  dB threshold costs 0.02 dB of deviation for nothing. -
  `apl_enable_tone_cancel` (default **false**). A canceller for persistent
  lines above 4 kHz that are not the mains series or the CRT line whistle: a
  recorded whine, a buzz. Its first setting looked at the whole spectrum and
  on 50 real captures read sustained notes and missed mains lines as
  persistent lines, costing 0.06 dB of deviation and one capture 4.2 dB.
  Restricted to lines above 4 kHz it finds lines on 19 of the 50 -- mostly
  the field-rate sidebands the surgical notch leaves either side of the CRT
  line -- and moves the medians not at all (10.02/0.23 to 10.02/0.23) while
  one capture loses 10.45 dB of noise removal to it: a line that holds still
  is already in the noise profile and the subtraction removes it outright,
  where the tracker's smoothed envelope leaves a residual, and a line that
  wanders defeats both. Off; selectable for a whine the ear finds and the
  probe missed. - `apl_use_native_suppress` (default **false**),
  `apl_suppress_noise_bias` (1.0), `apl_suppress_gain_floor_db` (-20.0) and
  `apl_suppress_dd_alpha` (0.96). The mode's own noise suppressor in
  cathar's subtraction slot: a per-bin MMSE log-spectral gain with a tracked
  noise level. Off on real-tape evidence, the DeepFilterNet lesson again: on
  the calibrated fixtures it lands at 5.9-6.2 dB of log-spectral distance
  where the subtraction lands at 10-12.9, and on 50 real captures in the
  chain it removes 5.83 dB at 0.19 of deviation against the subtraction's
  10.02 at 0.22 (on the tonal 45, 3.06/0.22 against 7.96/0.28). A per-bin
  estimator keeps the low-level programme the quiet frames hold, which the
  trade metric reads as noise left behind. Selectable, with the cathar path
  as its fallback. - `apl_neural_model` names the UVR model to run outright;
  empty follows the chain's own choice. The default is the Mel-RoFormer
  denoiser (`denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt`) since the
  listener round: on the corpus trade metric it measured 9.39/0.22 against
  UVR-DeNoise's 10.02/0.23 on 50 captures, and the listener harness on the
  Tata tapes preferred it in every round. `apl_music_neural_model` (default
  empty) is the model on music (held partials at or above
  `apl_music_persistence_min`): the music loop took the Mel-RoFormer back off
  there in its first round (hard-gate failures 35 to 30 on 12 clips), so music
  follows the chain's choice. - `apl_expander_depth_db` (default 12.0) is this
  mode's polish expander depth (see the listener-round settings below).
  `apl_use_resemble_denoise` (default **false**) puts Resemble-Enhance's
  denoiser -- a masking model; its enhancer is generative and is not a
  candidate -- in UVR-DeNoise's place, falling back to UVR on absence or
  failure. On 50 real captures it removes 10.78 dB against UVR-DeNoise's
  10.02 and moves the programme 0.58 dB against 0.23 (upper quartile 0.49 to
  1.77), winning both halves of the trade against `cathar` on 12 captures
  where the shipped stage wins on 28: a studio-speech masking model takes
  tape programme for noise, the DeepFilterNet finding again. -
  `apl_noiseprint_tonal_s` (default 4.0, the general probe's length) and
  `apl_tonal_skip_neural` (default **false**). Two settings for tonal
  material, both measured on the most tonal 45 corpus clips and both held: a
  shorter probe there buys nothing (2.5 s reads 8.09/0.28 against 9.05/0.30
  and loses both halves of the trade to `cathar` on 6 clips against 4; 1 s
  reads 5.77/0.28 and loses on 10), and leaving the neural stage out is a
  trade rather than a gain (8.84/0.31, the deviation's upper quartile 0.96
  to 0.84, 27 clips won against 24 and 5 lost against 4). The tonal losses
  are neither the probe's nor the neural stage's. -
  `apl_spectral_alpha_tonal` (default 2.0) and `apl_tonal_flatness_max`
  (default 0.035). On tonal material -- median spectral flatness in 100-5000
  Hz below the threshold, the most tonal third of the corpus -- subtraction
  runs at the gentler factor. At 3.0 that material deviated 0.49 dB against
  cathar's 0.32; at 2.0 it is 0.33 with noise removal still ahead. The gate
  fires on none of the noisier material. The threshold also decides
  whether the plosive tamer runs and how long the hum canceller's series
  is, so it stays live with `apl_enable_spectral_denoise` off: the Tata
  tapes read 0.022, and the tuned finals' 0.01 ran the tamer and the
  40-harmonic series where the shipped 0.035 does not. -
  `auto_cathar_tonal` (default **true**), `auto_cathar_flatness_max`
  (default 0.04) and
  `auto_cathar_probe_similarity` (default 0.9). Where `auto` prefers
  `cathar`'s fidelity: sustained tonal programme with no silence for
  `auto_pure_linear`'s 4 s noise probe to learn from. The tape reads as
  tonal under the flatness ceiling, the quietest 4 s carries the programme
  (its speech-band spectrum correlates with the loud frames' above the
  similarity), and there is no sustained beat. Measured on 136 corpus clips
  and 81 excerpts of 21 local tapes, that is the one condition under which
  `cathar` deviates less on most clips (10 of 14 and 9 of 15; 0.21 dB
  against 0.54 and 0.15 against 0.19), always for 2-3 dB less noise
  removed; wins on both halves are a wash, and every other reading leaves
  `cathar` behind on both. A fidelity preference rather than a win: set
  `auto_cathar_tonal` to false to keep `auto_pure_linear` there. -
  `apl_use_deepfilternet` (default **false**). DeepFilterNet3 in place of
  UVR-DeNoise as the neural stage,
  where it is installed. It wins on synthetic fixtures and loses on real
  tape -- programme deviation 0.22 to 0.66 dB across 50 captures -- so it is
  opt-in. It is a from-source dependency (Rust core built under MSVC,
  installed with `--no-deps`); absence falls back to UVR-DeNoise. -
  `apl_mute_silence_db` (default -45.0), `apl_mute_min_ms` (default 15.0)
  and `apl_mute_max_ms` (default 200.0) define what counts as a dropout
  rather than a pause. Depth alone is not enough: synthesised speech carries
  genuine digital silence between phrases, so a span also has to be short
  and bounded by programme on both sides. - `apl_enable_tonal_cleanup`
  (default **false**). Rumble removal (`dewind`), now separate from dehum.
  Off because it did not measure as a gain on real tape. - Output suffix:
  `*_PureLinear_Cleaned`. - `cathar` / `cathar_vhs`: - Deterministic
  high-fidelity DSP restoration engine. - Applies 8-harmonic adaptive
  de-hum, surgical CRT whistle notch filter, spectral noise print
  subtraction, de-click/de-crackle, and azimuth phase alignment. -
  `cathar_vhs` is an alias for `cathar`. - Output suffix:
  `*_Cathar_Cleaned`. - The binary is the one beside the interpreter (the
  venv), then `~/.cargo/bin`; `AI_RESTORE_CATHAR_BIN` in the environment
  names another build, so an upgrade can be measured on the tuning excerpts
  before it replaces the validated binary; the installers provision 0.8.0,
  bit-identical to 0.7.6 (the build the listener round tuned on) on every
  stage the app calls. - Speech settings since the listener
  round of 2026-09-25 (the self-driving loop on the four full Tata tapes,
  judging on the listener readings; `docs/validation.md`): `cathar_alpha` 1.0
  (from 2.0), `cathar_beta` 0.02, `cathar_repair_strength` 2,
  `cathar_enable_coherent` and `cathar_enable_enhance` off,
  `cathar_noiseprint_duration_s` 4.5 (six stitched pauses, from 90 s of
  material) and `cathar_deesser_threshold` 12 (only the loud 's' is caught).
  `cathar_noiseprint_mute_guard` (false) keeps digital mutes out of the
  print's pick; see "cathar Noise-Print Mute Guard" below.
  `cathar_noiseprint_min_depth_db` (0, off) and
  `cathar_noiseprint_shallow_action` (`scale`) scale, drop or replace with
  the quantile print a print that sits too close to the programme; see
  "cathar Noise-Print Depth Guard".
  `cathar_noiseprint_method` (`window`) learns the print from the quietest
  window, or on `quantile` writes it from the whole material, per bin; see
  "cathar Quantile Noise Print".
  - Music profile: a tape whose
    tonal persistence (the scanner's median share of held spectral peaks over
    15 s windows; music 0.064-0.225, speech over a bed 0.007-0.033, dry
    dialogue under 0.003) reaches `cathar_music_persistence_min` (0.05) is
    denoised at `cathar_music_alpha` (0.5) with `cathar_music_enable_noiseprint`
    (false), `cathar_music_enable_coherent` (true) and
    `cathar_music_enable_deplosive` (false) in place of the speech settings,
    because a print learned from music is programme and subtracting it at the
    speech factor shaves 8-16 kHz by 14 dB while removing no noise. The music
    loop of the listener round (four rounds on 12 clips, hard-gate failures 31
    to 27) also switches the de-esser off (`cathar_music_enable_deesser`,
    false), sets the polish expander to 4 dB (`cathar_music_expander_depth_db`)
    and narrows the CRT notch to Q 60 (`cathar_music_crt_notch_q`). Round C3
    gives music its own subtraction floor (`cathar_music_beta`, 0.02), spike
    repair switch (`cathar_music_enable_repair`, true) and dewind cutoff
    (`cathar_music_dewind_cutoff`, 80 Hz) in place of `cathar_beta`,
    `cathar_enable_repair` and `cathar_dewind_cutoff`, which music ran until
    2026-10-09, so the defaults keep every output's bytes; none is judged by
    ear yet. A cathar tape is high-passed twice: the pre-conditioning graph's
    `highpass` at the scanner's rumble pick (0, 45, 60 or 75 Hz; 80 when the
    scan fails), then cathar's `dewind`, a high-pass at its cutoff (80 Hz, the
    higher of the two). The music cutoff replaces the dewind's only, so a tape
    the scanner cut at 60 or 75 Hz keeps that cut under a lower dewind.
    `cathar_music_profile` false keeps the speech settings everywhere. - `auto`
    (default): - Intelligent
    acoustic profile scan
    (speech, music, rhythm, tonality, noise floor, hum, clicks) that names the
    material, tunes the pre-conditioning and the models, and runs
    `auto_pure_linear`, the engine that leads `cathar` on every class
    measured on real tape; runs `cathar` on sustained tonal programme with no
    silence for the noise probe (`auto_cathar_tonal`) and when the neural
    denoiser is not installed (when cathar is installed; with neither engine
    available the `auto_ffmpeg_native` chain is the last resort). - Suffix:
    `*_Auto_Cleaned`. -
    `multipass_auto` / `multipass`: - Maximum-quality 4-pass cascaded
    restoration. - Dual-resolution acoustic scan -> analog pre-conditioning ->
    stem separation & Resemble-Enhance -> residual polish -> DTW Sync -> final
    master mix. - `multipass` is an alias for `multipass_auto`. - Suffix:
    `*_MultiPass_Cleaned`. - `auto_pure` / `pure`: - Pure speech & ambient
    restoration without generative vocoder synthesis. - Dual-resolution
    acoustic scan -> analog pre-conditioning -> AI stem separation ->
    dedicated speech/background UVR-DeNoise + de-esser -> DTW/shift sync ->
    32-bit float mix with EBU R128 loudness normalization. - `pure` is an
    alias for `auto_pure`. - Output suffix: `*_Pure_Cleaned`. - `hybrid`: -
    Separation + vocal enhancement + background denoise + sync + final mix. -
    Output suffix: `*_Hybrid_Cleaned`. - `denoise_only`: - Full-audio denoise
    \+ sync + final remux. - No separation or vocal enhancement. - Output
    suffix: `*_Denoised_Cleaned`. - `auto_ffmpeg_native` / `auto_vhs_native`:
    \- Intelligent adaptive FFmpeg DSP restoration with acoustic profile
    scanning. - Automatically analyzes tape hiss noise floor, mains hum /
    head-switching buzz frequencies, motor rumble power, and impulsive click
    density to auto-tune FFmpeg native DSP parameters. - `auto_vhs_native` is
    an alias for `auto_ffmpeg_native`. - Output suffix:
    `*_AutoFFmpeg_Cleaned`. - `ffmpeg_native` / `vhs_native`: - Ultra-fast
    native FFmpeg DSP restoration (`highpass` + `adeclick` + `afftdn` +
    optional `bandreject` notch). - Best for continuous tape hiss, mechanical
    rumble, and impulsive electrical clicks without GPU. - `vhs_native` is an
    alias for `ffmpeg_native`. - Parameters: `afftdn_nr` (dB reduction,
    default 10.0), `afftdn_nf` (dB noise floor, default -55.0), `afftdn_tn`
    (adaptive noise tracking), `highpass_freq` (rumble cutoff, default 80 Hz),
    `enable_adeclick` (click removal), `notch_freq` (mains hum, default 50.0
    Hz; set 60.0 Hz for NTSC). - Output suffix: `*_FFmpeg_Cleaned`. -
    `arnndn_speech`: - FFmpeg Recurrent Neural Network (RNNoise)
    speech/dialogue denoiser. - Best for dialogue-heavy VHS recordings. -
    Parameters: `arnndn_model` (default `"cb.rnnn"` in `models/arnndn/`),
    `arnndn_highpass_freq`, `arnndn_enable_adeclick`. - Output suffix:
    `*_Speech_Cleaned`. - `arnndn_model` must be a bare `.rnnn` filename inside
    `models/arnndn/` or `models/` (like the UVR model names): a path separator,
    a `..`, a leading dot, a drive prefix (`D:model.rnnn`), a non-`.rnnn`
    suffix, or any FFmpeg filtergraph metacharacter (`'` `"` `;` `,` `[` `]`)
    is rejected with a `[Config]` warning and falls back to `cb.rnnn`. The name
    reaches a single-quoted FFmpeg filter option, so this blocks a planted
    `config.yaml` from breaking out of the quote and injecting filters that
    open attacker-chosen local files. The name is looked up only in the
    application's own `models/arnndn/` and `models/` folders, never in a
    `models/` folder of the directory the restorer is launched from, and a
    model file there that is a link pointing outside its folder is skipped; a
    model that is not found stops the run with a missing-model error.

## Listener-Round Settings

Keys both engines share, set by the self-driving loops of the listener round
(2026-09-25: both engines on the four full Tata tapes and on 12 music clips,
judged by the harness's listener readings; the listening set that confirms or
overrides them is `D:\Tata\New folder\variants\v2`):

- `expander_depth_db` (7.0) and `expander_knee_offset_db` (8.0, from 4.0):
  the polish expander's depth under its knee and the knee's offset above the
  scanner's noise floor. Every loop moved the knee to +8 dB; the depth is
  7 dB for `cathar` on speech, `apl_expander_depth_db` (12.0) for
  `auto_pure_linear` and `cathar_music_expander_depth_db` (4.0) for `cathar`
  on music.

- `enable_pause_floor` (true, from false), `pause_floor_fill_db` (12.0) and
  `pause_floor_quiet_percentile` (15.0): in the frames the source calls
  quiet, an attenuated copy of the source is put back under the restored
  audio so a pause keeps its own texture, the fill 12 dB under the source's
  pause level. All four loops accepted it and none moved the fill.

- `loudnorm_target_lra` (20.0, from 11.0): the mux's loudness-range target;
  above it ffmpeg's loudnorm turns dynamic and rides the gain between words.
  All four loops accepted 20. The range is one half of ffmpeg's rule; the
  true peak is the other (see `loudnorm_linear_fallback` below).

- `crt_notch_q` (30.0): the CRT line-whistle notch's Q in the pre-conditioning
  graph; `cathar_music_crt_notch_q` (60.0) on music.

- `apl_enable_sibilant_guard` (true, from false) and `apl_sibilant_mix` (0.8,
  from 0.5): on the fricative frames of the pre-neural audio this share of
  the high band is put back, so the 's' keeps the body the neural stage
  empties. The Tata loop's answer to the listener's "thin s"; Vaccin still
  reads thin at it.

- `linear_air_gain_db` (1.0, from 2.0): `auto_pure_linear`'s presence shelf
  at `linear_air_freq_hz` (7500 Hz by default; `enable_linear_air`). Set by
  ear on 2026-10-08, every shelf at 7500 Hz: at +2 dB the user heard the 's'
  as thin; in an A/B of +2 dB, +1 dB and off on the four Tata tapes they
  chose +1 dB. The shelf is what brightens the 's' (the
  denoiser leaves fricatives within 1 dB of the other speech in every band;
  the +2 dB shelf lifts the whole output about 1.6 dB above 4 kHz). Each of
  the 16 rounds of the four APL loops scored a +1 dB candidate and none
  accepted it: it ranked behind the +2 dB incumbent in 15 and lost to
  another move in the 16th, under grids that rank
  `dsp.hf_4k8k.delta.median` up beside HF gates that only bound a loss.
  Ear v3 reads the tilt two-sided (`docs/validation.md`, "Ear v3").

- `linear_air_freq_hz` (7500.0), `expander_attack_s` (0.04) and
  `expander_decay_s` (0.18): added on 2026-10-09 for rounds A1 (brightness)
  and A3 (pause texture), at the values `modules/filters.py` hard-coded
  until then, so every output keeps its bytes (a unit test pins the default
  polish graphs). The first is the air shelf's corner (`treble=f`,
  1000-16000 Hz), `auto_pure_linear`'s alone: cathar's polish runs without
  the shelf. The other two are the polish expander's `compand` attack and
  decay in seconds (0.001-1.0 and 0.001-5.0), how fast its level follower
  rises into a word and lets go after it; `cathar`, `auto_pure_linear` and
  `denoise_only` share that expander. None is judged by ear yet. The tuning
  loop judges a corner with its gain against round 3's +2 dB verdict, heard
  at 7500 Hz: +1.5 dB at 6000 Hz lifts 4-8 kHz more than that shelf and is
  refused (`scripts/autotune_guards.py`).

A tuning result that carries a knob an accepted switch made inert (the
subtraction's factor, probe and native suppressor once
`apl_enable_spectral_denoise` is off) leaves that knob at its shipped value.

## Mastering and Pre-Conditioning Switches

Two keys added with ear v3 (2026-10-09), both defaulting to what every
earlier release did, so every output keeps its bytes until a listening
session decides otherwise:

- `loudnorm_linear_fallback` (`ffmpeg`, or `gain_limiter`): what the mux
  does when ffmpeg's loudnorm would leave its linear mode only because of
  the true-peak rule. Linear mode (one gain for the whole programme) holds
  only while the measured true peak plus the gain to -16 LUFS stays at or
  under -1 dBTP, the measured range stays within `loudnorm_target_lra`, and
  the measurement is set (I and LRA not 0, TP not 99, threshold not -70);
  otherwise loudnorm rides the gain and lifts the pause floors after the
  engine has finished. A programme under 3 s, shorter than loudnorm's frame
  buffer, runs linear whatever was measured. The run log used to check the
  range half only: of the 616 distinct decisions it logged as "linear" in
  the repository's run logs on 2026-10-09, 229 (37%) broke the true-peak
  half and ran dynamic. The log now applies the rule (`linear_mode_blockers`
  in `modules/mastering.py`, with the short-programme case taken from the
  duration the mux knows), which reproduced ffmpeg 8.0.1's own decision in
  all nine synthetic cases checked and its 3 s boundary (linear at 2.9 s,
  dynamic at 3.0 s). `ffmpeg` keeps the dynamic ride. `gain_limiter`
  applies one gain to the target and the mux's existing limiter instead; a
  range over its target still goes to loudnorm. That limiter is `alimiter`
  at -1 dBFS on the samples, not a true-peak limiter: on three synthetic
  programmes with clicks the render peaked at up to +0.24 dBTP where
  loudnorm's dynamic mode held -1.8 to -1.9. The pause-texture round judges
  it by ear.
- `azimuth_reject_inverted_pair` (false): the pre-conditioning's azimuth
  delay in every mode, cathar included. The gate that admits a channel pair
  read |r|, so a pair with one channel's polarity inverted passed it, and
  the lag it then reads is half a period of the programme's strongest
  partial, not a skew: on L/-L pairs a 1 kHz tone reads -0.50 ms, a 2 kHz
  tone -0.25 ms and noise low-passed at 3 kHz -0.23 ms, each of which the
  pre-conditioning would apply as a delay. A pair at r -0.3 or below is
  always logged as inverted; with the key on, no skew is read from it.

## Pre-Conditioning Azimuth Estimate

`azimuth_estimate_method` (`head`) was added on 2026-10-12. It picks where
the scan reads the stereo azimuth skew that the pre-conditioning of every
mode, `cathar` included, applies as an `adelay` on one channel. The default
keeps every output's bytes: `tests/unit/test_azimuth_estimate_spans.py` pins
HEAD 57bfdb8's readings, graphs and one whole scan.

- `head` (default): the first 32768 samples (0.74 s at 44.1 kHz)
  cross-correlated and the integer peak taken, as every release has. Its
  sign is the one the graph does not correct: a right channel one sample
  late reads -0.02 ms, and the graph delays the right channel by one more
  (ffmpeg, a 5-sample skew became 10).
- `spans`: GCC-PHAT on up to seven of the loudest 1.5 s spans, voting
  (`modules/azimuth_spans.py`). On each span the cross-spectrum's phase
  alone, over 150-7000 Hz and in the bins where both channels carry power,
  gives a peak refined to a fraction of a sample. A span counts at phase
  coherence 0.3 or more; the lag is the median of the counted spans that
  agree within half a sample, and at least two of them and more than half
  of those counted must agree. No delay is applied when the spans disagree
  (a tie included: a print stitched from two parts skewed opposite ways
  would otherwise have one part's skew doubled), cohere weakly, or agree on
  less than half a sample (the graph delays by whole samples). The result
  has the sign the graph corrects. The |r| gate and
  `azimuth_reject_inverted_pair` apply as before, read over the whole
  scanned audio rather than its head; with the switch off, an inverted pair
  gives its true skew (the magnitude of the peak) instead of half a period
  of its strongest partial.

Measured on the IA hunt on 2026-10-12: 232 stereo clips through the app's
own scan and pre-conditioning step on CPU, each method against the same
graph without the delay.

- `head` delayed a channel on 43 clips. The mono sum's 4-8 kHz fell on 40
  of them (median -1.43 dB, worst -3.14), and the programme's skew grew by
  a median 1.00 sample (never under 0.47). Of the three it raised, one is a
  polarity-inverted pair whose 23-sample delay decorrelates two cancelling
  channels (+17.7 dB; `spans` reads its true lag, 0); the others carry a
  3-4 sample skew.
- `spans` delayed a channel on 28 clips. The mono 4-8 kHz rose on 27
  (median +0.45 dB, up to +8.2 on a 4-sample skew) and fell on one
  (capture-a-3102, -0.36 dB: its spans agree on 0.56 samples, the whole
  clip reads 0.31). The residual skew is a median 0.15 sample, at most
  0.69. On the 17 clips where `head` delays and `spans` does not, the head
  delay cost a median 0.82 dB.
- The failing cathar clips where the head reads one sample and a
  full-band whole-file GCC-PHAT reads lag 0 (joan-armatrading,
  great-outdoors, entertainment-daily, 9-october, abc-primetime,
  channel-4-1999-70, jefferson) are not lag-0 sources. Above the programme
  band (from 8-16 kHz up) their channels share a lag-0 component that holds
  most of a full-band reading's bins (coherence up to 0.99 on the four
  checked), while in 150-7000 Hz they carry a real skew of 0.6-0.9 samples
  (jefferson 1.3). The head read it to the nearest sample and the graph
  doubled it. `spans` corrects it to a residual of 0.13-0.37 samples: the
  mono 4-8 kHz gains 0.16-0.45 dB against no delay, where the head delay
  loses 1.24-1.78 dB.
- The Tata tapes read no delay with either method: four have a dead right
  channel (r 0.02, the gate refuses the pair) and the fifth has identical
  channels (all seven spans agree on 0.00 samples). Both Gaudeamus
  captures (`D:\Tata`) also have a dead right channel (r 0.019) and read
  no delay either way.
- A review re-run on 2026-10-12 with its own estimators (a
  coherence-weighted fit of the Welch cross-spectrum's phase, not GCC-PHAT)
  reproduced these counts and gains to within 0.03 dB. Its residual skew
  after `spans` is a median 0.26 sample in 300-4000 Hz and 0.15 in
  4000-8000 Hz (at most 0.82).

`auto_pure_linear` gains most. `cathar` runs its own GCC-PHAT `azimuth`
stage after `dewind`, and on ten of these clips it left the same residual
skew (within 0.07 sample) and mono 4-8 kHz (within 0.02 dB) from either
pre-conditioning. `auto_pure_linear` runs that stage only inside its
physical repair, and only when the scan reads 0.05 ms or more, so a
one-sample head reading (0.02 ms) stays in its output. Not judged by ear
yet: a listening session and the `artifacts/realistic-v2` and IA
confirmation come before any default changes.

## CRT Line Search

`crt_detect_mode` (`head`) was added on 2026-10-12, from backlog gap 5 (a CRT
whistle the denoise seemed to expose). It picks where the scan looks for the
CRT line whistle. The pre-conditioning of every mode notches the line it finds
(`crt_notch_q`), and its standard holds the mains search to 50/100 or
60/120 Hz (`filters.MAINS_CANDIDATES_BY_LINE_RATE`). The default keeps every
output's bytes: `tests/unit/test_crt_detect_mode.py` pins HEAD e429683's
verdicts and whole-scan graphs on five fixtures, and 91 `cathar` renders of
IA hunt clips on CPU decode to the same PCM as the unchanged worktree.

- `head` (default): one FFT of the first 32768 samples (0.74 s at 44.1 kHz),
  a line called when the band's strongest bin (15450-15900 Hz) stands 1.7x
  over its median magnitude, as every release has. Hiss in the band reads
  2.9x on its own, the maximum of 335 Rayleigh bins over their median: white
  noise clears the threshold 200 times in 200. So a hissy tape gets a notch
  and a mains family by chance (on a fixture with 50 Hz hum under hiss it
  reads NTSC and the hum is not notched), and a line that starts late or
  wanders is missed or named the wrong standard.
- `welch`: up to 256 segments of 32768 samples spread over the whole file,
  their Hann-windowed power averaged (`modules/crt_line.py`). A line is
  called when the band's strongest bin stands 10 dB over the band's median
  and 6 dB over the median of the bins 80-160 away on each side (108-215 Hz),
  each at least 21 dB over the square root of the segments averaged (1.5
  times the noise's 99.9th percentile: 12.5 dB on one segment, 2.7 on 39),
  and the band holds fewer than five other local peaks 27 Hz or more away,
  within 3 dB of it and over that floor. A codec cut inside the band leaves
  one side as loud as the peak; a line wandering 30 Hz, or a whine spread
  55 Hz by flutter, keeps both sides clear; a comb is no line. The verdict is
  the nearer line rate, as `head` returns.

Measured on the IA hunt's 235 sources, against R0's line track
(`source_profile.line_profile`):

- `head` named R0's standard on 67 of R0's 71 lines. It missed
  et-el-extraterrestre (a 23 dB PAL line) and smokymountainsundayvhs (21 dB
  NTSC), named thevistagroup's NTSC line PAL and 2003-commercial-for-toshiba's
  PAL line NTSC, and fired on 59 sources with no line. The IA corpus clip
  vid-20220730-091707, whose PAL line wanders down from about 15700 Hz, reads
  NTSC: it is the clip whose whistle gate refused all six
  `cathar_music_alpha` 0.25 candidates.
- `welch` named it on 68, none wrong, and missed three weak lines (7.0-9.9 dB
  on the averaged spectrum). It fired on 15 sources R0 does not track, narrow
  tones such as a 15656 Hz line on five UK tapes, and rejected five combs: the
  BBC iPlayer captures' 15703 Hz tone among four others, and two renders' 43 Hz
  comb (44100 / 1024), each of which `head` or the band test alone reads as a
  line. White and pink noise of 0.8-60 s never make it fire.
- realistic-v2: `head` fires on 284 of the 440 captures without a planted line
  and on all 20 with one; `welch` on 51 and 19. 34 of those 51 carry a line R0
  tracks from their noise bed, 50-170 Hz off the rates
  `realistic_defects._without_tones` cuts.
- The hum notch moves on 14 sources: toward R0's mains on 8 (terrytoons' 50 Hz
  hum, 23 dB of evidence, read as NTSC by `head`, is notched again;
  teddy-bear's 60 Hz notch on a 50 Hz tape becomes 50 Hz) and elsewhere on 6,
  where the mains detector's own guess (R0 evidence 3.2 dB or less on four of
  them) is no longer held to the family the noise verdict picked.
- A lower threshold for a weak line at a rate (3.4 dB on 15 s) caught the
  hunt's weak PAL lines and also 50 Hz buzz harmonics near 15734 Hz on PAL
  tapes, which took the 50 Hz notch from tempo-pre-schools (55.7 dB of R0
  evidence); it is not part of the search.

The `cathar` renders on CPU (91 hunt clips; `welch` against the default,
whistle and hum read with the harness's floored readings over the runner's
windows): 64 differ. The two lines `head` got wrong are taken out
(et-el-extraterrestre's whistle reading goes from +1.06 to -23.62 dB against
the source, 2003-commercial-for-toshiba's from -0.90 to -12.14), terrytoons'
hum falls from -0.01 to -1.41 dB and thevistagroup's from +0.38 to -0.57.
Where `head` had notched hiss by chance the band is left as it was: on 22
clips the whistle reading rises by more than 1 dB, mostly from a carved hole
(-4 to -12 dB) to about 0. One of them passes the 3 dB gate:
9-october-2008-comedy-channel-promo-break-8 carries a 4.1 dB PAL line that
`welch` does not call and the denoise exposes (+4.63 dB); it stands out of the
hiss only after the denoise, so a search after the denoise is the place to
catch it. Not judged by ear yet: a listening session and the
`artifacts/realistic-v2` and IA confirmation come before any default changes.

## Pre-Conditioning Mains Vote

`scanner_mains_vote` (`head`) was added on 2026-10-12 (backlog B2). It
picks which mains notch the scan leaves. That notch sets the Q 15
bandrejects at the fundamental and its second harmonic in every mode's
pre-conditioning, `cathar`'s dehum family (under `cathar_dehum_freq`
`scanner`), and `auto_pure_linear`'s third-to-fifth harmonic and
post-denoise notches. The default keeps every output's bytes:
`tests/unit/test_scanner_mains_vote.py` pins HEAD e429683's votes, and the
five `cathar_ab_head.json` clips render 5/5 identical.

- `head` (default): the first 32768 samples (0.74 s at 44.1 kHz), each
  candidate's bin against the bins two either side, as every release has.
- `confirmed` (`modules/mains_vote.py`): the head's notch, kept only where
  the whole analysed signal's eight harmonics of its family stand 6 dB or
  more over their own neighbourhood (`spectral_denoise.harmonic_excess_db`,
  the reading `detect_mains_hz` votes on; 100 Hz reads as 50 and 120 as
  60). It never adds a notch, and a dropped one is logged with the excess
  it read. The two families share 300 and 600 Hz, so a wrong-family notch
  over strong hum can still be confirmed.

Measured on 2026-10-12 with `cathar` 0.8.0 on CPU (scratch evidence:
`D:/ai_restore_work/scratch_hum`, local). Clips without hum are the IA hunt
clips R0 reads under 6 dB of mains excess and the realistic-v2 fixtures
labelled `tape_noise` alone (speech and the three chord classes); the hum
tapes are the 75 distinct tapes of the hunt and `catalog_hum48` that R0
reads at 6 dB or more, and the hum fixtures realistic-v2's 100 speech and
music fixtures with 50 Hz hum.

| Reading | `head` | `confirmed` |
| --- | --- | --- |
| hunt clips without hum, notched | 123 / 203 | 1 / 203 |
| hum tapes, right / wrong family | 35 / 3 | 35 / 2 |
| fixtures without hum, notched | 29 / 200 | 12 / 200 |
| hum fixtures, right / wrong / none | 71 / 5 / 24 | 66 / 0 / 34 |

The two IA rows do not test the gate. R0's `mains_excess_db` is
`measure_hum.hum_excess_db`, the same computation as
`harmonic_excess_db` (same Welch frame, windows and sum; at R0's family
the two read a median 0.002 dB apart on the 284 IA sources), held to the
same 6 dB. So on the IA sets `confirmed` agrees with R0 by construction:
on 163 of the 166 head notches, the three others being where R0's family
is not the head's (the-lion-king twice, cj96video). The fixture rows are
the test, since their labels do not come from the reading. There the
gate clears 17 of the 29 false fires and keeps 12, which read 6.1-10.7 dB
(8 of them music fixtures). It also drops 5 of the 71 right-family
notches, where the injected 50 Hz hum reads 3.9-5.9 dB over the whole
15 s. The 6 dB line itself calls 132 of the 200 hum-free fixtures hum at
50 Hz, so an IA clip R0 reads under 6 dB may carry mild hum and one over
it may carry none.

- What the pair costs a programme with no hum: on the 200 hum-free
  targets the two notches alone move LSD 40-400 Hz by a median 1.73 dB at
  50 Hz and 1.58 dB at 60 Hz, and R11 by -0.22 and -0.36 dB (down to -2.2
  on speech). `auto_pure_linear`'s five notches move them 2.88 / 2.84 dB
  and -0.69 / -0.50 dB.
- In `cathar`, on the 122 hunt clips the confirmation clears, R11 sits
  nearer 0 on 99 of 121 (median -1.39 to -1.05 dB) and the output's hum excess
  rises a median 0.03 dB (at most 1.38 dB over the source, inside the
  harness's 3 dB `dsp.hum` gate). On the 17 realistic-v2 false fires it
  clears, LSD 40-400 Hz falls a median 0.23 dB against the target, on 14
  of 17, and R11 nears 0 on 14. The real key renders those fixtures, and the
  ones it keeps, bit for bit as the measured variant (105 of 105).
- The 35 right-family notches on hum tapes read 6.07 dB of excess at the
  least (hey-abbott) only because those tapes were chosen at 6 dB on the
  same reading; the fixtures, not this, show the gate's margin. The one
  hum-tape change drops the wrong-family 50 Hz notch of the-lion-king
  (R0: 60 Hz, 6.2 dB, no frame evidence).
- On the realistic-v2 hum fixtures it drops 10 of 76 notches, 5 of the
  right family. The fixtures cannot price a lost catch: dropping any notch
  there reads nearer the target (LSD 40-400 Hz on 51 of the 71
  right-family ones; with the notch and dehum the harmonic bins sit a
  median 11 dB under the target, without them 4 dB).
- A whole-signal vote that also adds notches (the same reading over 50 and
  60 Hz at 6 dB, as `detect_mains_hz` decides) was measured and not built.
  The chord beds read as hum: the musicled and musiconly targets, which
  carry none, read a median 13.4 and 15.2 dB of excess, so it notched 112
  of the 200 hum-free fixtures (all 100 beds), and where it changed the
  vote R11 came nearer 0 on only 33 of 109. hum_cancel's quiet-frame
  reading fired on 99 of them, since the beds never pause. R0's
  frame-median evidence (`source_profile`) separates the fixtures (hum p10
  53.7 against hum-free p90 34.4) and is the candidate if a vote that adds
  notches is ever wanted.

`auto_pure_linear` was not rendered (it needs the GPU). There a dropped
notch also leaves the canceller without the scan's report, so a series
with nothing at the fundamental is refused (`hum_cancel`'s
`fundamental_upstream`). Not judged by ear yet: the effect is small and
probably inaudible, and a listening session decides.

## DTW Sync on Length-Preserving Chains

`sync_dtw_enable` (true) was added on 2026-10-12 (backlog B1). The scanner
sends a tape it reads as drifting (`has_drift`) to DTW sync instead of
shift. The chains of `auto_pure_linear` (its music stem path too),
`cathar` and every single-track mode (`denoise_only`, `vhs_native`,
`auto_ffmpeg_native`, `arnndn_speech`, and `auto`, which runs one of them)
keep the length and the time base of the source they sync against, so a
warp can only add error. The DTW path interpolates linearly between whole
40 ms chroma frames (`dtw_resolution` 25), so every off-diagonal step plays
a frame at 0x (the output holds one sample's value) or at 2x. The ear's
time-base reading (R12, `docs/validation.md`) counts it on the IA hunt's
stored outputs: of the 30 clips sent to DTW, 24 cathar outputs carry 177
frozen holds and 15 `auto_pure_linear` outputs 49; none of the 205 per
engine sent to shift carries one.

- `true` keeps the scanner's choice, as every release has, so every output
  keeps its bytes.
- `false` sends those chains to shift sync whatever the scanner or
  `sync_method` says, and the run log says so ("DTW sync is off for
  length-preserving chains"). The stem modes (`hybrid`, `multipass_auto`,
  `auto_pure`) keep DTW. On five of the hunt's DTW clips rendered with
  cathar on the CPU the shift output reads 0 holds; the DTW warp, replayed
  offline on the same pre-sync audio, reads 71 (the hunt's GPU renders of
  those clips read 72).

Rendered on the GPU from the merged branch (2026-10-12, 26 of the hunt's 30
DTW clips, both engines, today's route against `false`, scored on the dsp
family): `auto_pure_linear` went from 42 holds on 14 clips (47 rate jumps,
30 plateaus) to none, cathar from 154 holds on 20 clips (65 jumps, 58
plateaus) to none, and R12's two gates from 14 and 20 failures to 0. The
sync held: the drift reading's median and worst case went 0.09 / 0.26 to
0.06 / 0.17 ms (`auto_pure_linear`) and 0.18 / 7.09 to 0.10 / 1.10 ms
(cathar), the offset stayed at the limiter's 5 ms, and 4 clips per variant
abstained alike (`D:/ai_restore_work/dtw_check`, local).

Not yet heard by a listener: the default changes only after a blind block.
The key is read in `modules/sync.py` only, after the stage cache's point,
so it is one of the keys the cache leaves out of its key (below).

The app's own CPU DTW path does not run today: its process pool cannot
take the `multiprocessing.Event` each chunk is handed ("Condition objects
should only be shared between processes through inheritance"), every chunk
fails, and the clip falls back to shift. A machine without a CUDA GPU of
4 GB or more therefore already syncs drift clips by shift; the hunt's DTW
outputs came from the GPU path.

## cathar Declip and Dehum

Two `cathar` keys added on 2026-10-10 for round C4, both defaulting to what
every earlier release ran (`tests/unit/test_cathar_round_c4.py` pins HEAD
549705a's command lines), so every output keeps its bytes until the C3
listening session decides otherwise:

- `cathar_declip_mode` (`always`, or `detected`): `always` runs cathar's
  `declip` on every tape. `detected` runs it only on a tape the scanner
  flagged as clipped (`profile.has_clipping_any_channel`: 20 or more
  samples at `|x| >= 0.985` counted over every channel of the minutes the
  scan reads, the first 60 s of a tape under 4 minutes or four spread 60 s
  blocks of a longer one). The count is per channel, not in the mono mix:
  on a tape with one dead channel, half of a 21-tape library, the mix
  halves the live side so it never reaches 0.985, while the
  pre-conditioning mirrors that side to full scale before cathar runs.
  Wherever the mix reaches the threshold one channel does too, so the count
  never misses a tape the mix flags. The mono-mix verdict
  (`profile.has_clipping`) still switches the pre-conditioning `adeclip`
  and `auto_pure_linear`'s own declip, unchanged. A scan that left no
  verdict (unreadable audio, so the scanner's fallback strategy, or a
  caller with no strategy) runs declip as on `always` and logs a WARNING;
  only a verdict of no clipping skips it. Clipping outside the scanned
  minutes can go unflagged. cathar 0.8.0's declip gave a
  synthetic file peaking at 0.5 back sample for sample and reshaped every
  sample above `cathar_declip_threshold` (0.95) on unclipped files peaking
  at 0.97 and 1.0, so the two modes differ on tapes whose peaks pass 0.95
  without being flagged; `detected` also saves the stage's runtime (5.29 s
  on an 8 s fixture). `cathar_enable_declip` false runs no declip in either
  mode.
- `cathar_dehum_freq` (`scanner`, or `0`): `scanner` gives `dehum --freq`
  the scanner's mains family (`notch_hz` 50 or 100 gives 50, 60 or 120
  gives 60; `notch_freq` when the strategy carries no reading) and runs no
  dehum on a tape the scan read as hum-free; the scanner votes on the first
  32768 samples, about 0.7 s. `0` runs `dehum --freq 0` on every tape, so
  cathar 0.8.0 detects the family from the recording itself. With
  `cathar_dehum_adaptive` (true) the tracker also picks 50 against 60 and
  passes a file without hum through unchanged: on synthetic files
  `--freq 0`, `50` and `60` gave identical output on 50 Hz hum, and a
  hum-free file came back sample for sample. The change therefore shows on
  tapes the scanner read as hum-free. `auto_pure_linear`'s own dehum
  (`apl_enable_dehum`, off) keeps the frequency it reads from the recording
  and never reads this key. Measured on 2026-10-12 with cathar 0.8.0 on CPU
  (backlog B2; `D:/ai_restore_work/scratch_hum`, local) on 82 clips (the 48
  of `catalog_hum48`, the 29 hum clips of the IA hunt not in it and the
  five identity clips), read with `measure_hum.py`'s `--mains auto` rule
  and R11: `0` rendered 73 byte for byte as `scanner`, the identity clips
  5/5 at both values. Of the 38 hum clips (source excess 6 dB or more) the
  scanner read as hum-free, 31 came out unchanged. Three readable tapes
  (wives-and-daughters, vid-20230624-142023, channel-4-continuity) lost
  2.88, 4.06 and 6.09 dB more excess and 2.8-6.7 dB more of their lines,
  at 0.05-0.66 dB more low-band deviation and R11 0.1-2.6 dB lower; three
  moved within 0.12 dB and one unreadable tape 3.06 dB. Of the 39 the
  scanner notched, 37 came out unchanged and two did worse
  (diamonds-are-forever 0.61 dB less removed and 1.44 dB less line drop,
  hey-abbott 0.17 dB). Over the readable scanner-zero hum tapes the paired
  change was 0.0 dB: 17 of 20 at 6-12 dB of source excess and 5 of 7 at
  12 dB and over came out unchanged, and the group medians moved only
  +0.05 and +0.40 dB because changed tapes crossed the median. That is not
  the consistent gain of 1.5 dB or more a flip needs, so `scanner` stays
  and the three tapes go to the listening session.

Four pause floor keys added on 2026-10-12, for both engines
(`modules/pause_floor.py`, after the polish expander in `auto_pure_linear`
and in `cathar`). The defaults are what every earlier release ran, so every
output keeps its bytes (`tests/unit/test_pause_floor_identity.py` checks
both callers against a frozen copy of HEAD bb5fa2a's stage), until a GPU
render and a listening session decide otherwise. The numbers below are a
CPU replay of the stage on the work directories of the hiss diagnosis of
2026-10-11 (the 13 diagnosis clips, the 235 hunt clips and cathar's two Tata
excerpts): the paused WAV against the source, 3-16 kHz band median on the
harness's true pauses, before the mux.

- `pause_floor_target` (`cluster`, or `frame`): `cluster` aims every pause
  at one level, the median of the quiet frames minus `pause_floor_fill_db`.
  The diagnosis found three ways it misses. On dense speech or music that
  median sits 9-15 dB over the true pauses. Where the neural stage passed
  the floor, the fill adds on top of it (+5.6 dB over the source on
  dvd-205405-0). The separator's peak normalisation drops the programme up
  to 10.6 dB across the neural stage, so the fill sits that much too loud
  against it (fourstarrevue's pauses +1.8 dB over the source). `frame` aims
  each quiet frame at `pause_floor_fill_db` under the source frame itself,
  scaled by the restored programme's gain on the source's loud frames (p70
  and up), and fills no frame whose restored level already meets that
  target; it fills as soon as one quiet frame sits 0.5 dB short, where
  `cluster` waits for 5 % of them. Replayed: the hunt's pause hiss went
  from -7.6 to -13.4 dB (SOTI -10.9 to -13.0, Tele7abc -3.5 to -12.6), and
  the hunt clips whose pause hiss sits over the source from 10 to 3 (pause
  low band over the source: 27 to 11).
- `pause_floor_shape` (`flat`, `shelf`, or `lowpass`),
  `pause_floor_shelf_hz` (3000.0) and `pause_floor_hf_cut_db` (18.0): `flat`
  fills with the source as it is, its tape hiss included. `shelf` fills with
  a copy cut `pause_floor_hf_cut_db` above `pause_floor_shelf_hz` (a
  4th-order Butterworth low-pass run forward and backward, plus the rest of
  the signal that many dB down); `lowpass` with a copy low-passed there (2nd
  order, forward and backward, a Linkwitz-Riley 4th order; the cut is
  unused). The corner and the cut are read only when the shape is not
  `flat`. With a shaped fill the deficit is read on the restored audio
  through the same shaping, so a hiss burst the neural stage left in a pause
  never stands in for the low-band body: on cathar's Tele7abc excerpt the
  shelf keeps the pause's 100-1000 Hz band at -14.5 dB against -17.8 dB when
  read broadband. Replayed with `frame`: the hunt's pause hiss -30.3 dB
  (`shelf`) and -40.3 dB (`lowpass`) with its pause low band at -7.6 dB, as
  with no fill; SOTI -30.8 and Tele7abc -30.5 dB with the low band at -11.3
  and -11.8 dB, `gap_air_db` -20.0 and -11.7 dB, and R4's high-band excess
  -18.0 and -17.7 dB. The -25 dB dead-air flag on `gap_air_db` trips on 16 of
  234 hunt clips with `shelf`, on 11 today and on 24 with no fill; the five
  new ones keep their pause low band at -5 to -13 dB. 3000 Hz and 18 dB are
  the prototype's guesses, for `scripts/audibility_check.py` and a blind A/B.

## auto_pure_linear Hiss After the Neural Stage

Three `auto_pure_linear` settings added on 2026-10-12 for the hiss the
Mel-RoFormer denoiser passes, all defaulting to what every earlier release
ran, so every output keeps its bytes until a GPU render and a listening
session decide otherwise (`tests/unit/test_post_neural_hiss_identity.py`
renders the step with and without the new opt-ins and compares the samples).
`cathar` cannot reach them, and `denoise_only` does not opt in.

The diagnosis of 2026-10-11 (the IA hunt's 235 clips and two Tata excerpts)
found the neural stage the only one in the mode that removes hiss: strong
where a tape's 8-16 kHz band is hiss-dominated, nearly an identity on music
and adverts, and weakest in 3-4.8 kHz, where in energy it took -4.4 and
-7.7 dB under the programme against cathar's -8.0 and -9.2 (a median over
bands hides that). The separator's peak normalisation also drops the
programme up to 10.7 dB across the stage on a hot or declipped tape.

- `apl_post_neural_suppress` (false), with
  `apl_post_neural_suppress_floor_gap_db` (3.0) and
  `apl_post_neural_suppress_min_residual_db` (6.0): the mode's own MMSE-LSA
  suppressor (`modules/spectral_suppress.py`) after the sibilant guard, at
  `apl_suppress_noise_bias`, `apl_suppress_gain_floor_db` and
  `apl_suppress_dd_alpha`, its noise learned from the quietest
  `apl_noiseprint_duration_s` of the post-neural audio. It runs on a tape
  only when one reading of that audio against the pre-conditioned source
  (`modules/hiss_floor.py`) opens both halves of a gate: the source's
  8-16 kHz band is hiss-dominated (median minus p10 of its frame energy,
  median over the bands that stand 10 dB over 16-bit dither, at or under the
  floor gap), and the restored floor under the programme stands within the
  residual threshold of the source's, on the 3-16 kHz band median or on the
  3-4.8 kHz bands' energy. The decision and its numbers go to the run log
  and, with `AI_RESTORE_EVENT_LOG`, to a `post_neural_suppress__...json`
  record.
- `apl_neural_level_restore` (false): the neural output scaled back by the
  least-squares projection gain of its input onto it on the input's loud
  frames in 100-4000 Hz (`modules/neural_level.py`), so the expander's knee
  and the pause floor's target read the level they were set for. A gain
  under 0.5 dB, or a pair whose programmes correlate under 0.5, is left
  alone. The intermediates are float, so a restored peak over full scale
  reaches the mux's limiter unclipped.
- `linear_air_gate` (`none`, or `hiss_dominated`): `hiss_dominated` skips
  the +1 dB air shelf on a tape the hiss gate opens on. With the shelf
  switched off (`enable_linear_air` false or `linear_air_gain_db` 0) there
  is nothing to skip, and the gate reads nothing.

CPU replay on the diagnosis's work folders (the stage WAVs it kept, its own
measurement code, no GPU and no new restoration), against the source:

- Gate: it opens on 47 of the 235 hunt clips, 10 of them on the band median
  alone (the prototype's count) and 37 more on the 3-4.8 kHz energy. It
  agrees with the diagnosis's own readings on 229 clips (233 on the band
  median alone); the rest sit within 0.3 dB of a threshold, or read the
  pre-conditioned reference where the diagnosis read the extracted source
  (a dead-channel tape the mirror cleans: -3.7 dB at 3-4.8 kHz against
  -9.9). Of the 13 diagnosis clips it opens on one (abc-primetime 1991,
  -1.6 dB at 3-4.8 kHz) and stays shut on both Tata excerpts and every music
  clip. Six hunt clips give no reading (programme correlation under 0.5: a
  31-sample shift across the neural stage, noise-dominated tapes, a slate)
  and stay shut.
- Suppressor: its output is bit-identical to the prototype's on all 13
  clips, and the chain (suppressor, the shipped polish, the prototype's
  shelved fill) reproduces the prototype's numbers on all 61 clips
  replayed. On the prototype's 10 (the gate opens on 9 of them;
  commercials-qtq reads a 3.3 dB gap on the pre-conditioned audio, 2.7 dB on
  the extracted source): the floor under the programme -15.7 dB (band
  median), `residual_noise_db` -5.3, loud-frame 8-16 kHz -3.5 dB, musical
  noise (lkr) 0.39.
- Against today's output at the same point (air shelf, shipped pause floor,
  before the mux), on the 48 clips the gate opens on (median, p10 to p90):
  the floor under the programme -8.9 dB on the band median (-13.8 to +1.4)
  and -8.3 dB at 3-4.8 kHz (-11.5 to -5.7); loud-frame 8-16 kHz -0.6 dB
  (-4.8 to +1.4); programme level -1.1 dB (-1.8 to -0.5); lkr +0.18 (0.0 to
  +0.3). The pauses get worse, +2.0 dB (-0.1 to +54.6): the pause floor's
  absolute target now sees a deficit and refills it with source hiss. The
  tails matter: the pause hiss rose on 19 of the 27 clips with true pauses,
  by 29-81 dB on 7, which end between -16 and +0.3 dB against the source
  (capture-a-1128202105 -85.7 to -4.3 dB), and the floor under the programme
  rose on 6, by up to 19 dB (f-1-1980-08-british-gp). With the prototype's
  per-frame shelved fill instead, the pauses go -18.0 dB and lkr +0.28, so
  this stage is never switched on with today's pause floor: it is judged
  together with the per-frame fill.
- Air gate: -0.35 dB under the programme on the band median (-0.5 to -0.3),
  -0.04 dB at 3-4.8 kHz, -0.67 dB in the loud frames' 8-16 kHz; the pauses do
  not move.
- Level restore: on the 26 clips whose neural input peaked over 0.9 the gain
  matched the separator's peak ratio within 0.1 dB (fourstarrevue +10.68 dB,
  fun-song +10.05, f1-1953 +8.14); on the other 218 it read +0.004 dB at the
  median (p95 +0.15), and 7 passed 0.5 dB, the model's own level change.
  Through the real sibilant guard, polish and pause floor, six hot clips end
  within 0.4 dB of the source's programme level; fun-song's
  `residual_noise_db` goes +6.37 to -0.32 dB, everton's pause 100-1000 Hz
  +6.39 to -0.30, pugwash's pause hiss -5.9 to -11.3. On fourstarrevue the
  pause floor then finds nothing to fill on its quiet cluster and the true
  pauses stay at the neural stage's -55 dB (their 100-1000 Hz body at
  -0.7 dB, `gap_air_db` -9.7 dB, clear of the -25 dB dead-air flag). With
  the expander's knee where it was set, musical noise rose on two hot clips
  whose floor the neural stage had emptied: chuckle-toons lkr 0.84 to 2.10,
  hogan-show 0.99 to 1.19; a listening question.

All three act after the stage cache's cached point
(`POST_NEURAL_CONFIG_KEYS`), so a candidate that changes only them replays
the neural stage. The suppressor's own settings stay keyed, because the
subtraction stage before the cached point reads them too.

## Reference-SNR Suppressor (auto_pure_linear and cathar)

`apl_reference_suppress` (false; 2026-10-17) reads the mode's MMSE-LSA gain
on the pre-conditioned source and applies it to the audio after the sibilant
guard (`spectral_suppress.ReferenceSuppressor`, run by
`modules/post_neural_suppress.py`). Model the neural stage as a mask, the
restored magnitude over the source's: the hiss it leaves is that mask
squared times the source's hiss, so the restored audio's a posteriori ratio
against it is the source's own. The two masks multiply. A cell where the
source held only hiss loses up to `apl_suppress_gain_floor_db` (-20); a
cell where the programme stood over the hiss keeps its level. It can only
remove, so it runs ungated. With `apl_post_neural_suppress` also on, the
gated suppressor runs first and this one on its output.

Why the gated suppressor could not do this on the Tata tapes:

- Its gate stays shut there. The Mel-RoFormer already took the floor under
  the programme 16-33 dB under the source's at 3-4.8 kHz and 24-68 dB on the
  3-16 kHz band median; Vaccin's 8-16 kHz band also reads 3.7 dB of
  stationarity, over the 3.0 the gate wants.
- Its noise is the post-neural audio's quietest stretch, the RoFormer's
  near-empty pauses, and its level tracker stays within 6 dB of that, while
  the hiss it passes under speech follows the speech about 20 dB higher.
- The quietest stretch is the wrong estimate on any speech tape. The
  probe both suppressors learn from (`noise_profile`, the quietest 4 s,
  cathar's rule) read 13-25 dB over the floor of the true pauses below
  8 kHz on all four Tata tapes (0-3 dB at 12-16 kHz): the quietest 4 s of
  an interview holds speech. Read against it, the reference gain took the
  loud programme over 3 kHz down 4 dB on Tele7abc (`hiss_prog_kept_hf_db`
  0.18 to -4.01, `balance_top_db` -1.77 and `sib_texture_db` 2.43, both
  past their v4 guards).

So the reference suppressor learns its own noise, the source's whole-file
quantile print (`reference_noise`): per bin, the 5th percentile of the
16-frame (0.37 s) moving mean of the channel-mean power, the cathar quantile
print's estimator read in power (`noiseprint_quantile.MovingQuantile`, now
parametrised; cathar's print is unchanged), +2.0 dB, held at its level
(`HeldLevel`). It read the true pauses' floor within -2.7 to +2.2 dB per band
before its bias; a 16-frame mean's 5th percentile sits 2.3 dB under the mean
on white noise. With no 0.37 s of hiss alone in 5 % of the windows the print
learns the programme, and the gain took a gapless fixture's partials down
15 dB, so a print whose loud frames (P90, 300-3000 Hz) stand under
`MIN_DEPTH_DB` (18) over it runs no suppressor. Measured depths: the Tata
tapes 24.7-31.9 dB, the twelve music clips 1.7-58.6 (14.6, 19.9 and 21.0
nearest the line; one gapless clip 1.7).

CPU replay of the v9 incumbent's post-neural stages on the four Tata tapes
(the kept work folders of one render on hardlinked copies, the stages run
in-process from the neural output, before the mux; scored with the dsp
family against the tape). "Frame" is `pause_floor_target: frame` with
`pause_floor_shape: shelf` at 3000 Hz, 18 dB; "+ ref" adds this suppressor
with the shelf at 2000 Hz, 30 dB. Hiss readings are dB over the threshold
at 75 dB SPL; `kept_hf` and `lkr` are the guards they trade against:

| tape | reading | shipped | frame | + ref |
|---|---|---|---|---|
| Tele7abc | under speech | 6.35 | 1.02 | 0.00 |
| | pause tail | 29.50 | 11.16 | 0.00 |
| | pump | -8.75 | 10.89 | 9.58 |
| | kept HF / lkr | 0.18 / 0.15 | 0.18 / 0.20 | -0.39 / 0.44 |
| SOTI | under speech | 0.00 | 0.00 | 0.00 |
| | pause tail | 23.24 | 8.10 | 0.00 |
| | pump | -7.84 | 8.64 | 8.79 |
| | kept HF / lkr | -1.73 / 0.66 | -1.75 / 0.95 | -2.17 / 1.21 |
| Vaccin | under speech | 9.43 | 6.42 | 0.00 |
| | pause tail | 21.15 | 9.47 | 0.00 |
| | kept HF / lkr | -0.21 / 0.36 | -0.22 / 0.40 | -0.79 / 0.65 |
| Gaudeamus5 | under speech | 0.00 | 0.00 | 0.00 |
| | pause tail | 31.34 | 21.67 | 6.43 |
| | pump | -11.22 | -2.24 | -3.60 |
| | kept HF / lkr | -3.28 / 0.43 | -3.31 / 1.12 | -4.05 / 1.41 |

Every v4 guard passes on every tape (pump bound 12, kept HF -6, balance top
-1.0, `sib_texture_db` 2.0, the 35 dB pause depth, the islands, `lkr` 1.747);
`balance_top_db` moves 0.04-0.19 dB. Vaccin's windows hold no 0.5 s of
pause over the dither, so its pump reads none. Gaudeamus5's remaining pause
reading includes the soft singing the pause reading takes for hiss (see
`docs/ear_v4_hiss.md`).

What did not help, on Tele7abc: smoothing the reference's power over a third
or a whole octave (and over time) before the gain, the textbook cure for a
noise-like fricative's fading bins, left the probe-noise dulling at
-3.9..-6.7 dB, and with the quantile print it lets more hiss through under
speech (pump 16.6, past the bound); a faster decision-directed smoothing
(0.8) does the same (pump 14.7). The level tracker on the quantile print
lowered the pump to 7.2 at a little more dulling (kept HF -0.87); the held
level was kept as the more principled for stationary tape hiss.

The suppressor needs the pause fill's shelf at 2 kHz and 30 dB to empty the
pauses: with today's 3 kHz and 18 dB shelf the pause tail stays at 10.1
(Tele7abc), 7.3 (SOTI), 8.6 (Vaccin) and 13.9 (Gaudeamus5), while the hiss
under speech still reads 0. Without the suppressor the 2 kHz fill trips the
pump guard (Tele7abc 40.9, SOTI 35.7): with the hiss under speech at
-21 dB of the source's, emptying the pauses gates the hiss. The suppressor is
a loop knob (`scripts/autotune_restoration.py`), so the loop can reach the
pair one move at a time (the suppressor, then the fill's corner and cut), if
each step wins on its own.

The key is read after the stage cache's cached point
(`POST_NEURAL_CONFIG_KEYS`), so a candidate that changes only it replays the
neural stage. Off, every output keeps its bytes; the loop and a listening
session judge it.

`cathar_reference_suppress` (false) runs the same stage
(`modules/reference_suppress.py`) on cathar's output before the polish, at
the estimator's shipped settings (noise bias 1.0, gain floor -20 dB,
decision-directed 0.96); cathar's subtraction is the mask the source's SNR
multiplies. The same replay on cathar (one render per tape with its work
folder kept, then the polish and the pause floor in-process; the shipped
replay reproduced cathar's own pre-mux output exactly):

| tape | reading | shipped | frame | + ref |
|---|---|---|---|---|
| Tele7abc | under speech | 14.53 | 13.25 | 0.00 |
| | pause tail | 26.83 | 18.78 | 3.31 |
| | pump / kept HF | -5.36 / -0.60 | 2.00 / -0.60 | 3.39 / -1.36 |
| SOTI | under speech | 8.97 | 7.98 | 0.00 |
| | pause tail | 19.94 | 12.67 | 0.00 |
| | pump / kept HF | -3.37 / -2.01 | 0.38 / -2.04 | 2.00 / -3.56 |
| Vaccin | under speech | 7.75 | 4.31 | 0.00 |
| | pause tail | 19.46 | 13.44 | 0.00 |
| | kept HF | -0.88 | -0.89 | -1.32 |
| Gaudeamus5 | under speech | 11.55 | 11.00 | 0.00 |
| | pause tail | 30.98 | 24.95 | 11.28 |
| | pump / kept HF | -4.60 / -1.74 | 0.32 / -1.84 | 0.78 / -4.20 |

Every v4 guard passes: `balance_top_db` moves 0.01-0.22 dB, `sib_texture_db`
reads 1.32 at most, `lkr` 1.47 at most (SOTI), the pause islands 0.99. Here
"frame" is the shelf at 3000 Hz, 18 dB, and "+ ref" the shelf at 2000 Hz,
30 dB. cathar never reads the stage cache; off, cathar keeps its bytes.

## auto_pure_linear Burst Ceiling

Two `auto_pure_linear` keys added on 2026-10-12 for the hiss the
Mel-RoFormer leaves next to speech (`modules/burst_ceiling.py`, after the
polish and before the pause floor). `apl_burst_ceiling` defaults to `none`:
nothing is read and every output keeps its bytes
(`tests/unit/test_burst_ceiling_identity.py` renders the step with and
without the opt-in and compares the samples). `cathar` cannot reach the
stage and `denoise_only` does not opt in.

What the bursts are. With the pause floor off, 2-11 % of the true-pause
frames carry 80 % of the pauses' 3-16 kHz energy on the hiss diagnosis's
stage WAVs, so the pauses' energy mean stands far over their median. Read
cell by cell (20 ms, one ERB band) against the source's own band floor,
most of that energy is programme, not hiss. On Tele7abc the four frames
that carry it are a word's tail 50 ms after the activity detector's last
speech frame: the source stands 10.9 dB over its pause level there and
10-13 dB over its own floor in the 3.2-3.6 kHz bands the output keeps
within 1 dB. On abc-primetime 1991 it is the file's first frame, where the
clip starts on programme. The cells where the source holds nothing but hiss
carry 2 % (Tele7abc) and 14 % (SOTI) of the pauses' output energy, at -35
and -49 dB under the source. The hiss the RoFormer passes sits beside the
programme: in that tail's 4.5-7 kHz bands (9-15 dB under the source's
floor), in the short gaps between words (-24 / -29 dB in the hiss-only
cells), at the word edges (-21 / -25 dB) and under the programme (-16 /
-18 dB). The polish's expander does not make them: across it the burst
frames move 0.0-0.4 dB on the Tata excerpts and abc-primetime (c4-2009's
lose 2.1 dB).

- `apl_burst_ceiling` (`none`, `gaps`, or `edges`) and
  `apl_burst_ceiling_depth_db` (30.0, 0-120): in the source's non-speech
  blocks (`gaps`; the four-band activity detector of `modules/hiss_floor.py`)
  or in those and the speech blocks within 150 ms of a gap (`edges`), every
  restored cell (one ERB band of 3-16 kHz over a 23 ms block) is held to a
  ceiling read on the pre-conditioned source, at the restored programme's
  level: `apl_burst_ceiling_depth_db` under the source's band floor (the
  p10 of the band over +-1 s, the blocks mirrored at the file's ends) where
  the source holds only hiss; the source's closed envelope plus 3 dB where
  it surely carries programme; between the two, in power, by how surely
  (the envelope averaged over three neighbouring bands from 6 dB over the
  floor, a cell on its own from 10 dB, each surely so 3 dB further on). A
  sure programme cell's tail, the cells joined to it along time in its band
  whose three-band average stands over 3 dB (the hiss's own p95 in the Tata
  gaps), counts from 3 dB. A cell under its ceiling passes untouched; a cell
  of hiss alone takes the deepest cut of its block and the blocks either
  side, so a burst's onset and tail go with it; no STFT frame is cut deeper
  than the block it was read in. A band acts only where the neural stage
  emptied the gaps (its restored energy at least the depth under the
  source's floor, the median over the non-speech blocks) and where the
  source's floor stands 10 dB over 16-bit dither. The output is the
  restored audio plus the overlap-added STFT of `X * (gain - 1)`, so every
  sample no capped frame reaches keeps its bytes; it is named
  `capped_<scope><depth>_<input>`. The depth is the shelved pause fill's
  own level (12 + 18 dB). The run log and, with `AI_RESTORE_EVENT_LOG`, a
  `burst_ceiling__...json` record carry the decision, the active bands, the
  gaps' restored floor and the blocks capped.

CPU replay on the diagnosis's stage WAVs (`pause_off`, the 13 clips, the
polished WAV, then the `frame` + `shelf` pause floor; no GPU, no new
restoration), read with the diagnosis's own `measure.py` against the
extracted source. "Hiss-only cells" are the cells where the source stands
under 3 dB over its p10 floor (output minus source, energy); "tail" the
word-edge cells where it stands 3-6 dB over (output against the same chain
without the ceiling, the fill alone against the polished WAV: what the
stage took from the word's edges).

```text
clip           variant       prog sum  3-4.8k   edge   gaps   near  pause mean  gap air   lkr   tail
Tele7abc       fill alone       -25.2   -17.4  -20.5  -16.5  -27.2       -18.7    -11.7  0.18   0.07
Tele7abc       gaps + fill      -25.5   -17.6  -20.8  -17.2  -28.8       -19.2    -11.7  0.18  -0.08
Tele7abc       edges + fill     -26.1   -18.3  -22.9  -17.2  -28.9       -19.2    -11.8  0.18  -0.91
SOTI           fill alone       -37.4   -31.8  -24.6  -22.5  -28.6       -27.5    -20.0  2.36   0.25
SOTI           gaps + fill      -37.5   -31.9  -24.7  -28.5  -29.2       -27.6    -20.0  2.36  -0.01
SOTI           edges + fill     -39.1   -33.5  -25.9  -28.5  -29.2       -27.6    -20.8  2.42  -1.27
abc-primetime  fill alone        -3.6    -2.1   -5.5   -3.7   -5.9       -20.1     -5.0  0.00   0.00
abc-primetime  edges + fill      -4.2    -2.8  -15.5  -26.8  -27.5       -20.2     -5.0  0.00  -2.60
barry          fill alone       -17.1   -12.2  -14.6  -17.4  -23.3           -     -3.9  0.21   0.03
barry          edges + fill     -17.4   -12.3  -18.5  -17.7  -24.3           -     -3.9  0.21  -1.43
```

Columns: the floor under the programme (3-16 kHz energy sum) and its
3-4.8 kHz bands; the hiss-only cells at the word edges (speech frames within
150 ms of a gap); the short gaps' energy mean (non-speech frames outside the
true pauses); the hiss-only cells of the non-speech frames within 300 ms of
speech; the true pauses' energy mean; `gap_air_db` (output); lkr; the tail
cells. All in dB against the source, but lkr and the tail.

- The stage acts on six of the 13 clips. It skips the seven whose gaps
  the RoFormer left at the source's level (c4-2009, dvd-205405-0, cj96,
  itv1230819: "every band's gaps within 30 dB") or that hold under 0.2 s of
  gap (c4-1999, fun-song, agatha), and those keep every sample;
  fourstarrevue and bbc-news act in 8 and 6 bands and move under 0.1 dB.
- Guards: the loud frames' 4-8 and 8-16 kHz within 0.04 dB on every clip,
  sibilance (R2 level and texture, the body ratio) unchanged, gap air at
  -20.8 dB at the lowest (SOTI, -20.0 with the fill alone; the dead-air
  flag is -25), lkr up 0.06 at most, the pause low band unchanged. The
  word-edge cells 6-10 dB over the floor lose 1.0 dB at most
  (abc-primetime; 0.7 dB on SOTI).
- The true pauses' median and energy mean barely move: what carries the
  energy mean is programme the ceiling keeps (the tail on Tele7abc, the
  first frame on abc-primetime), and on SOTI the fill itself (-30 dB). The
  hiss-only cells in the true pauses read -29.0 / -28.8 dB with the ceiling
  and the fill, the fill's own level.
- Alone (the pause floor off) `edges` takes Tele7abc's word-edge hiss from
  -20.8 to -23.4 dB and SOTI's from -25.5 to -27.0 dB, SOTI's short gaps
  from -23.7 to -46.6 dB.
- Pumping is the open risk. Where the RoFormer leaves the hiss under the
  programme near the source's level, the ceiling makes it follow the words:
  on abc-primetime the floor under the programme stays near the source's
  (-3.6 dB with the fill alone, -4.2 with the ceiling) while the short gaps
  go from -3.7 to -26.8 dB, and the frame-to-frame change of the
  3-16 kHz level in the gaps and word edges rises from 2.6 to 4.3 dB (the
  Tata excerpts 3.4 to 3.8 and 6.1 to 6.2 dB). A listening session needs
  such a clip; a guard would hold the hiss ceiling no deeper than the
  restored floor under the programme.
- Cost: about 1.5 s per minute of tape on one CPU core (the reading and
  the writing, on the Tata excerpts).

As first committed (2026-10-12) the ceiling read a word's decaying tail as
hiss once the three-band average fell under 6 dB, and cut the cells the
RoFormer had kept at the source's level to the depth: on abc-primetime at
3.88 s five 3-5.3 kHz bands stood 5.4-7.1 dB over the floor and three were
cut 34-37 dB. Such tail cells carried 22-48 % of the energy it removed on
five of the six clips it acted on, the word-edge cells 3-6 dB over the floor
lost 3.7-6.1 dB with `edges` (Tele7abc -3.65, SOTI -4.16, abc-primetime
-5.81, barry -6.05) and 6-10 dB over 1.1-3.3 dB, and the short-gap and
word-edge numbers it reported were mostly those tails (Tele7abc `edges` +
fill: gaps -25.2, edge -27.7 dB, gap air -12.9 dB). The cuts were also
interpolated in dB from block to block, so the last frame of a word's block
lost 0.375 of the next block's cut, and the floor's percentile repeated the
last block (partly past the end) over the last 0.8 s, holding the floor up
to 6 dB low there on the unit test's tape. The tail rule, the frame rule
and the mirrored floor
replace them; with the tail threshold at 4 or 5 dB instead of 3 the
word-edge hiss on Tele7abc reads -24.1 / -26.0 dB and the tail cells lose
1.3 / 1.7 dB.

The candidates prototyped on the same WAVs (Tele7abc, each followed by the
shelf fill): a VAD-gated gain on everything over 3 kHz in the non-speech
frames within 300 ms of speech took the pause energy mean to -25.0 dB, and
a cap on the pause floor's quiet frames at the fill's own high band to
-25.1 dB, both by cutting the word's tail (the non-speech programme cells
-7.9 and -8.0 dB; the cap also took gap air to -18.1 dB, lkr to 0.53 and
the quiet speech frames' highs). A per-band expander with a 4-10 dB source
SNR knee cut the burst cells but also the tails beside them (non-speech
cells 6 dB over the floor, dilated over three frames, -17.4 dB on SOTI and
-7.7 dB on barry; the ceiling as first committed read -23.7 and -3.7 dB on
the same cells, the ceiling with the tail rule -22.1 and -0.05 dB; SOTI's
are non-speech cells beside 6-10 dB peaks that no sure programme cell
reaches, in pauses the RoFormer emptied, and read -1.8 dB with the fill).
A ceiling that read programme 4 dB over the floor let
enough hiss cells through as programme to leave the unit test's synthetic
bursts 17.5 dB down instead of 31 (one of five within 11 dB); one that set
the ceiling between the two levels in dB cut a clip's 10-15 dB tails by
16-24 dB. Over every frame (scope "all") the prototype ceiling lowered the
floor under the programme further (Tele7abc sum -34.8 dB) but raised the
's' body ratio 0.8 dB and its texture 0.16 dB and took 0.3 dB more off the
loud 8-16 kHz, so it is not offered.

Not rendered on the GPU or heard; `scripts/audibility_check.py` and a blind
A/B decide `edges` against the fill alone. Both keys act after the stage
cache's cached point (`POST_NEURAL_CONFIG_KEYS`), so a candidate that
changes only them replays the neural stage.

## Mux Timing and Rate

The mux now places the restored track where the source's audio started
against the video, in every mode and container, with no key: it reads the
start time of the audio stream the extraction decodes (`ffprobe`) and gives
each restored input `-itsoffset <start>` with `-copyts`
(`modules/mux_format.py`). MPEG-PS and MPEG-TS sources take no `-copyts`,
because their demuxer corrects a timestamp jump (two recordings joined
byte-wise, a recorder's pause) only without it: on two 6 s halves joined
byte-wise, `-copyts` piled the second half's video timestamps at the jump
and moved its clicks -254 and -620 ms against their flash frames in
MPEG-PS, +84 and +83 ms in MPEG-TS. Without it ffmpeg starts such an input
at the earliest stream the command maps (the copied video, and the first
audio stream when `preserve_original_audio_track` keeps it), so the
restored input goes to the audio's start against that stream's; all four
clicks then held within 0.01 ms, and a real capture cut whose audio led its
video by 554.5 ms sat at the limiter's delay. It used to start the track
with the video. On
3 s synthetic sources with the audio 300 ms after the video, a click 1 s
into the audio landed 273.7 ms early in MP4 and MOV, 295.0 ms in MKV and
285.0 ms in MPEG-PS; with the audio 300 ms early, MPEG-PS put it 315.0 ms
late, and with no offset 15.0 ms late (MP2's 481-sample decoder delay
re-based to zero). After the change the click sits at the limiter's delay
alone in all six containers tried (MP4, MPEG-PS, MOV, MKV, AVI, TS). An AVI
carries no start offset; ffmpeg writes none, so its audio starts with its
video either way. The cathar identity clips start at zero: muxed through
HEAD and through this code, their files come out byte for byte the same.

An MPEG-PS output (`.mpg`, `.mpeg`) now keeps its source's sample rate when
MP2 can encode it (32, 44.1 and 48 kHz and the half rates; any other rate
goes to 48 kHz, an unreadable source stays at 44.1 kHz). It used to be
44.1 kHz, which DVD-Video does not allow for MP2. Every other container
keeps 44.1 kHz unless `mux_sample_rate` says otherwise. The capture
`E:/Test/2015_12_17_10_56_3.mpg` (MP2 at 48 kHz, audio 5.578 ms after the
video) came out of the 16 Sep renders as 44.1 kHz MP2 with its audio
10.911 ms before the video; a 12 s cut of it now keeps 48 kHz and its
restored audio sits 4.989 ms after the source's (the limiter), 0.010 ms with
the compensation on.

Two keys, both defaulting to today's bytes outside MPEG-PS:

- `mux_limiter_latency_compensation` (false): the mux's limiter (`alimiter`
  at -1 dBFS) delays what it passes by its lookahead. A synthetic impulse at
  sample 10000 came out at 10219, 219 samples or 4.97 ms at 44.1 kHz (239
  samples at 48 kHz), with or without loudnorm's dynamic mode before it, so
  every output of every mode plays about 5 ms late against its video and
  against the preserved original track. `true` sets `latency=1`, which kept
  the impulse at 10000 and the length unchanged. It is a pure time shift,
  but it changes the bytes the cathar identity check compares, so it waits
  for the user's sign-off on re-basing that check.
- `mux_sample_rate` (`pipeline`, or `source`): `pipeline` keeps 44.1 kHz in
  every container but MPEG-PS. `source` keeps the source's rate in every
  container whose codec encodes it (AAC's and MP2's rates; PCM takes any),
  which changes every 48 kHz source's bytes, three of the five identity
  clips among them.

Both are read only by the mux, so the stage cache leaves them out of its
key, and the tuning driver knows them.

## Exit Status and Prompts

`restore_audio_hybrid.py` exits 0 when every file was restored (or already
was), 1 when any file failed or was skipped, and 2 when nothing ran
(dependencies missing, or no files). A source with no audio stream, or one
whose audio decodes to no samples (an edit list that hides every packet),
is a `[Skip]` line with the reason and counts as failed; it used to retry
six times or fail as "Output audio is invalid". An ffmpeg failure that no
retry changes (no stream in the output, a `-map` matching nothing, an
unreadable input, an unknown filter or option; `modules/ffmpeg_failures.py`)
is raised at once instead of retried on fewer threads. A memory message
keeps the retry, and a damaged capture's decode warnings ("Invalid data
found when processing input", printed for every broken MP2 frame on a run
that succeeds) never count: only ffmpeg's "Error opening input" names an
unreadable input. "Press Enter to
exit..." and the drag-and-drop prompt are asked only when stdin is a
console (on Windows the handle must pass `GetConsoleMode`: a detached
stdin, the NUL device, reports a tty too); with no paths and no console the
app asks nothing and exits 2.

## cathar Noise-Print Mute Guard

`cathar_noiseprint_mute_guard` (false), added on 2026-10-12. cathar learns
its noise print from the quietest 0.75 s of a clip (a grid of 40 windows)
or, from 90 s of material, from six stitched pauses. The single-window
pick's only guard was against an all-zero window, and the stitched pick had
none (it always keeps the quietest window, exact zeros included), so on an
edited upload the pick can be a digital mute (a fade to black, a blank
lead-in, a DVD's silence), and a print learned there removes nothing. On
the 136 IA clips of
`experiments/auto_engines_v140/ia_features.json` (cathar 0.7.3), 7 picked a
window under -70 dBFS (-80.0 to -128.4), and there APL removed a median
9.7 dB more than cathar, against 3.38 dB on the other 114.

With the key on (`modules/noiseprint_guard.py`), a candidate window is left
out when half or more of its samples repeat the one before (exact zeros, a
held DC level), or when it sits under -70 dBFS and at least 30 dB under the
p20 of the material's 23 ms frame levels. Frames under -90 dBFS do not count
toward that p20, so a mute filling a fifth of a clip cannot become its own
reference. Both picks keep their grid and their spread, so with nothing
rejected the pick is the same; with every window rejected the denoise runs
without `--noiseprint`. The log names every rejected window (one line per
run of windows on the stitched path), the level of the window learned from
and the p20, and the guarded print's file name ends in `_muteguard`. Off,
the picks, the slice and every command line are HEAD bb5fa2a's
(`tests/unit/test_cathar_mute_guard.py`). The key is read in cathar mode
only; auto_pure_linear's subtraction stage learns its print unguarded.

Measured on CPU on 2026-10-12 on the 7 clips and the counterexample
bbc-two-continuity (15 s each, so the single-window path). Each clip ran
through the scanner, the pre-conditioning, the cathar chain, the polish
and the pause floor keeper (no mux). The readings use the harness's own
functions over the whole clip: removal and deviation (`measure_tradeoff`),
the loud-frame 4-8 kHz and 8-16 kHz ratios as output minus source, and the
gap air of the output. The window is the one the print is learned from, in
dBFS; everything else is in dB.

| Clip | Window off | Window on | Removed off | Removed on |
| :--- | ---: | ---: | ---: | ---: |
| bbc-two-continuity | -86.1 | -86.1 | 15.64 | 15.64 |
| duck-tales | -91.3 | -36.4 | 0.70 | 3.30 |
| fun-and-fancy | -81.9 | -48.0 | -0.28 | 0.14 |
| superstition | -118.3 | -74.6 | 0.78 | 1.65 |
| tale-spin | -82.4 | -63.4 | 1.80 | 2.13 |
| animal-shelf | -119.8 | -78.5 | 1.72 | 4.15 |
| lion-king-dodge | -119.2 | -51.8 | -0.45 | -0.18 |
| scooby-doo | -94.9 | -69.2 | -0.85 | -0.84 |

| Clip (off / on) | 4-8 kHz | 8-16 kHz | Gap air |
| :--- | ---: | ---: | ---: |
| bbc-two-continuity | -0.42 / -0.42 | -0.56 / -0.56 | -64.2 / -64.2 |
| duck-tales | -0.43 / -0.58 | -0.60 / -0.69 | -12.7 / -13.8 |
| fun-and-fancy | -0.44 / -0.44 | 0.33 / 0.36 | -8.2 / -8.4 |
| superstition | 0.46 / 0.46 | 0.29 / 0.29 | -17.9 / -17.9 |
| tale-spin | -0.34 / -0.34 | -2.13 / -2.14 | -15.8 / -15.8 |
| animal-shelf | 0.30 / -0.88 | -0.64 / 0.25 | -1.0 / -1.2 |
| lion-king-dodge | 0.29 / -0.58 | 0.58 / -2.82 | -11.8 / -14.6 |
| scooby-doo | -0.72 / -0.70 | -0.47 / -0.45 | -1.2 / -1.2 |

The pick moved on all 7 and bbc-two-continuity came out sample for sample
the same. Removal rose by a median 0.42 dB (0.01 to 2.60), to a median of
1.65 dB, still short of the 2.78 dB cathar removes at HEAD on the other 206
clips the hunt had scored on 2026-10-12 (`tune_ia_hunt_v3`,
`cathar__baseline`), because on these dense clips the next quietest window
is rarely a clean pause. Programme deviation moved by
-0.90 to +0.11 dB. The cost is on lion-king-dodge: its new window straddles
the mute and the end of the programme before it, so the print carries
programme, and 8-16 kHz on the loud frames fell 3.4 dB, with 2.8 dB less
gap air. animal-shelf lost 1.2 dB at 4-8 kHz while its deviation fell
0.9 dB.

Two alternatives were measured and set aside:

- Dropping `--noiseprint` whenever the quietest window is a mute (not
  taking a louder one) read within 0.03 dB of off on all 7 clips.
  cathar's own estimate falls for the mute as well: on a synthetic 10 s
  file, `denoise --alpha 1.0` without a print removed 0.48 dB of hiss net
  of its +3.5 dB gain change, and 0.0 dB once the file held a 1 s digital
  mute, at its start or in the middle; a print learned from 0.75 s of the
  same hiss removed 9.7 dB. The fallback therefore matters only on material
  that is mute throughout, and there it removes little.
- Also rejecting windows a quarter or more of whose frames are digital,
  so that no window straddles a mute, moved three picks louder (read on
  the chain's own output, before the polish, against the same stage off).
  superstition rose to 6.01 dB removed with its highs unchanged. On
  lion-king-dodge the window was programme (-32.6 dBFS): 6.67 dB removed,
  but 9.6 dB less gap air and 1.3 dB less at 4-8 kHz. animal-shelf reached
  4.33 dB removed for 4.4 dB less at 4-8 kHz. This needs the ear before it
  becomes a rule.

The opt-in digital-mute fixture class (`--digital-mute` on
`make_realistic_fixtures_v2.py`, `docs/validation.md`) carries the case the
guard is for. On five of its fixtures, one per voice, the window cathar
learned from with the key off lay inside a gap, or mostly in one, on three.
Its spectrum stood 56.0, 63.2 and 14.3 dB (log-spectral distance, 100 Hz to
16 kHz) from the noise the fixture carries, |STFT(vhs - target)| outside
the gaps; with the key on, 3.9, 2.3 and 5.1 dB. The other two open on exact
zeros, which the old all-zero guard already skipped (2.0 and 3.3 dB either
way).

Key on, the whole local tapes (2026-10-12, CPU, each capture's audio
decoded to mono, so read before the pre-conditioning): on the seven
distinct full tapes, 134 to 1692 s long and so on the stitched path, the
level rule rejected nothing, since no window sat under -69 dBFS. The held
rule rejected windows on two: a 1 s lead-in of exact zeros, and a 0.75 s
window at the very end of a file with 75% of its samples held. The
unguarded stitched pick keeps the quietest window, so on both one of the
six pauses was that digital stretch; the key moves the stitched pick on
those two tapes and on no other. The single-window pick moved on none.

The guard stays off until a blind listening session judges it.

## cathar Noise-Print Depth Guard

`cathar_noiseprint_min_depth_db` (0, off) and
`cathar_noiseprint_shallow_action` (`scale`, `drop` or `quantile`), added
on 2026-10-12 (`quantile` the same day, after the quantile print). On a
15 s clip with no pause cathar's quietest 0.75 s is programme, and the
learned print then strips the background bed. On the
IA hunt (`tune_ia_hunt_v3`, 235 clips) cathar failed
`stems.octave_programme` on 64 of the 176 clips that read, and the
learned-print subtraction took the octave on 56 of them. On the failing
speech-profile clips the print's window sat a median 8.3 dB under the
loud frames, against 13.0 dB on the passing ones.

With the key over 0 (`modules/noiseprint_depth.py`), the chain picks the
print's window or windows again, exactly as the extractor picked them
(`cathar._probe_windows`), and measures their RMS against the material's
loud frames: the p90 of the mono mix's 1024-sample frame RMS, streamed.
A print shallower than the key is programme:

- `scale` multiplies the print's magnitudes by 10^(-(D - depth)/20), so
  it sits exactly D dB down, and writes `<print>_depth<D>db.np.json`
  beside it. cathar 0.8.0 writes the print as a JSON list of 1025 mean
  magnitudes (2048-sample symmetric Hann, hop 512) of the probe's left
  channel, and it is linear in level: a probe 20 dB quieter learned a
  print 0.1 times the first, within 6e-8. Scaling the file is learning
  from a quieter probe.
- `drop` runs `denoise` without `--noiseprint`, as the music profile does.
- `quantile` hands the denoise the whole-file quantile print in its place
  (`modules/noiseprint_quantile.py`, "cathar Quantile Noise Print" below),
  written beside the learned print as
  `noise_<stage>_quantile_k32_q0.05.np.json`, the file
  `cathar_noiseprint_method: quantile` writes. Material it cannot be learned
  from (fewer than 20 live windows, or a read or write error) is scaled as
  on `scale`, after a WARNING; with `cathar_noiseprint_method: quantile`,
  whose print already failed on that material, the guard scales without
  learning it again, after a WARNING of its own
  (`noiseprint_depth.shallow_action`).

A print at least D dB down is handed on untouched, and a depth that
cannot be measured keeps the learned print with a WARNING: no window to
read, or material whose loud level is 0 (digital silence on nine frames
in ten, or shorter than one frame), which would otherwise read as a
print 0 dB down, or scale it by hundreds of dB. The log names
the depth, both levels and the action. At 0 nothing is measured and the
slice, the noiseprint run and every command line are HEAD 57bfdb8's
(`tests/unit/test_cathar_print_depth_guard.py`). The depth is read on the
mono mix the pick reads, while cathar learns from the left channel only.
With the mute guard on as well, its lines are logged twice, once for the
print and once for the depth.

Measured on CPU on 2026-10-12, with the guard's own code at cathar's
denoise stage: the investigation's 159 speech-profile hunt clips, each
base render's denoise input and print copied and run through
`guard_print` and `_cathar_denoise_step` (cathar 0.8.0). Columns: the
clips the guard touched, the 62 failing clips still failing, the 56 of
the denoise class still failing, new failures among the 96 passing
clips, and the quiet-frame removal (`measure_tradeoff`'s median, failing
/ passing clips, in dB; the base is 5.49 / 3.23).

| D (dB) | Action | Touched | Failing | Denoise | New | Removal |
| ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| 13 | scale | 88 | 38 | 32 | 0 | 3.47 / 2.76 |
| 13 | drop | 88 | 22 | 16 | 0 | 0.20 / 0.83 |
| 20 | scale | 109 | 25 | 19 | 0 | 1.70 / 1.66 |
| 20 | drop | 109 | 17 | 11 | 0 | 0.11 / 0.30 |

The D = 20 rows are the investigation's emulation exactly. The depth
matched it within 3.6e-5 dB on every clip but two digital-mute windows
63 and 104 dB down, `drop` wrote the emulation's bytes on all 109, the
scaled prints sat within 1.5e-7 of its own, and the readings within
3e-5 dB. The 50 clips whose print is deep enough wrote the shipped stage
output byte for byte. At 20 the guard cannot reach nine denoise-class
clips, whose prints sit 21-38 dB down; at 13 it leaves 14.

The four local Tata tapes, whole (CPU, the full app in cathar mode, each
variant on its own copies), take the stitched print, and its six pauses
sit only 13.7 to 16.9 dB under the loud frames. So at 20 the guard fires
on every one of them. The denoise stage's quiet-frame removal falls by
about half on `scale` and to nothing on `drop`:

| Tape | Depth (dB) | Removal off | 20, scale | 20, drop |
| :--- | ---: | ---: | ---: | ---: |
| gaudeamus5 | 14.9 | 12.64 | 6.86 | 0.02 |
| soti | 16.9 | 17.01 | 13.02 | 0.00 |
| tele7abc | 13.7 | 13.05 | 6.58 | -0.48 |
| vaccin | 14.5 | 15.85 | 7.64 | 0.00 |

At 13 the guard keeps all four prints, and the four outputs' decoded audio
and their denoise stages are byte for byte those with the guard off. The
depth alone does not separate a dialogue tape's real pauses (13.7-16.9 dB
on these four) from a dense clip's programme (8.3 dB median on the
failing clips, 13.0 on the passing ones), so 20 would change the tapes
the listener round accepted. The candidates for the listening session
were 13 with `scale` or `drop`, which left these four tapes alone, or 20
restricted to the single-window path (15 s clips), which this key does not
offer; `quantile` at 13 has since replaced them ("The three actions on one
reading" below). Not judged by ear.

### The three actions on one reading

Measured on CPU on 2026-10-12 with the worktree's `guard_print` and
`_cathar_denoise_step` (cathar 0.8.0) at the denoise stage of the IA
hunt's 56 denoise-class failing clips and its 96 passing ones (the
quantile print's clip sets), each clip's base stage input and learned
print copied (D in dB). Every guarded output was byte for byte the earlier
validations' (`scale` and `drop` 380 of 380, `quantile` 190 of 190), no
clip fell back from `quantile` to `scale`, and every kept print wrote the
shipped output. The octave gate is re-read with the harness at 03afd4a,
as the quantile print's validation read it: two clips
(the-riddlers, channel-4-adverts-treasure-hunt) read -2.6 and -2.9 dB
where the guard's first validation read -5.5 and -5.7, so 53 of the 56
fail at the stage and each failing count here is two under the Denoise
column above. Columns: clips touched (failing / passing), failing clips
still failing, passing clips that start failing, then medians over each
group (failing / passing, in dB, output minus source): quiet-frame
removal (`measure_tradeoff`), and the APL hiss diagnosis's floor under
the programme and level in the true pauses (3-16 kHz, its `measure.py`).

| D | Action | Touched | Failing | New | Removal | Prog. hiss | Pause hiss |
| ---: | :--- | :--- | ---: | ---: | :--- | :--- | :--- |
| off | | 0 / 0 | 53 | 0 | 5.48 / 3.23 | -14.1 / -10.7 | -18.8 / -11.8 |
| 13 | scale | 42 / 43 | 30 | 0 | 3.46 / 2.76 | -9.3 / -8.8 | -12.4 / -10.9 |
| 13 | drop | 42 / 43 | 14 | 0 | 0.13 / 0.83 | -2.1 / -4.6 | -7.8 / -10.3 |
| 13 | quantile | 42 / 43 | 30 | 1 | 3.04 / 2.68 | -10.6 / -9.5 | -14.1 / -11.0 |
| 20 | scale | 47 / 58 | 17 | 0 | 1.46 / 1.66 | -5.1 / -5.5 | -9.8 / -9.8 |
| 20 | drop | 47 / 58 | 9 | 0 | 0.06 / 0.30 | -2.0 / -3.3 | -3.5 / -8.8 |
| 20 | quantile | 47 / 58 | 30 | 1 | 2.67 / 2.58 | -10.6 / -9.0 | -13.4 / -10.9 |

At 13, `quantile` cures as many clips as `scale` (23 of 53, 19 of them
the same) and keeps more of the hiss removal: on the clips it touches the
floor under the programme comes down 10.3 dB on the failing clips and 9.1
on the passing ones (shipped 15.0 and 12.0, `scale` 8.1 and 7.8), the
pauses 14.5 and 10.8 (19.9 and 13.5, `scale` 11.3 and 9.9). Its
quiet-frame removal is a little lower than `scale`'s (2.4 against 2.6 dB
on the touched failing clips): the quantile print sits lower under 3 kHz.
Of the 18 touched clips still failing, 9 read between -3.0 and -3.6 dB.
The quantile print does not depend on D, so at 20 it cures no more clips
than at 13: the five failing clips it adds (prints 13-20 dB down) all
still fail, and the 15 passing ones it adds stay passing. The one new
failure is potsworth-nick-super-switch (print 9.4 dB down), already near
the gate on the shipped print (-2.4 dB, -4.4 on the quantile print, which
removes more there: programme hiss -17.1 against -16.2 dB). The guard
spares the quantile method's other two new failures, whose prints sit
37 and 104 dB down (itv-continuity, lion-king-dodge).

The four local Tata tapes at 13 with `quantile` (whole, CPU, the full app,
each variant on its own copies): the guard keeps all four stitched prints
(13.7-16.9 dB down) and the decoded output and the denoise stage are byte
for byte those with the guard off, which are also the guard
investigation's guard-off outputs. At 20 the guard fires on all four and
the outputs are byte for byte `cathar_noiseprint_method: quantile`'s, with
its 1.5-3.1 dB more hiss above 7 kHz ("The four Tata tapes" below).

For the listening session: 13 with `quantile`, against the shipped print
on the 85 hunt clips it touches. It cures what `scale` at 13 cures, keeps
more hiss removal than `scale`, and leaves the family tapes alone; 20 with
`scale` cures 13 more failing clips, but the hiss floor under the
programme then comes down 5 dB instead of 14 on the failing clips and the
removal halves on all four Tata tapes. Not judged by ear.

## cathar Quantile Noise Print

`cathar_noiseprint_method` (`window` or `quantile`), added on 2026-10-12.
`window` (the default) learns the print from the quietest 0.75 s of a clip,
or from six stitched 0.75 s pauses on 90 s of material or more, as every
release has. A 15 s clip with no pause has no quiet window, so that print is
programme and the subtraction strips the background bed (the depth guard's
evidence above); a digital mute teaches it nothing. `quantile`
(`modules/noiseprint_quantile.py`) writes the print from the whole stage
input instead, streamed block by block, in the format cathar 0.8.0 reads: a
JSON list of 1025 mean magnitudes of cathar's own frames (2048-sample
symmetric Hann, hop 512, whole frames only, the left channel; checked to
4.5e-7 against `cathar noiseprint` on a probe). Per bin it takes the 5th
percentile of the 32-frame (0.37 s at 44.1 kHz) moving mean of the
magnitude over the windows that hold no digital mute (a frame under
-90 dBFS, or with half its samples repeating the one before), plus
0.62 dB, read off a 0.1 dB histogram so memory does not grow with the
material. The file is `noise_<stem>_quantile_k32_q0.05.np.json`. Material
with fewer than 20 such windows, or a print that cannot be read or written,
goes back to the window pick with a WARNING. The window path's mute and
depth guards do not apply (this print has no window), the music profile
learns no print either way, and auto_pure_linear's subtraction never reads
the key. At `window` the quantile module is never read: every command line
is HEAD 053ba30's (`tests/unit/test_cathar_quantile_print.py`), the five
cathar reference clips decode 5/5 identical on CPU, and the four Tata tapes'
outputs decode byte for byte as the depth guard's guard-off run. The depth
guard's `quantile` action writes the same print and subtracts it only where
the window print is shallow ("The three actions on one reading" above).

### Calibration on realistic-v2

The fixtures add the hiss after the recorder's gain control, so vhs minus
clean is each fixture's exact tape noise. Each estimator's print is read
against the print of that noise by its log-spectral distance over the bins
within 40 dB of the noise's loudest, 100 Hz to 16 kHz (LSD), its median
error at 3-16 kHz (bias) and the share of bins more than 6 dB over (over).
The bank's tape noise is no Rayleigh magnitude: its 5 kHz bin's mean sits
3.2-4.7 times its p10 (medians per class and language over the 200
tape-noise-only fixtures, a tenth of them past 9; 2.73 for Rayleigh, 2.6-3.8
in the Tata tapes' true pauses) and codec holes empty bins above
10 kHz, so a raw low quantile with the Rayleigh correction misses by 5 to
20 dB. A moving mean first narrows each bin to its local mean, holes
included. Over 1 to 64 frames and quantiles 0.05 to 0.5, each with its
correction fitted as the median per-bin error over the 300 tape-noise-only
fixtures, the 5th percentile of the 32-frame mean reads closest, at
+0.62 dB (on white noise it reads 1.93 dB under the mean; the programme
lifts most of that back). Medians of the LSD over each class's fixtures (dB):

| Class (50 fixtures each) | Window | Stitched | Quantile |
| :--- | ---: | ---: | ---: |
| speech | 3.51 | 10.12 | 4.61 |
| speech with hum | 3.86 | 8.67 | 4.21 |
| music bed | 4.14 | 9.06 | 4.33 |
| music bed with hum | 3.39 | 7.78 | 3.70 |
| music-led | 7.48 | 8.21 | 4.88 |
| music only | 7.98 | 8.49 | 4.16 |
| all 300 | 4.93 | 8.59 | 4.22 |

The bias at 3-16 kHz and the share of bins over +6 dB:

| Class | Window | Stitched | Quantile |
| :--- | ---: | ---: | ---: |
| speech | +2.39, 21 % | +7.21, 55 % | +0.06, 18 % |
| speech with hum | +1.76, 24 % | +8.26, 62 % | +0.84, 22 % |
| music bed | +1.45, 22 % | +5.89, 56 % | +1.09, 20 % |
| music bed with hum | +1.26, 16 % | +4.85, 43 % | -0.35, 14 % |
| music-led | +5.83, 45 % | +5.95, 50 % | -0.31, 15 % |
| music only | +5.00, 48 % | +5.89, 49 % | -1.41, 11 % |

Where the fixture has pauses the window is closer overall but reads the
hiss band 1.3-2.4 dB high; with no pause it learns programme. Every print
also learns the room tone (+1.5 to +2.8 dB under 3 kHz on speech and music).
The stitched pick never runs on 15 s material; it is read here for its
choice of pauses. The
module's streamed print reads within 0.06 dB (median per bin) of the
in-memory estimator, a 3 s digital mute moves it -0.1 to +1.9 dB (the
quiet stretch goes with it, never down to the mute), and a 15 s fixture
takes 0.3-0.4 s.

### IA hunt clips

On CPU at cathar's denoise stage (cathar 0.8.0, alpha 1.0, beta 0.02), each
of the investigation's speech-profile hunt clips' own stage input and
window print copied: the window print re-run (byte for byte the shipped
stage output on 152 of 152) against the quantile print, read against the
source with the octave reading's tail (`stems.octave_programme`, failing
under -3 dB) and the APL hiss diagnosis's readings: the hiss floor under
the programme (minimum statistics, 3-16 kHz ERB bands, median and
energy-sum deltas, and the bands up to 4.8 kHz), the true pauses (median and
energy mean), the pause low band, musical noise (lkr), the loud-frame
4-8 / 8-16 kHz loss where the programme reaches the band, gap air and the
's' level. The last column is informative only: the quantile print with
`cathar_alpha` 1.25.

The 56 denoise-class failing clips (medians, dB):

| Reading | Window | Quantile | Alpha 1.25 |
| :--- | ---: | ---: | ---: |
| Clips still failing | 53 | 28 | 37 |
| Octave tail | -5.69 | -3.01 | -3.90 |
| Quiet-frame removal | 5.48 | 2.67 | 3.51 |
| Hiss under programme | -14.07 | -10.22 | -13.15 |
| Same, energy sum | -14.09 | -9.93 | -13.22 |
| Same, 3-4.8 kHz | -14.23 | -10.13 | -12.97 |
| Pauses, median (26 clips) | -18.75 | -12.95 | -16.32 |
| Pauses, energy mean | -13.68 | -11.92 | -13.79 |
| Pause low band | -14.88 | -10.34 | -14.72 |
| lkr | 0.30 | 0.22 | 0.28 |
| 4-8 kHz loud loss | -0.36 | -0.58 | -0.67 |
| 8-16 kHz loud loss | -0.92 | -0.50 | -0.60 |
| Gap air under -25 dB (clips) | 3 | 1 | 1 |

The 96 passing clips:

| Reading | Window | Quantile | Alpha 1.25 |
| :--- | ---: | ---: | ---: |
| Clips failing | 1 | 4 | 8 |
| Quiet-frame removal | 3.23 | 2.58 | 3.22 |
| Hiss under programme | -10.74 | -8.85 | -11.10 |
| Pauses, median (53 clips) | -11.82 | -10.37 | -13.72 |
| Pauses, energy mean | -9.80 | -8.42 | -10.52 |
| lkr | 0.25 | 0.22 | 0.27 |
| 4-8 kHz loud loss | -0.18 | -0.17 | -0.17 |
| 8-16 kHz loud loss | -0.35 | -0.36 | -0.33 |

The quantile print sits a median 13.5 dB under the loud frames on the
failing clips' programme band (100 Hz to the programme bandwidth, at most
8 kHz) against 8.6 for the window, and only one failing clip's sits within
6 dB; on the passing clips 17.4 against 17.2. It sits 3.5 dB under the
window print below 3 kHz and 3.0 above on the failing clips, 0.3 and 1.6 on
the passing ones. Of the three new failures, the-lion-king-dodge-commercials
is a clip where the window had found nothing to subtract: its window print
was a digital mute (the clip the mute guard cost 3.4 dB at 8-16 kHz).
itv-continuity is the other way round: its window print is its 1 s
noise-only intro (within 0.25 dB of the noise in its true pauses, 300 Hz
to 16 kHz), while the quantile print sits 12-18 dB over that noise from
300 Hz to 10 kHz, the programme's own floor: the intro is under 5 % of the
clip's windows (see "Real tape and limits" below). The third, potsworth,
moves from -2.4 to -4.4 dB.

### The four Tata tapes

Whole tapes through the app in cathar mode on CPU, each variant on its own
copies; the output against the tape's own audio. The stitched print's six
pauses carry speech: the quantile print sits 1.2-3.5 dB under it above
6 kHz and 2-13 dB under it below. The hiss floor under the programme is
shown above 7 kHz (the programme ends at 4.5-7.1 kHz on these tapes) and
at 3-6 kHz, window then quantile; gap air is absolute, and the 's' level is
net of the plain frames:

| Output (dB) | gaudeamus5 | soti | tele7abc | vaccin |
| :--- | ---: | ---: | ---: | ---: |
| Floor, 7 kHz up | -13.8/-11.6 | -14.6/-13.1 | -14.1/-12.4 | -13.1/-10.0 |
| Floor, 3-6 kHz | -14.1/-11.7 | -31.6/-12.5 | -24.1/-13.1 | -15.9/-10.7 |
| Pauses, median | -8.5/-9.5 | -12.1/-11.2 | -8.2/-7.6 | -11.3/-11.3 |
| Pauses, mean | -8.4/-9.0 | -11.0/-10.6 | -6.8/-6.8 | -9.9/-9.5 |
| Pause low band | -13.2/-11.9 | -14.7/-15.9 | -10.2/-9.7 | -16.7/-15.8 |
| lkr | 0.67/0.62 | 0.76/0.57 | 0.35/0.22 | 0.45/0.35 |
| 4-8 kHz loss | -2.98/-2.61 | -1.19/-1.04 | -1.07/-1.01 | -0.67/-0.64 |
| Gap air | -5.0/-4.6 | -9.0/-8.7 | -10.3/-9.6 | -8.3/-7.0 |
| 's' level | - | +5.0/+1.1 | +3.8/+2.0 | +2.7/+1.8 |

The quantile print leaves 1.5-3.1 dB more hiss above 7 kHz under the
programme, and stops the 24-32 dB cut at 3-6 kHz on soti and tele7abc,
which was speech; the pauses come out alike (the pause floor keeper refills
them), with less musical noise and a smaller lift of the 's'. Replayed at
the denoise stage (informative), `cathar_alpha` 1.25 with the quantile
print brings the floor above 7 kHz to within 1.5 dB of the window's on all
four (-14.5, -16.6, -16.0, -13.2 against -13.9, -15.2, -14.5, -14.3) at
lkr 0.80, 0.71, 0.32, 0.38 (window 0.86, 1.25, 0.46, 0.46). Not judged by
ear: the listening session gets `quantile` at alpha 1.0 and 1.25 against
`window`, on the hunt clips that stop failing and on the Tata tapes.

### Real tape and limits

Read against the noise of the true pauses (the mean magnitude of cathar's
frames lying wholly inside `pause_metrics.true_pause_mask`'s pauses, the
left channel), the quantile print sits under the hiss above 3 kHz on real
tape. On the four Tata tapes it reads 0.5, 1.1 and 1.6 dB under it at 3-6,
6-10 and 10-16 kHz (medians); the stitched print reads +4.3, +1.2 and
+0.1 dB there and +8 to +11 dB under 3 kHz, where its pauses carry speech.
On the 52 IA hunt clips with at least 40 such frames it reads 0.2, 1.0 and
1.8 dB under, the window print (which found a pause there) -0.1 to +0.05.
On the 51 realistic-v2 speech and music-bed fixtures with pauses the same
reading puts the quantile print +0.3 dB over the pause noise at 3-6 and at
6-10 kHz, so the calibrated +0.62 dB leaves it 0.5-0.8 dB lower at 3-6 kHz
and 1.3-1.5 dB lower at 6-10 kHz on real tape than on the bank, whose noise
spreads wider than real hiss (above). Part of the hiss the quantile
print leaves is that offset: recheck the correction on real tape before a
default moves.

Limits, measured on the hunt clips and on synthetic 15 s material (hiss at
-50 dBFS):

- The 5th percentile needs noise-only material in 5 % of the windows, about
  1.1 s of a 15 s clip in runs longer than the moving mean; with less, the
  print is the programme's own floor (itv-continuity, above).
- A digital mute is left out frame by frame, but a near-silent stretch over
  -90 dBFS whose bins a lossy codec emptied is learned as it is:
  gogs-vhs-collection's 3 s at -73 to -84 dBFS put the print at the
  histogram floor above 300 Hz, so nothing is subtracted there (its window
  print was as empty).
- A fade to digital silence counts until it passes -90 dBFS: a 100 dB fade
  over 2 s at each end pulls the print 4.2 dB down in every band, over 1 s
  0.1 dB.
- A partial held through 95 % of the windows is learned in full (a chord
  held 15 s reads +52 dB over the hiss at its partials) and subtracted, as
  the window pick does on such material.

## Event Log

`AI_RESTORE_EVENT_LOG=<dir>` in the environment (not a `config.yaml` key)
makes the event-gated stages write what they found into that folder, one
JSON record each, `<stage>__<recording>__<mode>__<track>.json`: the
sibilant guard and the plosive tamer (the spans they detected, on the
pre-neural audio's timeline, with their thresholds), the pause floor keeper
(the pauses it found and the frames it filled, on the restored file's
timeline), `auto_pure_linear`'s post-neural hiss gate (its decision, its
thresholds and the reading it decided on, when its key is on), and the mux's
loudness stage (the measured values, the mode
ffmpeg's rule predicts, the fallback, and what the applied pass reported,
`normalization_type` included; also written beside the track it masters).
A stage that did not look records why (switched off, refused material, or
`failed: <error>`). Unset, nothing is written, no extra pass runs, and the
audio is the same either way. The records are written for the
output-quality harness's coverage reading (ear v3's R3,
`meta.fricative_coverage`), which will measure how many events a stage
reaches; that reading is not built yet, so until then compare a stage's
`events_s` with the harness's own locator by hand
(`.agent/workflows/tune_on_real_tapes.md`). Two runs in the same mode share
record names, so give each its own folder.

## Stage Cache

`AI_RESTORE_STAGE_CACHE=<absolute folder>` in the environment (not a
`config.yaml` key) keeps what `auto_pure_linear`'s and `denoise_only`'s
neural stage hands on (everything up to and including the neural denoiser:
the surgical notch, the deterministic chain and the model) and replays it
when the stage runs again on the same input under the same settings. The
self-driving loop sets it with `--stage-cache`
(`scripts/autotune_restoration.py`), so a candidate whose knobs all act after
the denoiser skips the chain and the model. Unset (the default), the cache is
one environment read and every output keeps its bytes; a relative path leaves
it off, with a warning, because each candidate runs in its own working folder.

- An entry, `<folder>/v1/<key>/`, holds the chain's output (the sibilant
  guard's reference) and the neural output, restored on a hit at the paths
  the stage wrote them at under its folder, with a manifest of their sizes
  and SHA-256 digests and the full key document (`key.json`, for diffing a
  miss). A hit checks every digest while it copies; an entry that does not
  read back is discarded and the stage renders afresh, while a replay that
  fails with the entry still intact (the work folder's disk full, a file
  there locked) keeps the entry. A cache folder that cannot be read is a
  miss, with a warning: no cache problem ends a restoration. Final outputs
  are never cached.
- The key covers the input by its decoded samples (a soundfile WAV carries a
  PEAK chunk with a write timestamp, so two writes of the same samples differ
  in their bytes), its name and layout; every argument of the stage, the
  scanner's whole strategy and the seven stage switches included; the
  resolved configuration minus the keys below; the sources of every module
  but `sibilant_guard`, `pause_floor`, `mastering`, `sync`, `hiss_floor`,
  `post_neural_suppress` and `neural_level`, what importing those seven runs
  (their import-time code, without the bodies of the
  functions import never calls) and `restore_audio_hybrid.py`, read when the
  process starts; ffmpeg (content and `-version` output) and cathar
  (content), which the chain runs; the model files the stage can load (name,
  size, modification time) in the folder audio-separator loads them from
  (`AUDIO_SEPARATOR_MODEL_DIR` when set, else `models/`);
  `assets/blend_weights.npz`; the interpreter, the installed packages (read
  at start-up too), torch, the GPU and its driver, the CPU and the
  denoiser's chunk length; and, deny by default, every environment variable
  whose name starts with `AI_RESTORE_`, `AUDIO_SEPARATOR_`, `CUDA`,
  `CUBLAS`, `CUDNN`, `NVIDIA_`, `TORCH`, `PYTORCH_`, `OMP_`, `KMP_`, `MKL_`,
  `OPENBLAS_`, `GOTO`, `BLIS_`, `VECLIB_`, `NUMEXPR_`, `NUMBA_`, `NPY_`,
  `RAYON_`, `ORT_` or `ONNXRUNTIME`, and `PYTHONHASHSEED`, but the cache's
  own two and `AI_RESTORE_EVENT_LOG`.
- The 36 keys left out of the key, each read only after the denoiser
  (`POST_NEURAL_CONFIG_KEYS` in `modules/stage_cache_key.py`, held to the
  code by `tests/unit/test_stage_cache_allowlist.py` and
  `tests/unit/test_stage_cache_reach.py`): `enable_linear_air`,
  `linear_air_gain_db`, `linear_air_freq_hz`, `enable_dynamic_expander`,
  `expander_depth_db`, `expander_knee_offset_db`, `expander_attack_s`,
  `expander_decay_s`, `apl_expander_depth_db`,
  `apl_enable_sibilant_guard`, `apl_sibilant_mix`, `apl_sibilant_guard_hz`,
  `apl_sibilant_hf_share_min`, `enable_pause_floor`, `pause_floor_fill_db`,
  `pause_floor_quiet_percentile`, `pause_floor_target`, `pause_floor_shape`,
  `pause_floor_shelf_hz`, `pause_floor_hf_cut_db`, `apl_post_neural_suppress`,
  `apl_post_neural_suppress_floor_gap_db`,
  `apl_post_neural_suppress_min_residual_db`, `linear_air_gate`,
  `apl_neural_level_restore`, `apl_burst_ceiling`,
  `apl_burst_ceiling_depth_db`, `enable_loudnorm`, `loudnorm_target_lra`,
  `loudnorm_linear_fallback`, `mux_limiter_latency_compensation`,
  `mux_sample_rate`, `preserve_original_audio_track`, `dtw_resolution`,
  `sync_dtw_enable` and `apl_music_bg_floor_db`. Every other key, one the
  code does not know included, is in the key; the `cathar_*` keys among
  them, because the chain's repair and subtraction run cathar's stages at
  those settings.
- Nothing is stored from a render during which a stage logged a warning or
  an error (a fallback after a failure is never frozen), audio-separator's
  own included (it logs through Python logging; the line
  `Using soundfile for writing.`, logged at WARNING on every render, and its
  note on inputs under 10 s do not count); from a stage folder that already
  held files (a resumed work folder, whose stages reuse what they find);
  when a module source was written after the process started, or the
  sources or the installed
  packages no longer hash as they did then (an edit or an install while a
  candidate renders); when an output lies outside the stage folder; or when
  the entry would put the cache over its cap or leave under 20 GB free on
  the volume, both read again from the disk once eviction ran.
  `apl_use_deepfilternet` and `apl_use_resemble_denoise` (weights outside
  `models/`) and `AI_RESTORE_EVENT_LOG` (the plosive tamer records before the
  cached stage) bypass the cache.
- `AI_RESTORE_STAGE_CACHE_MAX_GB` (50) is the cap; the least recently used
  entries go first, a hit counts as a use, and the entries other processes
  are still building count toward it. An entry takes about 42 MB per minute
  of stereo tape, 2.5 GB per hour. Only folders named like a key (64 hex
  digits) or like an entry in progress (`tmp-<12 hex digits>-<pid>`) are
  counted, evicted or swept: anything else under `v1/` is left alone.
- cathar never uses it. On the music stem path only the vocal stem's neural
  stage is cached; the separation still runs.
- The run log says `[Stage Cache] Miss`, `Stored` (the entry's size, the
  render time it saves and the cache's total), `Hit`, `Evicted`, `Bypassed`
  and `Not stored` with the reason.

## Model Files

`vocals_model`, `denoise_model` and `apl_neural_model` name UVR model files
that `audio-separator` downloads into `models/` (`background_model` is
accepted and validated the same way but no mode reads it yet). Each must be a
bare filename such as `UVR-DeNoise.pth`: a value that is not a
string, or contains a path separator, a drive prefix (`D:`), a `..` segment,
or a leading dot, is ignored with a warning on stderr and the setting falls
back to its default. A model file that fails to
load is deleted from `models/` so it can be re-downloaded, and only a file
inside that directory is ever deleted.

The output-quality harness keeps its own weights beside them, in
`models/sigmos/`, `models/whisper-large-v3-turbo/`,
`models/wavlm-base-plus-sv/`, `models/audiobox-aesthetics/`,
`models/utmos/`, `models/scoreq/` (the NeurIPS 2024 no-reference MOS, ONNX
weights from Zenodo, CC-BY-4.0) and `models/mert-v1-95m/` (the music
embedding whose distance reads the non-vocal stem, CC-BY-NC-4.0, research
use), fetched by `scripts/download_quality_models.py` from pinned upstream
revisions; each directory carries a README with the licence and a
`MANIFEST.json` with the sha256 of every file. No `config.yaml` key names
them. See `docs/validation.md`, "Output validation harness".

## Batches

`batch_jobs` (default 1) is how many files restore at the same time when a
folder or several files are given. Each file runs in a child interpreter on
its own, with its own work directory and a log under `logs/<name>.log`, so
its output is bit-identical to a solo run and a failure in one file leaves the
others alone; the parent prints start, finish and the failed files. cathar is
single-threaded on every stage (measured on a 134 s tape: the SBR enhance
stage uses 12 s of CPU for 12 s of wall, `RAYON_NUM_THREADS` changes nothing)
and its stages run one after another, so parallel files are the only way it
uses more cores: four jobs restore a folder of four tapes in the time of the
longest one. Neural modes load their models once per job; count about 6 GB of
GPU memory per `auto_pure_linear` job.

## Long Captures

`neural_chunk_seconds` is the longest track the neural denoiser (the UVR
models, or the Mel-RoFormer that `apl_neural_model` names) is given in one
pass. The UVR separator keeps several full-length copies of the track in
memory after inference, about 30 GB per hour of audio (48 GB measured on a
94-minute chunk), so a long capture fails outright: a 3h00m tape runs out of
memory on a 62 GB machine. The RoFormer's memory per hour has never been
measured; its chunks are sized by the same UVR figure. The default, `0`,
sizes the chunk from the machine: half its
memory at that rate, between five minutes and two hours: 57 minutes on a
62 GB desktop, 28 on a 30 GB laptop, 11 on a 12 GB machine. A track longer than
that is cut into equal chunks no longer than it, each denoised on its own and
rejoined; tracks up to it are processed whole, exactly as before. A positive
value fixes the length, and one longer than any capture keeps every track
whole at the cost of memory. Inside a container the sizing reads the cgroup
memory limit rather than the host's total, so a `--memory 8g` container gets
seven-minute chunks and not a chunk sized to the machine it runs on.

The floor is a real one. Measured on an 88-minute capture in memory-limited
containers on the same GPU: a 16 GB container restored it in six
fifteen-minute chunks with nothing to spare on the way in; an 8 GB container
was killed after the spectral subtraction, before the chunked denoiser was
reached, because the stages ahead of it hold the whole track. Plan on 16 GB
for captures of an hour or more.

Chunking is built so its result is the whole-file result to within float
rounding, not merely a good approximation of it, and each part of that is
measured on an 88-minute capture cut into three:

- Chunks overlap by two seconds and are rejoined with a linear crossfade.
  Both sides of a seam are the same audio denoised twice, so a linear fade
  sums to unity where an equal-power fade would leave a +3 dB bump.
- Each chunk starts on a multiple of the models' patch stride (192 frames of
  a 1024-sample hop for `UVR-DeNoise-Lite`, of a 480-sample hop for
  `UVR-DeNoise`; 66.9 s at 44.1 kHz), so its frames and windows fall on the
  grid the whole file's would.
- The model normalises its spectrogram by the file's maximum before
  inference, and a chunk on its own would be scaled by its own loudest moment
  instead. That alone left a chunk's body 33 dB below the signal from the
  whole-file result. Every chunk that does not already contain the file's
  loudest stride-aligned block is given it as a lead-in, dropped at the join.
- Each chunk also carries one stride of the true audio before and after it,
  dropped at the join, so its edges are denoised in real context rather than
  against the model's reflection padding.
- Loudness is normalised once over the joined result, under the same rule a
  whole track gets. The separator also peak-limits each file it writes; a
  chunk it limited on its own would step the level at a seam, so such a chunk
  is denoised again at half scale, which the model is indifferent to since it
  normalises its input, and the join doubles it back. Only a chunk that limits
  pays for that, in the fidelity of half-precision inference on a rescaled
  input, about -57 dB; on the second three-hour capture in the test folder it
  was one chunk of two.

With all of that in place the join matches the whole-file output on 99.97% of
samples, with the body at -103 dB and the seams at -156 dB relative to the
signal; the worst single sample is -68 dB. Chunks shorter than the automatic
length buy nothing: the lead-in already gives every chunk the file's maximum,
and each chunk costs a stride of context either side.

### The Mel-RoFormer and Other MDXC Models

Everything above is the VR models' design. The shipped speech denoiser of
`auto_pure_linear` (`apl_neural_model`, the Mel-RoFormer) is an MDXC model,
and until 2026-10-11 it never finished a track longer than one chunk: the
chunked path took only a `(No Noise)` stem, the RoFormer writes `(dry)` and
`(other)`, so every speech tape over the chunk length (57 minutes on a 62 GB
machine, 14 on a 16 GB one) failed in `auto_pure_linear` and in `auto`, from
v1.3.3, where the RoFormer became the default, on. Music tapes, which run
UVR (`apl_music_neural_model` is empty), were not affected.

- One rule picks the clean stem wherever a denoised file is taken, whole or
  chunked, fresh or left by an earlier run (`modules/denoise_model.py`):
  `(No Noise)` first, then the training target in the `_config.yaml` beside
  the checkpoint (`(dry)`), then the first file in name order that is not a
  residual; never `(Noise)` or `(other)`. `models-scores.json`'s
  `target_stem` is never read: it names the noise stem for the UVR models.
  The whole-file pass already took `(dry)` by name order, so no output under
  the chunk length changes; all 318 denoise directories found on this
  machine get the file they got before.
- The RoFormer denoises 8 s windows overlap-added every 2 s, 88,200 samples
  (from its yaml: hop 441 x (dim_t 801 - 1) = 352,800, four windows deep),
  each window on its own. Its chunks start on that step rather than on the
  VR patch stride (2,949,120 samples is 33.4 steps) and carry a full window
  of true audio either side, so every window that touches a chunk's range
  is one the whole file ran. There is no file-wide spectrogram maximum, so
  no lead-in. The chunk directory's name ends in `_mdxc88200`.
- The separator scales its input down to 0.9 when the file's peak is above
  it, and a chunk on its own would be scaled by its own peak, or not at all.
  Each chunk is written at the gain the whole file is fed at, computed as the
  separator computes it, so the network sees the whole file's samples bit
  for bit. The half-scale retry of a chunk whose output reached full scale
  stays as the VR path has it; for the RoFormer it changes the level the
  network sees, so such a chunk is not the whole file's to the bit.
- The grid is read from the yaml, which the separator downloads with the
  checkpoint on the model's first load; the installers do not provision it.
  A fresh model store whose first long speech tape goes straight to the
  chunked path would have been cut to the VR grid, so the chunked pass first
  fetches the model's files (`download_model_files`, without loading the
  model) when a `.ckpt` model's yaml is missing. A `.ckpt` model left without
  a usable yaml still runs, on the VR grid, with a WARNING in the log.

Checked on the CPU, without the network: a replica of the separator's
RoFormer overlap-add, with a level- and framing-dependent mask standing in
for the network, over a 7.2-minute speech tape cut into two 300 s chunks. The
join equalled the whole-file render on every body sample and stayed within
-158 dB of the signal at the seams (crossfade rounding), at the tape's own
level and with its peak pushed to 1.07. The same replica cut to the VR grid
drifted to -47 dB (-30 dB with the hot input). A second check ran the
library's own code (audio-separator 0.47.0's `MDXCSeparator.separate` and
`demix`, with a stub in place of the network) through the app's own denoise
runner, on a synthetic 150 s stereo file cut into three 60 s chunks: every
body sample equal to the whole-file render, the seams at -161 dB, at input
peaks of 0.6 and 1.2; cut to the VR grid it drifted to -33 and -23 dB.

Checked on the GPU with the real network (2026-10-12): a 374 s cut of the
SOTI tape in `auto_pure_linear`, rendered whole (`neural_chunk_seconds: 0`)
and in four 120 s chunks (`neural_chunk_seconds: 120`; on bb5fa2a the same
run failed on its first chunk, "no clean stem"). The chunked `(dry)` output
equals the whole-file one on 99.67 % of its samples; the rest sit in the
seconds around the three seams, at most -150.5 dBFS (-167.8 dB against the
signal overall). Both renders peaked at 2.3-2.4 GB of process memory on
this host, the model and the interpreter included, so the RoFormer is far
below UVR's 32 GB per hour, which still sets the chunk length.
