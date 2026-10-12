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
    `*_Speech_Cleaned`.

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
abstained alike (`D:i_restore_work\dtw_check`, local).

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
  and never reads this key.

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
- The 34 keys left out of the key, each read only after the denoiser
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
  `apl_neural_level_restore`, `enable_loudnorm`, `loudnorm_target_lra`,
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
