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
  and the canceller do better together. - `apl_enable_dehum` (default
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

## Event Log

`AI_RESTORE_EVENT_LOG=<dir>` in the environment (not a `config.yaml` key)
makes the event-gated stages write what they found into that folder, one
JSON record each, `<stage>__<recording>__<mode>__<track>.json`: the
sibilant guard and the plosive tamer (the spans they detected, on the
pre-neural audio's timeline, with their thresholds), the pause floor keeper
(the pauses it found and the frames it filled, on the restored file's
timeline), and the mux's loudness stage (the measured values, the mode
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
  but `sibilant_guard`, `pause_floor`, `mastering` and `sync`, what importing
  those four runs (their import-time code, without the bodies of the
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
- The 22 keys left out of the key, each read only after the denoiser
  (`POST_NEURAL_CONFIG_KEYS` in `modules/stage_cache_key.py`, held to the
  code by `tests/unit/test_stage_cache_allowlist.py` and
  `tests/unit/test_stage_cache_reach.py`): `enable_linear_air`,
  `linear_air_gain_db`, `linear_air_freq_hz`, `enable_dynamic_expander`,
  `expander_depth_db`, `expander_knee_offset_db`, `expander_attack_s`,
  `expander_decay_s`, `apl_expander_depth_db`,
  `apl_enable_sibilant_guard`, `apl_sibilant_mix`, `apl_sibilant_guard_hz`,
  `apl_sibilant_hf_share_min`, `enable_pause_floor`, `pause_floor_fill_db`,
  `pause_floor_quiet_percentile`, `enable_loudnorm`, `loudnorm_target_lra`,
  `loudnorm_linear_fallback`, `preserve_original_audio_track`,
  `dtw_resolution` and `apl_music_bg_floor_db`. Every other key, one the code
  does not know included, is in the key; the `cathar_*` keys among them,
  because the chain's repair and subtraction run cathar's stages at those
  settings.
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

`neural_chunk_seconds` is the longest track the UVR denoiser is given in one
pass. The separator keeps several full-length copies of the track in memory
after inference, about 30 GB per hour of audio (48 GB measured on a 94-minute
chunk), so a long capture fails outright: a 3h00m tape runs out of memory on
a 62 GB machine. The default, `0`, sizes the chunk from the machine: half its
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
