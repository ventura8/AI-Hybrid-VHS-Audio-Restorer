# Ear v4: Critical of Hiss

The acceptance record of ear v4 (built 2026-10-12 on `wip/ear-hiss`), the
readings that make the harness very critical of VHS hiss under speech or
singing and in silence and pauses (the user's rule of 2026-10-11). The
first acceptance run failed; validation round 1 (the same day) traced each
failure to a defect in the readings or in the transforms that set their tie
bands, fixed the implementation, and re-measured every row. The bars were
not moved: where a row's own check contradicts the user's verdicts or the
signal, the row is listed as not passed, with the evidence, under "Rows
Whose Check Contradicts the Evidence". Nothing here shipped a default: the
blind block (`audibility_check.py`, then `listen_ab` with the device and
volume recorded) and the realistic-v2 plus IA confirmation come first. T,
S, V and G are Tele7abc, SOTI, Vaccin and Gaudeamus5. The readings and the
grids are described in the output-quality-harness skill, "Critical of hiss
(ear v4, 2026-10-12)".

## Validation Round 1: Defects and Fixes

Each defect as measured in the first run, its cause, and the fix with the
measurement after it:

- **A perfect fixture output read +9..+22 dB** (0 of 43 at or under 0). A
  clean output reveals soft programme the source's hiss hid in band-edge
  bands (9-16 dB under the floor); the top tenth of noise-dominated cells
  held 74 % of that leak, and it read as hiss. Free and busy band-frames are
  read apart, a spectral cap holds each band 6 dB over the other bands' free
  cells, and a 3-frame band-power bound holds the rest: 41 of 43 at or under
  0 (median -17.4 on T's shape).
- **The programme reading sat 0.16 dB per band low.** Each cell was capped at
  its own peak (3.7 % of cells). The peak test now reads the frames two away
  (t +-2).
- **A fresh realisation moved the fixtures' programme reading +0.4..+0.9
  dB.** The masker came from the output's own realisation and source-chosen
  frames. The masker now comes from the source and the output plays at its
  own loud frames.
- **A held note read +20 dB.** A band the anchor left unread (NaN) took no
  cap. The anchor's unread bands are interpolated in log level.
- **`fresh_floor` moved G's pause median +1.6..+2.1 dB.** The transform
  redrew a held note as a band of noise. `same_percept.noise_led` gained a
  spectral-peak test.
- **`fresh_floor` lost 0.32 dB of a stationary white noise, correlated 0.26
  with its old draw, and lost up to 3 dB in its crossfades.** Its steady test,
  2x the P20 of the 5 x 5 envelope (1.5x a stationary floor's mean),
  failed about 5 % of pure-noise cells, whose islands of the old draw lost
  their energy in the inverse STFT; each block drew its own noise. The test
  is now 3x the P20 (a spectral subtractor's musical-noise islands and held
  tones still stand over it), a noise-led cell mixes power-complementarily
  with the old one as its 5 x 5 neighbourhood goes 50-90 % noise-led, and
  one noise covers the file: a white noise within 0.04 dB at correlation
  0.07, each level of a 20 dB gated noise within 0.05 dB. On six pairs the
  fresh draws' p95 move fell from 0.28 to 0.22 (programme), 0.63 to 0.54
  (pause median) and 0.65 to 0.52 (pause tail). Deciding on 9 x 9 means
  instead redrew the islands too: a change of texture that moved cathar
  renders' programme reading -0.3..-0.8 dB.
- **The kept-HF guard missed a 4-6 kHz cut.** One median over every cell was
  dominated by the 3-4 kHz cells. It is now the mean of per-band medians,
  each cell held at the 16-bit dither.
- **Requantisation moved V's kept HF 2.9 dB.** Dither landed in the guard's
  cells. The dither floor took it away (jitter p95 now 0.003).
- **Gaudeamus renders read no window.** A waveform-correlation guard
  rejected their polarity-inverted programme, and a 6 dB gain guard the
  least-squares shrink. The guard is now the 300-4000 Hz envelope
  correlation (at least 0.5) and a 24 dB sanity bound on the gain (G1
  8.2-8.3 dB, G5 14.6-14.9 dB, at envelope correlations of 0.995-0.998).
- **A band-filtered pause read -1843 dB.** Nothing held the margin. It is
  now held at L_crit - 140 = -65.
- **The pause margin scattered sd 0.25 dB per realisation.** The max over
  single ERB bands picked their upward draws. Each band is power-averaged
  with its two neighbours: sd 0.13-0.18.
- **`hiss_islands` failed** (2.25 floors at the mild level).
  `fresh_noise_linear` re-imprinted the old draw (benign floor 0.224). It is
  now an exact fresh draw of the tape hiss: floor 0.015, 33.6 floors.
- **The two SPL conventions disagreed by up to 0.88 dB.** The programme
  reading set the output's level through the least-squares gain. Each side
  now plays at its own loud frames (75 dB SPL), as in the pause reading, so
  the two agree by construction.

Tried and dropped, with the reason: selecting noise cells by the output's
neighbours and a valley bound on every band (a fresh draw shifted the tapes
+1.2-1.7 dB, and the keeper's gap hiss read low: SOTI APL -25); requiring
two selections to agree (unstable); selecting on free cells only or on the
source only (weak revealed programme stayed in); the mean pause floor (a
perfect output's pauses read up to +43); a rule emptying pause bands under
the 16-bit dither (the room delta broke: cathar -117).

## What Ran

- The jitter report: the eight pairs of the specification (v4_air
  `B_air_1dB` and v2 `final2_cathar` of T, S, V and G against their
  sources), `--transforms hiss`, two repeats, `--families dsp` and the
  first 125 s, through `reward_noise_floor`'s own functions with the final
  build; the pooled report is the worktree's
  `experiments/reward/noise_floor_hiss.json` (gitignored).
- The fixtures: `python -m scripts.check_hiss_fixtures --programme linear`
  over de, en, es, fr and it of `artifacts/realistic-v2` (40 gapped m12
  tape-noise programmes), with three noise shapes: `--shape corpus` (the
  fixtures' recorded noise), the hiss shape and T's raw tape
  (`artifacts/hiss-v1/shape_tele7abc.npy`, gitignored, never committed).
- The calibration: `calibrate_quality_metrics` with `--languages en,fr`
  and `--metrics dsp`, before the change (the base tree, bb5fa2a) and with
  the final build.
- The tapes: every stored render re-scored with the final build on
  `validate_restoration`'s windows (15 s, hop 7.5 s), the hiss readings
  through the runner's own functions: `r1` (the round-one cache of
  `experiments/tata_listen`), `v2` (the v2 finals and the identity),
  `v4air` (v4_air A, B, C, Session 0's shipped APL and three synthetic
  cheats built from B), `fill` (the autotune_v6 APL and cathar candidates
  with `pause_floor_fill_db` 8, 12, 18 and the keeper off), `known` (the
  known-ordering excerpts) and `music` (the 11 autotune_v6 music clips and
  Gaudeamus1).
- The IA defect hunt: the cathar and APL baselines of
  `experiments/tune_ia_hunt_v3` against their sources (`python -I`).
- The family tapes never left the machine; nothing was uploaded and no web
  service saw their audio. CPU only, no GPU, no learned model.

## Same-Percept Rows (the Jitter Report)

Eight pairs, nine scorings each (the reference and the eight transforms),
two repeats; the pause keys pool 136 deviations (all eight pairs), the
programme keys 102 (T, S and G: V's first 125 s validates no programme
window). The DSP repeat jitter is exactly 0 on every dsp and file reading.
Per transform (p95 over the pairs):

- Gain +-0.1 dB: 0.000 on every margin and excess key, median and tail,
  and on the fixtures. **Pass** (limit 0.01).
- Delay 0.37 samples: programme margin and excess median 0.036, all-band
  0.011, pause margin and excess 0.003; largest 0.13 (programme margin
  tail). **Pass** (limits 0.05 programme, 0.02 pause, max 0.30).
- soxr round trip: 0.000 everywhere. Requantisation: at most 0.038
  (all-band), programme margin 0.024, pause at most 0.008. **Pass** (limits
  0.05 / 0.10 and 0.10 / 0.30).
- Fresh realisation (`fresh_floor_a` / `_b` / `_c`, each over the eight
  pairs): programme excess median 0.26 / 0.05 / 0.08 (limit 0.65), all-band
  0.31 / 0.25 / 0.37 (0.45): **pass**. Pause excess tail 0.80 / 0.14 / 0.50
  (0.55) and median 0.39 / 0.36 / 0.99 (0.35): **not passed**. The
  outliers are Gaudeamus5's cathar final (tail 0.80, median 0.99), a music
  tape whose "pauses" hold quiet music the transform partly redraws as
  noise; on the six speech pairs the fresh draws move the pause median at
  most 0.36 (T's B) and the tail at most 0.24.
- The ties: each grid `tie` is the pooled p95 floor of its flat key
  rounded up to 0.05: programme excess median 0.073 -> **0.10**, all-band
  0.247 -> **0.25**, pause excess tail 0.174 -> **0.20**, median 0.327 ->
  **0.35**. H1's `max_move` takes the same values; H2's bound takes its
  delta key's own floor: programme margin delta median 0.130 -> **0.15**,
  pause margin delta tail 0.181 -> **0.20**, music's all-band delta 0.244 ->
  **0.25**. **Pass.** With the first build's `fresh_floor` the pause ties
  read 0.35 / 0.45 and H2's 0.30 / 0.25: its power loss read every fresh
  pause low.

## Fixture Rows (`check_hiss_fixtures`)

Validated fixtures: 23 of 40 on T's shape, 20 of 40 on the hiss shape, 12
of 40 on the corpus shape. The corpus noise is rumble-heavy, so its anchor
validates on under half; T's shape is the representative one.

- Gain +-0.1 dB: p95 and max 0.000 on every hiss key, all three shapes.
  **Pass.**
- Delay 0.37 samples: programme margin p95 0.001-0.003 (max 0.008), pause
  0.001. soxr: programme p95 at most 0.007 (max 0.012), pause at most
  0.003. Requantisation: programme p95 at most 0.009 (max 0.013), pause at
  most 0.006 (max 0.011). **Pass.**
- Fresh realisation: programme margin p95 0.238 (T's shape), 0.340 (hiss),
  0.597 (corpus); all-band 0.295, 0.310, 0.912; pause margin 0.511, 0.542,
  1.359. **Pass on T's and the hiss shape for the programme (0.65), the
  all-band (0.45) and the pause tail (0.55); not against the pause
  median's 0.35** (a fixture holds two 1.2 s gaps, one pause estimate).
- Oracle ladder monotone (identity, -6, -12, -20, -30 dB, prog_only, +0.05
  dB slack): 23 of 23 (T's shape), 20 of 20 (hiss), 9 of 12 (corpus).
  **Pass on T's and the hiss shape.**
- The perfect output (prog_only) at or under 0 dB: 22 of 23 (T's shape,
  median -17.4, max +0.38), 19 of 20 (hiss, median -11.0, max +4.5), 7 of
  9 (corpus). **Not passed (41 of 43 against 100 %).** Both misses are the
  German music fixture `mid00_music_m12`, whose programme ends at 2.2 kHz:
  its -30 dB oracle still reads +2.3 (T's shape) and +8.1 (hiss), so its
  few readable bands of 1 kHz and up hold the revealed programme. At the
  tape relation the anchor errs -0.1 dB (median) and the pause ladder is
  exact (6 / 12 / 20 / 30 dB).
- Physical removal (`hiss_prog_removed_hf_db` - g at 6 and 12 dB): -0.009 /
  -0.030 (T's shape), -0.004 / -0.011 (hiss), -0.017 / -0.028 (corpus).
  **Pass.**
- hf_left (noise above 4 kHz kept, the rest taken 20 dB down): |median
  delta| 0.104 (T's shape), 0.108 (hiss), 0.090 (corpus). **Pass.**
- gated_m0: margin within +0.5 dB of identity on 23 of 23 and 20 of 20;
  pump at least 15 dB on 23 of 23 (+20.4..+25.4) and 20 of 20 (+20.3..
  +25.1). Corpus: 9 of 12, no pump read. **Pass on T's and the hiss
  shape.**
- islands_15: island kurtosis +0.63..+0.73 (T's shape), +0.66..+0.72
  (hiss); corpus median +0.37. **Pass on T's and the hiss shape.**
- lp4k_all: kept HF at or under -1 dB on 23 of 23 (T's shape, -4.4..
  -10.5), 19 of 20 (hiss), 9 of 9 (corpus). **Not passed on one hiss-shape
  fixture** (Spanish speech, kept HF -0.35): its programme ends at 3.6
  kHz, so a 4 kHz low-pass takes no programme from the bands the guard
  reads.

## Calibration Rows (en, fr, dsp)

- Before: 108 of 165 checks pass, 3 non-blind failures (`gated_pauses`
  pause pumping, `pause_residual_islands` twice). With the final build: 132
  of 195 pass, the same 3 non-blind failures and no other.
- Every pre-existing check status is unchanged (165 of 165); every non-hiss
  benign floor is unchanged (51 of 51). Only the `dsp.hiss_*` floors come
  from the `*_linear` benign cases (`benign_only`). **Pass.**
- The new degradations, non-blind checks, all **pass**:
  - `hiss_under_programme`: programme margin down on the oracle ladder
    (-5.97 / -11.87 / -19.53; the first build saturated at -12.86), pause
    margin flat, pump down, kept HF flat, tilt flat.
  - `hiss_in_pauses_only`: pause margin down (-5.99 / -11.99 / -19.99),
    programme margin steady (-19.53..-19.58), `gap_atten_db` up.
  - `hiss_hf_tilt`: both margins down, kept HF flat.
  - `low_floor_kept`: pause margin steady (span 0.000) at -29.99, far below
    the centre; programme margin steady (span 0.000).
  - `dead_pauses`: pause margin, room delta, pause depth and pump move as
    specified; programme margin steady.
  - `hiss_islands`: island kurtosis up and monotone (+0.50 / +0.77 /
    +1.12), the mild level 33.6 floors (floor 0.015).
  - `hiss_removed_dulled`: kept HF down (-17.4 / -47.5 / -53.9), programme
    margin below (-29.7..-29.9).
- The benign floors of the hiss readings: programme margin 0.035, all-band
  0.068, pause margin 0.337, kept HF 0.098, pump 0.109, island kurtosis
  0.015, room delta 0.034.

## Tape Rows

- **Identity** (source against itself, four tapes, v2 and round one):
  every `dsp.hiss_*` delta exactly 0. **Pass.**
- **Programme-only change** (v4_air A / B / C: the APL air shelf +2, +1 dB
  and off): pause margin tail spread 0.01 (T), 0.00 (S), 0.01 (V), 0.09
  (G): **pass**. All-band programme spread 0.03 (T), 0.91 (S), 0.77 (V),
  0.25 (G) and kept HF spread 0.71, 0.65, 0.51, 0.17: **not passed on S, V
  and G (all-band) and T, S and V (kept HF)**; see "Rows Whose Check
  Contradicts the Evidence".
- **The two SPL conventions**: both readings place each side at its own
  loud frames (75 dB SPL), so they agree by construction. **Pass.**
- **Raw tape hissier than every render**: programme margin, smallest
  source-minus-render gap per tape: T 9.66 (27 renders, cathar fill 8), S
  9.89 (20), V 6.94 (20, APL fill 8), G 9.22 (21, cathar fill 8; source
  21.78). Pause margin tail: S 3.17, V 2.80, G 4.32; on T every render but
  APL fill 8, which reads 0.44 dB over the raw tape (33.77 against 33.33),
  0.24 past the pause tie (0.20) the specification allowed it. **Pass on
  S, V and G; T misses on fill 8** (an engine finding: the keeper's
  `cluster` target aims every pause at the quiet frames' median minus 8 dB,
  which lands over T's own pause floor; the loud frames match within 0.4
  dB).
- **Session 0** (T raw against the shipped APL, which the user called
  cleaner 3 of 3 blind): programme 25.79 against 8.47 (17.3 dB, over 5);
  pause tail 33.54 against 31.01 (2.5 dB, over 1.5). **Pass.**
- **"cathar alpha 2 has hiss"** (T, round one, not blind): programme
  margin -2.89 (alpha 2.0 on 0.7.3) and -3.27 (beta 0.005) against -6.09
  (0.7.5 alpha 2.0), -7.01 (alpha 2.25) and -9.64 (baseline); pause tail
  +0.32 and -0.09 against -4.22..-4.78. `listener.hiss_pause` fires on
  0.7.3 alpha 2 only; `listener.hiss_programme` fires on none. Against the
  baseline, alpha 2 on 0.7.3 trips H1 on the pause tail (excess 0.32 over
  the 0.20 tie); its beta 0.005 trips nothing. S: programme -5.87 against
  -12.80 and pause tail -0.30 against -3.46 (0.7.3 hissier in both
  places). **The ordering passes, the programme flag does not**; see "Rows
  Whose Check Contradicts the Evidence".
- **"light hiss (single4s, T)"** (round one, not blind): programme margin
  -9.21 against stitched's -9.64 (+0.43; the specification wants +2 and at
  least 0), pause tail -4.36 against -4.78. **Not passed.** Single4s on S
  and V reads -12.06 and -4.73 (under 0): **pass**. Single4s is the duller
  render on T (kept HF -7.88 against -3.86), which the dulling guard
  excludes.
- **"hiss on every v2 plateau"**: both flags fire on all nine speech files
  (T, S, V: `final2_apl`, `final2_cathar`, B; programme +0.72..+14.63,
  pause tails 22.4-31.0) and on G's `final2_cathar` (+11.88, 38.68); G's
  APL final and B fire the pause flag only (programme -0.56 and -0.65).
  **10 of 12**; see "Rows Whose Check Contradicts the Evidence".
- **Dead air excluded** (T round one, tata_v4, incumbent `cathar__baseline`):
  every APL render trips pause depth (over 35) and pump (over 12); the two
  roformer renders also trip the room delta (-118, -119).
  `cathar075__alpha_2_0` and `cathar__alpha_2_25` trip nothing, with the two
  bounds the build added (`gap_island_kurt` 1.85, `balance_top_db` -1.0).
  **Pass.**
- **Less removal not refused** (alpha 2.0 against the baseline): V trips
  nothing; programme excess 0 on both tapes. S trips H1 on the pause tail:
  its excess rises 0.53 dB (margin tail -0.30 against -3.41), within the
  specification's 0.55 but over the measured pause tie (0.20). **Pass on
  V, not on S**; see "Rows Whose Check Contradicts the Evidence".
- **Underwater renders trip the dulling guard** (known excerpts, against
  stitched): the de-esser bug reads kept HF -35.33 / -35.55 / -24.30 against
  -3.86 / -5.52 / -2.38 (T / S / V) and trips on every tape: **pass**.
  Single4s reads -7.88 / -5.46 / -4.34 and trips on T and V, not on S
  (0.06 over stitched).
- **Pause fill ladder** (autotune_v6 APL, incumbent fill 12): pause margin
  tail T 33.77 / 31.01 / 25.04 / 5.16 (fill 8 / 12 / 18 / keeper off), S
  29.94 / 25.89 / 20.14 / 13.11, V 28.77 / 24.81 / 19.71 / 13.80: monotone,
  0.7-1.0 dB per dB of fill. G's tail does not move (39.53, a music tape;
  its median does: 36.56 / 32.42 / 26.53 / -23.43). On T fill 8 trips H1
  and the pause H2, fill 18 trips nothing and wins the pause family, and
  keeper off trips the room delta (G2, -67.5), the pump and the modulation.
  **Pass.**
- **No win by deleting programme or by a pass-through** (T, synthetic from
  B, incumbent the shipped APL): the 4 kHz low-pass trips kept HF (-35.0);
  the raw tape trips H1 (programme excess +17.3 dB, pause +2.3); the gated
  pauses trip the room delta (-120.4) and the pump (+15.0). **Pass.**

## Loop Rows

- **Grid and loop wiring**: both v4 grids parse; `autotune_restoration`
  and `tune_restoration` default to `tata_v4.yaml`; the guards line reads
  "8 learned vetoes, 1 verdict reversals, 16 constraints, audibility tie at
  a 24 dB masking offset" (music_v4: 4, 1, 8); a misspelt constraint
  reading stops the loop; `reward.tie_floors` refuses the main checkout's
  `experiments/reward/noise_floor_benign.json` for both v4 grids, naming
  their first hiss reading. **Pass** (and the unit tests).
- **Tie bucketing**: 8.00 / 8.20 with tie 0.65 rank tied, 8.0 / 9.0 do
  not; entries without a tie rank as before. **Pass** (unit tests).
- **Gates fixed by the user's rule**: `load_gates` with injected entries
  for both flags keeps threshold 0.0, `flag` severity and
  `calibrated=False`; `derive_gates` with `clean["*"]` lists both as
  skipped, "fixed by the user's rule of 2026-10-11". **Pass.**

## Music Coverage per Clip

The programme reading reads Gaudeamus1 on 4 of its 8 windows per render
(the other four hold under two readable bands of 1 kHz and up: dense
singing over a programme that ends at 2.8 kHz) and Gaudeamus5 on all 38.
It abstains on the other eleven clips; unvalidated windows read None,
never a number. The all-band excess orders Gaudeamus1's `final2_cathar`
(+2.83) under `final2_apl` (+10.94). **Pass.**

| clip | anchor | read | pause tail src / APL / cathar |
|---|---|---|---|
| Gaudeamus1 | pauses, valid | 4 of 8 | 43.96 / 41.46 / 39.07 |
| m_vid-20220730-091707 | pauses | 0 of 2 | 10.42 / -1.39 / 0.34 |
| m_vid-20220925-115124 | all frames | 0 of 2 | 16.08 / 6.66 / 15.83 |
| m_vid-20221003-142320 | pauses | 0 of 2 | 9.85 / -2.26 / 3.25 |
| m_videoplayback-2021-07-11-t-1 | all frames | 0 of 2 | none |
| m_wbbddanceswithwolves | all frames | 0 of 2 | none |
| x_the-adventures-of-captain-pu | pauses | 0 of 2 | 6.68 / 0.62 / -3.91 |
| x_the-agatha-christie-hour-in- | all frames | 0 of 2 | none |
| x_vid-20230606-121054 | pauses | 0 of 2 | 26.53 / 11.78 / 21.77 |
| x_vid-20230624-142023 | pauses, valid | 0 of 2 | 20.12 / 5.30 / 17.44 |
| x_videoplayback-20260504-2342 | pauses | 0 of 2 | 7.94 / -6.55 / -0.53 |
| x_y2matecom-opening-and-closin | pauses | 0 of 2 | 39.19 / 36.91 / 40.78 |

Why the programme reading abstains: the anchor is unvalidated on nine
clips; the Agatha Christie clip's programme band ends at 500 Hz and the
y2matecom clip's at 891 Hz; x_vid-20230624-142023 holds one readable band
of 1 kHz and up. On Gaudeamus1 the programme margins read +6.18 (source),
+1.82 (APL) and -7.13 (cathar).

On the last clip cathar's pauses read 1.6 dB over the source's (40.78
against 39.19; v6 cathar 41.27): H2 would exclude it.

## IA Defect Hunt

235 excerpts with both renders, the cathar and APL baselines.

- The programme reading validates on 27 clips. Median programme margin:
  source 21.03, cathar 13.30, APL 6.36; the source is hissier on 26 of 27
  clips against cathar (the exception, das-chaos-duo, reads 0.36 dB over
  its source) and 27 of 27 against APL. **Pass.**
- Pause-margin removal (source tail minus output tail) on the 88 clips with
  true pauses: median +5.20 (cathar), +3.79 (APL). Under -0.55 on 5 clips
  for cathar (doctor-who -5.97, bart-peeters -4.41, toshiba commercial
  -2.30, pingu -1.73, lion-king commercials -1.46) and 14 for APL
  (doctor-who -5.07, wombles -3.12, frank-leaves -3.10, et-el-extraterrestre
  -2.92, impaktita -1.94, how-do-you-want-me -1.87, jadeworld -1.81,
  get-there-quick -1.59 and six more). **Not passed** against "every
  clip": on those clips the render's pauses sit higher re its own loud
  frames than the source's, which a listener at a fixed loud level hears
  as more hiss in the pauses. The cause per clip was not traced.

## Cost

T's first 125 s (v4_air source against B, 17 windows), timed directly on
this machine while eight jitter scorers ran: the two hiss readings take
0.30 s per 15 s window (median; max 0.35) and the anchor 0.78 s per source
(the whole tape), cached afterwards. CPU only. **Pass** (limits 0.35 s per
window and 1 s per anchor), with no headroom under that load.

## Rows Whose Check Contradicts the Evidence

These rows were not gamed and are reported as not passed; the evidence says
the check, not the reading, is wrong:

- **Programme-only change.** The air shelf is not a programme-only change:
  it lifts everything above 4 kHz in the programme stretches, the residual
  hiss included (the r3 record measured A 0.5 dB over C at 4-8 kHz and 1.3
  dB at 8-12 kHz; `linear_air_gate`'s note in `modules/config.py` measured
  the +1 dB shelf lifting the residual 0.3-0.9 dB under the programme). The
  all-band margin and kept HF must move by about that; they move 0.03-0.91
  and 0.17-0.71 dB. The pause reading, whose pauses the keeper refills
  after the shelf, holds within 0.09 dB, as the check wants. The user
  could not tell A from B blind (Session 0, SOTI, 10 of 20), so the move
  sits under audibility, but a hiss reading that ignored a 0.5-1.3 dB lift
  of the hiss would be blind to the hiss.
- **"cathar alpha 2 has hiss" (the programme flag).** The label is round
  one's, not blind. In Session 0 the user could not tell alpha 2 on 0.7.3
  from 0.7.5 blind (ABX 8 of 16, p 0.60, whole sound) nor their pauses (T:
  6 of 7 "same", the seventh picked 0.7.5 as hissier; S: 7 of 7 "same").
  The readings order 0.7.3 hissier in both places (programme +3.2 dB,
  pauses +5 dB over 0.7.5), fire the pause flag on it and, with the
  measured pause tie, H1 excludes it against the baseline; under the
  programme both sit under the masked threshold. The first build fired at
  +2.4 because a clean output's revealed programme read as hiss (the
  saturation above).
- **"hiss on every v2 plateau".** The ledger holds no such verdict: the
  phrase is the harness's own note on the round-two records ("the harness
  flagged listener.hiss on every v2 plateau ... while the listener heard
  natural pauses"), and those records mark both v2 finals clean of
  `listener.hiss` on all four tapes (2026-10-05, not blind). The user's
  rule of 2026-10-11 came later and asks for criticism, not for a flag on
  every file; the readings flag 10 of 12, and the two exceptions (G's APL
  and B under the singing, -0.56 / -0.65) carry the pause flag.
- **Less removal on S.** H1's `max_move` is the measured pause tie (0.20),
  and S's alpha 2.0 raises the pause tail excess 0.53 dB: its hissiest
  pauses cross the threshold at 75 dB where the baseline's do not. The
  preference ("alpha 2 sounds better") predates the user's rule of
  2026-10-11; a guard that let 0.53 dB of new audible hiss through would
  break that rule.
- **Single4s and the dulling guard.** The specification wants single4s
  excluded on every tape; the user flagged it dull on T only, and on S
  ranked it tied first with stitched. The guard excludes it on T (kept HF
  -7.88 against -3.86) and leaves it on S (-5.46 against -5.52); it also
  excludes it on V (-4.34 against -2.38), where the user did not flag it.

## What Did Not Pass

Besides the rows above: the fresh draws' p95 on the pause keys (median
0.99 against 0.35, tail 0.80 against 0.55, both from Gaudeamus5's cathar
final; the speech pairs at most 0.36 / 0.24) and the fixtures' pause
fresh p95 (0.51-0.54); two perfect fixture outputs over 0 (one music
fixture, +0.38 / +4.5); one `lp4k_all` fixture (-0.35); the corpus shape's
validation share (12 of 40); single4s on T (+0.43 over stitched, not +2);
T's fill 8 pauses 0.44 dB over the raw tape (0.24 past the tie); the IA
pause removal (5 cathar and 14 APL clips under -0.55). Everything else
passed, including identity, Session 0, the raw-hissier rows, the dead-air,
cheat, fill and underwater exclusions, music coverage on both Gaudeamus
clips and every calibration check.

## Quality Gate and Docs

Per touched file: black, isort, ruff, flake8, pylint, radon (CC A, MI A),
the unit tests, `mdformat --check` and `pymarkdown`. The full
`run_pipeline_locally.ps1` gate was not run in this build (the task kept it
for the orchestrator).

## What the Readings Say About the Engines

Reported, not built (no restoration code changed in this item):

- Pause hiss is pinned by `modules/pause_floor.py`: it refills the pauses
  with tape hiss at `pause_floor_fill_db` 12, so no denoiser knob moves the
  pause reading and the pause family plateaus at once. Fill 18 is allowed
  and lowers T's pause tail 6.0 dB; keeper off is excluded (G2); fill 8
  puts T's pauses 0.44 dB over the raw tape. The follow-up is a hiss-free
  shaped fill (room tone kept, 3 dB under the critical threshold above
  about 1-1.2 kHz) behind a switch.
- Under the programme the speech v2 plateaus read +0.7..+14.6 dB over the
  masked threshold at 75 dB (cathar over APL on T and S, APL over cathar
  on V); G's cathar final reads +11.9 under the singing, its APL final
  -0.6. The loop will move `cathar_split_band_hz` 4000 with
  `cathar_alpha_high` 2-3, `cathar_beta`, `apl_suppress_gain_floor_db`,
  `apl_sibilant_mix`, `apl_enable_learned_blend` and `linear_air_gain_db`.
- On 5 cathar and 14 APL IA clips the render's pauses sit higher re its
  loud frames than the source's; a refill that follows the render's loud
  frames rather than the source's is the first candidate to test.
