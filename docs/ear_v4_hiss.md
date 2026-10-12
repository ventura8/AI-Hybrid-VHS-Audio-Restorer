# Ear v4: Critical of Hiss

The acceptance record of ear v4 (built 2026-10-12 on `wip/ear-hiss`), the
readings that make the harness very critical of VHS hiss under speech or
singing and in silence and pauses (the user's rule of 2026-10-11). Every
row of the specification's acceptance table is listed with what was
measured, and whether it passed. Nothing here shipped a default: the blind
block (`audibility_check.py`, then `listen_ab` with the device and volume
recorded) and the realistic-v2 plus IA confirmation come first. T, S, V
and G are Tele7abc, SOTI, Vaccin and Gaudeamus5. The readings and the
grids are described in the output-quality-harness skill, "Critical of
hiss (ear v4, 2026-10-12)".

## What ran

- The jitter report: `python -m scripts.reward_noise_floor` on the eight
  pairs of the specification (v4_air `B_air_1dB` and v2 `final2_cathar` of
  T, S, V and G against their sources), `--transforms hiss`,
  `--repeats 2`, `--families dsp` and `--max-seconds 125`, into
  `experiments/reward/noise_floor_hiss.json`.
- The fixtures: `python -m scripts.check_hiss_fixtures --programme linear`
  over de, en, es, fr and it of `artifacts/realistic-v2` (40 gapped m12
  tape-noise programmes), with three noise shapes: `--shape corpus` (the
  fixtures' recorded noise), the hiss shape and `--shape-from` T's raw tape
  (`artifacts/hiss-v1/shape_tele7abc.npy`, gitignored, never committed).
  Reports: `experiments/hiss_v4/fixtures_<shape>.jsonl`.
- The calibration: `calibrate_quality_metrics` with `--languages en,fr`
  and `--metrics dsp` before the change (the base tree, bb5fa2a) and after
  it, into the scratch folder.
- The tapes: `validate_restoration --metrics dsp` (windows 15 s, hop 7.5 s)
  on the stored renders per tape, into `experiments/hiss_v4/<set>_<tape>.json`
  (gitignored): `r1` (the round-one cache of `experiments/tata_listen`),
  `v2` (the v2 finals and the identity), `v4air` (v4_air A, B, C, Session
  0's shipped APL and three synthetic cheats built from B), `fill` (the
  autotune_v6 APL candidates with `pause_floor_fill_db` 8, 12, 18 and the
  keeper off), `known` (the known-ordering excerpts) and `music` (the 12
  autotune_v6 music clips and Gaudeamus1).
- The IA defect hunt: the cathar and APL baselines of
  `experiments/tune_ia_hunt_v3` against their 238 sources (`python -I`).
- The family tapes never left the machine; nothing was uploaded and no web
  service saw their audio. CPU only, no GPU, no learned model.

## Same-percept rows (the jitter report)

`experiments/reward/noise_floor_hiss.json`: eight pairs, nine scorings each
(the reference and the eight transforms), two repeats, 136 deviations per
pause key and 68 per programme key (the programme reading validates on T
and S of the first 125 s, not on V or G). The DSP repeat jitter is exactly
0 on every dsp and file reading. Per transform (the report's largest
deviation, `by_transform`; the per-transform p95 from re-scoring the same
cached near-copies with the hiss readings alone, which the report now also
writes as `percentile_by_transform`):

- Gain +-0.1 dB: largest deviation 0.000 on every margin and excess key,
  median and tail. **Pass.**
- Delay 0.37 samples: p95 0.009 (programme margin median), 0.029
  (all-band), 0.003 (pause margin); largest 0.124 (programme margin tail).
  **Pass** (limits 0.05, 0.02 and 0.30).
- soxr round trip: 0.000 everywhere. Requantisation: p95 at most 0.032,
  largest 0.037 (programme margin delta). **Pass** (limits 0.05 / 0.10 and
  0.10 / 0.30).
- Fresh realisation p95: programme excess median 0.312 (limit 0.65),
  all-band 0.299 (0.45), pause excess tail 0.275 (0.55): **pass**; pause
  excess median 1.780 (0.35): **fail**, all of it from Gaudeamus5's B, a
  music tape (1.6-2.1 dB per realisation on that pair alone; 0.47 without
  it). On the fixtures the fresh case
  reads more (programme p95 1.02-1.60, above), and so does the calibration's
  `fresh_noise_linear` (-1.49 / -1.52 dB on the programme margin).
- The ties: each grid `tie` is the report's pooled p95 floor for its flat
  key rounded up to 0.05. Programme excess median 0.202 -> **0.25**,
  all-band 0.196 -> **0.20**, pause excess tail 0.244 -> **0.25**, median
  0.473 -> **0.50**. H1's `max_move` takes the same values; H2's bound takes
  its delta key's own floor: programme margin delta median 0.109 -> **0.15**,
  pause margin delta tail 0.321 -> **0.35**, music's all-band delta 0.334 ->
  **0.35**. **Pass.**
- Found on the way: requantisation moves `hiss_prog_kept_hf_db` on Vaccin
  from -3.73 to -0.79 (B) and -3.12 to -1.45 (cathar) dB: V's high band sits
  near the 16-bit floor, and the dither lands in the cells the guard reads.
  The guard only trips downward, so dither can hide dulling but never
  exclude a render; candidates of one loop share a format. A dither limit
  on those cells is a follow-up.

## Fixture rows (`check_hiss_fixtures`)

Validated fixtures: 12 of 40 on the corpus shape, 20 of 40 on the hiss
shape, 23 of 40 on T's shape. The corpus noise is rumble-heavy (its anchor
validates on 12 fixtures and the hiss cap reads it 7-19 dB low elsewhere),
so T's shape is the representative one.

- Gain +-0.1 dB: p95 and max 0.000 on every hiss key, all three shapes.
  **Pass.**
- Delay 0.37 samples: programme margin p95 0.008-0.012 dB (max 0.017),
  all-band p95 up to 0.033, pause margin p95 0.001-0.002. **Pass.**
- soxr round trip: programme p95 0.007-0.015 (max 0.022), pause 0.000-0.002.
  Requantisation: programme p95 0.019-0.042 (max 0.058), pause up to 0.009
  (max 0.017). **Pass.**
- Fresh realisation: programme margin p95 1.02 (T's shape), 1.25 (hiss),
  1.60 (corpus); pause margin p95 0.48, 0.57 and 1.95. Over the
  specification's 0.65 and 0.55 on the fixtures: the m12 programmes carry
  more noise-led cells than a tape, and one fresh draw moves them all.
- Oracle ladder monotone (identity, -6, -12, -20, -30 dB, prog_only, +0.05
  dB slack): 22 of 23 (T's shape), 20 of 20 (hiss), 6 of 12 (corpus).
  **Pass on T's and the hiss shape** (spec: at least 50 % validated, every
  validated fixture monotone: 22 of 23 misses by one).
- The perfect output (prog_only) at or under 0 dB: 0 of 23 (T's shape,
  median +14.8, range +9.2..+22.4), 0 of 20 (hiss, median +20.2), 2 of 9
  (corpus). **Fail.** The ladder saturates 13-14 dB under the source: past
  that, the programme's own leak into the noise-dominated cells, not hiss,
  sets the reading. The prototype's validation passed because its shaped
  noise was 30.1 dB quieter than its spectrum (no FRAME / 2 scaling in its
  `shaped_noise`; 18 dB in the loud variant): its hiss sat 30 dB under the
  tape's relation, so the saturation never showed. At the tape relation the
  anchor error is -0.1 dB median and the pause ladder is exact (6 / 12 / 20
  / 30 dB).
- Physical removal (`hiss_prog_removed_hf_db` - g at 6 and 12 dB): -0.009 /
  -0.031 (T's shape), -0.004 / -0.011 (hiss), -0.017 / -0.032 (corpus).
  **Pass.**
- hf_left (noise above 4 kHz kept, the rest taken 20 dB down): |median
  delta| 0.054 (T's shape), 0.117 (hiss), 0.126 (corpus). **Pass.**
- gated_m0: margin within +0.5 dB of identity on 23 of 23 and 20 of 20;
  pump at least 15 dB on 23 of 23 (+20.4..+25.4) and 20 of 20. Corpus: 7 of
  12, pump read on one fixture (+9.6). **Pass on T's and the hiss shape.**
- islands_15: island kurtosis +0.63..+0.73 (T's shape), +0.66..+0.72
  (hiss); corpus median +0.38. **Pass on T's and the hiss shape.**
- lp4k_all: kept HF at or under -1 dB on 14 of 23 (T's shape, median
  -1.35), 10 of 20 (hiss, median -0.99), 8 of 9 (corpus). **Fail** against
  "100 %": a 4 kHz low-pass of a programme already low-passed at 4.5 kHz
  leaves little programme above 3 kHz standing 18 dB over the floor. On the
  tapes the guard catches every dulling render (below).

## Calibration rows (en, fr, dsp)

- Before: 108 of 165 checks pass, 3 non-blind failures (`gated_pauses`
  pause pumping, `pause_residual_islands` twice). After: 132 of 196 pass,
  the same 3 plus `hiss_islands`.
- Every pre-existing check status is unchanged (165 of 165); every non-hiss
  benign floor is unchanged (51 of 51). Only the `dsp.hiss_*` floors come
  from the `*_linear` benign cases (`benign_only`). **Pass.**
- The new degradations, non-blind checks:
  - `hiss_under_programme`: programme margin down (-4.79 / -9.11 / -12.86,
    3.2 floors at the mild level), pause margin flat (moves 0.008-0.011),
    pump down, kept HF flat, tilt flat. **Pass.**
  - `hiss_in_pauses_only`: pause margin down (-5.99 / -11.99 / -19.99),
    programme margin steady (span 0.10), `gap_atten_db` up. **Pass.**
  - `hiss_hf_tilt`: both margins down, kept HF flat. **Pass.**
  - `low_floor_kept`: pause margin steady (span 0.00005) and below the
    centre (-29.99), programme margin steady (span 0.0003). **Pass.**
  - `dead_pauses`: pause margin, room delta and pump move as specified,
    programme margin steady. **Pass.**
  - `hiss_islands`: island kurtosis up and monotone (+0.50 / +0.77 /
    +1.12) but the mild level reads 2.25 floors, under the rule's 3 (floor
    0.224, from `fresh_noise_linear`). **Fail.**
  - `hiss_removed_dulled`: kept HF down (-7.8 / -45.6 / -54.1), programme
    margin below (-24.6..-25.6 against the edge -4.5). **Pass.**
- The benign `fresh_noise_linear` moves the programme margin -1.49 / -1.52
  dB (en / fr) and the pause margin -0.009.

## Tape rows

- **Identity** (source against itself, four tapes): every `dsp.hiss_*`
  delta exactly 0. **Pass.**
- **Programme-only change** (v4_air A / B / C): pause margin tail spread
  0.003 (T), 0.008 (S), 0.001 (V), 0.109 (G); all-band programme spread
  0.065 (T), 0.209 (S), 0.261 (V), unread on G; kept HF spread 0.252 (T),
  0.31 (S), 1.257 (V). **Pass on T; G misses the pause spread by 0.009; S
  and V exceed the all-band and kept-HF spreads** (the air shelf is a
  programme change in the very band kept HF reads).
- **The two SPL conventions** (|loud-frame gain - programme gain|, v2
  finals and B): 0.16 / 0.75 / 0.16 (T apl, cathar, B), 0.78 / 0.18 / 0.78
  (S), 0.88 / 0.47 / 0.88 (V), 0.03 / 0.46 / 0.03 (G). Within 0.3 dB on 5
  of 12. **Fail.** The least-squares alignment matches the waveform, not
  the loud frames, and a render that compresses its loud frames moves the
  two apart; each reading keeps its own convention.
- **Raw tape hissier than every render**: source programme margin T 26.74
  (smallest source-minus-render gap 5.36, fill 8), S 21.71 (7.53, fill 8), V
  19.96 (1.70, fill 8); pause margin tail T 33.2-33.7 (gap 0.24, fill 8, within
  the tie of 0.25, as the specification foresaw), S 32.93 (3.22), V 31.9-32.0
  (3.43), G 43.3-44.6 (2.77, round one's final APL). G's programme is unread
  outside v2 (no window validates). **Pass** on 72 renders.
- **Session 0** (T raw against the shipped APL): programme 26.74 against
  18.93 (7.81 dB, over 5); pause tail 33.67 against 31.02 (2.65, over 1.5).
  **Pass.**
- **"cathar alpha 2 has hiss"** (T, round one): programme margin +2.79
  (alpha 2.0, 0.7.3) and +2.52 (beta 0.005) against -0.63 (cathar075 alpha
  2.0), -0.54 (alpha 2.25) and -2.94 (baseline); `listener.hiss_programme`
  fires on both flagged renders and on none of the cleared ones. Pause tail
  +0.93 (0.7.3) against -4.35 (0.7.5): `listener.hiss_pause` fires on 0.7.3
  only (onset tails 74.07 against 79.35). S: programme -2.62 against -8.30,
  pause onset tail 74.90 against 78.59 (0.7.3 hissier on both). **Pass.**
- **"light hiss (single4s, T)"**: programme margin -2.76 against stitched's
  -2.94 (+0.18; the specification wants +2 dB and at least 0). **Fail.**
  Pause onset tail 79.29 against 79.61 (tie). Single4s on S and V reads
  -9.24 and -4.44 (under 0). **Pass.** The prototype's probe read +0.01
  against -3.01; this build reads the two 0.18 dB apart. Single4s is the
  duller render (kept HF -8.76 against -2.64), which the dulling guard
  catches (below).
- **"hiss on every v2 plateau"**: both flags fire on all nine speech files
  (T, S, V: `final2_apl`, `final2_cathar`, B; programme +11.0..+20.0, onset
  tails 43.9-52.5). On G (music) only `listener.hiss_pause` fires (onset
  tails 34.4-35.9); the programme reading abstains. **Pass on 9 of 12;**
  G's programme is unread.
- **Dead air excluded** (T round one, tata_v4, incumbent `cathar__baseline`):
  every APL render trips pause depth (36.89-40.31, over 35) and pump (15.4-95.7,
  over 12); the two roformer renders also trip the room delta (-118, -119).
  `cathar075__alpha_2_0` and `cathar__alpha_2_25` trip nothing. **Pass**,
  with two bounds this build added to the specified constraints
  (`gap_island_kurt` 1.85, tata_v3's dead-zone edge; `balance_top_db` -1.0,
  the dull flag's edge): the move alone excluded both cleared renders
  (cathar075 alpha 2 reads gap islands 1.734 and top -0.963 against the
  baseline's 0.959 and -0.231).
- **Less removal not refused** (alpha 2.0 against the baseline): V trips
  nothing. S trips the pause tail H1: the pause tail excess rises 0.638 dB
  (0 to 0.638), over the tie of 0.25 and over the row's own 0.55; the
  programme excess reads 0 on both tapes. **Fail on S, pass on V.** S's
  alpha 2.0 removes less in the pauses (pause margin tail +0.10 against the
  baseline's -3.39), just over audibility, and H1 reads that as new hiss;
  the user's preference was heard on the whole programme.
- **Underwater renders trip the dulling guard** (known excerpts, against
  stitched): de-esser bug kept HF -37.07 / -45.73 / -48.28 against -2.64 /
  -6.32 / -18.11 (T / S / V); single4s -8.76 / -7.62 / -22.10. Each trips
  `hiss_prog_kept_hf_db`. **Pass.**
- **Pause fill ladder** (autotune_v6 APL, incumbent fill 12): pause margin
  tail T 33.43 / 31.03 / 24.68 / 5.67 (fill 8 / 12 / 18 / keeper off),
  onset tails 41.57 / 43.97 / 50.32 / 69.33, monotone at about 1 dB per dB
  of fill; S 29.71 / 25.77 / 20.04 / 13.15; V 28.52 / 24.82 / 19.84 /
  15.60; G does not move (40.5, a music tape). On T fill 8 trips H1 (both
  places) and the pause H2, fill 18 trips nothing, and keeper off trips the
  room delta (G2, -67.5), the pump and the modulation. **Pass.** On S fill
  18 trips `gap_mod_dist_db` (+1.2 is its limit).
- **No win by deleting programme or by a pass-through** (T, synthetic from
  B, incumbent the shipped APL): the 4 kHz low-pass trips kept HF (-23.0),
  pump, modulation, top and the 's'; the raw tape trips H1 (programme
  excess +7.8 dB, pause +2.2); the gated pauses trip the room delta (-120)
  and the pump (+17.1). **Pass.**

## Loop rows

- **Grid and loop wiring**: both v4 grids parse; `autotune_restoration`
  and `tune_restoration` default to `tata_v4.yaml`; the guards line reads
  "8 learned vetoes, 1 verdict reversals, 16 constraints, audibility tie at
  a 24 dB masking offset" (music_v4: 4, 1, 8); a misspelt constraint
  reading stops the loop; `reward.tie_floors` refuses the main checkout's
  `experiments/reward/noise_floor_benign.json` (read-only) for both v4
  grids, naming their first hiss reading. **Pass** (and the unit tests).
- **Tie bucketing**: 8.00 / 8.20 with tie 0.65 rank tied, 8.0 / 9.0 do
  not; entries without a tie rank as before (`test_tune_restoration_v3.py`
  unchanged). **Pass** (unit tests).
- **Gates fixed by the user's rule**: `load_gates` on the main checkout's
  `experiments/quality_calibration/gates.json` (read-only) with injected
  entries for both flags keeps threshold 0.0, `flag` severity and
  `calibrated=False`; `derive_gates` with `clean["*"]` lists both as
  skipped, "fixed by the user's rule of 2026-10-11". **Pass.**

## Music coverage per clip

The programme reading validated **no window on any of the 13 clips**
(music_v4 abstains under the programme on all of them; unvalidated windows
read None, never a number). The pause reading reads where pauses exist.
**Fail** against "Gaudeamus1 validates": its anchor validates (4.76 s of
pauses) but every window fails the window guard.

| clip | anchor | windows | why the programme reading abstains |
|---|---|---|---|
| Gaudeamus1 | pauses, valid | 0 of 8 | gain -8..-11 dB (2), corr < 0.5 (6) |
| m_vid-20220730-091707 | pauses | 0 of 2 | anchor unvalidated |
| m_vid-20220925-115124 | all frames | 0 of 2 | anchor unvalidated |
| m_vid-20221003-142320 | pauses | 0 of 2 | anchor unvalidated |
| m_videoplayback-2021-07-11-t-1 | all frames | 0 of 2 | anchor unvalidated |
| m_wbbddanceswithwolves | all frames | 0 of 2 | anchor; corr < 0.5 |
| x_the-adventures-of-captain-pu | pauses | 0 of 2 | anchor; corr < 0.5 (2) |
| x_the-agatha-christie-hour-in- | all frames | 0 of 2 | band 500 Hz, no band |
| x_vid-20230606-121054 | pauses | 0 of 2 | anchor; corr < 0.5 (APL) |
| x_vid-20230624-142023 | pauses, valid | 0 of 2 | one readable band of 1 kHz+ |
| x_videoplayback-20260504-2342 | pauses | 0 of 2 | anchor; no readable band |
| x_y2matecom-opening-and-closin | pauses | 0 of 2 | band 891 Hz |

On Gaudeamus1 the source's right channel is dead (50 dB down) and the
renders correlate 0.39-0.60 with the source; the least-squares alignment
then shrinks the output by the correlation, so the programme gain reads
-8..-11 dB, past the guard's 6 dB. The pause tail excess still orders
`final2_cathar` (39.29) under `final2_apl` (41.63).

## IA defect hunt

238 excerpts, the cathar and APL baselines. The programme reading validates
on 26 clips (cathar 47 windows, APL 50).

- Median programme margin on the validated clips: source 22.58, cathar
  17.40, APL 14.57; the source is hissier on 23 of 26 clips against cathar
  and 25 of 26 against APL. **Pass.** The exceptions read within 0.64 dB:
  hey-abbott (-0.64), das-chaos-duo (-0.60), yorkshire-television (-0.13)
  for cathar; spider-im-only-scary (-0.63) for APL.
- Pause-margin removal (source tail minus output tail) on the 88 clips with
  true pauses: median +4.98 (cathar), +3.77 (APL). Under -0.55 on 6 clips
  for cathar (doctor-who -8.0, bart-peeters -4.4, superstition -1.8, pingu
  -1.8, toshiba commercial -1.7, lion-king commercials -1.4) and 15 for APL
  (making-of-snow-white -8.5, doctor-who -5.0, wombles -3.6, frank-leaves
  -3.1, et-el-extraterrestre -2.7, and 10 more down to -0.6). Within the
  tie, under 0: 2 (cathar), 9 (APL; channel-4, potsworth, dragon-ball and
  barker among them, as the specification foresaw). **Fail** against
  "every clip": on those clips the render's pauses sit higher re its own
  loud frames than the source's, which a listener at a fixed loud level
  hears as more hiss in the pauses. The cause per clip (a compressed loud
  level, a refilled pause floor) was not traced.

## Cost

The dsp family on T's first 125 s (v4_air source against B, 17 windows),
on this machine while other scorers ran (`cost_check.py` in the scratch
folder): 21.79 s before the change (base tree bb5fa2a; 20.71 s with the
profile cached) and 25.01 s after it (21.94 s with the anchor cached).
Timed directly, the two hiss readings take 0.18-0.21 s per 15 s window
(max 0.24) and the anchor 0.43-0.50 s per source, cached afterwards. CPU
only. **Pass** (limits 0.35 s per window and 1 s per anchor).

## Quality gate and docs

Per touched file: black, isort, ruff, flake8, pylint, radon (CC A, MI A),
the unit tests, `mdformat --check` and `pymarkdown`. The full
`run_pipeline_locally.ps1` gate was not run in this build (the task kept it
for the orchestrator); see the build's report.

## What the readings say about the engines

Reported, not built (no restoration code changed in this item):

- Pause hiss is pinned by `modules/pause_floor.py`: it refills the pauses
  with tape hiss at `pause_floor_fill_db` 12, so no denoiser knob moves the
  pause reading and the pause family plateaus at once. Fill 18 is allowed
  and lowers the pause margin 6.4 dB on T; keeper off is excluded (G2).
  The follow-up is a hiss-free shaped fill (room tone kept, 3 dB under the
  critical threshold above about 1-1.2 kHz) behind a switch.
- Under the programme every v2 plateau render is 11-20 dB over the masked
  threshold at 75 dB. The loop will move `cathar_split_band_hz` 4000 with
  `cathar_alpha_high` 2-3, `cathar_beta`, `apl_suppress_gain_floor_db`,
  `apl_sibilant_mix`, `apl_enable_learned_blend` and `linear_air_gain_db`.
- The programme reading saturates 13-14 dB under the source, so a render
  that removes more than that reads the same; the excess still separates
  round one's labelled renders and every plateau from its raw tape.
