# Ear v3 Round 0

Round 0 (plan 1.6) re-scored every file the verdict ledger judged with the
ear v3 readings on 2026-10-09. When it ran nothing was adopted. After
Session 0, the same day, the grid notes below went into `tata_v3.yaml` and
`music_v3.yaml` (see "What the grids took"); no gate, flag or reversal
changed, and the proposed flags stay proposals. T, S, V and G are Tele7abc,
SOTI, Vaccin and Gaudeamus5; A, B and C are round three's air shelf at
+2 dB, +1 dB (shipped) and off.

## What was scored and how

`experiments/tata_listen/score_round0.py` (new; untracked, `experiments/` is
gitignored) scores the ledger's files per set and tape with
`scripts/validate_restoration.py` (dsp family, the v2 `gates.json`) into
`experiments/tata_listen/scores_v3/<set>/<tape>.json`: `r1_known`, the
known-ordering excerpts (13 outputs); `r1_listen`, the round-one full-tape
variants (17); `r2`, the v2 plateaus on four speech tapes and 12 music
clips; `r3`, `variants/v4_air` on T, S, V, G. `check_verdicts.py --scores`
on each set, under the v2 flags, reproduces every checked verdict on
`r1_known` (3 records), `r1_listen` (3) and `r3` (4, trivially: a
preference check counts hard gates). On `r2` 9 checks fail (exit 1): v2
`sibilance_thin` misses "APL's 's' still thin" on `final2_apl` of T, S and
G, and v2 `dead_air`, `dull`, `attack` and `sibilance_thin` fire on six
music outputs the user called fine.

## Must-reproduce (plan 1.6)

- **+2 dB air bright, +1 dB clean**: **no** on the shipped `listener.bright`
  (R1 `balance_top_db` over +1.0; A reads -0.064 / +0.002 / +0.005 / -0.089
  on T/S/V/G); **yes** on the paired `hf_8k16k` (delta minus B over +0.345)
  as a shelf check: A-B +0.689 (T, derived) / +0.703 / +0.671 / +0.650, C-B
  -0.695 / -0.690 / -0.668 / -0.698, so B over A and C holds.
- **+2 dB thin_abs**: **partly** (R2 `sib_abs_level_db` minus B over
  0.1398). T +0.280 (derived); V +0.307 fires by 0.167 on 5 windows; S
  misses on the medians (+0.134, by 0.006), fires window-paired (+0.1405, by
  0.0007); G is unread (0 of 40 windows).
- **de-esser bug dull** (`balance_presence_db` at -1.5 or under): **yes**,
  -4.67 / -9.71 / -6.35 (T/S/V), margins 3.17 / 8.21 / 4.85.
- **single4s "underwater" on T only**: **yes on the speech-route statistic,
  thin margin**: presence T -2.039 against V -1.810 (V all-route -2.108),
  `balance_top_db` -2.039 / -1.860. An edge near -1.92 separates them by
  0.115 / 0.114, keeps T known `cathar075` (-1.06) and S single4s (+0.34)
  clean and fires on every de-esser bug; it rests on one record. At -1.5, V
  single4s (ranked first) fires. R2 texture: +3.49 / +0.37 (V, 5 windows).
- **alpha 2 hissy on 0.7.3, clean on 0.7.5** (T, R4 `gap_hf_excess_db`):
  **yes, T only**: +3.11 / +0.49, window-paired +2.29 on 13 of 13 windows;
  attenuation (42.31 / 42.84 dB) does not separate them.
- **The hiss rule off T** (HF excess over +1.80 where `gap_atten_db` is at
  least 32, speech route): **inconclusive**. S alpha 2 (+7.51, 16 of 18
  gated windows) is preferred only by assumption: "alpha 2 sounds better"
  names no binary (`context.assumed`) and may mean `cathar075__alpha_2_0`,
  which has no R4 in any report. S known `cathar075` (+3.05) is a weak
  clean, S `cathar__baseline` (+1.73) clean by 0.07, V clean. It misses T
  single4s "light hiss" (-0.27).
- **Round-one APL "silent in pauses"** (speech-route `pause_depth_db` delta
  at least 35.0): **yes**: `apl__baseline` +37.22, roformer +40.31; T
  cathar 29.01-32.98, S and V cathar 27.28-32.38, r2 and r3 3.48-13.80.
- **v3 sibilance final equals v2**: **yes**, audible share 0..0.0015.
  **"Distortion of spoken 's'"** (`sib_texture_db` at least 1.75): **yes,
  T only**: flagged APL +2.18..+2.26, cleared cathar +0.83..+1.34.
- **r2 "pauses natural", "'s' still thin", "music fine"** under the v2
  flags: **no** (below). **r3 leaves the pauses alone**: **yes** (0.07 dB).

## Proposed v3 flags

Anchor: the same tape at the accepted setting through the same neural stage
(today r3's B). One unblinded listener; r2's "thin" file is r3's A. Two are
live candidates, five display-only until Session 0 or a second tape agrees.

- `bright`, candidate, **2 verdict groups** (`r2-pauses-and-s`,
  `r3-air-preference`): `hf_8k16k` delta minus the anchor's over +0.345
  (half T's A-B); abstains without an anchor or when `gap_hf_excess_db`
  moved over 1.0 dB; needs a paired gate. A fires on S / V / G by 0.358 /
  0.326 / 0.305 (on G, A over B in 38 of 38 windows, two inside the dead
  zone): a check of the shelf filter, not the percept or the edge.
  `final2_cathar` abstains on all four (guard deltas 1.57 / 2.79 / 1.75 /
  6.67 dB), so no second engine was scored; the guard was fitted on
  held-out G (on T any value from 0.03 to 1.57 acts alike).
- `dead_air`, candidate, **1 record**, replacing v2 `dead_air` and
  `pause_collapse`: speech-route `pause_depth_db` delta at least 35.0,
  midway between T `cathar__baseline` (32.98) and `apl__no_expander`
  (36.89), whose flag the ledger inferred; on the files the user named the
  gap is 32.98-37.22 (midpoint 35.10). In it: T single4s 33.42, S single4s
  33.09 (ranked first, clean by 1.91), S `deesser_bug` 35.63 (fires, 0.63).
- `thin_abs`, display, **2 verdict groups**, 3 tapes: `sib_abs_level_db`
  delta minus the anchor's over +0.14, abstaining under 5 windows, where its
  one held-out fire (V) sits. Pooled absolute fails (S A +0.273 under V B
  +0.319); scaling per tape by (A-C)/2 pools by construction.
- `dull`, display, **3 records, edge on 1**: `balance_presence_db` at -1.5
  or under, midway between T `cathar075` (-1.06) and T single4s (-2.04). It
  fires on V single4s; smallest unflagged margins 0.27 (V `apl__baseline`)
  and 0.44 (T `cathar075`). Music: 20 of 24 outputs read -0.61..+0.41, 4 none.
- `pause_gated`, display, **1 record**: `gap_atten_db` at least 54 (R4
  floors at 60), shape readings None there. Untested off T; `r1_known`, r2
  and r3 read 3.32-42.88 (43.11 speech-route).
- `pause_hiss`, display, **1 record**: the hiss rule, speech route only (on
  speech plus mixed, accepted music `x_vid-20230606` `final2_apl` fires on a
  33.02 dB window); the deepest r2 speech-tape window is 29.51 dB, r3's 29.74.
- `s_texture`, display, **1 record**: `sib_texture_db` at least 1.75; also
  fires on T known `apl` (+2.257, ranked first), three APL renders with no
  's' verdict (+2.12..+2.26) and T single4s (+3.49).

## Grid target notes (`tata_v3.yaml`)

These reverse a verdict or charge accepted files; do not adopt them:

- `gap_atten_db` 14.5 / 11.3: dead APL (34.95-38.45) costs 9.2-12.7, the
  cleared cathar (42.31-42.84) 16.5-17.0: it reverses the round-one
  dead-air verdict, as today's 8.7 / 5.4 does (20.9-24.4 against 28.2-28.7).
- `balance_air_db` +0.17 / 0.25 (one Vaccin file): V's preferred alpha 2
  (-0.875) is out by 0.795, the less preferred `apl__baseline` (+0.963) by
  0.543, reversing the round-one Vaccin preference.
- `pause_depth_db` +5.0 / 9.0: T's renders cleared of dead air cost
  15.0-19.0, preferred alpha 2 13.3 / 16.2 (S / V), the dead APL 22.9-26.3.
- `gap_hf_excess_db` gated 0.49 / 1.31: its low side charges T and V known
  `cathar075` (-2.355 / -5.165) and S `apl__baseline` (-6.09).
- `gain_ride_lu` 2.1 leaves out V known single4s (ranked first, 2.257): it
  needs at least 2.26. Presence 1.1, tilt 1.15 and body 1.4 admit accepted
  files by 0.04, 0.028 and 0.007; V known single4s sits out of all three.
- `sib_abs_level_db` +0.25 / 0.14 ties S's A with B; drop `sib_centroid_hz`
  (ranks C over B on V). `rejected_below: 0.0` for `linear_air_gain_db`
  rests on one unblinded verdict group: Session 0's B-against-C pair first.

## What the grids took (2026-10-09, after Session 0)

Each grid comment carries the numbers. Every value is still an unblinded
hypothesis (Session 0 told no two restorations apart). This list is the grid
as taken on the pre-fix readings; the calibration v3 re-score changed five
of its values (next section).

- `tata_v3.yaml`, 12 entries (from 15): `balance_air_db` and
  `sib_centroid_hz` dropped, `gap_atten_db` and `gap_hf_excess_db` shown,
  not ranked. Pause depth is ranked as a band of 2.95-35.0 dB: the high
  edge is `dead_air`'s 35.0, the low edge one scale unit under the
  shallowest accepted speech file (S r3 C, 3.990), so a pass-through scores
  2.95 instead of tying every accepted file. Tilt 2.25, presence 2.25, body
  2.35, `sib_abs_level_db` 3.7 and `gain_ride_lu` 2.65 admit every accepted
  file by at least one scale unit; texture keeps 1.75. The slope's 2.1
  honours Session 0's blind "same" on the 0.7.3 / 0.7.5 pair and still
  charges V known single4s (ranked first) and S / V alpha 2 (preferred);
  pause depth charges V's ranked-first files on its mixed-route aggregate.
- Every round-two and round-three file scores 0, so their preferences tie
  rather than reproduce; round one's are charged where the comments say,
  and T known `apl` (ranked first) now scores above `cathar075` (second):
  33.40 against 14.31.
- `music_v3.yaml`: presence 0.75 (all 24 "music fine" outputs inside); its
  other entries still charge 16 of the 24.
- The sibilance family ties every A1 / A2 candidate (r3's A-to-C span sits
  inside 3.7), so those rounds rest on the paired `thin_abs` check.

## The grid after calibration v3 (2026-10-10)

Round 0 was re-scored with the readings calibration v3 fixed (commit
1d644fb) into `experiments/tata_listen/scores_v3`; the pre-fix reports are
kept in `experiments/tata_listen/scores_v3_precal`. On the 50 speech-tape
files the grid reads, R1 (tilt, presence, body), pause depth and the gain
ride read exactly as before. R2 moved on every file it reads: the gain is
matched around each 's', and the texture needs 30 fricative frames a
window, so V goes unread (G already was). R4's shape readings moved on V
only (the dither floor). `check_verdicts.py --scores` prints the same as on
the pre-fix reports: `r1_known`, `r1_listen` and `r3` reproduce every
checked verdict (3, 3 and 4 records, exit 0); `r2` fails the same 9 checks
(exit 1).

An entry was re-derived where a file its value was set by moved, or where
its old value broke the rule on the new readings:

- `sib_abs_level_db` 3.7 to 3.4: the edge was V known single4s (+3.612),
  which now reads +2.176; T known `cathar075` (-3.306, ranked second) sets
  it, admitted by 0.094. The old value charged no accepted file.
- `sib_texture_db` 1.75 to 2.0, the midpoint of the accepted top (T
  `cathar075__alpha_2_0`, ranked first, +1.652) and the APL bottom (T
  roformer, +2.375), to the nearest 0.05. At 1.75 the ranked-first file sat
  inside by 0.098 and the cleared baseline (+1.601) by 0.149, under one
  scale unit (0.2). The APL renders are charged by 0.375-0.420.
- `gap_spread_db` 3.0 to 6.85 and `gap_lsd_db` 3.5 to 9.05: the starting
  values rested on V `final2_cathar`'s pre-fix 2.742 / 3.492 (now 1.089 /
  1.841). On the new readings they reversed a verdict Round 0 kept: V's
  preferred alpha 2 summed 19.52 against `apl__baseline`'s 11.94 (pre-fix
  26.91 against 37.56), because `apl__baseline`'s spread fell from 7.255 to
  2.877 and its LSD from 8.716 to 4.183 while alpha 2 stayed out (6.301 /
  7.569). They also charged V's ranked-first known single4s and stitched,
  T's ranked-first `cathar075__alpha_2_0` (LSD) and S's preferred alpha 2.
  By the rule, spread admits V alpha 2 by 0.549 and charges no judged file.
  LSD cannot charge T known `apl` (8.610, ranked first, the APL build round
  one flagged dead air) without charging V known `cathar075` (8.280, ranked
  second), so it admits both by a scale unit: T `apl` by 0.440, `cathar075`
  by 0.770. It charges only flagged files: T beta (hiss, 9.163, by 0.113),
  T `apl__no_air` (dead air, 9.248, by 0.198) and the de-esser bugs. The
  first pass of this re-derivation set 8.75, which admitted T `apl` by
  0.140, under one unit; no ledger pair changes state between the two.
- Island kurtosis 1.8 to 1.85: Round 0 had kept 1.8 under the one-unit
  margin (T `cathar__alpha_2_25`, 1.736, by 0.064; the ranked-first
  `cathar075__alpha_2_0`, 1.734, by 0.066); 1.85 admits them by 0.114 /
  0.116. The largest judged reading, the 0.7.3 hiss render's 1.754, sits
  inside both, so neither value charges a judged file.
- Kept: the slope 2.1 (set by T's 0.7.3 render, +1.913; V's ranked-first and
  preferred files moved inside, so besides T known `apl`, ranked first, -4.410,
  by 2.310, the dead APL build, S alpha 2, +3.012, is the one accepted file it
  charges) and modulation 10.0 (V moved by 0.124 at most; it still charges the
  three known `cathar075`, ranked second, by 2.59-3.07). Modulation is the one
  known margin exception: it admits eight accepted files by under one scale
  unit, 1.2 (S known stitched and S `cathar__baseline` 9.866 by 0.134, T
  `cathar075__alpha_2_0` 9.863 by 0.137, S alpha 2 9.585 by 0.415, T
  `cathar__alpha_2_25` 9.489 by 0.511, S known single4s 9.229 by 0.771, T
  `cathar__baseline` and T known stitched 9.097 by 0.903). The rule would put it
  at 14.3, but that drops S known `cathar075`'s 2.236, the one charge that
  orders S's known ranking, which would then tie.

The whole grid on the new reports, summed as the yaml quotes:

- Accepted files charged: by pause depth, V known single4s and stitched and
  V's round-one renders (mixed-route aggregate) and S `apl__baseline` (in
  the dead-air range); by the slope, S alpha 2; by modulation, the three
  known `cathar075`; by texture, pause depth and the slope, T known `apl`
  (ranked first, the APL build round one flagged).
- 15 reversed pairs, against 20 under the old values on either report set. T
  known `apl` (21.11) above `cathar075` (2.56), stitched (0) and single4s
  (12.49). V known single4s (3.00) and stitched (5.33, pause depth) above
  `cathar075` (2.16, modulation), a verdict this re-derivation newly reverses:
  Round 0's grid on the pre-fix reports read 17.76 / 22.21 against 24.68, its
  spread and LSD charging `cathar075` (9.41 + 13.11) more than the others. The
  rule's values admit `cathar075` (6.072 / 8.280), which leaves pause depth's
  mixed-route aggregate to order the pair. No dead zone under the rule restores
  it: spread cannot charge `cathar075` without charging the preferred V alpha 2
  (6.301) more, and LSD only between 6.840 and 6.883, charging alpha 2 as well.
  A speech-only pause-depth aggregate would; until it exists
  `tests/unit/test_tune_restoration_v3.py` pins the reversal (the loop's
  worst-family win rate reverses the stitched pair only, its weighted mean rank
  neither). S alpha 2 (6.08, the slope) above `apl__baseline` (4.56) and
  `cathar__baseline` (0). On T's round-one ranking the roformer renders (ranked
  last, 8.05 / 8.12) below the other APL renders (16.12-23.55, the slope) in 8
  pairs. V's alpha 2 now holds (3.68 against 5.33 and 10.39), and the three
  cathar tiers on top of T's round-one ranking tie at 0 instead of reversing. By
  the loop's own readings of the grid, `reward.py`'s worst-family win rate and
  `tune_restoration.py`'s weighted mean rank, 14 and 14 pairs reverse (15 and 20
  under the old values on the new reports).
- Rounds two and three: every file scores 0, so their 11 preference pairs
  tie.
- A pass-through scores 2.95 (pause depth's low edge), above every
  round-two and round-three file.

## v2 flags to retire or restrict

- Retire `listener.hiss` (still `flag` in
  `experiments/quality_calibration/gates.json`; all 8 accepted r2 speech
  finals fire), `dead_air` (misses the roformer, fires on 4 accepted music
  outputs), `sibilance_thin`, `sibilance_dull` and the shipped `bright`.
- Restrict `pause_collapse` to speech-route windows inside the new
  `dead_air` (mixed windows read V's alpha 2 +38.68, speech-route +30.18),
  `dull` to presence, `attack` to speech: every music false fire came from
  a 15 s window `route_window` called mixed.
- Hard gates against verdicts: `dsp.hf_8k16k` under -12 fails G's preferred
  B (-18.195; abstain under an 8 kHz band); `dsp.output_silent` fails G
  `final2_cathar` and two music outputs.

## R1 cannot see the shelf on linear-track tapes

R1 is clipped to R0's programme band, and air needs a 5292 Hz band: T, S and
G end at 4490 / 5040 / 5040 Hz, so between +2 / +1 / off presence moves
0.03-0.05 dB and the tilt 0.02-0.10 dB/oct. The paired `hf_8k16k` detector
and R2 `sib_abs_level_db` (clipped to the brickwall) carry the shelf.

## Provenance erratum: the hiss pair is 0.7.3 against 0.7.5

The round-one "has hiss" file `tele7abc__cathar__alpha_2_0.mov` was
rendered 2026-09-20 19:11 with the venv's cathar 0.7.3, the validated
binary then (`install_dependencies.ps1` pinned 0.7.3 until b54c217,
2026-09-26). 0.7.6 arrived later that evening and is bit-identical to 0.7.5
on the app's stages. The append-only ledger (`r1-tele7abc-listen-*`
`context.note`) says 0.7.6 and stays as written: read it as 0.7.3.

## The learned vetoes' floors are too tight

`experiments/reward/noise_floor.json` (`python -m scripts.reward_noise_floor`
on v4_air B and v2 `final2_cathar` of T/S/V/G, four families, two repeats,
125 s) gives every v3 veto a floor. At 3x the floor the limits are CER
0.0025, speaker cosine 2.9e-5, MERT 2.3e-4, UTMOS 0.0027, SCOREQ 0.0033 and
SIGMOS 0.20-0.25: near-copies barely move Whisper or the embeddings, a real
restoration change moves them more, so the vetoes would refuse moves on
jitter. Proposed, not done: before the first v3 loop round, measure them on
pairs the audibility check calls inaudible (v3 sibilance final against v2).
The benign-pair mode exists now (`--benign-pair SOURCE OUTPUT_A OUTPUT_B`
of `python -m scripts.reward_noise_floor`); its run on Session 0's four
pairs nobody told apart, written to
`experiments/reward/noise_floor_benign.json` for the vetoes, is pending.

## What Session 0 asks

Blind, `python -m scripts.listen_ab` on 127.0.0.1, blocks of 16-20 trials,
every new render through `scripts/audibility_check.py` first:

1. Hiss, T: ABX 20 of `tele7abc__cathar__alpha_2_0.mov` (0.7.3) against
   `tele7abc__cathar075__alpha_2_0.mov` (0.7.5) at 105 s, on the pauses (15
   of 20 is p 0.021), then a pair with "same". Not heard: drop `pause_hiss`.
1. Air, S: ABX 20 of `v4_air/soti__B_air_1dB.wav` against
   `soti__A_as_heard.wav` at 165 s or 270 s (window-paired A-B +0.188 /
   +0.187), then B against C with "same". Not heard: `MASKING_OFFSET_DB`
   moves. Not G: its A and B differ beyond the shelf (NMR 22.9 dB).
1. Pause depth, T: blend 20 of B against B rendered with
   `enable_pause_floor: false`, if its speech-route depth reaches about
   37 dB, to place `dead_air`'s edge in 32.98-37.22.
1. If time remains: T and V `known/*_single4s.wav` against `*_stitched.wav`
   (muffled or watery?); S pauses at 180 s, 0.7.3 against 0.7.5 alpha 2 and
   baseline (settles SOTI); G baseline against 0.7.5 alpha 2: dead pauses?

## Session 0 results (2026-10-09, blind)

One listener, blind, `python -m scripts.listen_ab` on 127.0.0.1, 10 s cuts
loudness-matched on speech-active frames. The playback device and volume were
not recorded (`playback` is null in the records). Every pair below was called
audible by `scripts/audibility_check.py` first. Records: the six
`round: session0` lines of `assets/quality_calibration/verdicts.jsonl`.

Each block: the pair (tape, cut, question), what the check read, what was
heard.

- Hiss: 0.7.3 against 0.7.5 alpha 2 (T, 105 s, "more hiss in the pauses?").
  Check: diff -17.6 dB, 97% of frames over the mask, NMR 41.6 dB. Heard: 6
  same, 1 for 0.7.5.
- Hiss: the same pair (S, 180 s). Check: audible. Heard: 7 same.
- Air: ABX +1 dB against +2 dB shelf (S, 220 s). Check: 34% of frames over
  the mask. Heard: 10/20, p 0.59.
- Drift: shipped v1.3.8 APL against round-three B (T, 80 s, "more
  natural?"). Check: diff -39.4 dB, 12 event frames, NMR 28 dB. Heard: 7
  same.
- Sanity: raw tape against shipped APL (T, 80 s, "cleaner?"). Heard: 3/3
  APL.
- Control: ABX 0.7.3 against 0.7.5 alpha 2 (T, 105 s, whole sound). Check:
  the served cuts differ by -15 dB. Heard: 8/16, p 0.60.

The listener's words: "I chose B every time because they all sounded the
same." On both ABX blocks every answer was B.

What it means:

- The playback chain works (raw tape against a restoration is heard every
  time), and the page serves different audio for A and B (checked: the
  served cuts differ by -15 dB, loudness-matched).
- No difference between two restorations measured so far is heard blind on
  this setup, including the largest one available: 0.7.3 against 0.7.5,
  which removes 5.6 against 11.0 dB of noise.
- The unblinded verdicts this ledger rests on do not reproduce blind: round
  one's "cathar alpha 2 has hiss" (two tapes) and round three's "B is better"
  (SOTI).
- The audibility check's "audible" does not predict this listener at all.
  Its masking model is calibrated for a nominal playback level
  (`PLAYBACK_SPL_LOUD_FRAMES = 65`), and its anchors came from the unblinded
  round-three verdict. A tie is still trusted; "audible" is not, until the
  check is recalibrated against a blind threshold.
- The A0 drift (shipped `apl_tonal_flatness_max` 0.035 and
  `apl_sibilant_mix` 0.8, against round-three B's 0.01 and 1.0, which tamed 9
  plosives on Tele7abc) is not heard: the shipped values stay.

Next measurement, before any tuning round: the listener's blind threshold on a
blend continuum from the shipped restoration toward the raw tape (the
`blend` mode of `listen_ab`), at a recorded device and volume. The percentage
of raw tape at which the change is heard is the unit every later "audible"
is judged in.
