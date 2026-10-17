# Ear v4: Critical of Hiss

The acceptance record of ear v4 (built 2026-10-12 on `wip/ear-hiss`), the
readings that make the harness very critical of VHS hiss under speech or
singing and in silence and pauses (the user's rule of 2026-10-11). The
first acceptance run failed; validation round 1 (the same day) traced each
failure to a defect in the readings or in the transforms that set their tie
bands and fixed the implementation; validation round 2 failed that build
on 20 of 33 rows, and the fixes below answer it; validation round 3 and
the adversarial review that followed it (first section) found the guards
could still be won by emptying the pauses and the pause reading charged
room tone and soft singing, and fixed both. The bars were not moved:
where a row's own check contradicts the user's verdicts or the signal, the
row is listed as not passed, with the evidence, under "Rows Whose Check
Contradicts the Evidence". Nothing here shipped a default: the blind block
(`audibility_check.py`, then `listen_ab` with the device and volume
recorded) and the realistic-v2 plus IA confirmation come first. T, S, V and
G are Tele7abc, SOTI, Vaccin and Gaudeamus5. The readings and the grids are
described in the output-quality-harness skill, "Critical of hiss (ear v4,
2026-10-12)". The continuous-programme round of 2026-10-17 (first section)
added a second anchor for the programme reading and measured why it cannot
reach dense singing.

## Continuous Programme: the Second Anchor (2026-10-17)

The programme reading reports its margins only on a validated anchor, and
the anchor validated only from R4's true pauses. Music and singing rarely
hold 1 s of them, so the reading read 26 of 235 IA hunt clips and 2 of 12
music clips: the ear was blind to hiss under singing, half of the user's
rule. This round (`wip/ear-music-anchor`) built a second validated anchor
for continuous programme, `hiss_stationary.py`, and measured why it can
only be narrow. CPU only, stored files only; the family tapes never left
the machine.

### What No Anchor Fixes

The programme reading cannot tell programme that a hiss-free output reveals
from residual hiss in a noise-dominated cell; the spectral cap (round 1)
tells them apart only where the programme leaves bands or gaps to the hiss.
On dense programme no anchor removes the ambiguity:

- **The exact noise as the anchor.** On realistic-v2's music fixtures (de,
  en, es, fr, it at m09, m12, m18 and m24; the target as the perfect output;
  `vhs - clean` as the anchor, validated by hand) the perfect output read at
  or under 0 on 0 of 36 music-bed fixtures (+2.6..+29.1 dB, median +21.1),
  on 1 of 40 music-led and on 23 of 40 music-only fixtures, which hold real
  silences. The clean reference, which keeps the generator's ambience bed
  (up to 23 dB over the tape noise in some bands), read at or under 0 on one
  of 116 fixtures. The fixtures' corpus noise is not stationary hiss: its
  1 s block levels span 2-15 dB per band, and it is often band-limited at
  5-7 kHz.
- **Stationary hiss under the same programmes.** The target plus stationary
  tape hiss (white noise plus a 2 kHz high-passed copy at the recorded
  noise's 3-10 kHz level, or Tele7abc's pause spectrum placed against the
  programme as on the tape), full band or low-passed at 4.5 kHz, at m12:
  with the exact hiss as the anchor the full-band music-bed and music-led
  perfect outputs read over 0 on 39 of 40 (up to +17.0 dB); the low-passed
  ones read at or under 0 on 50 of 60, the misses dense programme (nine
  music-led, one music bed; +0.4..+11.1).
- **Each dB the anchor stands over the true hiss reads about 3 dB on a
  perfect output.** A stationary anchor 0.7-0.9 dB over the exact hiss in
  three note bands read a perfect output +0.22 where the exact hiss reads
  -2.96.

The task's second candidate, the bands over the programme extrapolated
down, failed both ways. An anchor from the hiss-alone bands (mean within 1
dB of the floor, 1 s spread under 1.5 dB, no correlation with the programme
envelope), held at their median plus 3 dB under them, read perfect outputs
up to +23.7 dB on the stationary-hiss variants: the tape-hiss shape sits
about 9.5 dB lower under 2 kHz than over 4 kHz, so a flat extrapolation
stands over the hiss there. Holding the reading to those bands alone fixed
the perfect output and went blind under the programme: a render that kept
the hiss under 5 kHz and removed it above, which the exact anchor reads
+13..+30 dB over its masked threshold on music-led and music-only
programme, read -21.9..+20.8, at or under 0 on 37 of the 67 variants it
validated. That render is the user's complaint itself, so neither form was
kept.

### The Second Anchor

`scripts/restoration_quality/hiss_stationary.py` is tried only when the
pause anchor is not validated, and is carried as `Anchor.programme`. The
pause reading keeps the pause anchor; the programme reading takes the second
one only when it alone validated (`Anchor.for_programme`), so a source whose
pause anchor validates reads exactly as before.

1. Per 15 s segment, each band's floor per 3 s block is C_b x P10 of its
   valley. A band is accepted where the block floors span at most 1 dB from
   p10 to p90 and every block holds over the 16-bit dither plus 10 dB; its
   level is the blocks' P25. The median read one perfect output +0.22; the
   lowest block biased a fresh draw of the raw tape +0.44 dB (p95 of three
   draws per source); the P25 reads +0.34 with every perfect output under
   -1.7.
1. A band carries programme where its mean stands more than 1 dB over the
   segment's floor or its frame power correlates over 0.15 with the
   300-4000 Hz envelope; it is hiss alone where neither holds and its 1 s
   block means span at most 1.5 dB.
1. Over the segments a band counts where half of them say so. The anchor is
   validated when (a) every band of 1 kHz and up that holds a floor is
   accepted but at most four, and six at least: the programme leaves the
   hiss exposed in every band within every 3 s; and (b) two hiss-alone
   bands lie from half an octave over the programme's top to 12 kHz: the
   programme stops, so the spectral cap has bands where nothing can be
   revealed. Bands not accepted read NaN and the reading interpolates them.

The runner caches it with the pause anchor (`runner.HISS_CODE` covers the
module) and writes `meta.hiss_anchor_how` (1 pauses, 2 stationary, 0 none)
and `card.meta["hiss_anchor"]["programme_how"]`; `check_hiss_fixtures`
reads the same anchor.

### Validation

`D:\ai_restore_work\scratch_manchor` holds the scripts and their JSON. The
fixture rows read realistic-v2's music, music-led and music-only fixtures
(5 languages x 3 kinds x 2 programmes x m09-m24, 150) and, built on
their targets, the stationary-hiss variants: tape hiss over the full-band
and the 4.5 kHz programme at every margin (150 each) and Tele7abc's shape
over both (30 each, the shape does not follow the margin).

Coverage, before (the pause path alone) and after (with the second anchor):

| sources | before | after |
|---|---|---|
| corpus fixtures | 8 of 150 | 8 + 0 |
| tape hiss, full band | 20 of 150 | 20 + 0 |
| tape hiss, 4.5 kHz programme | 22 of 150 | 22 + 36 |
| Tele7abc shape, full band | 4 of 30 | 4 + 0 |
| Tele7abc shape, 4.5 kHz programme | 5 of 30 | 5 + 2 |

The rows of the 38 sources the second anchor validates:

- **Perfect output** (the programme alone) at or under 0 on 38 of 38:
  -29.98..-1.70 dB, median -19.6; the all-band margin the same. **Pass.**
- **Raw tape** reads the hiss on 38 of 38: +24.0..+37.3. **Pass.**
- **Ladder** (raw, hiss -6, -12, -20 dB, perfect) monotone within 0.05 dB
  on 38 of 38. **Pass.**
- **Hiss kept under 5 kHz, removed above** read on 38 of 38: +19.6..+32.6
  (the render the bands-above candidate read as clean). **Pass.**
- **Same-percept**, +-0.1 dB, a 0.37-sample delay and a soxr round trip:
  p95 at most 0.001 dB, max 0.005. **Pass.**
- **Same-percept, a fresh realisation** of the hiss from its own process
  (three draws per source, at the raw level and 12 dB down): p95 0.36 dB,
  max 0.54 (the pause path on the variants it validates: 0.34, max 0.56);
  `same_percept.fresh_like`, an envelope-imposed redraw: p95 0.69 (pause
  path 0.52). **Not passed** against the 0.30 / 0.25 ties (below).

Through the runner's own pair, anchor and windows, base code (the main
checkout at ee14637) against this branch:

- **Tata v2 pairs** (Tele7abc, SOTI, Vaccin, Gaudeamus5, Gaudeamus1; the
  identity, the APL and the cathar final): all five validate from their
  pauses; 0 of 11,655 hiss readings moved. **Pass.**
- **Music clips** (12, the source against itself and every stored
  autotune_v6_music candidate, 2,484 pairs): 2 of 12 read before and after
  (`x_vid-20230624-142023` 207 of 207 windows, Gaudeamus1 1,035 of 1,656);
  no source validates the second anchor; 0 of 30,870 readings moved on the
  942 pairs read with both codes.
- **IA hunt** (470 pairs, 235 sources): 34 sources validate from their
  pauses, 26 clips and 92 of 902 windows read, before and after; 0 of
  18,942 readings moved.
- **Fixture probe** (`check_hiss_fixtures`, the 40 gapped m12 fixtures):
  the hiss shape validates 20 from the pauses and 3 more through the second
  anchor (perfect -20.7..-16.7), Tele7abc's shape 23 and 1 more (-15.7),
  the corpus shape 12 and none more.

Not passed, with the evidence:

- **The fresh realisation.** A redraw of the hiss from its own process
  reads p95 0.36 dB (median 0.15, max 0.54) on the sources the second anchor
  validates, over the grids' ties (0.30 on `hiss_prog_excess_db`, 0.25 on
  the all-band excess). It is a bias of the reading, not of the anchor: the
  deviations are positive (a fresh draw reads more hiss than the source's
  own, whose low cells the noise-dominated test picked), and the pause path
  reads p95 0.34 (max 0.56) on the variants it validates. The ties are p95s
  of window medians over the first 125 s of eight tape pairs (16 windows
  each); a 15 s clip has one or two windows.
- **The realistic-v2 corpus fixtures.** No anchor validates them without
  reading revealed programme as hiss (the exact noise above); the second
  anchor abstains on all 150 (a median of 3 of 23 bands accepted, none
  with a band edge).
- **Real continuous programme.** The second anchor validates no music clip
  and no IA hunt source. On the music clips 0-9 of 12-24 bands of 1 kHz and
  up are accepted and no band is hiss alone over the programme; on the IA
  hunt sources a median of 21.5 bands miss (p10 15.3), 4 of 234 are covered
  and 3 have a band edge, none both; even the Tata speech tapes accept 0-8
  of 20-24 bands (their programme bands' 3 s floors span 2-4 dB; over the
  programme 0.4-1.3 dB, within -0.2..+1.3 dB of the pause anchor). Real
  programme exposes the hiss in its pauses, not in a stationary floor, so
  the ear stays blind to hiss under dense singing.

Found on the way: the pause path validates dense music and then reads
revealed programme as hiss. On the variants it validates, the full-band
programmes' perfect outputs read +1.4..+18.8 dB (24 of 24), the 4.5 kHz ones
over 0 on 3 of 27; on the corpus fixtures it validates, 5 of 6 read
+3.9..+15.3 and the clean reference 0 of 6 at or under 0. Its validation
reads R0's programme bandwidth (2 kHz or more), which on music reads the
gap frames' programme (630-1414 Hz on most of these fixtures, 500 and 891
Hz on two music clips, 561-794 Hz on four IA clips), and has no band-edge
test like the second anchor's (b).

### The Pause Path's Hole: No Band Edge Closes It

The follow-up task (the same day) was to close that hole with a band-edge
condition on the pause anchor's validation, accepting that sources which
validate today might change, but keeping the four Tata tapes (Tele7abc,
SOTI, Vaccin, Gaudeamus5) and their pinned readings. No condition does both;
nothing was changed. Per band, on the active frames (live, outside R4's true
pauses) against the uncapped pause floor, `probe_edge2.py` read the median
excess, its P90, the share of frames at the floor and the correlation with
the 300-4000 Hz envelope, on every source the pause path validates (the 59
fixtures and variants, the Tata sources, 34 IA hunt sources, 2 music clips):

- **The second anchor's (b) on the pause path** (a band carries programme
  where its median excess is over 1 dB or it follows the envelope, r over
  0.15; two hiss-alone bands from half an octave over the programme's top
  to 12 kHz) drops 19 of the 24 dense variants and all 5 failing corpus
  fixtures and keeps all 27 linear ones, but it also drops all five Tata
  tapes, all 34 IA hunt sources (26 clips, 92 windows read today) and both
  music clips: the ear would read no tape at all.
- **A sweep of 576 settings** (excess 0.5-8 dB, guard 1-2, a top of 12 or
  16.5 kHz, 1-3 edge bands, the correlation test at 0.15, 0.3 or off): 71
  keep all four Tata tapes, and the best of those still validates 25 of the
  32 sources whose perfect output reads over 0 (20 of the 24 dense
  variants). Giving up Vaccin, the best keeps 10 of them.
- **Why the family cannot work.** By every source statistic the Tata tapes
  look more like programme over the hiss than the failing music does. From
  9.7 to 15 kHz the median excess reads +0.6..+1.0 dB on Tele7abc and SOTI
  and +1.0..+1.4 on Gaudeamus5; Vaccin reads +3.1 at 9.7 kHz, and no band
  above 10.8 kHz holds a floor over the dither. The dense variants read a
  median of -0.1..+0.4 dB per source from 8.7 kHz up (90 % of their bands
  under +0.6), with 37-100 % of their active frames at the pause floor.
  Their programme's top (the highest band whose P90 excess passes 3 dB)
  lies at 7.8-10.9 kHz, the Tata tapes' at 8.3-15.1 kHz (SOTI 8.3, Tele7abc
  and Vaccin 9.7, Gaudeamus5 15.1). SOTI's top bands follow the speech
  envelope (r 0.2-0.3 at 8.7-13 kHz: the tape's gain pumps the hiss with the
  speech). The reading's own free share in 1-4 kHz is 0.000-0.030 per window
  (median) on the Tata tapes and 0.000-0.055 on the dense variants.
  Speech tolerates a wide, sparse transition into the hiss; dense music
  fades under it continuously over 4-10 kHz, and no band-edge statistic
  tells the two apart.
- **Not a band edge either: tonal persistence** (`modules/tonal_persistence`,
  music against speech) reads 0.000-0.006 on the four Tata tapes and
  0.012-0.029 on the dense variants (speech over a music bed), but
  0.002-0.010 on the failing corpus fixtures, so a class gate does not close
  the hole and would blind the ear on every music clip.

The defect sits in the reading, not in the validation: with the exact hiss
as the anchor the dense perfect outputs still read over 0 (39 of 40 at m12,
above). On `it mid01_music_m12` with tape hiss, full band, the music fades
under the hiss over 3.5-10 kHz; those bands' noise-dominated ratios read
-5.5..-17 dB (-13..-22 after the spectral cap) and +2..+11 dB over their
masked threshold, while the hiss-alone 12-15 kHz bands read -38..-49: the
revealed music fills the cap's pool. Closing it needs a reading change that
still sees the hiss a render keeps under the programme (a pool held to the
hiss-alone bands reads that render as clean, above).

### Revealed Programme Is Not Told Apart in Time

The next task (the same day) tested the hypothesis that residual hiss and
revealed programme differ in time: residual hiss stationary, revealed
programme following the music. It fails, and nothing was changed. Per band,
in the noise-dominated cells of each frame, the output's mean power against
the source's (`probe_rp1.py` to `probe_rp4.py` in the scratch folder):

- **Revealed programme does follow the music.** On `it mid01_music_m12`
  with tape hiss, full band, the perfect output's NDC power correlates
  0.15-0.83 (0.48 or more in 7 of 11 bands) with the output's own programme
  cells in the same band over 3.5-10 kHz. Its P10 over frames sits 12-29 dB
  under the source's, where the mean sits 3-16 dB under; renders keeping the
  hiss 6, 12 and 20 dB down read P10 ratios of -4.3..-5.9, -8.4..-11.6 and
  -11.1..-18.6 dB there.
- **So does hiss a render lets through with the programme.** A render that
  keeps the hiss at full level under the louder half of the programme and
  20 dB down under the quieter half (an expander's gate; every
  SNR-dependent gain, a subtractor's or a mask's, lets hiss through where
  the programme is) correlates 0.55-0.83 with the programme cells over
  5-15 kHz on `en mid00_music_m12` with tape hiss over the 4.5 kHz
  programme, as strongly as revealed programme. Its NDC power sits 1.0-3.3
  dB under the hiss floor on average, its P10 ratio 15-19 dB under (the 20 dB
  render's 17.5-20), and a regression on the programme envelope leaves an
  intercept 4.7-10.9 dB under the floor.
- **Capping each band's removal at its stationary ratio** (the output's P10
  over the source's per-frame NDC power) lowers the perfect outputs, under 0
  on one of the three dense sources, and reads the gated hiss as removed:

Margins in dB. A is Italian `mid01_music_m12` with tape hiss, full band;
B English `mid00_music_m12`, the same; C is B's programme low-passed at 4.5
kHz; D English `mid00_music_m18` with its corpus noise; -6 to -20 keep the
hiss that far down:

| source | cap | perfect | -6 | -12 | -20 | kept under | gated |
|---|---|---|---|---|---|---|---|
| A | none | 16.65 | 27.77 | 23.52 | 19.76 | 19.52 | 29.63 |
| A | P10 | 11.26 | 27.51 | 22.15 | 16.26 | 15.58 | 17.90 |
| B | none | 5.42 | 23.29 | 18.95 | 13.64 | 5.58 | 25.67 |
| B | P10 | -0.34 | 23.17 | 17.79 | 12.31 | 0.17 | 15.79 |
| C | none | -34.31 | 24.82 | 18.87 | 11.07 | -5.83 | 27.30 |
| C | P10 | -35.52 | 24.82 | 18.84 | 10.98 | -5.92 | 12.99 |
| D | none | 15.26 | 21.60 | 18.69 | 16.08 | 16.26 | 21.70 |
| D | P10 | 2.62 | 21.52 | 16.86 | 11.25 | 11.34 | 12.38 |

The gated render keeps the hiss at full level under the programme's louder
half and reads 21.7..29.6 dB today; capped, it reads 12.4..17.9, under the
render that keeps the hiss 12 dB down everywhere. A P25 cap is between
(perfect 14.16 / 1.50 / -34.96 / 10.48, gated 23.57 / 20.36 / 16.54 /
17.69). Hiss gated out of the quiet moments and left under the programme is
the user's complaint itself, so a split by time cannot be the fix: the
residual of the engines this ear judges follows the programme in time just
as revealed programme does. A later attempt has to tell the two apart by
something they do not share.

## Adversarial Review: Defects and Fixes

Validation round 3 (on 6fd0295) and the adversarial review that followed
it found four holes a loop could use, each against the user's rule that no
hiss never means dead air and that room tone mostly under about 2-3 kHz
may stay. Each defect, its cause and the fix with the measurement after it
(the review's own runs through the runner's pair, anchor and windows; the
other rows below were re-measured where these changes move them):

- **Gating the pauses won the pause family and passed every guard.** B
  with the source's true pauses attenuated 20 or 30 dB passed all 16
  tata_v4 constraints against the shipped APL and won `hiss_pause` (pause
  tail +11.0 / +2.8 against +30.7). The pump read +8.2 / +11.2, and +11.5
  at 40 and 50 dB, under its 12 dB bound at every depth: its pause ratio
  was a ratio of means over the pause frames, and the few STFT frames that
  straddle a gate's edges carry the ungated audio and set the mean. The
  pause ratio is now each band's median band power over the pause frames,
  output over source (`hiss_programme.paused_ratio`); a band whose source
  pauses sit under the 16-bit dither plus 10 dB is left out, so a muted
  Hi-Fi break reads no pump at all. The output is not held at the dither:
  T's pause hiss stands only 28-31 dB over it per band and its dead APL
  pauses 3-17 dB under it, and a dither floor read them at +5.9 instead of
  +19.4. T's gates of 10 / 20 / 30 / 40 / 50 dB now read +2.1 / +11.8 /
  +21.8 / +31.8 / +41.8, S's 10 / 20 / 30 dB +1.8 / +11.6 / +21.6 and G's
  -4.6 / +5.0 / +15.0; V's programme windows hold no 0.5 s of pause, so its
  pump reads None and the other tapes carry the guard. The bound stays
  12.0 on new evidence: midway between the round-one renders the user did
  not call dead (at most +6.49: T's beta 0.005; the cleared cathar075 alpha
  2 +2.93 and alpha 2 +4.88; known cathar075 +6.29) and the dead APL
  (+18.09..+19.40; roformer +95), 12.29, rounded down. A 30 dB gate now
  trips the pump and nothing else on T; a 20 dB gate keeps audible hiss in
  the pauses (T pause tail +10.1) and passes, left to the listening
  session. The synthetic gated cheat reads +103.9 (11.5 before), the
  calibration's `dead_pauses` +10.0 / +25.0 / +40.0 at 30 / 45 / 60 dB.
- **A bounded guard could be crept past.** A constraint tripped only when
  a candidate was worse than its incumbent by more than `max_move` and past
  the bound, and the incumbent moves every accepted round: the pump could
  go 11 -> 14 -> 17 against its bound, dulling and islands (no bound) could
  grow by `max_move` every round. `reward.Constraint` now also excludes a
  candidate worse than its incumbent and more than `max_move` past the
  bound, whatever the incumbent read; the dulling guard takes a bound of
  -6.0 (every judged file the user did not call dull reads -5.60 or more,
  and with `max_move` the floor holds at -7.0, over T single4s' -7.04,
  "underwater again") and the programme islands 2.0 (the largest judged
  reading 1.87 plus one 0.1 unit, rounded up), in tata_v4 and music_v4.
- **Room tone read as pause hiss.** The hiss weight was 0.5 at 2.5 kHz, a
  quarter octave per e-fold, applied after the five-band smoothing, which
  carried a room floor under 2 kHz into the 2.4-2.7 kHz bands at their full
  weight: a hiss-free pink room floor low-passed at 2 kHz, at the source's
  own 100-1000 Hz pause level, read +9.7 dB median on the ten speech
  fixtures and was flagged on 9 (on 6 at 6 dB down). The weight is now
  half at 3.15 kHz, 0.15 octave per e-fold, applied before the smoothing,
  and the margin starts at 2 kHz (`hiss_pause.weighted`): that floor reads
  -4.2 (none flagged), -10.2 at 6 dB down; low-passed at 2.5 kHz and 12 dB
  down -6.1 (none). A floor reaching past 3 kHz is still counted (low-passed
  at 3 kHz, at the source's level +12.6, 9 of 10). Tape hiss set the margin
  at 3.5-3.9 kHz on every fixture: the source reads 0.5 dB lower (39.98
  against 40.46) and the -40 dB oracle -0.00 (+0.48 before). On the tapes
  the speech pause readings fall 0.7-2.7 dB and keep their order (v2 finals
  T 29.86 / 27.32, S 23.35 / 20.22, V 22.06 / 20.63, APL / cathar).
- **Soft singing still set the pause floor.** A 200 ms block counted as
  silent unless its band stood 6 dB over the anchor; Gaudeamus5's soft
  singing stands about 4 dB over it, so most singing blocks passed: the APL
  keeper ladder read 37.53 / 34.10 / 32.62 / 32.62 (fill 8 / 12 / 18 /
  keeper off) and a render that took the soft singing out would have won
  the family on G. The test is now 3 dB (Tele7abc's pause blocks stand
  +0.2..+1.2 dB over its anchor per window, SOTI -1.0..+2.0, Vaccin
  -1.2..+2.9): the ladder reads 34.66 / 31.21 / 25.09 / 20.50 (cathar
  34.00 / 31.01 / 26.18 / 24.40). G's windows of soft singing that every
  render keeps still read the same for every fill, but no longer set the
  tail. The corpus fixture whose perfect output read +24.25 in its pauses
  (es mid00, soft speech in R4's mask) reads +15.31: that noise is not
  stationary (below).

Re-measured with the fixed build:

- **The jitter report** (the eight pairs, two repeats, DSP repeat jitter
  exactly 0): gain 0; delay p95 at most 0.005; soxr 0.0001; requantisation
  0.059 (all-band margin tail). Fresh draws a / b / c: programme excess
  median 0.17 / 0.29 / 0.18 and all-band 0.15 / 0.19 / 0.24 (as round 2),
  pause excess tail 0.21 / 0.32 / 0.18, median 0.20 / 0.33 / 0.13. The
  pause ties become 0.319 -> **0.35** (tail, was 0.50) and 0.331 -> **0.35**
  (median, was 0.25), H1's pause `max_move` 0.35; the programme ties (0.30,
  0.25) and H2 (0.35 / 0.35; music 0.55) stay.
- **Fixtures**: fresh pause margin p95 0.166 (T's shape), 0.125 (hiss),
  0.722 (corpus), against 0.212 / 0.177 / 0.762; every perfect output of
  T's and the hiss shape reads its pauses at or under 0; `gated_m0` pump
  +30.2 on all 43; the programme rows read as round 2.
- **Calibration** (en, fr, dsp): 133 of 196 checks pass, every status as
  in round 3, every hiss check passes; the benign floors of the pause
  margin 0.077 (0.080) and the pump 0.077 (0.106), the non-hiss floors
  unchanged.
- **Tapes**: the rows below carry the fixed build's numbers.

Not fixed, reported: the corpus noise shape and the IA programme coverage
(below); a 20 dB pause gate passes the pump (above); and on T the APL fill
8 render reads its pause tail 0.22 dB over the raw tape's, inside the 0.35
tie, while H2's per-window delta (+1.03) excludes it.

## Validation Round 2: Defects and Fixes

Each defect as the round-2 report measured it, its cause, and the fix with
the measurement after it:

- **A 0.37-sample delay moved T's programme margin tail 0.13 dB.** The
  runner's alignment absorbs the delay by a one-sample lag change, so a
  window's first and last frames change; the centred STFT's first and last
  frames are half padding, read the masker 3 dB low under the full hiss
  estimate (one T window read 0.26 dB higher through its first two frames
  alone) and sat among the quietest frames that set the shape, the tonal
  bins and the P10 floor: one sample at the end of Gaudeamus5's last window
  moved a band across the leak test (-2.00 / -1.90 dB) and the window 2.4
  dB. The programme reading now keeps the STFT frames wholly inside the
  window (`hiss_programme.whole_frames`) and leaves the two frames at
  either end out of the active and the pause frames (`edge_frames`). Every
  window of both pairs now moves at most 0.01 dB; the report's delay p95 is
  0.004 (programme margin median) and 0.002 (tail). The half-padded frames
  had inflated some windows' P90 by up to 2 dB (T's B window 15 +4.9 ->
  +2.8).
- **The ties were diluted.** Each grid `tie` was the pooled p95 over all
  eight transforms; gain, delay and soxr move the hiss readings by about 0,
  so the pause-tail tie read 0.20 against a fresh-draw p95 of 0.50-0.80 and
  H1 refused a same-percept redraw. A tie band must cover every same-percept
  transform: `reward_noise_floor` now reports `tie_floor`, the widest
  transform's p95, and `reward.tie_floors` reads it (`tie_band_floors`);
  the grid ties and H1 / H2 take it (below).
- **The pause reading read singing as hiss.** On a choir tape R4's quiet
  frames inside the VAD's non-speech runs are soft singing: Gaudeamus5's
  source pause blocks stood +4.1 dB (median) over its own stationary hiss
  (the anchor) at 2-8 kHz, and +5.6..+8.7 in the windows that every render
  kept unchanged (room delta about 0, keeper off included), against
  +0.7..+1.2 on the speech tapes (their 90th percentile +2.3..+7.1). The
  singing pinned G's pause tail at 39.53 for every keeper fill and set the
  fresh draws' pause p95 (0.80 tail, 0.99 median on its cathar final, the
  transform redrawing the soft singing). Only the source's silent blocks now
  count, band by band on both sides: a 200 ms block whose source band of 1
  kHz and up stands 6 dB or more over the anchor holds programme there
  (`hiss_pause.silent_rows`, `anchor_band_power`); a window whose every such
  band held programme reads None. G's pause tail now moves with the fill
  (37.5 / 34.1 / 32.6 at fill 8 / 12 / 18).
- **The pause margin's realisation scatter set ties near the bars.** The max
  over three-band power means picked upward draws: T's B read 0.28-0.36 dB
  lower on each fresh draw (pause median). Each band now takes the power mean
  of five bands (two either side). With the silent blocks the fixtures'
  fresh-realisation p95 of the pause margin fell from 0.51 / 0.54 / 1.36 to
  0.21 / 0.18 / 0.76 (T's, the hiss and the corpus shape), the calibration's
  benign pause floor from 0.337 to 0.080, and the tapes' fresh p95 from 0.80
  to 0.49 (tail) and 0.99 to 0.23 (median).

Tried and dropped, with the measurement: the programme reading's threshold
in quiet 3 dB under Terhardt's, as the pause reading's (margins +1.3..+1.5
dB, no round-one label crossed 0, a T-shape perfect output +0.70); an anchor
from the median of the pause rows with a slope-following cap and validated
all-frames anchors (the corpus shape validated 20 of 40, but its perfect
outputs read up to +22.7 dB and its fresh p95 3.4 dB: that noise is not
stationary enough for any floor estimate, below); three-band smoothing with
the silent blocks (the pause median's fresh p95 0.41) and five bands without
them (0.87).

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

## What Ran (Round 2)

- The jitter report: the eight pairs of the specification (v4_air
  `B_air_1dB` and v2 `final2_cathar` of T, S, V and G against their
  sources), `--transforms hiss`, two repeats, `--families dsp` and the
  first 125 s, through `reward_noise_floor`'s own functions with the final
  build; the pooled report is the worktree's
  `experiments/reward/noise_floor_hiss.json` (gitignored).
- The fixtures: `check_hiss_fixtures`' own functions (`--programme linear`)
  over de, en, es, fr and it of `artifacts/realistic-v2` (40 gapped m12
  tape-noise programmes), with three noise shapes: `--shape corpus` (the
  fixtures' recorded noise), the hiss shape and T's raw tape
  (`artifacts/hiss-v1/shape_tele7abc.npy`, gitignored, never committed).
- The calibration: `calibrate_quality_metrics` with `--languages en,fr`
  and `--metrics dsp`, compared with the base tree's (bb5fa2a).
- The tapes: every stored render's hiss readings through the runner's own
  pair, anchor and windows (15 s, hop 7.5 s), merged with the round-2
  validation's full dsp scorings for the other readings: `r1` (the
  round-one cache of `experiments/tata_listen`), `v2` (the v2 finals and the
  identity), `v4air` (v4_air A, B, C and Session 0's shipped APL), the three
  synthetic cheats built from B, `fill` (the autotune_v6 APL and cathar
  candidates with `pause_floor_fill_db` 8, 12, 18 and the keeper off),
  `known` (the known-ordering excerpts), `music` (the 11 autotune_v6 music
  clips and Gaudeamus1) and the IA defect hunt (the cathar and APL
  baselines of `experiments/tune_ia_hunt_v3`, `python -I`).
- The family tapes never left the machine; nothing was uploaded and no web
  service saw their audio. CPU only, no GPU, no learned model.

## Same-Percept Rows (the Jitter Report)

Eight pairs, nine scorings each, two repeats; the pause keys pool 136
deviations, the programme keys 102 (V's first 125 s validates no programme
window). The DSP repeat jitter is exactly 0. Per transform (p95 over the
pairs, on every margin and excess key):

- Gain +-0.1 dB: at most 0.0001. **Pass** (limit 0.01).
- Delay 0.37 samples: programme margin median 0.004 and tail 0.002, pause
  margin median 0.011 and tail 0.004; largest 0.011. **Pass** (limits 0.05
  programme, 0.02 pause, max 0.30). Round 2's 0.13 came from the
  half-padded edge frames (above).
- soxr round trip: at most 0.0001. Requantisation: at most 0.059 (all-band
  margin tail), programme margin median 0.049, pause at most 0.004.
  **Pass** (limits 0.05 / 0.10 and 0.10 / 0.30).
- Fresh realisation (`fresh_floor_a` / `_b` / `_c`): programme excess median
  0.17 / 0.29 / 0.18 (limit 0.65), all-band 0.15 / 0.19 / 0.24 (0.45), pause
  excess tail 0.49 / 0.24 / 0.28 (0.55), median 0.12 / 0.20 / 0.23 (0.35).
  **Pass.** The adversarial review's re-run on the fixed pause reading:
  pause tail 0.21 / 0.32 / 0.18, median 0.20 / 0.33 / 0.13, the programme
  keys as here.
- The ties: each grid `tie` is the report's `tie_floor` (the widest
  transform's p95) rounded up to 0.05: programme excess median 0.294 ->
  **0.30**, all-band 0.239 -> **0.25**, pause excess tail 0.494 -> 0.50,
  median 0.232 -> 0.25 (the review's re-run: 0.319 -> **0.35** and 0.331
  -> **0.35**, the grids' values now). H1's `max_move` takes the same
  values; H2's bound takes its delta key's own tie floor: programme margin
  delta median 0.327 -> **0.35**, pause margin delta tail 0.341 -> **0.35**
  (0.319 in the re-run), music's all-band delta 0.506 -> **0.55**.
  **Pass.**

## Fixture Rows (`check_hiss_fixtures`)

Validated fixtures: 23 of 40 on T's shape, 20 of 40 on the hiss shape, 12
of 40 on the corpus shape.

- Gain +-0.1 dB: 0.000 on every hiss key, all three shapes. Delay: p95 at
  most 0.001 (max 0.002). soxr: at most 0.002 (max 0.004). Requantisation:
  p95 at most 0.014. **Pass.**
- Fresh realisation: programme margin p95 0.238 (T's shape), 0.308 (hiss),
  0.666 (corpus); all-band 0.267, 0.315, 0.908; pause margin 0.212, 0.177,
  0.762 (0.166, 0.125, 0.722 on the review's fixed pause reading). **Pass
  on T's and the hiss shape**; the corpus shape is not stationary noise
  (below).
- Oracle ladder monotone (identity, -6, -12, -20, -30 dB, prog_only, +0.05
  dB slack): 23 of 23 (T's shape), 20 of 20 (hiss), 9 of 12 (corpus; the
  other three validated music fixtures read None). **Pass on T's and the
  hiss shape.**
- The perfect output (prog_only) at or under 0 dB: 22 of 23 (T's shape,
  median -17.5, max +0.004), 19 of 20 (hiss, median -11.6, max +4.40), 8 of
  9 read (corpus, max +3.19). **Not passed.** The misses are music
  fixtures whose programme, low-passed at 4.5 kHz, sits under the hiss up to
  5 kHz while R0 reads its band as 2.2 kHz: the clean output reveals it in
  12 of the 24 bands of 1 kHz and up (the noise-dominated cells' ratio -9..
  -21 dB, audible at 75 dB: +9.7 over the masked threshold at 4 kHz), so
  the spectral cap's pool is itself half revealed programme.
- Physical removal (`hiss_prog_removed_hf_db` - g at 6 and 12 dB): -0.003 /
  -0.004 (T's shape), -0.001 / -0.002 (hiss), -0.017 / -0.028 (corpus).
  **Pass.**
- hf_left: |median delta| 0.101 (T's shape), 0.104 (hiss), 0.078 (corpus).
  **Pass.**
- gated_m0: margin within +0.5 dB of identity on 23 of 23 and 20 of 20, pump
  +20.4..+25.4 and +20.2..+25.1 (the review's medians: +30.2 on all 43, the
  gate's 30 dB). Corpus: 9 of 12, no pump read. **Pass on T's and the hiss
  shape.**
- islands_15: island kurtosis +0.63..+0.73 (T's shape), +0.66..+0.72
  (hiss); corpus median +0.385, 4 of 12 at 0.4 or more. **Pass on T's and
  the hiss shape.**
- lp4k_all: kept HF at or under -1 dB on 23 of 23 (T's shape, -4.4..-10.6),
  19 of 20 (hiss), 9 of 9 read (corpus). **Not passed on one hiss-shape
  fixture** (Spanish speech, kept HF -0.35): its programme ends at 3.6 kHz,
  so a 4 kHz low-pass takes nothing of the programme in the bands the guard
  reads; it takes the hiss above 4 kHz, which is the cleaning, not dulling.

## Calibration Rows (en, fr, dsp)

- Before (bb5fa2a): 108 of 165 checks pass, 3 non-blind failures
  (`gated_pauses` pause pumping, `pause_residual_islands` twice). With the
  final build: 133 of 196 pass, the same 3 non-blind failures and no other.
- Every pre-existing check status is unchanged (165 of 165); every non-hiss
  benign floor is unchanged (51 of 51); only the `dsp.hiss_*` floors come
  from the `*_linear` benign cases. **Pass.**
- The 21 new non-blind checks all **pass**: `hiss_under_programme`
  (programme margin -5.97 / -11.87 / -19.55, pump down, pause margin, kept
  HF and tilt flat), `hiss_in_pauses_only` (pause margin -5.99 / -11.99 /
  -19.99, programme margin steady, span 0.038), `hiss_hf_tilt`,
  `low_floor_kept` (pause margin steady at -29.99, span 0.00002, and below
  the centre; programme steady), `dead_pauses`, `hiss_islands` (+0.50 /
  +0.77 / +1.11, mild level 34.6 floors), `hiss_removed_dulled` (kept HF
  -17.4 / -47.5 / -53.9, margin below).
- The benign floors of the hiss readings: programme margin 0.052, all-band
  0.101, pause margin 0.080 (0.337 in round 2), kept HF 0.095, pump 0.106,
  island kurtosis 0.014, room delta 0.034. The review's fixed build: the
  same 133 of 196 with every status unchanged, every hiss check passing
  (`dead_pauses` pump +10.0 / +25.0 / +40.0), pause margin 0.077 and pump
  0.077, every other floor unchanged.

## Tape Rows

- **Identity** (source against itself, v2, round one, v4_air, known): every
  `dsp.hiss_*` delta exactly 0, output equal to source. **Pass.**
- **Programme-only change** (v4_air A / B / C: the APL air shelf +2, +1 dB
  and off): pause margin spread at most 0.024 (tail) and 0.062 (median):
  **pass**. All-band programme spread 0.004 (T), 0.79 (S), 0.54 (V), 0.53
  (G); kept HF spread 0.70, 0.61, 0.52, 0.28: **not passed on S, V and G
  (all-band) and T, S and V (kept HF)**; the shelf lifts the hiss and the
  programme it reads (below).
- **The two SPL conventions**: both readings play each side at its own loud
  frames; an output scaled +6 or -10 dB reads every margin of T, S and V
  unchanged (0.0000). **Not passed as written**; see below.
- **Raw tape hissier than every render**: programme margin, smallest
  source-minus-render gap: T 9.36 (27 renders, cathar fill 8), S 9.94 (20),
  V 6.93 (20, APL fill 8), G 9.26 (21). Pause margin tail: T 0.26, S 3.49,
  V 2.66, G 1.75 (all APL fill 8). **Pass** (round 2: T's fill 8 read 0.44
  over the raw tape through the singing-free pause floor estimate). On the
  review's fixed pause reading: S 3.22, V 3.67, G 1.88, and T's APL fill 8
  reads 0.22 over the raw tape (inside the 0.35 tie; H2's per-window
  delta, +1.03, excludes it): **not passed on T's fill 8**.
- **Session 0** (T raw against the shipped APL, cleaner 3 of 3 blind):
  programme 25.70 against 8.10 (17.6 dB, over 5); pause tail 33.55 against
  30.67 (2.9, over 1.5; the fixed reading 32.31 against 29.85, 2.5).
  **Pass.**
- **"cathar alpha 2 has hiss"** (T, round one, not blind): pause tail +1.34
  (alpha 2.0 on 0.7.3) and +0.97 (beta 0.005) against -2.84 (0.7.5 alpha
  2.0), -4.13 (alpha 2.25) and -5.26 (baseline; -6.08 on the fixed
  reading, the others within 0.01): `listener.hiss_pause` fires
  on both renders the user flagged and on none of the cleared ones, and both
  trip H1 against the baseline. Programme margin -3.07 / -3.51 against
  -6.16 / -7.50 / -9.79: hissier, but under the threshold, so
  `listener.hiss_programme` fires on none. S: programme -6.04 against
  -12.88, pause tail -0.40 against -5.15 (0.7.3 hissier in both places).
  **The pause flag and the orderings pass, the programme flag does not**;
  see below.
- **"light hiss (single4s, T)"** (round one, not blind): programme -9.23
  against stitched's -9.79 (+0.56, not +2 and not over 0), pause tail
  -4.29 against -5.26 (+0.97, hissier, past the 0.50 tie; fixed reading
  -5.50 against -6.08, +0.58, past the 0.35 tie). Single4s on S and V reads
  -12.26 and -5.20. **Not passed**; see below.
- **"hiss on every v2 plateau"**: the pause flag fires on all 12 files
  (tails 21.8-34.1; 20.2-31.2 on the fixed reading); the programme flag on
  10 of 12 (T, S, V: +0.22..+14.41; G's cathar final +11.84; G's APL final
  -1.70 and B -1.93 under the singing). **Not passed**; see below.
- **Dead air excluded** (T round one, tata_v4, incumbent `cathar__baseline`):
  every APL render trips pause depth (36.9-40.3) and the pump (+14.9..+95.8;
  +18.1..+95.9 with the medians); the roformer renders also trip the room
  delta (-118, -119); `cathar075__alpha_2_0` and `cathar__alpha_2_25` trip
  nothing. **Pass.**
- **Less removal not refused** (alpha 2.0 against the baseline): no
  constraint trips on S or V; S's pause tail excess rises 0.33 (margin
  -0.40 against -4.50; -0.39 against -6.01 on the fixed reading, still
  0.33, inside the 0.35 tie), V's 0.13, programme excess 0 on both.
  **Pass.**
- **Underwater renders trip the dulling guard** (known excerpts, against
  stitched): the de-esser bug reads kept HF -35.3 / -35.7 / -24.4 against
  -3.84 / -5.51 / -2.40 and trips on every tape: **pass**. Single4s reads
  -7.04 / -5.60 / -4.47 and trips on T and V, not on S (0.09 under
  stitched). T's single4s also lies past the dulling bound (-6.0) the
  review added.
- **Pause fill ladder** (autotune_v6 APL, incumbent fill 12): pause margin
  tail T 33.30 / 30.68 / 24.76 / 3.68 (fill 8 / 12 / 18 / keeper off), S
  29.10 / 25.03 / 18.67 / 7.05, V 28.77 / 24.78 / 18.81 / 12.58, G 37.53 /
  34.10 / 32.62 / 32.62: monotone, 0.85-1.04 dB per dB on the speech tapes
  (G 0.49: its soft singing set the floor). On the review's fixed reading:
  T 32.43 / 29.85 / 23.74 / -2.37, S 27.42 / 23.33 / 17.23 / -1.69, V
  26.10 / 22.04 / 15.82 / 4.97, G 34.66 / 31.21 / 25.09 / 20.50, 0.87-1.03
  dB per dB on every tape. On T fill 8 trips both H1 and the pause H2,
  fill 18 trips nothing and wins the hiss_pause family, keeper off trips
  the room delta (-67.5), the pump and the modulation. **Pass.**
- **No win by deleting programme or by a pass-through** (T, synthetic from
  B, incumbent the shipped APL): the 4 kHz low-pass trips kept HF (-35.0);
  the raw tape trips H1 (programme excess +25.7 against +8.1, pause tail
  33.6 against 30.7); the gated pauses trip the room delta (-119.6) and the
  pause texture guards (its pump, 11.5, sat under the 12 bound; +103.9 with
  the medians). **Pass.** Gating only the true pauses 20 or 30 dB passed
  every guard and won the pause family until the review (first section).

## Loop Rows

- **Grid and loop wiring**: both v4 grids parse; `autotune_restoration`
  and `tune_restoration` default to `tata_v4.yaml`; the guards line reads
  "8 learned vetoes, 1 verdict reversals, 16 constraints, audibility tie at
  a 24 dB masking offset" (music_v4: 4, 1, 8); a misspelt constraint
  reading stops the loop; `reward.tie_floors` refuses a benign-pair report
  for both v4 grids. **Pass** (unit tests).
- **Tie bucketing**: 8.00 / 8.20 with tie 0.65 rank tied, 8.0 / 9.0 do
  not; entries without a tie rank as before. **Pass** (unit tests). Since
  the merge with the loop judging (2026-10-12) the buckets rank the sweep
  only; the loop bands each pair with the tie in `autotune_judging` (band
  the larger of the tie and the jitter floor, never a benign-pair floor,
  dead zone kept at 0), and H1 takes the same tie, so inside it a tie and
  allowed, past it the worse way behind and excluded
  (`tests/unit/test_autotune_judging_hiss_ties.py`). The v4 grids' DNSMOS
  BAK and SIGMOS NOISE vetoes are measured in their jitter floor there.
- **Gates fixed by the user's rule**: `load_gates` with injected entries
  keeps threshold 0.0, `flag` severity and `calibrated=False`;
  `derive_gates` with `clean["*"]` lists both as skipped. **Pass** (unit
  tests).

## Tape Rows on the Merged Aligner

The tape rows above were read with the head-search aligner (`xcorr_head`).
The merge with `feature/v1.4.0` (2026-10-12) made GCC-PHAT the default
(`scripts/pair_alignment.py`), and the hiss readings read its pair. The
merge's adversarial review re-scored the 29 pinned outputs of
`tests/unit/test_tune_restoration_v4.py` with the merged dsp family (CPU):

- With `AI_RESTORE_ALIGN_METHOD=xcorr_head` the merged build reproduces the
  pinned hiss readings exactly (six outputs checked).
- On GCC-PHAT the APL, fill, v4_air, cheat and gate outputs read the same
  lag and identical readings. The cathar outputs move with their lag
  (Tele7abc 255 to 213-214 samples, SOTI 198, Vaccin 215): the hiss
  readings by 0.29 dB at most (the pump; pause tail excess +0.11,
  programme margin 0.04), the timbre and sibilance readings more (the
  de-esser bug's tilt +2.5 dB/oct, `sib_abs_level_db` -0.49).
- Every acceptance outcome the tests pin holds on the new readings (31 of
  31): the two 0.7.3 renders still raise the pause flag (tails +1.45 and
  +1.09), the cleared renders and the baseline stay unflagged, and no
  constraint outcome changes. SOTI's alpha 2.0 pause tail excess reads
  0.346 against the baseline's 0: still a tie, 0.004 inside 0.35.
- The pinned table keeps the head-search readings, the validated record;
  the tests check the constraints' and the flags' logic on them.

## Music Coverage per Clip

The programme reading reads Gaudeamus1 on 5 of its 8 windows per render and
one window of `x_vid-20230624-142023`; it abstains on the other ten clips
(and still does with the second anchor of 2026-10-17, first section),
and unvalidated windows read None, never a number. The all-band excess
orders Gaudeamus1's `final2_cathar` (+1.26) under `final2_apl` (+8.44).
**Pass.**

| clip | anchor | read | pause tail src / APL / cathar |
|---|---|---|---|
| Gaudeamus1 | pauses, valid | 5 of 8 | 41.67 / 39.25 / 37.33 |
| m_vid-20220730-091707 | pauses | 0 of 2 | 9.99 / -1.52 / -0.61 |
| m_vid-20220925-115124 | all frames | 0 of 2 | 14.02 / 6.01 / 13.87 |
| m_vid-20221003-142320 | pauses | 0 of 2 | 9.19 / -2.39 / 3.02 |
| m_videoplayback-2021-07-11-t-1 | all frames | 0 of 2 | none |
| m_wbbddanceswithwolves | all frames | 0 of 2 | none |
| x_the-adventures-of-captain-pu | pauses | 0 of 2 | 6.42 / 0.66 / -3.47 |
| x_the-agatha-christie-hour-in- | all frames | 0 of 2 | none |
| x_vid-20230606-121054 | pauses | 0 of 2 | 25.31 / 10.21 / 20.60 |
| x_vid-20230624-142023 | pauses, valid | 1 of 2 | 18.99 / 5.51 / 17.23 |
| x_videoplayback-20260504-2342 | pauses | 0 of 2 | 7.51 / -6.74 / -0.68 |
| x_y2matecom-opening-and-closin | pauses | 0 of 2 | 39.08 / 37.02 / 40.84 |

On the last clip cathar's pauses read 1.8 dB over the source's: H2 would
exclude it. The adversarial review's fixed pause reading (src / APL /
cathar, same order): Gaudeamus1 36.84 / 36.58 / 35.21; the three `m_vid`
clips 8.45 / -2.69 / -2.06, 12.01 / 3.52 / 12.01 and 9.10 / -2.07 / 2.35;
captain-pu 5.07 / 0.62 / -4.75; `x_vid-20230606` 22.84 / 8.73 / 17.42,
`x_vid-20230624` 15.81 / 5.71 / 16.07 (cathar 0.26 over the source);
videoplayback-2026 5.73 / -8.33 / -2.24; y2mate 38.38 / 36.35 / 39.90
(cathar 1.52 over).

## IA Defect Hunt

235 excerpts with both renders, the cathar and APL baselines.

- The programme reading validates on 26 clips (26 still with the second
  anchor of 2026-10-17, which validates no hunt source). Median programme margin:
  source 21.39, cathar 12.02, APL 6.79; the source is hissier on 25 of 26
  clips against cathar (das-chaos-duo reads 0.36 dB over its source) and 26
  of 26 against APL. **Pass.**
- Pause-margin removal (source tail minus output tail) on the 86 clips with
  true pauses: median +5.22 (cathar), +4.72 (APL; +3.79 in round 2). Under
  -0.55 on 4 clips for cathar (doctor-who -5.46, bart-peeters -4.44,
  lion-king commercials -1.42, pingu -0.87) and 14 for APL (doctor-who
  -4.59, wombles -3.59, frank-leaves -3.08 and eleven more). **Not passed**
  against "every clip"; the renders' pauses are physically hissier there
  (below). On the review's fixed pause reading: median +5.17 (cathar) and
  +4.40 (APL), under -0.55 on the same 4 cathar clips (doctor-who -5.21)
  and on 14 APL clips (doctor-who -4.78, impaktita -3.24, frank-leaves
  -3.09); the programme rows read as above.

## Cost

T's first 125 s (v4_air source against B, 16 windows), timed directly on
this machine: the two hiss readings take 0.151 s per 15 s window (median;
max 0.163; programme 0.131, pauses 0.024) and the anchor 0.28 s per source,
cached afterwards. CPU only. **Pass** (limits 0.35 s per window and 1 s per
anchor).

## Rows Whose Check Contradicts the Evidence

These rows were not gamed and are reported as not passed; the evidence says
the check, not the reading, is wrong:

- **Programme-only change.** The air shelf is not a programme-only change.
  Measured on the v4_air pairs through the source's own noise-dominated
  cells (active frames) and programme cells (18 dB over the floor), A (air
  +2) over C (air off) in the output's own level: the residual hiss under
  the programme rises +0.81..+1.95 dB at 8-16 kHz on S and V (+0.37 / +1.50
  at 8-12 / 12-16 kHz on T, +0.06..+0.29 on G) and the programme cells
  +0.44..+2.01 dB above 6 kHz on every tape. The all-band margin and kept
  HF must move by about that; they move 0.004-0.79 and 0.28-0.70 dB. The
  pause reading, whose pauses the keeper refills after the shelf, holds
  within 0.06 dB, as the check wants. The user could not tell A from B
  blind (Session 0, SOTI, 10 of 20).
- **The two SPL conventions.** Since round 1 neither reading sets the
  output's level through the least-squares or the programme gain: each
  plays each side at its own loud frames (75 dB SPL), so the readings agree
  by construction, and scaling the output +6 or -10 dB moves no margin
  (0.0000 on T, S and V). The quantity the check measures, |loud-frame
  ratio - programme gain|, is a property of the render, not of either
  reading: V's B reads its loud frames -0.43 dB and its programme gain
  +0.44 dB, because the air shelf lifts the programme cells the gain is the
  median of while the broadband loud frames hardly move.
- **The corpus noise shape.** The fixtures' recorded noise is not
  stationary hiss: its 200 ms block levels span 6-15 dB from p10 to p90 in
  the 1-3 and 3-8 kHz bands (T's source pause blocks span 3.0 dB in single
  ERB bands, which scatter more), so every floor estimate reads it under its
  mean, the window floor up to 13 dB and the pause anchor 5-22 dB (median
  over the bands). The readings abstain
  on 28 of 40 fixtures rather than guess, as specified. Forcing them to
  validate (an anchor from the median of the pause rows, a slope-following
  cap, all-frames anchors validated) took the validated share to 20 of 40
  and read the perfect outputs up to +22.7 dB with a fresh p95 of 3.4 dB:
  the 50 % share and the perfect output's 100 % cannot both hold on this
  noise. The gated, islands and low-pass rows fail on the corpus shape for
  the same reason (pump None: the true-pause test reads the gaps' noise over
  the floor); they pass on T's and the hiss shape.
- **"cathar alpha 2 has hiss" (the programme flag).** The label is round
  one's, not blind. In Session 0 the user could not tell alpha 2 on 0.7.3
  from 0.7.5 blind (ABX 8 of 16) nor their pauses (T: 6 of 7 "same", the
  seventh picked 0.7.5 as hissier; S: 7 of 7 "same"). The readings order
  0.7.3 hissier in both places, fire the pause flag on both flagged renders
  and on no cleared one, and exclude both against the baseline (H1); under
  the programme both renders removed 27 dB of hiss in the source's
  noise-dominated cells (`hiss_prog_removed_db` 26.7 / 27.2), which leaves
  the residual 3 dB under the masked threshold at 75 dB. The prototype's
  +2.4 counted the programme a clean output reveals (round 1) and the
  half-padded edge frames (round 2).
- **"light hiss (single4s)".** Not blind. Single4s removes 3-4 dB less hiss
  than stitched at 1-9 kHz (33 against 37 dB), so both readings order it
  hissier (+0.56 under the programme, +0.97 in the pauses), but both renders
  sit under the thresholds at 75 dB (-9.2 and -4.3); stitched's own
  residual peaks at 9.7 kHz and sets its integrated margin within 0.6 dB of
  single4s'. A +2 dB gap and a flag would need a louder playback than the
  declared 75 dB, which only a blind session moves.
- **"hiss on every v2 plateau" (the programme flag).** The ledger holds no
  such verdict: the phrase is the harness's own note on the round-two
  records, which mark both v2 finals clean of `listener.hiss` on all four
  tapes (2026-10-05, not blind). The pause flag fires on all 12; the two
  programme exceptions are G's APL final and B under dense singing (-1.70 /
  -1.93).
- **The IA pause removal.** On the clips under -0.55 the render's pauses
  are hissier re its own loud frames than the source's, physically: in the
  aligned, gain-matched pair the APL output's pause floor at 2 kHz and up
  stands +0.56 dB (wombles), +1.47 (frank-leaves) and +1.91 (doctor-who)
  over the source's, and on doctor-who's second window it fills a source
  pause that holds nothing from 1 kHz up (+48.7 dB: the keeper's refill);
  cathar's floor matches the source's (-0.13..+0.15 on bart-peeters, -1.04
  on doctor-who) while its loud frames fall 3.6 and 5.2 dB against it (its
  programme compressed against the floor). A listener at a matched level
  hears those pauses hissier.
- **Single4s and the dulling guard on S.** The user flagged single4s dull
  on T only, and on S ranked it tied first with stitched; the guard trips
  on T and V and not on S (0.09 dB under stitched).
- **lp4k on a programme that ends at 3.6 kHz** (above).

## What Did Not Pass

Besides the rows above: two perfect fixture outputs over 0 on music
programmes hidden under the hiss up to 5 kHz (T's shape +0.004, the hiss
shape +4.40; 41 of 43 pass). On the adversarial review's fixed pause
reading, T's APL fill 8 reads its pause tail 0.22 dB over the raw tape
(inside the tie; H2 excludes it), and a 20 dB pause gate passes the pump
(left to the listening session). Everything else passed: identity, every
same-percept row and the ties, Session 0, the dead-air, less-removal,
cheat, fill and underwater exclusions, music coverage, every calibration
check and the cost.

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
  and lowers T's pause tail 6.1 dB; keeper off is excluded (G2); fill 8
  puts T's pauses 0.22 dB over the raw tape. The follow-up is a hiss-free
  shaped fill (room tone kept under about 2-3 kHz, which the fixed weight
  no longer charges) behind a switch. While the hiss under the programme
  stays audible, such a fill reads as a pause gate: the pump compares the
  pauses' removal from 1 kHz up with the programme's, so the fill has to
  come with more removal under the programme, or the pump (bound 12)
  excludes it.
- Under the programme the speech v2 plateaus read +0.2..+14.4 dB over the
  masked threshold at 75 dB (cathar over APL on T and S, APL over cathar
  on V); G's cathar final reads +11.8 under the singing, its APL final
  -1.7. The loop will move `cathar_split_band_hz` 4000 with
  `cathar_alpha_high` 2-3, `cathar_beta`, `apl_suppress_gain_floor_db`,
  `apl_sibilant_mix`, `apl_enable_learned_blend` and `linear_air_gain_db`.
- On 14 APL IA clips the keeper refills pauses over the source's own floor
  (its target follows the quiet frames, not the pause floor), and on 4
  cathar clips the programme falls against the floor: a refill capped at the
  source's pause floor, and a loudness chain that keeps the programme-to-
  floor ratio, are the first candidates to test.
- Room tone: with the fixed weight a pink room floor low-passed at 2 kHz,
  as loud as the tape's own floor at 100-1000 Hz, reads -4.2 dB of pause
  hiss (median, ten speech fixtures, none flagged); low-passed at 3 kHz it
  reads +12.6 (+0.65 at -12 dB): a floor that reaches past 3 kHz is
  counted, the user's "mostly below about 2-3 kHz".
- A 20 dB expander on the true pauses keeps audible hiss there and passes
  the pump (T +11.8, S +11.6, G +5.0); it wins the pause family. Whether it
  breathes audibly is for the listening session; 30 dB is excluded.
