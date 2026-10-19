---
name: output-quality-harness
description: >-
  Judge a restored output the way a listener would (the "AI human ear"),
  calibrate the judge, tune both engines on real tapes with it, and run the
  self-driving tuning loop until it plateaus before asking the user to listen.
---

# Output-Quality Harness

Use this skill whenever an output has to be judged, an engine setting has to
be chosen, or the user asks to "fine-tune" cathar or `auto_pure_linear`. The
harness is the arbiter; the user's ear is the final judge and is asked only
once the loop cannot improve any further. `docs/validation.md` carries the
full description; this file is the operating manual.

## What the harness measures

`scripts/validate_restoration.py SOURCE LABEL=PATH... --metrics all` scores
every output against its own source over 15 s windows (7.5 s hop), as paired
deltas (output minus source) with a median and a bad-end tail, in four
families:

- `dsp` (no model): presence and air on the loud frames (muffling), a
  kurtosis ratio on quiet frames (musical noise), clicks (an impulse must
  stand 12x over its 50 ms floor and clear -60 dBFS), holes, the 15.6 kHz
  line, mains harmonics, LUFS / LRA, the trade metric, and the pauses:
  pumping (spread of the quiet frames' level; a source pause moves 4 dB,
  denoisers 11-23 dB) and tilt (low-band residual minus high-band; rumble
  kept and air taken reads positive, hiss left reads negative).
- `stems`: the app's BS-RoFormer splits both; SI-SDR, log-spectral distance,
  worst octave and envelope correlation on the non-vocal stem, only where the
  source stem carries a background (above -45 dBFS, within 20 dB of the mix).
  The gate and `music_v3` read the worst octave on the programme the source's
  background leads (`stems.octave_programme_db`, "Gates that refused the
  denoising itself"); the raw octave is displayed only.
- `speech`: Whisper large-v3-turbo CER / WER / log-prob (per-tape floor, the
  tail gate is soft: hissy tape makes Whisper unstable), WavLM speaker
  cosine, UTMOS.
- `mos`: SIGMOS (P.804 colouration / discontinuity / noise / ...), DNSMOS
  (cross-check only), Audiobox PQ / PC / CE / CU.

Windows are routed speech / music / mixed / silence. The app's scanner ratio
reads 0.03-0.06 on VHS music, so the router also reads tonal persistence
(prominent spectral peaks held for eight 93 ms frames: instrumental 0.02,
speech over a bed 0.005-0.011, dry speech 0.000-0.001) and syllabic
modulation (3-8 Hz envelope density over 0.5-3 Hz) to tell singing from
instrumental. Speech metrics run on speech and mixed windows only.

Weights live under `models/<name>/` with pinned revisions and recorded
hashes: `scripts/download_quality_models.py --set all`, `--check`. SIGMOS is
fetched from `github.com/.../raw/<commit>` (raw.githubusercontent serves an
LFS pointer). SCOREQ is loaded, not dropped: `mos.scoreq_nr` reads the
vendored ONNX export `models/scoreq/adapt_nr_telephone.onnx` (the package's
"natural" no-reference model) through `onnxruntime-gpu`. Never
`pip install scoreq`: the package pulls plain `onnxruntime` beside
`onnxruntime-gpu`. Which SCOREQ variant URGENT 2024 evaluated is not
verified, so its listener agreement is not claimed for this one: it is a
median-only veto. MERT (`stems.mert_dist`) has CC-BY-NC-4.0 weights, kept
for research use by the user's decision of 2026-10-08.

## Rules learned on real tape

- Learned MOS predictors are guardrails, never objectives (URGENT 2024: the
  teams that topped DNSMOS ranked bottom with humans). Rank-aggregate across
  metrics, veto with hard gates, show the trade metric, never rank it.
- Calibrate before trusting: `scripts/calibrate_quality_metrics.py` moves
  every metric on controlled degradations and on the known-ordering set of
  real outputs, and derives `experiments/quality_calibration/gates.json`.
  A detector change is followed by a `--metrics dsp` re-check into a
  separate `--out`, never over the gates in use.
- Read HF ratios on source-loud frames only; pauses confound them.
- Block DC before comparing (the loader does): a capture can be almost
  entirely offset (three IA music clips at +0.49, programme 23-27 dB below),
  and against the raw source the app's correct 2 Hz blocker scored as if it
  had destroyed the music (SI-SDR -42, octave -58). When a whole clip scores
  absurdly, check the source before the engine. The same captures are 96 %
  power below 80 Hz (5 Hz harmonics): the router calls such a window silence.
- Waveform readings punish what the ear does not hear: cathar's 80 Hz dewind
  shifts phase on bass-heavy programme and SI-SDR goes to -0.8 correlation;
  the worst-octave reading starts at 125 Hz for the same reason (both engines
  high-pass at 80 Hz by design). Rank magnitude readings (LSD, octave,
  envelope, HF), show SI-SDR.
- Before a tuning set is used, scan its sources: DC share, sub-80 Hz share,
  route; drop degenerate captures rather than letting the loop chase them.
- What the user hears and the standard metrics miss has to become a metric:
  "silent pauses" became pumping and tilt, a dither-scale "click" at
  -90 dBFS became the absolute click floor. Measure the specific thing on the
  specific file first, then generalise.
- A synthetic test voice with zero-phase harmonics is a pulse train; randomise
  phases. A chord's beating reads as syllables; use a harmonic series.
- cathar's stitched noise probe engages only from 20 x
  `cathar_noiseprint_duration_s` (6 s -> 120 s); tuning excerpts are cut at
  125 s, whole clips shorter than that keep the 0.75 s probe.
- Whole tapes are the ground truth: an excerpt's noise probe is not the full
  tape's. Confirm any winner on the full tapes.
- A scored document names its aligner (`alignment.method`). Scores written
  before 2026-10-11 carry none and were aligned by the head search
  (`xcorr_head`): re-score them before comparing with new ones, or compare
  like with like under `AI_RESTORE_ALIGN_METHOD=xcorr_head`. An abstained
  alignment (`alignment.status`) is a reading the scorer cannot vouch for:
  leave such clips out of a ranking rather than trusting their readings.
  A one-family re-score (`tune_restoration.merge_families`, also behind
  `validate_restoration.py --merge-into`) keeps the other families only
  when the pair is trimmed at the stored `lag_samples`; where the aligner
  moved the lag it keeps the fresh families alone, so re-score the rest
  rather than read a document that mixes two timelines.
- The hunt's `dsp.output_silent` failures are mostly the gain match, not
  the lag (2026-10-11, CPU). Of the 10 (6 cathar, 4 APL), the vote aligns 7
  at the true lag and 6 of those still fail; 3 abstain (no phase coherence
  on any span; APL's et-el carries a 431 ms sync shift, outside the
  search); one, a DTW clip the head search read at -1499 samples, passes
  once aligned. `align_pair`'s least-squares gain is the waveform
  correlation times the level ratio: on the five measured, the outputs
  correlate 0.09-0.13 in magnitude with their source at the true lag
  (phase coherence 0.55-0.95 in 150-7000 Hz), so the gain puts them
  18-21 dB under their energy-matched level and their windows route as
  silence. Fixed on 2026-10-12: the output is routed on an RMS-matched copy
  (`audio_io.output_routes`); on the re-scored hunt all 12 failures (8
  cathar, 4 APL) cleared, and a window an output really drops still reads
  as silence.
- The listener readings (v2, 2026-09-23) exist because the median level of
  the quiet frames could not tell "silent in pauses" from "has hiss" (every
  restoration drives the deepest pauses to -96..-104 dBFS): read the gaps
  (`dsp.gap_air_db`, 3-10 kHz on the p15-p40 frames over the loud frames)
  and the pause depth relative to the speech (`dsp.pause_depth_db`), both
  immune to the gain match; read the 's' on the source's fricative frames
  (`dsp.sib_centroid_hz`, net of the plain-frame shift; hiss frames have a
  high centroid too, so a fricative must also stand 10 dB over the gaps'
  4-12 kHz power); read music attacks on the source's onsets
  (`dsp.attack_db`, level-normalised). Their gates are `flag`s
  (`listener.*`): the loop counts them (`_beats` refuses more flags), they
  do not veto until a second listening round confirms the thresholds, and
  the calibration derives them from per-gate `flags` / `clean` label lists
  in `known_ordering_v2.json` (a single by-ear rank cannot say "hissy but
  not dead"). Split a flag when one direction is accepted by ear and the
  other is not (`sibilance_thin` above +300 Hz, `sibilance_dull` below
  -600: cathar's de-esser reads -100..-380 and nobody objected).
- Second listening round (2026-10-05, the v2 plateaus in
  `D:\Tata\New folder\variants\v2`): APL preferred over cathar on the Tata
  speech tapes, the pauses natural on both (the pause floor confirmed), the
  music clips fine on both, and APL's 's' still thin. The harness read
  `sibilance_thin` only on Vaccin, so the flag is right in direction and may
  under-call; the loop's sibilance grid stopped at its own top (mix 0.8) and
  never moved the guard's 4 kHz crossover, below which the harness locates
  the emptied body. `experiments/run_autotune_v3_apl_sibilance.cmd` reruns
  APL from the shipped defaults with mix up to 1.0 and the crossover 2-5 kHz.
  A plateau sitting on the edge of a knob's grid is not a plateau: widen the
  grid before asking the user.
- A plateau whose outputs barely differ is not a result. The v3 sibilance
  loop (2026-10-06) accepted mix 0.9 then 1.0 and stopped, but its final
  differed from the v2 plateau by -70..-100 dBFS on 0.5-32% of the samples:
  the guard's fricative detector (`HF_SHARE_MIN` 0.5) found 3 events in five
  minutes of Vaccin where the harness found 30 (12.5% of its fricative frames;
  tape rolls the highs off, the harness's fricatives carry a median 0.45 of
  their energy above 4 kHz). Before asking the user to listen, diff the final
  against what they heard (samples changed, level of the difference); a stage
  that only acts on detected events needs its detector checked against the
  harness's own. `apl_sibilant_hf_share_min` is now a knob
  (`experiments/run_autotune_v4_apl_sibilance.cmd`).
- Third listening round (2026-10-08, `D:\Tata\New folder\variants\v4_air`):
  the thin 's' on APL was the air shelf, not the neural stage. Measured on
  Vaccin's fricatives the output moved no more than on its other loud frames
  (under 1 dB per band), while the whole output sat ~+1.6 dB brighter above
  4 kHz; the user chose `linear_air_gain_db` 1.0 over 2.0 and off. The
  harness's `sibilance_thin` reads the centroid net of the plain frames, so a
  tilt that lifts every frame's top does not register: a complaint about the
  's' that the loops cannot move is tested by ear with an A/B of the
  whole-spectrum stages (air, expander) before the detector-bound ones. Ear
  v3 (below) gives that reading its absolute twin, `dsp.sib_abs_level_db`.
- Re-scoring one family into a stored report:
  `validate_restoration.py ... --metrics dsp --merge-into report.json`
  (`score_listen.py <slug> --dsp-rescore`), then
  `experiments/tata_listen/check_verdicts.py` must reproduce every verdict
  the user gave before a reading is trusted.
- Learned judges stay guardrails: SCOREQ (`mos.scoreq_nr`, vendored ONNX,
  never `pip install scoreq`: it drags plain onnxruntime beside
  onnxruntime-gpu), `mos.dnsmos_gap`, `speech.ssl_dist`, `stems.mert_dist`
  (MERT under transformers 5.17 returns no hidden states; the last hidden
  state is layer 12). Zimtohrli's PyPI wheel is published by a third party;
  the user approved installing it on 2026-10-08 (its dependency check found
  numpy only), but it is not provisioned yet, so the loader raises
  ImportError and the runner records the reading unavailable. No audio-LLM
  judge inside the loop.
- Speech and music want different engine settings (cathar: alpha 1.5 with
  the stitched print on dialogue, alpha 0.5 with no print on music), so a
  tuning result is only as good as the material gate that applies it. Before
  proposing one, run the app's own scanner on the tuning set, the identity
  clips and the speech tapes (`_extract_profile_from_signal`) and look for a
  reading that separates them with margin: the band ratios read every music
  clip as dialogue; the whole-file median tonal persistence separated the
  sets 0.064-0.225 against under 0.033. The gate lives in the app
  (`modules/tonal_persistence.py`, `cathar_music_persistence_min`) and the
  harness router imports the same functions.

## Ear v3 (Phase 0, 2026-10-09)

The plan approved on 2026-10-08 rebuilt the judge around three blind spots
the user's ear found and the v2 harness could not see:

- **The air-shelf tilt.** The +2 dB shelf at 7.5 kHz lifted the whole APL
  output about 1.6 dB above 4 kHz on every frame, and the user heard the
  's' as thin. No reading moved. The sibilance readings are net of the plain
  frames and the gap readings are relative to the same side's loud frames,
  so a change that lifts every frame cancels in both. The least-squares
  gain match can absorb a tilt. `tata_v1.yaml` and `tata_v2.yaml` rank
  `dsp.hf_4k8k.delta.median: up`, and the HF gates are one-sided (-8 and
  -12 dB). Each of the 16 rounds of the four APL loops scored a +1 dB
  candidate: it ranked behind the +2 dB incumbent in 15 and lost to another
  accepted move in the 16th (`experiments/autotune*/apl/state.json`). The
  user chose +1 dB by ear.
- **The guard's coverage.** At `apl_sibilant_hf_share_min` 0.5 the sibilant
  guard found 3 events in five minutes of Vaccin where the harness found 30
  (12.5% of its fricative frames). Two loops tuned the guard's mix and
  crossover on a stage that changed the audio by -70..-100 dBFS.
- **The hiss flag that never moved.** `listener.hiss` passes at an output
  `gap_air` of -20 dB or less. The round-2 finals the user heard as natural
  pauses read -8..-12; cathar alpha 2 ("has hiss", round 1) read -15..-18.5.
  Gap level alone does not order the verdicts, so the flag fired on accepted
  files too and no move could clear it.

Principles (plan Part 1), each closing one of those failures:

1. Rank the two-sided distance from what the user accepted,
   `weight * max(0, |x - target| - dead_zone) / scale`, never a raw
   direction. "More noise removed" is shown, never ranked.
1. Every net reading gets an absolute twin.
1. Timbre is read on programme cells only (source bin at least 12 dB over
   its own floor on loud frames), gain-matched on the 300-3000 Hz programme
   cells (`auditory.programme_gain_db`), not by
   `score_reference._match_gain`.
1. A per-source capture profile (R0) clips the readings and replaces the
   50 Hz and 15625 Hz assumptions.
1. Coverage before content: an event reading says how many events it
   stands on, and a stage whose detector covers few events is a finding,
   not a knob to tune.
1. Audibility before ranking: a candidate inaudibly different from the
   incumbent is a tie.
1. Learned judges (UTMOS, SCOREQ, SIGMOS, Whisper CER, WavLM, MERT) are
   median-only vetoes beyond their benign floor, never in the score (the
   plan said 3x near-copy floors; since Session 0, 1x the floor of pairs a
   blind listener could not tell apart).
1. Speech, music and mixed have separate grids and thresholds.
1. A reading is admitted by a monotonic response on the calibration
   degradations and a benign floor. The user's verdicts are falsifiers, and
   each reading's target complaint and direction is written down before
   Round 0 (the critique: about 25 readings against 10-20 verdicts overfit
   if chosen by "reproduces the verdicts").

### The new readings

Every threshold and target below is **uncalibrated** until Round 0
(re-scoring the stored listening files) derives it on Tele7abc and tests it
on SOTI and Vaccin. Round 0 ran on 2026-10-09 and proposed values (see
"Round 0 (2026-10-09)"); after Session 0 the v3 grids took its grid notes,
while no gate or flag is adopted. The readings pass their synthetic tests;
their module docstrings hold the full measurements.

- **R0, the capture profile** (`source_profile.py`, `meta.*`, source side,
  displayed). Programme bandwidth (the highest 1/6-octave band where the
  loud frames stand at least 3 dB over the gap frames), the brickwall cut,
  the mains (50 or 60 Hz, evidence, excess), the CRT line (Hz, ppm, spread,
  hum correlation, origin `playback_chain`, `speed_locked` or `unresolved`)
  and the channel state (signed r, energy ratio, dual mono, dead channel,
  inverted). It is read once per source on the raw multichannel file and
  cached at `<cache>/profile/<source key>_<code hash>.json`; the hash covers
  the code it reads with (`runner.PROFILE_CODE`, `pause_metrics.py`
  included), so any change there, a docstring edit too, re-reads every
  profile once (about 1 s per source). Tata: Tele7abc band
  4490 Hz, brickwall 18.9 kHz, right channel dead (43 dB); SOTI 5040 Hz,
  15.4 kHz, right dead (51 dB); Vaccin 7127 Hz, no brickwall, r 1.00; all
  50 Hz. The Tele7abc and SOTI lines are steady (7 and 13 ppm), classed
  `playback_chain`. The hum and whistle readings take its `mains_hz` and
  `line_hz` (falling back to 50 Hz and 15625 Hz); the whistle reading is
  floored since 2026-10-12 (see the gates section below). The summed rule of
  `measure_hum._mains_by_evidence`, which the app's own mains detector
  shares, names 60 Hz on SOTI: there the 150-250 Hz speech peaks sit
  21-29 dB over a lone 50 Hz fundamental. R0 decides on the 1 s frames
  instead (at least 3 dB of evidence on at least 10 frames). That is
  evidence for F3, the source-profile-driven frequencies in the app.
- **R1, spectral balance** (`balance_metrics.py`: `dsp.balance_tilt_db_oct`,
  `dsp.balance_presence_db` 2-5 kHz, `dsp.balance_air_db` from 5 kHz to the
  band, `dsp.balance_body_db` 100-300 Hz; complaints: bright or thin, dull or
  underwater). Per 1-ERB band, the median output-over-source dB on
  programme cells after the programme gain match, clipped to R0's programme
  bandwidth. Two guards keep hiss out: a cell must also stand 18 dB over its
  bin's bias-corrected mean noise, and a band whose expected noise-only
  cells exceed 20% reads NaN. Without them, oracle hiss removal on
  `artifacts/realistic-v2` read as up to -15 dB of air. Shipped benign floor
  (p90 of that oracle): tilt 0.02-0.09 dB/oct, presence 0.01-0.12, air
  0.18-0.24, body 0.03-0.10 dB. `dsp.balance_top_db` (presence or air,
  whichever moved further) drives `listener.bright` and `listener.dull` at
  +1 and -1 dB. Limits: R1 reads voiced loud frames and cannot see the 's'
  (on the sibilance tests' fixture, 0 of 388 's' frames are loud frames,
  and a 12 dB dulling of the 's' reads 0.00). A reading needs a band of R1's
  own 1-ERB layout under the clip, not just its span's nominal start
  (`balance_metrics.bands_under`, `readable_under`, `readable_from_hz`): air
  gets its first band only under a 5292 Hz band, presence under 2125 Hz,
  the tilt its third band under 1408 Hz. So air is unread on Tele7abc
  (4490 Hz) and SOTI (5040 Hz), where the +2 / +1 dB / off shelf moves
  presence and tilt by 0.04 at most; Vaccin's air reads
  +0.43 / +0.17 / -0.05.
- **R1's mute guard** (`runner.holds_mute`, `mute_floor_db`, `median_floor_db`,
  `span_floor_db`). A window whose source holds an analog mute reads every R1
  reading None: there R1's quiet frames measure the mute, not the hiss under the
  programme, and read hiss removal as dulling. A mute is 5% of the window's live
  20 ms frames 15 dB or more under the source's floor, decided on the source, so
  every variant skips the same windows. On the runner tests' speech (15 s
  window, oracle hiss removal) a stretch 25-40 dB under the hiss reads air
  -19 dB and tilt -7 dB/oct once it fills 10.7% of the window and 0.0 at 10.0%;
  20 dB under misreads at 4 s but not 2 s, 15 dB under stays below 0.4 dB. The
  floor is the median of each programme window's p10 from seven windows up
  (`MUTE_MEDIAN_WINDOWS`; a mute up to 7.5 s touches three windows at most, so
  it cannot own it), and below that the higher of the median (`median_floor_db`)
  and the p25 of every live frame in them (`span_floor_db`), the median taking
  the higher middle p10 of an even count (`upper=True`; their mean put the floor
  half-way, -52.6 dBFS over a -65 dBFS lead-in under -40 dBFS hiss, and missed
  it): a mute sets the floor only when it fills more than half the windows and
  a quarter of the span. Each
  alone missed one: a 15 s pair has two windows, both over its last 7.5 s, so a
  2 s mute owned the median (floor -90 dBFS); a 4 s mute at 13 s owned the
  median of five windows on 37.5-40 s pairs and sat half-way in six on 45 s
  (-52.5 dBFS over a -65 dBFS mute; windows 0-2 read air -19 dB, tilt -6.4 to
  -7.4 dB/oct); a blank lead-in counts in the span of the window it opens, so
  4 s of blank before 11 s of speech, 8 s before 15 s, 10 s before 20 s and 60 s
  before 15 s owned the p25 (air -14 to -19 dB). Over 1-4 s mutes at -90, -70
  and -65 dBFS at every 0.5 s position on 15, 20, 30, 37.5, 40, 45 and 50 s
  pairs, the guard covers every misreading window except a 4 s mute in 15 s
  (24% at every depth). The median from five windows with the p25 alone below
  covered 88-91% of the 4 s cases on 37.5 and 40 s pairs; the median alone
  covered 17-32% of the 2-4 s cases on 15 and 20 s pairs at -90 dBFS (none on
  15 s at -65 dBFS). Windows skipped on
  material with no planted mute (2026-10-09, the median alone in brackets):
  realistic-v2 `_vhs` 0 of 920 (0), `_clean` 23 (0), `_target` 293 (95; 249 of
  them window 0 over the Piper lead-in); IA corpus 20 of 109 (6), three of them
  over two -86 dBFS stretches the median missed (3.5 s at 16.5-20 s in windows
  1-2, the 1.06 s opening in window 0), two the 8.2 s at -62 dBFS
  that open jakki-brambles (41% of the clip; the median skips them too, the p25
  alone did not; whether that is a mute or the tape's own hiss is open, and
  either way a reading is lost, not misread); the Tata sources 1 of 137 (1,
  Vaccin's last window). The cost: R1 reads the calibration's spectral-tilt
  cases on window 1 only for en, es and fr and not at all for it (de's 1122 Hz
  band clips the tilt either way). A p20 in place of the p25 keeps those windows
  (IA 14, targets 245 skipped) but misses the -86 dBFS stretches, 22% of the
  clip together (the 3.5 s one alone is 17.5%), so
  p25 stays: a wrong reading costs more than a lost one.
- **R2, the 's' read absolutely** (`sibilance.py`: `dsp.sib_abs_level_db`
  and `dsp.sib_texture_db`; complaints: thin 's', "distortion of spoken
  's'"). The first is the fricative frames' level change from 4 kHz up after
  R1's gain match, not net of the plain frames. The second is the roughness
  of the 's' spectrum (each cell's spread around its frame's median), output
  minus source. The runner clips both to R0's **brickwall**, not the
  programme bandwidth: the bandwidth is read on voiced loud frames, and no
  's' is one. With the brickwall, the v4_air set (+2 / +1 / off) reads
  +0.53 / +0.25 / -0.04 on Tele7abc, +0.27 / +0.14 / -0.01 on SOTI and
  +0.63 / +0.32 / +0.03 on Vaccin; clipped to the band it reads a quarter of
  that. The texture reads round 1's APL baseline, no_air and roformer
  ("distorted") at +2.18..+2.26 against +0.83..+1.34 for the cathar renders.
  APL no_air reads level +0.05 against the baseline's +0.72, so round one's
  "thin" on no_air was texture, not level. Benign floor: level 0.05 dB;
  texture under 0.05 for static filters and -0.1..-0.2 when hiss above the
  band is cleared. Since calibration v3 the level's gain is matched around
  each 's' (+-0.5 s of R1's programme cells, so a slow level rider is no
  change) and the texture needs 30 fricative frames a window (10 read
  noise); both changed Round 0 numbers (see "Calibration v3 with the Tata
  cuts"). `sib_peak_hz`, `sib_rel_amp_db` and the Romanian s-comma
  class are not built: they rest on a refuted claim until a verdict backs
  them.
- **R3, coverage** (plan: `meta.fricative_coverage`). The engine side is
  built: `AI_RESTORE_EVENT_LOG=<dir>` makes the sibilant guard, the plosive
  tamer, the pause floor and the mux write what they found (see the
  audio-restoration-engine skill). The harness reading that compares the
  guard's spans with `fricative_frames` (recall in frames, with one
  hand-labelled Vaccin minute as truth) is not built yet. Give every tuning
  candidate its own event-log folder: candidates of one mode share record
  names.
- **R4, the residual in the true pauses** (`pause_metrics.py`:
  `dsp.gap_atten_db`, `dsp.gap_slope_db_oct`, `dsp.gap_spread_db`,
  `dsp.gap_hf_excess_db` (4-10 kHz minus 0.2-1 kHz), `dsp.gap_lsd_db`,
  `dsp.gap_mod_dist_db`, `dsp.gap_island_kurt`, `dsp.gap_pause_s`;
  complaints: hiss, dead air, musical islands). The mask is every source
  frame at or under p40 inside a non-speech run of at least 200 ms. On
  Tele7abc the DSP VAD calls 89% of the p15-p40 frames speech, so a p15-p40
  mask would keep 0.7 s of 133 s. The DSP VAD averages each of four bands
  over 60 ms: single frames crossed the threshold on 5.4% of white noise,
  the average on 0.02%. `AI_RESTORE_PAUSE_VAD=silero` runs silero-vad once
  it is installed (approved, not provisioned); the card records which VAD
  ran. Tones (hum) leave both sides first, and the output is held no lower
  than 60 dB under the source per cell. Speech and mixed windows only,
  clipped to R0's brickwall. On the must-separate pair (Tele7abc, cathar
  alpha 2, only the binary differs) the HF excess reads +3.11 dB on 0.7.3
  (heard: hiss; the ledger's 0.7.6 is an erratum, see "Round 0") against
  +0.49 on 0.7.5 (clean); clipped to the programme band both read -0.32.
  Attenuation (42.3 against 42.8 dB), modulation distance and island
  kurtosis do not separate them. APL baseline (dead air) reads -13.7 dB HF
  excess. No threshold is set: `listener.hiss` is display-only, and Round 0's
  R4 flags are proposals. A gates
  file sets its threshold but not its severity (`gates.DISPLAY_ONLY`): the
  v2 `gates.json` (the drivers' default `--gates`, and what
  `score_listen.py` passes) still lists it as a flag, and merged as it stood
  it made the loops count it again (an output at gap_air -10 dB failed it
  as a flag). Bins whose source pause floor sits under the 16-bit dither
  plus 10 dB leave every R4 reading (`above_dither`): requantising alone
  had lifted Vaccin's notched top band 9-16 dB.
- **R7, gain riding** (`file_metrics.py`: `file.gain_ride_lu`, the p95, with
  `file.gain_ride_std_lu` and `file.gain_ride_deriv_lu`). EBU short-term
  loudness of the raw source and output over windows standing 10 LU over the
  source's floor, the static offset removed, the sync lag applied first.
  The accepted v2 finals of the speech tapes read p95 0.82-1.80 LU, so the
  design's 1.5 LU flag would fire on three accepted files: display-only
  until the `loudnorm_ride` degradation calibrates it. Each output window
  is read along the sync's lag track (the matched anchors' lags), so a
  speed drift is no ride either.
- **Sync** (`file.sync_drift_ms`, `file.sync_offset_ms`,
  `file.sync_unmatched`, the per-anchor lags). The envelope correlation of
  8 s anchors at the start, centre and end of the raw pair. Hard gates: drift
  and offset at most 40 ms (one PAL video frame), no unmatched anchor. The
  32 v2 final pairs read lags of 4.7-5.4 ms, drift 0.02-0.26 ms, none
  unmatched.
- **Alignment** (`scripts/pair_alignment.py`, under every paired reading;
  2026-10-11). `gcc_phat_spans`: GCC-PHAT (150-7000 Hz) on up to seven of
  the source's loudest 1.5 s spans, the lag at least two and half of the
  spans with coherence 0.1 agree on within 2 samples, else an abstention
  (lag None, trimmed at `align_prior_s` or 0, `alignment.reason` says why).
  The prior, or 0 without one, breaks ties between distinct peaks only:
  letting the highest peak's own shoulders contend pulled lags a sample or
  two short (review, 2026-10-12: of the 204 shift-synced hunt clips with a
  logged shift, exact on 175 cathar and 196 APL outputs once only local
  maxima contend, 169 and 177 before).
  Each scored variant carries `alignment` (method, lag, applied lag,
  polarity, spans, agree, coherence); `lag_samples` is the applied lag. The
  old head search (`xcorr_head`: |corr| over the first 1.86 s) stays
  selectable: `AI_RESTORE_ALIGN_METHOD=xcorr_head`, `--align-method`. The
  true lag is the mux's 219 samples plus the shift the app's sync stage
  printed in its run log; on the hunt's 205 shift-synced clips the head
  search read it within 8 samples on 137 cathar / 199 APL outputs, the vote
  on 203 / 202; the Tata shipped outputs read exactly (head search: 5 of 8
  off by 6-181).
- **Noise removed, per frame** (`file.noise_removed_median_db`, beside the
  pooled `file.noise_removed_db`). SOTI, shipped APL: the hottest 1 % of the
  quiet frames hold 34 % of their energy, pooled -5.17 dB, median +3.00;
  cathar +9.61 / +10.40.
- **R11, the programme's low end** (`lf_metrics.py`: `dsp.lf_programme_db`;
  complaints: thin bass, rumble left; built 2026-10-10 for round C3, the
  cathar music profile's dewind cutoff and repair). The output's 40-200 Hz
  power over the source's, summed over the source's loud frames, less R1's
  programme gain match (`matched_gain_db`: the median over
  `balance_metrics.guarded_cells` in 300-3000 Hz), with +-2 bins round each
  harmonic of the mains R0 names left out (none where R0 names none; no
  50 Hz fallback). Two STFTs: the gain match on 8192/2048, whose 0.19 s
  frames see the floor between notes its noise guard needs, and the band on
  a frame that follows the rate to hold about 1.35 Hz bins
  (`lf_metrics.frame_length`: 32768 at 44.1 and 48 kHz, quarter-frame hop),
  so a mains line takes +-2.7 Hz out. On the 8192 frame the same +-2 bins
  took 39-61 Hz out round 50 Hz, the band the dewind corners move: 40-60 Hz
  20 dB down read -0.007 dB on a bass-led bed (-0.99 on the band frame), the
  dewind corners leaned on the skirt above 61 Hz, and at 96 kHz 1 of 14
  band bins was left. The band keeps about 100 bins at any rate and reads
  None under 8. One frame for both failed: 0.74 s frames straddle the notes,
  and a bed gated 0.5 s on, 0.5 s off kept no guarded cell. Power, not R1's
  per-cell median: music's bass never falls to a floor in 15 s, so R1's body
  read 6 of 150 realistic-v2 music windows, and the per-cell median moved
  p90 1.3-1.4 dB on music under the oracle against the power's 0.35-0.47.
  Two-sided, target 0; noise in the band counts, so rumble left reads nearer
  0 than rumble removed and the accepted file's reading decides. None under
  100 guarded gain cells (two realistic-v2 beds ending at 1.3 kHz had 78,
  half of them hiss, and read +18.9 and +12.0 dB; the 12 such windows of 784
  moved p90 11.7 dB, the 20 at 100-200 cells 0.3 at most), under a 1/10000
  share of the core's power in the band, on a dead render, and where R1's
  mute guard skips (`_dsp_window` decides the mute once for R1 and R11).
  Calibration on the band frame (en / fr, floor 8.4e-6): FFmpeg's `bass` at
  100 Hz +1.5 / 3 / 6 dB reads +0.56 / +1.15 / +2.41 on the en bed (fr +0.62
  / +1.27 / +2.66), the cut the mirror; `highpass` 40 / 60 / 80 Hz -0.12 /
  -0.47 / -1.09 on the bed (fr -0.15 / -0.57 / -1.27), -0.014 / -0.073 /
  -0.224 on speech; hum at R0's mains 0.022 at most; a tilt above 1 kHz
  reads the other way (the gain match takes in 1-3 kHz); the 8192 frame read
  every row within 0.16 dB. Oracle removal of the 200 tape-noise-only
  fixtures' low-heavy noise (7 dB more per hertz at 40-200 Hz than at
  300-3000 Hz), 20 dB down: beds p90 0.15 dB (max 0.40), speech over a bed
  0.61 (max 3.5), speech 1.04 (max 2.5), against 0.17 / 0.77 / 0.70 on the
  8192 frame, so on speech R11 is shown, not ranked. Read on the 8192 frame
  and not again since: the 12-clip music listening set, APL median -0.50 dB,
  cathar music profile -0.68, cathar speech settings -1.28 (to -16.8);
  gaudeamus5's music windows, cathar round-one baseline -3.9, APL -0.74, the
  plateau finals -1.55 and -0.98. In no grid, gate or listening pick yet,
  and out of the stand-in grid `calibration_ordering.default_grid` builds
  without `--grid` (`RANKING_EXCLUDED`): there it would add every variant's
  low-end noise removal to its score, speech tapes included. The round's
  orchestrator decides.
- **R12, the time base** (`timebase_metrics.py`, backlog B1, 2026-10-12;
  complaint: the app's DTW sync playing 40 ms frames at 0x or 2x). Per file,
  on the raw pair at the runner's lag (`runner.timebase_entry`):
  `file.timebase_holds` / `_holds_per_min` (a 25 dB dip of the output's
  300 Hz high-passed 5 ms level against the source's, 20 ms or longer, the
  source not dipping 20 dB under its own 205 ms median, over which the output
  sat at one value: mean 10 dB over its spread, within 50 dB of its p99),
  `file.timebase_plateaus` (a 10 ms lag track, 60 ms windows over +-100 ms,
  confident at a peak of 0.85 and 0.1 over the best peak 2 ms away, displaced
  by exactly 40 or 80 ms from a 4.5 s running local lag),
  `file.timebase_rate_jumps` (confident lags at most 200 ms apart, through a
  five-frame median, differing by one or two whole frames),
  `file.timebase_speed_dev_pct` (shown) and `file.timebase_tracked_share`.
  Hard gates `timebase.holds` and `timebase.rate_jumps` at none,
  `calibrated: false`. IA hunt (`tune_ia_hunt_v3`, all 30 DTW and 205 shift
  clips per engine): cathar 177 holds on 24 DTW clips, APL 49 on 15, the
  gates fail 24 and 16; shift clips read nothing. The 32 accepted v2 finals
  read nothing but one hold on the APL final of the one music clip that took
  DTW, which round two heard as fine (unblinded): a single hold in 15 s of
  music may pass this listener, so the gate stays uncalibrated until a blind
  block. What each constant cost to choose, on the same data: a peak of 0.8
  read one cathar shift clip's pitch-lock excursion (11.4 ms, three frames)
  as two jumps and another a plateau; a rate jump of any 5 ms read a sound
  repeating 96 ms apart at the start of one shift clip (both engines) as one,
  while 136 of the 137 DTW jumps were whole frames; a 100 ms pair gap missed
  every 2x catch-up, whose windows stay unconfident 120-140 ms; the frozen
  test replaced a raw-over-high-passed rule that read 14 holds on four APL
  shift clips (digital silence holding the loader's DC residue, LF left
  where the highs were cut). The calibration's `sync_holds` case (1 / 3 / 6
  glitches) reads holds 1 / 3 / 6 on en and fr and on the three tape cuts
  (Vaccin, SOTI, Tele7abc); `sync_drift` on the stored v3b tape-cut cases
  reads its speed error and no event, though the drift outruns the track's
  +-100 ms search (tracked share 0.10-0.26 at 0.2 %); `SCORE_SCHEMA` 7.
- Read a signed correlation on the raw pair at the lag, never on the
  gain-matched pair: `score_reference._match_gain`'s least-squares scale came
  out negative on the 0.1 % slow en fixture (-0.016) and on the hunt's APL
  output of `frequently-asked-questions-about-time-travel` (-0.45), so the
  aligned output is inverted there. Magnitude readings do not care; R12's lag
  track went blind on both until it moved to the raw pair. Reading the
  correlation's magnitude instead read a periodic stretch half a period away
  (one shift clip: a 9.6 % speed error and a jump).

Also changed: the scorecard's direction "two-sided" (its tail is whichever
of p10 and p90 sits further from the neutral target), counts report `max`
and `p95` beside the median, every gate carries `calibrated`, and
`listening.DEFAULT_PICK_METRICS` and `tune_restoration.LISTEN_METRICS`
include R1, R2 and R4. The scored document carries `source.profile`.

### Audibility and the audio hash

`scripts/audibility_check.py INCUMBENT CANDIDATE [--json OUT]`
(`auditory.compare_files`) asks whether a candidate is audibly different at all.
It aligns the pair (lag within +-4096 samples, and the sign) and takes the
time-domain difference. Per 15 s window it compares the difference, per 1-ERB
band, with the incumbent's masked threshold: spread 27 dB per ERB downwards and
12 dB upwards, lowered by `MASKING_OFFSET_DB`, floored at the threshold in
quiet. The pair is audible when a window reaches `TIE_AUDIBLE_FRAC` (5%) of
frames with NMR over 0 dB, or holds 2 frames at +6 dB or more (a click or a
dropout, which the share alone called a tie), or any frame reaches +12 dB
(`SINGLE_EVENT_NMR_DB`), or when it cannot be vouched for: two different
multichannel counts (a mono side against a multichannel one is repeated on every
channel and compared per channel, not refused; `matched_channels`), NaN samples,
or more than 50 ms left uncompared once aligned (`length_mismatch_s` against
`LENGTH_TOLERANCE_S`: the |lag| head the alignment drops plus the tail one side
holds past the other).

- The single frame: a click near a hop boundary lands in one Hann frame
  (the two frames over a sample share its amplitude, so the louder keeps at
  least half, -6 dB). Swept across one hop in 32-sample steps, the 2-frame
  rule alone read a 1 ms click as a tie in 4 of 32 positions at 0.3
  (NMR peak +40..+46 dB), 13 at 0.03 and 24 at 0.01; with the +12 dB frame
  none is a tie. A coherent 1 dB change caps the NMR at +5.7 dB, so a
  change near the threshold stays the share's call.
- The length: the NMR reads only the common span, so a render cut to half,
  empty, with 10 s of noise appended, missing its first 91 ms or with 91 ms
  of noise in front read nmr_max -200, a tie, before both ends were
  counted. A delay over 50 ms, or an equal-length shift over 25 ms (counted
  twice: the silent head and the lost tail), now reads audible, the
  conservative side. The anchor pairs have equal frame counts and lag 0 or
  -1, so they leave 0 or 2 samples.
- The NaN: `compare_files` takes its identical shortcut (equal PCM hashes)
  only for finite samples. `audio_io.remove_dc` spreads one NaN over its
  whole channel, so two different broken renders decoded alike (a 440 Hz
  and a 660 Hz tone, one NaN each, read identical); such a pair now goes
  through the NMR and reads audible.

Declared assumptions: `PLAYBACK_SPL_LOUD_FRAMES` 65 dB SPL (the harness
cannot know the playback level unless `AI_RESTORE_LISTENER` names a
listener whose `listen_ab threshold` run measured it; see "The verdict
ledger and the listening tool") and `MASKING_OFFSET_DB` 24 (at 20, SOTI's
+1 dB against off becomes a tie; at 30 every anchor still holds). The 5%
share is a starting value.

**One-sided**: "inaudible" is trusted and makes the candidate a tie (it
cannot win, is never shown to the listener and skips the learned judges);
"audible" proves nothing about better (a fractional-sample delay reads
audible). Calibration on the stored Tata files, 2026-10-09: air +1 against
+2 dB is audible on all four tapes (audible share 0.34-0.97, NMR peak
5.5-22.9 dB, in the 4.9-8.6 kHz bands); the v3 sibilance final against the
v2 plateau is inaudible on all four (NMR peak -21.2..0.0 dB); a 1-sample
shift reads nothing. The +1 against +2 anchor rests on one unblinded
preference: if Session 0's ABX cannot tell them apart, the offset moves.
Paths outside the repository need `AI_RESTORE_DATA_ROOTS` (`D:\Tata`).

`auditory.audio_sha256` hashes the decoded PCM after `load_audio` removes DC, so
equal hashes mean equal decoded samples and nothing more. The loop does not use
it to mark candidates inert (see "The self-driving loop"); its audibility tie
reaches `pcm_sha256` through `compare_files`, whose identical shortcut it stores
as `identical` in `<slug>.audibility.json`, so a change to the hash or the
shortcut changes the loop's tie verdicts. The stored verdicts are keyed on
`auditory.VERDICT_RULE`: bump it with such a change, as with any change to what
`compare_files` calls audible, and the loop reads every pair again.

### The verdict ledger and the listening tool

Every by-ear verdict is one line of
`assets/quality_calibration/verdicts.jsonl` (`ledger.py`, append-only):
`id`, `date`, `round`, `tape`, `windows`, `files` (label, path and sha256
for files up to 1 GiB), `question` (`abx`, `pair`, `blend`, `flag` or
`preference`), `answer`, `n_trials` and `n_correct`, `confidence`,
`playback`, `context`. In a `flag` answer `clean["*"]` lists outputs
accepted with no complaint at all. `context.heard` lists variants heard
without a verdict, and `context.verdict_group` ties records that restate one
verdict across tapes, so a fit counts it once. A stimulus label names one
file per tape across the ledger (`source` is exempt). Flags and clean lists
come only from the user's words: a rank is not a flag. A hand-added last
line without its newline is safe: `append_many` writes the owed LF before
its block, as `read` already takes such a line whole. The backfill holds
33 records: round one (the known set, the Tele7abc listening set, SOTI and
Vaccin "alpha 2 sounds better"), round two (APL better on speech, pauses
natural, music fine, the 's' still thin) and round three (+1 dB air over +2
and off, unblinded, so a `preference`, not an `abx`).

`experiments/tata_listen/check_verdicts.py` reads the ledger: a flagged label
must carry its flag, a label under `clean["*"]` must carry none and one under
`clean[F]` must not carry F, and the top tier of a two-tier preference must not
fail more hard gates than the rest. A flag on a display-only gate
(`gates.DISPLAY_ONLY` or a soft gate; `listener.hiss` today) is reported as not
asserted and checked neither flagged nor clean, and a stored report that still
lists it (scored under the v2 gates file before ear v3 demoted it) does not fail
on it. A flag record that judges labels on display-only gates alone is counted
on its own ("on display-only gates alone", `DISPLAY_ONLY_ALONE`), neither
checked, waiting nor among the not asserted (longer rankings, trials, empty flag
records). It iterates the reports in sorted order and exits 0 when every checked
verdict holds, 1 when one breaks, and 2 when nothing was checked. On 2026-10-09
it checked 3 records, did not assert 4, found none on display-only gates alone,
and 26 wait for the Round 0 re-score; it lists 6 `listener.hiss` flags as not
asserted (the Tele7abc known set and listening flags, the four round-two pause
records). That is the default `--scores` (`experiments/tata_listen/scores`);
the Round 0 reports are checked per set (see "Round 0 (2026-10-09)").

```text
python -m scripts.listen_ab {abx|pair|blend} --tape SLUG --stim LABEL=PATH ...
    --device NAME --volume SETTING [--connection T] [--eq T] [--anc T]
    [--surround T] [--anchor LABEL=PATH] [--start S] [--seconds 8-12]
    [--trials N] [--level-dbfs -23] [--catch N] [--catch-raw PATH]
    [--familiar N]
python -m scripts.listen_ab threshold --listener SLUG --device NAME
    --volume SETTING [--level-dbfs -23] [--trials 60] [--calibration PATH]
```

The page binds 127.0.0.1 only, so family tapes never leave the machine.
ABX runs 24 trials by default and at least 16 (17 of 24 is p 0.032 at
power 0.77). Pair plays every pair once with an explicit "same", plus one
replicate at the end (with the `--anchor`, the round-one alpha 2 hissy cut,
when one is given). Blend mixes the candidate into the incumbent and places
30 trials QUEST-style. Cuts are 8-12 s, aligned (lag and polarity) and
placed where the NMR says the first two stimuli differ most. The block's
answers reach the ledger in one write at its end. Set
`AI_RESTORE_DATA_ROOTS=D:\Tata` first.

Since 2026-10-12 (backlog E4, salvaged) a blind block is built to say
something even when nothing is heard:

- `--device` and `--volume` are required; `--connection`, `--eq`, `--anc`
  and `--surround` keep the rest of the chain as text in `playback` (the
  ledger takes any extra playback key as text or null).
- One speech level: every cut is brought to -23 dBFS on the first
  stimulus's speech-active frames (`listen_cuts.TARGET_ACTIVE_DBFS`,
  `--level-dbfs`), replacing the match to the first stimulus within
  +-20 dB. A silent cut, or one whose peak would pass -0.18 dBFS at the
  level, refuses the block and names the level every stimulus fits at.
  Measured on the Session 0 and 1 stimuli (10 s cuts): restorations read
  -13.8 to -19.0 dBFS on those frames (crest 12.8-17.7 dB), the raw
  Tele7abc tape at 80 s -36.1 dBFS with a 20.0 dB crest. Session 1 Block 1
  (APL first) would have held the tape at +20 dB where it needs +21.5; at
  -23 dBFS it fits with 2.9 dB to spare, and the restorations play 4-9 dB
  quieter than in Session 0 at the same volume setting.
- The page (`listen_page.py`) decodes the cuts into WebAudio buffers and
  switches A, B and X at the same playhead with an 8 ms linear crossfade
  on a sample boundary (keys 1, 2, 3; space plays and pauses; loops of 1-3
  s from the playhead, key L, or over the quietest 2 s of the reference,
  "Loop the pause"). Checked in the Browser pane on synthetic audio
  (2026-10-12): switching between identical buffers in an
  `OfflineAudioContext` leaves a residual of exactly 0 after the fade, an
  inverted copy nulls to exactly 0 mid-fade, and restarting the new buffer
  from zero instead leaves 0.99.
- Practice first (`listen_protocol`, `--familiar`: abx and blend 2,
  threshold 5, pair none), over an exaggerated pair (ABX: the difference
  doubled; blend: the full candidate), answered with feedback and never
  scored. Catch trials among the scored ones (`--catch`, one per 8, at
  least one), shaped like scored trials: the first stimulus against +6 dB
  of air at 7.5 kHz levelled back to the same speech level, and half the
  raw tape with `--catch-raw`. They are scored apart (`context.catch`).
  The air catch is only as strong as the clip's air: on the Session 1
  cuts (review, 2026-10-12, CPU) the content above 7.5 kHz sits 30.1 dB
  under the total on Tele7abc's APL and 54.7 dB under it on the wbbd
  music clip; the check reads both air catches audible (diff -27.9 and
  -28.1 dB, 96% and 100% of frames over the mask), but its "audible" is
  not trusted. A "disengaged" on a dull clip without `--catch-raw` may be
  a catch nobody could hear: read the catch trials before the flag.
- ABX, blend and threshold take a "can't tell" answer, kept in the record
  (`ledger.CANT_TELL`, `answer.cant_tell`) and out of the binomial score and
  the QUEST posterior; the prompt still asks for a guess when unsure.
- `context.block_flags`: "no signal" when every decided scored answer was
  one button ("can't tell" aside), never "disengaged" (Session 0's listener:
  "I chose B every time because they all sounded the same"); "disengaged"
  only when a catch trial was missed in a block that was not one button.
- `context.power` per record: ABX the critical score and the power at a
  true hit rate of 0.6, 0.75 and 0.9 on the decided trials (16 trials: 12
  needed, power 0.167 at 60%, Leventhal 1986; 24: 17, 0.77 at 75%); pair
  the same binomial per pair; blend the simulated share of 200 QUEST blocks
  whose 90% band ends under x = 1 for a listener whose 75% point is x = 0.5.
- `threshold` (`listen_threshold.py`): three-interval forced choice, a
  300 ms tone at 1, 2, 4, 8 and 12.5 kHz in one of A, B and C, 12 trials
  per frequency QUEST-placed on the server under a hard cap of -33 dBFS
  RMS or 10 dB under the session's speech level, whichever is lower; the
  prior is the check's own 65 dB SPL assumption. Simulated: median error
  1.1-1.6 dB, 90% of runs within 4.4-5.4 dB (two intervals with 10 trials:
  1.7-2.2 and 8.3-11.8 dB; a run of lucky guesses at a 1/2 guess rate put
  one threshold 25 dB low). A frequency not heard reliably at the cap reads
  `censored`. The run appends one record to
  `assets/quality_calibration/playback_calibration.jsonl`
  (`playback_calibration.py`): `sl_map` (dBFS at 0 dB sensation level per
  frequency at that device and volume), `spl_full_scale_db` (median of
  Terhardt's threshold minus each measured one) and `loud_frames_spl` (that
  plus the session level plus 1.5 dB, the loud-over-active gap of the
  Session 0/1 restorations, 1.1-2.1 dB). With `AI_RESTORE_LISTENER=<slug>`
  `auditory.PLAYBACK_SPL_LOUD_FRAMES` takes that listener's latest
  `loud_frames_spl` at import and the record's id joins
  `auditory.VERDICT_RULE`; a named listener without a usable run stops the
  import, `listen_ab`'s too (its cuts and tones import `auditory`), so a
  listener's first threshold block, and the one after a run that placed no
  level, runs with the variable unset (the refusal says so). Error budget
  7-10 dB (Masalski et al. 2014: the normal-hearing
  assumption); the map itself repeats within about 4-5 dB. Thresholds near
  -96 dBFS meet a 16-bit output's floor: record the format in
  `--connection`.

Session 1 commands written before 2026-10-12 need `--device` and
`--volume` (they pass both), and Block 1 now plays at -23 dBFS instead of
APL's own -14.6 dBFS: raise the volume by the difference, record the
setting, and run `threshold` at that setting first.

### The reward pieces (built; the loop uses the floors and the constraints)

- `python -m scripts.reward_noise_floor SOURCE=OUTPUT ... --out OUT.json`
  (as a module from the repository root: it does not patch `sys.path`)
  scores (source, output), then (source, near-copy of the output) for
  `shift_1`, `requantise_16` and the opt-in `resample_roundtrip`. Per
  reading it writes `readings.{key}.floor` (the p95 of the deviations),
  `max` and `n`. Run it on the shipped APL and cathar outputs, the point the
  reward ranks at: at the identity point (a bare `SOURCE`) the floors read
  16-200x too narrow. Each repeat after the first scores through its own
  cache directory, so the Whisper and stem caches cannot hide jitter.
  `--benign-pair SOURCE OUTPUT_A OUTPUT_B [LEDGER_ID]` (repeatable, each
  path its own value, so '=' and ',' survive) scores (source, A) as the
  reference and (source, B) as `benign_pair`: the deviation between two
  restorations a blind listener could not tell apart, with the ledger record
  that says so kept in `sources[].ledger_id` (shape-checked, not looked
  up). No near-copy is made of either; a repeat adds A's own jitter under
  `reference`. Near-copies barely move Whisper or the embeddings, so the
  learned vetoes read the benign-pair report
  (`--noise-floors experiments/reward/noise_floor_benign.json`) and the
  group reward's tie band keeps the output-point one. A report holds one
  operating point (identity, output, benign_pair): a mix is refused unless
  `--pool`, since a pooled p95 depends on how many deviations each point
  brought; `by_transform` shows what each kind contributed (its largest
  deviation) and `percentile_by_transform` each kind's own percentile.
- `scripts/restoration_quality/reward.py`: per reading the two-sided
  distance. A reading that one compared candidate lacks counts as the
  group's worst plus one scale unit (`fill_missing`), so an optimiser cannot
  win by making a reading unread. Per family, the within-group pairwise win
  rate, with a tie band summed over the family's readings: per side
  `weight * min(floor, max(0, |x - target| + floor - dead_zone)) / scale`,
  and the larger side counts. The reward is the worst family's win rate.
  Hard gates, verdict reversal and the learned veto exclude a candidate
  (0.0), inaudibility ties it (0.5), and extra listener flags cap it at 0.5.
  `parse_vetoes` / `vetoed_readings` and `parse_reversals` /
  `reversals_refusing` compute two of those booleans from a grid's
  `vetoes:` and `reversals:` sections; the loop applies them (see "The
  self-driving loop"). The group reward itself is not used by the loop yet;
  since 2026-10-11 the loop's ranking takes its tie band per reading
  (`scripts/autotune_judging.py`).
- `scripts/restoration_quality/preference.py`: the Bradley-Terry head, a
  regularised logistic fit on standardised reading differences, validated
  leave-one-tape-out. It picks the next pair for `listen_ab.py`, lowest
  `|P - 0.5| * (1 - disagreement)` first, inaudible pairs excluded. It sets the
  grids' weights and targets and acts as a falsifier, never the loop
  objective.

### Calibration v3 and the v3 grids

`scripts/calibrate_quality_metrics.py` now scores every family every time.
The v2 `gates.json` came from a dsp-only run, so the speech, mos and stems
gates kept hand-set values, and the v2 `harness_ranking` composite read 0
for every variant. It now ranks the known-ordering tapes by the v3 score
(`--grid`; `calibration_ordering.load_grid` reads the entries with a
`target` under `reward:`, else under the v3 grids' `ranking:` (tata_v3 12
entries, music_v3 10), else at the top level, and refuses a file with
none; `main` reads it before the synthetic suite, so a wrong `--grid` stops
the run in seconds, not after hours of scoring; without `--grid` it ranks
on the default grid of two-sided dsp readings at no change), writes
`gates_speech.json`, `gates_music.json` and `gates_mixed.json` beside
`gates.json`, and records each gate's `n_verdicts`, `source`,
`verdict_source`, `round`, `rounds` and `rounds_agreeing`.

A listener flag becomes hard only after two rounds agree, counted per gate
(`calibration_gates.rounds_agreeing`): a round agrees when the previous
entry's threshold still separates this run's good and bad readings, and it
counts only when the verdicts the gate rests on hold a ledger round the
previous entry's did not (`rounds`: the rounds of the records that name the
gate on the tapes that side it; report.md shows e.g. `2 (1, 4)`). The run's
`round`, the union of every tape's rounds, is display only: counted on it,
a round-4 flag about `listener.bright` on another tape promoted `dead_air`,
`pause_collapse` and `sibilance_thin` to hard on round-1 evidence alone.
Manifest lists and by-ear ranks name no ledger round, so a gate resting on
them counts 1, and a previous entry with no `rounds` (the v2 gates files, a
`--no-ledger` run) restarts the count. Flag verdicts come from the ledger
(`--no-ledger` reads the manifest's lists), and `--excerpts` runs the
degradations a real tape can carry on Tata cuts.

The benign floor leaves out what a benign case changes by construction
(`quality_degradations.benign_changes`): the shifts delay the raw pair, so
the raw-pair sync readings (`file.sync_offset_ms`, `file.sync_lag_*_ms`)
read the shift itself, the fault the 40 ms sync gate stands for. With them
in, shift_5ms / 30ms / 60ms set a 57 ms floor on realistic-v2 en + fr and
the floor rule relaxed the 40 ms sync-offset gate to 172 ms. The drift (one
constant lag) stays in the floor.

A check that reads nothing is `unscored` with a `reason`: `clipped` when the
widest programme band R0 read over the degradation's cases
(`calibration_checks._source_band`) leaves the reading no band of R1's layout,
so no case's band carries it (`balance_metrics.readable_under`; report.md's
"reading from" cell shows `readable_from_hz`, 5292 Hz for air), `not read`
otherwise. The widest, not the median (2026-10-09): with `--excerpts` the
fixture and tape cases of a degradation pool under one name, en (3000 Hz),
Tele7abc (4490), SOTI (5040) and Vaccin (7127), and their median of 4765 Hz
called an air reading Vaccin failed to give `clipped`, which the run prints as
expected; it is now `not read`. A `match` check is unscored only when no case
carries the source reading (the dsp family, which reads R0, did not run). The
summary (`print_summary`) counts only real passes, prints how many non-blind
checks are unscored and names every one R0's band clipped; the exit code follows
non-blind failures only.

The new degradations (`scripts/degradations_v3.py`) declare what must move
and what must not: `air_shelf_*`, `air_corner`, `spectral_tilt_*`,
`pause_residual_*`, `hifi_compander_mistrack`, `loudnorm_ride`,
`phasey_resynth`, `linear_bandwidth`, `treble_dropouts`, `sibilant_islands`,
`sync_drift`. `air_corner` (2026-10-09) is the shelf at +1.5 dB with its
corner at round A1's 9000 / 7500 / 6000 Hz: the readings A1 is judged by
were calibrated on the 7500 Hz shelf only. Read with the runner's dsp family
on en / fr they follow the corner: R2's absolute 's' level +0.140 / +0.301 /
+0.603 (fr +0.607 / +0.906 / +1.189), `hf_4k8k` +0.129 / +0.280 / +0.570,
`hf_8k16k` +0.936 / +1.182 / +1.358 (en), the net level under the absolute
one, the texture under 0.01. The calibration run has to confirm it before
A1 is judged.

Measured while building them: FFmpeg 8.0.1's `treble=g:f=7500` is the RBJ
shelf at Q 1/sqrt(2) (matched to 9e-8). Under that shelf the net sibilance
reading moves a median 53% (at most 96%) of the absolute one, so "the net
reading must not move" is asserted as "under its absolute twin". R1 reads
no air on any fixture under the runner's clip to R0's band: R0 reads
1.1-5.0 kHz on the speech targets (the Italian one 5040 Hz) and 1414 Hz on
the music beds, all under the 5292 Hz band air's first band needs. R0
reads the beds right: their 1.6 kHz band sits 84-94 dB under the loudest
and every band from 2 kHz up 106 dB or more, so the +0.48 / +0.95 /
+1.91 dB first read there were unclipped, the shelf's gain on residue
135-138 dB down. Chosen 2026-10-09 (review finding 12, over demoting the
music check to blind): both air-shelf entries assert R1 air, non-blind. On
every fixture the check reports unscored, reason `clipped`, and the run
names it; with `--excerpts` the speech entry asserts on Vaccin (R0
7127 Hz), where the runner's R1 path (R0 clip, mute guard) on the whole
cached source reads +0.135 / +0.27 / +0.54 dB for a +0.5 / 1 / 2 dB shelf
(the cut mirrors it) on 30 of 61 windows, over a benign floor of 8e-5 dB;
SOTI and Tele7abc clip it. That is a probe, not a calibration run: run the
calibration with `--excerpts` and a Vaccin cut (`--metrics dsp` is enough)
before its gates are trusted: a failing non-blind check makes it exit 1. Full
phase randomisation reads as a -0.83..-1.15 dB/oct tilt, so phasiness can
pass for a timbre change. The en and de fixtures hold no pause R4 can
read. On the stored v2 results
Tele7abc's "light hiss" on the single 4 s probe passes the v2 hiss flag, so
`flags_reproduced` reads False there: the flag that never moved.

`scripts/tune_grids/tata_v3.yaml` and `music_v3.yaml` rank readings by
`{target, dead_zone, scale, weight, family}`; `tune_restoration.py` and
`reward.py` read the same entries. Since 2026-10-12 the drivers default to
ear v4's `tata_v4.yaml` ("Critical of hiss"); the v3 grids stay unchanged
for reproducibility. `tata_v3` (12 entries) was re-derived
on Round 0's reports after Session 0, `music_v3` on presence only; every
value is still an unblinded hypothesis and its comment says what it stands
on, which accepted files it still charges and which rankings tie or
reverse. On the calibration v3 re-score (2026-10-10; pre-fix reports in
`experiments/tata_listen/scores_v3_precal`) `tata_v3` re-derived the entries
whose readings moved under a file that set them or broke the rule:
`sib_abs_level_db` 3.7 to 3.4, texture 1.75 to 2.0, `gap_spread_db` 3.0 to 6.85,
`gap_lsd_db` 3.5 to 9.05 (the old pair reversed Vaccin's "alpha 2 sounds better"
over `apl__baseline`) and island kurtosis 1.8 to 1.85 (Round 0 had kept it under
the one-unit margin); the slope and modulation keep 2.1 / 10.0, modulation as
the one known margin exception (eight accepted files inside by under one scale
unit, 1.2; the rule's 14.3 would tie Soti's known ranking). The tests hold the
eight entries the rule sets outright (all but texture's midpoint, pause depth's
band, the slope and modulation) to the smallest 0.05 step that admits the
furthest accepted file by one scale unit. It reverses 15 round-one pairs (20
before), each from a charge the comments name; two are new, Vaccin's known
single4s and stitched over `cathar075` (pause depth's mixed-route aggregate, no
longer masked by the old spread and LSD charge on `cathar075`), and the tests
pin them until a speech-only pause-depth aggregate exists. It ties every
round-two and round-three preference (`docs/ear_v3_round0.md`, "The grid after
calibration v3"). A re-derivation re-runs `check_verdicts.py --scores` on every
set and sums the grid over every judged file, ranking by ranking, before it is
written. Speech-side readings weigh 1.5 (the P.835 SIG/BAK prior,
speculative). Not ranked: `hf_4k8k` and `hf_8k16k` (backstop gates),
residual noise, `lkr`, `balance_top_db`, `gap_pause_s` and the sync; on
`tata_v3` also `balance_air_db` and `sib_centroid_hz` (dropped: unread on
the linear tapes, and ranking C over B on Vaccin) and `gap_atten_db` and
`gap_hf_excess_db` (shown only: the first orders round one backwards, the
second rests on the hiss pair Session 0 did not hear); learned judges sit
under `vetoes:` only. `tata_v3`'s pause depth band (2.95-35 dB) is the one
entry that charges a pass-through (2.95): every other entry reads an output
identical to its source as 0. Their `vetoes:`, `reversals:` and
`audibility:` sections are
the loop's guards ("The self-driving loop"); `tune_restoration.py` reads
none of them. The calibration writes under its `--out`, while the drivers
still default `--gates` to the v2 `experiments/quality_calibration/gates.json`:
a v3 round passes `--gates experiments/quality_calibration_v3/gates.json`.

Same-percept transforms (2026-10-11, backlog E2 part c,
`quality_degradations.SAME_PERCEPT`): `fresh_noise` (the fixture's
recorded noise against a fresh realisation of it, `fresh_realisation`:
the phase surrogate of the noise with its 50 ms level trajectory divided
out, under that trajectory again; on a tape cut, which has no recorded
noise, two noises shaped like its quietest frames), `allpass_200hz`,
`allpass_2khz`,
`frac_delay_0p37`, `gain_up_0p1db`, `gain_down_0p1db` and `tpdf_dither`
join `BENIGN`, each on a generator seeded from (language, name, level)
(`benign_rng`), so every older case stays bit-identical. Their floor reaches
only the sensitivity checks (`check_floor` in `report.json` and its own
table in `report.md`); the gates, the known-ordering stand-in grid, the
route floors and the report's benign floor leave them out, and no tuning
grid reads either. A frame-wise STFT re-synthesis of the noise lost about
6 dB to the incoherent overlap-add, hence the whole-signal surrogate. The
recorded noise is not steady (realistic-v2's +-3 dB wobble at 0.3-1 Hz on
top of the tape's own wander): over 100 ms frames it spreads 8.5 / 8.7 dB
p5-p95 on en / fr, a plain surrogate 1.9 / 1.6 dB (a steadier hiss, which
moved pause depth -0.47 / -0.73 dB and `hf_8k16k` -2.44 dB on en), the
envelope-kept realisation 8.5 / 8.9 dB (reviewer, 2026-10-11).
Measured with the dsp family on realistic-v2 en and fr (CPU, 2026-10-11,
the benign cases only): the check floor of `dsp.gap_atten_db` goes 0.13 ->
1.68 dB, `dsp.residual_noise_db` 0.0002 -> 1.34, `dsp.pause_depth_db`
0.0004 -> 0.20, `dsp.gap_mod_dist_db` 1.89 -> 2.66, `dsp.gap_hf_excess_db`
0.36 -> 0.41, `dsp.hf_8k16k` 0.005 -> 0.014, R1's tilt 0.0000 -> 0.005. The
200 Hz all-pass, a phase change nobody hears, moves residual noise
-3.1 / -1.7 dB and gap attenuation +3.2 / +1.9 dB (en / fr: both read the
aligned waveform difference); the dither moves fr's gap modulation 3.2 dB;
the fresh noise leaves the gap family unread on both (its noise fills the
source pauses R4 needs) and moves pause depth -0.08 / +0.23 dB, residual
noise -0.25 / -0.29 dB and `hf_8k16k` -0.54 / +0.01 dB. A full calibration
run with them has not been made (the GPU families wait for the hunt).

### Calibration v3 with the Tata cuts (2026-10-09)

The first v3 run (`--metrics dsp`, en, `--excerpts` Vaccin 60-360 s, SOTI
60-360 s, the whole 134 s Tele7abc) passed 98 of 144 checks with 8 non-blind
failures. Three (`hum` and the two dropout checks) had passed in v2, which
ran en alone; the other five are checks v2 did not have. Each was traced to
a reading, a degradation or a case set, and fixed there; no expectation was
loosened:

- `hum` (3 of 4 sources): the degradation put 50 Hz hum on every base,
  while the runner reads the mains R0 names, and R0 names 60 Hz on the
  hum-free en, es and it Piper targets (near-monotone partials hold fixed
  bins; the 3 dB frames floor was set on noise only). The case now injects
  at R0's mains (`quality_degradations.capture_mains_hz`): en +0.02 /
  +0.27 / +2.35, es and it order too, the tapes (50 Hz) unchanged. Not
  adopted: requiring R0's fundamental to stand about 10 dB out (Piper at
  most 8.1 dB, the tapes 13.1-19.9; it would also settle Vaccin's 60 Hz
  near miss) until the 24 IA clips and 300 noise clips confirm the cut.
- `words_muted` / `dropouts` on `dsp.dropouts` (medians 0 / 0 / 0): one
  hole sits in at most two of a 300 s cut's 40 windows, so the window
  median is blind. They now assert `file.dropouts`, the count over the
  whole aligned pair (`runner.dropout_entry`). The count measures each
  frame's drop against the programme frames' median drop and asks the
  100-1000 Hz body to fall too (`dsp_metrics.dropout_count`): on judged
  files the least-squares gain match had scaled drifting outputs down as a
  whole (gaudeamus5 845 holes, 0 re-centred; SOTI cathar075 276 to 94), and
  a 2 um treble lift read 10 holes on SOTI. `dropouts` draws nested 30-50 ms
  holes on programme frames from its own seed (`degradations_v3.holes`):
  3 / 6 / 12 on every source; `words_muted` orders on all four;
  `treble_dropouts` and every benign case read 0. Since 2026-10-12 the hard
  gate reads `file.dropouts` (see "Gates that refused the denoising
  itself").
- `pause_residual_hiss` slope and LSD (effect 2.6x and 1.5x the floor): the
  benign requantise set the floors (0.44 / 1.27). Vaccin's capture holds a
  notch 40 dB deep at 10.75-11.25 kHz, under the 16-bit dither, which
  lifted its top ERB band 9-16 dB. R4 now leaves out every bin whose source
  pause floor sits under the dither plus 10 dB (`pause_metrics.above_dither`):
  Vaccin's requantise reads 0.003 / 0.017 and en's 0.12 / 0.22. The floor is
  a p95 over every benign case, so it depends on the case set. On en and the
  tape cuts (this run) en's requantise is a lone outlier the p95 skips: the
  floors are about 0.003 / 0.024, which the mild hiss level clears about
  400x / 80x. On the default en and fr (`DEFAULT_LANGUAGES`) with the same
  cuts, fr's requantise reads 0.375 / 0.36: the floors are about 0.07 /
  0.13, cleared about 16x / 15x. Both are computed from the per-case
  readings (every other benign case reads 0), not yet from an en and fr
  calibration run.
- `sibilant_islands` texture (3 of 4): en's one readable window holds 10
  fricative frames, where the spread is noise (+-0.17..0.35 over 8 seeds).
  The texture needs 30 (`sibilance.TEXTURE_MIN_FRAMES`); en and Vaccin go
  unread, and the islands are placed on each runner window's own fricative
  frames (`degradations_v3.window_fricatives`): fr 0.16 / 0.22 / 0.62, SOTI
  0.13 / 0.31 / 0.64, Tele7abc 0.16 / 0.32 / 0.59. The calibration's
  default languages are now en and fr.
- `loudnorm_ride` on R2's level (Vaccin +0.56 dB at 0.9): one window gain
  from the loud frames cannot cancel a ride the 's' frames sample
  elsewhere. R2 matches the gain on R1's programme cells within 0.5 s of
  each 's' (`sibilance.frame_gains_db`): 0.045 dB at most.
- `sync_drift` on `file.gain_ride_lu` (0.18 / 0.30 / 0.71 LU against 0.1):
  the median lag cannot align a drift of 146-587 ms. R7 reads each output
  window along the sync's lag track (`file_metrics.lag_track`): 0.032 LU at
  most; a track that moves no window by half a 1 ms step leaves the old
  reading exactly.

`SCORE_SCHEMA` is 7 (2026-10-12: R12's `file.timebase_*` readings are new;
2026-10-10: 5 added R11, which a schema-4 score lacks and the benign cases
would give no floor; 6 moved R11's band to its own STFT and paired
`dsp.attack_db` on the onsets), so every cached v3 score is read again.
Cases are written once and reused, so a rerun deletes the cases these
changes rebuild (`hum`, `dropouts`, `sibilant_islands`, `sibilants_*`;
`dropouts` no longer draws from the shared generator, so every later
rng-driven case of a language changes on a fresh build: delete `cases/` to
reproduce).

R11's entries (2026-10-10): `lf_shelf_boost_music`, `lf_shelf_cut_music`,
`lf_highpass_music` and `lf_highpass` (speech, so it also runs on the tape
cuts) assert R11, hold `dsp.lkr` and R1's tilt flat and report R1's body;
`hum` asserts R11 flat (0.1 dB), `linear_bandwidth` and the tilts report it.
The generators (`degradations_v3.lf_shelf`, `lf_highpass`) apply the
magnitude of FFmpeg's `bass` and `highpass` zero-phase: run causally on a
bass-led bed the high pass turned `align_pair`'s lag by up to 101 samples
and lkr read 0.20-0.26, failing its flat check. None draws from the shared
generator. A subset run (en, fr, dsp) passed every R11 check on the 8192
frame; on the band frame the same rows, read through the runner's dsp family
on freshly built cases, order on both languages and clear the floor (the
`lf_highpass` mild level at least 1700x), and hum stays within 0.022 of 0.

### Calibration v3b, en and fr (2026-10-10)

Adding fr beside en failed two checks that had passed on en alone. Neither
was a code regression (en read bit-identical in v3 and v3b), and neither
degradation runs on the tape cuts (`hiss_in_pauses` needs the fixture's
noise, `transient_smear` the music bed), so in v3 both were en-only checks.
With two languages the median is their mean and the monotonic share is 0,
0.5 or 1.

- `hiss_in_pauses` on `dsp.gap_air_db` (monotonic share 0.5): the
  expectation was wrong for this material. The hiss lands where the
  expander's gain is closed, but gap_air reads the source's p15-p40 frames,
  60 % (en) and 71 % (fr) of them inside the speech mask at unity gain. In
  the median gap frame the added 3-10 kHz hiss sits 51 / 41 / 31 dB (en)
  and 73 / 63 / 53 dB (fr) under the frame's own air, so the reading moved
  by an order-statistic step: fr +0.0027 / +0.0027 / +0.0010, unordered; en
  +0.024 / +0.025 / +0.086, no real detection either. R4's true-pause
  attenuation reads the same cases -33.9 / -43.9 / -53.8 (en) and -41.0 /
  -51.0 / -60.9 (fr), 10 dB a step, about 4700x its 0.0079 floor. The entry
  now asserts `dsp.gap_atten_db` "down", keeps `dsp.residual_noise_db` "up"
  and reports gap_air (blind). No counted gate loses its calibration:
  gap_air "up" backs only the display-only `listener.hiss`, and its "down"
  side (`listener.dead_air`) stays asserted by `gated_pauses`. The unit
  speech's 0.58 s pauses put its p15-p40 class in the pauses, which is why
  gap_air orders there; a dense-speech unit case records the blindness.
- `transient_smear` on `dsp.attack_db` (mild level -10.6x the floor): a
  reading defect. The attack was the difference of the two sides' median
  rises, not a paired change: a 20 ms smear lowers 59-68 % of the onsets and
  lifts the rest a little (the HPSS re-estimates), so the medians landed on
  different onsets (fr +0.085, en windows -0.572 and +0.516). Paired
  (`transient_metrics.paired_attack_db`: the source's median rise, and that
  plus the median per-onset change) every 20 ms window reads negative and
  the levels order on both languages (en -0.17 / -1.90 / -4.15, fr -0.10 /
  -2.13 / -4.24). The same defect set the floor: SOTI's requantise read
  +0.00603 unpaired, -0.00014 paired, and that one dither draw is why the
  floor rose from 0.00032 (v3) to 0.0027 (v3b), the shared tape generator
  drawing differently since 1d644fb changed what the cases consume. On
  Gaudeamus1 and 5's round-2 finals the paired window values move about
  0.3 dB in the median and flip sign on 14-20 % of the windows, while the
  p10 tail `listener.attack` reads stays positive (+0.15..+0.59 paired,
  +0.06..+0.66 unpaired): its verdicts at -4 dB hold. The planning numbers
  (cathar 4.3-4.6 dB, APL 2.0) were read unpaired; re-check `music_v3`'s
  attack entry on a Round 0 music re-score before trusting its dead zone.

Still open: seed each case's generator from (language, name, level) in
`build_cases` / `build_tape_cases`, so the benign floors stop moving when
the degradation table changes.

### Round 0 (2026-10-09)

```text
python experiments/tata_listen/score_round0.py [--sets r1_known,r1_listen,r2,r3]
    [--metrics dsp] [--dry-run]
```

The driver (untracked: `experiments/` is gitignored) groups the ledger's
records by source and runs `scripts/validate_restoration.py` once per set
and tape (dsp family, `--language ro`, the v2 `gates.json`, the pair cache
`score_listen.py` uses) into `experiments/tata_listen/scores_v3/<set>/`.
`check_verdicts.py --scores experiments/tata_listen/scores_v3/<set>` then
checks each round on the files it judged: `r1_known` (3 records),
`r1_listen` (3) and `r3` (4, trivially: preference checks count hard gates)
hold; `r2` fails 9 checks (exit 1), the v2 `sibilance_thin` missing "APL's
's' still thin" on Tele7abc, SOTI and Gaudeamus5 and the v2 `dead_air`,
`dull`, `attack` and `sibilance_thin` firing on six accepted music outputs.
The report, with every number, is `docs/ear_v3_round0.md`. No gate or flag
is adopted; after Session 0 `tata_v3` and `music_v3` took the grid notes
(the report's "What the grids took"), and `tata_v3` was re-derived on the
calibration v3 re-score, whose `check_verdicts.py` results match the pre-fix
reports set by set ("The grid after calibration v3"). The facts that will
decide defaults:

- R1, clipped to the programme band, cannot see the 7.5 kHz shelf on
  linear-track tapes: between +2 / +1 dB / off presence moves 0.03-0.05 dB
  and the tilt 0.02-0.10 dB/oct, so the shipped `listener.bright` reads +2 dB
  at +0.005 at most. `hf_8k16k` minus the same tape's +1 dB render (+0.65 to
  +0.70 on all four tapes) and R2 `sib_abs_level_db` carry the shelf. The
  paired form checks the shelf, not the percept: an accepted
  `final2_cathar` abstains under its 1 dB `gap_hf_excess_db` guard on every
  tape.
- R4 `gap_hf_excess_db` separates the hiss pair on Tele7abc only (+3.11 on
  0.7.3, +0.49 on 0.7.5). Off Tele7abc it is inconclusive: SOTI's and
  Vaccin's "alpha 2 sounds better" names no binary (`context.assumed`).
- Speech-route `pause_depth_db` delta at 35.0 reproduces "silent in
  pauses". Between the files the user named the gap is 32.98-37.22, and
  three judged known excerpts fall inside 32.98-36.89.
- `gap_atten_db` reverses the dead-air verdict: the dead APL renders
  (34.95-38.45 dB) attenuate less than the cleared cathar (42.31-42.84), so
  today's 8.7 / 5.4 already charges the round-one winner more.
  `balance_air_db` at +0.17 reverses the round-one Vaccin preference, and
  `gain_ride_lu` needs a dead zone of at least 2.26 LU.
- Every music false fire came from a 15 s window `route_window` called
  mixed: pause and timbre flags count on speech windows only.
- The air verdicts are 2 verdict groups of one unblinded listener (round
  two's "thin" file is round three's A); every other proposed flag rests on
  one record. `thin_abs`, `dull`, `pause_gated`, `pause_hiss` and
  `s_texture` stay display-only until Session 0 or a second tape agrees.
- The noise-floor report of 2026-10-09 (8 pairs, four families, two
  repeats) lets both v3 grids' vetoes parse, but at 3x the floor the learned
  limits (CER 0.0025, speaker cosine 2.9e-5, UTMOS 0.0027, SCOREQ 0.0033,
  SIGMOS 0.20-0.25, MERT 2.3e-4) would veto on jitter: measure them on pairs
  a blind listener could not tell apart before the first v3 loop round.
  `reward_noise_floor --benign-pair` does that (built 2026-10-09; the run on
  Session 0's four pairs is pending).
- Erratum: the round-one hissy `tele7abc__cathar__alpha_2_0.mov` was
  rendered 2026-09-20 19:11 on cathar 0.7.3 (the installer pinned 0.7.3
  until b54c217, 2026-09-26); 0.7.6 came later that evening and is
  bit-identical to 0.7.5 on the app's stages. The ledger's
  `r1-tele7abc-listen-*` records say 0.7.6: read 0.7.3, leave the
  append-only ledger as written, and name Session 0's ABX 0.7.3 against
  0.7.5.

### Session 0 (2026-10-09, blind)

Six blind blocks through `listen_ab` (the `round: session0` ledger records;
device and volume not recorded). Raw tape against the shipped APL: 3/3
heard. Every pair of two restorations, each called audible by
`scripts/audibility_check.py`, was not: the 0.7.3 / 0.7.5 hiss pair on
Tele7abc and SOTI (13 of 14 "same"), ABX +1 / +2 dB air on SOTI (10/20),
shipped APL against round three's B (7 same), and an ABX control on the
largest restoration difference available, 0.7.3 against 0.7.5 alpha 2 on
Tele7abc (diff -17.6 dB, 97% of frames over the mask; 8/16). The listener:
"they all sounded the same". So:

- The unblinded verdicts of rounds one and three ("alpha 2 has hiss", "B is
  better") do not reproduce blind. Treat unblinded ledger records as
  hypotheses, not anchors.
- `audibility_check.py`'s "audible" does not predict this listener. Its
  tie stays trusted; nobody is asked to listen on its "audible" alone until
  it is recalibrated on a blind threshold (a `blend` continuum from the
  shipped restoration toward the raw tape, at a recorded device and volume).
- The A0 drift (shipped `apl_tonal_flatness_max` 0.035 skips the plosive
  tamer on the Tata tapes, flatness 0.0219; B ran 0.01 and tamed 9
  plosives) is not heard: keep the shipped values.
- Pass `--device` and `--volume` to every `listen_ab` block: the masking
  model's playback level (`PLAYBACK_SPL_LOUD_FRAMES`) is meaningless
  without them (required since 2026-10-12, with a threshold run at the
  session's volume to place that level).

### Overnight rounds v6 (2026-10-09/10)

`experiments/run_autotune_v6.cmd apl|cathar|apl_music|cathar_music` (output
in `D:\ai_restore_work\autotune_v6` and `autotune_v6_music`; APL with
`--stage-cache`), each from the shipped defaults, `tata_v3` on the four
speech tapes and `music_v3` on the 12 music clips plus Gaudeamus 1, with the
learned vetoes at 1x the Session 0 benign-pair floors, the audibility tie and
the air reversal guard:

- APL on speech (rounds A1/A2 and every other knob): plateau after round 1,
  the shipped defaults stay. 3 candidates hashed inert (music-only knobs on
  speech tapes), 1 audibility tie (`apl_sibilant_mix` 0.9), 2 learned vetoes
  (subtraction on: SCOREQ / SIGMOS down; pause floor off: SIGMOS DISC down),
  the rest 0 of 4 wins. The plateau is what the user heard, so nothing was
  asked.
- cathar on speech (rounds C1/C2): plateau after round 1, the shipped defaults
  stay. 8 learned vetoes (alpha 0.5 and 1.5, noise print off, repair strength
  4, split band 4 kHz, one de-esser band, pause floor off, fill 18 dB), the
  music-profile knobs hashed inert on the speech tapes, the rest 0 of 4 wins.
- APL on music (round A6): round 1 found a winner, `expander_decay_s` 0.12
  with `crt_notch_q` 60 (mean rank 12.53 against 16.07, ahead on 11 of 12
  clips, hard failures 28 against 31); MERT and Audiobox vetoed pause floor
  off, fill 18, expander depth 18, attack 0.02, subtraction on and the stem
  path. Still running; no default moves before a plateau, an audibility check
  against what the user heard, a blind block and the realistic-v2 plus
  IA-corpus confirmations.
- cathar on music: plateau after 4 rounds, the shipped defaults plus
  `expander_knee_offset_db` 0 and `expander_attack_s` 0.02 (shared polish keys);
  the audibility check calls 5 of 12 clips audible at -32 to -57 dB.
- cathar round C3 (music profile, from that plateau): plateau after round 2,
  `cathar_music_beta` 0.02 -> 0.01; music repair off ranked well (9 of 12
  wins) and was vetoed both rounds on MERT distance.
- cathar round C4 (speech tapes, from the shipped defaults, 2026-10-11,
  `D:\ai_restore_work\autotune_v7`): plateau after round 1, the defaults
  stay. `cathar_declip_mode` `detected` rendered byte-identical audio to
  `always` on all four tapes; `cathar_dehum_freq` 0 (cathar's own
  detection) won no tape. The round was resumed at its scoring step on the
  new loop code: 30 candidates instead of 44 (the 14 music-only moves left
  out by material, each byte-identical on every tape in the renders already
  made), dsp scoring only, and no learned stage since nothing beat the
  incumbent: about 1.6 h of scoring for 31 outputs on four full tapes.
  Most candidates tie at one mean rank (15.36) because every reading sits
  inside its dead zone around the accepted output; the stage-1 readings
  equal the full-family ones of round v6 for the same audio.
- The music plateaus differ from what the user approved by far less than
  Session 0's -15 dB control pair the user could not hear; none is adopted
  without a blind block.

The tied and plateaued rounds match Session 0: the knob moves the loops can
make on speech are below what this listener hears.

### Variation pre-check (plan 3.2a, 2026-10-11)

`python -m scripts.variation_precheck STATE.json ...` asks of finished loop
rounds whether a per-tape settings policy (plan 3.2b, GRPO) has anything
audible to choose, without rendering: per round, the single best candidate
and the per-tape oracle (each tape's lowest rank), every oracle pick
through `auditory.compare_files` against the best's render, `selectable`
only when audible and above `--heard-db` (default -17.6 dB, Session 0's
unheard control). Run on the five v6/v7 loops (report
`D:/ai_restore_work/variation_precheck_2026-10-11.json`):

- Speech (APL and cathar, `tata_v3`): the shipped settings rank first on
  every tape, oracle gain 0. Partly by construction: the v3 targets were
  read on those same accepted files.
- Music: the oracle picks another candidate on 7-11 of 12 clips for 3.1-4.4
  ranks, but six picks the check calls trusted ties (down to -94 dB) gained
  3.3-7.3 ranks: the per-clip oracle gain is the loop's ranking noise (a rank
  turns any reading difference outside its dead zone into a whole step).
  Most other picks are audible by the check but -20 to -52 dB, under what
  was not heard.
- The loop's acceptances are not that noise: re-judged offline from the
  stored `*.score.json` with readings closer than 0.25, 0.5 or 1x their
  benign floor (`experiments/reward/noise_floor_benign.json`) ranked as
  ties, every accepted move of the five loops still beats its incumbent
  (7-10 of 12 clips) and the speech rounds still have no winner. The noise
  sits in single-clip ranks; the mean over 12 clips and the half-the-tapes
  rule absorb it, so the ranking keeps exact ties.
- Selectable: only `apl_music_stem_path` on, on three clips (APL music
  round 2: `m_wbbddanceswithwolves` -7.9 dB, `m_vid-20220730-091707` -9.4,
  `x_vid-20230606-121054` -14.4), the switch the 2026-09-25 listener round
  did not accept and MERT / Audiobox vetoed in round A6.

So the GRPO policy stays parked (and its Optuna baseline with it) until
one blind block shows the stem path, the largest restoration difference
the loops produce, is heard: Session 1 Block 2 in
`experiments/session1/README.md` (ABX on `m_wbbddanceswithwolves`, shipped
against stem path on), after Block 1, the listener's blend threshold. When
that threshold is measured, re-run the pre-check with it as `--heard-db`.

### Gates that refused the denoising itself (E1, 2026-10-12)

Judge only; no restoration default moved. Several hard gates failed most
shipped outputs whatever the settings, because they read hiss removal as
damage. The scripts behind every number are in
`D:\ai_restore_work\scratch_e1_gates` (CPU only, cached stems and stored
scores read in place).

- **The stems' octave** (`stem_octave.py`, gate `stems.octave_programme`,
  `music_v3`'s background entry). The raw worst octave of the separated
  stem read hiss removal as stripping (the instrumental stem carries the
  hiss), and `_stems`' least-squares stem match, a waveform measure, scaled
  cathar's phase-moving output 1.3-7.9 dB under the mixes' magnitude match
  (five sampled hunt clips).
  The new reading compares the two MIXES on the cells where the source's
  stem stands guard (a)'s 18 dB over its noise and holds the mix within
  1 dB, per octave (median of the cell ratios, an octave needing 200 cells,
  guard (b) and a 1/10000 energy share of the loudest), levelled by the
  mix's median octave; only the source is separated for it. The source stem
  is brought back to its mix's level (audio-separator caps the input mix
  and each stem at a 0.9 peak; a capped stem reads None: 9 of the hunt's 235
  sources). Three designs failed on the way, each recorded so it is not
  retried: programme cells on the least-squares stems (cathar still 115 of
  152 hunt clips, every octave 3-9 dB down), the stem pair levelled by the
  mix (47 of 139 cathar and 31 of 162 APL clips under -3, against 31 of 140
  and 18 of 165 for the mixes read on the same cells: the output's own
  separation adds differences the restoration did not make), and a guard on
  the mix's
  own floor (it emptied a bed under speech, so `background_stripped` read
  nothing). Measured: the 24 v2 music finals read
  -2.42..0.00 on the 22 that read (raw octave: 13 of 21 under -3);
  `background_stripped` on the 2026-09-20 calibration stems -5.9 / -11.1 /
  -14.9 dB (raw -0.9 / -2.6 / -14.3); the hunt's shipped cathar 66 of 178
  clips under -3, APL 18 of 178 (raw 157 of 166, 126 of 192).
- **Guard (c), the absolute floor (2026-10-12).** A background cell counts
  only at `CELL_FLOOR_DBFS` (-90 dBFS) or more, its level read as the sine
  that would put its power in the bin (`FULL_SCALE_SINE_POWER` 0.25 under
  `stft_power`'s scaling): guard (a) is relative, so over a clean floor it
  passed cells under anything a capture carries. R1's -70 dBFS
  (`balance_metrics.MIN_LEVEL_DBFS`) is a broadband level and too high per
  cell: two of the 24 accepted v2 music finals went unread
  (m_wbbddanceswithwolves, both engines) and APL m_videoplayback-2021-07-11
  moved -0.79 to -2.24 dB; -80 left the same two unread. At -90: 22 finals
  read, three move by 0.19 dB at most; `background_stripped` unchanged; on
  the hunt 22 cathar and 18 APL readings move, two cathar verdicts pass
  (treasure-hunt -5.46 to -2.93, the-riddlers -5.64 to -2.86), APL keeps 18
  failures. Not et-el's fix, despite the investigation's "-94 dBFS cells":
  et-el-extraterrestre-tve still reads -54.0 / -54.3 because its 4-16 kHz
  cells are window 0's first two STFT frames (zero padding meeting a
  source that starts mid-signal, -68 dBFS median; the output starts
  quieter). Dropping two frames at each end clears it (-0.19 / -2.56) and
  moves every reading: a separate change with its own calibration. Scripts:
  `D:\ai_restore_work\scratch_npdepth` (`etel_check.py`, `check_finals.py`
  with the floor as its argument, `check_calibration.py`, `hunt_floor.py`).
- **Its blind spot.** Guard (a)'s floor is each bin's p10 in the window, so
  a bed that never falls to a floor in 15 s is unread: a sustained chord
  reads None stripped or not; on the realistic-v2 musiconly beds a planted
  one-octave mute fails nearly every window where the octave reads (17 of 17,
  36 of 36, 65 of 69 for 250-500 / 500-1000 / 1000-2000 Hz) but the octave
  reads on only 17, 36 and 69 of 100 windows. Lower margins did not help:
  at 10 dB the oracle fired on 5 of 48 windows, at 6 dB fewer octaves read.
- **Soft until a verdict sets them:** `speech.cer_median`,
  `mos.audiobox_pc`; the grids' learned vetoes stay.
- **`calibrated`:** False for every hand-set threshold but lkr's (1.747, the
  v2 known-good bound), and never for a gates-file `floor` entry resting on
  no verdict. Such an entry also never tightens the code's threshold
  (`gates.floor_threshold`), so the v3 / v3b files' lkr 0.3 reads as 1.747;
  under the old rules it failed 103 of the hunt's cathar clips.
- **Retired names** (`gates.RETIRED`): `dsp.dropouts` (the v2 file's 57.3
  belongs to the count 1d644fb replaced) and `stems.octave` are left out
  when a stored gates file still sets them.
- **`file.dropouts`** is the hole gate, bound 4: every benign case reads 0,
  and 4 is the smallest whole count with at most 5 % of the 920 realistic-v2
  oracle pairs over it (13.8 % over 1). It lets the `dropouts` degradation's
  3 holes through, and the count misreads hum removal (m09 hum fixtures
  against their clean programme: 10 of 10, up to 14 holes). It is soft
  (reported only) since 2026-10-12: it failed the shipped outputs the user
  accepted on 2 of the 4 Tata tapes (SOTI: cathar 16, APL 9), and a hard
  count would refuse the stronger hiss removal the user asked for.
- **The calibration's refusals:** no gates file without `--known-ordering`,
  and an oracle check (`ORACLE_MAX_SHARE`, 5 % of the pairs of each capture
  against its `_target` and `_clean`; `--oracle-max N` for a quick run)
  that refuses the gates files and exits 1. On the dsp family, all five
  languages: `dsp.whistle` 288 (none since its floor, below),
  `dsp.hf_8k16k` 102, `dsp.clicks` 45 (1 at the v2 file's 0.79),
  `file.dropouts` 44, `dsp.hum` 18, `dsp.hf_4k8k` 8, `file.lra` 1, so today
  no run writes a gates file until the air reading is fixed.
- **`dsp.whistle_db` is floored** (2026-10-12,
  `dsp_metrics.WHISTLE_FLOOR_DB`): the core's peak over the neighbourhood's
  median, the median read no lower than the bin a sine 80 dB under the
  loud frames (p70) would fill, and never under 0. Before, a ratio of
  nothing to nothing: realistic-v2's tape noise is notched within 40 Hz of
  both line rates (`realistic_defects._without_tones`), so its captures
  read -40 to -52 dB and the clean targets +2 to +5 (the maximum of
  near-empty bins: their neighbourhood sits a median 108 dB, p95 92.5,
  under the loud frames per bin). The IA hunt's 34 cathar clips rising
  over 1 dB (8 over the gate; channel-4-news, the-tube, spider and santas
  among them) held a line at
  the line rate on two only: et-el-extraterrestre (a 23 dB PAL line the
  scanner missed) and thevistagroup (an NTSC line it read as PAL); on 30
  the source's band lay over 80 dB under its loud frames, and the core's
  maximum sat at a random offset in both halves of the clip. Re-read on
  the same windows, a floor of 70 to 90 dB fails no oracle pair (100 dB
  fails 28). At 80 dB the hunt's clips over the gate fall from 8 to 2
  (cathar) and from 7 to 1 (APL): ein-halleluja on both, a polarity-
  inverted pair whose channel mean cancels the source (every paired
  reading on it is wrong; R0 knows the pair is inverted, the runner's
  downmix does not), and full-size-render202202012106's cathar output,
  its band 1-5 dB over the floor. The six `cathar_music_alpha` 0.25
  refusals stand: m_vid-20220730-091707 carries a real line 37 dB under
  its loud frames over hiss 68 dB under, so the floor never reads and the
  candidate lifts it +3.4 to +3.7 dB (the incumbent +2.4 to +2.7); the
  scanner's head search reads that wandering PAL line as NTSC
  (`crt_detect_mode`, the audio-restoration-engine skill). The planted
  lines keep reading: the whistle class (-39 dB) +41 dB, the recorded
  whine of the transport classes (-57 to -62 dB, spread by flutter) +10 to
  +21 dB. 34 of the 440 realistic-v2 captures without a planted line carry
  a line R0 tracks at 15574-15796 Hz, their noise bed's own, 50-170 Hz off
  the rates the bed's notch cuts. A weak-line fixture class (a line 3-8 dB
  over the capture's floor, on the three hissiest beds) was built and
  dropped: even there the line sat 81-86 dB under the loud frames, read 0
  by the floor, inaudible by construction. Tests:
  `tests/unit/test_whistle_reading.py`. Scratch:
  `D:\ai_restore_work\scratch_whistle\` (`diagnose.py` and `closer.py` for
  the core-maximum offsets, `candidates.py`, `sweep_floor.py`,
  `confirm_reading.py`, `music_alpha_recheck.py`, `gen_weak_line.py`).
- **The degradation table:** `background_stripped` asserts the programme
  octave (the raw one blind); `stem_octave_muted` plants a 1000-2000 Hz mute
  of the hiss-bearing bed at -6 / -12 / -24 dB; `music_hiss_removed` is the
  oracle, the programme octave flat within 0.5 dB. Both are built on the
  music fixture's clean bed and its own recorded noise (`vhs - clean`,
  `calibrate_quality_metrics._music_capture`), never on the speech
  fixture's `noise`: that residual carries the speech chain, so on it the
  500-1000 Hz mute read nothing on en or fr and its removal read -6.7 dB on
  de. With the bed as its own stem the mute's case median reads -1.4 /
  -2.8 / -4.8 dB (en) to -5.9 / -11.5 / -20.6 (es), monotonic on all five
  languages, and the oracle stays within 0.17 dB.
- **Re-gated offline** (the hunt, 235 clips per engine, the v2 file):
  clips failing a hard gate 216 to 112 (cathar), 205 to 161 (APL); with the
  programme octave read from the cached stems, 146 and 169.
  `tests/unit/test_gates_shipped_scores.py` holds ten of those cathar files
  (trimmed) and fails when a gate change makes them fail more.
- **On the GPU after the hunt:** score the stems family on the 24 v2 finals
  (`score_round0.py --sets r2` with stems), the calibration with
  `--known-ordering` and every family (the oracle pairs then show the stems
  and learned gates), and re-score the hunt's stems family so its files
  carry `stems.octave_programme_db`.

### Critical of hiss (ear v4, 2026-10-12)

The user's rule (2026-10-11, in capitals): the ear must be very critical of
VHS hiss in two places, under speech or singing and in silence and pauses.
The user had flagged hiss on every Tata tape for weeks ("hiss on every v2
plateau", "hiss (single4s, Tele7abc)", "hiss left?") while the harness never
moved: `listener.hiss` is display-only, `tata_v3` ranks pause depth with a
2.95..35 dB band (3 dB of cleaning ties with 30), `gap_atten_db` and
`gap_hf_excess_db` are shown only, and nothing read the hiss under the
programme. The same user rejected dead pauses ("silent in pauses"), so "no
hiss" never means dead air: a quiet floor of room tone mostly under about
2-3 kHz may stay. Gap level alone did not order the old verdicts (accepted v2
outputs read gap air -8..-12 dB, the "hissy" cathar alpha 2 -15..-18.5); R4's
HF excess did (+3.11 against +0.49): shape and audibility, not level.

Both readings use one declared critical playback level,
`hiss_floor.L_CRIT_DB` = 75 dB SPL for each file's loud frames (p70 of
1024-sample frame power), 10 dB over the audibility check's 65 dB: one
listener sets one volume. A reconciliation probe ran the programme
prototype at both levels: at 65 it reproduced none of round one's hiss
labels; at 75 it separates them on Tele7abc and orders single4s ("light
hiss") over stitched. The audibility tie keeps 65 dB and its 24 dB offset;
only a blind session at a recorded device and volume moves L_crit.

`scripts/restoration_quality/hiss_floor.py`, shared:

- `stft_power`: Hann 2048 / hop 1024, centred, scaled so a band's sum is
  its mean square; `band_layout`: 1-ERB bands from 100 Hz to min(16 kHz,
  Nyquist, R0's brickwall), CRT line bins (+-250 Hz) out, ERB-midpoint
  centres.
- The valley level: per frame and band, the 25th-percentile bin after
  whitening by a 31-bin median of the per-bin P20, divided by -ln(0.75),
  times the band's mean shape (a band under 8 bins takes its mean). Hiss
  fills a band's valleys, partials do not.
- `block_floor`: per 1 s block, C_b x P10 of the valley over the active
  frames within +-1.5 s (under 0.45 s: the window's P10). C_b
  (`white_calibration`, cached) maps the estimator onto the true mean of
  15 s of seeded white noise.
- `source_anchor` (once per source, on the raw mono, cached by the runner
  as `<cache>/profile/<key>_hiss_<hash>.json`, the hash over
  `runner.HISS_CODE`): R4's true-pause mask per 60 s chunk (memory on long
  tapes), per 15 s segment C_b x P10 of the valley over at least 1 s of pause
  frames (bands under the 16-bit dither plus 10 dB NaN); fewer pauses, or
  pauses that are digital silence, read every live frame (`all_frames`).
  `hiss_cap` holds every band at the median anchor over the bands from R0's
  programme bandwidth to 12 kHz plus 3 dB (reverb tails and room tone under
  the band otherwise read as hiss). `validated` only from the pauses, with
  a band of 2 kHz or more and at least two reference bands; an unvalidated
  window abstains from the ranked fields.
- `hiss_stationary.py` (2026-10-17), the second anchor, for continuous
  programme (music, singing, dense speech): read only when the pause anchor
  is not validated and carried as `Anchor.programme`; the runner and
  `check_hiss_fixtures` call `hiss_stationary.source_anchor`. Per 15 s
  segment each band's floor per 3 s block (C_b x P10 of its valley) is
  accepted where the blocks span at most 1 dB p10-p90, at their P25; a band
  carries programme where its mean stands over 1 dB above the segment's
  floor or its frame power follows the 300-4000 Hz envelope (correlation
  over 0.15). Validated when every band of 1 kHz and up holding a floor is
  accepted but at most four (six at least) and two hiss-alone bands lie from
  half an octave over the programme's top to 12 kHz. Only then does the
  programme reading cap with it and report margins
  (`Anchor.for_programme`); the pause reading keeps the pause anchor, so a
  source whose pause anchor validates reads bit for bit as before (the Tata
  v2 pairs: 11,655 readings, none moved). `meta.hiss_anchor_how` names the
  path (1 pauses, 2 stationary, 0 none). It validates where the reading is
  demonstrably right and nowhere else: 38 of 360 stationary-hiss music
  variants (music-only programme low-passed at 4.5 kHz; every perfect
  output at or under 0, max -1.70; a render keeping the hiss under the
  programme reads +19.6..+32.6), no realistic-v2 corpus fixture, no music
  clip, no IA hunt source. No anchor can do more there: with the exact noise
  as the anchor the music-bed and music-led fixtures' perfect outputs read
  over 0 on 75 of 76, and real programme exposes its hiss in pauses, not
  in a stationary floor (`docs/ear_v4_hiss.md`, "Continuous Programme").
  The pause path itself validates dense full-band music and reads its
  perfect outputs +1.4..+18.8 dB, and no band-edge condition on its
  validation closes that without dropping the Tata tapes: the strictest
  drops all five tapes and all 34 IA sources, and of 576 settings the best
  that keeps the four tapes still validates 25 of the 32 failing sources.
  The Tata tapes read more programme over the hiss above 9 kHz than the
  failing music does; the defect is the reading's spectral-cap pool, which
  revealed music fades into over 4-10 kHz (same section, "The Pause Path's
  Hole"). Telling revealed programme from residual hiss by time does not
  work: hiss an SNR-dependent gain lets through with the programme follows
  the music as closely (an expander-gated render correlates 0.55-0.83 with
  the programme cells), and capping each band at its stationary ratio reads
  that gated hiss 9-14 dB lower while still reading two of three dense
  perfect outputs over 0 (same document, "Revealed Programme Is Not Told
  Apart in Time"). Nor does the fine structure: reading the output's hiss
  from each band's within-frame valley (P20 over -ln 0.8) lowers the dense
  perfect outputs only 1.2-1.6 dB (+15.10, +14.10), because the revealed
  programme is as noise-like in band as the hiss, and it reads the cathar
  finals 2.1-5.5 dB lower, because a subtractor's residual is peaky in band
  (its valley 2.2-4.7 dB under its mean); its fresh-realisation p95 is
  0.56 (same document, "Revealed Programme Is Not Told Apart by Its
  Valleys").

`hiss_programme.py`, the hiss left under the programme (the `hup3`
prototype's estimator, rebuilt after validation round 1):

1. Window guard: a non-finite or still sample, a 300-4000 Hz power
   envelope correlating under 0.5 between the sides, or a programme gain
   over 24 dB (a sanity bound) reads every name None. Round 1's waveform
   correlation (0.5) and 6 dB gain rejected every Gaudeamus render: their
   programme band is polarity-inverted and 2 ms off the runner's LF-led
   lag (waveform correlation 0.31-0.34, envelope 0.96-1.00), and the
   least-squares gain match shrank them 8.2-8.3 dB (Gaudeamus1) and
   14.6-14.9 dB (Gaudeamus5) in every window, which R1's gain undoes.
1. Frames: the STFT frames wholly inside the window (`whole_frames`: the
   centred first and last are half padding, read the masker 3 dB low and sat
   among the quietest frames that set the shape and the P10 floor), and the
   two frames at either end are neither active nor pause (`edge_frames`: the
   tests two frames away read past the window). Validation round 2 found a
   0.37-sample delay moving T's margin tail 0.13 dB and one sample at the end
   of a Gaudeamus5 window moving it 2.4 dB through those frames; now every
   window moves at most 0.01 dB.
1. True pauses: DSP-VAD non-speech runs of 200 ms or more within 6 dB of
   the floor; the rest is active. Floor under the programme: `block_floor`,
   held no higher than the anchor, whose unread (NaN) bands take the anchor
   interpolated in log level across them (a held note in a music fixture's
   quiet frames left a band unread and its floor uncapped: the note, 15 dB
   over the hiss, read as a +20 dB hiss).
1. Noise-dominated cells (NDC): the band valley within 1.5 dB of the floor
   two frames before and after, ring mean (bins f+-2..4) at or under the
   floor, no partial in the same bin two frames away (10x its ring). Nothing
   selects on the cell's own power: round 1's cap on it (10x its ring) kept
   the low draws and read a fresh realisation 3.7 % over the source per band.
   Readable band: 40 cells, 2 % of its active cells, a source NDC mean within
   -2..+1 dB of the floor, a floor over the dither plus 10 dB.
1. Removal: sum O / sum S over NDC cells (output in the source's programme
   gain), local over +-0.5 s where 40 cells allow, read apart in two classes:
   free band-frames (the band's mean power two frames before and after within
   1.3x the floor: gaps between words, bands above the programme) and busy
   ones; each frame takes its own class's ratio, so the pause floor keeper's
   hiss in the gaps between words is read in those gaps.
1. Spectral cap: a band's ratio stands at most 6 dB over the ratio the free
   cells of the other bands of 1 kHz and up (outside +-2 bands, +-1 frame)
   show at the same instant. This is the round-1 fix: programme the source's
   hiss hid is revealed by a clean output and sat in the band-edge NDC cells
   at 9-16 dB under the floor, so a perfect output read +9..+22. Revealed
   programme stays in the bands the programme reaches; residual hiss is
   broadband. Unreadable bands interpolated as before; hiss band power =
   floor x bins x ratio, no more than the side's own band power over three
   frames.
1. Masking (Johnston 1988): offset alpha (14.5 + z) + (1 - alpha) 5.5 dB,
   alpha the programme's own tonality on the source with its floor removed;
   one masker for both sides, the source's band power less its hiss (read on
   the output, the threshold carried the output's own draw and a fresh
   realisation read +0.4..+0.9 dB more hiss). Each side plays at its own
   loud frames at L_crit (the output in the source's gain at L_crit less its
   loud level plus that gain), the pause reading's convention. Spread
   27 / 12 dB per ERB, PEAQ forward spread over two frames, floored at
   Terhardt's threshold.
1. NMR integrated as 10 log10 sqrt(sum NMR^2) over the readable bands of
   1 kHz and up (`margin`), every band of 1 kHz and up (`margin_all`, it
   carries singing) and the readable bands of 3 kHz and up (`margin_hf`);
   each the P90 over the active frames; `audible_frac` the share over 0 dB.
   Under two readable bands of 1 kHz and up the margins and guards read
   None.
1. Guards, paired (0, value): `hiss_pump_db` (hiss kept under the programme
   against the pauses' own removal: positive = gated out of the pauses and
   left under speech; needs 0.5 s of pause; the pauses' removal per band is
   the output's median band power over the pause frames over the source's,
   a band the source's pauses hold under the 16-bit dither plus 10 dB left
   out, so a muted Hi-Fi break reads None; the means let the frames that
   straddle a gate's edges carry it, and Tele7abc's 20-50 dB pause gates
   read +8.2..+11.5, under the old 12 dB bound at every depth),
   `hiss_prog_kept_hf_db` (per
   band of 3 kHz and up, the median output over source on active cells
   standing 18 dB over the floor, both held at the 16-bit dither, averaged
   over the bands with 20 such cells: dulling; round 1's median over all
   cells hid a cut of the band edge behind the 3-4 kHz cells and missed the
   4 kHz low pass on 9 of 23 fixtures), `hiss_prog_kept_soft_db` (its twin
   for the soft programme, 2026-10-17: per band of 1 kHz and up, the same
   median on active cells standing 6-18 dB over the floor spread over the
   band's bins by the source's shape, whose 3 x 3 neighbourhood mean stands
   6 dB over it too, averaged over the bands with 20 such cells; below) and
   `hiss_island_kurt` (R4's frame kurtosis on 20 ms frames, 3 kHz to 1.06 x
   the top readable band; 0.0 when the output's margin is under -10 dB, where
   no island can be heard, since 2026-10-19: as None it tripped the guard as a
   lost reading and excluded the v9 APL round 1's combined candidate, the one
   that removed the hiss best, on Gaudeamus5; None stays where nothing could
   be measured).

`hiss_pause.py`, the hiss in the pauses (the pause design's prototype with
block estimator, q 0.5, raw shape, no trim, Terhardt):

1. R4's true-pause mask on the source (25 frames at least, else every name
   None: the decision rests on the source, the same for every candidate);
   20 ms mean-square frames; tones of either side out (an output bin under
   the 16-bit dither is empty, not a tone, or a pause emptied to digital
   silence would take the source's floor with it).
1. 1-ERB bands from 50 Hz to min(16 kHz, Nyquist, brickwall); the
   stationary floor per band is the median over 200 ms blocks of each mask
   run (a tail of 5 frames kept), divided by the Wilson-Hilferty median/mean
   ratio max(1 - 1/(9k), 0.05)^3, k = max(kept bins / 1.5, 0.5) x 10. The
   median stays: the mean (tried in round 1's fixes) let the soft programme
   a clean output reveals in the source's quiet frames read up to +43 dB.
   Only the source's silent blocks count, band by band on both sides
   (`silent_rows`): a block whose source band of 1 kHz and up stands 3 dB or
   more over the anchor (`anchor_band_power`: the anchor's per-bin level
   interpolated over log f, rescaled to 20 ms bins, summed over the band)
   holds programme there; a window whose every band of 2 kHz and up held
   programme reads None. On a choir tape R4's quiet frames are soft singing:
   Gaudeamus5's pause blocks stood +4.1 dB (median) over its anchor at 2-8
   kHz, against +0.2..+1.2 per window on Tele7abc (SOTI -1.0..+2.0, Vaccin
   -1.2..+2.9). The first test (6 dB) let most singing blocks through: the
   APL keeper ladder's pause tail read 36.25 / 33.00 / 32.67 / 32.67 (fill
   8 / 12 / 18 / keeper off) and a render that took the soft singing out
   would have won the family; at 3 dB it reads 34.66 / 31.21 / 25.09 /
   20.50, and the speech tapes move under 0.6 dB.
1. rel_b = floor_b in dB re the side's own whole-file loud frames, weighted
   by 10 log10 w(f) first; then for the margin each band takes the power
   mean of itself and two neighbours either side (three bands took the
   realisation scatter from sd 0.25 to 0.13-0.18 dB on the fixtures; five
   with the silent blocks took the fixtures' fresh p95 of the pause margin
   from 0.51-0.54 to 0.18-0.21 dB, the tapes' from 0.80 to 0.49 on the tail
   and 0.99 to 0.23 on the median). w(f) = 1 / (1 + exp(-log2(f / 3150) /
   0.15)): 0.01 at 2 kHz, 0.10 at 2.5 kHz, 0.38 at 3 kHz, 0.5 at 3.15 kHz,
   0.91 at 4 kHz; tape hiss set the margin at 3.5-3.9 kHz on every fixture.
   The first weight (0.5 at 2.5 kHz, a quarter octave per e-fold, applied
   after the smoothing, which carried a room floor under 2 kHz into the
   2.4-2.7 kHz bands at their full weight) read a hiss-free pink room floor
   low-passed at 2 kHz, at the source's own 100-1000 Hz pause level, at +9.7
   dB median and flagged 9 of 10 speech fixtures: the user allows room tone
   mostly under 2-3 kHz.
1. `margin` = max over bands of 2 kHz and up of L_crit + the smoothed
   weighted rel_b minus (Tq - 3), never under L_crit - 140 = -65 (a
   band-filtered pause read -1843 in round 1); no band holding anything
   reads -65, never None (an empty output loses nothing for being unread).
   `onset` = L_crit - margin (level-free). `room_onset` (100-2000 Hz,
   unweighted), `room_delta` (the 100-1500 Hz floor re loud frames, output
   minus source: dead air reads far under 0), `sone` (Zwicker loudness x
   Bark width x w, bands of 2 kHz and up).

Card fields (all `dsp.*`, programme routes; `meta.*` source side):

| field | sides | direction | role |
|---|---|---|---|
| `hiss_prog_margin_db` | src, out | lower | flag (median); H2 (delta) |
| `hiss_prog_excess_db` | max(0, .) | lower | ranked (median 3.0); H1 |
| `hiss_prog_margin_all_db` | src, out | lower | display; music H2 |
| `hiss_prog_excess_all_db` | max(0, .) | lower | ranked (1.5; music 3.0) |
| `hiss_prog_margin_hf_db`, `_audible_frac` | src, out | lower | display |
| `hiss_prog_removed_db`, `_removed_hf_db` | 0, out | none | display |
| `hiss_pump_db` | 0, out | lower | constraint |
| `hiss_island_kurt` | 0, out | lower | constraint |
| `hiss_prog_kept_hf_db` | 0, out | higher | constraint |
| `hiss_prog_kept_soft_db` | 0, out | higher | constraint |
| `hiss_pause_margin_db` | src, out | lower | flag (tail); H2 (tail) |
| `hiss_pause_excess_db` | max(0, .) | lower | ranked (tail 3.0, median 1.5) |
| `hiss_pause_onset_db`, `_sone` | src, out | display | |
| `hiss_room_onset_db` | src, out | display | |
| `hiss_room_delta_db` | 0, out | two-sided | constraint (G2) |
| `hiss_prog_active_s`, `_bands`, `_validated` | both | none | coverage |
| `hiss_pause_s` | both | none | coverage |
| `pause_depth_speech_db` | as pause depth | lower | G1, speech route |
| `meta.hiss_anchor_pause_s` | source | none | display |
| `meta.hiss_anchor_validated` | source | none | display |
| `meta.hiss_anchor_how` | source | none | display (anchor path) |

Decisions that reconciled the two designs: one level (75) for both places;
the margin in dB over the threshold in both, ranked as the one-sided
excess with target 0 and dead zone 0 (no `reward.py` change); a grid `tie`
(reading units, the same-percept p95) that `reward.tolerance` widens the tie
band to, `tune_restoration.tie_buckets` ranks the sweep in and
`autotune_judging` bands the loop's ranking with (below, "One loop
judging"); two families,
`hiss_programme` and `hiss_pause`, so neither pays for the other; weights
programme median 3.0 and all-band 1.5, pause tail 3.0 and median 1.5 (each
place 4.5, hiss 9.0 of tata_v4's 20.5); the programme tail not ranked;
hard guards as grid `constraints:` applied by the loop alone (no new hard
gate in gates.py); DNSMOS BAK and SIGMOS NOISE as vetoes at floor multiple
1, measured in the loop in their jitter floor (the judging's hiss exemption),
never in the benign-pair floors (0.1205 / 0.1165) the branch first named;
R4's slope, spread and LSD demoted to display in v4 (their target is
the source floor's own shape, so a scaled copy of tape hiss reads perfect
and a room-tone-only pause is charged; a shaped fill reads slope
-16.6 dB/oct). The flags `listener.hiss_programme` (median) and
`listener.hiss_pause` (tail) fire at 0 dB and are the user's rule
(`gates.USER_RULE_FIXED`): a gates file entry for them is ignored, and
`calibration_gates` derives none, because 13 pre-rule `clean["*"]` records
count as clean on every listener flag and would relax them.

The v4 grids (`scripts/tune_grids/tata_v4.yaml`, `music_v4.yaml`) are the
default `--grid` of `autotune_restoration.py` and `tune_restoration.py`;
the v3 grids stay for reproducibility.

- `tata_v4`: v3's ranking without R4's slope, spread and LSD (display only),
  plus `dsp.hiss_prog_excess_db.output.median` (weight 3.0, tie 0.30),
  `dsp.hiss_prog_excess_all_db.output.median` (1.5, 0.25),
  `dsp.hiss_pause_excess_db.output.tail` (3.0, 0.35) and `.median` (1.5, 0.35),
  each one-sided (target 0, dead zone 0, scale 1). The vetoes are v3's six plus
  DNSMOS BAK and SIGMOS NOISE (floor multiple 1): the guards line reads "8
  learned vetoes, 1 verdict reversals, 17 constraints, audibility tie at a 24 dB
  masking offset".
- `music_v4`: every music_v3 entry and veto, the all-band excess at 3.0
  (it carries singing) and the programme excess at 1.5, the same pause
  entries, nine relative constraints (accepted music finals sit outside
  music_v3's uncalibrated R1 dead zones).
- `tie` is the jitter report's `tie_floor` for that flat key (the widest
  same-percept transform's p95), rounded up to 0.05: programme excess median
  0.294 -> 0.30, all-band 0.239 -> 0.25, pause excess tail 0.319 -> 0.35,
  median 0.331 -> 0.35 (`experiments/reward/noise_floor_hiss.json`'s recipe,
  re-run by the adversarial review of 2026-10-12 on the fixed pause reading
  and pump: eight pairs, two repeats, DSP repeat jitter exactly 0; round 2
  read the pause tail 0.494 and the median 0.232 before the 3 dB silent test
  and the new weight). H1's `max_move` equals the tie; H2's bound is its delta
  key's own tie floor (programme 0.327 -> 0.35, pause tail 0.319 -> 0.35,
  music all-band 0.506 -> 0.55). Round 1 took the pooled p95 over all eight
  transforms, which the five that barely move a hiss reading diluted (pause
  tail 0.20 against fresh draws of 0.50-0.80, so H1 refused a same-percept
  redraw); `reward.tie_floors` reads `tie_floor` (`tie_band_floors`).
- `constraints:` (`reward.Constraint`, `autotune_guards.mark_constraints` in
  both judging passes): a candidate whose reading, on some tape, is worse than
  the incumbent's by more than `max_move` (default 0) and lies beyond `bound`
  (default none) is "excluded" (`Constraints.dsp_guard`), and so is one worse
  than the incumbent's and more than `max_move` past the bound (past it the
  limit does not follow the incumbent, or a loop creeps `max_move` further
  every accepted round); a reading the
  incumbent has and the candidate lost trips too, an unread incumbent never
  does. `loop_guards` stops the loop on a reading the harness does not produce
  (a typo). tata_v4 holds H1 (no new audible hiss: both excess keys, `max_move`
  = their tie), H2 (never hissier than the raw tape: both margin deltas, `bound`
  = the delta key's floor), G1 (`pause_depth_speech_db` delta at most 35 dB), G2
  (`hiss_room_delta_db` not under -54), the pump (`max_move` 3, `bound` 12:
  midway between the round-one renders the user did not call dead, at most
  +6.49, and the dead APL, +18.09 and up; a 30 dB gate of the true pauses
  reads +21.8 on Tele7abc and is excluded, a 20 dB one +11.8 and passes),
  kept HF (`max_move` 1.0, `bound` -6.0: the floor holds at -7.0, over T
  single4s' -7.04, "underwater again"), the soft programme kept (`max_move`
  0.05, the jitter report's tie floor 0.045; `bound` -27.0, the next whole
  decibel under the lowest file the user did not call dull, V single4s
  -26.51, over T single4s' -27.46; music_v4 -28.0 under the captain-pu APL
  final's -27.32), the islands (0.3 each; the programme
  islands `bound` 2.0), modulation (1.2) and the
  timbre and 's' edges. Two bounds the specification did not give:
  `gap_island_kurt` 1.85 (tata_v3's dead-zone edge) and `balance_top_db` -1.0
  (the dull flag's edge), because the move alone excluded the cleared round-one
  cathar renders (cathar075 alpha 2: gap islands 1.734 and top -0.963 against
  the baseline's 0.959 and -0.231).

The benign transforms (`scripts/same_percept.py`, which since the merge also
holds the calibration's `fresh_realisation`, `floor_noise`, `allpass` and
`tpdf_dither` and names which consumer reads which subset; `reward_noise_floor`
with `--transforms hiss`): `gain_up_0_1`, `gain_down_0_1` (x 10^(+-0.1/20)),
`delay_0_37` (FFT fractional delay, pad 8192), `resample_soxr` (librosa soxr_hq
round trip through `round_trip_rate`, never `resample_poly`), `requantise_16`
and `fresh_floor_a/b/c` (30 s blocks, 1 s crossfade; STFT 1024 / 768). The
noise-led cells have a 5 x 5 envelope within 3x the block's per-bin P20 of
it and within 2x the median envelope of the 21 bins around it; each mixes
one file-long seeded white noise at the envelope power-complementarily with
the old cell, its share rising 0 to 1 as its 5 x 5 neighbourhood goes 50 to
90 % noise-led. Round 1 fixed three defects here: a held note sits at its
own P20 like the floor and was redrawn as a band of noise (Gaudeamus5's
pause median +1.6..+2.1 dB; the spectral-peak test); the 2x test (1.5x a
stationary floor's mean) failed about 5 % of pure-noise cells, whose islands
of the old draw lost their energy in the inverse STFT, so a white noise came
back 0.32 dB down, correlated 0.26 with the old draw, and every fresh pause
reading read low; and independent draws per block lost up to 3 dB in the
crossfades. Now a white noise keeps its power within 0.04 dB at correlation
0.07, and each level of a 20 dB gated noise within 0.05 dB; musical-noise
islands and held tones stand over 3x the P20 and stay (deciding on 9 x 9
means redrew them and moved cathar renders' programme reading -0.3..-0.8
dB, a change of texture). A render
whose floor sits at two levels (an expander-gated floor 20 dB under a
keeper-filled pause, APL's B above 4 kHz) keeps its pause cells there: the
block P20 is the gated level, so those pairs do not show the high bands'
realisation scatter. Run before a v4 round:

```text
python -m scripts.reward_noise_floor "<v4_air>\<tape>__source.wav=<v4_air>\<tape>__B_air_1dB.wav" (x4)
  "<v2>\<tape>__source.wav=<v2>\<tape>__final2_cathar.wav" (x4)
  --transforms hiss --repeats 2 --families dsp --max-seconds 125 --out experiments/reward/noise_floor_hiss.json
```

The report pools every transform into one p95 `floor` (the vetoes' kind);
`by_transform` keeps each one's largest deviation, `percentile_by_transform`
its own p95 and `tie_floor` the widest of those, which the hiss ties take.
On the round-2 report (2026-10-12): gain and soxr at most 0.0001 on every
margin and excess key, delay at most 0.011, requantisation at most 0.059;
the fresh realisations set every tie (programme excess median 0.17 / 0.29 /
0.18 for draws a / b / c, all-band 0.15 / 0.19 / 0.24, pause tail 0.49 /
0.24 / 0.28, pause median 0.12 / 0.20 / 0.23; the adversarial review's
re-run on the fixed pause reading: pause tail 0.21 / 0.32 / 0.18, median
0.20 / 0.33 / 0.13, delay at most 0.005). Round 1's report read the
fresh draws moving the pause median up to 0.99 and the tail up to 0.80 on
Gaudeamus5's cathar final (singing read as pause hiss, partly redrawn as
noise). The first build's report read the pause median at 1.78 (the
fresh floor redrawing Gaudeamus5's held notes), its `fresh_floor` lost 0.32
dB of a white noise (pause ties 0.35 / 0.45 before the fix), and
requantisation moved `hiss_prog_kept_hf_db` on Vaccin by 2.9 dB; the
spectral-peak test, the power-keeping redraw and the dither floor on the
guard took all three away (kept HF p95 0.003).

The degradations (`scripts/degradations_hiss.py`,
`quality_degradations.HISS_DEGRADATIONS`, fixtures only): a linear capture
(`linear_capture`: the clean programme low-passed at 4.5 kHz with two
1.2 s gaps, plus stationary tape hiss at the fixture noise's 3-10 kHz
level) and `hiss_under_programme`, `hiss_in_pauses_only`, `hiss_hf_tilt`,
`low_floor_kept`, `dead_pauses`, `hiss_islands`, `hiss_removed_dulled` and,
since 2026-10-17, `soft_programme_cut` (the programme with the cells the
capture holds 6-18 dB over its hiss's mean, from 1 kHz up, cut 4 / 8 / 12 dB:
the soft guard reads it down, kept HF moves less than it, 0.38-0.65 dB;
`hiss_under_programme` holds the soft guard flat within 1.0 dB and
`hiss_removed_dulled` reads it down; `SCORE_SCHEMA` 9).
The specification named the fixture's recorded noise for the capture; that
noise is rumble-heavy, the VAD found no true pauses in the gapped capture
and `low_floor_kept` read NaN, so the capture takes tape hiss instead. The
benign `*_linear` cases (identity, requantise, soxr, +0.1 dB, 0.37-sample
delay, `fresh_noise_linear`, since round 1 an exact fresh draw of the tape
hiss: `same_percept.fresh_like` carried the old draw's local fluctuation
into the new one) set the `dsp.hiss_*` floors alone
(`benign_only`); every other floor is untouched (51 of 51 identical, en
and fr). `calibration_checks` gains `steady` (the level medians span at
most max(tolerance, 3 floors)) and `below` (every median at or under the
centre minus that). `underwater` reads kept HF down and `musical_noise`
island kurtosis up, both blind.

The fixture probe:

```text
python -m scripts.check_hiss_fixtures --fixtures artifacts/realistic-v2 --languages de,en,es,fr,it
  --programme linear --shape corpus|white|<npy> [--shape-from WAV --shape-out artifacts/hiss-v1/shape.npy]
  --out experiments/hiss_v4/fixtures.jsonl
```

It scores the gapped m12 cases (identity,
oracle -6 / -12 / -20 / -30 dB, prog_only, fresh, delay, gain, soxr,
requantise, hf_left, lp4k_all, gated_m0, islands_15) and prints the
acceptance rows. A shape derived from a family tape stays under the
gitignored `artifacts/`.

The acceptance record, every row with its measured value, is
`docs/ear_v4_hiss.md`. Validation rounds 1 and 2 (2026-10-12) failed the
build and fixed the implementation, not the bars; the record lists each
defect, its cause and its fix. After round 2 every same-percept row passes
(delay 0.004, the fresh draws within every bar), and so do the raw-hissier
row (T's fill 8 0.26 under the raw tape; 0.22 over it on the adversarial
review's fixed pause reading, inside the tie, H2 excluding it), less
removal (S's alpha 2.0 +0.33 inside the tie, 0.35 now), the pause flag on
round one's labels (both
0.7.3 renders over 0, every cleared one under), the dead-air, cheat, fill
and underwater exclusions, music coverage, every calibration check and the
cost. What does not pass: two perfect fixture outputs over 0 (music hidden
under the hiss up to 5 kHz, +0.004 / +4.40; 41 of 43 pass). Rows that
contradict the evidence, reported and not gamed: the air shelf is no
programme-only change (A over C lifts the residual hiss +0.8..+2.0 dB at
8-16 kHz and the programme cells +0.4..+2.0 dB above 6 kHz); both readings
play each side at its own loud frames, so the "two SPL conventions" check
measures a property of the render (an output scaled +6 / -10 dB moves no
margin); the corpus noise is not stationary (block levels 6-15 dB p10-p90),
and forcing its anchors to validate read perfect outputs up to +22.7 dB;
"cathar alpha 2 has hiss" and "light hiss (single4s)" were not blind (the
user's blind ABX could not tell 0.7.3 from 0.7.5, 8 of 16; pauses 13 of 14
"same") and sit under the masked threshold at 75 dB after 27 dB of removal,
while the pause flag fires; "hiss on every v2 plateau" is the harness's own
note (the listener marked both v2 finals clean); the IA renders under -0.55
hold physically hissier pauses (APL's floor at 2 kHz and up +0.56..+1.91
dB over the source's, cathar's programme 3.6-5.2 dB lower against its
floor); single4s is dull on T only by the user's flags; a 4 kHz low-pass of
a programme ending at 3.6 kHz dulls nothing.

Pitfalls:

- A clean output reveals the soft programme the source's hiss hid (band-edge
  bands 9-16 dB under the floor): any reading of "what the output kept of
  the source's noise cells" counts it as hiss, and the first build read a
  perfect output +9..+22 (saturating 13-14 dB under the source). Read the
  free band-frames apart and cap each band's ratio 6 dB over the other
  bands' free cells (`hiss_programme.spectral_cap`).
- A transform that redraws some cells and keeps others must keep power
  across the seam: islands of the old draw lose their energy in the inverse
  STFT (the first `fresh_floor` lost 0.32 dB of a white noise and up to 3
  dB in its crossfades, and every fresh pause reading came out low). Set
  the steady test over the floor's own fluctuation (3x the P20, not 2x),
  mix the seam power-complementarily and draw one noise for the file. Check
  a white noise's power and its correlation with the old draw: a transform
  that keeps part of the old draw (a soft share everywhere correlated 0.41)
  reads smaller ties than a fresh draw would, and one that redraws a
  residual's musical-noise islands changes its texture, not its draw.
- The jitter report caches each near-copy as
  `<cache-dir>/noise_floor/<output key>_<transform>.wav` and reuses it on
  every later run: after changing a transform in `same_percept.py`, delete
  those files (or score through a new cache dir), or the report re-scores
  the old near-copies (round 1's first re-run did, and read the old floors
  to the third decimal).
- The pause median's realisation scatter is the estimator's own on short
  pauses: 0.5-1.5 s of pauses per window, the median over 200 ms blocks,
  then the median over windows. Three bands let the max pick upward draws
  (fresh draws moved T's B 0.28-0.36 dB, all downward); five bands and the
  silent blocks bring the tapes' fresh p95 to 0.23 (median) and 0.49
  (tail). The mean would halve it but let a perfect output's pauses read up
  to +43.
- Half-padded STFT frames: `stft_power` centres its frames, so the first and
  last frames of a window hold half padding. A reading per window must drop
  them and keep the frames whose two-away tests read past the window out of
  its statistics, or a one-sample change of the alignment (a 0.37-sample
  delay of the output) moves a band across a readability threshold.
- A tie band pooled over transforms that barely move a reading is diluted:
  take each transform's p95 and the widest (`tie_floor`).
- R4's true-pause mask holds whatever is quiet inside the VAD's non-speech
  runs: soft singing on a choir tape. A pause reading must read the hiss
  where the source is silent apart from it (blocks within 3 dB of the
  anchor; at 6 dB Gaudeamus5's soft singing, +4.1 dB, still passed), or
  every render ties at the source's singing and one that takes the singing
  out wins.
- A guard read as a mean over pause frames is carried by the few frames
  that straddle a gate's edges: `hiss_pump_db` read 20-50 dB pause gates at
  +8.2..+11.5 until it took medians.
- A bounded guard whose limit follows the incumbent can be crept past, one
  `max_move` per accepted round: `reward.Constraint` excludes a worse
  candidate more than `max_move` past the bound whatever its incumbent read.
- The reverse holds too: a guard with no bound follows the incumbent down.
  In the first ear v4 APL round (2026-10-17) the per-frame fill took
  Tele7abc's `gap_mod_dist_db` to 2.32, under every judged file, and from
  there the hiss-free shelf fill (6.74) tripped the 1.2 step while staying
  inside the accepted range. The guard now takes the ranking's dead-zone
  edge, 10.0, as its bound; keeper off (11.47) and the gated render (27.6)
  still trip it.
- The window guard is part of the programme reading: misaligned windows
  read +19..+28 on SOTI and Vaccin renders without it. It is the 300-4000
  Hz envelope correlation (at least 0.5), not the waveform's (Gaudeamus'
  renders are polarity-inverted), and a 24 dB sanity bound on the
  least-squares gain (Gaudeamus5's renders sit 14.6-14.9 dB under their
  source); the tail is not ranked.
- Each side plays at its own loud frames (75 dB SPL) in both readings: the
  least-squares gain only aligns the cell ratios. Setting the output's
  level through it disagreed with its loud frames by up to 0.88 dB.
- `clean["*"]` in the verdict ledger would relax the hiss flags: the 13
  pre-rule wildcard records read as clean on every listener flag, so
  `calibration_gates` never derives `USER_RULE_FIXED` gates. The ledger's
  "listener.hiss on every v2 plateau" is a harness note on the round-two
  records, which mark the v2 finals clean: not a verdict.
- `resample_poly` (the scipy round trip) puts images at -50..-60 dB into
  the noise cells of a deep residual: it is not a same-percept transform
  for hiss; `resample_soxr` is.
- `file.noise_removed_db` on a 48 / 44.1 kHz pair read -23.49 dB: the trade
  reads the files at their native rates, so `runner.trade_entries` skips it
  (`meta.trade_skipped`) when the rates differ.
- A hiss weight applied after the band smoothing carries a room floor's
  power into the bands above it at their full weight: weight first, then
  smooth (the first build read a 2 kHz low-passed room floor at +9.7 dB of
  pause hiss on the fixtures).
- The pause floor keeper (`modules/pause_floor.py`) refills the pauses with
  the pre-conditioned source at `pause_floor_fill_db` 12, so pause hiss is
  pinned at the tape's level and the pause family plateaus until the engine
  changes (fill 18 is allowed, keeper off excluded by G2; fill 8 lands T's
  pause tail 0.22 dB over the raw tape on the fixed reading, 0.26 under it
  in round 2, 0.44 over before the silent blocks). On 14 APL IA clips the
  keeper refills pauses over the source's
  own floor (+0.56..+1.91 dB at 2 kHz and up), so their pauses read
  hissier than the raw tape's.
- Tele7abc `apl__baseline`: its programme reads 7.1 dB over the cathar
  baseline's (-2.57 against -9.64, both under the threshold) and its pauses
  are dead (depth 37.2, pump +19.4); the constraints exclude it before the
  ranking sees its pause excess of 0.
- The prototype's `shaped_noise` lacked the FRAME / 2 scaling: its hiss sat
  30.1 dB under the spectrum it was shaped from, so its fixture check
  (perfect output -8.79, fresh p95 0.617) never met the saturation above.
- The corpus noise of the fixtures is rumble-heavy and not stationary: its
  200 ms block levels span 6-15 dB p10-p90 at 1-8 kHz, every floor estimate
  reads it under its mean (the window floor up to 13 dB, the pause anchor
  5-22 dB), its anchor validates on 12 of 40 (23 on T's shape, 20 on the
  hiss shape) and its fresh p95 reads 0.67 on the programme against
  0.24-0.31.
  Validating more of it (a median anchor, a slope-following cap, all-frames
  anchors) read perfect outputs up to +22.7 dB: abstaining is right there.
- Music: Gaudeamus1 reads 5 of its 8 windows (the rest hold under two
  readable bands of 1 kHz and up), Gaudeamus5 all 38, and one window of
  `x_vid-20230624-142023`; the other 10 music clips abstain (anchor
  unvalidated, or a programme band under 1 kHz).

The soft programme kept (2026-10-17, `hiss_programme.soft_cells`,
`kept_readings`; the record in `docs/ear_v4_hiss.md`, "The Soft Programme
Kept"). The reference-SNR suppressor took the Tata hiss under speech to
inaudible and the programme 3-12 dB over the hiss 3.5-8.7 dB down, which no
reading saw: kept HF reads 18 dB cells only. Facts that decided its form:

- A hiss cell stands 6 dB over its mean 1.9 % of the time, so the cell's own
  power alone filled the bands above a 4.5 kHz programme with hiss cells and
  read a perfect output -20.3 dB (fixtures); the 3 x 3 neighbourhood mean
  must stand 6 dB over the floor too (0 hiss cells of 800,000 synthetic; 0.8
  % of Tele7abc's true-pause cells).
- The band floor is the band's mean; a tape's roll-off spans 4.5-9.8 dB
  inside Tele7abc's 12-15 kHz bands, so the floor is spread over the bins by
  the source's shape (`shaped_floor`).
- The mean over bands, as kept HF: the median over bands read a 4 kHz low
  pass at 0.00 and put Tele7abc's de-esser bug over the accepted single4s of
  SOTI and Vaccin; the pooled median moved 0.35 dB under a fresh floor.
- It is valid only on stationary hiss: on the fixtures' wandering corpus
  noise (and the raw realistic-v2 pairs) a perfect output reads -23.4 / -7.2
  dB, as kept HF fails there; on the Tata tapes the floor holds (true-pause
  cells -1.12 dB under it, the exponential's -1.59).
- Same-percept p95 0.045 (fresh floor; the rest at most 0.004) on the
  jitter pairs; a perfect output -0.56 / -0.65 (Tele7abc's / tape-hiss
  shape), a 3 dB soft cut -2.76 / -2.87, 40 of 40 monotone.
- The judged files: accepted down to V single4s -26.51 (round one, ranked
  first; the round-one cathar renders -11..-22), rejected T single4s -27.46
  and the de-esser bugs -28.6..-33.9; the v2 / v4_air plateaus -1.5..-9.2.
  The guard: `max_move` 0.05, `bound` -27.0 (music -28.0, captain-pu APL
  -27.32). It does not exclude the suppressor (APL 4.1-7.0 dB, cathar
  3.1-7.8 dB under the same stages without it, the deepest -16.49); a
  move-only guard would, and round three's B (0.07-0.25 dB under A,
  preferred) says it must not be ranked. The listening session decides.

### One loop judging: ear v4 merged with the judging resolution (2026-10-12)

`wip/ear-hiss` (ear v4) and the judging resolution of `feature/v1.4.0`
(`scripts/autotune_judging.py`) each added a tie mechanism to the loop; the
merge keeps one ranking path, and `autotune_judging`'s docstring says which
layer owns what:

- **Ranking** (`autotune_judging`): each reading's band is
  `Floors.band(key, tie)`, the larger of its floor and its grid `tie`. A
  hiss or residual-noise reading's floor is its jitter floor
  (`--jitter-floors`) and never a benign pair's; every other reading's is
  the larger of the two. An entry with a grid `tie` (the v4 hiss excesses)
  keeps the dead zone its grid sets (0): a dead zone raised to the jitter
  floor beside a band of the tie would forgive the jitter twice. The tie
  bands its reading with or without a floors file. Before the merge the
  hiss branch's loop ranked through `tune_restoration.rank_variants`'s tie
  buckets, which the loop no longer calls; the sweep keeps them.
- **Constraints** (`autotune_guards.mark_constraints`, any one tape, explicit
  limits, no floor) are applied in both judging passes and in the offline
  re-judge (`rejudge_autotune.judge_with`, under every rule). Their families
  are first-stage families (`autotune_scoring.split_families`): a
  constraint read only in the learned stage would mark every candidate that
  never qualified as having lost the reading (the adversarial review of the
  merge; the v4 constraints are all dsp, so no round changes). H1's
  `max_move` is the same tie its ranking entry carries, so the two layers
  agree: inside the tie a tie and allowed, past it the worse way behind and
  excluded. A jitter floor wider than the tie (none is measured yet) would
  widen the band alone, leaving H1 the stricter layer. A candidate a
  constraint excludes counts as held back in the round's label
  (`plateau_label`), never "below the floor".
- **Vetoes** stay family-wise (`familywise_vetoes`); v4's DNSMOS BAK and
  SIGMOS NOISE vetoes are hiss readings, so their limits are their jitter
  floors (`veto_floors`).
- **The round's table** shows each reading's band and its source: the grid
  tie, the jitter floor, the benign pair or none.
- **The hiss exemption** (`HISS_PREFIXES`) covers every `dsp.hiss_*`
  reading, the hiss flags (`listener.hiss*`), the residual-noise, gap and
  pause readings (`dsp.pause_depth_speech_db` among them), DNSMOS BAK and
  SIGMOS NOISE.
- **Calibration**: two floors, the sensitivity checks' over every benign
  case (the speech-target `SAME_PERCEPT` set included) and the gates'
  without that set; the `dsp.hiss_*` readings read only the `*_linear`
  cases in both (`calibration_checks.in_scope`). `SCORE_SCHEMA` 8: a
  schema-7 score from either branch lacks the other's readings (R12's or
  the hiss readings).
- **The runner** reads the hiss readings on the pair `audio_io.aligned_pair`
  returns (the GCC-PHAT aligner's lag and polarity, the least-squares
  gain), never the level-matched copy the output's routes are read on; the
  trade keeps both the rate-mismatch skip and the aligner's options. The
  merge's review re-scored the 29 pinned outputs of
  `test_tune_restoration_v4.py` on the merged build: `xcorr_head`
  reproduces the pins exactly; on GCC-PHAT the APL, fill, v4_air and gate
  outputs read identically, the cathar outputs move with their lag (hiss
  readings 0.29 dB at most), and every pinned outcome holds, SOTI's alpha
  2.0 pause tail 0.004 inside its tie (`docs/ear_v4_hiss.md`, "Tape Rows on
  the Merged Aligner"). The pins keep the head-search readings.

### Not built yet

- From calibration v3: R0 naming a mains only when its fundamental stands
  about 10 dB out (after the IA and noise clips confirm it).
- From the oracle check (2026-10-12): `dsp.hf_4k8k` / `dsp.hf_8k16k`
  confirmed oracle-proof (they abstain above R0's programme bandwidth since
  2026-10-12, which cleared 43 of APL's 44 hunt failures, and need the
  oracle re-run; `dsp.whistle` is floored and fails no oracle pair), the
  runner's downmix of a polarity-inverted pair (it cancels the source:
  ein-halleluja), the dropout count no longer
  reading hum and noise removal as holes, and a rate rather than a count for
  `file.dropouts` on long tapes (the shipped Tata outputs read 0-16 holes).
- R3's harness coverage reading, `scripts/replay_post_neural.py` (it would
  rebuild a stage-cache entry's input and replay only the post-neural
  stages; the entry's manifest records the input's sample hash for that),
  the fidelity ladder, `known_ordering_v3.json`, and a gate on
  `file.gain_ride_lu`. The stage cache itself is built (see "The
  self-driving loop").
- The listener's blind threshold (Session 0 left it unmeasured; Session 1
  Block 1 is prepared), and `audibility_check.py` recalibrated on it. The
  tool for the absolute part is built (`listen_ab threshold`, feeding
  `PLAYBACK_SPL_LOUD_FRAMES` through `AI_RESTORE_LISTENER`); no listener
  has run it yet.
- R12's verdict: a blind block of a DTW render against its
  `sync_dtw_enable: false` render on a hunt clip with many holds (cathar's
  `bicentennialinauguration` reads 19 in 15 s), before the key's default or
  the time-base gates move; the stored DTW renders came from the GPU path,
  and the app's CPU DTW path falls back to shift today
  (`docs/configuration.md`, "DTW Sync on Length-Preserving Chains").
- Plan 3.2b (GRPO) and its Optuna baseline wait on the variation pre-check
  above; plan 3.3 (the Mel-RoFormer fine-tune in `tools/msst/`) is not
  started.
- From ear v4: a hiss-free shaped pause fill in `modules/pause_floor.py`
  (room tone kept, 3 dB under the critical threshold above about 1-1.2 kHz,
  behind a switch), without which the pause family plateaus; tape versions
  of the hiss degradations (the tape's own true-pause noise regenerated);
  BS.468, A-weighting and ISO 226 cross-checks and B's programme-shaped
  excess; the programme tail, ranked only after a clean re-measure; a dither
  limit on the kept-HF cells (requantisation moved them 2.9 dB on Vaccin); a
  programme gain for low-correlation music renders (least-squares alignment
  shrank Gaudeamus1's 8-11 dB, past the window guard); the programme
  reading's saturation 13-14 dB under the source; and the blind block of
  the specification (shipped B against a 6 dB stronger HF denoise, shaped
  fill against the shipped pauses, a tape-noise blend continuum in 3 dB
  steps that places L_crit), at the user's plateau only.

### The user's decisions (2026-10-08)

- Learned training only where it fits: GRPO for a per-tape settings policy,
  later, and only if a variation test beats the single best preset against
  an Optuna BO baseline; an anchored fine-tune of the Mel-RoFormer in the
  isolated `tools/msst/` venv; the judge is the preference head.
- No generative engines, and no audio-LLM or cloud judge.
- The user's ear is asked in short A/B batches (10-20 picks) through
  `listen_ab.py`, and no default ships without a listening session.
- Approved installs: cathar 0.8.0 (provisioned 2026-10-09, identical on
  every stage this app calls, the installers pin it); not provisioned yet:
  silero-vad, zimtohrli, optuna (dev group) and the `tools/msst/` venv.
  MERT's CC-BY-NC-4.0 weights stay (research use).

## Tuning on real tapes

```text
scripts/tune_restoration.py prepare|run|score|report|listen|all
    --name N --tapes-dir DIR --grid scripts/tune_grids/<grid>.yaml
```

The driver cuts excerpts, runs
every grid variant in a fresh interpreter with its own `config.yaml`
(overrides validated against the app's typed settings and read back from a
child before a run), scores, and writes `experiments/tune_N/scoreboard.md`.
The app is launched on batches of excerpts whose command line stays under
24,000 characters (`COMMAND_LINE_BUDGET`; 240 corpus clips in one call
raised WinError 206 on 2026-10-11), and a variant stopped part-way resumes
on the excerpts with no restored output yet.
Gates veto relative to the engine's own baseline; every variant is ranked;
`listen` renders the worst windows as WAV cuts. An engine in the grid may
carry `env:` (for example `AI_RESTORE_CATHAR_BIN` pointing at a second cathar
build under `experiments/cathar-<version>/`), so an upgrade is scored beside
the validated binary before anything is replaced. A `score` with
`--metrics dsp --rescore` merges one family into the stored pairs and
re-reads the verdicts. A v3 grid's `ranking` entry is a reading spec
(`{target, family, dead_zone, scale, weight}`), ranked by its two-sided
distance and weighted in the mean rank; the v1 and v2 grids' `up` / `down`
entries still work.

Full-tape listening sets go to `D:\Tata\New folder\variants\<tape>__<variant>`
through `experiments/tata_listen/run_listen_variants*.py` (the app writes
beside its input, so outputs are moved away between variants) and are scored
with `score_listen.py <slug>`. `score_round0.py` re-scores every file the
ledger judged, one report per set and tape (see "Round 0 (2026-10-09)").

## The self-driving loop

```text
scripts/autotune_restoration.py --engine cathar|apl --tapes tapes.json
```

This is the user's instruction made executable: "the AI human ear must
refine itself a few rounds and ask me to listen only when it cannot refine
any more". Each round proposes the neighbours of every knob's current value
(plus the combination of the moves that helped on their own), restores every
tape per candidate keeping only the audio, scores all tapes of the round in
parallel, and accepts a candidate only if its mean rank across tapes beats
the incumbent, it wins on at least half the tapes and it fails no more hard
gates. It stops only at a plateau (the user's rule: "stop only when no more
improvements are possible"; `--rounds`, default 20, is a safety cap); `log.md`
and `state.json` under
`experiments/autotune/<engine>/` make it resumable. Run both engines at once
only on different source paths: the app's work directory is
`.temp_work_<stem>` beside the source, so APL runs on NTFS hardlinks of the
tapes (`experiments/autotune/tapes_apl.json`). Run loops from the main
checkout only: a git worktree lacks the ignored gates file, benign floors
and models the defaults point at (the pipeline-runner skill, worktrees).

The loop scores only the families its grid reads (`--families auto`, the
default since 2026-10-11): dsp always, plus the family of every ranking
entry, learned veto and gate name, so `tata_v3` scores dsp, speech and mos
and `music_v3` dsp, stems and mos. The stems family separates every output
with BS-RoFormer and took about 74 of SOTI's 113 scoring minutes in the
APL v6 speech round, which ranks and vetoes on none of it; the rounds before
2026-10-11 paid for it on speech. Scoring then runs in two stages (plan 1.5,
`scripts/autotune_scoring.py`): the families the ranking and the gates read
on every candidate, the learned vetoes' families (speech and mos on
`tata_v3`, mos on `music_v3`) only on the incumbent and the candidates that
qualify without the vetoes, kept in `<slug>.learned.score.json`. A veto can
only take a qualification away, so the verdicts equal one pass over every
family; only contenders can carry a "vetoed" mark now, and a plateau round
runs no learned model at all. A named `--families` list runs in the same two
stages; a candidate scored on every family at once is not scored again.

Each round renders the incumbent first and reads the held partials (the
tonal persistence both engines' music settings key on) that the app's
scanner printed for every tape in its `run.log`
(`scripts/autotune_material.py`). When no tape reaches the incumbent's
music floor, the music-only knobs are left out before rendering (cathar's
11 `cathar_music_*` keys; APL's music model, stem path and background
floor), and so is a floor move no tape crosses. On the speech tapes the
readings are 0.0004-0.0055, so the floor's 0.005 move stays live:
Gaudeamus 5 would cross it. Before this, round C4 rendered every music key
on the speech tapes (about 3 minutes each) for the hash to discard.

Knobs must be the live ones: on the Tata tapes the scanner reads a spectral
flatness of 0.022, under `apl_tonal_flatness_max` 0.035, so APL takes its
tonal path (`apl_spectral_alpha_tonal`, `apl_noiseprint_tonal_s`, plosive
tamer skipped) and `apl_spectral_alpha` candidates came out bit-identical to
the incumbent. The knob tables were re-read from the code paths on
2026-10-09 (the loop's docstring holds the details):

- `apl_tonal_flatness_max` is live with the subtraction stage off: the
  plosive tamer and the hum canceller's series length read it. It left
  `SUBTRACTION_KNOBS`. The tuned finals carried 0.01 where the shipped
  default is 0.035, so tuned APL ran the tamer and the 40-harmonic hum
  series and shipped APL does not; round A0 settles which is better.
- `apl_music_persistence_min` is live with the stem path off: it picks the
  tapes that take `apl_music_neural_model`, now a knob (`MODEL_KNOBS`,
  validated as a model name).
- `apl_noiseprint_tonal_s` also feeds the stem path's background suppressor,
  so it is dead only with the subtraction off and the stem path off (or its
  floor at 0 dB).
- The cathar music profile's `cathar_music_enable_deesser`,
  `cathar_music_expander_depth_db` and `cathar_music_crt_notch_q` are knobs;
  the music rounds could not move them before.
- `env:AI_RESTORE_CATHAR_BIN [None, 0.7.6]` is gone: once the installers
  provisioned 0.7.6 it rendered one binary twice. Round C0 (2026-10-09) needed
  no binary knob either: cathar 0.8.0 gave identical output on the identity
  clips and the installers now provision it.
- Round C2 (2026-10-09): cathar's de-esser shape is a knob,
  `cathar_deesser_freq` `[4000, 5000, 6000]` and `cathar_deesser_bands`
  `[1, 3]` beside the threshold; all three are inert only when neither speech
  nor music runs the de-esser.
- Round C3 (2026-10-09, music profile): `cathar_music_beta`
  `[0.005, 0.01, 0.02, 0.04]`, `cathar_music_enable_repair` `[True, False]`
  and `cathar_music_dewind_cutoff` `[40, 60, 80]`, seeded at music's old
  values (0.02, on, 80 Hz: it ran `cathar_beta`, `cathar_enable_repair` and
  `cathar_dewind_cutoff`). All three join `_CATHAR_MUSIC_KEYS` (inert with
  `cathar_music_profile` false); the floor is inert under Wiener, the cutoff
  with `cathar_enable_dewind` false, and `cathar_repair_strength` now with
  both repair switches false. The profile's expander depth and CRT Q stay
  the round's other knobs. R11 (`dsp.lf_programme_db`) is the reading built
  for the low end these move, but no grid ranks it yet: `music_v3.yaml` and
  `tata_v3.yaml` hold no R11 entry, and their readings cannot see 40-200 Hz.
  It was read on round two's accepted files (2026-10-10, the r2 set scored
  from this branch): the 12 music clips and four speech tapes span -10.5 to
  +0.6 dB, most -4 to +0.6, with `m_wbbddanceswithwolves` at -10.5 (APL) and
  -9.2 (cathar), both "music fine". A two-sided dead zone admitting every
  accepted file would be about 5.7 dB wide, while the dewind cutoff moves R11
  by about 1 dB (calibration: -0.10 / -0.43 / -1.05 dB at 40 / 60 / 80 Hz), so
  ranked it could not see the round's own knob. R11 therefore stays
  display-only in `music_v3`; C3's low end is judged by the existing music
  readings, the MERT and stems vetoes, and the listener. A paired form (the
  candidate against the incumbent's R11 on the same clip, like the paired
  `hf_8k16k` air check) would see it; the loop has no paired entries yet.
- Round C4 (2026-10-10, declip and dehum): `cathar_declip_mode`
  `["always", "detected"]` and `cathar_dehum_freq` `["scanner", 0]`, seeded
  from the app at `always` / `scanner` (every release's behaviour), so each
  proposes its one alternative. Both stages run on speech and music, so
  neither joins `_CATHAR_MUSIC_KEYS`; each is inert with its stage switch
  off (`cathar_enable_declip`, `cathar_enable_dehum`, both now in
  `SWITCH_DEFAULTS`, true). `tune_restoration.known_config_fields` types
  them as choice sets, and `_choice_problem` refuses a bool (False equals
  zero) and sorts a mixed set by `str`. Expect inert renders: cathar 0.8.0's
  declip returned a synthetic file peaking at 0.5 unchanged, so on tapes
  peaking under 0.95 `detected` should hash equal to the incumbent, and
  with `--adaptive` on, `--freq 0` matched `--freq 50` on synthetic 50 Hz
  hum, so `0` moves only tapes the scanner read as hum-free (its vote reads
  the first 0.7 s). `detected` reads the clip count over every channel
  (`profile.has_clipping_any_channel`), so the dead-channel tapes, whose
  mono mix never reaches 0.985, are judged on their live side; a render
  whose scan left no verdict declips as `always` and logs a WARNING.
- Hiss and scan knobs (2026-10-17, for the ear v4 rounds): both engines
  take the hiss-free fill (`pause_floor_target`, `pause_floor_shape`,
  `pause_floor_shelf_hz`, `pause_floor_hf_cut_db`) and the scan's searches
  (`crt_detect_mode`, `scanner_mains_vote`, `azimuth_estimate_method`,
  `sync_dtw_enable`); cathar the print guards
  (`cathar_noiseprint_mute_guard`, `cathar_noiseprint_min_depth_db`,
  `cathar_noiseprint_shallow_action`, `cathar_noiseprint_method`); APL the
  post-neural stages (`apl_post_neural_suppress` and its residual gate,
  `apl_neural_level_restore`, `linear_air_gate`, `apl_burst_ceiling` and
  its depth, `apl_surgical_ring_guard`, `apl_depop_min_prominence_db`).
  Inert rules: the keeper off kills everything it reads
  (`PAUSE_FLOOR_KNOBS`); a flat fill reads no corner or cut, a low-pass no
  cut; no print kills every print guard (`NOISEPRINT_KNOBS`); the depth
  guard's action needs the guard on; the post-neural parameters need their
  switches; `apl_suppress_gain_floor_db` is dead only with both suppressors
  off (`tests/unit/test_autotune_hiss_knobs.py`).
  Default changes: they go to the C3 listening session. Tests:
  `tests/unit/test_autotune_cathar_c4_knobs.py`,
  `tests/unit/test_cathar_round_c4.py`.
- `apl_neural_model` was `[None, ROFORMER, ROFORMER_AGGR]`: once the
  Mel-RoFormer became the app's default, None and ROFORMER named one
  setting, so a run from the defaults rendered the default every round
  (hashed inert) and never proposed the aggressive model. It is
  `[ROFORMER, ROFORMER_AGGR]`, seeded from the app; a saved state holding
  None for a knob whose list has no None is seeded too. A list must never
  hold None and the app's default value together.
- Rounds A1 and A3 (2026-10-09): APL's `linear_air_gain_db` is
  `[0.5, 1.0, 1.5]`, the plan's set (0 is `enable_linear_air` false; +2 dB,
  which round 3 rejected, is left out so a v1 / v2 grid without the
  reversal cannot step to it either) and its new
  `linear_air_freq_hz` `[6000, 7500, 9000]`, both inert with the shelf off;
  the new `expander_attack_s` `[0.02, 0.04, 0.08]` and `expander_decay_s`
  `[0.12, 0.18, 0.3]` are shared knobs (cathar runs the same polish
  expander), inert with the expander off. Seeded from the shipped 7500 Hz
  and 0.04 / 0.18 s, each first proposes its two neighbours; all three new
  keys are post-neural, so their APL moves replay the stage cache. The
  grids' reversal `rejected_above: 2.0` (tata_v3 and music_v3 as of this
  date) judges gain and corner as one shelf (below): of A1's nine shelves it
  refuses +1.5 dB at 6000 Hz.

`INERT_WHEN` entries are `(when, knobs)`: the knobs are dead when every
switch in `when` holds its value (a switch the incumbent does not set reads
`SWITCH_DEFAULTS`, what the repository's `config.yaml` resolves to).
`DURATION_BOUND` drops a `cathar_noiseprint_duration_s` move when, on every
tape, the new value and the incumbent's give the same probe (ffprobe
lengths, a 1 s margin around the 20x switch). After rendering, each
candidate's audio is hashed per tape by the loop's own exact hash
(`exact_audio_sha256`: the samples as stored, DC kept, streamed in blocks;
cached in `cands/<cid>/<slug>.audio_sha256.json`, keyed by file key and
hasher name). `auditory.audio_sha256` is not used: it removes DC (an
offset-only difference would read as inert) and decodes the whole file
(about 1.3 GB per hour of stereo per copy). A candidate identical to the
incumbent, or to a candidate proposed before it, on every tape is logged as
`inert (= <id>)` in `log.md` with a "knob-table finding" line, kept in
`state.json` and never scored; inert renders had cost about 30% of a round.
Tests: `tests/unit/test_autotune_inert.py`, `test_autotune_knob_tables.py`
and `test_autotune_restoration_driver.py`.

An ear v3 grid adds three guards (plan 1.5, `scripts/autotune_guards.py`); a
v1 or v2 grid has none of these sections and its rounds are judged exactly
as before:

- `reversals:` (verdict reversal): `{key, rejected_above, verdict}` (or
  `rejected_below`) per ledger boundary. A move heading the rejected way
  that ends at the value or past it is refused before rendering and logged
  `refused (verdict reversal)`; a move back is never refused. `tata_v3` and
  `music_v3` carry `linear_air_gain_db` `rejected_above: 2.0`
  (`r3-air-preference`: +1 dB over +2 dB and off on all four tapes). A key
  no engine tunes stops the loop. Every round-3 shelf sat at 7500 Hz, so
  since `linear_air_freq_hz` is a knob the guard restates a candidate's air
  gain, and the incumbent's, as the 7500 Hz shelf with the same power gain
  over 4-8 kHz on a flat spectrum (`autotune_guards.in_ledger_units`,
  `ledger_air_gain`; unchanged at 7500 Hz or with no corner set). +1.5 dB at
  6000 Hz reads +2.57 and is refused, also when `combine` merges the two
  allowed winners +1.5 dB at 7500 Hz and +1.0 dB at 6000 Hz; the gain-only
  check let that through, and the grids' R1 readings cannot see above
  4.5-5 kHz on the linear tapes to catch it. The limit: one band stands for
  the percept. +1.0 dB at 6000 Hz reads +1.72 and passes, yet on the en
  fixture it lifts R2's 's' level and `hf_4k8k` as much as the rejected
  shelf (0.401 / 0.379 against 0.402 / 0.375), and over R2's 4-12 kHz band
  +1.5 dB at 6000 Hz would restate to +1.92 and pass; A1's listening session
  settles the lower corner, not the guard. Tests:
  `tests/unit/test_autotune_guards.py`.
- `audibility: {offset_db: 24.0}` (the tie): after the hash, each live candidate
  goes through `auditory.compare_files` against the incumbent per tape, stopping
  at the first audible tape (sidecar `cands/<cid>/<slug>.audibility.json`).
  Inaudible on every tape is a tie: not scored, never accepted, logged
  `tie (inaudible on every tape)`. A pair it cannot compare reads audible. It
  decodes whole files (about 1.3 GB per hour of stereo per copy) while no scorer
  runs; about 1 s per 2 min. The sidecar is keyed on `auditory.VERDICT_RULE`
  (`2026-10-09-head-tail`), the two file keys and the offset
  (`autotune_guards.audibility_key`), so a verdict an older rule wrote is read
  again, never reused: the 2026-10-09 changes (the single-frame event, the
  head-and-tail length flag, the NaN shortcut) invalidate stored verdicts with
  no delete. Bump `VERDICT_RULE` with every change to what `compare_files` calls
  audible: the masking model, the share, the event clause, the length, channel
  or NaN flags, or the identical shortcut.
- `vetoes:` (learned median veto), family-wise since 2026-10-11
  (`autotune_judging.familywise_vetoes`, backlog E3 part b). Per listed
  reading, each tape's median against the incumbent's counts as worse or
  better past the scorer's jitter floor (`--jitter-floors`); an exact
  one-sided sign test across the tapes, Holm-corrected over the grid's veto
  readings at 0.05, must reject, and at least a third of the tapes must have
  moved the `worse` way by more than `floor_multiple` times the reading's
  floor from `--noise-floors` (a `reward_noise_floor` report of the Session 0
  benign pairs; default `experiments/reward/noise_floor_benign.json`,
  `floor_multiple` 1 in both v3 grids), a reading the incumbent has and the
  candidate lost counting as such a move. `resolvable_clips` is seven over
  four to six readings (2^-n under 0.05 / m): on fewer tapes, the four speech
  tapes included, every tape must move the worse way past the jitter floor
  instead, a rule the formal test cannot back, kept so the speech loop keeps
  a learned guard. The vetoed candidate is logged `vetoed` and cannot win
  (`reward.gate_constraints(learned_veto=...)` must read "ok"). Until
  2026-10-11 one tape past the limit vetoed (`autotune_guards.learned_vetoes`,
  kept for the offline re-judge): 37 of the cathar music loop's 54 vetoes
  stood on one tape-reading and 15 vetoed APL music candidates won at least 6
  of 12 clips. A hiss reading's limit takes its jitter floor
  (`autotune_judging.veto_floors`). A veto without a
  floor stops the loop before the first render, and so does a floor that is not
  above 0: CER read at `--repeats 1` can have a p95 deviation of exactly 0, and
  a limit of 0 would veto any move the worse way, Whisper's own jitter included.
  The refusal names the reading and says which cause it met
  (`reward._floor_error` picks the message): a reading the report lacks, or
  holds no number for, gets `reward.NO_FLOOR`, which names `--families` (the
  report reads only the families it was run with, and
  `python -m scripts.reward_noise_floor` runs dsp alone by default); a floor of
  0 or below gets `reward.NO_BENIGN_FLOOR`, which asks for `--repeats 2` or
  more, or `resample_roundtrip` in `--transforms`.
- `constraints:` (ear v4; `tata_v4.yaml`, the loop's default `--grid` since
  2026-10-12, and `music_v4.yaml`): hard guards on dsp readings
  (`reward.parse_constraints`, `autotune_guards.mark_constraints`). A
  candidate whose reading, on some tape, is worse than the incumbent's by
  more than `max_move` (default 0) and beyond `bound` (default: none), or
  that lost a reading the incumbent has, is marked `constrained` (cell
  `guarded`, a `guard:` line in `log.md`) and cannot win
  (`reward.Constraints.dsp_guard` excludes). `judge` marks them on every
  candidate in both judgements of the two-stage scoring: their families are
  scored in the first stage (`autotune_scoring.split_families`). A key that
  does not split as `metric.side.stat` (a registered metric, side output or
  delta, a stat `runner.FLAT_STATS` holds) stops the loop.
  `describe_guards` counts them: tata_v4 reads "8 learned vetoes, 1 verdict
  reversals, 17 constraints, audibility tie at a 24 dB masking offset" with
  the benign-pair floors.

Before a v3 round, write the floor report on the shipped APL and cathar
outputs of the tuning tapes, with every family the grid's vetoes name
(`tata_v3`: speech and mos; `music_v3`: stems and mos):

```text
python -m scripts.reward_noise_floor SOURCE=OUTPUT ... --families speech,mos,stems
    --repeats 2 --out experiments/reward/noise_floor.json
```

Scoring every family on the four tapes took about 7.3 min per candidate in
the v4 round; the report pays that for the reference and each transform,
per repeat, and each repeat after the first scores through its own cache.

At 3x those near-copy floors the vetoes would refuse moves on the judges'
own jitter (see "Round 0"). Measure the vetoes' floors on pairs a blind
listener could not tell apart, in a report of their own, and pass that as
`--noise-floors` (the loop's default since 2026-10-09: CER 0.038, speaker
cosine 0.0063, UTMOS 0.051, SCOREQ 0.136, SIGMOS COL 0.226 / DISC 0.153,
MERT 0.0029, Audiobox PQ 0.205 / PC 0.167; the music floors come from speech
pairs and stay provisional until a music pair is heard); Session 0's pairs are
listed in `scripts/reward_noise_floor.py`'s docstring and the ledger's
`round: session0` records:

```text
python -m scripts.reward_noise_floor --benign-pair SOURCE OUTPUT_A OUTPUT_B LEDGER_ID ...
    --families dsp,speech,mos,stems --repeats 2 --max-seconds 125
    --out experiments/reward/noise_floor_benign.json
```

Session 0's listener told no two restorations apart at an unrecorded
playback level, so those floors are generous: weigh each veto's
`floor_multiple` against them before the first v3 round.

Each refused, tied or vetoed candidate gets a line in `log.md` naming the
boundary, or the reading, how many tapes moved which way and the test that
tripped. Tests: `tests/unit/test_autotune_guards.py` (the loop),
`test_autotune_guard_rules.py` (`reward.py`'s vetoes and boundaries) and
`test_autotune_familywise_veto.py` (the family-wise rule and its null).

Judging resolution (2026-10-11, `scripts/autotune_judging.py`, backlog E2
parts a and b). The ranking path takes `reward.py`'s tie band per reading:
two candidates tie when their distances differ by no more than the larger
side's `min(floor, max(0, |x - target| + floor - dead_zone)) / scale`, the
floor being the reading's Session 0 benign-pair floor (`--noise-floors`), and
every dead zone is bounded from below by that floor. A rank is one plus
every candidate ahead beyond the band plus half of every tie, which with no
floors is exactly `tune_restoration._ranks`' average rank: without the
report the loop warns and judges as before. A tape is a win only when the
candidate out-ranks the incumbent and moved past the band the better way on
at least one reading there (`TapeTable.better`): a rank among a group can
otherwise carry a tie, or a sub-floor gain beside a real loss, into a win
through third candidates (reviewer probe, 2026-10-11: tilt 0.7 dB/oct nearer
under its 0.85 floor, body 1.5 dB further past its 1.2, ten candidates in
the incumbent's band, two tapes won and accepted). Without floors this adds
nothing, since out-ranking on exact ranks needs some reading strictly
better; on the stored v6 and v7 rounds it changed no win. An incumbent
sitting less than a floor outside a zone reads as
inside it within noise, so a candidate pulling it into the zone ties.

The hiss exemption (the user's rule, 2026-10-11: the ear must be very
critical of VHS hiss under speech or singing and in pauses):
`autotune_judging.HISS_PREFIXES`, the user's list (every `dsp.hiss_*`,
`dsp.residual_noise_db`, every `dsp.gap_*`, `dsp.pause_depth_db`,
`dsp.hf_8k16k`, `mos.sigmos_noise`) plus the other residual-noise readings on
the same rule (`dsp.pause_*`, `file.noise_removed_db`, `mos.dnsmos_bak`),
never take a tie band, a dead-zone floor or a veto limit from the benign
pairs (residual noise 16.4 dB, noise removed 19.9 dB, pause depth 5.8 dB,
gap HF excess 4.5 dB there). They take `--jitter-floors` (default
`experiments/reward/noise_floor.json`: the reference repeat, `shift_1`,
`requantise_16`; residual noise 0.0018 dB, pause depth 0.0013, gap LSD
0.048), and `jitter_floors` refuses a report holding a benign pair. Tested
in `tests/unit/test_autotune_judging.py` and
`test_autotune_judging_loop.py`. On `tata_v3` the pause readings still tie
on every speech candidate, through the grid's own zones (pause depth +-16.0
dB around 19.0, gap LSD 9.05, gap modulation 10.0, set to admit every
accepted file), and `music_v3` ranks no hiss reading at all: making the
ear critical of hiss takes the hiss branch's readings ranked with zones of
their own, which this resolution leaves to that work.

`log.md` gains a per-reading distance table per round (each scored
candidate's distance per tape, `*` where it moved past its band against
the incumbent, the floor and its source under each reading). A round whose
every scored candidate stayed level with the incumbent, with no veto, gate
or reversal holding one back, ends "no candidate qualified: every move
below the Session 0 benign-pair floor (N readings unmoved)" (without benign
floors: "every candidate level with the incumbent inside the dead zones");
otherwise it reads `plateau` with the counts level, moved and lost, and
held back.

Offline re-judge (CPU, the stored `*.score.json` and `*.learned.score.json`;
the v6 rounds hold every family in `*.score.json`):

```text
python -m scripts.rejudge_autotune LOOP_DIR ... --grid GRID
    --noise-floors BENIGN.json --jitter-floors JITTER.json --null-draws 400
```

Run 2026-10-11 on `autotune_v6/{apl,cathar}`,
`autotune_v7/cathar` (`tata_v3`) and `autotune_v6_music/{apl,cathar}`,
`autotune_v7_music/cathar` (`music_v3`), the old rule reproducing every
stored decision:

- Speech: no decision changes. Vetoes 2 -> 1 (APL: pause floor off was
  vetoed on SIGMOS DISC at two tapes), 8 -> 3 (cathar: noise print off,
  repair strength 4 and pause floor off stay), 0 -> 0 (C4). Ties with the
  incumbent's mean rank 24 -> 25, 23 -> 23, 24 -> 24. The labels stay
  plateaus: fill 8 dB, pause floor off and a few cathar moves moved a
  reading past its band and lost.
- Music, the band alone: no accepted move survives. Each (APL decay 0.12 +
  notch Q 60, cathar attack 0.02, knee 4, knee 0, music beta 0.01) moved
  the benign-floored readings by a median across clips of a tenth of the
  floor or less; the music floors are speech pairs' and stay provisional.
- Music, the family-wise veto alone: 11-18 vetoes per round -> 0-2, and the
  old acceptances stand; cathar round 4 would accept the expander off and
  C3's rounds music repair off (9 of 12 wins, vetoed twice on MERT before).
- Music, both: APL round 1 accepts `pause_floor_fill_db` 8 (7 of 12 wins,
  old-vetoed on Audiobox PQ at two clips), C3 rounds 1 and 2
  `cathar_music_dewind_cutoff` 60 (6 of 12), both winning on `stems.*`,
  whose floors are jitter only (the Session 0 pairs had no background).
  Neither is adopted: each needs a blind block, and the fill reads more
  pause texture on a grid that ranks no hiss reading.
- Null simulation (400 draws a round): the any-tape rule vetoed 0% at
  jitter size, 44% (speech) and 68% (music) of candidates moved by
  benign-pair-size noise and 10-55% of sign-flipped real candidates; the
  family-wise rule 0%, 0.5% and 0-1.5%.

Same-percept transforms (backlog E2 part c) recalibrate only the
calibration's effect-versus-floor checks (see "Calibration v3 and the v3
grids"); they never reach a tuning dead zone.

A scorer holds about 4 GB of RAM; `--parallel` (default 2) caps the tape
scorers per engine, and two engines plus the corpus scorers exhausted a
62 GB machine.

Disk is the other budget. The scorer keeps every resampled array under
`<out>/cache` (three rates, about 1 GB per hour of tape per output) and the
loop keeps every candidate's audio under `<out>/cands`; two engines on full
tapes grew 250 GB of cache and 130 GB of candidates in three days and filled
the drive, which killed one loop mid-render (the app could not write its
WAV; the other loop survived). The driver now deletes an output's cache
entries the moment its `<slug>.score.json` is written (the source side is
reused every round and stays). Candidate audio of past rounds is only worth
keeping until the listening set is built from the finals
(`experiments/tata_listen/build_listen_v2.py` copies them); delete
`cands/*/*.wav` of a finished loop after that. When a loop dies this way,
remove the half-written candidate directory and the `.temp_work_*` folder
beside the source before relaunching, or the resume trips over them.

Render time is the third budget. `--stage-cache <out>/stage_cache` (with
`--stage-cache-gb`, 50 by default) sets `AI_RESTORE_STAGE_CACHE` for every
candidate's render: an APL candidate whose knobs all act after the neural
denoiser (air, the sibilant guard, the expanders, the pause floor,
`apl_expander_depth_db`, `apl_music_bg_floor_db`, `loudnorm_target_lra`:
17 of the 29 APL knobs on 2026-10-09, with the air corner and the expander
timing) replays the incumbent's chain and
model output instead of rendering them. It caches the neural stage's two
intermediate WAVs only, never a final output (the 250 GB lesson above):
about 2.5 GB per tape-hour per distinct pre-neural setting, least recently
used evicted first, so the incumbent's entries, hit every round, stay. Each
candidate's `timing.json` holds
`{"wall_s", "stage_cache": {"hits", "misses"}}`, counted from the
`[Stage Cache] Hit` / `Miss` lines of its `run.log`; a candidate
with misses moved a pre-neural knob (or the entry was evicted or refused:
the `Not stored` line says why; editing a module or installing a package
while a loop runs refuses the stores of the candidates then rendering, since
they may run other code than their key names). A hit renders the same
samples as a fresh render (`tests/unit/test_stage_cache_replay.py`; on real
tape, the
orchestrator's check in plan 1.5). The saving per candidate against the
uncached ~205 s on four tapes is measured on the first loop round run with
the flag; record it here. cathar never uses the cache. The key, its
allowlist and the invariants are in the audio-restoration-engine skill and
`docs/configuration.md`, "Stage Cache".

Only when both engines have plateaued: run
`scripts/audibility_check.py` on each final against what the user last heard
on the same tape, and do not ask when every tape reads inaudible (the v3
sibilance plateau would have been a wasted session). Then produce the
listening set (`experiments/tata_listen/build_listen_v2.py` copies each
final's outputs beside their sources and writes an index with the settings
diff and the harness readings), send the index, and ask the user, through
`python -m scripts.listen_ab` (ABX first, 24 trials) so the answer lands in
the ledger. Feed a winner back into
`config.yaml` and `modules/config.py` with the measured-effect comment, then
re-base the cathar identity reference (see the audio-restoration-engine
skill). Two rules from the 2026-09-26 feed-back: a shared key the speech and
music plateaus disagree on gets a music-profile key, never a compromise
value; and a final carries every knob the loop touched, including ones an
accepted switch made inert (the subtraction's factor, probe and native
suppressor once `apl_enable_spectral_denoise` is off), so read the code
before feeding a knob back and leave the inert ones at their shipped value.
Before a default ships, it also gets one `artifacts/realistic-v2` and
IA-corpus confirmation (AGENTS.md section 3) and a listening session.

## Working on the harness code

- `tests/conftest.py`'s session hook rebuilds `assets/coverage.svg` and runs the
  per-file coverage gate only from a `coverage.xml` / `coverage.json` this
  session wrote in the working directory (their modification times at start and
  finish compared): `--no-cov`, a run without `--cov` and a stale report another
  run left there touch neither. A targeted `--cov` run with an xml report
  moves the badge in its working directory, and one with a json report runs
  the per-file gate on partial coverage (exit 1), so run it from an empty
  directory, never the repository root.
- With other agents in the checkout, give coverage a private data file
  (`COVERAGE_FILE` under the temp directory) and point `--cov` at the
  package directory (`--cov=scripts/restoration_quality`); `--cov` on a
  single module loads numpy twice and fails. Pass `-p no:cacheprovider`.
- Never run a tool with a directory of stray `.py` files as the working
  directory: Python imports from it first. On 2026-10-09 radon, run from a
  session scratchpad, imported an old probe named `six.py`, which rewrote
  `scripts/cli_paths.py` before it failed. Run from the repository root or
  an empty directory, and keep probe scripts in their own folder, run by
  path (the test-runner skill).
- No torch or transformers import at module level under
  `scripts/restoration_quality/`: the dsp family must run without them.
- `python -m scripts.listen_ab` and `python -m scripts.reward_noise_floor`
  run as modules from the repository root; every script refuses a path
  outside the repository and the temp directory unless its root is in
  `AI_RESTORE_DATA_ROOTS`.
