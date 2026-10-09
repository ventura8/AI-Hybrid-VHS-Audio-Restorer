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
   median-only vetoes beyond 3x their benign floor, never in the score.
1. Speech, music and mixed have separate grids and thresholds.
1. A reading is admitted by a monotonic response on the calibration
   degradations and a benign floor. The user's verdicts are falsifiers, and
   each reading's target complaint and direction is written down before
   Round 0 (the critique: about 25 readings against 10-20 verdicts overfit
   if chosen by "reproduces the verdicts").

### The new readings

Every threshold and target below is **uncalibrated** until Round 0
(re-scoring the stored listening files) derives it on Tele7abc and tests it
on SOTI and Vaccin. Round 0 ran on 2026-10-09 and only proposed values (see
"Round 0 (2026-10-09)"); none is adopted yet. The readings pass their
synthetic tests; their module docstrings hold the full measurements.

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
  `line_hz` (falling back to 50 Hz and 15625 Hz). The summed rule of
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
  band is cleared. `sib_peak_hz`, `sib_rel_amp_db` and the Romanian s-comma
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
  as a flag).
- **R7, gain riding** (`file_metrics.py`: `file.gain_ride_lu`, the p95, with
  `file.gain_ride_std_lu` and `file.gain_ride_deriv_lu`). EBU short-term
  loudness of the raw source and output over windows standing 10 LU over the
  source's floor, the static offset removed, the sync lag applied first.
  The accepted v2 finals of the speech tapes read p95 0.82-1.80 LU, so the
  design's 1.5 LU flag would fire on three accepted files: display-only
  until the `loudnorm_ride` degradation calibrates it.
- **Sync** (`file.sync_drift_ms`, `file.sync_offset_ms`,
  `file.sync_unmatched`, the per-anchor lags). The envelope correlation of
  8 s anchors at the start, centre and end of the raw pair. Hard gates: drift
  and offset at most 40 ms (one PAL video frame), no unmatched anchor. The
  32 v2 final pairs read lags of 4.7-5.4 ms, drift 0.02-0.26 ms, none
  unmatched.

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
cannot know the playback level) and `MASKING_OFFSET_DB` 24 (at 20, SOTI's
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
    [--anchor LABEL=PATH] [--start S] [--seconds 8-12] [--trials N]
```

The page binds 127.0.0.1 only, so family tapes never leave the machine.
ABX runs 24 trials by default and at least 16 (17 of 24 is p 0.032 at
power 0.77). Pair plays every pair once with an explicit "same", plus one
replicate at the end (with the `--anchor`, the round-one alpha 2 hissy cut,
when one is given). Blend mixes the candidate into the incumbent and places
30 trials QUEST-style. Cuts are 8-12 s, aligned (lag and polarity),
loudness-matched on speech-active frames within +-20 dB, and placed where
the NMR says the first two stimuli differ most. The block's answers reach
the ledger in one write at its end. Set `AI_RESTORE_DATA_ROOTS=D:\Tata`
first.

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
  self-driving loop"). The group reward itself is not used by the loop yet.
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
`target` under `reward:`, else under the v3 grids' `ranking:` (tata_v3 15
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
and what must not: `air_shelf_*`, `spectral_tilt_*`, `pause_residual_*`,
`hifi_compander_mistrack`, `loudnorm_ride`, `phasey_resynth`,
`linear_bandwidth`, `treble_dropouts`, `sibilant_islands`, `sync_drift`.

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
`reward.py` read the same entries. Every value is uncalibrated and its
comment says what it stands on. Speech-side readings weigh 1.5 (the P.835
SIG/BAK prior, speculative). Not ranked: `hf_4k8k` and `hf_8k16k` (backstop
gates), residual noise, `lkr` and the sync; learned judges sit under
`vetoes:` only. Their `vetoes:`, `reversals:` and `audibility:` sections are
the loop's guards ("The self-driving loop"); `tune_restoration.py` reads
none of them. The calibration writes under its `--out`, while the drivers
still default `--gates` to the v2 `experiments/quality_calibration/gates.json`:
a v3 round passes `--gates experiments/quality_calibration_v3/gates.json`.

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
The report, with every number, is `docs/ear_v3_round0.md`. Nothing is
adopted into the gates or grids yet. The facts that will decide defaults:

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
  the audibility check calls inaudible before the first v3 loop round.
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
  without them.

### Not built yet

- In the loop (plan 1.5): two-stage scoring (the learned families on the
  best four candidates and the incumbent only) and inertness from each
  tape's material.
- The stage cache (`AI_RESTORE_STAGE_CACHE`), R3's harness coverage
  reading, `scripts/replay_post_neural.py`, the fidelity ladder,
  `known_ordering_v3.json`, and a gate on `file.gain_ride_lu`.
- Before any v3 tuning round (Round 0 is done): Session 0 with the user (ABX
  of +1 against +2 dB air, ABX of alpha 2 on 0.7.3 against 0.7.5 on pauses,
  and the pause-depth continuum against the shipped pause floor); the
  calibration with a Vaccin excerpt (the R1 air assertion); and learned
  veto floors measured on inaudible pairs, since the noise-floor report
  the v3 grids' vetoes read gives limits too tight (see "Round 0").

### The user's decisions (2026-10-08)

- Learned training only where it fits: GRPO for a per-tape settings policy,
  later, and only if a variation test beats the single best preset against
  an Optuna BO baseline; an anchored fine-tune of the Mel-RoFormer in the
  isolated `tools/msst/` venv; the judge is the preference head.
- No generative engines, and no audio-LLM or cloud judge.
- The user's ear is asked in short A/B batches (10-20 picks) through
  `listen_ab.py`, and no default ships without a listening session.
- Approved installs, not provisioned as of 2026-10-09: cathar 0.8.0 (into
  `experiments/cathar-0.8.0/`, behind `AI_RESTORE_CATHAR_BIN`, identity check
  first), silero-vad, zimtohrli, optuna (dev group) and the `tools/msst/`
  venv. MERT's CC-BY-NC-4.0 weights stay (research use).

## Tuning on real tapes

```text
scripts/tune_restoration.py prepare|run|score|report|listen|all
    --name N --tapes-dir DIR --grid scripts/tune_grids/<grid>.yaml
```

The driver cuts excerpts, runs
every grid variant in a fresh interpreter with its own `config.yaml`
(overrides validated against the app's typed settings and read back from a
child before a run), scores, and writes `experiments/tune_N/scoreboard.md`.
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
tapes (`experiments/autotune/tapes_apl.json`).

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
  provisioned 0.7.6 it rendered one binary twice. A binary knob returns only
  for a new build, `[None, experiments/cathar-0.8.0/cathar.exe]` in round
  C0.
- `apl_neural_model` was `[None, ROFORMER, ROFORMER_AGGR]`: once the
  Mel-RoFormer became the app's default, None and ROFORMER named one
  setting, so a run from the defaults rendered the default every round
  (hashed inert) and never proposed the aggressive model. It is
  `[ROFORMER, ROFORMER_AGGR]`, seeded from the app; a saved state holding
  None for a knob whose list has no None is seeded too. A list must never
  hold None and the app's default value together.

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
  no engine tunes stops the loop.
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
- `vetoes:` (learned median veto): on any tape, a listed median moved the
  `worse` way from the incumbent's by more than `floor_multiple` times the
  reading's floor from `--noise-floors` (a `reward_noise_floor` report of the
  shipped outputs; default `experiments/reward/noise_floor.json`), or a reading
  the incumbent has went unread: the candidate is logged `vetoed` and cannot win
  (`reward.gate_constraints(learned_veto=...)` must read "ok"). A veto without a
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

Each refused, tied or vetoed candidate gets a line in `log.md` naming the
boundary, or the tape, the reading and its move. Tests:
`tests/unit/test_autotune_guards.py` (the loop) and
`test_autotune_guard_rules.py` (`reward.py`'s vetoes and boundaries).

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
