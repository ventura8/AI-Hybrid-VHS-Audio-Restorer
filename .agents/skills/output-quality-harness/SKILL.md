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
LFS pointer). SCOREQ was dropped: it pulls plain `onnxruntime` beside
`onnxruntime-gpu`.

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
re-reads the verdicts.

Full-tape listening sets go to `D:\Tata\New folder\variants\<tape>__<variant>`
through `experiments/tata_listen/run_listen_variants*.py` (the app writes
beside its input, so outputs are moved away between variants) and are scored
with `score_listen.py <slug>`.

## The self-driving loop

```text
scripts/autotune_restoration.py --engine cathar|apl --tapes tapes.json
    --rounds 4
```

This is the user's instruction made executable: "the AI human ear must
refine itself a few rounds and ask me to listen only when it cannot refine
any more". Each round proposes the neighbours of every knob's current value
(plus the combination of the moves that helped on their own), restores every
tape per candidate keeping only the audio, scores all tapes of the round in
parallel, and accepts a candidate only if its mean rank across tapes beats
the incumbent, it wins on at least half the tapes and it fails no more hard
gates. It stops at a plateau; `log.md` and `state.json` under
`experiments/autotune/<engine>/` make it resumable. Run both engines at once
only on different source paths: the app's work directory is
`.temp_work_<stem>` beside the source, so APL runs on NTFS hardlinks of the
tapes (`experiments/autotune/tapes_apl.json`).

Knobs must be the live ones: on the Tata tapes the scanner reads a spectral
flatness of 0.022, under `apl_tonal_flatness_max` 0.035, so APL takes its
tonal path (`apl_spectral_alpha_tonal`, `apl_noiseprint_tonal_s`, plosive
tamer skipped) and `apl_spectral_alpha` candidates came out bit-identical to
the incumbent. When several candidates score exactly alike, hash their audio
before spending another round. A scorer holds about 4 GB of RAM; `--parallel`
(default 2) caps the tape scorers per engine, and two engines plus the corpus
scorers exhausted a 62 GB machine.

Only when both engines have plateaued: produce the listening set, send the
scoreboard, ask the user. Feed a winner back into `config.yaml` and
`modules/config.py` with the measured-effect comment, then re-base the cathar
identity reference (see the audio-restoration-engine skill).
