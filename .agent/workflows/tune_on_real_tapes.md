# Workflow: Tune an Engine on Real Tapes

Use this workflow when the user asks for cleaner output on their own tapes,
reports what they hear ("under water", hiss, dead pauses, harsh "s"), or asks
to fine-tune cathar or `auto_pure_linear`. The output-quality harness is the
judge; the user's ear decides only at the end. The output-quality-harness
skill holds the readings, their status and the measured facts behind them.

## Step 1: Measure the complaint

1. Extract the source and the output to WAV and score them:
   `scripts/validate_restoration.py SOURCE out=OUTPUT --metrics all`.
1. If no metric moves the way the user described, measure the specific
   thing on the specific file (band levels on the affected frames, pause
   floor per band, sibilant frames) until a number reproduces the
   complaint, then add that number to `scripts/restoration_quality` with a
   synthetic test and a docs sentence, and re-check the DSP calibration into
   a separate `--out`. Read it two-sided (distance from what the user
   accepted), give a net reading its absolute twin, and write down the
   complaint and direction it stands for before checking it against the
   verdicts. A whole-spectrum change (a shelf, a tilt) cancels in every
   reading that is net of plain or loud frames: the +2 dB air shelf thinned
   the 's' while no v2 reading moved.
1. For a stage that acts only on detected events (the sibilant guard, the
   plosive tamer, the pause floor), check its coverage first: run the
   restoration with `AI_RESTORE_EVENT_LOG=<dir>` and compare its spans with
   the harness's own locator. At a high-band share of 0.5 the guard found 3
   events where the harness found 30; tuning its mix moved nothing audible.
1. Record the verdict in the ledger,
   `assets/quality_calibration/verdicts.jsonl`
   (`scripts/restoration_quality/ledger.py`): a `flag` record (flags and
   clean lists only from the user's words; `clean["*"]` for an output
   accepted with no complaint), a `preference` for an unblinded choice, or
   the `abx` / `pair` / `blend` records `listen_ab.py` writes. Add a
   degradation to `scripts/quality_degradations.py` that must move the new
   reading, and a `listener.*` gate at severity `flag` with `calibrated`
   false until Round 0 derives it. Re-score the stored listening reports
   with `score_listen.py <slug> --dsp-rescore` and run
   `experiments/tata_listen/check_verdicts.py`: exit 0 means every checked
   verdict holds and nothing the user accepted is flagged, 1 means one
   broke, 2 means no report scored a judged label (a false green otherwise).
   A flag on a display-only gate (`gates.DISPLAY_ONLY` or a soft gate;
   `listener.hiss` today) is listed as not asserted and checked neither flagged
   nor clean, so a stored report that still carries it passes; a flag record
   that judges labels on display-only gates alone is counted on its own ("on
   display-only gates alone"). With the default `scores/` the run on 2026-10-09
   checks 3 records and leaves 26 waiting. Round 0 scored every judged file
   (`experiments/tata_listen/score_round0.py`, one folder per round under
   `scores_v3/`): pass `--scores experiments/tata_listen/scores_v3/<set>`.
   r1_known, r1_listen and r3 reproduce every checked verdict; r2 fails 9 on the
   v2 flags (see `docs/ear_v3_round0.md`).

## Step 2: Let the loop refine

1. List the tapes in `experiments/autotune/tapes.json` (whole tapes under
   20 min; cut longer ones, for example the first 300 s, with
   `ffmpeg -t 300 -c copy` into `D:\Tata\New folder\variants\`).
1. Check the knob table against the code paths before a round: a knob is
   dead only when every path that reads it is (`apl_tonal_flatness_max`
   stays live with the subtraction off), and no list may hold both None
   and the app's own default value (`apl_neural_model` did, and never
   proposed the aggressive model). `INERT_WHEN` and `DURATION_BOUND`
   leave dead moves out; the loop hashes every candidate's audio per tape
   and logs a byte-identical one as `inert (= <id>)` with a "knob-table
   finding" line in `log.md`, never scored. Read those lines: each is a
   table entry to fix.
1. Run `scripts/autotune_restoration.py --engine cathar --tapes ...` and the
   same for `apl` on hardlinked copies of the sources
   (`tapes_apl.json`), detached (`experiments/run_autotune*.cmd`), and watch
   `experiments/autotune/<engine>/log.md`. Launch the `apl` loop with
   `--stage-cache <out>/stage_cache` (an absolute or repo-relative folder;
   the loop resolves it): a candidate that moves only post-neural knobs then
   replays the incumbent's chain and model output instead of rendering them
   (`AI_RESTORE_STAGE_CACHE`, `docs/configuration.md`, "Stage Cache"), and
   each candidate's `timing.json` counts its hits and misses. cathar ignores
   the flag. Ear v3 rounds take
   `--grid scripts/tune_grids/tata_v3.yaml` (speech) or `music_v3.yaml`
   only after Round 0 and Session 0 have set their targets; until then
   their values are uncalibrated starting points. A v3 round also takes
   `--noise-floors`, a `python -m scripts.reward_noise_floor` report of the
   Session 0 benign pairs (default `experiments/reward/noise_floor_benign.json`,
   written with `--benign-pair ... --families speech,mos,stems --repeats 2`):
   the grid's learned vetoes are measured in those floors, one floor each, and
   the loop refuses to start when a veto has no floor or a floor of 0, and
   says which. A missing floor (`reward.NO_FLOOR`)
   names `--families`: the report reads only the families it was run with, dsp
   alone by default. A floor of 0 (`reward.NO_BENIGN_FLOOR`) asks for
   `--repeats 2` or more, or `resample_roundtrip` in `--transforms`: CER read
   once can show no jitter, and a limit of 0 would veto Whisper's own.
   Near-copy floors are so tight that 3x them vetoes moves on jitter, which is
   why the default is the benign-pair report (measured 2026-10-09). The stored
   tie verdicts (`cands/*/*.audibility.json`) are keyed on
   `auditory.VERDICT_RULE` as well as the files and the offset, so a verdict an
   older rule wrote is read again with no delete; bump `VERDICT_RULE` with any
   change to what `compare_files` calls audible. A v3 round takes
   `--gates experiments/quality_calibration_v3/gates.json` once the v3
   calibration has written it: the default `--gates` is still the v2 file (its
   `listener.hiss` severity is ignored, `gates.DISPLAY_ONLY`, but its other
   thresholds came from a dsp-only run). The v3 grids' guards then run in the
   loop (`scripts/autotune_guards.py`): a move past a ledger boundary the user
   rejected (`reversals:`) is refused before rendering, a candidate inaudible
   from the incumbent on every tape (`audibility:`) is a tie and never scored,
   and one whose learned median moved past its veto (`vetoes:`) cannot win.
   `log.md` names each with its reason; a refusal or a tie on every candidate is
   a plateau, not a fault.
1. Do not ask the user to listen while a round can still improve.

## Step 3: Confirm and hand over

1. At the plateau, run the winners on the full tapes into
   `D:\Tata\New folder\variants\<tape>__<variant>`.

1. Before asking, run `scripts/audibility_check.py` on each final against
   what the user last heard on the same tape. Inaudible on every tape means
   there is nothing to ask: the plateau ties the incumbent. Set
   `AI_RESTORE_DATA_ROOTS=D:\Tata` for paths on that drive.

1. Ask through the localhost-only listening tool, not by sending files
   around, ABX first (24 trials; 17 right is p 0.032), then `pair` or
   `blend` when the difference is heard:

   ```text
   python -m scripts.listen_ab abx --tape SLUG --stim A=PATH --stim B=PATH
   ```

   The cuts go where the noise-to-mask ratio says the two differ most, and
   the answers land in the ledger. Send the scoreboard and name the windows
   the harness found.

1. Feed the accepted settings back: `config.yaml` and `modules/config.py`
   with the measured-effect comment, `docs/configuration.md`, and re-base the
   cathar identity reference (`experiments/cathar_ab.py <tag>` with a tag
   never used before, since an existing run directory is resumed and proves
   nothing; then copy the run into `experiments/cathar_ab_head` and rewrite
   `cathar_ab_head.json`, keeping the previous reference as
   `cathar_ab_head_before_<tag>`). Where the speech and music plateaus
   disagree on a shared key, add a music-profile key; a knob the final
   carries but an accepted switch made inert stays at its shipped value.
   No default ships without a listening session and one
   `artifacts/realistic-v2` plus IA-corpus confirmation (AGENTS.md
   section 3).

1. Run the full local quality gate before pushing.
