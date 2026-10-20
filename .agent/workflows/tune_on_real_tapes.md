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
1. Run one loop at a time on a tape list: never two restorations on one
   source path (the user's rule). The loop renders each tape from its own
   path, and the app keeps its work folder beside the source
   (`.temp_work_<stem>`), so an APL loop and a cathar loop started
   together on `tapes.json` (2026-10-17) shared those folders: cathar's
   first render failed on two tapes with `WinError 32` and the loop
   stopped, and APL's renders of the same minutes could not be trusted.
   Chain the engines in one launcher instead
   (`experiments/run_autotune_v8.cmd both`), or give each loop copies of
   the tapes.
1. Stopping a loop to change its code (2026-10-17) leaves three traps:
   - The app writes each tape's output beside the tape and skips a tape
     whose output exists ("Output already exists. Skipping."), so a render
     stopped after one tape handed that output to the next candidate: three
     stale outputs of a stopped v8 run became the v9 incumbent's audio, and
     one stopped candidate's Tele7abc became the next candidate's. The loop
     now moves any `<stem><suffix>` beside a tape into
     `<out>/stale_outputs/` before it renders (and says so); still kill the
     launcher's `cmd` first (else `both` starts the next engine), then the
     Python tree, and remove the `.temp_work_<stem>` folder the kill left.
   - A knob added to the table is seeded into the incumbent on resume, and
     candidate ids hash the whole override set, so every id changes and the
     round renders again. With the new key off the audio is unchanged:
     copy each old candidate folder to its new id with the key added at its
     off value (scores read the output against the tape, not the
     incumbent; audibility verdicts are keyed on the files and are read
     again). A knob of the other engine changes no id.
   - The stage cache keys on every module's source, `config.py` included,
     so the first candidate of each pre-neural setting renders the neural
     stage again after any code change.
1. A render can fail, and the day is lost if nothing notices (2026-10-20:
   `apl_burst_ceiling` `gaps` made a stage name 261 characters long, the
   app could not write it on three Tata tapes, and APL's round 3 ended
   after two minutes while `both` went on to cathar; cathar then filled
   drive D: and crashed on "No space left on device"). The loop
   (`scripts/autotune_renders.py`) now sets a candidate whose render left
   no usable output aside with a `render failure:` line naming each tape's
   first `[Error]`, goes on, and ends such a round's plateau with "N
   render(s) failed, their moves untried": that plateau is not proven, so
   fix the cause and rerun before asking the user to listen. The
   incumbent failing, or three failures in a row, still stops the run, and
   so does a drive under the tapes, the loop's folder or the stage cache
   with less than 40 GB free, before the render (`MIN_FREE_GB`); a failure
   is retried on the next run. Watch the process, not a word in the log:
   a watcher that waited for "failed" or "Traceback" missed both stops.
   Free a drive by moving only folders with no junction in them: robocopy
   follows junctions (pipeline-runner skill).
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
1. Ear v4 (2026-10-12, the user's rule: be very critical of VHS hiss under
   the programme and in the pauses) is the loop's and the sweep's default:
   `--grid scripts/tune_grids/tata_v4.yaml` (speech) or `music_v4.yaml`.
   Before a v4 round, write the hiss jitter report on the shipped outputs
   (`python -m scripts.reward_noise_floor SOURCE=OUTPUT ...` with
   `--transforms hiss`, `--repeats 2`, `--families dsp`, `--max-seconds 600`
   so it reads whole tapes as the loop does (at 125 s the programme ties read
   0.24 / 0.21 where whole tapes read 0.359 / 0.413, and a tape whose
   excerpt keeps one pause window reads its pause scatter far too wide), and
   `--out experiments/reward/noise_floor_hiss.json`) and check each grid
   `tie` against it (the report's `tie_floor` for that flat key, the widest
   transform's p95, rounded up to 0.05; the pooled `floor` dilutes the fresh
   draws). The loop bands each hiss entry with that tie (or the entry's
   jitter floor where wider, never a benign pair's) and keeps its dead zone
   at 0; its DNSMOS BAK and SIGMOS NOISE vetoes are hiss readings, measured
   in `--jitter-floors` (default `experiments/reward/noise_floor.json`, which
   must hold the mos family). The v4 grids'
   `constraints:` exclude a candidate that adds audible hiss over the incumbent
   or the raw tape, empties the pauses or the room band, gates the hiss out of
   the pauses only, dulls the programme or adds musical noise; `log.md` marks it
   `guarded` with the tape, the reading and its move. The pause family plateaus
   until the engine's pause floor stops refilling the pauses with tape hiss
   (`modules/pause_floor.py`, see the output-quality-harness skill, "Critical of
   hiss").
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
       --device DEVICE --volume VOLUME
   ```

   The cuts go where the noise-to-mask ratio says the two differ most, and
   the answers land in the ledger. Record the playback device and volume in
   every block: ear v4's hiss readings assume each file's loud frames at
   75 dB SPL, and only a blind session at a recorded level can move that.
   Send the scoreboard and name the windows the harness found.

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
