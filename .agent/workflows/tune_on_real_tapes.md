# Workflow: Tune an Engine on Real Tapes

Use this workflow when the user asks for cleaner output on their own tapes,
reports what they hear ("under water", hiss, dead pauses, harsh "s"), or asks
to fine-tune cathar or `auto_pure_linear`. The output-quality harness is the
judge; the user's ear decides only at the end.

## Step 1: Measure the complaint

1. Extract the source and the output to WAV and score them:
   `scripts/validate_restoration.py SOURCE out=OUTPUT --metrics all`.
1. If no metric moves the way the user described, measure the specific
   thing on the specific file (band levels on the affected frames, pause
   floor per band, sibilant frames) until a number reproduces the
   complaint, then add that number to `scripts/restoration_quality` with a
   synthetic test and a docs sentence, and re-check the DSP calibration into
   a separate `--out`.
1. Record the verdict as flags: a `listener.*` gate at severity `flag`, a
   degradation in `scripts/quality_degradations.py`, and the flagged /
   clean labels per gate in `assets/quality_calibration/known_ordering_v2.json`.
   Re-score the stored listening reports with
   `score_listen.py <slug> --dsp-rescore` and run
   `experiments/tata_listen/check_verdicts.py`: every
   verdict must be reproduced and nothing the user accepted flagged.

## Step 2: Let the loop refine

1. List the tapes in `experiments/autotune/tapes.json` (whole tapes under
   20 min; cut longer ones, for example the first 300 s, with
   `ffmpeg -t 300 -c copy` into `D:\Tata\New folder\variants\`).
1. Run `scripts/autotune_restoration.py --engine cathar --tapes ...` and the
   same for `apl` on hardlinked copies of the sources
   (`tapes_apl.json`), detached (`experiments/run_autotune*.cmd`), and watch
   `experiments/autotune/<engine>/log.md`.
1. Do not ask the user to listen while a round can still improve.

## Step 3: Confirm and hand over

1. At the plateau, run the winners on the full tapes into
   `D:\Tata\New folder\variants\<tape>__<variant>` and send the scoreboard.
1. Ask for the verdict on the specific windows the harness names.
1. Feed the accepted settings back: `config.yaml` and `modules/config.py`
   with the measured-effect comment, `docs/configuration.md`, and re-base the
   cathar identity reference (`experiments/cathar_ab.py <tag>`, then copy the
   run into `experiments/cathar_ab_head` and rewrite `cathar_ab_head.json`,
   keeping the previous reference as `cathar_ab_head_before_<tag>`).
1. Run the full local quality gate before pushing.
