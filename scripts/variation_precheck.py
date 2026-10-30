#!/usr/bin/env python3
"""Could a per-tape settings policy beat the single best candidate audibly? (plan 3.2a, pre-check)

    variation_precheck.py STATE.json [STATE.json ...] [--round N] [--heard-db=-17.6] [--json out.json]

Reads what the self-driving loop already stored, renders nothing. Each `state.json` holds every
round's per-tape ranks; each candidate's renders sit beside it in `cands/<hash>/<tape>.wav`.
For the chosen round (default: the last one with two scored candidates, the plateau round):

- the single best candidate is the lowest mean rank among the scored ones;
- the oracle picks, per tape, the candidate with the lowest per-tape rank: the most any
  per-tape selector over these candidates could choose;
- every pick that is not the best goes through `auditory.compare_files` against the best's
  render of the same tape.

Each pick is then one of:

- `selectable`: the check calls it audible and its difference stands above `--heard-db`, the
  largest restoration difference the listener has failed to hear blind (Session 0: the 0.7.3
  against 0.7.5 control on Tele7abc, -17.6 dB, 8/16). These go to an ABX before anything else.
- `below_heard`: called audible, but smaller than a difference already not heard. The check's
  "audible" did not predict this listener (Session 0), so these are taken as unheard.
- `tie`: the check cannot tell it from the best. A tie is trusted, so its rank gain is what
  choosing among identical-sounding renders gains by chance: the per-tape ranking noise.
- `missing`: a render is gone (pruned or never written).

The round passes the pre-check when some pick is selectable; otherwise the round offers a
per-tape policy nothing this listener hears, and plan 3.2b (GRPO) stays parked. Measured on
2026-10-11 over the music rounds of `autotune_v6_music` and `autotune_v7_music` (29 picks): 5
trusted ties gained 3.3-7.3 ranks, as much as the oracle's mean gain (3.1-4.4), and the only
selectable picks were `apl_music_stem_path` on two clips (-7.8 and -14.4 dB). On the speech
tapes the shipped settings ranked first on every tape (the v3 targets were read on them).
When the listener's blend threshold is measured, pass it as `--heard-db`.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.cli_paths import confined_path, existing_path_arg, path_arg
    from scripts.restoration_quality import auditory
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.cli_paths import confined_path, existing_path_arg, path_arg
    from scripts.restoration_quality import auditory

# Session 0's ABX control (0.7.3 against 0.7.5 alpha 2, Tele7abc 105 s): -17.6 dB, heard 8/16.
DEFAULT_HEARD_DB = -17.6


def scored(round_record):
    """The candidates of one round that carry per-tape ranks, `{hash: verdict}`."""
    return {key: verdict for key, verdict in round_record.get("verdicts", {}).items() if verdict.get("per_tape")}


def tape_ranks(verdicts, tape):
    """`{hash: rank}` on one tape, over the candidates scored there."""
    return {key: verdict["per_tape"][tape] for key, verdict in verdicts.items() if tape in verdict["per_tape"]}


def oracle(round_record):
    """The round's single best candidate and the per-tape oracle's picks, or None under two scored candidates."""
    verdicts = scored(round_record)
    if len(verdicts) < 2:
        return None
    best = min(verdicts, key=lambda key: verdicts[key]["mean_rank"])
    picks, gains = {}, []
    for tape, best_rank in sorted(verdicts[best]["per_tape"].items()):
        ranks = tape_ranks(verdicts, tape)
        pick = min(ranks, key=ranks.get)
        gains.append(best_rank - ranks[pick])
        if pick != best:
            picks[tape] = {"pick": pick, "rank_gain": round(best_rank - ranks[pick], 2)}
    return {
        "round": round_record.get("round"),
        "best": best,
        "n_scored": len(verdicts),
        "oracle_gain": round(sum(gains) / len(gains), 2),
        "picks": picks,
    }


def scored_rounds(state):
    """The rounds with two scored candidates or more, in order."""
    return [record for record in state.get("rounds", []) if len(scored(record)) >= 2]


def choose_round(state, number=None):
    """The round `number`, or the last round with two scored candidates; None when there is none."""
    rounds = scored_rounds(state)
    if number is not None:
        rounds = [record for record in rounds if record.get("round") == number]
    return rounds[-1] if rounds else None


def classify(result, heard_db):
    """`selectable`, `below_heard` or `tie` for one compared pick."""
    if not result.get("audible"):
        return "tie"
    diff_db = result.get("diff_db")
    return "selectable" if diff_db is None or diff_db > heard_db else "below_heard"


def check_pick(cands, best, tape, pick, heard_db, compare=auditory.compare_files):
    """Compares the pick's render of `tape` with the best's and classifies it; `missing` when a render is gone."""
    incumbent, candidate = cands / best / f"{tape}.wav", cands / pick["pick"] / f"{tape}.wav"
    if not (incumbent.is_file() and candidate.is_file()):
        return {**pick, "class": "missing"}
    result = compare(incumbent, candidate)
    diff_db = result.get("diff_db")
    return {
        **pick,
        "diff_db": None if diff_db is None else round(diff_db, 1),
        "audible": bool(result.get("audible")),
        "class": classify(result, heard_db),
    }


def check_state(state_path, number=None, heard_db=DEFAULT_HEARD_DB, compare=auditory.compare_files):
    """The pre-check of one loop's state: its round's oracle with every pick compared, or None without a round."""
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    report = oracle(choose_round(state, number) or {})
    if report is None:
        return None
    cands = Path(state_path).parent / "cands"
    report["picks"] = {tape: check_pick(cands, report["best"], tape, pick, heard_db, compare) for tape, pick in report["picks"].items()}
    return summarize(report)


def summarize(report):
    """Adds the selectable picks, the ties' rank gains (the ranking noise) and the round's verdict."""
    picks = report["picks"]
    report["selectable"] = sorted(tape for tape, pick in picks.items() if pick["class"] == "selectable")
    report["tie_rank_gains"] = sorted(pick["rank_gain"] for pick in picks.values() if pick["class"] == "tie")
    report["passes"] = bool(report["selectable"])
    return report


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("states", nargs="+", type=existing_path_arg, help="the loops' state.json files")
    parser.add_argument("--round", type=int, default=None, help="the round to read (default: the last scored round)")
    parser.add_argument("--heard-db", type=float, default=DEFAULT_HEARD_DB, help="the largest difference not heard blind")
    parser.add_argument("--json", type=path_arg, default=None, help="also write the reports to this JSON file")
    return parser.parse_args(argv)


def _line(state, report):
    """One summary line per state."""
    if report is None:
        return f"{state}: no round with two scored candidates"
    verdict = "PASSES: ABX " + ", ".join(report["selectable"]) if report["passes"] else "nothing selectable"
    return (
        f"{state}: round {report['round']}, best {report['best']} of {report['n_scored']}, oracle gain "
        f"{report['oracle_gain']}, {len(report['picks'])} picks, tie rank gains {report['tie_rank_gains']}: {verdict}"
    )


def main(argv=None):
    """Runs the pre-check on every state, prints one line each and returns `{state: report}`."""
    args = _parse_args(argv)
    reports = {}
    for state in args.states:
        path = confined_path(state, "state", must_exist=True)
        reports[str(path)] = check_state(path, args.round, args.heard_db)
        print(_line(path, reports[str(path)]))
    if args.json is not None:
        target = confined_path(args.json, "--json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(reports, indent=1) + "\n", encoding="utf-8")
    return reports


if __name__ == "__main__":
    main()
