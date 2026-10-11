"""Re-judge the stored rounds of finished self-driving loops offline, from their score files (CPU, no render, no scorer).

    python -m scripts.rejudge_autotune LOOP_DIR [LOOP_DIR ...] --grid scripts/tune_grids/tata_v3.yaml
        --noise-floors experiments/reward/noise_floor_benign.json --jitter-floors experiments/reward/noise_floor.json
        [--null-draws 200] [--seed 0] [--json report.json]

A loop directory is `<out>/<engine>` of `scripts/autotune_restoration.py`: its `state.json`
holds each round's verdicts, and `cands/<cid>/<slug>.score.json` (plus
`<slug>.learned.score.json` from the two-stage scoring) each scored candidate's flat readings.
Every round is judged four ways on the candidates it scored, and the accepted candidate, the
qualifiers, the vetoes and the label are reported for each:

- `old`: exact ties and the any-tape veto, the loop before 2026-10-11 (it must reproduce the
  stored decision; the combination candidate a round rendered is judged with the others);
- `band`: the tie band and the dead zones' floor (`autotune_judging.Floors`), the any-tape veto;
- `familywise`: exact ties, the family-wise veto (`autotune_judging.familywise_vetoes`);
- `new`: both, the loop as it now judges.

The vetoes are checked on every scored candidate, not only on the contenders: a veto only takes
a qualification away, so the accepted candidate is the same, and the counts show what each rule
would refuse. A later round was rendered around the incumbent the stored run accepted, so a
round whose decision changes says what that round would have done, not what the run would have
become. Nothing here decides a default: any accepted move still needs a blind block.

`--null-draws N` adds the vetoes' null simulation, per round with learned readings: N null
candidates per null, each the incumbent's flat readings with every veto reading moved per tape
by noise of the scorer's jitter (`jitter`, normal with its p95 at the jitter floor) or of the
Session 0 benign pairs (`benign`, the size of differences a blind listener did not hear), and
every scored candidate with each tape's paired deltas given a random sign (`signflip`, all of a
tape's readings together). The rule must veto 5% of the null candidates or fewer.
"""

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

at = import_module("scripts.autotune_restoration")
judging = import_module("scripts.autotune_judging")
guards_mod = import_module("scripts.autotune_guards")
cli_paths = import_module("scripts.cli_paths")

STATE = "state.json"
NULL_LIMIT = 0.05
P95_Z = 1.959963984540054
NULLS = ("jitter", "benign", "signflip")


def _any_tape(tape_scores, incumbent_scores, vetoes, _floors):
    return guards_mod.learned_vetoes(tape_scores, incumbent_scores, vetoes)


def _familywise(tape_scores, incumbent_scores, vetoes, floors):
    return judging.familywise_vetoes(tape_scores, incumbent_scores, vetoes, floors)


# name -> (ranking floors?, veto rule)
RULES = {"old": (False, _any_tape), "band": (True, _any_tape), "familywise": (False, _familywise), "new": (True, _familywise)}


# ----------------------------------------------------------------------------- loading a stored round


def _flat(cand_dir, slug):
    """A candidate's flat readings on one tape: the first pass merged with the learned stage's, or None when unscored."""
    first = guards_mod.read_json(cand_dir / f"{slug}.score.json")
    if not first:
        return None
    return {**first, **guards_mod.read_json(cand_dir / f"{slug}.learned.score.json")}


def round_scores(loop_dir, record):
    """`({cid: {slug: flat}}, missing)` for the candidates the round scored; a candidate lacking a tape's file is missing."""
    verdicts = record.get("verdicts", {})
    tapes = list(verdicts.get(record["incumbent"], {}).get("per_tape", {}))
    scores, missing = {}, []
    for cid, verdict in verdicts.items():
        if not verdict.get("per_tape"):
            continue
        flats = {slug: _flat(Path(loop_dir) / "cands" / cid, slug) for slug in tapes}
        if None in flats.values():
            missing.append(cid)
        else:
            scores[cid] = flats
    return scores, missing


# ----------------------------------------------------------------------------- judging one round four ways


def judge_with(scores, incumbent_id, context, rule):
    """The round's verdicts under one rule (`RULES`): ranking floors or none, then the veto rule on every candidate.

    `context` holds the grid's `ranking`, its `guards` and the `floors`."""
    veto_rule, floors = RULES[rule][1], context["floors"]
    verdicts = at.judge(scores, incumbent_id, context["ranking"], floors=_ranking_floors(floors, rule))
    for cid, verdict in verdicts.items():
        vetoed = veto_rule(scores[cid], scores[incumbent_id], context["guards"].vetoes, floors)
        if vetoed:
            verdict["vetoed"] = vetoed
    at.qualify(verdicts, incumbent_id, len(scores[incumbent_id]))
    return verdicts


def _accepted(verdicts):
    """The best-ranked qualifier, as the loop's `_settle` picks it."""
    ordered = sorted(verdicts.items(), key=lambda item: (item[1]["mean_rank"] is None, item[1]["mean_rank"] or 0.0))
    return next((cid for cid, verdict in ordered if verdict["qualifies"]), None)


def _tied_with_incumbent(verdicts, incumbent_id):
    """The candidates whose mean rank equals the incumbent's."""
    base = verdicts[incumbent_id]["mean_rank"]
    return sorted(cid for cid, verdict in verdicts.items() if cid != incumbent_id and verdict["mean_rank"] == base)


def _qualifiers(verdicts):
    return sorted(cid for cid, verdict in verdicts.items() if verdict["qualifies"])


def _vetoes(verdicts):
    return {cid: verdict["vetoed"] for cid, verdict in verdicts.items() if verdict.get("vetoed")}


def _level(verdicts, incumbent_id):
    """The candidates level with the incumbent on every tape and reading."""
    return sorted(cid for cid, verdict in verdicts.items() if cid != incumbent_id and verdict["level"])


def summarise(verdicts, incumbent_id, floors):
    """What one rule decided: accepted, qualifiers, vetoes, ties with the incumbent, level candidates, the label, wins."""
    accepted = _accepted(verdicts)
    return {
        "accepted": accepted,
        "qualifiers": _qualifiers(verdicts),
        "vetoed": _vetoes(verdicts),
        "tied_with_incumbent": _tied_with_incumbent(verdicts, incumbent_id),
        "level": _level(verdicts, incumbent_id),
        "label": f"accepted {accepted}" if accepted else judging.plateau_label(verdicts, incumbent_id, floors),
        "wins": {cid: verdict["wins"] for cid, verdict in verdicts.items()},
    }


def _ranking_floors(floors, rule):
    """The floors a rule ranks with: the given ones, or none for the exact-tie rules."""
    return floors if RULES[rule][0] else judging.NO_FLOORS


def rejudge_round(loop_dir, record, context):
    """`{round, stored, n_scored, missing, rules: {rule: summary}}` for one stored round."""
    scores, missing = round_scores(loop_dir, record)
    incumbent_id = record["incumbent"]
    report = {"round": record["round"], "stored": record.get("accepted"), "n_scored": len(scores), "missing": missing}
    if incumbent_id not in scores:
        return {**report, "rules": {}}
    report["rules"] = {
        rule: summarise(
            judge_with(scores, incumbent_id, context, rule),
            incumbent_id,
            _ranking_floors(context["floors"], rule),
        )
        for rule in RULES
    }
    report["null"] = null_rates(scores, incumbent_id, context) if context.get("draws") else {}
    return report


# ----------------------------------------------------------------------------- the vetoes' null simulation


def _veto_readings(scores, incumbent_id, vetoes):
    """The veto readings the incumbent read on some tape: only those can veto."""
    return [reading for reading in vetoes if any(reading in flat for flat in scores[incumbent_id].values())]


def _noisy(flats, readings, scale, rng):
    """`flats` with each reading moved per tape by normal noise whose p95 magnitude is `scale[reading]`."""
    moved = {}
    for slug, flat in flats.items():
        noise = {reading: rng.normal(0.0, scale.get(reading, 0.0) / P95_Z) for reading in readings if reading in flat}
        moved[slug] = {**flat, **{reading: float(flat[reading]) + value for reading, value in noise.items()}}
    return moved


def _flipped(candidate, incumbent, readings, rng):
    """The candidate's paired deltas per tape, every reading of a tape given one random sign."""
    flipped = {}
    for slug, flat in candidate.items():
        sign = rng.choice((-1.0, 1.0))
        base = incumbent[slug]
        flipped[slug] = {
            **flat,
            **{r: float(base[r]) + sign * (float(flat[r]) - float(base[r])) for r in readings if _paired(flat, base, r)},
        }
    return flipped


def _paired(flat, base, reading):
    return judging.finite(flat.get(reading)) and judging.finite(base.get(reading))


def _null_candidates(scores, incumbent_id, readings, context, rng):
    """`{null: [candidate flats, ...]}` for the three nulls."""
    incumbent, draws, floors = scores[incumbent_id], context["draws"], context["floors"]
    scales = {"jitter": floors.jitter, "benign": judging.veto_floors(floors)}
    nulls = {name: [_noisy(incumbent, readings, scale, rng) for _ in range(draws)] for name, scale in scales.items()}
    nulls["signflip"] = _signflips(scores, incumbent_id, readings, draws, rng)
    return nulls


def _signflips(scores, incumbent_id, readings, draws, rng):
    """About `draws` sign-flipped copies of the scored candidates, the same number of each."""
    others = [flats for cid, flats in scores.items() if cid != incumbent_id]
    each = max(1, draws // max(1, len(others)))
    return [_flipped(candidate, scores[incumbent_id], readings, rng) for candidate in others for _ in range(each)]


def _rate(candidates, incumbent, vetoes, floors, rule):
    if not candidates:
        return None
    return sum(1 for flats in candidates if rule(flats, incumbent, vetoes, floors)) / len(candidates)


def null_rates(scores, incumbent_id, context):
    """`{null: {"any_tape": share vetoed, "familywise": share vetoed, "n": candidates}}`; {} without veto readings."""
    readings = _veto_readings(scores, incumbent_id, context["guards"].vetoes)
    if not readings:
        return {}
    vetoes = {reading: context["guards"].vetoes[reading] for reading in readings}
    rng = np.random.default_rng(context.get("seed", 0))
    nulls = _null_candidates(scores, incumbent_id, readings, context, rng)
    incumbent, floors = scores[incumbent_id], context["floors"]
    return {
        name: {
            "any_tape": _rate(candidates, incumbent, vetoes, floors, _any_tape),
            "familywise": _rate(candidates, incumbent, vetoes, floors, _familywise),
            "n": len(candidates),
        }
        for name, candidates in nulls.items()
    }


# ----------------------------------------------------------------------------- the loops and the command line


def rejudge_loop(loop_dir, context):
    """`{loop, rounds: [...]}` for every stored round of one loop directory."""
    state = json.loads((Path(loop_dir) / STATE).read_text(encoding="utf-8"))
    return {"loop": str(loop_dir), "rounds": [rejudge_round(loop_dir, record, context) for record in state.get("rounds", [])]}


def _changed(rules):
    """The rules whose accepted candidate differs from the old rule's."""
    old = rules.get("old", {}).get("accepted")
    return [rule for rule, summary in rules.items() if summary["accepted"] != old]


def _veto_count(rules, rule):
    return len(rules.get(rule, {}).get("vetoed", {}))


def round_line(loop, report):
    """One line per round: the decision per rule, the vetoes, the ties and the label."""
    rules = report["rules"]
    if not rules:
        return f"{loop} round {report['round']}: incumbent not scored, skipped"
    vetoes = "/".join(str(_veto_count(rules, rule)) for rule in RULES)
    ties = f"{len(rules['old']['tied_with_incumbent'])} -> {len(rules['new']['tied_with_incumbent'])}"
    return (
        f"{loop} round {report['round']} (stored {_name(report['stored'])}, {report['n_scored']} scored): {_decisions(rules)}; "
        f"vetoes old/band/familywise/new {vetoes}; tied with the incumbent {ties}; changed under {_changed(rules) or 'none'}; "
        f"new: {rules['new']['label']}"
    )


def _name(cid):
    return cid or "-"


def _decisions(rules):
    """`old <cid>, band <cid>, ...`: what each rule accepted."""
    return ", ".join(f"{rule} {_name(rules[rule]['accepted'])}" for rule in RULES)


def null_line(loop, report):
    """One line per round with a null simulation: the share of null candidates each rule vetoes."""
    cells = [
        f"{name} any-tape {rates['any_tape']:.1%} family-wise {rates['familywise']:.1%} (n {rates['n']})"
        for name, rates in report["null"].items()
        if rates["n"]
    ]
    return f"{loop} round {report['round']} null: " + "; ".join(cells)


def _context(args):
    benign = guards_mod.noise_floors(cli_paths.confined_path(args.noise_floors, "--noise-floors", must_exist=True))
    jitter_path = cli_paths.confined_path(args.jitter_floors, "--jitter-floors", must_exist=True)
    floors = judging.Floors(benign, judging.jitter_floors(guards_mod.read_json(jitter_path), jitter_path))
    grid = at.tune.load_grid(cli_paths.confined_path(args.grid, "--grid", must_exist=True))
    guards = guards_mod.loop_guards(grid, judging.veto_floors(floors), at.ALL_KNOBS)
    return {"ranking": grid["ranking"], "guards": guards, "floors": floors, "draws": args.null_draws, "seed": args.seed}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("loops", nargs="+", type=cli_paths.existing_path_arg, help="loop directories holding state.json and cands/")
    parser.add_argument("--grid", type=cli_paths.existing_path_arg, required=True, help="the grid the loops ran with")
    parser.add_argument("--noise-floors", type=cli_paths.existing_path_arg, required=True, help="the Session 0 benign-pair report")
    parser.add_argument("--jitter-floors", type=cli_paths.existing_path_arg, required=True, help="the repeat and near-copy report")
    parser.add_argument("--null-draws", type=int, default=0, help="null candidates per null and round (0: no null simulation)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=cli_paths.path_arg, default=None, help="also write the reports to this JSON file")
    return parser.parse_args(argv)


def main(argv=None):
    """Re-judges every round of every loop, prints one line per round (and per null) and returns the reports."""
    args = _parse_args(argv)
    context = _context(args)
    reports = []
    for loop in args.loops:
        report = rejudge_loop(cli_paths.confined_path(loop, "loop", must_exist=True), context)
        reports.append(report)
        for item in report["rounds"]:
            print(round_line(Path(loop).name, item))
            if item.get("null"):
                print(null_line(Path(loop).name, item))
    if args.json is not None:
        target = cli_paths.confined_path(args.json, "--json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(reports, indent=1, default=str) + "\n", encoding="utf-8")
    return reports


if __name__ == "__main__":
    main()
