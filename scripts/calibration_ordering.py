"""The known-ordering set: the user's tapes with outputs judged by ear, scored, ranked by the v3 score, checked.

`scripts/calibrate_quality_metrics.py --known-ordering MANIFEST` scores every tape of the
manifest once (cached under `<out>/ordering/<tape>.json`) and applies the rules here.

Why the ranking changed (ear v3 design 4.4, item 2). The v2 ranking ordered the variants by
hard failures, then by a composite: the mean z-score of three learned judges (UTMOS, SIGMOS
overall, Audiobox PQ) against the benign floor. The v2 calibration ran with `--metrics dsp`
(`experiments/quality_calibration_v2/ordering/*.json` carry `families: {"dsp": "ok"}` and
none of the three readings), and a missing judge read as 0: the composite was 0.0 for every
variant, the ranking fell back to the order the variants were listed in (alphabetical on the
13-variant Tele7abc set) and `good_in_top` / `bug_in_bottom_two` tested that order. Learned
judges are vetoes now, never in a score sum (plan principle 7), so the composite is gone:

- the v3 score is the grid's two-sided distance, `sum(weight * max(0, |x - target| -
  dead_zone) / scale)` over its readings (`reward.distances`), lower is better; a reading one
  variant lacks while another on the tape has it counts as the tape's worst plus one scale
  unit (`reward.fill_missing`), so destroying what a reading reads never helps;
- `--grid FILE` names the grid (a YAML or JSON mapping `{metric.side.stat: {target, family,
  dead_zone?, scale?, weight?}}`, under a `reward:` key, under `ranking:` as the v3 tuning
  grids keep it (`tata_v3.yaml` 15 entries, `music_v3.yaml` 10), or at the top level). A file
  with no such entry is refused: the v3 grids used to load empty and rank, without a word, on
  the default grid. Without `--grid`, `default_grid` takes every two-sided dsp reading
  of `scorecard.METRICS` at its neutral target (no change from the source), with a dead zone
  of three benign floors: a stand-in until Round 0 sets the accepted targets. Pause
  attenuation is left out (its target is the listener's preferred attenuation, which
  Session 0 measures; at 0 it would rank the least-denoised output first) and so is
  `balance_top_db` (presence or air again);
- a tape whose variants read none of the grid's readings is unranked: `ranked_on` is 0 and
  the rules that need a ranking read None instead of testing listing order.

Listener verdicts come from the ledger (`scripts/restoration_quality/ledger.py`,
`assets/quality_calibration/verdicts.jsonl`) when the module and records exist: per manifest
tape, the ledger tape it belongs to (the entry's `ledger_tape`, the tape's own name, or the
longest ledger tape the name starts with followed by `_`, so `tele7abc_listen` reads
`tele7abc`), the flag records whose stimuli include a label the manifest scores (round 2's
flags on `final2_*` say nothing about the round-1 set on the same tape), merged by
`ledger.flag_verdicts` and narrowed to those labels. A tape no such record covers keeps the
manifest's own lists, and so does every tape under `--no-ledger`, whatever `ledger_tape` it
names. Per flag the tape also records the rounds of the records that name it on a scored
label (`gate_rounds`): a gate counts agreeing rounds on the rounds of its own verdicts
(`calibration_gates.with_rounds`), not on every round the tape was ever judged in. By-ear
ranks stay the manifest's. Flags the ledger names that no gate
reads are reported (`unmapped_flags`), and a listed (gate, label) whose verdict read nothing
is counted by `flags_unread` instead of failing `flags_reproduced`. Read on the stored v2
results (2026-10-09), the ledger puts round 1 on all four manifest tapes; Tele7abc's
"light hiss" on the single 4 s probe passes the v2 hiss flag there, so `flags_reproduced`
reads False on that tape, the hiss flag that never moved.
"""

import json
from importlib import import_module
from pathlib import Path

import numpy as np
import yaml

from scripts.calibration_gates import ANY_FLAG, EFFECT_MULTIPLE, LISTENER_PREFIX
from scripts.restoration_quality import gates as gates_mod
from scripts.restoration_quality import reward, runner
from scripts.restoration_quality.scorecard import METRICS, TWO_SIDED, WindowRow, aggregate

KNOWN_BAD = ("deesser_bug", "single4s")
KNOWN_GOOD = ("cathar075", "stitched", "apl")
MUFFLED_GATES = ("dsp.hf_4k8k", "dsp.hf_8k16k", "mos.sigmos_col", "listener.dull")
ALTERED_GATES = ("speech.speaker", "speech.cer_median", "speech.cer_tail")
# A lower reading of any of these on the single 4 s probe than on the stitched print is "duller".
DULL_READINGS = ("dsp.hf_4k8k", "mos.sigmos_col", "dsp.balance_top_db", "dsp.sib_abs_level_db")
RANKING_EXCLUDED = ("dsp.gap_atten_db", "dsp.balance_top_db")
FAMILY_BY_PREFIX = {"dsp.balance_": "timbre", "dsp.sib_": "sibilance", "dsp.gap_": "pauses"}
RANKED_STAT = "delta.median"
GRID_SECTIONS = ("reward", "ranking")
FLAT_SIDES = ("output", "delta")
FLAT_STATS = ("median", "tail")


def _ledger():
    """The ledger module, or None when it is not there."""
    try:
        return import_module("scripts.restoration_quality.ledger")
    except ImportError:
        return None


def ledger_records(path=None):
    """The ledger's records, or None when the ledger module is not there (the manifest's lists then stand)."""
    ledger = _ledger()
    if ledger is None:
        return None
    return ledger.read(path) if path else ledger.read()


def _longest_prefix(name, tapes):
    """The longest tape in `tapes` that `name` starts with, followed by `_`; None when there is none."""
    prefixes = [tape for tape in tapes if name.startswith(f"{tape}_")]
    return max(prefixes, key=len) if prefixes else None


def ledger_tape(name, entry, tapes):
    """The ledger tape a manifest entry belongs to, or None."""
    if entry.get("ledger_tape"):
        return entry["ledger_tape"]
    return name if name in tapes else _longest_prefix(name, tapes)


def _narrowed(lists, labels):
    """`{flag: sorted labels}` keeping only the labels the manifest scores."""
    return {flag: sorted(set(names) & labels) for flag, names in lists.items()}


def _about(record, labels):
    """Whether a ledger record is a flag verdict on any of `labels`."""
    return record["question"]["type"] == "flag" and bool(set(record["question"]["stimuli"]) & labels)


def _named(record, labels):
    """The flags (`*` included) whose flagged or clean list in one record names any of `labels`."""
    answer = record["answer"]
    return {flag for lists in (answer["flagged"], answer["clean"]) for flag, names in lists.items() if set(names) & labels}


def gate_rounds(records, labels):
    """`{flag: sorted rounds}`: per flag, the rounds of the records whose lists name it on any of `labels`."""
    rounds = {}
    for record in records:
        for flag in _named(record, labels):
            rounds.setdefault(flag, set()).add(str(record["round"]))
    return {flag: sorted(found) for flag, found in rounds.items()}


def _ledger_flags(records, tape, labels):
    """`(flagged, clean, rounds, gate rounds)` merged over the tape's flag records that judged any of `labels`."""
    ledger = _ledger()
    chosen = [record for record in ledger.for_tape(records, tape) if _about(record, labels)]
    flagged, clean = ledger.flag_verdicts(chosen)
    return flagged, clean, sorted({str(record["round"]) for record in chosen}), gate_rounds(chosen, labels)


def tape_verdicts(name, entry, records):
    """`{flags, clean, verdict_source, rounds, gate_rounds}` for one manifest tape: the ledger's lists when it has any, else the manifest's.

    `records` None (`--no-ledger`, or no ledger module) reads the manifest's lists whatever
    ledger tape the entry names. `gate_rounds` holds, per flag, the ledger rounds whose records
    name it on the scored labels; the manifest's lists name no round.
    """
    manifest = {"flags": entry.get("flags", {}), "clean": entry.get("clean", {}), "verdict_source": "manifest", "rounds": []}
    manifest["gate_rounds"] = {}
    if records is None:
        return manifest
    tape = ledger_tape(name, entry, {record["tape"] for record in records})
    if tape is None:
        return manifest
    labels = set(entry["variants"])
    flagged, clean, rounds, per_gate = _ledger_flags(records, tape, labels)
    if not rounds:
        return manifest
    lists = {"flags": _narrowed(flagged, labels), "clean": _narrowed(clean, labels)}
    return {**lists, "verdict_source": "ledger", "rounds": rounds, "gate_rounds": per_gate}


def _named_flags(tape):
    return set(tape.get("flags") or {}) | set(tape.get("clean") or {})


def unmapped_flags(ordering, gates):
    """Listener flags the verdicts name that no gate reads, sorted."""
    named = set().union(*(_named_flags(tape) for tape in ordering.values()))
    return sorted(named - set(gates) - {ANY_FLAG})


def _flat_side(name, side, summary):
    return {f"{name}.{side}.{stat}": summary[stat] for stat in FLAT_STATS if stat in summary}


def _flat_entry(name, entry):
    flat = {}
    for side in (side for side in FLAT_SIDES if entry.get(side)):
        flat.update(_flat_side(name, side, entry[side]))
    return flat


def flat_readings(variant):
    """`{metric.side.stat: value}` from one variant's aggregate (the shape the tuning grids key on)."""
    flat = {}
    for name, entry in variant.get("aggregate", {}).items():
        flat.update(_flat_entry(name, entry))
    return flat


def _family(metric):
    return next((family for prefix, family in FAMILY_BY_PREFIX.items() if metric.startswith(prefix)), "other")


def default_grid(floor):
    """The stand-in v3 grid: every two-sided dsp reading at its neutral target, a dead zone of three benign floors."""
    grid = {}
    for name, spec in METRICS.items():
        if spec.family == "dsp" and spec.better == TWO_SIDED and name not in RANKING_EXCLUDED:
            dead_zone = EFFECT_MULTIPLE * floor.get(name, {}).get("floor", 0.0)
            grid[f"{name}.{RANKED_STAT}"] = {"target": spec.target, "family": _family(name), "dead_zone": dead_zone}
    return grid


def _grid_section(data):
    """The mapping a grid file keeps its entries in: `reward:`, else `ranking:` (the v3 tuning grids), else the top level."""
    return next((data[key] for key in GRID_SECTIONS if isinstance(data.get(key), dict)), data)


def _grid_entries(data):
    """The entries of a grid file's section that carry a `target`."""
    return {key: value for key, value in _grid_section(data).items() if isinstance(value, dict) and "target" in value}


def load_grid(path):
    """The ranking grid of a YAML or JSON file: the entries that carry a `target`; a file with none is refused.

    An empty grid would rank on `default_grid` without a word, so the run stops instead.
    """
    grid = _grid_entries(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    if not grid:
        raise SystemExit(f"--grid {path}: no ranking entries (a reward: or ranking: mapping, or top-level entries with a target)")
    reward.parse_grid(grid)
    return grid


def harness_scores(result, floor, grid=None):
    """`{label: v3 score or None}` for one tape's variants (lower is better; None reads no grid reading at all)."""
    specs = reward.parse_grid(default_grid(floor) if grid is None else grid)
    dists = {label: reward.distances(flat_readings(variant), specs) for label, variant in result["variants"].items()}
    filled = reward.fill_missing(dists, specs)
    return {label: _summed(values) for label, values in filled.items()}


def _summed(values):
    read = [value for value in values.values() if value is not None]
    return float(np.sum(read)) if read else None


def _rank_key(variant, score):
    return len(variant.get("hard_failures", [])), score if score is not None else np.inf


def _ranking(result, scores):
    return sorted(scores, key=lambda label: _rank_key(result["variants"][label], scores[label]))


def harness_ranking(result, floor, grid=None):
    """Variants best first: fewest hard failures, then the lowest v3 score (an unscored variant last; ties keep the listed order)."""
    return _ranking(result, harness_scores(result, floor, grid))


def ranked_on(scores):
    """How many variants the v3 score read."""
    return sum(score is not None for score in scores.values())


def _failed(variant, gates):
    return any(v["gate"] in gates and v["status"] == "failed" for v in variant["verdicts"])


def _delta(result, label, metric):
    entry = result["variants"].get(label, {}).get("aggregate", {}).get(metric, {}).get("delta")
    return entry["median"] if entry else None


def _lower(result, worse, better, metric):
    values = _delta(result, worse, metric), _delta(result, better, metric)
    return None not in values and values[0] < values[1]


def duller(result, worse, better):
    """Whether `worse` reads duller than `better` on any of the dull readings."""
    return any(_lower(result, worse, better, metric) for metric in DULL_READINGS)


def _bug_rules(variants, ranking, ranked):
    rules = {"bug_flagged_muffled": _failed(variants["deesser_bug"], MUFFLED_GATES)}
    rules["bug_in_bottom_two"] = "deesser_bug" in ranking[-2:] if ranked else None
    return rules


def _engine_rules(variants, ranking, ranked):
    """The de-esser bug must read muffled and rank in the bottom two; APL must not read as altered."""
    rules = _bug_rules(variants, ranking, ranked) if "deesser_bug" in variants else {}
    if "apl" in variants:
        rules["apl_not_altered"] = not _failed(variants["apl"], ALTERED_GATES)
    return rules


def _in_top(ranking, labels):
    return all(ranking.index(label) < len(labels) for label in labels)


def _good_in_top(variants, ranking, ranked):
    """Whether every known-good variant present sits inside the top `len(good)`; None when none is present or nothing ranked."""
    good = [label for label in KNOWN_GOOD if label in variants]
    return _in_top(ranking, good) if good and ranked else None


def _gate_status(variant, name):
    return next((v["status"] for v in variant["verdicts"] if v["gate"] == name), "skipped")


def _gates_for(flag, gate_names):
    """The gates a list key covers: itself, or every listener flag for `*`."""
    if flag != ANY_FLAG:
        return [flag]
    return [name for name in gate_names if name.startswith(LISTENER_PREFIX)]


def _demands(lists, status, gate_names):
    """`[(gate, label, status)]` for every label of `{flag: labels}`."""
    return [(gate, label, status) for flag, labels in lists.items() for gate in _gates_for(flag, gate_names) for label in labels]


def _wanted(flags, clean, gate_names):
    """What the lists demand: flagged labels fail their gate, clean labels pass it (`*`: every listener flag)."""
    return _demands(flags or {}, "failed", gate_names) + _demands(clean or {}, "passed", gate_names)


def _gate_names(variants):
    return {verdict["gate"] for variant in variants.values() for verdict in variant["verdicts"]}


def flag_checks(variants, flags, clean):
    """`[(status demanded, status read)]` for every listed label this tape scored ("skipped": the gate read nothing)."""
    wanted = [item for item in _wanted(flags, clean, _gate_names(variants)) if item[1] in variants]
    return [(status, _gate_status(variants[label], gate)) for gate, label, status in wanted]


def flags_reproduced(variants, flags, clean):
    """Every flagged label fails its gate and every clean label passes it; None when nothing listed was read.

    A gate that read nothing (its reading not scored, or no such gate: a ledger flag named before
    its gate existed) is left out here and counted by `flags_unread`.
    """
    read = [(wanted, got) for wanted, got in flag_checks(variants, flags, clean) if got != "skipped"]
    return all(wanted == got for wanted, got in read) if read else None


def flags_unread(variants, flags, clean):
    """How many listed (gate, label) pairs the tape's verdicts could not read."""
    return sum(got == "skipped" for _wanted, got in flag_checks(variants, flags, clean))


def ordering_rules(result, floor, flags=None, clean=None, grid=None):
    """The known-ordering rules on one tape's scored variants; rules whose variants are absent read None or are left out."""
    variants = result["variants"]
    scores = harness_scores(result, floor, grid)
    ranking = _ranking(result, scores)
    ranked = ranked_on(scores) > 0
    rules = {"ranking": ranking, "ranked_on": ranked_on(scores), **_engine_rules(variants, ranking, ranked)}
    if "single4s" in variants and "stitched" in variants:
        rules["single4s_duller_than_stitched"] = duller(result, "single4s", "stitched")
    rules["good_in_top"] = _good_in_top(variants, ranking, ranked)
    rules["flags_reproduced"] = flags_reproduced(variants, flags, clean)
    rules["flags_unread"] = flags_unread(variants, flags, clean)
    return rules


def tape_record(result, floor, meta, grid=None):
    """One tape's record: the rules, the scores, the result and the by-ear meta (ranks, lists, verdict source, rounds)."""
    rules = ordering_rules(result, floor, meta.get("flags"), meta.get("clean"), grid)
    return {"rules": rules, "scores": harness_scores(result, floor, grid), "result": result, **meta}


def requested_families(result):
    """The families a stored result was asked for (older files: the families its variants report)."""
    if "requested" in result:
        return set(result["requested"])
    return set().union(*(set(variant.get("families", {})) for variant in result.get("variants", {}).values()))


def covers(result, families):
    """Whether a stored result was scored with every family now asked for."""
    return set(families) <= requested_families(result)


def manifest_inside_repo(manifest_path):
    """The manifest as a resolved path inside the repository; anything else is refused."""
    root = Path(__file__).resolve().parents[1]
    resolved = Path(manifest_path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise SystemExit(f"--known-ordering must name a manifest file inside {root}: {manifest_path}")
    return resolved


def json_default(value):
    """JSON for numpy scalars and arrays."""
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value.tolist() if isinstance(value, np.ndarray) else str(value)


def scored_tape(tape, entry, scoring):
    """The stored result for one tape, re-scored when it was scored without a family now asked for.

    `scoring` holds `families`, `registry`, `cache_dir` and `out_dir` (results go to `<out_dir>/ordering/<tape>.json`).
    """
    families, registry, cache_dir = scoring["families"], scoring["registry"], scoring["cache_dir"]
    path = Path(scoring["out_dir"]) / "ordering" / f"{tape}.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if stored is not None and covers(stored, families):
        return stored
    result, _cards = runner.score_variants(
        entry["source"], entry["variants"], families=families, registry=registry, cache_dir=cache_dir, language="ro"
    )
    result["requested"] = sorted(families)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, default=json_default), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def run_known_ordering(manifest_path, scoring, floor, grid=None, records=None):
    """Scores every tape of the manifest (`scoring`: see `scored_tape`) and applies the rules.

    Each tape carries its own by-ear ranks: the single 4 s probe is a known-bad output on
    Tele7abc and a known-good one on SOTI and Vaccin, so good and bad are read per tape.
    """
    manifest = json.loads(manifest_inside_repo(manifest_path).read_text(encoding="utf-8"))
    out = {}
    for tape, entry in manifest.items():
        result = scored_tape(tape, entry, scoring)
        meta = {"ranks": entry.get("by_ear_rank", {}), **tape_verdicts(tape, entry, records)}
        out[tape] = tape_record(result, floor, meta, grid)
    return out


def _meta(entry):
    return {key: value for key, value in entry.items() if key not in ("rules", "scores", "result")}


def regated(result, gates):
    """A copy of `result` with every variant's verdicts re-read under `gates` (no re-scoring)."""
    result = json.loads(json.dumps(result, default=json_default))
    for variant in result["variants"].values():
        variant["verdicts"] = gates_mod.evaluate_gates(variant["aggregate"], gates, variant.get("speaker_floor"))
        variant["hard_failures"] = gates_mod.hard_failures(variant["verdicts"])
    return result


def reevaluate_ordering(ordering, gates, floor, grid=None):
    """The ordering rules again, with every variant's verdicts re-read under `gates` (no re-scoring)."""
    return {tape: tape_record(regated(entry["result"], gates), floor, _meta(entry), grid) for tape, entry in ordering.items()}


def _rows(variant):
    return [
        WindowRow(row["window"], row["start_s"], row["end_s"], row["route"], row["source"], row["output"], row.get("output_route", ""))
        for row in variant.get("rows", [])
    ]


def route_aggregate(variant, route):
    """The variant's aggregate over the windows on `route`, with its file-level and meta readings as they are."""
    routed = aggregate([row for row in _rows(variant) if row.route == route], METRICS)
    whole = {name: entry for name, entry in variant.get("aggregate", {}).items() if name.split(".", 1)[0] in ("file", "meta")}
    return {**routed, **whole}


def route_result(result, route, gates):
    """`result` re-aggregated on the windows of one route, its verdicts re-read under `gates`."""
    result = json.loads(json.dumps(result, default=json_default))
    for variant in result["variants"].values():
        variant["aggregate"] = route_aggregate(variant, route)
    return regated(result, gates)


def route_ordering(ordering, route, gates, floor, grid=None):
    """The known-ordering records re-read on the windows of one route."""
    return {tape: tape_record(route_result(entry["result"], route, gates), floor, _meta(entry), grid) for tape, entry in ordering.items()}
