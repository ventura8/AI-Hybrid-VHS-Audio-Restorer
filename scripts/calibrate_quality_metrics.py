#!/usr/bin/env python3
"""Prove every output-quality metric moves the right way, then derive the gate thresholds.

    calibrate_quality_metrics.py --fixtures artifacts/realistic-v2 [--languages en de ...]
        [--metrics all|dsp,...] [--known-ordering known_ordering.json]
        [--out experiments/quality_calibration] [--device cuda]

Two sources of truth. The synthetic suite applies each failure in
`scripts/quality_degradations.py` at three levels to realistic-v2 fixtures and asks,
per (metric, failure): does the delta point the expected way at the severest level, is
it monotonic across the levels, and does the mildest level already clear three times the
benign noise floor (identity, requantisation, a resample round trip, small shifts)? The
known-ordering set is the user's own tapes with outputs judged by ear this week: the
de-esser bug must be flagged muffled, the single 4 s probe must lose to the stitched one
on coloration and highs, APL must not read as altered.

Writes `report.json`, `report.md` and `gates.json` (the thresholds, in the format
`scripts/restoration_quality/gates.py:load_gates` reads).
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import quality_degradations as deg  # noqa: E402
from scripts.restoration_quality import gates as gates_mod  # noqa: E402
from scripts.restoration_quality import runner  # noqa: E402
from scripts.restoration_quality.gates import GATES  # noqa: E402

FLOOR_PERCENTILE = 95.0
EFFECT_MULTIPLE = 3.0
MONOTONIC_SHARE = 0.8
SPEECH_FIXTURE = "mid00_speech_m15"
DONOR_FIXTURE = "mid01_speech_m15"
MUSIC_FIXTURE = "mid00_musiconly_m15"
KNOWN_BAD = ("deesser_bug", "single4s")
KNOWN_GOOD = ("cathar075", "stitched", "apl")
GOOD_RANK_MAX = 2
BAD_RANK_MIN = 4
COMPOSITE = ("speech.utmos", "mos.sigmos_ovrl", "mos.audiobox_pq")
MUFFLED_GATES = ("dsp.hf_4k8k", "dsp.hf_8k16k", "mos.sigmos_col")
ALTERED_GATES = ("speech.speaker", "speech.cer_median", "speech.cer_tail")


@dataclass
class Case:
    """One (source, output) pair to score: a degradation at a level, or a benign transform."""

    kind: str
    name: str
    level: object
    language: str
    source: Path
    output: Path

    @property
    def case_id(self):
        return f"{self.language}/{self.kind}_{self.name}_{self.level}"


def _read(path):
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), int(rate)


def _fixture(fixtures_dir, language, stem, part):
    path = Path(fixtures_dir) / language / f"{stem}_{part}.wav"
    if not path.exists():
        raise SystemExit(f"fixture missing: {path}; run scripts/make_realistic_fixtures_v2.py first")
    return path


def _materials(fixtures_dir, language):
    """Base signals for one language: the speech target, its recorded noise, a donor voice and a music bed."""
    target, rate = _read(_fixture(fixtures_dir, language, SPEECH_FIXTURE, "target"))
    clean, _rate = _read(_fixture(fixtures_dir, language, SPEECH_FIXTURE, "clean"))
    vhs, _rate = _read(_fixture(fixtures_dir, language, SPEECH_FIXTURE, "vhs"))
    donor, _rate = _read(_fixture(fixtures_dir, language, DONOR_FIXTURE, "target"))
    music, _rate = _read(_fixture(fixtures_dir, language, MUSIC_FIXTURE, "target"))
    noise = (vhs - clean)[: len(target)]
    return {"speech": target, "clean": clean, "vhs": vhs, "noise": noise, "donor": donor, "voice": target, "music": music, "rate": rate}


def _write_pair(out_dir, case_id, source, output, rate):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = (out_dir / f"{case_id}_source.wav", out_dir / f"{case_id}_output.wav")
    for path, data in zip(paths, (source, output)):
        if not path.exists():
            sf.write(str(path), np.asarray(data, dtype=np.float32), rate, subtype="FLOAT")
    return paths


def _degradation_cases(materials, language, out_dir, rng):
    cases = []
    for name, spec in deg.DEGRADATIONS.items():
        base = materials[spec.base] if spec.base != "music" else materials["voice"]
        for level in spec.levels:
            source, output = deg.apply(name, level, base, materials["rate"], materials, rng)
            paths = _write_pair(out_dir / language, f"{name}_{level}", source, output, materials["rate"])
            cases.append(Case("degradation", name, level, language, *paths))
    return cases


def _benign_cases(materials, language, out_dir, rng):
    cases = []
    for name in deg.BENIGN:
        source, output = deg.apply_benign(name, materials["speech"], materials["rate"], rng)
        paths = _write_pair(out_dir / language, f"benign_{name}", source, output, materials["rate"])
        cases.append(Case("benign", name, None, language, *paths))
    return cases


def build_cases(fixtures_dir, languages, out_dir):
    """Every synthetic case for the requested languages, written to `out_dir/cases/<lang>/`."""
    cases = []
    for index, language in enumerate(languages):
        rng = np.random.default_rng(20260920 + index)
        materials = _materials(fixtures_dir, language)
        cases += _degradation_cases(materials, language, out_dir / "cases", rng)
        cases += _benign_cases(materials, language, out_dir / "cases", rng)
    return cases


def _deltas(card):
    """`{metric: delta median}` from a scored card (paired metrics read as their output value)."""
    return {name: entry["delta"]["median"] for name, entry in card.aggregate.items() if entry.get("delta")}


def score_cases(cases, families, registry, cache_dir, scores_dir):
    """Scores every case once (results persisted per case); returns `{case_id: deltas}`.

    Whisper is forced to the fixture's own language: the tapes are Romanian, the fixtures
    are Piper voices in five other languages.
    """
    scores = {}
    for case in cases:
        path = Path(scores_dir) / f"{case.case_id.replace('/', '_')}.json"
        if not path.exists():
            registry.language = case.language
            card, _pair = runner.score_pair(
                case.source, case.output, families=families, registry=registry, cache_dir=cache_dir, language=case.language
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"deltas": _deltas(card), "families": card.families}, indent=1), encoding="utf-8")
        scores[case.case_id] = json.loads(path.read_text(encoding="utf-8"))["deltas"]
    return scores


def noise_floor(cases, scores):
    """Per metric: the benign centre (median delta) and floor (p95 |delta - centre|) over every benign case."""
    values = {}
    for case in (c for c in cases if c.kind == "benign"):
        for metric, value in scores[case.case_id].items():
            values.setdefault(metric, []).append(value)
    return {metric: _centre_and_floor(v) for metric, v in values.items()}


def _centre_and_floor(values):
    centre = float(np.median(values))
    return {"centre": centre, "floor": float(np.percentile(np.abs(np.asarray(values) - centre), FLOOR_PERCENTILE))}


def _level_values(cases, scores, name, metric):
    """`{level: [delta per language]}` for one degradation and metric, levels in table order."""
    out = {level: [] for level in deg.DEGRADATIONS[name].levels}
    for case in (c for c in cases if c.kind == "degradation" and c.name == name):
        value = scores[case.case_id].get(metric)
        if value is not None:
            out[case.level].append(value)
    return out


def _signed(direction, value):
    return value if direction == "up" else -value


def _monotonic_share(per_level, direction, centre):
    """Share of languages on which the three levels are strictly ordered the expected way."""
    per_language = list(zip(*[values for values in per_level.values()]))
    if not per_language:
        return 0.0
    ordered = [all(_signed(direction, b - centre) > _signed(direction, a - centre) for a, b in zip(seq, seq[1:])) for seq in per_language]
    return float(np.mean(ordered))


def _status(direction, monotonic, effect_mild):
    return "pass" if direction and monotonic >= MONOTONIC_SHARE and effect_mild >= EFFECT_MULTIPLE else "fail"


def check_expectation(name, expectation, cases, scores, floor):
    """direction / monotonic / effect for one (degradation, metric); None readings make the check 'unscored'."""
    per_level = _level_values(cases, scores, name, expectation.metric)
    medians = [np.median(v) if v else None for v in per_level.values()]
    result = {"degradation": name, "metric": expectation.metric, "blind": expectation.blind}
    if any(m is None for m in medians):
        return {**result, "status": "unscored"}
    reference = floor.get(expectation.metric, {"centre": 0.0, "floor": 0.0})
    signed = [_signed(expectation.direction, m - reference["centre"]) for m in medians]
    monotonic = _monotonic_share(per_level, expectation.direction, reference["centre"])
    effect_mild = float(signed[0] / (reference["floor"] + 1e-9))
    direction = bool(signed[-1] > 0)
    return {
        **result,
        "direction": direction,
        "monotonic_share": monotonic,
        "effect_mild": effect_mild,
        "medians": [float(m) for m in medians],
        "status": _status(direction, monotonic, effect_mild),
    }


def run_checks(cases, scores, floor):
    return [
        check_expectation(name, expectation, cases, scores, floor)
        for name, spec in deg.DEGRADATIONS.items()
        for expectation in spec.expects
    ]


def _z(variant, metric, floor):
    entry = variant["aggregate"].get(metric, {}).get("delta")
    if not entry:
        return None
    reference = floor.get(metric, {"centre": 0.0, "floor": 1.0})
    return (entry["median"] - reference["centre"]) / (reference["floor"] + 1e-9)


def _composite(variant, floor):
    """Mean z-score of the composite metrics' deltas against the benign floor (higher = better)."""
    parts = [z for z in (_z(variant, metric, floor) for metric in COMPOSITE) if z is not None]
    return float(np.mean(parts)) if parts else 0.0


def harness_ranking(result, floor):
    """Variants ordered best first: fewest hard failures, then the highest composite."""
    keyed = {label: (len(variant["hard_failures"]), -_composite(variant, floor)) for label, variant in result["variants"].items()}
    return sorted(keyed, key=lambda label: keyed[label])


def _failed(variant, gates):
    return any(v["gate"] in gates and v["status"] == "failed" for v in variant["verdicts"])


def _delta(result, label, metric):
    entry = result["variants"].get(label, {}).get("aggregate", {}).get(metric, {}).get("delta")
    return entry["median"] if entry else None


def _lower(result, worse, better, metric):
    values = _delta(result, worse, metric), _delta(result, better, metric)
    return values[0] is not None and values[1] is not None and values[0] < values[1]


def _duller(result, worse, better):
    return _lower(result, worse, better, "dsp.hf_4k8k") or _lower(result, worse, better, "mos.sigmos_col")


def ordering_rules(result, floor):
    """The known-ordering rules on one tape's scored variants; rules whose variants are absent are left out."""
    variants = result["variants"]
    ranking = harness_ranking(result, floor)
    rules = {"ranking": ranking}
    if "deesser_bug" in variants:
        rules["bug_flagged_muffled"] = _failed(variants["deesser_bug"], MUFFLED_GATES)
        rules["bug_in_bottom_two"] = "deesser_bug" in ranking[-2:]
    if "single4s" in variants and "stitched" in variants:
        rules["single4s_duller_than_stitched"] = _duller(result, "single4s", "stitched")
    if "apl" in variants:
        rules["apl_not_altered"] = not _failed(variants["apl"], ALTERED_GATES)
    good = [label for label in KNOWN_GOOD if label in variants]
    rules["good_in_top"] = all(ranking.index(label) < len(good) for label in good) if good else None
    return rules


def run_known_ordering(manifest_path, families, registry, cache_dir, floor, out_dir):
    """Scores every tape in the manifest once (persisted under out_dir/ordering) and applies the rules.

    Each tape carries its own by-ear ranks: the single 4 s probe is a known-bad output on
    Tele7abc and a known-good one on SOTI and Vaccin, so good and bad are read per tape.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    out = {}
    for tape, entry in manifest.items():
        path = Path(out_dir) / "ordering" / f"{tape}.json"
        if not path.exists():
            result, _cards = runner.score_variants(
                entry["source"], entry["variants"], families=families, registry=registry, cache_dir=cache_dir, language="ro"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=1, default=runner._json_default), encoding="utf-8")
        result = json.loads(path.read_text(encoding="utf-8"))
        out[tape] = {"rules": ordering_rules(result, floor), "result": result, "ranks": entry.get("by_ear_rank", {})}
    return out


def reevaluate_ordering(ordering, gates, floor):
    """The ordering rules again, with every variant's verdicts re-read under `gates` (no re-scoring)."""
    out = {}
    for tape, entry in ordering.items():
        result = json.loads(json.dumps(entry["result"]))
        for variant in result["variants"].values():
            variant["verdicts"] = gates_mod.evaluate_gates(variant["aggregate"], gates, variant.get("speaker_floor"))
            variant["hard_failures"] = gates_mod.hard_failures(variant["verdicts"])
        out[tape] = {"rules": ordering_rules(result, floor), "result": result, "ranks": entry.get("ranks", {})}
    return out


def _verdict_value(variant, gate):
    """The value one variant's verdict holds for `gate`, or None."""
    for verdict in variant["verdicts"]:
        if verdict["metric"] == gate.metric and verdict["stat"] == gate.stat:
            return verdict["value"]
    return None


def _gate_metric_values(ordering, gate):
    """`(good, bad)` readings of the gate across the known-ordering set, good/bad by each tape's own by-ear ranks."""
    good, bad = [], []
    for tape in ordering.values():
        ranks = tape.get("ranks", {})
        for label, variant in tape["result"]["variants"].items():
            value = _verdict_value(variant, gate)
            if value is None or label not in ranks:
                continue
            (good if ranks[label] <= GOOD_RANK_MAX else bad if ranks[label] >= BAD_RANK_MIN else []).append(value)
    return good, bad


def _worst(values, op):
    return max(values) if op == "<=" else min(values)


def _best(values, op):
    return min(values) if op == "<=" else max(values)


def _from_ordering(gate, good, bad, spread):
    """The midpoint between the worst known-good and the best known-bad reading, when the bad side really is worse.

    A gap on the wrong side means this reading does not separate the two sets in the
    direction the gate reads; the floor rule applies instead of a threshold that would
    veto the good outputs.
    """
    if not good or not bad:
        return None
    worst_good, best_bad = _worst(good, gate.op), _best(bad, gate.op)
    gap = (best_bad - worst_good) if gate.op == "<=" else (worst_good - best_bad)
    if gap <= 2 * spread:
        return None
    return {"threshold": float((worst_good + best_bad) / 2.0), "severity": gate.severity, "source": "known ordering"}


def _from_floor(gate, spread):
    """Three floors past the benign centre, unless the hand-set threshold is already stricter than that."""
    floor_value = EFFECT_MULTIPLE * spread * (1.0 if gate.op == "<=" else -1.0)
    threshold = floor_value if abs(floor_value) > abs(gate.threshold) else gate.threshold
    return {"threshold": float(threshold), "severity": gate.severity, "source": "floor"}


def derive_gate(gate, floor, ordering):
    """A threshold for one gate, or None for relative and unmeasured gates.

    Order of preference: a clear gap between the known-good and known-bad outputs; else
    the hand-set threshold, relaxed to the worst known-good reading plus three floors when
    that threshold would veto an output the user accepted by ear; else three floors.
    """
    spread = floor.get(gate.metric, {}).get("floor")
    if not isinstance(gate.threshold, (int, float)) or spread is None:
        return None
    good, bad = _gate_metric_values(ordering, gate) if ordering else ([], [])
    return _from_ordering(gate, good, bad, spread) or _from_good_bound(gate, good, spread) or _from_floor(gate, spread)


def _from_good_bound(gate, good, spread):
    """The hand-set threshold pushed past the worst known-good reading, so no accepted output is vetoed."""
    if not good:
        return None
    worst_good = _worst(good, gate.op)
    tolerant = worst_good + 3 * spread if gate.op == "<=" else worst_good - 3 * spread
    if gates_mod._passes(worst_good, gate.op, float(gate.threshold)):
        return None
    return {"threshold": float(tolerant), "severity": gate.severity, "source": "known good bound"}


def derive_gates(floor, ordering):
    derived = {name: derive_gate(gate, floor, ordering) for name, gate in GATES.items()}
    return {name: value for name, value in derived.items() if value}


def _check_line(row):
    blind = " (blind)" if row["blind"] else ""
    if row["status"] == "unscored":
        return f"| {row['degradation']} | {row['metric']}{blind} | unscored | | | | |"
    medians = ", ".join(f"{m:+.3f}" for m in row["medians"])
    cells = f"{row['status']} | {row['direction']} | {row['monotonic_share']:.2f} | {row['effect_mild']:.1f}"
    return f"| {row['degradation']} | {row['metric']}{blind} | {cells} | {medians} |"


def _markdown(checks, floor, ordering, gates, ordering_after):
    head = "| degradation | metric | status | direction | monotonic | effect (x floor) | medians |"
    lines = ["# Output-quality metric calibration", "", "## Sensitivity", "", head, "|---|---|---|---|---|---|---|"]
    lines += [_check_line(row) for row in checks]
    lines += ["", "## Benign floor", "", "| metric | centre | floor (p95) |", "|---|---|---|"]
    lines += [f"| {metric} | {v['centre']:+.4f} | {v['floor']:.4f} |" for metric, v in sorted(floor.items())]
    lines += ["", "## Known ordering (default gates)", ""]
    lines += [f"- **{tape}**: " + ", ".join(f"{k}={v}" for k, v in entry["rules"].items()) for tape, entry in ordering.items()]
    lines += ["", "## Known ordering (derived gates)", ""]
    lines += [f"- **{tape}**: " + ", ".join(f"{k}={v}" for k, v in entry["rules"].items()) for tape, entry in ordering_after.items()]
    lines += ["", "## Derived gates", "", "| gate | threshold | severity | source |", "|---|---|---|---|"]
    lines += [f"| {name} | {g['threshold']:+.3f} | {g['severity']} | {g['source']} |" for name, g in gates.items()]
    return "\n".join(lines) + "\n"


def _write_reports(out, checks, floor, ordering, gates, cases, ordering_after=None):
    out.mkdir(parents=True, exist_ok=True)
    (out / "gates.json").write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    report = {
        "checks": checks,
        "floor": floor,
        "ordering": {t: e["rules"] for t, e in ordering.items()},
        "ordering_with_derived_gates": {t: e["rules"] for t, e in (ordering_after or {}).items()},
        "gates": gates,
        "cases": [asdict(c) for c in cases],
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(_markdown(checks, floor, ordering, gates, ordering_after or {}), encoding="utf-8")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, default=Path("artifacts/realistic-v2"))
    parser.add_argument("--languages", nargs="+", default=["en"])
    parser.add_argument("--metrics", default="all")
    parser.add_argument("--known-ordering", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("experiments/quality_calibration"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    families = runner.ALL_FAMILIES if args.metrics == "all" else tuple(args.metrics.split(","))
    registry = runner.ModelRegistry(device=args.device)
    cache_dir = args.out / "cache"
    cases = build_cases(args.fixtures, args.languages, args.out)
    scores = score_cases(cases, families, registry, cache_dir, args.out / "scores")
    floor = noise_floor(cases, scores)
    checks = run_checks(cases, scores, floor)
    ordering = run_known_ordering(args.known_ordering, families, registry, cache_dir, floor, args.out) if args.known_ordering else {}
    gates = derive_gates(floor, ordering)
    derived = {
        name: gates_mod.Gate(**{**gates_mod.asdict(GATES[name]), "threshold": g["threshold"], "severity": g["severity"]})
        for name, g in gates.items()
    }
    ordering_after = reevaluate_ordering(ordering, {**GATES, **derived}, floor)
    _write_reports(args.out, checks, floor, ordering, gates, cases, ordering_after)
    failed = [c for c in checks if c["status"] == "fail" and not c["blind"]]
    summary = f"{len(checks) - len(failed)}/{len(checks)} sensitivity checks pass; {len(failed)} non-blind failures"
    print(f"{summary}; report at {args.out / 'report.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
