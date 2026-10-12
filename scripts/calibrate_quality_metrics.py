#!/usr/bin/env python3
"""Prove every output-quality metric moves the right way, then derive the gate thresholds.

    calibrate_quality_metrics.py --fixtures artifacts/realistic-v2 [--languages en de ...]
        [--excerpts tape_cut.wav ...] [--metrics all|dsp,...]
        [--known-ordering known_ordering.json] [--grid tata_v3.yaml]
        [--ledger verdicts.jsonl | --no-ledger] [--round ID] [--previous DIR]
        [--oracle-max N] [--out experiments/quality_calibration] [--device cuda]

Two sources of truth. The synthetic suite applies each failure in
`scripts/quality_degradations.py` at three levels to realistic-v2 fixtures and asks, per
(metric, failure), what the table declares: a reading that must move ("up" / "down") must
point the expected way at the severest level, order the levels on most languages and clear
three benign floors (identity, requantisation, a resample round trip, small shifts, on the
speech and on the music bed; the shifts leave out the raw-pair sync lags, which read the
shift itself) at the mildest one; a reading that must not move ("flat") must stay inside
its tolerance, or under its absolute twin's move; a source reading ("match") must read the
condition the degradation built. A check that reads nothing is "unscored" with its reason,
and the run prints every non-blind one R0's programme band clipped away (R1's air on every
fixture: the music beds end under 1.6 kHz, the speech targets under the 5292 Hz band R1's
first air band needs; `--excerpts` asserts it on a cut whose band reaches that, as
Vaccin's 7127 Hz does). The known-ordering set is the user's own tapes with
outputs judged by ear (`scripts/calibration_ordering.py`); the verdicts come from the ledger
when it has them, and the gates are derived from both (`scripts/calibration_gates.py`).

What changed for ear v3 (design 4.4, critique 11), and why:

- Every family, every time. The v2 calibration ran `--metrics dsp`, so every speech, mos and
  stems gate kept its hand-set value while `gates.json` looked complete. A case or tape now
  records the families it was asked for and is scored again when a later run asks for one it
  lacks (the v2 caches carry only dsp and are re-scored, not reused); the report lists the
  families that scored and every gate left underived, with the reason. `--metrics dsp` is
  still the quick re-check after a detector change, into a separate `--out`.
- Per route. Besides the pooled `gates.json` (what `tune_restoration.py` and the loop load),
  `gates_speech.json`, `gates_music.json` and `gates_mixed.json` hold the gates read on that
  route, derived on that route's windows with that route's benign floor (the pooled floor
  where a route has no benign window).
- Provenance. Each derived gate records `n_verdicts`, `source`, `verdict_source`, `family`,
  `round`, `rounds` and `rounds_agreeing`; a listener flag turns hard only once two separate
  listening rounds agree (`calibration_gates.with_rounds`), counted on the ledger rounds of
  the verdicts that gate rests on (`rounds`). The previous round is read from `--previous`
  (default: the gates files already in `--out`, read before they are replaced); the run's
  id (`round`, display only) is `--round`, or the ledger rounds the verdicts came from.
- The ranking uses the v3 score (`calibration_ordering`); the v2 composite read 0 for every
  variant because its learned judges were never scored.
- Real excerpts (plan 1.3: effect sizes on real 5-minute Tata cuts, not only Piper fixtures).
  `--excerpts` adds, per WAV excerpt, every degradation a real tape can carry
  (`quality_degradations.on_tape`: no clean reference, second voice or separate bed needed)
  and the speech benign set, as language `ro-<stem>` (Whisper reads Romanian). On the
  fixtures gap_air moved 0.02-0.37 dB against thresholds of -20 / -25, and the en / de
  speech fixtures hold no 200 ms non-speech run for R4 to read at all.
- fr beside en by default (`DEFAULT_LANGUAGES`, 2026-10-09): R2's texture needs 30
  fricative frames a window, which en's speech target never holds; fr's does.

Two refusals (E1, 2026-10-12):

- No gates file without `--known-ordering`. Without one every gate falls back to the floor
  rule, which is the hand-set value in `gates.py` or three benign floors: the v3 and v3b
  files (round "manifest") were written that way and their lkr 0.3 vetoed the shipped Tata
  outputs on 3 of 4 tapes (APL) and 4 of 4 (cathar). Such a run (the `--metrics dsp`
  re-check after a detector change) still writes both reports.
- The oracle check. Every realistic-v2 capture (`*_vhs.wav`) is scored against the
  programme it was made from (`*_target.wav` and `*_clean.wav`), an output no restoration
  can beat; a hard gate of the gates being written that fails more than `ORACLE_MAX_SHARE`
  (5%) of those pairs refuses denoising itself, and the run writes no gates file and exits 1.
  The pairs share the score cache (kind `oracle`, with their aggregate kept for the gates);
  `--oracle-max N` spreads N captures per language over the set for a quick run.

Writes `report.json`, `report.md`, `gates.json` and `gates_<route>.json` (each in the format
`scripts/restoration_quality/gates.py:load_gates` reads).
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Imported by name once the repository is on sys.path, so the script runs from any directory.
deg = import_module("scripts.quality_degradations")
checks_mod = import_module("scripts.calibration_checks")
cgates = import_module("scripts.calibration_gates")
cordering = import_module("scripts.calibration_ordering")
creport = import_module("scripts.calibration_report")
runner = import_module("scripts.restoration_quality.runner")
scorecard = import_module("scripts.restoration_quality.scorecard")
source_profile = import_module("scripts.restoration_quality.source_profile")
gates_mod = import_module("scripts.restoration_quality.gates")
GATES = gates_mod.GATES

# 4 (2026-10-09): R2 matches the gain per frame and needs 30 frames for the texture, R4 leaves out the bins
# under the 16-bit dither, R7 reads along the sync track, `file.dropouts` is new: every v3 score is stale.
# 5 (2026-10-10): R11 (`dsp.lf_programme_db`) is new; a schema-4 score lacks it, and the benign cases
# would then give it no floor at all (every R11 check would clear "three floors" of zero).
# 6 (2026-10-10): R11 reads its band on a rate-scaled STFT (1.35 Hz bins) and `dsp.attack_db` is paired on the
# source's onsets: a schema-5 score (the R11 subset runs before either) reads both the old way.
SCORE_SCHEMA = 6
SPEECH_FIXTURE = "mid00_speech_m15"
DONOR_FIXTURE = "mid01_speech_m15"
MUSIC_FIXTURE = "mid00_musiconly_m15"
ROUTES = cgates.ROUTES
BASE_PREFIX = checks_mod.BASE_PREFIX
META_PREFIX = checks_mod.META_PREFIX
PROFILE_READINGS = ("prog_bandwidth_hz",)
GATE_FILE = creport.GATE_FILE
TAPE_LANGUAGE = "ro"
# fr beside en (2026-10-09): en's readable window holds 10 fricative frames, under the texture's 30; de and es
# hold none and it two; fr holds 102 and orders the `sibilant_islands` levels (0.14 / 0.36 / 0.75 over 8 seeds).
DEFAULT_LANGUAGES = ("en", "fr")
# The benign floor, for scripts that cite it from here (`reward_noise_floor.py`).
noise_floor = checks_mod.noise_floor
ORACLE = "oracle"
ORACLE_SUFFIX = "_vhs.wav"
ORACLE_OUTPUTS = ("target", "clean")
ORACLE_MAX_SHARE = 0.05
NO_ORDERING = "no --known-ordering: every gate would rest on the hand-set values (the floor rule)"


@dataclass
class Case:
    """One (source, output) pair to score: a degradation at a level, or a benign transform; `base` is the untouched material."""

    kind: str
    name: str
    level: object
    language: str
    source: Path
    output: Path
    base: Path | None = None

    @property
    def case_id(self):
        """The case's name in the score cache and the reports."""
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
    """Base signals for one language: the speech target, its recorded noise, a donor voice and a music bed (with its own noise)."""
    target, rate = _read(_fixture(fixtures_dir, language, SPEECH_FIXTURE, "target"))
    clean, _rate = _read(_fixture(fixtures_dir, language, SPEECH_FIXTURE, "clean"))
    vhs, _rate = _read(_fixture(fixtures_dir, language, SPEECH_FIXTURE, "vhs"))
    donor, _rate = _read(_fixture(fixtures_dir, language, DONOR_FIXTURE, "target"))
    music, _rate = _read(_fixture(fixtures_dir, language, MUSIC_FIXTURE, "target"))
    noise = (vhs - clean)[: len(target)]
    materials = {
        "speech": target,
        "clean": clean,
        "vhs": vhs,
        "noise": noise,
        "donor": donor,
        "voice": target,
        "music": music,
        "rate": rate,
    }
    return {**materials, **_music_capture(fixtures_dir, language)}


def _music_capture(fixtures_dir, language):
    """`{music_bed, music_noise}`: the music fixture's clean bed and the noise its own capture recorded (`vhs - clean`).

    The stems' octave cases (`stem_octave_muted`, `music_hiss_removed`) are built on these, so
    their source is the realistic-v2 music capture itself. The speech fixture's `noise` is no
    hiss for a bed: it carries the speech chain's residual, so the planted mute's octave never
    read on en / fr and removing it read -6.7 dB on de (E1 review, 2026-10-12).
    """
    bed, _rate = _read(_fixture(fixtures_dir, language, MUSIC_FIXTURE, "clean"))
    capture, _rate = _read(_fixture(fixtures_dir, language, MUSIC_FIXTURE, "vhs"))
    length = min(len(bed), len(capture))
    return {"music_bed": bed[:length], "music_noise": capture[:length] - bed[:length]}


def _write(path, data, rate):
    """Writes one float WAV unless it is already there (a case is built once and reused)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        sf.write(str(path), np.asarray(data, dtype=np.float32), rate, subtype="FLOAT")
    return path


def _write_pair(out_dir, case_id, source, output, rate):
    return _write(out_dir / f"{case_id}_source.wav", source, rate), _write(out_dir / f"{case_id}_output.wav", output, rate)


def _degradation_cases(materials, language, out_dir, rng, names=None):
    cases = []
    for name, spec in ((name, deg.DEGRADATIONS[name]) for name in (deg.DEGRADATIONS if names is None else names)):
        base = materials[spec.base]
        base_path = _write(out_dir / language / f"base_{spec.base}.wav", base, materials["rate"])
        for level in spec.levels:
            source, output = deg.apply(name, level, base, materials["rate"], materials, rng)
            paths = _write_pair(out_dir / language, f"{name}_{level}", source, output, materials["rate"])
            cases.append(Case("degradation", name, level, language, *paths, base_path))
    return cases


def _benign_cases(materials, language, out_dir, rng, names=None):
    cases = []
    for name in deg.BENIGN if names is None else names:
        source, output = deg.apply_benign(name, materials[deg.benign_base(name)], materials["rate"], rng)
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


def oracle_output(source, kind):
    """The fixture file beside a capture: `<stem>_vhs.wav` -> `<stem>_<kind>.wav`."""
    return Path(str(source)[: -len(ORACLE_SUFFIX)] + f"_{kind}.wav")


def spread(items, limit):
    """`limit` items spread evenly over `items` (all of them when `limit` is 0 or covers them)."""
    if not limit or limit >= len(items):
        return list(items)
    return [items[int(index)] for index in np.linspace(0, len(items) - 1, int(limit))]


def build_oracle_cases(fixtures_dir, languages, limit=0):
    """Every realistic-v2 capture against the programme it was made from, per language (`--oracle-max` spreads a subset)."""
    cases = []
    for language in languages:
        for source in spread(sorted((Path(fixtures_dir) / language).glob(f"*{ORACLE_SUFFIX}")), limit):
            cases += [_oracle_case(source, language, kind) for kind in ORACLE_OUTPUTS if oracle_output(source, kind).is_file()]
    return cases


def _oracle_case(source, language, kind):
    stem = source.name[: -len(ORACLE_SUFFIX)]
    return Case(ORACLE, f"{stem}__{kind}", None, language, source, oracle_output(source, kind))


def _tape_materials(path):
    """A real excerpt as every base material: it is the speech, the capture and the bed at once."""
    audio, rate = _read(path)
    return {"speech": audio, "vhs": audio, "voice": audio, "music": audio, "rate": rate}


def build_tape_cases(excerpts, out_dir):
    """The degradations a real excerpt can carry (`quality_degradations.on_tape`) and the speech benign set, per excerpt.

    A case's language is `ro-<excerpt stem>`: Whisper reads it as Romanian, and the stem keeps
    each excerpt's cases apart in the cache and in the monotonic share.
    """
    cases = []
    for index, path in enumerate(excerpts or ()):
        rng, language = np.random.default_rng(20261009 + index), f"{TAPE_LANGUAGE}-{Path(path).stem}"
        materials = _tape_materials(path)
        cases += _degradation_cases(materials, language, out_dir / "cases", rng, deg.on_tape())
        cases += _benign_cases(materials, language, out_dir / "cases", rng, deg.on_tape_benign())
    return cases


def _side_medians(aggregate, side, names):
    return {name: aggregate[name][side]["median"] for name in names if aggregate[name].get(side)}


def card_readings(aggregate):
    """`{metric: delta median}`, plus the source median of every `meta.*` reading (what the capture profile read)."""
    meta = [name for name in aggregate if name.startswith(META_PREFIX)]
    return {**_side_medians(aggregate, "delta", aggregate), **_side_medians(aggregate, "source", meta)}


def _whole_file(readings):
    return {name: value for name, value in readings.items() if name.split(".", 1)[0] in ("file", "meta")}


def route_readings(card, readings):
    """`{route: readings}` over each route's windows; the file-level and meta readings ride along on every route."""
    out = {}
    for route in ROUTES:
        routed = scorecard.aggregate([row for row in card.rows if row.route == route], scorecard.METRICS)
        out[route] = {**card_readings(routed), **_whole_file(readings)}
    return out


def _wants_base(case):
    spec = deg.DEGRADATIONS.get(case.name)
    return case.base is not None and spec is not None and any(e.direction == "match" for e in spec.expects)


def base_profile(case):
    """The capture profile of a case's untouched material (`base.*`), for the source-condition checks; {} when none is needed."""
    if not _wants_base(case):
        return {}
    audio, rate = _read(case.base)
    profile = source_profile.capture_profile(audio, rate)
    return {f"{BASE_PREFIX}{name}": profile.get(name) for name in PROFILE_READINGS}


def covers(doc, families):
    """Whether a cached score was made by this schema with every family now asked for."""
    return doc.get("schema") == SCORE_SCHEMA and set(families) <= set(doc.get("requested", ()))


def usable(doc, case, families):
    """Whether a cached score serves `case`: it covers the families, and an oracle pair's keeps the aggregate its gates read."""
    return covers(doc, families) and (case.kind != ORACLE or "aggregate" in doc)


def whisper_language(case):
    """The language Whisper is held to: the fixture's own, Romanian for a tape excerpt (`ro-<stem>`)."""
    return case.language.split("-", 1)[0]


def _score_case(case, families, registry, cache_dir):
    registry.language = whisper_language(case)
    card, _pair = runner.score_pair(
        case.source, case.output, families=families, registry=registry, cache_dir=cache_dir, language=registry.language
    )
    readings = card_readings(card.aggregate)
    return {
        "schema": SCORE_SCHEMA,
        "requested": sorted(families),
        "families": card.families,
        "deltas": {**readings, **base_profile(case)},
        "routes": route_readings(card, readings),
        "aggregate": card.aggregate,
        "speaker_floor": card.speaker_floor,
    }


def _cached(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def score_cases(cases, families, registry, cache_dir, scores_dir):
    """Scores every case once (persisted per case, re-scored when a family now asked for is missing); `{case_id: record}`.

    Whisper is forced to the fixture's own language: the tapes are Romanian, the fixtures
    are Piper voices in five other languages.
    """
    scores = {}
    for case in cases:
        path = Path(scores_dir) / f"{case.case_id.replace('/', '_')}.json"
        doc = _cached(path)
        if not usable(doc, case, families):
            doc = _score_case(case, families, registry, cache_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(doc, indent=1, default=cordering.json_default), encoding="utf-8")
        scores[case.case_id] = doc
    return scores


def readings_of(scores, route=None):
    """`{case_id: readings}`: pooled over every window, or over one route's windows (base readings kept)."""
    if route is None:
        return {case_id: doc["deltas"] for case_id, doc in scores.items()}
    return {case_id: {**doc["routes"].get(route, {}), **_based(doc["deltas"])} for case_id, doc in scores.items()}


def _based(readings):
    return {name: value for name, value in readings.items() if name.startswith(BASE_PREFIX)}


def family_status(scores):
    """`{family: {status: count}}` over every case: how many scored ok and why the rest did not."""
    out = {}
    for doc in scores.values():
        for family, status in doc.get("families", {}).items():
            key = "ok" if status == "ok" else status.split(":", 1)[0]
            out.setdefault(family, {}).setdefault(key, 0)
            out[family][key] += 1
    return out


def scored_families(status):
    """The families that scored ok on at least one case."""
    return sorted(family for family, counts in status.items() if counts.get("ok"))


def round_id(explicit, ordering):
    """The run's id, for display: `--round`, else the ledger rounds of the verdicts, else "manifest".

    Agreement is counted per gate on the rounds of its own verdicts (`calibration_gates.with_rounds`).
    """
    if explicit:
        return str(explicit)
    rounds = sorted({str(r) for entry in ordering.values() for r in entry.get("rounds", ())})
    return "+".join(rounds) if rounds else "manifest"


def _pooled_gates(context, previous):
    floor, ordering, families, current = context["floor"], context["ordering"], context["families"], context["round"]
    derived, skipped = cgates.derive_gates(floor, ordering, families=families)
    return cgates.with_rounds(derived, previous.get(GATE_FILE), current, ordering), skipped


def _route_gates(context, previous, route):
    floor = cgates.route_floor(context["route_floors"][route], context["floor"])
    routed = cordering.route_ordering(context["ordering"], route, GATES, floor, context["grid"])
    derived, skipped = cgates.derive_route_gates(route, floor, routed, context["families"])
    return cgates.with_rounds(derived, previous.get(f"gates_{route}.json"), context["round"], routed), skipped


def derive_all(context, previous):
    """`(gate files, report additions)`: the pooled and per-route gates, what was skipped, and the ordering re-read under them."""
    gates, skipped = _pooled_gates(context, previous)
    routes = {route: _route_gates(context, previous, route) for route in ROUTES}
    files = {GATE_FILE: gates, **{f"gates_{route}.json": routes[route][0] for route in ROUTES}}
    after = cordering.reevaluate_ordering(context["ordering"], {**GATES, **cgates.as_gates(gates)}, context["floor"], context["grid"])
    additions = {
        "gates": gates,
        "skipped_gates": skipped,
        "route_gates": {route: routes[route][0] for route in ROUTES},
        "route_skipped": {route: routes[route][1] for route in ROUTES},
        "ordering_after_records": after,
    }
    return files, additions


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, default=Path("artifacts/realistic-v2"))
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--excerpts", nargs="+", type=Path, default=None, help="real tape excerpts (WAV) to degrade beside the fixtures")
    parser.add_argument("--metrics", default="all")
    parser.add_argument("--known-ordering", type=Path, default=None)
    parser.add_argument("--grid", type=Path, default=None, help="ranking grid (YAML/JSON); default: two-sided dsp readings at no change")
    parser.add_argument("--ledger", type=Path, default=None, help="verdict ledger (default: assets/quality_calibration/verdicts.jsonl)")
    parser.add_argument("--no-ledger", action="store_true", help="read the manifest's own flag lists only")
    parser.add_argument("--round", default=None, help="id of this calibration round (default: the ledger rounds of the verdicts)")
    parser.add_argument("--previous", type=Path, default=None, help="directory of the previous round's gates files (default: --out)")
    parser.add_argument("--oracle-max", type=int, default=0, help="oracle captures per language, spread over the set (0: every one)")
    parser.add_argument("--out", type=Path, default=Path("experiments/quality_calibration"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _families(metrics):
    return runner.ALL_FAMILIES if metrics == "all" else tuple(metrics.split(","))


def _records(args):
    return None if args.no_ledger else cordering.ledger_records(args.ledger)


def _ordering(args, scoring, suite, grid):
    if not args.known_ordering:
        return {}
    return cordering.run_known_ordering(args.known_ordering, scoring, suite["floor"], grid, _records(args))


def synthetic(args, families, registry):
    """The synthetic suite: `{cases, status, floor, route_floors, checks}` (status = the families that scored, per case)."""
    cases = build_cases(args.fixtures, args.languages, args.out) + build_tape_cases(args.excerpts, args.out)
    cases += build_oracle_cases(args.fixtures, args.languages, args.oracle_max)
    scores = score_cases(cases, families, registry, args.out / "cache", args.out / "scores")
    readings = readings_of(scores)
    floor = checks_mod.noise_floor(cases, readings)
    route_floors = {route: checks_mod.noise_floor(cases, readings_of(scores, route)) for route in ROUTES}
    checks = checks_mod.run_checks(cases, readings, floor)
    suite = {"cases": cases, "status": family_status(scores), "floor": floor, "route_floors": route_floors, "checks": checks}
    return {**suite, "scores": scores}


def load_grid(args):
    """The `--grid` ranking grid, or None for the default; a file that names no entry stops the run (`calibration_ordering.load_grid`)."""
    return cordering.load_grid(args.grid) if args.grid else None


def known_ordering_context(args, families, registry, suite, grid=None):
    """What the gates are derived from: the floors, the scored known-ordering tapes, the grid, the families and the round."""
    ordering = _ordering(
        args, {"families": families, "registry": registry, "cache_dir": args.out / "cache", "out_dir": args.out}, suite, grid
    )
    context = {"floor": suite["floor"], "route_floors": suite["route_floors"], "ordering": ordering, "grid": grid}
    return {**context, "families": scored_families(suite["status"]), "round": round_id(args.round, ordering)}


def _report(context, suite):
    ordering = context["ordering"]
    report = {"round": context["round"], "checks": suite["checks"], "families": suite["status"], "floor": suite["floor"]}
    report.update({"route_floors": suite["route_floors"], "ordering_records": ordering, "cases": [asdict(c) for c in suite["cases"]]})
    return {**report, "unmapped_flags": cordering.unmapped_flags(ordering, GATES)}


def main(argv=None):
    """Runs the calibration and writes its report and gates; exit 1 when a non-blind sensitivity check fails.

    The grid is read first: a wrong `--grid` stops the run before the synthetic suite, which
    takes hours on every family, not after it.
    """
    args = _parse_args(argv)
    grid = load_grid(args)
    families, registry = _families(args.metrics), runner.ModelRegistry(device=args.device)
    previous = creport.previous_gates(args.previous or args.out)
    suite = synthetic(args, families, registry)
    context = known_ordering_context(args, families, registry, suite, grid)
    files, additions = derive_all(context, previous)
    oracle = oracle_check(suite["cases"], suite["scores"], {**GATES, **cgates.as_gates(files[GATE_FILE])})
    refusal = refusal_reason(args, oracle)
    report = {**_report(context, suite), **additions, "oracle": oracle, "gates_written": not refusal, "refusal": refusal}
    creport.write_reports(args.out, report, {} if refusal else files)
    notes = (oracle_line(oracle), refusal_line(refusal))
    return max(print_summary(args.out, suite["checks"], suite["status"], families, notes), int(oracle_failed(oracle)))


def oracle_check(cases, scores, gates):
    """`{pairs, gates, by_output, failing, passed}`: per hard gate, how many oracle pairs it fails, overall and per output kind.

    `failing` names every gate failing more than `ORACLE_MAX_SHARE` of the pairs; with no pair
    at all the check cannot vouch for the gates and does not pass.
    """
    failed = _oracle_failures(cases, scores, gates)
    rows = _shares(_counts(names for _kind, names in failed), len(failed))
    failing = _over_share(rows)
    by_output = {kind: _counts(_of_output(failed, kind)) for kind in ORACLE_OUTPUTS}
    return {"pairs": len(failed), "gates": rows, "by_output": by_output, "failing": failing, "passed": bool(failed) and not failing}


def _over_share(rows):
    """The gates failing more than `ORACLE_MAX_SHARE` of the pairs, by name."""
    return sorted(name for name, row in rows.items() if row["share"] > ORACLE_MAX_SHARE)


def _oracle_failures(cases, scores, gates):
    """`[(output kind, hard failures)]`, one per oracle pair."""
    return [(case.name.rsplit("__", 1)[-1], _hard_failures(scores[case.case_id], gates)) for case in cases if case.kind == ORACLE]


def _shares(counts, pairs):
    """`{gate: {failed, share}}` in name order."""
    return {name: {"failed": count, "share": count / pairs} for name, count in sorted(counts.items())}


def _of_output(failed, kind):
    """The failure lists of the pairs whose output is `kind`."""
    return [names for found, names in failed if found == kind]


def _hard_failures(doc, gates):
    """The hard gates one oracle pair's stored aggregate fails."""
    return gates_mod.hard_failures(gates_mod.evaluate_gates(doc.get("aggregate", {}), gates, doc.get("speaker_floor")))


def _counts(groups):
    """`{name: how many groups name it}`."""
    counts = {}
    for names in groups:
        for name in names:
            counts[name] = counts.get(name, 0) + 1
    return counts


def oracle_failed(oracle):
    """Whether the oracle check refuses the gates (a gate over the share, or no pair to check)."""
    return not oracle["passed"]


def refusal_reason(args, oracle):
    """Why no gates file is written, or None: no known ordering, or the oracle check failed."""
    if not args.known_ordering:
        return NO_ORDERING
    if not oracle["pairs"]:
        return "no oracle pair: no *_vhs.wav with its *_target.wav or *_clean.wav under --fixtures"
    if oracle["failing"]:
        return f"oracle: {', '.join(oracle['failing'])} fail more than {ORACLE_MAX_SHARE:.0%} of {oracle['pairs']} oracle pairs"
    return None


def non_blind_failures(checks):
    """The failed checks that are not blind (a blind pair is reported, never asserted)."""
    return [c for c in checks if c["status"] == "fail" and not c["blind"]]


def non_blind_unscored(checks):
    """The checks that are not blind and read nothing: asserted on paper only, each with its `reason`."""
    return [c for c in checks if c["status"] == "unscored" and not c["blind"]]


def clipped_line(checks):
    """The line naming every non-blind check R0's programme band clipped away, or None when there is none."""
    clipped = [c for c in non_blind_unscored(checks) if c.get("reason") == checks_mod.CLIPPED]
    names = ", ".join(f"{c['degradation']} {c['metric']} ({c['programme_band_hz']:.0f} Hz)" for c in clipped)
    return f"non-blind checks unread, the source's programme band ends under the reading's band: {names}" if clipped else None


def _missing_line(status, families):
    missing = sorted(set(families) - set(scored_families(status)))
    return f"families not scored (their gates keep the hand-set values): {', '.join(missing)}" if missing else None


def print_summary(out, checks, status, families, notes=()):
    """Prints the counts, the clipped checks, any family that did not score and the `notes` lines; returns the exit code."""
    failed, unscored = non_blind_failures(checks), non_blind_unscored(checks)
    passed = f"{sum(c['status'] == 'pass' for c in checks)}/{len(checks)} sensitivity checks pass"
    print(f"{passed}; {len(failed)} non-blind failures; {len(unscored)} non-blind unscored; report at {out / 'report.md'}")
    for line in filter(None, (clipped_line(checks), _missing_line(status, families), *notes)):
        print(line)
    return 1 if failed else 0


def oracle_line(oracle):
    """The oracle check in one line, or None when it did not run."""
    if oracle is None:
        return None
    counts = ", ".join(f"{name} {row['failed']}" for name, row in oracle["gates"].items()) or "none"
    return f"oracle pairs: {oracle['pairs']}; hard failures per gate: {counts}"


def refusal_line(refusal):
    """The refusal in one line, or None when the gates files were written."""
    return f"gates files not written: {refusal}" if refusal else None


if __name__ == "__main__":
    sys.exit(main())
