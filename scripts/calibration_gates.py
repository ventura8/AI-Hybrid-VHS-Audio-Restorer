"""Gate thresholds from the benign floor and the listener's verdicts, with where each one came from.

`scripts/calibrate_quality_metrics.py` measures a benign floor per metric and scores the
known-ordering tapes; this module turns both into thresholds (ear v3 design 4.4).

Per gate, in order of preference:

1. the midpoint between the worst known-good and the best known-bad reading, when that gap
   clears two floors in the gate's direction ("known ordering");
2. the hand-set threshold relaxed to the worst known-good reading plus three floors, when the
   hand-set value would veto an output the user accepted ("known good bound");
3. three floors past the benign centre, unless the hand-set threshold is stricter ("floor").

The sides come from the listener's words first: a tape whose flag lists (from the verdict
ledger, or the manifest when no ledger flag record judged the tape's labels) name a gate reads that gate by
its lists, flagged = bad, clean = good, and `clean["*"]` ("accepted with no complaint at all")
counts as clean for every listener flag (`listener.*`). Once any tape lists a gate, tapes
without lists stay out of it: a by-ear rank is an overall preference, not a verdict on one
reading. Gates no tape lists are sided by the by-ear ranks (rank 1-2 good, 4 and worse bad).

Every derived gate records `n_verdicts` (the verdicts its threshold rests on: good and bad
for a known ordering, good for a known-good bound, 0 for a floor), `verdict_source` (ledger,
manifest or by-ear rank) and, through `with_rounds`, how many separate listening rounds
agree on it. A listener flag becomes a hard gate only after two rounds agree
(`PROMOTE_AFTER_ROUNDS`): a round agrees when the previous calibration's threshold still
separates this one's good and bad readings, and it counts only when the verdicts the gate
rests on hold a ledger round the previous entry's did not (`rounds`: the rounds of the
records that name the gate on the tapes that side it, `calibration_ordering.gate_rounds`).
The run's own id (`round`, the union of every tape's rounds) is for display: with it, a
round-4 flag about `listener.bright` on another tape promoted `listener.dead_air`, resting on
round 1 alone, to hard. A rerun, or a new round about another gate, does not count; a
disagreement resets the count to 1 and the gate back to a flag. Manifest lists and by-ear
ranks name no ledger round, so a gate resting on them counts 1, and a previous entry that
names none (the v2 gates files, a `--no-ledger` run) cannot say which verdicts it came from:
the count starts afresh rather than promote a flag on verdicts both files share.

A gate whose metric's family was not scored in this run is not derived (`skip_reason`): the
v2 gates came from a dsp-only run, so every speech, mos and stems gate kept its hand-set
value without the gates file saying so. The report now names each one and why.

Per route: `derive_route_gates` keeps the gates whose metric is read on that route
(`scorecard.METRICS[...].routes`; file-level readings apply on every route), reads the route's
own benign floor where it has one and the pooled floor otherwise.
"""

from scripts.restoration_quality import gates as gates_mod
from scripts.restoration_quality.gates import GATES
from scripts.restoration_quality.scorecard import METRICS

EFFECT_MULTIPLE = 3.0
ORDERING_GAP_FLOORS = 2.0
GOOD_RANK_MAX = 2
BAD_RANK_MIN = 4
ANY_FLAG = "*"
LISTENER_PREFIX = "listener."
PROMOTE_AFTER_ROUNDS = 2
ROUTES = ("speech", "music", "mixed")
KNOWN_ORDERING = "known ordering"
GOOD_BOUND = "known good bound"
FLOOR = "floor"
BY_RANK = "by-ear rank"


def passes(value, op, threshold):
    """Whether `value` passes a gate reading `op` against `threshold` (`<=`, `>=` or `abs<=`)."""
    if op == "<=":
        return value <= threshold
    if op == ">=":
        return value >= threshold
    return abs(value) <= threshold


def verdict_value(variant, gate):
    """The value one variant's verdict holds for `gate`'s metric and statistic, or None."""
    for verdict in variant["verdicts"]:
        if verdict["metric"] == gate.metric and verdict["stat"] == gate.stat:
            return verdict["value"]
    return None


def _mapping(tape, key):
    """`tape[key]` as a mapping, empty when the tape has none."""
    return tape.get(key) or {}


def _is_listener(name):
    return bool(name) and name.startswith(LISTENER_PREFIX)


def _lists(tape, name):
    """`(flagged labels, clean labels)` the tape names for gate `name`; `*` counts as clean for listener flags."""
    flags, clean = _mapping(tape, "flags"), _mapping(tape, "clean")
    accepted = set(clean.get(ANY_FLAG, ())) if _is_listener(name) else set()
    return set(flags.get(name, ())), set(clean.get(name, ())) | accepted


def lists_gate(tape, name):
    """Whether the tape's flag lists say anything about gate `name` (an empty list for it counts: it was judged)."""
    named = set(_mapping(tape, "flags")) | set(_mapping(tape, "clean"))
    return name in named or bool(_lists(tape, name)[1])


def _by_flags(tape, label, name):
    """bad / good / neither by the tape's lists for gate `name`."""
    flagged, clean = _lists(tape, name)
    if label in flagged:
        return "bad"
    return "good" if label in clean else "neither"


def _by_rank(tape, label):
    """good / bad / neither by the tape's by-ear rank of `label`."""
    rank = (tape.get("ranks") or {}).get(label)
    if rank is None:
        return "neither"
    if rank <= GOOD_RANK_MAX:
        return "good"
    return "bad" if rank >= BAD_RANK_MIN else "neither"


def _side(tape, label, name, listed):
    """`(side, source)` of one variant: by the lists when the gate is listed anywhere, else by its by-ear rank."""
    if not listed:
        return _by_rank(tape, label), BY_RANK
    if not lists_gate(tape, name):
        return "neither", None
    return _by_flags(tape, label, name), tape.get("verdict_source", "manifest")


def _listed_anywhere(ordering, name):
    """Whether any tape's lists name gate `name`."""
    return bool(name) and any(lists_gate(tape, name) for tape in ordering.values())


def _tape_sided(tape, gate, name, listed):
    """`[(side, value, source)]` for every scored variant of one tape."""
    out = []
    for label, variant in tape["result"]["variants"].items():
        value = verdict_value(variant, gate)
        if value is not None:
            side, source = _side(tape, label, name, listed)
            out.append((side, value, source))
    return out


def sided_values(ordering, gate, name=None):
    """`[(side, value, source)]` for every scored variant of every tape."""
    listed = _listed_anywhere(ordering, name)
    return [item for tape in ordering.values() for item in _tape_sided(tape, gate, name, listed)]


def _tape_rounds(tape, name):
    """The ledger rounds of the tape's records that name gate `name` (`*` too for a listener flag)."""
    rounds = _mapping(tape, "gate_rounds")
    keys = (name, ANY_FLAG) if _is_listener(name) else (name,)
    return {str(found) for key in keys for found in rounds.get(key, ())}


def _rests_on(tape, gate, name, listed):
    """Whether the tape sides any of its readings of the gate good or bad."""
    return any(side != "neither" for side, _value, _source in _tape_sided(tape, gate, name, listed))


def verdict_rounds(ordering, gate, name=None):
    """The sorted ledger rounds of the verdicts the gate's good and bad readings rest on ([] for by-ear ranks)."""
    listed = _listed_anywhere(ordering, name)
    if not listed:
        return []
    tapes = [tape for tape in ordering.values() if _rests_on(tape, gate, name, listed)]
    return sorted(set().union(*(_tape_rounds(tape, name) for tape in tapes)))


def gate_metric_values(ordering, gate, name=None):
    """`(good, bad)` readings of the gate across the known-ordering set."""
    sided = sided_values(ordering, gate, name)
    return [value for side, value, _source in sided if side == "good"], [value for side, value, _source in sided if side == "bad"]


def _sources(ordering, gate, name):
    """The verdict sources the good and bad sides of a gate rest on, sorted."""
    return sorted({source for side, _value, source in sided_values(ordering, gate, name) if side != "neither" and source})


def _worst(values, op):
    return max(values) if op == "<=" else min(values)


def _best(values, op):
    return min(values) if op == "<=" else max(values)


def _entry(gate, threshold, source, n_verdicts):
    return {"threshold": float(threshold), "severity": gate.severity, "source": source, "n_verdicts": int(n_verdicts)}


def from_ordering(gate, good, bad, spread):
    """The midpoint between the worst known-good and the best known-bad reading, when the bad side really is worse."""
    if not good or not bad:
        return None
    worst_good, best_bad = _worst(good, gate.op), _best(bad, gate.op)
    gap = (best_bad - worst_good) if gate.op == "<=" else (worst_good - best_bad)
    if gap <= ORDERING_GAP_FLOORS * spread:
        return None
    return _entry(gate, (worst_good + best_bad) / 2.0, KNOWN_ORDERING, len(good) + len(bad))


def from_good_bound(gate, good, spread):
    """The hand-set threshold pushed past the worst known-good reading, so no accepted output is vetoed."""
    if not good:
        return None
    worst_good = _worst(good, gate.op)
    if passes(worst_good, gate.op, float(gate.threshold)):
        return None
    tolerant = worst_good + EFFECT_MULTIPLE * spread if gate.op == "<=" else worst_good - EFFECT_MULTIPLE * spread
    return _entry(gate, tolerant, GOOD_BOUND, len(good))


def from_floor(gate, spread):
    """Three floors past the benign centre, unless the hand-set threshold is already stricter than that."""
    floor_value = EFFECT_MULTIPLE * spread * (1.0 if gate.op == "<=" else -1.0)
    return _entry(gate, floor_value if abs(floor_value) > abs(gate.threshold) else gate.threshold, FLOOR, 0)


def _threshold(gate, good, bad, spread):
    return from_ordering(gate, good, bad, spread) or from_good_bound(gate, good, spread) or from_floor(gate, spread)


def _derivable(gate, spread):
    return isinstance(gate.threshold, (int, float)) and spread is not None


def derive_gate(gate, floor, ordering, name=None):
    """A threshold for one gate (`name` in `GATES`) with its provenance, or None for relative and unmeasured gates."""
    spread = floor.get(gate.metric, {}).get("floor")
    if not _derivable(gate, spread):
        return None
    good, bad = gate_metric_values(ordering, gate, name)
    derived = _threshold(gate, good, bad, spread)
    sources = _sources(ordering, gate, name) if derived["n_verdicts"] else []
    return {**derived, "verdict_source": "+".join(sources) if sources else None}


def gate_family(gate):
    """The harness family that computes the gate's metric (`dsp`, `speech`, `mos`, `stems`, `file`)."""
    spec = METRICS.get(gate.metric)
    return spec.family if spec else gate.metric.split(".", 1)[0]


def _unscored_family(gate, families):
    """The gate's family when this run did not score it, else None (file-level readings come with every family)."""
    family = gate_family(gate)
    if families is None or family == "file":
        return None
    return None if family in families else family


def skip_reason(gate, floor, families=None):
    """Why a gate gets no derived threshold, or None when it can be derived."""
    if not isinstance(gate.threshold, (int, float)):
        return "relative threshold (resolved per tape)"
    family = _unscored_family(gate, families)
    if family:
        return f"family {family} not scored in this run"
    return None if gate.metric in floor else "no benign floor (metric not read on the benign cases)"


def _derived_one(name, gate, floor, ordering, families):
    """`(entry, None)` or `(None, reason)` for one gate."""
    reason = skip_reason(gate, floor, families)
    entry = None if reason else derive_gate(gate, floor, ordering, name)
    if entry is None:
        return None, reason or "not derived"
    return {**entry, "family": gate_family(gate)}, None


def derive_gates(floor, ordering, gates=None, families=None):
    """`(derived, skipped)`: `{name: entry}` for every gate that gets a threshold, `{name: reason}` for the rest."""
    derived, skipped = {}, {}
    for name, gate in (GATES if gates is None else gates).items():
        entry, reason = _derived_one(name, gate, floor, ordering, families)
        if entry:
            derived[name] = entry
        else:
            skipped[name] = reason
    return derived, skipped


def applies_on(gate, route):
    """Whether the gate's metric is read on `route` (file-level and unregistered metrics apply everywhere)."""
    spec = METRICS.get(gate.metric)
    return spec is None or route in spec.routes


def route_floor(route_floors, pooled):
    """The route's own benign floor per metric, the pooled floor where the route has none."""
    return {**pooled, **route_floors}


def derive_route_gates(route, floor, ordering, families=None):
    """`(derived, skipped)` for the gates read on `route`, each entry tagged with the route."""
    gates = {name: gate for name, gate in GATES.items() if applies_on(gate, route)}
    derived, skipped = derive_gates(floor, ordering, gates, families)
    return {name: {**entry, "route": route} for name, entry in derived.items()}, skipped


def _verdict_derived(previous, gate):
    """Whether a previous entry is a verdict-derived threshold resting on known ledger rounds."""
    return gate is not None and bool(previous) and previous.get("source") == KNOWN_ORDERING and bool(previous.get("rounds"))


def _separates(threshold, op, good, bad):
    """Whether `threshold` passes every good reading and fails every bad one."""
    return all(passes(value, op, threshold) for value in good) and not any(passes(value, op, threshold) for value in bad)


def agrees(previous, gate, good, bad):
    """Whether a previous round's verdict-derived threshold still separates this round's good and bad readings."""
    if not _verdict_derived(previous, gate) or not (good and bad):
        return False
    return _separates(float(previous["threshold"]), gate.op, good, bad)


def _new_round(entry, previous):
    """Whether the verdicts behind the gate hold a ledger round the previous entry's did not."""
    return bool(set(entry.get("rounds") or ()) - set(previous.get("rounds") or ()))


def rounds_agreeing(entry, previous, agree):
    """How many separate listening rounds agree on this verdict-derived threshold (0 when it rests on no verdict).

    1 when it disagrees with the previous entry or rests on no ledger round; one more than the
    previous count only when its verdicts hold a round the previous entry's did not.
    """
    if entry["source"] != KNOWN_ORDERING:
        return 0
    if not (agree and entry.get("rounds")):
        return 1
    earlier = int(previous.get("rounds_agreeing", 1))
    return earlier + 1 if _new_round(entry, previous) else earlier


def _promoted(entry, count):
    """The entry promoted from a listener flag to a hard gate once enough rounds agree."""
    if entry["severity"] != gates_mod.FLAG or count < PROMOTE_AFTER_ROUNDS:
        return entry
    return {**entry, "severity": gates_mod.HARD, "promoted_from": gates_mod.FLAG}


def _evidence(ordering, name, entry):
    """`(good, bad, rounds)` of gate `name` across the known-ordering set; all empty for a gate `GATES` does not hold.

    A threshold resting on no verdict (the floor rule) rests on no round either.
    """
    gate = GATES.get(name)
    if gate is None:
        return [], [], []
    good, bad = gate_metric_values(ordering, gate, name)
    return good, bad, verdict_rounds(ordering, gate, name) if entry.get("n_verdicts") else []


def with_rounds(derived, previous, round_id, ordering):
    """`derived` with `round` (the run's id, for display), `rounds`, `rounds_agreeing` and the flag-to-hard promotion."""
    previous = previous or {}
    out = {}
    for name, entry in derived.items():
        good, bad, rounds = _evidence(ordering, name, entry)
        current = {**entry, "round": round_id, "rounds": rounds}
        prior = previous.get(name) or {}
        count = rounds_agreeing(current, prior, agrees(prior, GATES.get(name), good, bad))
        out[name] = _promoted({**current, "rounds_agreeing": count}, count)
    return out


def as_gates(derived, base=None):
    """`Gate` objects for the derived thresholds and severities, on top of the hand-set definitions."""
    base = base or GATES
    fields = {name: {**gates_mod.asdict(base[name]), "threshold": g["threshold"], "severity": g["severity"]} for name, g in derived.items()}
    return {name: gates_mod.Gate(**values) for name, values in fields.items()}
