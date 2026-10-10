"""Gate derivation: thresholds from the benign floor and the verdicts, their provenance, per route, and the round count."""

from scripts import calibration_gates as cg
from scripts.restoration_quality.gates import FLAG, GATES, HARD, Gate

KNOWN_GOOD = ("cathar075", "stitched", "apl")
KNOWN_BAD = ("deesser_bug", "single4s")
HF = Gate("dsp.hf_4k8k", "delta.median", ">=", -8.0)
FLOOR = {"dsp.hf_4k8k": {"centre": 0.0, "floor": 0.5}}


def _verdict(metric, value, stat="delta.median", gate="g", status="passed"):
    return {"gate": gate, "metric": metric, "stat": stat, "value": value, "status": status}


def _ranked(good_values, bad_values, metric="dsp.hf_4k8k"):
    """One tape sided by its by-ear ranks: the known-good labels ranked 1, the known-bad ones 5."""
    variants = {label: {"verdicts": [_verdict(metric, v)]} for label, v in zip(KNOWN_GOOD, good_values)}
    variants.update({label: {"verdicts": [_verdict(metric, v)]} for label, v in zip(KNOWN_BAD, bad_values)})
    ranks = {**{label: 1 for label in KNOWN_GOOD}, **{label: 5 for label in KNOWN_BAD}}
    return {"tele": {"result": {"variants": variants}, "ranks": ranks}}


def test_a_clear_known_ordering_gap_sets_the_midpoint_and_counts_its_verdicts():
    """A clear known ordering gap sets the midpoint and counts its verdicts."""
    derived = cg.derive_gate(HF, FLOOR, _ranked([-3.0, -4.0, -1.0], [-30.0, -7.0]))
    assert derived["source"] == cg.KNOWN_ORDERING
    assert derived["threshold"] == -5.5
    assert (derived["n_verdicts"], derived["verdict_source"]) == (5, cg.BY_RANK)


def test_without_verdicts_the_floor_rule_stands_and_says_so():
    """Without verdicts the floor rule stands and says so."""
    assert cg.derive_gate(HF, FLOOR, {}) == {
        "threshold": -8.0,
        "severity": "hard",
        "source": "floor",
        "n_verdicts": 0,
        "verdict_source": None,
    }
    assert cg.derive_gate(Gate("dsp.hf_4k8k", "delta.median", ">=", -1.0), FLOOR, {})["threshold"] == -1.5


def test_relative_and_unmeasured_gates_get_no_threshold():
    """Relative and unmeasured gates get no threshold."""
    assert cg.derive_gate(Gate("speech.speaker_cos", "tail", ">=", "floor-0.05"), FLOOR, {}) is None
    assert cg.derive_gate(Gate("mos.sigmos_col", "delta.median", ">=", -0.3), FLOOR, {}) is None


def test_a_hand_set_threshold_never_vetoes_a_known_good_output():
    """A hand set threshold never vetoes a known good output."""
    gate = Gate("dsp.lkr", "delta.median", "<=", 0.3)
    derived = cg.derive_gate(gate, {"dsp.lkr": {"centre": 0.0, "floor": 0.1}}, _ranked([0.6, 1.1, 0.8], [0.9, 1.2], "dsp.lkr"))
    assert derived["source"] == cg.GOOD_BOUND
    assert abs(derived["threshold"] - 1.4) < 1e-9
    assert derived["n_verdicts"] == 3


def _listed(bad_value, good_value, lists=None, rounds=None):
    """One tape read by flag lists for `listener.dead_air`: `bad` flagged, `good` clean, `spare` ranked 1 but on no list.

    `rounds` are the ledger rounds whose records name `listener.dead_air` (none: the manifest's lists).
    """
    variants = {
        label: {"verdicts": [_verdict("dsp.gap_air_db", value, "median", "listener.dead_air")]}
        for label, value in (("bad", bad_value), ("good", good_value), ("spare", -99.0))
    }
    lists = lists or {"flags": {"listener.dead_air": ["bad"]}, "clean": {"listener.dead_air": ["good"]}}
    gate_rounds = {"listener.dead_air": list(rounds)} if rounds else {}
    return {
        "tele": {"result": {"variants": variants}, "ranks": {"spare": 1}, "verdict_source": "ledger", "gate_rounds": gate_rounds, **lists}
    }


DEAD = Gate("dsp.gap_air_db", "median", ">=", -25.0, severity=FLAG)


def test_the_flag_lists_side_their_own_gate_and_ranks_side_the_rest():
    """The flag lists side their own gate and ranks side the rest."""
    assert cg.gate_metric_values(_listed(-33.0, -11.0), DEAD, "listener.dead_air") == ([-11.0], [-33.0])
    assert cg.gate_metric_values(_listed(-33.0, -11.0), DEAD, "listener.hiss") == ([-99.0], [])
    assert cg.gate_metric_values(_listed(-33.0, -11.0), DEAD) == ([-99.0], [])


def test_accepted_with_no_complaint_counts_as_clean_for_every_listener_flag():
    """Accepted with no complaint counts as clean for every listener flag."""
    lists = {"flags": {}, "clean": {"*": ["good"]}}
    assert cg.gate_metric_values(_listed(-33.0, -11.0, lists), DEAD, "listener.pause_collapse") == ([-11.0], [])
    assert cg.lists_gate(lists, "listener.hiss")
    assert not cg.lists_gate(lists, "dsp.hf_4k8k")


def test_a_listed_gate_records_where_its_verdicts_came_from():
    """A listed gate records where its verdicts came from."""
    derived = cg.derive_gate(DEAD, {"dsp.gap_air_db": {"centre": 0.0, "floor": 1.0}}, _listed(-33.0, -11.0), "listener.dead_air")
    assert (derived["threshold"], derived["severity"], derived["source"]) == (-22.0, FLAG, cg.KNOWN_ORDERING)
    assert (derived["n_verdicts"], derived["verdict_source"]) == (2, "ledger")


def test_an_empty_list_still_marks_the_gate_judged():
    """An empty list still marks the gate judged."""
    tape = {"flags": {"listener.sibilance_dull": []}, "clean": {}}
    assert cg.lists_gate(tape, "listener.sibilance_dull")
    assert not cg.lists_gate(tape, "listener.hiss")


def test_gates_whose_family_did_not_score_are_skipped_with_the_reason():
    """Gates whose family did not score are skipped with the reason."""
    derived, skipped = cg.derive_gates(FLOOR, {}, families=["dsp"])
    assert "dsp.hf_4k8k" in derived
    assert derived["dsp.hf_4k8k"]["family"] == "dsp"
    assert skipped["mos.sigmos_col"] == "family mos not scored in this run"
    assert skipped["speech.speaker"].startswith("relative threshold")


def test_a_scored_family_without_a_benign_floor_is_named_too():
    """A scored family without a benign floor is named too."""
    _derived, skipped = cg.derive_gates({}, {}, families=["dsp"])
    assert skipped["dsp.hf_4k8k"].startswith("no benign floor")
    assert cg.skip_reason(Gate("file.lra", "median", ">=", -3.0), {}, ["dsp"]).startswith("no benign floor")


def test_route_gates_keep_the_gates_read_on_that_route():
    """Route gates keep the gates read on that route."""
    derived, _skipped = cg.derive_route_gates("music", {**FLOOR, "dsp.attack_db": {"centre": 0.0, "floor": 0.1}}, {}, ["dsp"])
    assert "dsp.attack_db" in [GATES[name].metric for name in derived]
    assert "dsp.hf_4k8k" not in derived
    assert {entry["route"] for entry in derived.values()} == {"music"}


def test_a_route_floor_overrides_the_pooled_one_metric_by_metric():
    """A route floor overrides the pooled one metric by metric."""
    merged = cg.route_floor({"a": {"floor": 1.0}}, {"a": {"floor": 2.0}, "b": {"floor": 3.0}})
    assert merged == {"a": {"floor": 1.0}, "b": {"floor": 3.0}}


def test_a_previous_threshold_agrees_when_it_still_separates_the_verdicts():
    """A previous threshold agrees when it still separates the verdicts."""
    previous = {"threshold": -22.0, "source": cg.KNOWN_ORDERING, "rounds": ["1"]}
    assert cg.agrees(previous, DEAD, [-11.0], [-33.0])
    assert not cg.agrees({**previous, "threshold": -40.0}, DEAD, [-11.0], [-33.0])
    assert not cg.agrees({**previous, "source": cg.FLOOR}, DEAD, [-11.0], [-33.0])


def test_rounds_count_only_new_ledger_rounds_behind_an_agreeing_verdict_threshold():
    """A round counts when the gate's verdicts hold a ledger round the previous entry's did not."""
    entry = {"source": cg.KNOWN_ORDERING, "rounds": ["1", "2"]}
    assert cg.rounds_agreeing(entry, {"rounds_agreeing": 1, "rounds": ["1"]}, True) == 2
    assert cg.rounds_agreeing(entry, {"rounds_agreeing": 2, "rounds": ["1", "2"]}, True) == 2
    assert cg.rounds_agreeing(entry, {"rounds_agreeing": 2, "rounds": ["1"]}, False) == 1
    assert cg.rounds_agreeing({"source": cg.FLOOR}, {}, True) == 0


def test_a_gate_resting_on_no_ledger_round_counts_one():
    """Manifest lists and by-ear ranks name no round: they never add one, whatever the previous entry held."""
    entry = {"source": cg.KNOWN_ORDERING, "rounds": []}
    assert cg.rounds_agreeing(entry, {"rounds_agreeing": 2, "rounds": ["1"]}, True) == 1


DEAD_AIR_FLOOR = {"dsp.gap_air_db": {"centre": 0.0, "floor": 1.0}}


def _dead_air(ordering, previous=None, round_id="run"):
    """The derived `listener.dead_air` entry of one calibration, its rounds read against `previous`."""
    derived, _skipped = cg.derive_gates(DEAD_AIR_FLOOR, ordering, {"listener.dead_air": GATES["listener.dead_air"]})
    return cg.with_rounds(derived, {"listener.dead_air": previous} if previous else None, round_id, ordering)["listener.dead_air"]


def _dead_air_derived(previous_rounds, threshold=-22.0, rounds=("1", "2")):
    previous = {"threshold": threshold, "source": cg.KNOWN_ORDERING, "rounds": previous_rounds, "rounds_agreeing": 1}
    return _dead_air(_listed(-33.0, -11.0, rounds=rounds), previous)


def test_a_flag_turns_hard_once_two_rounds_agree():
    """A new round that names the gate and agrees with the previous threshold promotes the flag."""
    promoted = _dead_air_derived(["1"])
    assert (promoted["severity"], promoted["promoted_from"], promoted["rounds_agreeing"]) == (HARD, FLAG, 2)
    assert (promoted["round"], promoted["rounds"]) == ("run", ["1", "2"])


def test_a_rerun_or_a_disagreement_keeps_the_flag():
    """A rerun or a disagreement keeps the flag."""
    assert _dead_air_derived(["1", "2"])["severity"] == FLAG
    assert _dead_air_derived(["1"], threshold=-40.0)["severity"] == FLAG


def _bright_tape(round_id):
    """A second tape whose only verdict is a `listener.bright` flag from `round_id`; it reads dead air too."""
    variants = {"x": {"verdicts": [_verdict("dsp.gap_air_db", -50.0, "median", "listener.dead_air")]}}
    lists = {"flags": {"listener.bright": ["x"]}, "clean": {}, "gate_rounds": {"listener.bright": [round_id]}}
    return {"result": {"variants": variants}, "ranks": {}, "verdict_source": "ledger", **lists}


def test_a_new_round_about_another_gate_on_another_tape_does_not_promote():
    """The run's id turns from 1 to 1+4, yet dead air still rests on round 1 alone: no second round agrees."""
    first = _dead_air(_listed(-33.0, -11.0, rounds=("1",)), round_id="1")
    second = _dead_air({**_listed(-33.0, -11.0, rounds=("1",)), "soti": _bright_tape("4")}, first, "1+4")
    assert (first["rounds"], first["rounds_agreeing"]) == (["1"], 1)
    assert (second["severity"], second["rounds"], second["rounds_agreeing"]) == (FLAG, ["1"], 1)
    assert second["round"] == "1+4"
    assert "promoted_from" not in second


def test_a_manifest_run_then_a_ledger_run_on_the_same_verdicts_does_not_promote():
    """`--no-ledger` reads the manifest's lists (no round); the ledger run after it has no counted round to agree with."""
    manifest = _dead_air(_listed(-33.0, -11.0), round_id="manifest")
    ledger = _dead_air(_listed(-33.0, -11.0, rounds=("1",)), manifest, "1")
    assert (manifest["rounds"], manifest["rounds_agreeing"]) == ([], 1)
    assert (ledger["severity"], ledger["rounds_agreeing"]) == (FLAG, 1)


def test_verdict_rounds_come_from_the_tapes_that_side_the_gate():
    """The tapes whose lists side a reading count their rounds (`*` for a listener flag); by-ear ranks count none."""
    ordering = {**_listed(-33.0, -11.0, rounds=("1",)), "soti": _bright_tape("4")}
    ordering["tele"]["gate_rounds"]["*"] = ["3"]
    assert cg.verdict_rounds(ordering, DEAD, "listener.dead_air") == ["1", "3"]
    assert cg.verdict_rounds(ordering, DEAD, "dsp.gap_air") == []


def test_derived_entries_become_gates_over_the_hand_set_definitions():
    """Derived entries become gates over the hand set definitions."""
    gates = cg.as_gates({"listener.dead_air": {"threshold": -22.0, "severity": HARD}})
    assert gates["listener.dead_air"].threshold == -22.0
    assert gates["listener.dead_air"].severity == HARD
    assert gates["listener.dead_air"].metric == GATES["listener.dead_air"].metric


def test_passes_reads_every_operator():
    """Passes reads every operator."""
    assert cg.passes(1.0, "<=", 1.0)
    assert cg.passes(1.0, ">=", 1.0)
    assert cg.passes(-1.0, "abs<=", 1.5)
    assert not cg.passes(-2.0, "abs<=", 1.5)


def test_a_previous_entry_without_a_round_never_counts_as_an_agreeing_round():
    """The v2 gates name no round: they cannot promote a flag on verdicts both rounds share."""
    assert not cg.agrees({"threshold": -22.0, "source": cg.KNOWN_ORDERING, "round": "1"}, DEAD, [-11.0], [-33.0])
    assert _dead_air_derived(None)["rounds_agreeing"] == 1


def test_a_gate_the_table_does_not_hold_rests_on_no_verdict():
    """An entry named outside `GATES` has no readings to side and no rounds."""
    assert cg.with_rounds({"x.unknown": {"source": cg.FLOOR, "severity": FLAG}}, None, "1", {})["x.unknown"]["rounds"] == []


def test_a_floor_threshold_rests_on_no_round():
    """The floor rule takes no verdict, so the rounds that side the gate's readings are not its own."""
    floor_entry = {"listener.dead_air": {"source": cg.FLOOR, "severity": FLAG, "n_verdicts": 0}}
    ordering = _listed(-33.0, -11.0, rounds=("1",))
    assert cg.with_rounds(floor_entry, None, "1", ordering)["listener.dead_air"]["rounds"] == []
