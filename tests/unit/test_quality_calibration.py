"""The calibration's pure logic: floors, sensitivity checks, ordering rules and gate derivation."""

import functools
import json
from pathlib import Path

import pytest

from scripts import calibrate_quality_metrics as cal
from scripts import quality_degradations as deg
from scripts.restoration_quality.gates import GATES, Gate


def _case(kind, name, level, language="en"):
    return cal.Case(kind, name, level, language, Path("s.wav"), Path("o.wav"))


def _benign_cases_and_scores():
    """One case per benign name; the hf_4k8k deltas sit on a symmetric 0.1 grid, so the centre is 0 and the floor its edge."""
    cases = [_case("benign", name, None) for name in deg.BENIGN]
    half = (len(cases) - 1) / 2.0
    scores = {c.case_id: {"dsp.hf_4k8k": 0.1 * (i - half), "speech.cer": 0.02} for i, c in enumerate(cases)}
    return cases, scores, 0.1 * half


def test_noise_floor_centres_and_spreads_each_metric():
    cases, scores, edge = _benign_cases_and_scores()
    floor = cal.noise_floor(cases, scores)
    assert abs(floor["dsp.hf_4k8k"]["centre"]) < 1e-9
    assert abs(floor["dsp.hf_4k8k"]["floor"] - edge) < 1e-9
    assert floor["speech.cer"]["floor"] == 0.0


def _underwater_cases(values):
    cases = [_case("degradation", "underwater", level) for level in deg.DEGRADATIONS["underwater"].levels]
    scores = {c.case_id: {"dsp.hf_4k8k": v} for c, v in zip(cases, values)}
    return cases, scores


def test_check_expectation_passes_a_clean_monotonic_response():
    cases, scores = _underwater_cases([-3.0, -8.0, -20.0])
    floor = {"dsp.hf_4k8k": {"centre": 0.0, "floor": 0.5}}
    result = cal.check_expectation("underwater", deg.Expectation("dsp.hf_4k8k", "down"), cases, scores, floor)
    assert result["status"] == "pass"
    assert result["monotonic_share"] == 1.0
    assert abs(result["effect_mild"] - 6.0) < 1e-6


def test_check_expectation_fails_on_wrong_direction_or_weak_effect():
    cases, scores = _underwater_cases([-0.5, -8.0, -20.0])
    floor = {"dsp.hf_4k8k": {"centre": 0.0, "floor": 0.5}}
    weak = cal.check_expectation("underwater", deg.Expectation("dsp.hf_4k8k", "down"), cases, scores, floor)
    assert weak["status"] == "fail"
    wrong = cal.check_expectation("underwater", deg.Expectation("dsp.hf_4k8k", "up"), cases, scores, floor)
    assert wrong["direction"] is False


def test_check_expectation_reports_unscored_metrics():
    cases, scores = _underwater_cases([-3.0, -8.0, -20.0])
    result = cal.check_expectation("underwater", deg.Expectation("mos.sigmos_col", "down"), cases, scores, {})
    assert result["status"] == "unscored"


def _variant(failures, composite, extra=None):
    verdicts = [{"gate": g, "metric": "m", "stat": "median", "value": 1.0, "status": "failed", "severity": "hard"} for g in failures]
    aggregate = {metric: {"delta": {"median": composite}} for metric in cal.COMPOSITE}
    aggregate.update(extra or {})
    return {"hard_failures": list(failures), "verdicts": verdicts, "aggregate": aggregate}


def test_ordering_rules_reproduce_the_by_ear_verdicts():
    result = {
        "variants": {
            "deesser_bug": _variant(["dsp.hf_4k8k"], -3.0, {"dsp.hf_4k8k": {"delta": {"median": -30.0}}}),
            "single4s": _variant([], -1.0, {"dsp.hf_4k8k": {"delta": {"median": -7.0}}}),
            "stitched": _variant([], 0.5, {"dsp.hf_4k8k": {"delta": {"median": -4.0}}}),
            "apl": _variant([], 1.0),
            "cathar075": _variant([], 0.4),
        }
    }
    rules = cal.ordering_rules(result, {})
    assert rules["bug_flagged_muffled"] and rules["bug_in_bottom_two"]
    assert rules["single4s_duller_than_stitched"]
    assert rules["apl_not_altered"] and rules["good_in_top"]


def test_ranking_orders_by_failures_then_composite():
    result = {"variants": {"a": _variant([], 0.0), "b": _variant(["x"], 5.0), "c": _variant([], 2.0)}}
    assert cal.harness_ranking(result, {}) == ["c", "a", "b"]


def _ordering(good_values, bad_values):
    def tape(label, value):
        return {label: {"verdicts": [{"metric": "dsp.hf_4k8k", "stat": "delta.median", "value": value}]}}

    variants = {}
    for label, value in zip(cal.KNOWN_GOOD, good_values):
        variants.update(tape(label, value))
    for label, value in zip(cal.KNOWN_BAD, bad_values):
        variants.update(tape(label, value))
    ranks = {**{label: 1 for label in cal.KNOWN_GOOD}, **{label: 5 for label in cal.KNOWN_BAD}}
    return {"tele": {"result": {"variants": variants}, "ranks": ranks}}


def test_derive_gate_prefers_a_clear_known_ordering_gap():
    gate = Gate("dsp.hf_4k8k", "delta.median", ">=", -8.0)
    floor = {"dsp.hf_4k8k": {"centre": 0.0, "floor": 0.5}}
    derived = cal.derive_gate(gate, floor, _ordering([-3.0, -4.0, -1.0], [-30.0, -7.0]))
    assert derived["source"] == "known ordering"
    assert derived["threshold"] == -5.5


def test_derive_gate_falls_back_to_the_floor_and_skips_relative_gates():
    gate = Gate("dsp.hf_4k8k", "delta.median", ">=", -8.0)
    floor = {"dsp.hf_4k8k": {"centre": 0.0, "floor": 0.5}}
    assert cal.derive_gate(gate, floor, {}) == {"threshold": -8.0, "severity": "hard", "source": "floor"}
    loose = cal.derive_gate(Gate("dsp.hf_4k8k", "delta.median", ">=", -1.0), floor, {})
    assert loose["threshold"] == -1.5
    assert cal.derive_gate(Gate("speech.speaker_cos", "tail", ">=", "floor-0.05"), floor, {}) is None


def test_derive_gate_never_vetoes_a_known_good_output():
    gate = Gate("dsp.lkr", "delta.median", "<=", 0.3)
    floor = {"dsp.lkr": {"centre": 0.0, "floor": 0.1}}
    ordering = _ordering([0.6, 1.1, 0.8], [0.9, 1.2])
    for tape in ordering.values():
        for variant in tape["result"]["variants"].values():
            variant["verdicts"][0]["metric"] = "dsp.lkr"
    derived = cal.derive_gate(gate, floor, ordering)
    assert derived["source"] == "known good bound"
    assert abs(derived["threshold"] - 1.4) < 1e-9


def _flagged_ordering(bad_value, good_value):
    """One tape read by flag lists for `listener.dead_air`: `bad` flagged, `good` clean, and `spare` ranked 1 but on neither list."""

    def variant(value):
        return {"verdicts": [{"gate": "listener.dead_air", "metric": "dsp.gap_air_db", "stat": "median", "value": value}]}

    variants = {"bad": variant(bad_value), "good": variant(good_value), "spare": variant(-99.0)}
    lists = {"flags": {"listener.dead_air": ["bad"]}, "clean": {"listener.dead_air": ["good"]}}
    return {"tele": {"result": {"variants": variants}, "ranks": {"spare": 1}, **lists}}


def test_gate_metric_values_follow_the_flag_lists_when_the_tape_has_them():
    gate = Gate("dsp.gap_air_db", "median", ">=", -25.0, severity="flag")
    assert cal._gate_metric_values(_flagged_ordering(-33.0, -11.0), gate, "listener.dead_air") == ([-11.0], [-33.0])
    # Another gate has no lists on this tape: the by-ear ranks decide, and only `spare` is ranked.
    assert cal._gate_metric_values(_flagged_ordering(-33.0, -11.0), gate, "listener.hiss") == ([-99.0], [])
    assert cal._gate_metric_values(_flagged_ordering(-33.0, -11.0), gate) == ([-99.0], [])


def test_derive_gate_uses_the_flag_lists_of_its_own_gate():
    gate = Gate("dsp.gap_air_db", "median", ">=", -25.0, severity="flag")
    floor = {"dsp.gap_air_db": {"centre": 0.0, "floor": 1.0}}
    derived = cal.derive_gate(gate, floor, _flagged_ordering(-33.0, -11.0), "listener.dead_air")
    assert derived == {"threshold": -22.0, "severity": "flag", "source": "known ordering"}


def _listened(statuses):
    """Variants with one `listener.hiss` verdict each, at the given status."""
    verdict = {"metric": "dsp.gap_air_db", "stat": "median", "value": 0.0, "severity": "flag"}
    return {
        label: {"hard_failures": [], "verdicts": [{**verdict, "gate": "listener.hiss", "status": status}], "aggregate": {}}
        for label, status in statuses.items()
    }


def test_flags_reproduced_needs_every_flagged_label_failing_and_every_clean_one_passing():
    flags, clean = {"listener.hiss": ["hissy"]}, {"listener.hiss": ["quiet"]}
    reproduced = {"variants": _listened({"hissy": "failed", "quiet": "passed"})}
    assert cal.ordering_rules(reproduced, {}, flags, clean)["flags_reproduced"] is True
    clean_failed = {"variants": _listened({"hissy": "failed", "quiet": "failed"})}
    assert cal.ordering_rules(clean_failed, {}, flags, clean)["flags_reproduced"] is False
    flag_missed = {"variants": _listened({"hissy": "passed", "quiet": "passed"})}
    assert cal.ordering_rules(flag_missed, {}, flags, clean)["flags_reproduced"] is False


def test_flags_reproduced_is_unread_without_lists_or_with_labels_the_tape_lacks():
    flag_missed = {"variants": _listened({"hissy": "passed", "quiet": "passed"})}
    assert cal.ordering_rules(flag_missed, {})["flags_reproduced"] is None
    assert cal.ordering_rules(flag_missed, {}, {"listener.hiss": ["absent"]}, {})["flags_reproduced"] is None


def _aggregate(gap_air_db):
    return {"aggregate": {"dsp.gap_air_db": {"output": {"median": gap_air_db}}}}


def test_reevaluate_ordering_reads_the_flags_under_the_derived_gates():
    variants = {"dead": _aggregate(-33.0), "live": _aggregate(-11.0)}
    lists = {"flags": {"listener.dead_air": ["dead"]}, "clean": {"listener.dead_air": ["live"]}}
    ordering = {"tele": {"result": {"variants": variants}, "ranks": {}, **lists}}
    strict = {"listener.dead_air": Gate("dsp.gap_air_db", "median", ">=", -25.0, severity="flag")}
    assert cal.reevaluate_ordering(ordering, strict, {})["tele"]["rules"]["flags_reproduced"] is True
    loose = {"listener.dead_air": Gate("dsp.gap_air_db", "median", ">=", -40.0, severity="flag")}
    after = cal.reevaluate_ordering(ordering, loose, {})["tele"]
    assert after["rules"]["flags_reproduced"] is False
    assert after["flags"] == lists["flags"] and after["clean"] == lists["clean"]


REPO = Path(cal.__file__).resolve().parent.parent
LISTEN_GATES = ("listener.dead_air", "listener.pause_collapse", "listener.hiss", "listener.sibilance_thin", "listener.sibilance_dull")


@functools.lru_cache(maxsize=None)
def _manifest(name):
    return json.loads((REPO / "experiments" / "quality_calibration" / name).read_text(encoding="utf-8"))


def _listen():
    """The listened Tele7abc tape of the v2 manifest and its label set."""
    listen = _manifest("known_ordering_v2.json")["tele7abc_listen"]
    return listen, set(listen["variants"])


def test_v2_manifest_keeps_the_old_tapes():
    v1, v2 = _manifest("known_ordering.json"), _manifest("known_ordering_v2.json")
    assert {tape: v2[tape] for tape in v1} == v1


def test_the_listened_tele7abc_has_thirteen_variants_of_a_mov_with_lists_for_every_listen_gate():
    listen, labels = _listen()
    assert len(labels) == 13
    assert listen["source"].endswith(".mov")
    assert set(listen["flags"]) == set(listen["clean"]) == set(LISTEN_GATES)


@pytest.mark.parametrize("gate", LISTEN_GATES)
def test_each_listen_gate_is_a_flag_whose_lists_name_disjoint_listened_labels(gate):
    listen, labels = _listen()
    assert GATES[gate].severity == "flag"
    assert set(listen["flags"][gate]) | set(listen["clean"][gate]) <= labels
    assert not set(listen["flags"][gate]) & set(listen["clean"][gate])


def test_the_by_ear_ranks_name_listened_labels_and_leave_the_finals_unranked():
    listen, labels = _listen()
    assert set(listen["by_ear_rank"]) <= labels
    assert listen["by_ear_rank"]["cathar075__alpha_2_0"] == 1
    assert {"final_apl", "final_cathar"} <= labels - set(listen["by_ear_rank"])
