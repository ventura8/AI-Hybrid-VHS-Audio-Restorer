"""The calibration's pure logic: floors, sensitivity checks, ordering rules and gate derivation."""

from pathlib import Path

from scripts import calibrate_quality_metrics as cal
from scripts import quality_degradations as deg
from scripts.restoration_quality.gates import Gate


def _case(kind, name, level, language="en"):
    return cal.Case(kind, name, level, language, Path("s.wav"), Path("o.wav"))


def _benign_cases_and_scores():
    cases = [_case("benign", name, None) for name in deg.BENIGN]
    scores = {c.case_id: {"dsp.hf_4k8k": 0.1 * i - 0.25, "speech.cer": 0.02} for i, c in enumerate(cases)}
    return cases, scores


def test_noise_floor_centres_and_spreads_each_metric():
    cases, scores = _benign_cases_and_scores()
    floor = cal.noise_floor(cases, scores)
    assert abs(floor["dsp.hf_4k8k"]["centre"]) < 1e-9
    assert 0.2 < floor["dsp.hf_4k8k"]["floor"] <= 0.25
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
