"""The offline re-judge of stored loop rounds: the four rules, the stored decision reproduced, the vetoes' null simulation."""

import json

import numpy as np
import pytest

from scripts import autotune_judging as judging
from scripts import rejudge_autotune as rj
from scripts.restoration_quality import reward

TILT = "dsp.balance_tilt_db_oct.output.median"
CER = "speech.cer.output.median"
GRID = (
    "ranking:\n"
    f"  {TILT}: {{target: 0.0, dead_zone: 0.3, scale: 0.1, weight: 1.0, family: timbre}}\n"
    "vetoes:\n"
    f"  {CER}: {{worse: up, floor_multiple: 1}}\n"
)
# inc / small (a sub-floor tilt move, accepted by exact ranks) / vetoed_once (a real move, CER past the limit on one tape).
FLATS = {
    "inc": {"t1": {TILT: 2.00, CER: 0.10}, "t2": {TILT: 2.00, CER: 0.10}},
    "small": {"t1": {TILT: 1.98, CER: 0.10}, "t2": {TILT: 1.98, CER: 0.10}},
    "vetoed_once": {"t1": {TILT: 0.20, CER: 0.10}, "t2": {TILT: 0.20, CER: 0.20}},
}


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _loop(tmp_path):
    """A stored loop: round 1 accepted `small` (a set-aside and an unscored candidate beside it), round 2 lost its incumbent."""
    loop = tmp_path / "autotune" / "apl"
    for cid, tapes in FLATS.items():
        for slug, flat in tapes.items():
            _write(loop / "cands" / cid / f"{slug}.score.json", {**flat, "gates.hard_failures": 0.0})
    _write(loop / "cands" / "vetoed_once" / "t2.learned.score.json", {CER: 0.20})
    _write(loop / "cands" / "gone" / "t1.score.json", {TILT: 1.0})
    per_tape = {"t1": 1.0, "t2": 1.0}
    verdicts = {cid: {"per_tape": per_tape} for cid in [*FLATS, "gone"]}
    verdicts["inert1"] = {"per_tape": {}, "inert": "inc"}
    rounds = [
        {"round": 1, "incumbent": "inc", "accepted": "small", "verdicts": verdicts},
        {"round": 2, "incumbent": "small2", "accepted": None, "verdicts": {"small2": {"per_tape": per_tape}}},
    ]
    _write(loop / rj.STATE, {"incumbent": {}, "rounds": rounds, "candidates": {}})
    return loop


def _reports(tmp_path):
    benign = tmp_path / "benign.json"
    _write(benign, {"operating_points": ["benign_pair"], "readings": {TILT: {"floor": 0.85}, CER: {"floor": 0.038}}})
    jitter = tmp_path / "jitter.json"
    _write(jitter, {"operating_points": ["output"], "readings": {CER: {"floor": 0.001}}})
    grid = tmp_path / "grid.yaml"
    grid.write_text(GRID, encoding="utf-8")
    return ["--grid", str(grid), "--noise-floors", str(benign), "--jitter-floors", str(jitter)]


@pytest.fixture(name="rejudged")
def _rejudged(tmp_path, capsys):
    out = tmp_path / "rejudge.json"
    reports = rj.main([str(_loop(tmp_path)), *_reports(tmp_path), "--null-draws", "40", "--json", str(out)])
    return reports, capsys.readouterr().out, json.loads(out.read_text(encoding="utf-8"))


def test_the_old_rule_reproduces_the_stored_decision_and_the_new_ones_change_it(rejudged):
    """Exact ranks accept the sub-floor move; the band ties it; the family-wise veto lets the real move through."""
    reports, _out, _json = rejudged
    rules = reports[0]["rounds"][0]["rules"]
    accepted = {rule: rules[rule]["accepted"] for rule in rj.RULES}
    assert accepted == {"old": "small", "band": None, "familywise": "vetoed_once", "new": "vetoed_once"}
    assert reports[0]["rounds"][0]["stored"] == "small"


def test_the_band_makes_the_sub_floor_move_level_and_the_family_wise_rule_drops_the_one_tape_veto(rejudged):
    reports, _out, _json = rejudged
    rules = reports[0]["rounds"][0]["rules"]
    assert list(rules["old"]["vetoed"]) == ["vetoed_once"]
    assert rules["new"]["vetoed"] == {}
    assert rules["band"]["level"] == ["small"]


def test_set_aside_and_unscored_candidates_are_left_out_and_named(rejudged):
    reports, _out, _json = rejudged
    first = reports[0]["rounds"][0]
    assert first["n_scored"] == 3
    assert first["missing"] == ["gone"]


def test_a_round_whose_incumbent_was_not_scored_is_skipped(rejudged):
    reports, out, _json = rejudged
    assert reports[0]["rounds"][1]["rules"] == {}
    assert "apl round 2: incumbent not scored, skipped" in out


def test_one_line_per_round_and_per_null_reaches_the_console_and_the_report_the_json(rejudged):
    reports, out, written = rejudged
    assert "apl round 1 (stored small, 3 scored): old small, band -, familywise vetoed_once, new vetoed_once" in out
    assert "changed under ['band', 'familywise', 'new']" in out
    assert "apl round 1 null: jitter any-tape" in out
    assert written[0]["rounds"][0]["rules"]["new"]["accepted"] == reports[0]["rounds"][0]["rules"]["new"]["accepted"]


def test_the_null_simulation_rates_both_rules_on_every_null(rejudged):
    reports, _out, _json = rejudged
    null = reports[0]["rounds"][0]["null"]
    assert set(null) == set(rj.NULLS)
    assert null["jitter"]["n"] == 40
    assert null["signflip"]["n"] == 40
    assert all(0.0 <= rates[rule] <= 1.0 for rates in null.values() for rule in ("any_tape", "familywise"))


def test_without_a_json_path_or_null_draws_nothing_is_written_and_nothing_simulated(tmp_path, capsys):
    """The console lines alone; the round carries an empty null."""
    reports = rj.main([str(_loop(tmp_path)), *_reports(tmp_path)])
    assert reports[0]["rounds"][0]["null"] == {}
    assert "null:" not in capsys.readouterr().out
    assert sorted(path.name for path in tmp_path.iterdir()) == ["autotune", "benign.json", "grid.yaml", "jitter.json"]


def test_a_round_without_a_veto_reading_has_no_null():
    """A grid's veto reading the incumbent never read cannot veto, so it has nothing to simulate."""
    context = {"guards": rj.guards_mod.Guards(vetoes={CER: reward.Veto("up", 0.04)}), "floors": judging.NO_FLOORS, "draws": 5}
    scores = {"inc": {"t1": {TILT: 1.0}}, "a": {"t1": {TILT: 1.1}}}
    assert rj.null_rates(scores, "inc", context) == {}


def test_a_round_without_a_winner_is_labelled_and_a_rate_over_nothing_is_none():
    verdicts = {"inc": {"mean_rank": 1.0, "qualifies": False, "wins": 0, "level": True, "status": {}}}
    assert rj.summarise(verdicts, "inc", judging.NO_FLOORS)["label"] == judging.PLATEAU
    assert rj._rate([], {}, {}, judging.NO_FLOORS, rj._familywise) is None


def test_sign_flips_keep_each_tapes_delta_and_give_it_one_sign():
    rng = np.random.default_rng(3)
    incumbent = {"t1": {CER: 0.1, TILT: 1.0}}
    candidate = {"t1": {CER: 0.3, TILT: 1.5}}
    flipped = [rj._flipped(candidate, incumbent, [CER, TILT], rng)["t1"] for _ in range(20)]
    assert {(round(f[CER], 6), round(f[TILT], 6)) for f in flipped} == {(0.3, 1.5), (-0.1, 0.5)}
