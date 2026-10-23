"""The loop judged with its floors: the tie band in the verdict, the round's distances and label, the command line."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import autotune_guards as ag
from scripts import autotune_judging as judging
from scripts import autotune_restoration as at

TILT = "dsp.balance_tilt_db_oct.output.median"
GAP = "dsp.gap_lsd_db.output.median"
RANKING = {
    TILT: {"target": 0.0, "dead_zone": 0.3, "scale": 0.1, "weight": 1.0, "family": "timbre"},
    GAP: {"target": 0.0, "dead_zone": 0.0, "scale": 0.44, "weight": 1.0, "family": "pauses"},
}
FLOORS = judging.Floors({TILT: 0.85, GAP: 1.01}, {TILT: 0.0003, GAP: 0.048})
TAPES = ("t1", "t2")


def _flat(tilt, gap):
    return {TILT: tilt, GAP: gap, "gates.hard_failures": 0.0}


def _scores(tilt_by_cid, gap=1.0):
    return {cid: {slug: _flat(tilt, gap) for slug in TAPES} for cid, tilt in tilt_by_cid.items()}


def _sub_floor_scores():
    """Tilt 0.62 -> 0.60 dB/oct on both tapes, a move under the 0.85 floor."""
    return _scores({"inc": 0.62, "small": 0.60})


def test_a_sub_floor_move_ties_and_never_qualifies_with_the_floors():
    """Tilt 0.62 -> 0.60 dB/oct, the 0.85 floor: with the floors a tie, level, no win."""
    banded = at.judge(_sub_floor_scores(), "inc", RANKING, floors=FLOORS)
    assert banded["small"]["level"]
    assert banded["small"]["wins"] == 0
    assert not banded["small"]["qualifies"]


def test_a_sub_floor_move_won_both_tapes_where_exact_ranks_accepted_it():
    """The same 0.62 -> 0.60 dB/oct move on exact ranks: it won both tapes and qualified."""
    exact = at.judge(_sub_floor_scores(), "inc", RANKING)
    assert exact["small"]["wins"] == 2
    assert exact["small"]["qualifies"]


def test_a_move_past_the_floor_still_wins():
    scores = _scores({"inc": 2.00, "better": 0.50})
    verdicts = at.judge(scores, "inc", RANKING, floors=FLOORS)
    assert verdicts["better"]["qualifies"]
    assert verdicts["better"]["moved"] == {TILT: list(TAPES)}
    assert verdicts["better"]["status"] == {TILT: judging.MOVED, GAP: judging.UNMOVED}


def test_a_hiss_readings_move_is_judged_on_the_scorers_jitter_not_the_benign_pairs():
    """Gap LSD 1.00 -> 0.80 dB is inside its 1.01 dB benign-pair floor, but it is a hiss reading: the move counts."""
    scores = {"inc": {slug: _flat(0.0, 1.00) for slug in TAPES}, "quieter": {slug: _flat(0.0, 0.80) for slug in TAPES}}
    verdicts = at.judge(scores, "inc", RANKING, floors=FLOORS)
    assert verdicts["quieter"]["qualifies"]
    assert verdicts["quieter"]["status"][GAP] == judging.MOVED


def test_every_verdict_carries_its_per_reading_distances_per_tape():
    verdicts = at.judge(_scores({"inc": 0.62, "far": 2.0}), "inc", RANKING, floors=FLOORS)
    assert verdicts["far"]["distances"]["t1"][TILT] == pytest.approx((2.0 - 0.85) / 0.1)
    assert verdicts["inc"]["distances"]["t2"] == {TILT: 0.0, GAP: pytest.approx((1.0 - 0.048) / 0.44)}


# ----------------------------------------------------------------------------- a round's log


@pytest.fixture(name="level_round")
def _level_round(tmp_path, monkeypatch):
    """A round whose two candidates moved tilt (outside its zone) less than its floor: nobody wins, the log says why."""
    state = {"incumbent": {"k": 0}, "rounds": [], "candidates": {}}
    incumbent_id = at.candidate_id(state["incumbent"])
    flats = {incumbent_id: _flat(1.20, 1.0), "c1": _flat(1.18, 1.0), "c2": _flat(1.25, 1.0)}
    monkeypatch.setattr(at, "propose", lambda *_args: {"c1": {"k": 1}, "c2": {"k": 2}})
    monkeypatch.setattr(at, "run_candidate", Mock())
    monkeypatch.setattr(at, "audio_fingerprint", lambda _out, cid, _tapes: (cid,))
    monkeypatch.setattr(at, "score_round", lambda live, *_args: {cid: {"t1": flats[cid]} for cid in live})
    args = SimpleNamespace(families=("dsp",), gates=None, language="ro", parallel=1, guards=ag.NO_GUARDS, floors=FLOORS)
    moved = at.run_round(args, "apl", {"t1": "t1.mov"}, tmp_path, state, RANKING, 1)
    return SimpleNamespace(moved=moved, state=state, log=(tmp_path / at.LOG_MD).read_text(encoding="utf-8"))


def test_a_round_of_sub_floor_moves_ends_below_the_floor_not_on_a_bare_plateau(level_round):
    assert not level_round.moved
    assert f"{judging.BELOW_FLOOR} (1 readings unmoved; 1 moved less than their floor)" in level_round.log
    assert judging.PLATEAU not in level_round.log


def test_the_round_log_holds_every_candidates_distance_per_reading_and_the_floor(level_round):
    assert f"| candidate | change | {TILT} | {GAP} |" in level_round.log
    assert "| floor | | 0.85 (benign pair) | 0.048 (jitter) |" in level_round.log
    assert '| c2 | {"k": 2} | 4 | 2.16 |' in level_round.log


def test_without_floors_the_same_round_is_judged_on_exact_ranks(tmp_path, monkeypatch):
    """No floors on args (a v2 run, or no report): c1's 0.02 dB/oct is a win on its tape and is accepted."""
    state = {"incumbent": {"k": 0}, "rounds": [], "candidates": {}}
    flats = {at.candidate_id(state["incumbent"]): _flat(0.62, 1.0), "c1": _flat(0.60, 1.0)}
    narrow = {TILT: {**RANKING[TILT], "dead_zone": 0.0}}
    monkeypatch.setattr(at, "propose", lambda *_args: {"c1": {"k": 1}})
    monkeypatch.setattr(at, "run_candidate", Mock())
    monkeypatch.setattr(at, "audio_fingerprint", lambda _out, cid, _tapes: (cid,))
    monkeypatch.setattr(at, "score_round", lambda live, *_args: {cid: {"t1": flats[cid]} for cid in live})
    args = SimpleNamespace(families=("dsp",), gates=None, language="ro", parallel=1, guards=ag.NO_GUARDS)
    assert at.run_round(args, "apl", {"t1": "t1.mov"}, tmp_path, state, narrow, 1)
    assert state["rounds"][0]["accepted"] == "c1"


# ----------------------------------------------------------------------------- the command line


def _cli(tmp_path, *extra):
    tapes = tmp_path / "tapes.json"
    tapes.write_text(json.dumps({"t1": "t1.mov"}), encoding="utf-8")
    grid = tmp_path / "grid.yaml"
    grid.write_text("ranking:\n  m: down\n", encoding="utf-8")
    return ["--engine", "apl", "--tapes", str(tapes), "--grid", str(grid), "--out", str(tmp_path / "out"), *extra]


def _loop(monkeypatch, tmp_path):
    rounds = Mock(return_value=False)
    monkeypatch.setattr(at, "run_round", rounds)
    monkeypatch.setattr(at, "seed_defaults", lambda _engine, start, _out: start)
    monkeypatch.setattr(at.tune, "probe_duration", lambda _path: None)
    monkeypatch.setattr(ag, "DEFAULT_NOISE_FLOORS", tmp_path / "absent_benign.json")
    monkeypatch.setattr(judging, "DEFAULT_JITTER_FLOORS", tmp_path / "absent_jitter.json")
    return rounds


def _report(tmp_path, name, points, readings):
    path = tmp_path / name
    path.write_text(json.dumps({"operating_points": points, "readings": {k: {"floor": v} for k, v in readings.items()}}), encoding="utf-8")
    return str(path)


def test_the_command_line_reads_both_floor_reports_onto_every_round(tmp_path, monkeypatch, capsys):
    rounds = _loop(monkeypatch, tmp_path)
    benign = _report(tmp_path, "benign.json", ["benign_pair"], {TILT: 0.85, GAP: 1.01})
    jitter = _report(tmp_path, "jitter.json", ["output"], {GAP: 0.048})
    at.main(_cli(tmp_path, "--noise-floors", benign, "--jitter-floors", jitter))
    floors = rounds.call_args.args[0].floors
    assert floors.floor(TILT) == pytest.approx(0.85)
    assert floors.floor(GAP) == pytest.approx(0.048)
    assert "hiss readings: jitter only" in capsys.readouterr().out


def test_without_reports_the_loop_warns_and_keeps_exact_ties(tmp_path, monkeypatch, capsys):
    rounds = _loop(monkeypatch, tmp_path)
    at.main(_cli(tmp_path))
    assert rounds.call_args.args[0].floors == judging.NO_FLOORS
    out = capsys.readouterr().out
    assert "warning: no --noise-floors report" in out
    assert "warning: no --jitter-floors report" in out


def test_a_benign_pair_report_given_as_the_jitter_floors_stops_the_loop(tmp_path, monkeypatch):
    _loop(monkeypatch, tmp_path)
    benign = _report(tmp_path, "benign.json", ["benign_pair"], {GAP: 1.01})
    argv = _cli(tmp_path, "--jitter-floors", benign)
    with pytest.raises(SystemExit, match="benign-pair report"):
        at.main(argv)


def test_a_hiss_readings_veto_limit_takes_its_jitter_floor(tmp_path, monkeypatch):
    """A grid vetoing on SIGMOS noise: its limit is the scorer's 0.114, not the benign pairs' 0.5."""
    rounds = _loop(monkeypatch, tmp_path)
    noise = "mos.sigmos_noise.delta.median"
    grid = tmp_path / "grid.yaml"
    benign = _report(tmp_path, "benign.json", ["benign_pair"], {noise: 0.5})
    jitter = _report(tmp_path, "jitter.json", ["output"], {noise: 0.114})
    argv = _cli(tmp_path, "--noise-floors", benign, "--jitter-floors", jitter)
    grid.write_text(f"ranking:\n  m: down\nvetoes:\n  {noise}: {{worse: down}}\n", encoding="utf-8")
    at.main(argv)
    assert rounds.call_args.args[0].guards.vetoes[noise].limit == pytest.approx(3 * 0.114)
