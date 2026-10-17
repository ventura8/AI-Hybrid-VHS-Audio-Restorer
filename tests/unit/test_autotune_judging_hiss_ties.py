"""One loop judging after the ear v4 merge: the hiss ear's grid ties in the banded ranking, the guards beside it."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import autotune_guards as ag
from scripts import autotune_judging as judging
from scripts import autotune_restoration as at
from scripts import rejudge_autotune as rj
from scripts.restoration_quality import reward

EXCESS = "dsp.hiss_prog_excess_db.output.median"
TILT = "dsp.balance_tilt_db_oct.output.median"
# The tata_v4 entry: one-sided excess, target 0, dead zone 0, tie 0.3 (the jitter report's tie_floor, rounded up).
HISS_SPEC = {"target": 0.0, "dead_zone": 0.0, "scale": 1.0, "weight": 3.0, "tie": 0.3, "family": "hiss_programme"}
TILT_SPEC = {"target": 0.0, "dead_zone": 1.0, "scale": 0.1, "weight": 1.0, "family": "timbre"}
RANKING = {EXCESS: HISS_SPEC}
H1 = reward.parse_constraints({EXCESS: {"worse": "up", "max_move": 0.3}})
# A benign-pair floor of 5 dB on the hiss reading, as a Session 0 report could carry: it must never reach the ranking.
FLOORS = judging.Floors({EXCESS: 5.0, TILT: 0.85}, {EXCESS: 0.06})


# ----------------------------------------------------------------------------- the band and the dead zone


def test_a_grid_tie_bands_its_reading_without_any_floors_file():
    """Excess 0.0 / 0.3 tie on the tie alone; 0.31 is behind (the hiss branch's sweep and loop both did this)."""
    item = judging.entry(EXCESS, HISS_SPEC, {"inc": 0.0, "near": 0.3, "far": 0.31})
    assert item.status("near", "inc") == judging.BANDED
    assert item.status("far", "inc") == judging.MOVED


def test_the_band_is_the_larger_of_the_tie_and_the_jitter_floor_never_the_benign_pair():
    assert FLOORS.band(EXCESS, 0.3) == pytest.approx(0.3)
    assert FLOORS.band(EXCESS, 0.01) == pytest.approx(0.06)
    assert FLOORS.band(EXCESS) == pytest.approx(0.06)


def test_a_tie_entry_keeps_its_dead_zone_so_the_tie_forgives_once():
    """With the jitter floor also lifting the dead zone, excess 0.35 would tie with 0 (0.06 + 0.3); it must not."""
    item = judging.entry(EXCESS, HISS_SPEC, {"inc": 0.0, "hissier": 0.35}, FLOORS)
    assert item.distances["hissier"] == pytest.approx(0.35)
    assert item.status("hissier", "inc") == judging.MOVED


def test_an_entry_without_a_tie_keeps_the_dead_zone_floor():
    item = judging.entry(TILT, {**TILT_SPEC, "dead_zone": 0.0}, {"x": 0.5}, FLOORS)
    assert item.distances["x"] == 0.0


@pytest.mark.parametrize(
    ("tie", "source"),
    [(0.3, "grid tie"), (0.06, "grid tie"), (0.01, "jitter"), (0.0, "jitter")],
)
def test_the_band_says_where_it_comes_from(tie, source):
    assert FLOORS.source(EXCESS, tie) == source


def test_a_reading_with_neither_a_floor_nor_a_tie_has_no_band():
    assert judging.NO_FLOORS.source(EXCESS) == "none"
    assert judging.NO_FLOORS.band(EXCESS) == 0.0


@pytest.mark.parametrize(
    "key",
    ["listener.hiss_programme", "listener.hiss_pause", "dsp.pause_depth_speech_db.delta.median", "dsp.hiss_pump_db.output.median"],
)
def test_the_hiss_exemption_covers_the_flags_and_every_ear_v4_reading(key):
    assert judging.is_hiss_reading(key)


def test_the_grid_ties_are_read_off_the_ranking():
    assert judging.grid_ties({EXCESS: HISS_SPEC, TILT: TILT_SPEC, "m": "down"}) == {EXCESS: pytest.approx(0.3)}
    assert judging.grid_ties(None) == {}


def test_the_distance_table_shows_the_tie_as_the_hiss_readings_band():
    ordered = [("inc", {"distances": {"t1": {EXCESS: 0.0, TILT: 0.0}}}), ("a", {"distances": {"t1": {EXCESS: 0.4, TILT: 0.0}}})]
    lines = judging.distance_lines(ordered, str, FLOORS, {EXCESS: HISS_SPEC, TILT: TILT_SPEC})
    assert "| floor | | 0.3 (grid tie) | 0.85 (benign pair) |" in lines


# ----------------------------------------------------------------------------- the ranking and H1 agree


def _scores(excess):
    """The incumbent at excess 1.0 on two tapes, the candidate at `excess` on both."""
    return {cid: {slug: {EXCESS: value, "gates.hard_failures": 0.0} for slug in ("t1", "t2")} for cid, value in excess.items()}


def test_a_hiss_move_inside_the_tie_ties_in_the_ranking_and_passes_h1():
    verdicts = at.judge(_scores({"inc": 1.0, "c": 1.25}), "inc", RANKING, floors=FLOORS, constraints=H1)
    assert verdicts["c"]["status"] == {EXCESS: judging.BANDED}
    assert "constrained" not in verdicts["c"]


def test_a_hiss_move_past_the_tie_the_worse_way_loses_and_is_excluded():
    verdicts = at.judge(_scores({"inc": 1.0, "c": 1.4}), "inc", RANKING, floors=FLOORS, constraints=H1)
    assert verdicts["c"]["status"] == {EXCESS: judging.MOVED}
    assert verdicts["c"]["constrained"] == [
        "t1 dsp.hiss_prog_excess_db.output.median +0.4",
        "t2 dsp.hiss_prog_excess_db.output.median +0.4",
    ]
    assert not verdicts["c"]["qualifies"]


def test_a_hiss_move_past_the_tie_the_better_way_wins():
    verdicts = at.judge(_scores({"inc": 1.0, "c": 0.5}), "inc", RANKING, floors=FLOORS, constraints=H1)
    assert verdicts["c"]["wins"] == 2
    assert verdicts["c"]["qualifies"]


# ----------------------------------------------------------------------------- the round's label and the re-judge


def _scored(**extra):
    return {"failures": 0.0, "flags": 0.0, "level": True, "status": {EXCESS: judging.BANDED}, **extra}


def test_a_candidate_a_constraint_excluded_is_held_back_not_below_the_floor():
    verdicts = {"inc": _scored(), "a": _scored(constrained=["t1 dsp.hiss_pump_db.output.median +9"])}
    expected = "1 of 1 scored candidates level with the incumbent, 0 moved and lost, 1 held back"
    assert judging.plateau_label(verdicts, "inc", FLOORS) == f"{judging.PLATEAU} ({expected})"


def test_the_offline_re_judge_applies_the_grids_constraints_under_every_rule():
    context = {"ranking": RANKING, "guards": ag.Guards(constraints=H1), "floors": FLOORS}
    for rule in rj.RULES:
        verdicts = rj.judge_with(_scores({"inc": 1.0, "c": 1.4}), "inc", context, rule)
        assert "constrained" in verdicts["c"], rule


def test_the_round_log_carries_the_grid_tie(tmp_path, monkeypatch):
    """A whole round on the v4 hiss entry: the log's floor row names the grid tie."""
    state = {"incumbent": {"k": 0}, "rounds": [], "candidates": {}}
    flats = {at.candidate_id(state["incumbent"]): {EXCESS: 1.0}, "c1": {EXCESS: 1.1}}
    monkeypatch.setattr(at, "propose", lambda *_args: {"c1": {"k": 1}})
    monkeypatch.setattr(at, "run_candidate", Mock())
    monkeypatch.setattr(at, "audio_fingerprint", lambda _out, cid, _tapes: (cid,))
    monkeypatch.setattr(at, "score_round", lambda live, *_args: {cid: {"t1": flats[cid]} for cid in live})
    args = SimpleNamespace(families=("dsp",), gates=None, language="ro", parallel=1, guards=ag.Guards(constraints=H1), floors=FLOORS)
    assert not at.run_round(args, "apl", {"t1": "t1.mov"}, tmp_path, state, RANKING, 1)
    assert "| floor | | 0.3 (grid tie) |" in (tmp_path / at.LOG_MD).read_text(encoding="utf-8")
