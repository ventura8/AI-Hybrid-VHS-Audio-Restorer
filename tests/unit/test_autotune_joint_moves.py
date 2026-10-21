"""Joint moves: the pairs the loop cannot reach one knob at a time are proposed whole (cathar round 3, 2026-10-20)."""

from scripts import autotune_joint_moves as joint
from scripts import autotune_restoration as at

PAIR, RELEASED = joint.JOINT_MOVES["cathar"]
# cathar's round-3 winner as far as the pair is concerned: the fill off, the suppressor off.
CATHAR_INCUMBENT = {"cathar_alpha": 1.5, "cathar_reference_suppress": False, "enable_pause_floor": False, "pause_floor_target": "cluster"}


def test_cathar_s_suppressor_and_its_fill_are_proposed_together_while_the_incumbent_lacks_them():
    proposals = joint.joint_moves("cathar", CATHAR_INCUMBENT, at.candidate_id)
    expected = [{**CATHAR_INCUMBENT, **PAIR}, {**CATHAR_INCUMBENT, **RELEASED}]
    assert proposals == {at.candidate_id(overrides): overrides for overrides in expected}


def test_the_second_move_is_the_pair_with_the_release_at_six_db():
    assert RELEASED == {**PAIR, "cathar_reference_release_db": 6.0}


def test_an_incumbent_carrying_the_pair_gets_only_the_release_and_one_carrying_both_gets_none():
    carrying = {**CATHAR_INCUMBENT, **PAIR}
    assert joint.joint_moves("cathar", carrying, at.candidate_id) == {at.candidate_id({**carrying, **RELEASED}): {**carrying, **RELEASED}}
    assert joint.joint_moves("cathar", {**CATHAR_INCUMBENT, **RELEASED}, at.candidate_id) == {}


def test_apl_has_no_joint_move():
    assert joint.joint_moves("apl", {"apl_reference_suppress": False}, at.candidate_id) == {}


def test_the_loop_proposes_the_pair_beside_the_single_knob_neighbours(monkeypatch):
    monkeypatch.setattr(at, "neighbour_moves", lambda *_args: [("cathar_alpha", 1.0)])
    proposals = at.propose("cathar", CATHAR_INCUMBENT)
    moves = ({"cathar_alpha": 1.0}, PAIR, RELEASED)
    assert set(proposals) == {at.candidate_id({**CATHAR_INCUMBENT, **move}) for move in moves}


def test_the_moves_name_only_settings_the_loop_tunes():
    assert set(PAIR) | set(RELEASED) <= set(at.KNOBS["cathar"])
