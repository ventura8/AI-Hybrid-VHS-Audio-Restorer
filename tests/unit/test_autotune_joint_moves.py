"""Joint moves: the pairs the loop cannot reach one knob at a time are proposed whole (cathar round 3, 2026-10-20)."""

from scripts import autotune_joint_moves as joint
from scripts import autotune_restoration as at

PAIR = joint.JOINT_MOVES["cathar"][0]
# cathar's round-3 winner as far as the pair is concerned: the fill off, the suppressor off.
CATHAR_INCUMBENT = {"cathar_alpha": 1.5, "cathar_reference_suppress": False, "enable_pause_floor": False, "pause_floor_target": "cluster"}


def test_cathar_s_suppressor_and_its_fill_are_proposed_together_while_the_incumbent_lacks_them():
    proposals = joint.joint_moves("cathar", CATHAR_INCUMBENT, at.candidate_id)
    (cid, overrides), *_rest = proposals.items()
    assert (len(proposals), cid, overrides) == (1, at.candidate_id({**CATHAR_INCUMBENT, **PAIR}), {**CATHAR_INCUMBENT, **PAIR})


def test_an_incumbent_carrying_the_pair_gets_no_joint_move():
    assert joint.joint_moves("cathar", {**CATHAR_INCUMBENT, **PAIR}, at.candidate_id) == {}


def test_apl_has_no_joint_move():
    assert joint.joint_moves("apl", {"apl_reference_suppress": False}, at.candidate_id) == {}


def test_the_loop_proposes_the_pair_beside_the_single_knob_neighbours(monkeypatch):
    monkeypatch.setattr(at, "neighbour_moves", lambda *_args: [("cathar_alpha", 1.0)])
    proposals = at.propose("cathar", CATHAR_INCUMBENT)
    expected = {at.candidate_id({**CATHAR_INCUMBENT, "cathar_alpha": 1.0}), at.candidate_id({**CATHAR_INCUMBENT, **PAIR})}
    assert set(proposals) == expected


def test_the_pair_names_only_settings_the_loop_tunes():
    assert set(PAIR) <= set(at.KNOBS["cathar"])
