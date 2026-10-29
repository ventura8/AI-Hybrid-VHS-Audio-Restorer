"""`--freeze KNOB`: a knob held at the incumbent's value, for one a corpus confirmation refused that the tapes cannot show.

cathar's v10 loop accepted cathar_alpha 1.5 in its first round on the Tata tapes (2026-10-26), the setting that alone
tripped 29 of the v9 final's 57 IA clips (dulling bright programme those tapes do not carry). Frozen, no single move and
no joint move touches the knob; an unknown name stops the run before anything renders.
"""

from unittest.mock import patch

import pytest

from scripts import autotune_restoration as at

INCUMBENT = {"cathar_alpha": 1.0, "cathar_split_band_hz": 0}


def _moved_knobs(proposals):
    return {knob for overrides in proposals.values() for knob, value in overrides.items() if value != INCUMBENT.get(knob)}


def test_without_a_freeze_alpha_moves_and_frozen_it_does_not():
    moves = [("cathar_alpha", 1.5), ("cathar_beta", 0.01)]
    with patch.object(at, "neighbour_moves", return_value=moves), patch.object(at.joint, "joint_moves", return_value={}):
        free = at.propose("cathar", INCUMBENT)
        held = at.propose("cathar", INCUMBENT, frozen=("cathar_alpha",))
    assert _moved_knobs(free) == {"cathar_alpha", "cathar_beta"}
    assert _moved_knobs(held) == {"cathar_beta"}


def test_a_joint_move_that_touches_a_frozen_knob_is_dropped():
    joint_moves = {"j1": {**INCUMBENT, "cathar_split_band_hz": 4000, "cathar_beta": 0.01}, "j2": {**INCUMBENT, "cathar_beta": 0.01}}
    with patch.object(at, "neighbour_moves", return_value=[]), patch.object(at.joint, "joint_moves", return_value=joint_moves):
        held = at.propose("cathar", INCUMBENT, frozen=("cathar_split_band_hz",))
    assert list(held) == ["j2"]


def test_the_option_takes_known_knobs_and_refuses_an_unknown_one():
    args = at._parse_args(["--engine", "cathar", "--tapes", "t.json", "--freeze", "cathar_alpha", "--freeze", "cathar_split_band_hz"])
    assert args.freeze == ("cathar_alpha", "cathar_split_band_hz")
    assert at._parse_args(["--engine", "cathar", "--tapes", "t.json"]).freeze == ()
    with pytest.raises(SystemExit):
        at._parse_args(["--engine", "cathar", "--tapes", "t.json", "--freeze", "cathar_alphaa"])


def test_a_round_hands_the_frozen_knobs_to_the_proposal(tmp_path):
    seen = []
    args = at._parse_args(["--engine", "cathar", "--tapes", "t.json", "--freeze", "cathar_alpha"])
    state = {"incumbent": dict(INCUMBENT), "rounds": []}
    with (
        patch.object(at, "_incumbent_readings", return_value=None),
        patch.object(at, "propose", side_effect=lambda *a: seen.append(a[4]) or {}),
    ):
        assert at.run_round(args, "cathar", {}, tmp_path, state, {}, 1) is False
    assert seen == [("cathar_alpha",)]
