"""The tuner's round C3 knobs (cathar's music profile): their values, their start and when they are dead."""

import pytest

from modules import config as app_config
from scripts import autotune_restoration as at

C3_KNOBS = {
    "cathar_music_beta": [0.005, 0.01, 0.02, 0.04],
    "cathar_music_enable_repair": [True, False],
    "cathar_music_dewind_cutoff": [40, 60, 80],
}


def _moves(incumbent):
    moves = {}
    for knob, value in at.neighbour_moves("cathar", incumbent):
        moves.setdefault(knob, set()).add(value)
    return moves


def test_the_round_c3_knobs_are_in_the_cathar_table():
    assert {knob: at.KNOBS["cathar"][knob] for knob in C3_KNOBS} == C3_KNOBS
    assert set(C3_KNOBS) <= set(at._CATHAR_MUSIC_KEYS)


def test_from_the_shipped_defaults_each_c3_knob_steps_to_its_neighbours(tmp_path, monkeypatch):
    """Seeded from the app (music's old 0.02, on, 80 Hz), the floor steps both ways and the others one way."""
    monkeypatch.setattr(at.tune, "resolved_config", lambda *_args: dict(app_config.CONFIG))
    incumbent = at.seed_defaults("cathar", {}, tmp_path)
    assert [incumbent[knob] for knob in C3_KNOBS] == [0.02, True, 80]
    moves = _moves(incumbent)
    assert {knob: moves[knob] for knob in C3_KNOBS} == {
        "cathar_music_beta": {0.01, 0.04},
        "cathar_music_enable_repair": {False},
        "cathar_music_dewind_cutoff": {60},
    }


def test_the_c3_knobs_are_dead_with_the_music_profile_off():
    assert set(C3_KNOBS) <= at.inert_knobs("cathar", {"cathar_music_profile": False})
    assert not set(C3_KNOBS) & at.inert_knobs("cathar", {})


def test_the_music_floor_is_dead_under_the_wiener_denoiser():
    """`denoise --wiener` takes no beta on either material."""
    assert {"cathar_beta", "cathar_music_beta"} <= at.inert_knobs("cathar", {"cathar_denoise_method": "wiener"})


def test_the_music_cutoff_is_dead_with_the_dewind_off():
    assert "cathar_music_dewind_cutoff" in at.inert_knobs("cathar", {"cathar_enable_dewind": False})
    assert "cathar_music_dewind_cutoff" in _moves({})


@pytest.mark.parametrize(
    ("switches", "dead"),
    [
        ({"cathar_enable_repair": False, "cathar_music_enable_repair": False}, True),
        ({"cathar_enable_repair": False}, False),
        ({"cathar_music_enable_repair": False}, False),
    ],
)
def test_the_repair_strength_is_dead_only_when_no_material_repairs(switches, dead):
    """Speech takes cathar_enable_repair and music cathar_music_enable_repair."""
    assert ("cathar_repair_strength" in at.inert_knobs("cathar", switches)) is dead


def test_every_c3_value_is_a_setting_the_app_accepts():
    overrides = [{knob: value} for knob, values in C3_KNOBS.items() for value in values]
    assert [override for override in overrides if at.validation_problems(override)] == []
