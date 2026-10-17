"""The tuner's round C4 knobs (cathar's declip and dehum): their values, their start and when they are dead."""

import pytest

from modules import config as app_config
from scripts import autotune_restoration as at
from scripts import tune_restoration as tune

C4_KNOBS = {"cathar_declip_mode": ["always", "detected"], "cathar_dehum_freq": ["scanner", 0]}
SWITCHES = {"cathar_declip_mode": "cathar_enable_declip", "cathar_dehum_freq": "cathar_enable_dehum"}


def _moves(incumbent):
    moves = {}
    for knob, value in at.neighbour_moves("cathar", incumbent):
        moves.setdefault(knob, set()).add(value)
    return moves


def test_the_round_c4_knobs_are_in_the_cathar_table_only():
    assert {knob: at.KNOBS["cathar"][knob] for knob in C4_KNOBS} == C4_KNOBS
    assert not set(C4_KNOBS) & set(at.KNOBS["apl"])
    assert not set(C4_KNOBS) & set(at._CATHAR_MUSIC_KEYS)


def test_from_the_shipped_defaults_each_c4_knob_steps_to_its_alternative(tmp_path, monkeypatch):
    """Seeded from the app ('always', 'scanner'), each proposes its one other value."""
    monkeypatch.setattr(at.tune, "resolved_config", lambda *_args: dict(app_config.CONFIG))
    incumbent = at.seed_defaults("cathar", {}, tmp_path)
    assert [incumbent[knob] for knob in C4_KNOBS] == ["always", "scanner"]
    moves = _moves(incumbent)
    assert {knob: moves[knob] for knob in C4_KNOBS} == {"cathar_declip_mode": {"detected"}, "cathar_dehum_freq": {0}}


def test_a_zero_resolved_by_the_app_seeds_as_zero(tmp_path, monkeypatch):
    """A state that starts from cathar's own detection steps back to the scanner."""
    monkeypatch.setattr(at.tune, "resolved_config", lambda *_args: {**app_config.CONFIG, "cathar_dehum_freq": 0})
    incumbent = at.seed_defaults("cathar", {}, tmp_path)
    assert incumbent["cathar_dehum_freq"] == 0
    assert _moves(incumbent)["cathar_dehum_freq"] == {"scanner"}


@pytest.mark.parametrize("knob", sorted(C4_KNOBS))
def test_each_c4_knob_is_dead_with_its_stage_off(knob):
    assert knob in at.inert_knobs("cathar", {SWITCHES[knob]: False})
    assert knob not in at.inert_knobs("cathar", {})
    assert knob not in at.inert_knobs("cathar", {"cathar_music_profile": False})


def test_the_stage_switches_default_to_on():
    """inert_knobs reads an unset switch from SWITCH_DEFAULTS, the repository's config.yaml."""
    assert {at.SWITCH_DEFAULTS[switch] for switch in SWITCHES.values()} == {True}
    assert {app_config.CONFIG[switch] for switch in SWITCHES.values()} == {True}


def test_every_c4_value_is_a_setting_the_app_accepts():
    overrides = [{knob: value} for knob, values in C4_KNOBS.items() for value in values]
    assert [override for override in overrides if at.validation_problems(override)] == []


@pytest.mark.parametrize(("knob", "value"), [("cathar_declip_mode", "sometimes"), ("cathar_dehum_freq", 50), ("cathar_dehum_freq", False)])
def test_a_value_outside_the_choices_is_refused_before_the_render(knob, value):
    """False equals 0 but is no frequency: the app's normaliser refuses it, so the loop does too."""
    assert at.validation_problems({knob: value}) == [f"{knob}: {value!r} is not one of {sorted(C4_KNOBS[knob], key=str)}"]


@pytest.mark.parametrize(("knob", "value"), [(knob, value) for knob, values in C4_KNOBS.items() for value in values])
def test_the_app_resolves_each_value_as_given(knob, value):
    """The loop checks a candidate's resolved config (tune.assert_overrides_resolved); the normalisers keep every value."""
    normalise = {
        "cathar_declip_mode": app_config._normalize_cathar_declip_mode,
        "cathar_dehum_freq": app_config._normalize_cathar_dehum_freq,
    }
    assert tune.assert_overrides_resolved({knob: value}, {knob: normalise[knob](value)}) == []
