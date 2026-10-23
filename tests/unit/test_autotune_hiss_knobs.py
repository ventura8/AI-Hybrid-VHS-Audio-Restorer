"""The tuner's hiss and scan knobs (2026-10-12/17): their tables, the values the app accepts, and when they are dead."""

import pytest

from scripts import autotune_restoration as at

SHARED = ("pause_floor_target", "pause_floor_shape", "pause_floor_shelf_hz", "pause_floor_hf_cut_db")
SCAN = ("crt_detect_mode", "scanner_mains_vote", "azimuth_estimate_method", "sync_dtw_enable")
CATHAR_ONLY = (
    "cathar_noiseprint_mute_guard",
    "cathar_noiseprint_min_depth_db",
    "cathar_noiseprint_shallow_action",
    "cathar_noiseprint_method",
    "cathar_reference_suppress",
)
APL_ONLY = (
    "apl_post_neural_suppress",
    "apl_post_neural_suppress_min_residual_db",
    "apl_reference_suppress",
    "apl_neural_level_restore",
    "linear_air_gate",
    "apl_burst_ceiling",
    "apl_burst_ceiling_depth_db",
    "apl_surgical_ring_guard",
    "apl_depop_min_prominence_db",
)
SHAPE = "pause_floor_shape"
CUT = "pause_floor_hf_cut_db"
CORNER = "pause_floor_shelf_hz"


def _moves(engine, incumbent):
    moves = {}
    for knob, value in at.neighbour_moves(engine, incumbent):
        moves.setdefault(knob, set()).add(value)
    return moves


@pytest.mark.parametrize("engine", sorted(at.KNOBS))
def test_the_fill_and_scan_knobs_reach_both_engines(engine):
    assert set(SHARED + SCAN) <= set(at.KNOBS[engine])


def test_the_print_guards_are_cathar_knobs_only():
    assert set(CATHAR_ONLY) <= set(at.KNOBS["cathar"])
    assert not set(CATHAR_ONLY) & set(at.KNOBS["apl"])


def test_the_post_neural_stages_are_apl_knobs_only():
    assert set(APL_ONLY) <= set(at.KNOBS["apl"])
    assert not set(APL_ONLY) & set(at.KNOBS["cathar"])


def _new_knobs(engine):
    """The knobs of this change the engine's table holds."""
    return [knob for knob in SHARED + SCAN + CATHAR_ONLY + APL_ONLY if knob in at.KNOBS[engine]]


def _overrides(engine):
    """One single-knob override per value of every new knob."""
    return [{knob: value} for knob in _new_knobs(engine) for value in at.KNOBS[engine][knob]]


@pytest.mark.parametrize("engine", sorted(at.KNOBS))
def test_every_new_value_is_a_setting_the_app_accepts(engine):
    """validation_problems types each value against the app's own field table and choice sets."""
    assert [override for override in _overrides(engine) if at.validation_problems(override)] == []


@pytest.mark.parametrize("engine", sorted(at.KNOBS))
def test_a_flat_fill_reads_neither_the_corner_nor_the_cut(engine):
    assert {CORNER, CUT} <= at.inert_knobs(engine, {SHAPE: "flat"})


@pytest.mark.parametrize("engine", sorted(at.KNOBS))
def test_a_lowpass_fill_reads_its_corner_but_no_cut(engine):
    dead = at.inert_knobs(engine, {SHAPE: "lowpass"})
    assert (CUT in dead, CORNER in dead) == (True, False)


@pytest.mark.parametrize("engine", sorted(at.KNOBS))
def test_a_shelved_fill_reads_both(engine):
    assert not {CORNER, CUT} & at.inert_knobs(engine, {SHAPE: "shelf"})


@pytest.mark.parametrize("engine", sorted(at.KNOBS))
def test_with_the_keeper_off_nothing_it_reads_is_live(engine):
    assert set(at.PAUSE_FLOOR_KNOBS) <= at.inert_knobs(engine, {"enable_pause_floor": False, SHAPE: "shelf"})


def test_the_depth_guard_action_is_live_only_with_the_guard_on():
    action = "cathar_noiseprint_shallow_action"
    assert action in at.inert_knobs("cathar", {})
    assert action not in at.inert_knobs("cathar", {"cathar_noiseprint_min_depth_db": 13.0})


def test_without_a_print_none_of_its_guards_is_live():
    dead = at.inert_knobs("cathar", {"cathar_enable_noiseprint": False, "cathar_music_enable_noiseprint": False})
    assert set(at.NOISEPRINT_KNOBS) <= dead


def test_the_post_neural_parameters_follow_their_switches():
    shipped = at.inert_knobs("apl", {})
    on = at.inert_knobs("apl", {"apl_post_neural_suppress": True, "apl_burst_ceiling": "edges"})
    parameters = {"apl_post_neural_suppress_min_residual_db", "apl_burst_ceiling_depth_db"}
    assert (parameters <= shipped, parameters & on) == (True, set())


def test_the_gain_floor_is_live_when_either_suppressor_runs():
    floor = "apl_suppress_gain_floor_db"
    assert floor not in at.inert_knobs("apl", {"apl_enable_spectral_denoise": True, "apl_post_neural_suppress": True})
    assert floor in at.inert_knobs("apl", {"apl_enable_spectral_denoise": True})
    assert floor not in at.inert_knobs("apl", {"apl_enable_spectral_denoise": True, "apl_reference_suppress": True})


def test_the_reference_suppressor_is_one_move_from_the_shipped_state():
    """Ungated and needing no other switch, the reference suppressor is a single move the loop can make."""
    assert at.SWITCH_DEFAULTS["apl_reference_suppress"] is False
    assert _moves("apl", {"apl_reference_suppress": False})["apl_reference_suppress"] == {True}


def test_from_the_shipped_state_the_fill_steps_to_a_shaped_frame_fill():
    """From cluster + flat, the loop proposes the per-frame target and both shaped fills."""
    moves = _moves("apl", {"pause_floor_target": "cluster", SHAPE: "flat", "enable_pause_floor": True})
    assert (moves["pause_floor_target"], moves[SHAPE]) == ({"frame"}, {"shelf"})
