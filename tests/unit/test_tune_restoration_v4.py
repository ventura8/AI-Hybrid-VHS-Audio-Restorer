"""Ear v4 grids: v3 with the critical hiss entries, R4's shape demoted to display, two noise vetoes and the hard guards."""

from pathlib import Path

import pytest

from scripts import tune_restoration as tr
from scripts.restoration_quality import reward

GRIDS = Path(tr.REPO) / "scripts" / "tune_grids"
SHAPE = ("dsp.gap_slope_db_oct.output.median", "dsp.gap_spread_db.output.median", "dsp.gap_lsd_db.output.median")
HISS_TATA = {
    "dsp.hiss_prog_excess_db.output.median": (3.0, 0.65, "hiss_programme"),
    "dsp.hiss_prog_excess_all_db.output.median": (1.5, 0.45, "hiss_programme"),
    "dsp.hiss_pause_excess_db.output.tail": (3.0, 0.55, "hiss_pause"),
    "dsp.hiss_pause_excess_db.output.median": (1.5, 0.35, "hiss_pause"),
}
HISS_MUSIC = {**HISS_TATA, "dsp.hiss_prog_excess_all_db.output.median": (3.0, 0.45, "hiss_programme")}
HISS_MUSIC["dsp.hiss_prog_excess_db.output.median"] = (1.5, 0.65, "hiss_programme")
NOISE_VETOES = {
    "mos.dnsmos_bak.delta.median": {"worse": "down", "floor_multiple": 1},
    "mos.sigmos_noise.delta.median": {"worse": "down", "floor_multiple": 1},
}
TATA_CONSTRAINTS = {
    "dsp.hiss_prog_excess_db.output.median": {"worse": "up", "max_move": 0.65},
    "dsp.hiss_pause_excess_db.output.tail": {"worse": "up", "max_move": 0.55},
    "dsp.hiss_prog_margin_db.delta.median": {"worse": "up", "bound": 0.65},
    "dsp.hiss_pause_margin_db.delta.tail": {"worse": "up", "bound": 0.55},
    "dsp.pause_depth_speech_db.delta.median": {"worse": "up", "bound": 35.0},
    "dsp.hiss_room_delta_db.output.median": {"worse": "down", "bound": -54.0},
    "dsp.hiss_pump_db.output.median": {"worse": "up", "max_move": 3.0, "bound": 12.0},
    "dsp.hiss_prog_kept_hf_db.output.median": {"worse": "down", "max_move": 1.0},
    "dsp.hiss_island_kurt.output.median": {"worse": "up", "max_move": 0.3},
    "dsp.gap_island_kurt.output.median": {"worse": "up", "max_move": 0.3},
    "dsp.gap_mod_dist_db.output.median": {"worse": "up", "max_move": 1.2},
    "dsp.balance_top_db.output.median": {"worse": "down", "max_move": 0.25},
    "dsp.balance_tilt_db_oct.output.median": {"worse": "down", "bound": -2.25},
    "dsp.balance_presence_db.output.median": {"worse": "down", "bound": -2.25},
    "dsp.sib_abs_level_db.output.median": {"worse": "down", "bound": -3.4},
    "dsp.sib_texture_db.output.median": {"worse": "up", "bound": 2.0},
}
MUSIC_CONSTRAINTS = {
    "dsp.hiss_prog_excess_all_db.output.median": {"worse": "up", "max_move": 0.45},
    "dsp.hiss_pause_excess_db.output.tail": {"worse": "up", "max_move": 0.55},
    "dsp.hiss_prog_margin_all_db.delta.median": {"worse": "up", "bound": 0.45},
    "dsp.hiss_pump_db.output.median": {"worse": "up", "max_move": 3.0, "bound": 12.0},
    "dsp.hiss_prog_kept_hf_db.output.median": {"worse": "down", "max_move": 1.0},
    "dsp.hiss_island_kurt.output.median": {"worse": "up", "max_move": 0.3},
    "dsp.hiss_room_delta_db.output.median": {"worse": "down", "bound": -54.0},
    "dsp.balance_presence_db.output.median": {"worse": "down", "max_move": 0.25},
}


def _grid(name):
    return tr.load_grid(GRIDS / name)


def _hiss_entries(ranking):
    """`{reading: (weight, tie, family)}` of the hiss entries, each checked one-sided (target 0, dead zone 0, scale 1)."""
    specs = reward.parse_grid(ranking)
    hiss = {key: spec for key, spec in specs.items() if spec.family in reward.HISS_FAMILIES}
    assert {(spec.target, spec.dead_zone, spec.scale) for spec in hiss.values()} == {(0.0, 0.0, 1.0)}
    return {key: (spec.weight, spec.tie, spec.family) for key, spec in hiss.items()}


def test_the_speech_ranking_is_v3_less_r4s_shape_plus_the_hiss_entries():
    """Every v3 entry but the three shape readings is kept as it was; the four hiss entries join."""
    v3, v4 = _grid("tata_v3.yaml")["ranking"], _grid("tata_v4.yaml")["ranking"]
    kept = {key: entry for key, entry in v3.items() if key not in SHAPE}
    assert {**kept, **{key: v4[key] for key in HISS_TATA}} == v4
    assert _hiss_entries(v4) == HISS_TATA


def test_hiss_weighs_44_percent_of_the_speech_rank_weight():
    """Each place weighs 4.5; hiss is 9.0 of 20.5 and each hiss entry weighs twice a speech-side one."""
    specs = reward.parse_grid(_grid("tata_v4.yaml")["ranking"])
    hiss = sum(spec.weight for spec in specs.values() if spec.family in reward.HISS_FAMILIES)
    assert (hiss, sum(spec.weight for spec in specs.values())) == (9.0, 20.5)


def test_the_speech_vetoes_are_v3_plus_the_noise_judges():
    """DNSMOS BAK and SIGMOS NOISE join as vetoes only, at one benign-pair floor."""
    assert _grid("tata_v4.yaml")["vetoes"] == {**_grid("tata_v3.yaml")["vetoes"], **NOISE_VETOES}


def test_the_speech_constraints_are_the_specified_set():
    """Sixteen hard guards: no new hiss, never hissier than the tape, no dead air, no gated hiss, no dulling, no islands."""
    assert _grid("tata_v4.yaml")["constraints"] == TATA_CONSTRAINTS


def test_the_rest_of_the_speech_grid_is_v3():
    """Reversals, the audibility tie, the engines and the variants stay as in v3."""
    v3, v4 = _grid("tata_v3.yaml"), _grid("tata_v4.yaml")
    keys = ("reversals", "audibility", "engines", "variants", "excerpt_s", "margin_s", "gates")
    assert {key: v4[key] for key in keys} == {key: v3[key] for key in keys}


def test_the_music_grid_is_v3_plus_the_hiss_entries_and_its_guards():
    """music_v4 keeps every music_v3 entry and veto; the all-band hiss weighs 3.0; eight relative guards."""
    v3, v4 = _grid("music_v3.yaml"), _grid("music_v4.yaml")
    assert {key: v4["ranking"][key] for key in v3["ranking"]} == v3["ranking"]
    assert _hiss_entries(v4["ranking"]) == HISS_MUSIC
    assert (v4["vetoes"], v4["constraints"]) == (v3["vetoes"], MUSIC_CONSTRAINTS)


@pytest.mark.parametrize("name", ["tata_v4.yaml", "music_v4.yaml"])
def test_every_v4_constraint_parses(name):
    """The constraints parse into the loop's rules."""
    assert len(reward.parse_constraints(_grid(name)["constraints"])) == len(_grid(name)["constraints"])
