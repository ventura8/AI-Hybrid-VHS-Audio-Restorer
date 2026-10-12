"""Ear v4 grids: v3 with the critical hiss entries, R4's shape demoted to display, two noise vetoes and the hard guards;
the stored sets' readings pinned, so the constraints and the flags keep their acceptance outcomes."""

from pathlib import Path

import pytest

from scripts import tune_restoration as tr
from scripts.restoration_quality import gates, reward

GRIDS = Path(tr.REPO) / "scripts" / "tune_grids"
SHAPE = ("dsp.gap_slope_db_oct.output.median", "dsp.gap_spread_db.output.median", "dsp.gap_lsd_db.output.median")
HISS_TATA = {
    "dsp.hiss_prog_excess_db.output.median": (3.0, 0.25, "hiss_programme"),
    "dsp.hiss_prog_excess_all_db.output.median": (1.5, 0.2, "hiss_programme"),
    "dsp.hiss_pause_excess_db.output.tail": (3.0, 0.25, "hiss_pause"),
    "dsp.hiss_pause_excess_db.output.median": (1.5, 0.5, "hiss_pause"),
}
HISS_MUSIC = {**HISS_TATA, "dsp.hiss_prog_excess_all_db.output.median": (3.0, 0.2, "hiss_programme")}
HISS_MUSIC["dsp.hiss_prog_excess_db.output.median"] = (1.5, 0.25, "hiss_programme")
NOISE_VETOES = {
    "mos.dnsmos_bak.delta.median": {"worse": "down", "floor_multiple": 1},
    "mos.sigmos_noise.delta.median": {"worse": "down", "floor_multiple": 1},
}
TATA_CONSTRAINTS = {
    "dsp.hiss_prog_excess_db.output.median": {"worse": "up", "max_move": 0.25},
    "dsp.hiss_pause_excess_db.output.tail": {"worse": "up", "max_move": 0.25},
    "dsp.hiss_prog_margin_db.delta.median": {"worse": "up", "bound": 0.15},
    "dsp.hiss_pause_margin_db.delta.tail": {"worse": "up", "bound": 0.35},
    "dsp.pause_depth_speech_db.delta.median": {"worse": "up", "bound": 35.0},
    "dsp.hiss_room_delta_db.output.median": {"worse": "down", "bound": -54.0},
    "dsp.hiss_pump_db.output.median": {"worse": "up", "max_move": 3.0, "bound": 12.0},
    "dsp.hiss_prog_kept_hf_db.output.median": {"worse": "down", "max_move": 1.0},
    "dsp.hiss_island_kurt.output.median": {"worse": "up", "max_move": 0.3},
    "dsp.gap_island_kurt.output.median": {"worse": "up", "max_move": 0.3, "bound": 1.85},
    "dsp.gap_mod_dist_db.output.median": {"worse": "up", "max_move": 1.2},
    "dsp.balance_top_db.output.median": {"worse": "down", "max_move": 0.25, "bound": -1.0},
    "dsp.balance_tilt_db_oct.output.median": {"worse": "down", "bound": -2.25},
    "dsp.balance_presence_db.output.median": {"worse": "down", "bound": -2.25},
    "dsp.sib_abs_level_db.output.median": {"worse": "down", "bound": -3.4},
    "dsp.sib_texture_db.output.median": {"worse": "up", "bound": 2.0},
}
MUSIC_CONSTRAINTS = {
    "dsp.hiss_prog_excess_all_db.output.median": {"worse": "up", "max_move": 0.2},
    "dsp.hiss_pause_excess_db.output.tail": {"worse": "up", "max_move": 0.25},
    "dsp.hiss_prog_margin_all_db.delta.median": {"worse": "up", "bound": 0.35},
    "dsp.hiss_pump_db.output.median": {"worse": "up", "max_move": 3.0, "bound": 12.0},
    "dsp.hiss_prog_kept_hf_db.output.median": {"worse": "down", "max_move": 1.0},
    "dsp.hiss_island_kurt.output.median": {"worse": "up", "max_move": 0.3},
    "dsp.hiss_room_delta_db.output.median": {"worse": "down", "bound": -54.0},
    "dsp.balance_presence_db.output.median": {"worse": "down", "max_move": 0.25},
}

PROG_FLAG, PAUSE_FLAG = "dsp.hiss_prog_margin_db.output.median", "dsp.hiss_pause_margin_db.output.tail"
READINGS = (*TATA_CONSTRAINTS, PROG_FLAG, PAUSE_FLAG)
H1, H2 = "dsp.hiss_prog_excess_db.output.median", "dsp.hiss_pause_excess_db.output.tail"
PAUSE_DELTA, DEPTH = "dsp.hiss_pause_margin_db.delta.tail", "dsp.pause_depth_speech_db.delta.median"
ROOM, PUMP = "dsp.hiss_room_delta_db.output.median", "dsp.hiss_pump_db.output.median"
KEPT_HF, MODULATION = "dsp.hiss_prog_kept_hf_db.output.median", "dsp.gap_mod_dist_db.output.median"
DEAD_APL = (
    "apl__baseline",
    "apl__no_air",
    "apl__no_expander",
    "apl__no_expander_no_air",
    "apl__roformer",
    "apl__roformer_no_expander_no_a",
)
BOTH_FLAGS = {"listener.hiss_programme", "listener.hiss_pause"}
# SOTI's alpha 2.0 raises the pause tail excess 0.638 dB over the baseline's 0, past the pause tie (0.25).
LESS_REMOVAL_SOTI = {H2}
# The stored sets' readings (the acceptance run of 2026-10-12, docs/ear_v4_hiss.md): validate_restoration --metrics dsp at
# windows 15 s, hop 7.5 s, experiments/hiss_v4/<set>.json; per output, the aggregate a grid reads for each name in READINGS
# ("-" unread). The family tapes stay on the machine; these are numbers only.
PINNED_TABLE = """
r1_tele7abc/cathar__baseline
  0.0 0.0 -28.968 -37.408 32.975 -39.675 2.712 -2.643 1.831 0.959 9.097 -0.231 -0.066 -0.231 -1.892 1.601 -2.938 -4.608
r1_tele7abc/apl__baseline
  10.831 0.0 -15.611 -36.954 37.225 -34.252 18.655 -0.318 1.071 0.176 5.928 -0.257 -0.241 -0.257 0.967 2.42 10.831 -4.436
r1_tele7abc/apl__no_air
  10.34 0.0 -16.275 -36.998 37.491 -34.249 18.663 -0.761 1.072 0.174 5.97 -0.3 -0.291 -0.3 0.111 2.406 10.34 -4.48
r1_tele7abc/apl__no_expander
  10.83 0.0 -15.603 -33.304 36.888 -31.064 15.413 -0.305 1.073 0.315 5.699 -0.256 -0.241 -0.256 0.967 2.415 10.83 -1.511
r1_tele7abc/apl__no_expander_no_air
  10.339 0.0 -16.255 -33.355 37.201 -31.058 15.411 -0.706 1.074 0.308 5.784 -0.301 -0.292 -0.301 0.111 2.405 10.339 -1.564
r1_tele7abc/apl__roformer
  13.723 0.0 -12.653 -124.564 40.306 -118.992 95.677 -0.291 1.279 0.0 0.0 -0.254 -0.268 -0.254 0.948 2.375 13.723 -89.867
r1_tele7abc/apl__roformer_no_expander_no_a
  13.252 0.0 -13.057 -123.01 40.227 -117.958 94.926 -0.673 1.288 0.0 0.0 -0.299 -0.315 -0.299 0.117 2.376 13.252 -89.502
r1_tele7abc/cathar075__alpha_2_0
  0.0 0.0 -26.022 -37.855 31.531 -40.05 0.89 -3.583 1.711 1.734 9.863 -0.963 -0.857 -0.963 -2.466 1.652 -0.627 -4.35
r1_tele7abc/cathar__alpha_2_25
  0.0 0.0 -26.373 -36.778 31.57 -39.627 2.095 -2.363 1.824 1.736 9.489 -0.233 -0.061 -0.233 -2.074 1.37 -0.54 -3.439
r1_tele7abc/cathar__alpha_2_0
  2.787 0.973 -22.724 -32.443 29.012 -39.585 0.984 -2.141 1.682 1.754 9.469 -0.213 -0.052 -0.213 -2.096 1.184 2.787 0.93
r1_tele7abc/cathar__alpha_2_0_beta_0_005
  2.519 0.8 -22.981 -32.686 29.538 -45.615 1.144 -2.142 1.7 1.722 10.186 -0.214 -0.052 -0.214 -2.097 1.219 2.519 0.687
fill_tele7abc/fill12
  18.098 31.03 -7.703 -1.513 4.06 -2.627 -8.506 0.003 1.081 0.043 4.643 -0.078 -0.054 -0.078 0.253 0.394 18.098 31.03
fill_tele7abc/fill8
  21.412 33.435 -4.286 0.999 1.047 -0.492 -7.72 0.004 1.039 0.043 2.551 -0.079 -0.052 -0.079 0.253 0.394 21.412 33.435
fill_tele7abc/fill18
  13.934 24.676 -11.913 -7.698 8.04 -8.631 -7.188 0.003 1.122 0.083 4.998 -0.078 -0.054 -0.078 0.253 0.394 13.934 24.676
fill_tele7abc/keeper_off
  12.383 9.913 -13.818 -27.258 10.237 -67.51 68.394 0.003 1.157 0.12 11.47 -0.078 -0.056 -0.078 0.253 0.394 12.383 5.667
v4air_tele7abc/shipped_apl
  18.93 31.025 -7.365 -1.512 4.058 -2.626 -7.534 0.006 1.085 0.043 4.642 -0.083 -0.06 -0.083 0.255 0.388 18.93 31.025
v4air_tele7abc/synth_lp4k
  15.003 30.155 -12.107 -2.476 4.573 -2.591 12.051 -23.036 -0.001 0.055 9.242 -1.35 -1.882 -1.35 -18.055 -1.492 15.003 30.155
v4air_tele7abc/synth_raw
  26.736 33.204 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 26.736 33.204
v4air_tele7abc/synth_gated
  18.113 0.0 -7.854 -152.421 5.483 -120.404 17.109 0.007 1.078 0.0 0.0 -0.083 -0.061 -0.083 0.26 0.374 18.113 -119.853
r1_soti/cathar__baseline
  0.0 0.0 -29.407 -35.953 32.379 -38.24 0.919 -6.325 1.659 1.668 9.866 -0.225 -0.394 -0.225 -0.981 0.219 -8.671 -3.39
r1_soti/cathar__alpha_2_0
  0.0 0.638 -23.796 -30.754 27.281 -38.289 -0.18 -5.561 1.616 1.46 9.585 -0.276 -0.442 -0.276 -1.542 0.182 -2.624 0.098
r1_vaccin/cathar__baseline
  0.0 0.0 -29.718 -35.093 30.421 -39.181 3.684 -18.106 0.976 0.345 8.304 -1.04 -0.684 -0.93 0.456 - -9.391 -4.293
r1_vaccin/cathar__alpha_2_0
  0.0 0.0 -26.014 -34.795 30.183 -39.146 5.216 -12.781 0.982 1.186 7.971 -0.816 -0.554 -0.697 0.216 - -6.326 -4.099
known_tele/stitched
  0.0 0.0 -28.968 -37.408 32.975 -39.675 2.712 -2.643 1.831 0.959 9.097 -0.231 -0.066 -0.231 -1.892 1.601 -2.938 -4.608
known_tele/single4s
  0.007 0.0 -28.964 -36.934 33.423 -39.619 0.671 -8.762 1.727 0.152 7.738 -2.039 -1.648 -2.039 0.392 3.665 -2.757 -4.29
known_tele/deesser_bug
  0.0 0.0 -32.728 -37.981 38.645 -39.661 17.101 -37.069 1.03 0.087 1.914 -4.668 -6.054 -4.668 -21.029 1.807 -6.989 -5.283
"""


def _row(text):
    return dict(zip(READINGS, (None if token == "-" else float(token) for token in text.split())))


def _pinned():
    lines = [line.strip() for line in PINNED_TABLE.strip().splitlines()]
    return {name: _row(values) for name, values in zip(lines[::2], lines[1::2])}


PINNED = _pinned()


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


def _trips(candidate, incumbent):
    """The tata_v4 constraints `candidate` trips against `incumbent`, both pinned outputs of one set."""
    rules = reward.parse_constraints(_grid("tata_v4.yaml")["constraints"])
    return set(reward.constrained_readings(PINNED[candidate], PINNED[incumbent], rules))


def _flags(case):
    """The hiss flags a pinned output raises (the two `USER_RULE_FIXED` gates on its median and tail)."""
    row = PINNED[case]
    aggregate = {
        "dsp.hiss_prog_margin_db": {"output": {"median": row[PROG_FLAG]}},
        "dsp.hiss_pause_margin_db": {"output": {"tail": row[PAUSE_FLAG]}},
    }
    hiss = {name: gates.GATES[name] for name in sorted(gates.USER_RULE_FIXED)}
    return set(gates.flag_failures(gates.evaluate_gates(aggregate, hiss)))


def test_every_pinned_row_carries_every_reading():
    """The table parses: 26 outputs, each with the sixteen constraint readings and the two flag readings."""
    assert (len(PINNED), {len(row) for row in PINNED.values()}) == (26, {len(READINGS)})


@pytest.mark.parametrize("label", DEAD_APL)
def test_every_dead_apl_render_is_excluded_by_depth_and_pump(label):
    """Round one's APL renders on Tele7abc against cathar__baseline: speech pause depth over 35 dB, pump over 12 dB."""
    assert {DEPTH, PUMP} <= _trips(f"r1_tele7abc/{label}", "r1_tele7abc/cathar__baseline")


@pytest.mark.parametrize("label", ["apl__roformer", "apl__roformer_no_expander_no_a"])
def test_the_roformer_renders_empty_the_room_band(label):
    """The roformer pauses read the room band 118 dB down: G2 (-54) excludes them as well."""
    assert ROOM in _trips(f"r1_tele7abc/{label}", "r1_tele7abc/cathar__baseline")


@pytest.mark.parametrize("label", ["cathar075__alpha_2_0", "cathar__alpha_2_25"])
def test_the_cleared_cathar_renders_trip_no_constraint(label):
    """The renders the user cleared of hiss stay in the race (the gap-island and top bounds keep them in)."""
    assert not _trips(f"r1_tele7abc/{label}", "r1_tele7abc/cathar__baseline")


@pytest.mark.parametrize("label", ["cathar__alpha_2_0", "cathar__alpha_2_0_beta_0_005"])
def test_the_renders_heard_hissy_trip_both_hiss_guards(label):
    """'cathar alpha 2 has hiss' (0.7.3): new hiss under the programme and in the pauses against the baseline."""
    assert {H1, H2} <= _trips(f"r1_tele7abc/{label}", "r1_tele7abc/cathar__baseline")


@pytest.mark.parametrize(
    ("label", "expected"),
    [("fill8", {H1, H2, PAUSE_DELTA}), ("fill18", set()), ("keeper_off", {ROOM, PUMP, MODULATION})],
)
def test_the_pause_fill_ladder_against_the_shipped_fill(label, expected):
    """Fill 12 is the incumbent: fill 8 brings hiss back (H1, H2), fill 18 is allowed, keeper off empties the room (G2)."""
    assert _trips(f"fill_tele7abc/{label}", "fill_tele7abc/fill12") == expected


@pytest.mark.parametrize(("label", "guard"), [("synth_lp4k", KEPT_HF), ("synth_raw", H1), ("synth_gated", ROOM), ("synth_gated", PUMP)])
def test_no_cheat_wins_against_the_shipped_render(label, guard):
    """A 4 kHz low-pass trips the dulling guard, the raw tape H1 and digital-silence pauses G2 and the pump guard."""
    assert guard in _trips(f"v4air_tele7abc/{label}", "v4air_tele7abc/shipped_apl")


@pytest.mark.parametrize(("tape", "expected"), [("r1_soti", LESS_REMOVAL_SOTI), ("r1_vaccin", set())])
def test_less_removal_against_the_baseline(tape, expected):
    """'alpha 2 sounds better': cathar alpha 2.0 against the baseline (alpha 2.5) on SOTI and Vaccin."""
    assert _trips(f"{tape}/cathar__alpha_2_0", f"{tape}/cathar__baseline") == expected


@pytest.mark.parametrize("label", ["deesser_bug", "single4s"])
def test_the_underwater_renders_trip_the_dulling_guard(label):
    """The de-esser bug (kept HF -37 dB) and single4s (-8.8 dB) against stitched (-2.6 dB) on Tele7abc."""
    assert KEPT_HF in _trips(f"known_tele/{label}", "known_tele/stitched")


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("r1_tele7abc/cathar__alpha_2_0", BOTH_FLAGS),
        ("r1_tele7abc/cathar__alpha_2_0_beta_0_005", BOTH_FLAGS),
        ("r1_tele7abc/cathar075__alpha_2_0", set()),
        ("r1_tele7abc/cathar__alpha_2_25", set()),
        ("r1_tele7abc/cathar__baseline", set()),
        ("v4air_tele7abc/shipped_apl", BOTH_FLAGS),
    ],
)
def test_the_hiss_flags_reproduce_the_verdicts(case, expected):
    """Round one's hiss labels on Tele7abc, and the shipped plateau render the user hears hiss on (the 2026-10-11 rule)."""
    assert _flags(case) == expected
