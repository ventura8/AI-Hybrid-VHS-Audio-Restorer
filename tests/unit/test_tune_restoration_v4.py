"""Ear v4 grids: v3 with the critical hiss entries, R4's shape demoted to display, two noise vetoes and the hard guards;
the stored sets' readings pinned, so the constraints and the flags keep their acceptance outcomes."""

from pathlib import Path

import pytest

from scripts import tune_restoration as tr
from scripts.restoration_quality import gates, reward

GRIDS = Path(tr.REPO) / "scripts" / "tune_grids"
SHAPE = ("dsp.gap_slope_db_oct.output.median", "dsp.gap_spread_db.output.median", "dsp.gap_lsd_db.output.median")
HISS_TATA = {
    "dsp.hiss_prog_excess_db.output.median": (3.0, 0.3, "hiss_programme"),
    "dsp.hiss_prog_excess_all_db.output.median": (1.5, 0.25, "hiss_programme"),
    "dsp.hiss_pause_excess_db.output.tail": (3.0, 0.5, "hiss_pause"),
    "dsp.hiss_pause_excess_db.output.median": (1.5, 0.25, "hiss_pause"),
}
HISS_MUSIC = {**HISS_TATA, "dsp.hiss_prog_excess_all_db.output.median": (3.0, 0.25, "hiss_programme")}
HISS_MUSIC["dsp.hiss_prog_excess_db.output.median"] = (1.5, 0.3, "hiss_programme")
NOISE_VETOES = {
    "mos.dnsmos_bak.delta.median": {"worse": "down", "floor_multiple": 1},
    "mos.sigmos_noise.delta.median": {"worse": "down", "floor_multiple": 1},
}
TATA_CONSTRAINTS = {
    "dsp.hiss_prog_excess_db.output.median": {"worse": "up", "max_move": 0.3},
    "dsp.hiss_pause_excess_db.output.tail": {"worse": "up", "max_move": 0.5},
    "dsp.hiss_prog_margin_db.delta.median": {"worse": "up", "bound": 0.35},
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
    "dsp.hiss_prog_excess_all_db.output.median": {"worse": "up", "max_move": 0.25},
    "dsp.hiss_pause_excess_db.output.tail": {"worse": "up", "max_move": 0.5},
    "dsp.hiss_prog_margin_all_db.delta.median": {"worse": "up", "bound": 0.55},
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
# SOTI's alpha 2.0 raises the pause tail excess 0.33 dB over the baseline's 0, within the pause tie (0.50).
LESS_REMOVAL_SOTI = set()
# The stored sets' readings (validation round 2 of 2026-10-12, docs/ear_v4_hiss.md): validate_restoration --metrics dsp at
# windows 15 s, hop 7.5 s; per output, the aggregate a grid reads for each name in READINGS ("-" unread). The hiss
# readings are round 2's fixed build (the same windows through the runner's hiss path), the others the validation run's.
# The family tapes stay on the machine; these are numbers only.
PINNED_TABLE = """
r1_tele7abc/cathar__baseline
  0.0 0.0 -35.168 -37.762 32.975 -39.675 1.849 -3.839 1.869 0.959 9.097 -0.231 -0.066 -0.231 -1.892 1.601 -9.794 -5.258
r1_tele7abc/apl__baseline
  0.0 0.0 -28.207 -37.465 37.225 -34.448 18.201 -0.471 1.09 0.176 5.928 -0.257 -0.241 -0.257 0.967 2.42 -2.652 -5.08
r1_tele7abc/apl__no_air
  0.0 0.0 -29.161 -37.539 37.491 -34.444 18.199 -1.1 1.089 0.174 5.97 -0.3 -0.291 -0.3 0.111 2.406 -3.682 -5.154
r1_tele7abc/apl__no_expander
  0.0 0.0 -28.118 -34.328 36.888 -31.064 14.901 -0.442 1.096 0.315 5.699 -0.256 -0.241 -0.256 0.967 2.415 -2.634 -2.079
r1_tele7abc/apl__no_expander_no_air
  0.0 0.0 -29.088 -34.363 37.201 -31.058 14.902 -1.071 1.096 0.308 5.784 -0.301 -0.292 -0.301 0.111 2.405 -3.679 -2.134
r1_tele7abc/apl__roformer
  2.278 0.0 -23.29 -96.673 40.306 -118.992 95.75 -0.42 1.27 0 0 -0.254 -0.268 -0.254 0.948 2.375 2.278 -65
r1_tele7abc/apl__roformer_no_expander_no_a
  1.203 0.0 -24.356 -96.673 40.227 -117.958 95.052 -1.019 1.274 0 0 -0.299 -0.315 -0.299 0.117 2.376 1.203 -65
r1_tele7abc/cathar075__alpha_2_0
  0.0 0.0 -31.213 -35.907 31.531 -40.05 1.293 -4.078 1.707 1.734 9.863 -0.963 -0.857 -0.963 -2.466 1.652 -6.155 -2.842
r1_tele7abc/cathar__alpha_2_25
  0.0 0.0 -32.241 -37.411 31.57 -39.627 1.546 -3.677 1.829 1.736 9.489 -0.233 -0.061 -0.233 -2.074 1.37 -7.498 -4.133
r1_tele7abc/cathar__alpha_2_0
  0.0 1.337 -27.943 -31.992 29.012 -39.585 1.329 -3.428 1.672 1.754 9.469 -0.213 -0.052 -0.213 -2.096 1.184 -3.071 1.337
r1_tele7abc/cathar__alpha_2_0_beta_0_005
  0.0 0.975 -28.396 -32.424 29.538 -45.615 1.64 -3.431 1.697 1.722 10.186 -0.214 -0.052 -0.214 -2.097 1.219 -3.513 0.975
fill_tele7abc/fill12
  6.43 30.675 -18.072 -1.61 4.06 -2.627 -8.689 0.21 1.081 0.043 4.643 -0.078 -0.054 -0.078 0.253 0.394 6.43 30.675
fill_tele7abc/fill8
  11.682 33.303 -14.242 1.03 1.047 -0.492 -8.418 0.212 1.036 0.043 2.551 -0.079 -0.052 -0.079 0.253 0.394 11.682 33.303
fill_tele7abc/fill18
  3.201 24.764 -22.807 -7.629 8.04 -8.631 -6.733 0.209 1.12 0.083 4.998 -0.078 -0.054 -0.078 0.253 0.394 3.201 24.764
fill_tele7abc/keeper_off
  0.662 8.315 -25.501 -29.882 10.237 -67.51 68.583 0.209 1.154 0.12 11.47 -0.078 -0.056 -0.078 0.253 0.394 0.662 3.683
v4air_tele7abc/shipped_apl
  8.1 30.67 -17.916 -1.61 4.058 -2.626 -7.989 0.206 1.088 0.043 4.642 -0.083 -0.06 -0.083 0.255 0.388 8.1 30.67
v4air_tele7abc/synth_lp4k
  0.0 28.728 -28.626 -3.54 4.573 -2.591 -3.459 -35.007 -0.001 0.055 9.242 -1.35 -1.882 -1.35 -18.055 -1.472 -2.419 28.728
v4air_tele7abc/synth_raw
  25.697 33.554 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 25.697 33.554
v4air_tele7abc/synth_gated
  7.926 1.759 -18.179 -31.193 5.483 -119.638 11.532 0.207 1.079 2.221 27.614 -0.083 -0.061 -0.083 0.26 0.374 7.926 1.735
r1_soti/cathar__baseline
  0.0 0.0 -34.128 -36.277 32.379 -38.24 0.985 -5.51 1.438 1.668 9.866 -0.225 -0.394 -0.225 -0.981 0.219 -12.827 -4.503
r1_soti/cathar__alpha_2_0
  0.0 0.327 -26.238 -30.088 27.281 -38.253 0.556 -4.882 1.631 1.46 9.585 -0.276 -0.442 -0.276 -1.542 0.182 -6.035 -0.399
r1_vaccin/cathar__baseline
  0.0 0.711 -28.695 -29.708 30.421 -39.029 4.112 -2.396 1.043 0.345 8.304 -1.04 -0.684 -0.93 0.456 - -10.803 0.711
r1_vaccin/cathar__alpha_2_0
  0.0 0.844 -22.776 -29.586 30.183 -39.315 4.351 -1.927 1.003 1.186 7.971 -0.816 -0.554 -0.697 0.216 - -4.95 0.844
known_tele/stitched
  0.0 0.0 -35.168 -37.762 32.975 -39.675 1.849 -3.839 1.869 0.959 9.097 -0.231 -0.066 -0.231 -1.892 1.601 -9.794 -5.258
known_tele/single4s
  0.0 0.0 -35.107 -37.306 33.423 -39.619 1.313 -7.042 1.863 0.152 7.738 -2.039 -1.648 -2.039 0.392 3.665 -9.229 -4.288
known_tele/deesser_bug
  0.0 0.0 -49.198 -39.241 38.645 -39.661 14.208 -35.322 - 0.087 1.914 -4.668 -6.054 -4.668 -21.029 1.807 -23.991 -6.381
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
def test_the_renders_heard_hissy_read_hissier_than_every_cleared_render(label):
    """'cathar alpha 2 has hiss' (0.7.3, round one, not blind): hissier than every cleared render in both places.

    Under the programme both read under the threshold (-3.1 / -3.5 against -6.2..-9.8), in the pauses over it (+1.34 /
    +0.98 against -2.8..-5.3), and against the baseline both trip the pause H1 (excess over the 0.50 tie). The user's
    blind ABX of 2026-10-09 could not tell alpha 2 on 0.7.3 from 0.7.5 (8 of 16) nor their pauses (6 of 7 "same").
    """
    row = PINNED[f"r1_tele7abc/{label}"]
    for cleared in ("cathar075__alpha_2_0", "cathar__alpha_2_25", "cathar__baseline"):
        other = PINNED[f"r1_tele7abc/{cleared}"]
        assert row[PROG_FLAG] > other[PROG_FLAG] + 2.0
        assert row[PAUSE_FLAG] > other[PAUSE_FLAG] + 3.0
    assert {H1, H2} & _trips(f"r1_tele7abc/{label}", "r1_tele7abc/cathar__baseline") == {H2}


@pytest.mark.parametrize(
    ("label", "expected"),
    [("fill8", {H1, H2, PAUSE_DELTA}), ("fill18", set()), ("keeper_off", {ROOM, PUMP, MODULATION})],
)
def test_the_pause_fill_ladder_against_the_shipped_fill(label, expected):
    """Fill 12 is the incumbent: fill 8 brings hiss back (H1, H2), fill 18 is allowed, keeper off empties the room (G2)."""
    assert _trips(f"fill_tele7abc/{label}", "fill_tele7abc/fill12") == expected


@pytest.mark.parametrize(
    ("label", "guard"), [("synth_lp4k", KEPT_HF), ("synth_raw", H1), ("synth_gated", ROOM), ("synth_gated", MODULATION)]
)
def test_no_cheat_wins_against_the_shipped_render(label, guard):
    """A 4 kHz low-pass trips the dulling guard, the raw tape H1, digital-silence pauses G2 and the pause texture guards.

    The gated cheat's pump reads 11.5 dB, just under its 12 dB bound: G2 (room band 120 dB down) excludes it.
    """
    assert guard in _trips(f"v4air_tele7abc/{label}", "v4air_tele7abc/shipped_apl")


@pytest.mark.parametrize(("tape", "expected"), [("r1_soti", LESS_REMOVAL_SOTI), ("r1_vaccin", set())])
def test_less_removal_against_the_baseline(tape, expected):
    """'alpha 2 sounds better': cathar alpha 2.0 against the baseline (alpha 2.5) on SOTI and Vaccin."""
    assert _trips(f"{tape}/cathar__alpha_2_0", f"{tape}/cathar__baseline") == expected


@pytest.mark.parametrize("label", ["deesser_bug", "single4s"])
def test_the_underwater_renders_trip_the_dulling_guard(label):
    """The de-esser bug (kept HF -35.3 dB) and single4s (-7.9 dB) against stitched (-3.9 dB) on Tele7abc."""
    assert KEPT_HF in _trips(f"known_tele/{label}", "known_tele/stitched")


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("r1_tele7abc/cathar__alpha_2_0", {"listener.hiss_pause"}),
        ("r1_tele7abc/cathar__alpha_2_0_beta_0_005", {"listener.hiss_pause"}),
        ("r1_tele7abc/cathar075__alpha_2_0", set()),
        ("r1_tele7abc/cathar__alpha_2_25", set()),
        ("r1_tele7abc/cathar__baseline", set()),
        ("v4air_tele7abc/shipped_apl", BOTH_FLAGS),
    ],
)
def test_the_hiss_flags_reproduce_the_verdicts(case, expected):
    """Round one's hiss labels on Tele7abc, and the shipped plateau render the user hears hiss on (the 2026-10-11 rule)."""
    assert _flags(case) == expected
