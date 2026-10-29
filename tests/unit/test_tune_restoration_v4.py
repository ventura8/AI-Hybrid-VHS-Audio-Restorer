"""Ear v4 grids: v3 with the critical hiss entries, R4's shape demoted to display, two noise vetoes and the hard guards;
the stored sets' readings pinned, so the constraints and the flags keep their acceptance outcomes."""

from pathlib import Path

import pytest

from scripts import tune_restoration as tr
from scripts.restoration_quality import gates, reward

GRIDS = Path(tr.REPO) / "scripts" / "tune_grids"
TATA_V3, TATA_V4 = "tata_v3.yaml", "tata_v4.yaml"
H1, H2 = "dsp.hiss_prog_excess_db.output.median", "dsp.hiss_pause_excess_db.output.tail"
ALL_BAND, PAUSE_MEDIAN = "dsp.hiss_prog_excess_all_db.output.median", "dsp.hiss_pause_excess_db.output.median"
PAUSE_DELTA, DEPTH = "dsp.hiss_pause_margin_db.delta.tail", "dsp.pause_depth_speech_db.delta.median"
ROOM, PUMP = "dsp.hiss_room_delta_db.output.median", "dsp.hiss_pump_db.output.median"
KEPT_HF, MODULATION = "dsp.hiss_prog_kept_hf_db.output.median", "dsp.gap_mod_dist_db.output.median"
KEPT_SOFT = "dsp.hiss_prog_kept_soft_db.output.median"
ISLANDS = "dsp.hiss_island_kurt.output.median"
PAUSE_HISS = "listener.hiss_pause"
BASELINE, GATE_B = "r1_tele7abc/cathar__baseline", "gate_tele7abc/B"
SHAPE = ("dsp.gap_slope_db_oct.output.median", "dsp.gap_spread_db.output.median", "dsp.gap_lsd_db.output.median")
HISS_TATA = {
    H1: (3.0, 0.4, "hiss_programme"),
    ALL_BAND: (1.5, 0.45, "hiss_programme"),
    H2: (3.0, 0.35, "hiss_pause"),
    PAUSE_MEDIAN: (1.5, 0.35, "hiss_pause"),
}
HISS_MUSIC = {**HISS_TATA, ALL_BAND: (3.0, 0.25, "hiss_programme")}
HISS_MUSIC[H1] = (1.5, 0.3, "hiss_programme")
NOISE_VETOES = {
    "mos.dnsmos_bak.delta.median": {"worse": "down", "floor_multiple": 1},
    "mos.sigmos_noise.delta.median": {"worse": "down", "floor_multiple": 1},
}
TATA_CONSTRAINTS = {
    H1: {"worse": "up", "max_move": 0.4},
    H2: {"worse": "up", "max_move": 0.35},
    "dsp.hiss_prog_margin_db.delta.median": {"worse": "up", "bound": 0.35},
    PAUSE_DELTA: {"worse": "up", "bound": 0.35},
    DEPTH: {"worse": "up", "bound": 35.0},
    ROOM: {"worse": "down", "bound": -54.0},
    PUMP: {"worse": "up", "max_move": 3.0, "bound": 12.0},
    KEPT_HF: {"worse": "down", "max_move": 1.0, "bound": -6.0},
    KEPT_SOFT: {"worse": "down", "max_move": 0.05, "bound": -27.0},
    ISLANDS: {"worse": "up", "max_move": 0.3, "bound": 2.0},
    "dsp.gap_island_kurt.output.median": {"worse": "up", "max_move": 0.3, "bound": 1.85},
    MODULATION: {"worse": "up", "max_move": 1.2, "bound": 10.0},
    "dsp.balance_top_db.output.median": {"worse": "down", "max_move": 0.25, "bound": -1.0},
    "dsp.balance_tilt_db_oct.output.median": {"worse": "down", "bound": -2.25},
    "dsp.balance_presence_db.output.median": {"worse": "down", "bound": -2.25},
    "dsp.sib_abs_level_db.output.median": {"worse": "down", "bound": -3.4},
    "dsp.sib_texture_db.output.median": {"worse": "up", "bound": 2.0},
}
MUSIC_CONSTRAINTS = {
    ALL_BAND: {"worse": "up", "max_move": 0.25},
    H2: {"worse": "up", "max_move": 0.35},
    "dsp.hiss_prog_margin_all_db.delta.median": {"worse": "up", "bound": 0.55},
    PUMP: {"worse": "up", "max_move": 3.0, "bound": 12.0},
    KEPT_HF: {"worse": "down", "max_move": 1.0, "bound": -6.0},
    KEPT_SOFT: {"worse": "down", "max_move": 0.05, "bound": -28.0},
    ISLANDS: {"worse": "up", "max_move": 0.3, "bound": 2.0},
    ROOM: {"worse": "down", "bound": -54.0},
    "dsp.balance_presence_db.output.median": {"worse": "down", "max_move": 0.25},
}

PROG_FLAG, PAUSE_FLAG = "dsp.hiss_prog_margin_db.output.median", "dsp.hiss_pause_margin_db.output.tail"
# The pinned table's columns: the sixteen constraints of the acceptance run (the soft guard has its own table) and the flags.
READINGS = (*(key for key in TATA_CONSTRAINTS if key != KEPT_SOFT), PROG_FLAG, PAUSE_FLAG)
DEAD_APL = (
    "apl__baseline",
    "apl__no_air",
    "apl__no_expander",
    "apl__no_expander_no_air",
    "apl__roformer",
    "apl__roformer_no_expander_no_a",
)
BOTH_FLAGS = {"listener.hiss_programme", PAUSE_HISS}
# SOTI's alpha 2.0 raises the pause tail excess 0.33 dB over the baseline's 0, within the pause tie (0.35); 0.346 on the
# GCC-PHAT aligner (docs/ear_v4_hiss.md, "Tape Rows on the Merged Aligner").
LESS_REMOVAL_SOTI = set()
# The stored sets' readings (validation round 2 of 2026-10-12, docs/ear_v4_hiss.md): validate_restoration --metrics dsp at
# windows 15 s, hop 7.5 s; per output, the aggregate a grid reads for each name in READINGS ("-" unread). The hiss
# readings are the adversarial review's build (the same windows through the runner's hiss path: the pause reading's 3 dB
# silent test and weight, the pump's medians; the programme readings read exactly as round 2's), the others the
# validation run's. All of them were read with the head-search aligner (xcorr_head), which the merged build reproduces
# exactly under AI_RESTORE_ALIGN_METHOD=xcorr_head. On the GCC-PHAT aligner (the default since the merge with
# feature/v1.4.0) the APL, fill, v4_air, cheat and gate rows read identically, the cathar rows move with their lag (hiss
# readings 0.29 dB at most), and every outcome these tests pin still holds (docs/ear_v4_hiss.md).
# The family tapes stay on the machine; these are numbers only.
PINNED_TABLE = """
r1_tele7abc/cathar__baseline
  0.0 0.0 -35.168 -37.658 32.975 -39.675 4.379 -3.839 1.869 0.959 9.097 -0.231 -0.066 -0.231 -1.892 1.601 -9.794 -6.077
r1_tele7abc/apl__baseline
  0.0 0.0 -28.207 -38.104 37.225 -34.387 19.398 -0.471 1.09 0.176 5.928 -0.257 -0.241 -0.257 0.967 2.42 -2.652 -6.674
r1_tele7abc/apl__no_air
  0.0 0.0 -29.161 -38.199 37.491 -34.385 19.4 -1.1 1.089 0.174 5.97 -0.3 -0.291 -0.3 0.111 2.406 -3.682 -6.768
r1_tele7abc/apl__no_expander
  0.0 0.0 -28.118 -35.318 36.888 -31.064 18.099 -0.442 1.096 0.315 5.699 -0.256 -0.241 -0.256 0.967 2.415 -2.634 -3.684
r1_tele7abc/apl__no_expander_no_air
  0.0 0.0 -29.088 -35.399 37.201 -31.058 18.085 -1.071 1.096 0.308 5.784 -0.301 -0.292 -0.301 0.111 2.405 -3.679 -3.771
r1_tele7abc/apl__roformer
  2.278 0.0 -23.29 -95.954 40.306 -118.992 95.947 -0.42 1.27 0 0 -0.254 -0.268 -0.254 0.948 2.375 2.278 -65
r1_tele7abc/apl__roformer_no_expander_no_a
  1.203 0.0 -24.356 -95.954 40.227 -117.958 95.287 -1.019 1.274 0 0 -0.299 -0.315 -0.299 0.117 2.376 1.203 -65
r1_tele7abc/cathar075__alpha_2_0
  0.0 0.0 -31.213 -34.906 31.531 -40.05 2.933 -4.078 1.707 1.734 9.863 -0.963 -0.857 -0.963 -2.466 1.652 -6.155 -2.84
r1_tele7abc/cathar__alpha_2_25
  0.0 0.0 -32.241 -36.349 31.57 -39.627 4.656 -3.677 1.829 1.736 9.489 -0.233 -0.061 -0.233 -2.074 1.37 -7.498 -4.133
r1_tele7abc/cathar__alpha_2_0
  0.0 1.34 -27.943 -31.071 29.012 -39.585 4.877 -3.428 1.672 1.754 9.469 -0.213 -0.052 -0.213 -2.096 1.184 -3.071 1.34
r1_tele7abc/cathar__alpha_2_0_beta_0_005
  0.0 0.978 -28.396 -31.43 29.538 -45.615 6.49 -3.431 1.697 1.722 10.186 -0.214 -0.052 -0.214 -2.097 1.219 -3.513 0.978
fill_tele7abc/fill12
  6.43 29.851 -18.072 -1.521 4.06 -2.627 -8.708 0.21 1.081 0.043 4.643 -0.078 -0.054 -0.078 0.253 0.394 6.43 29.851
fill_tele7abc/fill8
  11.682 32.429 -14.242 1.034 1.047 -0.492 -8.437 0.212 1.036 0.043 2.551 -0.079 -0.052 -0.079 0.253 0.394 11.682 32.429
fill_tele7abc/fill18
  3.201 23.743 -22.807 -7.617 8.04 -8.631 -6.048 0.209 1.12 0.083 4.998 -0.078 -0.054 -0.078 0.253 0.394 3.201 23.743
fill_tele7abc/keeper_off
  0.662 3.159 -25.501 -34.77 10.237 -67.51 79.65 0.209 1.154 0.12 11.47 -0.078 -0.056 -0.078 0.253 0.394 0.662 -2.373
v4air_tele7abc/shipped_apl
  8.1 29.847 -17.916 -1.521 4.058 -2.626 -7.843 0.206 1.088 0.043 4.642 -0.083 -0.06 -0.083 0.255 0.388 8.1 29.847
v4air_tele7abc/synth_lp4k
  0.0 26.453 -28.626 -4.93 4.573 -2.591 -3.277 -35.007 -0.001 0.055 9.242 -1.35 -1.882 -1.35 -18.055 -1.472 -2.419 26.453
v4air_tele7abc/synth_raw
  25.697 32.307 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 25.697 32.307
v4air_tele7abc/synth_gated
  7.926 0.987 -18.179 -30.76 5.483 -119.638 103.949 0.207 1.079 2.221 27.614 -0.083 -0.061 -0.083 0.26 0.374 7.926 0.987
r1_soti/cathar__baseline
  0.0 0.0 -34.128 -35.966 32.379 -37.975 3.917 -5.51 1.438 1.668 9.866 -0.225 -0.394 -0.225 -0.981 0.219 -12.827 -6.012
r1_soti/cathar__alpha_2_0
  0.0 0.331 -26.238 -28.717 27.281 -37.919 3.611 -4.882 1.631 1.46 9.585 -0.276 -0.442 -0.276 -1.542 0.182 -6.035 -0.393
r1_vaccin/cathar__baseline
  0.0 0.712 -28.695 -27.871 30.421 -39.029 - -2.396 1.043 0.345 8.304 -1.04 -0.684 -0.93 0.456 - -10.803 0.712
r1_vaccin/cathar__alpha_2_0
  0.0 0.845 -22.776 -27.724 30.183 -39.315 - -1.927 1.003 1.186 7.971 -0.816 -0.554 -0.697 0.216 - -4.95 0.845
known_tele/stitched
  0.0 0.0 -35.168 -37.658 32.975 -39.675 4.379 -3.839 1.869 0.959 9.097 -0.231 -0.066 -0.231 -1.892 1.601 -9.794 -6.077
known_tele/single4s
  0.0 0.0 -35.107 -37.183 33.423 -39.619 3.509 -7.042 1.863 0.152 7.738 -2.039 -1.648 -2.039 0.392 3.665 -9.229 -5.503
known_tele/deesser_bug
  0.0 0.0 -49.198 -40.111 38.645 -39.661 19.728 -35.322 - 0.087 1.914 -4.668 -6.054 -4.668 -21.029 1.807 -23.991 -8.711
gate_tele7abc/B
  8.085 29.861 -17.91 -1.506 4.119 -2.619 -7.765 0.207 1.076 0.043 4.642 -0.083 -0.061 -0.083 0.26 0.374 8.085 29.861
gate_tele7abc/gate_20dB
  7.926 10.098 -18.179 -21.226 5.483 -22.104 11.842 0.207 1.077 0.079 4.313 -0.083 -0.061 -0.083 0.26 0.374 7.926 10.098
gate_tele7abc/gate_30dB
  7.926 2.18 -18.179 -29.568 5.483 -31.332 21.813 0.207 1.077 0.356 5.231 -0.083 -0.061 -0.083 0.26 0.374 7.926 2.18
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
    v3, v4 = _grid(TATA_V3)["ranking"], _grid(TATA_V4)["ranking"]
    kept = {key: entry for key, entry in v3.items() if key not in SHAPE}
    assert {**kept, **{key: v4[key] for key in HISS_TATA}} == v4
    assert _hiss_entries(v4) == HISS_TATA


def test_hiss_weighs_44_percent_of_the_speech_rank_weight():
    """Each place weighs 4.5; hiss is 9.0 of 20.5 and each hiss entry weighs twice a speech-side one."""
    specs = reward.parse_grid(_grid(TATA_V4)["ranking"])
    hiss = sum(spec.weight for spec in specs.values() if spec.family in reward.HISS_FAMILIES)
    assert (hiss, sum(spec.weight for spec in specs.values())) == (9.0, 20.5)


def test_the_speech_vetoes_are_v3_plus_the_noise_judges():
    """DNSMOS BAK and SIGMOS NOISE join as vetoes only, at one benign-pair floor."""
    assert _grid(TATA_V4)["vetoes"] == {**_grid(TATA_V3)["vetoes"], **NOISE_VETOES}


def test_the_speech_constraints_are_the_specified_set():
    """Seventeen hard guards: no new hiss, never hissier than the tape, no dead air, no gated hiss, no dulling (loud or
    soft programme), no islands."""
    assert _grid(TATA_V4)["constraints"] == TATA_CONSTRAINTS


def test_the_rest_of_the_speech_grid_is_v3():
    """Reversals, the audibility tie, the engines and the variants stay as in v3."""
    v3, v4 = _grid(TATA_V3), _grid(TATA_V4)
    keys = ("reversals", "audibility", "engines", "variants", "excerpt_s", "margin_s", "gates")
    assert {key: v4[key] for key in keys} == {key: v3[key] for key in keys}


def test_the_music_grid_is_v3_plus_the_hiss_entries_and_its_guards():
    """music_v4 keeps every music_v3 entry and veto; the all-band hiss weighs 3.0; nine relative guards."""
    v3, v4 = _grid("music_v3.yaml"), _grid("music_v4.yaml")
    assert {key: v4["ranking"][key] for key in v3["ranking"]} == v3["ranking"]
    assert _hiss_entries(v4["ranking"]) == HISS_MUSIC
    assert (v4["vetoes"], v4["constraints"]) == (v3["vetoes"], MUSIC_CONSTRAINTS)


@pytest.mark.parametrize("name", [TATA_V4, "music_v4.yaml"])
def test_every_v4_constraint_parses(name):
    """The constraints parse into the loop's rules."""
    assert len(reward.parse_constraints(_grid(name)["constraints"])) == len(_grid(name)["constraints"])


def _trips(candidate, incumbent):
    """The tata_v4 constraints `candidate` trips against `incumbent`, both pinned outputs of one set."""
    rules = reward.parse_constraints(_grid(TATA_V4)["constraints"])
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
    assert (len(PINNED), {len(row) for row in PINNED.values()}) == (29, {len(READINGS)})


@pytest.mark.parametrize("label", DEAD_APL)
def test_every_dead_apl_render_is_excluded_by_depth_and_pump(label):
    """Round one's APL renders on Tele7abc against cathar__baseline: speech pause depth over 35 dB, pump over 12 dB."""
    assert {DEPTH, PUMP} <= _trips(f"r1_tele7abc/{label}", BASELINE)


@pytest.mark.parametrize("label", ["apl__roformer", "apl__roformer_no_expander_no_a"])
def test_the_roformer_renders_empty_the_room_band(label):
    """The roformer pauses read the room band 118 dB down: G2 (-54) excludes them as well."""
    assert ROOM in _trips(f"r1_tele7abc/{label}", BASELINE)


@pytest.mark.parametrize("label", ["cathar075__alpha_2_0", "cathar__alpha_2_25"])
def test_the_cleared_cathar_renders_trip_no_constraint(label):
    """The renders the user cleared of hiss stay in the race (the gap-island and top bounds keep them in)."""
    assert not _trips(f"r1_tele7abc/{label}", BASELINE)


@pytest.mark.parametrize("label", ["cathar__alpha_2_0", "cathar__alpha_2_0_beta_0_005"])
def test_the_renders_heard_hissy_read_hissier_than_every_cleared_render(label):
    """'cathar alpha 2 has hiss' (0.7.3, round one, not blind): hissier than every cleared render in both places.

    Under the programme both read under the threshold (-3.1 / -3.5 against -6.2..-9.8), in the pauses over it (+1.34 /
    +0.98 against -2.8..-6.1), and against the baseline both trip the pause H1 (excess over the 0.35 tie). The user's
    blind ABX of 2026-10-09 could not tell alpha 2 on 0.7.3 from 0.7.5 (8 of 16) nor their pauses (6 of 7 "same").
    """
    row = PINNED[f"r1_tele7abc/{label}"]
    for cleared in ("cathar075__alpha_2_0", "cathar__alpha_2_25", "cathar__baseline"):
        other = PINNED[f"r1_tele7abc/{cleared}"]
        assert row[PROG_FLAG] > other[PROG_FLAG] + 2.0
        assert row[PAUSE_FLAG] > other[PAUSE_FLAG] + 3.0
    assert {H1, H2} & _trips(f"r1_tele7abc/{label}", BASELINE) == {H2}


@pytest.mark.parametrize(
    ("label", "expected"),
    [("fill8", {H1, H2, PAUSE_DELTA}), ("fill18", set()), ("keeper_off", {ROOM, PUMP, MODULATION})],
)
def test_the_pause_fill_ladder_against_the_shipped_fill(label, expected):
    """Fill 12 is the incumbent: fill 8 brings hiss back (H1, H2), fill 18 is allowed, keeper off empties the room (G2)."""
    assert _trips(f"fill_tele7abc/{label}", "fill_tele7abc/fill12") == expected


@pytest.mark.parametrize(
    ("label", "guard"),
    [("synth_lp4k", KEPT_HF), ("synth_raw", H1), ("synth_gated", ROOM), ("synth_gated", MODULATION), ("synth_gated", PUMP)],
)
def test_no_cheat_wins_against_the_shipped_render(label, guard):
    """A 4 kHz low-pass trips the dulling guard, the raw tape H1, digital-silence pauses G2, the pause texture guards and
    the pump (+103.9 with medians over the pause frames; the means read it at 11.5, under the 12 dB bound)."""
    assert guard in _trips(f"v4air_tele7abc/{label}", "v4air_tele7abc/shipped_apl")


def test_a_pause_gate_that_silences_the_pauses_is_excluded_by_the_pump():
    """B with the source's true pauses gated 30 dB (pause tail +2.2, room band 31 dB down, speech pause depth unmoved)
    trips the pump (+21.8 against B's -7.8) and only the pump; gated 20 dB its pauses keep audible hiss (+10.1) and its
    pump (+11.8) sits under the bound: it is left to the listening session. Before the review both passed every guard
    and won the hiss_pause family (the means read their pump at +8.2 / +11.2)."""
    assert _trips("gate_tele7abc/gate_30dB", GATE_B) == {PUMP}
    assert not _trips("gate_tele7abc/gate_20dB", GATE_B)
    assert PINNED["gate_tele7abc/gate_20dB"][PAUSE_FLAG] > 0.0 > PINNED[GATE_B][PUMP]


@pytest.mark.parametrize(("tape", "expected"), [("r1_soti", LESS_REMOVAL_SOTI), ("r1_vaccin", set())])
def test_less_removal_against_the_baseline(tape, expected):
    """'alpha 2 sounds better': cathar alpha 2.0 against the baseline (alpha 2.5) on SOTI and Vaccin."""
    assert _trips(f"{tape}/cathar__alpha_2_0", f"{tape}/cathar__baseline") == expected


@pytest.mark.parametrize("label", ["deesser_bug", "single4s"])
def test_the_underwater_renders_trip_the_dulling_guard(label):
    """The de-esser bug (kept HF -35.3 dB) and single4s (-7.0 dB) against stitched (-3.8 dB) on Tele7abc."""
    assert KEPT_HF in _trips(f"known_tele/{label}", "known_tele/stitched")


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("r1_tele7abc/cathar__alpha_2_0", {PAUSE_HISS}),
        ("r1_tele7abc/cathar__alpha_2_0_beta_0_005", {PAUSE_HISS}),
        ("r1_tele7abc/cathar075__alpha_2_0", set()),
        ("r1_tele7abc/cathar__alpha_2_25", set()),
        (BASELINE, set()),
        ("v4air_tele7abc/shipped_apl", BOTH_FLAGS),
    ],
)
def test_the_hiss_flags_reproduce_the_verdicts(case, expected):
    """Round one's hiss labels on Tele7abc, and the shipped plateau render the user hears hiss on (the 2026-10-11 rule)."""
    assert _flags(case) == expected


def _modulation_guard():
    """tata_v4's pause modulation guard, parsed from the grid file as the loop parses it."""
    return reward.parse_constraints(_grid(TATA_V4)["constraints"])[MODULATION]


def test_a_pause_texture_change_inside_the_accepted_range_passes_the_modulation_guard():
    """The first ear v4 APL round: the per-frame fill took Tele7abc to 2.32; the shelf fill reads 6.74 (2026-10-17)."""
    assert not _modulation_guard().trips(6.74, 2.32)


def test_past_the_bound_a_worse_pause_texture_still_trips_it():
    """Keeper off (11.47) against the shipped fill (4.64), and a small creep past the bound."""
    guard = _modulation_guard()
    assert (guard.trips(11.47, 4.64), guard.trips(11.5, 10.1)) == (True, True)


# The soft programme kept (`dsp.hiss_prog_kept_soft_db.output.median`, 2026-10-17; docs/ear_v4_hiss.md, "The Soft Programme
# Kept"): every judged file's reading and the reference suppressor's replays, through the runner's pair, anchor and
# windows (15 s, hop 7.5 s) on the GCC-PHAT aligner. Numbers only; the family tapes stay on the machine.
SOFT_PINNED = {
    "known_tele/stitched": -16.92,
    "known_tele/single4s": -27.46,
    "known_tele/deesser_bug": -28.62,
    "known_tele/cathar075": -11.28,
    "known_tele/apl": -11.29,
    "known_soti/stitched": -22.44,
    "known_soti/single4s": -26.09,
    "known_soti/deesser_bug": -33.91,
    "known_soti/cathar075": -14.64,
    "known_vaccin/stitched": -21.81,
    "known_vaccin/single4s": -26.51,
    "known_vaccin/deesser_bug": -30.00,
    "known_vaccin/cathar075": -12.14,
    "v4air_tele7abc/A_as_heard": -1.53,
    "v4air_tele7abc/B_air_1dB": -1.78,
    "v4air_tele7abc/C_air_off": -2.04,
    "v4air_soti/A_as_heard": -3.23,
    "v4air_soti/B_air_1dB": -3.43,
    "v4air_soti/C_air_off": -3.63,
    "v4air_vaccin/A_as_heard": -2.35,
    "v4air_vaccin/B_air_1dB": -2.48,
    "v4air_vaccin/C_air_off": -2.61,
    "v4air_gaudeamus5/A_as_heard": -9.00,
    "v4air_gaudeamus5/B_air_1dB": -9.07,
    "v4air_gaudeamus5/C_air_off": -9.19,
    "v2_x_the-adventures-of-captain-pu/final2_apl": -27.32,
    "v2_x_the-adventures-of-captain-pu/final2_cathar": -0.42,
    "replay_tele7abc/v1_frame_shelf": -1.85,
    "replay_tele7abc/w_f_qfix": -6.03,
    "replay_tele7abc/c1_frame_shelf": -5.76,
    "replay_tele7abc/c4_frame_sh2k30_ref": -10.41,
    "replay_soti/v1_frame_shelf": -3.47,
    "replay_soti/w_f_qfix": -7.55,
    "replay_soti/c1_frame_shelf": -8.42,
    "replay_soti/c4_frame_sh2k30_ref": -12.83,
    "replay_vaccin/v1_frame_shelf": -2.79,
    "replay_vaccin/w_f_qfix": -7.00,
    "replay_vaccin/c1_frame_shelf": -6.12,
    "replay_vaccin/c4_frame_sh2k30_ref": -9.22,
    "replay_gaudeamus5/v1_frame_shelf": -9.53,
    "replay_gaudeamus5/w_f_qfix": -16.49,
    "replay_gaudeamus5/c1_frame_shelf": -6.57,
    "replay_gaudeamus5/c4_frame_sh2k30_ref": -14.36,
}


def _soft_trips(candidate, incumbent, grid=TATA_V4):
    """Whether the grid's soft guard excludes `candidate` against `incumbent`, both pinned soft readings."""
    guard = reward.parse_constraints(_grid(grid)["constraints"])[KEPT_SOFT]
    return guard.trips(SOFT_PINNED[candidate], SOFT_PINNED[incumbent])


@pytest.mark.parametrize("tape", ["tele", "soti", "vaccin"])
def test_the_de_esser_bug_trips_the_soft_guard(tape):
    """'Speech from under water' on every tape: the de-esser bug reads -28.6..-33.9 against stitched, past the -27 bound."""
    assert _soft_trips(f"known_{tape}/deesser_bug", f"known_{tape}/stitched")


def test_single4s_trips_the_soft_guard_only_where_it_was_heard_underwater():
    """'Underwater again' (Tele7abc, -27.46) trips; SOTI's and Vaccin's single4s, ranked first (-26.09, -26.51), do not."""
    assert _soft_trips("known_tele/single4s", "known_tele/stitched")
    assert not _soft_trips("known_soti/single4s", "known_soti/stitched")
    assert not _soft_trips("known_vaccin/single4s", "known_vaccin/stitched")


@pytest.mark.parametrize("tape", ["tele7abc", "soti", "vaccin", "gaudeamus5"])
def test_the_air_shelf_round_passes_the_soft_guard(tape):
    """Round three preferred B (+1 dB air) over A (+2 dB) and C (off): each reads 0.07-0.26 dB under A, inside the bound."""
    assert not _soft_trips(f"v4air_{tape}/B_air_1dB", f"v4air_{tape}/A_as_heard")
    assert not _soft_trips(f"v4air_{tape}/C_air_off", f"v4air_{tape}/B_air_1dB")


def test_the_accepted_music_finals_pass_the_music_soft_guard():
    """'Music fine on both': the captain-pu APL final reads -27.32 (one window) against its cathar final's -0.42; music_v4's
    bound, -28.0, admits it where tata_v4's -27.0 would not."""
    candidate, incumbent = "v2_x_the-adventures-of-captain-pu/final2_apl", "v2_x_the-adventures-of-captain-pu/final2_cathar"
    assert not _soft_trips(candidate, incumbent, "music_v4.yaml")
    assert _soft_trips(candidate, incumbent)


def test_past_the_bound_a_soft_loss_beyond_jitter_trips_it():
    """Past -27 the guard does not follow the incumbent: a worse candidate more than 0.05 dB past it trips, one within the
    jitter of the bound does not, and inside the bound no move trips (the reference suppressor's replays stay there)."""
    guard = reward.parse_constraints(_grid(TATA_V4)["constraints"])[KEPT_SOFT]
    assert (guard.trips(-27.2, -27.1), guard.trips(-27.03, -27.01), guard.trips(-20.0, -1.8)) == (True, False, False)


@pytest.mark.parametrize("tape", ["tele7abc", "soti", "vaccin", "gaudeamus5"])
@pytest.mark.parametrize(("without", "with_it"), [("v1_frame_shelf", "w_f_qfix"), ("c1_frame_shelf", "c4_frame_sh2k30_ref")])
def test_the_reference_suppressor_takes_soft_programme_inside_the_bound(tape, without, with_it):
    """The suppressor takes the soft programme 3.1-7.8 dB down (APL and cathar replays), but no variant reaches -27: the
    bound set by the judged files does not exclude it; that question is the listening session's."""
    candidate, incumbent = f"replay_{tape}/{with_it}", f"replay_{tape}/{without}"
    assert SOFT_PINNED[candidate] <= SOFT_PINNED[incumbent] - 3.0
    assert not _soft_trips(candidate, incumbent)
