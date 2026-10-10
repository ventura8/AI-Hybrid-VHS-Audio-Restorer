"""Ear v3 grids: reading specs ranked as two-sided distances, the v3 grid files, the listening metrics."""

import math
from pathlib import Path

import pytest

from scripts import tune_restoration as tr
from scripts.restoration_quality import reward
from scripts.restoration_quality.scorecard import METRICS, TWO_SIDED

GRIDS = Path(tr.REPO) / "scripts" / "tune_grids"
SPEC = {"target": 1.0, "dead_zone": 0.5, "scale": 0.25, "weight": 2.0, "family": "timbre"}
LEARNED = ("speech.", "mos.", "stems.mert_dist")
# Round 0 re-scored with the calibration v3 readings (docs/ear_v3_round0.md, "The grid after calibration v3";
# experiments/tata_listen/scores_v3, the aggregate medians a grid reads): per entry, readings of files the user accepted
# (inside the dead zone; the furthest first) and of flagged or dead-air files, or accepted files a comment charges on purpose
# (charged: S alpha 2 and T known apl on the slope, the known cathar075 on modulation).
ROUND0 = [
    ("tata_v3.yaml", "dsp.balance_tilt_db_oct.output.median", (-2.145, -1.648, 0.099), (-6.054, -13.25)),
    ("tata_v3.yaml", "dsp.balance_presence_db.output.median", (-2.108, -2.039, 0.13), (-4.668, -9.709)),
    ("tata_v3.yaml", "dsp.balance_body_db.output.median", (-2.234, -2.16, 1.771), ()),
    ("tata_v3.yaml", "dsp.sib_abs_level_db.output.median", (-3.306, 2.176, 0.967, 0.624, 0.135), (-17.435, -21.029, -22.196)),
    ("tata_v3.yaml", "dsp.sib_texture_db.output.median", (1.652, 1.601, 1.329, 0.471), (2.375, 2.416, 2.42, 3.665)),
    ("tata_v3.yaml", "dsp.pause_depth_db.delta.median", (3.99, 13.734, 32.975, 33.234), (0.0, 2.9, 36.888, 37.59, 40.306)),
    (
        "tata_v3.yaml",
        "dsp.gap_slope_db_oct.output.median",
        (1.913, 1.775, 1.512, 1.11, 1.086, 1.055, -0.49),
        (3.012, -3.768, -4.41, -4.734),
    ),
    ("tata_v3.yaml", "dsp.gap_spread_db.output.median", (6.301, 6.072, 4.488, 4.471, 3.843, 1.559), ()),
    ("tata_v3.yaml", "dsp.gap_lsd_db.output.median", (8.61, 8.28, 7.569, 5.683, 4.688, 4.246, 2.645), (9.163, 9.248, 9.61, 10.884)),
    ("tata_v3.yaml", "dsp.gap_mod_dist_db.output.median", (9.866, 9.863, 9.585, 9.229, 9.097), (12.59, 12.683, 13.07, 10.186)),
    ("tata_v3.yaml", "dsp.gap_island_kurt.output.median", (1.736, 1.734, 1.579, 0.623), ()),
    ("tata_v3.yaml", "file.gain_ride_lu.output.median", (2.423, 2.257, 0.225), ()),
    ("music_v3.yaml", "dsp.balance_presence_db.output.median", (-0.61, 0.41), ()),
]
# The entries whose dead zone is the rule's own value: the smallest 0.05 step that admits the furthest accepted reading of the
# row above by one scale unit. Texture sits at a midpoint, pause depth is a band, and the slope and modulation keep values that
# charge accepted files on purpose (their comments say which).
RULE_DERIVED = (
    "dsp.balance_tilt_db_oct.output.median",
    "dsp.balance_presence_db.output.median",
    "dsp.balance_body_db.output.median",
    "dsp.sib_abs_level_db.output.median",
    "dsp.gap_spread_db.output.median",
    "dsp.gap_lsd_db.output.median",
    "dsp.gap_island_kurt.output.median",
    "file.gain_ride_lu.output.median",
)
TEXTURE = "dsp.sib_texture_db.output.median"
# The same re-score, Vaccin's round-one tape ("alpha 2 sounds better"): pause depth delta, spread and LSD of each file.
PAUSE_SHAPE = ("dsp.pause_depth_db.delta.median", "dsp.gap_spread_db.output.median", "dsp.gap_lsd_db.output.median")
VACCIN_ALPHA2 = {
    "cathar__alpha_2_0": (38.676, 6.301, 7.569),
    "cathar__baseline": (40.333, 4.488, 5.683),
    "apl__baseline": (45.389, 2.877, 4.183),
}
# Vaccin's known-ordering excerpt (single4s and stitched ranked first, cathar075 second), the same re-score: the four pause
# readings that charge any of the three (pause depth delta, spread, LSD, modulation).
VACCIN_KNOWN_READINGS = PAUSE_SHAPE + ("dsp.gap_mod_dist_db.output.median",)
VACCIN_KNOWN = {
    "single4s": (38.002, 4.471, 4.688, 6.599),
    "stitched": (40.333, 4.488, 5.683, 8.304),
    "cathar075": (27.187, 6.072, 8.28, 12.59),
}
# The one entry kept inside the rule's one-scale-unit margin, a known exception (the yaml header): modulation 10.0 admits eight
# accepted files by 0.134-0.903 of its 1.2, because the rule's 14.3 would tie Soti's known ranking.
UNMARGINED = ("dsp.gap_mod_dist_db.output.median",)
# Round 0's grid notes: each of these reversed a verdict or charged accepted files; they are shown, not ranked.
DROPPED = ("dsp.balance_air_db", "dsp.sib_centroid_hz", "dsp.gap_atten_db", "dsp.gap_hf_excess_db")


def _summary(**medians):
    return {"metrics": {key: {"median": value, "p10": value} for key, value in medians.items()}}


def test_a_direction_ranks_its_values_as_they_stand():
    """The v1/v2 grids' "up" / "down" entries are unchanged and weigh 1."""
    assert tr.ranked_values("k", [1.0, None], "up") == ([1.0, None], "up", 1.0)


def test_a_reading_spec_ranks_its_two_sided_distance_with_its_weight():
    """Inside the dead zone reads 0 (a tie); outside, the distance in scale units, either side of the target."""
    values, direction, weight = tr.ranked_values("k", [1.2, 0.0, 2.0, None], SPEC)
    assert values == [0.0, 2.0, 2.0, None]
    assert (direction, weight) == ("down", 2.0)


def test_a_bad_reading_spec_is_refused_by_name():
    """An entry without a target cannot rank anything."""
    with pytest.raises(ValueError, match="grid entry k: missing target"):
        tr.ranked_values("k", [1.0], {"family": "timbre"})


def test_rank_variants_ties_everything_inside_the_dead_zone():
    """Brighter is not better: +0.6 and +1.4 sit inside the band around +1.0 and tie; +3.0 ranks last."""
    summaries = {"a": _summary(k=0.6), "b": _summary(k=1.4), "c": _summary(k=3.0)}
    assert tr.rank_variants(summaries, {"k": SPEC}) == {"a": 1.5, "b": 1.5, "c": 3.0}


def test_rank_variants_weighs_each_entry_in_the_mean_rank():
    """A weight-2 reading counts twice against a weight-1 direction."""
    summaries = {"a": _summary(k=1.0, m=0.0), "b": _summary(k=3.0, m=1.0)}
    ranks = tr.rank_variants(summaries, {"k": SPEC, "m": "up"})
    assert ranks == {"a": pytest.approx((1.0 * 2 + 2.0) / 3), "b": pytest.approx((2.0 * 2 + 1.0) / 3)}


def test_rank_variants_leaves_out_a_reading_nobody_read():
    """An unread spec ties no one and leaves a variant with nothing ranked at None."""
    summaries = {"a": _summary(), "b": _summary()}
    assert tr.rank_variants(summaries, {"k": SPEC}) == {"a": None, "b": None}


def test_the_best_variant_on_a_tape_reads_reading_specs_too():
    """The per-tape winner is the one closest to the target, not the one furthest in a direction."""
    summaries = {
        label: {"per_tape": {"soti": {"k": value}}, "per_tape_vetoed": {}} for label, value in (("cathar__x", 3.0), ("apl__y", 0.9))
    }
    assert tr.best_engine_per_tape(summaries, {"k": SPEC}) == {"soti": {"variant": "apl__y", "engine": "apl", "rank_score": 1.0}}


def _grid(name):
    return tr.load_grid(GRIDS / name)


def _metric(key):
    return key.rsplit(".", 2)[0]


@pytest.mark.parametrize("name", ["tata_v3.yaml", "music_v3.yaml"])
def test_every_v3_ranking_entry_is_a_valid_reading_spec_on_a_registered_metric(name):
    """The reward module parses the very entries tune_restoration ranks; every key is a metric.side.stat it scores."""
    ranking = _grid(name)["ranking"]
    assert set(reward.parse_grid(ranking)) == set(ranking)
    assert all(
        _metric(key) in METRICS
        and key.rsplit(".", 2)[1:] in (["output", "median"], ["output", "tail"], ["delta", "median"], ["delta", "tail"])
        for key in ranking
    )


@pytest.mark.parametrize("name", ["tata_v3.yaml", "music_v3.yaml"])
def test_the_v3_grids_rank_no_learned_judge_and_no_one_sided_hf_band(name):
    """Learned judges veto only; hf_4k8k / hf_8k16k stay backstop gates; noise removed is shown, not ranked."""
    grid = _grid(name)
    assert not [key for key in grid["ranking"] if key.startswith(LEARNED)]
    assert not [key for key in grid["ranking"] if _metric(key) in ("dsp.hf_4k8k", "dsp.hf_8k16k", "dsp.residual_noise_db")]
    assert all(_metric(key).startswith(LEARNED) and entry["worse"] in ("up", "down") for key, entry in grid["vetoes"].items())


def test_the_speech_grid_weighs_the_speech_side_one_and_a_half():
    """Timbre and sibilance carry weight 1.5 [S], the noise side 1.0."""
    ranking = _grid("tata_v3.yaml")["ranking"]
    weights = {entry["family"]: entry["weight"] for entry in ranking.values()}
    assert weights == {"timbre": 1.5, "sibilance": 1.5, "pauses": 1.0, "level": 1.0}


@pytest.mark.parametrize(("name", "reading", "admitted", "charged"), ROUND0)
def test_a_v3_grid_admits_what_the_user_accepted_and_charges_what_was_flagged(name, reading, admitted, charged):
    """Every accepted file of Round 0 sits inside its dead zone; the de-esser bugs, the 's' distortion, dead air and hiss do not."""
    spec = reward.parse_grid({reading: _grid(name)["ranking"][reading]})[reading]
    assert [spec.distance(value) for value in admitted] == [0.0] * len(admitted)
    assert all(spec.distance(value) > 0.0 for value in charged)


@pytest.mark.parametrize(("name", "reading", "admitted", "_charged"), [row for row in ROUND0 if row[1] not in UNMARGINED])
def test_a_re_derived_entry_admits_each_accepted_file_by_at_least_one_scale_unit(name, reading, admitted, _charged):
    """The grid's stated rule: every accepted reading sits at least one `scale` inside the dead zone, not just inside it."""
    entry = _grid(name)["ranking"][reading]
    margins = [entry["dead_zone"] - abs(value - entry["target"]) for value in admitted]
    assert min(margins) >= entry["scale"] - 1e-9


@pytest.mark.parametrize(("name", "reading", "admitted", "_charged"), [row for row in ROUND0 if row[1] in RULE_DERIVED])
def test_a_rule_derived_dead_zone_is_the_smallest_step_admitting_the_furthest_accepted_file_by_one_unit(name, reading, admitted, _charged):
    """sib_abs 3.306 + 0.05 -> 3.4, spread 6.301 + 0.5 -> 6.85, LSD 8.610 + 0.44 -> 9.05, kurtosis 1.736 + 0.1 -> 1.85.

    The margin test alone lets a value drift above the rule (the old sib_abs 3.7 still passed it); this one holds each to the step.
    """
    entry = _grid(name)["ranking"][reading]
    furthest = max(abs(value - entry["target"]) for value in admitted)
    assert entry["dead_zone"] == pytest.approx(math.ceil(round((furthest + entry["scale"]) / 0.05, 6)) * 0.05)


def test_the_texture_dead_zone_sits_midway_between_the_accepted_top_and_the_apl_bottom():
    """2.0: the midpoint of T's ranked-first cathar075__alpha_2_0 (+1.652) and the lowest 's'-distortion APL render (+2.375)."""
    admitted, charged = next(row[2:] for row in ROUND0 if row[1] == TEXTURE)
    entry = _grid("tata_v3.yaml")["ranking"][TEXTURE]
    assert entry["dead_zone"] == pytest.approx(round((max(admitted) + min(charged)) / 2 / 0.05) * 0.05)


def test_vaccins_known_ranking_stays_reversed_by_pause_depths_mixed_route_aggregate():
    """A known reversal (the yaml header): the ranked-first single4s / stitched sum 3.00 / 5.33 against cathar075's 2.16.

    Round 0's spread and LSD charged cathar075 more on the pre-fix reports (17.76 / 22.21 against 24.68); the rule's values admit
    it, and no dead zone under the rule restores the order. A speech-only pause-depth aggregate would; until it exists this pins
    the reversal, so a grid change that moves it is seen.
    """
    ranking = _grid("tata_v3.yaml")["ranking"]
    grid = {reading: ranking[reading] for reading in VACCIN_KNOWN_READINGS}
    totals = {
        label: sum(reward.distances(dict(zip(VACCIN_KNOWN_READINGS, values)), grid).values()) for label, values in VACCIN_KNOWN.items()
    }
    assert totals["cathar075"] < totals["single4s"] < totals["stitched"]
    assert totals == pytest.approx({"single4s": 3.002, "stitched": 5.333, "cathar075": 2.158}, abs=1e-3)


def test_the_re_derived_pause_shape_keeps_vaccins_alpha_2_preference():
    """Spread and LSD admit the preferred render, so only pause depth's mixed-route charge orders the three: alpha 2 least.

    At the old 3.0 / 3.5 (set on the pre-fix readings) alpha 2 summed 19.52 against apl__baseline's 11.94 on these readings.
    """
    ranking = _grid("tata_v3.yaml")["ranking"]
    grid = {reading: ranking[reading] for reading in PAUSE_SHAPE}
    totals = {label: sum(reward.distances(dict(zip(PAUSE_SHAPE, values)), grid).values()) for label, values in VACCIN_ALPHA2.items()}
    assert totals["cathar__alpha_2_0"] < totals["cathar__baseline"] < totals["apl__baseline"]
    assert totals == pytest.approx({"cathar__alpha_2_0": 3.676, "cathar__baseline": 5.333, "apl__baseline": 10.389})


def test_a_pass_through_no_longer_ties_every_accepted_file():
    """An output identical to its source reads 0 on every speech entry; pause depth's low edge charges it 2.95."""
    ranking = _grid("tata_v3.yaml")["ranking"]
    found = reward.distances({reading: 0.0 for reading in ranking}, ranking)
    assert {reading: value for reading, value in found.items() if value} == pytest.approx({"dsp.pause_depth_db.delta.median": 2.95})


def test_the_speech_grid_ranks_none_of_the_readings_round_0_found_reversing_a_verdict():
    """Air (unread on linear tapes), the centroid (C over B), pause attenuation and the HF excess are shown, not ranked."""
    ranked = {_metric(key) for key in _grid("tata_v3.yaml")["ranking"]}
    assert not ranked & set(DROPPED)
    assert all(metric in METRICS for metric in DROPPED)


def test_every_speech_grid_entry_is_a_band_either_side_of_its_target():
    """Each entry states its target, dead zone and scale: a two-sided band, never a direction; pause depth spans 2.95..35 dB."""
    ranking = _grid("tata_v3.yaml")["ranking"]
    assert all({"target", "dead_zone", "scale", "weight", "family"} <= set(entry) and entry["dead_zone"] > 0 for entry in ranking.values())
    depth = ranking["dsp.pause_depth_db.delta.median"]
    assert (depth["target"] - depth["dead_zone"], depth["target"] + depth["dead_zone"]) == pytest.approx((2.95, 35.0))


def test_the_listening_metrics_are_window_readings_and_include_the_new_ones():
    """`listen` picks one window per metric: every entry is a registered per-window reading, R1/R2/R4 among them."""
    assert all(METRICS[name].family in ("dsp", "mos", "speech") for name in tr.LISTEN_METRICS)
    assert METRICS["dsp.balance_top_db"].better == TWO_SIDED
    assert {"dsp.balance_top_db", "dsp.sib_abs_level_db", "dsp.sib_texture_db", "dsp.gap_hf_excess_db"} <= set(tr.LISTEN_METRICS)


@pytest.mark.parametrize("name", ["tata_v3.yaml", "music_v3.yaml"])
def test_the_v3_grids_carry_the_round_three_air_boundary_and_the_audibility_tie(name):
    """The loop's guard sections parse; the air boundary names a knob and the ledger verdict it came from."""
    grid = _grid(name)
    (rule,) = reward.parse_reversals(grid["reversals"])
    assert (rule.key, rule.bound, rule.side, rule.verdict) == ("linear_air_gain_db", 2.0, 1.0, "r3-air-preference")
    assert grid["audibility"] == {"offset_db": 24.0}
