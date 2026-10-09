"""Ear v3 grids: reading specs ranked as two-sided distances, the v3 grid files, the listening metrics."""

from pathlib import Path

import pytest

from scripts import tune_restoration as tr
from scripts.restoration_quality import reward
from scripts.restoration_quality.scorecard import METRICS, TWO_SIDED

GRIDS = Path(tr.REPO) / "scripts" / "tune_grids"
SPEC = {"target": 1.0, "dead_zone": 0.5, "scale": 0.25, "weight": 2.0, "family": "timbre"}
LEARNED = ("speech.", "mos.", "stems.mert_dist")
# Round 0 (docs/ear_v3_round0.md; experiments/tata_listen/scores_v3, the aggregate medians a grid reads): per entry,
# readings of files the user accepted (inside the dead zone) and of flagged or dead-air files (charged).
ROUND0 = [
    ("tata_v3.yaml", "dsp.balance_tilt_db_oct.output.median", (-2.145, -1.648, 0.099), (-6.054, -13.25)),
    ("tata_v3.yaml", "dsp.balance_presence_db.output.median", (-2.108, -2.039, 0.13), (-4.668, -9.709)),
    ("tata_v3.yaml", "dsp.balance_body_db.output.median", (-2.16, 1.771, -2.234), ()),
    ("tata_v3.yaml", "dsp.sib_abs_level_db.output.median", (-3.128, 3.612, 0.625, 0.14), (-17.279, -21.788)),
    ("tata_v3.yaml", "dsp.sib_texture_db.output.median", (1.343, 0.334), (2.122, 2.257, 3.489)),
    ("tata_v3.yaml", "dsp.pause_depth_db.delta.median", (3.99, 13.734, 32.975, 33.234), (0.0, 2.9, 36.888, 37.59, 40.306)),
    ("tata_v3.yaml", "dsp.gap_slope_db_oct.output.median", (1.512, 1.086, -0.49, 1.913, 1.925), (2.146, 3.012, -3.768)),
    ("tata_v3.yaml", "dsp.gap_island_kurt.output.median", (1.736, 0.617), ()),
    ("tata_v3.yaml", "file.gain_ride_lu.output.median", (2.257, 2.423, 0.225), ()),
    ("music_v3.yaml", "dsp.balance_presence_db.output.median", (-0.61, 0.41), ()),
]
# The one entry Round 0 kept at its starting value inside the rule's one-scale-unit margin (1.736 inside 1.8 by 0.064).
UNMARGINED = ("dsp.gap_island_kurt.output.median",)
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
    """Every accepted file of Round 0 sits inside its dead zone; the de-esser bugs, the 's' distortion and dead air do not."""
    spec = reward.parse_grid({reading: _grid(name)["ranking"][reading]})[reading]
    assert [spec.distance(value) for value in admitted] == [0.0] * len(admitted)
    assert all(spec.distance(value) > 0.0 for value in charged)


@pytest.mark.parametrize(("name", "reading", "admitted", "_charged"), [row for row in ROUND0 if row[1] not in UNMARGINED])
def test_a_re_derived_entry_admits_each_accepted_file_by_at_least_one_scale_unit(name, reading, admitted, _charged):
    """The grid's stated rule: every accepted reading sits at least one `scale` inside the dead zone, not just inside it."""
    entry = _grid(name)["ranking"][reading]
    margins = [entry["dead_zone"] - abs(value - entry["target"]) for value in admitted]
    assert min(margins) >= entry["scale"] - 1e-9


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
