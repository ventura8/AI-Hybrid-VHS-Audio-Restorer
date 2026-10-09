"""The ranking grid of the known-ordering set: where a grid file keeps its readings, and a grid that names none."""

import json
from pathlib import Path

import pytest

from scripts import calibrate_quality_metrics as cal
from scripts import calibration_ordering as co
from tests.unit.test_calibration_ordering import AIR, _variant

REPO = Path(co.__file__).resolve().parent.parent


def test_a_grid_file_supplies_the_targets(tmp_path):
    """A grid file supplies the targets."""
    path = tmp_path / "grid.yaml"
    path.write_text("reward:\n  dsp.balance_air_db.delta.median: {target: 1.0, family: timbre, dead_zone: 0.25}\nexcerpt_s: 125\n", "utf-8")
    grid = co.load_grid(path)
    assert grid == {"dsp.balance_air_db.delta.median": {"target": 1.0, "family": "timbre", "dead_zone": 0.25}}
    result = {"variants": {"plus1": _variant(readings={AIR: 1.1}), "plus2": _variant(readings={AIR: 1.9})}}
    assert co.harness_ranking(result, {}, grid) == ["plus1", "plus2"]


def test_a_grid_file_without_a_reward_section_reads_its_target_entries(tmp_path):
    """A grid file without a reward section reads its target entries."""
    path = tmp_path / "grid.json"
    path.write_text(json.dumps({"dsp.lkr.delta.median": {"target": 0.0, "family": "artefacts"}, "variants": {"a": {}}}), "utf-8")
    assert list(co.load_grid(path)) == ["dsp.lkr.delta.median"]


def test_a_grid_file_reads_its_ranking_section(tmp_path):
    """The v3 tuning grids keep their readings under `ranking:`."""
    path = tmp_path / "grid.yaml"
    path.write_text("excerpt_s: 125\nranking:\n  dsp.lkr.delta.median: {target: 0.0, family: artefacts}\nvetoes: {}\n", "utf-8")
    assert list(co.load_grid(path)) == ["dsp.lkr.delta.median"]


@pytest.mark.parametrize(("name", "count"), [("tata_v3.yaml", 15), ("music_v3.yaml", 10)])
def test_the_v3_tuning_grids_load_their_ranking(name, count):
    """`--grid tata_v3.yaml` ranks on its own readings, not on the default grid."""
    grid = co.load_grid(REPO / "scripts" / "tune_grids" / name)
    assert len(grid) == count and "dsp.balance_air_db.output.median" in grid


def test_a_grid_file_without_entries_is_refused(tmp_path):
    """A grid that names nothing would rank on the default grid without a word: the run stops instead."""
    path = tmp_path / "grid.yaml"
    path.write_text("rewards:\n  dsp.lkr.delta.median: {target: 0.0, family: artefacts}\n", "utf-8")
    with pytest.raises(SystemExit, match="no ranking entries"):
        co.load_grid(path)


def test_an_explicit_empty_grid_ranks_nothing():
    """Only a missing grid means the default one."""
    result = {"variants": {"a": _variant(readings={AIR: 1.0})}}
    assert co.harness_scores(result, {}, {}) == {"a": None}


def test_a_bad_grid_stops_the_calibration_before_the_synthetic_suite(tmp_path, monkeypatch):
    """The refusal comes before hours of scoring every family, not after them."""
    path = tmp_path / "grid.yaml"
    path.write_text("rewards:\n  dsp.lkr.delta.median: {target: 0.0, family: artefacts}\n", "utf-8")
    monkeypatch.setattr(cal, "synthetic", lambda *_args: pytest.fail("the synthetic suite ran before the grid was read"))
    with pytest.raises(SystemExit, match="no ranking entries"):
        cal.main(["--grid", str(path), "--out", str(tmp_path / "out"), "--no-ledger"])
