"""The confirmation tabulator: strict trips as the loop judges, material trips past the allowance, split by group."""

import json

import pytest

from scripts import confirm_trips
from scripts.restoration_quality import reward

GRID = """constraints:
  dsp.hiss_pump_db.output.median: {worse: up, max_move: 3.0, bound: 12.0}
  dsp.hiss_prog_kept_hf_db.output.median: {worse: down, max_move: 1.0, bound: -6.0}
  dsp.balance_tilt_db_oct.output.median: {worse: down, bound: -2.25}
"""


def _variant(pump, kept, tilt=-1.0):
    aggregate = {
        "dsp.hiss_pump_db": {"output": {"median": pump}},
        "dsp.hiss_prog_kept_hf_db": {"output": {"median": kept}},
        "dsp.balance_tilt_db_oct": {"output": {"median": tilt}},
    }
    return {"aggregate": aggregate, "hard_failures": [], "passed": True}


def _report(folder, clip, incumbent, candidate, label="W"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{clip}.json").write_text(json.dumps({"variants": {"S": incumbent, label: candidate}}), encoding="utf-8")


@pytest.fixture(name="setup")
def _setup(tmp_path):
    grid = tmp_path / "grid.yaml"
    grid.write_text(GRID, encoding="utf-8")
    reports = tmp_path / "reports"
    _report(reports, "gated", _variant(-2.0, -1.0), _variant(20.0, -1.0))
    _report(reports, "already_dull", _variant(0.0, -11.71), _variant(0.0, -11.72))
    _report(reports, "clean", _variant(0.0, -1.0), _variant(1.0, -1.2))
    states = tmp_path / "states.tsv"
    states.write_text("refused\tgated\nrefused\talready_dull\nruns\tclean\n", encoding="utf-8")
    return grid, reports, states


def test_a_move_past_a_broken_bound_trips_the_loop_but_is_not_material(setup):
    grid, reports, _states = setup
    constraints = confirm_trips.load_constraints(grid)
    result = json.loads((reports / "already_dull.json").read_text(encoding="utf-8"))
    assert confirm_trips.clip_trips(result, "S", "W", constraints) == [("dsp.hiss_prog_kept_hf_db.output.median", -11.71, -11.72, False)]
    gated = json.loads((reports / "gated.json").read_text(encoding="utf-8"))
    assert confirm_trips.clip_trips(gated, "S", "W", constraints) == [("dsp.hiss_pump_db.output.median", -2.0, 20.0, True)]


def test_a_bound_alone_is_material_past_the_floor():
    tilt = reward.parse_constraints({"t": {"worse": "down", "bound": -2.25}})["t"]
    assert (confirm_trips.is_material(tilt, -3.0, -3.05), confirm_trips.is_material(tilt, -3.0, -3.2)) == (False, True)


def test_the_trips_split_by_group_and_count_material_apart(setup):
    grid, reports, states = setup
    counts, listed = confirm_trips.tabulate([("W", reports)], "S", confirm_trips.load_constraints(grid), confirm_trips.load_states(states))
    refused, runs = counts[("W", "refused")], counts[("W", "runs")]
    assert (refused["clips"], refused["tripped"], refused["material"]) == (2, 2, 1)
    assert (runs["clips"], runs["tripped"]) == (1, 0)
    assert set(listed[("W", "refused")]) == {"gated", "already_dull"}


def test_only_clips_every_folder_holds_are_counted_and_unmapped_ones_group_apart(tmp_path, setup):
    grid, reports, _states = setup
    other = tmp_path / "other"
    _report(other, "gated", _variant(-2.0, -1.0), _variant(-2.0, -1.0), label="F")
    candidates = [("W", reports), ("F", other)]
    assert confirm_trips.common_clips(candidates) == ["gated"]
    counts, _listed = confirm_trips.tabulate(candidates, "S", confirm_trips.load_constraints(grid), {})
    assert (counts[("W", "?")]["clips"], counts[("F", "?")]["tripped"]) == (1, 0)


def test_a_missing_reading_trips_strictly_but_never_materially(tmp_path, setup):
    grid, _reports, _states = setup
    folder = tmp_path / "missing"
    candidate = _variant(0.0, -1.0)
    del candidate["aggregate"]["dsp.hiss_prog_kept_hf_db"]
    _report(folder, "lost", _variant(0.0, -1.0), candidate)
    result = json.loads((folder / "lost.json").read_text(encoding="utf-8"))
    rows = confirm_trips.clip_trips(result, "S", "W", confirm_trips.load_constraints(grid))
    assert [(key, material) for key, _a, _b, material in rows] == [("dsp.hiss_prog_kept_hf_db.output.median", False)]


def test_the_command_prints_the_table_and_with_verbose_every_tripped_clip(setup, capsys):
    grid, reports, states = setup
    argv = ["--grid", str(grid), "--incumbent", "S", "--candidate", f"W={reports}", "--states", str(states)]
    assert confirm_trips.main(argv) == 0
    plain = capsys.readouterr().out
    assert "clips held by every report: 3" in plain and "W      refused    trips   2 material   1 /   2" in plain
    confirm_trips.main([*argv, "--verbose"])
    verbose = capsys.readouterr().out
    assert "*hiss_pump_db -2.00 -> 20.00" in verbose and "hiss_prog_kept_hf_db -11.71 -> -11.72" in verbose


@pytest.mark.parametrize("text", ["W", "=dir", "W="])
def test_a_candidate_names_its_label_and_its_folder(text):
    with pytest.raises(confirm_trips.argparse.ArgumentTypeError):
        confirm_trips.parse_candidate(text)


def test_without_a_states_file_every_clip_is_unmapped():
    assert confirm_trips.load_states(None) == {}
