"""The loop's scoring families: with `--families auto` only the families the grid and the gates read are scored."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import autotune_restoration as at
from scripts import tune_restoration as tune

GRIDS = Path(at.__file__).resolve().parent / "tune_grids"


def test_a_reading_belongs_to_its_prefix_family_or_to_the_dsp_pass():
    """file.*, listener.*, src.* and bare names come from the dsp pass; the four scorer families from themselves."""
    assert [at.reading_family(key) for key in ("mos.x", "speech.cer", "stems.mert_dist", "dsp.lkr")] == ["mos", "speech", "stems", "dsp"]
    assert {at.reading_family(key) for key in ("file.gain_ride_lu.output.median", "listener.hiss", "src.capture", "m")} == {"dsp"}


def test_the_speech_grid_skips_the_stems_family_and_the_music_grid_skips_speech():
    """tata_v3 ranks dsp and file readings and vetoes on speech and mos; music_v3 ranks stems and vetoes on stems and mos."""
    assert at.grid_families(tune.load_grid(GRIDS / "tata_v3.yaml")) == ("dsp", "speech", "mos")
    assert at.grid_families(tune.load_grid(GRIDS / "music_v3.yaml")) == ("dsp", "stems", "mos")


def test_dsp_is_always_scored_and_a_gate_adds_its_family():
    """An empty grid still scores dsp (every gate reads it); a learned gate brings its family in, in scoring order."""
    assert at.grid_families({}) == ("dsp",)
    assert at.grid_families({"ranking": {"m": "up"}, "vetoes": None}, ["mos.dnsmos", "listener.hiss"]) == ("dsp", "mos")


def test_auto_reads_the_gates_file_and_a_named_list_is_kept(tmp_path):
    """`auto` resolves from the grid and the gates file; any other value is scored exactly as given."""
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps({"dsp.hum": {}, "speech.cer": {}}), encoding="utf-8")
    grid = {"ranking": {"dsp.lkr.delta.median": "down"}}
    assert at.resolve_families(("auto",), grid, gates) == ("dsp", "speech")
    assert at.resolve_families(("auto",), grid, None) == ("dsp",)
    assert at.resolve_families(("dsp", "stems", "speech", "mos"), grid, gates) == ("dsp", "stems", "speech", "mos")


def test_a_missing_gates_file_stops_auto_with_its_name(tmp_path):
    """The gates path is checked where `auto` reads it."""
    with pytest.raises(SystemExit, match="--gates does not exist"):
        at.resolve_families(("auto",), {}, tmp_path / "missing.json")


def _cli(tmp_path, *extra):
    tapes = tmp_path / "tapes.json"
    tapes.write_text(json.dumps({"t1": "t1.mov"}), encoding="utf-8")
    grid = tmp_path / "grid.yaml"
    grid.write_text("ranking:\n  m: up\n  stems.lsd_db.output.median: down\n", encoding="utf-8")
    base = ["--engine", "apl", "--tapes", str(tapes), "--grid", str(grid), "--out", str(tmp_path / "out"), "--gates", str(_gates(tmp_path))]
    return [*base, *extra]


def _gates(tmp_path):
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps({"dsp.hum": {}}), encoding="utf-8")
    return gates


def _run(monkeypatch, tmp_path, *extra):
    """The command line with its rounds stood in for; returns the families the first round was given."""
    rounds = Mock(return_value=False)
    monkeypatch.setattr(at, "run_round", rounds)
    monkeypatch.setattr(at, "seed_defaults", lambda _engine, start, _out: start)
    monkeypatch.setattr(at.tune, "probe_duration", lambda _path: None)
    at.main(_cli(tmp_path, *extra))
    return rounds.call_args.args[0].families


def test_the_command_line_scores_the_grids_families_by_default(tmp_path, monkeypatch, capsys):
    """No --families: the round is given the grid's families and the run says which."""
    assert _run(monkeypatch, tmp_path) == ("dsp", "stems")
    assert "scoring families: dsp,stems" in capsys.readouterr().out


def test_the_command_line_keeps_a_named_family_list(tmp_path, monkeypatch):
    """--families dsp,stems,speech,mos (the old default) is scored as given."""
    assert _run(monkeypatch, tmp_path, "--families", "dsp,stems,speech,mos") == ("dsp", "stems", "speech", "mos")
