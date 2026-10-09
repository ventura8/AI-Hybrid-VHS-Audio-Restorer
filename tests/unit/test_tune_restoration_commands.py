"""The tuning driver's commands and the steps they run, with ffmpeg, the app and the scorer replaced by recorders."""

import json
import types
from pathlib import Path

import pytest
import yaml

from scripts import tune_restoration as tr
from scripts.restoration_quality import listening

GRID = {
    "excerpt_s": 125,
    "margin_s": 5,
    "engines": {"cathar": {"process_mode": "cathar", "suffix": "_Cathar_Cleaned"}},
    "gates": {"max_failed_excerpts": 1},
    "ranking": {"dsp.balance_top_db.output.median": {"target": 0.0, "dead_zone": 0.3, "scale": 0.1, "weight": 1.5, "family": "timbre"}},
    "variants": {"cathar": {"baseline": {}, "alpha": {"cathar_alpha": 2.0}}},
}
MANIFEST = {
    "tapes_dir": "D:/tapes",
    "config_sha256": "0" * 64,
    "excerpts": [{"excerpt": "soti_start.mov", "slug": "soti", "source_wav": "soti_start.src.wav", "probed_s": 124.98}],
}


def _completed(stdout):
    return types.SimpleNamespace(stdout=stdout)


def test_probe_duration_reads_ffprobe_and_none_when_it_cannot(monkeypatch):
    """A number is the duration; anything else (no file, no format) is None."""
    answers = iter(["12.5\n", "N/A\n"])
    monkeypatch.setattr(tr.subprocess, "run", lambda *_args, **_kwargs: _completed(next(answers)))
    assert tr.probe_duration("a.mov") == 12.5
    assert tr.probe_duration("b.mov") is None


def test_cut_excerpt_copies_the_audio_stream_bit_for_bit(monkeypatch, tmp_path):
    """The excerpt starts where asked and keeps the original audio (`-c:a copy`)."""
    calls = []
    monkeypatch.setattr(tr.subprocess, "run", lambda cmd, **_kwargs: calls.append(cmd))
    assert tr.cut_excerpt("tape.mov", tmp_path / "x.mov", 12.0, 125.0) == tmp_path / "x.mov"
    assert calls[0][calls[0].index("-ss") + 1] == "12.000"
    assert "copy" in calls[0]


def _fake_media(monkeypatch):
    """ffprobe reads 300 s for a tape and 124.98 s for an excerpt; cutting and extraction write small files."""
    monkeypatch.setattr(tr, "probe_duration", lambda path: 124.98 if "excerpts" in str(path) else 300.0)
    monkeypatch.setattr(tr, "cut_excerpt", lambda _tape, target, _start, _length: Path(target).write_bytes(b"cut"))
    monkeypatch.setattr(tr, "_extract", lambda _video, wav: Path(wav).write_bytes(b"wav") and wav)


def _tapes(tmp_path):
    tapes = tmp_path / "tapes"
    tapes.mkdir()
    for name in ("Interviu Tele7abc.mov", "Soti si alte.mov", "Soti si alte_Cathar_Cleaned.mov"):
        (tapes / name).write_bytes(b"tape")
    return tapes


def test_build_manifest_cuts_start_and_end_of_every_source_tape(monkeypatch, tmp_path):
    """A 300 s tape holds two 125 s excerpts; a restored output in the folder is not a tape."""
    _fake_media(monkeypatch)
    manifest = tr.build_manifest(_tapes(tmp_path), GRID, tmp_path / "out")
    assert [(row["slug"], row["part"]) for row in manifest["excerpts"]] == [
        ("tele7abc", "start"),
        ("tele7abc", "end"),
        ("soti", "start"),
        ("soti", "end"),
    ]
    assert manifest["excerpts"][0]["probed_s"] == 124.98
    assert json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))["whole"] is False


def test_build_manifest_stops_at_the_limit_and_reuses_what_was_cut(monkeypatch, tmp_path):
    """A second prepare finds its excerpts and wavs on disk; `--limit` caps the rows."""
    _fake_media(monkeypatch)
    tapes = _tapes(tmp_path)
    tr.build_manifest(tapes, GRID, tmp_path / "out")
    monkeypatch.setattr(tr, "cut_excerpt", lambda *_args: pytest.fail("cut again"))
    monkeypatch.setattr(tr, "_extract", lambda *_args: pytest.fail("extracted again"))
    assert len(tr.build_manifest(tapes, GRID, tmp_path / "out", limit=1)["excerpts"]) == 1


def test_resolved_config_reads_the_last_line_the_child_prints(monkeypatch, tmp_path):
    """The child may log before it prints the configuration; only the last line is the JSON."""
    monkeypatch.setattr(tr.subprocess, "run", lambda *_args, **_kwargs: _completed('loading\n{"cathar_alpha": 2.0}\n'))
    assert tr.resolved_config(tmp_path, "python") == {"cathar_alpha": 2.0}


def test_a_forced_run_clears_the_old_outputs_and_skips_a_complete_variant(monkeypatch, tmp_path):
    """`--force` removes excerpts, wavs and scores; a variant whose outputs are all valid is not run again."""
    variant = tmp_path / "runs" / "cathar__baseline"
    (variant / "scores").mkdir(parents=True)
    monkeypatch.setattr(tr, "variant_complete", lambda *_args: True)
    timing = tr.run_variant(variant, GRID["engines"]["cathar"], {}, MANIFEST, tmp_path / "excerpts", "python", force=True)
    assert timing == {"skipped": True}
    assert not (variant / "scores").exists()


def _scored_doc(label, top=0.5):
    entry = {"median": top, "tail": top, "n": 2}
    aggregate = {"dsp.balance_top_db": {"source": entry, "output": entry, "delta": entry}}
    variant = {"aggregate": aggregate, "hard_failures": [], "listener_flags": [], "passed": True, "rows": [], "file": {}, "families": {}}
    return {"variants": {label: variant}, "metrics": {}}


def test_score_variant_scores_valid_outputs_and_merges_a_one_family_rescore(monkeypatch, tmp_path):
    """Missing outputs are skipped; a rescore of one family merges into the stored pair instead of replacing it."""
    variant = tmp_path / "cathar__baseline"
    calls, merged = [], []
    monkeypatch.setattr(tr, "is_valid_video", lambda path: "soti_start" in str(path))
    monkeypatch.setattr(
        tr.runner, "score_variants", lambda *args, **kwargs: (calls.append(kwargs["families"]), (_scored_doc(variant.name), {}))[1]
    )
    monkeypatch.setattr(tr, "merge_families", lambda stored, fresh, families, gates: merged.append(families) or stored)
    manifest = {"excerpts": [*MANIFEST["excerpts"], {"excerpt": "soti_end.mov", "source_wav": "soti_end.src.wav"}]}
    first = tr.score_variant(variant, GRID["engines"]["cathar"], manifest, tmp_path, tr.runner.ALL_FAMILIES, None, None, tmp_path)
    tr.score_variant(variant, GRID["engines"]["cathar"], manifest, tmp_path, ("dsp",), None, None, tmp_path, rescore=True)
    assert list(first) == ["soti_start"]
    assert (calls, merged) == ([tr.runner.ALL_FAMILIES, ("dsp",)], [("dsp",)])


def _state(tmp_path, monkeypatch):
    """A tuning folder under the working directory, as the commands expect it, and the grid file."""
    monkeypatch.chdir(tmp_path)
    out_dir = Path("experiments") / "tune_t"
    out_dir.mkdir(parents=True)
    (out_dir / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(yaml.safe_dump(GRID, sort_keys=False), encoding="utf-8")
    return out_dir, grid_path


def _args(grid_path, **extra):
    fields = {"name": "t", "grid": grid_path, "engines": None, "variants": None, "gates": None, "python": "python", "force": False}
    return types.SimpleNamespace(
        **{**fields, "keep_work": False, "rescore": False, "metrics": "all", "device": "cpu", "language": "ro", **extra}
    )


def test_prepare_refuses_an_override_the_app_does_not_read(tmp_path, monkeypatch):
    """A typo in the grid stops the run before any excerpt is cut."""
    _out_dir, grid_path = _state(tmp_path, monkeypatch)
    grid_path.write_text(yaml.safe_dump({**GRID, "variants": {"cathar": {"typo": {"cathar_alfa": 2.0}}}}), encoding="utf-8")
    args = _args(grid_path, tapes_dir=str(tmp_path), limit=0, catalog=None, whole=False)
    with pytest.raises(SystemExit, match="cathar__typo"):
        tr.cmd_prepare(args)


def test_prepare_refuses_excerpts_too_short_for_the_cathar_probe(tmp_path, monkeypatch):
    """A cut excerpt that cannot feed the stitched probe is an error (whole corpus clips only get a note)."""
    _out_dir, grid_path = _state(tmp_path, monkeypatch)
    short = {**MANIFEST, "whole": False, "excerpts": [{**MANIFEST["excerpts"][0], "probed_s": 60.0}]}
    monkeypatch.setattr(tr, "build_manifest", lambda *_args: short)
    args = _args(grid_path, tapes_dir=str(tmp_path), limit=0, catalog=None, whole=False)
    with pytest.raises(SystemExit, match="cathar__baseline: needs"):
        tr.cmd_prepare(args)


def test_run_runs_every_variant_under_a_lock_it_removes(tmp_path, monkeypatch):
    """One driver at a time: the lock exists while the variants run and is gone after."""
    out_dir, grid_path = _state(tmp_path, monkeypatch)
    ran = []
    monkeypatch.setattr(
        tr, "run_variant", lambda variant_dir, *_args: ran.append(((out_dir / ".running").exists(), variant_dir.name)) or {}
    )
    tr.cmd_run(_args(grid_path))
    assert ran == [(True, "cathar__baseline"), (True, "cathar__alpha")]
    assert not (out_dir / ".running").exists()


def test_run_refuses_while_another_driver_holds_the_lock(tmp_path, monkeypatch):
    """Never two restorations at once."""
    out_dir, grid_path = _state(tmp_path, monkeypatch)
    (out_dir / ".running").write_text("1", encoding="utf-8")
    args = _args(grid_path)
    with pytest.raises(SystemExit, match="another driver is running"):
        tr.cmd_run(args)


def test_score_scores_the_chosen_families_with_the_chosen_gates(tmp_path, monkeypatch):
    """`--metrics dsp,mos` scores those two; `--gates` loads the calibrated thresholds."""
    _out_dir, grid_path = _state(tmp_path, monkeypatch)
    (tmp_path / "gates.json").write_text(json.dumps({"dsp.lkr": {"threshold": 0.9}}), encoding="utf-8")
    seen = []
    monkeypatch.setattr(tr.runner, "ModelRegistry", lambda device: device)
    monkeypatch.setattr(
        tr, "score_variant", lambda variant_dir, *args: seen.append((variant_dir.name, args[3], args[4], args[5]["dsp.lkr"].threshold))
    )
    tr.cmd_score(_args(grid_path, metrics="dsp,mos", gates=tmp_path / "gates.json", variants=["cathar__alpha"]))
    assert seen == [("cathar__alpha", ("dsp", "mos"), "cpu", 0.9)]


def _write_score(out_dir, variant_id, top):
    scores = out_dir / "runs" / variant_id / "scores"
    scores.mkdir(parents=True)
    (scores / "soti_start.json").write_text(json.dumps(_scored_doc(variant_id, top)), encoding="utf-8")


def test_report_ranks_a_v3_reading_spec_on_its_distance(tmp_path, monkeypatch):
    """The variant inside the dead zone of the two-sided target wins; the board is written as JSON and Markdown."""
    out_dir, grid_path = _state(tmp_path, monkeypatch)
    _write_score(out_dir, "cathar__baseline", 0.9)
    _write_score(out_dir, "cathar__alpha", 0.1)
    tr.cmd_report(_args(grid_path, gates=None))
    board = json.loads((out_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert board["recommendation"]["cathar"]["best"] == "cathar__alpha"
    assert (out_dir / "scoreboard.md").exists()


def test_listen_cuts_the_board_winners_picks(tmp_path, monkeypatch):
    """Without `--variants` the baseline and the board's best and runner-up are cut, once each."""
    out_dir, grid_path = _state(tmp_path, monkeypatch)
    board = {"recommendation": {"cathar": {"best": "cathar__alpha", "runner_up": "cathar__baseline"}}}
    (out_dir / "scoreboard.json").write_text(json.dumps(board), encoding="utf-8")
    pick = listening.Pick("dsp.balance_top_db", "v", 0, 0.0, 15.0, 0.4)
    cut = []
    monkeypatch.setattr(tr, "pick_worst_windows", lambda variant_dir, *_args: [("soti_start", pick)])
    monkeypatch.setattr(tr.listening, "render_from_files", lambda picks, _src, outputs, _dir: cut.append(list(outputs)) or ["- line"])
    tr.cmd_listen(_args(grid_path))
    assert cut == [["cathar__baseline"], ["cathar__alpha"]]
    assert "## cathar__alpha / soti_start" in (out_dir / "listen" / "index.md").read_text(encoding="utf-8")


def test_all_runs_every_step_in_order(monkeypatch):
    """`all` is prepare, run, score, report and listen, each with the same arguments."""
    steps = []
    for name in tr.COMMANDS:
        monkeypatch.setitem(tr.COMMANDS, name, lambda args, name=name: steps.append((name, args.name)))
    assert tr.main(["all", "--name", "t"]) == 0
    assert [step for step, _name in steps] == ["prepare", "run", "score", "report", "listen"]
