"""The loop survives a candidate whose render failed and stops before rendering on a full disk (2026-10-20).

APL's round 3 ended after two minutes when `apl_burst_ceiling` `gaps` left no output on three
Tata tapes, and cathar's crashed on "No space left on device"; both runs were lost for the day.
"""

from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import autotune_guards as ag
from scripts import autotune_restoration as at

SUFFIX = "_PureLinear_Cleaned"
TILT = "dsp.balance_tilt_db_oct.output.median"
RANKING = {TILT: {"target": 0.0, "dead_zone": 0.0, "scale": 0.1, "weight": 1.0, "family": "timbre"}}
Usage = namedtuple("Usage", "total used free")


def _tapes(tmp_path):
    folder = tmp_path / "tapes"
    folder.mkdir(exist_ok=True)
    tapes = {"a": folder / "alpha.mov", "b": folder / "beta.mov"}
    for tape in tapes.values():
        tape.write_bytes(b"source")
    return {slug: str(path) for slug, path in tapes.items()}


def _fake_extract(produced, cache_dir):
    target = Path(cache_dir) / f"extracted_{Path(produced).stem}.wav"
    target.write_bytes(b"RIFF")
    return target


def _cand_dir(tmp_path, log=""):
    cand_dir = tmp_path / "out" / "cands" / "abc123"
    cand_dir.mkdir(parents=True)
    (cand_dir / at.RUN_LOG).write_text(log, encoding="utf-8")
    return cand_dir, {"a": cand_dir / "a.wav", "b": cand_dir / "b.wav"}


def test_a_tape_without_output_fails_the_candidate_after_the_others_are_collected(tmp_path, monkeypatch):
    tapes = _tapes(tmp_path)
    produced = tmp_path / "tapes" / f"beta{SUFFIX}.mov"
    produced.write_bytes(b"video")
    _dir, wavs = _cand_dir(tmp_path, "   [System] Processing Task: alpha.mov\n   [Error] Failed to save audio x.wav: System error.\n")
    monkeypatch.setattr(at.audio_io, "extract_wav", _fake_extract)
    with pytest.raises(at.renders.RenderFailed) as failure:
        at._collect_outputs("abc123", tapes, SUFFIX, wavs)
    assert failure.value.reasons == {"a": "[Error] Failed to save audio x.wav: System error."}
    assert (wavs["b"].read_bytes(), produced.exists(), wavs["a"].exists()) == (b"RIFF", False, False)


def test_audio_that_cannot_be_extracted_fails_the_candidate_and_its_video_goes(tmp_path, monkeypatch):
    tapes = {"a": _tapes(tmp_path)["a"]}
    produced = tmp_path / "tapes" / f"alpha{SUFFIX}.mov"
    produced.write_bytes(b"truncated video")
    _dir, wavs = _cand_dir(tmp_path)

    def broken(*_args):
        raise at.subprocess.CalledProcessError(1, ["ffmpeg"])

    monkeypatch.setattr(at.audio_io, "extract_wav", broken)
    with pytest.raises(at.renders.RenderFailed, match="a: its audio could not be extracted"):
        at._collect_outputs("abc123", tapes, SUFFIX, {"a": wavs["a"]})
    assert not produced.exists()


def test_a_full_drive_stops_the_loop_before_the_candidate_is_prepared(tmp_path, monkeypatch):
    prepare = Mock()
    monkeypatch.setattr(at, "_prepare_candidate", prepare)
    monkeypatch.setattr(at.renders.shutil, "disk_usage", lambda _path: Usage(0, 0, 5 * at.renders.GB))
    with pytest.raises(SystemExit, match="5.0 GB free"):
        at.run_candidate("apl", {"linear_air_gain_db": 0.5}, _tapes(tmp_path), tmp_path / "out")
    assert not prepare.called


def test_the_render_folders_are_the_tapes_the_loops_and_the_stage_caches(tmp_path, monkeypatch):
    tapes = _tapes(tmp_path)
    monkeypatch.delenv(at.stage_cache.ENV_VAR, raising=False)
    assert at.render_paths(tapes, tmp_path / "out") == sorted({str(tmp_path / "tapes"), str(tmp_path / "out")})
    monkeypatch.setenv(at.stage_cache.ENV_VAR, str(tmp_path / "cache"))
    assert str(tmp_path / "cache") in at.render_paths(tapes, tmp_path / "out")


def _failing_for(bad):
    def run(_engine, overrides, _tapes, _out):
        if overrides.get("k") in bad:
            raise at.renders.RenderFailed(at.candidate_id(overrides), {"t1": f"no output for k={overrides['k']}"})

    return run


TILTS = {at.candidate_id({"k": 0}): 0.5, at.candidate_id({"k": 1}): 0.6}


def _tilt_scores(live, *_args):
    """The scorer's stand-in: every live candidate's tilt on its one tape."""
    return {cid: {"t1": {TILT: TILTS[cid], "gates.hard_failures": 0.0}} for cid in live}


def test_a_failed_candidate_is_set_aside_and_the_round_goes_on(tmp_path, monkeypatch, capsys):
    state = {"incumbent": {"k": 0}, "rounds": [], "candidates": {}}
    proposals = {at.candidate_id({"k": 1}): {"k": 1}, at.candidate_id({"k": 2}): {"k": 2}}
    monkeypatch.setattr(at, "propose", lambda *_args: proposals)
    monkeypatch.setattr(at, "run_candidate", Mock(side_effect=_failing_for({2})))
    monkeypatch.setattr(at, "audio_fingerprint", lambda _out, cid, _tapes: (cid,))
    monkeypatch.setattr(at, "score_round", _tilt_scores)
    args = SimpleNamespace(families=("dsp",), gates=None, language="ro", parallel=1, guards=ag.NO_GUARDS)
    assert not at.run_round(args, "apl", {"t1": "t1.mov"}, tmp_path, state, RANKING, 1)
    failed_id = at.candidate_id({"k": 2})
    verdict = state["rounds"][0]["verdicts"][failed_id]
    log = (tmp_path / at.LOG_MD).read_text(encoding="utf-8")
    assert (verdict["failed"], verdict["qualifies"]) == (["t1: no output for k=2"], False)
    assert (f"render failure: `{failed_id}`" in log, "1 render(s) failed, their moves untried" in log) == (True, True)
    assert "render failed" in capsys.readouterr().out


def test_the_incumbent_failing_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(at, "run_candidate", Mock(side_effect=_failing_for({0})))
    with pytest.raises(at.renders.RenderFailed):
        at._render("apl", {at.candidate_id({"k": 0}): {"k": 0}}, {"t1": "t1.mov"}, tmp_path, {"k": 0})


def test_three_failures_in_a_row_stop_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(at, "run_candidate", Mock(side_effect=_failing_for({1, 2, 3})))
    candidates = {at.candidate_id({"k": k}): {"k": k} for k in (1, 2, 3)}
    with pytest.raises(SystemExit, match="3 renders in a row failed"):
        at._render("apl", candidates, {"t1": "t1.mov"}, tmp_path, {"k": 0})


def test_a_render_that_failed_is_not_tried_again_in_the_same_run_but_is_on_the_next(tmp_path, monkeypatch):
    state = {"incumbent": {"k": 0}, "rounds": [], "candidates": {}, "failed": {"f1": {"t1": "why"}}}
    rendered = []
    monkeypatch.setattr(at, "_render", lambda _e, cands, *_a: rendered.append(sorted(cands)) or {})
    monkeypatch.setattr(at, "_split_inert", lambda live, *_a: (live, {}))
    monkeypatch.setattr(at, "score_and_judge", lambda *_a: {})
    args = SimpleNamespace(guards=ag.NO_GUARDS)
    everything = {"f1": {"k": 1}, at.candidate_id({"k": 0}): {"k": 0}}
    verdicts = at._run_and_judge(args, "apl", everything, {"t1": "t1.mov"}, tmp_path, state, RANKING, {"k": 0})
    assert (rendered, verdicts["f1"]["failed"]) == ([[at.candidate_id({"k": 0})]], ["t1: why"])
    monkeypatch.setattr(at, "seed_defaults", lambda _engine, start, _out: start)
    (tmp_path / "state.json").write_text(at.json.dumps(state), encoding="utf-8")
    resumed = at._resume(SimpleNamespace(start_file=None, start="{}", engine="apl"), tmp_path)
    assert "failed" not in resumed


def test_a_failed_render_has_its_own_cell_in_the_round_table():
    verdict = at._failed_verdict({"t1": "disk full"})
    assert (ag.aside_cell(verdict), verdict["failed"]) == ("render failed", ["t1: disk full"])
