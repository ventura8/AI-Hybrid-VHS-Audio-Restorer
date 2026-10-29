"""The loop's corpus guard: a winner rendered and scored on sentinel IA clips, refused past its trips."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import autotune_corpus as corpus
from scripts import autotune_restoration as at
from scripts.restoration_quality import reward

BASE_ID = "base000000"


def _clips(tmp_path, names=("c1", "c2", "c3")):
    folder = tmp_path / "clips"
    folder.mkdir(parents=True, exist_ok=True)
    clips = {}
    for name in names:
        clip = folder / f"{name}.mp4"
        clip.write_bytes(b"clip")
        clips[name] = str(clip)
    return clips


def _guard(tmp_path, max_trips=1, parallel=2):
    constraints = {"dsp.hiss_pump_db.output.median": reward.Constraint("up", 3.0, 12.0)}
    return corpus.Guard(_clips(tmp_path), (BASE_ID, {"a": 1}), max_trips, constraints, tmp_path / "out" / corpus.DIR, parallel)


def _fake_scoring(monkeypatch, tripping):
    """The scorer writes a report per clip; `tripping` names the clips whose report carries a material trip."""

    def score(guard, ids, slugs):
        for slug in slugs:
            path = corpus.report_path(guard, ids[1], slug)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"trip": slug in tripping}), encoding="utf-8")

    def clip_trips(report, *_args):
        return [("dsp.hiss_pump_db.output.median", 10.0, 18.0, True), ("x", 1.0, 1.0, False)] if report["trip"] else []

    monkeypatch.setattr(corpus, "score", Mock(side_effect=score))
    monkeypatch.setattr(corpus, "clip_trips", clip_trips)


def test_the_clips_load_and_a_missing_clip_stops_the_run(tmp_path):
    clips = _clips(tmp_path)
    good = tmp_path / "corpus.json"
    good.write_text(json.dumps(clips), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**clips, "gone": str(tmp_path / "gone.mp4")}), encoding="utf-8")
    assert corpus.load_clips(good) == clips
    with pytest.raises(SystemExit, match="gone"):
        corpus.load_clips(bad)


def test_the_guard_is_built_only_when_asked_for(tmp_path):
    grid = tmp_path / "grid.yaml"
    grid.write_text("constraints:\n  dsp.hiss_pump_db.output.median: {worse: up, max_move: 3, bound: 12}\n", encoding="utf-8")
    clips = tmp_path / "corpus.json"
    clips.write_text(json.dumps(_clips(tmp_path)), encoding="utf-8")
    args = SimpleNamespace(corpus=clips, corpus_max_trips=2, grid=grid, parallel=3)
    guard = corpus.guard_from_args(args, tmp_path / "out", (BASE_ID, {"a": 1}))
    assert corpus.guard_from_args(SimpleNamespace(corpus=None), tmp_path, (BASE_ID, {})) is None
    assert (guard.max_trips, guard.parallel, guard.out_dir, guard.baseline) == (2, 3, tmp_path / "out" / corpus.DIR, (BASE_ID, {"a": 1}))
    assert guard.constraints["dsp.hiss_pump_db.output.median"].bound == 12.0
    assert "3 sentinel clips, at most 2" in corpus.describe(guard)


def test_a_candidate_counts_the_clips_it_trips_and_is_read_once(tmp_path, monkeypatch):
    guard = _guard(tmp_path)
    _fake_scoring(monkeypatch, {"c2"})
    render = Mock()
    found = corpus.trips(guard, "apl", ("cand000001", {"a": 2}), render)
    again = corpus.trips(guard, "apl", ("cand000001", {"a": 2}), render)
    assert found == again == {"c2": [["dsp.hiss_pump_db.output.median", 10.0, 18.0]]}
    assert [call.args[1] for call in render.call_args_list] == [{"a": 1}, {"a": 2}]
    assert corpus.score.call_args.args[1] == (BASE_ID, "cand000001")


def test_a_failed_render_trips_and_a_clip_the_baseline_fails_on_is_left_out(tmp_path, monkeypatch):
    guard = _guard(tmp_path)
    _fake_scoring(monkeypatch, set())

    def render(_engine, overrides, _clips, _out):
        failed = {"c1": "baseline broke"} if overrides == {"a": 1} else {"c1": "x", "c3": "candidate broke"}
        raise at.renders.RenderFailed("id", failed)

    found = corpus.trips(guard, "apl", ("cand000002", {"a": 3}), render)
    stored = json.loads((guard.out_dir / "cands" / "cand000002" / corpus.TRIPS_JSON).read_text(encoding="utf-8"))
    assert found == {"c3": [[corpus.RENDER_FAILED, None, None]]}
    assert stored["left_out"] == ["c1"]
    assert corpus.score.call_args.args[2] == ["c2"]
    assert corpus.baseline_failed(guard, "apl", Mock(side_effect=AssertionError)) == {"c1"}


def test_a_clip_without_a_report_counts_as_a_trip(tmp_path):
    assert corpus.clip_rows(_guard(tmp_path), "cand", "c1") == [[corpus.UNSCORED, None, None]]


def test_the_scorer_runs_on_each_unscored_clip_a_batch_at_a_time(tmp_path, monkeypatch):
    guard = _guard(tmp_path, parallel=2)
    done = corpus.report_path(guard, "cand", "c3")
    done.parent.mkdir(parents=True)
    done.write_text("{}", encoding="utf-8")
    procs = []

    def launch(args, **kwargs):
        procs.append((args, kwargs))
        return Mock()

    monkeypatch.setenv("AI_RESTORE_DATA_ROOTS", "D:/roots")
    corpus.score(guard, (BASE_ID, "cand"), ["c1", "c2", "c3"], launch=launch)
    args, kwargs = procs[0]
    assert [Path(p[0][2]).stem for p in procs] == ["c1", "c2"]
    assert args[3:6] == [f"base={corpus.wav_path(guard, BASE_ID, 'c1')}", f"cand={corpus.wav_path(guard, 'cand', 'c1')}", "--metrics"]
    assert kwargs["env"]["AI_RESTORE_DATA_ROOTS"].startswith("D:/roots")
    assert str(Path(guard.clips["c1"]).resolve().parent) in kwargs["env"]["AI_RESTORE_DATA_ROOTS"]


def test_a_candidate_past_its_trips_is_refused_and_the_verdict_says_how_many(tmp_path, monkeypatch):
    guard = _guard(tmp_path, max_trips=1)
    _fake_scoring(monkeypatch, {"c1", "c2"})
    refused = {}
    assert not corpus.accepts(guard, "apl", ("bad0000000", {"a": 5}), Mock(), refused)
    assert refused["corpus_trips"] == 2
    assert corpus.finding("bad0000000", refused, guard).endswith("refused")


def test_a_candidate_within_its_trips_holds(tmp_path, monkeypatch):
    guard = _guard(tmp_path, max_trips=1)
    monkeypatch.setattr(corpus, "trips", lambda *_args: {"c1": []})
    held = {}
    assert corpus.accepts(guard, "apl", ("good000000", {"a": 4}), Mock(), held)
    assert corpus.finding("good000000", held, guard).endswith("held")
    assert corpus.finding("other", {}, guard) is None


def _verdicts(*ids):
    """Qualifying verdicts ranked in the order given."""
    return {cid: {"qualifies": True, "mean_rank": float(rank)} for rank, cid in enumerate(ids)}


def test_settling_asks_the_guard_best_first_and_stops_at_the_first_that_holds():
    state = {"incumbent": {}, "rounds": [], "candidates": {}}
    accept = Mock(side_effect=[False, True, True])
    everything = {"x": {"a": 1}, "y": {"a": 2}, "z": {"a": 3}}
    assert at._settle(state, 1, {}, everything, _verdicts("x", "y", "z"), accept) == "y"
    assert [call.args[0] for call in accept.call_args_list] == ["x", "y"]
    assert state["incumbent"] == {"a": 2}
    assert at._settle(state, 2, {}, everything, _verdicts("x"), Mock(return_value=False)) is None


def test_a_round_with_a_guard_settles_on_the_best_candidate_it_holds_and_logs_it(tmp_path, monkeypatch):
    state = {"incumbent": {"a": 1}, "rounds": [], "candidates": {}}
    monkeypatch.setattr(at, "propose", lambda *_args: {"c1": {"a": 2}, "c2": {"a": 3}})
    monkeypatch.setattr(at, "run_candidate", lambda *_args: None)
    monkeypatch.setattr(at, "audio_fingerprint", lambda _out, cid, _tapes: (cid,))
    monkeypatch.setattr(at, "score_round", lambda cands, *_a: {cid: {"t1": {"m": float(o["a"])}} for cid, o in cands.items()})

    def accepts(guard, _engine, candidate, _render, verdict):
        verdict["corpus_trips"] = 3 if candidate[1]["a"] == 3 else 0
        return verdict["corpus_trips"] <= guard.max_trips

    monkeypatch.setattr(at.corpus, "accepts", accepts)
    args = SimpleNamespace(families=("dsp",), gates=None, language="ro", parallel=1, corpus_guard=_guard(tmp_path))
    assert at.run_round(args, "apl", {"t1": "t1.mov"}, tmp_path, state, {"m": "up"}, 1)
    log = (tmp_path / at.LOG_MD).read_text(encoding="utf-8")
    assert state["incumbent"] == {"a": 2}
    assert f"corpus guard: `{at.candidate_id({'a': 3})}` tripped 3 of 3 sentinel clips (at most 1): refused" in log


def _guarded_cli(tmp_path, monkeypatch, seen):
    """The command line with a corpus file; its one round records the guard it was handed."""
    clips = tmp_path / "corpus.json"
    clips.write_text(json.dumps(_clips(tmp_path)), encoding="utf-8")
    tapes = tmp_path / "tapes.json"
    tapes.write_text(json.dumps({"t1": "t1.mov"}), encoding="utf-8")
    grid = tmp_path / "grid.yaml"
    grid.write_text("ranking:\n  m: up\n", encoding="utf-8")
    monkeypatch.setattr(at, "run_round", lambda args, *_a: seen.append(args.corpus_guard))
    monkeypatch.setattr(at, "seed_defaults", lambda _engine, start, _out: {**start, "seeded": True})
    monkeypatch.setattr(at.tune, "probe_duration", lambda _path: None)
    argv = ["--engine", "apl", "--tapes", str(tapes), "--grid", str(grid), "--out", str(tmp_path / "out")]
    return [*argv, "--start", '{"a": 1}', "--corpus", str(clips), "--corpus-max-trips", "4"]


def test_the_command_line_builds_the_guard_against_the_start_settings(tmp_path, monkeypatch, capsys):
    seen = []
    at.main(_guarded_cli(tmp_path, monkeypatch, seen))
    assert (seen[0].baseline, seen[0].max_trips) == ((at.candidate_id({"a": 1, "seeded": True}), {"a": 1, "seeded": True}), 4)
    assert "corpus guard: 3 sentinel clips" in capsys.readouterr().out
    assert at._corpus_check(SimpleNamespace(), "apl", {}, {}) is None


def test_a_round_whose_every_winner_the_guard_refuses_says_so(tmp_path, monkeypatch):
    state = {"incumbent": {"a": 1}, "rounds": [], "candidates": {}}
    monkeypatch.setattr(at, "propose", lambda *_args: {"c1": {"a": 2}})
    monkeypatch.setattr(at, "run_candidate", lambda *_args: None)
    monkeypatch.setattr(at, "audio_fingerprint", lambda _out, cid, _tapes: (cid,))
    monkeypatch.setattr(at, "score_round", lambda cands, *_a: {cid: {"t1": {"m": float(o["a"])}} for cid, o in cands.items()})
    monkeypatch.setattr(at.corpus, "accepts", lambda _g, _e, _c, _r, verdict: verdict.update(corpus_trips=9) and False)
    args = SimpleNamespace(families=("dsp",), gates=None, language="ro", parallel=1, corpus_guard=_guard(tmp_path))
    assert not at.run_round(args, "apl", {"t1": "t1.mov"}, tmp_path, state, {"m": "up"}, 1)
    assert "no candidate the corpus guard holds: plateau (1 qualified" in (tmp_path / at.LOG_MD).read_text(encoding="utf-8")
