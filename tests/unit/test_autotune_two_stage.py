"""Two-stage scoring: the ranked families on every candidate, the learned vetoes' families on the contenders only."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from scripts import autotune_guards as ag
from scripts import autotune_restoration as at
from scripts import autotune_scoring as scoring
from scripts.restoration_quality import reward

RANKING = {"dsp.m.delta.median": "down"}
CER = "speech.cer.output.median"
VETOES = {CER: reward.Veto(worse="up", limit=0.01)}
# dsp.m ranks (lower is better); speech.cer is the learned veto. `vetoed` ranks best and trips the veto.
FIRST = {
    "inc": {"dsp.m.delta.median": 2.0},
    "good": {"dsp.m.delta.median": 1.0},
    "worse": {"dsp.m.delta.median": 3.0},
    "vetoed": {"dsp.m.delta.median": 0.5},
}
SECOND = {"inc": {CER: 0.10}, "good": {CER: 0.10}, "worse": {CER: 0.90}, "vetoed": {CER: 0.50}}


def test_the_ranked_families_are_the_ones_the_ranking_and_the_gates_read():
    """Speech grid: dsp first, speech and mos second; music grid: stems ranks, so only mos waits; a gate pulls its family in."""
    assert scoring.split_families(("dsp", "speech", "mos"), RANKING) == (("dsp",), ("speech", "mos"))
    music = {"dsp.x": "down", "stems.lsd_db.output.median": {"target": 0.0}}
    assert scoring.split_families(("dsp", "stems", "mos"), music) == (("dsp", "stems"), ("mos",))
    assert scoring.split_families(("dsp", "speech", "mos"), RANKING, ["speech.cer"]) == (("dsp", "speech"), ("mos",))


def test_a_constraints_family_is_scored_in_the_first_stage():
    """A grid constraint is checked on every candidate in both judgements, so its family cannot wait for the contenders:
    scored in the second stage only, a candidate that never qualified would read as having lost it, and be guarded."""
    constraints = {"mos.sigmos_noise.delta.median": reward.Constraint("down", 0.1)}
    assert scoring.split_families(("dsp", "speech", "mos"), RANKING, (), constraints) == (("dsp", "mos"), ("speech",))


def _scorer(first=None, second=None):
    """A score_round stand-in recording each call's candidates, families and stage."""
    first, second = first or FIRST, second or SECOND
    calls = []

    def score(candidates, _tapes, _out, families, _gates, _language, _parallel, stage=""):
        calls.append((sorted(candidates), families, stage))
        table = second if stage == scoring.LEARNED_STAGE else first
        return {cid: {"t1": dict(table[cid])} for cid in candidates}

    return score, calls


def _judge(monkeypatch, score, families=("dsp", "speech")):
    monkeypatch.setattr(at, "score_round", score)
    args = SimpleNamespace(families=families, gates=None, language="ro", parallel=1)
    scorable = {cid: {} for cid in FIRST}
    return at.score_and_judge(args, scorable, ({"t1": "t1.mov"}, Path(".")), RANKING, "inc", ag.Guards(vetoes=VETOES))


def test_the_learned_families_run_only_on_the_incumbent_and_the_contenders(monkeypatch):
    """`worse` never beats the incumbent, so its speech readings are never computed."""
    score, calls = _scorer()
    _judge(monkeypatch, score)
    assert calls == [(sorted(FIRST), ("dsp",), ""), (["good", "inc", "vetoed"], ("speech",), scoring.LEARNED_STAGE)]


def test_score_and_judge_scores_a_constrained_family_on_every_candidate(monkeypatch):
    """A constraint on a speech reading pulls speech into the first stage: nothing is left for a second one."""
    score, calls = _scorer()
    monkeypatch.setattr(at, "score_round", score)
    args = SimpleNamespace(families=("dsp", "speech"), gates=None, language="ro", parallel=1)
    guards = ag.Guards(constraints={CER: reward.Constraint("up", 0.5)})
    at.score_and_judge(args, {cid: {} for cid in FIRST}, ({"t1": "t1.mov"}, Path(".")), RANKING, "inc", guards)
    assert calls == [(sorted(FIRST), ("dsp", "speech"), "")]


def test_the_two_stages_give_the_single_pass_verdict(monkeypatch):
    """Same qualifications as judging every family on every candidate."""
    score, _calls = _scorer()
    two_stage = _judge(monkeypatch, score)
    full = {cid: {"t1": {**FIRST[cid], **SECOND[cid]}} for cid in FIRST}
    single = at.judge(full, "inc", RANKING, VETOES)
    assert {cid: v["qualifies"] for cid, v in two_stage.items()} == {cid: v["qualifies"] for cid, v in single.items()}
    assert [cid for cid, v in two_stage.items() if v["qualifies"]] == ["good"]


def test_only_a_contender_carries_a_veto_mark(monkeypatch):
    """The vetoed contender is marked; `worse`, whose learned readings were never scored, is not."""
    score, _calls = _scorer()
    two_stage = _judge(monkeypatch, score)
    assert two_stage["vetoed"]["vetoed"]
    assert "vetoed" not in two_stage["worse"]


def test_without_a_contender_the_learned_families_never_run(monkeypatch):
    """A plateau round scores the ranked families only."""
    flat = {cid: {"dsp.m.delta.median": 0.0 if cid == "inc" else 5.0} for cid in FIRST}
    score, calls = _scorer(first=flat)
    verdicts = _judge(monkeypatch, score)
    assert len(calls) == 1
    assert not any(v["qualifies"] for v in verdicts.values())


def test_flats_scored_on_every_family_at_once_are_not_scored_again(monkeypatch):
    """A candidate scored before the two stages existed already holds its speech readings."""
    full = {cid: {**FIRST[cid], **SECOND[cid]} for cid in FIRST}
    score, calls = _scorer(first=full)
    verdicts = _judge(monkeypatch, score)
    assert len(calls) == 1
    assert [cid for cid, v in verdicts.items() if v["qualifies"]] == ["good"]


def test_with_every_family_ranked_there_is_no_second_stage(monkeypatch):
    """dsp alone: nothing waits for the contenders, and the veto reads what the first pass has."""
    score, calls = _scorer(first={cid: {**FIRST[cid], **SECOND[cid]} for cid in FIRST})
    _judge(monkeypatch, score, families=("dsp",))
    assert calls == [(sorted(FIRST), ("dsp",), "")]


def test_merging_keeps_the_first_pass_gates_and_adds_only_the_second_familys_readings():
    """The second pass's own gate counts and dsp readings never overwrite the first pass's."""
    scores = {"c": {"t1": {"dsp.m": 1.0, "gates.hard_failures": 2.0}}}
    learned = {"c": {"t1": {"speech.cer": 0.1, "gates.hard_failures": 0.0, "dsp.m": 9.0}}}
    scoring.merge_families(scores, learned, ("speech",))
    assert scores == {"c": {"t1": {"dsp.m": 1.0, "gates.hard_failures": 2.0, "speech.cer": 0.1}}}
    assert scoring.lacking(scores, ["c"], ()) == []
    assert scoring.lacking(scores, ["c"], ("mos",)) == ["c"]


def _fake_popen(args, **_kwargs):
    Path(args[args.index("--report") + 1]).write_text("{}", encoding="utf-8")
    return SimpleNamespace(returncode=0, wait=lambda: 0)


def _second_pass(tmp_path, monkeypatch):
    """One candidate with a first-pass score, scored again in the learned stage; returns the Popen stand-in, the result
    and the first-pass file."""
    tape = tmp_path / "t1.mov"
    tape.write_bytes(b"source")
    (tmp_path / "cands" / "c1").mkdir(parents=True)
    first = scoring._score_path(tmp_path, "c1", "t1")
    first.write_text(json.dumps({"dsp.m": 1.0}), encoding="utf-8")
    launcher = Mock(side_effect=_fake_popen)
    monkeypatch.setattr(scoring.subprocess, "Popen", launcher)
    monkeypatch.setattr(scoring.tune.runner, "aggregates_for_tuning", lambda *_args: {"speech.cer": 0.2})
    out = scoring.score_round({"c1": {}}, {"t1": str(tape)}, tmp_path, ("speech",), None, "ro", 1, scoring.LEARNED_STAGE)
    return launcher, out, first


def test_the_second_stage_writes_its_own_report_and_log(tmp_path, monkeypatch):
    """`learned_<slug>.json` and `.log` in rounds/, beside the first pass's `score_<slug>` files."""
    launcher, out, _first = _second_pass(tmp_path, monkeypatch)
    assert out == {"c1": {"t1": {"speech.cer": 0.2}}}
    assert launcher.call_args.args[0][-1] == str(tmp_path.resolve() / "rounds" / "learned_t1.json")
    assert (tmp_path / "rounds" / "learned_t1.log").exists()


def test_the_second_stage_keeps_the_first_pass_score_file(tmp_path, monkeypatch):
    """`<slug>.learned.score.json` holds the second pass; the first pass's file is left as it was."""
    _launcher, _out, first = _second_pass(tmp_path, monkeypatch)
    assert json.loads(first.read_text(encoding="utf-8")) == {"dsp.m": 1.0}
    learned = scoring._score_path(tmp_path, "c1", "t1", scoring.LEARNED_STAGE)
    assert (learned.name, json.loads(learned.read_text(encoding="utf-8"))) == ("t1.learned.score.json", {"speech.cer": 0.2})
