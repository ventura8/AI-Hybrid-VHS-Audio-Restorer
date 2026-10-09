"""The self-driving tuner's driver: rendering a candidate, seeding defaults, launching scorers, a round, the command line."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from scripts import autotune_restoration as at

APL_SUFFIX = "_PureLinear_Cleaned"


def _tape(tmp_path):
    tape = tmp_path / "tapes" / "t1.mov"
    tape.parent.mkdir(parents=True, exist_ok=True)
    tape.write_bytes(b"source")
    return tape


def _fake_restore(cmd, **_kwargs):
    """The app's run: a restored video beside every tape it was given."""
    for tape in map(Path, cmd[2:]):
        tape.with_name(f"{tape.stem}{APL_SUFFIX}{tape.suffix}").write_bytes(b"video")


def _fake_extract(produced, cache_dir):
    target = Path(cache_dir) / f"extracted_{Path(produced).stem}.wav"
    target.write_bytes(b"RIFF")
    return target


def _echo_config(cand_dir, *_args):
    """resolved_config's stand-in: the app honours every setting the candidate's config.yaml names."""
    return yaml.safe_load((Path(cand_dir) / "config.yaml").read_text(encoding="utf-8"))


def _app(monkeypatch, restore=_fake_restore, resolved=_echo_config):
    runner = Mock(side_effect=restore)
    monkeypatch.setattr(at.tune, "resolved_config", resolved)
    monkeypatch.setattr(at.audio_io, "extract_wav", _fake_extract)
    monkeypatch.setattr(at.subprocess, "run", runner)
    return runner


def test_a_candidate_keeps_its_audio_removes_the_restored_video_and_never_renders_twice(tmp_path, monkeypatch):
    """The audio lands in cands/<id>/<slug>.wav; a model-name knob passes the setting check like any other."""
    runner = _app(monkeypatch)
    tapes = {"t1": str(_tape(tmp_path))}
    overrides = {"linear_air_gain_db": 2.0, "apl_music_neural_model": at.ROFORMER}
    wavs = at.run_candidate("apl", overrides, tapes, tmp_path / "out")
    at.run_candidate("apl", overrides, tapes, tmp_path / "out")
    assert wavs["t1"].read_bytes() == b"RIFF"
    assert not (tmp_path / "tapes" / f"t1{APL_SUFFIX}.mov").exists()
    assert runner.call_count == 1
    assert json.loads((wavs["t1"].parent / "overrides.json").read_text(encoding="utf-8")) == overrides


def test_a_candidate_records_the_stage_cache_hits_and_misses_its_run_log_reports(tmp_path, monkeypatch):
    """timing.json says how many tapes replayed the neural stage and how many rendered it."""

    def restore_with_cache(cmd, **kwargs):
        kwargs["stdout"].write("\x1b[K    [Stage Cache] Hit 0123456789ab: replayed 1.0 MB\n    [Stage Cache] Miss abc: rendering\n")
        _fake_restore(cmd)

    _app(monkeypatch, restore=restore_with_cache)
    wavs = at.run_candidate("apl", {"linear_air_gain_db": 2.0}, {"t1": str(_tape(tmp_path))}, tmp_path / "out")
    timing = json.loads((wavs["t1"].parent / "timing.json").read_text(encoding="utf-8"))
    assert timing["stage_cache"] == {"hits": 1, "misses": 1}
    assert at.stage_cache_counts(tmp_path / "no_such.log") == {"hits": 0, "misses": 0}


REFUSED_CANDIDATES = [
    ({"no_such_setting": 1}, _echo_config, _fake_restore, "not a setting"),
    ({"linear_air_gain_db": 2.0}, lambda *_args: {}, _fake_restore, "did not honour"),
    ({"linear_air_gain_db": 2.0}, _echo_config, lambda *_args, **_kwargs: None, "no output"),
]


@pytest.mark.parametrize("case", REFUSED_CANDIDATES)
def test_a_candidate_the_app_would_not_run_as_asked_stops_the_loop(tmp_path, monkeypatch, case):
    """An unknown setting, a setting the app reverts and a run that left no output each raise."""
    overrides, resolved, restore, message = case
    _app(monkeypatch, restore, resolved)
    with pytest.raises(SystemExit, match=message):
        at.run_candidate("apl", overrides, {"t1": str(_tape(tmp_path))}, tmp_path / "out")


def test_seeding_takes_the_app_default_when_it_is_one_of_the_knob_values(tmp_path, monkeypatch):
    """2.0 and "True" match a knob value and are seeded; 7 dB of air matches none and stays unset."""
    resolved = {"apl_spectral_alpha_tonal": 2.0, "enable_linear_air": "True", "linear_air_gain_db": 7.0}
    monkeypatch.setattr(at.tune, "resolved_config", lambda *_args: resolved)
    seeded = at.seed_defaults("apl", {"apl_sibilant_mix": 0.9}, tmp_path)
    assert seeded["apl_spectral_alpha_tonal"] == 2.0
    assert seeded["enable_linear_air"] is True
    assert "linear_air_gain_db" not in seeded
    assert seeded["apl_sibilant_mix"] == 0.9


def test_seeding_asks_the_app_nothing_when_every_knob_is_named(tmp_path, monkeypatch):
    """No child interpreter starts when the start settings already name every knob."""
    resolver = Mock()
    monkeypatch.setattr(at.tune, "resolved_config", resolver)
    start = {knob: values[-1] for knob, values in at.KNOBS["cathar"].items()}
    assert at.seed_defaults("cathar", start, tmp_path) == start
    assert not resolver.called


def _fake_popen(args, **_kwargs):
    """The scorer's run: an empty report where it was asked to write one."""
    Path(args[args.index("--report") + 1]).write_text("{}", encoding="utf-8")
    return SimpleNamespace(returncode=0, wait=lambda: 0)


def _scoring(tmp_path, monkeypatch, popen=_fake_popen):
    """A tape, a gates file and a rendered candidate folder; returns the Popen stand-in and the tapes."""
    launcher = Mock(side_effect=popen)
    monkeypatch.setattr(at.subprocess, "Popen", launcher)
    monkeypatch.setattr(at.tune.runner, "aggregates_for_tuning", lambda *_args: {"m": 1.0})
    (tmp_path / "gates.json").write_text("{}", encoding="utf-8")
    (tmp_path / "cands" / "c1").mkdir(parents=True)
    return launcher, {"t1": str(_tape(tmp_path))}


def test_a_scorer_runs_on_the_checked_tape_and_gates_and_its_report_becomes_the_score(tmp_path, monkeypatch):
    """The gates path reaches the scorer resolved; the report's aggregates are the candidate's score on the tape."""
    launcher, tapes = _scoring(tmp_path, monkeypatch)
    scores = at.score_round({"c1": {}}, tapes, tmp_path, ("dsp",), tmp_path / "gates.json", "ro", parallel=1)
    assert scores == {"c1": {"t1": {"m": 1.0}}}
    assert launcher.call_args.args[0][-2:] == ["--gates", str((tmp_path / "gates.json").resolve())]


def test_a_failed_scorer_stops_the_loop(tmp_path, monkeypatch):
    """A scorer that exits non-zero leaves no score to judge on."""
    _, tapes = _scoring(tmp_path, monkeypatch, popen=lambda *_args, **_kwargs: SimpleNamespace(returncode=1, wait=lambda: 1))
    with pytest.raises(SystemExit, match="scoring t1 failed"):
        at.score_round({"c1": {}}, tapes, tmp_path, ("dsp",), None, "ro")


REFUSED_SCORER_INPUTS = [
    ({"language": "r0"}, "two- or three-letter"),
    ({"families": ("dsp", "nope")}, "unknown metric families"),
    ({"tape": "no_such_tape.mov"}, "does not exist"),
    ({"tape": "-rf"}, "looks like an option"),
]


@pytest.mark.parametrize("case", REFUSED_SCORER_INPUTS)
def test_the_scorer_is_never_launched_on_inputs_it_cannot_check(tmp_path, monkeypatch, case):
    """A malformed language, an unknown family, a missing tape and an option-shaped path are refused first."""
    changes, message = case
    _, tapes = _scoring(tmp_path, monkeypatch)
    inputs = {"tape": tapes["t1"], "families": ("dsp",), "language": "ro", **changes}
    with pytest.raises(SystemExit, match=message):
        at.score_round({"c1": {}}, {"t1": inputs["tape"]}, tmp_path, inputs["families"], None, inputs["language"])


def _score_by_settings(candidates, *_args):
    """Higher settings score higher on the one metric: each move wins and their combination wins most."""
    return {cid: {"t1": {"m": float(sum(overrides.values()))}} for cid, overrides in candidates.items()}


def _round(tmp_path, monkeypatch, proposals):
    state = {"incumbent": {"a": 1, "b": 1}, "rounds": [], "candidates": {}}
    monkeypatch.setattr(at, "propose", lambda *_args: proposals)
    monkeypatch.setattr(at, "run_candidate", lambda *_args: None)
    monkeypatch.setattr(at, "audio_fingerprint", lambda _out, cid, _tapes: (cid,))
    monkeypatch.setattr(at, "score_round", _score_by_settings)
    args = SimpleNamespace(families=("dsp",), gates=None, language="ro", parallel=1)
    return at.run_round(args, "apl", {"t1": "t1.mov"}, tmp_path, state, {"m": "up"}, 1), state


def test_two_winning_moves_are_combined_and_the_combination_is_accepted(tmp_path, monkeypatch):
    """a=2 and b=2 each beat the incumbent; the candidate carrying both beats them and becomes the incumbent."""
    moved, state = _round(tmp_path, monkeypatch, {"c1": {"a": 2, "b": 1}, "c2": {"a": 1, "b": 2}})
    assert moved
    assert state["incumbent"] == {"a": 2, "b": 2}
    assert state["rounds"][0]["accepted"] == at.candidate_id({"a": 2, "b": 2})
    assert "accepted `" in (tmp_path / at.LOG_MD).read_text(encoding="utf-8")


def test_a_candidate_without_any_ranked_reading_never_qualifies():
    """No ranking metric was read on the tape, so there is no mean rank to beat the incumbent's with."""
    verdicts = at.judge({"inc": {"t1": {}}, "c1": {"t1": {}}}, "inc", {"m": "up"})
    assert verdicts["c1"]["mean_rank"] is None
    assert not verdicts["c1"]["qualifies"]


def test_a_round_without_proposals_moves_nothing(tmp_path, monkeypatch):
    """No neighbour left to try is a plateau without a render."""
    moved, state = _round(tmp_path, monkeypatch, {})
    assert not moved
    assert not state["rounds"]


def _cli(tmp_path, *extra):
    tapes = tmp_path / "tapes.json"
    tapes.write_text(json.dumps({"t1": "t1.mov"}), encoding="utf-8")
    grid = tmp_path / "grid.yaml"
    grid.write_text("ranking:\n  m: up\n", encoding="utf-8")
    return ["--engine", "apl", "--tapes", str(tapes), "--grid", str(grid), "--out", str(tmp_path / "out"), *extra]


def _loop(monkeypatch, rounds, length=None):
    """The command line with its rounds, its seeding and ffprobe stood in for."""
    monkeypatch.setattr(at, "run_round", rounds)
    monkeypatch.setattr(at, "seed_defaults", lambda _engine, start, _out: start)
    monkeypatch.setattr(at.tune, "probe_duration", lambda _path: length)


def test_the_command_line_runs_rounds_until_a_plateau_and_writes_the_final_settings(tmp_path, monkeypatch):
    """Two rounds move the incumbent, the third does not; the log starts with the run's heading."""
    rounds = Mock(side_effect=[True, True, False])
    _loop(monkeypatch, rounds)
    assert at.main(_cli(tmp_path, "--start", '{"a": 1}')) == 0
    assert rounds.call_count == 3
    assert json.loads((tmp_path / "out" / "apl" / "final.json").read_text(encoding="utf-8")) == {"a": 1}
    assert (tmp_path / "out" / "apl" / at.LOG_MD).read_text(encoding="utf-8").startswith("# autotune apl")


def test_a_resumed_run_keeps_its_saved_incumbent_and_stops_at_the_round_cap(tmp_path, monkeypatch):
    """A saved state wins over --start-file, and the round numbers continue from the rounds already run."""
    out = tmp_path / "out" / "apl"
    out.mkdir(parents=True)
    (out / "state.json").write_text(json.dumps({"incumbent": {"b": 2}, "rounds": [{}], "candidates": {}}), encoding="utf-8")
    (out / at.LOG_MD).write_text("# autotune apl, resumed", encoding="utf-8")
    start = tmp_path / "final_before.json"
    start.write_text('{"a": 1}', encoding="utf-8")
    rounds = Mock(return_value=True)
    _loop(monkeypatch, rounds)
    at.main(_cli(tmp_path, "--start-file", str(start), "--rounds", "3"))
    assert [call.args[-1] for call in rounds.call_args_list] == [2, 3]
    assert json.loads((out / "final.json").read_text(encoding="utf-8")) == {"b": 2}


def test_the_stage_cache_flag_reaches_every_candidate_through_the_environment(tmp_path, monkeypatch, capsys):
    """--stage-cache sets AI_RESTORE_STAGE_CACHE (absolute) and --stage-cache-gb its cap; the candidates inherit both."""
    _loop(monkeypatch, Mock(return_value=False))
    at.main(_cli(tmp_path, "--stage-cache", str(tmp_path / "cache"), "--stage-cache-gb", "7.5"))
    assert at.os.environ[at.stage_cache.ENV_VAR] == str((tmp_path / "cache").resolve())
    assert at.os.environ[at.stage_cache.MAX_GB_ENV] == "7.5"
    assert "stage cache:" in capsys.readouterr().out


def test_without_the_flag_the_stage_cache_stays_off(tmp_path, monkeypatch):
    _loop(monkeypatch, Mock(return_value=False))
    at.main(_cli(tmp_path))
    assert at.stage_cache.ENV_VAR not in at.os.environ
    assert at.enable_stage_cache(None) is None


def test_every_round_gets_the_tape_lengths_read_once_at_the_start(tmp_path, monkeypatch):
    """The lengths decide which print-length moves are dead (DURATION_BOUND); ffprobe reads them before the first round."""
    rounds = Mock(return_value=False)
    _loop(monkeypatch, rounds, length=61.0)
    at.main(_cli(tmp_path))
    assert rounds.call_args.args[0].durations == {"t1": 61.0}


def test_a_round_proposes_with_the_tape_lengths_it_was_given(tmp_path, monkeypatch):
    """run_round hands args.durations to propose, so a dead print-length move is never rendered."""
    seen = []
    monkeypatch.setattr(at, "propose", lambda *args: seen.append(args) or {})
    state = {"incumbent": {"a": 1}, "rounds": [], "candidates": {}}
    args = SimpleNamespace(families=("dsp",), gates=None, language="ro", parallel=1, durations={"t1": 61.0})
    assert not at.run_round(args, "cathar", {"t1": "t1.mov"}, tmp_path, state, {"m": "up"}, 1)
    assert seen == [("cathar", {"a": 1}, {"t1": 61.0})]
