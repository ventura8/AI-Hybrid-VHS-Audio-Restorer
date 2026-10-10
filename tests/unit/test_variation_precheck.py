"""The plan 3.2a pre-check: the per-tape oracle over a loop round, each pick compared with the best, and its verdict."""

import builtins
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts import variation_precheck as vp

RATE = 44100
SCRIPT = Path(vp.__file__)


def _tone_bursts(seconds=3.0, seed=5):
    """A 440 Hz tone in 0.25 s bursts over a faint hiss."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    gate = ((t % 0.5) < 0.25).astype(np.float64)
    return (0.1 * np.sin(2 * np.pi * 440.0 * t) * gate + 1e-4 * rng.standard_normal(len(t))).astype(np.float32)


def _verdict(mean_rank, **per_tape):
    return {"mean_rank": mean_rank, "per_tape": per_tape}


def _round(number=1):
    """Three scored candidates on two tapes and one tie that carries no ranks.

    `best` leads on the mean; `same` leads on tape `a` and `loud` on tape `b`.
    """
    return {
        "round": number,
        "verdicts": {
            "best": _verdict(2.0, a=2.0, b=2.0),
            "same": _verdict(2.5, a=1.0, b=4.0),
            "loud": _verdict(2.6, a=3.0, b=1.5),
            "tied": {"mean_rank": None, "per_tape": {}},
        },
    }


def _loop(tmp_path, renders=True):
    """A loop output: state.json with one round and, when `renders`, the candidates' WAVs."""
    out = tmp_path / "loop"
    voice = _tone_bursts()
    audio = {"best": voice, "same": voice, "loud": voice * np.float32(1.5)}
    for cand, samples in audio.items():
        folder = out / "cands" / cand
        folder.mkdir(parents=True)
        if renders:
            for tape in ("a", "b"):
                sf.write(str(folder / f"{tape}.wav"), samples, RATE, subtype="FLOAT")
    state = out / "state.json"
    state.write_text(json.dumps({"rounds": [_round()], "candidates": {}}), encoding="utf-8")
    return state


def test_the_oracle_picks_each_tapes_best_rank_against_the_single_best():
    """The best is the lowest mean rank among the scored candidates; the oracle's gain is in rank units."""
    report = vp.oracle(_round())
    assert (report["best"], report["n_scored"]) == ("best", 3)
    assert report["picks"] == {"a": {"pick": "same", "rank_gain": 1.0}, "b": {"pick": "loud", "rank_gain": 0.5}}
    assert report["oracle_gain"] == 0.75


def test_a_round_with_one_scored_candidate_has_no_oracle():
    """Ties and inert candidates carry no per-tape ranks, so they never become the best or a pick."""
    single = {"round": 1, "verdicts": {"best": _verdict(1.0, a=1.0), "tied": {"per_tape": {}}}}
    assert vp.oracle(single) is None
    assert vp.oracle({}) is None


def test_the_last_scored_round_is_read_unless_one_is_named():
    """The plateau round is the last one with two scored candidates; a named round that is missing reads None."""
    empty = {"round": 3, "verdicts": {"x": _verdict(1.0, a=1.0)}}
    state = {"rounds": [_round(1), _round(2), empty]}
    assert vp.choose_round(state)["round"] == 2
    assert vp.choose_round(state, 1)["round"] == 1
    assert vp.choose_round(state, 3) is None
    assert vp.choose_round({}) is None


def test_a_pick_is_selectable_only_when_audible_and_above_what_was_not_heard():
    """Audible and larger than the unheard difference: selectable; audible but smaller: below_heard; else a tie."""
    assert vp.classify({"audible": False, "diff_db": -5.0}, -17.6) == "tie"
    assert vp.classify({"audible": True, "diff_db": -8.0}, -17.6) == "selectable"
    assert vp.classify({"audible": True, "diff_db": -30.0}, -17.6) == "below_heard"
    assert vp.classify({"audible": True, "diff_db": None}, -17.6) == "selectable"


def test_a_missing_render_is_reported_not_compared(tmp_path):
    """A pruned candidate's render is `missing`; the comparison is never called."""

    def never(*_args):
        raise AssertionError("compared a missing render")

    pick = vp.check_pick(tmp_path, "best", "a", {"pick": "same", "rank_gain": 1.0}, -17.6, compare=never)
    assert pick == {"pick": "same", "rank_gain": 1.0, "class": "missing"}


def test_a_state_is_checked_on_its_renders(tmp_path):
    """The identical render is a trusted tie whose rank gain is noise; the 1.5x louder one is selectable."""
    report = vp.check_state(_loop(tmp_path))
    assert report["picks"]["a"]["class"] == "tie"
    assert report["picks"]["b"]["class"] == "selectable"
    assert report["picks"]["b"]["diff_db"] > -17.6
    assert (report["selectable"], report["tie_rank_gains"], report["passes"]) == (["b"], [1.0], True)


def test_a_lower_heard_level_than_the_difference_makes_it_unheard(tmp_path):
    """With a listener who fails a -3 dB difference, the louder render no longer passes."""
    report = vp.check_state(_loop(tmp_path), heard_db=-3.0)
    assert report["picks"]["b"]["class"] == "below_heard"
    assert (report["selectable"], report["passes"]) == ([], False)


def test_a_state_without_a_scored_round_reads_none(tmp_path):
    """A loop stopped before its first scored round has nothing to check."""
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"rounds": []}), encoding="utf-8")
    assert vp.check_state(state) is None


def test_the_cli_prints_a_line_per_state_and_writes_the_json(tmp_path, capsys):
    """One line per state, the empty one included, and every report in the JSON file."""
    state = _loop(tmp_path)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"rounds": []}), encoding="utf-8")
    target = tmp_path / "out" / "precheck.json"
    reports = vp.main([str(state), str(empty), "--round", "1", "--json", str(target)])
    out = capsys.readouterr().out
    assert "PASSES: ABX b" in out
    assert "no round with two scored candidates" in out
    assert json.loads(target.read_text(encoding="utf-8")) == reports


def test_the_cli_says_nothing_is_selectable(tmp_path, capsys):
    """Missing renders leave nothing to select, and the line says so."""
    vp.main([str(_loop(tmp_path, renders=False))])
    assert "nothing selectable" in capsys.readouterr().out


def test_the_script_runs_standalone_when_scripts_is_not_on_the_path(tmp_path, monkeypatch, capsys):
    """Run as a file, the script puts the repository on sys.path itself when `scripts` does not import."""
    state = _loop(tmp_path, renders=False)
    real_import = builtins.__import__
    refused = []

    def first_import_fails(name, *args, **kwargs):
        if name == "scripts.cli_paths" and not refused:
            refused.append(name)
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", first_import_fails)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(state)])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    assert refused == ["scripts.cli_paths"]
    assert sys.path[0] == str(SCRIPT.resolve().parent.parent)
    assert "nothing selectable" in capsys.readouterr().out
