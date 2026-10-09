"""The three listening questions (ABX, pair with "same" and one replicate, blend threshold) and the records they write."""

import itertools

import numpy as np
import pytest

from scripts.restoration_quality import ledger, listen_modes


def _record(mode, rid, result=0):
    """A whole ledger record around one of the mode's results, so the ledger's own validation judges it."""
    base = {
        "id": rid,
        "date": "2026-10-09",
        "round": "session0",
        "tape": "vaccin",
        "windows": [[1.0, 11.0]],
        "files": [ledger.file_entry(label, f"{label}.wav") for label in mode.labels],
        "playback": {"device": None, "volume": None},
        "context": {},
    }
    return {**base, **mode.results()[result]}


def _answer_all(mode, pick):
    while not mode.finished():
        trial = mode.trial(len(mode.answers))
        mode.record(trial, pick(trial))


def test_abx_keys_follow_x():
    """X plays A's stimulus exactly when the key is A."""
    mode = listen_modes.AbxMode(["a", "b"], 24, np.random.default_rng(1))
    trials = [mode.trial(index) for index in range(24)]
    assert all((trial["play"][2][1] == {"a": 1.0}) == (trial["key"] == "A") for trial in trials)
    assert {trial["key"] for trial in trials} == {"A", "B"}


def test_abx_results_carry_the_binomial_p():
    """24 right of 24 is heard; the record body passes the ledger's validation."""
    mode = listen_modes.AbxMode(["a", "b"], 24, np.random.default_rng(2))
    _answer_all(mode, lambda trial: trial["key"])
    result = mode.results()[0]
    assert result["n_correct"] == 24 and result["answer"]["heard"]
    assert not ledger.validate_record(_record(mode, "abx-1"))
    assert mode.summary().startswith("24/24 right, p = 0.000: heard")


def test_abx_by_chance_is_not_heard():
    """Always answering A is about half right: not shown to be heard."""
    mode = listen_modes.AbxMode(["a", "b"], 24, np.random.default_rng(3))
    _answer_all(mode, lambda _trial: "A")
    assert not mode.results()[0]["answer"]["heard"]
    assert "not shown to be heard" in mode.summary()


def test_pair_counts_each_pair_with_same_and_the_replicate():
    """Three stimuli twice each plus one replicate: three pair records, every trial counted once, one trial marked."""
    mode = listen_modes.PairMode(["a", "b", "c"], 2, np.random.default_rng(4))
    picks = itertools.cycle(["A", "B", ledger.SAME])
    _answer_all(mode, lambda _trial: next(picks))
    results = mode.results()
    assert mode.total == 7 and len(results) == 3
    assert sum(result["n_trials"] for result in results) == 7
    assert [trial[3] for result in results for trial in result["answer"]["trials"]].count(True) == 1


def test_the_replicate_plays_its_pair_again_swapped():
    """The last trial repeats an earlier pair with A and B swapped and is the only one marked."""
    mode = listen_modes.PairMode(["a", "b", "c"], 1, np.random.default_rng(5))
    trials = [mode.trial(index) for index in range(mode.total)]
    assert trials[-1]["stimuli"] == trials[mode.replicate_of]["stimuli"][::-1]
    assert [trial["replicate"] for trial in trials] == [False, False, False, True]


def test_the_replicate_takes_the_anchor():
    """With an anchor, the replicated pair is always one the anchor plays in."""
    pairs = [listen_modes.PairMode(["a", "b", "c", "z"], 1, np.random.default_rng(seed), anchor="z").order[-1] for seed in range(8)]
    assert all("z" in pair for pair in pairs)


def test_the_summary_reports_whether_the_replicate_held():
    """The same answer twice reads as held; a changed answer names both."""
    steady = listen_modes.PairMode(["a", "b"], 1, np.random.default_rng(6))
    _answer_all(steady, lambda _trial: ledger.SAME)
    changing = listen_modes.PairMode(["a", "b"], 1, np.random.default_rng(6))
    _answer_all(changing, lambda trial: "A" if not trial["replicate"] else ledger.SAME)
    assert steady.summary().endswith("replicate a vs b: the same answer")
    assert changing.summary().endswith(f"replicate a vs b: {changing.order[0][0]}, then same")


def test_pair_records_are_valid_ledger_records():
    """Each pair's record passes the ledger's validation and names the pair it counts; no replicate note before the end."""
    mode = listen_modes.PairMode(["a", "b"], 3, np.random.default_rng(5))
    assert mode.replicate_note() == ""
    _answer_all(mode, lambda trial: "A")
    assert not ledger.validate_record(_record(mode, "pair-1"))
    assert "same 0" in mode.summary()


def test_blend_opens_on_the_full_candidate():
    """The first trial plays the whole candidate against the incumbent; R is always the incumbent."""
    mode = listen_modes.BlendMode(["inc", "cand"], 30, np.random.default_rng(6))
    trial = mode.trial(0)
    slot = 1 if trial["key"] == "A" else 2
    assert trial["x"] == 1.0 and trial["play"][0] == ("R", {"inc": 1.0})
    assert trial["play"][slot][1] == {"inc": 0.0, "cand": 1.0}


def test_blend_posterior_finds_a_simulated_listener():
    """A simulated listener with threshold 0.3: the estimate lands near it and the record is valid."""
    rng = np.random.default_rng(7)
    mode = listen_modes.BlendMode(["inc", "cand"], 60, np.random.default_rng(8))
    _answer_all(
        mode, lambda trial: trial["key"] if rng.random() < listen_modes.p_correct(trial["x"], 0.3) else "AB".replace(trial["key"], "")
    )
    answer = mode.results()[0]["answer"]
    assert answer["band"][0] <= answer["threshold_x"] <= answer["band"][1]
    assert 0.12 < answer["threshold_x"] < 0.6
    assert not ledger.validate_record(_record(mode, "blend-1")) and "90% band" in mode.summary()


def test_x_at_target_inverts_the_psychometric_function():
    """The reported x is the one heard 75% of the time."""
    assert listen_modes.p_correct(listen_modes.x_at_target(0.4), 0.4) == pytest.approx(listen_modes.TARGET_P)
