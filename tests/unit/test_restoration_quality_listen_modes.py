"""The three listening questions (ABX, pair with "same" and one replicate, blend threshold), "can't tell", catch and practice trials."""

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
    assert result["n_correct"] == 24
    assert result["answer"]["heard"]
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
    assert mode.total == 7
    assert len(results) == 3
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
    assert trial["x"] == 1.0
    assert trial["play"][0] == ("R", {"inc": 1.0})
    assert trial["play"][slot][1] == {"inc": 0.0, "cand": 1.0}


def _simulated_blend():
    """A blend run answered by a simulated listener whose threshold is 0.3."""
    rng = np.random.default_rng(7)
    mode = listen_modes.BlendMode(["inc", "cand"], 60, np.random.default_rng(8))
    _answer_all(
        mode, lambda trial: trial["key"] if rng.random() < listen_modes.p_correct(trial["x"], 0.3) else "AB".replace(trial["key"], "")
    )
    return mode


def test_blend_posterior_finds_a_simulated_listener():
    """A simulated listener with threshold 0.3: the estimate lands near it, inside its own band."""
    answer = _simulated_blend().results()[0]["answer"]
    assert answer["band"][0] <= answer["threshold_x"] <= answer["band"][1]
    assert 0.12 < answer["threshold_x"] < 0.6


def test_the_simulated_listeners_blend_record_is_valid():
    """The simulated listener's blend record passes the ledger's validation and the summary names its band."""
    mode = _simulated_blend()
    assert not ledger.validate_record(_record(mode, "blend-1"))
    assert "90% band" in mode.summary()


def test_x_at_target_inverts_the_psychometric_function():
    """The reported x is the one heard 75% of the time."""
    assert listen_modes.p_correct(listen_modes.x_at_target(0.4), 0.4) == pytest.approx(listen_modes.TARGET_P)


# ----------------------------------------------------------------------------- "can't tell", kept apart from the score


def _unsure_where(pattern):
    """A listener who answers "can't tell" where `pattern` says so and right everywhere else."""
    unsure = iter(pattern)
    return lambda trial: listen_modes.CANT_TELL if next(unsure) else trial["key"]


def test_abx_cant_tell_is_kept_but_not_scored():
    """Two "can't tell" answers of six: the score counts the four decided ones, the record keeps all six in order."""
    mode = listen_modes.AbxMode(["a", "b"], 6, np.random.default_rng(10))
    _answer_all(mode, _unsure_where([True, False, False, True, False, False]))
    result = mode.results()[0]
    assert (result["n_trials"], result["n_correct"], result["answer"]["cant_tell"]) == (4, 4, 2)
    choices = [choice for _key, choice in result["answer"]["trials"]]
    assert (choices.count(listen_modes.CANT_TELL), result["answer"]["p_value"]) == (2, ledger.binomial_p(4, 4))
    assert (ledger.validate_record(_record(mode, "abx-ct")), "2 can't tell (not scored)" in mode.summary()) == ([], True)


def test_an_abx_block_of_only_cant_tell_is_still_a_valid_record():
    """Every answer "can't tell": 0 of 0 decided, p 1, not heard, and the ledger takes it because the count says why."""
    mode = listen_modes.AbxMode(["a", "b"], 3, np.random.default_rng(11))
    _answer_all(mode, lambda _trial: listen_modes.CANT_TELL)
    result = mode.results()[0]
    assert (result["n_trials"], result["n_correct"], result["answer"]["heard"]) == (0, 0, False)
    assert not ledger.validate_record(_record(mode, "abx-none"))


def test_blend_cant_tell_leaves_the_posterior_and_is_recorded_apart():
    """A "can't tell" moves nothing: the next trial sits where it would have, and the record lists its level apart."""
    mode = listen_modes.BlendMode(["inc", "cand"], 3, np.random.default_rng(12))
    first = mode.trial(0)
    mode.record(first, first["key"])
    before = mode.log_posterior.copy()
    second = mode.trial(1)
    mode.record(second, listen_modes.CANT_TELL)
    assert (bool(np.array_equal(mode.log_posterior, before)), mode.trial(2)["x"]) == (True, second["x"])
    result = mode.results()[0]
    assert (result["answer"]["cant_tell"], result["n_trials"], result["n_correct"]) == ([second["x"]], 1, 1)
    assert "1 can't tell" in mode.summary()


def test_forced_choice_trials_offer_cant_tell_and_ask_for_a_guess():
    """ABX and blend offer A, B and "can't tell" and ask for a guess; pair keeps its "same"."""
    abx = listen_modes.AbxMode(["a", "b"], 2, np.random.default_rng(13)).trial(0)
    blend = listen_modes.BlendMode(["a", "b"], 2, np.random.default_rng(13)).trial(0)
    pair = listen_modes.PairMode(["a", "b"], 1, np.random.default_rng(13)).trial(0)
    assert abx["choices"] == blend["choices"] == ["A", "B", listen_modes.CANT_TELL]
    assert "Guess when unsure" in abx["prompt"] and "Guess when unsure" in blend["prompt"]
    assert pair["choices"] == ["A", "B", ledger.SAME]


# ----------------------------------------------------------------------------- catch and practice trials look like scored ones


def test_catch_trials_take_the_shape_of_each_mode():
    """A catch trial has the scored trial's buttons, prompt and choices, over the first stimulus and the control."""
    rng = np.random.default_rng(14)
    modes = [
        listen_modes.AbxMode(["a", "b"], 2, rng),
        listen_modes.BlendMode(["a", "b"], 2, rng),
        listen_modes.PairMode(["a", "b"], 1, rng),
    ]
    for mode in modes:
        scored, catch = mode.trial(0), mode.catch_trial(listen_modes.CATCH_AIR, "A")
        assert [name for name, _weights in catch["play"]] == [name for name, _weights in scored["play"]]
        assert (catch["prompt"], catch["choices"]) == (scored["prompt"], scored["choices"])
        assert catch["stimuli"] == ["a", listen_modes.CATCH_AIR]


def test_the_catch_control_sits_where_the_key_says():
    """ABX: X plays the control when the key is B; blend: the control is the odd one in the key's slot; pair: on the key's side."""
    rng = np.random.default_rng(15)
    abx = listen_modes.AbxMode(["a", "b"], 2, rng).catch_trial(listen_modes.CATCH_RAW, "B")
    blend = listen_modes.BlendMode(["a", "b"], 2, rng).catch_trial(listen_modes.CATCH_RAW, "A")
    pair = listen_modes.PairMode(["a", "b"], 1, rng).catch_trial(listen_modes.CATCH_RAW, "B")
    assert abx["play"][2] == ("X", {listen_modes.CATCH_RAW: 1.0})
    assert blend["play"][1] == ("A", {listen_modes.CATCH_RAW: 1.0}) and blend["play"][0] == ("R", {"a": 1.0})
    assert pair["play"][0] == ("A", {listen_modes.CATCH_RAW: 1.0}) and pair["key"] is None


def test_practice_trials_play_the_exaggerated_pair():
    """ABX practice plays the familiarisation pair as A and B; blend practice plays it as R and the odd one."""
    rng = np.random.default_rng(16)
    abx = listen_modes.AbxMode(["a", "b"], 2, rng).familiar_trial(0, "A")
    blend = listen_modes.BlendMode(["a", "b"], 2, rng).familiar_trial(0, "B")
    assert abx["play"][:2] == [("A", {listen_modes.FAMILIAR_A: 1.0}), ("B", {listen_modes.FAMILIAR_B: 1.0})]
    assert blend["play"] == [
        ("R", {listen_modes.FAMILIAR_A: 1.0}),
        ("A", {listen_modes.FAMILIAR_A: 1.0}),
        ("B", {listen_modes.FAMILIAR_B: 1.0}),
    ]


def test_scored_choices_include_cant_tell():
    """The buttons a block was answered with, "can't tell" among them: what tells a one-button block."""
    mode = listen_modes.AbxMode(["a", "b"], 2, np.random.default_rng(17))
    picks = iter(["A", listen_modes.CANT_TELL])
    _answer_all(mode, lambda _trial: next(picks))
    assert mode.scored_choices() == ["A", listen_modes.CANT_TELL]
