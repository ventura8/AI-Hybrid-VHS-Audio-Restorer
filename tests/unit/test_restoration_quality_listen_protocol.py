"""The blind-block protocol: familiarisation, interleaved catch trials scored apart, block flags and each block's power."""

import numpy as np
import pytest

from scripts.restoration_quality import ledger, listen_modes, listen_protocol

CATCH, SCORED, FAMILIAR = listen_protocol.CATCH, listen_protocol.SCORED, listen_protocol.FAMILIAR


def _run(protocol, pick):
    """Answers every presentation with `pick(trial)`; returns the feedback texts in order."""
    feedback = []
    while not protocol.finished():
        trial = protocol.trial()
        feedback.append(protocol.take(trial, pick(trial)))
    return feedback


# ----------------------------------------------------------------------------- the schedule


def test_one_catch_per_eight_scored_trials_at_least_one():
    """24 scored trials take 3 catches, 30 take 4, 2 take 1."""
    assert [listen_protocol.default_catches(n) for n in (24, 30, 2, 8, 9)] == [3, 4, 1, 1, 2]


@pytest.mark.parametrize("seed", range(6))
def test_the_schedule_puts_practice_first_and_one_catch_in_each_stretch(seed):
    """Two practice trials, then 24 scored with 3 catches: never a catch first, one in each third of the block."""
    order = listen_protocol.schedule(24, 2, 3, np.random.default_rng(seed))
    assert (order[:3], order.count(SCORED), order.count(CATCH)) == ([FAMILIAR, FAMILIAR, SCORED], 24, 3)
    assert [(count - 1) // 8 for count in _scored_before_each_catch(order)] == [0, 1, 2]


def _scored_before_each_catch(order):
    """How many scored trials precede each catch trial."""
    return [order[:index].count(SCORED) for index, kind in enumerate(order) if kind == CATCH]


def test_no_catches_and_more_catches_than_trials():
    """No catch asked, none placed; more catches than scored trials are cut to one per scored trial."""
    rng = np.random.default_rng(1)
    assert listen_protocol.schedule(3, 0, 0, rng) == [SCORED] * 3
    assert listen_protocol.schedule(2, 0, 5, rng) == [SCORED, CATCH, SCORED, CATCH]


# ----------------------------------------------------------------------------- catch trials and flags


def test_a_catch_is_caught_by_its_key_or_in_a_pair_by_naming_a_side():
    """With a key the key must be chosen; in a pair (no key) any side counts and "same" misses."""
    assert listen_protocol.caught({"key": "A"}, "A")
    assert not listen_protocol.caught({"key": "A"}, listen_modes.CANT_TELL)
    assert listen_protocol.caught({"key": None}, "B")
    assert not listen_protocol.caught({"key": None}, ledger.SAME)


def test_a_one_button_block_reads_no_signal_never_disengaged():
    """Session 0's case: B on every scored trial is "no signal", even with a missed catch; the listener's null stands."""
    missed = [["catch:air", "A", "B", False]]
    assert listen_protocol.block_flags(["B"] * 16, missed) == [listen_protocol.NO_SIGNAL]
    assert listen_protocol.block_flags([ledger.SAME] * 7, []) == [listen_protocol.NO_SIGNAL]


def test_only_a_missed_catch_reads_disengaged():
    """Mixed answers with a missed catch read disengaged; with every catch caught, or a single answer, nothing."""
    caught = [["catch:air", "A", "A", True]]
    missed = caught + [["catch:raw", "B", "A", False]]
    assert listen_protocol.block_flags(["A", "B", "A"], missed) == [listen_protocol.DISENGAGED]
    assert not listen_protocol.block_flags(["A", "B", "A"], caught)
    assert not listen_protocol.block_flags(["A"], [])


# ----------------------------------------------------------------------------- the protocol through a block


def _abx_protocol(seed=2, familiar=2, catches=3):
    rng = np.random.default_rng(seed)
    mode = listen_modes.AbxMode(["a", "b"], 24, rng)
    labels = (listen_modes.CATCH_AIR, listen_modes.CATCH_RAW)
    return mode, listen_protocol.Protocol(mode, rng, familiar=familiar, catches=catches, catch_labels=labels)


def test_practice_answers_get_feedback_and_never_reach_the_score():
    """Two practice trials answered wrongly: feedback names the answer, the mode counts none of it, the context counts both."""
    mode, protocol = _abx_protocol()
    feedback = _run(protocol, lambda trial: "B" if trial["key"] == "A" else "A")
    assert feedback[0].startswith("Not this time: the answer was ")
    assert (set(feedback[2:]), len(mode.answers), protocol.context()["familiarisation"]) == ({""}, 24, {"trials": 2, "right": 0})


def test_catch_answers_are_scored_apart():
    """Every answer right: 24 scored in the mode, 3 catches in the context, both kinds of control used, nothing flagged."""
    mode, protocol = _abx_protocol()
    _run(protocol, lambda trial: trial["key"])
    context = protocol.context()
    assert (mode.right_answers(), context["catch"]["caught"], context["catch"]["missed"], context["block_flags"]) == (24, 3, 0, [])
    controls = {entry[0] for entry in context["catch"]["trials"]}
    assert (controls, protocol.note()) == ({listen_modes.CATCH_AIR, listen_modes.CATCH_RAW}, "catch trials 3/3 caught")


def test_a_block_answered_at_random_with_a_missed_catch_is_disengaged():
    """Wrong on every catch but answering both buttons in the scored trials: disengaged, and the note says so."""
    _mode, protocol = _abx_protocol(seed=3)
    picks = iter(["A", "B"] * 20)

    def pick(trial):
        if trial["phase"] == CATCH:
            return "B" if trial["key"] == "A" else "A"
        return next(picks)

    _run(protocol, pick)
    assert protocol.context()["block_flags"] == [listen_protocol.DISENGAGED]
    assert protocol.note() == "catch trials 0/3 caught | disengaged"


def test_the_progress_counts_practice_then_test():
    """The page reads practice 1 of 2, then test 1 of 27 (24 scored and 3 catches)."""
    _mode, protocol = _abx_protocol()
    assert protocol.progress() == {"phase": "practice", "number": 1, "of": 2}
    for _practice in range(2):
        assert protocol.take(protocol.trial(), "A").endswith(".")
    assert protocol.progress() == {"phase": "test", "number": 1, "of": 27}


def test_a_block_without_catch_trials_says_so():
    """No catch asked: the note says none ran; a block of two scored trials is flagged on its buttons alone."""
    rng = np.random.default_rng(4)
    mode = listen_modes.PairMode(["a", "b"], 1, rng)
    protocol = listen_protocol.Protocol(mode, rng)
    _run(protocol, lambda _trial: ledger.SAME)
    assert protocol.note() == "no catch trials | no signal"
    assert protocol.current() is None


def test_trials_carry_their_phase_and_catch_label():
    """Every built trial names its phase; a catch trial names its control, which its buttons play."""
    _mode, protocol = _abx_protocol(familiar=0, catches=24)
    first = protocol.trial()
    protocol.take(first, "A")
    catch = protocol.trial()
    assert first["phase"] == SCORED
    assert catch["phase"] == CATCH
    assert catch["catch"] in catch["stimuli"]


# ----------------------------------------------------------------------------- power


def test_the_binomial_power_matches_leventhal_and_the_design_table():
    """16 trials: 12 needed, power 0.167 at 60% (a real difference missed 83% of the time); 24 trials: 17, power 0.77 at 75%."""
    assert listen_protocol.critical_correct(16) == 12
    assert listen_protocol.binomial_power(16, 0.6) == pytest.approx(0.167, abs=0.001)
    assert listen_protocol.critical_correct(24) == 17
    assert listen_protocol.binomial_power(24, 0.75) == pytest.approx(0.77, abs=0.01)


def test_too_few_trials_have_no_power():
    """Under five trials no score reaches p 0.05: no critical score, power 0."""
    assert listen_protocol.critical_correct(4) is None
    assert listen_protocol.binomial_power(4, 0.9) == 0.0
    assert listen_protocol.binomial_power_record(2)["at_hit_rate"] == {"0.6": 0.0, "0.75": 0.0, "0.9": 0.0}


def test_power_of_gives_an_abx_record_the_binomial_power():
    """An abx record gets the binomial power: 24 trials need 17 right, power 0.7662 at a 75% hit rate."""
    abx = listen_protocol.power_of({"question": {"type": "abx"}, "n_trials": 24})
    assert abx["critical_correct"] == 17
    assert abx["at_hit_rate"]["0.75"] == pytest.approx(0.7662, abs=1e-4)


def test_power_of_gives_a_pair_record_the_sign_test():
    """A pair record gets the binomial power as a one-sided sign test; two trials have no critical score."""
    pair = listen_protocol.power_of({"question": {"type": "pair"}, "n_trials": 2})
    assert pair["test"].startswith("one-sided sign test")
    assert pair["critical_correct"] is None


def test_a_blend_record_gets_the_simulated_power():
    """A blend record gets the simulated QUEST power; with no trials no band ends under the full candidate."""
    blend = listen_protocol.blend_power_record(0)
    assert blend["band_ends_under_full"] == 0.0
    assert blend["test"] == "simulated QUEST"


def test_blend_power_grows_with_the_block():
    """A listener at x = 0.5: a 30-trial blend shows the full candidate heard more often than a 6-trial one, deterministically."""
    short, full = listen_protocol.blend_power(6, runs=40), listen_protocol.blend_power(30, runs=40)
    assert 0.0 <= short < full <= 1.0
    assert listen_protocol.blend_power(30, runs=40) == full
    assert listen_protocol.blend_power_record(12, runs=10)["runs"] == 10


def test_cant_tell_answers_do_not_break_a_one_button_block():
    """One "can't tell" among fifteen A answers is still one button: "no signal"; only "can't tell" is one too; A and B are not."""
    missed = [["catch:air", "B", "A", False]]
    assert listen_protocol.block_flags([ledger.CANT_TELL] + ["A"] * 15, missed) == [listen_protocol.NO_SIGNAL]
    assert listen_protocol.one_button([ledger.CANT_TELL] * 3)
    assert not listen_protocol.one_button(["A", ledger.CANT_TELL, "B"])
