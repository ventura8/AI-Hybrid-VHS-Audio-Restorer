"""The loop's family-wise learned veto: a sign test across the clips, Holm across the readings, a third past the limit."""

import numpy as np
import pytest

from scripts import autotune_guards as ag
from scripts import autotune_judging as judging
from scripts.restoration_quality import reward

CER = "speech.cer.output.median"


def test_the_sign_test_gives_the_textbook_numbers():
    assert judging.sign_test_p(4, 0) == pytest.approx(1 / 16)
    assert judging.sign_test_p(10, 2) == pytest.approx(79 / 4096)
    assert judging.sign_test_p(0, 0) == 1.0


@pytest.mark.parametrize(
    ("pvalues", "rejected"),
    [
        ({"a": 0.001, "b": 0.02, "c": 0.04}, {"a", "b", "c"}),
        ({"a": 0.001, "b": 0.03, "c": 0.04}, {"a"}),
        ({"a": 0.02, "b": 0.03}, {"a", "b"}),
        ({"a": 0.03, "b": 0.04}, set()),
    ],
)
def test_holm_steps_down_until_the_first_p_value_over_its_threshold(pvalues, rejected):
    assert judging.holm(pvalues) == rejected


def test_seven_clips_are_the_fewest_a_sign_test_can_resolve_over_the_grids_vetoes():
    assert [judging.resolvable_clips(n) for n in (1, 4, 6)] == [5, 7, 7]


def _clips(values, base=0.10):
    incumbent = {f"t{i}": {CER: base} for i in range(len(values))}
    candidate = {f"t{i}": ({} if value is None else {CER: value}) for i, value in enumerate(values)}
    return candidate, incumbent


VETO = {CER: reward.Veto("up", 0.038)}
JITTERED = judging.Floors({}, {CER: 0.001})


def test_one_clip_far_past_the_limit_no_longer_vetoes_a_twelve_clip_candidate():
    """The any-clip rule's typical false veto: one clip-reading moved, eleven did not."""
    candidate, incumbent = _clips([0.30] + [0.10] * 11)
    assert ag.learned_vetoes(candidate, incumbent, VETO)
    assert judging.familywise_vetoes(candidate, incumbent, VETO, JITTERED) == []


def test_a_judge_worse_on_most_clips_and_past_the_limit_on_a_third_vetoes():
    candidate, incumbent = _clips([0.15] * 5 + [0.12] * 5 + [0.09] + [0.10])
    found = judging.familywise_vetoes(candidate, incumbent, VETO, JITTERED)
    assert found == [f"{CER} worse on 10 of 12 clips (1 better), 5 past the limit; sign test p 0.00586, Holm over 1"]


def test_consistent_moves_under_the_limit_do_not_veto():
    """Ten of twelve clips worse past the jitter floor, none past the veto's own limit: below what was not heard."""
    candidate, incumbent = _clips([0.12] * 10 + [0.10] * 2)
    assert judging.familywise_vetoes(candidate, incumbent, VETO, JITTERED) == []


def test_moves_inside_the_jitter_floor_are_still():
    moves = judging.clip_moves(*_clips([0.1005, 0.0995, 0.2]), CER, VETO[CER], 0.001)
    assert moves == judging.ClipMoves(clips=3, worse=1, better=0, past_limit=1)


def test_on_four_tapes_every_tape_must_move_the_worse_way():
    """Four tapes cannot reach significance: unanimity, with a third past the limit, stands in for the test."""
    candidate, incumbent = _clips([0.15, 0.15, 0.11, 0.12])
    assert judging.familywise_vetoes(candidate, incumbent, VETO, JITTERED) == [
        f"{CER} worse on 4 of 4 clips (0 better), 2 past the limit; unanimous (too few clips)"
    ]
    three_of_four = _clips([0.15, 0.15, 0.11, 0.10])
    assert judging.familywise_vetoes(*three_of_four, VETO, JITTERED) == []


def test_a_lost_reading_counts_as_a_move_past_the_limit():
    candidate, incumbent = _clips([None, None])
    assert judging.familywise_vetoes(candidate, incumbent, VETO) == [
        f"{CER} worse on 2 of 2 clips (0 better), 2 past the limit; unanimous (too few clips)"
    ]


def test_a_reading_the_incumbent_never_read_vetoes_nothing():
    candidate = {"t1": {CER: 0.9}}
    assert judging.clip_moves(candidate, {"t1": {}}, CER, VETO[CER]) == judging.ClipMoves()
    assert judging.familywise_vetoes(candidate, {"t1": {}}, VETO) == []


def test_holm_corrects_across_the_grids_veto_readings():
    """p 0.0107 (9 of 10 moved worse) passes alone, not as the first of six readings (0.05 / 6)."""
    values = [0.15] * 9 + [0.09] + [0.10] * 2
    candidate, incumbent = _clips(values)
    assert judging.familywise_vetoes(candidate, incumbent, VETO, JITTERED)
    six = {CER: VETO[CER], **{f"mos.other{i}.delta.median": reward.Veto("down", 1.0) for i in range(5)}}
    assert judging.familywise_vetoes(candidate, incumbent, six, JITTERED) == []


NULL_VETOES = {f"mos.r{i}.delta.median": reward.Veto("down", 0.2) for i in range(4)}
NULL_INCUMBENT = {f"t{c}": dict.fromkeys(NULL_VETOES, 3.0) for c in range(12)}


def _null_candidate(rng):
    """The incumbent with every clip-reading moved by normal noise whose p95 is the veto limit."""
    return {slug: {key: value + rng.normal(0.0, 0.2 / 1.96) for key, value in flat.items()} for slug, flat in NULL_INCUMBENT.items()}


def _null_rates(draws=400):
    """`(any-clip rate, family-wise rate)` over `draws` null candidates."""
    rng = np.random.default_rng(20261011)
    floors = judging.Floors({}, dict.fromkeys(NULL_VETOES, 0.01))
    candidates = [_null_candidate(rng) for _ in range(draws)]
    any_clip = sum(bool(ag.learned_vetoes(c, NULL_INCUMBENT, NULL_VETOES)) for c in candidates)
    familywise = sum(bool(judging.familywise_vetoes(c, NULL_INCUMBENT, NULL_VETOES, floors)) for c in candidates)
    return any_clip / draws, familywise / draws


def test_the_null_simulation_vetoes_under_five_percent_where_the_any_clip_rule_vetoed_most():
    """Twelve clips, four readings, each clip moved by differences nobody heard: the old rule vetoed most such candidates."""
    any_clip, familywise = _null_rates()
    assert familywise <= 0.05
    assert any_clip >= 0.5
