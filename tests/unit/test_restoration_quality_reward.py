"""The group reward: two-sided distances, dead zones, pairwise win rates per family, the worst family, and constraints."""

import math

import pytest

from scripts.restoration_quality import reward as rw

AIR = "dsp.balance_air_db.delta.median"
GAP = "dsp.gap_residual_db.output.median"
SIB = "sibilance.texture_db.delta.median"


def _grid(**families):
    """One reading per family, target 0, no dead zone, scale 1: `_grid(timbre=AIR, noise=GAP)`."""
    return {reading: {"target": 0.0, "family": family} for family, reading in families.items()}


def _two_families():
    return _grid(timbre=AIR, noise=GAP)


def _summary(entry):
    return entry.reward, entry.status, entry.eligible


def test_the_distance_is_two_sided_and_zero_inside_the_dead_zone():
    """Above or below the target by the same amount reads the same; missing, NaN and booleans read None."""
    spec = rw.ReadingSpec(target=1.0, family="timbre", dead_zone=0.5, scale=2.0, weight=3.0)
    assert spec.distance(1.4) == 0.0
    assert spec.distance(2.5) == pytest.approx(1.5)
    assert spec.distance(-0.5) == spec.distance(2.5)
    assert [spec.distance(value) for value in (None, math.nan, True)] == [None, None, None]


def test_a_grid_entry_takes_defaults_for_its_dead_zone_scale_and_weight():
    """Only target and family are required; a parsed grid parses to itself."""
    specs = rw.parse_grid({AIR: {"target": 1, "family": "timbre"}})
    assert specs[AIR] == rw.ReadingSpec(target=1.0, family="timbre", dead_zone=0.0, scale=1.0, weight=1.0)
    assert rw.parse_grid(specs) == specs


@pytest.mark.parametrize(
    "entry",
    [
        {"family": "timbre"},
        {"target": 0.0},
        {"target": math.inf, "family": "timbre"},
        {"target": 0.0, "family": "timbre", "scale": 0.0},
        {"target": 0.0, "family": "timbre", "dead_zone": -0.1},
        {"target": 0.0, "family": "timbre", "weight": -1.0},
        {"target": 0.0, "family": "timbre", "weight": "heavy"},
    ],
)
def test_an_unusable_grid_entry_is_refused_by_name(entry):
    """A missing key, a non-finite target or a number out of range names the reading."""
    with pytest.raises(ValueError, match=AIR):
        rw.parse_grid({AIR: entry})


def test_a_miss_inside_the_dead_zone_ties_with_the_target_and_one_outside_loses():
    """Inside the dead zone both read 0 and tie; outside, the distance decides."""
    grid = {AIR: {"target": 1.0, "dead_zone": 0.3, "scale": 0.5, "family": "timbre"}}
    rewards = rw.group_rewards({"at_target": {AIR: 1.0}, "inside": {AIR: 1.25}, "outside": {AIR: 2.0}}, grid)
    assert rewards["at_target"].reward == rewards["inside"].reward == 0.75
    assert rewards["outside"].reward == 0.0
    assert [entry.candidate for entry in rw.ranked(rewards)] == ["at_target", "inside", "outside"]


def test_too_bright_and_too_dull_by_the_same_amount_are_equally_far():
    """Two-sided: a lift and a cut of the same size are the same fault."""
    rewards = rw.group_rewards({"bright": {AIR: 2.0}, "dull": {AIR: -2.0}, "accepted": {AIR: 0.0}}, _grid(timbre=AIR))
    assert rewards["bright"].reward == rewards["dull"].reward == 0.25
    assert rewards["accepted"].reward == 1.0


def test_the_worst_family_decides_so_noise_cannot_buy_timbre():
    """The best noise reading with the worst timbre loses to the candidate that is fair on both."""
    group = {"a": {AIR: 0.0, GAP: 3.0}, "b": {AIR: 1.0, GAP: 1.0}, "c": {AIR: 2.0, GAP: 0.0}}
    rewards = rw.group_rewards(group, _two_families())
    assert rw.ranked(rewards)[0].candidate == "b"
    assert (rewards["a"].worst_family, rewards["c"].worst_family) == ("noise", "timbre")
    assert rewards["a"].win_rates == {"timbre": 1.0, "noise": 0.0}
    assert rewards["b"].reward == 0.5


def test_a_hard_gate_or_a_verdict_reversal_excludes_without_giving_free_wins():
    """Excluded candidates read 0 and are not compared, so beating them earns nobody a point."""
    group = {"good": {AIR: 0.0}, "worse": {AIR: 1.0}, "gated": {AIR: 9.0}, "reversed": {AIR: 8.0}}
    constraints = {"gated": {"hard_gate": True}, "reversed": rw.Constraints(verdict_reversal=True)}
    rewards = rw.group_rewards(group, _grid(timbre=AIR), constraints)
    assert _summary(rewards["gated"]) == _summary(rewards["reversed"]) == (0.0, "excluded", False)
    assert (rewards["good"].reward, rewards["worse"].reward) == (1.0, 0.0)


def test_an_inaudible_candidate_ties_and_a_learned_veto_excludes():
    """Inaudible reads the group baseline 0.5; a learned-judge veto excludes like a hard gate."""
    group = {"good": {AIR: 0.0}, "same": {AIR: 0.0}, "vetoed": {AIR: 0.0}, "worse": {AIR: 1.0}}
    constraints = {"same": {"inaudible": True}, "vetoed": {"learned_veto": True}}
    rewards = rw.group_rewards(group, _grid(timbre=AIR), constraints)
    assert _summary(rewards["same"]) == (0.5, "tie", False)
    assert _summary(rewards["vetoed"]) == (0.0, "excluded", False)
    assert rewards["good"].reward == 1.0


def test_a_flagged_candidate_is_compared_but_capped_and_cannot_win():
    """A flagged candidate still counts against the others, but its own reward stops at 0.5."""
    group = {"flagged": {AIR: 0.0}, "middle": {AIR: 1.0}, "last": {AIR: 2.0}}
    rewards = rw.group_rewards(group, _grid(timbre=AIR), {"flagged": {"listener_flag": True}})
    assert _summary(rewards["flagged"]) == (0.5, "capped", False)
    assert rewards["middle"].reward == 0.5
    assert [entry.candidate for entry in rw.ranked(rewards)] == ["middle", "last"]


def test_the_test_retest_floor_turns_a_sub_noise_difference_into_a_tie():
    """A difference inside the summed noise floors is a tie; without floors it is a win."""
    group = {"a": {AIR: 0.0}, "b": {AIR: 0.2}}
    report = {"readings": {AIR: {"floor": 0.3}, GAP: {"floor": math.nan}}}
    floors = rw.noise_floors(report)
    assert floors == {AIR: 0.3}
    assert rw.group_rewards(group, _grid(timbre=AIR), floors=floors)["a"].reward == 0.5
    assert rw.group_rewards(group, _grid(timbre=AIR))["a"].reward == 1.0


def test_a_candidate_without_anyone_to_compare_with_is_unrated():
    """Alone, or in a group where nobody has a grid reading: 0.5, unrated, never a winner."""
    lone = rw.group_rewards({"only": {AIR: 0.0}}, _grid(timbre=AIR))
    unread = rw.group_rewards({"a": {}, "b": None}, _two_families())
    assert _summary(lone["only"]) == (0.5, "unrated", False)
    assert [_summary(entry) for entry in unread.values()] == [(0.5, "unrated", False)] * 2


def _dulled_group():
    """The reviewer's probe: `dull` wiped out the fricative band, so its sibilance texture reads None."""
    return {"dull": {GAP: 0.0, SIB: None}, "fair": {GAP: 1.0, SIB: 0.5}, "other": {GAP: 2.0, SIB: 0.6}}


def test_a_candidate_cannot_escape_a_family_by_leaving_its_reading_unread():
    """The unread texture loses: `dull` is rated on both families and its worst one decides."""
    rewards = rw.group_rewards(_dulled_group(), _grid(sibilance=SIB, noise=GAP))
    assert rewards["dull"].win_rates == {"sibilance": 0.0, "noise": 1.0}
    assert _summary(rewards["dull"]) == (0.0, "ok", True)
    assert rw.ranked(rewards)[0].candidate == "fair"


def test_a_missing_reading_counts_as_the_worst_read_plus_one_scale_unit():
    """Only a reading somebody in the group has is filled; one unit of scale is one weight of distance."""
    grid = {SIB: {"target": 0.0, "family": "sibilance", "weight": 2.0, "scale": 4.0}, AIR: {"target": 0.0, "family": "timbre"}}
    dists = {name: rw.distances(flat, grid) for name, flat in _dulled_group().items()}
    filled = rw.fill_missing(dists, grid)
    assert filled["dull"][SIB] == pytest.approx(0.6 * 2.0 / 4.0 + 2.0)
    assert [filled[name][AIR] for name in filled] == [None] * 3


def test_each_side_of_two_readings_unread_loses_the_family_it_lacks():
    """`a` has only timbre, `b` only noise: each loses the family it left unread."""
    rewards = rw.group_rewards({"a": {AIR: 0.0}, "b": {GAP: 0.0}}, _two_families())
    assert rewards["a"].win_rates == {"timbre": 1.0, "noise": 0.0}
    assert (rewards["a"].reward, rewards["b"].reward) == (0.0, 0.0)


def test_points_need_a_reading_read_on_both_sides():
    """Only readings both candidates have are compared, inside the given tie band."""
    assert rw.pair_points({AIR: 0.1}, {AIR: None}, [AIR]) is None
    assert rw.pair_points({AIR: 0.1, GAP: 0.5}, {AIR: 0.4, GAP: None}, [AIR, GAP]) == 1.0
    assert rw.pair_points({AIR: 0.1}, {AIR: 0.4}, [AIR], {AIR: 0.5}) == 0.5


def _parked_grid():
    """`a` is read near its target; `b` and `c` sit 4 units inside a 5-unit dead zone."""
    parked = {"target": 0.0, "family": "timbre", "dead_zone": 5.0}
    return {"a": {"target": 0.0, "family": "timbre"}, "b": parked, "c": parked}


def test_readings_parked_inside_their_dead_zones_do_not_widen_the_tie_band():
    """A difference of eight floors on `a` decides; `b` and `c` cannot move under their floors and add no band."""
    group = {"x": {"a": 0.0, "b": 1.0, "c": 1.0}, "y": {"a": 0.8, "b": 1.0, "c": 1.0}}
    floors = {"a": 0.1, "b": 0.4, "c": 0.4}
    rewards = rw.group_rewards(group, _parked_grid(), floors=floors)
    assert (rewards["x"].reward, rewards["y"].reward) == (1.0, 0.0)
    assert rw.tolerance(group["x"], _parked_grid(), floors) == pytest.approx({"a": 0.1, "b": 0.0, "c": 0.0})


def test_a_readings_share_of_the_tie_band_is_what_its_floor_can_move():
    """Full outside the dead zone, partial within a floor of its edge, none deeper in or when missing."""
    spec = rw.ReadingSpec(target=0.0, family="timbre", dead_zone=1.0, scale=2.0, weight=4.0)
    assert spec.noise(2.0, 0.4) == pytest.approx(0.8)
    assert spec.noise(-0.8, 0.4) == pytest.approx(0.4)
    assert (spec.noise(0.5, 0.4), spec.noise(None, 0.4)) == (0.0, 0.0)


def test_unweighted_distances_feed_a_preference_head():
    """Distances with and without the grid weight; a missing reading is None."""
    grid = {AIR: {"target": 1.0, "family": "timbre", "weight": 4.0, "scale": 2.0}}
    assert rw.distances({AIR: 3.0}, grid) == {AIR: 4.0}
    assert rw.distances({AIR: 3.0}, grid, weighted=False) == {AIR: 1.0}
    assert rw.distances({}, grid) == {AIR: None}


def test_gate_constraints_mirror_the_incumbents_gate_counts():
    """More hard failures excludes, more flags caps, as `_beats` refuses them today."""
    incumbent = {rw.HARD_KEY: 0.0, rw.FLAG_KEY: 1.0}
    worse = rw.gate_constraints({rw.HARD_KEY: 1.0, rw.FLAG_KEY: 1.0}, incumbent, inaudible=True)
    assert (worse.hard_gate, worse.listener_flag, worse.inaudible) == (True, False, True)
    assert rw.gate_constraints({rw.FLAG_KEY: 2.0}, incumbent).status() == "capped"
    assert rw.gate_constraints({rw.HARD_KEY: math.nan}, None).status() == "ok"
