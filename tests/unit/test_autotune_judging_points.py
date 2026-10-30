"""The judging's points are exact: `status` and `improved_readings` pick a tie and a full point out with `math.isclose`,
which equals an equality test only because `points_behind` returns nothing but 0, 0.5 and 1; and a floor is 0 or positive.
"""

from itertools import product

from scripts import autotune_judging as judging

TILT = "dsp.balance_tilt_db_oct.output.median"
DISTANCES = (None, 0.0, 0.3, 1.0, 1.0 + 1e-12, 2.5)
BANDS = (0.0, 0.5, 2.0)


def _pair(mine, theirs):
    """A two-candidate entry on one reading with no tie band: `a` at `mine`, the incumbent at `theirs`."""
    return judging.Entry(TILT, 1.0, {"a": mine, "inc": theirs}, {"a": 0.0, "inc": 0.0})


def test_points_behind_takes_only_the_three_exact_values():
    seen = {judging.points_behind(mine, theirs, band) for mine, theirs, band in product(DISTANCES, DISTANCES, BANDS)}
    assert seen == {0.0, 0.5, 1.0}


def test_a_tiny_move_past_a_zero_band_is_moved_not_banded():
    assert _pair(1.0, 1.0 + 1e-12).status("a", "inc") == judging.MOVED


def test_a_move_inside_the_band_is_banded():
    item = judging.Entry(TILT, 1.0, {"a": 1.0, "inc": 1.2}, {"a": 0.5, "inc": 0.0})
    assert item.status("a", "inc") == judging.BANDED


def test_a_tiny_move_past_a_zero_band_the_better_way_counts_as_improved():
    assert judging.improved_readings([_pair(1.0, 1.0 + 1e-12)], ["a", "inc"], "inc") == {"a": [TILT], "inc": []}


def test_a_tie_does_not_count_as_improved():
    assert judging.improved_readings([_pair(None, None)], ["a"], "inc") == {"a": []}


def test_a_tiny_positive_floor_is_a_floor():
    assert judging.Floors({TILT: 1e-300}, {}).source(TILT) == "benign pair"


def test_a_zero_floor_is_none():
    assert judging.Floors({TILT: 0.0}, {TILT: -1.0}).source(TILT) == "none"
