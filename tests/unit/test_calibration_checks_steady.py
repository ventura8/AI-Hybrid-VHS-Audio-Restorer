"""Ear v4's two check directions: `steady` (the level medians span little) and `below` (every level under the centre)."""

from scripts import calibration_checks as checks
from scripts import quality_degradations as deg
from tests.unit.test_calibration_checks import _Case

PAUSE = "dsp.hiss_pause_margin_db"
FLOOR = {PAUSE: {"centre": 0.0, "floor": 0.1}}


def _check(name, values, expectation, floor=None):
    """The check of `expectation` on one degradation whose levels read `values` (None leaves a level unread)."""
    cases = [_Case("degradation", name, level) for level in deg.DEGRADATIONS[name].levels]
    readings = {case.case_id: ({} if value is None else {expectation.metric: value}) for case, value in zip(cases, values)}
    return checks.check_expectation(name, expectation, cases, readings, FLOOR if floor is None else floor)


def test_steady_passes_within_its_tolerance_wherever_the_levels_sit():
    """A span of 0.4 under a 0.5 tolerance passes though every level sits 30 dB under the source; 0.6 fails."""
    steady = deg.Expectation(PAUSE, "steady", tolerance=0.5)
    assert _check("low_floor_kept", [-30.0, -29.8, -29.6], steady)["status"] == "pass"
    assert _check("low_floor_kept", [-30.0, -29.7, -29.4], steady)["status"] == "fail"


def test_steady_takes_three_floors_when_they_are_wider_than_the_tolerance():
    """With a 0.3 floor the allowance is 0.9."""
    result = _check(
        "low_floor_kept", [-30.0, -29.5, -29.2], deg.Expectation(PAUSE, "steady", tolerance=0.5), {PAUSE: {"centre": 0.0, "floor": 0.3}}
    )
    assert (result["status"], round(result["allowed"], 6), round(result["span"], 6)) == ("pass", 0.9, 0.8)


def test_below_needs_every_level_under_the_centre_by_the_allowance():
    """Under the centre less three floors (0.3) at every level passes; one level at -0.2 fails."""
    below = deg.Expectation(PAUSE, "below")
    assert _check("low_floor_kept", [-30.0, -20.0, -0.4], below)["status"] == "pass"
    assert _check("low_floor_kept", [-30.0, -20.0, -0.2], below)["status"] == "fail"


def test_a_missing_level_leaves_both_unscored():
    """No reading at one level asserts nothing, with the reason."""
    for direction in ("steady", "below"):
        result = _check("low_floor_kept", [-30.0, None, -29.0], deg.Expectation(PAUSE, direction, tolerance=0.5))
        assert (result["status"], result["reason"]) == ("unscored", checks.NOT_READ)


def test_the_hiss_degradations_declare_both_new_directions():
    """`low_floor_kept` asserts the pause margin steady and below; every direction the table uses has a check."""
    directions = {(e.metric, e.direction) for e in deg.DEGRADATIONS["low_floor_kept"].expects}
    assert {(PAUSE, "steady"), (PAUSE, "below")} <= directions
    assert {e.direction for spec in deg.DEGRADATIONS.values() for e in spec.expects} <= set(checks.CHECKS)


def test_the_linear_benign_cases_floor_only_the_hiss_readings():
    """A `*_linear` case adds to the hiss readings' floor only; every other case adds to everything but them."""
    cases = [_Case("benign", "identity", None), _Case("benign", "identity_linear", None)]
    readings = {cases[0].case_id: {"dsp.lkr": 0.0, PAUSE: 5.0}, cases[1].case_id: {"dsp.lkr": 9.0, PAUSE: 0.2}}
    floor = checks.noise_floor(cases, readings)
    assert (floor["dsp.lkr"]["centre"], floor[PAUSE]["centre"]) == (0.0, 0.2)
