"""The rules the loop's ear v3 guards apply (`reward.py`): learned-judge vetoes in benign floors, ledger boundaries."""

import math

import pytest

from scripts.restoration_quality import reward as rw

CER = "speech.cer.output.median"
UTMOS = "speech.utmos.delta.median"


def test_a_veto_is_floor_multiple_floors_of_a_move_the_worse_way():
    """CER is worse up, UTMOS worse down; the limit is floor_multiple x the reading's floor (3 by default)."""
    vetoes = rw.parse_vetoes({CER: {"worse": "up", "floor_multiple": 2}, UTMOS: {"worse": "down"}}, {CER: 0.01, UTMOS: 0.05})
    assert vetoes == {CER: rw.Veto("up", 0.02), UTMOS: rw.Veto("down", pytest.approx(0.15))}
    incumbent = {CER: 0.010, UTMOS: 0.0}
    assert rw.vetoed_readings({CER: 0.045, UTMOS: 0.1}, incumbent, vetoes) == [CER]
    assert rw.vetoed_readings({CER: 0.025, UTMOS: -0.2}, incumbent, vetoes) == [UTMOS]
    assert rw.vetoed_readings({CER: 0.0, UTMOS: 0.5}, incumbent, vetoes) == []


def test_a_lost_reading_vetoes_and_one_the_incumbent_lacks_vetoes_nothing():
    """A learned family that failed for the candidate alone cannot let it win; nothing to compare with is no veto."""
    veto = rw.Veto("up", 0.1)
    assert veto.trips(None, 0.2) and veto.trips(math.nan, 0.2)
    assert not veto.trips(0.9, None)
    assert not veto.trips(0.3, 0.2)
    assert rw.vetoed_readings(None, None, {CER: veto}) == []


@pytest.mark.parametrize(
    ("section", "floors", "message"),
    [
        ({CER: {"worse": "sideways"}}, {CER: 0.01}, "worse must be up or down"),
        ({CER: "up"}, {CER: 0.01}, "worse must be up or down"),
        ({CER: {"worse": "up", "floor_multiple": 0}}, {CER: 0.01}, "floor_multiple"),
        ({CER: {"worse": "up"}}, {}, "no benign floor in --noise-floors"),
        ({CER: {"worse": "up"}}, {CER: math.nan}, "no benign floor in --noise-floors"),
        ({CER: {"worse": "up"}}, {CER: "0.01"}, "no benign floor in --noise-floors"),
        ({CER: {"worse": "up"}}, {CER: 0.0}, "no benign floor above 0"),
        ({CER: {"worse": "up"}}, {CER: -0.01}, "no benign floor above 0"),
        ([CER], {CER: 0.01}, "vetoes must map"),
    ],
)
def test_a_veto_that_cannot_be_measured_is_refused_by_name(section, floors, message):
    """A bad direction, a non-positive multiple, a reading the report lacks, a floor not above 0 and a list each raise."""
    with pytest.raises(ValueError, match=message):
        rw.parse_vetoes(section, floors)


def _refusal(floors):
    with pytest.raises(ValueError) as refused:
        rw.parse_vetoes({CER: {"worse": "up", "floor_multiple": 3}}, floors)
    return str(refused.value)


def test_a_zero_floor_names_the_reading_and_how_to_measure_its_jitter():
    """CER read at --repeats 1 can have a p95 deviation of exactly 0; a 0 limit would veto Whisper's own jitter."""
    refused = _refusal({CER: 0.0})
    assert refused.startswith(f"veto {CER}: no benign floor above 0")
    assert "--repeats 2 or more" in refused and "resample_roundtrip" in refused
    assert "python -m scripts.reward_noise_floor" in refused and "--families" not in refused


def test_a_missing_floor_names_the_family_flag_not_the_repeats():
    """The default report reads dsp only, so a learned reading is absent: more repeats would not bring it back."""
    refused = _refusal({"dsp.lkr.output.median": 0.01})
    assert refused.startswith(f"veto {CER}: no benign floor in --noise-floors")
    assert "--families" in refused and "the default is dsp only" in refused
    assert "python -m scripts.reward_noise_floor" in refused and "--repeats" not in refused


def test_no_vetoes_and_no_reversals_parse_to_nothing():
    """A v1 or v2 grid has neither section."""
    assert (rw.parse_vetoes(None, {}), rw.parse_reversals(None)) == ({}, ())


AIR_GAIN = "linear_air_gain_db"


def test_a_reversal_entry_parses_to_its_boundary_and_says_why_it_refuses():
    """Round 3: the user rejected +2 dB of air for +1 dB."""
    (rule,) = rw.parse_reversals([{"key": AIR_GAIN, "rejected_above": 2.0, "verdict": "r3-air-preference"}])
    assert rule == rw.Reversal(AIR_GAIN, 2.0, 1.0, "r3-air-preference")
    assert rule.describe(2.0) == "linear_air_gain_db 2.0 is at or above 2, rejected by ear (r3-air-preference)"


AIR_RULE = rw.Reversal(AIR_GAIN, 2.0, 1.0, "r3-air-preference")


@pytest.mark.parametrize(
    ("value", "current", "refused"),
    [
        (2.0, 1.0, True),
        (3.0, 2.0, True),
        (2.0, None, True),
        (1.5, 1.0, False),
        (1.0, 2.0, False),
        (2.0, 2.0, False),
        (None, 1.0, False),
    ],
)
def test_a_reversal_refuses_a_move_that_heads_past_the_rejected_value(value, current, refused):
    """+1 -> +2 crosses the boundary, +2 -> +3 goes deeper, from the app's default to +2 is refused; +1.5, a move back,
    staying put and the app's default are not."""
    assert AIR_RULE.refuses(value, current) is refused


def test_a_lower_boundary_refuses_moves_down_to_it():
    """rejected_below mirrors rejected_above; an entry without a verdict names the ledger."""
    (rule,) = rw.parse_reversals([{"key": "pause_floor_fill_db", "rejected_below": 8.0}])
    assert rule.refuses(8.0, 12.0) and not rule.refuses(18.0, 12.0)
    assert rule.describe(8.0).endswith("at or below 8, rejected by ear (ledger)")
    assert rw.reversals_refusing({"pause_floor_fill_db": 8.0}, {"pause_floor_fill_db": 12.0}, [rule]) == [rule]
    assert rw.reversals_refusing({}, {"pause_floor_fill_db": 12.0}, [rule]) == []


@pytest.mark.parametrize(
    "entry",
    [
        {"rejected_above": 2.0},
        {"key": AIR_GAIN},
        {"key": AIR_GAIN, "rejected_above": 2.0, "rejected_below": 0.0},
        {"key": AIR_GAIN, "rejected_above": "two"},
        "linear_air_gain_db > 2",
    ],
)
def test_a_reversal_without_a_key_and_one_number_is_refused(entry):
    """No key, no side, both sides, a side that is not a number, or not a mapping at all."""
    with pytest.raises(ValueError, match="reversal 0"):
        rw.parse_reversals([entry])


def test_reversals_must_be_a_list():
    """A mapping of keys to bounds is not the list of boundaries the section holds."""
    with pytest.raises(ValueError, match="reversals must be a list"):
        rw.parse_reversals({AIR_GAIN: 2.0})
