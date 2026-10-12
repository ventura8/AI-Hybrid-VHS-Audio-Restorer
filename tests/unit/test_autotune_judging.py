"""The loop's judging resolution: the tie band, the dead zones' floor, the hiss exemption, the round's label and distances."""

import numpy as np
import pytest

from scripts import autotune_judging as judging
from scripts import tune_restoration as tune

TILT = "dsp.balance_tilt_db_oct.output.median"
RESIDUAL = "dsp.residual_noise_db.output.median"
GAP_LSD = "dsp.gap_lsd_db.output.median"
CER = "speech.cer.output.median"
SIGMOS = "mos.sigmos_col.delta.median"
SPEC = {"target": 0.0, "dead_zone": 1.0, "scale": 0.1, "weight": 1.0, "family": "timbre"}
# The Session 0 benign-pair floors and the scorer's jitter as the two reports hold them (rounded).
BENIGN = {TILT: 0.85, RESIDUAL: 16.44, GAP_LSD: 1.01, CER: 0.038}
JITTER = {TILT: 0.0003, RESIDUAL: 0.0018, GAP_LSD: 0.048, CER: 0.0008}
FLOORS = judging.Floors(BENIGN, JITTER)


# ----------------------------------------------------------------------------- the hiss exemption


@pytest.mark.parametrize(
    "key",
    [
        "dsp.hiss_speech_db.output.median",
        RESIDUAL,
        GAP_LSD,
        "dsp.gap_hf_excess_db.output.median",
        "dsp.pause_depth_db.delta.median",
        "dsp.hf_8k16k.delta.median",
        "mos.sigmos_noise.delta.median",
        "dsp.pause_tilt_db.output.median",
        "file.noise_removed_db.output.median",
        "mos.dnsmos_bak.delta.median",
    ],
)
def test_every_reading_of_hiss_or_residual_noise_is_exempt(key):
    """The user's list (2026-10-11), every dsp.hiss_* reading included, and the other residual-noise readings."""
    assert judging.is_hiss_reading(key)


@pytest.mark.parametrize("key", [TILT, CER, SIGMOS, "dsp.sib_abs_level_db.output.median", "file.gain_ride_lu.output.median"])
def test_timbre_sibilance_level_and_learned_readings_are_not(key):
    assert not judging.is_hiss_reading(key)


def test_a_hiss_reading_never_takes_its_floor_from_the_benign_pairs():
    """The residual-noise floor of the Session 0 pairs is 16.4 dB; the ear keeps the scorer's 0.0018 dB instead."""
    assert FLOORS.floor(RESIDUAL) == pytest.approx(0.0018)
    assert FLOORS.source(RESIDUAL) == "jitter"
    assert FLOORS.floor(GAP_LSD) == pytest.approx(0.048)


def test_a_hiss_reading_without_a_jitter_floor_has_none_at_all():
    """No scorer jitter measured: the hiss reading gets no floor rather than the benign pairs' 16.4 dB."""
    assert judging.Floors(BENIGN, {}).floor(RESIDUAL) == 0.0
    assert judging.Floors(BENIGN, {}).source(RESIDUAL) == "none"


def test_another_reading_takes_the_larger_floor():
    assert FLOORS.floor(TILT) == pytest.approx(0.85)
    assert FLOORS.source(TILT) == "benign pair"
    assert judging.Floors({}, {TILT: 0.2}).source(TILT) == "jitter"


def test_an_unmeasured_or_unusable_floor_reads_zero():
    assert FLOORS.floor("dsp.unmeasured.output.median") == 0.0
    assert judging.Floors({TILT: float("nan")}, {TILT: -1.0}).floor(TILT) == 0.0


def test_the_vetoes_keep_the_benign_pairs_except_on_a_hiss_reading():
    floors = judging.veto_floors(
        judging.Floors({**BENIGN, "mos.sigmos_noise.delta.median": 0.116}, {**JITTER, "mos.sigmos_noise.delta.median": 0.114})
    )
    assert floors[CER] == pytest.approx(0.038)
    assert floors["mos.sigmos_noise.delta.median"] == pytest.approx(0.114)
    assert floors[RESIDUAL] == pytest.approx(0.0018)


@pytest.mark.parametrize("report", [{"operating_points": ["benign_pair"]}, {"transforms": ["reference", "benign_pair"]}])
def test_a_benign_pair_report_is_refused_as_the_jitter_floors(report):
    with pytest.raises(SystemExit, match="benign-pair report"):
        judging.jitter_floors({**report, "readings": {RESIDUAL: {"floor": 16.4}}}, "noise_floor_benign.json")


def test_a_repeat_and_near_copy_report_gives_the_jitter_floors():
    report = {
        "operating_points": ["output"],
        "transforms": ["reference", "shift_1", "requantise_16"],
        "readings": {RESIDUAL: {"floor": 0.0018}},
    }
    assert judging.jitter_floors(report) == {RESIDUAL: pytest.approx(0.0018)}


def test_the_console_says_where_the_floors_come_from():
    assert judging.describe_floors(judging.NO_FLOORS).startswith("none")
    assert "hiss readings: jitter only" in judging.describe_floors(FLOORS)


# ----------------------------------------------------------------------------- the banded ranking


def _rank_variants(flats, ranking):
    """The ranking before 2026-10-11: tune_restoration's exact average ranks."""
    summaries = {cid: {"metrics": {key: {"median": value} for key, value in flat.items()}} for cid, flat in flats.items()}
    return tune.rank_variants(summaries, ranking)


def _random_flats(rng, keys, size):
    flats = {}
    for index in range(size):
        values = rng.choice([0.0, 0.5, 1.5, 2.0, 3.0, None], size=len(keys))
        flats[f"c{index}"] = {key: value for key, value in zip(keys, values) if value is not None}
    return flats


@pytest.mark.parametrize("seed", range(6))
def test_without_floors_the_ranks_are_exactly_the_old_average_ranks(seed):
    """No floors file: equal distances tie and nothing else, unread ranks last, direction entries as before."""
    ranking = {"a": SPEC, "b": {**SPEC, "weight": 2.0}, "c": "up", "d": "down", "e": SPEC}
    flats = _random_flats(np.random.default_rng(seed), list(ranking)[:4], 9)
    entries = judging.tape_entries(flats, ranking)
    assert judging.mean_ranks(entries, list(flats)) == _rank_variants(flats, ranking)
    assert "e" not in [item.key for item in entries]


def test_nothing_ranked_gives_no_rank():
    assert judging.mean_ranks([], ["a", "b"]) == {"a": None, "b": None}


def test_a_difference_under_the_floor_ties_and_one_past_it_orders():
    """Tilt 1.30 / 1.40 / 2.50 dB/oct against a 1.0 zone and the 0.85 floor: the first two tie, the third is behind."""
    item = judging.entry(TILT, SPEC, {"inc": 1.30, "near": 1.40, "far": 2.50}, FLOORS)
    assert judging.entry_ranks(item) == {"inc": 1.5, "near": 1.5, "far": 3.0}
    assert item.status("near", "inc") == judging.BANDED
    assert item.status("far", "inc") == judging.MOVED
    assert item.status("inc", "inc") == judging.UNMOVED


def test_every_dead_zone_is_bounded_from_below_by_the_floor():
    """A 0.0 zone reads 0.5 as a distance; floored at the 0.85 benign-pair floor it reads nothing."""
    narrow = {**SPEC, "dead_zone": 0.0}
    assert judging.entry(TILT, narrow, {"x": 0.5}).distances["x"] == pytest.approx(5.0)
    assert judging.entry(TILT, narrow, {"x": 0.5}, FLOORS).distances["x"] == 0.0


def test_the_hiss_exemption_keeps_a_one_db_residual_noise_move_ranked():
    """A 1 dB move ties on a reading with a 16 dB benign floor, but residual noise is a hiss reading: it stays ordered."""
    values = {"inc": 1.0, "hissier": 2.0}
    hiss = judging.entry(RESIDUAL, {**SPEC, "dead_zone": 0.0, "family": "pauses"}, values, FLOORS)
    benign_only = judging.Floors({"dsp.other.output.median": 16.44}, {})
    other = judging.entry("dsp.other.output.median", {**SPEC, "dead_zone": 0.0}, values, benign_only)
    assert judging.entry_ranks(hiss) == {"inc": 1.0, "hissier": 2.0}
    assert judging.entry_ranks(other) == {"inc": 1.5, "hissier": 1.5}


def test_a_candidate_level_with_the_incumbent_never_wins_a_tape_through_a_third_one():
    """Distances 1.0 / 0.92 / 1.05, band 0.1: `a` ties the incumbent, out-ranks it only by beating `c`; not a win."""
    floors = judging.Floors({TILT: 0.01}, {})
    spec = {**SPEC, "dead_zone": 0.0, "scale": 0.1}
    flats = {"inc": {TILT: 0.100}, "a": {TILT: 0.092}, "c": {TILT: 0.105}}
    table = judging.tape_table(flats, {TILT: spec}, "inc", floors)
    assert table.ranks["a"] < table.ranks["inc"]
    assert table.level("a")
    assert judging.tape_wins({"t1": table}, "a", "inc") == []
    plain = judging.tape_table(flats, {TILT: spec}, "inc")
    assert judging.tape_wins({"t1": plain}, "a", "inc") == ["t1"]


BODY = "dsp.balance_body_db.output.median"


def _pack_flats():
    """`worse`: tilt 0.7 dB/oct nearer (under the 0.85 floor), body 1.5 dB further (past the 1.2 floor); ten candidates
    sit inside the incumbent's band on both readings and outside `worse`'s on tilt."""
    flats = {"inc": {TILT: 2.0, BODY: 2.0}, "worse": {TILT: 1.3, BODY: 3.5}}
    flats.update({f"pack{index}": {TILT: 2.5, BODY: 2.75} for index in range(10)})
    return flats


def _pack_table():
    """The tape table of `_pack_flats` under the 0.85 tilt and 1.2 body floors, with no dead zone."""
    floors = judging.Floors({TILT: 0.85, BODY: 1.2}, {})
    spec = {**SPEC, "dead_zone": 0.0}
    return judging.tape_table(_pack_flats(), {TILT: spec, BODY: spec}, "inc", floors)


def test_a_candidate_whose_only_move_past_the_band_is_worse_out_ranks_the_incumbent_through_a_pack():
    """The pack lifts `worse` above the incumbent on exact ranks, on a sub-floor tilt gain and a real body loss."""
    table = _pack_table()
    assert table.ranks["worse"] < table.ranks["inc"]
    assert table.status["worse"] == {TILT: judging.BANDED, BODY: judging.MOVED}


def test_a_candidate_whose_only_move_past_the_band_is_worse_never_wins_a_tape_through_a_pack():
    """Out-ranking the incumbent through third candidates on a sub-floor gain does not make a real loss a win."""
    table = _pack_table()
    assert not table.level("worse")
    assert not table.better("worse")
    assert judging.tape_wins({"t1": table}, "worse", "inc") == []


@pytest.mark.parametrize("seed", range(6))
def test_without_floors_a_tape_win_is_exactly_the_old_better_rank(seed):
    """Out-ranking the incumbent on exact ranks needs some reading strictly better, so the better-move rule adds nothing."""
    ranking = {"a": SPEC, "b": {**SPEC, "weight": 2.0}, "c": "up", "d": "down"}
    flats = _random_flats(np.random.default_rng(100 + seed), list(ranking), 9)
    old = _rank_variants(flats, ranking)
    table = judging.tape_table(flats, ranking, "c0")
    for cid in flats:
        ahead = old[cid] is not None and old["c0"] is not None and old[cid] < old["c0"]
        assert judging.tape_wins({"t1": table}, cid, "c0") == (["t1"] if ahead else [])


def test_a_reading_nobody_read_on_a_tape_never_wins_it():
    table = judging.tape_table({"inc": {}, "a": {}}, {TILT: SPEC}, "inc", FLOORS)
    assert table.ranks == {"inc": None, "a": None}
    assert judging.tape_wins({"t1": table}, "a", "inc") == []


def test_unread_readings_rank_last_and_tie_with_each_other():
    assert judging.points_behind(None, None, 0.0) == 0.5
    assert judging.points_behind(None, 1.0, 0.0) == 1.0
    assert judging.points_behind(1.0, None, 0.0) == 0.0


def test_the_worst_status_on_any_tape_and_the_tapes_that_moved_are_kept_per_reading():
    tables = {
        "t1": judging.TapeTable({}, {}, {"a": {TILT: judging.UNMOVED, GAP_LSD: judging.MOVED}}),
        "t2": judging.TapeTable({}, {}, {"a": {TILT: judging.BANDED, GAP_LSD: judging.UNMOVED}}),
    }
    assert judging.reading_status(tables, "a") == {TILT: judging.BANDED, GAP_LSD: judging.MOVED}
    assert judging.moved_tapes(tables, "a") == {GAP_LSD: ["t1"]}


# ----------------------------------------------------------------------------- the round's label and distances


def _scored(level=True, status=None, **extra):
    return {"failures": 0.0, "flags": 0.0, "level": level, "status": status or {TILT: judging.UNMOVED, GAP_LSD: judging.BANDED}, **extra}


def test_a_round_whose_every_candidate_stayed_level_says_every_move_was_below_the_floor():
    verdicts = {"inc": _scored(), "a": _scored(), "b": _scored(status={TILT: judging.UNMOVED, GAP_LSD: judging.UNMOVED})}
    label = judging.plateau_label(verdicts, "inc", FLOORS)
    assert label == f"{judging.BELOW_FLOOR} (1 readings unmoved; 1 moved less than their floor)"
    assert judging.plateau_label(verdicts, "inc").startswith(judging.INSIDE_ZONES)


def test_inert_and_tied_candidates_leave_the_below_floor_label_standing():
    verdicts = {"inc": _scored(), "a": _scored(), "x": {"failures": 0.0, "flags": 0.0, "inert": "inc"}, "y": {"tie": True}}
    assert judging.plateau_label(verdicts, "inc", FLOORS).startswith(judging.BELOW_FLOOR)


@pytest.mark.parametrize(
    ("extra", "counts"),
    [
        ({"a": _scored(level=False)}, "0 of 1 scored candidates level with the incumbent, 1 moved and lost, 0 held back"),
        ({"a": _scored(vetoed=["why"])}, "1 of 1 scored candidates level with the incumbent, 0 moved and lost, 1 held back"),
        ({"a": _scored(), "r": {"refused": ["why"]}}, "1 of 1 scored candidates level with the incumbent, 0 moved and lost, 1 held back"),
        (
            {"a": {**_scored(level=False), "failures": 2.0}},
            "0 of 1 scored candidates level with the incumbent, 0 moved and lost, 1 held back",
        ),
    ],
)
def test_a_real_move_or_a_candidate_held_back_keeps_the_plateau(extra, counts):
    """A move past its band that lost, a veto, a reversal or more hard failures: the round is a plateau, with the counts."""
    assert judging.plateau_label({"inc": _scored(), **extra}, "inc", FLOORS) == f"{judging.PLATEAU} ({counts})"


def test_a_round_with_nothing_scored_is_a_plain_plateau():
    assert judging.plateau_label({"inc": _scored()}, "inc", FLOORS) == judging.PLATEAU


ORDERED = [
    ("inc", {"distances": {"t1": {TILT: 0.0, GAP_LSD: None}, "t2": {TILT: 0.0, GAP_LSD: 1.25}}}),
    ("a", {"distances": {"t1": {TILT: 3.5, GAP_LSD: None}, "t2": {TILT: 0.0, GAP_LSD: 1.25}}, "moved": {TILT: ["t1"]}}),
    ("x", {"inert": "inc"}),
]


def test_the_distance_table_shows_each_tapes_distance_the_floor_and_what_moved():
    lines = judging.distance_lines(ORDERED, lambda cid: f"move of {cid}", FLOORS)
    assert f"| candidate | change | {TILT} | {GAP_LSD} |" in lines
    assert "| floor | | 0.85 (benign pair) | 0.048 (jitter) |" in lines
    assert "| a | move of a | 3.5*/0 | -/1.25 |" in lines


def test_a_candidate_set_aside_has_no_row_and_a_round_of_them_no_table():
    lines = judging.distance_lines(ORDERED, str, FLOORS)
    assert [line for line in lines if line.startswith("| x ")] == []
    assert judging.distance_lines([("x", {"inert": "inc"})], str) == []
