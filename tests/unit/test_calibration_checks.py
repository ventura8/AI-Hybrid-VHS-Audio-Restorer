"""The calibration's sensitivity checks: the benign floor, and the up / down / flat / match rules on tiny synthetic scores."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts import calibration_checks as checks
from scripts import calibration_gates as cg
from scripts import quality_degradations as deg
from scripts.restoration_quality.gates import GATES


@dataclass
class _Case:
    kind: str
    name: str
    level: object
    language: str = "en"
    source: Path = Path("s.wav")
    output: Path = Path("o.wav")

    @property
    def case_id(self):
        """The case's key, as the calibration builds it."""
        return f"{self.language}/{self.kind}_{self.name}_{self.level}"


def _benign():
    """One case per benign name; the hf_4k8k deltas sit on a symmetric 0.1 grid, so the centre is 0 and the floor its edge."""
    cases = [_Case("benign", name, None) for name in deg.BENIGN]
    half = (len(cases) - 1) / 2.0
    readings = {
        c.case_id: {"dsp.hf_4k8k": 0.1 * (i - half), "speech.cer": 0.02, "meta.prog_bandwidth_hz": 4000.0} for i, c in enumerate(cases)
    }
    return cases, readings, 0.1 * half


def test_noise_floor_centres_and_spreads_each_metric():
    """Noise floor centres and spreads each metric."""
    cases, readings, edge = _benign()
    floor = checks.noise_floor(cases, readings)
    assert abs(floor["dsp.hf_4k8k"]["centre"]) < 1e-9
    assert abs(floor["dsp.hf_4k8k"]["floor"] - edge) < 1e-9
    assert floor["speech.cer"]["floor"] == 0.0


def test_the_floor_skips_profile_and_base_readings():
    """The floor skips profile and base readings."""
    cases, readings, _edge = _benign()
    readings[cases[0].case_id]["base.prog_bandwidth_hz"] = 9000.0
    floor = checks.noise_floor(cases, readings)
    assert "meta.prog_bandwidth_hz" not in floor
    assert "base.prog_bandwidth_hz" not in floor


def _cases(name, values, metric="dsp.hf_4k8k", language="en"):
    cases = [_Case("degradation", name, level, language) for level in deg.DEGRADATIONS[name].levels]
    return cases, {c.case_id: {metric: v} for c, v in zip(cases, values)}


FLOOR = {"dsp.hf_4k8k": {"centre": 0.0, "floor": 0.5}}


def test_a_clean_monotonic_response_passes():
    """A clean monotonic response passes."""
    cases, readings = _cases("underwater", [-3.0, -8.0, -20.0])
    result = checks.check_expectation("underwater", deg.Expectation("dsp.hf_4k8k", "down"), cases, readings, FLOOR)
    assert result["status"] == "pass"
    assert result["monotonic_share"] == 1.0
    assert abs(result["effect_mild"] - 6.0) < 1e-6


def test_a_weak_mild_level_or_the_wrong_direction_fails():
    """A weak mild level or the wrong direction fails."""
    cases, readings = _cases("underwater", [-0.5, -8.0, -20.0])
    weak = checks.check_expectation("underwater", deg.Expectation("dsp.hf_4k8k", "down"), cases, readings, FLOOR)
    assert weak["status"] == "fail"
    wrong = checks.check_expectation("underwater", deg.Expectation("dsp.hf_4k8k", "up"), cases, readings, FLOOR)
    assert wrong["direction"] is False


def test_an_unread_metric_is_unscored():
    """An unread metric is unscored."""
    cases, readings = _cases("underwater", [-3.0, -8.0, -20.0])
    result = checks.check_expectation("underwater", deg.Expectation("mos.sigmos_col", "down"), cases, readings, {})
    assert result["status"] == "unscored"
    assert result["reason"] == checks.NOT_READ
    assert result["expect"] == "down"


def test_a_flat_reading_inside_its_tolerance_passes_and_one_outside_fails():
    """A flat reading inside its tolerance passes and one outside fails."""
    cases, readings = _cases("treble_dropouts", [0.0, 0.5, 1.0], "file.dropouts")
    inside = checks.check_expectation("treble_dropouts", deg.Expectation("file.dropouts", "flat", tolerance=1.0), cases, readings, {})
    assert inside["status"] == "pass"
    outside = checks.check_expectation("treble_dropouts", deg.Expectation("file.dropouts", "flat", tolerance=0.4), cases, readings, {})
    assert outside["status"] == "fail"
    assert outside["moves"] == [0.0, 0.5, 1.0]


def test_a_flat_reading_may_move_three_benign_floors():
    """A flat reading may move three benign floors."""
    cases, readings = _cases("treble_dropouts", [0.0, 1.0, 1.4], "file.dropouts")
    floor = {"file.dropouts": {"centre": 0.0, "floor": 0.5}}
    result = checks.check_expectation("treble_dropouts", deg.Expectation("file.dropouts", "flat", tolerance=0.1), cases, readings, floor)
    assert result["status"] == "pass"
    assert result["allowed"] == [1.5, 1.5, 1.5]


def _twinned(net, absolute):
    cases = [_Case("degradation", "air_shelf_boost", level) for level in deg.DEGRADATIONS["air_shelf_boost"].levels]
    readings = {c.case_id: {deg.SIB_NET: n, deg.SIB_ABS: a} for c, n, a in zip(cases, net, absolute)}
    return checks.check_expectation("air_shelf_boost", deg.Expectation(deg.SIB_NET, "flat", twin=deg.SIB_ABS), cases, readings, {})


def test_a_net_reading_must_stay_strictly_under_its_absolute_twin():
    """A net reading must stay strictly under its absolute twin."""
    assert _twinned([0.08, 0.16, 0.25], [0.10, 0.20, 0.40])["status"] == "pass"
    assert _twinned([0.08, 0.20, 0.25], [0.10, 0.20, 0.40])["status"] == "fail"


def test_a_twin_that_is_not_read_leaves_the_check_unscored():
    """A twin that is not read leaves the check unscored."""
    assert _twinned([0.08, 0.16, 0.25], [0.10, None, 0.40])["status"] == "unscored"


def _bandwidth(values, bases):
    cases = [_Case("degradation", "linear_bandwidth", level) for level in deg.DEGRADATIONS["linear_bandwidth"].levels]
    readings = {c.case_id: {"meta.prog_bandwidth_hz": v, "base.prog_bandwidth_hz": b} for c, v, b in zip(cases, values, bases)}
    expectation = deg.Expectation("meta.prog_bandwidth_hz", "match", tolerance=1.0 / 6.0)
    return checks.check_expectation("linear_bandwidth", expectation, cases, readings, {})


def test_a_source_reading_matches_the_cut_or_the_materials_own_lower_band():
    """A source reading matches the cut or the materials own lower band."""
    assert _bandwidth([10200.0, 7800.0, 5100.0], [16000.0] * 3)["status"] == "pass"
    assert _bandwidth([2830.0, 2830.0, 2830.0], [2830.0] * 3)["match_share"] == 1.0


def test_a_source_reading_off_the_condition_fails():
    """A source reading off the condition fails."""
    result = _bandwidth([10200.0, 12000.0, 9000.0], [16000.0] * 3)
    assert result["status"] == "fail"
    assert result["match_share"] == pytest.approx(1.0 / 3.0)


def test_a_missing_source_reading_does_not_match():
    """R0 ran and read no band on one case: that case is a miss, not a reason to skip the check."""
    assert _bandwidth([None, 0.0, 5000.0], [None, None, None])["match_share"] == pytest.approx(1.0 / 3.0)
    assert _bandwidth([None, 10200.0, 5000.0], [16000.0] * 3)["status"] == "fail"


def test_a_match_without_cases_is_unscored():
    """A match without cases is unscored."""
    expectation = deg.Expectation("meta.prog_bandwidth_hz", "match", tolerance=1.0 / 6.0)
    assert checks.check_expectation("linear_bandwidth", expectation, [], {}, {})["status"] == "unscored"


def test_a_match_no_case_carries_is_unscored_not_failed():
    """A run without the dsp family reads no `meta.*` at all: the check reads nothing, it does not fail."""
    cases, readings = _cases("linear_bandwidth", [0.01, 0.01, 0.01], "speech.cer")
    expectation = deg.Expectation("meta.prog_bandwidth_hz", "match", tolerance=1.0 / 6.0)
    result = checks.check_expectation("linear_bandwidth", expectation, cases, readings, {})
    assert (result["status"], result["reason"]) == ("unscored", checks.NOT_READ)


def _air_bed(band, air=None):
    """The music air shelf's cases on a bed R0 reads at `band` Hz; R1 air reads `air` (None: clipped away)."""
    cases, readings = _cases("air_shelf_boost_music", [air] * 3, "dsp.balance_air_db")
    for read in readings.values():
        read[checks.PROGRAMME_BAND] = band
    return cases, readings


def test_a_reading_r0s_band_clips_away_says_so():
    """The realistic-v2 music beds read 1414 Hz: R1's air (from 5 kHz) is clipped away, and the check says why."""
    cases, readings = _air_bed(1414.2)
    result = checks.check_expectation("air_shelf_boost_music", deg.Expectation("dsp.balance_air_db", "up"), cases, readings, {})
    assert (result["status"], result["reason"], result["programme_band_hz"], round(result["reading_from_hz"])) == (
        "unscored",
        checks.CLIPPED,
        1414.2,
        5292,
    )


@pytest.mark.parametrize(
    ("metric", "band"),
    [("dsp.balance_air_db", 5000.0), ("dsp.balance_air_db", 5100.0), ("dsp.balance_air_db", 5200.0), ("dsp.balance_presence_db", 2100.0)],
)
def test_a_band_past_the_readings_nominal_start_still_clips_it_when_the_layout_has_no_band_there(metric, band):
    """R1's layout, not the span's start, decides: a 5.0-5.29 kHz band leaves air no band (Tele7abc's R0 ends at 5.0 kHz)."""
    assert checks.why_unscored("air_shelf_boost_music", metric, *_air_bed(band))["reason"] == checks.CLIPPED


def test_the_tilt_needs_three_bands_of_the_layout():
    """1100 Hz carries one band from 1 kHz up: clipped; the music beds' 1414 Hz carries three, so an unread tilt there is data."""
    assert checks.why_unscored("air_shelf_boost_music", "dsp.balance_tilt_db_oct", *_air_bed(1100.0))["reason"] == checks.CLIPPED
    assert checks.why_unscored("air_shelf_boost_music", "dsp.balance_tilt_db_oct", *_air_bed(1414.2)) == {"reason": checks.NOT_READ}


@pytest.mark.parametrize(
    ("metric", "band"),
    [
        ("dsp.balance_air_db", 7000.0),
        ("dsp.balance_air_db", 5300.0),
        ("dsp.balance_air_db", None),
        ("dsp.balance_top_db", 1414.2),
        ("dsp.lkr", 1414.2),
    ],
)
def test_an_unread_reading_inside_the_band_is_not_called_clipped(metric, band):
    """A band wide enough for the reading, no band read at all, or a reading that is not R1's leaves the plain reason."""
    assert checks.why_unscored("air_shelf_boost_music", metric, *_air_bed(band)) == {"reason": checks.NOT_READ}


# R0's programme band per source of a pooled `--excerpts` run: a fixture and three tapes (2026-10-09).
POOLED_BANDS = {"en": 3000.0, "tele7abc": 4490.0, "soti": 5040.0, "vaccin": 7127.0}


def _pooled_air(bands):
    """The speech air shelf's cases on each source of `bands`, R1 air read on none of them."""
    cases = [
        _Case("degradation", "air_shelf_boost", level, language)
        for language in bands
        for level in deg.DEGRADATIONS["air_shelf_boost"].levels
    ]
    readings = {c.case_id: {"dsp.balance_air_db": None, checks.PROGRAMME_BAND: bands[c.language]} for c in cases}
    return checks.check_expectation("air_shelf_boost", deg.Expectation("dsp.balance_air_db", "up"), cases, readings, {})


def test_one_source_wide_enough_for_the_reading_makes_an_unread_reading_not_read():
    """The median band of the four (4765 Hz) called it clipped; Vaccin's 7127 Hz carries air, so its miss is data."""
    result = _pooled_air(POOLED_BANDS)
    assert (result["status"], result["reason"]) == ("unscored", checks.NOT_READ)
    assert "programme_band_hz" not in result


def test_clipped_names_the_widest_band_when_no_source_carries_the_reading():
    """Without Vaccin every band ends under 5292 Hz: clipped, quoting the widest (SOTI's 5040 Hz)."""
    result = _pooled_air({language: band for language, band in POOLED_BANDS.items() if language != "vaccin"})
    assert (result["reason"], result["programme_band_hz"]) == (checks.CLIPPED, 5040.0)


def test_a_scored_check_carries_no_reason():
    """Only an unscored check says why."""
    cases, readings = _air_bed(7000.0, 1.0)
    for case, value in zip(cases, (0.5, 1.0, 2.0)):
        readings[case.case_id]["dsp.balance_air_db"] = value
    result = checks.check_expectation("air_shelf_boost_music", deg.Expectation("dsp.balance_air_db", "up"), cases, readings, {})
    assert result["status"] == "pass"
    assert "reason" not in result


# The raw-pair sync offset (ms) per benign case on realistic-v2 en (review probe, `file_metrics.file_entries`).
SYNC_OFFSETS = {"identity": 0.009, "requantise": 0.009, "resample": 0.011, "shift_5ms": 4.99, "shift_30ms": 29.98, "shift_60ms": 59.97}


def _sync_floor():
    """The benign floor over every benign case: the offsets above, a drift of 0.02 ms on the shifts and 0.01 elsewhere."""
    cases = [_Case("benign", name, None) for name in deg.BENIGN]
    readings = {c.case_id: {"file.sync_offset_ms": SYNC_OFFSETS.get(c.name), "file.sync_drift_ms": _drift(c.name)} for c in cases}
    return checks.noise_floor(cases, readings)


def _drift(name):
    return 0.02 if name.startswith(deg.SHIFT_PREFIX) else 0.01


def test_the_shifts_raw_pair_lag_stays_out_of_the_benign_floor():
    """The shifts move the raw-pair sync by construction; their 5-60 ms must not relax the 40 ms sync gates."""
    floor = _sync_floor()
    assert floor["file.sync_offset_ms"]["floor"] < 0.01
    assert [cg.derive_gate(GATES[name], floor, {})["threshold"] for name in ("file.sync_offset", "file.sync_drift")] == [40.0, 40.0]


def test_a_shifts_drift_stays_in_the_benign_floor():
    """One constant lag is no drift: the shifts' drift readings stay in (their 0.02 widens the floor)."""
    assert _sync_floor()["file.sync_drift_ms"]["floor"] > 0.0


def _two_languages(name, metric, values):
    """`(cases, readings)` of one degradation on en and fr, `values` per language in level order."""
    cases, readings = [], {}
    for language, levels in values.items():
        more, read = _cases(name, levels, metric, language)
        cases, readings = cases + more, {**readings, **read}
    return cases, readings


HISS_FLOOR = {"dsp.gap_atten_db": {"centre": 0.0, "floor": 0.0079}, "dsp.gap_air_db": {"centre": 0.0, "floor": 0.0015}}


def _hiss_expectations():
    """`{metric: (direction, blind)}` of the `hiss_in_pauses` entry."""
    return {e.metric: (e.direction, e.blind) for e in deg.DEGRADATIONS["hiss_in_pauses"].expects}


def test_hiss_in_pauses_asserts_r4s_attenuation_and_reports_gap_air():
    """R4's true-pause attenuation and the residual noise are asserted; gap_air rides along blind."""
    expects = _hiss_expectations()
    assert expects["dsp.gap_atten_db"] == ("down", False)
    assert expects["dsp.gap_air_db"] == ("up", True)
    assert expects["dsp.residual_noise_db"] == ("up", False)


def test_on_calibration_v3bs_readings_r4_passes_where_gap_air_would_fail():
    """en and fr: R4 orders the hiss on both (10 dB a step); asserted, gap_air's order-statistic step orders on en alone."""
    atten = _two_languages("hiss_in_pauses", "dsp.gap_atten_db", {"en": [-33.9, -43.9, -53.8], "fr": [-41.0, -51.0, -60.9]})
    result = checks.check_expectation("hiss_in_pauses", deg.Expectation("dsp.gap_atten_db", "down"), *atten, HISS_FLOOR)
    assert (result["status"], result["monotonic_share"]) == ("pass", 1.0)
    air = _two_languages("hiss_in_pauses", "dsp.gap_air_db", {"en": [0.024, 0.025, 0.086], "fr": [0.00265, 0.00265, 0.00097]})
    asserted = checks.check_expectation("hiss_in_pauses", deg.Expectation("dsp.gap_air_db", "up"), *air, HISS_FLOOR)
    assert (asserted["status"], asserted["monotonic_share"]) == ("fail", 0.5)


def test_run_checks_covers_every_declared_expectation():
    """Run checks covers every declared expectation."""
    rows = checks.run_checks([], {}, {})
    assert len(rows) == sum(len(spec.expects) for spec in deg.DEGRADATIONS.values())
    assert {row["status"] for row in rows} == {"unscored"}
