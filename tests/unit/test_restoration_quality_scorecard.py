"""Median/tail aggregation and paired deltas."""

import numpy as np

from scripts.restoration_quality import scorecard as sc


def _rows():
    rows = []
    for index, (src, out) in enumerate([(1.0, 2.0), (1.0, 0.0), (1.0, 3.0), (1.0, None)]):
        source = {"dsp.hf_4k8k": src, "speech.cer": 0.1}
        output = {"dsp.hf_4k8k": out, "speech.cer": 0.1 + index / 10.0}
        rows.append(sc.WindowRow(index, index * 7.5, index * 7.5 + 15.0, "speech", source, output))
    return rows


def test_delta_skips_missing_and_non_numeric_readings():
    row = sc.WindowRow(0, 0.0, 15.0, "speech", {"a": 1.0, "b": 2.0, "c": float("nan")}, {"a": 1.5, "c": 1.0, "d": 9.0})
    assert row.delta() == {"a": 0.5}


def test_summarise_points_the_tail_at_the_bad_end():
    values = np.arange(1.0, 11.0)
    higher = sc.summarise(values, "higher")
    lower = sc.summarise(values, "lower")
    assert higher["median"] == 5.5
    assert higher["tail"] < higher["median"]
    assert lower["tail"] > lower["median"]


def test_summarise_counts_and_handles_empty_input():
    assert sc.summarise(np.arange(1.0, 11.0), "higher")["n"] == 10
    assert sc.summarise(np.zeros(0), "higher") is None


def test_aggregate_reports_every_scored_metric():
    result = sc.aggregate(_rows())
    assert set(result) == {"dsp.hf_4k8k", "speech.cer"}
    assert result["dsp.hf_4k8k"]["output"]["n"] == 3
    assert result["dsp.hf_4k8k"]["delta"]["median"] == 1.0


def test_aggregate_tails_follow_each_metric_direction():
    result = sc.aggregate(_rows())
    assert result["dsp.hf_4k8k"]["delta"]["tail"] < result["dsp.hf_4k8k"]["delta"]["median"]
    assert result["speech.cer"]["delta"]["tail"] > result["speech.cer"]["delta"]["median"]


def test_paired_deltas_flattens_a_card():
    card = sc.ScoreCard(rows=_rows())
    card.aggregate = sc.aggregate(card.rows)
    flat = sc.paired_deltas(card)
    assert flat["dsp.hf_4k8k"] == 1.0
    assert set(flat) == {"dsp.hf_4k8k", "speech.cer"}


def test_every_metric_spec_has_a_family_and_direction():
    for name, spec in sc.METRICS.items():
        assert name.startswith(spec.family + ".")
        assert spec.better in sc.DIRECTIONS


def test_a_two_sided_tail_is_the_decile_further_from_the_target():
    """A reading wrong either way points its tail at whichever end moved further from its neutral point."""
    values = np.array([-3.0, -2.0, -1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert sc.summarise(values, sc.TWO_SIDED)["tail"] < 0.0
    assert sc.summarise(-values, sc.TWO_SIDED)["tail"] > 0.0
    assert sc.summarise(values, sc.TWO_SIDED, target=-3.0)["tail"] > 0.0


def test_counts_report_their_worst_window_and_p95():
    """One click in one window is not diluted by the median: the max and p95 sit beside it."""
    rows = [
        sc.WindowRow(i, i * 7.5, i * 7.5 + 15.0, "speech", {"dsp.clicks_per_s": 0.0}, {"dsp.clicks_per_s": c})
        for i, c in enumerate([0, 0, 0, 2.0])
    ]
    summary = sc.aggregate(rows)["dsp.clicks_per_s"]["output"]
    assert (summary["median"], summary["max"]) == (0.0, 2.0)
    assert 0.0 < summary["p95"] <= 2.0
    assert "max" not in sc.summarise(np.arange(4.0), sc.LOWER)


def test_the_ear_v3_readings_are_registered_with_their_family_and_direction():
    """R1 and R2's level are two-sided, R4's texture lower-is-better, R7 and the sync per file, R0 displayed."""
    two_sided = ("dsp.balance_tilt_db_oct", "dsp.balance_top_db", "dsp.sib_abs_level_db", "dsp.gap_hf_excess_db", "dsp.gap_atten_db")
    assert all(sc.METRICS[name].better == sc.TWO_SIDED for name in two_sided)
    assert sc.METRICS["dsp.gap_mod_dist_db"].routes == sc.SPEECH_ROUTES
    assert {sc.METRICS[name].family for name in ("file.gain_ride_lu", "file.sync_drift_ms")} == {"file"}
    assert all(spec.better == sc.NONE for name, spec in sc.METRICS.items() if spec.family == "meta")
