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
        assert spec.better in {"higher", "lower", "none"}
