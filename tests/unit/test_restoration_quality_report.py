"""The Markdown report renders every metric, including one whose source side aggregated to nothing."""

from scripts.restoration_quality import report


def _result():
    verdict = {
        "gate": "dsp.lkr",
        "metric": "dsp.lkr",
        "stat": "delta.median",
        "value": 0.1,
        "threshold": 0.3,
        "status": "passed",
        "severity": "hard",
        "note": "",
    }
    flag = {**verdict, "gate": "listener.hiss", "status": "failed", "severity": "flag"}
    variant = {
        "families": {"dsp": "ok", "stems": "ok"},
        "passed": True,
        "hard_failures": [],
        "listener_flags": ["listener.hiss"],
        "lag_samples": 3,
        "verdicts": [verdict, flag],
        "aggregate": {
            "dsp.lkr": {
                "source": {"median": 0.0, "tail": 0.0, "n": 2},
                "output": {"median": 0.1, "tail": 0.2, "n": 2},
                "delta": {"median": 0.1, "tail": 0.2, "n": 2},
            },
            # An output-only reading: the source side is None, the delta absent.
            "stems.has_background": {"source": None, "output": {"median": 1.0, "tail": 1.0, "n": 2}, "delta": None},
        },
    }
    return {
        "source": {"path": "tape.mov"},
        "windows": {"seconds": 15.0, "hop": 7.5, "count": 2},
        "variants": {"a": variant},
        "metrics": {"dsp.lkr": {"family": "dsp"}, "stems.has_background": {"family": "stems"}},
    }


def test_a_metric_without_a_source_side_renders_as_dashes():
    text = report.render_markdown(_result())
    assert "| stems.has_background | - | - | - |" in text
    assert "| dsp.lkr | +0.00 | +0.10 | +0.20 |" in text


def test_flags_and_soft_failures_are_named_in_the_report():
    text = report.render_markdown(_result())
    assert "listener flags: listener.hiss" in text
    assert "FAIL (flag)" in text
