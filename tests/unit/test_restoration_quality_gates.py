"""Gates are data: every operator, the speaker floor, JSON overrides, and skipped readings."""

import json

from scripts.restoration_quality import gates as g


def _aggregate(**entries):
    """`{metric: {...}}` from `metric=(output_median, output_tail, delta_median, delta_tail)`."""
    result = {}
    for name, (om, ot, dm, dt) in entries.items():
        result[name] = {
            "source": {"median": 0.0, "tail": 0.0, "n": 4},
            "output": {"median": om, "tail": ot, "n": 4},
            "delta": {"median": dm, "tail": dt, "n": 4},
        }
    return result


def _status(verdicts, name):
    return next(v["status"] for v in verdicts if v["gate"] == name)


def _threshold(verdicts, name):
    return next(v["threshold"] for v in verdicts if v["gate"] == name)


def test_output_statistics_pass_and_fail():
    aggregate = _aggregate(**{"speech.cer": (0.02, 0.10, 0.0, 0.0), "dsp.hf_4k8k": (-20.0, -25.0, -9.0, -12.0)})
    verdicts = g.evaluate_gates(aggregate, g.GATES)
    assert _status(verdicts, "speech.cer_tail") == "passed"
    assert _status(verdicts, "speech.cer_median") == "passed"
    assert _status(verdicts, "dsp.hf_4k8k") == "failed"


def test_absolute_operator_and_unscored_metrics():
    aggregate = _aggregate(**{"file.lufs": (-20.0, -20.0, 1.0, 1.0)})
    verdicts = g.evaluate_gates(aggregate, g.GATES)
    assert _status(verdicts, "file.lufs") == "passed"
    assert _status(verdicts, "dsp.lkr") == "skipped"


def test_speaker_floor_is_required_for_the_relative_gate():
    aggregate = _aggregate(**{"speech.speaker_cos": (0.90, 0.84, 0.0, 0.0)})
    verdicts = g.evaluate_gates(aggregate, g.GATES)
    assert _status(verdicts, "speech.speaker") == "skipped"


def test_speaker_floor_resolves_the_relative_threshold():
    aggregate = _aggregate(**{"speech.speaker_cos": (0.90, 0.84, 0.0, 0.0)})
    passing = g.evaluate_gates(aggregate, g.GATES, speaker_floor=0.88)
    failing = g.evaluate_gates(aggregate, g.GATES, speaker_floor=0.92)
    assert _status(passing, "speech.speaker") == "passed"
    assert _status(failing, "speech.speaker") == "failed"
    assert _threshold(failing, "speech.speaker") == 0.87


def test_hard_failures_ignore_soft_gates():
    aggregate = _aggregate(**{"file.lufs": (-20.0, -20.0, 4.0, 4.0), "dsp.lkr": (0.0, 0.0, 0.9, 0.9)})
    verdicts = g.evaluate_gates(aggregate, g.GATES)
    assert _status(verdicts, "file.lufs") == "failed"
    assert g.hard_failures(verdicts) == ["dsp.lkr"]


def test_listener_flags_are_counted_apart_from_hard_failures():
    aggregate = _aggregate(**{"dsp.gap_air_db": (-30.0, -30.0, -5.0, -5.0), "dsp.lkr": (0.0, 0.0, 0.9, 0.9)})
    verdicts = g.evaluate_gates(aggregate, g.GATES)
    assert _status(verdicts, "listener.dead_air") == "failed"
    assert _status(verdicts, "listener.hiss") == "passed"
    assert g.flag_failures(verdicts) == ["listener.dead_air"]
    assert g.hard_failures(verdicts) == ["dsp.lkr"]


def _write_overrides(tmp_path):
    path = tmp_path / "gates.json"
    custom = {"metric": "dsp.hum_excess_db", "stat": "delta.tail", "op": "<=", "threshold": 1.0}
    path.write_text(json.dumps({"dsp.lkr": {"threshold": 0.8, "severity": "soft"}, "custom": custom}), encoding="utf-8")
    return path


def test_load_gates_merges_overrides_over_the_defaults(tmp_path):
    merged = g.load_gates(_write_overrides(tmp_path))
    assert merged["dsp.lkr"].threshold == 0.8
    assert merged["dsp.lkr"].severity == "soft"
    assert merged["dsp.lkr"].metric == "dsp.lkr"


def test_load_gates_adds_new_rules(tmp_path):
    merged = g.load_gates(_write_overrides(tmp_path))
    assert merged["custom"].op == "<="
    assert len(merged) == len(g.GATES) + 1


def test_read_handles_missing_sides():
    gate = g.Gate("x", "delta.median", "<=", 1.0)
    assert g._read({"x": {"output": {"median": 1.0}}}, gate) is None
    assert g._read({}, gate) is None
    assert g._read({"x": {"delta": {"median": 0.5}}}, gate) == 0.5


def test_load_gates_ignores_keys_the_gate_does_not_carry(tmp_path):
    """A calibration gates.json also records where each threshold came from; that is not a Gate field."""
    path = tmp_path / "gates.json"
    path.write_text(json.dumps({"dsp.lkr": {"threshold": 1.75, "severity": "hard", "source": "known good bound"}}), encoding="utf-8")
    merged = g.load_gates(path)
    assert merged["dsp.lkr"].threshold == 1.75
