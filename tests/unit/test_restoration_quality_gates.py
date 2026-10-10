"""Gates are data: every operator, the speaker floor, JSON overrides, and skipped readings."""

import json

import pytest

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


def test_the_hiss_flag_is_shown_but_no_longer_counted():
    """Gap level does not order the verdicts (ear v3, R4): a hissy gap reading fails a soft gate, not a flag."""
    verdicts = g.evaluate_gates(_aggregate(**{"dsp.gap_air_db": (-10.0, -10.0, 5.0, 5.0)}), g.GATES)
    assert _status(verdicts, "listener.hiss") == "failed"
    assert g.GATES["listener.hiss"].severity == g.SOFT
    assert "listener.hiss" not in g.flag_failures(verdicts)


@pytest.mark.parametrize(("top", "flags"), [(1.5, ["listener.bright"]), (-1.5, ["listener.dull"]), (0.5, []), (-0.9, [])])
def test_bright_and_dull_read_presence_or_air_two_sided(top, flags):
    """One reading, two flags: a lifted top is bright, a lost one dull, the accepted band between them flags nothing."""
    verdicts = g.evaluate_gates(_aggregate(**{"dsp.balance_top_db": (top, top, top, top)}), g.GATES)
    assert [name for name in g.flag_failures(verdicts) if name in ("listener.bright", "listener.dull")] == flags


def test_the_bright_and_dull_thresholds_say_they_are_uncalibrated():
    """+1.0 / -1.0 dB are starting values until Round 0 reads the accepted +1 dB air file."""
    verdicts = g.evaluate_gates(_aggregate(**{"dsp.balance_top_db": (0.0, 0.0, 0.0, 0.0)}), g.GATES)
    calibrated = {v["gate"]: v["calibrated"] for v in verdicts}
    assert calibrated["listener.bright"] is calibrated["listener.dull"] is False
    assert calibrated["dsp.lkr"] is True


@pytest.mark.parametrize(
    ("entries", "failed"),
    [
        ({"file.sync_drift_ms": 0.3, "file.sync_offset_ms": 5.0, "file.sync_unmatched": 0.0}, []),
        ({"file.sync_drift_ms": 55.0, "file.sync_offset_ms": 5.0, "file.sync_unmatched": 0.0}, ["file.sync_drift"]),
        ({"file.sync_drift_ms": 0.3, "file.sync_offset_ms": 1500.0, "file.sync_unmatched": 0.0}, ["file.sync_offset"]),
        ({"file.sync_offset_ms": 5.0, "file.sync_unmatched": 2.0}, ["file.sync_unmatched"]),
    ],
)
def test_a_sync_fault_of_a_video_frame_is_a_hard_failure(entries, failed):
    """40 ms of drift or offset parts lips from picture; an anchor the output does not follow is a broken sync."""
    aggregate = _aggregate(**{name: (value, value, value, value) for name, value in entries.items()})
    assert [name for name in g.hard_failures(g.evaluate_gates(aggregate, g.GATES)) if name.startswith("file.sync")] == failed


def test_a_calibration_entry_that_names_its_source_is_calibrated(tmp_path):
    """A threshold a calibration derived carries its `source`; a hand override without one keeps the default."""
    path = tmp_path / "gates.json"
    entries = {"listener.bright": {"threshold": 0.3, "source": "known ordering"}, "listener.dull": {"threshold": -0.4}}
    path.write_text(json.dumps(entries), encoding="utf-8")
    merged = g.load_gates(path)
    assert (merged["listener.bright"].threshold, merged["listener.bright"].calibrated) == (0.3, True)
    assert (merged["listener.dull"].threshold, merged["listener.dull"].calibrated) == (-0.4, False)


def test_a_stored_v2_gates_file_cannot_make_the_hiss_reading_count_again(tmp_path):
    """The v2 gates.json lists listener.hiss as a flag; its threshold merges, its severity does not (display-only)."""
    path = tmp_path / "gates.json"
    entry = {"threshold": -19.95, "severity": "flag", "source": "known ordering"}
    path.write_text(json.dumps({"listener.hiss": entry, "listener.dead_air": {"threshold": -26.0, "severity": "hard"}}), encoding="utf-8")
    merged = g.load_gates(path)
    assert (merged["listener.hiss"].threshold, merged["listener.hiss"].severity, merged["listener.hiss"].calibrated) == (
        -19.95,
        g.SOFT,
        True,
    )
    assert merged["listener.dead_air"].severity == g.HARD
    verdicts = g.evaluate_gates(_aggregate(**{"dsp.gap_air_db": (-10.0, -10.0, 5.0, 5.0)}), merged)
    assert _status(verdicts, "listener.hiss") == "failed"
    assert "listener.hiss" not in g.flag_failures(verdicts)


def test_a_base_without_the_display_only_gate_takes_the_file_entry_whole(tmp_path):
    """Only a gate the base defines keeps the base's severity; a file-only gate needs the severity it names."""
    path = tmp_path / "gates.json"
    hiss = {"metric": "dsp.gap_air_db", "stat": "median", "op": "<=", "threshold": -20.0, "severity": "flag"}
    path.write_text(json.dumps({"listener.hiss": hiss}), encoding="utf-8")
    assert g.load_gates(path, base={})["listener.hiss"].severity == g.FLAG
