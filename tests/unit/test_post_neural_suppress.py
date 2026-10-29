"""The post-neural suppressor and the air gate: off by default, gated per file on one hiss-floor reading."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from modules import hiss_floor, post_neural_suppress, utils
from tests.unit.test_hiss_floor import RATE, _bursts, _hiss, _write

OPEN = {"floor_gap_db": 2.2, "median_delta_db": -20.0, "presence_delta_db": -2.5}


@pytest.fixture(autouse=True)
def _quiet_log(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "LOG_FILE", tmp_path / "session_log.txt")
    monkeypatch.delenv("AI_RESTORE_EVENT_LOG", raising=False)


@pytest.fixture(name="keys_on")
def keys_on_fixture(monkeypatch):
    monkeypatch.setattr(post_neural_suppress, "APL_POST_NEURAL_SUPPRESS", True)
    monkeypatch.setattr(post_neural_suppress, "LINEAR_AIR_GATE", "hiss_dominated")


def _apply(tmp_path, reading, **stages):
    with (
        patch.object(hiss_floor, "read", return_value=reading) as read,
        patch.object(post_neural_suppress, "suppress", return_value=tmp_path / "suppressed.wav") as run,
    ):
        result = post_neural_suppress.apply_when_needed(tmp_path / "source.wav", tmp_path / "restored.wav", tmp_path, stages)
    return result, read, run


ALL = {"apply_air": True, "post_neural_suppress": True, "air_gate": True}


def test_the_keys_off_read_nothing_and_hand_back_both_values_unchanged(tmp_path):
    """The post-neural keys at the releases' values (the reference suppressor too, on by default since 2026-10-23)."""
    restored = tmp_path / "restored.wav"
    with (
        patch.object(hiss_floor, "read", side_effect=AssertionError("read")),
        patch.object(post_neural_suppress, "APL_REFERENCE_SUPPRESS", False),
    ):
        result = post_neural_suppress.apply_when_needed(tmp_path / "s.wav", restored, tmp_path, ALL)
    assert result[0] is restored
    assert result[1] is True
    assert not (tmp_path / post_neural_suppress.STAGE).exists()


def test_keys_on_but_not_opted_in_read_nothing(tmp_path, keys_on):
    del keys_on
    result, read, run = _apply(tmp_path, OPEN, apply_air=True)
    assert result == (tmp_path / "restored.wav", True)
    read.assert_not_called()
    run.assert_not_called()


def test_an_open_gate_suppresses_and_skips_the_air_shelf(tmp_path, keys_on):
    del keys_on
    result, read, run = _apply(tmp_path, OPEN, **ALL)
    assert result == (tmp_path / "suppressed.wav", False)
    assert read.call_args.args == (tmp_path / "source.wav", tmp_path / "restored.wav")
    run.assert_called_once_with(tmp_path / "restored.wav", tmp_path)


def test_the_air_gate_alone_keeps_the_audio_and_an_absent_shelf_stays_absent(tmp_path, keys_on):
    del keys_on
    result, _read, run = _apply(tmp_path, OPEN, apply_air=True, air_gate=True)
    assert result == (tmp_path / "restored.wav", False)
    run.assert_not_called()
    result, read, _run = _apply(tmp_path, OPEN, apply_air=False, air_gate=True)
    assert result == (tmp_path / "restored.wav", False)
    read.assert_not_called()


@pytest.mark.parametrize("switch, value", [("ENABLE_LINEAR_AIR", False), ("LINEAR_AIR_GAIN_DB", 0.0)])
def test_a_shelf_the_configuration_switched_off_leaves_the_air_gate_nothing_to_read(tmp_path, keys_on, monkeypatch, switch, value):
    del keys_on
    monkeypatch.setattr(post_neural_suppress, switch, value)
    result, read, _run = _apply(tmp_path, OPEN, apply_air=True, air_gate=True)
    assert result == (tmp_path / "restored.wav", True)
    read.assert_not_called()


@pytest.mark.parametrize(
    "reading, why",
    [
        ({"reason": "too short to read"}, "no reading (too short to read)"),
        ({**OPEN, "floor_gap_db": 3.5}, "not hiss-dominated"),
        ({**OPEN, "floor_gap_db": None}, "not hiss-dominated"),
        ({**OPEN, "median_delta_db": -20.0, "presence_delta_db": -6.0}, "past the residual"),
        ({**OPEN, "median_delta_db": None, "presence_delta_db": None}, "past the residual"),
    ],
)
def test_a_shut_gate_leaves_the_audio_and_the_shelf(tmp_path, keys_on, reading, why):
    del keys_on
    result, _read, run = _apply(tmp_path, reading, **ALL)
    assert result == (tmp_path / "restored.wav", True)
    run.assert_not_called()
    assert why in post_neural_suppress._shut_because(reading)


def test_either_residual_reading_opens_the_gate():
    assert post_neural_suppress._shut_because({**OPEN, "presence_delta_db": None, "median_delta_db": -5.9}) is None
    assert post_neural_suppress._shut_because({**OPEN, "presence_delta_db": -5.9, "median_delta_db": -40.0}) is None
    assert post_neural_suppress._shut_because({**OPEN, "floor_gap_db": 3.0}) is None


def test_a_reading_that_fails_shuts_the_gate_and_says_why(tmp_path):
    with patch.object(hiss_floor, "read", side_effect=OSError("locked")):
        verdict = post_neural_suppress.decide(tmp_path / "s.wav", tmp_path / "r.wav")
    assert (verdict["open"], verdict["why"]) == (False, "no reading (failed: locked)")
    assert "Shut" in (tmp_path / "session_log.txt").read_text(encoding="utf-8")


def _open_run(tmp_path, monkeypatch):
    """Run the stage on an open reading with the event log on; returns the session log's text."""
    monkeypatch.setenv("AI_RESTORE_EVENT_LOG", str(tmp_path / "events"))
    _apply(tmp_path, OPEN, **ALL)
    return (tmp_path / "session_log.txt").read_text(encoding="utf-8")


def test_the_decision_is_logged_with_its_numbers(tmp_path, keys_on, monkeypatch):
    del keys_on
    log = _open_run(tmp_path, monkeypatch)
    assert "[Post-Neural Hiss Gate] Open" in log
    assert "+2.2 dB" in log
    assert "-2.5 dB (3-4.8 kHz)" in log
    assert "[Air Gate] Skipping the air shelf" in log


def test_the_decision_is_recorded_in_the_event_log(tmp_path, keys_on, monkeypatch):
    del keys_on
    _open_run(tmp_path, monkeypatch)
    (record,) = (tmp_path / "events").glob("post_neural_suppress__*.json")
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert (payload["open"], payload["suppressed"], payload["air_shelf"], payload["reading"]) == (True, True, False, OPEN)
    assert (payload["floor_gap_db"], payload["min_residual_db"]) == (3.0, 6.0)


def test_the_summary_of_a_missing_reading_is_its_reason():
    assert post_neural_suppress._summary({"reason": "x"}) == "x"
    assert "- dB (band median)" in post_neural_suppress._summary({"floor_gap_db": 1.0})


def _residual_pair(tmp_path, residual):
    programme, hiss = _bursts(8.0), _hiss(8.0)
    source = _write(tmp_path / "source.wav", programme + hiss)
    restored = _write(tmp_path / "restored.wav", programme + residual * hiss)
    return source, restored


def test_on_real_files_a_passed_floor_opens_the_gate_and_the_suppressor_lowers_it(tmp_path, keys_on):
    del keys_on
    source, restored = _residual_pair(tmp_path, residual=0.8)
    output, air = post_neural_suppress.apply_when_needed(source, restored, tmp_path, ALL)
    assert (output.parent.name, air, sf.info(str(output)).frames) == (post_neural_suppress.STAGE, False, sf.info(str(restored)).frames)
    before, after = hiss_floor.read(source, restored), hiss_floor.read(source, output)
    # The 8 s fixture's quietest 4 s holds programme bursts, so its probe is milder than a tape's.
    assert after["median_delta_db"] < before["median_delta_db"] - 1.5
    reused = post_neural_suppress.suppress(restored, tmp_path)
    assert (reused, "Reusing" in (tmp_path / "session_log.txt").read_text(encoding="utf-8")) == (output, True)


def test_on_real_files_a_removed_floor_keeps_the_gate_shut(tmp_path, keys_on):
    del keys_on
    source, restored = _residual_pair(tmp_path, residual=0.02)
    assert post_neural_suppress.apply_when_needed(source, restored, tmp_path, ALL) == (Path(restored), True)


def test_a_setting_change_never_reuses_a_stale_file(tmp_path, monkeypatch):
    first = post_neural_suppress._target_for(tmp_path / "r.wav", tmp_path)
    monkeypatch.setattr(post_neural_suppress, "APL_SUPPRESS_GAIN_FLOOR_DB", -25.0)
    assert post_neural_suppress._target_for(tmp_path / "r.wav", tmp_path) != first


def test_a_suppressor_that_cannot_run_hands_back_its_input(tmp_path):
    short = _write(tmp_path / "short.wav", np.zeros(int(0.5 * RATE)))
    assert post_neural_suppress.suppress(short, tmp_path) == short
    assert "Skipped: no quiet stretch" in (tmp_path / "session_log.txt").read_text(encoding="utf-8")
