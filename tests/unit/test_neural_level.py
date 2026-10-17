"""The neural level restore: off by default, the projection gain put back, small gains and unrelated pairs left alone."""

from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from modules import hiss_floor, neural_level, utils
from tests.unit.test_hiss_floor import _bursts, _hiss, _write


@pytest.fixture(autouse=True)
def _quiet_log(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "LOG_FILE", tmp_path / "session_log.txt")


@pytest.fixture(name="key_on")
def key_on_fixture(monkeypatch):
    monkeypatch.setattr(neural_level, "APL_NEURAL_LEVEL_RESTORE", True)


def _pair(tmp_path, level):
    """A neural input (programme plus hiss) and an output with the hiss removed, at `level` of the programme."""
    programme = _bursts(4.0)
    neural_in = _write(tmp_path / "in.wav", programme + _hiss(4.0))
    neural_out = _write(tmp_path / "out.wav", level * programme)
    return neural_in, neural_out


def _log(tmp_path):
    return (tmp_path / "session_log.txt").read_text(encoding="utf-8")


OPTED = {"neural_level_restore": True}


def test_off_by_default_or_not_opted_in_nothing_is_read(tmp_path, monkeypatch):
    with patch.object(hiss_floor, "frame_pair", side_effect=AssertionError("read")):
        off = neural_level.restore_when_needed(tmp_path / "in.wav", tmp_path / "out.wav", tmp_path, OPTED)
        monkeypatch.setattr(neural_level, "APL_NEURAL_LEVEL_RESTORE", True)
        not_opted = neural_level.restore_when_needed(tmp_path / "in.wav", tmp_path / "out.wav", tmp_path, {})
    assert (off, not_opted) == (tmp_path / "out.wav", tmp_path / "out.wav")


def test_a_stage_that_did_not_run_is_passed_through(tmp_path, key_on):
    del key_on
    with patch.object(hiss_floor, "frame_pair", side_effect=AssertionError("read")):
        assert neural_level.restore_when_needed(tmp_path / "same.wav", tmp_path / "same.wav", tmp_path, OPTED) == tmp_path / "same.wav"


def test_a_separator_rescale_is_put_back(tmp_path, key_on):
    del key_on
    neural_in, neural_out = _pair(tmp_path, level=0.3)
    levelled = neural_level.restore_when_needed(neural_in, neural_out, tmp_path, OPTED)
    assert (levelled.parent.name, levelled.name.startswith("levelled_+10.4")) == ("neural_level", True)
    original, _rate = sf.read(str(neural_out), dtype="float64")
    restored, _rate = sf.read(str(levelled), dtype="float64")
    assert (np.max(np.abs(restored - original / 0.3)) < 0.02, "restoring +10.4" in _log(tmp_path)) == (True, True)
    reused = neural_level.restore_when_needed(neural_in, neural_out, tmp_path, OPTED)
    assert (reused, "Reusing" in _log(tmp_path)) == (levelled, True)


def test_a_gain_within_the_tolerance_is_left_alone(tmp_path, key_on):
    del key_on
    neural_in, neural_out = _pair(tmp_path, level=0.98)
    assert neural_level.restore_when_needed(neural_in, neural_out, tmp_path, OPTED) == neural_out
    assert not (tmp_path / "neural_level").exists()
    assert "Skipped: the programme left the neural stage" in _log(tmp_path)


def test_an_unrelated_or_short_output_is_refused(tmp_path, key_on):
    del key_on
    neural_in = _write(tmp_path / "in.wav", _bursts(4.0) + _hiss(4.0))
    unrelated = _write(tmp_path / "noise.wav", _hiss(4.0, seed=8, level=0.05))
    assert neural_level.restore_when_needed(neural_in, unrelated, tmp_path, OPTED) == unrelated
    short = _write(tmp_path / "short.wav", _hiss(0.1))
    assert neural_level.restore_when_needed(short, short.with_name("short2.wav"), tmp_path, OPTED) == short.with_name("short2.wav")
    log = _log(tmp_path)
    assert ("does not follow" in log, "Skipped after failure" in log) == (True, True)
    assert neural_level.programme_gain(short, short) == (None, "too short to read")
