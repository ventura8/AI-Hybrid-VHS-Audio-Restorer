"""The pause floor keeper's event log: the pauses it found, the frames it filled, and why it did neither.

The fixture is the gated voice of `test_pause_floor.py`: tone bursts on the first half of
each second over a hiss, the restored copy's pauses pushed 40 dB down.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import modules.pause_floor as pause_floor
from modules import event_log
from tests.unit.test_pause_floor import RATE, _gated, _on, _voice, _write


def _fill_with_log(tmp_path, work, log_dir, monkeypatch):
    """The gated voice through the keeper with the event log on (or off, when `log_dir` is None); returns the filled samples."""
    if log_dir is None:
        monkeypatch.delenv("AI_RESTORE_EVENT_LOG", raising=False)
    else:
        monkeypatch.setenv("AI_RESTORE_EVENT_LOG", str(log_dir))
    source = _voice()
    reference = _write(tmp_path / "ref.wav", source)
    restored = _write(tmp_path / "res.wav", _gated(source))
    with _on()[0], _on()[1], _on()[2]:
        return sf.read(str(pause_floor.apply_when_needed(reference, restored, work)), dtype="float32")[0]


def _pause_record(log_dir):
    """The keeper's event-log record for the gated voice (the restored track, `res.wav`)."""
    return json.loads((log_dir / event_log.record_name(pause_floor.STAGE, Path("res.wav"))).read_text(encoding="utf-8"))


def _pause_middles(record):
    """Where in its second each logged pause longer than 0.2 s is centred: the voice is on for the first half of each."""
    return [(start + stop) / 2.0 % 1.0 for start, stop in record["events_s"] if stop - start > 0.2]


def test_the_event_log_holds_the_pauses_found_and_changes_no_sample(tmp_path, monkeypatch):
    """The logged pauses sit in the off half of each second; the filled samples are the same with the log off."""
    logged = _fill_with_log(tmp_path, tmp_path / "on", tmp_path / "events", monkeypatch)
    record = _pause_record(tmp_path / "events")
    middles = _pause_middles(record)
    assert (record["fill_db"], record["quiet_percentile"], record["count"]) == (12.0, 15.0, len(record["events_s"]))
    assert len(middles) >= 7
    assert 0.5 < min(middles) <= max(middles) < 1.0
    assert np.array_equal(logged, _fill_with_log(tmp_path, tmp_path / "off", None, monkeypatch))


def test_the_event_log_says_how_much_was_filled_or_why_nothing_was(tmp_path, monkeypatch):
    """A fill logs its frame count; a mismatched pair logs the reason it filled nothing."""
    _fill_with_log(tmp_path, tmp_path / "work", tmp_path / "events", monkeypatch)
    assert _pause_record(tmp_path / "events")["outcome"] > 0
    mono = _write(tmp_path / "res.wav", _gated(_voice())[:, :1])
    with _on()[0], _on()[1], _on()[2]:
        assert pause_floor.apply_when_needed(tmp_path / "ref.wav", mono, tmp_path / "work") == mono
    assert _pause_record(tmp_path / "events")["outcome"] == "reference and restored audio differ in rate or channels"


def test_pause_spans_land_on_the_restored_timeline():
    """A run of quiet frames moves by the lag when the restored file trails; a lead leaves it in place."""
    weight = np.array([0.0, 1.0, 1.0, 0.0, 0.7, 0.2])
    assert pause_floor.pause_spans(weight > 0.5, 3) == [(4, 6), (7, 8)]
    assert pause_floor.pause_spans(weight > 0.5, -2) == [(1, 3), (4, 5)]


def _outside(spans_s, length, pad_s):
    """True on the samples away from every span, each widened by `pad_s` either side."""
    mask = np.ones(length, dtype=bool)
    for start, stop in spans_s:
        low, high = max(0, int((start - pad_s) * RATE)), int((stop + pad_s) * RATE)
        mask[low:high] = False
    return mask


def _inside_one(span, spans):
    """Whether a span lies wholly within one of `spans`."""
    return any(start <= span[0] and span[1] <= stop for start, stop in spans)


def test_the_filled_spans_hold_every_sample_the_fill_changed(tmp_path, monkeypatch):
    """Away from the logged fill (and the one frame its smoothing reaches) the samples are the restored ones to the bit."""
    logged = _fill_with_log(tmp_path, tmp_path / "work", tmp_path / "events", monkeypatch)
    filled = _pause_record(tmp_path / "events")["filled_s"]
    restored = sf.read(str(tmp_path / "res.wav"), dtype="float32")[0]
    untouched = _outside(filled, len(restored), pause_floor.FRAME_S)
    assert untouched.sum() > 0.3 * len(restored)
    assert np.array_equal(logged[untouched], restored[untouched])


def test_the_fill_reaches_past_the_pauses_found(tmp_path, monkeypatch):
    """Every pause found lies inside a filled run, and the fill also takes the edge frames the smoothing weights at 1/3."""
    _fill_with_log(tmp_path, tmp_path / "work", tmp_path / "events", monkeypatch)
    record = _pause_record(tmp_path / "events")
    assert all(_inside_one(pause, record["filled_s"]) for pause in record["events_s"])
    assert sum(stop - start for start, stop in record["filled_s"]) > sum(stop - start for start, stop in record["events_s"])


def test_a_switched_off_keeper_logs_why_it_did_not_look(tmp_path, monkeypatch):
    """Off, the record says so with the settings, and no pauses."""
    monkeypatch.setenv("AI_RESTORE_EVENT_LOG", str(tmp_path / "events"))
    restored = _write(tmp_path / "res.wav", _gated(_voice()))
    with patch("modules.pause_floor.ENABLE_PAUSE_FLOOR", False):
        assert pause_floor.apply_when_needed(tmp_path / "ref.wav", restored, tmp_path / "work") == restored
    record = _pause_record(tmp_path / "events")
    assert (record["skipped"], record["events_s"], record["fill_db"]) == ("switched off", [], pause_floor.PAUSE_FLOOR_FILL_DB)


def test_a_failing_fill_logs_the_failure_in_place_of_the_pauses(tmp_path, monkeypatch):
    """The fill raised, so the output carries none: the record says the stage failed."""
    monkeypatch.setenv("AI_RESTORE_EVENT_LOG", str(tmp_path / "events"))
    source = _voice()
    reference = _write(tmp_path / "ref.wav", source)
    restored = _write(tmp_path / "res.wav", _gated(source))
    with _on()[0], _on()[1], _on()[2], patch("modules.pause_floor.fill_file", side_effect=OSError("disk")):
        assert pause_floor.apply_when_needed(reference, restored, tmp_path / "work") == restored
    record = _pause_record(tmp_path / "events")
    assert (record["skipped"], record["count"], record["quiet_percentile"]) == ("failed: disk", 0, 15.0)
