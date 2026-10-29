"""The engine's event log: off unless the environment names a folder, and then one JSON record per stage and track."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from modules import config, event_log

TRACK = Path("D:/tapes/.temp_work_Vaccin 1990/apl/preconditioned.wav")


@pytest.fixture(name="log_dir")
def log_dir_fixture(tmp_path, monkeypatch):
    """The event log switched on into a folder of the test's own, for a run in `auto_pure_linear`."""
    folder = tmp_path / "events"
    monkeypatch.setenv(event_log.ENV_VAR, str(folder))
    monkeypatch.setattr(config, "PROCESS_MODE", "auto_pure_linear")
    return folder


def _read(path):
    """A written record, parsed."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_the_log_is_off_unless_the_environment_names_a_folder(monkeypatch, tmp_path):
    """Unset or blank is off; a folder name turns the log on."""
    monkeypatch.delenv(event_log.ENV_VAR, raising=False)
    assert event_log.directory() is None
    monkeypatch.setenv(event_log.ENV_VAR, "   ")
    assert not event_log.enabled()
    monkeypatch.setenv(event_log.ENV_VAR, str(tmp_path))
    assert event_log.directory() == tmp_path


def test_a_work_file_is_named_after_its_tape_and_any_other_after_itself():
    """The `.temp_work_<stem>` folder names the tape."""
    assert event_log.recording_name(TRACK) == "Vaccin 1990"
    assert event_log.recording_name(Path("D:/tapes/clip.wav")) == "clip"


def test_sample_spans_become_seconds_to_the_microsecond():
    """NumPy integers convert like plain ones."""
    spans = [(0, 44100), (np.int64(22050), np.int64(66150)), (1, 3)]
    assert event_log.seconds(spans, 44100) == [[0.0, 1.0], [0.5, 1.5], [0.000023, 0.000068]]
    assert event_log.seconds([], 48000) == []


def test_the_record_path_is_safe_for_a_file_name(log_dir, monkeypatch):
    """Spaces and separators become underscores; an empty name is 'unnamed'."""
    expected = log_dir / "sibilant_guard__Vaccin_1990__auto_pure_linear__preconditioned.json"
    assert event_log.record_path("sibilant guard", TRACK) == expected
    assert event_log.record_path("", Path("x/.wav")) == log_dir / "unnamed__.wav__auto_pure_linear__.wav.json"
    monkeypatch.delenv(event_log.ENV_VAR)
    assert event_log.record_path("pause_floor", TRACK) is None


def test_two_modes_on_one_tape_keep_a_record_each(log_dir, monkeypatch):
    """Both modes' runs share the tape's work folder; the mode in the name and the header keeps them apart."""
    first = event_log.write_spans("pause_floor", TRACK, [(0, 10)], 100)
    monkeypatch.setattr(config, "PROCESS_MODE", "cathar")
    second = event_log.write_skip("pause_floor", TRACK, "switched off")
    assert first != second
    assert [_read(path)["mode"] for path in (first, second)] == ["auto_pure_linear", "cathar"]
    assert len(list(log_dir.glob("pause_floor__Vaccin_1990__*__preconditioned.json"))) == 2


def test_a_missing_mode_reads_unknown(monkeypatch):
    """A configuration without a mode still names the record."""
    monkeypatch.setattr(config, "PROCESS_MODE", "")
    assert event_log.process_mode() == "unknown"
    assert event_log.record_name("loudnorm", TRACK) == "loudnorm__Vaccin_1990__unknown__preconditioned.json"


def test_values_json_cannot_hold_are_written_plainly(tmp_path):
    """NumPy scalars become numbers, anything else its string."""
    path = event_log.write_json(tmp_path / "doc.json", {"count": np.int64(3), "level": np.float32(0.5), "where": Path("a")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"count": 3, "level": 0.5, "where": "a"}


def test_a_record_that_cannot_be_written_is_logged_and_dropped(tmp_path):
    """A file where the folder should be: no record, one log line, no exception."""
    blocker = tmp_path / "file"
    blocker.write_text("in the way", encoding="utf-8")
    with patch("modules.event_log.log_msg") as log:
        assert event_log.write_json(blocker / "doc.json", {"a": 1}) is None
    assert "Could not write doc.json" in log.call_args[0][0]


def test_spans_are_written_with_the_thresholds_and_the_track(log_dir):
    """One record per stage and track, the spans in seconds."""
    path = event_log.write_spans("plosive_tamer", TRACK, [(4410, 8820)], 44100, excess_db=12.0)
    record = _read(path)
    assert path.parent == log_dir
    assert (record["stage"], record["recording"], record["schema"]) == ("plosive_tamer", "Vaccin 1990", event_log.SCHEMA)
    assert (record["rate"], record["count"], record["excess_db"], record["events_s"]) == (44100, 1, 12.0, [[0.1, 0.2]])


def test_a_skipped_stage_writes_its_reason(log_dir):
    """A stage that did not look says why, with no events."""
    record = _read(event_log.write_skip("plosive_tamer", TRACK, "tonal material", flatness=0.01))
    assert (record["skipped"], record["count"], record["events_s"], record["flatness"]) == ("tonal material", 0, [], 0.01)
    assert record["track"] == str(TRACK.resolve())
    assert len(list(log_dir.glob("*.json"))) == 1


def test_nothing_is_written_while_the_log_is_off(monkeypatch, tmp_path):
    """Off, every writer returns None and the folder stays empty."""
    monkeypatch.delenv(event_log.ENV_VAR, raising=False)
    assert event_log.write_spans("pause_floor", tmp_path / "a.wav", [(0, 10)], 100) is None
    assert event_log.write_skip("pause_floor", tmp_path / "a.wav", "switched off") is None
    assert event_log.write("pause_floor", tmp_path / "a.wav", {}) is None
    assert not list(tmp_path.iterdir())
