"""Per-listener playback calibration: the append-only file, the selection `auditory` reads at import, the verdict rule it joins."""

import importlib
import json

import pytest

from scripts.restoration_quality import auditory, playback_calibration
from tests.unit.test_restoration_quality_ledger import _set

LISTENER_ENV, PATH_ENV = playback_calibration.LISTENER_ENV, playback_calibration.PATH_ENV


def _record(rid="2026-10-12-threshold-user-aaaaaa", listener="user", loud=67.5):
    """A valid calibration record; `loud` None stands for a run that heard no tone under its cap."""
    return {
        "id": rid,
        "date": "2026-10-12",
        "round": "session1",
        "listener": listener,
        "playback": {"device": "Arctis Nova Pro", "volume": "40", "connection": None, "eq": "off", "anc": "off", "surround": "off"},
        "level_dbfs": -23.0,
        "cap_dbfs": -33.0,
        "sl_map": {"1000": -84.0, "12500": None},
        "thresholds": [
            {
                "hz": 1000.0,
                "threshold_dbfs": -84.0,
                "band_dbfs": [-88.0, -80.0],
                "censored": False,
                "spl_offset_db": 87.4,
                "responses": [[-80.0, True]],
                "cant_tell": [],
            }
        ],
        "spl_full_scale_db": None if loud is None else 89.0,
        "loud_frames_spl": loud,
        "context": {},
    }


BROKEN = [
    (("id",), "two words", "id must"),
    (("date",), "2026-13-40", "date must"),
    (("listener",), "a/b", "listener must"),
    (("playback",), {"device": "", "volume": "40"}, "playback needs"),
    (("playback",), {"device": "hp", "volume": 40}, "playback needs"),
    (("playback",), {"device": "hp", "volume": "40", "eq": 3}, "playback needs"),
    (("level_dbfs",), "loud", "level_dbfs and cap_dbfs"),
    (("sl_map",), {}, "sl_map must"),
    (("sl_map",), {"1000": "low"}, "sl_map must"),
    (("thresholds",), [], "thresholds must"),
    (("thresholds", 0, "band_dbfs"), [-80.0, -88.0], "thresholds must"),
    (("thresholds", 0, "censored"), "no", "thresholds must"),
    (("spl_full_scale_db",), "x", "spl_full_scale_db must"),
    (("loud_frames_spl",), float("nan"), "loud_frames_spl must"),
    (("context",), [], "context must"),
]


@pytest.mark.parametrize(("path", "value", "fragment"), BROKEN)
def test_a_broken_record_names_its_problem(path, value, fragment):
    """Each broken field is named; the valid record has no problem."""
    assert not playback_calibration.validate(_record())
    assert any(fragment in problem for problem in playback_calibration.validate(_set(_record(), path, value)))


def test_a_missing_field_or_a_non_object_is_refused():
    """A record without a field names it; a list is not a record."""
    record = _record()
    del record["sl_map"]
    assert playback_calibration.validate(record) == ["missing field 'sl_map'"]
    assert playback_calibration.validate([]) == ["a calibration record must be a JSON object"]


def test_records_round_trip_as_lf_terminated_json(tmp_path):
    """What is appended reads back unchanged and in order, one JSON object per LF-terminated line."""
    path = tmp_path / "cal.jsonl"
    records = [_record("r-1"), _record("r-2", listener="guest")]
    playback_calibration.append_many(records, path)
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert playback_calibration.read(path) == records
    assert playback_calibration.read(tmp_path / "missing.jsonl") == []


def test_an_id_already_in_the_file_and_a_broken_record_are_refused(tmp_path):
    """Nothing is written for a taken id or a record that breaks the schema."""
    path = tmp_path / "cal.jsonl"
    playback_calibration.append_many([_record("r-1")], path)
    with pytest.raises(playback_calibration.CalibrationError, match="already in the calibration file"):
        playback_calibration.append_many([_record("r-1")], path)
    with pytest.raises(playback_calibration.CalibrationError, match="new record 1"):
        playback_calibration.append_many([_set(_record("r-2"), ("listener",), "a b")], path)
    assert len(playback_calibration.read(path)) == 1


def test_a_hand_edited_last_line_keeps_its_own_line(tmp_path):
    """A last line without its line end is ended before the next record, which keeps a line of its own."""
    path = tmp_path / "cal.jsonl"
    path.write_text(json.dumps(_record("r-1")), encoding="utf-8")
    playback_calibration.append_many([_record("r-2")], path)
    assert [record["id"] for record in playback_calibration.read(path)] == ["r-1", "r-2"]


def test_a_broken_line_is_named_with_its_place(tmp_path):
    """A line that is not JSON stops the read and says where it is."""
    path = tmp_path / "cal.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(playback_calibration.CalibrationError, match=r"cal\.jsonl:1: not JSON"):
        playback_calibration.read(path)


# ----------------------------------------------------------------------------- what auditory reads


def _file(tmp_path, records):
    path = tmp_path / "cal.jsonl"
    playback_calibration.append_many(records, path)
    return path


def test_nothing_is_read_without_a_listener():
    """No listener named, or a blank one: no calibration, and the default stays."""
    assert playback_calibration.selected({}) is None
    assert playback_calibration.selected({LISTENER_ENV: "  "}) is None
    assert playback_calibration.loud_frames_spl(None, 65.0) == 65.0
    assert playback_calibration.rule_suffix(None) == ""


def test_the_listeners_latest_record_is_selected(tmp_path):
    """Two runs of the user and one of a guest: the user's later run is the one read, with its level and its id."""
    path = _file(tmp_path, [_record("r-1", loud=67.5), _record("r-2", listener="guest", loud=50.0), _record("r-3", loud=70.0)])
    record = playback_calibration.selected({LISTENER_ENV: "user", PATH_ENV: str(path)})
    assert record["id"] == "r-3"
    assert playback_calibration.loud_frames_spl(record, 65.0) == 70.0
    assert playback_calibration.rule_suffix(record) == "+r-3"


def test_a_listener_without_a_usable_run_stops_the_read(tmp_path):
    """A listener with no run, or whose latest run placed no level, is refused rather than read as 65 dB SPL."""
    path = _file(tmp_path, [_record("r-1"), _record("r-2", loud=None)])
    with pytest.raises(playback_calibration.CalibrationError, match="no playback calibration for listener 'guest'"):
        playback_calibration.selected({LISTENER_ENV: "guest", PATH_ENV: str(path)})
    with pytest.raises(playback_calibration.CalibrationError, match="places no playback level") as refused:
        playback_calibration.selected({LISTENER_ENV: "user", PATH_ENV: str(path)})
    assert str(refused.value).endswith(f"unset {LISTENER_ENV} ({BLOCKED}), then run listen_ab threshold --listener user")


BLOCKED = "the threshold block imports auditory and cannot start while it names an uncalibrated listener"


def test_the_refusal_says_to_unset_the_listener_before_the_threshold_block(tmp_path):
    """The threshold block imports auditory itself: the way out a refusal names is to unset the variable, then run it."""
    with pytest.raises(playback_calibration.CalibrationError) as refused:
        playback_calibration.selected({LISTENER_ENV: "guest", PATH_ENV: str(tmp_path / "none.jsonl")})
    assert f"unset {LISTENER_ENV}" in str(refused.value) and str(refused.value).endswith("--listener guest")


def test_auditory_takes_the_listeners_level_at_import(tmp_path, monkeypatch):
    """With a calibrated listener named, the module constant and the verdict rule follow the run; unset, they are back."""
    path = _file(tmp_path, [_record("r-9", loud=58.25)])
    base_rule = auditory.VERDICT_RULE
    monkeypatch.setenv(LISTENER_ENV, "user")
    monkeypatch.setenv(PATH_ENV, str(path))
    try:
        reloaded = importlib.reload(auditory)
        observed = (reloaded.PLAYBACK_SPL_LOUD_FRAMES, reloaded.VERDICT_RULE, reloaded.PLAYBACK_CALIBRATION["id"])
    finally:
        monkeypatch.delenv(LISTENER_ENV)
        monkeypatch.delenv(PATH_ENV)
        importlib.reload(auditory)
    assert observed == (58.25, base_rule + "+r-9", "r-9")
    assert (auditory.PLAYBACK_SPL_LOUD_FRAMES, auditory.DEFAULT_PLAYBACK_SPL_LOUD_FRAMES, auditory.VERDICT_RULE) == (65.0, 65.0, base_rule)


def test_auditory_will_not_import_for_an_uncalibrated_listener(tmp_path, monkeypatch):
    """A named listener with no run stops auditory's import, and with it listen_ab's: hence the refusal's "unset" first."""
    monkeypatch.setenv(LISTENER_ENV, "guest")
    monkeypatch.setenv(PATH_ENV, str(tmp_path / "none.jsonl"))
    try:
        with pytest.raises(playback_calibration.CalibrationError, match=f"unset {LISTENER_ENV}"):
            importlib.reload(auditory)
    finally:
        monkeypatch.delenv(LISTENER_ENV)
        monkeypatch.delenv(PATH_ENV)
        importlib.reload(auditory)
    assert (auditory.PLAYBACK_CALIBRATION, auditory.PLAYBACK_SPL_LOUD_FRAMES) == (None, 65.0)
