"""The hiss readings' critical level under a calibrated listener, and the loop's score files that carry it.

Unmeasured, each file's loud frames play at 75 dB SPL (65 assumed plus a 10 dB allowance). A listener's threshold run
measured the volume their blind blocks play at, so the readings stand on that level plus a smaller allowance, the
least that made the hiss readings agree with every strong blind pair verdict at that setup. A score read at one level
must never be served at another: under a calibration the record's id joins the loop's score file names.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts import autotune_scoring as scoring
from scripts import rejudge_autotune as rejudge
from scripts.restoration_quality import hiss_floor
from tests.unit.test_restoration_quality_playback_calibration import LISTENER_ENV, PATH_ENV, _file, _record

REPO = Path(__file__).resolve().parents[2]


def test_unmeasured_the_level_is_75_and_a_record_takes_its_own_level_and_the_smaller_allowance():
    assert hiss_floor.critical_level_db(None, 65.0) == 75.0
    assert hiss_floor.critical_level_db(_record(loud=84.0), 84.0) == 84.0 + hiss_floor.MEASURED_ALLOWANCE_DB
    assert 0.0 <= hiss_floor.MEASURED_ALLOWANCE_DB < hiss_floor.UNMEASURED_ALLOWANCE_DB == 10.0


def _level_at_import(environ):
    """`hiss_floor.L_CRIT_DB` as a fresh interpreter imports it under `environ` (a reload here would redefine the
    module's dataclasses under the other tests)."""
    code = "from scripts.restoration_quality import hiss_floor; print(hiss_floor.L_CRIT_DB)"
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=environ, capture_output=True, text=True, check=True)
    return float(done.stdout.strip())


def test_the_module_level_follows_the_listener_named_at_import(tmp_path):
    """With a calibrated listener the constant is the record's level plus the allowance; unset, 75."""
    base = {key: value for key, value in os.environ.items() if key not in (LISTENER_ENV, PATH_ENV)}
    calibrated = {**base, LISTENER_ENV: "user", PATH_ENV: str(_file(tmp_path, [_record("r-9", loud=84.0)]))}
    assert _level_at_import(calibrated) == 84.0 + hiss_floor.MEASURED_ALLOWANCE_DB
    assert _level_at_import(base) == hiss_floor.L_CRIT_DB == 75.0


def test_score_files_keep_their_names_without_a_listener_and_carry_the_record_with_one(monkeypatch):
    assert (scoring.score_name("tele"), scoring.score_name("tele", scoring.LEARNED_STAGE)) == ("tele.score.json", "tele.learned.score.json")
    monkeypatch.setattr(scoring.auditory, "PLAYBACK_CALIBRATION", _record("r-9"))
    names = (scoring.score_name("tele"), scoring.score_name("tele", scoring.LEARNED_STAGE))
    assert names == ("tele.cal-r-9.score.json", "tele.learned.cal-r-9.score.json")


def test_a_score_read_at_the_unmeasured_level_is_not_served_to_a_calibrated_loop(tmp_path, monkeypatch):
    """The 75 dB score on disk does not count as scored once a listener is named: the candidate is scored again."""
    cand = tmp_path / "cands" / "c1"
    cand.mkdir(parents=True)
    (cand / "tele.score.json").write_text(json.dumps({"dsp.hiss_prog_excess_db.output.median": 0.0}), encoding="utf-8")
    assert scoring._unscored(["c1"], {"tele": "tape.wav"}, tmp_path) == []
    monkeypatch.setattr(scoring.auditory, "PLAYBACK_CALIBRATION", _record("r-9"))
    assert scoring._unscored(["c1"], {"tele": "tape.wav"}, tmp_path) == [("tele", "tape.wav", ["c1"])]


def test_the_offline_rejudge_reads_the_calibrated_files(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring.auditory, "PLAYBACK_CALIBRATION", _record("r-9"))
    (tmp_path / "tele.cal-r-9.score.json").write_text(json.dumps({"a": 1.0}), encoding="utf-8")
    (tmp_path / "tele.learned.cal-r-9.score.json").write_text(json.dumps({"b": 2.0}), encoding="utf-8")
    (tmp_path / "tele.score.json").write_text(json.dumps({"a": 9.0}), encoding="utf-8")
    assert rejudge._flat(tmp_path, "tele") == {"a": 1.0, "b": 2.0}
