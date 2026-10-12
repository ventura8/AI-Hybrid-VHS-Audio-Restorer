"""The listening server's protocol: practice with feedback, catch trials scored apart, "can't tell", and the threshold run."""

import json

import numpy as np

from scripts import listen_ab
from scripts.restoration_quality import ledger, listen_modes, listen_protocol, playback_calibration
from scripts.restoration_quality.listen_modes import AbxMode
from scripts.restoration_quality.listen_threshold import CATCH_TONE, ThresholdMode
from tests.unit.test_listen_ab import PORT, _answer_everything, _base, _cuts, _get, _post, _session, _state

CALIBRATION_ID = "2026-10-12-threshold-user-abc123"

# ----------------------------------------------------------------------------- practice and catch trials through the server


def _protocol_session(tmp_path, seed=5):
    """A 16-trial ABX with 2 practice trials and 2 catch trials over an air-shelf control."""
    rng = np.random.default_rng(seed)
    mode = AbxMode(["a", "b"], 16, rng)
    protocol = listen_protocol.Protocol(mode, rng, familiar=2, catches=2)
    cuts = {
        **_cuts(),
        listen_modes.CATCH_AIR: _cuts()["a"] * 0.7,
        listen_modes.FAMILIAR_A: _cuts()["a"],
        listen_modes.FAMILIAR_B: _cuts()["b"],
    }
    return listen_ab.Session(protocol, cuts, (_base(), tmp_path / "v.jsonl"), PORT, {"loops": {"pause": [1.0, 3.0]}})


def test_the_block_opens_on_practice(tmp_path):
    """Practice 1 of 2 comes first; the test phase counts 18 presentations (16 scored, 2 catches); the pause loop is offered."""
    state = _state(_protocol_session(tmp_path))
    assert (state["phase"], state["number"], state["of"], state["total"], state["loops"]) == ("practice", 1, 2, 18, {"pause": [1.0, 3.0]})


def test_a_practice_answer_comes_back_with_feedback_and_is_not_scored(tmp_path):
    """The practice answer's reply names the answer, moves to practice 2 and leaves the mode's score empty."""
    session = _protocol_session(tmp_path)
    key = session.view["trial"]["key"]
    reply = json.loads(_post(session, {"trial": 0, "choice": "A"})[2])
    observed = (reply["feedback"], reply["phase"], reply["number"], session.mode.answers)
    assert observed == (listen_protocol.feedback_text(key, key == "A"), "practice", 2, [])


def test_catch_answers_stay_out_of_the_score_and_reach_the_record(tmp_path):
    """Every answer right: 16 scored in the ledger record, 2 catches caught in its context, practice counted apart."""
    session = _protocol_session(tmp_path)
    _answer_everything(session, lambda trial: trial["key"])
    record = ledger.read(tmp_path / "v.jsonl")[0]
    context = record["context"]
    observed = (record["n_trials"], record["n_correct"], context["catch"]["caught"], context["familiarisation"], context["block_flags"])
    assert observed == (16, 16, 2, {"trials": 2, "right": 2}, [])
    assert (context["power"]["critical_correct"], "catch trials 2/2 caught" in session.view["summary"]) == (12, True)


def _catch_misser(trial, picks):
    """Misses every catch trial, alternates A and B on the scored ones, answers A in practice."""
    if trial["phase"] == listen_protocol.CATCH:
        return "B" if trial["key"] == "A" else "A"
    return next(picks) if trial["phase"] == listen_protocol.SCORED else "A"


def test_a_missed_catch_flags_the_block_disengaged(tmp_path):
    """Scored answers on both buttons, every catch missed: the record says disengaged, and so does the summary."""
    session = _protocol_session(tmp_path)
    picks = iter(["A", "B"] * 10)
    _answer_everything(session, lambda trial: _catch_misser(trial, picks))
    flags = ledger.read(tmp_path / "v.jsonl")[0]["context"]["block_flags"]
    assert (flags, session.view["summary"].endswith("disengaged")) == ([listen_protocol.DISENGAGED], True)


def test_a_cant_tell_is_taken_and_kept_out_of_the_score(tmp_path):
    """ "can't tell" is a choice the server takes; the record counts it apart from the decided trials."""
    session = _session(tmp_path)
    _answer_everything(session, lambda trial: ledger.CANT_TELL if trial["key"] == "A" else "B")
    record = ledger.read(tmp_path / "verdicts.jsonl")[0]
    assert (record["answer"]["cant_tell"] + record["n_trials"], record["n_correct"]) == (4, record["n_trials"])


# ----------------------------------------------------------------------------- a threshold run through the server


def _threshold_session(tmp_path):
    rng = np.random.default_rng(6)
    mode = ThresholdMode(15, rng, -23.0)
    protocol = listen_protocol.Protocol(mode, rng, familiar=5, catches=2, catch_labels=(CATCH_TONE,))
    base = {"id": CALIBRATION_ID, "date": "2026-10-12", "round": "session1", "listener": "user", "context": {"tool": "listen_ab"}}
    base["playback"] = {"device": "hp", "volume": "40", "connection": None, "eq": None, "anc": None, "surround": None}
    sink = listen_ab.CalibrationSink(base, tmp_path / "cal.jsonl")
    return listen_ab.Session(protocol, {}, sink, PORT, {"loops": {"tone": [0.0, 1.0]}, "autoloop": "tone"})


def test_a_threshold_run_serves_its_own_tones_and_loops_them(tmp_path):
    """Three intervals of a second, made by the mode, looped by the page; practice first."""
    session = _threshold_session(tmp_path)
    state = _state(session)
    assert [button["name"] for button in state["buttons"]] == ["A", "B", "C"]
    assert (state["autoloop"], state["loops"], state["phase"]) == ("tone", {"tone": [0.0, 1.0]}, "practice")
    assert _get(session, state["buttons"][0]["url"])[2].startswith(b"RIFF")


def test_a_threshold_run_ends_in_the_calibration_file(tmp_path):
    """Every answer right: one valid calibration record with the playback, the map, the level and the protocol's context."""
    session = _threshold_session(tmp_path)
    _answer_everything(session, lambda trial: trial["key"])
    record = playback_calibration.read(tmp_path / "cal.jsonl")[0]
    assert (record["id"], record["playback"]["device"], sorted(record["sl_map"])) == (
        CALIBRATION_ID,
        "hp",
        ["1000", "12500", "2000", "4000", "8000"],
    )
    context = record["context"]
    assert (context["catch"]["caught"], context["tool"], "tone" in context, "power" in context) == (2, "listen_ab", True, False)
    assert session.view["summary"].startswith("thresholds: 1 kHz")


def test_a_refused_calibration_write_keeps_the_answers_in_a_sidecar(tmp_path):
    """An id already in the calibration file: the run closes with a 500 naming the calibration file and the sidecar."""
    _answer_everything(_threshold_session(tmp_path), lambda trial: trial["key"])
    status, _type, body = _answer_everything(_threshold_session(tmp_path), lambda trial: trial["key"])
    sidecar = tmp_path / f"cal.jsonl.unsaved-{CALIBRATION_ID}.json"
    assert (status, b"NOT written to the calibration file" in body, sidecar.exists()) == (500, True, True)


def test_power_note_speaks_for_abx_only():
    """An abx record's power at a 75% hit rate is named; any other record, or none, adds nothing."""
    abx = {"question": {"type": "abx"}, "n_trials": 24, "context": {"power": {"at_hit_rate": {"0.75": 0.7662}}}}
    assert listen_ab.power_note([abx]) == "power 0.77 at a true 75% hit rate over 24 decided trials"
    notes = [listen_ab.power_note(records) for records in ([{"question": {"type": "pair"}}], [{"id": "calibration"}], [])]
    assert notes == ["", "", ""]
