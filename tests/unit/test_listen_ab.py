"""The listening page's server logic: routes, one-time cut names, the request checks and the ledger write at the end."""

import json

import numpy as np
import pytest

from scripts import listen_ab
from scripts.restoration_quality import ledger
from scripts.restoration_quality.listen_modes import AbxMode, BlendMode, PairMode

PORT = 8765
HOST = f"127.0.0.1:{PORT}"
RATE = 44100
JSON_HEADERS = {"Host": HOST, "Content-Type": "application/json"}


def _cuts():
    rng = np.random.default_rng(1)
    noise = (0.1 * rng.standard_normal((RATE, 2))).astype(np.float32)
    return {"a": noise, "b": noise * 0.5}


def _base(labels=("a", "b")):
    return {
        "id": "2026-10-09-test",
        "date": "2026-10-09",
        "round": "session0",
        "tape": "vaccin",
        "windows": [[1.0, 11.0]],
        "files": [ledger.file_entry(label, f"{label}.wav") for label in labels],
        "playback": {"device": None, "volume": None},
        "context": {},
    }


def _session(tmp_path, mode=None, path=None):
    mode = mode or AbxMode(["a", "b"], 4, np.random.default_rng(2))
    return listen_ab.Session(mode, _cuts(), (_base(), path or tmp_path / "verdicts.jsonl"), PORT)


def _get(session, path, host=HOST):
    return session.handle("GET", path, {"Host": host}, b"")


def _post(session, payload, headers=None):
    return session.handle("POST", "/api/answer", headers or JSON_HEADERS, json.dumps(payload).encode("utf-8"))


def _answer_everything(session, pick):
    """Answers every trial through the POST route (one pass, so a refused answer cannot loop); returns the last response."""
    response = None
    for index in range(session.mode.total):
        response = _post(session, {"trial": index, "choice": pick(session.view["trial"])})
    return response


def test_the_page_and_the_state(tmp_path):
    """The page is HTML; the state names the trial, its three play buttons and the two choices."""
    session = _session(tmp_path)
    status, content_type, body = _get(session, "/")
    state = json.loads(_get(session, "/api/state")[2])
    assert status == 200 and content_type.startswith("text/html") and b"<title>Listening test</title>" in body
    assert [button["name"] for button in state["buttons"]] == ["A", "B", "X"] and state["choices"] == ["A", "B"]
    assert state["index"] == 0 and state["total"] == 4 and not state["done"]


def test_cuts_are_served_under_one_time_names(tmp_path):
    """A play button's URL serves a WAV; an unknown name, another path or another method is not found."""
    session = _session(tmp_path)
    url = json.loads(_get(session, "/api/state")[2])["buttons"][0]["url"]
    status, content_type, body = _get(session, url)
    assert status == 200 and content_type == "audio/wav" and body.startswith(b"RIFF")
    assert _get(session, "/audio/0123.wav")[0] == 404 and _get(session, "/elsewhere")[0] == 404
    assert session.handle("PUT", "/", {"Host": HOST}, b"")[0] == 404


def test_another_host_is_refused(tmp_path):
    """DNS rebinding: a request naming any other host gets nothing."""
    session = _session(tmp_path)
    assert _get(session, "/", host="evil.example:8765")[0] == 403
    assert _get(session, "/api/state", host=f"localhost:{PORT}")[0] == 200


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": HOST, "Content-Type": "text/plain"},
        {"Host": HOST},
        {**JSON_HEADERS, "Origin": "https://evil.example"},
        {**JSON_HEADERS, "Origin": "null"},
    ],
)
def test_an_answer_from_another_site_is_refused(tmp_path, headers):
    """A cross-site form or no-cors post (form content type) or a foreign Origin is refused: no answer is taken."""
    session = _session(tmp_path)
    status, _type, body = _post(session, {"trial": 0, "choice": "A"}, headers)
    assert status == 403 and b"application/json" in body and not session.mode.answers


@pytest.mark.parametrize("origin", [None, f"http://{HOST}", f"http://localhost:{PORT}"])
def test_an_answer_from_this_page_is_taken(tmp_path, origin):
    """JSON from this page's origin, or with no Origin header at all, is taken; a charset parameter does not matter."""
    session = _session(tmp_path)
    headers = {"Host": HOST, "Content-Type": "application/json; charset=utf-8"}
    status, _type, _body = _post(session, {"trial": 0, "choice": "A"}, headers if origin is None else {**headers, "Origin": origin})
    assert status == 200 and len(session.mode.answers) == 1


def test_an_abx_block_ends_in_the_ledger(tmp_path):
    """Four right answers: the last response is the summary, the ledger holds one valid abx record."""
    session = _session(tmp_path)
    status, _type, body = _answer_everything(session, lambda trial: trial["key"])
    records = ledger.read(tmp_path / "verdicts.jsonl")
    assert status == 200 and json.loads(body) == {"done": True, "summary": session.mode.summary()}
    assert [record["id"] for record in records] == ["2026-10-09-test-1"] and records[0]["n_correct"] == 4
    assert records[0]["question"] == {"type": "abx", "stimuli": ["a", "b"], "blind": True, "text": ""}


def test_after_reply_closes_the_block_only_when_finished(tmp_path):
    """The server stops once the last answer's response has gone out, not before."""
    session = _session(tmp_path)
    session.after_reply()
    assert not session.closed.is_set()
    _answer_everything(session, lambda trial: "A")
    session.after_reply()
    assert session.closed.is_set()


def test_a_pair_block_writes_one_record_per_pair_with_its_replicate(tmp_path):
    """Pair mode through the server: three pairs and one replicate, every pair's counts in the ledger, the replicate marked."""
    mode = PairMode(["a", "b", "c"], 1, np.random.default_rng(3))
    session = listen_ab.Session(mode, {**_cuts(), "c": _cuts()["a"] * 0.2}, (_base(("a", "b", "c")), tmp_path / "v.jsonl"), PORT)
    _answer_everything(session, lambda _trial: ledger.SAME)
    records = ledger.read(tmp_path / "v.jsonl")
    assert [record["id"] for record in records] == ["2026-10-09-test-1", "2026-10-09-test-2", "2026-10-09-test-3"]
    assert sorted(record["n_trials"] for record in records) == [1, 1, 2]
    assert [trial[3] for record in records for trial in record["answer"]["trials"]].count(True) == 1


def test_a_blend_trial_plays_the_mix(tmp_path):
    """The blend's play button is the weighted sum of the two cuts."""
    session = _session(tmp_path, BlendMode(["a", "b"], 2, np.random.default_rng(4)))
    mixed = session.mix({"a": 0.25, "b": 0.75})
    assert np.allclose(mixed, 0.25 * _cuts()["a"] + 0.75 * _cuts()["b"])
    assert [button["name"] for button in session.state()["buttons"]] == ["R", "A", "B"]


def _conflicting_ledger(path):
    """A ledger where stimulus 'a' on vaccin is another file than the session's a.wav."""
    record = {**_base(), "id": "older", "files": [ledger.file_entry("a", "elsewhere.wav"), ledger.file_entry("b", "b.wav")]}
    record.update(question={"type": "preference", "stimuli": ["a", "b"]}, answer={"order": [["a"], ["b"]]})
    ledger.append({**record, "n_trials": None, "n_correct": None, "confidence": None}, path)
    return path


def test_a_refused_ledger_write_keeps_the_answers_in_a_sidecar(tmp_path):
    """The ledger refuses the block (label 'a' is another file there): 500 naming the sidecar, the block closes, nothing half-written."""
    session = _session(tmp_path, path=_conflicting_ledger(tmp_path / "v.jsonl"))
    status, _type, body = _answer_everything(session, lambda trial: trial["key"])
    sidecar = tmp_path / "v.jsonl.unsaved-2026-10-09-test.json"
    session.after_reply()
    assert status == 500 and b"NOT written to the ledger" in body and str(sidecar).encode("utf-8") in body
    assert [record["id"] for record in json.loads(sidecar.read_text(encoding="utf-8"))] == ["2026-10-09-test-1"]
    assert session.closed.is_set() and [record["id"] for record in ledger.read(tmp_path / "v.jsonl")] == ["older"]


def test_when_the_sidecar_fails_too_the_summary_carries_the_records(tmp_path, monkeypatch):
    """A disk error on the ledger and on the sidecar: the answers survive in the summary the terminal prints."""

    def broken(_records, _path):
        raise OSError("disk full")

    monkeypatch.setattr(listen_ab.ledger, "append_many", broken)
    session = _session(tmp_path, path=tmp_path / "missing" / "v.jsonl")
    status, _type, _body = _answer_everything(session, lambda trial: "A")
    assert status == 500 and "disk full" in session.view["summary"] and "failed too" in session.view["summary"]
    assert '"id": "2026-10-09-test-1"' in session.view["summary"] and json.loads(_get(session, "/api/state")[2])["done"]


def test_answers_that_cannot_be_taken(tmp_path):
    """Malformed JSON and an unknown choice are bad requests; a stale or repeated trial is a conflict."""
    session = _session(tmp_path)
    assert session.handle("POST", "/api/answer", JSON_HEADERS, b"{not json")[0] == 400
    assert _post(session, {"trial": 0, "choice": "C"})[0] == 400
    assert _post(session, {"trial": 3, "choice": "A"})[0] == 409
    assert _post(session, {"trial": 0, "choice": "A"})[0] == 200 and _post(session, {"trial": 0, "choice": "A"})[0] == 409


def test_parse_answer_names_what_is_wrong():
    """Missing keys and wrong types are refused with a reason."""
    assert listen_ab.parse_answer(b'{"trial": 2, "choice": "B"}') == (2, "B")
    with pytest.raises(ValueError, match="expected"):
        listen_ab.parse_answer(b'{"trial": 2}')
    with pytest.raises(ValueError, match="must be a number"):
        listen_ab.parse_answer(b'{"trial": "2", "choice": "B"}')


def test_body_length_and_media_type_are_read_defensively():
    """No header or a broken one reads as empty; a huge length is capped; the media type drops its parameters."""
    assert listen_ab.body_length({}) == 0 and listen_ab.body_length({"Content-Length": "abc"}) == 0
    assert listen_ab.body_length({"Content-Length": "999999"}) == listen_ab.MAX_BODY
    assert listen_ab.media_type({"Content-Type": "Application/JSON ; charset=utf-8"}) == "application/json"
    assert listen_ab.media_type({}) == ""
