"""The listening page's server logic: routes, one-time cut names, the request checks and the ledger write at the end."""

import json
from types import SimpleNamespace

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


def _state(session):
    """The page's state as the browser reads it."""
    return json.loads(_get(session, "/api/state")[2])


def _answer_everything(session, pick):
    """Answers every trial through the POST route (one pass, so a refused answer cannot loop); returns the last response."""
    response = None
    for index in range(session.mode.total):
        response = _post(session, {"trial": index, "choice": pick(session.view["trial"])})
    return response


# ----------------------------------------------------------------------------- the page, its state and the cuts


def test_the_page_is_html(tmp_path):
    """The page is HTML under its own title."""
    status, content_type, body = _get(_session(tmp_path), "/")
    assert status == 200
    assert content_type.startswith("text/html")
    assert b"<title>Listening test</title>" in body


def test_the_state_names_the_three_play_buttons_and_the_two_choices(tmp_path):
    """The state names the trial's three play buttons and the two choices."""
    state = _state(_session(tmp_path))
    assert [button["name"] for button in state["buttons"]] == ["A", "B", "X"]
    assert state["choices"] == ["A", "B"]


def test_the_state_names_the_trial(tmp_path):
    """A fresh block stands at the first of its four trials, not done."""
    state = _state(_session(tmp_path))
    assert state["index"] == 0
    assert state["total"] == 4
    assert not state["done"]


def test_cuts_are_served_under_one_time_names(tmp_path):
    """A play button's URL serves a WAV."""
    session = _session(tmp_path)
    status, content_type, body = _get(session, _state(session)["buttons"][0]["url"])
    assert status == 200
    assert content_type == "audio/wav"
    assert body.startswith(b"RIFF")


@pytest.mark.parametrize(("method", "path"), [("GET", "/audio/0123.wav"), ("GET", "/elsewhere"), ("PUT", "/")])
def test_an_unknown_name_another_path_or_another_method_is_not_found(tmp_path, method, path):
    """An unknown cut name, another path or another method is not found."""
    assert _session(tmp_path).handle(method, path, {"Host": HOST}, b"")[0] == 404


# ----------------------------------------------------------------------------- who may ask and who may answer


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
    assert status == 403
    assert b"application/json" in body
    assert not session.mode.answers


@pytest.mark.parametrize("origin", [None, f"http://{HOST}", f"http://localhost:{PORT}"])
def test_an_answer_from_this_page_is_taken(tmp_path, origin):
    """JSON from this page's origin, or with no Origin header at all, is taken; a charset parameter does not matter."""
    session = _session(tmp_path)
    headers = {"Host": HOST, "Content-Type": "application/json; charset=utf-8"}
    status, _type, _body = _post(session, {"trial": 0, "choice": "A"}, headers if origin is None else {**headers, "Origin": origin})
    assert status == 200
    assert len(session.mode.answers) == 1


# ----------------------------------------------------------------------------- a block through the server: abx, pair and blend


def _abx_block(tmp_path):
    """Four right answers through the server: the session, the last response and the ledger's records."""
    session = _session(tmp_path)
    status, _type, body = _answer_everything(session, lambda trial: trial["key"])
    return SimpleNamespace(session=session, status=status, body=body, records=ledger.read(tmp_path / "verdicts.jsonl"))


def test_an_abx_block_ends_with_its_summary(tmp_path):
    """Four right answers: the last response is the summary."""
    block = _abx_block(tmp_path)
    assert block.status == 200
    assert json.loads(block.body) == {"done": True, "summary": block.session.mode.summary()}


def test_an_abx_block_ends_in_the_ledger(tmp_path):
    """Four right answers: the ledger holds one valid abx record with all four right."""
    records = _abx_block(tmp_path).records
    assert [record["id"] for record in records] == ["2026-10-09-test-1"]
    assert records[0]["n_correct"] == 4
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


# ----------------------------------------------------------------------------- a ledger write that fails: the sidecar


def _conflicting_ledger(path):
    """A ledger where stimulus 'a' on vaccin is another file than the session's a.wav."""
    record = {**_base(), "id": "older", "files": [ledger.file_entry("a", "elsewhere.wav"), ledger.file_entry("b", "b.wav")]}
    record.update(question={"type": "preference", "stimuli": ["a", "b"]}, answer={"order": [["a"], ["b"]]})
    ledger.append({**record, "n_trials": None, "n_correct": None, "confidence": None}, path)
    return path


def _refused_block(tmp_path):
    """A block the ledger refuses (label 'a' is another file there), answered to the end and its last reply sent."""
    session = _session(tmp_path, path=_conflicting_ledger(tmp_path / "v.jsonl"))
    status, _type, body = _answer_everything(session, lambda trial: trial["key"])
    session.after_reply()
    return SimpleNamespace(session=session, status=status, body=body, sidecar=tmp_path / "v.jsonl.unsaved-2026-10-09-test.json")


def test_a_refused_ledger_write_answers_500_naming_the_sidecar(tmp_path):
    """The ledger refuses the block: the last response is a 500 that says so and names the sidecar."""
    block = _refused_block(tmp_path)
    assert block.status == 500
    assert b"NOT written to the ledger" in block.body
    assert str(block.sidecar).encode("utf-8") in block.body


def test_a_refused_ledger_write_keeps_the_answers_in_a_sidecar(tmp_path):
    """The ledger refuses the block: the answers are in the sidecar, the block closes, nothing half-written."""
    block = _refused_block(tmp_path)
    assert [record["id"] for record in json.loads(block.sidecar.read_text(encoding="utf-8"))] == ["2026-10-09-test-1"]
    assert block.session.closed.is_set()
    assert [record["id"] for record in ledger.read(tmp_path / "v.jsonl")] == ["older"]


def _sidecar_failure(tmp_path, monkeypatch):
    """A block whose ledger write and sidecar both hit a disk error: the session and the last response's status."""

    def broken(_records, _path):
        raise OSError("disk full")

    monkeypatch.setattr(listen_ab.ledger, "append_many", broken)
    session = _session(tmp_path, path=tmp_path / "missing" / "v.jsonl")
    return session, _answer_everything(session, lambda trial: "A")[0]


def test_when_the_sidecar_fails_too_the_response_names_both_failures(tmp_path, monkeypatch):
    """A disk error on the ledger and on the sidecar: a 500 whose summary names the error and the failed sidecar."""
    session, status = _sidecar_failure(tmp_path, monkeypatch)
    assert status == 500
    assert "disk full" in session.view["summary"]
    assert "failed too" in session.view["summary"]


def test_when_the_sidecar_fails_too_the_summary_carries_the_records(tmp_path, monkeypatch):
    """A disk error on the ledger and on the sidecar: the answers survive in the summary the terminal prints."""
    session, _status = _sidecar_failure(tmp_path, monkeypatch)
    assert '"id": "2026-10-09-test-1"' in session.view["summary"]
    assert _state(session)["done"]


# ----------------------------------------------------------------------------- answers the server cannot take, and reading the request


def test_answers_that_cannot_be_taken(tmp_path):
    """Malformed JSON and an unknown choice are bad requests, and neither moves the block on: trial 0 is still open."""
    session = _session(tmp_path)
    assert session.handle("POST", "/api/answer", JSON_HEADERS, b"{not json")[0] == 400
    assert _post(session, {"trial": 0, "choice": "C"})[0] == 400
    assert _post(session, {"trial": 0, "choice": "A"})[0] == 200


def test_a_stale_or_repeated_trial_is_a_conflict(tmp_path):
    """An answer to a trial not reached yet, or to one already answered, is a conflict."""
    session = _session(tmp_path)
    assert _post(session, {"trial": 3, "choice": "A"})[0] == 409
    assert _post(session, {"trial": 0, "choice": "A"})[0] == 200
    assert _post(session, {"trial": 0, "choice": "A"})[0] == 409


def test_parse_answer_names_what_is_wrong():
    """Missing keys and wrong types are refused with a reason."""
    assert listen_ab.parse_answer(b'{"trial": 2, "choice": "B"}') == (2, "B")
    with pytest.raises(ValueError, match="expected"):
        listen_ab.parse_answer(b'{"trial": 2}')
    with pytest.raises(ValueError, match="must be a number"):
        listen_ab.parse_answer(b'{"trial": "2", "choice": "B"}')


@pytest.mark.parametrize(
    ("headers", "length"),
    [({}, 0), ({"Content-Length": "abc"}, 0), ({"Content-Length": "999999"}, listen_ab.MAX_BODY)],
    ids=["no-header", "broken", "huge"],
)
def test_body_length_is_read_defensively(headers, length):
    """No header or a broken one reads as empty; a huge length is capped."""
    assert listen_ab.body_length(headers) == length


def test_the_media_type_is_read_defensively():
    """The media type drops its parameters and its case; no header reads as none."""
    assert listen_ab.media_type({"Content-Type": "Application/JSON ; charset=utf-8"}) == "application/json"
    assert listen_ab.media_type({}) == ""


def test_local_url_builds_the_loopback_address_and_origin():
    """The printed page address ends in a slash; an Origin is scheme and host only."""
    assert listen_ab.local_url(HOST, "/") == "http://127.0.0.1:8765/"
    assert listen_ab.local_url(f"localhost:{PORT}") == "http://localhost:8765"
