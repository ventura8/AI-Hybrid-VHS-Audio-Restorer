"""The listening page's HTTP handler, its server loop and its command line, without opening a socket."""

import argparse
import http.client
import io
import json

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

from scripts import listen_ab
from scripts.restoration_quality import ledger, listen_cuts
from scripts.restoration_quality.listen_modes import AbxMode

PORT = 8765
HOST = f"127.0.0.1:{PORT}"
RATE = 44100


def _session(tmp_path):
    """A four-trial ABX session over two short noise cuts; nothing in these tests finishes it."""
    noise = (0.1 * np.random.default_rng(1).standard_normal((RATE // 4, 2))).astype(np.float32)
    return listen_ab.Session(
        AbxMode(["a", "b"], 4, np.random.default_rng(2)), {"a": noise, "b": noise * 0.5}, ({}, tmp_path / "v.jsonl"), PORT
    )


def _handler(session, method, path, body=b""):
    """A handler instance over in-memory streams, as the server would build it for one request."""
    handler_class = listen_ab.handler_for(session)
    handler = handler_class.__new__(handler_class)
    headers = http.client.HTTPMessage()
    headers["Host"] = HOST
    headers["Content-Type"] = "application/json"
    headers["Content-Length"] = str(len(body))
    handler.headers, handler.path, handler.command = headers, path, method
    handler.rfile, handler.wfile = io.BytesIO(body), io.BytesIO()
    handler.request_version, handler.requestline, handler.client_address = "HTTP/1.1", f"{method} {path} HTTP/1.1", ("127.0.0.1", 0)
    return handler


def test_the_handler_carries_get_and_post(tmp_path):
    """GET and POST go through the session and come back as complete HTTP responses."""
    session = _session(tmp_path)
    getter = _handler(session, "GET", "/api/state")
    getter.do_GET()
    poster = _handler(session, "POST", "/api/answer", json.dumps({"trial": 0, "choice": "A"}).encode("utf-8"))
    poster.do_POST()
    response = getter.wfile.getvalue()
    assert response.startswith(b"HTTP/1.0 200")
    assert b"Cache-Control: no-store" in response
    assert b'"index": 1' in poster.wfile.getvalue()


class _FakeServer:
    """Stands in for ThreadingHTTPServer: records the address, never opens a socket."""

    instances = []

    def __init__(self, address, handler):
        self.address, self.handler, self.calls = address, handler, []
        _FakeServer.instances.append(self)

    def serve_forever(self):
        """Returns at once, as a server that was shut down."""
        self.calls.append("serve")

    def shutdown(self):
        """Records the shutdown."""
        self.calls.append("shutdown")

    def server_close(self):
        """Records the close."""
        self.calls.append("close")


def test_serve_binds_loopback_and_stops_when_the_block_closes(tmp_path, monkeypatch):
    """The server binds 127.0.0.1 only, opens the browser on it and is shut down at the end."""
    opened = []
    monkeypatch.setattr(listen_ab, "ThreadingHTTPServer", _FakeServer)
    monkeypatch.setattr(listen_ab.webbrowser, "open", opened.append)
    session = _session(tmp_path)
    session.view["summary"] = "done"
    session.closed.set()
    assert listen_ab.serve(session, 9000) == "done"
    assert _FakeServer.instances[-1].address == ("127.0.0.1", 9000)
    assert opened == ["http://127.0.0.1:9000/"]
    assert _FakeServer.instances[-1].calls[-2:] == ["shutdown", "close"]


def test_serve_stops_cleanly_on_ctrl_c(tmp_path, monkeypatch, capsys):
    """Ctrl+C before the block ends shuts the server down and says nothing was written."""
    monkeypatch.setattr(listen_ab, "ThreadingHTTPServer", _FakeServer)
    session = _session(tmp_path)

    def interrupted(_timeout):
        raise KeyboardInterrupt

    monkeypatch.setattr(session.closed, "wait", interrupted)
    assert listen_ab.serve(session, 9001, open_browser=False) == ""
    assert "not written to the ledger" in capsys.readouterr().out


def test_stimulus_and_tape_arguments(tmp_path):
    """LABEL=PATH needs an existing path and a label other than 'same'; tapes are slugs."""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    assert listen_ab.stimulus_arg(f"B_air_1dB={wav}") == ("B_air_1dB", wav.resolve())
    for bad in (f"same={wav}", "no-equals", f"a={tmp_path / 'missing.wav'}"):
        with pytest.raises(argparse.ArgumentTypeError):
            listen_ab.stimulus_arg(bad)
    with pytest.raises(argparse.ArgumentTypeError):
        listen_ab.tape_arg("a/b")


def _args(tmp_path, *extra, mode="abx", count=2):
    for label in "abc"[:count]:
        sf.write(str(tmp_path / f"{label}.wav"), np.zeros(RATE, dtype=np.float32), RATE)
    stims = [item for label in "abc"[:count] for item in ("--stim", f"{label}={tmp_path / f'{label}.wav'}")]
    return listen_ab.parse_args([mode, "--tape", "vaccin", *stims, "--ledger", str(tmp_path / "v.jsonl"), *extra])


@pytest.mark.parametrize(
    "mode, count, extra, fragment",
    [
        ("abx", 3, [], "abx takes 2 to 2"),
        ("blend", 2, ["--anchor", "PLACEHOLDER"], "--anchor belongs to pair mode"),
        ("abx", 2, ["--seconds", "5"], "--seconds must lie in 8-12"),
        ("abx", 2, ["--start", "-1"], "--start must be >= 0"),
        ("abx", 2, ["--trials", "8"], "--trials must be at least 16 for abx"),
        ("abx", 2, ["--trials", "-3"], "--trials must be at least 16 for abx"),
        ("blend", 2, ["--trials", "0"], "--trials must be at least 1 for blend"),
        ("pair", 2, ["--repeats", "0"], "--repeats must be at least 1"),
    ],
)
def test_argument_problems(tmp_path, mode, count, extra, fragment):
    """Each misuse of the command line is named."""
    extra = [f"z={tmp_path / 'a.wav'}" if item == "PLACEHOLDER" else item for item in extra]
    assert fragment in "; ".join(listen_ab.argument_problems(_args(tmp_path, *extra, mode=mode, count=count)))


def test_duplicate_labels_are_a_problem(tmp_path):
    """Two stimuli with one label could not be told apart in the record."""
    args = _args(tmp_path, "--anchor", f"a={tmp_path / 'b.wav'}", mode="pair")
    assert "stimulus labels must be distinct" in listen_ab.argument_problems(args)


def test_build_mode_follows_the_mode_and_keeps_the_question(tmp_path):
    """abx and blend take their default trial counts; pair takes every pair `--repeats` times plus one replicate."""
    rng = np.random.default_rng(5)
    abx = listen_ab.build_mode(_args(tmp_path, "--question", "is X A?"), ["a", "b"], rng)
    blend = listen_ab.build_mode(_args(tmp_path, mode="blend"), ["a", "b"], rng)
    pair = listen_ab.build_mode(_args(tmp_path, "--repeats", "3", mode="pair", count=3), ["a", "b", "c"], rng)
    assert (abx.total, abx.text, blend.total, pair.total) == (24, "is X A?", 30, 10)


def test_the_pair_replicate_takes_the_anchor(tmp_path):
    """With an anchor, the replicated pair is one the anchor plays in; by default every pair is played once."""
    args = _args(tmp_path, "--anchor", f"z={tmp_path / 'c.wav'}", mode="pair", count=3)
    mode = listen_ab.build_mode(args, ["a", "b", "c", "z"], np.random.default_rng(6))
    assert mode.total == 7
    assert "z" in mode.order[-1]


def _voice_file(path, seconds=12.0):
    t = np.arange(int(seconds * RATE)) / RATE
    audio = (0.1 * np.sin(2 * np.pi * 220.0 * t) * ((t % 1.0) < 0.5) + 1e-3 * np.sin(2 * np.pi * 5000.0 * t)).astype(np.float32)
    sf.write(str(path), audio, RATE, subtype="FLOAT")
    return path


def _soundfile_mono(path, _work_dir, rate=listen_cuts.PICK_RATE):
    """`listen_cuts.decode_mono` without ffmpeg (CI runners have none on PATH): soundfile, downmix, polyphase resample."""
    audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return scipy.signal.resample_poly(audio.mean(axis=1), rate, source_rate).astype(np.float32)


def _answering_serve(session, port, open_browser=True):
    """The stand-in server: answers every trial correctly and hands back the block's summary."""
    assert port == 9100
    assert not open_browser
    for index in range(session.mode.total):
        body = json.dumps({"trial": index, "choice": session.view["trial"]["key"]}).encode("utf-8")
        session.handle("POST", "/api/answer", {"Host": f"127.0.0.1:{port}", "Content-Type": "application/json"}, body)
    return session.view["summary"]


def _recorded_block(tmp_path, monkeypatch):
    """`main` end to end on two voice files with the stand-in server: its exit code, the ledger record and the first file."""
    first, second = _voice_file(tmp_path / "a.wav"), _voice_file(tmp_path / "b.wav")
    monkeypatch.setattr(listen_cuts, "decode_mono", _soundfile_mono)
    monkeypatch.setattr(listen_ab, "serve", _answering_serve)
    argv = ["abx", "--tape", "vaccin", "--stim", f"a={first}", "--stim", f"b={second}", "--seconds", "8", "--trials", "16"]
    code = listen_ab.main([*argv, "--port", "9100", "--no-browser", "--ledger", str(tmp_path / "v.jsonl"), "--seed", "1"])
    return code, ledger.read(tmp_path / "v.jsonl")[0], first


def test_main_cuts_serves_and_records_a_block(tmp_path, monkeypatch, capsys):
    """End to end without a socket: the stand-in server answers every trial; the ledger gets a valid record with hashes."""
    code, record, first = _recorded_block(tmp_path, monkeypatch)
    assert code == 0
    assert "16/16 right" in capsys.readouterr().out
    assert record["files"][0]["sha256"] == ledger.sha256_of(first)
    assert set(record["context"]["cut_sha256"]) == {"a", "b"}


def test_main_records_the_cut_window_and_no_held_gain(tmp_path, monkeypatch):
    """The recorded window spans the asked 8 s; two equal-level voices need no gain held at the clamp."""
    _code, record, _first = _recorded_block(tmp_path, monkeypatch)
    assert record["windows"][0][1] - record["windows"][0][0] == pytest.approx(8.0)
    assert record["context"]["gain_clamped"] == []


def test_main_refuses_bad_arguments(tmp_path):
    """A misuse stops before anything is cut or served."""
    first = _voice_file(tmp_path / "a.wav")
    with pytest.raises(SystemExit, match="abx takes 2 to 2"):
        listen_ab.main(["abx", "--tape", "vaccin", "--stim", f"a={first}", "--ledger", str(tmp_path / "v.jsonl")])


def _two_voices(tmp_path, *extra):
    first, second = _voice_file(tmp_path / "a.wav"), _voice_file(tmp_path / "b.wav")
    return ["abx", "--tape", "vaccin", "--stim", f"a={first}", "--stim", f"b={second}", "--ledger", str(tmp_path / "v.jsonl"), *extra]


def test_main_refuses_a_cut_past_the_end_of_the_files(tmp_path, monkeypatch):
    """A start past the end gives silent cuts: the block stops before it is served."""
    monkeypatch.setattr(listen_ab, "serve", lambda *_args, **_kwargs: pytest.fail("a silent block was served"))
    argv = _two_voices(tmp_path, "--start", "100", "--no-browser")
    with pytest.raises(SystemExit, match=r"silent at 100\.0 s .*: a, b"):
        listen_ab.main(argv)


def test_main_refuses_a_label_the_ledger_gives_another_file(tmp_path, monkeypatch):
    """Label 'a' already names another file on this tape: refused before anything is decoded."""
    older = {"id": "older", "date": "2026-10-09", "round": "1", "tape": "vaccin", "windows": [], "context": {}}
    older.update(files=[ledger.file_entry("a", "elsewhere.wav"), ledger.file_entry("b", "b2.wav")], n_trials=None, n_correct=None)
    older.update(question={"type": "preference", "stimuli": ["a", "b"]}, answer={"order": [["a"], ["b"]]}, confidence=None)
    ledger.append({**older, "playback": {"device": None, "volume": None}}, tmp_path / "v.jsonl")
    monkeypatch.setattr(listen_ab.cuts_mod, "prepare_cuts", lambda *_args: pytest.fail("cut before the label check"))
    argv = _two_voices(tmp_path, "--start", "1")
    with pytest.raises(SystemExit, match="give it a label of its own"):
        listen_ab.main(argv)


def test_clamped_note_names_the_held_gains():
    """No clamped gain, no note; otherwise the labels and the bound."""
    assert listen_ab.clamped_note([]) == ""
    assert listen_ab.clamped_note(["b", "c"]) == " | gain held at +-20 dB for b, c"
