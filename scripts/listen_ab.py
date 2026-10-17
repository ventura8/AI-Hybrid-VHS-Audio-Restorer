"""A/B listening on this machine only: ABX, pairwise preference with "same", a blend threshold and a tone-threshold run.

usage (from the repository root; the page binds 127.0.0.1, family tapes never leave the machine):

    python -m scripts.listen_ab abx --tape vaccin --stim A=<file> --stim B=<file> --device <name> --volume <setting>
    python -m scripts.listen_ab pair --tape vaccin --stim a=<file> --stim b=<file> [--stim c=<file> ...] [--anchor hiss=<file>] ...
    python -m scripts.listen_ab blend --tape vaccin --stim incumbent=<file> --stim candidate=<file> ...
    python -m scripts.listen_ab threshold --listener user --device <name> --volume <setting>

`--device` and `--volume` are required (Session 0 recorded neither, and the masking model's
playback level means nothing without them); `--connection`, `--eq`, `--anc` and `--surround`
record the rest of the playback chain as text (Bluetooth or USB and its format, the EQ preset,
noise cancelling on or off, virtual surround on or off). Files outside the repository and the
temp directory need their root in $AI_RESTORE_DATA_ROOTS (`D:\\Tata` for the family tapes). A
comparison block's records go to the verdict ledger (`assets/quality_calibration/verdicts.jsonl`,
schema in `scripts/restoration_quality/ledger.py`), a threshold run's to the playback
calibration file (`playback_calibration.py`), once the last answer is in, in one write after
every record is validated; Ctrl+C before that writes nothing. When the file refuses them (a
broken line, a full disk), the answers go to `<file>.unsaved-<block id>.json` beside it, the
page and the terminal say where, and the block still closes. A stimulus label that already names
another file on the same tape in the ledger is refused before anything is decoded: the ledger
merges a tape's verdicts by label.

- The questions (`restoration_quality/listen_modes.py`): ABX, 24 trials by default and 16 at
  least (17/24 right is p 0.032, power 0.77 at a true 75% hit rate; 12/16 is p 0.038 at power
  0.63; under 5 trials no score reaches p 0.05); pair with "same", every pair once
  (`--repeats`) in a fresh order plus one replicate pair at the end, with the `--anchor` output
  when there is one (the round-one cathar alpha 2 hissy cut, which keeps sessions comparable);
  blend, the candidate mixed into the incumbent at x in [0, 1] with QUEST-placed trials, 30 by
  default; threshold (`restoration_quality/listen_threshold.py`), tones at 1-12.5 kHz under a
  hard cap, 60 trials in three intervals, whose map feeds `auditory.PLAYBACK_SPL_LOUD_FRAMES` for that listener.
  ABX, blend and threshold take a "can't tell" answer that is kept apart and never scored.
- The protocol (`restoration_quality/listen_protocol.py`): familiarisation trials first (abx
  and blend 2, threshold 5, `--familiar`), answered with feedback and never scored; catch trials
  among the scored ones (one per 8, `--catch`; +6 dB of air, and half the raw tape with
  `--catch-raw`), shaped like the scored trials and scored apart; `context.block_flags` says "no
  signal" for a one-button block and "disengaged" for a missed catch; `context.power` keeps
  each record's power.
- The cuts (`restoration_quality/listen_cuts.py`): 8-12 s (10 s by default), aligned to the
  first stimulus (lag and polarity), each brought to one speech-active level (-23 dBFS,
  `--level-dbfs`), faded; without --start they go where the first two stimuli differ most
  audibly (NMR). A silent cut, or one that would clip at the level, stops the block before it is
  served, the latter naming the highest level every stimulus fits at.
- The page (`restoration_quality/listen_page.py`) decodes the cuts into WebAudio buffers and
  switches A, B and X at the same playhead with an 8 ms crossfade (keys 1, 2, 3; space plays
  and pauses; loops of 1-3 s, or over the quietest 2 s of the reference cut).
- Every trial serves its cuts under fresh random names, so the page cannot give X away. A
  request whose Host is not 127.0.0.1 or localhost on this port is refused (DNS rebinding), and
  an answer is taken only as `Content-Type: application/json` from this page's own origin (or
  with no Origin header). Another site open in the browser can send a form or a no-cors fetch to
  127.0.0.1 without a preflight, but only with a form content type; a JSON post from it needs a
  CORS preflight this server never grants, so it cannot put answers into the ledger.
"""

import argparse
import datetime
import json
import re
import secrets
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from scripts import cli_paths
from scripts.restoration_quality import ledger, listen_cuts, listen_protocol, playback_calibration
from scripts.restoration_quality.listen_modes import AbxMode, BlendMode, PairMode
from scripts.restoration_quality.listen_page import PAGE
from scripts.restoration_quality.listen_threshold import CATCH_TONE, PERIOD_S, ThresholdMode

HOST = "127.0.0.1"
# The family tapes never leave the machine: the page is served on the loopback address only, so it needs no TLS.
LOCAL_SCHEME = "http"
DEFAULT_PORT = 8765
CUT_SECONDS = (8.0, 12.0)
DEFAULT_SECONDS = 10.0
DEFAULT_TRIALS = {"abx": 24, "pair": 0, "blend": 30, "threshold": 60}
MIN_TRIALS = {"abx": 16, "pair": 1, "blend": 1, "threshold": 15}
PAIR_REPEATS = 1
MAX_STIMULI = 6
STIMULUS_COUNTS = {"abx": (2, 2), "pair": (2, MAX_STIMULI), "blend": (2, 2)}
LEVEL_RANGE_DBFS = (-40.0, -14.0)
PLAYBACK_STATE = ("connection", "eq", "anc", "surround")
CATCH_RAW_LABEL = "catch_raw"
DEFAULT_LISTENER = "user"
MAX_BODY = 4096
JSON_TYPE = "application/json"
STIMULUS_RE = re.compile(r"^([A-Za-z0-9_.+-]+)=(.+)$")


def _text(status, message):
    return status, "text/plain; charset=utf-8", message.encode("utf-8")


def _json(payload):
    return HTTPStatus.OK, "application/json", json.dumps(payload).encode("utf-8")


def parse_answer(body):
    """`(trial index, choice)` from the page's JSON body; ValueError says what is wrong."""
    try:
        payload = json.loads(body.decode("utf-8"))
        index, choice = payload["trial"], payload["choice"]
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError('expected {"trial": <number>, "choice": <text>}') from error
    if not isinstance(index, int) or not isinstance(choice, str):
        raise ValueError("trial must be a number and choice a text")
    return index, choice


def media_type(headers):
    """The request's media type without its parameters, lower case ("" when there is no Content-Type)."""
    return (headers.get("Content-Type") or "").split(";")[0].strip().lower()


def save_unsaved(records, path, block_id):
    """Writes a block the file refused to `<file>.unsaved-<block id>.json`; says where, or carries the records itself."""
    path = Path(path)
    sidecar = path.with_name(f"{path.name}.unsaved-{block_id}.json")
    try:
        sidecar.write_text(json.dumps(records, indent=1), encoding="utf-8")
    except OSError as error:
        return f"the sidecar {sidecar} failed too ({error}); the records: {json.dumps(records)}"
    return f"the answers are in {sidecar}"


class LedgerSink:
    """Where a comparison block's records go: the verdict ledger, one record per result with the block's context and power."""

    name = "ledger"
    errors = (ledger.LedgerError, OSError)

    def __init__(self, base, path):
        self.base, self.path = base, path

    def records(self, mode, extra):
        """The ledger records of the finished block, fields in schema order."""
        out = []
        for number, partial in enumerate(mode.results(), start=1):
            context = {**self.base["context"], **extra, "power": listen_protocol.power_of(partial)}
            record = {**self.base, **partial, "id": f"{self.base['id']}-{number}", "context": context}
            out.append({name: record[name] for name in ledger.FIELDS})
        return out

    def write(self, records):
        """Every record in one validated write."""
        ledger.append_many(records, self.path)


class CalibrationSink:
    """Where a threshold run's record goes: the playback calibration file."""

    name = "calibration file"
    errors = (playback_calibration.CalibrationError, OSError)

    def __init__(self, base, path):
        self.base, self.path = base, path

    def records(self, mode, extra):
        """The run's one calibration record, fields in schema order."""
        body = mode.results()[0]
        context = {**self.base["context"], **body.pop("context"), **extra}
        record = {**self.base, **body, "context": context}
        return [{name: record[name] for name in playback_calibration.FIELDS}]

    def write(self, records):
        """The record, validated, appended."""
        playback_calibration.append_many(records, self.path)


def power_note(records):
    """The summary's clause on an ABX block's power ("" for any other block)."""
    first = records[0] if records else {}
    if first.get("question", {}).get("type") != "abx":
        return ""
    rate = first["context"]["power"]["at_hit_rate"]["0.75"]
    return f"power {rate:.2f} at a true 75% hit rate over {first['n_trials']} decided trials"


class Session:
    """One listening block: its protocol and mode, the stimuli, the current trial's one-time audio names, the write at the end.

    `block` is a `listen_protocol.Protocol`, or a bare mode run without practice or catch trials;
    `out` a sink or `(base record, ledger path)`; `page` the page's loops, `{"loops": {name: [start_s,
    end_s]}, "autoloop": name or None}`.
    """

    def __init__(self, block, cuts, out, port, page=None):
        self.protocol = block if isinstance(block, listen_protocol.Protocol) else listen_protocol.Protocol(block, np.random.default_rng(0))
        self.cuts = cuts
        self.sink = out if hasattr(out, "write") else LedgerSink(*out)
        self.settings = {"loops": {}, "autoloop": None, **(page or {}), "hosts": {f"{HOST}:{port}", f"localhost:{port}"}}
        self.view = {"audio": {}, "trial": {}, "summary": "", "failed": False}
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self._next_trial()

    @property
    def mode(self):
        """The block's question."""
        return self.protocol.mode

    def mix(self, weights):
        """The weighted sum of cuts a play button stands for."""
        return sum(weight * self.cuts[label] for label, weight in weights.items()).astype(np.float32)

    def render(self, source):
        """The samples a play button stands for: a clip the mode made itself, or a weighted sum of cuts."""
        return source if isinstance(source, np.ndarray) else self.mix(source)

    def _next_trial(self):
        trial = self.protocol.trial()
        audio, tokens = {}, []
        for name, source in trial["play"]:
            token = secrets.token_hex(8)
            audio[token] = listen_cuts.wav_bytes(self.render(source))
            tokens.append((name, token))
        self.view.update(audio=audio, trial={**trial, "tokens": tokens})

    def state(self):
        """What the page shows now."""
        if self.protocol.finished():
            return {"done": True, "summary": self.view["summary"]}
        trial = self.view["trial"]
        buttons = [{"name": name, "url": f"/audio/{token}.wav"} for name, token in trial["tokens"]]
        practice = self.protocol.order.count(listen_protocol.FAMILIAR)
        return {
            "done": False,
            "index": self.protocol.position,
            "total": len(self.protocol.order) - practice,
            **self.protocol.progress(),
            "prompt": trial["prompt"],
            "buttons": buttons,
            "choices": trial["choices"],
            "loops": self.settings["loops"],
            "autoloop": self.settings["autoloop"],
        }

    def handle(self, method, path, headers, body):
        """`(status, content type, body)` for one request (`headers`: a mapping); the HTTP handler only carries it."""
        with self.lock:
            refusal = self.refusal(method, headers)
            if refusal:
                return _text(HTTPStatus.FORBIDDEN, refusal)
            route = _ROUTES.get((method, path))
            if route is not None:
                return route(self, body)
            return self._audio(path) if method == "GET" else _text(HTTPStatus.NOT_FOUND, "not found")

    def refusal(self, method, headers):
        """Why a request is refused, or "": another Host (DNS rebinding), or an answer not posted as JSON by this page."""
        if headers.get("Host", "") not in self.settings["hosts"]:
            return "this page answers on 127.0.0.1 only"
        if method == "POST" and not self._own_json(headers):
            return "answers are taken as application/json from this page only"
        return ""

    def _own_json(self, headers):
        origin = headers.get("Origin")
        return media_type(headers) == JSON_TYPE and (origin is None or origin in {local_url(host) for host in self.settings["hosts"]})

    def page(self, _body):
        """The listening page."""
        return HTTPStatus.OK, "text/html; charset=utf-8", PAGE.encode("utf-8")

    def state_json(self, _body):
        """The page's state as JSON."""
        return _json(self.state())

    def _audio(self, path):
        data = self.view["audio"].get(path.removeprefix("/audio/").removesuffix(".wav"))
        if not path.startswith("/audio/") or data is None:
            return _text(HTTPStatus.NOT_FOUND, "no such cut")
        return HTTPStatus.OK, "audio/wav", data

    def answer(self, body):
        """Takes one answer; returns the next trial's state, or the summary once the block is written (500 when it is not)."""
        try:
            index, choice = parse_answer(body)
        except ValueError as error:
            return _text(HTTPStatus.BAD_REQUEST, str(error))
        trial = self.view["trial"]
        if self.protocol.finished() or index != self.protocol.position:
            return _text(HTTPStatus.CONFLICT, "that trial is already answered")
        if choice not in trial["choices"]:
            return _text(HTTPStatus.BAD_REQUEST, f"choose one of {', '.join(trial['choices'])}")
        feedback = self.protocol.take(trial, choice)
        self._advance()
        return self._progress(feedback)

    def _progress(self, feedback):
        """The next trial's state (with a practice answer's feedback) or the summary; a 500 carrying it when the write failed."""
        if self.view["failed"]:
            return _text(HTTPStatus.INTERNAL_SERVER_ERROR, self.view["summary"])
        state = self.state()
        return _json({**state, "feedback": feedback} if feedback else state)

    def _advance(self):
        if not self.protocol.finished():
            self._next_trial()
            return
        records = self.records()
        parts = [self.mode.summary(), power_note(records), self.protocol.note()]
        self.view.update(summary=" | ".join(part for part in parts if part), audio={})
        self._write_records(records)

    def _write_records(self, records):
        """Every record of the block in one validated write; on failure the answers go to a sidecar and the summary says where."""
        try:
            self.sink.write(records)
        except self.sink.errors as error:
            self.view["failed"] = True
            where = save_unsaved(records, self.sink.path, self.sink.base["id"])
            self.view["summary"] += f" | NOT written to the {self.sink.name} ({error}); {where}"

    def records(self):
        """The records of the finished block, as its sink writes them."""
        return self.sink.records(self.mode, self.protocol.context())

    def after_reply(self):
        """Called once a response is on the wire: the block closes when its last answer has been sent back."""
        if self.protocol.finished():
            self.closed.set()


_ROUTES = {("GET", "/"): Session.page, ("GET", "/api/state"): Session.state_json, ("POST", "/api/answer"): Session.answer}


def body_length(headers):
    """The request body's length, at most MAX_BODY; 0 when the header is missing or not a number."""
    try:
        length = int(headers.get("Content-Length", "0"))
    except ValueError:
        return 0
    return max(0, min(length, MAX_BODY))


def handler_for(session):
    """The request handler class bound to `session`."""

    class Handler(BaseHTTPRequestHandler):
        """Carries requests to `Session.handle` and its answers back."""

        server_version = "listen_ab"

        def _get(self):
            self._reply(session.handle("GET", self.path, self.headers, b""))

        def _post(self):
            body = self.rfile.read(body_length(self.headers))
            self._reply(session.handle("POST", self.path, self.headers, body))

        do_GET = _get
        do_POST = _post

        def _reply(self, response):
            status, content_type, body = response
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            session.after_reply()

        def log_request(self, code="-", size="-"):
            """Quiet: a line per cut fetched would bury the session's own output."""

    return Handler


def local_url(netloc, path=""):
    """The page's own address, or an Origin it sends: plain HTTP is right here, the server binds 127.0.0.1 only."""
    return urllib.parse.urlunsplit((LOCAL_SCHEME, netloc, path, "", ""))


def serve(session, port, open_browser=True):
    """Serves the page on 127.0.0.1:`port` until the block's last answer is sent back; returns the summary."""
    server = ThreadingHTTPServer((HOST, port), handler_for(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = local_url(f"{HOST}:{port}", "/")
    print(f"listening page: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        # Half-second waits, not one unbounded wait: Ctrl+C reaches the loop between them (Windows).
        closed = False
        while not closed:
            closed = session.closed.wait(0.5)
    except KeyboardInterrupt:
        print("stopped: an unfinished block is not written to the ledger or the calibration file")
    finally:
        server.shutdown()
        server.server_close()
    return session.view["summary"]


def stimulus_arg(value):
    """An argparse type: `LABEL=PATH` with a label of letters, digits, '_', '.', '+' or '-' (not 'same')."""
    match = STIMULUS_RE.match(str(value))
    if match is None or match.group(1) == ledger.SAME:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH with a label other than {ledger.SAME!r}: {value!r}")
    return match.group(1), cli_paths.existing_path_arg(match.group(2))


def tape_arg(value):
    """An argparse type: a tape slug as the ledger stores it."""
    if ledger.NAME_RE.match(str(value)) is None:
        raise argparse.ArgumentTypeError(f"a tape slug is letters, digits, '_', '.', '+' or '-': {value!r}")
    return str(value)


def listener_arg(value):
    """An argparse type: a listener slug as the calibration file stores it."""
    if playback_calibration.NAME_RE.match(str(value)) is None:
        raise argparse.ArgumentTypeError(f"a listener is letters, digits, '_', '.', '+' or '-': {value!r}")
    return str(value)


def parse_args(argv=None):
    """The command line, parsed; stimulus paths are resolved and must exist (`cli_paths`)."""
    parser = argparse.ArgumentParser(description="Blind A/B listening on 127.0.0.1; answers go to the verdict ledger.")
    parser.add_argument("mode", choices=tuple(DEFAULT_TRIALS))
    parser.add_argument("--tape", type=tape_arg, default=None, help="tape slug, as in the ledger (vaccin, tele7abc, ...)")
    parser.add_argument("--stim", action="append", default=[], type=stimulus_arg, metavar="LABEL=PATH")
    parser.add_argument("--anchor", action="append", default=[], type=stimulus_arg, metavar="LABEL=PATH", help="pair: a fixed anchor")
    parser.add_argument("--start", type=float, default=None, help="cut start in seconds (default: where the first two differ most audibly)")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="cut length, 8-12 s")
    parser.add_argument("--trials", type=int, default=None, help="abx: 24 (16 at least), blend: 30, threshold: 60 by default")
    parser.add_argument("--repeats", type=int, default=PAIR_REPEATS, help="pair: plays of each pair before the one replicate")
    parser.add_argument("--round", default="listen_ab", help="the ledger's round (session0, ...)")
    parser.add_argument("--question", default="", help="what the listener is asked, kept in the record")
    parser.add_argument("--device", required=True, help="playback device (headphones model, speakers)")
    parser.add_argument("--volume", required=True, help="playback volume setting")
    parser.add_argument("--connection", default=None, help="how the device is connected (USB 24-bit/48 kHz, Bluetooth codec, ...)")
    parser.add_argument("--eq", default=None, help="EQ state (off, a preset's name, ...)")
    parser.add_argument("--anc", default=None, help="active noise cancelling state (off, on, transparency)")
    parser.add_argument("--surround", default=None, help="virtual surround state (off, on, ...)")
    parser.add_argument("--level-dbfs", type=float, default=listen_cuts.TARGET_ACTIVE_DBFS, help="speech-active level of every stimulus")
    parser.add_argument("--catch", type=int, default=None, help="catch trials (default: one per 8 scored trials, at least one)")
    parser.add_argument("--catch-raw", type=cli_paths.existing_path_arg, default=None, help="the raw tape, for half-raw catch trials")
    parser.add_argument("--familiar", type=int, default=None, help="familiarisation trials (abx and blend 2, threshold 5, pair 0)")
    parser.add_argument("--listener", type=listener_arg, default=DEFAULT_LISTENER, help="threshold: whose calibration this is")
    parser.add_argument("--ledger", type=cli_paths.path_arg, default=ledger.LEDGER_PATH)
    parser.add_argument("--calibration", type=cli_paths.path_arg, default=playback_calibration.CALIBRATION_PATH)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args(argv)


def _below(value, minimum):
    return value is not None and value < minimum


def _blank(value):
    return not str(value).strip()


def _raw_entry(args):
    return [(CATCH_RAW_LABEL, args.catch_raw)] if args.catch_raw else []


def _common_problems(args):
    low, high = LEVEL_RANGE_DBFS
    least = MIN_TRIALS[args.mode]
    checks = [
        (_blank(args.device), "--device must name the playback device"),
        (_blank(args.volume), "--volume must give the volume setting"),
        (not low <= args.level_dbfs <= high, f"--level-dbfs must lie in {low:g}..{high:g}"),
        (_below(args.trials, least), f"--trials must be at least {least} for {args.mode}"),
        (_below(args.catch, 0), "--catch must be >= 0"),
        (_below(args.familiar, 0), "--familiar must be >= 0"),
    ]
    return [message for bad, message in checks if bad]


def _stimulus_problems(args):
    stimuli = args.stim + args.anchor
    labelled = stimuli + _raw_entry(args)
    low, high = STIMULUS_COUNTS[args.mode]
    checks = [
        (args.tape is None, f"{args.mode} needs --tape"),
        (not low <= len(stimuli) <= high, f"{args.mode} takes {low} to {high} stimuli, anchors included"),
        (args.mode != "pair" and bool(args.anchor), "--anchor belongs to pair mode"),
        (len(dict(labelled)) != len(labelled), "stimulus labels must be distinct"),
        (not CUT_SECONDS[0] <= args.seconds <= CUT_SECONDS[1], f"--seconds must lie in {CUT_SECONDS[0]:g}-{CUT_SECONDS[1]:g}"),
        (_below(args.start, 0.0), "--start must be >= 0"),
        (args.repeats < 1, "--repeats must be at least 1"),
        (args.mode == "pair" and bool(args.familiar), "pair has no right answer to practise on: --familiar belongs to abx and blend"),
    ]
    return [message for bad, message in checks if bad]


def _threshold_problems(args):
    checks = [
        (bool(args.stim or args.anchor or args.catch_raw), "threshold takes no --stim, --anchor or --catch-raw"),
        (args.start is not None, "threshold takes no --start"),
    ]
    return [message for bad, message in checks if bad]


def argument_problems(args):
    """What is wrong with the parsed arguments, as messages."""
    specific = _threshold_problems if args.mode == "threshold" else _stimulus_problems
    return _common_problems(args) + specific(args)


def _anchor_label(args):
    return args.anchor[0][0] if args.anchor else None


def build_mode(args, labels, rng):
    """The mode object for `args.mode`, with the question text the record keeps."""
    trials = args.trials or DEFAULT_TRIALS[args.mode]
    builders = {
        "abx": lambda: AbxMode(labels, trials, rng),
        "pair": lambda: PairMode(labels, args.repeats, rng, anchor=_anchor_label(args)),
        "blend": lambda: BlendMode(labels, trials, rng),
        "threshold": lambda: ThresholdMode(trials, rng, args.level_dbfs),
    }
    mode = builders[args.mode]()
    mode.text = args.question
    return mode


def build_protocol(args, mode, rng, catch_labels):
    """The block's familiarisation and catch trials: the mode's defaults unless the command line names counts."""
    familiar = listen_protocol.FAMILIAR_TRIALS[args.mode] if args.familiar is None else args.familiar
    catches = listen_protocol.default_catches(mode.total) if args.catch is None else args.catch
    return listen_protocol.Protocol(mode, rng, familiar=familiar, catches=catches, catch_labels=catch_labels)


def playback_of(args):
    """The record's `playback`: device and volume, and the rest of the chain as text or null."""
    return {"device": args.device, "volume": args.volume, **{key: getattr(args, key) for key in PLAYBACK_STATE}}


def _rounded(mapping):
    return {label: round(float(value), 2) for label, value in mapping.items()}


def base_record(args, files, cuts, info, start_s):
    """The record fields every result of this block shares."""
    today = datetime.date.today().isoformat()
    context = {
        "tool": "listen_ab",
        "seed": args.seed,
        "anchors": [label for label, _path in args.anchor],
        "target_dbfs": info["target_dbfs"],
        "levels_dbfs": _rounded(info["levels_dbfs"]),
        "gains_db": _rounded(info["gains_db"]),
        "peaks_dbfs": _rounded(info["peaks_dbfs"]),
        "cut_sha256": {label: listen_cuts.cut_sha256(cut) for label, cut in cuts.items()},
    }
    return {
        "id": f"{today}-{args.mode}-{args.tape}-{secrets.token_hex(3)}",
        "date": today,
        "round": args.round,
        "tape": args.tape,
        "windows": [[round(start_s, 3), round(start_s + args.seconds, 3)]],
        "files": files,
        "playback": playback_of(args),
        "context": context,
    }


def refuse_conflicts(ledger_path, tape, files, stimuli):
    """Stops before any decoding when a stimulus label already names another file on this tape in the ledger."""
    probe = {"id": "new-block", "tape": tape, "files": files, "question": {"stimuli": list(stimuli)}}
    problems = ledger.label_conflicts(ledger.read(ledger_path), probe)
    if problems:
        raise SystemExit("; ".join(problems))


def levelling_problem(info, start_s):
    """Why the cuts cannot be played ("" when they can): a silent cut, or one that would clip at the target level."""
    if info["silent"]:
        return f"silent at {start_s:.1f} s (a start past the end of the file, or an empty one): {', '.join(info['silent'])}"
    if info["unlevelled"]:
        peaks = ", ".join(f"{label} peaks at {info['peaks_dbfs'][label]:+.1f} dBFS" for label in info["unlevelled"])
        fits = f"every stimulus fits at --level-dbfs {info['fits_at_dbfs']:.1f} or below, or cut elsewhere"
        return f"cannot be levelled at {info['target_dbfs']:g} dBFS: {peaks}; {fits}"
    return ""


def cut_block(args, paths):
    """`(start_s, cuts, info)` for the block; stops when a cut is silent or cannot be levelled, before anything is served."""
    first, second = list(paths.values())[:2]
    start_s = args.start if args.start is not None else listen_cuts.pick_start(first, second, args.seconds)
    cuts, info = listen_cuts.prepare_cuts(paths, start_s, args.seconds, target_dbfs=args.level_dbfs)
    problem = levelling_problem(info, start_s)
    if problem:
        raise SystemExit(problem)
    return start_s, cuts, info


def block_stimuli(args, cuts, labels, raw):
    """`(every cut the block plays, {labels, gains} of the catch controls)`: the stimuli, the controls and the practice pair."""
    first, second = labels[:2]
    catch, gains = listen_cuts.catch_cuts(cuts, first, CATCH_RAW_LABEL if raw else None)
    familiar = listen_cuts.familiar_cuts(cuts[first], cuts[second], listen_cuts.EXAGGERATION.get(args.mode, 1.0))
    return {**cuts, **catch, **familiar}, {"labels": tuple(catch), "gains": gains}


def block_inputs(args):
    """`(stimulus paths, raw-tape path or {}, ledger path, files)`, every path confined, every stimulus checked against the ledger."""
    paths = {label: cli_paths.confined_path(path, f"--stim {label}", must_exist=True) for label, path in args.stim + args.anchor}
    raw = {CATCH_RAW_LABEL: cli_paths.confined_path(args.catch_raw, "--catch-raw", must_exist=True)} if args.catch_raw else {}
    ledger_path = cli_paths.confined_path(args.ledger, "--ledger")
    files = [ledger.file_entry(label, path, ledger.sha256_of(path)) for label, path in {**paths, **raw}.items()]
    refuse_conflicts(ledger_path, args.tape, files, list(paths))
    return paths, raw, ledger_path, files


def comparison_session(args, port, rng):
    """An abx, pair or blend block: the stimuli cut, levelled and checked against the ledger, ready to serve."""
    paths, raw, ledger_path, files = block_inputs(args)
    start_s, cuts, info = cut_block(args, {**paths, **raw})
    mode = build_mode(args, list(paths), rng)
    stimuli, catch = block_stimuli(args, cuts, list(paths), bool(raw))
    base = base_record(args, files, cuts, info, start_s)
    base["context"].update(catch_gains_db=_rounded(catch["gains"]), loops={"pause": listen_cuts.quietest_span(cuts[next(iter(paths))])})
    protocol = build_protocol(args, mode, rng, catch["labels"])
    return Session(protocol, stimuli, LedgerSink(base, ledger_path), port, {"loops": base["context"]["loops"]})


def threshold_session(args, port, rng):
    """A threshold run: tones made on the server, the record bound for the playback calibration file."""
    path = cli_paths.confined_path(args.calibration, "--calibration")
    mode = build_mode(args, [], rng)
    today = datetime.date.today().isoformat()
    base = {
        "id": f"{today}-threshold-{args.listener}-{secrets.token_hex(3)}",
        "date": today,
        "round": args.round,
        "listener": args.listener,
        "playback": playback_of(args),
        "context": {"tool": "listen_ab", "seed": args.seed},
    }
    protocol = build_protocol(args, mode, rng, (CATCH_TONE,))
    return Session(protocol, {}, CalibrationSink(base, path), port, {"loops": {"tone": [0.0, PERIOD_S]}, "autoloop": "tone"})


def main(argv=None):
    """Prepares one block, serves it on 127.0.0.1 and prints its summary once it is written."""
    args = parse_args(argv)
    problems = argument_problems(args)
    if problems:
        raise SystemExit("; ".join(problems))
    port = int(cli_paths.checked_number(args.port, "--port", minimum=1024, maximum=65535, kind=int))
    builder = threshold_session if args.mode == "threshold" else comparison_session
    session = builder(args, port, np.random.default_rng(args.seed))
    print(serve(session, port, open_browser=not args.no_browser))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
