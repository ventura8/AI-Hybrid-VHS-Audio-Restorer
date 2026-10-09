"""A/B listening on this machine only: ABX, pairwise preference with "same", and a blend threshold.

usage (from the repository root; the page binds 127.0.0.1, family tapes never leave the machine):

    python -m scripts.listen_ab abx --tape vaccin --stim A=<file> --stim B=<file> [--start 63] [--seconds 10]
    python -m scripts.listen_ab pair --tape vaccin --stim a=<file> --stim b=<file> [--stim c=<file> ...] [--anchor hiss=<file>]
    python -m scripts.listen_ab blend --tape vaccin --stim incumbent=<file> --stim candidate=<file>

Files outside the repository and the temp directory need their root in $AI_RESTORE_DATA_ROOTS
(`D:\\Tata` for the family tapes). The block's records go to the verdict ledger
(`assets/quality_calibration/verdicts.jsonl`, schema in `scripts/restoration_quality/ledger.py`)
once its last answer is in, in one write after every record is validated; Ctrl+C before that
writes nothing. When the ledger refuses them (a broken ledger line, a full disk), the answers go
to `<ledger>.unsaved-<block id>.json` beside it, the page and the terminal say where, and the
block still closes. A stimulus label that already names another file on the same tape in the
ledger is refused before anything is decoded: the ledger merges a tape's verdicts by label.

- The questions (`restoration_quality/listen_modes.py`): ABX, 24 trials by default and 16 at
  least (17/24 right is p 0.032, power 0.77 at a true 75% hit rate; 12/16 is p 0.038 at power
  0.63; under 5 trials no score reaches p 0.05), the blind repeat of the round-three air verdict
  ("b is better", +1 dB over +2 dB and off, given unblinded); pair with "same", every pair once
  (`--repeats`) in a fresh order plus one replicate pair at the end, with the `--anchor` output
  when there is one (the round-one cathar alpha 2 hissy cut, which keeps sessions comparable);
  blend, the candidate mixed into the incumbent at x in [0, 1] with QUEST-placed trials, 30 by
  default.
- The cuts (`restoration_quality/listen_cuts.py`): 8-12 s (10 s by default), aligned to the
  first stimulus (lag and polarity), loudness-matched on speech-active frames within +-20 dB,
  faded; without --start they go where the first two stimuli differ most audibly (NMR). A silent
  cut (a start past the end of a file) stops the block before it is served; a gain held at the
  +-20 dB bound is named in the summary and kept in the record.
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
from scripts.restoration_quality import ledger
from scripts.restoration_quality import listen_cuts as cuts_mod
from scripts.restoration_quality.listen_modes import AbxMode, BlendMode, PairMode

HOST = "127.0.0.1"
# The family tapes never leave the machine: the page is served on the loopback address only, so it needs no TLS.
LOCAL_SCHEME = "http"
DEFAULT_PORT = 8765
CUT_SECONDS = (8.0, 12.0)
DEFAULT_SECONDS = 10.0
DEFAULT_TRIALS = {"abx": 24, "pair": 0, "blend": 30}
MIN_TRIALS = {"abx": 16, "pair": 1, "blend": 1}
PAIR_REPEATS = 1
MAX_STIMULI = 6
STIMULUS_COUNTS = {"abx": (2, 2), "pair": (2, MAX_STIMULI), "blend": (2, 2)}
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


def save_unsaved(records, ledger_path, block_id):
    """Writes a block the ledger refused to `<ledger>.unsaved-<block id>.json`; says where, or carries the records itself."""
    ledger_path = Path(ledger_path)
    sidecar = ledger_path.with_name(f"{ledger_path.name}.unsaved-{block_id}.json")
    try:
        sidecar.write_text(json.dumps(records, indent=1), encoding="utf-8")
    except OSError as error:
        return f"the sidecar {sidecar} failed too ({error}); the records: {json.dumps(records)}"
    return f"the answers are in {sidecar}"


class Session:
    """One listening block: the cuts, the mode, the current trial's one-time audio names, the ledger write at the end."""

    def __init__(self, mode, cuts, out, port):
        self.mode = mode
        self.cuts = cuts
        self.out = out
        self.hosts = {f"{HOST}:{port}", f"localhost:{port}"}
        self.view = {"audio": {}, "trial": {}, "summary": "", "note": "", "failed": False}
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self._next_trial()

    def mix(self, weights):
        """The weighted sum of cuts a play button stands for."""
        return sum(weight * self.cuts[label] for label, weight in weights.items()).astype(np.float32)

    def _next_trial(self):
        trial = self.mode.trial(len(self.mode.answers))
        audio, tokens = {}, []
        for name, weights in trial["play"]:
            token = secrets.token_hex(8)
            audio[token] = cuts_mod.wav_bytes(self.mix(weights))
            tokens.append((name, token))
        self.view.update(audio=audio, trial={**trial, "tokens": tokens})

    def state(self):
        """What the page shows now."""
        if self.mode.finished():
            return {"done": True, "summary": self.view["summary"]}
        trial = self.view["trial"]
        buttons = [{"name": name, "url": f"/audio/{token}.wav"} for name, token in trial["tokens"]]
        index = len(self.mode.answers)
        return {
            "done": False,
            "index": index,
            "total": self.mode.total,
            "prompt": trial["prompt"],
            "buttons": buttons,
            "choices": trial["choices"],
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
        if headers.get("Host", "") not in self.hosts:
            return "this page answers on 127.0.0.1 only"
        if method == "POST" and not self._own_json(headers):
            return "answers are taken as application/json from this page only"
        return ""

    def _own_json(self, headers):
        origin = headers.get("Origin")
        return media_type(headers) == JSON_TYPE and (origin is None or origin in {local_url(host) for host in self.hosts})

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
        """Takes one answer; returns the next trial's state, or the summary once the block is in the ledger (500 when it is not)."""
        try:
            index, choice = parse_answer(body)
        except ValueError as error:
            return _text(HTTPStatus.BAD_REQUEST, str(error))
        trial = self.view["trial"]
        if self.mode.finished() or index != len(self.mode.answers):
            return _text(HTTPStatus.CONFLICT, "that trial is already answered")
        if choice not in trial["choices"]:
            return _text(HTTPStatus.BAD_REQUEST, f"choose one of {', '.join(trial['choices'])}")
        self.mode.record(trial, choice)
        self._advance()
        return self._progress()

    def _progress(self):
        """The next trial's state or the summary; a 500 carrying the summary when the ledger refused the block."""
        if self.view["failed"]:
            return _text(HTTPStatus.INTERNAL_SERVER_ERROR, self.view["summary"])
        return _json(self.state())

    def _advance(self):
        if not self.mode.finished():
            self._next_trial()
            return
        self.view.update(summary=self.mode.summary() + self.view["note"], audio={})
        self._write_records()

    def _write_records(self):
        """Every record of the block in one validated write; on failure the answers go to a sidecar and the summary says where."""
        base, path = self.out
        records = self.records()
        try:
            ledger.append_many(records, path)
        except (ledger.LedgerError, OSError) as error:
            self.view["failed"] = True
            self.view["summary"] += f" | NOT written to the ledger ({error}); {save_unsaved(records, path, base['id'])}"

    def records(self):
        """The ledger records of the finished block, fields in schema order."""
        base, _path = self.out
        merged = [{**base, **partial, "id": f"{base['id']}-{number}"} for number, partial in enumerate(self.mode.results(), start=1)]
        return [{name: record[name] for name in ledger.FIELDS} for record in merged]

    def after_reply(self):
        """Called once a response is on the wire: the block closes when its last answer has been sent back."""
        if self.mode.finished():
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
        print("stopped: an unfinished block is not written to the ledger")
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


def parse_args(argv=None):
    """The command line, parsed; stimulus paths are resolved and must exist (`cli_paths`)."""
    parser = argparse.ArgumentParser(description="Blind A/B listening on 127.0.0.1; answers go to the verdict ledger.")
    parser.add_argument("mode", choices=tuple(DEFAULT_TRIALS))
    parser.add_argument("--tape", required=True, type=tape_arg, help="tape slug, as in the ledger (vaccin, tele7abc, ...)")
    parser.add_argument("--stim", action="append", required=True, type=stimulus_arg, metavar="LABEL=PATH")
    parser.add_argument("--anchor", action="append", default=[], type=stimulus_arg, metavar="LABEL=PATH", help="pair: a fixed anchor")
    parser.add_argument("--start", type=float, default=None, help="cut start in seconds (default: where the first two differ most audibly)")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="cut length, 8-12 s")
    parser.add_argument("--trials", type=int, default=None, help="abx: 24 (16 at least), blend: 30 by default")
    parser.add_argument("--repeats", type=int, default=PAIR_REPEATS, help="pair: plays of each pair before the one replicate")
    parser.add_argument("--round", default="listen_ab", help="the ledger's round (session0, ...)")
    parser.add_argument("--question", default="", help="what the listener is asked, kept in the record")
    parser.add_argument("--device", default=None, help="playback device (headphones model, speakers)")
    parser.add_argument("--volume", default=None, help="playback volume setting")
    parser.add_argument("--ledger", type=cli_paths.path_arg, default=ledger.LEDGER_PATH)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args(argv)


def _below(value, minimum):
    return value is not None and value < minimum


def argument_problems(args):
    """What is wrong with the parsed arguments, as messages."""
    stimuli = args.stim + args.anchor
    low, high = STIMULUS_COUNTS[args.mode]
    least = MIN_TRIALS[args.mode]
    checks = [
        (not low <= len(stimuli) <= high, f"{args.mode} takes {low} to {high} stimuli, anchors included"),
        (args.mode != "pair" and bool(args.anchor), "--anchor belongs to pair mode"),
        (len(dict(stimuli)) != len(stimuli), "stimulus labels must be distinct"),
        (not CUT_SECONDS[0] <= args.seconds <= CUT_SECONDS[1], f"--seconds must lie in {CUT_SECONDS[0]:g}-{CUT_SECONDS[1]:g}"),
        (_below(args.start, 0.0), "--start must be >= 0"),
        (_below(args.trials, least), f"--trials must be at least {least} for {args.mode}"),
        (args.repeats < 1, "--repeats must be at least 1"),
    ]
    return [message for bad, message in checks if bad]


def _anchor_label(args):
    return args.anchor[0][0] if args.anchor else None


def build_mode(args, labels, rng):
    """The mode object for `args.mode`, with the question text the record keeps."""
    trials = args.trials or DEFAULT_TRIALS[args.mode]
    builders = {
        "abx": lambda: AbxMode(labels, trials, rng),
        "pair": lambda: PairMode(labels, args.repeats, rng, anchor=_anchor_label(args)),
        "blend": lambda: BlendMode(labels, trials, rng),
    }
    mode = builders[args.mode]()
    mode.text = args.question
    return mode


def base_record(args, files, cuts, info, start_s):
    """The record fields every result of this block shares."""
    today = datetime.date.today().isoformat()
    context = {
        "tool": "listen_ab",
        "seed": args.seed,
        "anchors": [label for label, _path in args.anchor],
        "gains_db": {label: round(gain, 2) for label, gain in info["gains_db"].items()},
        "gain_clamped": info["clamped"],
        "cut_sha256": {label: cuts_mod.cut_sha256(cut) for label, cut in cuts.items()},
    }
    return {
        "id": f"{today}-{args.mode}-{args.tape}-{secrets.token_hex(3)}",
        "date": today,
        "round": args.round,
        "tape": args.tape,
        "windows": [[round(start_s, 3), round(start_s + args.seconds, 3)]],
        "files": files,
        "playback": {"device": args.device, "volume": args.volume},
        "context": context,
    }


def refuse_conflicts(ledger_path, tape, files):
    """Stops before any decoding when a stimulus label already names another file on this tape in the ledger."""
    probe = {"id": "new-block", "tape": tape, "files": files, "question": {"stimuli": [entry["label"] for entry in files]}}
    problems = ledger.label_conflicts(ledger.read(ledger_path), probe)
    if problems:
        raise SystemExit("; ".join(problems))


def cut_block(args, paths):
    """`(start_s, cuts, info)` for the block; stops when a cut is silent, before anything is served."""
    first, second = list(paths.values())[:2]
    start_s = args.start if args.start is not None else cuts_mod.pick_start(first, second, args.seconds)
    cuts, info = cuts_mod.prepare_cuts(paths, start_s, args.seconds)
    if info["silent"]:
        raise SystemExit(f"silent at {start_s:.1f} s (a start past the end of the file, or an empty one): {', '.join(info['silent'])}")
    return start_s, cuts, info


def clamped_note(clamped):
    """The summary's note on gains held at the +-MAX_GAIN_DB bound ("" when none was)."""
    return f" | gain held at +-{cuts_mod.MAX_GAIN_DB:g} dB for {', '.join(clamped)}" if clamped else ""


def main(argv=None):
    """Cuts the stimuli, serves one block on 127.0.0.1 and prints its summary once the ledger has it."""
    args = parse_args(argv)
    problems = argument_problems(args)
    if problems:
        raise SystemExit("; ".join(problems))
    paths = {label: cli_paths.confined_path(path, f"--stim {label}", must_exist=True) for label, path in args.stim + args.anchor}
    port = int(cli_paths.checked_number(args.port, "--port", minimum=1024, maximum=65535, kind=int))
    ledger_path = cli_paths.confined_path(args.ledger, "--ledger")
    files = [ledger.file_entry(label, path, ledger.sha256_of(path)) for label, path in paths.items()]
    refuse_conflicts(ledger_path, args.tape, files)
    start_s, cuts, info = cut_block(args, paths)
    mode = build_mode(args, list(paths), np.random.default_rng(args.seed))
    session = Session(mode, cuts, (base_record(args, files, cuts, info, start_s), ledger_path), port)
    session.view["note"] = clamped_note(info["clamped"])
    print(serve(session, port, open_browser=not args.no_browser))
    return 0


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listening test</title>
<style>
:root { color-scheme: light dark; --bg: #f7f7f5; --fg: #1b1b1b; --muted: #666; --accent: #2d5bd1; --card: #fff; --line: #d6d6d0; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #151515; --fg: #ececec; --muted: #a8a8a8; --accent: #86a8ff; --card: #222; --line: #383838; }
}
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }
main { max-width: 640px; margin: 0 auto; padding: 24px 16px; }
.row { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
button { font: inherit; padding: 12px 20px; min-width: 80px; border-radius: 8px; cursor: pointer;
  border: 1px solid var(--line); background: var(--card); color: var(--fg); }
button.playing { box-shadow: inset 0 0 0 2px var(--accent); }
button.choice { background: var(--accent); border-color: var(--accent); color: var(--bg); }
#progress, #note { color: var(--muted); }
</style>
</head>
<body>
<main>
<h1>Listening test</h1>
<p id="progress"></p>
<p id="prompt"></p>
<div class="row" id="play"></div>
<div class="row" id="choices"></div>
<p id="note"></p>
</main>
<script>
let state = null;
let player = new Audio();
const byId = (id) => document.getElementById(id);
function button(text, cls, action) {
  const el = document.createElement("button");
  el.textContent = text;
  if (cls) el.className = cls;
  el.onclick = () => action(el);
  return el;
}
function play(url, el) {
  player.pause();
  document.querySelectorAll("button.playing").forEach((b) => b.classList.remove("playing"));
  player = new Audio(url);
  el.classList.add("playing");
  player.onended = () => el.classList.remove("playing");
  player.play();
}
function render(next) {
  state = next;
  byId("play").textContent = "";
  byId("choices").textContent = "";
  if (next.done) {
    byId("progress").textContent = "Finished.";
    byId("prompt").textContent = next.summary;
    return;
  }
  byId("progress").textContent = "Trial " + (next.index + 1) + " of " + next.total;
  byId("prompt").textContent = next.prompt;
  next.buttons.forEach((b) => byId("play").appendChild(button("Play " + b.name, "", (el) => play(b.url, el))));
  next.choices.forEach((c) => byId("choices").appendChild(button(c, "choice", () => answer(c))));
}
async function load() {
  const response = await fetch("/api/state", { cache: "no-store" });
  render(await response.json());
}
async function answer(choice) {
  player.pause();
  const response = await fetch("/api/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trial: state.index, choice: choice }),
  });
  if (!response.ok) {
    byId("note").textContent = await response.text();
    return load();
  }
  byId("note").textContent = "";
  render(await response.json());
}
load();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
