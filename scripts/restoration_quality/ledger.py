"""The listener-verdict ledger: every by-ear verdict as one JSON object per line, append-only.

Until ear v3 the user's verdicts lived in three places: `known_ordering_v2.json` (round one),
sets hard-coded in `experiments/tata_listen/check_verdicts.py`, and prose in the skills (rounds
two and three). The ledger, `assets/quality_calibration/verdicts.jsonl`, gathers them so the
verdict check, the calibration and the listening tool (`scripts/listen_ab.py`) read and write
one source. A record:

- `id` (unique), `date` (ISO), `round` (text: "1", "2", "session0", ...), `tape` (slug);
- `windows`: `[[start_s, end_s], ...]` that were heard, `[]` for the whole file;
- `files`: `[{label, path, sha256}]`, the stimuli and any reference. A relative path is taken
  from the repository root. `sha256` is the file's digest, or null when it was not hashed:
  files over 1 GiB (`HASH_MAX_BYTES`) are not read, so the full-tape .mov captures and renders
  (2.6-8.2 GB) stay null, the trade `audio_io.file_key` makes too. `listen_ab.py` hashes its
  stimuli under the same cap and keeps each decoded cut's PCM hash
  (`auditory.pcm_sha256`) in `context.cut_sha256`;
- `question`: `{type, stimuli, blind, text}`, where `type` is
  - `abx`: two stimuli, `n_trials` / `n_correct`, answer `{heard, p_value}` with the one-sided
    binomial p (17/24 right is p 0.032, power 0.77 at a true 75% hit rate);
  - `pair`: two stimuli, answer `{counts: {<a>: n, <b>: n, "same": n}}`, "same" when the
    listener could not choose (`listen_ab.py` adds `trials: [[A, B, chosen, replicate], ...]`
    in the order heard, `replicate` true on the block's one consistency repeat);
  - `blend`: incumbent and candidate on a blend continuum x in [0, 1], answer
    `{threshold_x, band, responses: [[x, right], ...]}` (x heard 75% of the time);
  - `flag`: answer `{flagged: {flag: [labels]}, clean: {flag: [labels]}}`; `clean["*"]` lists
    outputs accepted with no complaint at all;
  - `preference`: answer `{order: [[best, ...], [next, ...], ...]}`; labels sharing a tier carry
    no claim between them (tied, or not compared);
- `n_trials` / `n_correct` (null for by-ear verdicts), `confidence` (0-1, or null when not
  measured), `playback` `{device, volume}`, `context` (the listener's words; `heard`: a list of
  other variants in the session that drew no verdict; `verdict_group`: a name shared by the
  records that restate one verdict across tapes, so a fit counts it once).

A stimulus label means one file per tape across the whole ledger: `read` and `append` refuse a
record whose stimulus label already names another file on the same tape (other digests when
both are known, else another path). `for_tape`, `flag_verdicts` and `heard_labels` merge the
records of a tape by label, so a label reused for another file would merge two stimuli's
verdicts. Reference entries that are not stimuli (`source`) are free to differ: on Tele7abc
`source` is four different excerpts across rounds.

The backfill (33 records) holds what the user said, one listener, playback not recorded; where
a record has to read the words (which comparisons they cover, which render they meant), its
`context` says so. Flags and clean lists come only from words: a rank is not a flag, and calibration
reads clean lists as by-ear bounds. Round one (2026-09-20): the known set from the de-esser and
probe fixes (the ranks of `known_ordering_v2.json`; the flags the words carry, "speech from under
water" on every tape's de-esser-bug output and "underwater again, light hiss" on Tele7abc's single
4 s probe; no clean lists, since no words cleared an output), the 13-variant Tele7abc listening
set, SOTI/Vaccin "alpha 2 sounds better" (over the cathar baseline at alpha 2.5, whose default it
replaced, and the APL baseline). Round two (2026-10-05, the v2 plateaus): APL better on speech,
pauses natural on both, music fine on both, APL's 's' still thin. Round three (2026-10-08, the
air shelf): "b is better", +1 dB over +2 dB and off, unblinded, so a `preference` and not an
`abx`; no flag records, since the user named no flaw there (round two's "thin" is round two's,
and +1 dB as a clean bound is what the Session 0 ABX is for). Every file up to 1 GiB carries its
digest, taken 2026-10-09 from files dated no later than their verdicts (the known set
2026-09-20, the v2 set 2026-09-22/25, v4_air 2026-10-06).
"""

import datetime
import hashlib
import itertools
import json
import math
import numbers
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "assets" / "quality_calibration" / "verdicts.jsonl"
QUESTION_TYPES = ("abx", "pair", "blend", "flag", "preference")
FIELDS = (
    "id",
    "date",
    "round",
    "tape",
    "windows",
    "files",
    "question",
    "answer",
    "n_trials",
    "n_correct",
    "confidence",
    "playback",
    "context",
)
ANY_FLAG = "*"
SAME = "same"
FLAG_PREFIX = "listener."
NAME_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HASH_MAX_BYTES = 1 << 30
HASH_CHUNK = 1 << 20
P_TOLERANCE = 1e-6
_SCHEMA_ERRORS = (KeyError, TypeError, ValueError, AttributeError, IndexError)


class LedgerError(ValueError):
    """A record or a ledger line that breaks the schema; the message names every problem."""


def binomial_p(n_correct, n_trials):
    """One-sided p of `n_correct` or more right out of `n_trials` guesses at 1/2 (the ABX test)."""
    tail = sum(math.comb(n_trials, k) for k in range(n_correct, n_trials + 1))
    return tail / 2.0**n_trials


def file_entry(label, path, sha256=None):
    """One `files` entry; `sha256` stays null when the file was not hashed."""
    return {"label": label, "path": str(path), "sha256": sha256}


def sha256_of(path, max_bytes=HASH_MAX_BYTES):
    """The file's SHA-256, or None when it is missing or larger than `max_bytes`."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size > max_bytes:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record):
    """Every schema problem of `record` as a list of messages; empty when the record is valid."""
    if not isinstance(record, dict):
        return ["a record must be a JSON object"]
    missing = _missing_fields(record)
    if missing:
        return missing
    problems = _field_problems(record) + _question_problems(record)
    return problems or _TYPE_CHECKS[record["question"]["type"]](record)


def check_record(record, where="record"):
    """`record` itself when it is valid; LedgerError naming every problem otherwise."""
    _raise_on(validate_record(record), where)
    return record


def read(path=LEDGER_PATH):
    """Every record of the ledger at `path`, validated, in file order; `[]` when the file does not exist yet."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = [_parse_line(path, number, line) for number, line in enumerate(lines, start=1) if line.strip()]
    _require_unique(records, path)
    _require_consistent(records, path)
    return records


def append_many(records, path=LEDGER_PATH):
    """Validates every record against the schema, the ledger's ids and its labels, then appends them in one write.

    Nothing is written unless every record passes, so a block's records land together or not
    at all. Each record is one LF-terminated line; a ledger whose last line has no line end
    (a hand edit) gets one first, so the block's first record never joins it.
    """
    path, records = Path(path), list(records)
    for number, record in enumerate(records, start=1):
        check_record(record, f"new record {number}")
    existing = read(path)
    _refuse_taken_ids(records, existing, path)
    _require_unique(records, "new records")
    _require_consistent(existing + records, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lead = _line_end_owed(path)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(lead + "".join(json.dumps(record) + "\n" for record in records))
    return records


def _line_end_owed(path):
    """`"\\n"` when the file at `path` ends without a line end, else `""`.

    `read` takes a last line without one as a whole record, so the appender must too: written
    straight after it, the first new record would share its line and the next read would fail
    on that line ("Extra data") with both records in it.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return ""
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return "" if handle.read(1) == b"\n" else "\n"


def append(record, path=LEDGER_PATH):
    """Validates `record` and appends it as one LF-terminated line; an id already in the ledger is refused."""
    return append_many([record], path)[0]


def label_conflicts(records, record):
    """Messages for each stimulus of `record` whose label names another file on the same tape in `records`."""
    known = {}
    for earlier in records:
        _remember(known, earlier)
    return _conflicts_with(known, record)


def for_tape(records, tape):
    """The records about `tape`."""
    return [record for record in records if record["tape"] == tape]


def heard_labels(records):
    """Every label a listener heard in `records`: the stimuli and the session's `context.heard`."""
    labels = set()
    for record in records:
        labels.update(record["question"]["stimuli"])
        labels.update(record["context"].get("heard", []))
    return labels


def flag_verdicts(records):
    """`(flagged, clean)`, each `{flag: set of labels}`, merged over the flag records in `records`."""
    flagged, clean = {}, {}
    for record in records:
        if record["question"]["type"] == "flag":
            _merge(flagged, record["answer"]["flagged"])
            _merge(clean, record["answer"]["clean"])
    return flagged, clean


def preference_pairs(record):
    """`(better, worse)` for every two labels in different tiers of a preference record."""
    tiers = itertools.combinations(record["answer"]["order"], 2)
    return [(better, worse) for upper, lower in tiers for better in upper for worse in lower]


def _merge(into, mapping):
    for flag, labels in mapping.items():
        into.setdefault(flag, set()).update(labels)


def _parse_line(path, number, line):
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise LedgerError(f"{path}:{number}: not JSON ({error.msg})") from error
    return check_record(record, f"{path}:{number}")


def _raise_on(problems, where):
    if problems:
        raise LedgerError(f"{where}: " + "; ".join(problems))


def _refuse_taken_ids(records, existing, where):
    taken = {record["id"] for record in existing}
    clashes = [record["id"] for record in records if record["id"] in taken]
    if clashes:
        raise LedgerError(f"{where}: id {clashes[0]!r} is already in the ledger")


def _require_unique(records, where):
    seen = set()
    for record in records:
        if record["id"] in seen:
            raise LedgerError(f"{where}: duplicate id {record['id']!r}")
        seen.add(record["id"])


def _stimulus_entries(record):
    stimuli = set(record["question"]["stimuli"])
    return [entry for entry in record["files"] if entry["label"] in stimuli]


def _remember(known, record):
    """Adds `record`'s stimuli to `known`, `{(tape, label): (record id, file entry)}`; the first record naming a label keeps it."""
    for entry in _stimulus_entries(record):
        known.setdefault((record["tape"], entry["label"]), (record["id"], entry))


def _path_key(path):
    """A path as compared: from the repository root when relative, normalised, case-folded where the platform folds case."""
    path = Path(path)
    return os.path.normcase(os.path.normpath(str(path if path.is_absolute() else REPO_ROOT / path)))


def _same_file(first, second):
    """Equal digests when both are known; otherwise the same path."""
    if first["sha256"] and second["sha256"]:
        return first["sha256"] == second["sha256"]
    return _path_key(first["path"]) == _path_key(second["path"])


def _conflicts_with(known, record):
    problems = []
    for entry in _stimulus_entries(record):
        earlier = known.get((record["tape"], entry["label"]))
        if earlier is not None and not _same_file(earlier[1], entry):
            problems.append(
                f"stimulus {entry['label']!r} on tape {record['tape']!r} is {earlier[1]['path']} in record {earlier[0]!r}, "
                f"{entry['path']} here: give it a label of its own"
            )
    return problems


def _require_consistent(records, where):
    known = {}
    for record in records:
        _raise_on(_conflicts_with(known, record), f"{where}: record {record['id']!r}")
        _remember(known, record)


def _missing_fields(record):
    return [f"missing field {name!r}" for name in FIELDS if name not in record]


def _failed(checks):
    """The messages of the `(predicate, message)` checks whose predicate is false or cannot be evaluated."""
    return [message for predicate, message in checks if not _holds(predicate)]


def _holds(predicate):
    try:
        return bool(predicate())
    except _SCHEMA_ERRORS:
        return False


def _is_name(value):
    return isinstance(value, str) and NAME_RE.match(value) is not None


def _is_text(value):
    return isinstance(value, str) and value.strip() != ""


def _is_optional_text(value):
    return value is None or isinstance(value, str)


def _is_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_optional_count(value):
    return value is None or _is_count(value)


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _is_unit(value):
    return _is_number(value) and 0.0 <= value <= 1.0


def _is_optional_unit(value):
    return value is None or _is_unit(value)


def _is_date(value):
    return datetime.date.fromisoformat(value).isoformat() == value


def _is_window(window):
    start, end = window
    return _is_number(start) and _is_number(end) and 0.0 <= start < end


def _is_digest(digest):
    return digest is None or SHA256_RE.match(digest) is not None


def _is_file(entry):
    return _is_name(entry["label"]) and _is_text(entry["path"]) and _is_digest(entry["sha256"])


def _name_list(values):
    """A list of distinct names, possibly empty."""
    return isinstance(values, list) and all(map(_is_name, values)) and len(set(values)) == len(values)


def _unique_names(values):
    return _name_list(values) and len(values) > 0


def _windows_ok(windows):
    return isinstance(windows, list) and all(map(_is_window, windows))


def _files_ok(files):
    return isinstance(files, list) and len(files) > 0 and all(map(_is_file, files)) and _unique_names([entry["label"] for entry in files])


def _playback_ok(playback):
    return isinstance(playback, dict) and all(map(_is_optional_text, playback.values()))


def _optional_key_ok(mapping, key, check):
    """True when `mapping` lacks `key` or its value passes `check`."""
    return key not in mapping or check(mapping[key])


def _field_problems(record):
    context = record["context"]
    return _failed(
        [
            (lambda: _is_name(record["id"]), "id must be letters, digits, '_', '.', '+' or '-'"),
            (lambda: _is_date(record["date"]), "date must be an ISO date (YYYY-MM-DD)"),
            (lambda: _is_text(record["round"]), "round must be a non-empty text"),
            (lambda: _is_name(record["tape"]), "tape must be a slug (letters, digits, '_', '.', '+' or '-')"),
            (lambda: _windows_ok(record["windows"]), "windows must be [[start_s, end_s], ...] with 0 <= start < end"),
            (lambda: _files_ok(record["files"]), "files must be a non-empty list of {label, path, sha256 or null} with distinct labels"),
            (lambda: _is_optional_count(record["n_trials"]), "n_trials must be a count or null"),
            (lambda: _is_optional_count(record["n_correct"]), "n_correct must be a count or null"),
            (lambda: _is_optional_unit(record["confidence"]), "confidence must lie in [0, 1] or be null"),
            (lambda: _playback_ok(record["playback"]), "playback must be {device, volume} as texts or nulls"),
            (lambda: isinstance(context, dict), "context must be an object"),
            (lambda: _optional_key_ok(context, "heard", _name_list), "context.heard must be a list of distinct labels"),
            (lambda: _optional_key_ok(context, "verdict_group", _is_name), "context.verdict_group must be a name"),
        ]
    )


def _file_labels(record):
    return {entry["label"] for entry in record["files"]}


def _question_problems(record):
    question = record["question"]
    return _failed(
        [
            (lambda: question["type"] in QUESTION_TYPES, f"question.type must be one of {', '.join(QUESTION_TYPES)}"),
            (lambda: _unique_names(question["stimuli"]), "question.stimuli must be distinct labels"),
            (lambda: set(question["stimuli"]) <= _file_labels(record), "every stimulus needs its entry in files"),
            (lambda: isinstance(question.get("blind", False), bool), "question.blind must be true or false"),
            (lambda: isinstance(question.get("text", ""), str), "question.text must be a text"),
        ]
    )


def _two_stimuli(record):
    return len(record["question"]["stimuli"]) == 2


def _trials_ok(record):
    n_trials, n_correct = record["n_trials"], record["n_correct"]
    return _is_count(n_trials) and n_trials >= 1 and _is_count(n_correct) and n_correct <= n_trials


def _no_trials(record):
    return record["n_trials"] is None and record["n_correct"] is None


def _abx_problems(record):
    answer = record["answer"]
    return _failed(
        [
            (lambda: _two_stimuli(record), "an abx record compares exactly two stimuli"),
            (lambda: _trials_ok(record), "an abx record needs n_trials >= 1 and 0 <= n_correct <= n_trials"),
            (lambda: isinstance(answer["heard"], bool), "abx answer.heard must be true or false"),
            (lambda: _p_matches(answer["p_value"], record), "abx answer.p_value must be the one-sided binomial p of n_correct of n_trials"),
        ]
    )


def _p_matches(p_value, record):
    return abs(p_value - binomial_p(record["n_correct"], record["n_trials"])) <= P_TOLERANCE


def _count_total(counts):
    return sum(counts.values()) if all(map(_is_count, counts.values())) else -1


def _pair_problems(record):
    stimuli, answer = record["question"]["stimuli"], record["answer"]
    return _failed(
        [
            (lambda: _two_stimuli(record), "a pair record compares exactly two stimuli"),
            (lambda: SAME not in stimuli, f"a pair stimulus cannot be named {SAME!r}"),
            (lambda: set(answer["counts"]) == set(stimuli) | {SAME}, f"pair answer.counts must count both stimuli and {SAME!r}"),
            (lambda: _count_total(answer["counts"]) == record["n_trials"], "pair answer.counts must add up to n_trials"),
            (lambda: record["n_correct"] is None, "a pair record has no right answer: n_correct must be null"),
        ]
    )


def _response_ok(response):
    x, right = response
    return _is_unit(x) and isinstance(right, bool)


def _responses_ok(responses, record):
    rights = [right for _x, right in responses]
    return all(map(_response_ok, responses)) and len(responses) == record["n_trials"] and rights.count(True) == record["n_correct"]


def _band_ok(band):
    low, high = band
    return _is_number(low) and _is_number(high) and 0.0 <= low <= high


def _blend_problems(record):
    answer = record["answer"]
    return _failed(
        [
            (lambda: _two_stimuli(record), "a blend record has exactly two stimuli: the incumbent and the candidate"),
            (lambda: _trials_ok(record), "a blend record needs n_trials >= 1 and 0 <= n_correct <= n_trials"),
            (lambda: _is_number(answer["threshold_x"]) and answer["threshold_x"] >= 0.0, "blend answer.threshold_x must be >= 0"),
            (lambda: _band_ok(answer["band"]), "blend answer.band must be [low, high] with 0 <= low <= high"),
            (
                lambda: _responses_ok(answer["responses"], record),
                "blend answer.responses must be n_trials [x, right] pairs, n_correct right",
            ),
        ]
    )


def _flag_key_ok(key, allow_any):
    if allow_any and key == ANY_FLAG:
        return True
    return isinstance(key, str) and key.startswith(FLAG_PREFIX)


def _label_list_ok(labels, stimuli):
    return isinstance(labels, list) and len(set(labels)) == len(labels) and set(labels) <= stimuli


def _flag_map_ok(mapping, stimuli, allow_any):
    keys_ok = all(_flag_key_ok(key, allow_any) for key in mapping)
    return keys_ok and all(_label_list_ok(labels, stimuli) for labels in mapping.values())


def _contradicts(answer):
    """A label both flagged with a flag and heard clean of it (or accepted with no complaint at all)."""
    flagged, clean = answer["flagged"], answer["clean"]
    every = set(itertools.chain.from_iterable(flagged.values()))
    per_flag = [set(labels) & set(clean.get(flag, [])) for flag, labels in flagged.items()]
    return bool(every & set(clean.get(ANY_FLAG, []))) or any(per_flag)


def _flag_problems(record):
    stimuli, answer = set(record["question"]["stimuli"]), record["answer"]
    return _failed(
        [
            (lambda: _flag_map_ok(answer["flagged"], stimuli, False), "flag answer.flagged must map listener.* flags to stimulus labels"),
            (lambda: _flag_map_ok(answer["clean"], stimuli, True), "flag answer.clean must map listener.* flags or '*' to stimulus labels"),
            (lambda: not _contradicts(answer), "a label cannot be flagged and heard clean of the same flag"),
            (lambda: _no_trials(record), "a flag record holds no trials: n_trials and n_correct must be null"),
        ]
    )


def _tier_ok(tier):
    return isinstance(tier, list) and len(tier) > 0


def _tiers_ok(order, stimuli):
    flat = list(itertools.chain.from_iterable(order))
    return len(order) >= 2 and all(map(_tier_ok, order)) and len(flat) == len(set(flat)) and set(flat) <= stimuli


def _preference_problems(record):
    stimuli, answer = set(record["question"]["stimuli"]), record["answer"]
    return _failed(
        [
            (
                lambda: _tiers_ok(answer["order"], stimuli),
                "preference answer.order must be two or more non-empty tiers of distinct stimuli",
            ),
            (lambda: _no_trials(record), "a preference record holds no trials: n_trials and n_correct must be null"),
        ]
    )


_TYPE_CHECKS = {
    "abx": _abx_problems,
    "pair": _pair_problems,
    "blend": _blend_problems,
    "flag": _flag_problems,
    "preference": _preference_problems,
}
