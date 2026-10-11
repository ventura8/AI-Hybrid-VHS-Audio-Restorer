"""Per-listener playback calibration: the threshold run's dBFS-to-sensation-level map, and what `auditory` reads from it.

`python -m scripts.listen_ab threshold` measures a listener's absolute thresholds at 1, 2, 4, 8
and 12.5 kHz at one device and volume setting (`listen_threshold`) and appends one record per
run to `assets/quality_calibration/playback_calibration.jsonl` (append-only, one JSON object per
LF-terminated line, like the verdict ledger). A record:

- `id`, `date`, `round`, `listener` (a slug), `playback` (`device` and `volume` as non-empty
  text, any other playback state as text or null: `connection`, `eq`, `anc`, `surround`);
- `level_dbfs`: the speech-active level the session's blocks play at (`listen_cuts`), and
  `cap_dbfs`, the hard cap no tone passed;
- `sl_map`: `{hz: dBFS}`, the tone level at 0 dB sensation level per frequency at that device
  and volume (null where the cap was not heard reliably): a sound at L dBFS sits L - sl_map[hz]
  dB above this listener's threshold;
- `thresholds`: per frequency the 75%-correct level, its 90% band, `censored`, the dB SPL a
  0 dBFS tone would play at if the listener's threshold were Terhardt's (`spl_offset_db`), and
  every response;
- `spl_full_scale_db` (the median `spl_offset_db` over the frequencies heard under the cap) and
  `loud_frames_spl`, the level `auditory.PLAYBACK_SPL_LOUD_FRAMES` takes for this listener;
  both null when no frequency was heard under the cap;
- `context`: the tone, the prior, the familiarisation, the catch trials and the block flags.

`auditory` reads the record at import, the way it reads its other constants: with
`AI_RESTORE_LISTENER=<listener>` set, `selected()` returns that listener's latest record (from
`AI_RESTORE_PLAYBACK_CALIBRATION`, or the default file) and `PLAYBACK_SPL_LOUD_FRAMES` becomes its
`loud_frames_spl`; unset, nothing is read and the 65 dB SPL assumption stays. A listener named
but not calibrated, or a latest record with no level, stops the import (`CalibrationError`)
rather than letting the check fall back to 65 dB SPL in silence. The record's id joins
`auditory.VERDICT_RULE`, so the loop's stored verdicts are read again under it.
"""

import datetime
import json
import math
import numbers
import os
import re
from pathlib import Path

from scripts.restoration_quality import ledger

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = REPO_ROOT / "assets" / "quality_calibration" / "playback_calibration.jsonl"
LISTENER_ENV = "AI_RESTORE_LISTENER"
PATH_ENV = "AI_RESTORE_PLAYBACK_CALIBRATION"
FIELDS = (
    "id",
    "date",
    "round",
    "listener",
    "playback",
    "level_dbfs",
    "cap_dbfs",
    "sl_map",
    "thresholds",
    "spl_full_scale_db",
    "loud_frames_spl",
    "context",
)
NAME_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")
REQUIRED_PLAYBACK = ("device", "volume")


class CalibrationError(ValueError):
    """A calibration record or file that breaks the schema, or a listener with no usable calibration."""


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _is_optional_number(value):
    return value is None or _is_number(value)


def _is_name(value):
    return isinstance(value, str) and NAME_RE.match(value) is not None


def _is_date(value):
    return isinstance(value, str) and datetime.date.fromisoformat(value).isoformat() == value


def _is_optional_text(value):
    return value is None or isinstance(value, str)


def _is_filled(value):
    return isinstance(value, str) and value.strip() != ""


def _playback_ok(playback):
    return all(map(_is_optional_text, playback.values())) and all(_is_filled(playback.get(key)) for key in REQUIRED_PLAYBACK)


def _band_ok(band):
    low, high = band
    return all(map(_is_number, band)) and low <= high


def _threshold_ok(entry):
    numbers_ok = all(map(_is_number, (entry["hz"], entry["threshold_dbfs"]))) and _band_ok(entry["band_dbfs"])
    return numbers_ok and isinstance(entry["censored"], bool) and isinstance(entry["responses"], list)


def _levels_ok(record):
    return _is_number(record["level_dbfs"]) and _is_number(record["cap_dbfs"])


def _thresholds_ok(entries):
    return isinstance(entries, list) and len(entries) > 0 and all(map(_threshold_ok, entries))


def _sl_map_ok(sl_map):
    return isinstance(sl_map, dict) and len(sl_map) > 0 and all(map(_is_optional_number, sl_map.values()))


def _holds(predicate):
    try:
        return bool(predicate())
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        return False


def _field_problems(record):
    checks = [
        (lambda: _is_name(record["id"]), "id must be letters, digits, '_', '.', '+' or '-'"),
        (lambda: _is_date(record["date"]), "date must be an ISO date (YYYY-MM-DD)"),
        (lambda: _is_name(record["listener"]), "listener must be a slug"),
        (lambda: _playback_ok(record["playback"]), "playback needs device and volume as text; any other key text or null"),
        (lambda: _levels_ok(record), "level_dbfs and cap_dbfs must be numbers"),
        (lambda: _sl_map_ok(record["sl_map"]), "sl_map must map frequencies to dBFS or null"),
        (lambda: _thresholds_ok(record["thresholds"]), "thresholds must be {hz, threshold_dbfs, band_dbfs, censored, responses}"),
        (lambda: _is_optional_number(record["spl_full_scale_db"]), "spl_full_scale_db must be a number or null"),
        (lambda: _is_optional_number(record["loud_frames_spl"]), "loud_frames_spl must be a number or null"),
        (lambda: isinstance(record["context"], dict), "context must be an object"),
    ]
    return [message for predicate, message in checks if not _holds(predicate)]


def validate(record):
    """Every schema problem of `record` as a list of messages; empty when it is valid."""
    if not isinstance(record, dict):
        return ["a calibration record must be a JSON object"]
    missing = [f"missing field {name!r}" for name in FIELDS if name not in record]
    return missing or _field_problems(record)


def _checked(record, where):
    problems = validate(record)
    if problems:
        raise CalibrationError(f"{where}: " + "; ".join(problems))
    return record


def _parse(path, number, line):
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise CalibrationError(f"{path}:{number}: not JSON ({error.msg})") from error
    return _checked(record, f"{path}:{number}")


def read(path=CALIBRATION_PATH):
    """Every record of the calibration file, validated, in file order; `[]` when the file does not exist yet."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [_parse(path, number, line) for number, line in enumerate(lines, start=1) if line.strip()]


def _refuse_taken(records, path):
    taken = {record["id"] for record in read(path)}
    clashes = [record["id"] for record in records if record["id"] in taken]
    if clashes:
        raise CalibrationError(f"{path}: id {clashes[0]!r} is already in the calibration file")


def append_many(records, path=CALIBRATION_PATH):
    """Validates the records, refuses an id already in the file, and appends them as LF-terminated lines in one write."""
    path, records = Path(path), list(records)
    for number, record in enumerate(records, start=1):
        _checked(record, f"new record {number}")
    _refuse_taken(records, path)
    return ledger.append_lines(records, path)


def latest_for(records, listener):
    """The last record of `listener` in file order, or None."""
    mine = [record for record in records if record["listener"] == listener]
    return mine[-1] if mine else None


def selected(environ=None):
    """The calibration `auditory` uses: the latest record of $AI_RESTORE_LISTENER, or None when it is unset.

    Raises CalibrationError when the listener has no record, or when the latest one placed no level.
    """
    environ = os.environ if environ is None else environ
    listener = environ.get(LISTENER_ENV, "").strip()
    if not listener:
        return None
    path = Path(environ.get(PATH_ENV) or CALIBRATION_PATH)
    return usable(latest_for(read(path), listener), listener, path)


def usable(record, listener, path):
    """`record` when it places a playback level; CalibrationError when there is none or it placed none."""
    if record is None:
        raise CalibrationError(
            f"no playback calibration for listener {listener!r} in {path}: run the threshold block or unset {LISTENER_ENV}"
        )
    if record["loud_frames_spl"] is None:
        raise CalibrationError(f"{record['id']} heard no tone under its cap: it places no playback level; run the threshold block again")
    return record


def loud_frames_spl(record, default):
    """The loud-frame playback level in dB SPL: the record's, or `default` without one."""
    return default if record is None else float(record["loud_frames_spl"])


def rule_suffix(record):
    """What a calibration adds to `auditory.VERDICT_RULE` ("" without one)."""
    return "" if record is None else f"+{record['id']}"
