"""The engine's event log: what an event-gated stage found, written where the harness can read it back.

The harness reads fricatives, plosives and pauses with locators of its own, and until now it
could only guess what the engine's stages saw. The APL sibilant guard is the case that made
this matter: at `apl_sibilant_hf_share_min` 0.5 it caught 3 events in five minutes of
Vaccin against the harness's 30, so the loops tuned its mix and crossover for two passes
while it touched almost nothing. With the stages' own spans on disk, the coverage check
(`meta.fricative_coverage`, plan R3) compares the real detector to the harness's, in one
unit (frames), instead of a re-implementation of it.

Set `AI_RESTORE_EVENT_LOG=<dir>` and the sibilant guard, the plosive tamer and the pause
floor keeper each write `<stage>__<recording>__<mode>__<track>.json` into that folder: the
stage, the recording (the tape's `.temp_work_<stem>` folder name, else the track's stem),
the configured `process_mode`, the track's path and rate, the stage's thresholds, and the
detected spans in seconds on the track's own timeline (`events_s`, [start, stop] pairs).
The mode is in the name because two modes run on one tape share its work folder and would
otherwise overwrite each other's records; two runs in one mode (tuning candidates) share
it too, so each candidate gets an event-log folder of its own. A stage that does not look
writes its reason (`skipped`): switched off, the material it refuses, or `failed: <error>`
when it raised. The mux writes `loudnorm__...json` the same way (see
`modules/mastering.py`). Unset, the default, nothing is written and nothing else changes:
the stages call this module only to ask whether the log is on, and the audio they write is
the same samples either way (tested per stage). A record that cannot be written is logged
and dropped; it never stops a restoration.
"""

import json
import os
import re
from pathlib import Path

from . import config
from .hygiene import atomic_target
from .utils import log_msg

ENV_VAR = "AI_RESTORE_EVENT_LOG"
WORK_DIR_PREFIX = ".temp_work_"
SCHEMA = 1
_UNSAFE_NAME = re.compile(r"[^\w.-]+")
_WRITE_FAILURES = (OSError, TypeError, ValueError)


def directory():
    """The folder the environment names for the event log, or None when the log is off."""
    raw = os.environ.get(ENV_VAR, "").strip()
    return Path(raw) if raw else None


def enabled():
    """Whether the event log is on for this process."""
    return directory() is not None


def recording_name(wav_path):
    """The tape a work file belongs to: its `.temp_work_<stem>` folder's stem, else the file's own stem."""
    path = Path(wav_path)
    for parent in path.parents:
        if parent.name.startswith(WORK_DIR_PREFIX):
            return parent.name.removeprefix(WORK_DIR_PREFIX)
    return path.stem


def process_mode():
    """The configured process mode, which tells two runs on one tape apart."""
    return str(getattr(config, "PROCESS_MODE", "") or "unknown")


def seconds(spans, rate):
    """Sample spans at `rate` as [start_s, stop_s] pairs, to the microsecond."""
    return [[round(float(start) / rate, 6), round(float(stop) / rate, 6)] for start, stop in spans]


def safe_name(name):
    """A name reduced to word characters, dots and dashes, so it can sit in a file name."""
    return _UNSAFE_NAME.sub("_", str(name)).strip("_") or "unnamed"


def _plain(value):
    """A NumPy scalar or anything else JSON cannot hold, as the nearest plain value."""
    item = getattr(value, "item", None)
    return item() if callable(item) else str(value)


def record_name(stage, wav_path):
    """The file name of a stage's record for a track: `<stage>__<recording>__<mode>__<track>.json`."""
    parts = (stage, recording_name(wav_path), process_mode(), Path(wav_path).stem)
    return "__".join(safe_name(part) for part in parts) + ".json"


def record_path(stage, wav_path):
    """Where a stage's record for a track goes in the event-log folder; None when the log is off."""
    folder = directory()
    if folder is None:
        return None
    return folder / record_name(stage, wav_path)


def document(stage, wav_path, record):
    """A stage's record for a track under the header every record carries."""
    header = {
        "schema": SCHEMA,
        "stage": stage,
        "recording": recording_name(wav_path),
        "mode": process_mode(),
        "track": str(Path(wav_path).resolve()),
    }
    return {**header, **record}


def write_json(path, payload):
    """Publishes a JSON document atomically; returns the path, or None after logging why it could not be written."""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=1, default=_plain)
        with atomic_target(target) as partial:
            Path(partial).write_text(text, encoding="utf-8")
    except _WRITE_FAILURES as exc:
        log_msg(f"    [Event Log] Could not write {target.name}: {exc}")
        return None
    return target


def write(stage, wav_path, record):
    """Writes one stage's record for a track into the event-log folder; returns the path, or None when the log is off."""
    target = record_path(stage, wav_path)
    if target is None:
        return None
    return write_json(target, document(stage, wav_path, record))


def write_spans(stage, wav_path, spans, rate, **details):
    """Writes the events a stage detected, given as sample spans at `rate`, with the thresholds it used."""
    if not enabled():
        return None
    return write(stage, wav_path, {"rate": rate, "count": len(spans), **details, "events_s": seconds(spans, rate)})


def write_skip(stage, wav_path, reason, **details):
    """Writes that a stage did not look for events on a track, or failed while it did, and why."""
    if not enabled():
        return None
    return write(stage, wav_path, {"skipped": reason, "count": 0, **details, "events_s": []})
