"""A replay of the neural stage's output for auto_pure_linear and denoise_only (`AI_RESTORE_STAGE_CACHE`).

The self-driving loop renders one candidate per knob change. For auto_pure_linear the cost is
everything up to and including the neural denoiser, while 14 of its 26 loop knobs act after
it. With `AI_RESTORE_STAGE_CACHE=<absolute folder>` set, the pair the neural stage hands on --
the chain's output (the sibilant guard's reference) and the neural output -- is stored under a
key of everything that could change it (`modules/stage_cache_key.py`) and replayed when the
key recurs, so such a candidate skips the chain and the model.

- Unset, the cache is one environment read and the stages run exactly as before.
- A hit restores both files at the paths the producer wrote them at, relative to the stage
  folder, so every later file name, and with it every later stage, is a fresh render's.
- A miss hands back the producer's own paths; storing never raises and never changes them.
  No cache problem ends a render: a cache folder that cannot be read is a miss.
- Nothing is stored when the render logged a WARNING or an ERROR, its own or audio-separator's
  (a fallback after a stage failure is never frozen), when the stage folder already held
  files (a resumed work folder, whose stages may reuse what they find), when a module source
  or a package changed after the process loaded its code, when an output lies outside the
  stage folder, when the entry would put the cache over its cap, or when storing it would
  leave the volume under `MIN_FREE_GB` (both read again after eviction).
- An entry that does not read back is discarded; a replay that fails while the entry still
  reads back intact (the stage folder's disk is full or a file in it is locked) keeps it.
- DeepFilterNet and Resemble (weights outside the model store) and the event log (the
  plosive tamer records before the cached point) bypass it.
- Entries are evicted least recently used first under `AI_RESTORE_STAGE_CACHE_MAX_GB` (50 by
  default). Only folders named like a key, or like an entry in progress, are ever counted,
  evicted or swept. Final outputs are never cached (a cache of outputs once filled 250 GB),
  and cathar never calls this module.
"""

import collections
import dataclasses
import datetime
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
from pathlib import Path

from . import event_log as _event_log
from . import stage_cache_key as _key
from . import utils as _utils
from .hygiene import atomic_target, remove_tree
from .utils import log_msg

ENV_VAR = _key.CACHE_ENV_VAR
MAX_GB_ENV = "AI_RESTORE_STAGE_CACHE_MAX_GB"
DEFAULT_MAX_GB = 50.0
MIN_FREE_GB = 20.0
GB = 1 << 30
MB = 1 << 20
LAYOUT = "v1"
MANIFEST_SCHEMA = 1
MANIFEST = "manifest.json"
KEY_FILE = "key.json"
TMP_PREFIX = "tmp-"
STALE_TMP_S = 6 * 3600
OUTPUTS = ("surgical", "denoised")
# What an output may name instead of a stored file, in the order it is tried: the chain that
# changed nothing hands on its input, and a skipped neural stage hands on the chain's output.
ALIASES = {"surgical": ("input",), "denoised": ("surgical", "input")}
BYPASS_FLAGS = ("deepfilternet", "resemble_denoise")
# What reading, keying or storing an entry may raise; any of it falls back to the render, never ends it.
CACHE_FAILURES = (OSError, ValueError, RuntimeError, TypeError, LookupError, ArithmeticError, AttributeError)
COPY_BLOCK_BYTES = 1 << 22
PREFIX = "    [Stage Cache]"
# audio-separator logs through `logging`. What it logs at WARNING on every render, or as a
# fixed function of the input's length, is no fallback; anything else it logs at WARNING or
# above is (a stem written at 16 bits when the input's format could not be read, say).
SEPARATOR_LOGGER = "audio_separator"
SEPARATOR_ROUTINE = ("Using soundfile for writing.", "Audio duration (", "Automatically enabling override_model_segment_size")
_UNSAFE_IN_PART = ("\\", ":", "\0")
_HEX = frozenset("0123456789abcdef")
# ASCII: `\d` then matches 0-9 only, the digits `os.getpid()` writes, as `[0-9]` did.
_IN_PROGRESS_NAME = re.compile(r"tmp-[0-9a-f]{12}-\d+", re.ASCII)
_WARNED = set()
_NOTED = set()
_SEPARATOR_PROBLEMS = collections.Counter()
_WATCHING = []


@dataclasses.dataclass(frozen=True)
class _Job:
    """One cacheable call: where the cache lives, its key and key document, the input and the stage folder."""

    root: Path
    key: str
    document: dict
    input_wav: Path
    audio_dir: Path


def _warn_once(message):
    if message not in _WARNED:
        _WARNED.add(message)
        log_msg(message, level="WARNING")


def directory():
    """The cache root the environment names, or None when the cache is off.

    A relative path would resolve against each candidate's own working folder and split the
    cache per candidate, so it leaves the cache off, with one warning.
    """
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return None
    root = Path(raw)
    if root.is_absolute():
        return root
    _warn_once(f"{PREFIX} {ENV_VAR}={raw!r} is not an absolute path; the cache is off.")
    return None


def _positive_gb(raw):
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def max_bytes():
    """The cap in bytes: `AI_RESTORE_STAGE_CACHE_MAX_GB`, or DEFAULT_MAX_GB when unset or not a positive number."""
    raw = os.environ.get(MAX_GB_ENV, "").strip()
    gb = _positive_gb(raw) if raw else DEFAULT_MAX_GB
    if gb is None:
        _warn_once(f"{PREFIX} {MAX_GB_ENV}={raw!r} is not a positive number; the cap is {DEFAULT_MAX_GB:g} GB.")
        gb = DEFAULT_MAX_GB
    return int(gb * GB)


def through(produce, input_wav, audio_dir, call):
    """`produce()`'s (surgical, denoised) pair, replayed from the cache when this call's key recurs.

    `call` is every argument the producer reads besides the input file (`stage`, `flags`,
    `strategy`, ...); it is part of the key. Off, this is `produce()` and nothing else.
    """
    root = directory()
    if root is None:
        return produce()
    key, document = _keyed(input_wav, call)
    if key is None:
        return produce()
    job = _Job(root, key, document, Path(input_wav), Path(audio_dir))
    replayed = _lookup(job)
    if replayed is not None:
        return replayed
    return _render_and_store(produce, job)


def _bypass_reason(call):
    """Why this call cannot be cached, or None."""
    named = [flag for flag in BYPASS_FLAGS if call.get("flags", {}).get(flag)]
    if named:
        return f"{' and '.join(named)} on (weights outside the model store)"
    if _event_log.enabled():
        return "the event log is on (the plosive tamer records before the cached point)"
    return None


def _keyed(input_wav, call):
    """(key, key document), or (None, None) with the reason logged when the call bypasses the cache."""
    reason = _bypass_reason(call)
    if reason is None:
        try:
            document = _key.key_document(input_wav, call)
            return _key.key_of(document), document
        except CACHE_FAILURES as exc:
            log_msg(f"{PREFIX} The key could not be computed ({type(exc).__name__}: {exc}).", level="WARNING")
            reason = "no key"
    log_msg(f"{PREFIX} Bypassed: {reason}.")
    return None, None


# ----------------------------------------------------------------------------- audio-separator's warnings


def _separator_problem(record):
    """An audio-separator record at WARNING or above that is not one of its routine ones."""
    name = record.name
    from_separator = name == SEPARATOR_LOGGER or name.startswith(f"{SEPARATOR_LOGGER}.")
    return from_separator and record.levelno >= logging.WARNING and not str(record.msg).startswith(SEPARATOR_ROUTINE)


def _counting_factory(previous):
    """A log-record factory that makes records as `previous` does and counts audio-separator's problems."""

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        if _separator_problem(record):
            _SEPARATOR_PROBLEMS["count"] += 1
        return record

    return factory


def watch_separator_log():
    """Counts audio-separator's warnings and errors from now on; installed once per process, never removed.

    The separator logs through `logging`, not `log_msg`, so its fallbacks would not move
    `utils.problem_count`. A record factory sees every record whatever handlers exist, so the
    console handler the separator adds only when no handler is reachable is left as it is.
    """
    if not _WATCHING:
        _WATCHING.append(True)
        logging.setLogRecordFactory(_counting_factory(logging.getLogRecordFactory()))


def problem_count():
    """The WARNING and ERROR lines logged so far: this process's own (`utils.problem_count`) and audio-separator's."""
    return _utils.problem_count() + _SEPARATOR_PROBLEMS["count"]


# ----------------------------------------------------------------------------- reading an entry


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _safe_part(part):
    return bool(part) and not part.startswith(".") and not any(ch in part for ch in _UNSAFE_IN_PART)


def safe_relative(relative):
    """The parts of a stored relative path; ValueError unless each is a bare name (no '..', leading dot, separator, drive or NUL)."""
    _require(isinstance(relative, str), "a stored path is not a string")
    parts = relative.split("/")
    _require(all(_safe_part(part) for part in parts), f"unsafe stored path {relative!r}")
    return parts


def is_key(name):
    """Whether a name is a key: 64 lowercase hex digits."""
    return isinstance(name, str) and len(name) == 64 and set(name) <= _HEX


def _check_record(name, record):
    """ValueError unless `record` is an alias this output may name, or a safe path with a size and a digest."""
    _require(isinstance(record, dict), f"{name}: the record is not an object")
    if "from" in record:
        _require(record["from"] in ALIASES[name], f"{name}: alias {record['from']!r} is not allowed")
        return
    safe_relative(record.get("path"))
    _require(record.get("file") == f"{name}.wav", f"{name}: the stored file is misnamed")
    _require(isinstance(record.get("bytes"), int) and is_key(record.get("sha256")), f"{name}: size or digest malformed")


def read_manifest(entry, key):
    """The entry's manifest, checked against its key and its own shape; ValueError when it does not hold."""
    manifest = json.loads((Path(entry) / MANIFEST).read_text(encoding="utf-8"))
    _require(isinstance(manifest, dict), "the manifest is not an object")
    _require(manifest.get("schema") == MANIFEST_SCHEMA and manifest.get("key") == key, "the manifest belongs to another key")
    outputs = manifest.get("outputs")
    _require(isinstance(outputs, dict) and sorted(outputs) == sorted(OUTPUTS), "the manifest's outputs are malformed")
    for name in OUTPUTS:
        _check_record(name, outputs[name])
    _require(all(isinstance(manifest.get(field), (int, float)) for field in ("bytes", "created", "producer_seconds")), "no totals")
    return manifest


def _hashed_blocks(source, sink):
    """Reads a file in blocks, handing each to `sink`; returns its SHA-256 and byte count."""
    digest, size = hashlib.sha256(), 0
    with open(source, "rb") as reader:
        for block in iter(lambda: reader.read(COPY_BLOCK_BYTES), b""):
            digest.update(block)
            sink(block)
            size += len(block)
    return digest.hexdigest(), size


def copy_hashing(source, target):
    """Copies a file in blocks; returns its SHA-256 and byte count, read in the same pass."""
    with open(target, "wb") as writer:
        return _hashed_blocks(source, writer.write)


def file_digest(path):
    """A file's SHA-256 and byte count."""
    return _hashed_blocks(path, lambda _block: None)


def _restore_one(entry, record, paths, audio_dir):
    """One output back at its place under `audio_dir`, checked against its size and digest; an alias names a path already known."""
    if "from" in record:
        return paths[record["from"]], None
    target = audio_dir.joinpath(*safe_relative(record["path"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    with atomic_target(target) as partial:
        digest, copied = copy_hashing(entry / record["file"], partial)
        _require(copied == record["bytes"] and digest == record["sha256"], f"{record['file']} does not match its manifest")
    return target, target


def _unlink_all(paths):
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log_msg(f"{PREFIX} Could not remove the half-restored {path.name}: {exc}", level="DEBUG")


def _restore(entry, manifest, job):
    """(surgical, denoised) restored under the stage folder; a failure removes what was restored before it re-raises.

    A half replay left behind would read to a fresh render as its own finished stage output.
    """
    paths, written = {"input": job.input_wav}, []
    try:
        for name in OUTPUTS:
            paths[name], restored = _restore_one(entry, manifest["outputs"][name], paths, job.audio_dir)
            written.append(restored)
    except CACHE_FAILURES:
        _unlink_all(path for path in written if path is not None)
        raise
    return paths["surgical"], paths["denoised"]


def _stored_file_intact(entry, record):
    return file_digest(entry / record["file"]) == (record["sha256"], record["bytes"])


def _entry_intact(entry, manifest):
    """Whether every file the manifest names still reads back at its size and digest."""
    try:
        return all(_stored_file_intact(entry, record) for record in manifest["outputs"].values() if "from" not in record)
    except CACHE_FAILURES:
        return False


def _mb(size):
    return f"{size / MB:.1f}"


def _date(timestamp):
    try:
        return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return "at an unknown time"


def _touch(path):
    """Marks an entry used now, which is what the eviction orders by."""
    try:
        os.utime(path)
    except OSError as exc:
        log_msg(f"{PREFIX} Could not mark {path.parent.name[:12]} used: {exc}", level="DEBUG")


def _discard(entry, key, exc):
    """Removes an entry that does not read back; the render goes ahead as on a miss."""
    remove_tree(entry)
    log_msg(f"{PREFIX} Entry {key[:12]} unreadable ({type(exc).__name__}: {exc}); discarded, rendering afresh.", level="WARNING")


def _after_failed_replay(entry, manifest, key, exc):
    """A replay that failed while its entry still reads back intact failed on the stage folder's side, and keeps the entry."""
    if not _entry_intact(entry, manifest):
        _discard(entry, key, exc)
        return
    log_msg(
        f"{PREFIX} Hit {key[:12]} could not be restored into the stage folder ({type(exc).__name__}: {exc}); "
        "the entry is kept, rendering afresh.",
        level="WARNING",
    )


def _hit_message(key, manifest):
    return (
        f"{PREFIX} Hit {key[:12]}: replayed {_mb(manifest['bytes'])} MB stored {_date(manifest['created'])}; "
        f"skipped ~{manifest['producer_seconds']:.0f} s."
    )


def _lookup(job):
    """The replayed pair, or None on a miss.

    No cache problem ends the render: a cache folder that cannot be read (another account's,
    a share whose credentials expired) reads as a miss, with a warning.
    """
    try:
        return _replay(job)
    except CACHE_FAILURES as exc:
        log_msg(f"{PREFIX} The cache cannot be read ({type(exc).__name__}: {exc}); rendering afresh.", level="WARNING")
        return None


def _replay(job):
    """The replayed pair, or None; an entry that does not read back is discarded and counts as a miss."""
    entry = Path(job.root) / LAYOUT / job.key
    if not entry.is_dir():
        return None
    try:
        manifest = read_manifest(entry, job.key)
    except CACHE_FAILURES as exc:
        _discard(entry, job.key, exc)
        return None
    try:
        replayed = _restore(entry, manifest, job)
    except CACHE_FAILURES as exc:
        _after_failed_replay(entry, manifest, job.key, exc)
        return None
    _touch(entry / MANIFEST)
    log_msg(_hit_message(job.key, manifest))
    return replayed


# ----------------------------------------------------------------------------- storing an entry


def _holds_files(folder):
    """Whether the stage folder already holds a file: a resumed work folder whose stages may reuse it.

    Empty folders do not count (a replay that failed leaves the ones it made), and a folder
    that cannot be read counts as holding files, so nothing doubtful is stored.
    """
    try:
        return any(path.is_file() for path in Path(folder).rglob("*"))
    except OSError:
        return True


def _render_and_store(produce, job):
    """Runs the producer and stores its pair; returns the producer's own pair whatever the store does."""
    log_msg(f"{PREFIX} Miss {job.key[:12]}: rendering the chain and the neural model.")
    resumed = _holds_files(job.audio_dir)
    watch_separator_log()
    before, started = problem_count(), time.monotonic()
    produced = produce()
    elapsed = time.monotonic() - started
    try:
        reason = _refusal(job, resumed, before) or _store(job, produced, elapsed)
    except CACHE_FAILURES as exc:
        reason = f"{type(exc).__name__}: {exc}"
    if reason:
        log_msg(f"{PREFIX} Not stored {job.key[:12]}: {reason}.", level="WARNING")
    return produced


def _refusal(job, resumed, before):
    """Why the render just made must not be stored, or None.

    A stage that fell back after a failure logs a WARNING or an ERROR, and a fallback is never
    frozen; a stage folder that held files before the render may have handed some of them on
    (each stage reuses a valid output it finds), and the key does not cover them; and code or
    packages that changed after the process loaded its own may not be what the key names.
    """
    if problem_count() != before:
        return "a stage logged a warning or an error while it rendered (a fallback is never frozen)"
    if resumed:
        return "the stage folder held files from an earlier run, which the stages may have reused"
    return _key.drift(job.document)


def _same_path(first, second):
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))


def _relative_to(path, audio_dir):
    """`path` relative to the stage folder, posix-spelt; ValueError when it lies outside it."""
    try:
        relative = Path(os.path.abspath(path)).relative_to(os.path.abspath(audio_dir)).as_posix()
    except ValueError as exc:
        raise ValueError(f"{Path(path).name} lies outside the stage folder") from exc
    safe_relative(relative)
    return relative


def _record(name, paths, audio_dir):
    """An alias when the output is a path the replay already knows, else where it lies under the stage folder."""
    for alias in ALIASES[name]:
        if _same_path(paths[name], paths[alias]):
            return {"from": alias}
    return {"path": _relative_to(paths[name], audio_dir), "file": f"{name}.wav"}


def _records(produced, job):
    """Each output's manifest record, and the (name, file) pairs to store."""
    paths, records = {"input": job.input_wav}, {}
    for name, produced_path in zip(OUTPUTS, produced):
        paths[name] = Path(produced_path)
        records[name] = _record(name, paths, job.audio_dir)
    files = [(name, paths[name]) for name in OUTPUTS if "from" not in records[name]]
    return records, files


def _store(job, produced, elapsed):
    """Stores the produced pair; returns why it was not stored, or None."""
    records, files = _records(produced, job)
    layout, cap = Path(job.root) / LAYOUT, max_bytes()
    if (layout / job.key).is_dir():
        log_msg(f"{PREFIX} {job.key[:12]} is stored already (its replay failed on the stage folder's side); it is kept.")
        return None
    reason = _make_room(layout, sum(source.stat().st_size for _name, source in files), cap)
    if reason is None:
        _log_stored(job.key, _commit(job, layout, records, files, elapsed), layout, cap)
    return reason


def _is_entry(folder):
    return is_key(folder.name) and folder.is_dir()


def _is_in_progress(folder):
    return _IN_PROGRESS_NAME.fullmatch(folder.name) is not None and folder.is_dir()


def _note_foreign(path):
    """Says once that a folder or file under the layout is not the cache's; it is never counted, evicted or swept."""
    if str(path) not in _NOTED:
        _NOTED.add(str(path))
        log_msg(f"{PREFIX} {path.name} under {path.parent} is not a cache entry; it is left alone.")


def _classified(layout):
    """(stored entries, entries in progress) under the layout; whatever else is there is left alone."""
    stored, building = [], []
    for path in Path(layout).iterdir():
        if _is_entry(path):
            stored.append(path)
        elif _is_in_progress(path):
            building.append(path)
        else:
            _note_foreign(path)
    return stored, building


def _last_used(folder):
    """When an entry was last stored or replayed; an entry without a manifest sorts first."""
    try:
        return (folder / MANIFEST).stat().st_mtime
    except OSError:
        return 0.0


def _file_bytes(path):
    """A file's size, or 0 for anything else, a file another process removed meanwhile included."""
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _tree_bytes(folder):
    return sum(_file_bytes(path) for path in folder.rglob("*"))


def entries(layout):
    """(last used, bytes, folder) of every stored entry, least recently used first."""
    found = [(_last_used(folder), _tree_bytes(folder), folder) for folder in _classified(layout)[0]]
    return sorted(found, key=lambda item: (item[0], item[2].name))


def _in_progress_bytes(layout):
    """What the entries other processes are building hold: it will count once they are stored."""
    return sum(_tree_bytes(folder) for folder in _classified(layout)[1])


def _stale(folder, now):
    try:
        return now - folder.stat().st_mtime > STALE_TMP_S
    except OSError:
        return False


def _sweep_stale(layout, now):
    """Removes the entries-in-progress older than STALE_TMP_S: a process killed while it stored."""
    for folder in _classified(layout)[1]:
        if _stale(folder, now):
            remove_tree(folder)


def eviction_plan(stored, incoming, cap):
    """The least recently used entries to remove so the cache and the incoming entry fit the cap."""
    total = sum(size for _used, size, _folder in stored)
    plan = []
    for used, size, folder in stored:
        if total + incoming <= cap:
            break
        plan.append((used, size, folder))
        total -= size
    return plan


def _evict(plan):
    for used, size, folder in plan:
        if remove_tree(folder):
            log_msg(f"{PREFIX} Evicted {folder.name[:12]} ({_mb(size)} MB, last used {_date(used)}).")
        else:
            log_msg(f"{PREFIX} Could not evict {folder.name[:12]} (a file in it is still open); it stays.", level="WARNING")


def _free_bytes(layout):
    return shutil.disk_usage(layout).free


def _floor_reason():
    return f"storing it would leave under {MIN_FREE_GB:g} GB free on the volume"


def _make_room(layout, incoming, cap):
    """Evicts least recently used entries until the incoming one fits; returns why it cannot be stored, or None.

    The free-space floor is checked before anything is evicted, so a refused entry costs no
    stored one; both limits are read again from the disk once eviction ran.
    """
    if incoming > cap:
        return f"{_mb(incoming)} MB is over the {cap / GB:g} GB cap"
    layout.mkdir(parents=True, exist_ok=True)
    _sweep_stale(layout, time.time())
    plan = eviction_plan(entries(layout), incoming + _in_progress_bytes(layout), cap)
    if _free_bytes(layout) + sum(size for _used, size, _folder in plan) - incoming < MIN_FREE_GB * GB:
        return _floor_reason()
    _evict(plan)
    return _shortfall(layout, incoming, cap)


def _shortfall(layout, incoming, cap):
    """Why the entry still does not fit once eviction ran, read from the disk as it is now, or None.

    An entry Windows would not remove, or a store by another process meanwhile, leaves the
    cache fuller than the plan said.
    """
    held = sum(size for _used, size, _folder in entries(layout)) + _in_progress_bytes(layout)
    if held + incoming > cap:
        return f"the cache still holds {held / GB:.2f} GB after eviction, no room for {_mb(incoming)} MB under the {cap / GB:g} GB cap"
    if _free_bytes(layout) - incoming < MIN_FREE_GB * GB:
        return _floor_reason()
    return None


def _write_json(path, document):
    with atomic_target(path) as partial:
        partial.write_text(json.dumps(document, indent=1, sort_keys=True, allow_nan=True, default=_key.jsonable), encoding="utf-8")


def _manifest(job, records, elapsed):
    return {
        "schema": MANIFEST_SCHEMA,
        "key": job.key,
        "stage": job.document["stage"],
        "created": time.time(),
        "producer_seconds": round(elapsed, 3),
        "input": {field: job.document["input"][field] for field in ("name", "pcm_sha256", "frames")},
        "outputs": records,
        "bytes": sum(record.get("bytes", 0) for record in records.values()),
    }


def _fill_entry(job, folder, records, files, elapsed):
    """Copies the outputs into an entry-in-progress, then its key document, then its manifest (whose presence marks it complete)."""
    for name, source in files:
        with atomic_target(folder / records[name]["file"]) as partial:
            digest, copied = copy_hashing(source, partial)
        records[name].update(bytes=copied, sha256=digest)
    manifest = _manifest(job, records, elapsed)
    _write_json(folder / KEY_FILE, job.document)
    _write_json(folder / MANIFEST, manifest)
    return manifest


def _publish_entry(folder, final):
    """Renames a complete entry into place; False when another process stored the same key first."""
    try:
        os.rename(folder, final)
    except OSError:
        if final.exists():
            return False
        raise
    return True


def _commit(job, layout, records, files, elapsed):
    """Builds the entry beside the cache and renames it into place; returns its manifest, or None when another process won.

    The folder in progress never outlives the call.
    """
    folder = layout / f"{TMP_PREFIX}{job.key[:12]}-{os.getpid()}"
    remove_tree(folder)
    folder.mkdir(parents=True)
    try:
        manifest = _fill_entry(job, folder, records, files, elapsed)
        published = _publish_entry(folder, layout / job.key)
    finally:
        remove_tree(folder)
    if published:
        return manifest
    log_msg(f"{PREFIX} {job.key[:12]} was stored by another process first; this copy is dropped.")
    return None


def _log_stored(key, manifest, layout, cap):
    """The per-entry size line, with the cache's size after it."""
    if manifest is None:
        return
    stored = entries(layout)
    total = sum(size for _used, size, _folder in stored)
    sizes = {name: manifest["outputs"][name].get("bytes", 0) for name in OUTPUTS}
    log_msg(
        f"{PREFIX} Stored {key[:12]}: {_mb(manifest['bytes'])} MB (surgical {_mb(sizes['surgical'])}, "
        f"denoised {_mb(sizes['denoised'])}; {manifest['producer_seconds']:.0f} s of render); "
        f"cache {len(stored)} entries, {total / GB:.2f}/{cap / GB:g} GB."
    )
