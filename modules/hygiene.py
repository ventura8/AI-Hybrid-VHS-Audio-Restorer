"""Temp and file hygiene shared by every stage.

Every stage writes through a same-directory partial and publishes it with one rename: a
power cut or a kill part-way through leaves a partial, never a target that reads as a
finished file. A WAV truncated mid-stream keeps a valid header, so the validity check
that lets a rerun resume would otherwise take the fragment for the whole.

While a video is processed, every temp-file API is pointed inside its work directory so
library scratch leaves with the work directory rather than accumulating in the system
temp.
"""

import contextlib
import os
import shutil
import tempfile
import time
from pathlib import Path

from .utils import log_msg

PARTIAL_MARKER = ".tmp"
_TEMP_ENV_KEYS = ("TMPDIR", "TEMP", "TMP")


def partial_path(target):
    """The same-directory partial a stage writes before publishing `target`."""
    target = Path(target)
    return target.with_name(f"{target.stem}{PARTIAL_MARKER}{target.suffix}")


def _remove_if_exists(path):
    if path.exists():
        path.unlink()


@contextlib.contextmanager
def atomic_target(target):
    """Yields the partial to write; publishes it over `target` on success, removes it otherwise.

    A stage that writes nothing (it returned before opening the partial) publishes nothing.
    """
    target = Path(target)
    partial = partial_path(target)
    _remove_if_exists(partial)
    try:
        yield partial
    except BaseException:
        _remove_if_exists(partial)
        raise
    if partial.exists():
        publish(partial, target)


# Directory entries are made durable on POSIX by syncing the directory; Windows cannot open
# a directory for that, and its rename is journalled by the filesystem.
SYNC_DIRECTORIES = os.name == "posix"


def _fsync(path, flags):
    """Flushes a file or directory to stable storage."""
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(partial, target):
    """Replaces `target` with `partial` in one step, durably.

    The rename alone makes the file visible; it does not make its bytes durable. After a
    power loss a file renamed into place can come back truncated, and a truncated WAV
    still reads as valid audio, which is the failure this module exists to prevent. The
    partial is flushed to storage before the rename, and on POSIX its directory after it,
    so the new entry is durable too.
    """
    partial, target = Path(partial), Path(target)
    _fsync(partial, os.O_RDWR)
    os.replace(str(partial), str(target))
    if SYNC_DIRECTORIES:
        _fsync(target.parent, os.O_RDONLY)


def is_partial(path):
    """Whether a file is an unpublished partial: ours, or a same-directory temp a library left."""
    name = path.name
    if f"{PARTIAL_MARKER}." in name or name.endswith(PARTIAL_MARKER):
        return True
    # audio-separator publishes through mkstemp(prefix=".<name>.", suffix=".wav", dir=<output>).
    return name.startswith(".") and name.endswith(".wav")


def _unlink_partial(path, removed):
    try:
        path.unlink()
        removed.append(path)
    except OSError as exc:
        log_msg(f"    [Hygiene] Could not remove partial {path.name}: {exc}", is_error=True)


def _partials_under(work_dir):
    """Every unpublished partial below a directory, or nothing when it does not exist."""
    if not work_dir.is_dir():
        return []
    return [path for path in work_dir.rglob("*") if path.is_file() and is_partial(path)]


def sweep_partials(work_dir):
    """Removes every unpublished partial under a work directory; returns the paths removed.

    Run when a work directory is (re)opened, so a resumed restoration never globs a
    fragment from the interrupted run as a finished stage output.
    """
    removed = []
    for path in _partials_under(Path(work_dir)):
        _unlink_partial(path, removed)
    if removed:
        log_msg(f"  [Hygiene] Removed {len(removed)} unfinished partial file(s) from the previous run.")
    return removed


def _restore_env(saved_env):
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@contextlib.contextmanager
def scoped_temp_dir(work_dir):
    """Points every temp-file API at a folder inside the work directory for the block.

    Python's `tempfile`, and any child process (ffmpeg, resemble-enhance, cathar) through
    the TMP/TEMP/TMPDIR variables it inherits.
    """
    scratch = Path(work_dir) / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    saved_env = {key: os.environ.get(key) for key in _TEMP_ENV_KEYS}
    saved_tempdir = tempfile.tempdir
    for key in _TEMP_ENV_KEYS:
        os.environ[key] = str(scratch)
    tempfile.tempdir = str(scratch)
    try:
        yield scratch
    finally:
        tempfile.tempdir = saved_tempdir
        _restore_env(saved_env)


def remove_tree(path, attempts=3, delay_s=0.5):
    """Removes a directory tree, retrying briefly for handles Windows releases late; returns True when gone."""
    path = Path(path)
    for attempt in range(attempts):
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass
        if not path.exists():
            return True
        if attempt < attempts - 1:
            time.sleep(delay_s)
    return not path.exists()
