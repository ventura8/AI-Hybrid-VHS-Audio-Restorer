"""Temp and file hygiene shared by every stage.

Every stage writes through a same-directory partial and publishes it with one rename: a
power cut or a kill part-way through leaves a partial, never a target that reads as a
finished file. A WAV truncated mid-stream keeps a valid header, so the validity check
that lets a rerun resume would otherwise take the fragment for the whole.

While a video is processed, every temp-file API is pointed inside its work directory so
library scratch leaves with the work directory rather than accumulating in the system
temp.

The work directory sits beside the tape, a path a folder of tapes can plant: shipped as a
link (`.temp_work_<stem>` pointing at the user's home), the partial sweep walked the tree it
points to and unlinked every `*.tmp*` and hidden `*.wav` there, and the stages wrote into
it (security review, 2026-10-26). `open_work_dir` refuses a link or a non-directory at the
path, the sweep never follows a link, and a stale partial that is a dangling link is
removed rather than written through. A real work directory can carry links too: one named
like a partial (`original.tmp.wav` pointing at a file the user owns) was skipped by the sweep
and the extraction's `ffmpeg -y` wrote through it (security review of PR #21, 2026-10-29).
The app makes no links, so the sweep removes every link it meets inside the work directory,
the link itself and never what it points to, and a stage that writes its partial with ffmpeg
removes a stale one first (`remove_stale`).
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


class WorkDirRefused(OSError):
    """A work directory's path holds a link or something other than a directory this run can own."""


def is_link(path):
    """Whether `path` is a symbolic link or a Windows directory junction: either leads out of the tree it sits in."""
    path = Path(path)
    return path.is_symlink() or path.is_junction()


def open_work_dir(work_dir):
    """Creates `work_dir`, or reopens one an earlier run left, and returns it; raises `WorkDirRefused` when the path is
    a link or not a directory, before and after the mkdir, so nothing is swept or written inside a tree it leads to."""
    work_dir = Path(work_dir)
    if is_link(work_dir):
        raise WorkDirRefused(f"{work_dir.name} is a link; nothing is written inside the tree it points to")
    work_dir.mkdir(exist_ok=True)
    if is_link(work_dir) or not work_dir.is_dir():
        raise WorkDirRefused(f"{work_dir.name} is not a directory this run can own")
    return work_dir


def remove_stale(path):
    """Removes `path`, a link included and a dangling one too: the next write would otherwise go through it."""
    path = Path(path)
    if is_link(path):
        _remove_link(path)
    elif path.exists():
        path.unlink()


def _remove_link(path):
    """Removes a link entry itself, never what it points to: a file link by unlink, a directory link or a junction by
    rmdir where the platform refuses to unlink one (Windows)."""
    try:
        path.unlink()
    except (IsADirectoryError, PermissionError):
        os.rmdir(path)


def _finish_partial(partial, target, completed):
    """Publishes a completed partial that was written; removes the partial of a failed stage."""
    if not completed:
        remove_stale(partial)
    elif partial.exists():
        publish(partial, target)


@contextlib.contextmanager
def atomic_target(target):
    """Yields the partial to write; publishes it over `target` on success, removes it otherwise.

    A stage that writes nothing (it returned before opening the partial) publishes nothing.
    """
    target = Path(target)
    partial = partial_path(target)
    remove_stale(partial)
    completed = False
    try:
        yield partial
        completed = True
    finally:
        _finish_partial(partial, target, completed)


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


def _remove_swept(path, removed, remove, what):
    try:
        remove(path)
        removed.append(path)
    except OSError as exc:
        log_msg(f"    [Hygiene] Could not remove {what} {path.name}: {exc}", is_error=True)


def _real_dirs(root, names):
    """The subdirectory names under `root` that are not links, so a walk stays inside the tree."""
    return [name for name in names if not is_link(Path(root) / name)]


def _swept_under(work_dir):
    """`(partials, links)` below a directory, walked without following a link: every unpublished partial that is a
    real file, and every link, whatever its name; nothing when the directory does not exist or is itself a link."""
    if is_link(work_dir) or not work_dir.is_dir():
        return [], []
    partials, links = [], []
    for root, dirs, files in os.walk(work_dir, followlinks=False):
        links += _links_among(root, dirs + files)
        dirs[:] = _real_dirs(root, dirs)
        partials += _partial_files(root, files)
    return partials, links


def _links_among(root, names):
    """The entries under `root` that are links: the app makes none, so each was planted."""
    return [path for path in (Path(root) / name for name in names) if is_link(path)]


def _partial_files(root, names):
    """The files under `root` named as partials, links left out (the sweep removes those as links)."""
    paths = [Path(root) / name for name in names]
    return [path for path in paths if is_partial(path) and not is_link(path)]


def sweep_partials(work_dir):
    """Removes every unpublished partial and every link under a work directory; returns the paths removed.

    Run when a work directory is (re)opened, so a resumed restoration never globs a
    fragment from the interrupted run as a finished stage output, and no stage writes
    through a link a folder of tapes planted inside it. A link is removed, never followed.
    """
    removed = []
    partials, links = _swept_under(Path(work_dir))
    for path in partials:
        _remove_swept(path, removed, Path.unlink, "partial")
    for path in links:
        _remove_swept(path, removed, _remove_link, "link")
    if removed:
        log_msg(f"  [Hygiene] Removed {len(removed)} unfinished partial file(s) or link(s) from the previous run.")
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
