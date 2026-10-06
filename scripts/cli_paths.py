"""Validators for command-line values that reach a file path or a subprocess.

Every path a script takes on its command line passes through one of these at
argparse time: a value shaped like an option is refused and the path is
resolved, and an input path must exist. Codes such as a language are matched
against a strict pattern.
"""

import argparse
import os
import re
import tempfile
from pathlib import Path

LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9_.]+$")
URL_RE = re.compile(r"^https?://[^\s]+$")
REPO_ROOT = Path(__file__).resolve().parents[1]
# Extra directories a script may read or write outside the repository (local tapes on another
# drive, a corpus elsewhere), separated by os.pathsep. The repository and the temp directory
# are always allowed.
DATA_ROOTS_ENV = "AI_RESTORE_DATA_ROOTS"


def _resolved(value):
    """The value resolved to an absolute path; a value shaped like an option is refused."""
    text = str(value)
    if not text or text.startswith("-"):
        raise argparse.ArgumentTypeError(f"looks like an option, not a path: {text!r}")
    return Path(text).resolve()


def path_arg(value):
    """An argparse type: a path that may not exist yet (an output), resolved."""
    return _resolved(value)


def existing_path_arg(value):
    """An argparse type: a path that must exist, resolved."""
    resolved = _resolved(value)
    if not resolved.exists():
        raise argparse.ArgumentTypeError(f"does not exist: {resolved}")
    return resolved


def language_arg(value):
    """An argparse type: a two- or three-letter language code."""
    text = str(value)
    if not LANGUAGE_RE.match(text):
        raise argparse.ArgumentTypeError(f"not a two- or three-letter language code: {text!r}")
    return text


def checked_path(value, what, must_exist=False):
    """The same checks outside argparse: SystemExit names the offending value."""
    try:
        return existing_path_arg(value) if must_exist else path_arg(value)
    except argparse.ArgumentTypeError as error:
        raise SystemExit(f"{what}: {error}") from error


def allowed_roots():
    """The directories a command-line path may resolve under: the repository, the temp dir, $AI_RESTORE_DATA_ROOTS."""
    extra = [Path(entry).resolve() for entry in os.environ.get(DATA_ROOTS_ENV, "").split(os.pathsep) if entry.strip()]
    return (REPO_ROOT, Path(tempfile.gettempdir()).resolve(), *extra)


def confined_path(value, what, must_exist=False):
    """A command-line path checked where it is used: not shaped like an option, resolved, and inside an allowed root.

    Validation at argparse time is invisible to a taint analysis that starts at `parse_args()`;
    this is called on the value right before it reaches a file or a subprocess. A path outside
    the repository and the temp directory is refused unless its root is listed in
    $AI_RESTORE_DATA_ROOTS; SystemExit names the offending value.
    """
    resolved = checked_path(value, what, must_exist=must_exist)
    roots = allowed_roots()
    if not any(resolved.is_relative_to(root) for root in roots):
        raise SystemExit(f"{what} must lie inside {', '.join(str(root) for root in roots)} (or a root in ${DATA_ROOTS_ENV}): {value}")
    return resolved


def checked_number(value, what, minimum=0, maximum=None, kind=float):
    """A number from the command line checked where it reaches a subprocess: converted and inside its range.

    The value is rebuilt with `kind(...)`, so what reaches the command line is a number,
    never the raw string; SystemExit names the offending value.
    """
    try:
        number = kind(value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"{what} is not a number: {value!r}") from error
    _require_range(number, what, minimum, maximum)
    return number


def _require_range(number, what, minimum, maximum):
    upper = float("inf") if maximum is None else maximum
    if not minimum <= number <= upper:
        raise SystemExit(f"{what} must lie between {minimum} and {upper}: {number!r}")


def checked_token(value, what):
    """A name from the command line (a variant, a configuration) checked where it reaches a subprocess."""
    text = str(value)
    if not TOKEN_RE.match(text):
        raise SystemExit(f"{what} must be letters, digits, '_' or '.': {text!r}")
    return text


def checked_url(value, what):
    """A stream URL checked where it reaches a subprocess: http(s), no whitespace, never option-shaped."""
    text = str(value)
    if not URL_RE.match(text):
        raise SystemExit(f"{what} must be an http(s) URL: {text!r}")
    return text
