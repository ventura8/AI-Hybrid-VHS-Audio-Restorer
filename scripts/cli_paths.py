"""Validators for command-line values that reach a file path or a subprocess.

Every path a script takes on its command line passes through one of these at
argparse time: a value shaped like an option is refused and the path is
resolved, and an input path must exist. Codes such as a language are matched
against a strict pattern.
"""

import argparse
import re
from pathlib import Path

LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$")


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
