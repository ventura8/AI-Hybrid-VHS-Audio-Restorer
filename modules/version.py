"""Single source of truth for the application version."""

import functools
import importlib.metadata
import tomllib
from pathlib import Path


def _read_version_from_pyproject(path: Path) -> str | None:
    """Reads the version string from pyproject.toml if present."""
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        version = data.get("project", {}).get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return None


def _read_version_from_metadata(pkg_name: str) -> str | None:
    """Reads the version string from installed package metadata."""
    try:
        version = importlib.metadata.version(pkg_name)
        if version and version.strip():
            return version.strip()
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return None
    return None


@functools.cache
def get_version() -> str:
    """Resolves the application version with pyproject.toml as primary source."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    from_file = _read_version_from_pyproject(pyproject_path)
    if from_file is not None:
        return from_file

    from_metadata = _read_version_from_metadata("ai-hybrid-vhs-audio-restorer")
    if from_metadata is not None:
        return from_metadata

    return "0.0.0"


__version__ = get_version()
