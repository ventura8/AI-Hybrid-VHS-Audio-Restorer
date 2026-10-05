"""Unit tests for modules.version."""

import importlib.metadata
from unittest.mock import patch

import modules.version
from modules.version import (
    __version__,
    _read_version_from_metadata,
    _read_version_from_pyproject,
    get_version,
)


def test_read_version_from_pyproject_valid(tmp_path):
    """Test reading valid version from pyproject.toml."""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_text('[project]\nversion = "2.4.6"\n', encoding="utf-8")
    assert _read_version_from_pyproject(toml_file) == "2.4.6"


def test_read_version_from_pyproject_nonexistent(tmp_path):
    """Test reading from nonexistent path returns None."""
    missing_file = tmp_path / "missing.toml"
    assert _read_version_from_pyproject(missing_file) is None


def test_read_version_from_pyproject_invalid_toml(tmp_path):
    """Test reading invalid TOML content returns None."""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_text("invalid [[ toml", encoding="utf-8")
    assert _read_version_from_pyproject(toml_file) is None


def test_read_version_from_pyproject_missing_version(tmp_path):
    """Test pyproject.toml without version field returns None."""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_text('[project]\nname = "pkg"\n', encoding="utf-8")
    assert _read_version_from_pyproject(toml_file) is None


def test_read_version_from_pyproject_blank_version(tmp_path):
    """Test pyproject.toml with blank version field returns None."""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_text('[project]\nversion = "   "\n', encoding="utf-8")
    assert _read_version_from_pyproject(toml_file) is None


def test_read_version_from_metadata_success():
    """Test reading version from package metadata successfully."""
    with patch("importlib.metadata.version", return_value="3.1.4"):
        assert _read_version_from_metadata("sample-pkg") == "3.1.4"


def test_read_version_from_metadata_not_found():
    """Test reading version when package is not found returns None."""
    with patch(
        "importlib.metadata.version",
        side_effect=importlib.metadata.PackageNotFoundError("sample-pkg"),
    ):
        assert _read_version_from_metadata("sample-pkg") is None


def test_read_version_from_metadata_blank():
    """Test reading version when metadata returns empty string returns None."""
    with patch("importlib.metadata.version", return_value="  "):
        assert _read_version_from_metadata("sample-pkg") is None


def test_get_version_from_pyproject(tmp_path):
    """Test get_version prioritizes pyproject.toml."""
    get_version.cache_clear()
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_text('[project]\nversion = "1.9.9"\n', encoding="utf-8")

    with patch.object(modules.version, "Path") as mock_path:
        mock_path.return_value.resolve.return_value.parent.parent.__truediv__.return_value = toml_file
        version = get_version()
        assert version == "1.9.9"
    get_version.cache_clear()


def test_get_version_fallback_to_metadata():
    """Test get_version falls back to importlib.metadata when pyproject is missing."""
    get_version.cache_clear()
    with patch("modules.version._read_version_from_pyproject", return_value=None):
        with patch("modules.version._read_version_from_metadata", return_value="2.0.1"):
            assert get_version() == "2.0.1"
    get_version.cache_clear()


def test_get_version_fallback_to_default():
    """Test get_version falls back to 0.0.0 when all sources fail."""
    get_version.cache_clear()
    with patch("modules.version._read_version_from_pyproject", return_value=None):
        with patch("modules.version._read_version_from_metadata", return_value=None):
            assert get_version() == "0.0.0"
    get_version.cache_clear()


def test_module_version_exports():
    """Test that __version__ matches current pyproject.toml version."""
    get_version.cache_clear()
    assert __version__ == "1.3.3"
    assert modules.version.get_version() == "1.3.3"
