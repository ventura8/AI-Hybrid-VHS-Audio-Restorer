"""Shared opt-in fixtures for physical hardware validation."""

import os
from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip real hardware runs unless explicitly enabled by the operator."""
    del config
    if os.environ.get("AI_RESTORE_HARDWARE_TESTS") == "1":
        return
    marker = pytest.mark.skip(reason="Set AI_RESTORE_HARDWARE_TESTS=1 for physical hardware validation.")
    allowed = ("test_longform_stress.py", "test_mid_validation.py", "test_short_smoke.py")
    for item in items:
        fspath_str = str(item.fspath)
        if ("tests\\hardware" in fspath_str or "tests/hardware" in fspath_str) and not any(a in fspath_str for a in allowed):
            item.add_marker(marker)


@pytest.fixture
def audio_matrix_dir():
    """Return the conventional generated-fixture directory."""
    return Path("artifacts/audio-matrix")
