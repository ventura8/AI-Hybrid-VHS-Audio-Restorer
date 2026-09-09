"""Shared opt-in fixtures for physical hardware validation."""

import os
from pathlib import Path

import pytest

HARDWARE_TESTS_ENABLED = os.environ.get("AI_RESTORE_HARDWARE_TESTS") == "1"

# Gate collection, not execution. These modules import the ml dependency group
# (librosa, by way of scripts.run_hardware_validation), which CI deliberately does not
# install -- it syncs only main and dev. A skip marker applied during
# pytest_collection_modifyitems runs far too late: the module is already imported by
# then, so the run dies with ModuleNotFoundError before any marker is consulted.
# Ignoring the files outright is what keeps `pytest tests/` working on a machine that
# has no GPU stack at all.
#
# The previous guard skipped items carrying the `physical_hardware` marker, but no test
# in this directory ever carried it, so nothing was gated and these tests ran wherever
# their imports happened to resolve.
collect_ignore_glob = [] if HARDWARE_TESTS_ENABLED else ["test_*.py"]


@pytest.fixture
def audio_matrix_dir():
    """Return the conventional generated-fixture directory."""
    return Path("artifacts/audio-matrix")
