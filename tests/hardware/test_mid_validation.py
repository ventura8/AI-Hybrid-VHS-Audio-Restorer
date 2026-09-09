"""Opt-in mid-duration fixture metadata checks."""

from scripts.audio_matrix.manifest import load_manifest


def test_mid_manifest_contains_required_analog_defects():
    """Mid validation retains the signals needed by native filter checks."""
    defects = load_manifest()["mid"].defects
    assert {"hum", "whistle", "rumble", "azimuth"}.issubset(defects)
