"""Opt-in long-form fixture planning checks."""

from scripts.audio_matrix.manifest import load_manifest


def test_longform_fixture_is_sustained_and_has_drift():
    """Long-form runs must exercise chunk and drift handling."""
    fixture = load_manifest()["longform"]
    assert fixture.duration_seconds >= 300
    assert "drift" in fixture.defects
