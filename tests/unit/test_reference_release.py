"""The reference suppressor's programme release (`apl_reference_release_db`, `cathar_reference_release_db`, 2026-10-20).

The two masks multiply, so a cell the restoring stage kept as programme lost the source-SNR gain
too (5 of 63 IA hunt clips past the depth line tripped a v4 guard); the release eases the gain
toward 1 where the restored frame stands over the source's print, and off it changes nothing.
"""

from unittest.mock import patch

import numpy as np
import pytest

from modules import post_neural_suppress, reference_suppress, spectral_suppress, utils
from modules.blend_weights import FRAME
from tests.unit.test_reference_suppress import SETTINGS, _band_db, _leaky_pair

BINS = FRAME // 2 + 1
INTERIOR = slice(1, BINS - 1)


@pytest.fixture(autouse=True)
def _quiet_log(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "LOG_FILE", tmp_path / "session_log.txt")
    monkeypatch.delenv("AI_RESTORE_EVENT_LOG", raising=False)


def _suppressor(release_db=None):
    noise = np.ones(BINS)
    if release_db is None:
        return spectral_suppress.ReferenceSuppressor(noise, 1.0, -20.0, 0.96)
    return spectral_suppress.ReferenceSuppressor(noise, 1.0, -20.0, 0.96, release_db)


def _frames(level, count=4):
    """`count` one-channel frames whose every bin holds `level` times the print's power."""
    return np.full((count, 1, BINS), np.sqrt(level), dtype=np.complex128)


def _gains(release_db, restored_level, reference_level=1.5):
    """The gain each bin of the last frame took: output over input."""
    restored = _frames(restored_level)
    out = _suppressor(release_db).apply_pair(_frames(reference_level), restored)
    return np.abs(out[-1, 0]) / np.abs(restored[-1, 0])


def test_off_the_gain_is_the_estimator_s_own_bit_for_bit():
    reference, restored = _frames(1.5), _frames(100.0)
    plain = _suppressor()
    expected = np.stack([frame * plain._gain(np.mean(np.abs(ref) ** 2, axis=0))[None, :] for ref, frame in zip(reference, restored)])
    assert np.array_equal(_suppressor(0.0).apply_pair(reference, restored), expected)


def test_a_restored_frame_well_over_the_print_keeps_its_level():
    assert np.allclose(_gains(3.0, restored_level=100.0), 1.0)


def test_a_restored_frame_at_the_print_keeps_the_whole_suppression():
    assert np.allclose(_gains(3.0, restored_level=1.0), _gains(0.0, restored_level=1.0))
    assert np.max(_gains(0.0, restored_level=1.0)) < 0.5


def test_halfway_up_the_span_the_gain_is_eased_halfway_in_db():
    halfway = 10.0 ** ((3.0 + spectral_suppress.RELEASE_SPAN_DB / 2.0) / 10.0)
    eased, whole = _gains(3.0, restored_level=halfway)[INTERIOR], _gains(0.0, restored_level=halfway)[INTERIOR]
    assert np.allclose(eased, np.sqrt(whole))


def test_on_the_leaky_pair_the_release_keeps_the_gaps_clean_and_the_programme(tmp_path):
    source, restored, programme, gaps = _leaky_pair(tmp_path)
    output = spectral_suppress.suppress_by_reference(source, restored, tmp_path / "out.wav", **SETTINGS, release_db=3.0)
    assert _band_db(output, 3000, 16000, gaps) < _band_db(restored, 3000, 16000, gaps) - 12.0
    assert abs(_band_db(output, 100, 3000, ~gaps) - _band_db(programme, 100, 3000, ~gaps)) < 0.5


def test_the_settings_carry_the_release_only_when_it_is_on(tmp_path):
    assert reference_suppress.with_release(SETTINGS, 0.0) == SETTINGS
    assert reference_suppress.with_release(SETTINGS, 3) == {**SETTINGS, "release_db": 3.0}
    off, on = (reference_suppress.target_for(tmp_path / "r.wav", tmp_path, s) for s in (SETTINGS, {**SETTINGS, "release_db": 3.0}))
    assert (off.name != on.name, off == reference_suppress.target_for(tmp_path / "r.wav", tmp_path, dict(SETTINGS))) == (True, True)


@pytest.mark.parametrize("release, expected", [(0.0, None), (6.0, 6.0)])
def test_auto_pure_linear_hands_its_release_to_the_suppressor(monkeypatch, release, expected):
    monkeypatch.setattr(post_neural_suppress, "APL_REFERENCE_RELEASE_DB", release)
    assert post_neural_suppress._reference_settings().get("release_db") == expected


def test_cathar_hands_its_release_to_the_suppressor(tmp_path, monkeypatch):
    from modules.modes import cathar as cathar_mode

    monkeypatch.setattr(cathar_mode, "CATHAR_REFERENCE_SUPPRESS", True)
    monkeypatch.setattr(cathar_mode, "CATHAR_REFERENCE_RELEASE_DB", 3.0)
    with patch.object(reference_suppress, "suppress", return_value=tmp_path / "ref.wav") as run:
        cathar_mode.reference_suppressed(tmp_path / "s.wav", tmp_path / "r.wav", tmp_path)
    run.assert_called_once_with(tmp_path / "s.wav", tmp_path / "r.wav", tmp_path, {**reference_suppress.CATHAR_SETTINGS, "release_db": 3.0})


def test_both_releases_ship_off():
    from modules import config as cfg

    assert (cfg.CONFIG["apl_reference_release_db"], cfg.CONFIG["cathar_reference_release_db"]) == (0.0, 0.0)
