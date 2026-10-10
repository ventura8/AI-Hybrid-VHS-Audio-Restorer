"""The pre-cache fallbacks log at WARNING, so the stage cache never freezes one (`utils.problem_count` moves).

A render during which a stage fell back is not stored: a fresh render may not fall back, and
a replay of the degraded one would then differ from it. Each fallback below used to hand its
input on quietly (or at INFO or DEBUG); the chain stages' own "after failure" lines are pinned
by their modules' tests.
"""

from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from modules import cathar, filters, hum_cancel, processing, tone_cancel, utils


@pytest.fixture(name="counted")
def counted_fixture(tmp_path, monkeypatch):
    """The problem count before the test, with the session log kept in the test folder."""
    monkeypatch.setattr(utils, "LOG_FILE", tmp_path / "session_log.txt")
    return utils.problem_count()


def _wav(path, seconds=0.5):
    sf.write(str(path), np.zeros((int(44100 * seconds), 2), dtype=np.float32), 44100, subtype="FLOAT")
    return path


def test_an_analysis_read_that_fails_warns(tmp_path, counted):
    """Every stage's "unreadable" (hum, tones, plosives, the subtraction's margin and tonality, the repair's gate) starts here."""
    assert filters._read_stereo_audio_for_analysis(tmp_path / "missing.wav") == (None, None)
    assert utils.problem_count() == counted + 1


@pytest.mark.parametrize(
    ("module", "switch"),
    [(hum_cancel, "APL_ENABLE_HUM_CANCEL"), (tone_cancel, "APL_ENABLE_TONE_CANCEL")],
)
def test_a_cancel_stage_skipped_as_unreadable_after_a_failed_read_moves_the_count(tmp_path, counted, monkeypatch, module, switch):
    """Their own "Skipped: unreadable" stays at INFO (a recording too short to scan says it too); the read's warning counts."""
    monkeypatch.setattr(module, switch, True)
    missing = tmp_path / "missing.wav"
    assert module.apply_when_needed(missing, tmp_path) == missing
    assert utils.problem_count() > counted


def test_a_stem_that_cannot_be_rewritten_as_float_warns(tmp_path, counted):
    stem = _wav(tmp_path / "stem.wav")
    with patch.object(processing.sf, "info", side_effect=RuntimeError("locked")):
        assert processing._ensure_float_pcm(stem) == stem
    assert utils.problem_count() == counted + 1


def test_a_filter_that_wrote_no_audio_hands_its_input_on_with_a_warning(tmp_path, counted):
    source = _wav(tmp_path / "in.wav")
    with patch.object(processing, "run_command_with_progress"):
        assert processing._run_dsp_filter_file(source, tmp_path / "out.wav", "anull", "Polishing", 0.5) == source
    assert utils.problem_count() == counted + 1


def test_an_input_that_is_not_audio_skips_the_surgical_notch_with_a_warning(tmp_path, counted):
    missing = tmp_path / "missing.wav"
    assert processing._pre_denoise_surgical_step(missing, tmp_path) == missing
    assert utils.problem_count() == counted + 1


def test_a_quiet_window_search_that_fails_warns(tmp_path, counted):
    """The subtraction's noise print would come from the opening instead of the quietest window."""
    with patch.object(cathar, "_read_mono_samples", side_effect=MemoryError()):
        assert cathar._find_quiet_window(_wav(tmp_path / "in.wav")) == 0.0
    assert utils.problem_count() == counted + 1
