"""Tests for the physical tape damage repair stage.

The stage exists because `auto_pure_linear` repaired none of it: clicks, dropouts,
saturation and azimuth skew all survived the mode. Four cathar stages earned a place against
paired fixtures and three were rejected for making undamaged material worse, so what these
tests pin is the gating -- a stage that runs on material without its defect is the failure
mode, not an inefficiency. Applied blanket-fashion `decrackle` scores -11.09 dB on material
whose only defect is azimuth skew.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from modules import physical_repair

RATE = 44100


@pytest.fixture(name="quiet_wav")
def _quiet_wav(tmp_path):
    """A stand-in path; gating tests patch detection rather than reading audio."""
    wav = tmp_path / "source.wav"
    wav.write_text("audio")
    return wav


def _speech_like(tmp_path, name, mutes=(), pause=False, bias=0.0):
    """Writes running audio, optionally punched through with mutes, given a long pause, or DC-biased."""
    rng = np.random.default_rng(7)
    samples = rng.normal(0.0, 0.15, RATE * 4).astype(np.float32)
    for start_s in mutes:
        start = int(start_s * RATE)
        stop = start + RATE // 20
        samples[start:stop] = 0.0
    if pause:
        pause_end = RATE * 2
        samples[RATE:pause_end] = 0.0
    samples = samples + np.float32(bias)
    path = tmp_path / name
    sf.write(str(path), samples, RATE, subtype="FLOAT")
    return path


def test_a_dropout_is_detected(tmp_path):
    """A short hole punched through running audio is what the inpainter is for."""
    assert physical_repair.detect_mute_spans(_speech_like(tmp_path, "drop.wav", mutes=(1.0, 2.0))) == 2


def test_a_dropout_on_a_biased_capture_is_still_detected(tmp_path):
    """A digitiser's DC bias puts the dropout at the bias level, not at zero.

    On a worn-tape fixture carrying both faults every dropout went uncounted: the hole sat
    at 0.03, some 30 dB above the silence threshold, until the bias was taken out first.
    """
    assert physical_repair.detect_mute_spans(_speech_like(tmp_path, "biased.wav", mutes=(1.0, 2.0), bias=0.03)) == 2


def test_running_audio_reports_no_dropout(tmp_path):
    """Undamaged material must not trigger the stage."""
    assert physical_repair.detect_mute_spans(_speech_like(tmp_path, "clean.wav")) == 0


def test_a_long_silence_is_not_a_dropout(tmp_path):
    """A pause between phrases is silence too, and filling it would invent audio.

    Depth alone cannot separate the two: synthesised speech carries genuine digital silence
    between phrases, and a depth-only test found fourteen of them in every clean reference.
    """
    assert physical_repair.detect_mute_spans(_speech_like(tmp_path, "pause.wav", pause=True)) == 0


def test_unreadable_audio_reports_nothing(tmp_path):
    """An unmeasurable input is skipped rather than guessed at."""
    with patch.object(physical_repair, "_read_audio_for_analysis", return_value=(None, 0)):
        assert physical_repair.detect_mute_spans(tmp_path / "missing.wav") == 0


def test_the_stage_is_switchable(quiet_wav, tmp_path):
    """The whole stage can be switched off without touching the chain."""
    with patch.object(physical_repair, "APL_ENABLE_PHYSICAL_REPAIR", False):
        assert physical_repair.apply_when_needed(quiet_wav, tmp_path) == quiet_wav


def test_undamaged_audio_is_returned_untouched(quiet_wav, tmp_path):
    """Nothing detected means nothing runs, and the chain continues with its input."""
    with patch.object(physical_repair, "detect_mute_spans", return_value=0):
        assert physical_repair.apply_when_needed(quiet_wav, tmp_path, strategy={"profile": {}}) == quiet_wav


def test_each_stage_is_gated_on_its_own_defect(quiet_wav):
    """Only the detected defects are scheduled; the rest are not run at all."""
    strategy = {"profile": {"has_clicks": True, "has_clipping": False, "azimuth_delay_ms": 0.0}}
    with patch.object(physical_repair, "detect_mute_spans", return_value=0):
        plan = physical_repair._stage_plan(quiet_wav, strategy)
    assert plan == {"depop": True, "decrackle": True, "declip": False, "azimuth": False, "inpaint": False}


def test_pop_removal_runs_ahead_of_decrackle_and_can_be_switched_off(quiet_wav):
    """The pop stage shares decrackle's gate, runs first, and a zero threshold leaves it out."""
    strategy = {"profile": {"has_clicks": True}}
    with patch.object(physical_repair, "detect_mute_spans", return_value=0):
        plan = physical_repair._stage_plan(quiet_wav, strategy)
        assert list(plan)[:2] == ["depop", "decrackle"]
        with patch.object(physical_repair, "APL_DEPOP_THRESHOLD", 0.0):
            assert physical_repair._stage_plan(quiet_wav, strategy)["depop"] is False


def test_negligible_azimuth_skew_is_left_alone(quiet_wav):
    """Below the detector's own resolution the correction is noise, not a fix."""
    strategy = {"profile": {"azimuth_delay_ms": 0.01}}
    with patch.object(physical_repair, "detect_mute_spans", return_value=0):
        assert physical_repair._stage_plan(quiet_wav, strategy)["azimuth"] is False


def test_a_malformed_strategy_does_not_decide_a_stage(quiet_wav):
    """Strategy shapes vary across callers, so the gate reads defensively."""
    assert physical_repair._profile_value(None, "has_clicks", False) is False
    assert physical_repair._profile_value({"profile": {"has_clicks": "yes"}}, "has_clicks", False) is False
    assert physical_repair._profile_value({"profile": {"has_clicks": 1}}, "has_clicks", False) is False


def test_an_integer_delay_counts_as_a_number_and_a_boolean_does_not(quiet_wav):
    """A profile that carries the skew as 1 rather than 1.0 still triggers the repair; True does not."""
    assert physical_repair._profile_value({"profile": {"azimuth_delay_ms": 1}}, "azimuth_delay_ms", 0.0) == 1
    assert physical_repair._profile_value({"profile": {"azimuth_delay_ms": True}}, "azimuth_delay_ms", 0.0) == 0.0
    with patch.object(physical_repair, "detect_mute_spans", return_value=0):
        assert physical_repair._stage_plan(quiet_wav, {"profile": {"azimuth_delay_ms": 1}})["azimuth"] is True


def test_missing_cathar_binary_falls_back_to_the_unrepaired_chain(quiet_wav, tmp_path):
    """A host that never provisioned the CLI must still be able to run this mode."""
    with (
        patch.object(physical_repair, "detect_mute_spans", return_value=3),
        patch("modules.cathar._require_cathar_binary", side_effect=FileNotFoundError("cathar")),
    ):
        assert physical_repair.apply_when_needed(quiet_wav, tmp_path, strategy={"profile": {}}) == quiet_wav


def test_detected_damage_is_repaired(quiet_wav, tmp_path):
    """A capture with clicks gets decrackle, which is the stage that measured best for them."""
    produced = tmp_path / "decrackled.wav"
    produced.write_text("audio")
    strategy = {"profile": {"has_clicks": True}}
    with (
        patch.object(physical_repair, "detect_mute_spans", return_value=0),
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._cathar_decrackle_step", return_value=produced) as mock_decrackle,
        patch("modules.cathar._cathar_declip_step") as mock_declip,
    ):
        assert physical_repair.apply_when_needed(quiet_wav, tmp_path, strategy=strategy) == produced
    mock_decrackle.assert_called_once()
    mock_declip.assert_not_called()


def test_a_failed_repair_leaves_the_audio_usable(quiet_wav, tmp_path):
    """A failure here must not take the restoration down with it."""
    strategy = {"profile": {"has_clicks": True}}
    with (
        patch.object(physical_repair, "detect_mute_spans", return_value=0),
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._cathar_decrackle_step", side_effect=RuntimeError("cathar exploded")),
    ):
        assert physical_repair.apply_when_needed(quiet_wav, tmp_path, strategy=strategy) == Path(quiet_wav)


def test_the_rejected_stages_are_not_wired_in():
    """`repair` and `deplosive` make undamaged material measurably worse.

    Measured on fixtures carrying no physical damage, `repair` scores -15.89 dB and lifts
    injected error to -4.68 dB against the programme; `deplosive` -5.60 and -15.61. Neither
    belongs in a mode whose advantage is that it disturbs the programme less.
    """
    source = Path(physical_repair.__file__).read_text(encoding="utf-8")
    assert "_cathar_repair_step" not in source
    assert "_cathar_deplosive_step" not in source
    assert "_cathar_declick_step" not in source
