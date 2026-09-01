"""Unit tests for the object-oriented processing modes package."""

from unittest.mock import MagicMock, patch

import pytest

from modules.modes import (
    ArnndnSpeechMode,
    AutoFFmpegNativeMode,
    AutoPureLinearMode,
    AutoPureMode,
    BaseRestorationMode,
    CatharMode,
    DenoiseOnlyMode,
    FFmpegNativeMode,
    HybridMode,
    MultiPassMode,
    get_mode_instance,
)


def test_base_mode_raises_not_implemented():
    class DummyMode(BaseRestorationMode):
        pass

    with pytest.raises(TypeError):
        DummyMode()


def test_resolve_strategy_val():
    assert BaseRestorationMode.resolve_strategy_val(None, "key", "def") == "def"
    assert BaseRestorationMode.resolve_strategy_val("invalid", "key", "def") == "def"
    assert BaseRestorationMode.resolve_strategy_val({"key": "val"}, "key", "def") == "val"


@pytest.mark.parametrize(
    "mode_key,expected_type",
    [
        ("auto_pure_linear", AutoPureLinearMode),
        ("cathar", CatharMode),
        ("cathar_vhs", CatharMode),
        ("hybrid", HybridMode),
        ("multipass_auto", MultiPassMode),
        ("multipass", MultiPassMode),
        ("auto_pure", AutoPureMode),
        ("pure", AutoPureMode),
        ("denoise_only", DenoiseOnlyMode),
        ("ffmpeg_native", FFmpegNativeMode),
        ("vhs_native", FFmpegNativeMode),
        ("auto_ffmpeg_native", AutoFFmpegNativeMode),
        ("auto_vhs_native", AutoFFmpegNativeMode),
        ("arnndn_speech", ArnndnSpeechMode),
        ("unknown_mode", HybridMode),
    ],
)
def test_registry_resolution(mode_key, expected_type):
    assert isinstance(get_mode_instance(mode_key), expected_type)


def test_auto_pure_linear_mode_execute(tmp_path):
    mode = AutoPureLinearMode()
    work_dir = tmp_path / "work"
    orig = tmp_path / "orig.wav"
    vid = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"

    with (
        patch("modules.processing._resolve_preconditioned_audio", return_value=(orig, {})) as mock_pre,
        patch("modules.processing._process_single_track_pipeline") as mock_pipeline,
        patch("modules.processing._denoise_and_polish_full_audio_step", return_value=orig) as mock_denoise,
    ):
        mode.execute(work_dir, orig, vid, out, 20.0, strategy={"denoise_model": "test.pth"})
        mock_pre.assert_called_once()
        mock_pipeline.assert_called_once()
        # Verify step callable passed to single track pipeline
        step_func = mock_pipeline.call_args[0][5]
        step_func(orig, work_dir)
        mock_denoise.assert_called_once()


def test_cathar_mode_execute(tmp_path):
    mode = CatharMode()
    work_dir = tmp_path / "work"
    orig = tmp_path / "orig.wav"
    vid = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"

    with (
        patch("modules.processing._resolve_preconditioned_audio", return_value=(orig, {})) as mock_pre,
        patch("modules.processing._process_single_track_pipeline") as mock_pipeline,
        patch("modules.filters.filter_cathar_vhs_pipeline", return_value=orig) as mock_cathar,
        patch("modules.processing._polish_full_audio_step", return_value=orig) as mock_polish,
    ):
        mode.execute(work_dir, orig, vid, out, 20.0, strategy={})
        mock_pre.assert_called_once()
        mock_pipeline.assert_called_once()
        step_func = mock_pipeline.call_args[0][5]
        step_func(orig, work_dir)
        mock_cathar.assert_called_once()
        mock_polish.assert_called_once()


def test_hybrid_mode_execute(tmp_path):
    mode = HybridMode()
    work_dir = tmp_path / "work"
    orig = tmp_path / "orig.wav"
    vid = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"

    with (
        patch("modules.processing._resolve_preconditioned_audio", return_value=(orig, {})) as mock_pre,
        patch("modules.processing._execute_hybrid_restoration") as mock_exec,
    ):
        mode.execute(work_dir, orig, vid, out, 20.0)
        mock_pre.assert_called_once()
        mock_exec.assert_called_once()


def test_multipass_mode_execute(tmp_path):
    mode = MultiPassMode()
    work_dir = tmp_path / "work"
    orig = tmp_path / "orig.wav"
    vid = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"

    with (
        patch("modules.processing._resolve_preconditioned_audio", return_value=(orig, {})) as mock_pre,
        patch("modules.processing._execute_hybrid_restoration") as mock_exec,
    ):
        mode.execute(work_dir, orig, vid, out, 20.0)
        mock_pre.assert_called_once()
        mock_exec.assert_called_once()


def test_auto_pure_mode_execute(tmp_path):
    mode = AutoPureMode()
    work_dir = tmp_path / "work"
    orig = tmp_path / "orig.wav"
    vid = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"

    with (
        patch("modules.processing._resolve_preconditioned_audio", return_value=(orig, {})) as mock_pre,
        patch("modules.processing._execute_pure_restoration") as mock_exec,
    ):
        mode.execute(work_dir, orig, vid, out, 20.0)
        mock_pre.assert_called_once()
        mock_exec.assert_called_once()


def test_denoise_only_mode_execute(tmp_path):
    mode = DenoiseOnlyMode()
    work_dir = tmp_path / "work"
    orig = tmp_path / "orig.wav"
    vid = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"

    with (
        patch("modules.processing._resolve_preconditioned_audio", return_value=(orig, {})) as mock_pre,
        patch("modules.processing._process_single_track_pipeline") as mock_pipeline,
        patch("modules.processing._denoise_and_polish_full_audio_step", return_value=orig) as mock_denoise,
    ):
        mode.execute(work_dir, orig, vid, out, 20.0)
        mock_pre.assert_called_once()
        mock_pipeline.assert_called_once()
        step_func = mock_pipeline.call_args[0][5]
        step_func(orig, work_dir)
        mock_denoise.assert_called_once()


def test_native_dsp_modes_execute(tmp_path):
    ffmpeg_mode = FFmpegNativeMode()
    auto_ffmpeg_mode = AutoFFmpegNativeMode()
    work_dir = tmp_path / "work"
    orig = tmp_path / "orig.wav"
    vid = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"

    with patch("modules.processing._process_single_track_pipeline") as mock_pipeline:
        ffmpeg_mode.execute(work_dir, orig, vid, out, 20.0)
        assert mock_pipeline.call_count == 1
        auto_ffmpeg_mode.execute(work_dir, orig, vid, out, 20.0)
        assert mock_pipeline.call_count == 2


def test_arnndn_speech_mode_execute(tmp_path):
    mode = ArnndnSpeechMode()
    work_dir = tmp_path / "work"
    orig = tmp_path / "orig.wav"
    vid = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"

    with (
        patch("modules.processing._resolve_preconditioned_audio", return_value=(orig, {})) as mock_pre,
        patch("modules.processing._process_single_track_pipeline") as mock_pipeline,
        patch("modules.processing._bind_step_model", return_value=MagicMock()) as mock_bind,
    ):
        mode.execute(work_dir, orig, vid, out, 20.0)
        mock_pre.assert_called_once()
        mock_bind.assert_called_once()
        mock_pipeline.assert_called_once()
