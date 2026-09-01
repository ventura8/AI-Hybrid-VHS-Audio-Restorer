"""Unit tests for native DSP processing, remuxing, stem combining, and audio polish.

Tests coverage across:
- Output suffix resolution across all supported processing modes.
- Audio codec and container parameter determination (PCM, AAC, MP2).
- FFmpeg process command construction and execution with retry logic.
- ARNNDN neural speech filtering, temporary file management, and error handling.
- Smart audio synchronization and atomic output file promotion.
- Dynamic speech de-essing, background expansion, and loudness normalization.
- Multi-stream container mapping for archival audio track preservation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import modules.filters
import modules.processing


@pytest.mark.parametrize(
    "mode,expected_suffix",
    [
        ("auto", "_Auto_Cleaned"),
        ("multipass_auto", "_MultiPass_Cleaned"),
        ("multipass", "_MultiPass_Cleaned"),
        ("auto_pure", "_Pure_Cleaned"),
        ("pure", "_Pure_Cleaned"),
        ("hybrid", "_Hybrid_Cleaned"),
        ("denoise_only", "_Denoised_Cleaned"),
        ("ffmpeg_native", "_FFmpeg_Cleaned"),
        ("auto_ffmpeg_native", "_AutoFFmpeg_Cleaned"),
        ("vhs_native", "_FFmpeg_Cleaned"),
        ("auto_vhs_native", "_AutoFFmpeg_Cleaned"),
        ("arnndn_speech", "_Speech_Cleaned"),
        ("unknown", "_Hybrid_Cleaned"),
    ],
)
def test_get_output_suffix_all_modes(mode, expected_suffix):
    """Ensure output suffixes match for all process modes."""
    assert modules.processing._get_output_suffix(mode) == expected_suffix


@pytest.mark.parametrize(
    "suffix,expected_args",
    [
        (".mp4", ["-c:a", "aac", "-b:a", "320k"]),
        (".m4v", ["-c:a", "aac", "-b:a", "320k"]),
        (".mpg", ["-c:a", "mp2", "-b:a", "384k"]),
        (".mpeg", ["-c:a", "mp2", "-b:a", "384k"]),
        (".ts", ["-c:a", "aac", "-b:a", "320k"]),
        (".m2ts", ["-c:a", "aac", "-b:a", "320k"]),
        (".avi", ["-c:a", "pcm_s16le"]),
        (".mkv", ["-c:a", "pcm_f32le"]),
        (".mov", ["-c:a", "pcm_f32le"]),
        (".unknown", ["-c:a", "pcm_f32le"]),
    ],
)
def test_get_audio_encoding_args_all_containers(suffix, expected_args):
    """Ensure container-specific audio encoding args are correctly generated."""
    assert modules.processing._get_audio_encoding_args(suffix) == expected_args


def test_build_vhs_native_filter_string():
    """Verify vhs_native filter string construction under various parameter combinations."""
    s_default = modules.filters._build_vhs_native_filter_string(12.0, -45.0, True, 60, True, 0.0)
    assert s_default == "highpass=f=60,adeclick,afftdn=nr=12.0:nf=-45.0:tn=1"

    s_notch = modules.filters._build_vhs_native_filter_string(10.0, -50.0, False, 60, True, 59.94)
    expected_notch = (
        "highpass=f=60,adeclick,bandreject=f=59.94:width_type=q:w=15,bandreject=f=119.88:width_type=q:w=15,afftdn=nr=10.0:nf=-50.0:tn=0"
    )
    assert s_notch == expected_notch

    s_clean = modules.filters._build_vhs_native_filter_string(15.0, -40.0, True, 0, False, 0.0)
    assert s_clean == "afftdn=nr=15.0:nf=-40.0:tn=1"


def test_escape_ffmpeg_filter_path():
    """Verify colon escaping in file paths for FFmpeg filter syntax."""
    raw_path = "C:/models/cb.rnnn"
    expected = Path(raw_path).resolve().as_posix().replace(":", r"\:")
    assert modules.filters._escape_ffmpeg_filter_path(raw_path) == expected
    assert expected.endswith(r"C\:/models/cb.rnnn")


def test_build_arnndn_filter_string():
    """Verify arnndn filter string with highpass, adeclick, and escaped model."""
    s = modules.filters._build_arnndn_filter_string("C:/models/cb.rnnn", highpass_freq=60, enable_adeclick=True)
    assert "highpass=f=60" in s
    assert "adeclick" in s
    assert "arnndn=m=" in s

    s_no_hp = modules.filters._build_arnndn_filter_string("model.rnnn", highpass_freq=0, enable_adeclick=False)
    assert s_no_hp.startswith("arnndn=m=")


def test_build_filter_audio_cmd():
    """Verify filter audio FFmpeg command generation."""
    cmd = modules.filters._build_filter_audio_cmd("in.wav", "out.tmp.wav", "highpass=f=60", 4)
    expected_fragments = [modules.filters.FFMPEG_BIN, "-af", "highpass=f=60", "-threads", "4"]
    assert all(frag in cmd for frag in expected_fragments)


@patch("modules.filters.is_valid_audio", return_value=True)
def test_filter_vhs_native_step_skips_when_valid(mock_valid, tmp_path):
    """Step skips filtering if output is already valid."""
    orig = tmp_path / "orig.wav"
    out_dir = tmp_path / "out_dir"
    out_dir.mkdir()
    result = modules.filters._filter_vhs_native_step(orig, out_dir)
    assert result == out_dir / "vhs_filtered_orig.wav"


@patch("modules.filters.attempt_cpu_run_with_retry")
@patch("modules.filters.is_valid_audio")
def test_filter_vhs_native_step_success(mock_valid, mock_retry, tmp_path):
    """Step runs filter command and promotes valid output."""
    mock_valid.side_effect = [False, True]
    orig = tmp_path / "orig.wav"
    orig.write_text("audio")
    out_dir = tmp_path / "out_dir"
    out_dir.mkdir()

    output_wav = out_dir / "vhs_filtered_orig.wav"
    tmp_wav = output_wav.with_suffix(".tmp.wav")

    def fake_retry(cmd_builder, threads, description, total_duration):
        tmp_wav.write_text("filtered")
        return True

    mock_retry.side_effect = fake_retry
    result = modules.filters._filter_vhs_native_step(orig, out_dir)
    assert result == output_wav
    assert output_wav.exists()


@patch("modules.filters.attempt_cpu_run_with_retry")
@patch("modules.filters.is_valid_audio", return_value=False)
def test_filter_vhs_native_step_fails_on_invalid_output(mock_valid, mock_retry, tmp_path):
    """Step raises RuntimeError if output audio is invalid."""
    orig = tmp_path / "orig.wav"
    out_dir = tmp_path / "out_dir"
    out_dir.mkdir()

    output_wav = out_dir / "vhs_filtered_orig.wav"
    tmp_wav = output_wav.with_suffix(".tmp.wav")
    tmp_wav.write_text("corrupted")

    with pytest.raises(RuntimeError, match="Native VHS filtering failed"):
        modules.filters._filter_vhs_native_step(orig, out_dir)
    assert not tmp_wav.exists()


@patch("modules.filters.is_valid_audio", return_value=True)
def test_filter_arnndn_step_skips_when_valid(mock_valid, tmp_path):
    """ARNNDN step skips if valid output exists."""
    orig = tmp_path / "orig.wav"
    out_dir = tmp_path / "out_dir"
    out_dir.mkdir()
    result = modules.filters._filter_arnndn_step(orig, out_dir)
    assert result == out_dir / "arnndn_filtered_orig.wav"


@patch("modules.filters.is_valid_audio", return_value=False)
def test_filter_arnndn_step_missing_model_raises(mock_valid, tmp_path):
    """ARNNDN step raises FileNotFoundError if model is missing."""
    orig = tmp_path / "orig.wav"
    out_dir = tmp_path / "out_dir"
    out_dir.mkdir()

    with patch("modules.filters._resolve_arnndn_model_path", return_value=Path(tmp_path / "missing.rnnn")):
        with pytest.raises(FileNotFoundError, match="ARNNDN model file not found"):
            modules.filters._filter_arnndn_step(orig, out_dir)


@patch("modules.filters.attempt_cpu_run_with_retry")
@patch("modules.filters.is_valid_audio")
def test_filter_arnndn_step_success(mock_valid, mock_retry, tmp_path):
    """ARNNDN step successfully filters audio and renames tmp output."""
    mock_valid.side_effect = [False, True]
    orig = tmp_path / "orig.wav"
    orig.write_text("audio")
    out_dir = tmp_path / "out_dir"
    out_dir.mkdir()

    fake_model = tmp_path / "cb.rnnn"
    fake_model.write_text("model")

    output_wav = out_dir / "arnndn_filtered_orig.wav"
    tmp_wav = output_wav.with_suffix(".tmp.wav")

    def fake_retry(cmd_builder, threads, description, total_duration):
        tmp_wav.write_text("filtered")
        return True

    mock_retry.side_effect = fake_retry

    with patch("modules.filters._resolve_arnndn_model_path", return_value=fake_model):
        result = modules.filters._filter_arnndn_step(orig, out_dir)
        assert result == output_wav
        assert output_wav.exists()


@patch("modules.filters.attempt_cpu_run_with_retry")
@patch("modules.filters.is_valid_audio", return_value=False)
def test_filter_arnndn_step_fails_on_invalid_output(mock_valid, mock_retry, tmp_path):
    """ARNNDN step raises RuntimeError and cleans tmp file when output is invalid."""
    orig = tmp_path / "orig.wav"
    out_dir = tmp_path / "out_dir"
    out_dir.mkdir()

    fake_model = tmp_path / "cb.rnnn"
    fake_model.write_text("model")

    output_wav = out_dir / "arnndn_filtered_orig.wav"
    tmp_wav = output_wav.with_suffix(".tmp.wav")
    tmp_wav.write_text("bad")

    with patch("modules.filters._resolve_arnndn_model_path", return_value=fake_model):
        with pytest.raises(RuntimeError, match="ARNNDN Speech filtering failed"):
            modules.filters._filter_arnndn_step(orig, out_dir)
    assert not tmp_wav.exists()


@patch("modules.processing._final_mux_single_audio_step")
@patch("modules.processing._align_stems")
@patch("modules.processing._filter_vhs_native_step")
def test_process_ffmpeg_native_mode(mock_filter, mock_align, mock_mux, tmp_path):
    """Verify ffmpeg_native mode pipeline orchestrates filter -> sync -> remux."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    orig = tmp_path / "orig.wav"
    video = tmp_path / "video.mp4"
    final_out = tmp_path / "video_FFmpeg_Cleaned.mp4"

    mock_filter.return_value = work_dir / "vhs_filtered_orig.wav"
    modules.processing._process_ffmpeg_native_mode(work_dir, orig, video, final_out, 10.0)

    mock_filter.assert_called_once()
    mock_align.assert_called_once()
    mock_mux.assert_called_once()


@patch("modules.processing._filter_precondition_step", side_effect=lambda orig, precond, *args, **kwargs: orig)
@patch("modules.processing._final_mux_single_audio_step")
@patch("modules.processing._align_stems")
@patch("modules.processing._filter_arnndn_step")
def test_process_arnndn_speech_mode(mock_filter, mock_align, mock_mux, mock_precond, tmp_path):
    """Verify arnndn_speech mode pipeline orchestrates filter -> sync -> remux."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    orig = tmp_path / "orig.wav"
    video = tmp_path / "video.mp4"
    final_out = tmp_path / "video_Speech_Cleaned.mp4"

    mock_filter.return_value = work_dir / "arnndn_filtered_orig.wav"
    modules.processing._process_arnndn_speech_mode(work_dir, orig, video, final_out, 10.0)

    mock_filter.assert_called_once()
    mock_align.assert_called_once()
    mock_mux.assert_called_once()


@patch("modules.processing._final_mux_single_audio_step")
@patch("modules.processing._align_stems")
@patch("modules.processing._filter_auto_vhs_native_step")
def test_process_auto_ffmpeg_native_mode(mock_filter, mock_align, mock_mux, tmp_path):
    """Verify auto_ffmpeg_native mode pipeline orchestrates auto-tuned filter -> sync -> remux."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    orig = tmp_path / "orig.wav"
    video = tmp_path / "video.mp4"
    final_out = tmp_path / "video_AutoFFmpeg_Cleaned.mp4"

    mock_filter.return_value = work_dir / "auto_vhs_filtered_orig.wav"
    modules.processing._process_auto_ffmpeg_native_mode(work_dir, orig, video, final_out, 10.0)

    mock_filter.assert_called_once()
    mock_align.assert_called_once()
    mock_mux.assert_called_once()


@patch("modules.auto_scanner.scan_and_decide_restoration_strategy")
@patch("modules.modes.registry.get_mode_instance")
def test_process_auto_mode_dispatches_selected_strategy(mock_instance, mock_scan, tmp_path):
    """Verify _process_auto_mode scans audio and runs the chosen pipeline without duplicate scanning."""
    strategy = {"mode": "hybrid", "reason": "Test", "enhance_nfe": 256}
    mock_scan.return_value = strategy
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    orig = tmp_path / "orig.wav"
    video = tmp_path / "video.mp4"
    final_out = tmp_path / "video_Auto_Cleaned.mp4"
    handler = MagicMock()
    mock_instance.return_value.execute = handler

    with patch("modules.config.ENABLE_MULTIPASS", False):
        modules.processing._process_auto_mode(work_dir, orig, video, final_out, 10.0)
    mock_scan.assert_called_once_with(orig)
    mock_instance.assert_called_once_with("hybrid")
    handler.assert_called_once()

    mock_scan.reset_mock()
    mock_instance.reset_mock()
    handler.reset_mock()
    with patch("modules.config.ENABLE_MULTIPASS", True):
        modules.processing._process_auto_mode(work_dir, orig, video, final_out, 10.0)
    mock_scan.assert_called_once_with(orig)
    mock_instance.assert_called_once_with("multipass_auto")
    handler.assert_called_once_with(work_dir, orig, video, final_out, 10.0, strategy=strategy)


def test_build_precondition_filter_string():
    """Verify pre-conditioning filter string construction."""
    s1 = modules.filters._build_precondition_filter_string(60, True, 59.94)
    assert "highpass=f=60" in s1
    assert "adeclick" in s1
    assert "bandreject=f=59.94" in s1

    s2 = modules.filters._build_precondition_filter_string(0, False, 0.0)
    assert s2 == "anull"


@patch("modules.filters._run_ffmpeg_filter_step")
def test_filter_precondition_step(mock_run, tmp_path):
    """Verify _filter_precondition_step executes filter pipeline."""
    orig = tmp_path / "orig.wav"
    out_wav = tmp_path / "precond.wav"
    mock_run.return_value = out_wav

    cfg = {"highpass_hz": 60, "notch_hz": 0.0, "enable_adeclick": True}
    res = modules.filters._filter_precondition_step(orig, out_wav, cfg, total_duration=10.0)
    assert res == out_wav
    mock_run.assert_called_once()


@patch("modules.processing._execute_hybrid_restoration")
@patch("modules.processing._filter_precondition_step")
@patch("modules.auto_scanner.scan_and_decide_restoration_strategy")
def test_process_multipass_mode(mock_scan, mock_precond, mock_exec, tmp_path):
    """Verify _process_multipass_mode coordinates 4-pass pipeline and reuses passed strategy."""
    strat = {"mode": "hybrid", "precondition_filters": {"highpass_hz": 60}}
    mock_scan.return_value = strat
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    orig = tmp_path / "orig.wav"
    video = tmp_path / "video.mp4"
    final_out = tmp_path / "video_MultiPass_Cleaned.mp4"
    clean_wav = work_dir / "preconditioned_audio.wav"
    mock_precond.return_value = clean_wav

    # Call without strategy -> calls scanner
    modules.processing._process_multipass_mode(work_dir, orig, video, final_out, 10.0)
    mock_scan.assert_called_once_with(orig, executed_mode="multipass_auto")
    mock_precond.assert_called_once()
    mock_exec.assert_called_once_with(work_dir, clean_wav, orig, video, final_out, 10.0, strategy=strat)

    # Call with pre-computed strategy -> skips duplicate scanner
    mock_scan.reset_mock()
    modules.processing._process_multipass_mode(work_dir, orig, video, final_out, 10.0, strategy=strat)
    mock_scan.assert_not_called()


@patch("modules.processing._execute_pure_restoration")
@patch("modules.processing._filter_precondition_step")
@patch("modules.auto_scanner.scan_and_decide_restoration_strategy")
def test_process_auto_pure_mode(mock_scan, mock_precond, mock_exec, tmp_path):
    """Verify _process_auto_pure_mode coordinates pure 4-pass restoration."""
    strat = {"mode": "auto_pure", "precondition_filters": {"highpass_hz": 60}}
    mock_scan.return_value = strat
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    orig = tmp_path / "orig.wav"
    video = tmp_path / "video.mp4"
    final_out = tmp_path / "video_Pure_Cleaned.mp4"
    clean_wav = work_dir / "preconditioned_audio.wav"
    mock_precond.return_value = clean_wav

    modules.processing._process_auto_pure_mode(work_dir, orig, video, final_out, 10.0)
    mock_scan.assert_called_once_with(orig, executed_mode="auto_pure")
    mock_precond.assert_called_once()
    mock_exec.assert_called_once_with(work_dir, clean_wav, orig, video, final_out, 10.0, strategy=strat)


@patch("modules.processing._final_mix_step")
@patch("modules.processing._align_stems")
@patch("modules.processing._expand_background_step")
@patch("modules.processing._deess_vocals_step")
@patch("modules.processing._denoise_background_step")
@patch("modules.processing._denoise_vocals_step")
@patch("modules.processing._separate_stems_step")
def test_execute_pure_restoration(mock_sep, mock_den_voc, mock_den_bg, mock_deess, mock_expand, mock_align, mock_mix, tmp_path):
    """Verify _execute_pure_restoration separates stems and denoises without generative vocoder."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    inp = tmp_path / "input.wav"
    orig = tmp_path / "orig.wav"
    video = tmp_path / "video.mp4"
    final_out = tmp_path / "out_Pure_Cleaned.mp4"

    mock_sep.return_value = (work_dir / "vocals.wav", work_dir / "bg.wav")
    mock_den_voc.side_effect = lambda wav, out_dir, **kw: out_dir / f"denoised_{wav.name}"
    mock_den_bg.side_effect = lambda wav, out_dir, **kw: out_dir / f"denoised_{wav.name}"
    mock_deess.side_effect = lambda wav, out_dir, **kw: out_dir / f"deessed_{wav.name}"
    mock_expand.side_effect = lambda wav, out_dir, **kw: out_dir / f"expanded_{wav.name}"

    modules.processing._execute_pure_restoration(work_dir, inp, orig, video, final_out, 10.0)
    mock_sep.assert_called_once()
    mock_den_voc.assert_called_once()
    mock_den_bg.assert_called_once()
    mock_deess.assert_called_once()
    mock_expand.assert_called_once()
    assert mock_align.call_count == 2
    mock_mix.assert_called_once()


@patch("modules.modes.registry.get_mode_instance")
def test_run_processing_mode_dispatch(mock_instance, tmp_path):
    """Verify registry dispatches every non-auto mode to its mode instance."""
    work_dir = tmp_path / "work"
    orig = tmp_path / "orig.wav"
    video = tmp_path / "video.mp4"
    final_out = tmp_path / "out.mp4"

    handler = MagicMock()
    mock_instance.return_value.execute = handler
    for mode in (
        "multipass_auto",
        "auto_pure",
        "auto_pure_linear",
        "denoise_only",
        "ffmpeg_native",
        "auto_ffmpeg_native",
        "arnndn_speech",
        "hybrid",
        "cathar",
    ):
        with patch("modules.processing.PROCESS_MODE", mode):
            modules.processing._run_processing_mode(work_dir, orig, video, final_out, 5.0)
    assert mock_instance.call_args_list == [
        ((mode,), {})
        for mode in (
            "multipass_auto",
            "auto_pure",
            "auto_pure_linear",
            "denoise_only",
            "ffmpeg_native",
            "auto_ffmpeg_native",
            "arnndn_speech",
            "hybrid",
            "cathar",
        )
    ]
    assert handler.call_count == 9

    with patch("modules.processing.PROCESS_MODE", "auto"), patch("modules.processing._process_auto_mode") as mock_auto:
        modules.processing._run_processing_mode(work_dir, orig, video, final_out, 5.0)
    mock_auto.assert_called_once()


def test_deess_vocals_step_disabled(tmp_path):
    """Verify de-esser passthrough when disabled."""
    v = tmp_path / "vocals.wav"
    with patch("modules.processing.ENABLE_DEESSER", False):
        assert modules.processing._deess_vocals_step(v, tmp_path) == v


@patch("modules.processing.run_command_with_progress")
@patch("modules.processing.is_valid_audio")
def test_deess_vocals_step_active(mock_valid, mock_run, tmp_path):
    """Verify de-esser execution when enabled."""
    v = tmp_path / "vocals.wav"
    v.write_text("audio")
    mock_valid.side_effect = [True, False, True]
    with patch("modules.processing.ENABLE_DEESSER", True):
        res = modules.processing._deess_vocals_step(v, tmp_path)
        assert res.name == "deessed_vocals.wav"
        mock_run.assert_called_once()
        assert any("deesser" in str(arg) for arg in mock_run.call_args[0][0])


def test_expand_background_step_disabled(tmp_path):
    """Verify expander passthrough when disabled."""
    b = tmp_path / "bg.wav"
    with patch("modules.processing.ENABLE_DYNAMIC_EXPANDER", False):
        assert modules.processing._expand_background_step(b, tmp_path) == b


@patch("modules.processing.run_command_with_progress")
@patch("modules.processing.is_valid_audio")
def test_expand_background_step_active(mock_valid, mock_run, tmp_path):
    """Verify expander execution when enabled."""
    b = tmp_path / "bg.wav"
    b.write_text("audio")
    mock_valid.side_effect = [True, False, True]
    with patch("modules.processing.ENABLE_DYNAMIC_EXPANDER", True):
        res = modules.processing._expand_background_step(b, tmp_path)
        assert res.name == "expanded_bg.wav"
        mock_run.assert_called_once()
        assert any("compand" in str(arg) for arg in mock_run.call_args[0][0])


def test_build_mix_filter_expression_variations():
    """Verify loudness normalization inclusion in amix filter expression."""
    with patch("modules.config.ENABLE_LOUDNORM", False):
        expr = modules.processing._build_mix_filter_expression()
        assert "loudnorm" not in expr

    with patch("modules.config.ENABLE_LOUDNORM", True):
        expr = modules.processing._build_mix_filter_expression()
        assert f"loudnorm={modules.processing._loudnorm_target_args()}" in expr
        # loudnorm leaves the graph at 96 kHz unless it is resampled back, and its
        # linear mode does not guarantee the true-peak ceiling on its own.
        assert f"aresample={modules.processing.PIPELINE_SAMPLE_RATE}" in expr
        assert expr.endswith(f"{modules.processing.LOUDNORM_TRUE_PEAK_LIMITER}[mixed]")


@pytest.mark.parametrize(
    ("raw_input", "expected"),
    [
        (1.5, 1.5),
        ("0.8", 0.8),
        ("1.0[v];ametadata=file=wipe", 1.0),
        (float("nan"), 1.0),
        (-5.0, 1.0),
        (20.0, 1.0),
    ],
)
def test_sanitize_mix_level_variations(raw_input, expected):
    """Verify _sanitize_mix_level handles various inputs safely."""
    assert modules.processing._sanitize_mix_level(raw_input) == expected


def test_build_mix_filter_expression_neutralizes_injection():
    """Verify _build_mix_filter_expression neutralizes filter injection in mix volumes."""
    with (
        patch("modules.config.VOCAL_MIX_VOL", "1.0;movie=http://attacker"),
        patch("modules.config.BACKGROUND_MIX_VOL", 0.5),
        patch("modules.config.ENABLE_LOUDNORM", False),
    ):
        expr = modules.processing._build_mix_filter_expression()
        assert expr == "[1:a]volume=1.0[v];[2:a]volume=0.5[b];[v][b]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mixed]"


def test_scope_audio_args_for_stream():
    """Verify audio argument scoping for multi-stream muxing."""
    raw_args = ["-c:a", "aac", "-b:a", "320k", "-ar", "44100"]
    scoped_0 = modules.processing._scope_audio_args_for_stream(raw_args, 0)
    assert scoped_0 == ["-c:a:0", "aac", "-b:a:0", "320k", "-ar", "44100"]

    scoped_1 = modules.processing._scope_audio_args_for_stream(["-c:a", "mp2"], 1)
    assert scoped_1 == ["-c:a:1", "mp2"]


def test_final_mix_command_preservation_enabled(tmp_path):
    """Verify final mix stream mapping and codec scoping with track preservation."""
    v = tmp_path / "video.mp4"
    a1 = tmp_path / "a1.wav"
    a2 = tmp_path / "a2.wav"
    out = tmp_path / "out.mp4"
    args = ["-c:a", "aac", "-b:a", "320k"]

    with patch("modules.processing.PRESERVE_ORIGINAL_AUDIO_TRACK", True):
        mix_cmd = modules.processing._final_mix_output_command(v, a1, a2, out, args, 4)
        expected_flags = ["-map", "[mixed]", "0:a:0?", "-c:a:0", "-c:a:1", "copy"]
        assert all(flag in mix_cmd for flag in expected_flags)


def test_single_audio_mux_command_preservation_enabled(tmp_path):
    """Verify single-track mux mapping and codec scoping with track preservation."""
    v = tmp_path / "video.mp4"
    a1 = tmp_path / "a1.wav"
    out = tmp_path / "out.mp4"
    args = ["-c:a", "aac", "-b:a", "320k"]

    with patch("modules.processing.PRESERVE_ORIGINAL_AUDIO_TRACK", True):
        single_cmd = modules.processing._build_single_audio_mux_command(v, a1, out, args, 4)
        expected_flags = ["0:a:0?", "-c:a:0", "-c:a:1", "copy"]
        assert all(flag in single_cmd for flag in expected_flags)


@pytest.mark.parametrize("suffix", [".avi", ".mpg", ".mpeg"])
def test_preserved_audio_uses_container_codec_when_copy_is_incompatible(tmp_path, suffix):
    """Restricted containers receive a second track encoded for that container."""
    video = tmp_path / f"video{suffix}"
    audio = tmp_path / "audio.wav"
    output = tmp_path / f"output{suffix}"
    audio_args = modules.processing._get_audio_encoding_args(suffix)

    with patch("modules.processing.PRESERVE_ORIGINAL_AUDIO_TRACK", True):
        command = modules.processing._build_single_audio_mux_command(video, audio, output, audio_args, 4)

    audio_pairs = list(zip(command, command[1:]))
    assert ("-c:a:1", "copy") not in audio_pairs
    assert ("-c:a:1", audio_args[1]) in audio_pairs


def test_mux_commands_with_preservation_disabled(tmp_path):
    """Verify multi-stream mapping when track preservation is disabled."""
    v = tmp_path / "video.mp4"
    a1 = tmp_path / "a1.wav"
    a2 = tmp_path / "a2.wav"
    out = tmp_path / "out.mp4"
    args = ["-c:a", "aac"]

    with patch("modules.processing.PRESERVE_ORIGINAL_AUDIO_TRACK", False):
        mix_cmd_no_pres = modules.processing._final_mix_output_command(v, a1, a2, out, args, 4)
        assert "-map" in mix_cmd_no_pres
        assert "0:v:0" in mix_cmd_no_pres
        assert "[mixed]" in mix_cmd_no_pres
        assert "0:a:0?" not in mix_cmd_no_pres
