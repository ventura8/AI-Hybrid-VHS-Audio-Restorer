from unittest.mock import MagicMock, patch

import pytest

import modules.cathar as cathar
import modules.config as cfg
import modules.filters as flt
import modules.processing as prc
import modules.utils as utl


def test_normalize_cathar_denoise_method():
    """Validates normalization of denoise method options with default fallback."""
    assert cfg._normalize_cathar_denoise_method("spectral") == "spectral"
    assert cfg._normalize_cathar_denoise_method("wiener") == "wiener"
    assert cfg._normalize_cathar_denoise_method("invalid_method") == "spectral"
    assert cfg._normalize_cathar_denoise_method(123) == "spectral"


def test_normalize_cathar_azimuth_method():
    """Validates normalization of azimuth method options with default fallback."""
    assert cfg._normalize_cathar_azimuth_method("gcc-phat") == "gcc-phat"
    assert cfg._normalize_cathar_azimuth_method("correlation") == "correlation"
    assert cfg._normalize_cathar_azimuth_method("invalid_method") == "gcc-phat"
    assert cfg._normalize_cathar_azimuth_method(None) == "gcc-phat"


def test_normalize_cathar_enhance_method():
    """Validates normalization of enhance method options with default fallback."""
    assert cfg._normalize_cathar_enhance_method("replicate") == "replicate"
    assert cfg._normalize_cathar_enhance_method("interpolate") == "interpolate"
    assert cfg._normalize_cathar_enhance_method("invalid_method") == "replicate"
    assert cfg._normalize_cathar_enhance_method(42) == "replicate"


def test_cathar_dependency_check():
    """Verifies check_dependencies checks Cathar binary when in cathar mode."""
    record_mock = MagicMock()
    utl._check_mode_specific_tools("cathar", record_mock)
    record_mock.assert_called_once_with("Cathar", [utl.CATHAR_BIN, "--version"])


def test_promote_cathar_tmp_success(tmp_path):
    """Promotes valid tmp WAV to target destination."""
    tmp_wav = tmp_path / "test.tmp.wav"
    out_wav = tmp_path / "test.wav"
    tmp_wav.write_bytes(b"dummy")

    with patch("modules.cathar.is_valid_audio", return_value=True):
        result = cathar._promote_cathar_tmp(tmp_wav, out_wav, "TestStep")
    assert result == out_wav
    assert out_wav.exists()


def test_promote_cathar_tmp_failure(tmp_path):
    """Raises error and removes tmp when audio output is invalid."""
    tmp_wav = tmp_path / "test.tmp.wav"
    out_wav = tmp_path / "test.wav"
    tmp_wav.write_bytes(b"dummy")

    with patch("modules.cathar.is_valid_audio", return_value=False):
        with pytest.raises(RuntimeError, match="Cathar TestStep failed"):
            cathar._promote_cathar_tmp(tmp_wav, out_wav, "TestStep")
    assert not tmp_wav.exists()


def test_run_cathar_step_skips_when_exists(tmp_path):
    """Skips execution when valid output audio file is already present."""
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    out_wav.write_bytes(b"existing")

    with patch("modules.cathar.is_valid_audio", return_value=True), patch("modules.cathar.run_command_with_progress") as mock_run:
        result = cathar._run_cathar_step(["denoise"], in_wav, out_wav, "Denoise", "Cathar Task")
    assert result == out_wav
    mock_run.assert_not_called()


def test_run_cathar_step_executes_command(tmp_path):
    """Executes command and promotes output upon completion."""
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    tmp_wav = out_wav.with_suffix(".tmp.wav")

    def fake_run(cmd, description=None, total_duration=None):
        tmp_wav.write_bytes(b"valid_audio_data")

    with (
        patch("modules.cathar.is_valid_audio", side_effect=[False, True]),
        patch("modules.cathar.run_command_with_progress", side_effect=fake_run) as mock_run,
    ):
        result = cathar._run_cathar_step(["dewind", "--cutoff", "60"], in_wav, out_wav, "Dewind", "Cathar Dewind")
    assert result == out_wav
    mock_run.assert_called_once()


def test_run_cathar_step_cleans_tmp_on_exception(tmp_path):
    """Ensures temp file is unlinked if command execution raises an exception."""
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    tmp_wav = out_wav.with_suffix(".tmp.wav")

    def fake_fail(cmd, description=None, total_duration=None):
        tmp_wav.write_bytes(b"corrupt")
        raise OSError("Process failed")

    with (
        patch("modules.cathar.is_valid_audio", return_value=False),
        patch("modules.cathar.run_command_with_progress", side_effect=fake_fail),
    ):
        with pytest.raises(RuntimeError, match="Cathar Test failed"):
            cathar._run_cathar_step(["declick"], in_wav, out_wav, "Test", "Test Task")
    assert not tmp_wav.exists()


def test_cathar_is_stereo_audio(tmp_path):
    """Verifies channel inspection with soundfile and fallbacks."""
    fake_wav = tmp_path / "test.wav"
    mock_info = MagicMock()
    mock_info.channels = 2
    with patch("modules.cathar.sf.info", return_value=mock_info):
        assert cathar._is_stereo_audio(fake_wav) is True

    mock_info.channels = 1
    with patch("modules.cathar.sf.info", return_value=mock_info):
        assert cathar._is_stereo_audio(fake_wav) is False

    with patch("modules.cathar.sf.info", side_effect=OSError("Read error")):
        assert cathar._is_stereo_audio(fake_wav) is True

    with patch("modules.cathar.sf", None):
        assert cathar._is_stereo_audio(fake_wav) is True


def test_cathar_mono_below_step(tmp_path):
    """Applies stereo mono below for stereo input and bypasses for mono input."""
    in_wav = tmp_path / "in.wav"
    with (
        patch("modules.cathar._is_stereo_audio", return_value=True),
        patch("modules.cathar._run_cathar_step") as mock_step,
    ):
        cathar._cathar_mono_below_step(in_wav, tmp_path, cutoff_hz=100)
        mock_step.assert_called_once()
        assert "--mono-below" in mock_step.call_args[0][0]

    with (
        patch("modules.cathar._is_stereo_audio", return_value=False),
        patch("modules.cathar._run_cathar_step") as mock_step,
    ):
        res = cathar._cathar_mono_below_step(in_wav, tmp_path, cutoff_hz=100)
        mock_step.assert_not_called()
        assert res == in_wav


def test_cathar_dehum_selects_50hz(tmp_path):
    """Dehum selects 50 Hz when closest to 50 Hz."""
    in_wav = tmp_path / "in.wav"
    with patch("modules.cathar._run_cathar_step") as mock_step:
        cathar._cathar_dehum_step(in_wav, tmp_path, freq=49.9, adaptive=True, harmonics=5)
        cmd_50 = mock_step.call_args[0][0]
        assert "--freq" in cmd_50
        assert cmd_50[cmd_50.index("--freq") + 1] == "50"
        assert "--adaptive" in cmd_50
        assert cmd_50[cmd_50.index("--harmonics") + 1] == "5"


def test_cathar_dehum_selects_60hz(tmp_path):
    """Dehum selects 60 Hz when closest to 60 Hz."""
    in_wav = tmp_path / "in.wav"
    with patch("modules.cathar._run_cathar_step") as mock_step:
        cathar._cathar_dehum_step(in_wav, tmp_path, freq=59.94, adaptive=False, harmonics=4)
        cmd_60 = mock_step.call_args[0][0]
        assert "--freq" in cmd_60
        assert cmd_60[cmd_60.index("--freq") + 1] == "60"
        assert "--adaptive" not in cmd_60
        assert cmd_60[cmd_60.index("--harmonics") + 1] == "4"


def test_cathar_dereverb_step(tmp_path):
    """Dereverb invokes WPE or strength mode as configured."""
    in_wav = tmp_path / "in.wav"
    with patch("modules.cathar._run_cathar_step") as mock_step:
        cathar._cathar_dereverb_step(in_wav, tmp_path, wpe=True)
        assert mock_step.call_args[0][0] == ["dereverb", "--wpe"]

    with patch("modules.cathar._run_cathar_step") as mock_step:
        cathar._cathar_dereverb_step(in_wav, tmp_path, wpe=False, strength=3.0)
        assert mock_step.call_args[0][0] == ["dereverb", "-s", "3.0"]


def test_cathar_deesser_step(tmp_path):
    """De-esser formats command arguments with equals threshold."""
    in_wav = tmp_path / "in.wav"
    with patch("modules.cathar._run_cathar_step") as mock_step:
        cathar._cathar_deesser_step(in_wav, tmp_path, bands=3, freq=4000, threshold=-24.0)
        cmd = mock_step.call_args[0][0]
        assert cmd == ["deesser", "--bands", "3", "-f", "4000", "--threshold=-24.0"]


def test_build_cathar_denoise_cmd_wiener(tmp_path):
    """Wiener denoise builds proper flags with optional phase-coherence and noiseprint."""
    np_json = tmp_path / "noise.np.json"
    np_json.write_text("{}")
    cmd_coherent = cathar._build_cathar_denoise_cmd("wiener", 3.0, 0.01, True, noiseprint_path=np_json)
    assert cmd_coherent == ["denoise", "--wiener", "--coherent", "--noiseprint", str(np_json)]

    cmd_mono = cathar._build_cathar_denoise_cmd("wiener", 3.0, 0.01, False)
    assert cmd_mono == ["denoise", "--wiener"]


def test_build_cathar_denoise_cmd_spectral(tmp_path):
    """Spectral subtraction denoise builds alpha, beta, and coherent flags."""
    np_json = tmp_path / "noise.np.json"
    np_json.write_text("{}")
    cmd = cathar._build_cathar_denoise_cmd("spectral", 4.0, 0.02, True, noiseprint_path=np_json)
    assert cmd == ["denoise", "--alpha", "4.0", "--beta", "0.02", "--coherent", "--noiseprint", str(np_json)]


def test_find_quiet_window(tmp_path):
    """Finds quiet window using RMS probe evaluation."""
    fake_wav = tmp_path / "silence.wav"
    import numpy as np

    fake_data = np.ones((44100 * 3, 2), dtype=np.float32)
    mid_start = 22050
    mid_end = mid_start + 44100
    fake_data[mid_start:mid_end] = 0.01  # Quieter middle section

    with patch("modules.cathar._read_mono_samples", return_value=(fake_data[:, 0], 44100)):
        start_s = cathar._find_quiet_window(fake_wav, duration_s=0.5)
        assert 0.0 <= start_s <= 3.0
        # The quiet region spans 0.5s–1.5s; probe must land inside it.
        assert start_s >= 0.4, f"Quiet window {start_s}s should be >= 0.4s (near quiet zone)"

    with patch("modules.cathar.sf", None):
        assert cathar._find_quiet_window(fake_wav) == 0.0

    with patch("modules.cathar._read_mono_samples", side_effect=OSError("Read error")):
        assert cathar._find_quiet_window(fake_wav) == 0.0


def test_cathar_noiseprint_step(tmp_path):
    """Extracts quiet window and executes cathar noiseprint command."""
    in_wav = tmp_path / "input.wav"
    out_dir = tmp_path / "work"
    out_dir.mkdir()

    with (
        patch("modules.cathar._extract_noiseprint_slice", return_value=True),
        patch("modules.cathar._execute_noiseprint", return_value=out_dir / "noise_input.np.json") as mock_exec,
    ):
        result = cathar._cathar_noiseprint_step(in_wav, out_dir, duration_s=0.75)
    assert result == out_dir / "noise_input.np.json"
    mock_exec.assert_called_once()

    # Pre-existing noiseprint JSON is returned directly
    existing_json = out_dir / "noise_input.np.json"
    existing_json.write_text("{}")
    assert cathar._cathar_noiseprint_step(in_wav, out_dir) == existing_json

    # Bypassed when extraction fails
    with patch("modules.cathar._extract_noiseprint_slice", return_value=False):
        assert cathar._cathar_noiseprint_step(tmp_path / "other.wav", out_dir) is None


def test_filter_cathar_vhs_pipeline(tmp_path):
    """Orchestrates all passes and stages through the pipeline."""
    original_wav = tmp_path / "orig.wav"
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with (
        patch("modules.cathar._cathar_precondition_pass", return_value=original_wav) as mock_pre,
        patch("modules.cathar._cathar_repair_pass", return_value=original_wav) as mock_rep,
        patch("modules.cathar._cathar_noiseprint_step", return_value=work_dir / "noise.np.json") as mock_np,
        patch("modules.cathar._cathar_denoise_step", return_value=original_wav) as mock_den,
        patch("modules.cathar._cathar_polish_pass", return_value=original_wav) as mock_pol,
    ):
        strategy = {"precondition_filters": {"notch_hz": 50.0}}
        result = flt.filter_cathar_vhs_pipeline(original_wav, work_dir, strategy=strategy)
    assert result == original_wav
    mock_pre.assert_called_once_with(original_wav, work_dir, total_duration=None)
    mock_rep.assert_called_once_with(original_wav, work_dir, notch_freq=50.0, total_duration=None)
    mock_np.assert_called_once()
    mock_den.assert_called_once_with(original_wav, work_dir, noiseprint_path=work_dir / "noise.np.json", total_duration=None)
    mock_pol.assert_called_once_with(original_wav, work_dir, total_duration=None)


def test_process_cathar_mode(tmp_path):
    """Executes single track pipeline with cathar handler."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    orig_wav = tmp_path / "orig.wav"
    video_path = tmp_path / "video.mp4"
    out_video = tmp_path / "out.mp4"

    with (
        patch("modules.processing._resolve_preconditioned_audio", return_value=(orig_wav, {"sync_method": "shift"})),
        patch("modules.processing._process_single_track_pipeline") as mock_single,
    ):
        prc._process_cathar_mode(work_dir, orig_wav, video_path, out_video, 10.0, strategy={"sync_method": "shift"})
    mock_single.assert_called_once()
    assert mock_single.call_args[0][6] == "cathar_restored"
    assert mock_single.call_args[0][7] == "Cathar DSP"


def test_cathar_dewow_fallback_on_error(tmp_path):
    """Gracefully falls back to input audio stream when dewow raises error."""
    in_wav = tmp_path / "in.wav"
    with patch("modules.cathar._run_cathar_step", side_effect=RuntimeError("Index out of bounds")):
        res = cathar._cathar_dewow_step(in_wav, tmp_path)
    assert res == in_wav


def test_polish_full_audio_step(tmp_path):
    """Dynamic noise expander polishes full mix audio."""
    denoised_wav = tmp_path / "denoised.wav"
    denoised_wav.write_bytes(b"dummy")
    polish_dir = tmp_path / "polish"
    polish_dir.mkdir()

    with (
        patch("modules.filters.ENABLE_DYNAMIC_EXPANDER", True),
        patch("modules.processing.is_valid_audio", side_effect=[True, False]),
        patch("modules.processing._run_dsp_filter_file") as mock_filter,
    ):
        mock_filter.return_value = polish_dir / "polished_denoised.wav"
        res = prc._polish_full_audio_step(denoised_wav, polish_dir, 10.0)
    mock_filter.assert_called_once()
    assert res == polish_dir / "polished_denoised.wav"


def test_filters_re_exports_cathar():
    """Verifies that modules.filters exposes Cathar pipeline functions for backward compatibility."""
    exported_names = (
        "filter_cathar_vhs_pipeline",
        "_promote_cathar_tmp",
        "_run_cathar_step",
        "_cathar_dewind_step",
        "_cathar_azimuth_step",
        "_cathar_mono_below_step",
        "_cathar_declick_step",
        "_cathar_decrackle_step",
        "_cathar_inpaint_step",
        "_cathar_deplosive_step",
        "_cathar_enhance_step",
        "_cathar_declip_step",
        "_cathar_dehum_step",
        "_cathar_repair_step",
        "_cathar_dereverb_step",
        "_cathar_deesser_step",
        "_cathar_dewow_step",
        "_cathar_noiseprint_step",
        "_cathar_denoise_step",
        "_cathar_precondition_pass",
        "_cathar_analog_repair_pass",
        "_cathar_repair_pass",
        "_cathar_polish_pass",
    )
    for name in exported_names:
        assert getattr(flt, name) is getattr(cathar, name)


def test_cathar_dehum_bypasses_for_invalid_or_zero_frequency(tmp_path):
    """Dehum returns input unchanged when frequency is None or non-positive."""
    in_wav = tmp_path / "in.wav"
    assert cathar._cathar_dehum_step(in_wav, tmp_path, freq=None) == in_wav
    assert cathar._cathar_dehum_step(in_wav, tmp_path, freq=0.0) == in_wav
    assert cathar._cathar_dehum_step(in_wav, tmp_path, freq=-50.0) == in_wav


def test_promote_cathar_tmp_unlinks_existing_output(tmp_path):
    """Unlinks pre-existing output before renaming temporary audio."""
    tmp_wav = tmp_path / "test.tmp.wav"
    out_wav = tmp_path / "test.wav"
    tmp_wav.write_bytes(b"new_valid_audio")
    out_wav.write_bytes(b"old_audio")

    with patch("modules.cathar.is_valid_audio", return_value=True):
        res = cathar._promote_cathar_tmp(tmp_wav, out_wav, "PromoteTest")
    assert res == out_wav
    assert out_wav.read_bytes() == b"new_valid_audio"


def test_read_mono_samples_stereo_and_mono(tmp_path):
    """Verifies _read_mono_samples reads bounded float32 samples and handles mono/stereo."""
    fake_wav = tmp_path / "test.wav"
    import numpy as np

    mock_info = MagicMock()
    mock_info.frames = 1000
    mock_info.samplerate = 44100
    stereo_data = np.ones((500, 2), dtype=np.float32)
    mono_data = np.ones((500,), dtype=np.float32)

    with patch("modules.cathar.sf.info", return_value=mock_info):
        with patch("modules.cathar.sf.read", return_value=(stereo_data, 44100)):
            mono, sr = cathar._read_mono_samples(fake_wav)
            assert mono.shape == (500,)
            assert sr == 44100

        with patch("modules.cathar.sf.read", return_value=(mono_data, 44100)):
            mono, sr = cathar._read_mono_samples(fake_wav)
            assert mono.shape == (500,)
            assert sr == 44100


def test_noiseprint_helpers_and_error_handling(tmp_path):
    """Tests _extract_noiseprint_slice, _execute_noiseprint, and noiseprint error handling."""
    slice_wav = tmp_path / "slice.wav"
    out_json = tmp_path / "out.np.json"

    with (
        patch("modules.cathar._find_quiet_window", return_value=1.5),
        patch("modules.cathar.run_command_with_progress") as mock_run,
        patch("modules.cathar.is_valid_audio", return_value=True),
    ):
        assert cathar._extract_noiseprint_slice(tmp_path / "in.wav", slice_wav, 0.75) is True
        mock_run.assert_called_once()

    with patch("modules.cathar.run_command_with_progress") as mock_run:
        out_json.write_text("{}")
        assert cathar._execute_noiseprint(slice_wav, out_json) == out_json
        mock_run.assert_called_once()

    # Exception inside _cathar_noiseprint_step logs warning and returns None
    in_wav = tmp_path / "in_exc.wav"
    with patch("modules.cathar._extract_noiseprint_slice", side_effect=RuntimeError("Extraction crashed")):
        assert cathar._cathar_noiseprint_step(in_wav, tmp_path) is None
