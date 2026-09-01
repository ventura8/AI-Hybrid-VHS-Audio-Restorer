import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import modules.sync

# ---------------------------------------------------------
# Sync Logic
# ---------------------------------------------------------


@patch("modules.sync.sf.read")
@patch("modules.sync._save_audio_atomic", return_value=True)
def test_align_stems_shift_branch_no_lag(mock_save, mr):
    """Shift sync should preserve audio when no lag is detected."""
    sr = 44100
    base_audio = np.arange(200, dtype=np.float32).reshape(100, 2)
    shifted_audio = np.arange(200, dtype=np.float32).reshape(100, 2) + 1000
    mr.side_effect = [(base_audio, sr), (shifted_audio, sr), (shifted_audio, sr)] * 3

    with patch("modules.sync._calculate_cross_correlation_lag", return_value=0):
        with patch("modules.sync.SYNC_METHOD", "shift"):
            modules.sync._align_stems(Path("o.wav"), Path("p.wav"), Path("out.wav"))
            assert mock_save.called
            no_lag_written = mock_save.call_args_list[-1][0][1]
            assert no_lag_written.shape == shifted_audio.shape
            assert no_lag_written[0, 0] == shifted_audio[0, 0]
            assert no_lag_written[-1, -1] == shifted_audio[-1, -1]


@patch("modules.sync.sf.read")
@patch("modules.sync._save_audio_atomic", return_value=True)
def test_align_stems_shift_branch_positive_lag(mock_save, mr):
    """Shift sync should roll audio forward when a positive lag is detected."""
    sr = 44100
    base_audio = np.arange(200, dtype=np.float32).reshape(100, 2)
    shifted_audio = np.arange(200, dtype=np.float32).reshape(100, 2) + 1000
    mr.side_effect = [(base_audio, sr), (shifted_audio, sr), (shifted_audio, sr)] * 3

    with patch("modules.sync._calculate_cross_correlation_lag", return_value=50):
        with patch("modules.sync.SYNC_METHOD", "shift"):
            modules.sync._align_stems(Path("o.wav"), Path("p.wav"), Path("out.wav"))
            positive_written = mock_save.call_args_list[-1][0][1]
            assert np.array_equal(positive_written[:-50], shifted_audio[50:])
            assert np.all(positive_written[-50:] == 0)


@patch("modules.sync.sf.read")
@patch("modules.sync._save_audio_atomic", return_value=True)
def test_align_stems_shift_branch_negative_lag(mock_save, mr):
    """Shift sync should roll audio backward when a negative lag is detected."""
    sr = 44100
    base_audio = np.arange(200, dtype=np.float32).reshape(100, 2)
    shifted_audio = np.arange(200, dtype=np.float32).reshape(100, 2) + 1000
    mr.side_effect = [(base_audio, sr), (shifted_audio, sr), (shifted_audio, sr)] * 3

    with patch("modules.sync._calculate_cross_correlation_lag", return_value=-50):
        with patch("modules.sync.SYNC_METHOD", "shift"):
            modules.sync._align_stems(Path("o.wav"), Path("p.wav"), Path("out.wav"))
            negative_written = mock_save.call_args_list[-1][0][1]
            assert np.all(negative_written[:50] == 0)
            assert np.array_equal(negative_written[50:], shifted_audio[:-50])


@patch("modules.sync.sf.read")
@patch("modules.sync.sf.write")
def test_align_stems_mono_to_stereo(mock_write, mock_read):
    """Test align_stems converts mono to stereo."""
    sr = 44100
    # Return mono audio
    mock_read.return_value = (np.zeros((100, 1)), sr)

    with patch("modules.sync.SYNC_METHOD", "shift"):
        modules.sync._align_stems(Path("o.wav"), Path("p.wav"), Path("out.wav"))

    # Verify stereo output was written
    call_args = mock_write.call_args
    written_data = call_args[0][1]  # Second positional arg is the data
    assert written_data.shape[1] == 2  # Should be stereo


@patch("modules.sync.sf.read")
@patch("modules.sync.sf.write")
def test_align_stems_empty_audio(mock_write, mock_read):
    """Test align_stems handles empty audio."""
    sr = 44100
    # Return empty audio
    mock_read.return_value = (np.zeros((0, 2)), sr)

    with patch("modules.sync.SYNC_METHOD", "shift"):
        modules.sync._align_stems(Path("o.wav"), Path("p.wav"), Path("out.wav"))
    mock_write.assert_called()


@patch("modules.sync.sf.read")
@patch("modules.sync.sf.write")
@patch("modules.sync.log_msg")
def test_align_stems_exception_fallback(mock_log, mock_write, mock_read):
    """Test align_stems fallback on exception."""
    mock_read.side_effect = [Exception("Read error"), (np.zeros((10, 2)), 44100)]

    with patch("modules.sync.SYNC_METHOD", "shift"):
        modules.sync._align_stems(Path("o.wav"), Path("p.wav"), Path("out.wav"))
    assert mock_read.call_count == 2
    mock_write.assert_called_once()
    mock_log.assert_any_call("    [Warning] Sync failed (Read error). Using unaligned.", is_error=True)


@patch("modules.sync.sf.read")
@patch("modules.sync._save_audio_atomic", return_value=True)
def test_align_stems_large_negative_lag(mock_save, mock_read):
    """Test align_stems with very large negative lag."""
    sr = 44100
    # Return small audio
    input_audio = np.ones((50, 2), dtype=np.float32)
    mock_read.return_value = (input_audio, sr)

    # Simulate very large negative lag (cut more than available)
    with patch("modules.sync._calculate_cross_correlation_lag", return_value=-1000):
        with patch("modules.sync.SYNC_METHOD", "shift"):
            modules.sync._align_stems(Path("o.wav"), Path("p.wav"), Path("out.wav"))
            mock_save.assert_called_once()
            saved_audio = mock_save.call_args[0][1]
            assert np.all(saved_audio == 0)


def test_apply_warp_gpu_mocked():
    """Test _apply_warp_gpu with mocked PyTorch."""
    mock_torch = MagicMock()
    # Setup mocks
    mock_torch.cuda.is_available.return_value = True
    mock_device = MagicMock()
    mock_torch.device.return_value = mock_device

    # Mock tensor creation and methods
    mock_tensor = MagicMock()
    mock_torch.from_numpy.return_value = mock_tensor
    mock_tensor.float.return_value = mock_tensor
    mock_tensor.unsqueeze.return_value = mock_tensor
    mock_tensor.to.return_value = mock_tensor

    # Mock grid_sample result
    mock_warped = MagicMock()
    mock_torch.nn.functional.grid_sample.return_value = mock_warped

    # Mock chain: squeeze -> permute -> cpu -> numpy
    mock_cpu = MagicMock()
    mock_warped.squeeze.return_value.squeeze.return_value.permute.return_value.cpu.return_value = mock_cpu
    expected_output = np.array([[0.1, 0.2]])
    mock_cpu.numpy.return_value = expected_output

    # Input data
    audio_np = np.zeros((10, 2))
    indices_np = np.zeros(10)

    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = modules.sync._apply_warp_gpu(audio_np, indices_np)

        assert result is expected_output
        mock_torch.nn.functional.grid_sample.assert_called_once()


@patch("modules.sync.sf.read")
@patch("modules.sync.sf.write")
def test_align_stems_shift_empty_mono(mock_write, mock_read, tmp_path):
    """Test clean fallback for empty audio in align steps."""
    # First read (ref/proc check) -> returns empty arrays
    # Second read (fallback write) -> returns mono array

    mock_read.side_effect = [
        (np.array([]), 44100),  # ref
        (np.array([]), 44100),  # proc
        (np.zeros((100, 1)), 44100),  # fallback read, mono
    ]

    wav = tmp_path / "test.wav"
    out = tmp_path / "out.wav"

    modules.sync._align_stems_shift(wav, wav, out)

    # Verify write was called with stereo data (tiled)
    args = mock_write.call_args
    data = args[0][1]
    assert data.shape[1] == 2  # Should be tiled to stereo


@patch("modules.sync.sf.read")
@patch("modules.sync.sf.write")
def test_align_shift_failure_fallback(mock_write, mock_read, tmp_path, capsys):
    """Test fallback when shift sync fails."""
    wav = tmp_path / "in.wav"
    out = tmp_path / "out.wav"

    # Mock read failure
    # 1. read ref (fail) -> Exception
    # 2. Catch -> read raw (succeed) -> write
    mock_read.side_effect = [Exception("Corrupt Ref"), (np.zeros((10, 2)), 44100)]

    modules.sync._align_stems_shift(wav, wav, out)

    assert mock_write.called
    captured = capsys.readouterr()
    assert "Sync failed" in captured.err or "Sync failed" in captured.out


@patch("modules.sync.map_coordinates")
@patch("modules.sync._save_audio_atomic")
def test_warp_aligned_audio_cpu(mock_save, mock_map, tmp_path):
    """Test CPU warping fallback."""
    # Setup inputs
    proc = np.zeros((100, 2))
    indices = np.zeros(100)
    out = tmp_path / "out.wav"

    mock_map.return_value = np.zeros(100)
    mock_save.return_value = True

    modules.sync._warp_aligned_audio_cpu(proc, indices, 2, out, 44100)

    assert mock_map.call_count == 2  # Once per channel
    assert mock_save.called


def test_run_fastdtw_chunk():
    """Test fastdtw worker function directly."""
    # The worker function imports fastdtw internally.
    # We must patch the usage of the imported module.
    # Since we can't easily patch an import inside a function from outside without
    # complex sys.modules hacks, we'll patch sys.modules dictionary for 'fastdtw'.
    mock_fastdtw_module = MagicMock()
    mock_fastdtw_module.fastdtw.return_value = (0, [(0, 0), (1, 1)])
    mock_distance_module = MagicMock()
    mock_distance_module.euclidean = MagicMock(return_value=0.0)

    def _import_side_effect(module_name):
        if module_name == "fastdtw":
            return mock_fastdtw_module
        if module_name == "scipy.spatial.distance":
            return mock_distance_module
        raise ModuleNotFoundError(f"No module named '{module_name}'")

    with patch("modules.sync.importlib.import_module", side_effect=_import_side_effect):
        # fastdtw expects 1D arrays if checking simple distance, or 2D if features.
        # The actual implementation passes features with shape (Frames, 12).
        args = (np.zeros((10, 12)), np.zeros((10, 12)), 10)
        path = modules.sync._run_fastdtw_chunk(args)

    assert path == [(0, 0), (1, 1)]


@patch("modules.sync.librosa", None)
@patch("modules.sync.fastdtw", None)
@patch("modules.sync._align_stems_shift")
def test_align_stems_dtw_missing_deps(mock_shift, tmp_path):
    """Test DTW sync fallback when deps missing."""
    wav = tmp_path / "a.wav"
    out = tmp_path / "out.wav"
    modules.sync._align_stems_dtw(wav, wav, out)
    assert mock_shift.called


def test_apply_warp_gpu_exception(capsys):
    """Test exception handler in GPU warp."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    # Raise exception during processing
    mock_torch.from_numpy.side_effect = Exception("GPU Boom")

    with patch.dict(sys.modules, {"torch": mock_torch}):
        res = modules.sync._apply_warp_gpu(np.zeros((10, 2)), np.zeros(10))
        assert res is None
        captured = capsys.readouterr()
        assert "GPU Warp failed" in captured.out


def test_calculate_cross_correlation_lag_raises_when_scipy_missing():
    """Cross-correlation sync should fail clearly when scipy.signal is unavailable."""
    ref_audio = np.zeros(100)
    proc_audio = np.zeros(100)

    with patch.object(modules.sync, "scipy_signal", None):
        with pytest.raises(ImportError, match="scipy is required for shift synchronization"):
            modules.sync._calculate_cross_correlation_lag(ref_audio, proc_audio, 44100)


@pytest.mark.parametrize("invalid_val", [np.nan, np.inf, -np.inf])
def test_calculate_cross_correlation_lag_handles_nan_processed_audio(invalid_val):
    """NaN/inf samples in the processed stream must not produce a spurious full-clip lag.

    Regression guard for the case where ARNNDN (or any denoiser) produces NaN/inf
    on heavily-degraded audio; the guard must clamp the returned lag to ±25 % of
    the reference length rather than returning an index that swallows the entire clip.
    """
    sr = 44100
    # Reference: clean sine, processed: non-finite values (fully suppressed denoiser output).
    ref_audio = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
    proc_audio = np.full(1000, invalid_val, dtype=np.float32)

    lag = modules.sync._calculate_cross_correlation_lag(ref_audio, proc_audio, sr)

    # Lag must be within ±25 % of reference length (250 samples) and must be finite.
    assert np.isfinite(lag), "Lag must be finite even when processed audio is non-finite"
    assert abs(lag) <= len(ref_audio) // 4, "Lag must be clamped to ±25 % of ref length"


def test_calculate_cross_correlation_lag_clamps_to_quarter_length():
    """All-zeros processed audio should produce lag=0, not an out-of-bounds full-clip shift.

    Verifies the ±25 % clamping mask: when the processed stream carries no signal,
    the best valid-lag index must still fall within the allowed window.
    """
    sr = 44100
    ref_audio = np.random.default_rng(42).standard_normal(4000).astype(np.float32)
    proc_audio = np.zeros(4000, dtype=np.float32)

    lag = modules.sync._calculate_cross_correlation_lag(ref_audio, proc_audio, sr)

    max_allowed = len(ref_audio) // 4
    assert abs(lag) <= max_allowed, f"Lag {lag} exceeds ±25 % clamp ({max_allowed} samples)"


def test_correlation_lag_never_exceeds_short_processed_stream():
    """A much shorter processed stream cannot select an unusable lag."""
    reference = np.ones(1000, dtype=np.float32)
    processed = np.ones(100, dtype=np.float32)
    lag = modules.sync._calculate_cross_correlation_lag(reference, processed, 44100)
    assert abs(lag) <= len(processed) - 1


def test_load_optional_returns_none_on_import_error():
    """_load_optional should gracefully return None when import fails."""
    with patch("modules.sync.importlib.import_module", side_effect=ImportError("missing")):
        assert modules.sync._load_optional("totally_missing") is None


def test_apply_warp_gpu_returns_none_when_cuda_unavailable():
    """GPU warp should return None when CUDA is unavailable."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = modules.sync._apply_warp_gpu(np.zeros((10, 2)), np.zeros(10))

    assert result is None


def test_apply_warp_gpu_returns_none_for_tiny_input():
    """GPU warp should return None when input frame count is <= 1."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.from_numpy.return_value.float.return_value.unsqueeze.return_value.unsqueeze.return_value.to.return_value = MagicMock()

    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = modules.sync._apply_warp_gpu(np.zeros((1, 2)), np.zeros(1))

    assert result is None


def test_execute_parallel_dtw_uses_cpu_executor_when_gpu_not_allowed():
    """CPU executor path should run when GPU path is disallowed by config."""
    chunks = [(np.zeros((2, 2)), np.zeros((2, 2)), 5)]
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with (
        patch.dict(sys.modules, {"torch": mock_torch}),
        patch("modules.sync.GPU_VRAM_GB", 0),
        patch("modules.sync.concurrent.futures.ProcessPoolExecutor") as MockPool,
        patch("modules.sync.concurrent.futures.as_completed", side_effect=lambda futures: list(futures)),
        patch("modules.sync._run_fastdtw_chunk", return_value=[[0, 0]]),
    ):
        mock_pool = MagicMock()
        MockPool.return_value = mock_pool
        mock_pool.__enter__.return_value = mock_pool

        future = MagicMock()
        future.result.return_value = [[0, 0]]
        mock_pool.submit.return_value = future

        result = modules.sync._execute_parallel_dtw(chunks)

    assert len(result) == 1
    assert MockPool.called


def test_execute_parallel_dtw_raises_on_worker_failure():
    """Worker exceptions should be logged and re-raised."""
    chunks = [(np.zeros((2, 2)), np.zeros((2, 2)), 5)]
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with (
        patch.dict(sys.modules, {"torch": mock_torch}),
        patch("modules.sync.GPU_VRAM_GB", 0),
        patch("modules.sync.concurrent.futures.ProcessPoolExecutor") as MockPool,
        patch("modules.sync.concurrent.futures.as_completed", side_effect=lambda futures: list(futures)),
    ):
        mock_pool = MagicMock()
        MockPool.return_value = mock_pool
        mock_pool.__enter__.return_value = mock_pool

        future = MagicMock()
        future.result.side_effect = RuntimeError("boom")
        mock_pool.submit.return_value = future

        with pytest.raises(RuntimeError, match="boom"):
            modules.sync._execute_parallel_dtw(chunks)


def test_collect_dtw_results_cancels_outstanding_futures_on_failure():
    """Failed chunk collection should cancel non-completed futures and re-raise."""
    chunks = ["c0", "c1", "c2"]
    failed_future = MagicMock()
    outstanding_future_1 = MagicMock()
    outstanding_future_2 = MagicMock()

    failed_future.result.side_effect = RuntimeError("boom")
    failed_future.done.return_value = True
    outstanding_future_1.done.return_value = False
    outstanding_future_2.done.return_value = False

    executor = MagicMock()
    executor.submit.side_effect = [failed_future, outstanding_future_1, outstanding_future_2]

    with (
        patch("modules.sync.concurrent.futures.as_completed", return_value=[failed_future]),
        patch("modules.sync.log_msg") as mock_log,
        patch("modules.sync._update_dtw_progress") as mock_progress,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            modules.sync._collect_dtw_results(executor, MagicMock(), chunks, len(chunks), 0.0)

    outstanding_future_1.cancel.assert_called_once()
    outstanding_future_2.cancel.assert_called_once()
    mock_log.assert_called_once_with("    Chunk 0 failed: boom", is_error=True)
    mock_progress.assert_not_called()


def test_warp_aligned_audio_handles_savgol_error_and_gpu_success(tmp_path):
    """Warp should continue when savgol fails and use GPU output when available."""
    processed = tmp_path / "proc.wav"
    output = tmp_path / "out.wav"
    path = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

    def mock_interp(*_args, **_kwargs):
        return lambda values: values

    mock_signal = MagicMock()
    mock_signal.savgol_filter.side_effect = ValueError("savgol fail")
    gpu_audio = np.zeros((60, 2))

    with (
        patch.object(modules.sync, "interp1d", mock_interp),
        patch.object(modules.sync, "scipy_signal", mock_signal),
        patch("modules.sync.sf.read", return_value=(np.zeros((60, 2)), 44100)),
        patch("modules.sync._apply_warp_gpu", return_value=gpu_audio),
        patch("modules.sync._save_audio_atomic", return_value=True) as mock_save,
    ):
        modules.sync._warp_aligned_audio(processed, output, path, 4, 4)

    assert mock_save.called


def test_align_stems_shift_propagates_fallback_write_error(tmp_path):
    """Fallback write failure should propagate so an invalid output path is not returned."""
    wav = tmp_path / "in.wav"
    out = tmp_path / "out.wav"

    with (
        patch("modules.sync.sf.read", side_effect=[Exception("x"), (np.zeros((2, 1)), 44100)]),
        patch("modules.sync._save_audio_atomic", side_effect=Exception("write failed")),
    ):
        with pytest.raises(Exception, match="write failed"):
            modules.sync._align_stems_shift(wav, wav, out)


def test_align_stems_dtw_missing_deps_logs_import_error_context(tmp_path):
    """DTW fallback log should include import error details when available."""
    wav = tmp_path / "a.wav"
    out = tmp_path / "out.wav"

    with (
        patch.object(modules.sync, "librosa", None),
        patch.object(modules.sync, "fastdtw", None),
        patch.object(modules.sync, "DTW_IMPORT_ERROR", "No module named fastdtw"),
        patch("modules.sync.log_msg") as mock_log,
        patch("modules.sync._align_stems_shift", return_value=out),
    ):
        modules.sync._align_stems_dtw(wav, wav, out)

    logged = " ".join(str(arg) for arg in mock_log.call_args[0])
    assert "No module named fastdtw" in logged


def test_align_stems_dispatches_to_dtw_branch(tmp_path):
    """_align_stems should dispatch to DTW path when configured."""
    wav = tmp_path / "a.wav"
    out = tmp_path / "out.wav"

    with (
        patch("modules.sync.SYNC_METHOD", "dtw"),
        patch("modules.sync._align_stems_dtw", return_value=out) as mock_dtw,
    ):
        result = modules.sync._align_stems(wav, wav, out)

    assert result == out
    mock_dtw.assert_called_once()
