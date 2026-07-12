import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

MODE_BY_SUFFIX = {
    "Denoised_Cleaned": "denoise_only",
    "Hybrid_Cleaned": "hybrid",
}


def _setup_test_environment(base_dir):
    """Sets up input/output directories and cleans up old artifacts."""
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    log_file = base_dir / "session_log.txt"

    # Clean previous run artifacts
    if output_dir.exists():
        for f in output_dir.glob("*_Hybrid_Cleaned*"):
            f.unlink()
        for f in output_dir.glob("*_Denoised_Cleaned*"):
            f.unlink()
    if log_file.exists():
        log_file.unlink()

    # Ensure directories exist (important for CI)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    return input_dir, output_dir, log_file


def _get_test_video(input_dir):
    """Finds a video file for testing."""
    exts = {".mp4", ".mkv", ".avi", ".mov"}
    video_files = [f for f in input_dir.iterdir() if f.suffix.lower() in exts]

    if not video_files:
        raise AssertionError(f"No video files found in {input_dir}.")

    return video_files[0]


def _run_script(base_dir, repo_root):
    """Executes the restore script via subprocess."""
    print("=== STARTING E2E TEST ===")
    start_time = time.time()
    timeout_sec = 900

    cmd = [sys.executable, str(repo_root / "restore_audio_hybrid.py")]

    # Inject Test Mode Env Var
    test_env = os.environ.copy()
    test_env["AI_RESTORE_TEST_MODE"] = "1"

    try:
        subprocess.run(cmd, check=True, env=test_env, timeout=timeout_sec, cwd=base_dir)
    except subprocess.TimeoutExpired as e:
        print(f"Script timed out after {e.timeout} seconds")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Script failed with code {e.returncode}")
        raise

    end_time = time.time()
    print(f"=== EXECUTION FINISHED in {end_time - start_time:.2f}s ===")


def _verify_output(output_dir, log_file, input_video):
    """Verifies that output files exist and checks logs."""
    hybrid_output = output_dir / f"{input_video.stem}_Hybrid_Cleaned{input_video.suffix}"
    denoised_output = output_dir / f"{input_video.stem}_Denoised_Cleaned{input_video.suffix}"
    assert not (hybrid_output.exists() and denoised_output.exists()), (
        "FAILURE: Both hybrid and denoise-only outputs exist; expected exactly one mode-specific output."
    )
    expected_output = hybrid_output if hybrid_output.exists() else denoised_output

    assert expected_output.exists(), "FAILURE: Output file NOT found for either mode-specific suffix."
    print(f"SUCCESS: Output file generated: {expected_output.name}")
    print(f"Size: {expected_output.stat().st_size / 1024 / 1024:.2f} MB")

    # Verify Log content
    assert log_file.exists(), "FAILURE: Log file not found."
    with open(log_file, "r", encoding="utf-8") as log_f:
        logs = log_f.read()

    print("\n--- LOG ANALYSIS ---")
    if "Attempting execution on GPU" in logs:
        print("GPU Usage Attempt: DETECTED (Good)")
    else:
        print("GPU Usage Attempt: NOT DETECTED (Bad)")

    if "GPU Failed" in logs:
        print("GPU Status: FAILED (Fallback to CPU used)")
    else:
        print("GPU Status: SUCCESS (Presumably)")


@pytest.mark.parametrize(
    "mode_suffix",
    ["Denoised_Cleaned", "Hybrid_Cleaned"],
)
def test_pipeline_modes(tmp_path, monkeypatch, mode_suffix):
    repo_root = Path(__file__).resolve().parents[2]
    base_dir = tmp_path
    process_mode = MODE_BY_SUFFIX[mode_suffix]
    monkeypatch.chdir(base_dir)
    input_dir, output_dir, log_file = _setup_test_environment(base_dir)
    (base_dir / "config.yaml").write_text(f"process_mode: {process_mode}\n", encoding="utf-8")

    # Create a deterministic input video placeholder so this test always runs in CI.
    seeded_video = input_dir / "seed_input.mp4"
    seeded_video.write_bytes(b"fake-video")

    input_video = _get_test_video(input_dir)

    print(f"Using test video: {input_video.name}")

    def _fake_run(cmd, check, env, timeout, cwd):
        assert check is True
        assert timeout == 900
        assert cwd == base_dir
        assert env.get("AI_RESTORE_TEST_MODE") == "1"
        assert str(repo_root / "restore_audio_hybrid.py") in cmd
        assert (base_dir / "config.yaml").read_text(encoding="utf-8") == f"process_mode: {process_mode}\n"

        # Simulate generated output artifact and session log from a successful run.
        out_path = output_dir / f"{input_video.stem}_{mode_suffix}{input_video.suffix}"
        out_path.write_bytes(b"processed-video")
        log_file.write_text("[INFO] Attempting execution on GPU\n", encoding="utf-8")

        return subprocess.CompletedProcess(cmd, 0)

    with patch("subprocess.run", side_effect=_fake_run):
        _run_script(base_dir, repo_root)

    _verify_output(output_dir, log_file, input_video)


def test_run_script_timeout_propagates(tmp_path):
    """E2E runner should propagate subprocess timeout failures."""
    base_dir = tmp_path
    repo_root = Path(__file__).resolve().parents[2]

    timeout_exc = subprocess.TimeoutExpired(cmd=["python", "restore_audio_hybrid.py"], timeout=900)
    with patch("subprocess.run", side_effect=timeout_exc):
        with pytest.raises(subprocess.TimeoutExpired):
            _run_script(base_dir, repo_root)


def test_run_script_called_process_error_propagates(tmp_path):
    """E2E runner should propagate non-zero process exits."""
    base_dir = tmp_path
    repo_root = Path(__file__).resolve().parents[2]

    process_err = subprocess.CalledProcessError(returncode=1, cmd=["python", "restore_audio_hybrid.py"])
    with patch("subprocess.run", side_effect=process_err):
        with pytest.raises(subprocess.CalledProcessError):
            _run_script(base_dir, repo_root)


def test_verify_output_accepts_hybrid_mode_artifact(tmp_path):
    """E2E verification should accept hybrid output naming path."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = tmp_path / "session_log.txt"

    input_video = tmp_path / "clip.mp4"
    input_video.write_bytes(b"in")
    hybrid_output = output_dir / "clip_Hybrid_Cleaned.mp4"
    hybrid_output.write_bytes(b"out")
    log_file.write_text("[INFO] Attempting execution on GPU\n[INFO] GPU Failed\n", encoding="utf-8")

    log_contents = log_file.read_text(encoding="utf-8")
    assert "Attempting execution on GPU" in log_contents
    assert "GPU Failed" in log_contents

    _verify_output(output_dir, log_file, input_video)


def test_verify_output_rejects_dual_mode_outputs(tmp_path):
    """E2E verification should fail when both hybrid and denoise-only outputs exist."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = tmp_path / "session_log.txt"
    log_file.write_text("[INFO] Attempting execution on GPU\n", encoding="utf-8")

    input_video = tmp_path / "clip.mp4"
    input_video.write_bytes(b"in")
    (output_dir / "clip_Hybrid_Cleaned.mp4").write_bytes(b"hybrid")
    (output_dir / "clip_Denoised_Cleaned.mp4").write_bytes(b"denoise")

    with pytest.raises(AssertionError, match="Both hybrid and denoise-only outputs exist"):
        _verify_output(output_dir, log_file, input_video)
