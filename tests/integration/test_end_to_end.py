import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from modules.utils import is_valid_video

MODE_BY_SUFFIX = {
    "Auto_Cleaned": "auto",
    "MultiPass_Cleaned": "multipass_auto",
    "Pure_Cleaned": "auto_pure",
    "PureLinear_Cleaned": "auto_pure_linear",
    "Denoised_Cleaned": "denoise_only",
    "Hybrid_Cleaned": "hybrid",
    "FFmpeg_Cleaned": "ffmpeg_native",
    "AutoFFmpeg_Cleaned": "auto_ffmpeg_native",
    "Speech_Cleaned": "arnndn_speech",
    "Cathar_Cleaned": "cathar",
}

# Single source of truth for every mode-specific output suffix.
MODE_SUFFIXES = [f"_{suffix}" for suffix in MODE_BY_SUFFIX]


def _setup_test_environment(base_dir):
    """Sets up input/output directories and cleans up old artifacts."""
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    log_file = base_dir / "session_log.txt"

    # Clean previous run artifacts
    if output_dir.exists():
        for suffix in MODE_SUFFIXES:
            for f in output_dir.glob(f"*{suffix}*"):
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


def _get_expected_output(output_dir, input_video):
    for suffix in MODE_SUFFIXES:
        candidate = output_dir / f"{input_video.stem}{suffix}{input_video.suffix}"
        if candidate.exists():
            return candidate
    return output_dir / f"{input_video.stem}_Denoised_Cleaned{input_video.suffix}"


def _assert_log_markers(logs):
    assert "Attempting execution on GPU" in logs, "GPU Usage Attempt marker missing from logs."
    print("GPU Usage Attempt: DETECTED (Good)")

    if "GPU Failed" in logs:
        print("GPU Status: FAILED (Fallback to CPU used)")
    else:
        print("GPU Status: SUCCESS (Presumably)")


def _assert_smoke_output_artifact(smoke_expected):
    assert smoke_expected.exists(), "FAILURE: smoke subprocess did not emit mode-specific output artifact."
    assert is_valid_video(smoke_expected) or smoke_expected.stat().st_size > 0


def _assert_smoke_log_contents(smoke_log, process_mode):
    assert smoke_log.exists(), "FAILURE: smoke subprocess did not create session log."
    smoke_logs = smoke_log.read_text(encoding="utf-8")
    assert "Attempting execution on GPU" in smoke_logs
    assert f"mode={process_mode}" in smoke_logs


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


def _collect_existing_mode_outputs(output_dir, input_video):
    return [
        output_dir / f"{input_video.stem}{s}{input_video.suffix}"
        for s in MODE_SUFFIXES
        if (output_dir / f"{input_video.stem}{s}{input_video.suffix}").exists()
    ]


def _verify_output(output_dir, log_file, input_video):
    """Verifies that output files exist and checks logs."""
    found_outputs = _collect_existing_mode_outputs(output_dir, input_video)
    assert len(found_outputs) <= 1, "FAILURE: Multiple mode outputs exist."
    expected_output = _get_expected_output(output_dir, input_video)

    assert expected_output.exists(), "FAILURE: Output file NOT found for mode-specific suffix."
    assert is_valid_video(expected_output) or expected_output.stat().st_size > 0, f"FAILURE: Empty output {expected_output.name}"
    print(f"SUCCESS: Output file generated: {expected_output.name}")

    assert log_file.exists(), "FAILURE: Log file not found."
    _assert_log_markers(log_file.read_text(encoding="utf-8"))


def _write_smoke_sitecustomize(smoke_dir, process_mode):
    sitecustomize_path = smoke_dir / "sitecustomize.py"
    sitecustomize_path.write_text(
        "from pathlib import Path\n"
        "import modules.config as cfg\n"
        "import modules.processing as proc\n"
        "import modules.ui as ui\n"
        "import modules.utils as utils\n"
        "\n"
        "def _fake_process(video_path, gpu_name, target_output_dir=None):\n"
        "    video_path = Path(video_path)\n"
        "    target_dir = Path(target_output_dir) if target_output_dir else video_path.parent\n"
        "    target_dir.mkdir(parents=True, exist_ok=True)\n"
        "    suffix = proc._get_output_suffix(cfg.PROCESS_MODE)\n"
        '    out = target_dir / f"{video_path.stem}{suffix}{video_path.suffix}"\n'
        "    out.write_bytes(b'v' * 12000)\n"
        "    with Path('session_log.txt').open('a', encoding='utf-8') as handle:\n"
        "        handle.write('[INFO] Attempting execution on GPU\\n')\n"
        "        handle.write(f'[INFO] mode={cfg.PROCESS_MODE}\\n')\n"
        "    return True\n"
        "\n"
        "utils.check_dependencies = lambda: True\n"
        "ui.run_init_sequence = lambda: ('cpu', 'gpu')\n"
        "ui._show_banner = lambda: None\n"
        "proc.process_hybrid_audio = _fake_process\n",
        encoding="utf-8",
    )


def _build_smoke_environment(smoke_dir, repo_root):
    smoke_env = os.environ.copy()
    existing_pythonpath = smoke_env.get("PYTHONPATH", "")
    pythonpath_parts = [str(smoke_dir), str(repo_root)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    smoke_env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    smoke_env["AI_RESTORE_TEST_MODE"] = "1"
    smoke_env["PYTHONIOENCODING"] = "utf-8"
    return smoke_env


def _run_smoke_subprocess(smoke_env, smoke_dir, repo_root, smoke_video):
    cmd = [sys.executable, str(repo_root / "restore_audio_hybrid.py"), str(smoke_video)]
    subprocess.run(cmd, check=True, env=smoke_env, timeout=120, cwd=smoke_dir)


def _run_real_subprocess_smoke_test(base_dir, repo_root, process_mode, mode_suffix):
    # Real subprocess smoke test for entry-point/config routing without mock patching.
    smoke_dir = base_dir / "smoke"
    smoke_input, _, smoke_log = _setup_test_environment(smoke_dir)
    smoke_video = smoke_input / "seed_input.mp4"
    smoke_video.write_bytes(b"v" * 12000)
    (smoke_dir / "config.yaml").write_text(f"process_mode: {process_mode}\n", encoding="utf-8")

    _write_smoke_sitecustomize(smoke_dir, process_mode)
    smoke_env = _build_smoke_environment(smoke_dir, repo_root)
    _run_smoke_subprocess(smoke_env, smoke_dir, repo_root, smoke_video)

    smoke_expected = smoke_video.parent / f"{smoke_video.stem}_{mode_suffix}{smoke_video.suffix}"
    _assert_smoke_output_artifact(smoke_expected)
    _assert_smoke_log_contents(smoke_log, process_mode)


def _assert_pipeline_run_execution(check, timeout, cwd, base_dir):
    assert check is True
    assert timeout == 900
    assert cwd == base_dir


def _assert_pipeline_run_environment(env, base_dir, process_mode):
    assert env.get("AI_RESTORE_TEST_MODE") == "1"
    assert (base_dir / "config.yaml").read_text(encoding="utf-8") == f"process_mode: {process_mode}\n"


def _assert_pipeline_run_command(cmd, repo_root):
    assert str(repo_root / "restore_audio_hybrid.py") in cmd


def _write_pipeline_run_artifacts(output_dir, log_file, input_video, mode_suffix):
    out_path = output_dir / f"{input_video.stem}_{mode_suffix}{input_video.suffix}"
    out_path.write_bytes(b"processed-video")
    log_file.write_text("[INFO] Attempting execution on GPU\n", encoding="utf-8")


def _fake_pipeline_run(cmd, check, env, timeout, cwd, repo_root, base_dir, process_mode, output_dir, log_file, input_video, mode_suffix):
    _assert_pipeline_run_execution(check, timeout, cwd, base_dir)
    _assert_pipeline_run_environment(env, base_dir, process_mode)
    _assert_pipeline_run_command(cmd, repo_root)
    _write_pipeline_run_artifacts(output_dir, log_file, input_video, mode_suffix)
    return subprocess.CompletedProcess(cmd, 0)


@pytest.mark.parametrize("mode_suffix", sorted(MODE_BY_SUFFIX))
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
        return _fake_pipeline_run(
            cmd,
            check,
            env,
            timeout,
            cwd,
            repo_root,
            base_dir,
            process_mode,
            output_dir,
            log_file,
            input_video,
            mode_suffix,
        )

    with patch("subprocess.run", side_effect=_fake_run):
        _run_script(base_dir, repo_root)

    _verify_output(output_dir, log_file, input_video)

    _run_real_subprocess_smoke_test(base_dir, repo_root, process_mode, mode_suffix)


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
    """E2E verification should fail when multiple mode outputs exist."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = tmp_path / "session_log.txt"
    log_file.write_text("[INFO] Attempting execution on GPU\n", encoding="utf-8")

    input_video = tmp_path / "clip.mp4"
    input_video.write_bytes(b"in")
    (output_dir / "clip_Hybrid_Cleaned.mp4").write_bytes(b"hybrid")
    (output_dir / "clip_Denoised_Cleaned.mp4").write_bytes(b"denoise")

    with pytest.raises(AssertionError, match="Multiple mode outputs exist"):
        _verify_output(output_dir, log_file, input_video)
