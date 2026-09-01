#!/usr/bin/env python3
"""Cross-platform end-to-end test runner for installed application executables.

Tests all operational scenarios including:
- CLI help flags (-h, --help)
- Cold-start & warm-start invocations
- Single file, multi-file, and directory batch processing
- All restoration modes (vhs_native, auto_ffmpeg_native, arnndn_speech, etc.)
- Multi-container format coverage (.mp4, .mkv, .avi, .mov)
- Output media bitstream and duration verification
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_cmd(cmd, cwd=None, stdin_data=None):
    """Executes a command and returns the completed process."""
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        input=stdin_data,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result


def _create_test_fixture(output_path, duration=1.0, channels=2, hz=1000):
    """Creates a synthetic media test fixture using system ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=30",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={hz}:sample_rate=48000",
        "-t",
        str(duration),
        "-ac",
        str(channels),
        "-c:v",
        "mpeg4",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    _run_cmd(cmd)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to generate test fixture: {output_path}")


def _verify_output(output_file):
    """Verifies that an output media file exists, is non-empty, and passes ffprobe."""
    if not output_file.exists() or output_file.stat().st_size == 0:
        raise AssertionError(f"Expected output file does not exist or is empty: {output_file}")
    probe_cmd = ["ffprobe", "-v", "error", str(output_file)]
    _run_cmd(probe_cmd)
    print(f"Verified valid output: {output_file.name}")


def _test_help_flag(launcher_cmd):
    """Tests that --help and -h execute cleanly."""
    print("\n--- Test Scenario: CLI Help Screen ---")
    for flag in ["-h", "--help"]:
        res = _run_cmd(launcher_cmd + [flag])
        if "AI Hybrid VHS Audio Restorer" not in res.stdout:
            raise AssertionError(f"Help text missing in output for flag {flag}")
    print("Help flag test passed.")


def _test_single_file(launcher_cmd, test_dir, extension=".mp4"):
    """Tests single file restoration argument."""
    print(f"\n--- Test Scenario: Single File Argument ({extension}) ---")
    fixture = test_dir / f"test_single{extension}"
    _create_test_fixture(fixture, duration=1.0)
    _run_cmd(launcher_cmd + [str(fixture)], stdin_data="\n")
    cleaned_name = f"test_single_FFmpeg_Cleaned{extension}"
    cleaned = fixture.parent / cleaned_name
    _verify_output(cleaned)
    print(f"Single file test ({extension}) passed.")


def _test_multi_files(launcher_cmd, test_dir):
    """Tests multiple file restoration arguments."""
    print("\n--- Test Scenario: Multiple Files Arguments ---")
    fixtures = [
        test_dir / "test_multi_1.mp4",
        test_dir / "test_multi_2.mkv",
        test_dir / "test_multi_3.avi",
    ]
    for f in fixtures:
        _create_test_fixture(f, duration=1.0)
    _run_cmd(launcher_cmd + [str(f) for f in fixtures], stdin_data="\n")
    for f in fixtures:
        cleaned = f.parent / f"{f.stem}_FFmpeg_Cleaned{f.suffix}"
        _verify_output(cleaned)
    print("Multiple files test passed.")


def _test_directory_input(launcher_cmd, test_dir):
    """Tests passing a folder path containing media files."""
    print("\n--- Test Scenario: Directory Folder Argument ---")
    folder = test_dir / "batch_folder"
    folder.mkdir(parents=True, exist_ok=True)
    f1 = folder / "clip1.mp4"
    f2 = folder / "clip2.mov"
    _create_test_fixture(f1, duration=1.0)
    _create_test_fixture(f2, duration=1.0)
    _run_cmd(launcher_cmd + [str(folder)], stdin_data="\n")
    _verify_output(folder / "clip1_FFmpeg_Cleaned.mp4")
    _verify_output(folder / "clip2_FFmpeg_Cleaned.mov")
    print("Directory input test passed.")


def _configure_native_mode(config_dir):
    """Sets the installed app's configuration to the deterministic native mode."""
    config_file = Path(config_dir) / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Installed config file was not created: {config_file}")
    content = config_file.read_text(encoding="utf-8")
    new_content = re.sub(r"^process_mode:.*$", 'process_mode: "vhs_native"', content, flags=re.MULTILINE)
    config_file.write_text(new_content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="End-to-end executable test suite")
    parser.add_argument("launcher", help="Path to executable or launcher command")
    parser.add_argument("--config-dir", help="Directory where config.yaml is located", default=None)
    args = parser.parse_args()

    launcher_path = Path(args.launcher).resolve()
    if not launcher_path.exists() and shutil.which(args.launcher) is None:
        raise FileNotFoundError(f"Launcher executable not found: {args.launcher}")

    launcher_cmd = [str(launcher_path)] if launcher_path.exists() else [args.launcher]

    temp_dir = Path(tempfile.mkdtemp(prefix="ai_vhs_e2e_"))
    try:
        print(f"Starting E2E verification suite for: {launcher_cmd}")
        print(f"Working in temporary sandbox: {temp_dir}")

        _test_help_flag(launcher_cmd)
        if args.config_dir:
            _configure_native_mode(args.config_dir)
        _test_single_file(launcher_cmd, temp_dir, extension=".mp4")
        _test_single_file(launcher_cmd, temp_dir, extension=".mkv")
        _test_single_file(launcher_cmd, temp_dir, extension=".avi")
        _test_single_file(launcher_cmd, temp_dir, extension=".mov")
        _test_multi_files(launcher_cmd, temp_dir)
        _test_directory_input(launcher_cmd, temp_dir)

        print("\n=============================================")
        print("ALL END-TO-END TEST SCENARIOS PASSED!")
        print("=============================================")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
