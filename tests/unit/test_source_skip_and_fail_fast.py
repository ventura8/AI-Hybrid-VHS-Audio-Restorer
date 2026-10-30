"""Deterministic failures end at once: no retry on fewer threads, and a source with nothing to restore is a logged skip.

Hunt v3 (2026-10-11): everest-the-mountain-... has no audio stream and its extraction ran six
times; Pooh, Powerpuff and Rocky decode to a 92-byte WAV with no samples. The real-ffmpeg
cases rebuild both shapes synthetically: a video-only MP4, and an MP4 whose AAC packets all
sit before its edit list (as Pooh's do), which ffmpeg decodes to nothing with exit code 0.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

import modules.processing as processing
from modules import ffmpeg_failures, source_checks, utils
from modules.source_checks import SourceSkipped

EVEREST_TAIL = (
    "[out#0/wav @ 000002406a9af440] Output file does not contain any stream\n"
    "Error opening output file original.tmp.wav.\n"
    "Error opening output files: Invalid argument\n"
)
# ffmpeg 9.0.1 on an MPEG-PS capture with broken MP2 frames: printed for every bad frame, exit code 0.
DAMAGED_CAPTURE_TAIL = (
    "[mp2 @ 000002517436ccc0] Header missing\n"
    "[aist#0:1/mp2 @ 0000025174365cc0] [dec:mp2 @ 000002517436cb40] Error submitting packet to decoder: "
    "Invalid data found when processing input\n"
)


def _failure(output):
    return subprocess.CalledProcessError(1, ["ffmpeg"], output=output)


@pytest.mark.parametrize("marker", ffmpeg_failures.DETERMINISTIC_MARKERS)
def test_every_marker_reads_as_deterministic(marker):
    assert ffmpeg_failures.is_deterministic_failure(_failure(f"[x @ 0] {marker.upper()}\n"))


@pytest.mark.parametrize("output", ["Cannot allocate memory\n", "", None, b"Killed\n"])
def test_memory_and_silent_failures_keep_the_retry(output):
    assert not ffmpeg_failures.is_deterministic_failure(_failure(output))


def test_bytes_output_is_read_leniently():
    assert ffmpeg_failures.is_deterministic_failure(_failure(b"\x81 Stream map '' matches no streams.\n"))


def test_the_everest_tail_is_deterministic():
    assert ffmpeg_failures.is_deterministic_failure(_failure(EVEREST_TAIL))


def test_an_unreadable_input_is_read_from_the_open_error():
    tail = "[in#0 @ 0] Error opening input: Invalid data found when processing input\nError opening input file tape.mp4.\n"
    assert ffmpeg_failures.is_deterministic_failure(_failure(tail))


@pytest.mark.parametrize(
    "tail", [DAMAGED_CAPTURE_TAIL, DAMAGED_CAPTURE_TAIL + "Cannot allocate memory\n", EVEREST_TAIL + "Not enough space\n"]
)
def test_a_damaged_capture_or_a_lack_of_memory_keeps_the_retry(tail):
    """Broken MP2 frames print 'Invalid data found when processing input' on a run that succeeds; memory always retries."""
    assert not ffmpeg_failures.is_deterministic_failure(_failure(tail))


def _child(script):
    return [sys.executable, "-c", script]


def test_a_failed_command_carries_its_tail_on_the_error(capsys):
    command = _child("print('Output file does not contain any stream'); raise SystemExit(3)")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        utils.run_command_with_progress(command, description="Probe")
    assert caught.value.returncode == 3
    assert "does not contain any stream" in caught.value.output
    capsys.readouterr()


def _attempts(script, threads=8):
    built = []

    def build(count):
        built.append(count)
        return _child(script)

    with pytest.raises(subprocess.CalledProcessError), patch("modules.utils.log_msg") as log:
        utils.attempt_cpu_run_with_retry(build, threads, description="Extracting Audio", total_duration=1.0)
    return built, [call.args[0] for call in log.call_args_list]


def test_a_deterministic_failure_runs_once(capsys):
    built, logged = _attempts("import sys; sys.stdout.write('Output file does not contain any stream\\n'); raise SystemExit(1)")
    assert built == [8]
    assert any("not retrying" in line for line in logged)
    capsys.readouterr()


def test_any_other_failure_still_halves_the_threads(capsys):
    built, _logged = _attempts("import sys; sys.stdout.write('Cannot allocate memory\\n'); raise SystemExit(1)")
    assert built == [8, 4, 2, 1]
    capsys.readouterr()


@pytest.mark.parametrize(("types", "skipped"), [({"video"}, True), ({"video", "audio"}, False), (set(), False)])
def test_only_a_readable_source_without_audio_is_skipped(monkeypatch, types, skipped):
    monkeypatch.setattr(source_checks, "_probe_stream_types", lambda path: types)
    if skipped:
        with pytest.raises(SourceSkipped, match="no audio stream"):
            source_checks.require_audio_stream("tape.mp4")
    else:
        source_checks.require_audio_stream("tape.mp4")


def _wav(path, frames):
    sf.write(str(path), np.zeros((frames, 2), dtype=np.float32), 44100, subtype="FLOAT")
    return path


def test_an_empty_decode_is_removed_and_skipped(tmp_path):
    empty = _wav(tmp_path / "original.tmp.wav", 0)
    with pytest.raises(SourceSkipped, match="decodes to no samples"):
        source_checks.reject_empty_decode(empty)
    assert not empty.exists()


@pytest.mark.parametrize("content", ["frames", "garbage", "missing"])
def test_anything_but_an_empty_decode_is_left_to_the_caller(tmp_path, content):
    path = tmp_path / "original.tmp.wav"
    if content == "frames":
        _wav(path, 10)
    elif content == "garbage":
        path.write_bytes(b"not a wav")
    source_checks.reject_empty_decode(path)
    assert path.exists() is (content != "missing")


def test_without_soundfile_nothing_is_called_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(source_checks, "sf", None)
    source_checks.reject_empty_decode(_wav(tmp_path / "a.wav", 0))


def test_the_extraction_skips_a_source_without_audio_before_ffmpeg(tmp_path):
    with (
        patch.object(source_checks, "_probe_stream_types", return_value={"video"}),
        patch.object(processing, "attempt_cpu_run_with_retry") as run,
        pytest.raises(SourceSkipped),
    ):
        processing._extract_audio_step(tmp_path / "tape.mp4", tmp_path / "original.wav")
    run.assert_not_called()


def _extract_writing(tmp_path, writer):
    """Runs the extraction with ffmpeg replaced by `writer`, which fills the temporary WAV."""
    original = tmp_path / "original.wav"

    def fake_run(_build, _threads, **_kwargs):
        writer(original.with_suffix(".tmp.wav"))

    with (
        patch.object(source_checks, "_probe_stream_types", return_value={"video", "audio"}),
        patch.object(processing, "attempt_cpu_run_with_retry", side_effect=fake_run),
    ):
        processing._extract_audio_step(tmp_path / "tape.mp4", original)
    return original


def test_the_extraction_turns_an_empty_decode_into_a_skip(tmp_path):
    with pytest.raises(SourceSkipped):
        _extract_writing(tmp_path, lambda path: _wav(path, 0))
    assert not (tmp_path / "original.tmp.wav").exists()


def test_the_extraction_still_fails_on_an_unreadable_output(tmp_path):
    with pytest.raises(RuntimeError, match="Extraction failed"):
        _extract_writing(tmp_path, lambda path: path.write_bytes(b"x" * 2048))
    assert not (tmp_path / "original.tmp.wav").exists()


def test_the_extraction_publishes_a_valid_decode(tmp_path):
    original = _extract_writing(tmp_path, lambda path: _wav(path, 44100))
    assert sf.info(str(original)).frames == 44100


def test_a_skipped_task_logs_the_reason_and_returns_false(tmp_path):
    video = tmp_path / "everest.mp4"
    video.write_bytes(b"x")
    with (
        patch.object(processing, "_log_video_duration", return_value=15.1),
        patch.object(processing, "_extract_audio_step", side_effect=SourceSkipped(source_checks.NO_AUDIO_STREAM)),
        patch.object(processing, "log_msg") as log,
    ):
        assert processing.process_hybrid_audio(video, "GPU", target_output_dir=tmp_path) is False
    lines = [call.args[0] for call in log.call_args_list]
    assert any(line.startswith("  [Skip] everest.mp4: the source has no audio stream") for line in lines)
    assert not any("Processing failed" in line for line in lines)


def _available(binary):
    return shutil.which(binary) is not None or Path(binary).is_file()


needs_ffmpeg = pytest.mark.skipif(
    not (_available(utils.FFMPEG_BIN) and _available(utils.FFPROBE_BIN)), reason="needs the ffmpeg and ffprobe binaries"
)


def _make(path, *args):
    lavfi = ["-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=1"]
    subprocess.run([utils.FFMPEG_BIN, "-v", "error", "-y", *lavfi, *args, str(path)], check=True, capture_output=True)
    return path


@needs_ffmpeg
def test_a_real_video_only_source_is_skipped_without_running_ffmpeg(tmp_path):
    source = _make(tmp_path / "video_only.mp4", "-c:v", "mpeg4")
    with patch.object(processing, "attempt_cpu_run_with_retry") as run, pytest.raises(SourceSkipped):
        processing._extract_audio_step(source, tmp_path / "original.wav", total_duration=1.0)
    run.assert_not_called()


@needs_ffmpeg
def test_a_real_source_whose_audio_decodes_to_nothing_is_skipped(tmp_path, capsys):
    hidden = ["-itsoffset", "-5", "-f", "lavfi", "-i", "sine=sample_rate=44100:duration=1"]
    source = _make(
        tmp_path / "hidden_audio.mp4",
        *hidden,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "mpeg4",
        "-c:a",
        "aac",
        "-avoid_negative_ts",
        "disabled",
    )
    with pytest.raises(SourceSkipped, match="decodes to no samples"):
        processing._extract_audio_step(source, tmp_path / "original.wav", total_duration=1.0)
    assert not list(tmp_path.glob("*.wav"))
    capsys.readouterr()


@needs_ffmpeg
def test_a_real_unreadable_source_runs_ffmpeg_once(tmp_path, capsys):
    source = tmp_path / "garbage.mp4"
    source.write_bytes(bytes(range(256)) * 12)
    calls = []
    real_run = utils.run_command_with_progress

    def counted(cmd, *args, **kwargs):
        calls.append(cmd)
        return real_run(cmd, *args, **kwargs)

    with (
        patch.object(utils, "run_command_with_progress", side_effect=counted),
        patch.object(processing, "CPU_THREADS", 8),
        pytest.raises(subprocess.CalledProcessError) as caught,
    ):
        processing._extract_audio_step(source, tmp_path / "original.wav", total_duration=1.0)
    assert len(calls) == 1
    assert "error opening input" in caught.value.output.lower()
    capsys.readouterr()


def _damaged_capture(folder):
    """A 4 s MPEG-PS capture (MPEG-2 video, MP2 audio) with garbage written over packs in its middle."""
    clean = _make(folder / "clean.mpg", "-f", "lavfi", "-i", "sine=sample_rate=48000:duration=4", "-c:v", "mpeg2video", "-c:a", "mp2")
    data = bytearray(clean.read_bytes())
    garbage = bytes((i * 37) & 0xFF for i in range(600))
    for start in range(len(data) // 3, len(data) // 3 + 120_000, 9000):
        end = start + len(garbage)
        data[start:end] = garbage
    damaged = folder / "damaged.mpg"
    damaged.write_bytes(bytes(data))
    return damaged


@needs_ffmpeg
def test_a_real_damaged_capture_does_not_read_as_deterministic(tmp_path):
    """ffmpeg decodes it (exit 0) while printing 'Invalid data found when processing input'; an OOM after that still retries."""
    source = _damaged_capture(tmp_path)
    decoded = subprocess.run(
        [utils.FFMPEG_BIN, "-hide_banner", "-i", str(source), "-acodec", "pcm_f32le", "-ar", "44100", "-y", str(tmp_path / "o.wav")],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert decoded.returncode == 0
    assert "invalid data found when processing input" in decoded.stderr.lower()
    assert not ffmpeg_failures.is_deterministic_failure(_failure(decoded.stderr))
    assert not ffmpeg_failures.is_deterministic_failure(_failure(decoded.stderr + "Cannot allocate memory\n"))
