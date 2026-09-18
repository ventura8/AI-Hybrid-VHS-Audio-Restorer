"""Hygiene of the processing stages: FFmpeg renders, the staged final mux, resume, and cleanup."""

import errno
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

import modules.processing as processing
from modules import hygiene, utils

RATE = 44100


def _wav(path, seconds=1.0, channels=2):
    sf.write(str(path), np.full((int(RATE * seconds), channels), 0.1, dtype=np.float32), RATE, subtype="FLOAT")
    return path


# --------------------------------------------------------------------------- FFmpeg filter steps


def test_dsp_filter_publishes_only_a_complete_render(tmp_path):
    source = _wav(tmp_path / "in.wav")
    output = tmp_path / "out.wav"

    def render(cmd, **kwargs):
        assert Path(cmd[-1]) == hygiene.partial_path(output)
        _wav(Path(cmd[-1]))

    with patch("modules.processing.run_command_with_progress", side_effect=render):
        assert processing._run_dsp_filter_file(source, output, "anull", "Test", 1.0) == output
    assert utils.is_valid_audio(output) and not hygiene.partial_path(output).exists()


def test_dsp_filter_failure_leaves_no_partial_and_falls_back(tmp_path):
    """A killed encoder leaves a header-valid WAV; the stage must not publish it."""
    source = _wav(tmp_path / "in.wav")
    output = tmp_path / "out.wav"

    def cut_off(cmd, **kwargs):
        _wav(Path(cmd[-1]))
        assert utils.is_valid_audio(Path(cmd[-1]))
        raise RuntimeError("power cut")

    with patch("modules.processing.run_command_with_progress", side_effect=cut_off):
        assert processing._run_dsp_filter_file(source, output, "anull", "Test", 1.0) == source
    assert not output.exists() and not hygiene.partial_path(output).exists()


def test_copy_result_to_final_dir_leaves_no_partial(tmp_path):
    result = _wav(tmp_path / "joined.wav")
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    published = processing._copy_result_to_final_dir(result, final_dir)
    assert published == final_dir / "joined.wav" and utils.is_valid_audio(published)
    assert list(final_dir.iterdir()) == [published]


# --------------------------------------------------------------------------- staged final render


def test_staged_output_lives_inside_the_work_directory(tmp_path):
    final = tmp_path / "clip_PureLinear_Cleaned.mp4"
    staged = processing._staged_output_path(final, tmp_path / ".temp_work_clip")
    assert staged == tmp_path / ".temp_work_clip" / "clip_PureLinear_Cleaned.tmp.mp4"


def _cross_device(staged):
    """os.replace that refuses to move `staged` itself, as a cross-volume rename would."""
    real_replace = os.replace

    def replace(src, dst):
        if Path(src) == staged:
            raise OSError(errno.EXDEV, "cross-device link")
        real_replace(src, dst)

    return replace


def test_publish_staged_output_stages_a_copy_when_a_rename_cannot_cross_volumes(tmp_path):
    staged = tmp_path / "work" / "clip.tmp.mp4"
    staged.parent.mkdir()
    staged.write_text("render")
    final = tmp_path / "clip.mp4"
    final.write_text("old")
    with patch("modules.hygiene.os.replace", side_effect=_cross_device(staged)):
        processing._publish_staged_output(staged, final)
    assert final.read_text() == "render" and not staged.exists()
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_publish_staged_output_raises_any_error_that_is_not_a_cross_device_rename(tmp_path):
    """A failure other than EXDEV must not be masked by an attempt to copy a file that may already have moved."""
    staged = tmp_path / "work" / "clip.tmp.mp4"
    staged.parent.mkdir()
    staged.write_text("render")
    final = tmp_path / "clip.mp4"
    final.write_text("old")
    with (
        patch("modules.hygiene.os.replace", side_effect=PermissionError(errno.EACCES, "locked")),
        patch("modules.processing.shutil.copy2") as copy,
        pytest.raises(PermissionError),
    ):
        processing._publish_staged_output(staged, final)
    assert copy.call_count == 0 and final.read_text() == "old" and staged.exists()


def test_publish_staged_output_keeps_the_previous_output_when_the_copy_fails(tmp_path):
    """A cross-volume copy that dies must not have removed or half-written the old output."""
    staged = tmp_path / "work" / "clip.tmp.mp4"
    staged.parent.mkdir()
    staged.write_text("render")
    final = tmp_path / "clip.mp4"
    final.write_text("old")
    with (
        patch("modules.hygiene.os.replace", side_effect=_cross_device(staged)),
        patch("modules.processing.shutil.copy2", side_effect=OSError("disk full")),
        pytest.raises(OSError),
    ):
        processing._publish_staged_output(staged, final)
    assert final.read_text() == "old"
    assert staged.exists()
    assert list(tmp_path.glob("*.tmp.*")) == []


@patch("modules.processing.attempt_cpu_run_with_retry")
@patch("modules.processing.is_valid_video")
def test_final_remux_renders_inside_the_work_directory(mock_valid_video, mock_retry, tmp_path):
    work = tmp_path / ".temp_work_clip"
    work.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_text("video")
    aligned = work / "aligned.wav"
    aligned.write_text("audio")
    final = tmp_path / "clip_Cleaned.mp4"
    mock_valid_video.side_effect = [False, True]

    def render(cmd_builder, threads, **kwargs):
        rendered = Path(cmd_builder(threads)[-1])
        assert rendered.parent == work
        rendered.write_text("render")

    mock_retry.side_effect = render
    processing._final_mux_single_audio_step(video, aligned, final)
    assert final.read_text() == "render"
    assert sorted(path.name for path in tmp_path.iterdir()) == [".temp_work_clip", "clip.mp4", "clip_Cleaned.mp4"]


# --------------------------------------------------------------------------- resume and cleanup


def test_skipping_an_existing_output_still_removes_a_stale_work_directory(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_text("video")
    work = tmp_path / ".temp_work_clip"
    work.mkdir()
    (work / "original.wav").write_text("left behind")
    with patch("modules.processing.is_valid_video", return_value=True), patch("modules.processing.KEEP_INPUT_FILES", False):
        assert processing.process_hybrid_audio(video, "GPU") is True
    assert not work.exists()


@patch("modules.processing._run_processing_mode")
@patch("modules.processing._extract_audio_step")
@patch("modules.processing.get_video_duration_sec", return_value=1.0)
def test_a_resumed_run_sweeps_partials_and_scopes_temp_to_the_work_directory(mock_dur, mock_extract, mock_mode, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_text("video")
    work = tmp_path / ".temp_work_clip"
    work.mkdir()
    (work / "original.tmp.wav").write_text("cut off")
    seen = {}

    def run_mode(*args, **kwargs):
        seen["tempdir"] = tempfile.gettempdir()
        seen["partial"] = (work / "original.tmp.wav").exists()

    mock_mode.side_effect = run_mode
    with patch("modules.processing.is_valid_video", return_value=False), patch("modules.processing.KEEP_INPUT_FILES", True):
        assert processing.process_hybrid_audio(video, "GPU") is True
    assert seen == {"tempdir": str(work / "tmp"), "partial": False}
    assert tempfile.gettempdir() != str(work / "tmp")


def test_cleanup_warns_when_the_work_directory_cannot_be_removed(tmp_path, capsys):
    work = tmp_path / ".temp_work_clip"
    work.mkdir()
    with (
        patch("modules.processing.is_valid_video", return_value=True),
        patch("modules.processing.KEEP_INPUT_FILES", False),
        patch("modules.processing.remove_tree", return_value=False),
    ):
        processing._cleanup_work_dir(work, tmp_path / "clip_Cleaned.mp4")
    assert "Could not remove" in capsys.readouterr().out
