"""Temp and file hygiene: every stage publishes atomically, partials are swept on resume, nothing lands outside the work folder."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

import modules.blend_weights as blend_weights
import modules.deepfilter_denoise as deepfilter_denoise
import modules.hum_cancel as hum_cancel
import modules.impulse_repair as impulse_repair
import modules.plosive_tamer as plosive_tamer
import modules.processing as processing
import modules.spectral_suppress as spectral_suppress
from modules import hygiene, utils

RATE = 44100


def _wav(path, seconds=1.0, channels=2):
    sf.write(str(path), np.full((int(RATE * seconds), channels), 0.1, dtype=np.float32), RATE, subtype="FLOAT")
    return path


# --------------------------------------------------------------------------- atomic_target


def test_partial_path_keeps_the_directory_and_extension():
    assert hygiene.partial_path(Path("a/b/stage_out.x.wav")) == Path("a/b/stage_out.x.tmp.wav")


def test_atomic_target_publishes_a_complete_write(tmp_path):
    target = tmp_path / "out.wav"
    with hygiene.atomic_target(target) as partial:
        assert partial.parent == tmp_path and partial != target
        partial.write_text("done")
    assert target.read_text() == "done"
    assert not partial.exists()


def test_atomic_target_removes_the_partial_when_the_stage_fails(tmp_path):
    target = tmp_path / "out.wav"
    with pytest.raises(RuntimeError):
        with hygiene.atomic_target(target) as partial:
            partial.write_text("half")
            raise RuntimeError("cut off")
    assert not partial.exists() and not target.exists()


def test_atomic_target_publishes_nothing_when_nothing_was_written(tmp_path):
    target = tmp_path / "out.wav"
    with hygiene.atomic_target(target):
        pass
    assert not target.exists()


def test_atomic_target_clears_a_stale_partial_first(tmp_path):
    target = tmp_path / "out.wav"
    hygiene.partial_path(target).write_text("stale")
    with hygiene.atomic_target(target) as partial:
        assert not partial.exists()
        partial.write_text("fresh")
    assert target.read_text() == "fresh"


# --------------------------------------------------------------------------- sweep_partials


def test_sweep_partials_removes_only_unfinished_files(tmp_path):
    work = tmp_path / ".temp_work_clip"
    (work / "stage").mkdir(parents=True)
    keep = [work / "original.wav", work / "stage" / "done.json", work / "stage" / "clip_(Vocals).wav"]
    drop = [
        work / "original.tmp.wav",
        work / "stage" / "polished_x.tmp.wav",
        work / "clip_Cleaned.tmp.mp4",
        work / "stage" / ".clip_(Vocals).wav.k3j2h.wav",
        work / "stage" / "scratch.tmp",
    ]
    for path in keep + drop:
        path.write_text("x")
    removed = hygiene.sweep_partials(work)
    assert sorted(removed) == sorted(drop)
    assert all(path.exists() for path in keep)
    assert not any(path.exists() for path in drop)


def test_sweep_partials_on_a_missing_directory_is_a_no_op(tmp_path):
    assert hygiene.sweep_partials(tmp_path / "nowhere") == []


def test_sweep_partials_reports_a_file_it_cannot_remove(tmp_path, capsys):
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.tmp.wav").write_text("x")
    with patch("modules.hygiene.Path.unlink", side_effect=OSError("locked")):
        assert hygiene.sweep_partials(work) == []
    assert "Could not remove partial" in capsys.readouterr().out


# --------------------------------------------------------------------------- scoped_temp_dir


def test_scoped_temp_dir_redirects_python_and_child_temp(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMP", "C:/somewhere")
    with hygiene.scoped_temp_dir(tmp_path) as scratch:
        assert scratch == tmp_path / "tmp" and scratch.is_dir()
        assert tempfile.gettempdir() == str(scratch)
        assert os.environ["TEMP"] == str(scratch) and os.environ["TMPDIR"] == str(scratch)
        assert Path(tempfile.mkstemp()[1]).parent == scratch


def test_scoped_temp_dir_restores_python_and_child_temp(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMP", "C:/somewhere")
    monkeypatch.delenv("TMPDIR", raising=False)
    before = tempfile.gettempdir()
    with hygiene.scoped_temp_dir(tmp_path):
        pass
    assert tempfile.gettempdir() == before
    assert os.environ["TEMP"] == "C:/somewhere" and "TMPDIR" not in os.environ


# --------------------------------------------------------------------------- session log


def test_log_rotates_once_past_the_cap_and_keeps_one_generation(tmp_path):
    log = tmp_path / "session_log.txt"
    log.write_text("old")
    assert utils._rotate_log_file(log, limit=2) is True
    assert not log.exists() and (log.with_name("session_log.txt.1")).read_text() == "old"
    log.write_text("newer")
    assert utils._rotate_log_file(log, limit=2) is True
    assert (log.with_name("session_log.txt.1")).read_text() == "newer"


def test_log_below_the_cap_or_missing_is_left_alone(tmp_path):
    log = tmp_path / "session_log.txt"
    assert utils._rotate_log_file(log) is False
    log.write_text("small")
    assert utils._rotate_log_file(log) is False and log.read_text() == "small"


def test_append_log_file_rotates_before_writing(tmp_path):
    log = tmp_path / "session_log.txt"
    log.write_text("x" * 10)
    with patch("modules.utils.LOG_FILE", log), patch("modules.utils.LOG_ROTATE_BYTES", 5):
        utils._append_log_file("INFO", "hello")
    assert "hello" in log.read_text() and log.with_name("session_log.txt.1").read_text() == "x" * 10


# --------------------------------------------------------------------------- remove_tree


def test_remove_tree_removes_a_populated_directory(tmp_path):
    work = tmp_path / "work" / "deep"
    work.mkdir(parents=True)
    (work / "f.wav").write_text("x")
    assert hygiene.remove_tree(tmp_path / "work") is True
    assert not (tmp_path / "work").exists()


def test_remove_tree_retries_then_reports_failure_without_raising(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    with patch("modules.hygiene.shutil.rmtree", side_effect=OSError("busy")) as rmtree, patch("modules.hygiene.time.sleep") as sleep:
        assert hygiene.remove_tree(work, attempts=3, delay_s=0.0) is False
    assert rmtree.call_count == 3 and sleep.call_count == 2


# --------------------------------------------------------------------------- processing stages


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
    source = _wav(tmp_path / "in.wav")
    output = tmp_path / "out.wav"

    def cut_off(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"RIFF" + b"\0" * 2000)
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


def test_staged_output_lives_inside_the_work_directory(tmp_path):
    final = tmp_path / "clip_PureLinear_Cleaned.mp4"
    staged = processing._staged_output_path(final, tmp_path / ".temp_work_clip")
    assert staged == tmp_path / ".temp_work_clip" / "clip_PureLinear_Cleaned.tmp.mp4"


def test_publish_staged_output_copies_when_a_rename_cannot_cross_volumes(tmp_path):
    staged = tmp_path / "work" / "clip.tmp.mp4"
    staged.parent.mkdir()
    staged.write_text("render")
    final = tmp_path / "clip.mp4"
    final.write_text("old")
    with patch("modules.processing.Path.rename", side_effect=OSError("cross-device")):
        processing._publish_staged_output(staged, final)
    assert final.read_text() == "render" and not staged.exists()


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


# --------------------------------------------------------------------------- streaming stage writers


def _only_files(directory):
    return sorted(path.name for path in Path(directory).rglob("*") if path.is_file())


def test_hum_subtract_interrupted_mid_stream_leaves_nothing(tmp_path):
    source = _wav(tmp_path / "in.wav")
    target = tmp_path / "out" / "humcancel_in.wav"
    target.parent.mkdir()
    with patch("modules.hum_cancel._synthesise", side_effect=RuntimeError("cut")), pytest.raises(RuntimeError):
        hum_cancel.subtract(source, target, np.zeros((1, 2, 1)), [50.0])
    assert _only_files(target.parent) == []


def test_hum_subtract_publishes_a_complete_file(tmp_path):
    source = _wav(tmp_path / "in.wav")
    target = tmp_path / "humcancel_in.wav"
    envelope = np.zeros((4, 2, 1), dtype=np.complex128)
    assert hum_cancel.subtract(source, target, envelope, [50.0]) == target
    assert sf.info(str(target)).frames == RATE and _only_files(tmp_path) == ["humcancel_in.wav", "in.wav"]


def test_plosive_tame_interrupted_mid_stream_leaves_nothing(tmp_path):
    source = _wav(tmp_path / "in.wav")
    target = tmp_path / "out" / "tamed_in.wav"
    target.parent.mkdir()
    with patch("modules.plosive_tamer.gain_curve", side_effect=RuntimeError("cut")), pytest.raises(RuntimeError):
        plosive_tamer.tame_file(source, target, [], 0.0, 0.0)
    assert _only_files(target.parent) == []


def test_suppress_interrupted_mid_stream_leaves_nothing(tmp_path):
    source = _wav(tmp_path / "in.wav", seconds=3.0)
    target = tmp_path / "out" / "suppressed_in.wav"
    target.parent.mkdir()
    with patch("modules.spectral_suppress.Suppressor", side_effect=RuntimeError("cut")), pytest.raises(RuntimeError):
        spectral_suppress.suppress_file(source, target, 1.0, -20.0, 0.9, 0.5)
    assert _only_files(target.parent) == []


def test_deepfilter_interrupted_mid_stream_leaves_nothing(tmp_path):
    source = _wav(tmp_path / "in.wav")
    out_dir = tmp_path / "dfn"
    with (
        patch("modules.deepfilter_denoise._load", return_value=object()),
        patch("modules.deepfilter_denoise._stream_denoise", side_effect=RuntimeError("cut")),
        pytest.raises(RuntimeError),
    ):
        deepfilter_denoise.denoise(source, out_dir)
    assert _only_files(out_dir) == []


def test_blend_refusal_leaves_nothing(tmp_path):
    original = _wav(tmp_path / "orig.wav")
    denoised = _wav(tmp_path / "den.wav")
    target = tmp_path / "out" / "blended_den.wav"
    with (
        patch("modules.blend_weights._recording_statistics", return_value=None),
        patch("modules.blend_weights._blend_block", return_value=None),
        sf.SoundFile(str(original)) as a,
        sf.SoundFile(str(denoised)) as b,
    ):
        assert blend_weights._write_blend((a, b), RATE, RATE, 2, object(), target) is None
    assert _only_files(target.parent) == []


def test_depop_interrupted_mid_write_leaves_nothing(tmp_path, monkeypatch):
    source = _wav(tmp_path / "in.wav", channels=1)
    out_dir = tmp_path / "repair"

    def cut(path, *args, **kwargs):
        Path(path).write_bytes(b"RIFF" + b"\0" * 2000)
        raise RuntimeError("cut")

    monkeypatch.setattr(impulse_repair.sf, "write", cut)
    with pytest.raises(RuntimeError):
        impulse_repair.depop(source, out_dir, threshold=0.5)
    assert _only_files(out_dir) == []
