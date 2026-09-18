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


def test_publish_flushes_the_file_and_on_posix_its_directory(tmp_path):
    """A rename is visible at once but not durable; the bytes and the entry are synced."""
    partial, target = tmp_path / "out.tmp.wav", tmp_path / "out.wav"
    partial.write_text("done")
    with patch("modules.hygiene._fsync") as fsync, patch("modules.hygiene.SYNC_DIRECTORIES", True):
        hygiene.publish(partial, target)
    assert [call.args[0] for call in fsync.call_args_list] == [partial, tmp_path]
    assert target.read_text() == "done" and not partial.exists()


def test_publish_skips_the_directory_sync_where_it_is_unsupported(tmp_path):
    partial, target = tmp_path / "out.tmp.wav", tmp_path / "out.wav"
    partial.write_text("done")
    with patch("modules.hygiene._fsync") as fsync, patch("modules.hygiene.SYNC_DIRECTORIES", False):
        hygiene.publish(partial, target)
    assert [call.args[0] for call in fsync.call_args_list] == [partial]


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
        descriptor, temp_path = tempfile.mkstemp()
        try:
            assert Path(temp_path).parent == scratch
        finally:
            os.close(descriptor)


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


def test_blend_refusal_keeps_an_earlier_result(tmp_path):
    """A refusal must not publish a prefix over a complete blend, nor delete it."""
    original = _wav(tmp_path / "orig.wav")
    denoised = _wav(tmp_path / "den.wav")
    target = tmp_path / "out" / "blended_den.wav"
    target.parent.mkdir()
    target.write_text("an earlier complete blend")
    with (
        patch("modules.blend_weights._recording_statistics", return_value=None),
        patch("modules.blend_weights._blend_block", return_value=None),
        sf.SoundFile(str(original)) as a,
        sf.SoundFile(str(denoised)) as b,
    ):
        assert blend_weights._write_blend((a, b), RATE, RATE, 2, object(), target) is None
    assert target.read_text() == "an earlier complete blend"
    assert _only_files(target.parent) == ["blended_den.wav"]


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
