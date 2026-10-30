"""A planted `.temp_work_<stem>` link never leads the sweep, the stages or the cleanup into another tree (2026-10-26).

The vulnerability scan's finding: a folder of tapes could ship `.temp_work_<stem>` as a link to the user's home;
mkdir(exist_ok=True) accepted it, the partial sweep walked the tree it points to and unlinked every `*.tmp*` and hidden
`*.wav` there, and the stages wrote original.wav and their scratch into it. Now the work directory is refused, the
sweep follows no link, and a stale partial that is a dangling link is removed instead of written through. A real
work directory's own links (one named original.tmp.wav, which the extraction's ffmpeg -y wrote through) are
removed by the sweep since the security review of PR #21 (2026-10-29): the app makes none.
"""

import errno
import importlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import modules.processing as processing
from modules import hygiene

VICTIM_FILES = ("todo.tmp", "document.tmp", "proj/build/out.tmp.wav", ".voice.wav", "notes.txt")


def _dir_link(link, target):
    """A directory symbolic link, or a junction where the platform refuses one (Windows without Developer Mode)."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        importlib.import_module("_winapi").CreateJunction(str(target), str(link))


def _file_link(link, target):
    try:
        os.symlink(target, link)
    except OSError as refused:
        pytest.skip(f"file links need a privilege this platform withholds: {refused}")


def _victim_home(tmp_path):
    home = tmp_path / "home"
    for name in VICTIM_FILES:
        (home / name).parent.mkdir(parents=True, exist_ok=True)
        (home / name).write_text("keep", encoding="utf-8")
    return home


def _all_kept(home):
    return all((home / name).exists() for name in VICTIM_FILES)


def test_a_planted_work_dir_link_is_refused_and_the_sweep_leaves_its_target(tmp_path):
    home = _victim_home(tmp_path)
    tapes = tmp_path / "tapes"
    tapes.mkdir()
    planted = tapes / ".temp_work_holiday"
    _dir_link(planted, home)
    with pytest.raises(hygiene.WorkDirRefused, match="link"):
        hygiene.open_work_dir(planted)
    assert hygiene.sweep_partials(planted) == []
    assert _all_kept(home)


def test_the_orchestrator_skips_the_tape_and_writes_nothing_into_the_linked_tree(tmp_path):
    home = _victim_home(tmp_path)
    tapes = tmp_path / "tapes"
    tapes.mkdir()
    video = tapes / "holiday.mp4"
    video.write_bytes(b"not decoded")
    _dir_link(tapes / ".temp_work_holiday", home)
    with patch.object(processing, "is_valid_video", return_value=False):
        assert processing.process_hybrid_audio(video, "GPU") is False
    assert _all_kept(home)
    assert not (home / "original.wav").exists()
    assert hygiene.is_link(tapes / ".temp_work_holiday")


def test_a_real_work_dir_loses_its_partials_and_a_planted_directory_link_but_nothing_through_it(tmp_path):
    home = _victim_home(tmp_path)
    work = hygiene.open_work_dir(tmp_path / ".temp_work_tape")
    (work / "stage").mkdir()
    ours = [work / "stage" / "out.tmp.wav", work / ".sep.wav", work / "x.tmp"]
    for path in ours:
        path.write_bytes(b"partial")
    (work / "stage" / "done.wav").write_bytes(b"finished")
    _dir_link(work / "escape", home)
    removed = hygiene.sweep_partials(work)
    assert sorted(removed) == sorted([*ours, work / "escape"])
    assert ((work / "stage" / "done.wav").exists(), hygiene.is_link(work / "escape")) == (True, False)
    assert _all_kept(home)


def test_a_planted_file_link_named_like_a_partial_is_swept_and_its_target_kept(tmp_path):
    """The security review of PR #21 (2026-10-29): a real work directory shipping `original.tmp.wav` as a link to a
    file the user owns kept it through the sweep, and the extraction's `ffmpeg -y` overwrote the target."""
    victim = tmp_path / "home" / ".bashrc"
    victim.parent.mkdir()
    victim.write_text("keep", encoding="utf-8")
    work = hygiene.open_work_dir(tmp_path / ".temp_work_tape")
    _file_link(work / "original.tmp.wav", victim)
    _file_link(work / "original.wav", victim)
    assert sorted(hygiene.sweep_partials(work)) == sorted([work / "original.tmp.wav", work / "original.wav"])
    assert (list(work.iterdir()), victim.read_text(encoding="utf-8")) == ([], "keep")


def test_a_link_the_platform_will_not_unlink_is_removed_with_rmdir(tmp_path):
    link = tmp_path / "junction"
    with patch.object(Path, "unlink", side_effect=PermissionError("a directory")), patch.object(os, "rmdir") as rmdir:
        hygiene._remove_link(link)
    rmdir.assert_called_once_with(link)


def test_a_link_the_sweep_cannot_remove_is_reported_and_the_sweep_goes_on(tmp_path):
    work = hygiene.open_work_dir(tmp_path / ".temp_work_tape")
    (work / "x.tmp").write_bytes(b"partial")
    _dir_link(work / "escape", _victim_home(tmp_path))
    with patch.object(hygiene, "_remove_link", side_effect=OSError("in use")), patch.object(hygiene, "log_msg") as log:
        removed = hygiene.sweep_partials(work)
    assert removed == [work / "x.tmp"]
    assert any("Could not remove link escape" in call.args[0] for call in log.call_args_list)


def test_remove_stale_takes_a_file_a_link_or_nothing(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    plain = tmp_path / "plain.tmp.wav"
    plain.write_bytes(b"partial")
    hygiene.remove_stale(plain)
    hygiene.remove_stale(tmp_path / "absent.wav")
    _file_link(tmp_path / "linked.tmp.wav", victim)
    hygiene.remove_stale(tmp_path / "linked.tmp.wav")
    assert (plain.exists(), hygiene.is_link(tmp_path / "linked.tmp.wav"), victim.read_text(encoding="utf-8")) == (False, False, "keep")


def test_the_extraction_removes_a_stale_partial_or_link_before_ffmpeg_writes(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    work = tmp_path / ".temp_work_tape"
    work.mkdir()
    original = work / "original.wav"
    _file_link(work / "original.tmp.wav", victim)
    seen = []

    def run(build, *_args, **_kwargs):
        seen.append(hygiene.is_link(work / "original.tmp.wav"))
        Path(build(1)[-1]).write_bytes(b"audio")

    with (
        patch.object(processing, "is_valid_audio", return_value=False),
        patch.object(processing._source_checks, "require_audio_stream"),
        patch.object(processing, "attempt_cpu_run_with_retry", side_effect=run),
        patch.object(processing, "_publish_extraction"),
    ):
        processing._extract_audio_step(tmp_path / "tape.mp4", original)
    assert (seen, victim.read_text(encoding="utf-8")) == ([False], "keep")


def test_a_path_holding_a_file_is_not_a_work_dir(tmp_path):
    occupied = tmp_path / ".temp_work_tape"
    occupied.write_text("a file", encoding="utf-8")
    with pytest.raises(OSError):
        hygiene.open_work_dir(occupied)


def test_a_dangling_link_at_a_partial_is_removed_not_written_through(tmp_path):
    outside = tmp_path / "outside.txt"
    target = tmp_path / "work" / "stage.wav"
    target.parent.mkdir()
    _file_link(hygiene.partial_path(target), outside)
    with hygiene.atomic_target(target) as partial:
        partial.write_bytes(b"audio")
    assert target.read_bytes() == b"audio"
    assert not outside.exists()


def test_a_cross_volume_publish_unlinks_a_planted_partial_before_copying(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    staged = tmp_path / "work" / "out.tmp.mp4"
    staged.parent.mkdir()
    staged.write_bytes(b"video")
    final = tmp_path / "out" / "out.mp4"
    final.parent.mkdir()
    _file_link(hygiene.partial_path(final), victim)
    calls = []

    def publish(partial, target):
        calls.append(Path(partial))
        if len(calls) == 1:
            raise OSError(errno.EXDEV, "cross-device link")
        os.replace(partial, target)

    with patch.object(processing, "publish", side_effect=publish):
        processing._publish_staged_output(staged, final)
    assert (final.read_bytes(), victim.read_text(encoding="utf-8")) == (b"video", "keep")


def test_the_cleanup_never_removes_through_a_link(tmp_path):
    home = _victim_home(tmp_path)
    planted = tmp_path / ".temp_work_tape"
    _dir_link(planted, home)
    with patch.object(processing, "is_valid_video", return_value=True), patch.object(processing, "remove_tree") as remove:
        processing._cleanup_work_dir(planted, tmp_path / "out.mp4")
    remove.assert_not_called()
    assert _all_kept(home)
