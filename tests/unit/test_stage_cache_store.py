"""The stage cache's store: the cap, least-recently-used eviction, the free-space floor, races and the folders in progress."""

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules import stage_cache
from tests.unit.test_stage_cache import cache_on, capture_log, render, said, stored_entries, warned


@pytest.fixture(name="cache")
def cache_fixture(tmp_path, monkeypatch):
    return cache_on(tmp_path, monkeypatch)


@pytest.fixture(name="logged")
def logged_fixture(monkeypatch):
    return capture_log(monkeypatch)


def _stored_by(cache, value, tmp_path):
    """Renders `value` and returns the entry it added."""
    before = set(stored_entries(cache))
    render(tmp_path, value=value)
    (added,) = set(stored_entries(cache)) - before
    return added


def _set_last_used(entry, when):
    os.utime(entry / stage_cache.MANIFEST, (when, when))


def _cap_for(cache, entries_that_fit):
    """A cap, as the environment spells it, that holds this many of the stored entries."""
    size = max(size for _used, size, _folder in stage_cache.entries(cache / stage_cache.LAYOUT))
    return f"{entries_that_fit * size / stage_cache.GB:.12f}"


def test_an_entry_over_the_cap_is_not_stored(tmp_path, cache, monkeypatch, logged):
    monkeypatch.setenv(stage_cache.MAX_GB_ENV, "0.000001")
    render(tmp_path)
    assert (stored_entries(cache), warned(logged, "over the")) == ([], True)


def test_the_least_recently_used_entry_is_evicted_and_a_hit_counts_as_a_use(tmp_path, cache, monkeypatch, logged):
    first, second, third = (_stored_by(cache, value, tmp_path) for value in (0.1, 0.2, 0.3))
    for when, entry in enumerate((first, second, third), start=1):
        _set_last_used(entry, 1000.0 * when)
    replayed = render(tmp_path, value=0.1)[1].calls == 0
    monkeypatch.setenv(stage_cache.MAX_GB_ENV, _cap_for(cache, 3.5))
    fourth = _stored_by(cache, 0.4, tmp_path)
    assert (replayed, set(stored_entries(cache)), said(logged, "Evicted")) == (True, {first, third, fourth}, True)


def test_nothing_is_stored_or_evicted_when_the_volume_would_drop_under_the_floor(tmp_path, cache, monkeypatch, logged):
    kept = _stored_by(cache, 0.1, tmp_path)
    monkeypatch.setenv(stage_cache.MAX_GB_ENV, _cap_for(cache, 1.5))
    monkeypatch.setattr(stage_cache.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
    _, producer, _ = render(tmp_path, value=0.2)
    assert (producer.calls, stored_entries(cache), warned(logged, "GB free")) == (1, [kept], True)


def test_a_key_another_process_stored_first_drops_this_copy_quietly(tmp_path, cache, monkeypatch, logged):
    real_fill = stage_cache._fill_entry

    def racing(job, folder, *args):
        manifest = real_fill(job, folder, *args)
        (folder.parent / job.key).mkdir()
        (folder.parent / job.key / "other").write_text("x", encoding="utf-8")
        return manifest

    monkeypatch.setattr(stage_cache, "_fill_entry", racing)
    (surgical, _denoised), producer, audio_dir = render(tmp_path)
    in_progress = list((cache / stage_cache.LAYOUT).glob(f"{stage_cache.TMP_PREFIX}*"))
    assert (producer.calls, surgical.is_relative_to(audio_dir), in_progress) == (1, True, [])
    assert (said(logged, "another process"), logged.warnings) == (True, [])


def test_a_rename_that_fails_for_another_reason_is_not_stored(tmp_path, cache, monkeypatch, logged):
    def refuse(*_args):
        raise PermissionError("denied")

    monkeypatch.setattr(stage_cache.os, "rename", refuse)
    render(tmp_path)
    assert (stored_entries(cache), warned(logged, "PermissionError")) == ([], True)


def _age(folder):
    old = time.time() - stage_cache.STALE_TMP_S - 60
    os.utime(folder, (old, old))


def test_an_entry_in_progress_is_swept_only_once_it_is_stale(tmp_path, cache):
    layout = cache / stage_cache.LAYOUT
    stale, fresh = layout / "tmp-0123456789ab-1", layout / "tmp-ba9876543210-2"
    for folder in (stale, fresh):
        folder.mkdir(parents=True)
    _age(stale)
    render(tmp_path)
    assert (stale.exists(), fresh.exists()) == (False, True)


def test_folders_the_cache_did_not_make_are_never_counted_evicted_or_swept(tmp_path, cache, monkeypatch, logged):
    """A cache pointed at a folder that already had a `v1` of its own: only key-named and tmp-<key>-<pid> folders are the cache's."""
    layout = cache / stage_cache.LAYOUT
    foreign = [layout / "mine", layout / "tmp-mine", layout / ("0" * 63)]
    for folder in foreign:
        folder.mkdir(parents=True)
        (folder / "keep.txt").write_text("kept", encoding="utf-8")
        _age(folder)
    first = _stored_by(cache, 0.1, tmp_path)
    monkeypatch.setenv(stage_cache.MAX_GB_ENV, _cap_for(cache, 1.5))
    second = _stored_by(cache, 0.2, tmp_path)
    assert ([folder.exists() for folder in foreign], first.exists(), second.exists()) == ([True, True, True], False, True)
    assert said(logged, "not a cache entry")


def test_an_entry_that_could_not_be_evicted_refuses_the_store(tmp_path, cache, monkeypatch, logged):
    """The plan is checked against the disk once eviction ran: an entry Windows kept open still counts."""
    kept = _stored_by(cache, 0.1, tmp_path)
    monkeypatch.setenv(stage_cache.MAX_GB_ENV, _cap_for(cache, 1.5))
    monkeypatch.setattr(stage_cache, "remove_tree", lambda _path: False)
    render(tmp_path, value=0.2)
    assert (stored_entries(cache), warned(logged, "Could not evict"), warned(logged, "still holds")) == ([kept], True, True)


def test_a_volume_another_writer_filled_meanwhile_refuses_the_store(tmp_path, cache, monkeypatch, logged):
    readings = iter([1 << 50, 0])
    monkeypatch.setattr(stage_cache, "_free_bytes", lambda _layout: next(readings))
    render(tmp_path)
    assert (stored_entries(cache), warned(logged, "GB free")) == ([], True)


def test_entries_other_processes_are_building_count_toward_the_cap(tmp_path, cache, monkeypatch):
    first = _stored_by(cache, 0.1, tmp_path)
    building = cache / stage_cache.LAYOUT / "tmp-0123456789ab-99"
    building.mkdir()
    (building / "denoised.wav").write_bytes(b"x" * stage_cache._tree_bytes(first))
    monkeypatch.setenv(stage_cache.MAX_GB_ENV, _cap_for(cache, 2.5))
    second = _stored_by(cache, 0.2, tmp_path)
    assert (first.exists(), second.exists(), building.exists()) == (False, True, True)


def test_a_folder_that_vanishes_is_neither_stale_nor_used():
    missing = Path("no") / "such" / "folder"
    assert (stage_cache._stale(missing, time.time()), stage_cache._last_used(missing)) == (False, 0.0)


def test_a_hit_that_cannot_be_marked_used_still_replays(tmp_path, cache, monkeypatch, logged):
    del cache
    render(tmp_path)

    def refuse(*_args):
        raise PermissionError("read-only")

    monkeypatch.setattr(stage_cache.os, "utime", refuse)
    assert (render(tmp_path)[1].calls, said(logged, "Could not mark")) == (0, True)


def test_an_entry_that_cannot_be_evicted_stays_and_says_so(tmp_path, monkeypatch, logged):
    folder = tmp_path / "entry"
    folder.mkdir()
    monkeypatch.setattr(stage_cache, "remove_tree", lambda _path: False)
    stage_cache._evict([(0.0, 10, folder)])
    assert (folder.exists(), warned(logged, "Could not evict")) == (True, True)


def test_a_half_restored_file_that_cannot_be_removed_is_logged(logged):
    class Stuck:
        name = "stuck.wav"

        def unlink(self, missing_ok=False):
            raise PermissionError(f"open ({missing_ok})")

    stage_cache._unlink_all([Stuck()])
    assert said(logged, "half-restored")


def _evicted(stored, incoming, cap):
    return [folder for _used, _size, folder in stage_cache.eviction_plan(stored, incoming, cap)]


def test_the_eviction_plan_takes_the_oldest_until_the_new_entry_fits():
    stored = [(1.0, 10, "a"), (2.0, 10, "b"), (3.0, 10, "c")]
    assert (_evicted(stored, 5, 35), _evicted(stored, 15, 35), _evicted(stored, 30, 35)) == ([], ["a"], ["a", "b", "c"])


@pytest.mark.parametrize(
    ("raw", "gb"),
    [(None, 50.0), ("2.5", 2.5), ("abc", 50.0), ("-1", 50.0), ("0", 50.0), ("nan", 50.0), ("inf", 50.0)],
)
def test_the_cap_is_a_positive_number_of_gigabytes_or_the_default(monkeypatch, raw, gb):
    stage_cache._WARNED.clear()
    monkeypatch.setenv(stage_cache.MAX_GB_ENV, raw or "")
    assert stage_cache.max_bytes() == int(gb * stage_cache.GB)
