"""The stage cache's guards: a cache folder that cannot be read, a replay that fails on the stage folder's side,
audio-separator's warnings, and code that moved under the render."""

import errno
import logging
from pathlib import Path

import pytest

from modules import stage_cache
from tests.unit.test_stage_cache import Producer, cache_on, capture_log, render, said, stored_entries, warned

SEPARATOR = logging.getLogger("audio_separator.separator.separator")


@pytest.fixture(name="cache")
def cache_fixture(tmp_path, monkeypatch):
    return cache_on(tmp_path, monkeypatch)


@pytest.fixture(name="logged")
def logged_fixture(monkeypatch):
    return capture_log(monkeypatch)


def test_a_cache_folder_that_cannot_be_read_is_a_miss_and_never_ends_the_render(tmp_path, cache, monkeypatch, logged):
    """Python 3.12's `Path.exists` and `is_dir` raise on EACCES: a folder another account owns must not end the restoration."""
    real_stat = Path.stat

    def denied(self, *args, **kwargs):
        if cache in self.parents:
            raise PermissionError(errno.EACCES, "Access is denied", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    (_surgical, denoised), producer, audio_dir = render(tmp_path)
    assert (producer.calls, denoised.is_relative_to(audio_dir)) == (1, True)
    assert (warned(logged, "cannot be read"), warned(logged, "Not stored")) == (True, True)


def test_a_replay_that_fails_on_the_stage_folders_side_keeps_the_entry(tmp_path, cache, monkeypatch, logged):
    """A full disk under the work folder is not the entry's fault: it renders afresh and the next candidate still hits."""
    render(tmp_path)
    (entry,) = stored_entries(cache)
    real_copy = stage_cache.copy_hashing

    def full_disk(source, target):
        if Path(source).parent == entry:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_copy(source, target)

    monkeypatch.setattr(stage_cache, "copy_hashing", full_disk)
    (_surgical, denoised), producer, audio_dir = render(tmp_path)
    assert (producer.calls, denoised.is_relative_to(audio_dir), stored_entries(cache)) == (1, True, [entry])
    assert (warned(logged, "the entry is kept"), said(logged, "stored already"), warned(logged, "Not stored")) == (True, True, False)
    monkeypatch.setattr(stage_cache, "copy_hashing", real_copy)
    assert render(tmp_path)[1].calls == 0


def _separating(message):
    """A producer that logs one audio-separator line while it renders, as the neural stage would."""
    real_call = Producer.__call__

    def call(self):
        SEPARATOR.warning(message)
        return real_call(self)

    return call


@pytest.mark.parametrize(
    ("message", "stored"),
    [
        ("Could not read audio file info, defaulting to 16-bit output: locked", False),
        ("Fell back to legacy Roformer implementation successfully", False),
        ("Using soundfile for writing.", True),
        ("Audio duration (8.00s) is less than 10 seconds.", True),
    ],
)
def test_an_audio_separator_warning_refuses_the_store_unless_it_is_routine(tmp_path, cache, monkeypatch, message, stored):
    monkeypatch.setattr(Producer, "__call__", _separating(message))
    render(tmp_path)
    assert bool(stored_entries(cache)) is stored


def _record(name, level, message):
    return logging.LogRecord(name, level, __file__, 1, message, None, None)


@pytest.mark.parametrize(
    ("record", "counted"),
    [
        (_record("audio_separator", logging.ERROR, "Failed to load model"), True),
        (_record("audio_separator.separator.roformer.roformer_loader", logging.WARNING, "Fell back"), True),
        (_record("audio_separator.separator.separator", logging.INFO, "Loading model"), False),
        (_record("audio_separator_other", logging.WARNING, "x"), False),
        (_record("torch", logging.WARNING, "x"), False),
    ],
)
def test_only_the_separators_own_warnings_and_errors_count(record, counted):
    assert stage_cache._separator_problem(record) is counted


def test_the_separator_watch_is_installed_once_and_makes_records_as_before():
    stage_cache.watch_separator_log()
    installed = logging.getLogRecordFactory()
    stage_cache.watch_separator_log()
    record = logging.getLogRecordFactory()("audio_separator", logging.WARNING, __file__, 1, "Model failed", None, None)
    assert (logging.getLogRecordFactory() is installed, record.getMessage()) == (True, "Model failed")


def test_code_that_moved_under_the_render_refuses_the_store(tmp_path, cache, monkeypatch, logged):
    monkeypatch.setattr(stage_cache._key, "drift", lambda _document: "the module sources changed while it rendered")
    _, producer, _ = render(tmp_path)
    assert (producer.calls, stored_entries(cache), warned(logged, "sources changed")) == (1, [], True)


def test_a_date_out_of_range_reads_as_unknown():
    assert stage_cache._date(1e20) == "at an unknown time"


@pytest.mark.parametrize(("name", "expected"), [("a" * 64, True), ("A" * 64, False), ("a" * 63, False), (None, False)])
def test_a_key_is_64_lowercase_hex_digits(name, expected):
    assert stage_cache.is_key(name) is expected
