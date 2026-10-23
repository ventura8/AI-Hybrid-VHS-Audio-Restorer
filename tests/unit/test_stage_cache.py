"""The stage cache: off it is the producer; on, a recurring key replays the pair, and nothing doubtful is stored.

The helpers here (the producer stand-in, the cache switched on in a test folder, the log
capture) are shared with `test_stage_cache_store.py`.
"""

import json
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from modules import stage_cache, utils

RATE = 44100
CALL = {"stage": "full_mix_neural", "flags": {"hum_cancel": True, "deepfilternet": False}, "strategy": {"profile": {}}}
INPUT_NAME = "preconditioned_x_audio.wav"
Captured = namedtuple("Captured", "messages warnings")


def write_wav(path, value, seconds=0.25):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.full((int(RATE * seconds), 2), value, dtype=np.float32), RATE, subtype="FLOAT")
    return path


class Producer:
    """The cached stages' stand-in: a chain output under hum_cancel/ and a neural output in a model folder.

    `mode` "input" hands the input on twice (nothing changed it), "surgical" skips the neural stage.
    """

    def __init__(self, audio_dir, input_wav, mode="both", log=None):
        self.audio_dir, self.input_wav, self.mode, self.log, self.calls = Path(audio_dir), Path(input_wav), mode, log, 0

    def __call__(self):
        self.calls += 1
        if self.log:
            utils.log_msg(self.log, level="WARNING", console=False)
        if self.mode == "input":
            return self.input_wav, self.input_wav
        surgical = write_wav(self.audio_dir / "hum_cancel" / f"humcancel_{self.input_wav.name}", 0.25)
        if self.mode == "surgical":
            return surgical, surgical
        return surgical, write_wav(self.audio_dir / "neural_denoised_abc" / f"{surgical.stem}_(No Noise)_m.wav", 0.125)


def small_document(input_wav, call):
    """A key of the input's samples and the call alone, so these tests stay hermetic."""
    return {"stage": call["stage"], "input": stage_cache._key.input_identity(input_wav), "call": call}


def hold_the_host_still(monkeypatch):
    """The host's real free space and the checkout's edits kept out of a test that stores.

    The volume reads as nearly empty (a CI runner has 14 GB, under the 20 GB floor), and the
    code-drift check, tested on its own, passes (another agent editing the checkout would
    otherwise refuse a store mid-test).
    """
    monkeypatch.setattr(stage_cache.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1 << 50))
    monkeypatch.setattr(stage_cache._key, "drift", lambda _document: None)


def cache_on(tmp_path, monkeypatch):
    """The cache on in a folder of its own, with the small key and the session log kept in the test folder."""
    root = tmp_path / "cache"
    monkeypatch.setattr(utils, "LOG_FILE", tmp_path / "session_log.txt")
    monkeypatch.setenv(stage_cache.ENV_VAR, str(root))
    monkeypatch.delenv("AI_RESTORE_EVENT_LOG", raising=False)
    monkeypatch.setattr(stage_cache._key, "key_document", small_document)
    hold_the_host_still(monkeypatch)
    stage_cache._WARNED.clear()
    return root


def capture_log(monkeypatch):
    """What the cache logs, and which of it is a warning."""
    captured = Captured([], [])

    def capture(message, level="INFO", **_kwargs):
        captured.messages.append(message)
        if level == "WARNING":
            captured.warnings.append(message)

    monkeypatch.setattr(stage_cache, "log_msg", capture)
    return captured


def warned(logged, text):
    return any(text in message for message in logged.warnings)


def said(logged, text):
    return any(text in message for message in logged.messages)


def render(base, call=CALL, value=0.5, mode="both", log=None):
    """One render in a fresh work folder of its own: the input copied in, the stage folder beside it."""
    work = base / f"run{len(list(base.glob('run*')))}"
    input_wav = write_wav(work / INPUT_NAME, value)
    audio_dir = work / "denoised_preconditioned_audio"
    audio_dir.mkdir(parents=True)
    producer = Producer(audio_dir, input_wav, mode=mode, log=log)
    return stage_cache.through(producer, input_wav, audio_dir, call), producer, audio_dir


def stored_entries(root):
    layout = root / stage_cache.LAYOUT
    return sorted(path for path in layout.iterdir() if path.is_dir()) if layout.exists() else []


def relative(pair, folder):
    return tuple(path.relative_to(folder) for path in pair)


def contents(pair):
    return tuple(path.read_bytes() for path in pair)


@pytest.fixture(name="cache")
def cache_fixture(tmp_path, monkeypatch):
    return cache_on(tmp_path, monkeypatch)


@pytest.fixture(name="logged")
def logged_fixture(monkeypatch):
    return capture_log(monkeypatch)


def test_off_the_producer_runs_once_and_its_own_result_comes_back(tmp_path, monkeypatch):
    monkeypatch.delenv(stage_cache.ENV_VAR, raising=False)
    result, calls = object(), []
    returned = stage_cache.through(lambda: calls.append(1) or result, tmp_path / "in.wav", tmp_path, CALL)
    assert (returned is result, calls, stage_cache.directory()) == (True, [1], None)


def test_a_relative_folder_leaves_the_cache_off_with_one_warning(monkeypatch, logged):
    monkeypatch.setenv(stage_cache.ENV_VAR, "relative/cache")
    stage_cache._WARNED.clear()
    assert (stage_cache.directory(), stage_cache.directory(), len(logged.warnings)) == (None, None, 1)


def test_a_miss_stores_the_pair_and_a_hit_replays_it_at_the_same_paths(tmp_path, cache):
    first_pair, first, first_dir = render(tmp_path)
    second_pair, second, second_dir = render(tmp_path)
    assert (first.calls, second.calls) == (1, 0)
    assert relative(second_pair, second_dir) == relative(first_pair, first_dir)
    assert contents(second_pair) == contents(first_pair)
    assert {path.name for path in stored_entries(cache)[0].iterdir()} == {"manifest.json", "key.json", "surgical.wav", "denoised.wav"}


def test_the_log_names_the_miss_the_store_and_the_hit(tmp_path, cache, logged):
    del cache
    render(tmp_path)
    render(tmp_path)
    kinds = [message.split("] ", 1)[1].split(" ", 1)[0] for message in logged.messages]
    assert (kinds, logged.warnings) == (["Miss", "Stored", "Hit"], [])


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("input", INPUT_NAME), ("surgical", f"denoised_preconditioned_audio/hum_cancel/humcancel_{INPUT_NAME}")],
)
def test_an_output_that_is_a_known_path_is_stored_as_an_alias(tmp_path, cache, mode, expected):
    render(tmp_path, mode=mode)
    (surgical, denoised), producer, audio_dir = render(tmp_path, mode=mode)
    outputs = json.loads((stored_entries(cache)[0] / stage_cache.MANIFEST).read_text(encoding="utf-8"))["outputs"]
    assert (producer.calls, outputs["denoised"], "from" in outputs["surgical"]) == (0, {"from": "surgical"}, mode == "input")
    assert (surgical.relative_to(audio_dir.parent), denoised) == (Path(expected), surgical)


def _corrupt_truncated(entry):
    (entry / "denoised.wav").write_bytes((entry / "denoised.wav").read_bytes()[:-8])


def _corrupt_garbage(entry):
    (entry / stage_cache.MANIFEST).write_text("{not json", encoding="utf-8")


def _edit_manifest(entry, change):
    manifest = json.loads((entry / stage_cache.MANIFEST).read_text(encoding="utf-8"))
    change(manifest)
    (entry / stage_cache.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


def _corrupt_key(entry):
    _edit_manifest(entry, lambda manifest: manifest.update(key="0" * 64))


def _corrupt_path(entry):
    _edit_manifest(entry, lambda manifest: manifest["outputs"]["denoised"].update(path="../escaped.wav"))


def _corrupt_missing_manifest(entry):
    (entry / stage_cache.MANIFEST).unlink()


def _corrupt_missing_file(entry):
    (entry / "denoised.wav").unlink()


@pytest.mark.parametrize(
    "corrupt", [_corrupt_truncated, _corrupt_garbage, _corrupt_key, _corrupt_path, _corrupt_missing_manifest, _corrupt_missing_file]
)
def test_an_entry_that_does_not_read_back_is_discarded_and_rendered_afresh(tmp_path, cache, logged, corrupt):
    render(tmp_path)
    corrupt(stored_entries(cache)[0])
    (_surgical, denoised), producer, audio_dir = render(tmp_path)
    escaped = (audio_dir.parent / "escaped.wav").exists()
    assert (producer.calls, denoised.is_relative_to(audio_dir), escaped, warned(logged, "unreadable")) == (1, True, False, True)
    (entry,) = stored_entries(cache)
    stage_cache.read_manifest(entry, entry.name)
    assert render(tmp_path)[1].calls == 0


def test_a_failed_replay_removes_what_it_restored_before_the_render(tmp_path, cache, monkeypatch):
    render(tmp_path)
    _corrupt_truncated(stored_entries(cache)[0])
    seen = []
    original_call = Producer.__call__

    def watching(self):
        chain_dir = self.audio_dir / "hum_cancel"
        seen.append(chain_dir.exists() and any(chain_dir.iterdir()))
        return original_call(self)

    monkeypatch.setattr(Producer, "__call__", watching)
    render(tmp_path)
    assert seen == [False]


def test_a_producer_that_raises_stores_nothing(tmp_path, cache):
    input_wav = write_wav(tmp_path / "in.wav", 0.5)

    def fail():
        raise RuntimeError("model crashed")

    with pytest.raises(RuntimeError, match="model crashed"):
        stage_cache.through(fail, input_wav, tmp_path, CALL)
    assert not stored_entries(cache)


def test_a_render_that_logged_a_warning_is_not_stored(tmp_path, cache, logged):
    _, producer, _ = render(tmp_path, log="    [Hum Cancel] Skipped after failure: boom")
    assert (producer.calls, stored_entries(cache), warned(logged, "Not stored")) == (1, [], True)


def test_a_render_into_a_folder_that_already_held_files_is_not_stored(tmp_path, cache, logged):
    """A resumed work folder: the stages may have reused what an earlier run left, which the key does not cover."""
    input_wav = write_wav(tmp_path / "work" / INPUT_NAME, 0.5)
    audio_dir = tmp_path / "work" / "stage"
    write_wav(audio_dir / "neural_denoised_old" / "left_over.wav", 0.1)
    producer = Producer(audio_dir, input_wav)
    stage_cache.through(producer, input_wav, audio_dir, CALL)
    assert (producer.calls, stored_entries(cache), warned(logged, "earlier run")) == (1, [], True)


class _Vanished:
    """A file another process removed between the listing and the size query."""

    @staticmethod
    def is_file():
        return True

    @staticmethod
    def stat():
        raise FileNotFoundError("gone")


def test_an_unreadable_stage_folder_counts_as_holding_files_and_a_vanished_file_as_empty(tmp_path, monkeypatch):
    def refuse(*_args):
        raise PermissionError("denied")

    monkeypatch.setattr(stage_cache.Path, "rglob", refuse)
    assert (stage_cache._holds_files(tmp_path), stage_cache._file_bytes(_Vanished())) == (True, 0)


def test_an_output_outside_the_stage_folder_is_not_stored(tmp_path, cache, logged):
    outside = write_wav(tmp_path / "elsewhere" / "out.wav", 0.5)
    input_wav = write_wav(tmp_path / "in.wav", 0.5)
    (tmp_path / "stage").mkdir()
    returned = stage_cache.through(lambda: (outside, outside), input_wav, tmp_path / "stage", CALL)
    assert (returned, stored_entries(cache), warned(logged, "outside the stage folder")) == ((outside, outside), [], True)


def test_deepfilternet_and_resemble_bypass_the_cache(tmp_path, cache, logged):
    _, producer, _ = render(tmp_path, call={**CALL, "flags": {"deepfilternet": True, "resemble_denoise": True}})
    assert (producer.calls, stored_entries(cache), said(logged, "Bypassed: deepfilternet and resemble_denoise")) == (1, [], True)


def test_the_event_log_bypasses_the_cache(tmp_path, cache, monkeypatch, logged):
    monkeypatch.setenv("AI_RESTORE_EVENT_LOG", str(tmp_path / "events"))
    _, producer, _ = render(tmp_path)
    assert (producer.calls, stored_entries(cache), said(logged, "event log")) == (1, [], True)


def test_a_key_that_cannot_be_computed_bypasses_the_cache_with_a_warning(tmp_path, cache, monkeypatch, logged):
    def broken(*_args):
        raise OSError("unreadable input")

    monkeypatch.setattr(stage_cache._key, "key_document", broken)
    _, producer, _ = render(tmp_path)
    assert (producer.calls, stored_entries(cache), warned(logged, "could not be computed")) == (1, [], True)
