"""Inert candidates found by their audio: the decoded-audio hash, the fingerprint cache and the round that skips them."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from scripts import autotune_restoration as at


def _tone(frames=48000, rate=44100):
    wave = 0.1 * np.sin(2.0 * np.pi * 440.0 * np.arange(frames) / rate)
    return np.stack([wave, 0.5 * wave], axis=1).astype(np.float32)


def _write(path, samples, rate=44100, subtype="FLOAT", title=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(path), "w", rate, samples.shape[1], subtype=subtype) as handle:
        if title:
            handle.title = title
        handle.write(samples)
    return path


def test_the_loop_hash_reads_the_decoded_samples_not_the_file_bytes(tmp_path):
    """A tag in the header changes the file, not the audio."""
    plain = _write(tmp_path / "a.wav", _tone())
    tagged = _write(tmp_path / "b.wav", _tone(), title="tagged")
    assert plain.read_bytes() != tagged.read_bytes()
    assert at.exact_audio_sha256(plain) == at.exact_audio_sha256(tagged)


def test_the_loop_hash_tells_one_changed_sample_another_rate_and_another_format_apart(tmp_path):
    """One float sample nudged by 1e-6, the same samples at 48 kHz, the same samples as 24-bit PCM."""
    changed = _tone()
    changed[1000, 0] += 1e-6
    base = at.exact_audio_sha256(_write(tmp_path / "a.wav", _tone()))
    assert base != at.exact_audio_sha256(_write(tmp_path / "b.wav", changed))
    assert base != at.exact_audio_sha256(_write(tmp_path / "c.wav", _tone(), rate=48000))
    assert base != at.exact_audio_sha256(_write(tmp_path / "d.wav", _tone(), subtype="PCM_24"))


def test_the_loop_hash_does_not_depend_on_the_block_size(tmp_path, monkeypatch):
    """The file is hashed in blocks; 48 blocks of 1000 frames give the digest one block gives."""
    wav = _write(tmp_path / "a.wav", _tone())
    whole = at.exact_audio_sha256(wav)
    monkeypatch.setattr(at, "HASH_BLOCK_FRAMES", 1000)
    assert at.exact_audio_sha256(wav) == whole


def test_the_fingerprint_the_loop_takes_tells_a_dc_offset_and_one_nudged_sample_apart(tmp_path):
    """Design 5.2 asks for a byte-identical match: an offset of 0.01 on every sample, or 1e-6 on one, is another render."""
    offset, nudged = _tone() + np.float32(0.01), _tone()
    nudged[1000, 0] += 1e-6
    for cid, samples in (("same", _tone()), ("offset", offset), ("nudged", nudged)):
        _write(tmp_path / "cands" / cid / "t1.wav", samples)
    prints = {cid: at.audio_fingerprint(tmp_path, cid, {"t1": "t1.mov"}) for cid in ("same", "offset", "nudged")}
    assert len(set(prints.values())) == 3


def test_find_inert_names_the_incumbent_or_the_first_candidate_that_sounds_the_same():
    """Equal on every tape is inert; equal on one tape out of two is not."""
    prints = {"c0": ("a", "b"), "inc": ("a", "b"), "c2": ("c", "d"), "c3": ("c", "d"), "c4": ("a", "x")}
    assert at.find_inert(prints, "inc") == {"c0": "inc", "c3": "c2"}


def _render(out_dir, cid, slugs):
    for slug in slugs:
        _write(out_dir / "cands" / cid / f"{slug}.wav", _tone(4000))


def test_the_fingerprint_reuses_its_hash_while_the_file_is_unchanged(tmp_path, monkeypatch):
    """A second fingerprint of the same files reads the sidecars instead of decoding the audio again."""
    hashed = []
    monkeypatch.setattr(at, "exact_audio_sha256", lambda path: hashed.append(path) or "h")
    _render(tmp_path, "c1", ("t1", "t2"))
    tapes = {"t1": "t1.mov", "t2": "t2.mov"}
    assert at.audio_fingerprint(tmp_path, "c1", tapes) == at.audio_fingerprint(tmp_path, "c1", tapes) == ("h", "h")
    assert len(hashed) == 2


def _key(wav):
    return f"{at.audio_io.file_key(wav)}|{at.HASH_NAME}"


def _sidecar_text(kind, key):
    """The sidecar's contents: another hash function's entry, an entry without a digest, or `kind` itself."""
    built = {"other_hash": {"key": key.replace(at.HASH_NAME, "auditory"), "sha256": "h"}, "no_digest": {"key": key}}
    return json.dumps(built[kind]) if kind in built else kind


UNTRUSTED_SIDECARS = ["{", "[]", "null", "other_hash", "no_digest"]


@pytest.mark.parametrize("sidecar", UNTRUSTED_SIDECARS)
def test_a_sidecar_from_another_hash_or_not_holding_a_hash_is_rehashed(tmp_path, sidecar):
    """Unparsable JSON, a list, null, the harness's DC-blind hash and a key without a digest are never trusted."""
    _render(tmp_path, "c1", ("t1",))
    wav = tmp_path / "cands" / "c1" / "t1.wav"
    wav.with_name("t1" + at.HASH_SIDECAR_SUFFIX).write_text(_sidecar_text(sidecar, _key(wav)), encoding="utf-8")
    assert at.audio_fingerprint(tmp_path, "c1", {"t1": "t1.mov"}) == (at.exact_audio_sha256(wav),)


def _equal_scores(candidates, *_args):
    return {cid: {"t1": {"m": 1.0}} for cid in candidates}


def test_a_round_marks_candidates_that_sound_alike_inert_logs_them_and_never_scores_them(tmp_path, monkeypatch):
    """c1 renders the incumbent's audio: it is set aside before scoring and reported as a knob-table finding."""
    state = {"incumbent": {"a": 1}, "rounds": [], "candidates": {}}
    inc = at.candidate_id(state["incumbent"])
    prints = {inc: ("x",), "c1": ("x",), "c2": ("y",)}
    scorer = Mock(side_effect=_equal_scores)
    monkeypatch.setattr(at, "propose", lambda *_args: {"c1": {"a": 2}, "c2": {"a": 3}})
    monkeypatch.setattr(at, "run_candidate", lambda *_args: None)
    monkeypatch.setattr(at, "audio_fingerprint", lambda _out, cid, _tapes: prints[cid])
    monkeypatch.setattr(at, "score_round", scorer)
    args = SimpleNamespace(families=("dsp",), gates=None, language="ro", parallel=1)
    assert not at.run_round(args, "apl", {"t1": "t1.mov"}, tmp_path, state, {"m": "up"}, 1)
    assert set(scorer.call_args.args[0]) == {inc, "c2"}
    assert state["rounds"][0]["verdicts"]["c1"]["inert"] == inc
    assert "knob-table finding: `c1`" in (tmp_path / at.LOG_MD).read_text(encoding="utf-8")
