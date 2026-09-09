"""Tests for the optional DeepFilterNet3 stage.

The stage is off by default and the model is a from-source optional dependency, so what
these pin is everything that has to hold *without* it: the chain must fall back cleanly, and
the chunked processing must reconstruct audio exactly at the seams. The model's quality was
measured on real tape and is the reason the stage is off -- against UVR-DeNoise in the same
chain it triples programme deviation, 0.22 to 0.66 dB, on 50 real captures, where paired
synthetic fixtures had it winning by 3.52 dB of log-spectral distance. That is the fifth
constant on this branch a fixture set wrongly.
"""

from unittest.mock import patch

import numpy as np

from modules import deepfilter_denoise


def _chunks_of(audio, chunk, overlap):
    """Splits the way the stream does, so the join can be tested against identity."""
    pieces, start = [], 0
    while start < len(audio):
        stop = min(start + chunk, len(audio))
        pieces.append(audio[start:stop].copy())
        if stop >= len(audio):
            break
        start = stop - overlap
    return pieces


def test_chunk_join_reconstructs_identity_exactly():
    """Identity processing must come back sample for sample, or the seams are artefacts."""
    rng = np.random.default_rng(0)
    audio = rng.normal(0.0, 0.1, (44100 * 7 + 123, 2)).astype(np.float32)
    pieces = _chunks_of(audio, 44100 * 2, 44100 // 2)
    joined = pieces[0]
    for piece in pieces[1:]:
        joined = deepfilter_denoise._join_two(joined, piece, 44100 // 2)
    assert joined.shape == audio.shape
    assert np.allclose(joined, audio, atol=1e-6)


def test_chunk_join_without_an_overlap_concatenates():
    """No overlap means plain concatenation; a piece shorter than the overlap fades over its length."""
    head = np.ones((100, 1), dtype=np.float32)
    tail = np.zeros((3, 1), dtype=np.float32)
    assert np.array_equal(deepfilter_denoise._join_two(head, tail, 0), np.concatenate([head, tail]))
    joined = deepfilter_denoise._join_two(head, tail, 500)
    assert joined.shape == (100, 1) and joined[-1, 0] == 0.0


def test_stream_chunks_reads_overlapping_blocks(tmp_path):
    """Blocks overlap by the configured amount and between them cover the file exactly."""
    import soundfile as sf

    path, audio = _wav(tmp_path, seconds=1.1, rate=8000)
    with sf.SoundFile(str(path)) as source:
        blocks = list(deepfilter_denoise._stream_chunks(source, 3000, 500))
    assert [len(b) for b in blocks] == [3000, 3000, 3000, 1300]
    assert np.array_equal(blocks[1][:500], blocks[0][-500:])
    assert np.array_equal(blocks[-1], audio[-1300:])


def test_unavailable_model_reports_unavailable(tmp_path):
    """A host without the build must be told so, not crash."""
    with patch.dict(deepfilter_denoise._MODEL, {"model": None}, clear=True):
        assert deepfilter_denoise.available() is False
        assert deepfilter_denoise.denoise(tmp_path / "in.wav", tmp_path) is None


def test_a_failed_import_is_cached_as_unavailable():
    """The import is attempted once; every later call reads the cached answer."""
    with patch.dict(deepfilter_denoise._MODEL, {}, clear=True):
        with patch("builtins.__import__", side_effect=ImportError("no df")):
            assert deepfilter_denoise._load() is None
        assert deepfilter_denoise._MODEL["model"] is None
        assert deepfilter_denoise.available() is False


def test_the_torchaudio_shim_only_fills_a_gap():
    """The shim provides the removed module and never overrides a real one."""
    import sys

    with patch.dict(sys.modules, {}, clear=False):
        sys.modules.pop("torchaudio.backend.common", None)
        sys.modules.pop("torchaudio.backend", None)
        deepfilter_denoise._shim_torchaudio_backend()
        assert sys.modules["torchaudio.backend.common"].AudioMetaData is deepfilter_denoise._AudioMetaData
        assert sys.modules["torchaudio.backend"].common is sys.modules["torchaudio.backend.common"]
        marker = object()
        sys.modules["torchaudio.backend.common"] = marker
        deepfilter_denoise._shim_torchaudio_backend()
        assert sys.modules["torchaudio.backend.common"] is marker


def test_the_torchaudio_shim_keeps_an_existing_backend_module():
    """A `torchaudio.backend` the library already loaded keeps its state and gains `common`."""
    import sys
    import types

    with patch.dict(sys.modules, {}, clear=False):
        sys.modules.pop("torchaudio.backend.common", None)
        existing = types.ModuleType("torchaudio.backend")
        existing.list_audio_backends = lambda: ["soundfile"]
        sys.modules["torchaudio.backend"] = existing
        deepfilter_denoise._shim_torchaudio_backend()
        assert sys.modules["torchaudio.backend"] is existing
        assert existing.list_audio_backends() == ["soundfile"]
        assert existing.common is sys.modules["torchaudio.backend.common"]


def test_without_soundfile_the_module_loads_and_the_stage_is_off(monkeypatch):
    """The optional import is guarded: without soundfile the module imports and denoise() is None."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "soundfile", None)
    reloaded = importlib.reload(deepfilter_denoise)
    try:
        assert reloaded.sf is None
        with patch.dict(reloaded._MODEL, {"model": _fake_loaded(8000)}, clear=True):
            assert reloaded.denoise("in.wav", "out") is None
    finally:
        monkeypatch.delitem(sys.modules, "soundfile")
        importlib.reload(deepfilter_denoise)
    assert deepfilter_denoise.sf is not None


def test_the_stage_is_off_by_default():
    """Measured on real tape it loses to UVR-DeNoise, so it is opt-in."""
    from modules.config import APL_USE_DEEPFILTERNET

    assert APL_USE_DEEPFILTERNET is False


class _IdentityState:
    """Stands in for the model state; the sample rate matches the file so no resample runs."""

    def __init__(self, rate):
        self._rate = rate

    def sr(self):
        return self._rate


class _Tensorish:
    """The slice of a tensor the stage touches: .cpu().numpy(), and shape/len for the join."""

    def __init__(self, array):
        self.array = array

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def _fake_loaded(rate):
    """A loaded tuple whose 'model' returns its input, so the whole path runs without torch.

    CI's static job has no torch, and the stage's processing path must still be exercised
    there. The fakes provide exactly the calls the stage makes: from_numpy, resample and
    cpu().numpy(), with resample an identity because the rates match.
    """
    import types

    torch = types.SimpleNamespace(from_numpy=_Tensorish, no_grad=lambda: _NoGrad())
    torchaudio = types.SimpleNamespace(functional=types.SimpleNamespace(resample=lambda x, _a, _b: x))
    return (object(), _IdentityState(rate), lambda _model, _state, x, **_kw: x, torch, torchaudio)


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _wav(tmp_path, seconds, rate=8000):
    import soundfile as sf

    rng = np.random.default_rng(1)
    audio = rng.normal(0.0, 0.1, (int(rate * seconds), 2)).astype(np.float32)
    path = tmp_path / "in.wav"
    sf.write(str(path), audio, rate, subtype="FLOAT")
    return path, audio


def test_denoise_runs_the_model_over_chunks_and_writes_the_result(tmp_path):
    """A capture longer than one chunk is processed in overlapping pieces and reassembled.

    With an identity model the output must equal the input exactly, which is what proves
    the chunk seams introduce nothing of their own.
    """
    import soundfile as sf

    rate = 8000
    path, audio = _wav(tmp_path, seconds=5.0, rate=rate)
    with (
        patch.dict(deepfilter_denoise._MODEL, {"model": _fake_loaded(rate)}, clear=True),
        patch.object(deepfilter_denoise, "CHUNK_SECONDS", 2.0),
        patch.object(deepfilter_denoise, "OVERLAP_SECONDS", 0.25),
    ):
        produced = deepfilter_denoise.denoise(path, tmp_path / "out")
    assert produced is not None and produced.is_file()
    result, _rate = sf.read(str(produced), dtype="float32", always_2d=True)
    assert result.shape == audio.shape
    assert np.allclose(result, audio, atol=1e-6)


def test_denoise_or_uses_the_fallback_when_not_wanted_or_unavailable(tmp_path):
    """The chain never depends on the build: off, or absent, means the fallback runs."""
    calls = []

    def fallback():
        calls.append(1)
        return "fallback"

    assert deepfilter_denoise.denoise_or(tmp_path / "x.wav", tmp_path, False, fallback) == "fallback"
    with patch.dict(deepfilter_denoise._MODEL, {"model": None}, clear=True):
        assert deepfilter_denoise.denoise_or(tmp_path / "x.wav", tmp_path, True, fallback) == "fallback"
    assert len(calls) == 2


def test_denoise_or_falls_back_when_the_stage_fails_at_run_time(tmp_path):
    """A model that loads but fails on the capture is the same as no model: the chain goes on."""
    rate = 8000
    path, _audio = _wav(tmp_path, seconds=1.0, rate=rate)

    def broken(_model, _state, _x, **_kw):
        raise RuntimeError("CUDA error: out of memory")

    loaded = _fake_loaded(rate)
    loaded = (loaded[0], loaded[1], broken, loaded[3], loaded[4])
    with patch.dict(deepfilter_denoise._MODEL, {"model": loaded}, clear=True):
        assert deepfilter_denoise.denoise_or(path, tmp_path / "out", True, lambda: "fallback") == "fallback"


def test_denoise_or_prefers_the_model_when_it_produces_output(tmp_path):
    """When asked for and working, the stage's output is what the chain continues with."""
    rate = 8000
    path, _audio = _wav(tmp_path, seconds=1.0, rate=rate)
    with patch.dict(deepfilter_denoise._MODEL, {"model": _fake_loaded(rate)}, clear=True):
        produced = deepfilter_denoise.denoise_or(path, tmp_path / "out", True, lambda: "fallback")
    assert produced != "fallback" and produced.is_file()


def test_load_succeeds_with_a_fake_package(monkeypatch):
    """The success path of the loader is exercised without the real dependency."""
    import sys
    import types

    fake_df = types.ModuleType("df")
    fake_enhance = types.ModuleType("df.enhance")
    fake_enhance.init_df = lambda **_kw: (object(), _IdentityState(48000), None)
    fake_enhance.enhance = lambda *_a, **_k: None
    fake_df.enhance = fake_enhance
    # torch is absent on the static CI job, so the loader's own imports are faked too.
    for name in ("torch", "torchaudio"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "df", fake_df)
    monkeypatch.setitem(sys.modules, "df.enhance", fake_enhance)
    with patch.dict(deepfilter_denoise._MODEL, {}, clear=True):
        loaded = deepfilter_denoise._load()
        assert loaded is not None and loaded[1].sr() == 48000
        assert deepfilter_denoise.available() is True


def test_a_model_that_will_not_initialise_is_unavailable(monkeypatch):
    """Importable but broken -- a missing checkpoint, a CUDA fault -- is the same as absent."""
    import sys
    import types

    def boom(**_kw):
        raise RuntimeError("checkpoint missing")

    fake_enhance = types.ModuleType("df.enhance")
    fake_enhance.init_df = boom
    fake_enhance.enhance = lambda *_a, **_k: None
    fake_df = types.ModuleType("df")
    fake_df.enhance = fake_enhance
    for name in ("torch", "torchaudio"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "df", fake_df)
    monkeypatch.setitem(sys.modules, "df.enhance", fake_enhance)
    with patch.dict(deepfilter_denoise._MODEL, {}, clear=True):
        assert deepfilter_denoise._load() is None
        assert deepfilter_denoise.available() is False


def test_a_failed_run_leaves_no_partial_output_behind(tmp_path):
    """The stage streams into its output, so a failure part-way must take the partial file with it."""
    rate = 8000
    path, _audio = _wav(tmp_path, seconds=3.0, rate=rate)
    calls = []

    def breaks_on_the_second_block(_model, _state, x, **_kw):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("CUDA error")
        return x

    loaded = _fake_loaded(rate)
    loaded = (loaded[0], loaded[1], breaks_on_the_second_block, loaded[3], loaded[4])
    with (
        patch.dict(deepfilter_denoise._MODEL, {"model": loaded}, clear=True),
        patch.object(deepfilter_denoise, "CHUNK_SECONDS", 1.0),
        patch.object(deepfilter_denoise, "OVERLAP_SECONDS", 0.1),
    ):
        assert deepfilter_denoise.denoise_or(path, tmp_path / "out", True, lambda: "fallback") == "fallback"
    assert not deepfilter_denoise.target_path(path, tmp_path / "out").exists()
