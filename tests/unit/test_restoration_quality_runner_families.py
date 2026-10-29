"""The runner's family plumbing: the lazy model registry, the pair's resampling and MOS gain, unavailable families."""

import json
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from scripts.restoration_quality import mos_models, runner, speech_models, stem_metrics
from tests.unit.test_restoration_quality_sibilance import RATE, _voice_with_esses


def _fake_torch(calls):
    """A torch stand-in whose `cuda.empty_cache` records the call."""
    cuda = types.SimpleNamespace(empty_cache=lambda: calls.append("empty"))
    return types.SimpleNamespace(cuda=cuda)


def test_the_registry_loads_a_model_once_and_frees_it_between_families(monkeypatch):
    """`get` calls the loader on first use only; `release` drops every model and empties the CUDA cache."""
    calls = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls))
    registry = runner.ModelRegistry(device="cpu", models_dir="models")
    first = registry.get("m", lambda device, models_dir: (device, str(models_dir), calls.append("load")))
    assert registry.get("m", lambda *_args: "other") is first
    registry.release()
    assert calls == ["load", "empty"]


def test_releasing_without_torch_is_harmless(monkeypatch):
    """The dsp-only path never needs torch: a missing import is skipped."""
    monkeypatch.setitem(sys.modules, "torch", None)
    registry = runner.ModelRegistry()
    registry.release()
    assert registry.models_dir is None


def _write(tmp_path):
    """A 44.1 kHz source and a quieter 48 kHz output the pair resamples."""
    voice, _ess = _voice_with_esses(6.0)
    sf.write(str(tmp_path / "s.wav"), voice, RATE, subtype="FLOAT")
    output = np.interp(np.arange(int(len(voice) * 48000 / RATE)) * RATE / 48000.0, np.arange(len(voice)), voice)
    sf.write(str(tmp_path / "o.wav"), (0.5 * output).astype(np.float32), 48000, subtype="FLOAT")
    return tmp_path / "s.wav", tmp_path / "o.wav"


@pytest.fixture(name="pair")
def fixture_pair(tmp_path):
    """A real pair over the two files."""
    source, output = _write(tmp_path)
    return runner.Pair(source, output, tmp_path / "cache", 15.0, 7.5)


def test_the_pair_resamples_each_side_once(pair):
    """`at` resamples to a model's rate once, then serves the same array; the native rate is the side itself."""
    first = pair.at("output", 16000)
    assert pair.at("output", 16000) is first
    assert abs(len(first) - len(pair.output) * 16000 / RATE) <= 2
    assert np.array_equal(pair.at("source", RATE), pair.source)


def test_the_mos_gain_holds_every_peak_under_the_guard(pair):
    """One gain for both sides; it never pushes a peak past 0.99, and no profile is read outside the dsp family."""
    peak = max(float(np.abs(pair.source).max()), float(np.abs(pair.output).max()))
    assert pair.mos_gain() * peak <= runner.PEAK_GUARD + 1e-6
    assert pair.profile is None


def _registry(released):
    """A registry with no models that counts its releases."""
    return types.SimpleNamespace(language="ro", release=lambda: released.append(1))


@pytest.mark.parametrize(("name", "module"), [("stems", stem_metrics), ("speech", speech_models), ("mos", mos_models)])
def test_each_model_family_hands_the_pair_to_its_module(name, module, monkeypatch, tmp_path):
    """The model families are thin: each calls its module's `score(pair, card, registry)` and frees its models."""
    calls, released = [], []
    monkeypatch.setattr(module, "score", lambda pair, card, registry: calls.append(pair.rate))
    card, _pair = runner.score_pair(*_write(tmp_path), families=(name,), registry=_registry(released), cache_dir=tmp_path / "cache")
    assert calls == [RATE]
    assert (card.families, released) == ({name: "ok"}, [1])


def test_a_family_that_cannot_run_is_recorded_not_raised(monkeypatch, tmp_path):
    """A missing model or library costs only its family, with the reason."""

    def broken(_pair, _card, _registry):
        raise ValueError("no weights")

    monkeypatch.setattr(mos_models, "score", broken)
    card, _pair = runner.score_pair(*_write(tmp_path), families=("mos",), registry=_registry([]), cache_dir=tmp_path / "cache")
    assert card.families["mos"] == "unavailable: ValueError: no weights"


def test_the_result_document_writes_numpy_values_and_anything_else_as_text(tmp_path):
    """numpy numbers, booleans and arrays become JSON values; any other object its text."""
    path = tmp_path / "out" / "r.json"
    runner.write_result({"a": np.float32(0.5), "b": np.bool_(True), "c": np.arange(2), "d": types.SimpleNamespace(x=1)}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 0.5, "b": True, "c": [0, 1], "d": "namespace(x=1)"}
