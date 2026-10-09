"""The stage cache's key: the input by its samples, the call, and every setting but the post-neural allowlist move it.

The code, binaries, models, assets and runtime parts are in `test_stage_cache_key_outside.py`.
"""

import numpy as np
import pytest
import soundfile as sf

from modules import config
from modules import stage_cache_key as key_mod

RATE = 44100


def samples(seconds=0.5):
    t = np.arange(int(RATE * seconds)) / RATE
    tone = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    return np.stack([tone, 0.5 * tone], axis=1).astype(np.float32)


def write(path, data, subtype="FLOAT"):
    sf.write(str(path), data, RATE, subtype=subtype)
    return path


def _with_peak_timestamp(path, stamp):
    """The same file with another PEAK chunk timestamp: what two soundfile writes seconds apart differ by."""
    data = bytearray(path.read_bytes())
    start = data.index(b"PEAK") + 12
    stop = start + 4
    data[start:stop] = stamp.to_bytes(4, "little")
    path.write_bytes(bytes(data))
    return path


def _pcm(path, **kwargs):
    return key_mod.input_identity(path, **kwargs)["pcm_sha256"]


def test_canonical_json_spells_numpy_values_sets_and_key_order_one_way():
    numpy_doc = {"b": np.float32(0.5), "a": np.int64(3), "c": np.array([1, 2]), "d": {3, 1, 2}, "e": np.bool_(True), "f": float("nan")}
    plain_doc = {"a": 3, "b": 0.5, "c": [1, 2], "d": [1, 2, 3], "e": True, "f": float("nan")}
    assert key_mod.canonical_json(numpy_doc) == key_mod.canonical_json(plain_doc)
    assert (key_mod.key_of(numpy_doc), key_mod.jsonable(object)) == (key_mod.key_of(plain_doc), repr(object))


def test_the_input_is_read_by_its_samples_not_its_header(tmp_path):
    data = samples()
    first = _with_peak_timestamp(write(tmp_path / "a.wav", data), 1)
    second = _with_peak_timestamp(write(tmp_path / "b.wav", data), 2)
    assert first.read_bytes() != second.read_bytes()
    assert (_pcm(first), _pcm(first, block_frames=1000)) == (_pcm(second), _pcm(second))
    identity = key_mod.input_identity(first)
    assert (identity["name"], identity["frames"], identity["subtype"]) == ("a.wav", len(data), "FLOAT")


def test_one_sample_a_dc_offset_and_the_sample_type_each_change_the_input(tmp_path):
    data = samples()
    nudged = data.copy()
    nudged[100, 0] += np.float32(1e-6)
    changed = [
        _pcm(write(tmp_path / "b.wav", nudged)),
        _pcm(write(tmp_path / "c.wav", data + np.float32(1e-3))),
        _pcm(write(tmp_path / "d.wav", data, subtype="PCM_24")),
    ]
    assert _pcm(write(tmp_path / "a.wav", data)) not in changed


def test_without_soundfile_the_input_cannot_be_keyed(tmp_path, monkeypatch):
    monkeypatch.setattr(key_mod, "sf", None)
    with pytest.raises(RuntimeError):
        key_mod.input_identity(tmp_path / "a.wav")


PERTURB = {bool: lambda value: not value, int: lambda value: value + 1, float: lambda value: value + 1.5}
PERTURB.update({str: lambda value: value + "x", list: lambda value: [*value, "x"]})


def _perturbed(value):
    """A different value of the same kind; anything else becomes a string."""
    return PERTURB.get(type(value), lambda _value: "x")(value)


def _config_key(resolved):
    return key_mod.key_of(key_mod.config_identity(resolved))


def test_every_setting_outside_the_allowlist_moves_the_key_and_every_one_on_it_does_not():
    base = dict(config.CONFIG)
    reference = _config_key(base)
    moved = {name for name, value in base.items() if _config_key({**base, name: _perturbed(value)}) != reference}
    assert moved == set(base) - key_mod.POST_NEURAL_CONFIG_KEYS


def test_a_setting_nobody_knows_moves_the_key():
    assert _config_key({**config.CONFIG, "zz_new": 1}) != _config_key(dict(config.CONFIG))


def test_the_configuration_is_read_when_the_key_is_computed(monkeypatch):
    before = key_mod.config_identity()
    monkeypatch.setitem(config.CONFIG, "apl_tonal_flatness_max", 0.5)
    assert key_mod.config_identity() != before


@pytest.fixture(name="steady")
def steady_fixture(monkeypatch):
    """The parts that reach outside the process held still: no ffmpeg run, no torch query."""
    monkeypatch.setattr(key_mod, "binaries", lambda: {"ffmpeg": {"absent": True}, "cathar": {"absent": True}})
    monkeypatch.setattr(key_mod, "runtime", lambda: {"python": "3"})


def _call(**changes):
    call = {"stage": key_mod.STAGE_FULL_MIX_NEURAL, "strategy": {"profile": {"noise_floor_db": -50.0}}, "flags": {"hum_cancel": True}}
    call.update(changes)
    return call


CALL_CHANGES = [
    {"flags": {"hum_cancel": False}},
    {"strategy": {"profile": {"noise_floor_db": -49.0}}},
    {"denoise_model": "UVR-DeNoise.pth"},
    {"total_duration": 61.0},
]


@pytest.mark.parametrize("changes", CALL_CHANGES)
def test_the_call_its_flags_and_its_strategy_move_the_key(tmp_path, steady, changes):
    del steady
    wav = write(tmp_path / "in.wav", samples())
    base = key_mod.key_of(key_mod.key_document(wav, _call()))
    again = key_mod.key_of(key_mod.key_document(wav, _call()))
    assert (again, key_mod.key_of(key_mod.key_document(wav, _call(**changes))) != base) == (base, True)


def test_the_document_holds_every_part(tmp_path, steady):
    del steady
    document = key_mod.key_document(write(tmp_path / "in.wav", samples()), _call())
    parts = {"schema", "stage", "input", "call", "config", "code", "binaries", "models", "assets", "runtime"}
    assert (set(document), document["schema"], document["stage"]) == (parts, key_mod.KEY_SCHEMA, key_mod.STAGE_FULL_MIX_NEURAL)
