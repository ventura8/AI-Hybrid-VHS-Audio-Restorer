"""The stage cache reads back only what it wrote: bare relative paths, known aliases, sizes, digests and totals."""

import json

import pytest

from modules import stage_cache


@pytest.mark.parametrize("relative", ["../x.wav", ".hidden/x.wav", "a\\b.wav", "C:x.wav", "a\0b.wav", "", "/abs.wav", "a//b.wav", 7])
def test_a_stored_path_must_be_bare_names(relative):
    with pytest.raises(ValueError):
        stage_cache.safe_relative(relative)


def test_a_bare_stored_path_splits_into_its_parts():
    assert stage_cache.safe_relative("neural_denoised_abc/x_(No Noise)_m.wav") == ["neural_denoised_abc", "x_(No Noise)_m.wav"]


GOOD_RECORD = {"path": "a/b.wav", "file": "denoised.wav", "bytes": 4, "sha256": "0" * 64}
ALIASED = {"from": "input"}


def _write(folder, document):
    (folder / stage_cache.MANIFEST).write_text(json.dumps(document), encoding="utf-8")


def _manifest(outputs):
    return {"schema": stage_cache.MANIFEST_SCHEMA, "key": "k", "outputs": outputs, "bytes": 0, "created": 0, "producer_seconds": 0}


@pytest.mark.parametrize(
    "outputs",
    [
        "not a dict",
        {"surgical": ALIASED},
        {"surgical": {"from": "surgical"}, "denoised": ALIASED},
        {"surgical": ALIASED, "denoised": [1]},
        {"surgical": ALIASED, "denoised": {**GOOD_RECORD, "file": "other.wav"}},
        {"surgical": ALIASED, "denoised": {**GOOD_RECORD, "sha256": "xyz"}},
        {"surgical": ALIASED, "denoised": {**GOOD_RECORD, "bytes": "4"}},
        {"surgical": ALIASED, "denoised": {**GOOD_RECORD, "path": "../b.wav"}},
    ],
)
def test_a_malformed_manifest_is_refused(tmp_path, outputs):
    _write(tmp_path, _manifest(outputs))
    with pytest.raises(ValueError):
        stage_cache.read_manifest(tmp_path, "k")


@pytest.mark.parametrize(
    "document",
    [[1, 2], {**_manifest({"surgical": ALIASED, "denoised": ALIASED}), "created": None}, {**_manifest({}), "key": "other"}],
)
def test_a_manifest_without_its_shape_its_key_or_its_totals_is_refused(tmp_path, document):
    _write(tmp_path, document)
    with pytest.raises(ValueError):
        stage_cache.read_manifest(tmp_path, "k")


def test_a_well_formed_manifest_reads_back(tmp_path):
    document = _manifest({"surgical": ALIASED, "denoised": GOOD_RECORD})
    _write(tmp_path, document)
    assert stage_cache.read_manifest(tmp_path, "k") == document
