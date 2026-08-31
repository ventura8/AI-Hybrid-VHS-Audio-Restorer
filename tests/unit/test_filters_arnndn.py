import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

import modules.filters


def test_resolve_arnndn_model_path_direct_and_searched(tmp_path, monkeypatch):
    """Verify model path resolution checks direct path and candidate search folders."""
    monkeypatch.chdir(tmp_path)
    direct_model = tmp_path / "direct_test.rnnn"
    direct_model.write_bytes(b"model")
    assert modules.filters._resolve_arnndn_model_path(str(direct_model)) == direct_model.resolve()

    model_dir = tmp_path / "models" / "arnndn"
    model_dir.mkdir(parents=True)
    searched_model = model_dir / "custom.rnnn"
    searched_model.write_bytes(b"model")
    assert modules.filters._resolve_arnndn_model_path("custom.rnnn") == searched_model.resolve()

    nonexistent_custom_path = tmp_path / "subdir" / "missing.rnnn"
    assert modules.filters._resolve_arnndn_model_path(str(nonexistent_custom_path)) == nonexistent_custom_path.resolve()


def test_fetch_remote_model_bytes_success(monkeypatch):
    """Verify _fetch_remote_model_bytes requests model over HTTPS and returns bytes."""
    dummy_bytes = b"fake_rnnn_model_data_bytes_long_enough_for_validation" * 5

    class DummyResponse:
        status = 200
        reason = "OK"

        def read(self):
            return dummy_bytes

    class DummyConn:
        def __init__(self, host, timeout):
            pass

        def request(self, method, url, headers):
            pass

        def getresponse(self):
            return DummyResponse()

        def close(self):
            pass

    monkeypatch.setitem(modules.filters.ARNNDN_MODEL_SHA256, "cb.rnnn", hashlib.sha256(dummy_bytes).hexdigest())
    with patch("http.client.HTTPSConnection", side_effect=DummyConn):
        data = modules.filters._fetch_remote_model_bytes("cb.rnnn")
        assert data == dummy_bytes


def test_fetch_remote_model_bytes_http_error():
    """Verify _fetch_remote_model_bytes raises RuntimeError on non-200 HTTP response."""

    class ErrorResponse:
        status = 404
        reason = "Not Found"

    class ErrorConn:
        def __init__(self, host, timeout):
            pass

        def request(self, method, url, headers):
            pass

        def getresponse(self):
            return ErrorResponse()

        def close(self):
            pass

    with patch("http.client.HTTPSConnection", side_effect=ErrorConn):
        with pytest.raises(RuntimeError, match="HTTP response error 404"):
            modules.filters._fetch_remote_model_bytes("cb.rnnn")


def test_fetch_remote_model_bytes_rejects_unpinned_model():
    """Verify a model without a pinned digest is rejected before any download."""
    with pytest.raises(ValueError, match="no pinned SHA-256 digest"):
        modules.filters._fetch_remote_model_bytes("unknown.rnnn")


def test_fetch_remote_model_bytes_rejects_hash_mismatch(monkeypatch):
    """A model with an unexpected digest cannot enter the local model store."""
    dummy_bytes = b"fake_rnnn_model_data_bytes_long_enough_for_validation" * 5

    class DummyResponse:
        status = 200
        reason = "OK"

        def read(self):
            return dummy_bytes

    class DummyConn:
        def __init__(self, host, timeout):
            pass

        def request(self, method, url, headers):
            pass

        def getresponse(self):
            return DummyResponse()

        def close(self):
            pass

    monkeypatch.setitem(modules.filters.ARNNDN_MODEL_SHA256, "cb.rnnn", "0" * 64)
    with patch("http.client.HTTPSConnection", side_effect=DummyConn), pytest.raises(ValueError, match="SHA-256"):
        modules.filters._fetch_remote_model_bytes("cb.rnnn")


def test_download_arnndn_model_success(tmp_path):
    """Verify _download_arnndn_model downloads, writes, and returns resolved model path."""
    dest = tmp_path / "models" / "arnndn" / "cb.rnnn"
    dummy_bytes = b"fake_rnnn_model_data_bytes_long_enough_for_validation" * 5

    with patch("modules.filters._fetch_remote_model_bytes", return_value=dummy_bytes):
        res = modules.filters._download_arnndn_model("cb.rnnn", dest)
        assert res == dest.resolve()
        assert dest.exists()
        assert dest.read_bytes() == dummy_bytes


def test_download_arnndn_model_fetch_failure(tmp_path):
    """Verify _download_arnndn_model safely returns None and logs warning on exception."""
    dest = tmp_path / "models" / "arnndn" / "missing.rnnn"
    with patch("modules.filters._fetch_remote_model_bytes", side_effect=RuntimeError("Connection failed")):
        res = modules.filters._download_arnndn_model("missing.rnnn", dest)
        assert res is None
        assert not dest.exists()


def test_try_auto_download_arnndn_rejects_unpinned_model():
    """Verify automatic downloads are limited to models with pinned digests."""
    with patch("modules.filters._download_arnndn_model") as download:
        assert modules.filters._try_auto_download_arnndn("unknown.rnnn") is None
    download.assert_not_called()


def test_resolve_arnndn_model_path_triggers_download(tmp_path, monkeypatch):
    """Auto-download must land in the package model store, not the working directory."""
    monkeypatch.chdir(tmp_path)
    store = tmp_path / "package_models"
    monkeypatch.setattr(modules.filters, "MODELS_DIR", store)
    target_path = store / "arnndn" / "bd.rnnn"

    def fake_download(model_name, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"downloaded_bd_model_data_bytes" * 5)
        return dest.resolve()

    with patch("modules.filters._download_arnndn_model", side_effect=fake_download) as mock_dl:
        resolved = modules.filters._resolve_arnndn_model_path("bd.rnnn")
        assert resolved == target_path.resolve()
        mock_dl.assert_called_once_with("bd.rnnn", target_path)


def test_resolve_arnndn_model_path_download_failed_returns_candidate(tmp_path, monkeypatch):
    """Verify _resolve_arnndn_model_path falls back to candidate path if download fails."""
    monkeypatch.chdir(tmp_path)
    with patch("modules.filters._download_arnndn_model", return_value=None):
        resolved = modules.filters._resolve_arnndn_model_path("unknown.rnnn")
        assert resolved == Path("unknown.rnnn").resolve()


def test_cb_arnndn_model_resolution():
    """Verify the canonical cb.rnnn model resolves to its full upstream path."""
    assert modules.filters._get_remote_model_url_path("cb.rnnn") == "/GregorR/rnnoise-models/master/conjoined-burgers-2018-08-28/cb.rnnn"


def test_validate_arnndn_file_integrity_corrupt_deleted(tmp_path):
    """Corrupt model file is automatically deleted when SHA-256 verification fails."""
    model_file = tmp_path / "cb.rnnn"
    model_file.write_bytes(b"corrupt model bytes")

    assert modules.filters._validate_arnndn_file_integrity(model_file, "cb.rnnn") is False
    assert not model_file.exists()


def test_validate_arnndn_file_integrity_valid_retained(tmp_path):
    """Valid model file passes integrity check and is retained."""
    model_file = tmp_path / "cb.rnnn"
    dummy_bytes = b"valid_model_bytes_12345" * 10
    model_file.write_bytes(dummy_bytes)

    with patch.dict(modules.filters.ARNNDN_MODEL_SHA256, {"cb.rnnn": hashlib.sha256(dummy_bytes).hexdigest()}):
        assert modules.filters._validate_arnndn_file_integrity(model_file, "cb.rnnn") is True
        assert model_file.exists()


def test_resolve_arnndn_model_path_redownloads_corrupt_file(tmp_path, monkeypatch):
    """A corrupt local model is purged and replaced via auto-download."""
    monkeypatch.chdir(tmp_path)
    store = tmp_path / "package_models"
    store_dir = store / "arnndn"
    store_dir.mkdir(parents=True)
    target_path = store_dir / "cb.rnnn"
    target_path.write_bytes(b"corrupt")
    monkeypatch.setattr(modules.filters, "MODELS_DIR", store)

    def fake_download(model_name, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"downloaded_good_model" * 10)
        return dest.resolve()

    with patch("modules.filters._download_arnndn_model", side_effect=fake_download) as mock_dl:
        resolved = modules.filters._resolve_arnndn_model_path("cb.rnnn")
        assert resolved == target_path.resolve()
        mock_dl.assert_called_once_with("cb.rnnn", target_path)
