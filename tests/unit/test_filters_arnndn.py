import hashlib
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import modules.filters

_SYNTH_INPUT = "sine=frequency=440:duration=0.1"


def _ffmpeg_arnndn_ready():
    """True when a real FFmpeg with the arnndn filter is available for a round-trip test."""
    if not Path(modules.filters.FFMPEG_BIN).exists():
        return False
    try:
        return modules.filters.ffmpeg_has_filter("arnndn")
    except OSError:
        return False


def test_resolve_arnndn_model_path_searches_store(tmp_path, monkeypatch):
    """A bare model filename is found inside the model store search folders."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(modules.filters, "MODELS_DIR", tmp_path / "pkg_models")
    model_dir = tmp_path / "models" / "arnndn"
    model_dir.mkdir(parents=True)
    searched_model = model_dir / "custom.rnnn"
    searched_model.write_bytes(b"model")
    assert modules.filters._resolve_arnndn_model_path("custom.rnnn") == searched_model.resolve()


def test_resolve_arnndn_model_path_confines_absolute_to_store(tmp_path, monkeypatch):
    """An absolute model path is reduced to its basename under the store, never opened as given."""
    monkeypatch.chdir(tmp_path)
    store = tmp_path / "pkg_models"
    monkeypatch.setattr(modules.filters, "MODELS_DIR", store)
    outside = tmp_path / "outside" / "evil.rnnn"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"model")
    assert modules.filters._resolve_arnndn_model_path(str(outside)) == (store / "arnndn" / "evil.rnnn").resolve()


def test_resolve_arnndn_model_path_confines_traversal_to_store(tmp_path, monkeypatch):
    """A parent-traversal model name cannot escape the model store."""
    monkeypatch.chdir(tmp_path)
    store = tmp_path / "pkg_models"
    monkeypatch.setattr(modules.filters, "MODELS_DIR", store)
    assert modules.filters._resolve_arnndn_model_path("../../secret.rnnn") == (store / "arnndn" / "secret.rnnn").resolve()


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


def test_resolve_arnndn_model_path_unknown_falls_back_to_store(tmp_path, monkeypatch):
    """An unknown model absent from the store falls back to a (non-existent) store path."""
    monkeypatch.chdir(tmp_path)
    store = tmp_path / "pkg_models"
    monkeypatch.setattr(modules.filters, "MODELS_DIR", store)
    resolved = modules.filters._resolve_arnndn_model_path("unknown.rnnn")
    assert resolved == (store / "arnndn" / "unknown.rnnn").resolve()


@pytest.mark.parametrize("payload", ["m'.rnnn", "a;b.rnnn", "a[b].rnnn", "a,b.rnnn", 'a"b.rnnn'])
def test_escape_ffmpeg_filter_path_refuses_filtergraph_metacharacters(payload):
    """A model path with a quote or filtergraph metacharacter is refused, not escaped."""
    with pytest.raises(ValueError, match="Unsafe character"):
        modules.filters._escape_ffmpeg_filter_path(f"models/arnndn/{payload}")


def test_build_arnndn_filter_string_fails_closed_on_injection():
    """The injection payload cannot build a graph; _build_arnndn_filter_string raises."""
    with pytest.raises(ValueError, match="Unsafe character"):
        modules.filters._build_arnndn_filter_string("models/arnndn/x';amovie=target.wav")


def test_escape_ffmpeg_filter_path_escapes_drive_colon():
    """A Windows drive colon is backslash-escaped so it cannot split the filter option."""
    escaped = modules.filters._escape_ffmpeg_filter_path("C:/models/cb.rnnn")
    assert escaped == Path("C:/models/cb.rnnn").resolve().as_posix().replace(":", r"\:")


def _run_arnndn_graph(tmp_path, model_path):
    """Runs the arnndn graph for model_path on a synthetic input and returns FFmpeg stderr."""
    graph = modules.filters._build_arnndn_filter_string(str(model_path), highpass_freq=0, enable_adeclick=False)
    cmd = [modules.filters.FFMPEG_BIN, "-hide_banner", "-nostdin", "-f", "lavfi", "-i", _SYNTH_INPUT, "-af", graph, "-f", "null", "-"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False).stderr


@pytest.mark.skipif(not _ffmpeg_arnndn_ready(), reason="FFmpeg with arnndn filter is not available")
def test_escape_roundtrips_spaced_path_through_ffmpeg(tmp_path):
    """FFmpeg receives a spaced, drive-coloned model path literally, with no injected filter."""
    model = tmp_path / "sub dir" / "mo del.rnnn"
    model.parent.mkdir(parents=True)
    stderr = _run_arnndn_graph(tmp_path, model)
    assert model.resolve().as_posix() in stderr
    assert "amovie" not in stderr.lower()


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
