"""The stage cache's key outside the call: the code, ffmpeg and cathar, the model files, the blend weights and the runtime."""

import os
import subprocess
from types import SimpleNamespace

import pytest

from modules import config
from modules import stage_cache_key as key_mod
from tests.unit.test_stage_cache_key import samples, write


def _tree(root):
    (root / "modules" / "modes").mkdir(parents=True)
    for relative, text in (("modules/a.py", "A = 1\n"), ("modules/modes/b.py", "B = 1\n"), ("modules/sibilant_guard.py", "S = 1\n")):
        (root / relative).write_text(text, encoding="utf-8")
    (root / "restore_audio_hybrid.py").write_text("E = 1\n", encoding="utf-8")
    return root


def _fingerprint(root):
    key_mod.code_fingerprint.cache_clear()
    return key_mod.code_fingerprint(root)


def _fingerprint_after(root, relative):
    (root / relative).write_text(f"X = '{relative}'\n", encoding="utf-8")
    return _fingerprint(root)


@pytest.mark.parametrize("relative", ["modules/a.py", "modules/modes/b.py", "restore_audio_hybrid.py", "modules/sibilant_guard.py"])
def test_the_code_key_follows_every_pre_cache_source(tmp_path, relative):
    root = _tree(tmp_path / "repo")
    base = _fingerprint(root)
    assert _fingerprint_after(root, relative) != base


def test_a_tree_without_sources_cannot_be_keyed(tmp_path):
    with pytest.raises(ValueError):
        _fingerprint(tmp_path)


def test_the_document_names_its_code_root(tmp_path, monkeypatch):
    monkeypatch.setattr(key_mod, "binaries", lambda: {})
    monkeypatch.setattr(key_mod, "runtime", lambda: {})
    root = _tree(tmp_path / "repo")
    wav = write(tmp_path / "in.wav", samples())
    document = key_mod.key_document(wav, {"stage": key_mod.STAGE_FULL_MIX_NEURAL}, root=root)
    assert document["code"] == _fingerprint(root)


def test_a_binary_is_known_by_its_content(tmp_path):
    exe = tmp_path / "tool.exe"
    exe.write_bytes(b"one")
    first = key_mod.binary_identity(exe)
    exe.write_bytes(b"other")
    assert (first["name"], key_mod.binary_identity(exe)["sha256"] != first["sha256"]) == ("tool.exe", True)


def test_a_missing_binary_is_absent_and_a_bare_name_is_resolved(tmp_path, monkeypatch):
    exe = tmp_path / "found.exe"
    exe.write_bytes(b"found")
    monkeypatch.setattr(key_mod.shutil, "which", lambda name: str(exe) if name == "found" else None)
    absent, found = key_mod.binary_identity(tmp_path / "missing.exe"), key_mod.binary_identity("found")
    assert (absent, found["sha256"]) == ({"absent": True}, key_mod.file_sha256(exe))


def test_ffmpeg_is_also_known_by_its_version_output(tmp_path, monkeypatch):
    exe = tmp_path / "ffmpeg.exe"
    exe.write_bytes(b"ffmpeg build")
    outputs = iter([b"version 1", b"version 2"])
    commands = []

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return SimpleNamespace(stdout=next(outputs))

    monkeypatch.setattr(key_mod.subprocess, "run", fake_run)
    key_mod._version_sha256.cache_clear()
    first = key_mod.binary_identity(exe, with_version=True)["version_sha256"]
    key_mod._version_sha256.cache_clear()
    second = key_mod.binary_identity(exe, with_version=True)["version_sha256"]
    assert (first != second, commands[0]) == (True, [str(exe), "-version"])


def test_a_version_query_that_hangs_cannot_be_keyed(tmp_path, monkeypatch):
    exe = tmp_path / "ffmpeg.exe"
    exe.write_bytes(b"ffmpeg hangs")

    def hang(cmd, **_kwargs):
        raise subprocess.TimeoutExpired(cmd, 30)

    monkeypatch.setattr(key_mod.subprocess, "run", hang)
    key_mod._version_sha256.cache_clear()
    with pytest.raises(RuntimeError):
        key_mod.binary_identity(exe, with_version=True)


def test_the_binaries_are_the_ones_the_stages_resolve(tmp_path, monkeypatch):
    ffmpeg, cathar = tmp_path / "ffmpeg.exe", tmp_path / "cathar.exe"
    ffmpeg.write_bytes(b"f")
    cathar.write_bytes(b"c")
    monkeypatch.setattr(key_mod.utils, "FFMPEG_BIN", str(ffmpeg))
    monkeypatch.setattr(key_mod.utils, "CATHAR_BIN", str(cathar))
    monkeypatch.setattr(key_mod.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=b"v"))
    key_mod._version_sha256.cache_clear()
    found = key_mod.binaries()
    assert (found["ffmpeg"]["sha256"], "version_sha256" in found["ffmpeg"]) == (key_mod.file_sha256(ffmpeg), True)
    assert found["cathar"] == {"name": "cathar.exe", "sha256": key_mod.file_sha256(cathar)}


def test_a_model_is_known_with_its_config_and_the_parameter_tables(tmp_path):
    for name in ("deep.ckpt", "deep_config.yaml", "vr_model_data.json", "unrelated.pth"):
        (tmp_path / name).write_bytes(b"m")
    (tmp_path / "deep_folder").mkdir()
    names = [entry[0] for entry in key_mod.model_identity(tmp_path, ["deep.ckpt"])]
    assert (names, key_mod.model_identity(tmp_path / "absent", ["deep.ckpt"])) == (
        ["deep.ckpt", "deep_config.yaml", "vr_model_data.json"],
        [],
    )


def test_the_model_files_are_read_from_the_folder_the_separator_uses(tmp_path, monkeypatch):
    """AUDIO_SEPARATOR_MODEL_DIR replaces the folder the stage passes: a fine-tune under the stock name must miss."""
    stock, tuned = tmp_path / "stock", tmp_path / "tuned"
    for folder, size in ((stock, 1), (tuned, 2)):
        folder.mkdir()
        (folder / "deep.ckpt").write_bytes(b"m" * size)
    monkeypatch.setattr(key_mod.utils, "MODELS_DIR", stock)
    monkeypatch.delenv(key_mod.MODEL_DIR_ENV, raising=False)
    call = {"denoise_model": "deep.ckpt"}
    before = key_mod.models(call)
    monkeypatch.setenv(key_mod.MODEL_DIR_ENV, str(tuned))
    after = key_mod.models(call)
    assert (before["dir"], after["dir"]) == (os.path.abspath(stock), os.path.abspath(tuned))
    assert (before["files"][0][:2], after["files"][0][:2]) == (["deep.ckpt", 1], ["deep.ckpt", 2])


def test_the_model_candidates_take_the_call_and_the_strategy_and_drop_empty_names():
    names = set(key_mod.model_candidates({"denoise_model": "call.pth", "strategy": {"denoise_model": "scan.pth"}}))
    assert ({"call.pth", "scan.pth", config.DENOISE_MODEL} <= names, {"", "None"} & names) == (True, set())


def test_the_blend_weights_are_known_by_content_or_as_absent(tmp_path, monkeypatch):
    weights = tmp_path / "w.npz"
    monkeypatch.setattr(key_mod.blend_weights, "DEFAULT_WEIGHTS_PATH", weights)
    absent = key_mod.assets()
    weights.write_bytes(b"w")
    assert (absent, key_mod.assets()) == ({"blend_weights": None}, {"blend_weights": key_mod.file_sha256(weights)})


def _fake_torch(cuda):
    return SimpleNamespace(
        __version__="2.0",
        cuda=SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda index: f"gpu{index}"),
        version=SimpleNamespace(cuda="13.0" if cuda else None),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 9 if cuda else None)),
    )


def test_torch_is_known_by_its_build_and_device():
    found = (
        key_mod.torch_identity(None),
        key_mod.torch_identity(_fake_torch(True))["device"],
        key_mod.torch_identity(_fake_torch(False))["device"],
    )
    assert found == (None, "gpu0", None)


def _driver(monkeypatch, which, run):
    monkeypatch.setattr(key_mod.shutil, "which", which)
    monkeypatch.setattr(key_mod.subprocess, "run", run)
    key_mod.gpu_driver.cache_clear()
    try:
        return key_mod.gpu_driver()
    finally:
        key_mod.gpu_driver.cache_clear()


def _fails(*_args, **_kwargs):
    raise subprocess.TimeoutExpired("nvidia-smi", 30)


def _reports(stdout):
    return lambda *_args, **_kwargs: SimpleNamespace(stdout=stdout)


def _absent(_name):
    return None


def _found(_name):
    return "nvidia-smi.exe"


@pytest.mark.parametrize(
    ("which", "run", "expected"),
    [(_absent, _fails, None), (_found, _reports("581.15\r\n"), "581.15"), (_found, _reports(""), None), (_found, _fails, None)],
)
def test_the_gpu_driver_is_what_nvidia_smi_reports_or_none(monkeypatch, which, run, expected):
    """A driver update can move the GPU's numerics, so it is in the key; without nvidia-smi it is None."""
    assert _driver(monkeypatch, which, run) == expected


def test_the_runtime_names_the_steering_environment_the_cpu_and_never_the_temp_folders(monkeypatch):
    monkeypatch.setattr(key_mod, "torch", None)
    monkeypatch.setattr(key_mod, "gpu_driver", lambda: "581.15")
    monkeypatch.setattr(key_mod, "cpu_name", lambda: "AMD Ryzen 9")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("TEMP", "C:/somewhere")
    found = key_mod.runtime()
    assert (found["torch"], found["env"]["OMP_NUM_THREADS"], "TEMP" in found["env"]) == (None, "4", False)
    assert (found["chunk_seconds"] > 0, len(found["distributions"]), found["gpu_driver"]) == (True, 64, "581.15")
    assert (found["cpu"], bool(found["machine"])) == ("AMD Ryzen 9", True)


ENVIRONMENT = {
    "AUDIO_SEPARATOR_FORCE_CPU_COMPLEX": "1",
    "AUDIO_SEPARATOR_MODEL_DIR": "D:/tuned",
    "omp_num_threads": "2",
    "NPY_DISABLE_CPU_FEATURES": "AVX512F",
    "PYTHONHASHSEED": "0",
    "AI_RESTORE_CATHAR_BIN": "cathar.exe",
    "AI_RESTORE_STAGE_CACHE": "C:/cache",
    "AI_RESTORE_STAGE_CACHE_MAX_GB": "5",
    "AI_RESTORE_EVENT_LOG": "C:/events",
    "TEMP": "C:/work/tmp",
    "PATH": "C:/bin",
}


def test_the_environment_is_keyed_by_prefix_but_the_caches_own_the_event_log_and_the_temp_folders():
    """Deny by default, as the configuration is: a loop's `env:` knob on a numerics variable cannot share a key."""
    keyed = set(key_mod.keyed_environment(ENVIRONMENT))
    expected = {"AUDIO_SEPARATOR_FORCE_CPU_COMPLEX", "AUDIO_SEPARATOR_MODEL_DIR", "omp_num_threads", "NPY_DISABLE_CPU_FEATURES"}
    assert keyed == expected | {"PYTHONHASHSEED", "AI_RESTORE_CATHAR_BIN"}


def test_the_cpu_is_named_once_per_process(monkeypatch):
    calls = []
    monkeypatch.setattr(key_mod.hardware, "get_cpu_name", lambda: calls.append(1) or "Ryzen")
    key_mod.cpu_name.cache_clear()
    try:
        names = (key_mod.cpu_name(), key_mod.cpu_name())
    finally:
        key_mod.cpu_name.cache_clear()
    assert (names, calls) == (("Ryzen", "Ryzen"), [1])
