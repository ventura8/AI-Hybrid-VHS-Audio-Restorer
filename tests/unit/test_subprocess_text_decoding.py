"""Every subprocess read as text decodes UTF-8 with replacement, whatever the platform's code page.

On Windows `text=True` alone decodes with the ANSI code page (cp1252 here). The IA clip
1994-6-ntsc20241207 carries a Cyrillic title that ffmpeg prints with every run; its UTF-8
bytes include 0x8f, which cp1252 cannot map, so the reader thread raised UnicodeDecodeError
and the mux's loudness measurement silently fell back to single-pass normalisation. These
tests run a real child that writes such bytes (and bytes that are not UTF-8 either, so a
strict UTF-8 decode on Linux fails too) through each parsing path, with the keyword
arguments the code under test passes; the AST check keeps new calls in line.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.apply_patches as apply_patches
import scripts.audit_hardware as audit_hardware
import scripts.curate_ia_corpus as curate_ia_corpus
import scripts.curate_massive_ia_corpus as curate_massive
import scripts.run_multi_pass_tuning as run_multi_pass_tuning
import scripts.sweep_denoise_settings as sweep
import scripts.tune_restoration as tune
from modules import mastering, stage_cache_key
from scripts.audio_matrix import piper

REPO = Path(__file__).resolve().parents[2]
TITLE = "Олег Газманов - Загулял (1994)"
# The title as UTF-8 (0x8f among its bytes: undefined in cp1252), then bytes valid in neither encoding.
PAYLOAD = TITLE.encode("utf-8") + b" \x81\x8d\x8f\x90\x9d"
REAL_RUN = subprocess.run
REAL_CHECK_OUTPUT = subprocess.check_output
EMIT = "import sys; sys.stdout.buffer.write(bytes.fromhex(sys.argv[1])); sys.stderr.buffer.write(bytes.fromhex(sys.argv[2]))"
LOUDNORM_JSON = json.dumps(
    {"input_i": "-20.10", "input_tp": "-3.20", "input_lra": "5.00", "input_thresh": "-30.50", "target_offset": "0.10"}, indent=1
).encode("ascii")
SUBPROCESS_FUNCS = {"run", "Popen", "check_output", "check_call", "call"}
# On Windows a capture's reader thread decodes each stream: a decode error there only prints
# a thread traceback (as the hunt log shows) and leaves that stream empty, so it fails here.
pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")


def _child(stdout, stderr):
    """A Python child that writes these exact bytes to stdout and stderr."""
    return [sys.executable, "-c", EMIT, stdout.hex(), stderr.hex()]


@pytest.fixture
def emit(monkeypatch):
    """Swaps each subprocess call's command for a child writing given bytes; the caller's keyword arguments stay as written."""
    seen = []

    def install(stdout=b"", stderr=b""):
        def fake_run(_cmd, *args, **kwargs):
            seen.append(kwargs)
            return REAL_RUN(_child(stdout, stderr), *args, **kwargs)

        def fake_check_output(_cmd, *args, **kwargs):
            seen.append(kwargs)
            return REAL_CHECK_OUTPUT(_child(stdout, stderr), *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(subprocess, "check_output", fake_check_output)
        return seen

    return install


def test_the_payload_breaks_both_legacy_decodes():
    """The bytes these tests send defeat cp1252 and strict UTF-8 alike, so each test below fails on today's text=True."""
    with pytest.raises(UnicodeDecodeError):
        PAYLOAD.decode("cp1252")
    with pytest.raises(UnicodeDecodeError):
        PAYLOAD.decode("utf-8")


def test_the_loudness_measurement_reads_through_a_cyrillic_title(emit):
    seen = emit(stderr=b"title : " + PAYLOAD + b"\n[Parsed_loudnorm_0]\n" + LOUDNORM_JSON)
    measurements = mastering._run_loudness_analysis(["v.mp4", "a.wav"], "[1:a]loudnorm=print_format=json[mastered]")
    assert measurements["input_i"] == "-20.10"
    assert seen[0]["encoding"] == "utf-8"
    assert seen[0]["errors"] == "replace"


def test_the_gpu_driver_query_reads_any_bytes(emit, monkeypatch):
    emit(stdout=PAYLOAD + b"\n")
    monkeypatch.setattr(stage_cache_key.shutil, "which", lambda name: "nvidia-smi")
    stage_cache_key.gpu_driver.cache_clear()
    try:
        driver = stage_cache_key.gpu_driver()
    finally:
        stage_cache_key.gpu_driver.cache_clear()
    assert driver.startswith(TITLE)
    assert "�" in driver


def test_the_hardware_audit_reads_any_bytes(emit, monkeypatch):
    emit(stdout=b"GPU " + PAYLOAD + b"\n")
    monkeypatch.setattr(audit_hardware.shutil, "which", lambda name: "nvidia-smi")
    assert audit_hardware._nvidia_smi() == [f"GPU {TITLE} �����"]


def test_the_corpus_curator_reads_a_duration_beside_a_cyrillic_warning(emit):
    emit(stdout=b"12.5\n", stderr=PAYLOAD)
    assert curate_ia_corpus._get_clip_duration(Path("clip.mp4")) == pytest.approx(12.5)


def test_the_massive_curator_reads_a_duration_beside_a_cyrillic_warning(emit):
    emit(stdout=b"12.5\n", stderr=PAYLOAD)
    assert curate_massive._source_duration("clip.mp4") == pytest.approx(12.5)


def test_the_multi_pass_probe_turns_undecodable_output_into_none(emit):
    seen = emit(stdout=PAYLOAD)
    assert run_multi_pass_tuning._probe_duration(Path("tape.mpg")) is None
    assert (seen[0]["encoding"], seen[0]["errors"]) == ("utf-8", "replace")


def test_the_tuning_probe_reads_a_duration_beside_a_cyrillic_warning(emit):
    emit(stdout=b"90.25\n", stderr=PAYLOAD)
    assert tune.probe_duration(Path("tape.mpg")) == pytest.approx(90.25)


def test_the_tuning_config_probe_reads_past_a_cyrillic_line(emit, tmp_path):
    emit(stdout=PAYLOAD + b'\n{"process_mode": "cathar"}\n')
    assert tune.resolved_config(tmp_path, sys.executable) == {"process_mode": "cathar"}


def test_the_sweep_names_dirty_files_with_any_bytes(emit):
    emit(stdout=b" M " + PAYLOAD + b"\n")
    with pytest.raises(SystemExit, match="uncommitted changes"):
        sweep._require_clean({"modules/config.py"})


def test_the_sweep_config_probe_reads_past_a_cyrillic_line(emit, monkeypatch):
    emit(stdout=PAYLOAD + b'\n{"denoise_model": "x"}\n')
    monkeypatch.setattr(sweep, "_fresh_interpreter", lambda: ([sys.executable], {}))
    assert sweep._resolved_config("variant") == {"denoise_model": "x"}


def test_poetry_is_asked_for_utf8_and_read_back_as_utf8(emit, tmp_path):
    seen = emit(stdout=str(tmp_path / "Олег").encode("utf-8") + b"\n")
    candidates = apply_patches._prepend_poetry_site_packages([])
    assert candidates[0] == tmp_path / "Олег" / "Lib/site-packages"
    assert seen[0]["env"]["PYTHONIOENCODING"] == "utf-8"


def test_piper_sends_its_text_as_utf8(monkeypatch, tmp_path):
    received = tmp_path / "stdin.bin"
    code = f"import sys; open({str(received)!r}, 'wb').write(sys.stdin.buffer.read())"

    def fake_run(_cmd, *args, **kwargs):
        return REAL_RUN([sys.executable, "-c", code], *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(piper, "resolve_piper_python", lambda override=None: Path(sys.executable))
    piper.synthesize(TITLE, tmp_path / "voice.onnx", tmp_path / "out.wav")
    assert received.read_bytes() == TITLE.encode("utf-8")


def _keyword(call, name):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _is_true(node):
    return isinstance(node, ast.Constant) and node.value is True


def _call_name(call):
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


def _text_mode(call):
    return _is_true(_keyword(call, "text")) or _is_true(_keyword(call, "universal_newlines")) or _keyword(call, "encoding") is not None


def _constant(node):
    return node.value if isinstance(node, ast.Constant) else None


def _lenient_utf8(call):
    return _constant(_keyword(call, "encoding")) == "utf-8" and _constant(_keyword(call, "errors")) == "replace"


def _is_subprocess_call(node):
    return isinstance(node, ast.Call) and _call_name(node) in SUBPROCESS_FUNCS


def _text_calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if _is_subprocess_call(node) and _text_mode(node)]


def _sources():
    yield REPO / "restore_audio_hybrid.py"
    for folder in ("modules", "scripts"):
        yield from sorted((REPO / folder).rglob("*.py"))


def test_every_text_mode_subprocess_call_decodes_utf8_with_replacement():
    offenders = [
        f"{path.relative_to(REPO).as_posix()}:{call.lineno}" for path in _sources() for call in _text_calls(path) if not _lenient_utf8(call)
    ]
    assert not offenders, offenders


def test_the_ast_check_sees_a_bare_text_call():
    call = ast.parse("subprocess.run(cmd, capture_output=True, text=True)").body[0].value
    assert _text_mode(call)
    assert not _lenient_utf8(call)
