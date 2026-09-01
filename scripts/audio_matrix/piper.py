"""Thin, deterministic Piper command-line adapter."""

import hashlib
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PIPER_RUNTIME_ROOT = REPOSITORY_ROOT / "tools" / "piper-tts"


def verify_checksum(model_path, expected_checksum):
    """Verify the upstream MD5 or SHA-256 checksum of a downloaded model."""
    algorithm = hashlib.md5 if len(expected_checksum) == 32 else hashlib.sha256
    digest = algorithm(Path(model_path).read_bytes()).hexdigest()
    return digest.lower() == expected_checksum.lower()


def _piper_python_candidates():
    """Return the isolated Piper interpreter locations for the current OS."""
    return (PIPER_RUNTIME_ROOT / ".venv" / "Scripts" / "python.exe", PIPER_RUNTIME_ROOT / ".venv" / "bin" / "python")


def resolve_piper_python(override=None):
    """Return the isolated Piper Python interpreter or raise an actionable error."""
    candidates = (Path(override),) if override else _piper_python_candidates()
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("Piper runtime is missing; run the project installer to provision tools/piper-tts/.venv.")


def synthesize(text, model_path, output_path, python_executable=None):
    """Create a WAV using Piper's zero-noise deterministic settings."""
    piper_python = resolve_piper_python(python_executable)
    command = [
        str(piper_python),
        "-m",
        "piper",
        "--model",
        str(model_path),
        "--output-file",
        str(output_path),
        "--noise-scale",
        "0",
        "--noise-w",
        "0",
        "--length-scale",
        "1.0",
    ]
    subprocess.run(command, input=text, text=True, check=True)


def ensure_voice(voice, expected_md5, voices_dir, python_executable=None):
    """Download a pinned voice once and return its verified ONNX model path."""
    voices_dir = Path(voices_dir)
    model_path = voices_dir / f"{voice}.onnx"
    if not model_path.exists():
        piper_python = resolve_piper_python(python_executable)
        command = [str(piper_python), "-m", "piper.download_voices", "--download-dir", str(voices_dir), voice]
        subprocess.run(command, check=True)
    if not verify_checksum(model_path, expected_md5):
        model_path.unlink(missing_ok=True)
        raise RuntimeError(f"Piper checksum mismatch for {voice}: {model_path}")
    return model_path
