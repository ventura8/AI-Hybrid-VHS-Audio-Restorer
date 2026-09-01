"""Profile generated fixtures through selected restoration modes when opted in."""

import argparse
import json
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
build_report = import_module("scripts.audit_hardware").build_report

DEFAULT_MODES = (
    "auto_pure_linear",
    "auto",
    "multipass_auto",
    "auto_pure",
    "cathar",
    "hybrid",
    "denoise_only",
    "auto_ffmpeg_native",
    "vhs_native",
    "arnndn_speech",
)


def _resolve_language_names(fixtures_dir, languages):
    """Returns sorted language directory names if not explicitly specified."""
    if not fixtures_dir.exists() or not fixtures_dir.is_dir():
        raise FileNotFoundError(f"Generate fixtures first: fixtures directory not found at {fixtures_dir}")
    if languages:
        return languages
    available = sorted([path.name for path in fixtures_dir.iterdir() if path.is_dir()])
    if not available:
        raise FileNotFoundError(f"Generate fixtures first: no language fixture directories found in {fixtures_dir}")
    return available


def _find_missing_paths(paths):
    """Finds missing fixture files and raises FileNotFoundError if any."""
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Generate fixtures first: " + ", ".join(missing))


def collect_fixture_paths(fixtures_dir, profile, languages):
    """Return generated VHS fixture paths for the selected profile."""
    names = ("short", "mid") if profile == "core" else (profile,)
    language_names = _resolve_language_names(fixtures_dir, languages)
    paths = [fixtures_dir / language / f"{name}_vhs.wav" for language in language_names for name in names]
    _find_missing_paths(paths)
    return paths


def build_dry_run_report(paths, modes):
    """Report exactly what an explicit execution run will process."""
    return {"hardware": build_report(), "fixtures": [str(path) for path in paths], "modes": list(modes), "execution": "not-run"}


def require_nvidia_cuda():
    """Fail before a physical run unless PyTorch is executing on NVIDIA CUDA."""
    hardware = import_module("modules.hardware")
    settings = hardware.get_optimal_settings()
    if settings["is_nvidia"] and not settings["cpu_only_fallback"]:
        return settings
    raise RuntimeError("NVIDIA CUDA is required for --execute hardware validation.")


def _make_video_fixture(wav_path, output_path):
    """Mux a generated WAV with a black video stream for the normal pipeline."""
    utilities = import_module("modules.utils")
    command = [
        utilities.FFMPEG_BIN,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=640x480:r=25",
        "-i",
        str(wav_path),
        "-shortest",
        "-c:v",
        "mpeg4",
        "-c:a",
        "pcm_f32le",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)


def _peak_vram_mb():
    """Read PyTorch's peak allocation when CUDA is available."""
    try:
        torch = import_module("torch")
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0
    except ImportError:
        return 0.0


def _reset_peak_vram():
    """Reset PyTorch peak CUDA memory stats if available."""
    try:
        torch = import_module("torch")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, AttributeError):
        pass


def execute_validation(paths, modes, work_dir):
    """Run each mode through generated video fixtures and profile the result."""
    processing = import_module("modules.processing")
    hardware = import_module("modules.hardware")
    work_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for wav_path in paths:
        lang = wav_path.parent.name
        lang_work_dir = work_dir / lang
        lang_work_dir.mkdir(parents=True, exist_ok=True)
        video_path = lang_work_dir / f"{wav_path.stem}.mkv"
        _make_video_fixture(wav_path, video_path)
        for mode in modes:
            processing.PROCESS_MODE = mode
            _reset_peak_vram()
            started = time.monotonic()
            success = processing.process_hybrid_audio(video_path, hardware.get_gpu_name(), lang_work_dir)
            elapsed = time.monotonic() - started
            results.append(
                {
                    "fixture": f"{lang}/{wav_path.name}",
                    "mode": mode,
                    "success": success,
                    "elapsed_seconds": round(elapsed, 2),
                    "peak_vram_mb": _peak_vram_mb(),
                }
            )
    return results


def main(argv=None):
    """Create a hardware-validation report; execution requires --execute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("short", "mid", "longform", "core"), default="core", nargs="?")
    parser.add_argument("--fixtures-dir", type=Path, default=Path("artifacts/audio-matrix"))
    parser.add_argument("--language", action="append", dest="languages", default=[])
    parser.add_argument("--mode", action="append", dest="modes")
    parser.add_argument("--report", type=Path, default=Path("artifacts/hardware-validation.json"))
    parser.add_argument("--execute", action="store_true", help="Run each selected mode through generated MKV fixtures.")
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/hardware-work"))
    args = parser.parse_args(argv)
    paths = collect_fixture_paths(args.fixtures_dir, args.profile, args.languages)
    report = build_dry_run_report(paths, args.modes or DEFAULT_MODES)
    if args.execute:
        require_nvidia_cuda()
        report["execution"] = execute_validation(paths, report["modes"], args.work_dir)
    report["created_at_epoch"] = round(time.time())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
