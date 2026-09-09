"""Profile generated fixtures through selected restoration modes when opted in."""

import argparse
import json
import subprocess
import sys
import threading
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


def _device_used_mb():
    """Returns VRAM currently in use on the device, across every process."""
    try:
        torch = import_module("torch")
        if not torch.cuda.is_available():
            return 0.0
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return (total_bytes - free_bytes) / (1024 * 1024)
    except (ImportError, AttributeError, RuntimeError):
        return 0.0


class _VramSampler:
    """Tracks device-level peak VRAM for the duration of one mode.

    torch.cuda.max_memory_allocated() reports only this process's PyTorch allocator, and the
    heaviest consumer here -- resemble-enhance -- runs as a child process. Its allocations were
    therefore absent from the figure this validation exists to record: an 8 GB card logged a
    1.6 GB "peak" in the same run that died with "CUDA out of memory". Sampling the device
    counts every process on it.
    """

    def __init__(self, interval=0.25):
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.peak_mb = 0.0

    def _run(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, _device_used_mb())
            self._stop.wait(self._interval)

    def __enter__(self):
        self.peak_mb = _device_used_mb()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return False


def _fixture_duration_seconds(wav_path):
    """Returns the fixture's duration, so throughput can be reported as a real-time factor."""
    try:
        import soundfile as sf

        with sf.SoundFile(str(wav_path)) as handle:
            return handle.frames / handle.samplerate
    except Exception:
        return 0.0


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
            expected_output = lang_work_dir / f"{video_path.stem}{processing._get_output_suffix(mode)}{video_path.suffix}"
            expected_output.unlink(missing_ok=True)
            started = time.monotonic()
            with _VramSampler() as sampler:
                success = processing.process_hybrid_audio(video_path, hardware.get_gpu_name(), lang_work_dir)
            elapsed = time.monotonic() - started
            fixture_seconds = _fixture_duration_seconds(wav_path)
            results.append(
                {
                    "fixture": f"{lang}/{wav_path.name}",
                    "mode": mode,
                    "success": success,
                    "elapsed_seconds": round(elapsed, 2),
                    # Real-time factor: how many seconds of tape are restored per second of
                    # wall clock. Comparable across fixtures of different lengths, which raw
                    # elapsed time is not.
                    "realtime_factor": round(fixture_seconds / elapsed, 2) if elapsed > 0 and fixture_seconds else None,
                    "peak_vram_mb": round(sampler.peak_mb, 2),
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
    # A run where every mode failed used to exit 0, so any caller -- CI, a shell wrapper, an
    # operator reading $? -- recorded it as a pass while the report said otherwise. Observed on
    # a host missing the Cathar binary: ten failed modes, exit status 0.
    failed = [run["mode"] for run in report["execution"] if not run["success"]] if isinstance(report.get("execution"), list) else []
    if failed:
        print(f"FAILED modes: {', '.join(sorted(set(failed)))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
