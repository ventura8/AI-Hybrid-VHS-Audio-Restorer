#!/usr/bin/env python3
"""Benchmark audio quality and restoration artifacts across restoration modes."""

import argparse
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import scipy.signal
import soundfile as sf

# Ensure modules package can be resolved
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.utils import FFMPEG_BIN, is_valid_audio
from scripts.ia_benchmark_common import _compute_noise_floor_db, _compute_rumble_pct, _compute_spectral_ratio, _split_channels


def analyze_audio_quality(wav_path, crt_hz=15625.0):
    """Measures key quality metrics for restored audio."""
    data, sr = sf.read(str(wav_path), dtype="float32")
    mono, left, right = _split_channels(data)

    peak = float(np.max(np.abs(data)))
    peak_db = 20.0 * np.log10(peak + 1e-9)
    rms = float(np.sqrt(np.mean(data**2)))
    rms_db = 20.0 * np.log10(rms + 1e-9)

    nf_db = _compute_noise_floor_db(mono, sr, rms_db)
    freqs, psd = scipy.signal.welch(mono, sr, nperseg=8192)
    whistle_ratio = _compute_spectral_ratio(psd, freqs, crt_hz) if sr > 2 * crt_hz else None
    rumble_pct = _compute_rumble_pct(psd, freqs)

    rms_l = float(np.sqrt(np.mean(left**2)))
    rms_r = float(np.sqrt(np.mean(right**2)))
    bal_db = abs(20.0 * np.log10((rms_l + 1e-9) / (rms_r + 1e-9)))

    return {
        "peak_db": round(peak_db, 2),
        "rms_db": round(rms_db, 2),
        "noise_floor_db": round(nf_db, 2),
        "peak_to_noise_db": round(peak_db - nf_db, 2),
        "crt_whistle_ratio": round(whistle_ratio, 2) if whistle_ratio is not None else None,
        "rumble_energy_pct": round(rumble_pct, 2),
        "stereo_balance_diff_db": round(bal_db, 2),
    }


def _extract_video_pcm(video_path, temp_wav):
    """Runs ffmpeg extraction and returns temp_wav if successful."""
    if temp_wav.exists():
        temp_wav.unlink()
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_f32le",
        "-ar",
        "44100",
        str(temp_wav),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=60)
        if res.returncode == 0 and is_valid_audio(temp_wav):
            return temp_wav
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        pass
    if temp_wav.exists():
        temp_wav.unlink()
    return None


def _resolve_evaluation_wav(file_path, work_dir):
    """Extracts 32-bit PCM audio if input is a video container, or returns audio path."""
    if file_path.suffix.lower() in [".mp4", ".mpg", ".mkv", ".avi", ".mov", ".m4v"]:
        temp_wav = work_dir / f"{file_path.stem}_eval.wav"
        return _extract_video_pcm(file_path, temp_wav)
    return file_path if is_valid_audio(file_path) else None


def _print_benchmark_table(results):
    """Outputs structured benchmark table for analyzed files."""
    header = f"\n{'Label':<15} | {'Noise Floor':<12} | {'SNR':<8} | " f"{'CRT Whistle':<12} | {'Rumble %':<10} | {'Stereo Imbal':<12}"
    print(header)
    print("-" * 80)
    for r in results:
        lbl = r["label"]
        nf = r["noise_floor_db"]
        snr = r.get("peak_to_noise_db", r.get("snr_db", 0))
        crt = f"{r['crt_whistle_ratio']}x" if r.get("crt_whistle_ratio") is not None else "N/A"
        rumble = r["rumble_energy_pct"]
        bal = r["stereo_balance_diff_db"]
        print(f"{lbl:<15} | {nf:>7} dB    | {snr:>5} dB | {crt:>10}   | {rumble:>8}%  | {bal:>8} dB")
    print("=" * 80)


def _collect_comparison_metrics(files, work_dir, crt_hz):
    """Evaluates each candidate file and gathers quality metrics."""
    results = []
    for label, file_path in files:
        if not file_path.exists():
            print(f"Skipping {label}: '{file_path}' not found")
            continue

        eval_wav = _resolve_evaluation_wav(file_path, work_dir)
        if not eval_wav:
            print(f"Skipping {label}: could not extract valid audio from '{file_path}'")
            continue

        metrics = analyze_audio_quality(eval_wav, crt_hz=crt_hz)
        metrics["label"] = label
        results.append(metrics)
    return results


def compare_files(original_file, restored_files, work_dir=None, crt_hz=15625.0):
    """Compares original audio vs multiple restored candidate outputs."""
    print("=" * 80)
    print("AUDIO RESTORATION QUALITY BENCHMARK")
    print("=" * 80)

    if not work_dir:
        work_dir = Path(__file__).resolve().parent.parent / "experiments" / "eval_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    files = [("ORIGINAL", Path(original_file))] + [(name, Path(p)) for name, p in restored_files]
    results = _collect_comparison_metrics(files, work_dir, crt_hz)
    _print_benchmark_table(results)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True, help="Path to original audio/video file")
    parser.add_argument("--restored", nargs="+", required=True, help="List of 'Label=Path' pairs to compare")
    parser.add_argument("--work-dir", type=Path, default=None, help="Directory for temporary WAV extractions")
    parser.add_argument("--crt-hz", type=float, default=15625.0, help="CRT whistle frequency in Hz")
    return parser.parse_args()


def main():
    args = _parse_args()
    restored_pairs = []
    for pair in args.restored:
        if "=" in pair:
            lbl, path = pair.split("=", 1)
        else:
            lbl = Path(pair).stem
            path = pair
        restored_pairs.append((lbl, Path(path)))

    compare_files(args.original, restored_pairs, work_dir=args.work_dir, crt_hz=args.crt_hz)


if __name__ == "__main__":
    main()
