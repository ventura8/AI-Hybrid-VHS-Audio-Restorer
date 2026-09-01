#!/usr/bin/env python3
"""Scan VHS captures from a directory to extract empirical acoustic profiles."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import soundfile as sf

# Ensure modules package can be resolved
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import EXTS
from modules.filters import (
    _detect_analog_clipping,
    _detect_click_density,
    _detect_crt_flyback_notch,
    _detect_dc_offset_bias,
    _detect_enclosure_resonance_notch,
    _detect_low_frequency_rumble,
    _detect_mains_buzz_notch,
    _detect_stereo_azimuth_skew,
    _detect_stereo_balance_imbalance,
    _estimate_noise_floor_and_reduction,
)
from modules.utils import FFMPEG_BIN


def _extract_audio_sample(tape_path, temp_wav, start_sec, duration_sec):
    """Extracts raw stereo 32-bit float audio from capture."""
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(tape_path),
        "-vn",
        "-acodec",
        "pcm_f32le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(temp_wav),
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return res.returncode == 0 and temp_wav.exists() and temp_wav.stat().st_size > 1000


def _profile_audio(stereo_data, sr):
    """Extracts all acoustic parameters via modules.filters analysis routines."""
    mono = np.mean(stereo_data, axis=1) if stereo_data.ndim > 1 else stereo_data
    nf_db, nr_db = _estimate_noise_floor_and_reduction(mono)
    crt = _detect_crt_flyback_notch(mono, sr)
    mains = _detect_mains_buzz_notch(mono, sr, crt)
    rumble = _detect_low_frequency_rumble(mono, sr)
    clicks = _detect_click_density(mono)
    clipping = _detect_analog_clipping(mono)
    azimuth = _detect_stereo_azimuth_skew(stereo_data, sr) if stereo_data.ndim > 1 else 0.0
    dc_block = _detect_dc_offset_bias(mono)
    balance = _detect_stereo_balance_imbalance(stereo_data) if stereo_data.ndim > 1 else 0.0
    resonance = _detect_enclosure_resonance_notch(mono, sr)

    peak = float(np.max(np.abs(stereo_data)))
    peak_db = 20.0 * np.log10(peak + 1e-9)
    rms = float(np.sqrt(np.mean(stereo_data**2)))
    rms_db = 20.0 * np.log10(rms + 1e-9)

    return {
        "peak_db": round(peak_db, 1),
        "rms_db": round(rms_db, 1),
        "noise_floor_db": round(nf_db, 1),
        "snr_est_db": round(peak_db - nf_db, 1),
        "recommended_nr_db": round(nr_db, 1),
        "rumble_cutoff_hz": rumble,
        "click_density_high": clicks,
        "clipping_detected": clipping,
        "mains_hum_hz": mains,
        "crt_whistle_hz": crt,
        "dc_offset_bias": dc_block,
        "azimuth_skew_ms": azimuth,
        "stereo_balance_imbalance_db": balance,
        "resonance_hz": resonance,
    }


def extract_and_analyze(tape_path, work_dir, start_sec=30, duration_sec=90):
    """Extracts a representative segment and computes acoustic profile."""
    temp_wav = work_dir / f"{tape_path.stem}_sample.wav"
    if not _extract_audio_sample(tape_path, temp_wav, start_sec, duration_sec):
        return None

    try:
        stereo_data, sr = sf.read(str(temp_wav), dtype="float32")
        profile = _profile_audio(stereo_data, sr)
        profile["tape"] = tape_path.name
        return profile
    except Exception as exc:
        sys.stderr.write(f"Failed to profile {tape_path.name}: {exc}\n")
        return None
    finally:
        if temp_wav.exists():
            try:
                temp_wav.unlink()
            except OSError:
                pass


def _is_valid_capture(path, valid_exts):
    return path.is_file() and path.suffix.lower() in valid_exts


def _scan_directory_captures(directory):
    valid_exts = {e.lower() for e in EXTS}
    return sorted([p for p in directory.iterdir() if _is_valid_capture(p, valid_exts)])


def _collect_video_files(input_path):
    """Collects candidate video files from a file or folder path."""
    target = Path(input_path)
    if target.is_file():
        return [target]
    if target.is_dir():
        return _scan_directory_captures(target)
    return []


def run_batch_scan(input_target, output_json, start_sec=30, duration_sec=90):
    """Scans video captures from input target and saves acoustic profile JSON."""
    candidates = _collect_video_files(input_target)
    if not candidates:
        print(f"No valid video files found at '{input_target}'", file=sys.stderr)
        return []

    work_dir = output_json.parent / "scan_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for tape_path in candidates:
        print(f"Analyzing {tape_path.name}...", flush=True)
        res = extract_and_analyze(tape_path, work_dir, start_sec=start_sec, duration_sec=duration_sec)
        if res:
            results.append(res)
            nf = res["noise_floor_db"]
            snr = res["snr_est_db"]
            rumble = res["rumble_cutoff_hz"]
            mains = res["mains_hum_hz"]
            crt = res["crt_whistle_hz"]
            bal = res["stereo_balance_imbalance_db"]
            print(f"  NF: {nf} dB | SNR: {snr} dB | Rumble: {rumble} Hz | Mains: {mains} Hz | CRT: {crt} Hz | Bal: {bal} dB", flush=True)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved acoustic profiles for {len(results)} files to {output_json}", flush=True)
    return results


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input directory or specific video file to analyze")
    parser.add_argument("--output", type=Path, default=Path("tapes_profile_analysis.json"), help="Output JSON profile destination")
    parser.add_argument("--duration", type=int, default=90, help="Sample duration in seconds")
    parser.add_argument("--start", type=int, default=30, help="Sample start offset in seconds")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_batch_scan(args.input, args.output, start_sec=args.start, duration_sec=args.duration)


if __name__ == "__main__":
    main()
