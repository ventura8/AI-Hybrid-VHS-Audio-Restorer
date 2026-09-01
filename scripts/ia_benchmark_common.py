#!/usr/bin/env python3
"""Shared acoustic metrics and restoration evaluation helpers for IA benchmarks."""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.signal
import soundfile as sf

# Ensure modules package can be resolved
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modules.config as config_mod
import modules.processing as processing_mod
from modules.processing import _get_output_suffix, process_hybrid_audio
from modules.utils import is_valid_video


def _compute_noise_floor_db(mono: np.ndarray, sr: int, fallback_rms_db: float) -> float:
    """Estimates quiet noise floor via 10th percentile windowed RMS."""
    hop = sr // 10
    frame_len = sr // 5
    if len(mono) <= frame_len:
        return fallback_rms_db
    frames = [mono[i : i + frame_len] for i in range(0, len(mono) - frame_len, hop)]
    if not frames:
        return fallback_rms_db
    frame_rms = [float(np.sqrt(np.mean(f**2))) for f in frames]
    p10 = float(np.percentile(frame_rms, 10))
    return float(20.0 * np.log10(p10 + 1e-9))


def _compute_spectral_ratio(psd: np.ndarray, f: np.ndarray, target_hz: float) -> float:
    """Computes peak energy ratio at target frequency vs local median background."""
    if target_hz >= float(f[-1]):
        return 1.0
    idx = int(np.argmin(np.abs(f - target_hz)))
    peak_energy = float(psd[idx])
    width = min(20, idx - 1, len(psd) - idx - 2)
    if width <= 0:
        return round(peak_energy / 1e-12, 2)
    left = psd[idx - width : idx - 1]
    right = psd[idx + 2 : idx + 2 + width]
    background = np.concatenate((left, right))
    bg = float(np.median(background)) + 1e-12 if background.size else 1e-12
    return round(peak_energy / bg, 2)


def _compute_rumble_pct(psd: np.ndarray, f: np.ndarray, cutoff_hz: float = 100.0) -> float:
    """Calculates sub-cutoff rumble energy percentage of total spectrum."""
    cutoff_idx = int(np.argmin(np.abs(f - cutoff_hz)))
    rumble_sum = float(np.sum(psd[:cutoff_idx]))
    total_sum = float(np.sum(psd)) + 1e-12
    return round((rumble_sum / total_sum) * 100.0, 2)


def _split_channels(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits audio data into mono, left, and right channels."""
    if data.ndim > 1:
        mono = np.mean(data, axis=1)
        left = data[:, 0]
        right = data[:, 1]
    else:
        mono = data
        left = data
        right = data
    return mono, left, right


def _measure_audio_metrics(wav_path: Path, crt_hz: float = 15625.0, mains_hz: float = 50.0) -> Dict[str, float]:
    """Computes comprehensive acoustic and distortion metrics from a WAV file."""
    data, sr = sf.read(str(wav_path), dtype="float32")
    mono, left, right = _split_channels(data)

    peak = float(np.max(np.abs(data)))
    peak_db = round(float(20.0 * np.log10(peak + 1e-9)), 2)
    rms = float(np.sqrt(np.mean(data**2)))
    rms_db = round(float(20.0 * np.log10(rms + 1e-9)), 2)
    nf_db = round(_compute_noise_floor_db(mono, sr, rms_db), 2)
    snr_db = round(peak_db - nf_db, 2)

    f, psd = scipy.signal.welch(mono, sr, nperseg=8192)
    crt_ratio = _compute_spectral_ratio(psd, f, crt_hz)
    mains_ratio = _compute_spectral_ratio(psd, f, mains_hz)
    rumble_pct = _compute_rumble_pct(psd, f, 100.0)

    rms_l = float(np.sqrt(np.mean(left**2)))
    rms_r = float(np.sqrt(np.mean(right**2)))
    bal_diff = round(abs(float(20.0 * np.log10((rms_l + 1e-9) / (rms_r + 1e-9)))), 2)

    return {
        "peak_db": peak_db,
        "rms_db": rms_db,
        "noise_floor_db": nf_db,
        "snr_db": snr_db,
        "crt_whistle_ratio": crt_ratio,
        "mains_hum_ratio": mains_ratio,
        "rumble_energy_pct": rumble_pct,
        "stereo_balance_diff_db": bal_diff,
    }


def _calculate_deltas(orig: Dict[str, float], rest: Dict[str, float]) -> Dict[str, float]:
    """Calculates improvement deltas between original and restored metrics."""
    nf_red = round(orig["noise_floor_db"] - rest["noise_floor_db"], 2)
    snr_gain = round(rest["snr_db"] - orig["snr_db"], 2)
    crt_att = round(orig["crt_whistle_ratio"] / max(rest["crt_whistle_ratio"], 0.05), 2)
    mains_att = round(orig["mains_hum_ratio"] / max(rest["mains_hum_ratio"], 0.05), 2)
    rumble_red = round(orig["rumble_energy_pct"] - rest["rumble_energy_pct"], 2)
    bal_imp = round(orig["stereo_balance_diff_db"] - rest["stereo_balance_diff_db"], 2)

    return {
        "noise_reduction_db": nf_red,
        "snr_gain_db": snr_gain,
        "crt_attenuation_ratio": crt_att,
        "mains_attenuation_ratio": mains_att,
        "rumble_reduction_pct": rumble_red,
        "balance_improvement_db": bal_imp,
    }


def _run_mode_restoration(video_path: Path, mode: str, output_dir: Path, gpu_name: str) -> Optional[Path]:
    """Runs a single restoration mode on the input video clip."""
    if mode not in config_mod.VALID_PROCESS_MODES:
        return None
    suffix = _get_output_suffix(mode)
    expected_out = output_dir / f"{video_path.stem}{suffix}{video_path.suffix}"
    if is_valid_video(expected_out):
        return expected_out

    apply_mode_override(mode)

    ok = process_hybrid_audio(video_path, gpu_name, target_output_dir=output_dir)
    return expected_out if ok and is_valid_video(expected_out) else None


def apply_mode_override(mode: str) -> None:
    """Applies a mode consistently to processing and configuration bindings."""
    processing_mod.PROCESS_MODE = mode
    config_mod.CONFIG["process_mode"] = mode
    config_mod.PROCESS_MODE = mode
