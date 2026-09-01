#!/usr/bin/env python3
"""Multi-pass empirical tuning runner for VHS restoration modes.

Executes multiple passes of restoration across diverse tape captures and
restoration modes, collects objective quality metrics per pass, aggregates
statistics across tapes, and derives data-driven parameter recommendations.

Usage:
    python scripts/run_multi_pass_tuning.py \
        --input-dir /path/to/tapes \
        --modes vhs_native auto_ffmpeg_native arnndn_speech \
        --passes 2 \
        --min-duration 60 \
        --max-duration 120 \
        --clip-duration 60 \
        --report experiments/tuning_report.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.signal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.hardware import get_gpu_name
from modules.processing import process_hybrid_audio
from modules.utils import FFMPEG_BIN, FFPROBE_BIN, is_valid_video

_VIDEO_EXTS = {".mp4", ".mkv", ".mpg", ".mpeg", ".avi", ".mov", ".ts", ".m2ts"}


def _probe_duration(video_path):
    """Returns duration in minutes or None on failure."""
    cmd = [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    try:
        return float(subprocess.check_output(cmd, text=True, timeout=10).strip()) / 60.0
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


def _collect_eligible_tapes(input_dir, min_duration_min, max_duration_min):
    """Returns tapes whose durations fall within the inclusive minute window."""
    tapes = []
    for f in sorted(Path(input_dir).iterdir()):
        if f.suffix.lower() not in _VIDEO_EXTS:
            continue
        dur = _probe_duration(f)
        if dur is not None and min_duration_min <= dur <= max_duration_min:
            tapes.append((f, round(dur, 1)))
    return tapes


def _extract_sample(tape_path, out_path, start_sec, duration_sec):
    """Extracts a short video clip with pcm_f32le audio for fast evaluation."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(tape_path),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-c:a",
        "pcm_f32le",
        "-ar",
        "44100",
        str(out_path),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=60)
        return res.returncode == 0 and is_valid_video(out_path)
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        return False


def _extract_eval_wav(video_path, wav_path):
    """Extracts audio from a video container into a 32-bit PCM WAV."""
    cmd = [FFMPEG_BIN, "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_f32le", "-ar", "44100", str(wav_path)]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=60)
        return res.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        return False


def _compute_audio_features(mono, left, right, sr, crt_hz=15625.0):
    """Computes noise floor, SNR, CRT whistle ratio, rumble%, and balance dB."""

    from scripts.ia_benchmark_common import _compute_noise_floor_db, _compute_rumble_pct, _compute_spectral_ratio

    peak_db = float(20.0 * np.log10(float(np.max(np.abs(mono))) + 1e-9))
    rms_db = float(20.0 * np.log10(float(np.sqrt(np.mean(mono**2))) + 1e-9))
    nf_db = _compute_noise_floor_db(mono, sr, rms_db)
    f, psd = scipy.signal.welch(mono, sr, nperseg=4096)
    crt_ratio = _compute_spectral_ratio(psd, f, crt_hz)
    rumble_pct = _compute_rumble_pct(psd, f)
    rms_l = float(np.sqrt(np.mean(left**2)))
    rms_r = float(np.sqrt(np.mean(right**2)))
    bal_db = abs(float(20.0 * np.log10((rms_l + 1e-9) / (rms_r + 1e-9))))
    return {
        "noise_floor_db": round(nf_db, 2),
        "snr_db": round(peak_db - nf_db, 2),
        "crt_whistle_ratio": crt_ratio,
        "rumble_energy_pct": rumble_pct,
        "stereo_balance_db": round(bal_db, 2),
    }


def _measure_quality(video_path, work_dir, crt_hz=15625.0):
    """Returns quality metrics dict for a video file audio track."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return {}
    wav_path = work_dir / f"{video_path.stem}_mq.wav"
    if not _extract_eval_wav(video_path, wav_path):
        return {}
    try:
        try:
            data, sr = sf.read(str(wav_path), dtype="float32")
        except (OSError, RuntimeError):
            return {}
        if data.ndim > 1:
            mono = np.mean(data, axis=1)
            left, right = data[:, 0], data[:, 1]
        else:
            mono, left, right = data, data, data
        return _compute_audio_features(mono, left, right, sr, crt_hz=crt_hz)
    finally:
        wav_path.unlink(missing_ok=True)


def _mode_output_suffix(mode):
    """Returns the expected output filename suffix for a restoration mode."""
    from modules.processing import _get_output_suffix

    return _get_output_suffix(mode)


def _apply_mode_override(mode):
    """Patches all module-level PROCESS_MODE bindings for a tuning experiment."""
    from scripts.ia_benchmark_common import apply_mode_override

    apply_mode_override(mode)


def _run_restoration(sample_video, mode, output_dir, gpu_name):
    """Restores a sample clip with the given mode; returns output path or None."""
    _apply_mode_override(mode)
    suffix = _mode_output_suffix(mode)
    expected_out = output_dir / f"{sample_video.stem}{suffix}{sample_video.suffix}"
    if is_valid_video(expected_out):
        expected_out.unlink()
    success = process_hybrid_audio(sample_video, gpu_name, target_output_dir=output_dir)
    return expected_out if success and is_valid_video(expected_out) else None


def _run_single_pass(tape_path, mode, clip_dur, start_sec, output_dir, gpu_name, work_dir, crt_hz=15625.0):
    """Extracts sample, restores, measures quality delta. Returns dict or None."""
    sample = output_dir / f"{tape_path.stem}_sample_{clip_dur}s.mp4"
    if not is_valid_video(sample):
        if not _extract_sample(tape_path, sample, start_sec, clip_dur):
            return None
    orig_metrics = _measure_quality(sample, work_dir, crt_hz=crt_hz)
    restored = _run_restoration(sample, mode, output_dir, gpu_name)
    if not restored:
        return None
    rest_metrics = _measure_quality(restored, work_dir, crt_hz=crt_hz)
    return {"original": orig_metrics, "restored": rest_metrics}


def _compute_delta(original, restored):
    """Returns per-metric deltas and improvement scores."""
    if not original or not restored:
        return {}
    nf_delta = restored.get("noise_floor_db", 0) - original.get("noise_floor_db", 0)
    snr_delta = restored.get("snr_db", 0) - original.get("snr_db", 0)
    crt_attn = original.get("crt_whistle_ratio", 1) / max(restored.get("crt_whistle_ratio", 1), 1e-9)
    rumble_red = original.get("rumble_energy_pct", 0) - restored.get("rumble_energy_pct", 0)
    bal_fix = original.get("stereo_balance_db", 0) - restored.get("stereo_balance_db", 0)
    return {
        "nf_delta_db": round(nf_delta, 2),
        "snr_delta_db": round(snr_delta, 2),
        "crt_attenuation_x": round(crt_attn, 2),
        "rumble_reduction_pct": round(rumble_red, 3),
        "balance_correction_db": round(bal_fix, 2),
    }


def _aggregate_mode_results(deltas):
    """Aggregates per-tape deltas into mean/min/max statistics per metric."""
    import numpy as np

    keys = ["nf_delta_db", "snr_delta_db", "crt_attenuation_x", "rumble_reduction_pct", "balance_correction_db"]
    agg = {}
    for key in keys:
        vals = [d[key] for d in deltas if key in d]
        if not vals:
            continue
        agg[key] = {
            "mean": round(float(np.mean(vals)), 2),
            "min": round(float(np.min(vals)), 2),
            "max": round(float(np.max(vals)), 2),
            "n": len(vals),
        }
    return agg


def _rec_filter_checks(rec, nf_mean, crt_mean, rumble_mean):
    """Adds filter-related recommendations for afftdn_nf, notch and highpass."""
    if nf_mean > -3.0:
        rec["afftdn_nf"] = f"LOWER by 5 dB: noise floor delta only {nf_mean:+.1f} dB"
    if crt_mean < 5.0:
        rec["notch_freq"] = f"CHECK: CRT attenuation only {crt_mean:.1f}x; verify notch_freq={crt_hz}"
    if rumble_mean < 0.05:
        rec["highpass_freq"] = f"RAISE by 10 Hz: rumble reduction only {rumble_mean:.3f}%"


def _rec_snr_status(rec, snr_mean):
    """Adds SNR-related recommendations or positive status message."""
    if snr_mean < -2.0:
        rec["afftdn_nr"] = f"REDUCE: mean SNR delta {snr_mean:+.1f} dB suggests over-processing"
    if snr_mean > 1.0:
        rec["status"] = f"GOOD: mean SNR improvement +{snr_mean:.1f} dB across tapes"


def _rec_for_mode(stats):
    """Builds a recommendation dict for one mode from its aggregate stats."""
    rec = {}
    snr_mean = stats.get("snr_delta_db", {}).get("mean", 0)
    nf_mean = stats.get("nf_delta_db", {}).get("mean", 0)
    crt_mean = stats.get("crt_attenuation_x", {}).get("mean", 1)
    rumble_mean = stats.get("rumble_reduction_pct", {}).get("mean", 0)
    _rec_snr_status(rec, snr_mean)
    _rec_filter_checks(rec, nf_mean, crt_mean, rumble_mean)
    return rec


def _recommend_parameters(pass_aggregates):
    """Derives config parameter recommendations from aggregated multi-pass data."""
    return {mode: _rec_for_mode(stats) for mode, stats in pass_aggregates.items()}


def _save_report(report_data, report_path):
    """Serializes tuning report to JSON."""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report_data, fh, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")


def _print_pass_result(mode, tape_name, pass_num, delta):
    """Prints a single tape/mode/pass result line."""
    nf = delta.get("nf_delta_db", 0)
    snr = delta.get("snr_delta_db", 0)
    crt = delta.get("crt_attenuation_x", 1)
    rumble = delta.get("rumble_reduction_pct", 0)
    bal = delta.get("balance_correction_db", 0)
    print(f"  P{pass_num} {mode:<22} {tape_name:<40} " f"NF{nf:+.1f} SNR{snr:+.1f} CRT{crt:.1f}x Rum{rumble:.2f}% Bal{bal:.1f}")


def _print_aggregate_summary(aggregated, recommendations):
    """Prints per-mode aggregate statistics and recommendations."""
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS BY MODE")
    print("=" * 80)
    for mode, stats in aggregated.items():
        snr_s = stats.get("snr_delta_db", {})
        nf_s = stats.get("nf_delta_db", {})
        crt_s = stats.get("crt_attenuation_x", {})
        n = snr_s.get("n", 0)
        print(
            f"\n  {mode:<25} ({n} tape-passes)"
            f"\n    NF  delta: mean={nf_s.get('mean', 0):+.2f} dB "
            f"[{nf_s.get('min', 0):+.2f}, {nf_s.get('max', 0):+.2f}]"
            f"\n    SNR delta: mean={snr_s.get('mean', 0):+.2f} dB "
            f"[{snr_s.get('min', 0):+.2f}, {snr_s.get('max', 0):+.2f}]"
            f"\n    CRT atten: mean={crt_s.get('mean', 1):.2f}x"
        )
        for param, msg in recommendations.get(mode, {}).items():
            print(f"    [REC] {param}: {msg}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing VHS tape files")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["vhs_native", "auto_ffmpeg_native", "arnndn_speech"],
        help="Restoration modes to evaluate",
    )
    parser.add_argument("--passes", type=int, default=2, help="Number of tuning passes")
    parser.add_argument("--min-duration", type=float, default=0.0, help="Min tape duration in minutes, inclusive")
    parser.add_argument("--max-duration", type=float, default=60.0, help="Max tape duration in minutes, inclusive")
    parser.add_argument("--clip-duration", type=int, default=60, help="Clip length in seconds")
    parser.add_argument("--start", type=int, default=30, help="Clip start offset in seconds")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for clips/restored files")
    parser.add_argument("--report", type=Path, default=None, help="Path to save JSON tuning report")
    parser.add_argument("--device", type=str, default=None, help="Processing device override")
    parser.add_argument("--crt-hz", type=float, default=15625.0, help="CRT line whistle target frequency in Hz")
    args = parser.parse_args()
    if args.min_duration < 0 or args.max_duration < args.min_duration:
        parser.error("duration bounds require 0 <= --min-duration <= --max-duration")
    return args


def _resolve_dirs(args):
    """Resolves output and work directories from args."""
    output_dir = args.output_dir or Path(__file__).resolve().parent.parent / "experiments" / "multi_pass_tuning"
    report_path = args.report or output_dir / "tuning_report.json"
    return output_dir, output_dir / "_eval_work", report_path


def _run_pass_for_mode(
    mode, tapes, pass_num, clip_dur, start, output_dir, gpu_name, work_dir, all_results, pass_aggregates, crt_hz=15625.0
):
    """Runs one mode over all tapes for a single pass, recording results in-place."""
    print(f"\n  Mode: {mode}")
    for tape_path, _ in tapes:
        result = _run_single_pass(tape_path, mode, clip_dur, start, output_dir, gpu_name, work_dir, crt_hz=crt_hz)
        if not result:
            print(f"  SKIP {tape_path.name} [{mode}] - restoration failed")
            continue
        delta = _compute_delta(result["original"], result["restored"])
        _print_pass_result(mode, tape_path.name, pass_num, delta)
        all_results[f"p{pass_num}|{tape_path.name}|{mode}"] = {"metrics": result, "delta": delta}
        pass_aggregates[mode].append(delta)


def _run_all_passes(tapes, modes, passes, clip_dur, start, output_dir, gpu_name, work_dir, crt_hz=15625.0):
    """Executes all tuning passes; returns (all_results, pass_aggregates)."""
    all_results = {}
    pass_aggregates = {mode: [] for mode in modes}
    for pass_num in range(1, passes + 1):
        print(f"\n{'=' * 80}\nPASS {pass_num} / {passes}\n{'=' * 80}")
        for mode in modes:
            _run_pass_for_mode(
                mode, tapes, pass_num, clip_dur, start, output_dir, gpu_name, work_dir, all_results, pass_aggregates, crt_hz=crt_hz
            )
    return all_results, pass_aggregates


def main():
    args = _parse_args()
    output_dir, work_dir, report_path = _resolve_dirs(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = args.device or get_gpu_name()
    print(f"GPU: {gpu_name} | Passes: {args.passes} | Modes: {args.modes}")
    print(f"Probing tapes in: {args.input_dir}")

    tapes = _collect_eligible_tapes(args.input_dir, args.min_duration, args.max_duration)
    print(f"Eligible tapes ({args.min_duration:g}m to {args.max_duration:g}m, inclusive): {len(tapes)}")
    for tp, dur in tapes:
        print(f"  {tp.name:<55} {dur:.1f}m")

    all_results, pass_aggregates = _run_all_passes(
        tapes,
        args.modes,
        args.passes,
        args.clip_duration,
        args.start,
        output_dir,
        gpu_name,
        work_dir,
        crt_hz=args.crt_hz,
    )

    aggregated = {mode: _aggregate_mode_results(pass_aggregates[mode]) for mode in args.modes}
    recommendations = _recommend_parameters(aggregated)
    _print_aggregate_summary(aggregated, recommendations)

    report = {
        "passes": args.passes,
        "modes": args.modes,
        "duration_window_minutes": {"minimum": args.min_duration, "maximum": args.max_duration},
        "tapes_evaluated": [str(t) for t, _ in tapes],
        "results": all_results,
        "aggregated": aggregated,
        "recommendations": recommendations,
    }
    _save_report(report, report_path)


if __name__ == "__main__":
    main()
