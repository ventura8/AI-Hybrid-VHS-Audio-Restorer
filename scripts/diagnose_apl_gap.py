#!/usr/bin/env python3
"""Attributes the auto_pure_linear vs cathar quality gap to specific detector behaviour.

Read-only: this measures what the current detectors decide on a real corpus, so each
proposed fix can be sized before any of it is written. It changes no restoration code and
writes only its own report.

The benchmark in docs/cathar_vs_auto_pure_linear_1000_benchmark.md shows auto_pure_linear
losing to cathar on CRT attenuation (26x vs 96x), mains attenuation (1.85x vs 4.66x) and,
on NTSC, on noise reduction (+0.40 dB vs +5.33 dB). Four candidate causes live in the
detectors rather than in the filters, and this script measures all four:

1. recall  -- both detectors read only signal_data[:32768], the first ~0.74 s at 44.1 kHz
2. cutoff  -- below a prominence threshold they return 0.0 and NO notch is applied at all
3. nominal -- the measured peak is discarded in favour of a nominal constant
4. model   -- the deep UVR-DeNoise model may be effectively unreachable
"""

import argparse
import json
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.filters import (
    CRT_MIN_PROMINENCE,
    CRT_SEARCH_HIGH_HZ,
    CRT_SEARCH_LOW_HZ,
    MAINS_CANDIDATES_BY_LINE_RATE,
    MAINS_CANDIDATES_DEFAULT,
    _detect_crt_flyback_notch,
    _detect_mains_buzz_notch,
    _read_audio_for_analysis,
)
from scripts.benchmark_ia_corpus_batch import _extract_audio_pcm

# The window the current detectors use, and the cap on how many such windows the
# whole-file comparison walks. 120 windows of 32768 covers ~89 s at 44.1 kHz, well beyond
# the 15 s corpus clips, so the cap only matters for long captures.
DETECTOR_WINDOW = 32768
MAX_WINDOWS = 120


def _band_peak(mag, freqs, low_hz, high_hz):
    """Returns (peak_hz, prominence) for the strongest bin in a band, against its median."""
    band = np.flatnonzero((freqs >= low_hz) & (freqs <= high_hz))
    if len(band) == 0:
        return 0.0, 0.0
    peak = int(band[int(np.argmax(mag[band]))])
    background = float(np.median(mag[band])) + 1e-12
    return float(freqs[peak]), float(mag[peak]) / background


def _spectrum(chunk, sr, window=False):
    """Returns (magnitude, freqs); Hann windowing optional to expose leakage effects."""
    data = chunk * np.hanning(len(chunk)) if window else chunk
    return np.abs(np.fft.rfft(data)), np.fft.rfftfreq(len(chunk), 1.0 / sr)


def _iter_windows(mono, sr):
    """Yields successive detector-sized windows across the whole file."""
    del sr
    count = 0
    for start in range(0, max(1, len(mono) - DETECTOR_WINDOW + 1), DETECTOR_WINDOW):
        stop = start + DETECTOR_WINDOW
        yield mono[start:stop]
        count += 1
        if count >= MAX_WINDOWS:
            return


def _crt_head_vs_whole(mono, sr):
    """Measures the CRT tone in the first window only, then across every window."""
    head_mag, head_freqs = _spectrum(mono[:DETECTOR_WINDOW], sr)
    head_hz, head_prom = _band_peak(head_mag, head_freqs, CRT_SEARCH_LOW_HZ, CRT_SEARCH_HIGH_HZ)

    # Per-window, so the persistent tone can be told apart from a one-off noise peak: a
    # CRT whistle is recorded continuously and shows up in nearly every window, whereas
    # the maximum across windows rises for transient content just as readily.
    per_window = []
    best_hz, best_prom = head_hz, head_prom
    for chunk in _iter_windows(mono, sr):
        if len(chunk) < DETECTOR_WINDOW:
            continue
        mag, freqs = _spectrum(chunk, sr)
        peak_hz, prom = _band_peak(mag, freqs, CRT_SEARCH_LOW_HZ, CRT_SEARCH_HIGH_HZ)
        per_window.append((peak_hz, prom))
        if prom > best_prom:
            best_hz, best_prom = peak_hz, prom

    proms = [p for _, p in per_window]
    median_prom = float(np.median(proms)) if proms else head_prom
    median_hz = float(np.median([hz for hz, p in per_window if p >= median_prom])) if proms else head_hz
    return {
        "head_hz": head_hz,
        "head_prominence": head_prom,
        "whole_hz": best_hz,
        "whole_prominence": best_prom,
        "median_prominence": median_prom,
        "median_hz": median_hz,
        "windows": len(per_window),
    }


def _local_ratio(mag, freqs, target_hz, offset_bins):
    """Peak-to-neighbour ratio at a target frequency, with a configurable reference offset.

    The shipped detector uses offset_bins=2 on an unwindowed spectrum. A rectangular
    window smears a strong tone into its immediate neighbours, so the reference bins rise
    with the peak and the ratio understates a tone that is plainly present.
    """
    idx = int(np.argmin(np.abs(freqs - target_hz)))
    if not (offset_bins <= idx < len(mag) - offset_bins):
        return 0.0
    neighbours = float(mag[idx - offset_bins] + mag[idx + offset_bins]) / 2.0 + 1e-9
    return float(mag[idx]) / neighbours


def _mains_variants(mono, sr, line_rate_hz):
    """Compares the shipped mains ratio against windowed and wider-reference variants."""
    candidates = MAINS_CANDIDATES_BY_LINE_RATE.get(line_rate_hz, MAINS_CANDIDATES_DEFAULT)
    head = mono[:DETECTOR_WINDOW]
    plain_mag, plain_freqs = _spectrum(head, sr)
    hann_mag, hann_freqs = _spectrum(head, sr, window=True)

    best = {"shipped": 0.0, "hann": 0.0, "hann_wide": 0.0, "whole_file": 0.0}
    for target in candidates:
        best["shipped"] = max(best["shipped"], _local_ratio(plain_mag, plain_freqs, target, 2))
        best["hann"] = max(best["hann"], _local_ratio(hann_mag, hann_freqs, target, 2))
        best["hann_wide"] = max(best["hann_wide"], _local_ratio(hann_mag, hann_freqs, target, 8))

    per_window = []
    for chunk in _iter_windows(mono, sr):
        if len(chunk) < DETECTOR_WINDOW:
            continue
        mag, freqs = _spectrum(chunk, sr, window=True)
        ratios = [_local_ratio(mag, freqs, target, 8) for target in candidates]
        per_window.append(max(ratios) if ratios else 0.0)
        best["whole_file"] = max(best["whole_file"], max(ratios) if ratios else 0.0)
    # Mains hum is continuous; a median across windows rejects the transient bass peaks
    # that inflate a whole-file maximum.
    best["whole_file_median"] = float(np.median(per_window)) if per_window else 0.0
    return best


def _measure_clip(wav_path, nominal_crt_hz):
    """Runs the shipped detectors and the comparison measurements on one clip."""
    mono, sr = _read_audio_for_analysis(wav_path)
    if mono is None or sr is None or len(mono) < DETECTOR_WINDOW:
        return None

    crt_shipped = _detect_crt_flyback_notch(mono, sr)
    mains_shipped = _detect_mains_buzz_notch(mono, sr, crt_shipped)
    crt = _crt_head_vs_whole(mono, sr)
    return {
        "crt_shipped_hz": crt_shipped,
        "mains_shipped_hz": mains_shipped,
        "crt": crt,
        "crt_offset_from_nominal_hz": round(crt["whole_hz"] - nominal_crt_hz, 2) if crt["whole_prominence"] > 0 else None,
        "mains_ratios": _mains_variants(mono, sr, crt_shipped),
    }


def _scan_strategy(wav_path):
    """Returns the strategy fields that drive the auto_pure_linear chain."""
    from modules.auto_scanner import scan_and_decide_restoration_strategy

    strategy = scan_and_decide_restoration_strategy(wav_path, executed_mode="auto_pure_linear")
    profile = strategy.get("profile", {})
    precondition = strategy.get("precondition_filters", {})
    return {
        "denoise_model": strategy.get("denoise_model"),
        "noise_floor_db": profile.get("noise_floor_db"),
        "highpass_hz": precondition.get("highpass_hz"),
        "notch_hz": precondition.get("notch_hz"),
        "crt_notch_hz": precondition.get("crt_notch_hz"),
    }


def _load_catalog(catalog_path, corpus_dir):
    """Returns [(clip_path, region, nominal_crt_hz)] from the corpus catalog."""
    records = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries = []
    for record in records:
        clip = corpus_dir / record["file"]
        if clip.exists():
            entries.append((clip, record.get("region", "unknown"), float(record.get("crt_hz", 15625.0))))
    return entries


def _percentiles(values):
    """Returns a compact distribution summary, or None when there is nothing to describe."""
    usable = sorted(v for v in values if v is not None)
    if not usable:
        return None
    return {
        "n": len(usable),
        "min": round(usable[0], 3),
        "p50": round(statistics.median(usable), 3),
        "p90": round(usable[int(len(usable) * 0.9) - 1] if len(usable) > 1 else usable[0], 3),
        "max": round(usable[-1], 3),
    }


def _summarise(rows):
    """Aggregates per-clip rows into the attribution counts the plan asks for."""
    total = len(rows)
    crt_missing = sum(1 for r in rows if not r["crt_shipped_hz"])
    mains_missing = sum(1 for r in rows if not r["mains_shipped_hz"])
    lite = sum(1 for r in rows if (r.get("denoise_model") or "").startswith("UVR-DeNoise-Lite"))
    rescued = sum(1 for r in rows if not r["crt_shipped_hz"] and r["crt"]["whole_prominence"] >= CRT_MIN_PROMINENCE)
    return {
        "clips": total,
        "crt_notch_not_applied": crt_missing,
        "mains_notch_not_applied": mains_missing,
        "lite_model_chosen": lite,
        "crt_recoverable_by_whole_file_scan": rescued,
        "crt_prominence_head": _percentiles([r["crt"]["head_prominence"] for r in rows]),
        "crt_prominence_whole": _percentiles([r["crt"]["whole_prominence"] for r in rows]),
        "crt_prominence_median": _percentiles([r["crt"]["median_prominence"] for r in rows]),
        "crt_offset_from_nominal_hz": _percentiles(
            [abs(r["crt_offset_from_nominal_hz"]) for r in rows if r["crt_offset_from_nominal_hz"] is not None]
        ),
        "noise_floor_db": _percentiles([r.get("noise_floor_db") for r in rows]),
        "mains_ratio_shipped": _percentiles([r["mains_ratios"]["shipped"] for r in rows]),
        "mains_ratio_hann": _percentiles([r["mains_ratios"]["hann"] for r in rows]),
        "mains_ratio_hann_wide": _percentiles([r["mains_ratios"]["hann_wide"] for r in rows]),
        "mains_ratio_whole_file": _percentiles([r["mains_ratios"]["whole_file"] for r in rows]),
        "mains_ratio_whole_median": _percentiles([r["mains_ratios"]["whole_file_median"] for r in rows]),
        "models": dict(Counter(r.get("denoise_model") for r in rows)),
    }


def _process_clip(clip, region, nominal_crt, work_dir, with_scan):
    """Extracts one clip's audio and returns its measurement row, or None if unusable."""
    wav = work_dir / f"{clip.stem}.wav"
    if not _extract_audio_pcm(clip, wav):
        return None
    try:
        measured = _measure_clip(wav, nominal_crt)
        if measured is None:
            return None
        measured.update({"clip": clip.name, "region": region, "nominal_crt_hz": nominal_crt})
        if with_scan:
            measured.update(_scan_strategy(wav))
        return measured
    finally:
        wav.unlink(missing_ok=True)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("experiments/ia_corpus_1000"))
    parser.add_argument("--catalog", type=Path, default=None, help="Defaults to <corpus-dir>/catalog_1000.json")
    parser.add_argument("--output", type=Path, default=Path("experiments/apl_gap_diagnosis.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-scan", action="store_true", help="Skip the full scanner (faster, omits model choice)")
    return parser.parse_args()


def main():
    """Measures the corpus and writes the attribution report."""
    args = _parse_args()
    catalog = args.catalog or (args.corpus_dir / "catalog_1000.json")
    entries = _load_catalog(catalog, args.corpus_dir)
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        raise SystemExit(f"No clips found from {catalog} under {args.corpus_dir}")

    rows = []
    with tempfile.TemporaryDirectory(prefix="apl_gap_") as tmp:
        work_dir = Path(tmp)
        for index, (clip, region, nominal_crt) in enumerate(entries, start=1):
            row = _process_clip(clip, region, nominal_crt, work_dir, not args.no_scan)
            if row is not None:
                rows.append(row)
            if index % 10 == 0 or index == len(entries):
                sys.stdout.write(f"  [{index}/{len(entries)}] measured {len(rows)} clips\n")
                sys.stdout.flush()

    if not rows:
        raise SystemExit("No clip could be measured; the corpus audio is unreadable.")

    report = {"overall": _summarise(rows), "by_region": {}, "clips": rows}
    for region in sorted({r["region"] for r in rows}):
        report["by_region"][region] = _summarise([r for r in rows if r["region"] == region])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(f"\n{json.dumps(report['overall'], indent=2)}\n\nWrote {args.output}\n")


if __name__ == "__main__":
    main()
