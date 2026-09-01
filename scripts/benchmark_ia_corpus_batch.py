#!/usr/bin/env python3
"""High-throughput parallel batch benchmark runner for massive VHS corpora.

Evaluates cathar and auto_pure_linear on large datasets with atomic checkpointing.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.utils import FFMPEG_BIN, is_valid_audio
from scripts.benchmark_ia_corpus import _aggregate_by_genre, _aggregate_by_region, _average_metrics
from scripts.ia_benchmark_common import (
    _calculate_deltas,
    _measure_audio_metrics,
    _run_mode_restoration,
)


def _extract_audio_pcm(video_path: Path, temp_wav: Path) -> bool:
    """Extracts raw audio as 32-bit float PCM."""
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
        return res.returncode == 0 and is_valid_audio(temp_wav)
    except (subprocess.SubprocessError, OSError):
        return False


def _remove_file(path: Path) -> None:
    """Remove a temporary file without masking a benchmark result."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _audio_metrics(video_path: Path, wav_path: Path, crt_hz: float, mains_hz: float) -> Optional[Dict[str, float]]:
    """Extract and measure a video's audio, reusing an existing valid WAV."""
    if not is_valid_audio(wav_path) and not _extract_audio_pcm(video_path, wav_path):
        return None
    return _measure_audio_metrics(wav_path, crt_hz=crt_hz, mains_hz=mains_hz)


def _benchmark_mode(
    clip_path: Path,
    mode: str,
    output_dir: Path,
    eval_dir: Path,
    gpu_name: str,
    original_metrics: Dict[str, float],
    crt_hz: float,
    mains_hz: float,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Restore one clip in one mode and calculate its metrics and deltas."""
    started = time.time()
    restored_video = _run_mode_restoration(clip_path, mode, output_dir, gpu_name)
    if not restored_video:
        return None
    restored_wav = eval_dir / f"{restored_video.stem}.wav"
    try:
        metrics = _audio_metrics(restored_video, restored_wav, crt_hz, mains_hz)
        if metrics is None:
            return None
    finally:
        _remove_file(restored_wav)
    deltas = _calculate_deltas(original_metrics, metrics)
    deltas["latency_sec"] = round(time.time() - started, 2)
    return {"metrics": metrics, "deltas": deltas}


def benchmark_clip(
    clip_path: Path,
    meta: Dict[str, Any],
    modes: List[str],
    output_dir: Path,
    eval_dir: Path,
    gpu_name: str,
) -> Optional[Dict[str, Any]]:
    """Benchmark all requested modes for one catalogued clip."""
    crt_hz = float(meta.get("crt_hz", 15625.0))
    mains_hz = float(meta.get("notch_hz", 50.0))
    orig_wav = eval_dir / f"{clip_path.stem}_orig.wav"
    try:
        orig_metrics = _audio_metrics(clip_path, orig_wav, crt_hz, mains_hz)
        if orig_metrics is None:
            return None
        mode_results = {
            mode: result
            for mode in modes
            if (result := _benchmark_mode(clip_path, mode, output_dir, eval_dir, gpu_name, orig_metrics, crt_hz, mains_hz))
        }
    finally:
        _remove_file(orig_wav)

    return {
        "identifier": meta.get("identifier", clip_path.stem),
        "title": meta.get("title", clip_path.stem),
        "genre": meta.get("genre", "general"),
        "region": meta.get("region", "europe"),
        "standard": meta.get("standard", "PAL"),
        "original_metrics": orig_metrics,
        "restored": mode_results,
    }


def aggregate_results(results: List[Dict[str, Any]], modes: List[str]) -> Dict[str, Any]:
    summary = {"overall": {}, "by_region": {}, "by_genre": {}}
    for m in modes:
        deltas_m = [r["restored"][m]["deltas"] for r in results if m in r.get("restored", {})]
        if not deltas_m:
            continue
        summary["overall"][m] = _average_metrics(deltas_m)
        summary["by_region"][m] = _aggregate_by_region(results, m)
        summary["by_genre"][m] = _aggregate_by_genre(results, m)

    return summary


def _parse_arguments():
    """Parse command-line arguments for the batch benchmark."""
    parser = argparse.ArgumentParser(description="Massive parallel benchmark runner for VHS corpora")
    parser.add_argument("--corpus-dir", type=Path, default=Path("experiments/ia_corpus_1000"))
    parser.add_argument("--catalog", type=Path, default=Path("experiments/ia_corpus_1000/catalog_1000.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/benchmark_1000_results"))
    parser.add_argument("--modes", nargs="+", default=["cathar", "auto_pure_linear"])
    parser.add_argument("--gpu", default="CPU")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _load_catalog(catalog_path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Load a catalog and discard entries that cannot identify a fixture."""
    with open(catalog_path, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    if limit:
        catalog = catalog[:limit]
    return [item for item in catalog if item.get("identifier")]


def _load_checkpoint(path: Path) -> List[Dict[str, Any]]:
    """Load prior batch results when a usable checkpoint exists."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            results = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    sys.stdout.write(f"Loaded {len(results)} existing benchmark results from checkpoint.\n")
    return results


def _save_json(path: Path, content: Any) -> None:
    """Atomically save JSON content."""
    temporary_path = path.with_suffix(".tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2)
    temporary_path.replace(path)


def _ready_clip_path(corpus_dir: Path, item: Dict[str, Any], processed_ids: set[str]) -> Optional[Path]:
    """Return an eligible, available fixture path, or None when it should be skipped."""
    identifier = item["identifier"]
    relative_file = item.get("file")
    if identifier in processed_ids or not relative_file:
        return None
    clip_path = corpus_dir / relative_file
    return clip_path if clip_path.exists() else None


def _run_clip_benchmark(args, item, clip_path, eval_dir) -> Optional[Dict[str, Any]]:
    """Run one clip, reporting expected operational failures without ending the batch."""
    identifier = item["identifier"]
    try:
        return benchmark_clip(clip_path, item, args.modes, args.output_dir, eval_dir, args.gpu)
    except (OSError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"  [Warning] Failed benchmarking clip {identifier}: {exc}\n")
        return None


def _record_result(results, result, identifier, processed_ids, checkpoint_file, index, total) -> None:
    """Record a successful result and persist each checkpoint boundary."""
    if result is None:
        return
    results.append(result)
    processed_ids.add(identifier)
    if len(results) % 5 == 0 or index == total:
        _save_json(checkpoint_file, results)


def _benchmark_catalog(args, catalog, results, eval_dir, checkpoint_file) -> None:
    """Benchmark each available unprocessed catalog entry and checkpoint results."""
    processed_ids = {result["identifier"] for result in results}
    total = len(catalog)
    for index, item in enumerate(catalog, 1):
        identifier = item["identifier"]
        clip_path = _ready_clip_path(args.corpus_dir, item, processed_ids)
        if clip_path is None:
            continue
        sys.stdout.write(f"[{index}/{total}] Benchmarking {identifier}...\n")
        result = _run_clip_benchmark(args, item, clip_path, eval_dir)
        _record_result(results, result, identifier, processed_ids, checkpoint_file, index, total)


def main():
    """Run the checkpointed batch benchmark and write its final report."""
    args = _parse_arguments()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = args.output_dir / "eval_tmp"
    eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = args.output_dir / "checkpoint.json"

    catalog = _load_catalog(args.catalog, args.limit)
    results = _load_checkpoint(checkpoint_file)
    _benchmark_catalog(args, catalog, results, eval_dir, checkpoint_file)

    summary = aggregate_results(results, args.modes)
    final_report = {"summary": summary, "results": results}

    report_json = args.output_dir / "report_1000.json"
    _save_json(report_json, final_report)

    sys.stdout.write(f"\nBenchmark completed! Processed {len(results)} clips. Report saved to {report_json}\n")


if __name__ == "__main__":
    main()
