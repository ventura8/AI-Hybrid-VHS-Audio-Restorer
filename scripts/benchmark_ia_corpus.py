#!/usr/bin/env python3
"""Benchmark audio restoration modes across Internet Archive VHS corpus."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure modules package can be resolved
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.hardware import get_gpu_name
from modules.utils import FFMPEG_BIN, is_valid_audio, is_valid_video
from scripts.curate_ia_corpus import _sanitize_slug as _plain_slug
from scripts.curate_massive_ia_corpus import _sanitize_slug as _massive_slug
from scripts.ia_benchmark_common import (
    _aggregate_by_genre,
    _aggregate_by_region,
    _average_metrics,
    _calculate_deltas,
    _compute_noise_floor_db,
    _compute_rumble_pct,
    _compute_spectral_ratio,
    _measure_audio_metrics,
    _run_mode_restoration,
    _split_channels,
)

analyze_audio = _measure_audio_metrics

__all__ = [
    "_calculate_deltas",
    "_compute_noise_floor_db",
    "_compute_rumble_pct",
    "_compute_spectral_ratio",
    "_measure_audio_metrics",
    "_run_mode_restoration",
    "_split_channels",
    "analyze_audio",
]


def _extract_pcm(media_path: Path, temp_wav: Path) -> bool:
    """Extracts 32-bit float PCM audio from video or audio container."""
    temp_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-acodec",
        "pcm_f32le",
        "-ar",
        "44100",
        str(temp_wav),
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=60,
        )
        return res.returncode == 0 and is_valid_audio(temp_wav)
    except (subprocess.SubprocessError, OSError):
        return False


def _unlink_eval_wav(wav_path: Path) -> None:
    """Safely removes an extracted evaluation WAV file."""
    if wav_path.exists():
        try:
            wav_path.unlink()
        except OSError:
            pass


def _eval_single_mode(
    clip_path: Path,
    mode: str,
    orig_metrics: Dict[str, float],
    out_dir: Path,
    eval_dir: Path,
    gpu_name: str,
    crt_hz: float,
    mains_hz: float,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Runs a single mode restoration and computes metric deltas."""
    rest_vid = _run_mode_restoration(clip_path, mode, out_dir, gpu_name)
    if not rest_vid:
        return None
    rest_wav = eval_dir / f"{rest_vid.stem}.wav"
    if not _extract_pcm(rest_vid, rest_wav):
        return None
    try:
        rest_metrics = analyze_audio(rest_wav, crt_hz, mains_hz)
        deltas = _calculate_deltas(orig_metrics, rest_metrics)
        return mode, {"metrics": rest_metrics, "deltas": deltas}
    finally:
        _unlink_eval_wav(rest_wav)


def _eval_single_clip(
    clip_path: Path,
    meta: Dict[str, Any],
    modes: List[str],
    out_dir: Path,
    eval_dir: Path,
    gpu_name: str,
) -> Optional[Dict[str, Any]]:
    """Restores and evaluates a single clip across configured modes."""
    defaults = _default_clip_meta(clip_path.stem, meta.get("region") == "america")
    crt_hz = float(meta.get("crt_hz", defaults["crt_hz"]))
    mains_hz = float(meta.get("notch_hz", defaults["notch_hz"]))

    orig_wav = eval_dir / f"{clip_path.stem}_orig.wav"
    if not _extract_pcm(clip_path, orig_wav):
        return None

    try:
        orig_metrics = analyze_audio(orig_wav, crt_hz, mains_hz)
        mode_results: Dict[str, Any] = {}
        for mode in modes:
            res = _eval_single_mode(clip_path, mode, orig_metrics, out_dir, eval_dir, gpu_name, crt_hz, mains_hz)
            if res:
                mode_results[res[0]] = res[1]

        return {
            "identifier": meta.get("identifier", clip_path.stem),
            "title": meta.get("title", clip_path.stem),
            "genre": meta.get("genre", "general"),
            "region": meta.get("region", "europe"),
            "standard": meta.get("standard", "PAL"),
            "original_metrics": orig_metrics,
            "restored": mode_results,
        }
    finally:
        _unlink_eval_wav(orig_wav)


def _matches_filter(clip: Dict[str, Any], key: Optional[str], val: Optional[str]) -> bool:
    """Checks if clip metadata matches an optional key-value filter."""
    return not key or clip.get(key) == val


def _filter_mode_deltas(
    clips_data: List[Dict[str, Any]],
    mode: str,
    filter_key: Optional[str] = None,
    filter_val: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Extracts delta dictionaries for a given mode matching optional metadata filter."""
    matching = [c for c in clips_data if _matches_filter(c, filter_key, filter_val)]
    return [c["restored"][mode]["deltas"] for c in matching if mode in c.get("restored", {})]


def _aggregate_by_category(clips_data: List[Dict[str, Any]], modes: List[str]) -> Dict[str, Any]:
    """Aggregates benchmark results by region, genre, and overall per mode."""
    overall = {m: _average_metrics(_filter_mode_deltas(clips_data, m)) for m in modes}
    by_region = {m: _aggregate_by_region(clips_data, m) for m in modes}
    by_genre = {m: _aggregate_by_genre(clips_data, m) for m in modes}
    return {"overall": overall, "by_region": by_region, "by_genre": by_genre}


def _format_overall_table(summary: Dict[str, Any], modes: List[str]) -> List[str]:
    """Generates overall summary markdown table."""
    lines = [
        "## Overall Mode Comparison (Average Deltas vs Original)",
        "",
        "| Mode | Noise Red (dB) | SNR Gain (dB) | CRT Attenuation | Mains Attenuation | Rumble Red (%) | Balance Imp (dB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for mode in modes:
        ov = summary.get("overall", {}).get(mode, {})
        nr = ov.get("noise_reduction_db", 0.0)
        snr = ov.get("snr_gain_db", 0.0)
        crt = ov.get("crt_attenuation_ratio", 0.0)
        mains = ov.get("mains_attenuation_ratio", 0.0)
        rum = ov.get("rumble_reduction_pct", 0.0)
        bal = ov.get("balance_improvement_db", 0.0)
        lines.append(f"| `{mode}` | {nr:+.2f} dB | {snr:+.2f} dB | {crt}x | {mains}x | {rum:+.2f}% | {bal:+.2f} dB |")
    return lines


def _format_regional_table(summary: Dict[str, Any], modes: List[str]) -> List[str]:
    """Generates regional breakdown markdown table."""
    lines = [
        "",
        "## Regional Breakdown (Europe PAL vs America NTSC)",
        "",
        "| Mode | Region | Noise Red | SNR Gain | CRT Atten | Mains Atten | Rumble Red |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    reg_labels = (("europe", "Europe (PAL 50Hz)"), ("america", "America (NTSC 60Hz)"))
    for mode in modes:
        by_reg = summary.get("by_region", {}).get(mode, {})
        for reg, label in reg_labels:
            r = by_reg.get(reg, {})
            nr = r.get("noise_reduction_db", 0.0)
            snr = r.get("snr_gain_db", 0.0)
            crt = r.get("crt_attenuation_ratio", 0.0)
            mains = r.get("mains_attenuation_ratio", 0.0)
            rum = r.get("rumble_reduction_pct", 0.0)
            lines.append(f"| `{mode}` | {label} | {nr:+.2f} dB | " f"{snr:+.2f} dB | {crt}x | " f"{mains}x | {rum:+.2f}% |")
    return lines


def _format_genre_table(summary: Dict[str, Any], modes: List[str]) -> List[str]:
    """Generates genre breakdown markdown table."""
    lines = [
        "",
        "## Genre Breakdown (Home Videos vs Broadcast TV vs Music)",
        "",
        "| Mode | Genre | Noise Red | SNR Gain | CRT Atten | Mains Atten | Rumble Red |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for mode in modes:
        by_gen = summary.get("by_genre", {}).get(mode, {})
        for gen in ("home", "tv", "music"):
            g = by_gen.get(gen, {})
            nr = g.get("noise_reduction_db", 0.0)
            snr = g.get("snr_gain_db", 0.0)
            crt = g.get("crt_attenuation_ratio", 0.0)
            mains = g.get("mains_attenuation_ratio", 0.0)
            rum = g.get("rumble_reduction_pct", 0.0)
            lines.append(f"| `{mode}` | {gen.capitalize()} | {nr:+.2f} dB | " f"{snr:+.2f} dB | {crt}x | " f"{mains}x | {rum:+.2f}% |")
    return lines


def _format_markdown_report(summary: Dict[str, Any], modes: List[str], total_clips: int) -> str:
    """Formats aggregated benchmark results as Markdown table and narrative."""
    lines = [
        "# Internet Archive VHS Audio Restoration Benchmark Report",
        "",
        f"Evaluated across **{total_clips}** VHS captures spanning Europe (PAL 50 Hz / 15.625 kHz) ",
        "and America (NTSC 60 Hz / 15.734 kHz) across Home Video, Broadcast TV, and Music categories.",
        "",
    ]
    # Stated in the report, not just on stderr. Without it the headline count reads as the whole
    # corpus, and a reader has no way to tell that some clips never contributed.
    coverage = summary.get("coverage") or {}
    if coverage.get("clips_skipped"):
        lines.extend(
            [
                f"> **Coverage:** {coverage['clips_evaluated']} of {coverage['clips_found']} clips in the corpus "
                f"were evaluated; {coverage['clips_skipped']} were skipped.",
                "",
            ]
        )
        lines.extend([f"> - `{entry['clip']}` — {entry['reason']}" for entry in coverage.get("skipped", [])])
        lines.append("")
    lines.extend(_format_overall_table(summary, modes))
    lines.extend(_format_regional_table(summary, modes))
    lines.extend(_format_genre_table(summary, modes))
    return "\n".join(lines)


def _parse_region_items(region: str, items: Any) -> Dict[str, Dict[str, Any]]:
    """Builds metadata lookup entries for a single catalog region."""
    reg_norm = "america" if "america" in region.lower() else "europe"
    region_index: Dict[str, Dict[str, Any]] = {}
    for it in (items if isinstance(items, list) else []):
        ident = str(it.get("identifier", "")).strip()
        if ident:
            it_meta = dict(it)
            it_meta["region"] = reg_norm
            defaults = _default_clip_meta(ident, reg_norm == "america")
            it_meta.setdefault("crt_hz", defaults["crt_hz"])
            it_meta.setdefault("notch_hz", defaults["notch_hz"])
            region_index[ident] = it_meta
    return region_index


def _clip_stems(meta: Dict[str, Any], duration_sec: int = 20) -> List[str]:
    """Returns every filename stem this catalog item could carry on disk.

    The two curation scripts slugify differently -- curate_ia_corpus.py keeps the identifier
    intact, curate_massive_ia_corpus.py truncates it and appends a digest -- so a stem computed
    with one cannot match a corpus built by the other. A miss is silent: _find_clip_meta falls
    back to _default_clip_meta and the clip is then measured against guessed mains and CRT
    frequencies rather than its catalogued ones.
    """
    if meta.get("file"):
        return [Path(str(meta["file"])).stem]
    identifier = str(meta["identifier"])
    genre = str(meta.get("genre", "general"))
    stems = [f"{slug(identifier)}_{slug(genre)}_{duration_sec}s" for slug in (_plain_slug, _massive_slug)]
    return list(dict.fromkeys(stems))


def _load_meta_index(catalog_path: Path) -> Dict[str, Dict[str, Any]]:
    """Loads catalog JSON and indexes entries by their canonical clip stem."""
    with open(catalog_path, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)

    # Two catalog shapes exist: curate_ia_corpus.py writes {region: [records]}, while
    # curate_massive_ia_corpus.py writes a flat list whose records carry their own "region".
    # Calling .items() on the latter raised AttributeError, so pointing --catalog at
    # catalog_1000.json crashed before a single clip was measured.
    if isinstance(catalog, list):
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in catalog:
            grouped.setdefault(str(record.get("region", "unknown")), []).append(record)
        catalog = grouped

    index: Dict[str, Dict[str, Any]] = {}
    for region, items in catalog.items():
        for meta in _parse_region_items(region, items).values():
            for stem in _clip_stems(meta):
                index.setdefault(stem, meta)
    return index


def _infer_genre_from_stem(stem: str) -> str:
    """Infers genre tag from filename stem conventions (_home_, _tv_, _music_)."""
    lowered = stem.lower()
    for tag in ("_home_", "_tv_", "_music_"):
        if tag in lowered:
            return tag.strip("_")
    return "general"


def _default_clip_meta(stem: str, is_america: bool) -> Dict[str, Any]:
    """Builds fallback metadata when item is not found in catalog index."""
    reg = "america" if is_america else "europe"
    return {
        "identifier": stem,
        "region": reg,
        "standard": "NTSC" if is_america else "PAL",
        "crt_hz": 15734.0 if is_america else 15625.0,
        "notch_hz": 60.0 if is_america else 50.0,
        "genre": _infer_genre_from_stem(stem),
    }


def _find_clip_meta(clip_path: Path, meta_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Matches a clip path to exact canonical catalog metadata when available."""
    stem = clip_path.stem
    if stem in meta_index:
        return meta_index[stem]
    return _default_clip_meta(stem, "america" in str(clip_path).lower())


def _collect_region_clips(reg_dir: Path, limit: Optional[int], skipped: List[Dict[str, str]]) -> List[Path]:
    """Collects and sorts valid video clips, recording the ones rejected as unreadable."""
    if not reg_dir.exists():
        return []
    clips: List[Path] = []
    for path in sorted(reg_dir.glob("*.mp4")):
        # The limit stops the scan rather than trimming afterwards. Recording skips for files
        # past the limit counted clips the run never intended to touch, so a --limit run
        # reported coverage against the whole directory instead of the subset it asked for.
        if limit is not None and len(clips) >= limit:
            break
        if is_valid_video(path):
            clips.append(path)
        else:
            # Recorded rather than dropped: a corpus can contain a truncated or zero-byte
            # download, and averaging over the survivors while reporting the smaller count as
            # if it were the corpus is how a benchmark quietly overstates its coverage.
            skipped.append({"clip": path.name, "reason": "unreadable or not a valid video"})
    return clips


def _evaluate_all_clips(
    all_clips: List[Path],
    meta_index: Dict[str, Dict[str, Any]],
    modes: List[str],
    out_dir: Path,
    eval_dir: Path,
    gpu_name: str,
    skipped: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Iterates through and evaluates all clips, recording every clip that produced no result."""
    results: List[Dict[str, Any]] = []
    for idx, clip in enumerate(all_clips, 1):
        meta = _find_clip_meta(clip, meta_index)
        print(f"[{idx}/{len(all_clips)}] Benchmarking: {clip.name}")
        try:
            eval_res = _eval_single_clip(clip, meta, modes, out_dir, eval_dir, gpu_name)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"  [Warning] Skipping {clip.name}: {exc}", file=sys.stderr)
            skipped.append({"clip": clip.name, "reason": str(exc)})
            continue
        if eval_res:
            results.append(eval_res)
        else:
            # A None result means audio extraction or analysis produced nothing usable. It used
            # to vanish here, so the averages were taken over an unstated subset.
            print(f"  [Warning] Skipping {clip.name}: produced no usable audio", file=sys.stderr)
            skipped.append({"clip": clip.name, "reason": "no usable audio extracted"})
    return results


def benchmark_corpus(
    corpus_dir: Path,
    catalog_path: Path,
    output_dir: Path,
    modes: List[str],
    limit_per_region: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Runs complete benchmarking across curated corpus directory."""
    meta_index = _load_meta_index(catalog_path)
    eval_dir = output_dir / "eval_tmp"
    eval_dir.mkdir(parents=True, exist_ok=True)
    gpu_name = get_gpu_name()

    skipped: List[Dict[str, str]] = []
    all_clips = _collect_region_clips(corpus_dir / "europe", limit_per_region, skipped)
    us_dir = corpus_dir / "american" if (corpus_dir / "american").exists() else corpus_dir / "america"
    all_clips.extend(_collect_region_clips(us_dir, limit_per_region, skipped))
    # Counted before evaluation runs: clips rejected during collection are not in all_clips,
    # while clips that fail later are. Adding the two lists blindly double-counts the latter,
    # so the reported total would not equal evaluated + skipped.
    rejected_during_collection = len(skipped)

    results = _evaluate_all_clips(all_clips, meta_index, modes, output_dir, eval_dir, gpu_name, skipped)
    summary = _aggregate_by_category(results, modes)
    summary["coverage"] = {
        "clips_found": len(all_clips) + rejected_during_collection,
        "clips_evaluated": len(results),
        "clips_skipped": len(skipped),
        "skipped": skipped,
    }
    if skipped:
        print(f"\n[Coverage] {len(results)} clips evaluated, {len(skipped)} skipped:", file=sys.stderr)
        for entry in skipped:
            print(f"  - {entry['clip']}: {entry['reason']}", file=sys.stderr)
    return summary, results


def _parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("experiments/ia_corpus"))
    parser.add_argument("--catalog", type=Path, default=Path("experiments/ia_corpus_catalog.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/benchmark_run"))
    parser.add_argument("--modes", nargs="+", default=["cathar", "auto_pure_linear"])
    parser.add_argument("--report-json", type=Path, default=Path("experiments/benchmark_ia_corpus_report.json"))
    parser.add_argument("--report-md", type=Path, default=Path("experiments/benchmark_ia_corpus_report.md"))
    parser.add_argument("--limit", type=int, default=None, help="Max clips per region")
    return parser.parse_args()


def main() -> None:
    """Main CLI entry point."""
    args = _parse_args()
    summary, results = benchmark_corpus(
        args.corpus_dir,
        args.catalog,
        args.output_dir,
        args.modes,
        limit_per_region=args.limit,
    )

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": results}, handle, indent=2)

    md_report = _format_markdown_report(summary, args.modes, len(results))
    with open(args.report_md, "w", encoding="utf-8") as handle:
        handle.write(md_report)

    print("\n" + md_report)
    print(f"\nSaved JSON report: {args.report_json}")
    print(f"Saved Markdown report: {args.report_md}")


if __name__ == "__main__":
    main()
