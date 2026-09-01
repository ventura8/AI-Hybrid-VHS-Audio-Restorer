#!/usr/bin/env python3
"""Curate representative test clips from Internet Archive VHS corpus."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure modules package can be resolved
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.utils import FFMPEG_BIN, FFPROBE_BIN, is_valid_video


def _sanitize_slug(name: str) -> str:
    """Creates a filesystem-safe slug from a title or filename."""
    cleaned = re.sub(r"[^\w\-_.]", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def _build_copy_cmd(stream_url: str, target_path: Path, offset_sec: int, duration_sec: int) -> List[str]:
    """Constructs fast stream-copy FFmpeg command line."""
    return [
        FFMPEG_BIN,
        "-y",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "2",
        "-ss",
        str(offset_sec),
        "-i",
        stream_url,
        "-t",
        str(duration_sec),
        "-c",
        "copy",
        str(target_path),
    ]


def _build_transcode_cmd(stream_url: str, target_path: Path, offset_sec: int, duration_sec: int) -> List[str]:
    """Constructs transcode fallback FFmpeg command line."""
    return [
        FFMPEG_BIN,
        "-y",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "2",
        "-ss",
        str(offset_sec),
        "-i",
        stream_url,
        "-t",
        str(duration_sec),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "26",
        "-c:a",
        "pcm_f32le",
        "-ar",
        "44100",
        str(target_path),
    ]


def _get_clip_duration(clip_path: Path) -> float:
    """Probes container duration in seconds using ffprobe."""
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(clip_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
        return float(res.stdout.strip()) if res.stdout.strip() else 0.0
    except (subprocess.SubprocessError, OSError, ValueError):
        return 0.0


def _execute_ffmpeg(cmd: List[str], timeout_sec: int = 40) -> bool:
    """Executes FFmpeg subprocess with timeout."""
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout_sec,
        )
        return res.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def _attempt_download_step(cmd: List[str], tmp_path: Path, target_path: Path, duration_sec: int) -> bool:
    """Executes download command to tmp_path, validating duration before replacement."""
    if _execute_ffmpeg(cmd, timeout_sec=max(40, duration_sec * 2)) and _get_clip_duration(tmp_path) >= duration_sec * 0.8:
        tmp_path.replace(target_path)
        return True
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass
    return False


def _try_download_at_offset(stream_url: str, target_path: Path, offset: int, duration_sec: int) -> bool:
    """Tries stream copy, then transcode at given offset to a temporary file."""
    tmp_path = target_path.with_name(f"{target_path.stem}.tmp.mp4")
    copy_cmd = _build_copy_cmd(stream_url, tmp_path, offset, duration_sec)
    if _attempt_download_step(copy_cmd, tmp_path, target_path, duration_sec):
        return True
    transcode_cmd = _build_transcode_cmd(stream_url, tmp_path, offset, duration_sec)
    return _attempt_download_step(transcode_cmd, tmp_path, target_path, duration_sec)


def _download_clip(stream_url: str, target_path: Path, offset_sec: int, duration_sec: int) -> bool:
    """Downloads a clip trying preferred offset first, falling back to 0."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if _try_download_at_offset(stream_url, target_path, offset_sec, duration_sec):
        return True
    if offset_sec != 0 and _try_download_at_offset(stream_url, target_path, 0, duration_sec):
        return True
    return False


def _process_item(
    item: Dict[str, Any],
    region_dir: Path,
    offset_sec: int,
    duration_sec: int,
    force: bool,
) -> bool:
    """Extracts a single catalog item to destination file."""
    identifier = item.get("identifier", "unknown")
    genre = item.get("genre", "general")
    stream_url = item.get("stream_url", "")
    if not stream_url:
        return False

    slug = _sanitize_slug(identifier)
    target_path = region_dir / f"{slug}_{genre}_{duration_sec}s.mp4"
    if not force and is_valid_video(target_path) and _get_clip_duration(target_path) >= duration_sec * 0.8:
        return True

    return _download_clip(stream_url, target_path, offset_sec, duration_sec)


def _curate_region(
    items: List[Dict[str, Any]],
    region_name: str,
    output_dir: Path,
    offset_sec: int,
    duration_sec: int,
    limit: Optional[int],
    force: bool,
) -> int:
    """Processes all items for a specific region."""
    region_dir = output_dir / region_name
    region_dir.mkdir(parents=True, exist_ok=True)
    selected = items[:limit] if limit is not None else items
    success_count = 0

    for idx, item in enumerate(selected, 1):
        title = item.get("title", item.get("identifier", "unknown"))
        ok = _process_item(item, region_dir, offset_sec, duration_sec, force)
        status = "OK" if ok else "FAILED"
        print(f"[{region_name.upper()} {idx}/{len(selected)}] {status}: {title[:60]}", flush=True)
        if ok:
            success_count += 1
    return success_count


def curate_corpus(
    catalog_path: Path,
    output_dir: Path,
    offset_sec: int = 30,
    duration_sec: int = 20,
    limit_per_region: Optional[int] = None,
    force: bool = False,
) -> Dict[str, int]:
    """Curates clips for all regions in catalog."""
    with open(catalog_path, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)

    stats: Dict[str, int] = {}
    for region_name, items in catalog.items():
        if isinstance(items, list):
            count = _curate_region(
                items,
                region_name,
                output_dir,
                offset_sec,
                duration_sec,
                limit_per_region,
                force,
            )
            stats[region_name] = count
    return stats


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments for corpus curation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("experiments/ia_corpus_catalog.json"),
        help="Path to JSON corpus catalog",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/ia_corpus"),
        help="Destination directory for curated video clips",
    )
    parser.add_argument("--offset", type=int, default=30, help="Start offset in seconds")
    parser.add_argument("--duration", type=int, default=20, help="Clip duration in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Limit items per region")
    parser.add_argument("--force", action="store_true", help="Force re-download existing clips")
    return parser.parse_args()


def main() -> None:
    """Main CLI entry point."""
    args = _parse_args()
    print(f"Curating Internet Archive VHS corpus from: {args.catalog}")
    stats = curate_corpus(
        args.catalog,
        args.output_dir,
        offset_sec=args.offset,
        duration_sec=args.duration,
        limit_per_region=args.limit,
        force=args.force,
    )
    print("\nCuration Complete:")
    for region, count in stats.items():
        print(f"  - {region}: {count} clips curated")


if __name__ == "__main__":
    main()
