#!/usr/bin/env python3
"""Curates a massive 1,000-tape Internet Archive VHS benchmark corpus.

Queries archive.org for diverse VHS collections (Home Video, TV, Music)
across European (PAL) and American (NTSC) broadcast standards.
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.utils import FFMPEG_BIN, FFPROBE_BIN, is_valid_video

EUROPE_SEARCH_QUERIES = [
    ('collection:(vhsvault OR home_movies) AND ("UK VHS" OR "PAL VHS" OR "Home Video UK" OR "camcorder")', "home", 50.0, 15625.0),
    ('collection:(vhsvault) AND (BBC OR ITV OR "Channel 4" OR "British Television" OR "TV PAL")', "tv", 50.0, 15625.0),
    ('collection:(vhsvault OR musicvideos) AND ("Top of the Pops" OR Eurodance OR "MTV Europe" OR "TOTP")', "music", 50.0, 15625.0),
    ('collection:(vhsvault) AND ("PAL" OR "Tele 7" OR "SOTI" OR "Romanian VHS" OR "German VHS" OR "French VHS")', "tv", 50.0, 15625.0),
]

AMERICA_SEARCH_QUERIES = [
    (
        'collection:(vhsvault OR home_movies) AND ("Home Video" OR "Christmas 19" OR "Family Vacation" OR "VHS Camcorder")',
        "home",
        60.0,
        15734.0,
    ),
    (
        'collection:(vhsvault OR classic_tv_commercials) AND ("NTSC" OR NBC OR CBS OR ABC OR "Commercials 19" OR "Broadcast")',
        "tv",
        60.0,
        15734.0,
    ),
    ('collection:(vhsvault OR musicvideos) AND ("MTV" OR "Music Video" OR "Concert VHS" OR "VH1")', "music", 60.0, 15734.0),
    ('collection:(vhsvault) AND ("Disney VHS" OR "Cartoon VHS" OR "Promo VHS" OR "Feature Presentation")', "home", 60.0, 15734.0),
]


def _search_archive(query: str, rows: int = 150) -> List[Dict[str, Any]]:
    """Queries Archive.org search API for movie items."""
    url = (
        f"https://archive.org/advancedsearch.php?q={urllib.parse.quote(query)}"
        f"&fl[]=identifier,title,year,collection&rows={rows}&output=json"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "AI-Restorer-Curator/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", {}).get("docs", [])
    except Exception as e:
        sys.stderr.write(f"Search failed for {query[:30]}: {e}\n")
        return []


def _resolve_mp4_url(identifier: str) -> Optional[str]:
    """Resolves primary MP4 stream URL from item metadata."""
    url = f"https://archive.org/metadata/{identifier}"
    req = urllib.request.Request(url, headers={"User-Agent": "AI-Restorer-Curator/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        files = [f["name"] for f in data.get("files", []) if f.get("name", "").endswith(".mp4")]
        if not files:
            return None
        non_derivs = [f for f in files if "512kb" not in f and "ia.mp4" not in f]
        chosen = non_derivs[0] if non_derivs else files[0]
        escaped_file = urllib.parse.quote(chosen)
        return f"https://archive.org/download/{identifier}/{escaped_file}"
    except Exception:
        return None


def _sanitize_slug(name: str) -> str:
    """Maps an archive identifier to a filesystem-safe stem that is unique to that identifier.

    The digest is unconditional, not merely a truncation aid. Every character outside
    [\\w\\-_.] collapses to "_", so two distinct identifiers can sanitize to one slug and
    resolve to the same clip path. Curation downloads through a ThreadPoolExecutor with eight
    workers, so colliding items can write the same temporary target and replace() it
    concurrently -- leaving one truncated or overwritten clip and a manifest entry pointing at
    the wrong source.
    """
    cleaned = re.sub(r"[^\w\-_.]", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    digest = hashlib.md5(name.encode(), usedforsecurity=False).hexdigest()[:8]
    if len(cleaned) > 51:
        cleaned = cleaned[:51]
    return f"{cleaned}_{digest}"


def _source_duration(source: str, timeout: int = 10):
    """Seconds of media in a file or stream, or None when ffprobe cannot say."""
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        return float(result.stdout.strip()) if result.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _has_expected_duration(path: Path, duration_sec: int) -> bool:
    """Confirm a downloaded clip is long enough for a meaningful benchmark."""
    duration = _source_duration(path)
    return duration is not None and duration >= duration_sec * 0.8


def _extract_clip(stream_url: str, target_path: Path, offset_sec: int = 60, duration_sec: int = 15, fallback_to_start=True) -> bool:
    """Extracts a 15-second slice from stream.

    When the slice at `offset_sec` cannot be taken and `fallback_to_start` is set, the
    opening of the tape is taken instead; a caller that records the offset passes False,
    so that what it records is where the clip came from.
    """
    if is_valid_video(target_path) and _has_expected_duration(target_path, duration_sec):
        return True
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_suffix(".tmp.mp4")

    # Fast stream copy
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "3",
        "-ss",
        str(offset_sec),
        "-i",
        stream_url,
        "-t",
        str(duration_sec),
        "-c",
        "copy",
        str(temp_target),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=40)
        if res.returncode == 0 and is_valid_video(temp_target) and _has_expected_duration(temp_target, duration_sec):
            temp_target.replace(target_path)
            return True
    except (subprocess.SubprocessError, OSError):
        pass

    # Transcode fallback
    cmd_trans = [
        FFMPEG_BIN,
        "-y",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "3",
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
        "28",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(temp_target),
    ]
    try:
        res = subprocess.run(cmd_trans, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=60)
        if res.returncode == 0 and is_valid_video(temp_target) and _has_expected_duration(temp_target, duration_sec):
            temp_target.replace(target_path)
            return True
    except (subprocess.SubprocessError, OSError):
        pass
    finally:
        if temp_target.exists():
            try:
                temp_target.unlink()
            except OSError:
                pass
    if offset_sec != 0 and fallback_to_start:
        return _extract_clip(stream_url, target_path, offset_sec=0, duration_sec=duration_sec)
    return False


def _harvest_region(queries, region, standard, target, items_by_id):
    """Collect unique catalog records for one broadcast region."""
    records = []
    for query, genre, notch, crt in queries:
        for doc in _search_archive(query, rows=200):
            record = _catalog_record(doc, region, standard, genre, notch, crt)
            if record and record["identifier"] not in items_by_id:
                items_by_id[record["identifier"]] = record
                records.append(record)
                if len(records) >= target:
                    return records
    return records


def _catalog_record(doc, region, standard, genre, notch, crt):
    """Build one normalized catalog record, or None for an unidentified item."""
    ident = doc.get("identifier")
    if not ident:
        return None
    return {
        "identifier": ident,
        "title": doc.get("title") or ident,
        "region": region,
        "standard": standard,
        "genre": genre,
        "notch_hz": notch,
        "crt_hz": crt,
    }


def curate_catalog(target_count: int = 1000) -> List[Dict[str, Any]]:
    """Builds a diverse 1,000-tape manifest across European and American queries."""
    items_by_id: Dict[str, Dict[str, Any]] = {}
    half_target = target_count // 2
    euro_items = _harvest_region(EUROPE_SEARCH_QUERIES, "europe", "PAL", half_target, items_by_id)
    us_items = _harvest_region(AMERICA_SEARCH_QUERIES, "america", "NTSC", half_target, items_by_id)

    total = euro_items + us_items
    sys.stdout.write(f"Harvested {len(total)} candidate items (Europe: {len(euro_items)}, America: {len(us_items)})\n")
    return total[:target_count]


def download_corpus(catalog: List[Dict[str, Any]], output_dir: Path, max_workers: int = 8) -> List[Dict[str, Any]]:
    """Downloads 15-second representative clips in parallel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_catalog = []

    def process_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ident = item["identifier"]
        stream_url = _resolve_mp4_url(ident)
        if not stream_url:
            return None
        slug = _sanitize_slug(ident)
        reg_dir = output_dir / item["region"]
        clip_name = f"{slug}_{item['genre']}_15s.mp4"
        clip_path = reg_dir / clip_name

        ok = _extract_clip(stream_url, clip_path, offset_sec=60, duration_sec=15)
        if ok:
            record = dict(item)
            record["file"] = clip_path.relative_to(output_dir).as_posix()
            record["stream_url"] = stream_url
            return record

        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, it): it for it in catalog}
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
            except Exception as exc:
                sys.stderr.write(f"  [Warning] Item download failed: {exc}\n")
                continue
            if res:
                downloaded_catalog.append(res)
                if len(downloaded_catalog) % 25 == 0:
                    sys.stdout.write(f"  [Progress] Downloaded {len(downloaded_catalog)} valid clips...\n")

    return downloaded_catalog


def main():
    parser = argparse.ArgumentParser(description="Curate massive 1,000-tape IA VHS benchmark corpus")
    parser.add_argument("--target-count", type=int, default=1000, help="Target clip count")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ia_corpus_1000"))
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "catalog_1000.json"

    catalog_candidates = curate_catalog(target_count=args.target_count)
    downloaded = download_corpus(catalog_candidates, args.output_dir, max_workers=args.workers)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(downloaded, f, indent=2)

    sys.stdout.write(f"Successfully curated {len(downloaded)} clips. Catalog saved to {manifest_path}\n")


if __name__ == "__main__":
    main()
