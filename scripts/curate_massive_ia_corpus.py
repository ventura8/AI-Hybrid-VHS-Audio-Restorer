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
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.utils import FFMPEG_BIN, is_valid_video

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
    cleaned = re.sub(r"[^\w\-_.]", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if len(cleaned) > 60:
        digest = hashlib.md5(name.encode()).hexdigest()[:8]
        cleaned = f"{cleaned[:51]}_{digest}"
    return cleaned


def _extract_clip(stream_url: str, target_path: Path, offset_sec: int = 60, duration_sec: int = 15) -> bool:
    """Extracts a 15-second slice from stream."""
    if is_valid_video(target_path):
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
        if res.returncode == 0 and is_valid_video(temp_target):
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
        if res.returncode == 0 and is_valid_video(temp_target):
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
    return False


def curate_catalog(target_count: int = 1000) -> List[Dict[str, Any]]:
    """Builds a diverse 1,000-tape manifest across European and American queries."""
    items_by_id: Dict[str, Dict[str, Any]] = {}
    half_target = target_count // 2

    # 1. Harvest European items
    euro_items = []
    for query, genre, notch, crt in EUROPE_SEARCH_QUERIES:
        docs = _search_archive(query, rows=200)
        for doc in docs:
            ident = doc.get("identifier")
            if ident and ident not in items_by_id:
                record = {
                    "identifier": ident,
                    "title": doc.get("title") or ident,
                    "region": "europe",
                    "standard": "PAL",
                    "genre": genre,
                    "notch_hz": notch,
                    "crt_hz": crt,
                }
                items_by_id[ident] = record
                euro_items.append(record)
                if len(euro_items) >= half_target:
                    break
        if len(euro_items) >= half_target:
            break

    # 2. Harvest American items
    us_items = []
    for query, genre, notch, crt in AMERICA_SEARCH_QUERIES:
        docs = _search_archive(query, rows=200)
        for doc in docs:
            ident = doc.get("identifier")
            if ident and ident not in items_by_id:
                record = {
                    "identifier": ident,
                    "title": doc.get("title") or ident,
                    "region": "america",
                    "standard": "NTSC",
                    "genre": genre,
                    "notch_hz": notch,
                    "crt_hz": crt,
                }
                items_by_id[ident] = record
                us_items.append(record)
                if len(us_items) >= half_target:
                    break
        if len(us_items) >= half_target:
            break

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
