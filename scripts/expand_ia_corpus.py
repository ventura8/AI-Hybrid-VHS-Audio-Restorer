#!/usr/bin/env python3
"""Expands the Internet Archive VHS corpus: more tapes, more kinds of tape, more of each tape.

The 192-clip corpus behind every real-tape figure on this branch came from eight searches
over two collections, one 15 s slice per tape, and its genre field reads "home" for all of
them. That is a narrow base for verdicts that decide what ships, and the fixture set is
calibrated against its statistics. This widens it three ways:

- **More searches.** Children's television, sport, news, documentary, comedy, concerts,
  adverts, weddings and school events, and tapes in languages other than English, over the
  collections that hold digitised VHS rather than only two of them. Each query carries a
  genre label that means what it says.
- **Paging.** The search API returns at most a page at a time; this walks pages until a
  query is exhausted or the target is met.
- **More of each tape.** A tape is not the same at minute one as at minute eight -- the
  titles, the music bed, the talking, the credits -- so up to three slices are taken from
  different offsets, each a clip of its own.

Tapes already in a catalog are skipped, so the result is strictly additional, and the
merged catalog it writes is what the measurement scripts take through `--catalog`.
"""

import argparse
import concurrent.futures
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.curate_massive_ia_corpus import _catalog_record, _extract_clip, _resolve_mp4_url, _sanitize_slug, _source_duration

COLLECTIONS = "(vhsvault OR home_movies OR opensource_movies OR community_video OR television OR ephemera)"
PAL = ("europe", "PAL", 50.0, 15625.0)
NTSC = ("america", "NTSC", 60.0, 15734.0)
# (query terms, genre, region). Every query is restricted to movies in the collections above.
QUERIES = (
    ('("UK VHS" OR "PAL VHS") AND (children OR kids OR cartoon OR "Thomas" OR "Postman Pat" OR nursery)', "kids", PAL),
    ('("UK VHS" OR "British") AND (football OR cricket OR snooker OR "Grand Prix" OR wrestling OR sport)', "sport", PAL),
    ('("UK VHS" OR BBC OR ITV OR "Channel 4") AND (news OR documentary OR "Panorama" OR "World in Action")', "documentary", PAL),
    ('("UK VHS" OR BBC OR ITV) AND (comedy OR sitcom OR "Only Fools" OR "Blackadder" OR "stand-up")', "comedy", PAL),
    ('("UK VHS" OR "PAL") AND (concert OR "live at" OR "Top of the Pops" OR gig OR "music video")', "music", PAL),
    ('("UK VHS" OR ITV OR "Channel 4") AND (adverts OR commercials OR "ad break" OR continuity OR idents)', "adverts", PAL),
    ('(camcorder OR "home video" OR wedding OR "school play" OR "family video") AND (UK OR PAL OR Ireland OR Australia)', "home", PAL),
    ('VHS AND (Deutsch OR German OR "ZDF" OR "ARD" OR "RTL" OR "Sat.1" OR "Fernsehen")', "foreign", PAL),
    ('VHS AND (French OR "français" OR "TF1" OR "France 2" OR "Antenne 2" OR "Canal+")', "foreign", PAL),
    ('VHS AND (Spanish OR "español" OR "TVE" OR "Telecinco" OR Italian OR "Rai" OR "Canale 5")', "foreign", PAL),
    ('VHS AND (Dutch OR "Nederland" OR Polish OR "Polsat" OR "TVP" OR Romanian OR "TVR" OR Russian OR Czech)', "foreign", PAL),
    ('("NTSC" OR "USA") AND VHS AND (children OR kids OR cartoon OR "Sesame Street" OR "Barney" OR Nickelodeon)', "kids", NTSC),
    ('VHS AND (NFL OR NBA OR MLB OR NHL OR "Monday Night" OR "WWF" OR wrestling OR "Super Bowl")', "sport", NTSC),
    ('VHS AND (NBC OR CBS OR ABC OR CNN OR "local news" OR "Eyewitness News" OR "60 Minutes" OR documentary)', "documentary", NTSC),
    ('VHS AND (sitcom OR "Saturday Night Live" OR "stand-up" OR comedy OR "Tonight Show" OR "Letterman")', "comedy", NTSC),
    ('VHS AND (MTV OR VH1 OR concert OR "live in" OR "music video" OR "American Bandstand")', "music", NTSC),
    ('VHS AND (commercials OR "commercial break" OR "station ID" OR promos OR "TV ads")', "adverts", NTSC),
    ('(camcorder OR "home video" OR "home movie" OR wedding OR "birthday party" OR "family vacation" OR "Christmas")', "home", NTSC),
    ('VHS AND (Spanish OR "Telemundo" OR "Univision" OR Japanese OR "Japan" OR anime OR Korean OR Chinese OR Filipino)', "foreign", NTSC),
)
PAGE_ROWS = 100
USER_AGENT = "AI-Restorer-Curator/2.1"


def _search_page(query, page):
    """One page of the search API, or an empty list on any failure."""
    url = (
        f"https://archive.org/advancedsearch.php?q={urllib.parse.quote(query)}"
        f"&fl[]=identifier,title,year,collection&rows={PAGE_ROWS}&page={page}&output=json"
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8")).get("response", {}).get("docs", [])
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"  search failed ({query[:40]}..., page {page}): {exc}\n")
        return []


def _harvest(target, known, max_pages):
    """Walks every query, page by page, taking an equal share of the target from each.

    A first pass that stopped as soon as the total was met filled the target from the first
    four queries alone -- 1,145 clips, all PAL, four genres -- so each query now gets its
    share and no more, and the set comes out balanced across genre and region.
    """
    share = -(-target // len(QUERIES))
    records = []
    for terms, genre, (region, standard, notch, crt) in QUERIES:
        query = f"mediatype:movies AND collection:{COLLECTIONS} AND {terms}"
        taken, page = 0, 1
        while taken < share and page <= max_pages:
            docs = _search_page(query, page)
            page += 1
            if not docs:
                break
            for doc in docs:
                record = _catalog_record(doc, region, standard, genre, notch, crt)
                if record and record["identifier"] not in known and taken < share:
                    known.add(record["identifier"])
                    records.append(record)
                    taken += 1
    return records


def _fetch_slices(item, output_dir, offsets):
    """Downloads up to one clip per offset for one tape; returns the catalog records that succeeded.

    Only offsets the tape actually reaches are asked for, and the extractor is told not to
    fall back to the opening: with the fallback, a tape shorter than the last offset gave
    its opening three times over under three different offsets, and the first harvest
    carried 167 such duplicates across 117 tapes before they were found by hash.
    """
    stream_url = _resolve_mp4_url(item["identifier"])
    if not stream_url:
        return []
    duration = _source_duration(stream_url, timeout=30)
    if duration is None:
        return []
    slug = _sanitize_slug(item["identifier"])
    found = []
    for offset in (offset for offset in offsets if offset + 15 <= duration):
        clip_path = output_dir / item["region"] / f"{slug}_{item['genre']}_{offset}s_15s.mp4"
        if _extract_clip(stream_url, clip_path, offset_sec=offset, duration_sec=15, fallback_to_start=False):
            record = dict(item)
            record["file"] = clip_path.relative_to(output_dir).as_posix()
            record["stream_url"] = stream_url
            record["offset_sec"] = offset
            found.append(record)
    return found


FLUSH_EVERY = 25


def _download(catalog, output_dir, offsets, workers, flush=None):
    """Fetches every tape's slices in parallel, handing the records so far to `flush` as it goes.

    A harvest is hours long, so what has landed is written out every few tapes rather than
    only at the end: an interruption keeps its records, and a rerun skips the tapes they name.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded, completed = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_slices, item, output_dir, offsets) for item in catalog]
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            try:
                downloaded.extend(future.result())
            except Exception as exc:  # one tape must not stop the harvest, whatever it failed with
                sys.stderr.write(f"  download failed: {type(exc).__name__}: {exc}\n")
            if completed % FLUSH_EVERY == 0:
                sys.stdout.write(f"  {completed} tapes done, {len(downloaded)} clips so far\n")
                sys.stdout.flush()
                if flush is not None:
                    flush(downloaded)
    return downloaded


def _load_catalog(path):
    """A catalog's records, or none when the file is absent."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _write_atomic(path, records):
    """Writes a catalog through a temporary file, so a stop mid-write leaves the previous one whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2), encoding="utf-8")
    temporary.replace(path)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ia_corpus_1000"))
    parser.add_argument("--existing", type=Path, nargs="*", default=[Path("experiments/ia_corpus_1000/catalog_1000.json")])
    parser.add_argument("--target-tapes", type=int, default=300, help="New tapes to look for")
    parser.add_argument("--offsets", type=int, nargs="+", default=[60, 240, 480], help="Slice offsets in seconds")
    parser.add_argument("--max-pages", type=int, default=5, help="Search pages per query")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--catalog", type=Path, default=Path("experiments/ia_corpus_1000/catalog_extra.json"))
    parser.add_argument("--merged", type=Path, default=Path("experiments/ia_corpus_1000/catalog_all.json"))
    return parser.parse_args()


def main():
    """Harvests, downloads, and writes both the new and the merged catalogs."""
    args = _parse_args()
    existing = [record for path in args.existing for record in _load_catalog(path)]
    known = {record["identifier"] for record in existing} | {record["identifier"] for record in _load_catalog(args.catalog)}
    candidates = _harvest(args.target_tapes, set(known), args.max_pages)
    sys.stdout.write(f"{len(candidates)} new tapes found; fetching {len(args.offsets)} slices of each\n")
    sys.stdout.flush()
    previous = _load_catalog(args.catalog)

    def write_catalogs(new_records):
        _write_atomic(args.catalog, previous + new_records)
        _write_atomic(args.merged, existing + previous + new_records)

    fetched = _download(candidates, args.output_dir, args.offsets, args.workers, flush=write_catalogs)
    write_catalogs(fetched)
    downloaded = previous + fetched
    sys.stdout.write(f"{len(downloaded)} new clips in {args.catalog}; {len(existing) + len(downloaded)} in {args.merged}\n")


if __name__ == "__main__":
    main()
