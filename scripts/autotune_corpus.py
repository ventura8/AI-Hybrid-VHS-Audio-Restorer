"""The loop's corpus guard: a round's winner is accepted only while a sentinel set of IA clips holds.

Three plateaus in a row passed the Tata tapes and failed their IA confirmation afterwards (the APL
v9 plateau: 31 of 202 clips, cathar's v9 final: 57 of 235, the APL v10 plateau: 31 of 235), each
attributed by hand. With `--corpus FILE` (JSON `{slug: clip path}`: copies of IA clips the loop
owns, since the app writes its outputs beside them) the loop renders each qualifying candidate,
best first, on those clips, scores it beside the corpus baseline (the run's start settings,
rendered once) at the same level, and counts the clips on which it trips one of the grid's
constraints materially (`confirm_trips.clip_trips`, the rule a plateau's confirmation reads). A
candidate past `--corpus-max-trips` is refused and the next qualifying one is tried, so the loop
settles on the best move the sentinels allow; the confirmation on the whole corpus still decides a
default. A clip the candidate fails to render counts as a trip, and a clip the baseline fails on is
left out. Each candidate's trips are kept in `corpus/cands/<id>/corpus_trips.json`, so a resumed
run does not render or score it again.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import NamedTuple

from scripts import autotune_renders as renders
from scripts.autotune_guards import read_json
from scripts.autotune_scoring import PYTHON, REPO
from scripts.confirm_trips import clip_trips, load_constraints

DIR = "corpus"
TRIPS_JSON = "corpus_trips.json"
BASE_FAILED_JSON = "baseline_failed.json"
BASE, CAND = "base", "cand"
RENDER_FAILED = "render failed"
UNSCORED = "not scored"


class Guard(NamedTuple):
    """`clips` `{slug: path}`; `baseline` `(id, settings)`, what every candidate is read against; `max_trips` the
    clips a candidate may trip; `constraints` the grid's; `out_dir` the loop's `<out>/<engine>/corpus`; `parallel`
    the scorers at once."""

    clips: dict
    baseline: tuple
    max_trips: int
    constraints: dict
    out_dir: Path
    parallel: int = 2


def load_clips(path):
    """`{slug: clip path}` from the corpus file; stops the run when a clip is missing."""
    clips = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = sorted(slug for slug, clip in clips.items() if not Path(clip).exists())
    if missing:
        raise SystemExit(f"--corpus: clips not found: {', '.join(missing)}")
    return {slug: str(Path(clip)) for slug, clip in clips.items()}


def guard_from_args(args, out_dir, baseline):
    """The guard `--corpus` asks for, reading against `baseline` (`(id, settings)`); None without one."""
    if not getattr(args, "corpus", None):
        return None
    constraints = load_constraints(args.grid)
    base_id, settings = baseline
    return Guard(
        load_clips(args.corpus), (base_id, dict(settings)), int(args.corpus_max_trips), constraints, Path(out_dir) / DIR, args.parallel
    )


def describe(guard):
    """One line for the run's header."""
    return f"corpus guard: {len(guard.clips)} sentinel clips, at most {guard.max_trips} may trip a constraint"


def wav_path(guard, cid, slug):
    """Where the loop's render keeps a candidate's audio on one clip (`autotune_restoration.run_candidate`)."""
    return guard.out_dir / "cands" / cid / f"{slug}.wav"


def report_path(guard, cid, slug):
    """The scorer's report of the baseline and the candidate on one clip."""
    return guard.out_dir / "reports" / cid / f"{slug}.json"


def rendered(render, engine, overrides, guard):
    """Renders `overrides` on every clip; returns the slugs whose render failed."""
    try:
        render(engine, overrides, guard.clips, guard.out_dir)
    except renders.RenderFailed as exc:
        return set(exc.reasons)
    return set()


def scorer_args(guard, ids, slug):
    """The scorer's command for one clip: the clip, the baseline's audio and the candidate's, dsp only."""
    base_id, cid = ids
    return [
        PYTHON,
        str(REPO / "scripts" / "validate_restoration.py"),
        guard.clips[slug],
        f"{BASE}={wav_path(guard, base_id, slug)}",
        f"{CAND}={wav_path(guard, cid, slug)}",
        "--metrics",
        "dsp",
        "--cache-dir",
        str(guard.out_dir / "cache"),
        "--report",
        str(report_path(guard, cid, slug)),
    ]


def scorer_env(clip):
    """The scorer's environment, the clip's folder joining AI_RESTORE_DATA_ROOTS (the scorer confines its paths)."""
    roots = [entry for entry in os.environ.get("AI_RESTORE_DATA_ROOTS", "").split(os.pathsep) if entry]
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "AI_RESTORE_DATA_ROOTS": os.pathsep.join([*roots, str(Path(clip).resolve().parent)])}


def score(guard, ids, slugs, launch=subprocess.Popen):
    """Scores each clip of `slugs` without a report yet, `guard.parallel` scorers at once."""
    pending = [slug for slug in slugs if not report_path(guard, ids[1], slug).exists()]
    (guard.out_dir / "reports" / ids[1]).mkdir(parents=True, exist_ok=True)
    step = max(1, guard.parallel)
    for start in range(0, len(pending), step):
        stop = start + step
        _score_batch(guard, ids, pending[start:stop], launch)


def _score_batch(guard, ids, batch, launch):
    """Launches one scorer per clip of `batch`, then waits for all of them."""
    procs = [
        launch(scorer_args(guard, ids, slug), cwd=str(REPO), env=scorer_env(guard.clips[slug]), stdout=subprocess.DEVNULL) for slug in batch
    ]
    for proc in procs:
        proc.wait()


def clip_rows(guard, cid, slug):
    """The material trips on one clip as `[key, baseline, candidate]` rows; one `UNSCORED` row without a report."""
    report = read_json(report_path(guard, cid, slug))
    if not report:
        return [[UNSCORED, None, None]]
    return [[key, before, after] for key, before, after, material in clip_trips(report, BASE, CAND, guard.constraints) if material]


def count_trips(guard, cid, scored, failed):
    """`{slug: rows}` of the clips the candidate trips: the scored ones past a constraint, the failed renders."""
    found = {slug: [[RENDER_FAILED, None, None]] for slug in sorted(failed)}
    for slug in scored:
        rows = clip_rows(guard, cid, slug)
        if rows:
            found[slug] = rows
    return found


def baseline_failed(guard, engine, render):
    """The clips the baseline fails to render, read once and kept (a clip it cannot render is left out of every count)."""
    path = guard.out_dir / BASE_FAILED_JSON
    stored = read_json(path)
    if "failed" in stored:
        return set(stored["failed"])
    failed = rendered(render, engine, guard.baseline[1], guard)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"failed": sorted(failed)}), encoding="utf-8")
    return failed


def trips(guard, engine, candidate, render):
    """`{slug: rows}` of the clips `candidate` (`(cid, overrides)`) trips, from its file or rendered and scored now."""
    cid, overrides = candidate
    stored_path = guard.out_dir / "cands" / cid / TRIPS_JSON
    stored = read_json(stored_path)
    if "trips" in stored:
        return stored["trips"]
    base_failed = baseline_failed(guard, engine, render)
    failed = rendered(render, engine, overrides, guard) - base_failed
    scored = [slug for slug in guard.clips if slug not in base_failed | failed]
    score(guard, (guard.baseline[0], cid), scored)
    found = count_trips(guard, cid, scored, failed)
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    stored_path.write_text(json.dumps({"trips": found, "left_out": sorted(base_failed)}, indent=1), encoding="utf-8")
    return found


def accepts(guard, engine, candidate, render, verdict):
    """Whether `candidate` holds the sentinels; records `corpus_trips` (the count) in its verdict."""
    found = trips(guard, engine, candidate, render)
    verdict["corpus_trips"] = len(found)
    return len(found) <= guard.max_trips


def finding(cid, verdict, guard):
    """The log line of a candidate the guard read, or None when it read none."""
    count = verdict.get("corpus_trips")
    if count is None:
        return None
    word = "refused" if count > guard.max_trips else "held"
    return f"corpus guard: `{cid}` tripped {count} of {len(guard.clips)} sentinel clips (at most {guard.max_trips}): {word}"
