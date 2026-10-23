"""Scoring for the self-driving loop (`scripts/autotune_restoration.py`): the families, the scorers and the two stages.

- Which families: `--families auto` (the loop's default) scores only the families the grid
  reads, dsp always (every gate and listener flag reads it, and the file.*, listener.* and src.*
  readings come from that pass) plus the family of every ranking entry, learned veto and gate
  name (`grid_families`). `tata_v3` reads dsp, speech and mos; `music_v3` dsp, stems and mos. The
  stems family separates every output with BS-RoFormer and was the dearest part of a speech
  round while `tata_v3` reads none of it: about 74 of the 113 minutes SOTI's scorer took in the
  APL v6 round (2026-10-10, 29 outputs). A named list is scored as given, in the two stages.
- Two stages (plan 1.5, `autotune_restoration.score_and_judge`): the families the ranking, the
  gates and the grid's constraints read (`split_families`) on every candidate; the rest, the
  learned vetoes' families, only on the incumbent and the candidates that qualified without the
  vetoes, in `<slug>.learned.score.json` beside the first pass's `<slug>.score.json`. A veto can
  only take a qualification away, so the verdicts equal a single pass over every family. A candidate whose
  first-pass flat already holds a family (scored before the stages existed) is not scored again.
- The scorers: one `scripts/validate_restoration.py` process per tape for every candidate of a
  batch, at most `parallel` at once (about 4 GB of RAM each), their reports flattened into each
  candidate's score file, the scored output's cache entries dropped.
"""

import json
import os
import re
import subprocess
import sys
from importlib import import_module
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

tune = import_module("scripts.tune_restoration")
audio_io = import_module("scripts.restoration_quality.audio_io")

PYTHON = str(REPO / ".venv" / "Scripts" / "python.exe") if (REPO / ".venv" / "Scripts" / "python.exe").exists() else sys.executable
SCORER_PARALLEL = 2
# `--families auto`: the families the grid reads (`grid_families`).
AUTO_FAMILIES = "auto"
# The second scoring stage: the families the ranking and the gates do not read, on the incumbent and the contenders.
LEARNED_STAGE = "learned"


def score_round(candidates, tapes, out_dir, families, gates, language, parallel=SCORER_PARALLEL, stage=""):
    """Scores every candidate on every tape, at most `parallel` tape processes at once; returns `{cid: {slug: flat}}`.

    A scorer holds about 4 GB of RAM with every model resident; two engines tuning at
    once with four scorers each ran a 62 GB machine out of memory, so the default is two.
    `stage` names a second scoring pass (`LEARNED_STAGE`), kept in its own score and report files.
    """
    (out_dir / "rounds").mkdir(parents=True, exist_ok=True)
    pending = _unscored(candidates, tapes, out_dir, stage)
    step = max(1, parallel)
    for start in range(0, len(pending), step):
        stop = start + step
        _score_batch(pending[start:stop], out_dir, (families, gates, language), stage)
    return _load_scores(candidates, tapes, out_dir, stage)


def _score_batch(batch, out_dir, scorer_options, stage=""):
    """One scorer per `(slug, tape, cids)` of the batch, all launched before the first is waited for."""
    procs = [(slug, cids, _launch_scorer(slug, tape, cids, out_dir, *scorer_options, stage=stage)) for slug, tape, cids in batch]
    for slug, cids, (log, proc) in procs:
        _harvest(slug, cids, log, proc, out_dir, stage)


def _load_scores(candidates, tapes, out_dir, stage=""):
    """`{cid: {slug: flat}}` read back from the score files."""
    return {
        cid: {slug: json.loads(_score_path(out_dir, cid, slug, stage).read_text(encoding="utf-8")) for slug in tapes} for cid in candidates
    }


def _unscored_ids(candidates, out_dir, slug, stage=""):
    return [cid for cid in candidates if not _score_path(out_dir, cid, slug, stage).exists()]


def _unscored(candidates, tapes, out_dir, stage=""):
    """`[(slug, tape, cids)]` for every tape that still has candidates without a score on it."""
    pending = [(slug, tape, _unscored_ids(candidates, out_dir, slug, stage)) for slug, tape in tapes.items()]
    return [row for row in pending if row[2]]


def _report_stem(slug, stage=""):
    """The scorer's report and log name in `rounds/`: `score_<slug>`, or `<stage>_<slug>` for a second pass."""
    return f"{stage or 'score'}_{slug}"


def _harvest(slug, cids, log, proc, out_dir, stage=""):
    proc.wait()
    log.close()
    rounds = out_dir / "rounds"
    if proc.returncode != 0:
        raise SystemExit(f"scoring {slug} failed (see {rounds / f'{_report_stem(slug, stage)}.log'})")
    report = json.loads((rounds / f"{_report_stem(slug, stage)}.json").read_text(encoding="utf-8"))
    for cid in cids:
        flat = tune.runner.aggregates_for_tuning(report, cid)
        _score_path(out_dir, cid, slug, stage).write_text(json.dumps(flat, indent=1), encoding="utf-8")
        _drop_cache_entries(out_dir, cid, slug)


def _drop_cache_entries(out_dir, cid, slug):
    """The scorer's cache entries for a scored output (its resampled arrays, its DC-free copy), which nothing reads again.

    A candidate output is scored once; its entries (three rates, ~1 GB per hour of tape) only
    fill the disk afterwards: two loops on full tapes grew 250 GB of them in three days and
    ran the machine out of space. The source side of the cache stays, every round reuses it.
    """
    wav = out_dir / "cands" / cid / f"{slug}.wav"
    if not wav.exists():
        return
    for entry in (out_dir / "cache").glob(f"{audio_io.file_key(wav)}_*"):
        entry.unlink()


def _score_path(out_dir, cid, slug, stage=""):
    """`cands/<cid>/<slug>.score.json`, or `<slug>.<stage>.score.json` for a second scoring pass."""
    return out_dir / "cands" / cid / (f"{slug}.{stage}.score.json" if stage else f"{slug}.score.json")


LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$")


def existing_path(value, what):
    """A command-line path resolved and checked to exist; a value shaped like an option is refused."""
    text = str(value)
    if text.startswith("-"):
        raise SystemExit(f"{what} looks like an option, not a path: {text}")
    resolved = Path(text).resolve()
    if not resolved.exists():
        raise SystemExit(f"{what} does not exist: {resolved}")
    return resolved


def _scorer_arguments(slug, tape, cids, out_dir, families, gates, language, stage=""):
    """The scorer's argument list from checked inputs: the tape and the gates must exist, the language is a code."""
    if not LANGUAGE_RE.match(language):
        raise SystemExit(f"--language must be a two- or three-letter code: {language!r}")
    unknown = set(families) - set(tune.runner.ALL_FAMILIES)
    if unknown:
        raise SystemExit(f"unknown metric families: {sorted(unknown)}")
    out_dir = existing_path(out_dir, "--out")
    args = [PYTHON, str(REPO / "scripts" / "validate_restoration.py"), str(existing_path(tape, "tape"))]
    args += [f"{cid}={out_dir / 'cands' / cid / f'{slug}.wav'}" for cid in cids]
    args += ["--metrics", ",".join(families), "--language", language, "--cache-dir", str(out_dir / "cache")]
    args += ["--report", str(out_dir / "rounds" / f"{_report_stem(slug, stage)}.json")]
    if gates:
        args += ["--gates", str(existing_path(gates, "--gates"))]
    return args


def _scorer_env(tape):
    """The scorer's environment: the tape's folder joins AI_RESTORE_DATA_ROOTS, since the scorer confines its paths."""
    roots = [entry for entry in os.environ.get("AI_RESTORE_DATA_ROOTS", "").split(os.pathsep) if entry]
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "AI_RESTORE_DATA_ROOTS": os.pathsep.join([*roots, str(Path(tape).resolve().parent)])}


def _launch_scorer(slug, tape, cids, out_dir, families, gates, language, stage=""):
    args = _scorer_arguments(slug, tape, cids, out_dir, families, gates, language, stage)
    log = open(out_dir / "rounds" / f"{_report_stem(slug, stage)}.log", "a", encoding="utf-8")
    proc = subprocess.Popen(args, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, env=_scorer_env(tape))
    return log, proc


def reading_family(key):
    """The scorer family that produces reading `key`: its prefix when that names a family, else dsp (file.*, listener.*, src.*)."""
    prefix = str(key).split(".", 1)[0]
    return prefix if prefix in tune.runner.ALL_FAMILIES else "dsp"


def grid_families(grid, gate_names=()):
    """The scorer families a grid reads, in scoring order: dsp always, plus each ranking entry's, veto's, constraint's and
    gate's family (the v4 constraints read dsp readings)."""
    keys = [*grid.get("ranking", {}), *(grid.get("vetoes") or {}), *(grid.get("constraints") or {}), *gate_names]
    wanted = {"dsp", *map(reading_family, keys)}
    return tuple(name for name in tune.runner.ALL_FAMILIES if name in wanted)


def gate_names(gates):
    """The gate names of a gates file (`{name: {threshold, ...}}`); none without one."""
    if gates is None:
        return ()
    return tuple(json.loads(existing_path(gates, "--gates").read_text(encoding="utf-8")))


def split_families(families, ranking, gate_names=(), constraints=()):
    """`(ranked, rest)` of the requested families: the ones the ranking, the gates and the grid's constraints read, scored
    on every candidate, and the others (the learned vetoes'), scored only on the incumbent and the candidates that beat it.

    A constraint is checked on every candidate in both judgements, so its family is a first-stage one: scored in the second
    stage only, a candidate that did not qualify would read as having lost the reading and be marked guarded."""
    needed = set(grid_families({"ranking": ranking, "constraints": dict.fromkeys(constraints)}, gate_names))
    return tuple(name for name in families if name in needed), tuple(name for name in families if name not in needed)


def resolve_families(families, grid, gates):
    """`--families` as given, or for `auto` the families the grid and the gates file read."""
    if tuple(families) != (AUTO_FAMILIES,):
        return tuple(families)
    return grid_families(grid, gate_names(gates))


def lacking(scores, ids, families):
    """The ids among `ids` whose flats do not hold every family yet; none when `families` is empty."""
    return [cid for cid in ids if families and not has_families(scores[cid], families)]


def has_families(tape_scores, families):
    """Whether every tape's flat already holds readings of each family (a candidate scored on every family at once)."""
    return all(any(reading_family(key) == name for key in flat) for flat in tape_scores.values() for name in families)


def merge_families(scores, learned, families):
    """Adds each second-pass flat's readings of `families` to the candidate's first-pass flat on the same tape."""
    for cid, tapes in learned.items():
        for slug, flat in tapes.items():
            scores[cid][slug].update({key: value for key, value in flat.items() if reading_family(key) in families})
