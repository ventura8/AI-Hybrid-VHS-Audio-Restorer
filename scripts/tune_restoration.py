#!/usr/bin/env python3
"""Tune cathar and auto_pure_linear on the user's own tapes, judged by the output-quality harness.

    tune_restoration.py prepare --name tata_v1 --tapes-dir "D:\\Tata\\New folder" --grid scripts/tune_grids/tata_v1.yaml
    tune_restoration.py run     --name tata_v1 [--engines cathar apl] [--variants cathar__baseline ...] [--limit N] [--force]
    tune_restoration.py score   --name tata_v1 [--metrics all|dsp,...] [--gates gates.json] [--rescore]
    tune_restoration.py report  --name tata_v1
    tune_restoration.py listen  --name tata_v1
    tune_restoration.py all     --name tata_v1 --tapes-dir ... --grid ...

Excerpts (125 s by default, so cathar's stitched noise probe engages) are cut from every
tape once; every variant of the grid is then run in a fresh interpreter with its own
config.yaml in the launch directory, which is how the app resolves configuration. Every
output is scored against its source excerpt with `scripts/restoration_quality`, the
hard gates veto variants that alter words, timbre, highs or background, and the rest
are rank-aggregated per metric. The scoreboard names the best variant per engine and
the best engine per tape, and prints the exact commands for a confirmation run on the
full tapes. Layout: experiments/tune_<name>/{manifest.json, excerpts/, runs/<variant>/,
scoreboard.json, scoreboard.md, listen/}.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from modules import config as config_mod  # noqa: E402
from modules.utils import FFMPEG_BIN, is_valid_video  # noqa: E402
from scripts.measure_tradeoff import _extract  # noqa: E402
from scripts.restoration_quality import gates as gates_mod  # noqa: E402
from scripts.restoration_quality import listening, runner  # noqa: E402

FFPROBE_BIN = str(Path(FFMPEG_BIN).with_name("ffprobe" + Path(FFMPEG_BIN).suffix)) if Path(FFMPEG_BIN).suffix else "ffprobe"
VIDEO_EXTENSIONS = (".mov", ".mp4", ".mkv", ".avi")
CUT_ARGS = [
    "-map",
    "0:v:0",
    "-map",
    "0:a:0",
    "-c:v",
    "libx264",
    "-preset",
    "ultrafast",
    "-crf",
    "35",
    "-vf",
    "scale=160:-2",
    "-r",
    "5",
    "-c:a",
    "copy",
]
PROBE_RATIO_KEY = "cathar_noiseprint_duration_s"
RUN_TIMEOUT_PER_EXCERPT_S = 900
DEFAULT_GATES = Path("experiments/quality_calibration/gates.json")
DEFAULT_TAPE_MAP = {"campanie": "campanie", "spitalul": "spitalul", "soti": "soti", "tele7abc": "tele7abc", "vaccinat": "vaccin"}


# ----------------------------------------------------------------------------- preparation


def probe_duration(path):
    """Seconds of media, or None when ffprobe cannot read it."""
    cmd = [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def plan_excerpts(duration_s, excerpt_s, margin_s):
    """`[(part, start_s)]`: start/middle/end when the tape holds three, two when it holds two, else one in the middle (margins waived)."""
    usable = duration_s - 2 * margin_s
    if duration_s < excerpt_s:
        return []
    if usable >= 3 * excerpt_s:
        return [("start", margin_s), ("middle", (duration_s - excerpt_s) / 2.0), ("end", duration_s - margin_s - excerpt_s)]
    if usable >= 2 * excerpt_s:
        return [("start", margin_s), ("end", duration_s - margin_s - excerpt_s)]
    return [("whole", (duration_s - excerpt_s) / 2.0)]


def slug_for(name, tape_map=DEFAULT_TAPE_MAP):
    """An ASCII slug for a tape from the first mapped keyword in its name, else its first word (48 chars at most)."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    for keyword, slug in tape_map.items():
        if keyword in folded:
            return slug
    return "".join(ch for ch in folded.split()[0] if ch.isalnum() or ch == "-")[:48].strip("-") or "tape"


def cut_excerpt(source, target, start_s, excerpt_s):
    """One excerpt with a tiny video track and the original audio stream copied bit for bit."""
    cmd = [FFMPEG_BIN, "-y", "-v", "error", "-ss", f"{start_s:.3f}", "-i", str(source), "-t", f"{excerpt_s:.3f}", *CUT_ARGS, str(target)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return target


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tapes(tapes_dir):
    """Source tapes in the folder: videos that are not themselves restored outputs."""
    return sorted(p for p in Path(tapes_dir).iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS and "_Cleaned" not in p.stem)


def _excerpt_rows(tape, grid, excerpts_dir, limit):
    duration = probe_duration(tape)
    rows = []
    for part, start_s in plan_excerpts(duration or 0.0, grid["excerpt_s"], grid["margin_s"]):
        target = excerpts_dir / f"{slug_for(tape.stem)}_{part}{tape.suffix.lower()}"
        if not target.exists():
            cut_excerpt(tape, target, start_s, grid["excerpt_s"])
        source_wav = (
            _extract(target, target.with_suffix(".src.wav"))
            if not target.with_suffix(".src.wav").exists()
            else target.with_suffix(".src.wav")
        )
        rows.append(
            {
                "tape": tape.name,
                "slug": slug_for(tape.stem),
                "part": part,
                "start_s": start_s,
                "excerpt": target.name,
                "source_wav": Path(source_wav).name,
                "probed_s": probe_duration(target),
                "sha256": sha256_of(target),
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def _whole_row(clip, slug, excerpts_dir):
    """One row for a clip taken whole (a corpus clip): hardlinked into the excerpts dir under its slug."""
    target = excerpts_dir / f"{slug}{clip.suffix.lower()}"
    if not target.exists():
        _link_or_copy(clip, target)
    source_wav = target.with_suffix(".src.wav")
    if not source_wav.exists():
        _extract(target, source_wav)
    return {
        "tape": clip.name,
        "slug": slug,
        "part": "whole",
        "start_s": 0.0,
        "excerpt": target.name,
        "source_wav": source_wav.name,
        "probed_s": probe_duration(target),
        "sha256": sha256_of(target),
    }


def _catalog_clips(tapes_dir, catalog):
    """`[(clip path, slug)]` from a corpus catalog (a list of records with `file` and `identifier`)."""
    records = json.loads(Path(catalog).read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = [r for region in records.values() for r in region]
    # The corpus catalog was written on Windows; its backslashes are separators on every platform.
    clips = [(Path(tapes_dir) / _portable(r["file"]), slug_for(r.get("identifier") or _portable(r["file"]).stem, {})) for r in records]
    return [(clip, slug) for clip, slug in clips if clip.exists()]


def _portable(catalog_path):
    return Path(str(catalog_path).replace(chr(92), "/"))


def build_manifest(tapes_dir, grid, out_dir, limit=0, catalog=None, whole=False):
    """Excerpts cut from every tape in the folder, or corpus clips taken whole (from a catalog or the folder)."""
    excerpts_dir = out_dir / "excerpts"
    excerpts_dir.mkdir(parents=True, exist_ok=True)
    if catalog or whole:
        clips = _catalog_clips(tapes_dir, catalog) if catalog else [(t, slug_for(t.stem, {})) for t in _tapes(tapes_dir)]
        rows = [_whole_row(clip, slug, excerpts_dir) for clip, slug in (clips[:limit] if limit else clips)]
    else:
        rows = []
        for tape in _tapes(tapes_dir):
            rows += _excerpt_rows(tape, grid, excerpts_dir, limit)
            if limit and len(rows) >= limit:
                break
    manifest = {
        "tapes_dir": str(tapes_dir),
        "excerpt_s": grid["excerpt_s"],
        "whole": bool(catalog or whole),
        "config_sha256": sha256_of(REPO / "config.yaml"),
        "excerpts": rows[:limit] if limit else rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


# ----------------------------------------------------------------------------- variants


def load_grid(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def known_config_fields():
    """`{name: (type, minimum, maximum)}` for every typed setting, plus the choice and string ones."""
    fields = {}
    for row in config_mod._NUMERIC_CONFIG_FIELDS:
        fields[row[0]] = (row[1], row[3] if len(row) > 3 else None, row[4] if len(row) > 4 else None)
    for name, _default in config_mod._BOOL_CONFIG_FIELDS:
        fields[name] = (bool, None, None)
    fields["cathar_denoise_method"] = (set(config_mod.VALID_CATHAR_DENOISE_METHODS), None, None)
    fields["apl_neural_model"] = (str, None, None)
    return fields


def _problem(name, value, spec):
    kind, low, high = spec
    if isinstance(kind, set):
        return None if value in kind else f"{name}: {value!r} is not one of {sorted(kind)}"
    if kind is bool:
        return None if isinstance(value, bool) else f"{name}: expected true/false, got {value!r}"
    if kind is str:
        return None if config_mod._is_bare_filename(str(value)) else f"{name}: model names must be bare filenames"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{name}: expected a number, got {value!r}"
    return _range_problem(name, value, low, high)


def _range_problem(name, value, low, high):
    if low is not None and value < low:
        return f"{name}: {value} is under the minimum {low}"
    if high is not None and value > high:
        return f"{name}: {value} is over the maximum {high}"
    return None


def validate_overrides(overrides, fields=None):
    """Problems with a variant's overrides against the app's own typed settings; empty when it is sound."""
    fields = fields or known_config_fields()
    problems = []
    for name, value in overrides.items():
        if name not in fields:
            problems.append(f"{name}: not a setting the app reads")
            continue
        problem = _problem(name, value, fields[name])
        if problem:
            problems.append(problem)
    return problems


def check_probe_material(grid, manifest):
    """cathar variants whose stitched probe would not engage on the shortest excerpt."""
    shortest = min([row["probed_s"] or 0.0 for row in manifest["excerpts"]], default=0.0)
    problems = []
    for engine, spec in grid["engines"].items():
        if spec["process_mode"] == "cathar":
            problems += _probe_problems(engine, grid["variants"].get(engine, {}), shortest)
    return problems


def _probe_problems(engine, variants, shortest):
    problems = []
    for name, overrides in variants.items():
        needed = 20.0 * float((overrides or {}).get(PROBE_RATIO_KEY, config_mod.CATHAR_NOISEPRINT_DURATION_S))
        if needed > shortest:
            problems.append(f"{engine}__{name}: needs {needed:.0f} s of material, shortest excerpt is {shortest:.1f} s")
    return problems


def engine_env(engine_spec):
    """Extra environment for an engine's child processes (`env:` in the grid); a value naming a repo path is made absolute."""
    env = {}
    for key, value in (engine_spec.get("env") or {}).items():
        candidate = REPO / str(value)
        env[key] = str(candidate.resolve()) if candidate.exists() else str(value)
    return env


def materialise_config(repo_config_text, process_mode, overrides):
    """The repo config.yaml with `process_mode` and the overrides applied; unknown keys survive the round trip."""
    config = yaml.safe_load(repo_config_text) or {}
    config["process_mode"] = process_mode
    config.update(overrides)
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def resolved_config(variant_dir, python, extra_env=None):
    """CONFIG as a fresh interpreter launched in `variant_dir` resolves it."""
    code = "import json, modules.config as c; print(json.dumps(c.CONFIG, default=str))"
    env = {**os.environ, **(extra_env or {}), "PYTHONPATH": str(REPO), "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([python, "-c", code], cwd=str(variant_dir), env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def assert_overrides_resolved(overrides, resolved):
    """Overrides the app silently reverted (a bound, a normaliser, a typo)."""
    reverted = []
    for name, value in overrides.items():
        seen = resolved.get(name)
        same = str(seen).lower() == str(value).lower() if isinstance(value, bool) else _close(seen, value)
        if not same:
            reverted.append(f"{name}: wanted {value!r}, app resolved {seen!r}")
    return reverted


def _close(seen, value):
    try:
        return abs(float(seen) - float(value)) < 1e-9
    except (TypeError, ValueError):
        return str(seen) == str(value)


def variant_ids(grid, engines=None, only=None):
    """`[(variant_id, engine, name, overrides)]` in grid order, filtered by engine and by explicit ids."""
    out = []
    for engine, variants in grid["variants"].items():
        if engines and engine not in engines:
            continue
        out += [(f"{engine}__{name}", engine, name, overrides or {}) for name, overrides in variants.items()]
    return [row for row in out if not only or row[0] in only]


# ----------------------------------------------------------------------------- execution


def link_excerpts(excerpts_dir, target_dir, manifest):
    """Hardlinks of the excerpts inside the variant directory (the app writes outputs beside its input)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in manifest["excerpts"]:
        source, target = excerpts_dir / row["excerpt"], target_dir / row["excerpt"]
        if not target.exists():
            _link_or_copy(source, target)
        paths.append(target)
    return paths


def _link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def output_path(excerpt, suffix):
    return excerpt.with_name(f"{excerpt.stem}{suffix}{excerpt.suffix}")


def variant_complete(variant_dir, manifest, suffix):
    return all(is_valid_video(output_path(variant_dir / "excerpts" / row["excerpt"], suffix)) for row in manifest["excerpts"])


def _clear_variant(variant_dir):
    for name in ("excerpts", "wav", "scores"):
        shutil.rmtree(variant_dir / name, ignore_errors=True)


def run_variant(variant_dir, engine_spec, overrides, manifest, excerpts_dir, python, force=False, keep_work=False):
    """Runs the app once, in a fresh interpreter, over every excerpt; returns the timing record."""
    if force:
        _clear_variant(variant_dir)
    suffix = engine_spec["suffix"]
    if variant_complete(variant_dir, manifest, suffix):
        return (
            json.loads((variant_dir / "timing.json").read_text(encoding="utf-8"))
            if (variant_dir / "timing.json").exists()
            else {"skipped": True}
        )
    variant_dir.mkdir(parents=True, exist_ok=True)
    # One candidate at a time: the app's own batch stays sequential inside a variant.
    overrides = {"batch_jobs": 1, **overrides}
    (variant_dir / "config.yaml").write_text(
        materialise_config((REPO / "config.yaml").read_text(encoding="utf-8"), engine_spec["process_mode"], overrides), encoding="utf-8"
    )
    extra_env = engine_env(engine_spec)
    resolved = resolved_config(variant_dir, python, extra_env)
    (variant_dir / "resolved.json").write_text(json.dumps(resolved, indent=1, default=str), encoding="utf-8")
    reverted = assert_overrides_resolved(overrides, resolved)
    if reverted:
        raise SystemExit(f"{variant_dir.name}: the app did not honour the overrides: {reverted}")
    excerpts = link_excerpts(excerpts_dir, variant_dir / "excerpts", manifest)
    timing = _launch(variant_dir, excerpts, python, keep_work, extra_env)
    (variant_dir / "timing.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
    return timing


def _launch(variant_dir, excerpts, python, keep_work, extra_env=None):
    env = {**os.environ, **(extra_env or {}), "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(REPO)}
    if keep_work:
        env["AI_RESTORE_TEST_MODE"] = "1"
    started = time.time()
    with open(variant_dir / "run.log", "w", encoding="utf-8") as log:
        subprocess.run(
            [python, str(REPO / "restore_audio_hybrid.py"), *(str(Path(e).resolve()) for e in excerpts)],
            cwd=str(variant_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=RUN_TIMEOUT_PER_EXCERPT_S * max(1, len(excerpts)),
            check=False,
        )
    wall = time.time() - started
    return {"wall_s": wall, "n": len(excerpts), "s_per_excerpt": wall / max(1, len(excerpts))}


# ----------------------------------------------------------------------------- scoring


def score_variant(variant_dir, engine_spec, manifest, excerpts_dir, families, registry, gates, cache_dir, rescore=False, language="ro"):
    """Scores every excerpt's output against its source; one JSON per pair under `scores/`."""
    scores_dir = variant_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for row in manifest["excerpts"]:
        excerpt = variant_dir / "excerpts" / row["excerpt"]
        target = scores_dir / f"{excerpt.stem}.json"
        if rescore or not target.exists():
            output = output_path(excerpt, engine_spec["suffix"])
            if not is_valid_video(output):
                continue
            result, _cards = runner.score_variants(
                excerpts_dir / row["source_wav"],
                {variant_dir.name: output},
                families=families,
                registry=registry,
                cache_dir=cache_dir,
                gates=gates,
                language=language,
            )
            if target.exists() and set(families) != set(runner.ALL_FAMILIES):
                result = merge_families(json.loads(target.read_text(encoding="utf-8")), result, families, gates)
            target.write_text(json.dumps(result, indent=1, default=runner._json_default), encoding="utf-8")
        results[excerpt.stem] = json.loads(target.read_text(encoding="utf-8"))
    return results


def merge_families(stored, fresh, families, gates):
    """`stored` with the re-scored `families` replaced from `fresh` (rows, aggregates, file metrics), verdicts re-read.

    A detector fix is then a re-score of one family, not of the four; the windows are the
    same because source, output and window settings are.
    """
    prefixes = tuple(f"{family}." for family in families)
    for label, variant in fresh["variants"].items():
        old = stored["variants"].get(label)
        if old is None or len(old["rows"]) != len(variant["rows"]):
            stored["variants"][label] = variant
        else:
            _merge_variant(old, variant, prefixes, gates)
    stored["metrics"].update(fresh["metrics"])
    return stored


def _merge_variant(old, fresh, prefixes, gates):
    def keep(mapping):
        return {k: v for k, v in mapping.items() if k.startswith(prefixes)}

    for old_row, new_row in zip(old["rows"], fresh["rows"]):
        for side in ("source", "output", "delta"):
            old_row[side].update(keep(new_row[side]))
    old["aggregate"].update(keep(fresh["aggregate"]))
    old["file"], old["families"] = fresh["file"], {**old["families"], **fresh["families"]}
    old["verdicts"] = gates_mod.evaluate_gates(old["aggregate"], gates, old.get("speaker_floor"))
    old["hard_failures"] = gates_mod.hard_failures(old["verdicts"])
    old["listener_flags"] = gates_mod.flag_failures(old["verdicts"])
    old["passed"] = not old["hard_failures"]


def flatten_pair(result, label, gates=None):
    """`{metric.side.stat: value}` plus the trade metric and the hard failures of one scored pair.

    With `gates` the verdicts are re-read from the stored aggregates, so a recalibrated
    gates.json changes the scoreboard without re-scoring every pair.
    """
    variant = result["variants"][label]
    if gates is not None:
        verdicts = gates_mod.evaluate_gates(variant["aggregate"], gates, variant.get("speaker_floor"))
        variant["hard_failures"] = gates_mod.hard_failures(verdicts)
        variant["listener_flags"] = gates_mod.flag_failures(verdicts)
        variant["passed"] = not variant["hard_failures"]
    flat = runner.aggregates_for_tuning(result, label)
    flat["hard_failures"] = variant["hard_failures"]
    return flat


# ----------------------------------------------------------------------------- aggregation


def _median_or_none(values):
    return float(np.median(values)) if values else None


def aggregate_variant(pairs, manifest):
    """Per metric medians (and p10 tails) across excerpts, plus per-tape medians and the gate failures."""
    by_slug = {row["excerpt"].rsplit(".", 1)[0]: row["slug"] for row in manifest["excerpts"]}
    metrics, per_tape, failures = {}, {}, {}
    for stem, flat in pairs.items():
        slug = by_slug.get(stem, stem)
        for key, value in flat.items():
            if key == "hard_failures":
                failures.setdefault(slug, []).extend([(stem, gate) for gate in value])
            elif value is not None:
                metrics.setdefault(key, []).append(value)
                per_tape.setdefault(slug, {}).setdefault(key, []).append(value)
    summary = {key: {"median": _median_or_none(v), "p10": float(np.percentile(v, 10))} for key, v in metrics.items()}
    tapes = {slug: {key: _median_or_none(v) for key, v in entries.items()} for slug, entries in per_tape.items()}
    return {"metrics": summary, "per_tape": tapes, "failures": failures, "n_excerpts": len(pairs)}


def apply_gates(summary, max_failed_excerpts, baseline_failed=0):
    """Vetoed when more excerpts failed a hard gate than the policy allows AND more than the engine's baseline.

    The gates were calibrated on three tapes; a 22-minute tape's lead-in fails them for
    every variant, the shipped defaults included. A variant is therefore judged against
    its own engine's baseline: it may not fail on more excerpts than the baseline does.
    Per tape any failure vetoes that tape for the per-tape recommendation.
    """
    failed_excerpts = {stem for stems in summary["failures"].values() for stem, _gate in stems}
    reasons = sorted({gate for stems in summary["failures"].values() for _stem, gate in stems})
    per_tape = {slug: bool(stems) for slug, stems in summary["failures"].items()}
    summary["failed_excerpts"] = len(failed_excerpts)
    return len(failed_excerpts) > max(max_failed_excerpts, baseline_failed), reasons, per_tape


def _ranks(values, direction):
    """Average ranks (1 = best) with ties averaged; None ranks last."""
    order = sorted(
        range(len(values)),
        key=lambda i: (values[i] is None, -values[i] if values[i] is not None and direction == "up" else values[i] or 0.0),
    )
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        same = [j for j in order[position:] if values[j] == values[order[position]]]
        for j in same:
            ranks[j] = position + (len(same) + 1) / 2.0
        position += len(same)
    return ranks


def rank_variants(summaries, ranking):
    """`{variant: rank_score}` = mean rank over the ranked metrics (lower is better) for every variant; the veto is applied later."""
    labels = list(summaries)
    scores = {label: [] for label in labels}
    for key, direction in ranking.items():
        values = [summaries[label]["metrics"].get(key, {}).get("median") for label in labels]
        if all(v is None for v in values):
            continue
        for label, rank in zip(labels, _ranks(values, direction)):
            scores[label].append(rank)
    return {label: float(np.mean(r)) if r else None for label, r in scores.items()}


def best_engine_per_tape(summaries, ranking):
    """For every tape, the variant (of either engine) that ranks best on that tape's own excerpts, ignoring vetoed tapes."""
    tapes = sorted({slug for summary in summaries.values() for slug in summary["per_tape"]})
    out = {}
    for slug in tapes:
        candidates = [
            label for label, summary in summaries.items() if slug in summary["per_tape"] and not summary["per_tape_vetoed"].get(slug)
        ]
        if not candidates:
            continue
        per_label = {label: summaries[label]["per_tape"][slug] for label in candidates}
        out[slug] = _best_on_tape(per_label, ranking)
    return out


def _best_on_tape(per_label, ranking):
    labels = list(per_label)
    totals = {label: [] for label in labels}
    for key, direction in ranking.items():
        values = [per_label[label].get(key) for label in labels]
        for label, rank in zip(labels, _ranks(values, direction)):
            totals[label].append(rank)
    scored = {label: float(np.mean(r)) for label, r in totals.items()}
    best = min(scored, key=scored.get)
    return {"variant": best, "engine": best.split("__")[0], "rank_score": scored[best]}


def config_diff(overrides):
    return dict(overrides)


# ----------------------------------------------------------------------------- scoreboard


def _table(labels, summaries, keys):
    head = "| variant | rank | failed excerpts | vetoed | " + " | ".join(keys) + " |"
    lines = [head, "|---|---|---|---|" + "---|" * len(keys)]
    for label in labels:
        summary = summaries[label]
        cells = [_cell(summary["metrics"].get(key)) for key in keys]
        veto = ", ".join(summary["veto_reasons"]) if summary["vetoed"] else "-"
        rank = "-" if summary.get("rank_score") is None else f"{summary['rank_score']:.2f}"
        lines.append(
            f"| {label} | {rank} | {summary.get('failed_excerpts', 0)}/{summary.get('n_excerpts', 0)} | {veto} | "
            + " | ".join(cells)
            + " |"
        )
    return lines


def _cell(entry):
    return "-" if not entry or entry["median"] is None else f"{entry['median']:+.3f} ({entry['p10']:+.3f})"


def render_scoreboard_md(board):
    """The scoreboard as Markdown: per-engine tables, per-tape winners, recommendations, confirmation commands."""
    ranking = list(board["ranking"])
    trade = ["file.noise_removed_db.output.median", "file.programme_deviation_db.output.median"]
    lines = [
        f"# Tuning scoreboard: {board['name']}",
        "",
        f"{len(board['excerpts'])} excerpts, config sha {board['config_sha256'][:12]}",
        "",
    ]
    for engine in board["engines"]:
        labels = sorted(
            (label for label in board["variants"] if label.startswith(engine + "__")),
            key=lambda label: (
                board["variants"][label]["vetoed"],
                board["variants"][label]["rank_score"] is None,
                board["variants"][label]["rank_score"] or 0.0,
            ),
        )
        lines += [f"## {engine}", "", "median (p10) across excerpts; rank order, vetoed last", ""]
        lines += _table(labels, board["variants"], ranking + trade) + [""]
    lines += (
        ["## Best variant per tape", ""]
        + [f"- **{slug}**: {entry['variant']} (rank {entry['rank_score']:.2f})" for slug, entry in board["per_tape_engine"].items()]
        + [""]
    )
    lines += ["## Recommendation", ""]
    for engine, rec in board["recommendation"].items():
        lines.append(
            f"- **{engine}**: {rec['best'] or 'no variant survived the gates'} (runner-up {rec['runner_up'] or '-'}); "
            f"overrides: `{json.dumps(rec['config_diff'])}`"
        )
    lines += ["", "## Warnings", ""] + [f"- {w}" for w in board["warnings"]] + ["", "## Confirmation run", ""] + board["confirmation"]
    return "\n".join(lines) + "\n"


def _recommendation(board, engine):
    ranked = [
        label for label in board["variants"] if label.startswith(engine + "__") and board["variants"][label]["rank_score"] is not None
    ]
    ranked.sort(key=lambda label: board["variants"][label]["rank_score"])
    survivors = [label for label in ranked if not board["variants"][label]["vetoed"]]
    best = survivors[0] if survivors else None
    return {
        "best": best,
        "runner_up": survivors[1] if len(survivors) > 1 else None,
        "config_diff": board["variants"][best]["overrides"] if best else {},
        "ranked_all": ranked,
    }


WARNINGS = (
    "Any change to a cathar default alters the five reference clips' outputs (experiments/cathar_ab_head.json bit-identity) "
    "unless it is gated by material length as the noise probe is; that is the user's call.",
    "The auto_cathar_* routing thresholds were calibrated with cathar on its shipped settings; a cathar change means re-reading them.",
    "An excerpt's noise probe is not the full tape's: confirm a winner on the full tapes before it becomes a default.",
)


def build_scoreboard(name, grid, manifest, summaries):
    ranking = grid["ranking"]
    rank_scores = rank_variants(summaries, ranking)
    for label, summary in summaries.items():
        summary["rank_score"] = rank_scores.get(label)
    board = {
        "name": name,
        "config_sha256": manifest["config_sha256"],
        "excerpts": manifest["excerpts"],
        "engines": list(grid["engines"]),
        "ranking": ranking,
        "variants": summaries,
    }
    board["per_tape_engine"] = best_engine_per_tape(summaries, ranking)
    board["recommendation"] = {engine: _recommendation(board, engine) for engine in grid["engines"]}
    board["warnings"] = list(WARNINGS)
    board["confirmation"] = _confirmation_lines(board, manifest)
    return board


def _confirmation_lines(board, manifest):
    lines = []
    for engine, rec in board["recommendation"].items():
        if rec["best"]:
            lines.append(
                f"- {engine}: write `{json.dumps(rec['config_diff'])}` into a config.yaml beside the tapes and run "
                f"`restore_audio_hybrid.py \"{manifest['tapes_dir']}\"` with process_mode {engine}"
            )
    return lines or ["- nothing to confirm"]


# ----------------------------------------------------------------------------- listening


def pick_worst_windows(variant_dir, manifest, metrics=listening.DEFAULT_PICK_METRICS, count=2):
    """`[(excerpt stem, Pick)]` over the variant's scored pairs."""
    picks = []
    for row in manifest["excerpts"]:
        target = variant_dir / "scores" / f"{Path(row['excerpt']).stem}.json"
        if not target.exists():
            continue
        result = json.loads(target.read_text(encoding="utf-8"))
        for metric in metrics:
            picks += [(Path(row["excerpt"]).stem, pick) for pick in listening.select_worst(result, metric, count)]
    return picks


# ----------------------------------------------------------------------------- CLI


def _out_dir(args):
    return Path("experiments") / f"tune_{args.name}"


def _load_state(args):
    out_dir = _out_dir(args)
    grid = load_grid(args.grid)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return out_dir, grid, manifest


def cmd_prepare(args):
    out_dir = _out_dir(args)
    grid = load_grid(args.grid)
    for engine, variants in grid["variants"].items():
        for name, overrides in variants.items():
            problems = validate_overrides(overrides or {})
            if problems:
                raise SystemExit(f"{engine}__{name}: {problems}")
    manifest = build_manifest(Path(args.tapes_dir), grid, out_dir, args.limit, args.catalog, args.whole)
    problems = check_probe_material(grid, manifest)
    if problems and not manifest["whole"]:
        raise SystemExit("\n".join(problems))
    for problem in problems:
        print(f"note: {problem}; on whole clips cathar keeps its single 0.75 s probe")
    print(f"{len(manifest['excerpts'])} excerpts under {out_dir / 'excerpts'}")


def _lock(out_dir):
    lock = out_dir / ".running"
    if lock.exists():
        raise SystemExit(f"{lock} exists: another driver is running (never run two restorations at once); remove it if that run died")
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return lock


def cmd_run(args):
    out_dir, grid, manifest = _load_state(args)
    lock = _lock(out_dir)
    try:
        for variant_id, engine, _name, overrides in variant_ids(grid, args.engines, args.variants):
            print(f"running {variant_id}", flush=True)
            timing = run_variant(
                out_dir / "runs" / variant_id,
                grid["engines"][engine],
                overrides,
                manifest,
                out_dir / "excerpts",
                args.python,
                args.force,
                args.keep_work,
            )
            print(f"  {timing}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


def cmd_score(args):
    out_dir, grid, manifest = _load_state(args)
    families = runner.ALL_FAMILIES if args.metrics == "all" else tuple(args.metrics.split(","))
    gates = gates_mod.load_gates(args.gates) if args.gates else gates_mod.GATES
    registry = runner.ModelRegistry(device=args.device)
    for variant_id, engine, _name, _overrides in variant_ids(grid, args.engines, args.variants):
        print(f"scoring {variant_id}", flush=True)
        score_variant(
            out_dir / "runs" / variant_id,
            grid["engines"][engine],
            manifest,
            out_dir / "excerpts",
            families,
            registry,
            gates,
            out_dir / "cache",
            args.rescore,
            args.language,
        )


def cmd_report(args):
    out_dir, grid, manifest = _load_state(args)
    gates = gates_mod.load_gates(args.gates) if args.gates else gates_mod.GATES
    summaries = {}
    for variant_id, _engine, _name, overrides in variant_ids(grid, args.engines, args.variants):
        pairs = _pairs_for(out_dir / "runs" / variant_id, variant_id, manifest, gates)
        if not pairs:
            continue
        summary = aggregate_variant(pairs, manifest)
        summary["overrides"] = overrides
        summaries[variant_id] = summary
    baseline_failed = {engine: _failed_count(summaries.get(f"{engine}__baseline")) for engine in grid["engines"]}
    for variant_id, summary in summaries.items():
        allowed = baseline_failed.get(variant_id.split("__")[0], 0)
        summary["vetoed"], summary["veto_reasons"], summary["per_tape_vetoed"] = apply_gates(
            summary, grid["gates"]["max_failed_excerpts"], allowed
        )
    board = build_scoreboard(args.name, grid, manifest, summaries)
    (out_dir / "scoreboard.json").write_text(json.dumps(board, indent=1, default=str), encoding="utf-8")
    (out_dir / "scoreboard.md").write_text(render_scoreboard_md(board), encoding="utf-8")
    print(render_scoreboard_md(board))


def _failed_count(summary):
    """Excerpts on which a summary's variant failed any hard gate (0 when the variant was not scored)."""
    if not summary:
        return 0
    return len({stem for stems in summary["failures"].values() for stem, _gate in stems})


def _pairs_for(variant_dir, label, manifest, gates=None):
    pairs = {}
    for row in manifest["excerpts"]:
        target = variant_dir / "scores" / f"{Path(row['excerpt']).stem}.json"
        if target.exists():
            pairs[Path(row["excerpt"]).stem] = flatten_pair(json.loads(target.read_text(encoding="utf-8")), label, gates)
    return pairs


LISTEN_METRICS = ("mos.sigmos_col", "mos.sigmos_disc", "speech.cer", "dsp.hf_8k16k")


def _listen_variants(out_dir, grid, args):
    """The variants worth an ear: the explicit list, else each engine's baseline, best and runner-up from the scoreboard."""
    if args.variants:
        return args.variants
    chosen = [f"{engine}__baseline" for engine in grid["engines"]]
    board_path = out_dir / "scoreboard.json"
    if board_path.exists():
        for rec in json.loads(board_path.read_text(encoding="utf-8"))["recommendation"].values():
            chosen += [label for label in (rec["best"], rec["runner_up"]) if label]
    return list(dict.fromkeys(chosen))


def _excerpt_row(manifest, stem):
    return next(r for r in manifest["excerpts"] if Path(r["excerpt"]).stem == stem)


def cmd_listen(args):
    """Cuts the worst windows of the chosen variants from source and output into listen/<variant>/<excerpt>/."""
    out_dir, grid, manifest = _load_state(args)
    listen_dir = out_dir / "listen"
    lines = [
        "# Listening picks",
        "",
        "Each pick: the window where the variant moved a metric furthest the wrong way, source beside output.",
        "",
    ]
    suffixes = {engine: spec["suffix"] for engine, spec in grid["engines"].items()}
    for variant_id in _listen_variants(out_dir, grid, args):
        picks = pick_worst_windows(out_dir / "runs" / variant_id, manifest, LISTEN_METRICS, 1)
        for stem in sorted({stem for stem, _pick in picks}):
            row = _excerpt_row(manifest, stem)
            outputs = {
                variant_id: output_path(out_dir / "runs" / variant_id / "excerpts" / row["excerpt"], suffixes[variant_id.split("__")[0]])
            }
            chosen = [pick for s, pick in picks if s == stem]
            rendered = listening.render_from_files(
                chosen, out_dir / "excerpts" / row["source_wav"], outputs, listen_dir / variant_id / stem
            )
            lines += [f"## {variant_id} / {stem}", ""] + rendered + [""]
    listen_dir.mkdir(parents=True, exist_ok=True)
    (listen_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "run", "score", "report", "listen", "all"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--tapes-dir", default=None)
    parser.add_argument("--grid", type=Path, default=Path("scripts/tune_grids/tata_v1.yaml"))
    parser.add_argument("--engines", nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--metrics", default="all")
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES if DEFAULT_GATES.exists() else None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--catalog", type=Path, default=None, help="corpus catalog: take its clips whole")
    parser.add_argument("--whole", action="store_true", help="take every video in --tapes-dir whole instead of cutting excerpts")
    parser.add_argument("--language", default="ro", help="Whisper language for the speech family")
    return parser.parse_args(argv)


COMMANDS = {"prepare": cmd_prepare, "run": cmd_run, "score": cmd_score, "report": cmd_report, "listen": cmd_listen}


def main(argv=None):
    args = _parse_args(argv)
    steps = ("prepare", "run", "score", "report", "listen") if args.command == "all" else (args.command,)
    for step in steps:
        COMMANDS[step](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
