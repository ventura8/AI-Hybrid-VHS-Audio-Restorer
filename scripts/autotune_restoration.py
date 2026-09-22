"""Self-driving fine-tuning of one engine on whole tapes, judged by the output-quality harness.

Each round proposes one-knob moves from the incumbent settings (the neighbours of every
knob's current value, plus the combination of every move that helped on its own), restores
every tape with each candidate in a fresh interpreter, keeps only the audio, scores it with
`scripts/restoration_quality` (all tapes of one candidate in parallel) and ranks candidates
against the incumbent with the grid's `ranking` metrics. A candidate replaces the incumbent
when its mean rank across tapes is better, it wins on at least half the tapes and it fails
no more hard gates than the incumbent. The loop stops when no candidate qualifies (a
plateau); `--rounds` (default 20) is only a safety cap. The listener is asked at the plateau.

usage:
  autotune_restoration.py --engine cathar|apl --tapes tapes.json [--rounds 20] [--out experiments/autotune]
                          [--grid scripts/tune_grids/tata_v1.yaml] [--families dsp,stems,speech,mos]
                          [--start '{"cathar_alpha": 2.0}' | --start-file final.json] [--language ro] [--gates gates.json]
                          [--parallel 2]

`tapes.json` maps a slug to a video path. Everything is resumable: candidates are keyed by
their settings, and a scored candidate is never run or scored twice.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import tune_restoration as tune  # noqa: E402
from scripts.restoration_quality import audio_io  # noqa: E402

ENV_PREFIX = "env:"
ROFORMER = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
CATHAR_076 = "experiments/cathar-0.7.6/cathar.exe"
SCORER_PARALLEL = 2

# Ordered candidate values per knob; None means "the app's default" (the override is dropped).
KNOBS = {
    "cathar": {
        # 0.5 and 1.0, and no print at all, exist for music: on a music-dominant clip the print
        # learned from the music's own quietest window removes no noise and only shaves the top
        # octave (alpha 1.5: -14 dB at 8-16 kHz, quiet frames +1.3 dB; alpha 0.5: -1.6 dB).
        "cathar_alpha": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "cathar_enable_noiseprint": [True, False],
        "cathar_beta": [0.005, 0.01, 0.02],
        "cathar_deesser_threshold": [6.0, 9.0, 12.0],
        "cathar_enable_deesser": [True, False],
        "cathar_noiseprint_duration_s": [4.5, 6.0],
        "cathar_enable_coherent": [True, False],
        "cathar_enable_enhance": [True, False],
        "cathar_enable_deplosive": [True, False],
        "cathar_repair_strength": [2, 4],
        f"{ENV_PREFIX}AI_RESTORE_CATHAR_BIN": [None, CATHAR_076],
    },
    # On the Tata tapes the scanner reads a spectral flatness of 0.022 (< apl_tonal_flatness_max 0.035),
    # so APL takes its tonal path: apl_spectral_alpha_tonal and apl_noiseprint_tonal_s are the live
    # knobs there (apl_spectral_alpha and apl_noiseprint_duration_s produced bit-identical audio),
    # and the flatness threshold itself decides whether an interview is treated as tonal at all.
    "apl": {
        "apl_spectral_alpha_tonal": [1.5, 2.0, 2.5, 3.0],
        "apl_noiseprint_tonal_s": [2.5, 4.0, 6.0],
        "apl_tonal_flatness_max": [0.01, 0.035],
        "enable_linear_air": [True, False],
        "linear_air_gain_db": [1.0, 2.0],
        "apl_neural_model": [None, ROFORMER],
        "apl_enable_learned_blend": [True, False],
        "enable_dynamic_expander": [True, False],
        "apl_enable_spectral_denoise": [True, False],
        "apl_use_native_suppress": [False, True],
    },
}
ENGINES = {
    "cathar": {"process_mode": "cathar", "suffix": "_Cathar_Cleaned"},
    "apl": {"process_mode": "auto_pure_linear", "suffix": "_PureLinear_Cleaned"},
}
PYTHON = str(REPO / ".venv" / "Scripts" / "python.exe") if (REPO / ".venv" / "Scripts" / "python.exe").exists() else sys.executable


# ----------------------------------------------------------------------------- candidates


def candidate_id(overrides):
    """A stable short id for a set of overrides (None values dropped first)."""
    clean = {k: v for k, v in sorted(overrides.items()) if v is not None}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]


def split_overrides(overrides):
    """`(config_overrides, env)`: `env:` keys become environment for the app's children."""
    config = {k: v for k, v in overrides.items() if not k.startswith(ENV_PREFIX) and v is not None}
    env = {k[len(ENV_PREFIX) :]: v for k, v in overrides.items() if k.startswith(ENV_PREFIX) and v is not None}
    return config, env


def seed_defaults(engine, start, out_dir):
    """`start` completed with the app's own value for every numeric knob it does not name.

    A knob whose candidates include None (the app's default) needs no seeding; the others
    are read back from a child interpreter so the incumbent is the real starting point and
    a neighbour never re-runs the default under another name.
    """
    seeded = dict(start)
    missing = [knob for knob, values in KNOBS[engine].items() if knob not in seeded and None not in values]
    if not missing:
        return seeded
    probe = out_dir / "defaults"
    probe.mkdir(parents=True, exist_ok=True)
    (probe / "config.yaml").write_text(
        tune.materialise_config((REPO / "config.yaml").read_text(encoding="utf-8"), ENGINES[engine]["process_mode"], {}), encoding="utf-8"
    )
    resolved = tune.resolved_config(probe, PYTHON)
    for knob in missing:
        match = _matching_value(resolved.get(knob), KNOBS[engine][knob])
        if match is not None:
            seeded[knob] = match
    return seeded


def _matching_value(seen, values):
    """The candidate value the app resolved to, or None."""
    for value in values:
        same = str(seen).lower() == str(value).lower() if isinstance(value, bool) else tune._close(seen, value)
        if same:
            return value
    return None


def neighbour_moves(engine, incumbent):
    """`[(knob, value)]`: the values adjacent to each knob's current value (or every value when unset)."""
    moves = []
    for knob, values in KNOBS[engine].items():
        current = incumbent.get(knob, None)
        if current not in values:
            moves += [(knob, v) for v in values if v != current]
            continue
        index = values.index(current)
        moves += [(knob, values[i]) for i in (index - 1, index + 1) if 0 <= i < len(values)]
    return moves


def propose(engine, incumbent):
    """`{id: overrides}` for every single-knob neighbour of the incumbent."""
    proposals = {}
    for knob, value in neighbour_moves(engine, incumbent):
        overrides = {**incumbent, knob: value}
        proposals[candidate_id(overrides)] = overrides
    proposals.pop(candidate_id(incumbent), None)
    return proposals


def combine(incumbent, winners):
    """One candidate carrying every move that beat the incumbent on its own (two or more of them)."""
    if len(winners) < 2:
        return {}
    overrides = dict(incumbent)
    for candidate in winners:
        overrides.update({k: v for k, v in candidate.items() if candidate.get(k) != incumbent.get(k)})
    return {candidate_id(overrides): overrides} if candidate_id(overrides) != candidate_id(incumbent) else {}


# ----------------------------------------------------------------------------- running


def run_candidate(engine, overrides, tapes, out_dir, language):
    """Restores every tape with `overrides` and keeps the audio as WAV under `out_dir/<id>/`; returns the WAV paths."""
    cid = candidate_id(overrides)
    cand_dir = out_dir / "cands" / cid
    wavs = {slug: cand_dir / f"{slug}.wav" for slug in tapes}
    if all(w.exists() for w in wavs.values()):
        return wavs
    config, env = split_overrides(overrides)
    config = {"batch_jobs": 1, **config}  # one candidate at a time; the app's own batch stays sequential inside it
    spec = {**ENGINES[engine], "env": env}
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "overrides.json").write_text(json.dumps(overrides, indent=1), encoding="utf-8")
    problems = tune.validate_overrides(config)
    if problems:
        raise SystemExit(f"{cid}: {problems}")
    (cand_dir / "config.yaml").write_text(
        tune.materialise_config((REPO / "config.yaml").read_text(encoding="utf-8"), spec["process_mode"], config), encoding="utf-8"
    )
    extra_env = tune.engine_env(spec)
    reverted = tune.assert_overrides_resolved(config, tune.resolved_config(cand_dir, PYTHON, extra_env))
    if reverted:
        raise SystemExit(f"{cid}: the app did not honour {reverted}")
    started = time.time()
    with open(cand_dir / "run.log", "w", encoding="utf-8") as log:
        subprocess.run(
            [PYTHON, str(REPO / "restore_audio_hybrid.py"), *(str(Path(p)) for p in tapes.values())],
            cwd=str(cand_dir),
            env={**os.environ, **extra_env, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(REPO)},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=tune.RUN_TIMEOUT_PER_EXCERPT_S * 4 * max(1, len(tapes)),
            check=False,
        )
    for slug, tape in tapes.items():
        produced = Path(tape).with_name(f"{Path(tape).stem}{spec['suffix']}{Path(tape).suffix}")
        if not produced.exists():
            raise SystemExit(f"{cid}: no output for {slug} ({produced.name})")
        shutil.move(str(audio_io.extract_wav(produced, cand_dir)), str(wavs[slug]))
        produced.unlink()
    (cand_dir / "timing.json").write_text(json.dumps({"wall_s": time.time() - started}), encoding="utf-8")
    return wavs


def score_round(candidates, tapes, out_dir, families, gates, language, parallel=SCORER_PARALLEL):
    """Scores every candidate on every tape, at most `parallel` tape processes at once; returns `{cid: {slug: flat}}`.

    A scorer holds about 4 GB of RAM with every model resident; two engines tuning at
    once with four scorers each ran a 62 GB machine out of memory, so the default is two.
    """
    (out_dir / "rounds").mkdir(parents=True, exist_ok=True)
    pending = [(slug, tape, [cid for cid in candidates if not _score_path(out_dir, cid, slug).exists()]) for slug, tape in tapes.items()]
    pending = [(slug, tape, cids) for slug, tape, cids in pending if cids]
    for start in range(0, len(pending), max(1, parallel)):
        batch = pending[start : start + max(1, parallel)]
        procs = [(slug, cids, _launch_scorer(slug, tape, cids, out_dir, families, gates, language)) for slug, tape, cids in batch]
        for slug, cids, (log, proc) in procs:
            _harvest(slug, cids, log, proc, out_dir)
    return {cid: {slug: json.loads(_score_path(out_dir, cid, slug).read_text(encoding="utf-8")) for slug in tapes} for cid in candidates}


def _harvest(slug, cids, log, proc, out_dir):
    proc.wait()
    log.close()
    if proc.returncode != 0:
        raise SystemExit(f"scoring {slug} failed (see {out_dir / 'rounds' / f'score_{slug}.log'})")
    report = json.loads((out_dir / "rounds" / f"score_{slug}.json").read_text(encoding="utf-8"))
    for cid in cids:
        flat = tune.runner.aggregates_for_tuning(report, cid)
        _score_path(out_dir, cid, slug).write_text(json.dumps(flat, indent=1), encoding="utf-8")


def _score_path(out_dir, cid, slug):
    return out_dir / "cands" / cid / f"{slug}.score.json"


def _launch_scorer(slug, tape, cids, out_dir, families, gates, language):
    args = [PYTHON, str(REPO / "scripts" / "validate_restoration.py"), str(tape)]
    args += [f"{cid}={out_dir / 'cands' / cid / f'{slug}.wav'}" for cid in cids]
    args += ["--metrics", ",".join(families), "--language", language, "--cache-dir", str(out_dir / "cache")]
    args += ["--report", str(out_dir / "rounds" / f"score_{slug}.json")]
    if gates:
        args += ["--gates", str(gates)]
    log = open(out_dir / "rounds" / f"score_{slug}.log", "a", encoding="utf-8")
    proc = subprocess.Popen(args, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return log, proc


# ----------------------------------------------------------------------------- judging


def rank_on_tape(flats, ranking):
    """`{cid: rank_score}` on one tape from flat aggregates, with the grid's directions."""
    summaries = {cid: {"metrics": {key: {"median": value} for key, value in flat.items()}} for cid, flat in flats.items()}
    return tune.rank_variants(summaries, ranking)


def judge(scores, incumbent_id, ranking):
    """`{cid: {"mean_rank", "wins", "failures", "per_tape", "qualifies"}}` against the incumbent across tapes."""
    tapes = list(next(iter(scores.values())))
    per_tape = {slug: rank_on_tape({cid: scores[cid][slug] for cid in scores}, ranking) for slug in tapes}
    verdicts = {cid: _verdict(cid, scores[cid], per_tape, incumbent_id) for cid in scores}
    base = verdicts[incumbent_id]
    for cid, verdict in verdicts.items():
        verdict["qualifies"] = cid != incumbent_id and _beats(verdict, base, len(tapes))
    return verdicts


def _verdict(cid, tape_scores, per_tape, incumbent_id):
    ranks = {slug: per_tape[slug][cid] for slug in tape_scores}
    known = [r for r in ranks.values() if r is not None]
    wins = sum(1 for slug, r in ranks.items() if r is not None and r < per_tape[slug][incumbent_id])
    failures = sum(flat.get("gates.hard_failures", 0.0) for flat in tape_scores.values())
    return {"mean_rank": float(np.mean(known)) if known else None, "wins": wins, "failures": failures, "per_tape": ranks}


def _beats(verdict, base, n_tapes):
    if verdict["mean_rank"] is None or base["mean_rank"] is None:
        return False
    return verdict["mean_rank"] < base["mean_rank"] and verdict["wins"] * 2 >= n_tapes and verdict["failures"] <= base["failures"]


def describe(overrides, incumbent):
    changed = {k: v for k, v in overrides.items() if incumbent.get(k) != v}
    return json.dumps(changed if changed else overrides, default=str) if overrides else "defaults"


# ----------------------------------------------------------------------------- driver


def load_state(out_dir, start):
    path = out_dir / "state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"incumbent": start, "rounds": [], "candidates": {}}


def save_state(out_dir, state):
    (out_dir / "state.json").write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")


def run_round(args, engine, tapes, out_dir, state, ranking, number):
    """One round: neighbours, runs, scores, the combination of the winners, the verdict; True when the incumbent moved."""
    incumbent = state["incumbent"]
    incumbent_id = candidate_id(incumbent)
    proposals = propose(engine, incumbent)
    if not proposals:
        return False
    everything = {incumbent_id: incumbent, **proposals}
    print(f"round {number}: incumbent {incumbent_id} {describe(incumbent, {})}; {len(proposals)} candidates", flush=True)
    verdicts = _run_and_judge(args, engine, everything, tapes, out_dir, state, ranking, incumbent)
    winners = [everything[cid] for cid, v in verdicts.items() if v["qualifies"]]
    combo = combine(incumbent, winners)
    if combo:
        everything.update(combo)
        verdicts = _run_and_judge(args, engine, everything, tapes, out_dir, state, ranking, incumbent)
    ordered = sorted(verdicts.items(), key=lambda kv: (kv[1]["mean_rank"] is None, kv[1]["mean_rank"] or 0.0))
    qualifying = [cid for cid, v in ordered if v["qualifies"]]
    state["rounds"].append(
        {"round": number, "incumbent": incumbent_id, "verdicts": verdicts, "accepted": qualifying[0] if qualifying else None}
    )
    if qualifying:
        state["incumbent"] = everything[qualifying[0]]
    _report_round(out_dir, number, incumbent, everything, ordered, qualifying, len(tapes))
    save_state(out_dir, state)
    return bool(qualifying)


def _run_and_judge(args, engine, everything, tapes, out_dir, state, ranking, incumbent):
    state["candidates"].update(everything)
    save_state(out_dir, state)
    for cid, overrides in everything.items():
        print(f"  running {cid} {describe(overrides, incumbent)}", flush=True)
        run_candidate(engine, overrides, tapes, out_dir, args.language)
    scores = score_round(everything, tapes, out_dir, args.families, args.gates, args.language, args.parallel)
    return judge(scores, candidate_id(incumbent), ranking)


def _report_round(out_dir, number, incumbent, everything, ordered, qualifying, n_tapes):
    lines = [f"## round {number}", "", f"incumbent `{candidate_id(incumbent)}` {describe(incumbent, {})}", ""]
    lines += ["| candidate | change | mean rank | wins | hard failures | qualifies |", "|---|---|---|---|---|---|"]
    for cid, v in ordered:
        rank = "-" if v["mean_rank"] is None else f"{v['mean_rank']:.2f}"
        flag = "yes" if v["qualifies"] else "-"
        lines.append(f"| {cid} | {describe(everything[cid], incumbent)} | {rank} | {v['wins']}/{n_tapes} | {v['failures']:.0f} | {flag} |")
    outcome = (
        f"accepted `{qualifying[0]}` -> {describe(everything[qualifying[0]], incumbent)}"
        if qualifying
        else "no candidate qualified: plateau"
    )
    lines += ["", outcome, ""]
    with open(out_dir / "log.md", "a", encoding="utf-8") as log:
        log.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=list(KNOBS), required=True)
    parser.add_argument("--tapes", type=Path, required=True, help="JSON {slug: video path}")
    parser.add_argument("--out", type=Path, default=Path("experiments/autotune"))
    parser.add_argument("--grid", type=Path, default=Path("scripts/tune_grids/tata_v1.yaml"))
    # The loop stops at a plateau (user: "stop only when no more improvements are possible"); the cap is a safety net.
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--families", default="dsp,stems,speech,mos")
    parser.add_argument("--start", default="{}", help="JSON overrides to start from")
    parser.add_argument("--start-file", type=Path, default=None, help="a final.json of an earlier run to start from (wins over --start)")
    parser.add_argument("--language", default="ro")
    parser.add_argument("--parallel", type=int, default=SCORER_PARALLEL, help="tape scorers running at once (about 4 GB RAM each)")
    parser.add_argument("--gates", type=Path, default=tune.DEFAULT_GATES if tune.DEFAULT_GATES.exists() else None)
    args = parser.parse_args(argv)
    args.families = tuple(args.families.split(","))
    tapes = {slug: str(Path(p)) for slug, p in json.loads(args.tapes.read_text(encoding="utf-8")).items()}
    out_dir = args.out / args.engine
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking = tune.load_grid(args.grid)["ranking"]
    start = json.loads(args.start_file.read_text(encoding="utf-8")) if args.start_file else json.loads(args.start)
    state = load_state(out_dir, seed_defaults(args.engine, start, out_dir))
    # A knob added to the table after the run began is seeded too, so its first moves are neighbours of the real value.
    state["incumbent"] = seed_defaults(args.engine, state["incumbent"], out_dir)
    if not (out_dir / "log.md").exists():
        (out_dir / "log.md").write_text(f"# autotune {args.engine}\n\ntapes: {', '.join(tapes)}\n\n", encoding="utf-8")
    for number in range(len(state["rounds"]) + 1, args.rounds + 1):
        if not run_round(args, args.engine, tapes, out_dir, state, ranking, number):
            break
    final = state["incumbent"]
    print(f"final {args.engine}: {candidate_id(final)} {describe(final, {})}", flush=True)
    (out_dir / "final.json").write_text(json.dumps(final, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
