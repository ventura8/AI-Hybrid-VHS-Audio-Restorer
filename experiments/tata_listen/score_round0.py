"""Round 0 of ear v3 (plan 1.6): re-score every file the verdict ledger judged with the v3 readings.

usage: score_round0.py [--sets r1_known,r1_listen,r2,r3] [--metrics dsp] [--dry-run]

The ledger (`assets/quality_calibration/verdicts.jsonl`) names, per record, a source and the
outputs the user heard. Records sharing a source are scored together, one report per tape and
set: `experiments/tata_listen/scores_v3/<set>/<tape>.json` (and `.md`), so
`check_verdicts.py --scores scores_v3/<set>` asserts each round on the files it judged. The sets:
r1_known (the 5-minute known-ordering excerpts), r1_listen (the round-one full-tape variants),
r2 (the v2 plateaus, `variants/v2`) and r3 (the air-shelf set, `variants/v4_air`). Only the dsp
family is scored by default: every v3 reading (R0, R1, R2, R4, R7, sync) lives there. The pair
cache is shared with `score_listen.py` (`experiments/tata_listen/cache`).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = str(REPO / ".venv" / "Scripts" / "python.exe")
LEDGER = REPO / "assets" / "quality_calibration" / "verdicts.jsonl"
OUT = REPO / "experiments" / "tata_listen" / "scores_v3"
SETS = ("r1_known", "r1_listen", "r2", "r3")


def set_of(source_path):
    """The set a record belongs to, from where its source lives."""
    path = source_path.replace("\\", "/")
    if "/known/" in path:
        return "r1_known"
    if "/variants/v2/" in path:
        return "r2"
    if "/variants/v4_air/" in path:
        return "r3"
    return "r1_listen"


def groups(records):
    """`{(set, tape): {"source": path, "outputs": {label: path}}}` over every record with a source."""
    found = {}
    for record in records:
        files = {f["label"]: f["path"] for f in record["files"]}
        source = files.pop("source", None)
        if source is None:
            continue
        group = found.setdefault((set_of(source), record["tape"]), {"source": source, "outputs": {}})
        if group["source"] != source:
            raise SystemExit(f"{record['id']}: two sources for one tape and set ({group['source']} / {source})")
        group["outputs"].update(files)
    return found


def command(key, group, metrics):
    """The validate_restoration.py call that scores one group."""
    set_name, tape = key
    folder = OUT / set_name
    labels = [f"{label}={_absolute(path)}" for label, path in sorted(group["outputs"].items())]
    return [
        PYTHON,
        str(REPO / "scripts" / "validate_restoration.py"),
        str(_absolute(group["source"])),
        *labels,
        "--metrics",
        metrics,
        "--language",
        "ro",
        "--gates",
        str(REPO / "experiments" / "quality_calibration" / "gates.json"),
        "--cache-dir",
        str(REPO / "experiments" / "tata_listen" / "cache"),
        "--markdown",
        str(folder / f"{tape}.md"),
        "--report",
        str(folder / f"{tape}.json"),
    ]


def _absolute(path):
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def _env():
    roots = [os.environ.get("AI_RESTORE_DATA_ROOTS", ""), r"D:\Tata\New folder", str(REPO / "experiments")]
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "AI_RESTORE_DATA_ROOTS": os.pathsep.join(filter(None, roots))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", default=",".join(SETS))
    parser.add_argument("--metrics", default="dsp")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    wanted = set(args.sets.split(","))
    records = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    failed = []
    for key, group in sorted(groups(records).items()):
        if key[0] not in wanted:
            continue
        (OUT / key[0]).mkdir(parents=True, exist_ok=True)
        print(f"{key[0]}/{key[1]}: {len(group['outputs'])} outputs", flush=True)
        if args.dry_run:
            continue
        if subprocess.run(command(key, group, args.metrics), cwd=str(REPO), check=False, env=_env()).returncode:
            failed.append("/".join(key))
    print("failed: " + (", ".join(failed) if failed else "none"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
