"""Tabulate a plateau's confirmation: which of a grid's constraints each candidate trips against the incumbent, per clip.

usage: python -m scripts.confirm_trips --grid scripts/tune_grids/tata_v4.yaml --incumbent S
           --candidate W=DIR [--candidate F=DIR ...] [--states TSV] [--verbose]

Each DIR holds one `scripts/validate_restoration.py` report per clip (the file stem names the clip),
scoring the incumbent and the candidate against the clip's source. Only clips every DIR holds are
counted, so the columns compare like with like. `--states` maps a clip to a group (lines of
`group<TAB>clip`), so the trips split by what the candidate does to each clip; unmapped clips form
the group '?'. The grid, the groups file and each DIR must lie inside the repository, the temp
directory or a root listed in AI_RESTORE_DATA_ROOTS (`cli_paths.confined_path`).

Two counts per group. Strict trips are the loop's own rule (`reward.constrained_readings`).
Material trips keep only the constraints whose move exceeds their `max_move` (MATERIAL_FLOOR_DB
for a bound alone): the loop's rule trips any move past a bound the incumbent already broke (kept
HF -11.71 to -11.71 tripped on an IA clip), which is right for a loop, where the incumbent may not
creep, and noise for a confirmation (the APL v9 plateau's, 2026-10-21: 31 strict trips on 202 clips,
10 of 15 on one group material). See the harness skill, "Confirming a plateau on the IA corpus".
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from scripts.cli_paths import confined_path
from scripts.restoration_quality import reward, runner

# A bound-only constraint's move counts as material over this many dB (the hiss readings' jitter is 0.3-0.4).
MATERIAL_FLOOR_DB = 0.1
UNMAPPED = "?"


def parse_candidate(text):
    """`LABEL=DIR` as (label, Path)."""
    label, sep, folder = text.partition("=")
    if not sep or not label or not folder:
        raise argparse.ArgumentTypeError(f"expected LABEL=DIR, got {text!r}")
    return label, Path(folder)


def load_states(path):
    """`{clip: group}` from lines of `group<TAB>clip`; empty without a file."""
    if path is None:
        return {}
    lines = confined_path(path, "--states", must_exist=True).read_text(encoding="utf-8").splitlines()
    return {clip: group for group, clip, *_rest in (line.split("\t") for line in lines if "\t" in line)}


def load_constraints(grid_path):
    """The grid's `constraints:` section, parsed as the loop parses it."""
    grid = yaml.safe_load(confined_path(grid_path, "--grid", must_exist=True).read_text(encoding="utf-8"))
    return reward.parse_constraints(grid.get("constraints") or {})


def is_material(constraint, incumbent_value, candidate_value):
    """Whether the move the worse way exceeds the constraint's `max_move` (MATERIAL_FLOOR_DB for a bound alone)."""
    move = candidate_value - incumbent_value if constraint.worse == "up" else incumbent_value - candidate_value
    return move > max(constraint.max_move, MATERIAL_FLOOR_DB)


def clip_trips(result, incumbent, label, constraints):
    """`[(key, incumbent value, candidate value, material)]` of the constraints `label` trips against `incumbent`."""
    held, moved = runner.aggregates_for_tuning(result, incumbent), runner.aggregates_for_tuning(result, label)
    rows = []
    for key in reward.constrained_readings(moved, held, constraints):
        before, after = held.get(key), moved.get(key)
        material = before is not None and after is not None and is_material(constraints[key], before, after)
        rows.append((key, before, after, material))
    return rows


def confined_candidates(candidates):
    """`[(label, folder)]` with each folder confined where it is read (`cli_paths.confined_path`)."""
    return [(label, confined_path(folder, f"--candidate {label}", must_exist=True)) for label, folder in candidates]


def common_clips(candidates):
    """The clips every candidate's folder holds a report for, sorted."""
    held = [{path.stem for path in folder.glob("*.json")} for _label, folder in candidates]
    return sorted(set.intersection(*held)) if held else []


def tabulate(candidates, incumbent, constraints, states):
    """`(counts, listed)`: per (label, group) a Counter of clips, tripped, material and each reading's trips; and per
    (label, group) the tripped clips' rows."""
    counts, listed = defaultdict(Counter), defaultdict(dict)
    for clip in common_clips(candidates):
        group = states.get(clip, UNMAPPED)
        for label, folder in candidates:
            result = json.loads((folder / f"{clip}.json").read_text(encoding="utf-8"))
            rows = clip_trips(result, incumbent, label, constraints)
            _count(counts[(label, group)], rows)
            if rows:
                listed[(label, group)][clip] = rows
    return counts, listed


def _count(counter, rows):
    counter["clips"] += 1
    counter["tripped"] += bool(rows)
    counter["material"] += any(row[3] for row in rows)
    counter.update(f"by:{_short(row[0])}" for row in rows)


def _short(key):
    """`dsp.hiss_pump_db.output.median` as `hiss_pump_db`."""
    parts = key.split(".")
    return parts[1] if len(parts) > 1 else key


def _value(number):
    return "None" if number is None else f"{number:.2f}"


def summary_lines(counts, labels):
    """One line per group and label: trips, material trips, clips, and the trips by reading."""
    groups = sorted({group for _label, group in counts})
    return [_summary_line(counts.get((label, group), Counter()), label, group) for group in groups for label in labels]


def _summary_line(counter, label, group):
    by = {name[3:]: n for name, n in counter.most_common() if name.startswith("by:")}
    return f"{label:6} {group:10} trips {counter['tripped']:3d} material {counter['material']:3d} / {counter['clips']:3d}  {by}"


def clip_lines(listed):
    """Every tripped clip's readings, `*` marking a material move."""
    lines = []
    for (label, group), clips in sorted(listed.items()):
        lines.append(f"{label} {group}")
        for clip, rows in sorted(clips.items()):
            text = "; ".join(f"{'*' if material else ''}{_short(key)} {_value(a)} -> {_value(b)}" for key, a, b, material in rows)
            lines.append(f"  {clip[:44]:44} {text}")
    return lines


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--grid", required=True, help="the grid whose constraints judge, e.g. scripts/tune_grids/tata_v4.yaml")
    parser.add_argument("--incumbent", required=True, help="the incumbent's label inside each report")
    parser.add_argument("--candidate", required=True, action="append", type=parse_candidate, help="LABEL=DIR, repeatable")
    parser.add_argument("--states", help="lines of group<TAB>clip")
    parser.add_argument("--verbose", action="store_true", help="list every tripped clip")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    candidates = confined_candidates(args.candidate)
    counts, listed = tabulate(candidates, args.incumbent, load_constraints(args.grid), load_states(args.states))
    labels = [label for label, _folder in candidates]
    print(f"clips held by every report: {len(common_clips(candidates))}")
    print("\n".join(summary_lines(counts, labels)))
    if args.verbose:
        print("\n".join(clip_lines(listed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
