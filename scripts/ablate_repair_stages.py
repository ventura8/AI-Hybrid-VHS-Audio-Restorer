#!/usr/bin/env python3
"""Measures each physical-repair stage against ground truth, one stage at a time.

`auto_pure_linear` repairs no physical tape damage: no clicks, no crackle, no dropouts, no
clipping, no azimuth skew. Closing that gap by switching on the stages `cathar` runs is the
one approach already known to fail here -- the deterministic tonal cleanup was switched on
that way and measured 6.7-10.7 dB *below* the clean reference, because it removed programme
along with the defect. So each stage is measured on its own, against the defect it targets
and against material that does not carry it, and only the ones that earn it get wired in.

Stages are applied directly to the degraded fixture rather than through the full pipeline.
That isolates the stage from the denoising around it, and it is fast enough to run the whole
matrix: every stage against every fixture, so a stage that repairs its own defect but damages
unrelated material cannot hide.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import cathar
from scripts.score_defect_repair import score_repair
from scripts.score_reference import _fixture_variant

# Every stage cathar runs that auto_pure_linear does not, mapped to the defect it targets.
STAGES = {
    "declick": (cathar._cathar_declick_step, "clicks"),
    "decrackle": (cathar._cathar_decrackle_step, "clicks"),
    "inpaint": (cathar._cathar_inpaint_step, "dropout"),
    "declip": (cathar._cathar_declip_step, "clip"),
    "azimuth": (cathar._cathar_azimuth_step, "azimuth"),
    "repair": (cathar._cathar_repair_step, "clicks"),
    "deplosive": (cathar._cathar_deplosive_step, None),
}

# Fixtures carrying physical damage, plus controls that carry none of it. A stage is only
# worth shipping if it repairs the first group without disturbing the second.
DAMAGE_VARIANTS = ("clicks_only", "dropout_only", "clip_only", "azimuth_only", "physical_combo", "damaged_tape")
CONTROL_VARIANTS = ("hiss_only", "hum_only")


def _variant_of(name):
    """Returns the variant portion of a fixture name: what follows the two-digit segment index."""
    return _fixture_variant(name)


def _apply_stage(step, degraded, work):
    """Runs one cathar stage over one fixture, returning the output path or None."""
    try:
        cathar._require_cathar_binary()
        produced = step(degraded, work)
        return produced if produced and Path(produced).is_file() else None
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"    {step.__name__} failed: {exc}")
        return None


def _score_stage(name, step, records, fixtures_dir, temp_dir):
    """Scores one stage across every requested fixture."""
    rows = []
    for language, record in records:
        language_dir = fixtures_dir / language
        degraded = language_dir / record["degraded"]
        work = temp_dir / name / language / record["name"]
        work.mkdir(parents=True, exist_ok=True)
        produced = _apply_stage(step, degraded, work)
        if produced is None:
            continue
        scored = score_repair(language_dir / record["clean"], degraded, produced)
        shutil.rmtree(work, ignore_errors=True)
        if scored:
            scored.update({"stage": name, "variant": _variant_of(record["name"]), "language": language})
            rows.append(scored)
    return rows


def _summarise(rows, variants, label):
    """Prints median repair and collateral per stage for one group of fixtures."""
    print(f"\n{label}")
    print(f"  {'stage':<12}{'repaired dB':>13}{'spectral dB':>14}{'collateral dB':>15}{'n':>5}")
    for stage in ["none"] + list(STAGES):
        selected = [r for r in rows if r["stage"] == stage and r["variant"] in variants]
        if not selected:
            continue
        repaired = np.median([r["repaired_db"] for r in selected])
        spectral = [r["spectral_repaired_db"] for r in selected if r.get("spectral_repaired_db") is not None]
        collateral = np.median([r["collateral_db"] for r in selected])
        spectral_text = f"{np.median(spectral):>14.2f}" if spectral else f"{'-':>14}"
        print(f"  {stage:<12}{repaired:>13.2f}{spectral_text}{collateral:>15.2f}{len(selected):>5}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=Path("artifacts/repair-fixtures"))
    parser.add_argument("--report", type=Path, default=Path("experiments/repair_stages.json"))
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    parser.add_argument("--damage-variants", nargs="+", default=list(DAMAGE_VARIANTS), help="Fixture families carrying the damage")
    parser.add_argument("--control-variants", nargs="+", default=list(CONTROL_VARIANTS), help="Fixture families carrying none")
    return parser.parse_args()


def main():
    """Runs every stage over every fixture and reports the matrix."""
    args = _parse_args()
    manifest = json.loads((args.fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    wanted = set(args.damage_variants) | set(args.control_variants)
    records = [(lang, rec) for lang, recs in manifest["languages"].items() for rec in recs if _variant_of(rec["name"]) in wanted]
    print(f"{len(records)} fixtures, {len(args.stages)} stages\n")

    rows = []
    with tempfile.TemporaryDirectory(prefix="repair_") as temp:
        temp_dir = Path(temp)
        # The untreated baseline: whatever a stage scores has to beat leaving it alone.
        for language, record in records:
            language_dir = args.fixtures_dir / language
            scored = score_repair(language_dir / record["clean"], language_dir / record["degraded"], language_dir / record["degraded"])
            if scored:
                scored.update({"stage": "none", "variant": _variant_of(record["name"]), "language": language})
                rows.append(scored)
        for name in args.stages:
            step, _target = STAGES[name]
            print(f"=== {name} ===")
            rows.extend(_score_stage(name, step, records, args.fixtures_dir, temp_dir))

    _summarise(rows, args.damage_variants, "ON DAMAGED FIXTURES (repair should be high)")
    _summarise(rows, args.control_variants, "ON UNDAMAGED CONTROLS (collateral should stay low)")

    print("\nPER DEFECT (median repaired dB)")
    header = f"  {'stage':<12}" + "".join(f"{v.replace('_only', ''):>14}" for v in args.damage_variants)
    print(header)
    for stage in ["none"] + list(STAGES):
        cells = ""
        for variant in args.damage_variants:
            selected = [r["repaired_db"] for r in rows if r["stage"] == stage and r["variant"] == variant]
            cells += f"{np.median(selected):>14.2f}" if selected else f"{'-':>14}"
        if any(r["stage"] == stage for r in rows):
            print(f"  {stage:<12}{cells}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
