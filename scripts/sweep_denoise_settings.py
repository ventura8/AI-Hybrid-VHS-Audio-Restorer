#!/usr/bin/env python3
"""Sweeps denoise settings against the real-tape trade metric.

Three constants on this branch were set from synthetic fixtures and turned out wrong on real
tape, so the remaining untested choices are swept the same way the corrected ones were:
against noise removal and programme damage measured separately, on real captures.

The settings under test are shared with the `cathar` mode. Sweeping them here only measures;
anything that wins has to become an `apl_*` setting of its own, because `cathar` shipped in
v1.2.0 and must not move.

**A setting has to be patched where it is actually resolved.** `load_config` merges the
tracked `config.yaml` over the defaults dict, so patching `modules/config.py` for a key the
YAML also names is silently overwritten -- which made three variants report byte-identical
medians and no measurable difference on any clip. The `cathar_*` keys are pinned in
`config.yaml`; the `apl_*` keys are not, and resolve from the typed defaults in
`modules/config.py`. Each variant therefore names its own file, every substitution must
match, and the resolved configuration is re-read before a run is spent on it.
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CONFIG_PY = Path("modules/config.py")
CONFIG_YAML = Path("config.yaml")
SPECTRAL_PY = Path("modules/spectral_denoise.py")
PROCESSING_PY = Path("modules/processing.py")

# name -> [(file, pattern, replacement), ...]. YAML patterns are multiline-anchored so a
# commented-out example of the same key cannot absorb the substitution.
VARIANTS = {
    "wiener": [(CONFIG_YAML, r'^cathar_denoise_method: "spectral"$', 'cathar_denoise_method: "wiener"')],
    "noiseprint_2s5": [(CONFIG_YAML, r"^cathar_noiseprint_duration_s: [0-9.]+$", "cathar_noiseprint_duration_s: 2.5")],
    "beta_008": [(CONFIG_YAML, r"^cathar_beta: [0-9.]+$", "cathar_beta: 0.08")],
    "no_coherent": [(CONFIG_YAML, r"^cathar_enable_coherent: true$", "cathar_enable_coherent: false")],
    "lite_model": [(SPECTRAL_PY, r'^DEEP_DENOISE_MODEL = "UVR-DeNoise\.pth"$', 'DEEP_DENOISE_MODEL = "UVR-DeNoise-Lite.pth"')],
    "alpha_4": [(CONFIG_PY, r'\("apl_spectral_alpha", float, [0-9.]+, 0\.0\)', '("apl_spectral_alpha", float, 4.0, 0.0)')],
    "alpha_2_5": [(CONFIG_PY, r'\("apl_spectral_alpha", float, [0-9.]+, 0\.0\)', '("apl_spectral_alpha", float, 2.5, 0.0)')],
    # A 2.5 s noise probe bought +3.39 dB of removal for +0.12 dB of deviation, the largest
    # single gain measured on this branch. These bracket it to find where it stops paying:
    # the probe takes the quietest stretch of the capture, so a long enough one must
    # eventually reach into programme and start subtracting content.
    "np_1s5": [(CONFIG_YAML, r"^cathar_noiseprint_duration_s: [0-9.]+$", "cathar_noiseprint_duration_s: 1.5")],
    "np_2s5_repeat": [(CONFIG_YAML, r"^cathar_noiseprint_duration_s: [0-9.]+$", "cathar_noiseprint_duration_s: 2.5")],
    "np_4s": [(CONFIG_YAML, r"^cathar_noiseprint_duration_s: [0-9.]+$", "cathar_noiseprint_duration_s: 4.0")],
    "np_6s": [(CONFIG_YAML, r"^cathar_noiseprint_duration_s: [0-9.]+$", "cathar_noiseprint_duration_s: 6.0")],
    # Does the neural separator still earn its place now that a 2.5 s profile is subtracted
    # before it? Choosing between the deep and the lite model moved both metrics by 0.01,
    # which says the choice does not matter but not that the stage does nothing. Skipping it
    # outright is the question the ablation plan listed as "none" and never measured.
    "no_neural": [
        (
            PROCESSING_PY,
            r"    denoised_wav = _denoise_full_audio_step\(surgical_wav, denoise_sub_dir, total_duration=total_duration, denoise_model=model_to_use\)",
            "    denoised_wav = surgical_wav",
        )
    ],
    # The blend was last judged at a 0.75 s probe. Settings on this branch have interacted
    # before -- the blend looked like a coin flip at alpha 1.8 and won clearly at 3.0 -- so
    # it is re-checked rather than assumed at the probe that now ships.
    "no_blend": [(CONFIG_PY, r'\("apl_enable_learned_blend", True\)', '("apl_enable_learned_blend", False)')],
    # Tonal material: on the most tonal third of the corpus auto_pure_linear deviates 0.49
    # against cathar's 0.32, where on the noisiest third it is 0.32 against 0.56. These test
    # the candidate mechanisms -- subtraction too strong for sustained tones, a probe that
    # learns a held note as noise, the blend reading stationarity as noise.
    "alpha_2": [(CONFIG_PY, r'\("apl_spectral_alpha", float, [0-9.]+, 0\.0\)', '("apl_spectral_alpha", float, 2.0, 0.0)')],
    "no_subtraction": [(CONFIG_PY, r'\("apl_enable_spectral_denoise", True\)', '("apl_enable_spectral_denoise", False)')],
    "apl_probe_0s75": [
        (CONFIG_PY, r'\("apl_noiseprint_duration_s", float, [0-9.]+, 0\.0\)', '("apl_noiseprint_duration_s", float, 0.75, 0.0)')
    ],
    "no_repair": [(CONFIG_PY, r'\("apl_enable_physical_repair", True\)', '("apl_enable_physical_repair", False)')],
    "deepfilternet": [(CONFIG_PY, r'\("apl_use_deepfilternet", False\)', '("apl_use_deepfilternet", True)')],
    # Fixture-realism checks: the factor real tape rejected, and the tonal gate switched off
    # (a flatness ceiling of 0 fires on nothing) so alpha can be compared without it.
    "alpha_1_8": [
        (CONFIG_PY, r'\("apl_spectral_alpha", float, [0-9.]+, 0\.0\)', '("apl_spectral_alpha", float, 1.8, 0.0)'),
        (CONFIG_PY, r'\("apl_tonal_flatness_max", float, [0-9.]+, 0\.0\)', '("apl_tonal_flatness_max", float, 0.0, 0.0)'),
    ],
    "no_tonal_gate": [(CONFIG_PY, r'\("apl_tonal_flatness_max", float, [0-9.]+, 0\.0\)', '("apl_tonal_flatness_max", float, 0.0, 0.0)')],
    "np_2s5_alpha4": [
        (CONFIG_YAML, r"^cathar_noiseprint_duration_s: [0-9.]+$", "cathar_noiseprint_duration_s: 2.5"),
        (CONFIG_PY, r'\("apl_spectral_alpha", float, [0-9.]+, 0\.0\)', '("apl_spectral_alpha", float, 4.0, 0.0)'),
    ],
}

# Variants whose effect is invisible to the resolved configuration, because they patch a
# module constant rather than a setting. These skip the resolved-change check.
NOT_A_SETTING = {"lite_model", "no_neural"}


def _fresh_interpreter():
    """Runs a child interpreter that cannot read or leave stale bytecode.

    Python validates a cached .pyc by the source's (mtime, size). Every value this sweep
    substitutes is the same byte length as the one it replaces, and a restore-then-patch
    lands inside one mtime tick, so the pair matches and a child silently imports the
    *previous* variant's module: patched to 2.5 on disk, resolved as 4.0. Discarding the
    caches and refusing to write new ones is what makes each variant independent.
    """
    for cache in Path("modules").rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    return [sys.executable, "-B"], dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


def _git_restore(paths):
    """Returns tracked files to HEAD exactly.

    Restoring from a copy held in memory is not enough: this script crashed part-way once
    and left a cathar setting patched in the working tree. `git checkout --` is exact and
    works even when the run died before any bookkeeping could happen.
    """
    if paths:
        subprocess.run(["git", "checkout", "--", *sorted(paths)], check=False, timeout=120)


def _require_clean(paths):
    """Refuses to start while any file the sweep will patch carries uncommitted changes.

    The restore is `git checkout --`, which discards whatever is in the working tree; run
    over an edit in progress it would take the edit with the patch. This bit once, with an
    uncommitted setting wiped part-way through a run.
    """
    if not paths:
        return
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *sorted(paths)], capture_output=True, text=True, check=False, timeout=120
    )
    dirty = [line[3:] for line in status.stdout.splitlines() if line.strip()]
    if dirty:
        raise SystemExit("uncommitted changes in files the sweep patches and restores; commit or stash first: " + ", ".join(dirty))


def _files_touched(names):
    """Every file any requested variant patches, for restoring afterwards."""
    return {str(target) for name in names for target, _pattern, _replacement in VARIANTS[name]}


def _apply(name):
    """Applies one variant's substitutions, raising if any pattern fails to match."""
    for target, pattern, replacement in VARIANTS[name]:
        source = target.read_text(encoding="utf-8")
        patched, count = re.subn(pattern, replacement, source, flags=re.MULTILINE)
        if count == 0:
            raise SystemExit(f"variant {name}: pattern did not match in {target}, refusing to measure the default: {pattern}")
        target.write_text(patched, encoding="utf-8", newline="\n")


def _resolved_config(name):
    """Returns the configuration as the pipeline will actually see it.

    A variant that patches the losing file resolves back to the default, and the sweep then
    reports a real measurement of nothing. Reading the merged result is the only check that
    catches that, and it is cheap next to the run it guards.
    """
    probe = "import json; from modules import config; print(json.dumps({k: str(v) for k, v in config.CONFIG.items()}))"
    interpreter, env = _fresh_interpreter()
    result = subprocess.run([*interpreter, "-c", probe], capture_output=True, text=True, check=False, timeout=300, env=env)
    if result.returncode != 0:
        raise SystemExit(f"variant {name}: configuration failed to import:\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _run(name, limit, catalog=None):
    """Measures one variant and returns its report path."""
    work = Path(f"experiments/sweep_{name}")
    report = Path(f"experiments/sweep_{name}.json")
    shutil.rmtree(work, ignore_errors=True)
    # An earlier run's report must not stand in for this one if the child fails.
    report.unlink(missing_ok=True)
    interpreter, env = _fresh_interpreter()
    command = [
        *interpreter,
        "scripts/measure_tradeoff.py",
        "--limit",
        str(limit),
        "--modes",
        "auto_pure_linear",
        "--work-dir",
        str(work),
        "--report",
        str(report),
    ]
    if catalog is not None:
        command += ["--catalog", str(catalog)]
    subprocess.run(command, check=False, timeout=7200, env=env)
    shutil.rmtree(work, ignore_errors=True)
    return report if report.exists() else None


def _median(report, key):
    """Median of one metric from a variant report."""
    rows = json.loads(report.read_text(encoding="utf-8"))["auto_pure_linear"]
    return statistics.median(row[key] for row in rows) if rows else None


def _report_line(name, report, base_noise, base_dev):
    """Formats one variant's comparison against the in-run baseline."""
    if report is None:
        return f"{name:<20}{'(not measured)':>15}"
    noise, dev = _median(report, "noise_removed_db"), _median(report, "programme_deviation_db")
    if noise is None or dev is None:
        return f"{name:<20}{'(no clips measured)':>15}"
    if noise > base_noise and dev < base_dev:
        verdict = "BETTER ON BOTH"
    elif noise < base_noise and dev > base_dev:
        verdict = "worse on both"
    else:
        verdict = "trade"
    return f"{name:<20}{noise:>15.2f}{dev:>12.2f}   {noise - base_noise:+.2f} / {dev - base_dev:+.2f}  {verdict}"


def _measure_variant(name, baseline_config, limit, catalog=None):
    """Patches, verifies the patch reaches the pipeline, and measures one variant."""
    _apply(name)
    resolved = _resolved_config(name)
    changed = {key: (baseline_config.get(key), value) for key, value in resolved.items() if baseline_config.get(key) != value}
    if name in NOT_A_SETTING:
        print(f"  module constant patched (not a setting); resolved config unchanged: {not changed}")
    elif not changed:
        print("  SKIPPED: the patched file is not what resolves this setting, so the run would measure the default")
        return None
    else:
        print(f"  resolved change: {changed}")
    return _run(name, limit, catalog)


def main():
    """Runs every variant and prints the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=25)
    # No external baseline: it is measured here, on the same clips, in the same run. A
    # variant scored on 2 clips against a baseline scored on 40 different ones reports the
    # difference between the samples, not the settings -- which briefly made two unrelated
    # variants look identically and dramatically better.
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--catalog", type=Path, default=None, help="Sweep a clip subset instead of the whole corpus")
    args = parser.parse_args()

    targets = _files_touched(args.variants)
    _require_clean(targets)
    results = {}
    try:
        baseline_config = _resolved_config("current")
        print("\n=== current (baseline) ===")
        results["current"] = _run("current", args.limit, args.catalog)
        for name in args.variants:
            print(f"\n=== {name} ===")
            results[name] = _measure_variant(name, baseline_config, args.limit, args.catalog)
            _git_restore(targets)
    finally:
        _git_restore(targets)

    base_report = results.pop("current", None)
    if base_report is None:
        raise SystemExit("The in-run baseline failed; nothing can be compared against it.")
    base_noise, base_dev = _median(base_report, "noise_removed_db"), _median(base_report, "programme_deviation_db")
    if base_noise is None or base_dev is None:
        raise SystemExit("The in-run baseline measured no clips; nothing can be compared against it.")
    print(f"\n{'variant':<20}{'noise removed':>15}{'deviation':>12}   verdict")
    print(f"{'current':<20}{base_noise:>15.2f}{base_dev:>12.2f}")
    for name, report in results.items():
        print(_report_line(name, report, base_noise, base_dev))


if __name__ == "__main__":
    main()
