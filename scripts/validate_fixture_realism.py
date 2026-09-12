#!/usr/bin/env python3
"""Does a fixture set predict real tape? Rank agreement on the comparisons that inverted.

A fixture set is only useful if a change that scores better on it also scores better on
real tape. Five changes on this branch failed that test against the old sets, and each is a
known real-tape verdict now. This runs those same comparisons on a candidate fixture set and
reports whether each ranks the way real tape did. A set that agrees on all of them is fit to
tune against; one that does not is telling you which property it still gets wrong.

Real-tape verdicts, from `scripts/measure_tradeoff.py` over the corpus:

- subtraction factor 3.0 beats 1.8         (25 clips: removal +1.27 dB for +0.09 deviation)
- a 2.5 s noise probe beats 0.75 s           (25 clips: +3.39 dB removal for +0.12 deviation)
- UVR-DeNoise beats DeepFilterNet3           (50 clips: deviation 0.22 against 0.66)
- the tonal gate helps on tonal material     (45 clips: deviation 0.49 to 0.33 at cathar parity)

The verdicts were reached with the trade metric -- noise removed against programme
deviation, reference-free -- so that is the metric the agreement is judged on here, applied
to the fixture outputs exactly as it was applied to the tapes. Two kinds of verdict exist:
ones where removal dominates (a stronger subtraction buys many dB of removal per dB of
deviation) and ones where deviation decides (a stage that damages programme loses whatever
it removes). Each comparison carries its rule.

The full-reference scores against the clean fixture -- SI-SDR and log-spectral distance on
the finished output -- are reported alongside, but they cannot rank subtraction strength at
realistic margins and the reason is measured: fed a noise-free fixture, the finished chain
comes back at 16.9 dB SI-SDR (11.0 dB for unconditioned speech), because the polish,
expander and loudness stages colour the waveform. At a 14 dB margin that colour is a larger
error than the noise was, so every configuration scores below the degraded input and the
trade between them is buried. They still catch destruction -- DeepFilterNet on music-led
programme lands at -9 dB -- which is what the secondary table is for. To tune against
ground truth at these margins, score the chain's intermediate output instead of the mux.

The defect classes get two checks of their own. A coverage matrix runs the scanner and the
dropout detector over every class and says which detector each trips, against what the
class was built to trip: a crackle fixture that does not read as clicks exercises no stage.
And the repair stages are run on and off over every class, held to the real-tape verdict
that gated repair is close to free -- across 174 captures it moved removal from 9.66 to
9.74 dB and deviation from 0.48 to 0.50 -- so on a class carrying no physical damage the
two must land within that distance of each other. On a class that does carry damage the
stage's effect is reported, not gated: the trade metric reads the finished mux, and a pop
or a dropout is a few milliseconds inside it, so an impulse repair that measures +5.5 dB
at the damaged samples (scripts/score_defect_repair.py) can read as nothing here.

Each comparison patches the configuration the way the settings sweep does, so it measures
the setting that actually resolves, and restores the tree afterwards.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import physical_repair
from modules.auto_scanner import scan_and_decide_restoration_strategy
from scripts import sweep_denoise_settings as sweep
from scripts.measure_tradeoff import measure

REMOVAL_DOMINATES = "removal dominates"
DEVIATION_DECIDES = "deviation decides"
# A stronger subtraction agrees with real tape when it buys at least this much removal
# per dB of deviation it adds; real tape showed 14:1 and 28:1 for the two such verdicts.
MIN_REMOVAL_PER_DEVIATION = 5.0
MUSIC_CLASSES = ("music", "musicled", "musiconly")

# (label, better variant, worse variant, classes to judge on, rule, what real tape found)
COMPARISONS = (
    ("alpha 3.0 vs 1.8", "no_tonal_gate", "alpha_1_8", None, REMOVAL_DOMINATES, "+1.27 dB removal for +0.09 deviation"),
    ("probe 2.5 s vs 0.75 s", "current", "apl_probe_0s75", None, REMOVAL_DOMINATES, "+3.39 dB removal for +0.12 deviation"),
    ("UVR-DeNoise vs DeepFilterNet", "current", "deepfilternet", None, DEVIATION_DECIDES, "deviation 0.22 against 0.66"),
    ("tonal gate on vs off (music)", "current", "no_tonal_gate", MUSIC_CLASSES, DEVIATION_DECIDES, "deviation 0.49 to 0.33"),
)
DEFAULT_VARIANTS = (
    "speech_m12",
    "speech_m15",
    "speech_m15_hum",
    "music_m12",
    "music_m15",
    "music_m15_hum",
    "musicled_m15",
    "musiconly_m15",
)
# The defect classes and the detector each was built to trip. Classes with no entry carry
# no physical damage and must trip nothing.
DEFECT_VARIANTS = (
    "crackle_m15",
    "dropout_m15",
    "clip_m15",
    "azimuth_m15",
    "imbalance_m15",
    "whistle_m15",
    "rumble_m15",
    "flutter_m15",
    "dc_m15",
    "low_level_m15",
    "bandlimited_m15",
    "hifibuzz_m15",
    "crosstalk_m15",
    "sprolloff_m15",
    "ep_m12",
    "resonance_m15",
    "plosive_m15",
    "handling_m15",
    "printthrough_m15",
    "squeal_m15",
    "modnoise_m15",
    "breathing_m15",
    "trackswitch_m15",
    "dolbyb_m15",
    "edgedamage_m15",
    "headclog_m15",
    "codec_m15",
    "ghost_m15",
    "crowd_m15",
    "scrapeflutter_m15",
    "buzz_m15",
    "worn_m09",
)
DETECTORS = ("clicks", "dropouts", "clipping", "azimuth", "imbalance", "hum", "whistle", "rumble", "drift", "dc", "resonance")
EXPECTED_DETECTIONS = {
    "crackle": {"clicks"},
    "dropout": {"dropouts"},
    "clip": {"clipping"},
    "azimuth": {"azimuth"},
    "imbalance": {"imbalance"},
    "whistle": {"whistle"},
    "rumble": {"rumble"},
    "flutter": {"drift"},
    "dc": {"dc"},
    "hifibuzz": {"clicks"},
    "ep": {"dropouts", "drift"},
    "resonance": {"resonance"},
    "buzz": {"hum"},
    "worn": {"clicks", "dropouts", "hum", "drift", "dc"},
    "speech_hum": {"hum"},
}
# (class, detector) pairs the scanner is known not to read, so their miss is documented
# rather than failed. The enclosure resonance is a 16 dB ring at a Q of 5 that the scanner
# never reports; the worn tape's mains series sits under a 9 dB margin of noise and reads on
# two voices in five where the same hum at 15 dB reads on seven captures in ten. The scanner
# is cathar's too, so it is left alone (docs/validation.md).
KNOWN_BLIND = {"resonance": {"resonance"}, "worn": {"hum"}}
# Undamaged baselines the matrix carries alongside the defect classes, and the hum class,
# keyed by class name (the hum class reads as "speech" with the hum flag set).
MATRIX_BASELINES = ("speech_m15", "speech_m15_hum", "musicled_m15")
# Classes carrying damage a repair stage may legitimately act on; the rest must come out of
# the repair stages untouched, as real tape did.
PHYSICAL_CLASSES = ("crackle", "dropout", "clip", "azimuth", "hifibuzz", "ep", "worn", "handling", "plosive")
# Gated repair on real tape: removal 9.66 to 9.74 dB, deviation 0.48 to 0.50 across 174 captures.
REPAIR_FREE_REMOVAL_DB = 0.5
REPAIR_FREE_DEVIATION_DB = 0.1


def _score(config_name, fixtures_dir, variants, limit, work_root):
    """Runs auto_pure_linear over the fixture subset under one configuration and returns the rows."""
    report = work_root / f"scores_{config_name}.json"
    interpreter, env = sweep._fresh_interpreter()
    command = [
        *interpreter,
        "scripts/score_reference.py",
        "--fixtures-dir",
        str(fixtures_dir),
        "--modes",
        "auto_pure_linear",
        "--variants",
        *variants,
        "--limit",
        str(limit),
        "--report",
        str(report),
        "--work-dir",
        str(work_root / config_name),
    ]
    # The report of an earlier run is removed first, so a scorer that fails leaves nothing
    # that could be read as this configuration's result.
    report.unlink(missing_ok=True)
    completed = subprocess.run(command, check=False, timeout=7200, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if completed.returncode != 0 or not report.exists():
        print(f"  scoring {config_name} failed (exit {completed.returncode})")
        return []
    return json.loads(report.read_text(encoding="utf-8"))["fixtures"]


def _trade_rows(config_name, fixtures_dir, work_root):
    """The trade metric over every restored fixture of one configuration, keyed by class."""
    rows = []
    if not (work_root / config_name).is_dir():
        return rows
    for language_dir in sorted(p for p in (work_root / config_name).iterdir() if p.is_dir()):
        mode_dir = language_dir / "auto_pure_linear"
        if not mode_dir.is_dir():
            continue
        for fixture_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
            restored = fixture_dir / f"{fixture_dir.name}_vhs_auto_pure_linear.wav"
            degraded = fixtures_dir / language_dir.name / f"{fixture_dir.name}_vhs.wav"
            if not restored.exists() or not degraded.exists():
                continue
            scored = measure(degraded, restored)
            if scored:
                rows.append({"fixture": fixture_dir.name, "class": _class_of(fixture_dir.name), **scored})
    return rows


def _class_of(fixture_name):
    """The programme class a fixture name encodes: speech, music, musicled or musiconly."""
    family = fixture_name.split("_", 1)[1] if "_" in fixture_name else fixture_name
    return family.split("_m", 1)[0]


def _median(rows, key, classes=None):
    """Median of one metric over the rows, optionally only fixtures of the given classes."""
    values = [row[key] for row in rows if row.get(key) is not None and (classes is None or row["class"] in classes)]
    return statistics.median(values) if values else None


def _reference_rows(rows):
    """Flattens score_reference rows to the mode's full-reference metrics, keyed by class."""
    flat = []
    for row in rows:
        scored = row.get("scores", {}).get("auto_pure_linear")
        if scored:
            flat.append({"fixture": row["fixture"], "class": _class_of(row["fixture"]), **scored})
    return flat


def _agrees(rule, removal, deviation):
    """Applies a comparison's rule to (better - worse) deltas of the trade metric."""
    if rule == REMOVAL_DOMINATES:
        return removal > 0 and deviation < removal / MIN_REMOVAL_PER_DEVIATION
    return deviation < 0


def _measure_all(configs, fixtures_dir, variants, limit, work_root):
    """Scores every configuration once, patching and restoring around each."""
    results = {}
    targets = sweep._files_touched([c for c in configs if c != "current"])
    try:
        for name in configs:
            print(f"=== {name} ===")
            if name != "current":
                sweep._apply(name)
            reference = _score(name, fixtures_dir, variants, limit, work_root)
            results[name] = {"reference": _reference_rows(reference), "trade": _trade_rows(name, fixtures_dir, work_root)}
            print(f"  scored {len(results[name]['trade'])} fixtures")
            sweep._git_restore(targets)
    finally:
        sweep._git_restore(targets)
    return results


def _report_trade(results):
    """The primary table: the real-tape metric on the fixture outputs, one row per verdict."""
    agree = 0
    print(f"\n{'trade metric (as on real tape)':<32}{'removed':>9}{'removed':>9}{'deviation':>11}{'deviation':>11}   agrees?")
    print(f"{'comparison':<32}{'better':>9}{'worse':>9}{'better':>11}{'worse':>11}")
    for label, better, worse, classes, rule, verdict in COMPARISONS:
        rb = _median(results[better]["trade"], "noise_removed_db", classes)
        rw = _median(results[worse]["trade"], "noise_removed_db", classes)
        db = _median(results[better]["trade"], "programme_deviation_db", classes)
        dw = _median(results[worse]["trade"], "programme_deviation_db", classes)
        if None in (rb, rw, db, dw):
            print(f"{label:<32}{'unmeasured':>18}")
            continue
        ok = _agrees(rule, rb - rw, db - dw)
        agree += ok
        print(f"{label:<32}{rb:>9.2f}{rw:>9.2f}{db:>11.2f}{dw:>11.2f}   {'YES' if ok else 'no '}   ({rule}; real: {verdict})")
    print(f"\nagreement with real tape: {agree}/{len(COMPARISONS)}")
    return agree


def _report_reference(results):
    """The secondary table: full-reference scores on the finished output, for destruction only."""
    print(f"\n{'full reference (finished output)':<32}{'SI-SDR better':>14}{'SI-SDR worse':>13}{'LSD better':>11}{'LSD worse':>10}")
    for label, better, worse, classes, _rule, _verdict in COMPARISONS:
        sb = _median(results[better]["reference"], "si_sdr_db", classes)
        sw = _median(results[worse]["reference"], "si_sdr_db", classes)
        lb = _median(results[better]["reference"], "lsd_db", classes)
        lw = _median(results[worse]["reference"], "lsd_db", classes)
        if None in (sb, sw, lb, lw):
            print(f"{label:<32}{'unmeasured':>14}")
            continue
        print(f"{label:<32}{sb:>14.2f}{sw:>13.2f}{lb:>11.2f}{lw:>10.2f}")


def _report_classes(results):
    """Per-class trade metric for the current configuration, so a miss can be located."""
    rows = results["current"]["trade"]
    print(f"\n{'current, by class':<16}{'n':>4}{'removed':>9}{'deviation':>11}")
    for name in sorted({row["class"] for row in rows}):
        subset = [row for row in rows if row["class"] == name]
        print(f"{name:<16}{len(subset):>4}{_median(subset, 'noise_removed_db'):>9.2f}{_median(subset, 'programme_deviation_db'):>11.2f}")


def _detections(wav_path):
    """Which detectors a fixture trips, read the way the chain reads them."""
    profile = scan_and_decide_restoration_strategy(wav_path, executed_mode="auto_pure_linear").get("profile", {})
    return {
        "clicks": bool(profile.get("has_clicks")),
        "dropouts": physical_repair.detect_mute_spans(wav_path) > 0,
        "clipping": bool(profile.get("has_clipping")),
        "azimuth": abs(float(profile.get("azimuth_delay_ms", 0.0))) >= 0.05,
        "imbalance": abs(float(profile.get("balance_db", 0.0))) >= 1.0,
        "hum": float(profile.get("notch_hz", 0.0)) > 0.0,
        "whistle": float(profile.get("crt_notch_hz", 0.0)) > 0.0,
        "rumble": int(profile.get("highpass_hz", 0)) >= 60,
        "drift": bool(profile.get("has_drift")),
        "dc": bool(profile.get("has_dc_offset")),
        "resonance": float(profile.get("resonance_hz", 0.0)) > 0.0,
    }


def _coverage_matrix(fixtures_dir, variants):
    """Runs the detectors over one fixture per class and language; returns {class: {detector: hits}}."""
    manifest = json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    hits, counts = {}, {}
    for language, records in manifest["languages"].items():
        for record in records:
            family = record["name"].split("_", 1)[1]
            if family not in variants:
                continue
            name = _class_of(record["name"]) + ("_hum" if family.endswith("_hum") else "")
            found = _detections(fixtures_dir / language / record["degraded"])
            counts[name] = counts.get(name, 0) + 1
            hits.setdefault(name, {d: 0 for d in DETECTORS})
            for detector, seen in found.items():
                hits[name][detector] += int(seen)
    return hits, counts


def _report_coverage(hits, counts):
    """Prints the class-by-detector matrix and counts the misses against what each class was built for.

    Every cell that disagrees with the design counts as a miss, false positives included,
    because the matrix exists to document the scanner; only the misses `_blind_classes`
    returns fail the run.
    """
    misses = 0
    print(f"\n{'detector coverage':<14}" + "".join(f"{d:>10}" for d in DETECTORS) + "   as built?")
    for name in sorted(hits):
        expected = EXPECTED_DETECTIONS.get(name, set())
        row, wrong = "", 0
        for detector in DETECTORS:
            share = hits[name][detector] / max(counts[name], 1)
            should = detector in expected
            wrong += int((share >= 0.5) != should)
            row += f"{share:>9.0%}{'*' if should else ' '}"
        misses += wrong
        print(f"{name:<14}{row}   {'yes' if wrong == 0 else f'{wrong} off'}")
    print("  * marks the detector the class was built to trip; a share is the fraction of its fixtures that trip it")
    return misses


def _blind_classes(hits, counts):
    """Classes that fail to trip a detector they were built to trip, on half their fixtures or more.

    A class its own detector does not read exercises no stage, so the run fails on it,
    except for the classes the scanner is documented not to read.
    """
    blind = []
    for name, expected in EXPECTED_DETECTIONS.items():
        required = expected - KNOWN_BLIND.get(name, set())
        if name in hits and any(hits[name][detector] / max(counts[name], 1) < 0.5 for detector in required):
            blind.append(name)
    return blind


def _report_repair(results):
    """Repair on against off, class by class: free where nothing is damaged, and fired where something is."""
    print(f"\n{'repair on vs off':<14}{'n':>4}{'removed on':>12}{'off':>8}{'deviation on':>14}{'off':>8}   verdict")
    failures = 0
    for name in sorted({row["class"] for row in results["no_repair"]["trade"]}):
        on = [row for row in results["current"]["trade"] if row["class"] == name]
        off = [row for row in results["no_repair"]["trade"] if row["class"] == name]
        if not on or not off:
            continue
        ro, rf = _median(on, "noise_removed_db"), _median(off, "noise_removed_db")
        do, df = _median(on, "programme_deviation_db"), _median(off, "programme_deviation_db")
        # Free means nothing got worse past the band; a stage that helps is not a failure.
        worse = (rf - ro) > REPAIR_FREE_REMOVAL_DB or (do - df) > REPAIR_FREE_DEVIATION_DB
        moved = abs(ro - rf) > REPAIR_FREE_REMOVAL_DB or abs(do - df) > REPAIR_FREE_DEVIATION_DB
        if name in PHYSICAL_CLASSES:
            verdict = "damaged: stage fired" if moved else "damaged: within the free band"
        elif worse:
            verdict = "NOT free: a stage hurt undamaged material"
            failures += 1
        else:
            verdict = "free, as on real tape" + (" (a stage helped)" if moved else "")
        print(f"{name:<14}{len(on):>4}{ro:>12.2f}{rf:>8.2f}{do:>14.2f}{df:>8.2f}   {verdict}")
    return failures


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=Path("artifacts/realistic-v2"))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--defect-variants", nargs="+", default=list(DEFECT_VARIANTS))
    parser.add_argument("--skip-defects", action="store_true", help="Only the four programme comparisons")
    parser.add_argument("--limit", type=int, default=1, help="Fixtures per variant per language")
    parser.add_argument("--work-dir", type=Path, default=Path("experiments/fixture_realism"))
    return parser.parse_args()


def _defect_checks(args, results):
    """The coverage matrix and the repair rule over the defect classes.

    Returns the matrix's miss count, the classes blind to their own detector, and the
    number of undamaged classes a repair stage hurt.
    """
    hits, counts = _coverage_matrix(args.fixtures_dir, set(args.defect_variants) | set(MATRIX_BASELINES))
    misses = _report_coverage(hits, counts)
    every = list(args.variants) + list(args.defect_variants)
    results["current"] = _rescore("current", args, every)
    results["no_repair"] = _rescore("no_repair", args, every)
    return misses, _blind_classes(hits, counts), _report_repair(results)


def _failures(summary):
    """What, if anything, makes this run a failed validation."""
    reasons = []
    if summary["agreement"] < summary["comparisons"]:
        reasons.append(f"agreement {summary['agreement']}/{summary['comparisons']}")
    if summary.get("blind_classes"):
        reasons.append("classes blind to their own detector: " + ", ".join(summary["blind_classes"]))
    if summary.get("repair_not_free"):
        reasons.append(f"repair hurt {summary['repair_not_free']} undamaged classes")
    return reasons


def _rescore(name, args, variants):
    """Scores one configuration over the given variants, patching the tree around it."""
    targets = sweep._files_touched([name]) if name != "current" else []
    try:
        if name != "current":
            sweep._apply(name)
        reference = _score(name, args.fixtures_dir, variants, args.limit, args.work_dir)
        return {"reference": _reference_rows(reference), "trade": _trade_rows(name, args.fixtures_dir, args.work_dir)}
    finally:
        sweep._git_restore(targets)


def main():
    """Runs the comparisons and reports rank agreement with real tape."""
    args = _parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    configs = sorted({c for _, a, b, _, _, _ in COMPARISONS for c in (a, b)}, key=lambda c: c != "current")
    # Every configuration this run patches is restored with `git checkout --`, which would
    # take an uncommitted edit with it; refuse to start over one.
    patched = [c for c in configs if c != "current"] + ([] if args.skip_defects else ["no_repair"])
    sweep._require_clean(sweep._files_touched(patched))
    results = _measure_all(configs, args.fixtures_dir, args.variants, args.limit, args.work_dir)
    agree = _report_trade(results)
    _report_reference(results)
    _report_classes(results)
    summary = {"agreement": agree, "comparisons": len(COMPARISONS), "scored": {k: len(v["trade"]) for k, v in results.items()}}
    if not args.skip_defects:
        summary["detector_misses"], summary["blind_classes"], summary["repair_not_free"] = _defect_checks(args, results)
    (args.work_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    reasons = _failures(summary)
    if reasons:
        print("\nFAILED: " + "; ".join(reasons))
        sys.exit(1)
    print("\nPASSED: every comparison agrees with real tape" + ("" if args.skip_defects else ", every class reads, repair is free"))


if __name__ == "__main__":
    main()
