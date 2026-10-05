"""Does the AI human ear v2 flag what the listener flagged on the Tata tapes, and nothing the listener accepted?

usage: check_verdicts.py            (reads experiments/tata_listen/scores/<slug>.json)

The user's verdicts (2026-09-20): Tele7abc "roformer and baseline are silent in pauses",
"cathar alpha 2 has hiss", "apl creates distortion of spoken 's'"; SOTI and Vaccin "alpha 2
sounds better" (than the APL baseline). `final_cathar` / `final_apl` and Gaudeamus were never
judged by ear, so they are reported, not asserted.

Second round (2026-10-05, on the v2 plateaus in variants/v2): APL better than cathar on speech,
pauses natural on both, music fine on both, APL's 's' still thin.
"""

import json
import sys
from pathlib import Path

SCORES = Path(__file__).resolve().parent / "scores"
EXPECTED = {
    "tele7abc": {
        "listener.pause_collapse": {"apl__roformer", "apl__roformer_no_expander_no_air"},
        "listener.dead_air": {"apl__baseline", "apl__no_air", "apl__no_expander", "apl__no_expander_no_air"},
        "listener.hiss": {"cathar__alpha_2_0", "cathar__alpha_2_0_beta_0_005"},
        "listener.sibilance_thin": {"apl__baseline", "apl__no_air", "apl__roformer"},
    }
}
CLEAN = {"tele7abc": {"cathar075__alpha_2_0", "cathar__baseline"}}
UNJUDGED = ("final_cathar", "final_apl")


def flags_of(report):
    """`{label: set of failed listener flags}`."""
    return {label: set(variant.get("listener_flags", [])) for label, variant in report["variants"].items()}


def check_tape(slug):
    path = SCORES / f"{slug}.json"
    if not path.exists():
        print(f"{slug}: no scores at {path}")
        return []
    flags = flags_of(json.loads(path.read_text(encoding="utf-8")))
    problems = []
    for gate, labels in EXPECTED.get(slug, {}).items():
        for label in labels:
            if label in flags and gate not in flags[label]:
                problems.append(f"{slug}: {label} should be flagged {gate}, has {sorted(flags[label]) or 'nothing'}")
    for label in CLEAN.get(slug, set()):
        if flags.get(label):
            problems.append(f"{slug}: {label} was accepted by ear but carries {sorted(flags[label])}")
    for label, found in sorted(flags.items()):
        tag = " (unjudged)" if label in UNJUDGED else ""
        print(f"{slug}: {label}{tag}: {', '.join(sorted(found)) or '-'}")
    return problems


def check_ordering(slug):
    """cathar alpha 2 must still rank above the APL baseline where the user preferred it (fewer hard failures first)."""
    path = SCORES / f"{slug}.json"
    if not path.exists():
        return []
    report = json.loads(path.read_text(encoding="utf-8"))
    hard = {label: len(variant["hard_failures"]) for label, variant in report["variants"].items()}
    if hard.get("cathar__alpha_2_0", 0) > hard.get("apl__baseline", 0):
        return [
            f"{slug}: cathar__alpha_2_0 fails more hard gates ({hard['cathar__alpha_2_0']}) than apl__baseline ({hard['apl__baseline']})"
        ]
    return []


def main():
    problems = check_tape("tele7abc") + check_tape("soti") + check_tape("vaccin") + check_tape("gaudeamus5")
    problems += check_ordering("soti") + check_ordering("vaccin")
    print("\n".join(problems) if problems else "every listener verdict is reproduced")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
