"""Does the AI human ear flag what the listener flagged on the Tata tapes, and nothing the listener accepted?

usage: check_verdicts.py [--ledger assets/quality_calibration/verdicts.jsonl] [--scores experiments/tata_listen/scores]

The verdicts come from the ledger (`scripts/restoration_quality/ledger.py`); this script used to
hard-code them. For every scored tape (`<scores>/<slug>.json`) and every label that report scored:

- a label a `flag` record flags with F must carry F;
- a label listed under `clean["*"]` (accepted by ear with no complaint) must carry no flag at all,
  and one listed under `clean[F]` must not carry F;
- in a `preference` record of two tiers (one verdict: the top tier over the rest) a better
  output must not fail more hard gates than a worse one; longer rankings are not asserted (the
  by-ear rank was never a gate);
- labels no record names (in its stimuli or `context.heard`) are printed as unjudged;
- a flag on a display-only gate (`gates.DISPLAY_ONLY`, or soft: never counted as a listener
  flag, so a new report cannot carry it) is reported as not asserted and checked nowhere,
  neither flagged nor clean, and a stored report scored before the gate was demoted that still
  lists it does not fail on it (its flags are printed whole). `listener.hiss` is the one so far
  (ear v3, 2026-10-09): the -20 dB gap-air flag also fires on the round-2 finals the user
  heard as natural pauses, so R4's residual readings take over once Round 0 derives their flags.

The check says how much of the ledger it asserted: a flag record or a two-tier preference is
checked when a report scores at least one of its judged labels (both sides of an ordering),
waiting while no report does. A flag record that judges labels on display-only gates alone is
not asserted and counted on its own ("on display-only gates alone"); every other record (a
longer ranking, a trial, a flag record that names no label) is not asserted; the display-only
flags are listed by record. Exit 0 when every checked verdict holds, 1 when one breaks, 2 when
nothing was checked: without a report that scores a judged label the check proves nothing, and
saying "reproduced" then would be a false green. On 2026-10-09 the reports in `scores/` (the
round-one full-tape variants of Tele7abc, SOTI, Vaccin and Gaudeamus) check 3 of the ledger's
33 records (the Tele7abc listening flags, SOTI and Vaccin "alpha 2 sounds better"), 4 are not
asserted, none is on display-only gates alone and 26 wait: the known set and rounds two and
three judged files no report scores yet, which the Round 0 re-score (plan 1.6) does. Six
`listener.hiss` flags are not asserted (the known set and the listening flags of
Tele7abc, the four round-two pause records); the stored reports still list hiss, scored under
the v2 gates file before ear v3 demoted it, and pass.

Round one (2026-09-20): Tele7abc "roformer and baseline are silent in pauses", "cathar alpha 2
has hiss", "apl creates distortion of spoken 's'"; SOTI and Vaccin "alpha 2 sounds better"
(than the cathar and APL baselines). `final_cathar` / `final_apl` and Gaudeamus were never
judged by ear. Rounds two (2026-10-05, the v2 plateaus: APL better on speech, pauses natural on
both, music fine on both, APL's 's' still thin) and three (2026-10-08, the air shelf: +1 dB over
+2 dB and off) judged files these reports do not score.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCORES = HERE / "scores"
CHECKED, WAITING, NOT_ASSERTED = "checked", "waiting", "not asserted"
# A flag record that judges labels on display-only gates alone: not asserted, counted on its own.
DISPLAY_ONLY_ALONE = "on display-only gates alone"
DISPLAY_ONLY = "display-only flags"


def restoration_module(name):
    """`scripts.restoration_quality.<name>`, with the repository on the import path when run as a script."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    return importlib.import_module(f"scripts.restoration_quality.{name}")


def ledger_module():
    """`scripts.restoration_quality.ledger`."""
    return restoration_module("ledger")


def display_only_gates(gates):
    """The gates a ledger flag is not checked on: ear v3's display-only set and every soft gate (never a listener flag)."""
    return frozenset(gates.DISPLAY_ONLY) | {name for name, gate in gates.GATES.items() if gate.severity == gates.SOFT}


def asserted(by_gate, display_only):
    """`{gate: labels}` without the display-only gates."""
    return {gate: labels for gate, labels in by_gate.items() if gate not in display_only}


def flags_of(report, display_only=frozenset()):
    """`{label: set of failed listener flags}`, without the display-only gates a stored report may still list."""
    return {label: set(variant.get("listener_flags", [])) - display_only for label, variant in report["variants"].items()}


def is_ordering(record):
    """A preference record of two tiers: one verdict, the top tier preferred to everything below it."""
    return record["question"]["type"] == "preference" and len(record["answer"]["order"]) == 2


def asserted_pairs(ledger, records):
    """`(better, worse)` from the two-tier preference records: the orderings the hard gates must keep."""
    return [pair for record in records if is_ordering(record) for pair in ledger.preference_pairs(record)]


def verdicts_for(ledger, records, slug, display_only=frozenset()):
    """What the ledger says about `slug`: flagged and clean sets on the gates it asserts, the labels heard, the orderings."""
    mine = ledger.for_tape(records, slug)
    flagged, clean = (asserted(by_gate, display_only) for by_gate in ledger.flag_verdicts(mine))
    return {
        "flagged": flagged,
        "clean": clean,
        "heard": ledger.heard_labels(mine),
        "pairs": asserted_pairs(ledger, mine),
        "display_only": display_only,
    }


def _lacks(flags, label, gate):
    return label in flags and gate not in flags[label]


def _listed(found):
    return sorted(found) or "nothing"


def expected_problems(slug, flags, flagged):
    """Scored labels the listener flagged with a gate the ear does not raise."""
    problems = []
    for gate, labels in sorted(flagged.items()):
        problems += [
            f"{slug}: {label} should be flagged {gate}, has {_listed(flags[label])}"
            for label in sorted(labels)
            if _lacks(flags, label, gate)
        ]
    return problems


def accepted_problems(slug, flags, accepted):
    """Scored labels accepted by ear with no complaint that carry any flag."""
    return [f"{slug}: {label} was accepted by ear but carries {sorted(flags[label])}" for label in sorted(accepted) if flags.get(label)]


def _carries(flags, label, gate, accepted):
    return label not in accepted and gate in flags.get(label, ())


def clean_problems(slug, flags, clean, accepted):
    """Scored labels heard clean of a gate that carry it (the accepted ones are reported once, above)."""
    problems = []
    for gate, labels in sorted(clean.items()):
        problems += [
            f"{slug}: {label} was heard clean of {gate} but carries it"
            for label in sorted(labels)
            if _carries(flags, label, gate, accepted)
        ]
    return problems


def check_tape(slug, report, verdicts, any_flag):
    """Prints every scored label's flags and returns the flag verdicts the report breaks (display-only flags shown only)."""
    flags = flags_of(report, verdicts.get("display_only", frozenset()))
    clean = verdicts["clean"]
    accepted = clean.get(any_flag, set())
    problems = expected_problems(slug, flags, verdicts["flagged"]) + accepted_problems(slug, flags, accepted)
    problems += clean_problems(slug, flags, clean, accepted)
    for label, found in sorted(flags_of(report).items()):
        tag = "" if label in verdicts["heard"] else " (unjudged)"
        print(f"{slug}: {label}{tag}: {', '.join(sorted(found)) or '-'}")
    return problems


def _fails_more(hard, better, worse):
    return better in hard and worse in hard and hard[better] > hard[worse]


def check_ordering(slug, report, pairs):
    """A preferred output must not fail more hard gates than the one it was preferred to."""
    hard = {label: len(variant["hard_failures"]) for label, variant in report["variants"].items()}
    return [
        f"{slug}: {better} fails more hard gates ({hard[better]}) than {worse} ({hard[worse]})"
        for better, worse in pairs
        if _fails_more(hard, better, worse)
    ]


def check_report(slug, report, ledger, records, display_only=frozenset()):
    """Every verdict problem of one scored report."""
    verdicts = verdicts_for(ledger, records, slug, display_only)
    return check_tape(slug, report, verdicts, ledger.ANY_FLAG) + check_ordering(slug, report, verdicts["pairs"])


def _is_flag(record):
    return record["question"]["type"] == "flag"


def _flag_labels(record, display_only):
    """The labels a flag record judges on the gates the check asserts."""
    answer = record["answer"]
    return set().union(*asserted(answer["flagged"], display_only).values(), *asserted(answer["clean"], display_only).values())


def _display_only_alone(record, display_only):
    """A flag record that judges labels, every one of them on a display-only gate."""
    return _is_flag(record) and not _flag_labels(record, display_only) and bool(_flag_labels(record, frozenset()))


def _asserts(record, display_only):
    """A flag record with a label on a gate the check reads, or a two-tier preference."""
    return bool(_flag_labels(record, display_only)) if _is_flag(record) else is_ordering(record)


def _covered(ledger, record, scored, display_only):
    """Whether a report scores enough of `record` for the check to assert it."""
    if _is_flag(record):
        return bool(_flag_labels(record, display_only) & scored)
    return any(better in scored and worse in scored for better, worse in ledger.preference_pairs(record))


def status(ledger, record, scored, display_only=frozenset()):
    """CHECKED, WAITING, DISPLAY_ONLY_ALONE or NOT_ASSERTED for `record`, given the labels scored on its tape."""
    if _display_only_alone(record, display_only):
        return DISPLAY_ONLY_ALONE
    if not _asserts(record, display_only):
        return NOT_ASSERTED
    return CHECKED if _covered(ledger, record, scored, display_only) else WAITING


def display_only_flags(records, display_only):
    """`record id gate` for every ledger flag (flagged or clean) on a display-only gate: reported, never asserted."""
    return [
        f"{record['id']} {gate}"
        for record in filter(_is_flag, records)
        for gate in sorted(set(record["answer"]["flagged"]) | set(record["answer"]["clean"]))
        if gate in display_only
    ]


def coverage(ledger, records, scored, display_only=frozenset()):
    """`{status: [record ids]}` over `records` (DISPLAY_ONLY_ALONE among them), plus the display-only flags under `DISPLAY_ONLY`.

    `scored` is `{tape slug: set of scored labels}`.
    """
    tally = {CHECKED: [], WAITING: [], NOT_ASSERTED: [], DISPLAY_ONLY_ALONE: [], DISPLAY_ONLY: display_only_flags(records, display_only)}
    for record in records:
        tally[status(ledger, record, scored.get(record["tape"], set()), display_only)].append(record["id"])
    return tally


def _not_asserted_flags(tally):
    flags = tally.get(DISPLAY_ONLY, [])
    return f"; {len(flags)} display-only flags not asserted: {', '.join(flags)}" if flags else ""


def coverage_line(tally):
    """How much of the ledger the run asserted, why the rest was not, the records still waiting and the display-only flags."""
    waiting = f": {', '.join(tally[WAITING])}" if tally[WAITING] else ""
    return (
        f"{len(tally[CHECKED])} verdict records checked, "
        f"{len(tally[NOT_ASSERTED])} not asserted (longer rankings, trials, empty flag records), "
        f"{len(tally[DISPLAY_ONLY_ALONE])} {DISPLAY_ONLY_ALONE}, "
        f"{len(tally[WAITING])} waiting for a report{waiting}{_not_asserted_flags(tally)}"
    )


def outcome(problems, tally):
    """The closing line and the exit code: 1 on a broken verdict, 2 when nothing was checked, 0 otherwise."""
    if problems:
        return "\n".join(problems), 1
    if not tally[CHECKED]:
        return "nothing checked: no report scores a label the ledger judged", 2
    return "every checked listener verdict is reproduced", 0


def load_reports(scores):
    """`{tape slug: report}` for every `<slug>.json` under `scores`."""
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(scores).glob("*.json"))}


def _parse(argv):
    parser = argparse.ArgumentParser(description="Check the ear's listener flags against the verdict ledger.")
    parser.add_argument("--ledger", type=Path, default=REPO / "assets" / "quality_calibration" / "verdicts.jsonl")
    parser.add_argument("--scores", type=Path, default=SCORES)
    return parser.parse_args(argv)


def main(argv=None):
    """Checks every scored report against the ledger; 0 all reproduced, 1 a verdict broken, 2 nothing checked."""
    args = _parse(argv)
    ledger = ledger_module()
    display_only = display_only_gates(restoration_module("gates"))
    records = ledger.read(args.ledger)
    reports = load_reports(args.scores)
    if not reports:
        print(f"no scores under {args.scores}")
    problems = [problem for slug, report in reports.items() for problem in check_report(slug, report, ledger, records, display_only)]
    tally = coverage(ledger, records, {slug: set(report["variants"]) for slug, report in reports.items()}, display_only)
    print(coverage_line(tally))
    line, code = outcome(problems, tally)
    print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
