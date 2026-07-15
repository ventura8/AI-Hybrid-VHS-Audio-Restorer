import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Enforce radon maintainability index grade A for all reported files.")
    parser.add_argument("report_json", type=Path, help="Path to a radon mi JSON report.")
    return parser.parse_args(argv)


def _iter_report_entries(payload):
    for path, entry in payload.items():
        if not entry:
            continue
        if not isinstance(entry, dict):
            yield path, "error", f"Malformed Radon MI entry: expected mapping, got {type(entry).__name__}"
            continue
        if isinstance(entry, dict) and entry.get("error"):
            yield path, "error", entry["error"]
            continue
        grade = entry.get("rank")
        score = entry.get("mi")
        yield path, grade, score


def main(argv=None):
    args = parse_args(argv)
    if not args.report_json.exists():
        print(f"ERROR: radon MI report not found: {args.report_json}")
        return 2

    with args.report_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    failing = [(path, grade, score) for path, grade, score in _iter_report_entries(payload) if grade != "A"]
    if failing:
        print("FAIL: Radon MI must be A for all reported files")
        for path, grade, score in sorted(failing):
            score_text = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
            print(f"  - {path}: {grade} ({score_text})")
        return 1

    print("PASS: Radon MI is A for all reported files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
