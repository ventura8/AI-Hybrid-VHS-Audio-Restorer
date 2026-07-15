import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Enforce radon cyclomatic complexity grade A for all reported blocks.")
    parser.add_argument("report_json", type=Path, help="Path to a radon cc JSON report.")
    return parser.parse_args(argv)


def _iter_blocks(blocks, parent_name=None):
    if isinstance(blocks, dict):
        error_message = blocks.get("error")
        if error_message:
            yield parent_name or "unknown", "error", error_message
            return
        blocks = [blocks]

    for block in blocks:
        if not isinstance(block, dict):
            yield parent_name or "unknown", "error", f"Unexpected Radon CC entry: {block!r}"
            continue

        if "error" in block:
            yield parent_name or "unknown", "error", block.get("error")
            continue

        block_name = block.get("name", "unknown")
        qualified_name = f"{parent_name}.{block_name}" if parent_name else block_name
        yield qualified_name, block.get("rank"), block.get("complexity")
        closures = block.get("closures", [])
        if closures:
            yield from _iter_blocks(closures, qualified_name)


def main(argv=None):
    args = parse_args(argv)
    if not args.report_json.exists():
        print(f"ERROR: radon CC report not found: {args.report_json}")
        return 2

    with args.report_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    failing = []
    for path, blocks in payload.items():
        for block_name, rank, complexity in _iter_blocks(blocks):
            if rank != "A":
                failing.append((path, block_name, rank, complexity))

    if failing:
        print("FAIL: Radon CC must be A for all reported blocks")
        for path, block_name, rank, complexity in sorted(failing):
            print(f"  - {path}::{block_name}: {rank} ({complexity})")
        return 1

    print("PASS: Radon CC is A for all reported blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
