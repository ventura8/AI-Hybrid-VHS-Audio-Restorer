#!/usr/bin/env python3
"""Validate restored outputs against their source the way a listener would.

    validate_restoration.py SOURCE LABEL=PATH [LABEL=PATH ...] [--outputs-dir DIR]
        [--windows 15 --hop 7.5] [--metrics all|dsp,stems,speech,mos]
        [--report out.json] [--markdown out.md] [--listen-dir DIR --worst 3]
        [--language ro] [--device cuda] [--cache-dir experiments/quality_cache]
        [--gates gates.json]

SOURCE and every PATH may be a video (audio is extracted once into the cache) or a WAV.
With --outputs-dir the outputs are found next to the source by the mode suffixes
(`<stem>_Cathar_Cleaned.*` -> cathar, `<stem>_PureLinear_Cleaned.*` -> apl, ...).
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import OUTPUT_SUFFIX_BY_MODE  # noqa: E402
from scripts.restoration_quality import gates as gates_mod  # noqa: E402
from scripts.restoration_quality import listening, report, runner  # noqa: E402

LABEL_BY_SUFFIX = {suffix: mode for mode, suffix in OUTPUT_SUFFIX_BY_MODE.items()}
LABEL_BY_SUFFIX["_PureLinear_Cleaned"] = "apl"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("outputs", nargs="*", help="LABEL=PATH pairs")
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument("--windows", type=float, default=15.0)
    parser.add_argument("--hop", type=float, default=7.5)
    parser.add_argument("--metrics", default="all")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--listen-dir", type=Path, default=None)
    parser.add_argument("--worst", type=int, default=3)
    parser.add_argument("--language", default="ro")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/quality_cache"))
    parser.add_argument("--gates", type=Path, default=None)
    parser.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help="an earlier --report JSON: the families scored now replace theirs in it (the rest is kept), verdicts re-read",
    )
    return parser.parse_args(argv)


def _merged(result, args, gates):
    """`result` merged into the stored report named by --merge-into, or `result` itself."""
    if not args.merge_into or not args.merge_into.exists():
        return result
    import json

    from scripts.tune_restoration import merge_families

    stored = json.loads(args.merge_into.read_text(encoding="utf-8"))
    return merge_families(stored, result, _families(args.metrics), gates)


def _outputs_from_dir(source, outputs_dir):
    """`{label: path}` for every mode output beside the source."""
    stem = Path(source).stem
    found = {}
    for candidate in sorted(Path(outputs_dir).iterdir()):
        if not candidate.stem.startswith(stem) or candidate.is_dir():
            continue
        suffix = candidate.stem[len(stem) :]
        if suffix in LABEL_BY_SUFFIX:
            found[LABEL_BY_SUFFIX[suffix]] = candidate
    return found


def _outputs(args):
    outputs = {label: Path(path) for label, _eq, path in (item.partition("=") for item in args.outputs)}
    if args.outputs_dir:
        outputs.update(_outputs_from_dir(args.source, args.outputs_dir))
    if not outputs:
        raise SystemExit("no outputs given: pass LABEL=PATH or --outputs-dir")
    return outputs


def _families(spec):
    return runner.ALL_FAMILIES if spec == "all" else tuple(part.strip() for part in spec.split(",") if part.strip())


def main(argv=None):
    args = _parse_args(argv)
    gates = gates_mod.load_gates(args.gates) if args.gates else gates_mod.GATES
    registry = runner.ModelRegistry(device=args.device)
    result, cards = runner.score_variants(
        args.source,
        _outputs(args),
        cache_dir=args.cache_dir,
        gates=gates,
        windows=args.windows,
        hop=args.hop,
        families=_families(args.metrics),
        registry=registry,
        language=args.language,
    )
    listen_index = listening.render_listening_set(result, cards, args.listen_dir, count=args.worst) if args.listen_dir else None
    result = _merged(result, args, gates)
    if args.report:
        runner.write_result(result, args.report)
    if args.markdown:
        report.write_markdown(result, args.markdown, listen_index)
    print(report.render_markdown(result, listen_index))
    return 0


if __name__ == "__main__":
    sys.exit(main())
