"""The calibration's report: `report.md`, `report.json` and the gates files, plus the previous round's gates.

The markdown lists every sensitivity check with what it expected ("up", "down", "flat",
"match") and what it read (an unscored one, why: "not read", or "clipped" with the source's
programme band), the families that scored, the benign floor, the known-ordering rules before
and after the derived gates (and where each tape's verdicts came from), the pooled and
per-route gates with their provenance (source, how many verdicts, how many rounds agree and
the ledger rounds behind them), every gate left underived with the reason, and any verdict
flag no gate reads.
`report.json` holds the same, the rules in place of the full scored results.
"""

import json
from pathlib import Path

from scripts.calibration_gates import ROUTES

GATE_FILE = "gates.json"


def gate_files():
    """The gates file names a calibration writes: the pooled one first, then one per route."""
    return [GATE_FILE] + [f"gates_{route}.json" for route in ROUTES]


def _number(value, spec):
    return "" if value is None else format(value, spec)


def _moving_cells(row):
    medians = ", ".join(f"{m:+.3f}" for m in row["medians"])
    return f"{row['direction']} | {row['monotonic_share']:.2f} | {row['effect_mild']:.1f} | {medians}"


def _flat_cells(row):
    return "| | | " + ", ".join(f"{m:.3f} / {a:.3f}" for m, a in zip(row["moves"], row["allowed"]))


def _match_cells(row):
    return f"| {row['match_share']:.2f} | | " + ", ".join(_number(v, ".0f") for v in row["values"])


CELLS = {"up": _moving_cells, "down": _moving_cells, "flat": _flat_cells, "match": _match_cells}


def _check_cells(row):
    """The table cells after the status: direction, monotonic or match share, effect, and the readings."""
    return CELLS[row["expect"]](row)


def _why(row):
    """Why a check read nothing, in words: "not read", or the clip with the source's band and the reading's start."""
    if row.get("programme_band_hz") is None:
        return row.get("reason", "")
    return f"{row['reason']}: programme band {row['programme_band_hz']:.0f} Hz, reading from {row['reading_from_hz']:.0f} Hz"


def _unscored_cells(row):
    """The empty cells of an unscored check, its reason in the readings cell."""
    why = _why(row)
    return f" | | |{' ' + why if why else ''} |"


def _check_line(row):
    blind = " (blind)" if row["blind"] else ""
    head = f"| {row['degradation']} | {row['metric']}{blind} | {row['expect']} | {row['status']} |"
    return head + (_unscored_cells(row) if row["status"] == "unscored" else f" {_check_cells(row)} |")


def _ordering_lines(title, ordering):
    """A markdown section listing every tape's rules and where its verdicts came from."""
    lines = ["", f"## {title}", ""]
    for tape, entry in ordering.items():
        rules = ", ".join(f"{k}={v}" for k, v in entry["rules"].items())
        lines.append(f"- **{tape}** ({entry.get('verdict_source', 'manifest')}): {rules}")
    return lines


def _floor_lines(floor):
    lines = ["", "## Benign floor", "", "| metric | centre | floor (p95) |", "|---|---|---|"]
    return lines + [f"| {metric} | {v['centre']:+.4f} | {v['floor']:.4f} |" for metric, v in sorted(floor.items())]


def _rounds_cell(gate):
    """How many rounds agree, and the ledger rounds of the verdicts the gate rests on."""
    rounds = gate.get("rounds")
    count = gate.get("rounds_agreeing", 0)
    return f"{count} ({', '.join(rounds)})" if rounds else str(count)


def _gate_lines(title, gates, skipped):
    lines = ["", f"## {title}", "", "| gate | threshold | severity | source | verdicts | rounds |", "|---|---|---|---|---|---|"]
    for name, g in gates.items():
        lines.append(f"| {name} | {g['threshold']:+.3f} | {g['severity']} | {g['source']} | {g['n_verdicts']} | {_rounds_cell(g)} |")
    return lines + [f"- not derived: {name}: {reason}" for name, reason in sorted(skipped.items())]


def _family_lines(status):
    lines = ["", "## Families", ""]
    return lines + [f"- {family}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) for family, counts in sorted(status.items())]


def _markdown(report):
    head = "| degradation | metric | expect | status | direction | monotonic / match | effect (x floor) | readings |"
    lines = [
        "# Output-quality metric calibration",
        "",
        f"Round: {report['round']}",
        "",
        "## Sensitivity",
        "",
        head,
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [_check_line(row) for row in report["checks"]] + _family_lines(report["families"]) + _floor_lines(report["floor"])
    lines += _ordering_lines("Known ordering (default gates)", report["ordering_records"])
    lines += _ordering_lines("Known ordering (derived gates)", report["ordering_after_records"])
    lines += _gate_lines("Derived gates (pooled)", report["gates"], report["skipped_gates"])
    for route in ROUTES:
        lines += _gate_lines(f"Derived gates ({route})", report["route_gates"][route], report["route_skipped"][route])
    unmapped = report["unmapped_flags"]
    return "\n".join(lines + (["", f"Verdict flags no gate reads: {', '.join(unmapped)}"] if unmapped else [])) + "\n"


def _rules_of(ordering):
    return {
        tape: {**entry["rules"], "scores": entry.get("scores"), "verdict_source": entry.get("verdict_source")}
        for tape, entry in ordering.items()
    }


def write_reports(out, report, files):
    """Writes every gates file (`{name: gates}`), `report.json` and `report.md`."""
    out.mkdir(parents=True, exist_ok=True)
    for name, gates in files.items():
        (out / name).write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    serial = {key: value for key, value in report.items() if not key.endswith("_records")}
    serial.update(
        {"ordering": _rules_of(report["ordering_records"]), "ordering_with_derived_gates": _rules_of(report["ordering_after_records"])}
    )
    (out / "report.json").write_text(json.dumps(serial, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(_markdown(report), encoding="utf-8")


def previous_gates(directory):
    """`{file name: gates}` of an earlier calibration round, read before anything is replaced."""
    return {
        name: json.loads((Path(directory) / name).read_text(encoding="utf-8"))
        for name in gate_files()
        if (Path(directory) / name).is_file()
    }
