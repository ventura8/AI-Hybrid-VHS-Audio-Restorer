"""Markdown rendering of a `score_variants` result: verdicts first, then one table per family."""

from pathlib import Path

FAMILY_ORDER = ("file", "dsp", "speech", "mos", "stems")


def _fmt(value, digits=2):
    if value is None:
        return "-"
    return f"{value:+.{digits}f}" if isinstance(value, float) else str(value)


def _verdict_table(result):
    labels = list(result["variants"])
    lines = ["| gate | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    gates = result["variants"][labels[0]]["verdicts"] if labels else []
    for index, gate in enumerate(gates):
        cells = [_verdict_cell(result["variants"][label]["verdicts"][index]) for label in labels]
        lines.append(f"| {gate['gate']} | " + " | ".join(cells) + " |")
    return lines


def _verdict_cell(verdict):
    mark = {"passed": "PASS", "failed": "FAIL", "skipped": "skip"}[verdict["status"]]
    soft = " (soft)" if verdict["severity"] == "soft" and verdict["status"] == "failed" else ""
    return f"{mark}{soft} {_fmt(verdict['value'])}"


def _scored_names(result):
    return {name for variant in result["variants"].values() for name in variant["aggregate"]}


def _metric_names(result, family):
    scored = _scored_names(result)
    return [name for name, spec in result["metrics"].items() if spec["family"] == family and name in scored]


def _family_table(result, family):
    labels = list(result["variants"])
    names = _metric_names(result, family)
    if not names:
        return []
    header = "| metric | source median | " + " | ".join(f"{label} \u0394 median | {label} \u0394 tail" for label in labels) + " |"
    lines = [f"### {family}", "", header, "|---|---|" + "---|---|" * len(labels)]
    for name in names:
        lines.append(_metric_row(result, name, labels))
    return lines + [""]


def _metric_row(result, name, labels):
    first = result["variants"][labels[0]]["aggregate"].get(name, {})
    source = first.get("source", {}).get("median") if first else None
    cells = []
    for label in labels:
        entry = result["variants"][label]["aggregate"].get(name, {}).get("delta") or {}
        cells.append(f"{_fmt(entry.get('median'))} | {_fmt(entry.get('tail'))}")
    return f"| {name} | {_fmt(source)} | " + " | ".join(cells) + " |"


def _status_lines(result):
    lines = ["## Families", ""]
    for label, variant in result["variants"].items():
        summary = ", ".join(f"{family}: {status}" for family, status in variant["families"].items())
        verdict = "PASS" if variant["passed"] else "FAIL " + ", ".join(variant["hard_failures"])
        lines.append(f"- **{label}** \u2014 {verdict}; lag {variant['lag_samples']} samples; {summary}")
    return lines + [""]


def render_markdown(result, listen_index=None):
    """The whole report as Markdown text."""
    source = result["source"]["path"]
    windows = result["windows"]
    lines = [
        "# Restoration quality",
        "",
        f"Source: `{source}`  ",
        f"Windows: {windows['seconds']} s every {windows['hop']} s, {windows['count']} per output.",
        "",
    ]
    lines += _status_lines(result)
    lines += ["## Verdicts", ""] + _verdict_table(result) + [""]
    lines += ["## Metrics (output \u2212 source; tail = worst decile)", ""]
    for family in FAMILY_ORDER:
        lines += _family_table(result, family)
    if listen_index:
        lines += [f"Listening set: `{listen_index}`", ""]
    return "\n".join(lines)


def write_markdown(result, path, listen_index=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_markdown(result, listen_index), encoding="utf-8")
