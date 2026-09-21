"""Hard gates as data: the readings that veto an output the way a listener would.

A gate names a metric, which statistic to read (`median` or `tail` of the output,
or `delta.median` / `delta.tail` of output minus source), a comparison, and a
threshold. Thresholds are numbers, or `"floor-0.05"` for the per-tape speaker
floor. `scripts/calibrate_quality_metrics.py` writes a `gates.json` that
overrides these values without touching code.
"""

import json
from dataclasses import asdict, dataclass

SOFT = "soft"


@dataclass(frozen=True)
class Gate:
    """One veto rule. `severity` is "hard" (vetoes) or "soft" (reported only)."""

    metric: str
    stat: str
    op: str
    threshold: object
    note: str = ""
    severity: str = "hard"


GATES = {
    "speech.cer_tail": Gate(
        "speech.cer",
        "tail",
        "<=",
        0.15,
        "ASR consistency with the source, worst decile of windows (Whisper is unstable on hissy tape)",
        SOFT,
    ),
    "speech.cer_median": Gate("speech.cer", "median", "<=", 0.05, "ASR consistency with the source, typical window"),
    "speech.speaker": Gate(
        "speech.speaker_cos", "tail", ">=", "floor-0.05", "timbre: cosine to the source under the tape's own intra-speaker floor"
    ),
    "speech.logprob": Gate("speech.avg_logprob", "delta.tail", ">=", -0.15, "ASR confidence must not fall"),
    "mos.sigmos_col": Gate("mos.sigmos_col", "delta.median", ">=", -0.3, "coloration: muffled / metallic timbre"),
    "mos.sigmos_disc": Gate("mos.sigmos_disc", "delta.tail", ">=", -0.3, "discontinuity: chopping, musical noise"),
    "mos.sigmos_noise": Gate("mos.sigmos_noise", "delta.median", ">=", 0.0, "the output must not read noisier than the source"),
    "stems.octave": Gate("stems.octave_ratio_db", "tail", ">=", -3.0, "background stripped: non-vocal stem down in some octave"),
    "dsp.hf_4k8k": Gate("dsp.hf_4k8k", "delta.median", ">=", -8.0, "presence band lost: muffled speech"),
    "dsp.pause_pumping": Gate("dsp.pause_pumping_db", "delta.median", "<=", 12.0, "pause floor switches on and off between words", "soft"),
    "dsp.hf_8k16k": Gate("dsp.hf_8k16k", "delta.median", ">=", -12.0, "air band lost: sibilance eaten"),
    "dsp.lkr": Gate("dsp.lkr", "delta.median", "<=", 0.3, "musical noise"),
    "dsp.hum": Gate("dsp.hum_excess_db", "delta.median", "<=", 3.0, "hum must not appear"),
    "dsp.clicks": Gate("dsp.clicks_per_s", "delta.tail", "<=", 0.5, "clicks must not appear"),
    "dsp.dropouts": Gate("dsp.dropouts", "delta.tail", "<=", 1.0, "no new holes in the programme"),
    "dsp.whistle": Gate("dsp.whistle_db", "delta.median", "<=", 3.0, "line whistle must not appear"),
    "file.lra": Gate("file.lra", "delta.median", ">=", -3.0, "dynamics squashed"),
    "file.lufs": Gate("file.lufs", "delta.median", "abs<=", 1.5, "level moved", SOFT),
    "speech.utmos": Gate("speech.utmos", "delta.median", ">=", -0.3, "naturalness", SOFT),
    "mos.audiobox_pq": Gate("mos.audiobox_pq", "delta.median", ">=", -0.3, "production quality", SOFT),
    "mos.audiobox_pc": Gate("mos.audiobox_pc", "delta.median", ">=", -0.5, "scene simplified: background stripped"),
}


def load_gates(path, base=GATES):
    """`base` with the entries of a JSON file merged over it: `{name: {metric, stat, op, threshold, note?, severity?}}`."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    merged = dict(base)
    for name, fields in raw.items():
        known = {key: value for key, value in fields.items() if key in Gate.__dataclass_fields__}
        merged[name] = Gate(**{**asdict(base[name]), **known}) if name in base else Gate(**known)
    return merged


def _read(aggregate, gate):
    """The number a gate looks at, or None when the metric was not scored."""
    entry = aggregate.get(gate.metric)
    if not entry:
        return None
    side, _dot, stat = gate.stat.partition(".")
    if not stat:
        side, stat = "output", side
    summary = entry.get(side)
    return None if not summary else summary.get(stat)


def _threshold(gate, speaker_floor):
    """A numeric threshold, resolving `floor-x` against the tape's speaker floor."""
    if isinstance(gate.threshold, str) and gate.threshold.startswith("floor"):
        if speaker_floor is None:
            return None
        return float(speaker_floor) + float(gate.threshold[5:] or 0.0)
    return float(gate.threshold)


def _passes(value, op, threshold):
    if op == "<=":
        return value <= threshold
    if op == ">=":
        return value >= threshold
    return abs(value) <= threshold


def evaluate_gates(aggregate, gates=GATES, speaker_floor=None):
    """Verdicts for every gate: passed / failed / skipped (metric absent or floor unknown)."""
    verdicts = []
    for name, gate in gates.items():
        value, threshold = _read(aggregate, gate), _threshold(gate, speaker_floor)
        status = "skipped" if value is None or threshold is None else ("passed" if _passes(value, gate.op, threshold) else "failed")
        verdicts.append(
            {
                "gate": name,
                "metric": gate.metric,
                "stat": gate.stat,
                "value": value,
                "threshold": threshold,
                "status": status,
                "severity": gate.severity,
                "note": gate.note,
            }
        )
    return verdicts


def hard_failures(verdicts):
    """Names of the hard gates that failed."""
    return [verdict["gate"] for verdict in verdicts if verdict["status"] == "failed" and verdict["severity"] != SOFT]
