"""The metric registry, per-window rows and the median/tail aggregation every family shares."""

from dataclasses import dataclass, field

import numpy as np

TAIL_PERCENTILE = 10.0


@dataclass(frozen=True)
class MetricSpec:
    """What a metric is, which family computes it, and which way is better ("higher" | "lower" | "none")."""

    name: str
    family: str
    better: str
    unit: str = ""
    routes: tuple = ("speech", "music", "mixed")


@dataclass
class WindowRow:
    """One window's readings for the source, the output, and their difference."""

    window: int
    start_s: float
    end_s: float
    route: str
    source: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)

    def delta(self):
        """output - source for every metric read on both sides."""
        return {name: value - self.source[name] for name, value in self.output.items() if _paired(self.source, name, value)}


def _paired(source, name, value):
    """Whether `name` has a finite reading on both sides."""
    return _is_number(value) and _is_number(source.get(name))


@dataclass
class ScoreCard:
    """Everything scored for one output: its rows, file-level readings, aggregates and family status."""

    rows: list = field(default_factory=list)
    file: dict = field(default_factory=dict)
    families: dict = field(default_factory=dict)
    speaker_floor: float | None = None
    aggregate: dict = field(default_factory=dict)
    verdicts: list = field(default_factory=list)


METRICS = {
    spec.name: spec
    for spec in (
        MetricSpec("dsp.residual_noise_db", "dsp", "lower", "dB", ("speech", "music", "mixed", "silence")),
        MetricSpec("dsp.lkr", "dsp", "lower", "log-ratio", ("speech", "music", "mixed", "silence")),
        MetricSpec("dsp.hf_4k8k", "dsp", "higher", "dB", ("speech", "mixed")),
        MetricSpec("dsp.hf_8k16k", "dsp", "higher", "dB", ("speech", "mixed")),
        MetricSpec("dsp.clicks_per_s", "dsp", "lower", "1/s", ("speech", "music", "mixed", "silence")),
        MetricSpec("dsp.pause_pumping_db", "dsp", "lower", "dB", ("speech", "music", "mixed")),
        MetricSpec("dsp.pause_tilt_db", "dsp", "lower", "dB", ("speech", "music", "mixed")),
        MetricSpec("dsp.pause_tilt_abs_db", "dsp", "lower", "dB", ("speech", "music", "mixed")),
        MetricSpec("dsp.dropouts", "dsp", "lower", "count", ("speech", "music", "mixed")),
        MetricSpec("dsp.whistle_db", "dsp", "lower", "dB", ("speech", "music", "mixed", "silence")),
        MetricSpec("dsp.hum_excess_db", "dsp", "lower", "dB", ("speech", "music", "mixed", "silence")),
        MetricSpec("stems.si_sdr_db", "stems", "higher", "dB"),
        MetricSpec("stems.lsd_db", "stems", "lower", "dB"),
        MetricSpec("stems.octave_ratio_db", "stems", "higher", "dB"),
        MetricSpec("stems.envelope_corr", "stems", "higher", ""),
        MetricSpec("speech.cer", "speech", "lower", "", ("speech", "mixed")),
        MetricSpec("speech.wer", "speech", "lower", "", ("speech", "mixed")),
        MetricSpec("speech.avg_logprob", "speech", "higher", "", ("speech", "mixed")),
        MetricSpec("speech.no_speech_prob", "speech", "lower", "", ("speech", "mixed")),
        MetricSpec("speech.speaker_cos", "speech", "higher", "", ("speech", "mixed")),
        MetricSpec("speech.utmos", "speech", "higher", "MOS", ("speech", "mixed")),
        MetricSpec("mos.sigmos_sig", "mos", "higher", "MOS", ("speech", "mixed")),
        MetricSpec("mos.sigmos_col", "mos", "higher", "MOS", ("speech", "mixed")),
        MetricSpec("mos.sigmos_disc", "mos", "higher", "MOS", ("speech", "mixed")),
        MetricSpec("mos.sigmos_loud", "mos", "higher", "MOS", ("speech", "mixed")),
        MetricSpec("mos.sigmos_noise", "mos", "higher", "MOS", ("speech", "mixed")),
        MetricSpec("mos.sigmos_reverb", "mos", "higher", "MOS", ("speech", "mixed")),
        MetricSpec("mos.sigmos_ovrl", "mos", "higher", "MOS", ("speech", "mixed")),
        MetricSpec("mos.dnsmos_sig", "mos", "higher", "MOS", ("speech", "mixed")),
        MetricSpec("mos.dnsmos_bak", "mos", "higher", "MOS", ("speech", "mixed")),
        MetricSpec("mos.dnsmos_ovrl", "mos", "higher", "MOS", ("speech", "mixed")),
        MetricSpec("mos.dnsmos_p808", "mos", "higher", "MOS", ("speech", "mixed")),
        MetricSpec("mos.audiobox_pq", "mos", "higher", ""),
        MetricSpec("mos.audiobox_pc", "mos", "higher", ""),
        MetricSpec("mos.audiobox_ce", "mos", "higher", ""),
        MetricSpec("mos.audiobox_cu", "mos", "higher", ""),
        MetricSpec("file.lufs", "file", "none", "LUFS"),
        MetricSpec("file.lra", "file", "none", "LU"),
        MetricSpec("file.noise_removed_db", "file", "higher", "dB"),
        MetricSpec("file.programme_deviation_db", "file", "lower", "dB"),
    )
}


def _is_number(value):
    return isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value)


def _values(rows, side, name):
    """Finite readings of `name` on `side` across rows, in row order."""
    return np.array([getattr(row, side)[name] for row in rows if _is_number(getattr(row, side).get(name))], dtype=np.float64)


def _delta_values(rows, name):
    return np.array([row.delta()[name] for row in rows if name in row.delta()], dtype=np.float64)


def summarise(values, better):
    """Median plus the tail that points at the bad end: p10 when higher is better, p90 when lower is."""
    if len(values) == 0:
        return None
    tail = TAIL_PERCENTILE if better == "higher" else 100.0 - TAIL_PERCENTILE
    return {"median": float(np.median(values)), "tail": float(np.percentile(values, tail)), "n": int(len(values))}


def aggregate(rows, specs=METRICS):
    """Per metric: source, output and delta summaries over the rows on the routes the metric applies to."""
    out = {}
    for name, spec in specs.items():
        routed = [row for row in rows if row.route in spec.routes]
        source, output, delta = _values(routed, "source", name), _values(routed, "output", name), _delta_values(routed, name)
        if len(output) == 0:
            continue
        out[name] = {
            "source": summarise(source, spec.better),
            "output": summarise(output, spec.better),
            "delta": summarise(delta, spec.better),
        }
    return out


def paired_deltas(card):
    """`{metric: delta median}` for a quick, flat view of one card."""
    return {name: entry["delta"]["median"] for name, entry in card.aggregate.items() if entry.get("delta")}
