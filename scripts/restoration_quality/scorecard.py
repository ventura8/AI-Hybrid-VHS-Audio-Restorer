"""The metric registry, per-window rows and the median/tail aggregation every family shares.

A metric's direction says where its bad tail points: "higher" is better (the tail is p10),
"lower" is better (p90), "none" is displayed only (p90), and "two-sided" is a reading a
listener hears as wrong either way, read as a distance from a neutral `target` (ear v3,
principle 1). Its tail is whichever of p10 and p90 sits further from that target, and the
listening set picks its windows by |delta - target|. The listener's own target (the reading
of the file the user accepted) and the dead zone around it live in the tuning grids
(`scripts/tune_grids/*_v3.yaml`), not here: a two-sided spec's `target` is only its neutral
point, no change from the source (0) unless the reading says otherwise.

Ear v3 families: `dsp` (per window), `file` (one reading per file: loudness, the gain ride
and the sync of `file_metrics.py`) and `meta` (what the capture itself carries, the source
profile of `source_profile.py`; source side only, displayed). Counts (clicks, dropouts) also
report their worst window (`max`) and `p95` beside the median and tail, so one click in one
window is not diluted by the median of the windows around it.
"""

from dataclasses import dataclass, field

import numpy as np

TAIL_PERCENTILE = 10.0
EXTREME_PERCENTILE = 95.0
HIGHER, LOWER, NONE, TWO_SIDED = "higher", "lower", "none", "two-sided"
DIRECTIONS = (HIGHER, LOWER, NONE, TWO_SIDED)
SPEECH_ROUTES = ("speech", "mixed")
PROGRAMME_ROUTES = ("speech", "music", "mixed")
ALL_ROUTES = ("speech", "music", "mixed", "silence")


@dataclass(frozen=True)
class MetricSpec:
    """What a metric is, which family computes it, and which way is better (one of `DIRECTIONS`).

    `target` is the neutral point of a two-sided reading; `extremes` adds the worst window
    (`max`) and the `p95` to a count's summaries.
    """

    name: str
    family: str
    better: str
    unit: str = ""
    routes: tuple = PROGRAMME_ROUTES
    target: float = 0.0
    extremes: bool = False


@dataclass
class WindowRow:
    """One window's readings for the source, the output, and their difference."""

    window: int
    start_s: float
    end_s: float
    route: str
    source: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    output_route: str = ""

    def delta(self):
        """output - source for every metric read on both sides."""
        return {name: value - self.source[name] for name, value in self.output.items() if _paired(self.source, name, value)}


def _paired(source, name, value):
    """Whether `name` has a finite reading on both sides."""
    return _is_number(value) and _is_number(source.get(name))


@dataclass
class ScoreCard:
    """Everything scored for one output: its rows, file-level readings, aggregates, family status and run notes."""

    rows: list = field(default_factory=list)
    file: dict = field(default_factory=dict)
    families: dict = field(default_factory=dict)
    speaker_floor: float | None = None
    aggregate: dict = field(default_factory=dict)
    verdicts: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


METRICS = {
    spec.name: spec
    for spec in (
        MetricSpec("dsp.residual_noise_db", "dsp", LOWER, "dB", ALL_ROUTES),
        MetricSpec("dsp.lkr", "dsp", LOWER, "log-ratio", ALL_ROUTES),
        MetricSpec("dsp.hf_4k8k", "dsp", HIGHER, "dB", SPEECH_ROUTES),
        MetricSpec("dsp.hf_8k16k", "dsp", HIGHER, "dB", SPEECH_ROUTES),
        MetricSpec("dsp.clicks_per_s", "dsp", LOWER, "1/s", ALL_ROUTES, extremes=True),
        MetricSpec("dsp.pause_pumping_db", "dsp", LOWER, "dB"),
        MetricSpec("dsp.pause_tilt_db", "dsp", LOWER, "dB"),
        MetricSpec("dsp.pause_tilt_abs_db", "dsp", LOWER, "dB"),
        MetricSpec("dsp.dropouts", "dsp", LOWER, "count", extremes=True),
        MetricSpec("dsp.whistle_db", "dsp", LOWER, "dB", ALL_ROUTES),
        MetricSpec("dsp.hum_excess_db", "dsp", LOWER, "dB", ALL_ROUTES),
        # Listener readings (scripts/restoration_quality/pause_metrics.py, sibilance.py, transient_metrics.py).
        MetricSpec("dsp.gap_air_db", "dsp", LOWER, "dB", SPEECH_ROUTES),
        MetricSpec("dsp.pause_depth_db", "dsp", LOWER, "dB", SPEECH_ROUTES),
        MetricSpec("dsp.output_silent", "dsp", LOWER, ""),
        MetricSpec("dsp.sib_centroid_hz", "dsp", LOWER, "Hz", SPEECH_ROUTES),
        MetricSpec("dsp.sib_centroid_abs_hz", "dsp", LOWER, "Hz", SPEECH_ROUTES),
        MetricSpec("dsp.sib_body_db", "dsp", LOWER, "dB", SPEECH_ROUTES),
        MetricSpec("dsp.sib_level_db", "dsp", LOWER, "dB", SPEECH_ROUTES),
        MetricSpec("dsp.attack_db", "dsp", HIGHER, "dB", ("music", "mixed")),
        MetricSpec("dsp.onset_corr", "dsp", HIGHER, "", ("music", "mixed")),
        MetricSpec("dsp.percussive_share_db", "dsp", HIGHER, "dB", ("music", "mixed")),
        MetricSpec("dsp.zimtohrli_loud", "dsp", LOWER, ""),
        # Ear v3, R1 (balance_metrics.py): the programme-cell spectral balance on every route
        # (music takes its own grid). `balance_top_db` is presence or air, whichever moved
        # further from the source: what the bright / dull flags read.
        MetricSpec("dsp.balance_tilt_db_oct", "dsp", TWO_SIDED, "dB/oct"),
        MetricSpec("dsp.balance_presence_db", "dsp", TWO_SIDED, "dB"),
        MetricSpec("dsp.balance_air_db", "dsp", TWO_SIDED, "dB"),
        MetricSpec("dsp.balance_body_db", "dsp", TWO_SIDED, "dB"),
        MetricSpec("dsp.balance_top_db", "dsp", TWO_SIDED, "dB", SPEECH_ROUTES),
        # R11 (lf_metrics.py): the programme's 40-200 Hz against its mids on the loud frames, the
        # mains lines left out: a thinner bass on one side, low end (rumble) left on the other.
        # Read on every programme route; built for round C3's music dewind cutoff and repair, in no grid yet.
        MetricSpec("dsp.lf_programme_db", "dsp", TWO_SIDED, "dB"),
        # R2 (sibilance.py): the level of the 's' after R1's gain match, not net of the plain
        # frames, and the roughness of its spectrum (the round-one "distorted s").
        MetricSpec("dsp.sib_abs_level_db", "dsp", TWO_SIDED, "dB", SPEECH_ROUTES),
        MetricSpec("dsp.sib_texture_db", "dsp", LOWER, "dB", SPEECH_ROUTES),
        # R4 (pause_metrics.gap_residual_readings): the residual in the true pauses. The
        # attenuation's neutral point is the source itself; the listener's own target comes
        # from Session 0's sweep. `gap_pause_s` is the coverage the readings stand on.
        MetricSpec("dsp.gap_atten_db", "dsp", TWO_SIDED, "dB", SPEECH_ROUTES),
        MetricSpec("dsp.gap_slope_db_oct", "dsp", TWO_SIDED, "dB/oct", SPEECH_ROUTES),
        MetricSpec("dsp.gap_spread_db", "dsp", LOWER, "dB", SPEECH_ROUTES),
        MetricSpec("dsp.gap_hf_excess_db", "dsp", TWO_SIDED, "dB", SPEECH_ROUTES),
        MetricSpec("dsp.gap_lsd_db", "dsp", LOWER, "dB", SPEECH_ROUTES),
        MetricSpec("dsp.gap_mod_dist_db", "dsp", LOWER, "dB", SPEECH_ROUTES),
        MetricSpec("dsp.gap_island_kurt", "dsp", LOWER, "log-ratio", SPEECH_ROUTES),
        MetricSpec("dsp.gap_pause_s", "dsp", NONE, "s", SPEECH_ROUTES),
        MetricSpec("stems.si_sdr_db", "stems", HIGHER, "dB"),
        MetricSpec("stems.lsd_db", "stems", LOWER, "dB"),
        MetricSpec("stems.octave_ratio_db", "stems", HIGHER, "dB"),
        MetricSpec("stems.envelope_corr", "stems", HIGHER, ""),
        MetricSpec("stems.has_background", "stems", HIGHER, ""),
        MetricSpec("stems.mert_dist", "stems", LOWER, ""),
        MetricSpec("speech.hallucinated", "speech", LOWER, "", SPEECH_ROUTES),
        MetricSpec("speech.ssl_dist", "speech", LOWER, "", SPEECH_ROUTES),
        MetricSpec("speech.cer", "speech", LOWER, "", SPEECH_ROUTES),
        MetricSpec("speech.wer", "speech", LOWER, "", SPEECH_ROUTES),
        MetricSpec("speech.avg_logprob", "speech", HIGHER, "", SPEECH_ROUTES),
        MetricSpec("speech.no_speech_prob", "speech", LOWER, "", SPEECH_ROUTES),
        MetricSpec("speech.speaker_cos", "speech", HIGHER, "", SPEECH_ROUTES),
        MetricSpec("speech.utmos", "speech", HIGHER, "MOS", SPEECH_ROUTES),
        MetricSpec("mos.sigmos_sig", "mos", HIGHER, "MOS", SPEECH_ROUTES),
        MetricSpec("mos.sigmos_col", "mos", HIGHER, "MOS", SPEECH_ROUTES),
        MetricSpec("mos.sigmos_disc", "mos", HIGHER, "MOS", SPEECH_ROUTES),
        MetricSpec("mos.sigmos_loud", "mos", HIGHER, "MOS", SPEECH_ROUTES),
        MetricSpec("mos.sigmos_noise", "mos", HIGHER, "MOS", SPEECH_ROUTES),
        MetricSpec("mos.sigmos_reverb", "mos", HIGHER, "MOS", SPEECH_ROUTES),
        MetricSpec("mos.sigmos_ovrl", "mos", HIGHER, "MOS", SPEECH_ROUTES),
        MetricSpec("mos.dnsmos_sig", "mos", HIGHER, "MOS", SPEECH_ROUTES),
        MetricSpec("mos.dnsmos_bak", "mos", HIGHER, "MOS", SPEECH_ROUTES),
        MetricSpec("mos.dnsmos_ovrl", "mos", HIGHER, "MOS", SPEECH_ROUTES),
        MetricSpec("mos.dnsmos_p808", "mos", HIGHER, "MOS", SPEECH_ROUTES),
        MetricSpec("mos.dnsmos_gap", "mos", HIGHER, "MOS", SPEECH_ROUTES),
        MetricSpec("mos.scoreq_nr", "mos", HIGHER, "MOS", SPEECH_ROUTES),
        MetricSpec("mos.audiobox_pq", "mos", HIGHER, ""),
        MetricSpec("mos.audiobox_pc", "mos", HIGHER, ""),
        MetricSpec("mos.audiobox_ce", "mos", HIGHER, ""),
        MetricSpec("mos.audiobox_cu", "mos", HIGHER, ""),
        MetricSpec("file.lufs", "file", NONE, "LUFS"),
        MetricSpec("file.lra", "file", NONE, "LU"),
        MetricSpec("file.noise_removed_db", "file", HIGHER, "dB"),
        MetricSpec("file.programme_deviation_db", "file", LOWER, "dB"),
        # The holes over the whole aligned pair (runner.dropout_entry): the window median of
        # `dsp.dropouts` reads 0 wherever a long file holds a few (calibration v3).
        MetricSpec("file.dropouts", "file", LOWER, "count"),
        # R7 and the sync (file_metrics.py): the raw pair, once per file. The ride is shown,
        # not flagged, until the loudnorm_ride degradation calibrates it (the accepted v2
        # finals read up to 1.80 LU p95); the sync readings are gated in gates.py.
        MetricSpec("file.gain_ride_lu", "file", LOWER, "LU"),
        MetricSpec("file.gain_ride_std_lu", "file", LOWER, "LU"),
        MetricSpec("file.gain_ride_deriv_lu", "file", LOWER, "LU/s"),
        MetricSpec("file.sync_drift_ms", "file", LOWER, "ms"),
        MetricSpec("file.sync_offset_ms", "file", LOWER, "ms"),
        MetricSpec("file.sync_unmatched", "file", LOWER, "count"),
        MetricSpec("file.sync_lag_start_ms", "file", NONE, "ms"),
        MetricSpec("file.sync_lag_middle_ms", "file", NONE, "ms"),
        MetricSpec("file.sync_lag_end_ms", "file", NONE, "ms"),
        # R0 (source_profile.py): the capture's own profile, read once per source, displayed.
        MetricSpec("meta.prog_bandwidth_hz", "meta", NONE, "Hz"),
        MetricSpec("meta.brickwall_hz", "meta", NONE, "Hz"),
        MetricSpec("meta.mains_hz", "meta", NONE, "Hz"),
        MetricSpec("meta.mains_evidence_db", "meta", NONE, "dB"),
        MetricSpec("meta.line_hz", "meta", NONE, "Hz"),
        MetricSpec("meta.line_ppm", "meta", NONE, "ppm"),
        MetricSpec("meta.line_sd_ppm", "meta", NONE, "ppm"),
    )
}


def _is_number(value):
    return isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value)


def _values(rows, side, name):
    """Finite readings of `name` on `side` across rows, in row order."""
    return np.array([getattr(row, side)[name] for row in rows if _is_number(getattr(row, side).get(name))], dtype=np.float64)


def _delta_values(rows, name):
    return np.array([row.delta()[name] for row in rows if name in row.delta()], dtype=np.float64)


def summarise(values, better, target=0.0, extremes=False):
    """Median plus the tail that points at the bad end; with `extremes`, the worst window (`max`) and the p95 too.

    The tail is p10 when higher is better, p90 when lower is (or nothing is), and for a
    two-sided reading whichever of the two sits further from `target`.
    """
    if len(values) == 0:
        return None
    summary = {"median": float(np.median(values)), "tail": _tail(values, better, target), "n": int(len(values))}
    if extremes:
        summary.update({"max": float(np.max(values)), "p95": float(np.percentile(values, EXTREME_PERCENTILE))})
    return summary


def _tail(values, better, target):
    low, high = (float(np.percentile(values, percentile)) for percentile in (TAIL_PERCENTILE, 100.0 - TAIL_PERCENTILE))
    if better == TWO_SIDED:
        return low if abs(low - target) > abs(high - target) else high
    return low if better == HIGHER else high


def aggregate(rows, specs=METRICS):
    """Per metric: source, output and delta summaries over the rows on the routes the metric applies to."""
    out = {}
    for name, spec in specs.items():
        summaries = _metric_summaries([row for row in rows if row.route in spec.routes], name, spec)
        if summaries:
            out[name] = summaries
    return out


def _metric_summaries(routed, name, spec):
    """`{"source", "output", "delta"}` summaries of one metric over its routed rows, or None when no output reads it."""
    sides = {"source": _values(routed, "source", name), "output": _values(routed, "output", name), "delta": _delta_values(routed, name)}
    if len(sides["output"]) == 0:
        return None
    return {side: summarise(values, spec.better, spec.target, spec.extremes) for side, values in sides.items()}


def paired_deltas(card):
    """`{metric: delta median}` for a quick, flat view of one card."""
    return {name: entry["delta"]["median"] for name, entry in card.aggregate.items() if entry.get("delta")}
