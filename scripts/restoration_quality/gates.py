"""Hard gates as data: the readings that veto an output the way a listener would.

A gate names a metric, which statistic to read (`median` or `tail` of the output,
or `delta.median` / `delta.tail` of output minus source), a comparison, and a
threshold. Thresholds are numbers, or `"floor-0.05"` for the per-tape speaker
floor. `scripts/calibrate_quality_metrics.py` writes a `gates.json` that
overrides these values without touching code. A gate whose threshold is a starting guess
no verdict has tested yet is `calibrated=False`; an entry a calibration file sets (it
carries the `source` the threshold came from) is calibrated.

Ear v3 (2026-10-09):

- `listener.hiss` is display-only (soft). Gap level does not order the verdicts: the round-2
  finals the user heard as natural pauses read gap_air -8..-12 dB, cathar alpha 2 ("has
  hiss", round 1) -15..-18.5, so the -20 dB flag fires on the accepted files as well as on
  the hissy one and cannot tell them apart. R4's residual readings
  (`pause_metrics.gap_residual_readings`) take over once Round 0 derives their flags from
  the ledger. A gates file sets its threshold but not its severity (`DISPLAY_ONLY`): the v2
  `experiments/quality_calibration/gates.json` (2026-09-22), still the default `--gates` of
  `tune_restoration.py` and `autotune_restoration.py` and what `score_listen.py` passes, lists
  it as a `flag`, and merged as it stands it made the loops count the hiss reading again (an output
  at gap_air -10 dB failed it as a flag, so `_beats` and `reward.gate_constraints` refused or
  capped a candidate on it).
- `listener.bright` / `listener.dull` read R1's `balance_top_db` (presence or air, whichever
  moved further from the source) at +1.0 / -1.0 dB, uncalibrated. The plan's threshold is
  the midpoint between R1's reading of the accepted +1 dB air file and the rejected +2 dB
  one, the dull side set on `deesser_bug` / `single4s` ("underwater"). A read-only probe of
  the v4_air set (APL, air shelf at 7.5 kHz +2 / +1 / off) found R1 cannot carry that on two
  of the three speech tapes: their programme band ends at 4.5 / 5.0 kHz, so air is unread,
  and presence reads -0.06 / -0.08 / -0.10 (Tele7abc), 0.00 / -0.02 / -0.05 (SOTI). On
  Vaccin (7.1 kHz) air reads +0.43 / +0.17 / -0.05 dB. On the accepted round-2 finals the
  presence and air medians sit within -0.60..+0.43 dB, so neither flag fires on them at the
  starting values.
- the sync gates (`file_metrics.sync_drift`): one PAL video frame (40 ms) of drift or of
  constant offset parts lips from picture, and an anchor the output does not follow is a
  broken sync stage. The 32 v2 final pairs read drift 0.02-0.26 ms, offset 4.7-5.4 ms and
  no unmatched anchor.
- `dsp.hf_4k8k` / `dsp.hf_8k16k` stay as backstops only; the v3 grids never rank them.
"""

import json
from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields

SOFT = "soft"
HARD = "hard"
# A listener flag: a reading that reproduced what the user heard on the Tata tapes (dead
# pauses, hiss left, thinned sibilants, softened attacks) and is counted by the tuning
# loop, but does not veto on its own until a second listening round confirms its threshold.
FLAG = "flag"
# The two aggregate statistics a gate reads: the window median and the bad-end tail.
MEDIAN = "delta.median"
TAIL = "delta.tail"


@dataclass(frozen=True)
class Gate:
    """One veto rule. `severity` is "hard" (vetoes), "flag" (counted by the loop) or "soft" (reported only)."""

    metric: str
    stat: str
    op: str
    threshold: object
    note: str = ""
    severity: str = "hard"
    calibrated: bool = True


# The keys a gates file entry may set; anything else (its `source`, `n_verdicts`, ...) is provenance.
GATE_FIELDS = frozenset(item.name for item in dataclass_fields(Gate))

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
    "speech.logprob": Gate("speech.avg_logprob", TAIL, ">=", -0.15, "ASR confidence must not fall"),
    "mos.sigmos_col": Gate("mos.sigmos_col", MEDIAN, ">=", -0.3, "coloration: muffled / metallic timbre"),
    "mos.sigmos_disc": Gate("mos.sigmos_disc", TAIL, ">=", -0.3, "discontinuity: chopping, musical noise"),
    "mos.sigmos_noise": Gate("mos.sigmos_noise", MEDIAN, ">=", 0.0, "the output must not read noisier than the source"),
    "stems.octave": Gate("stems.octave_ratio_db", "tail", ">=", -3.0, "background stripped: non-vocal stem down in some octave"),
    "dsp.hf_4k8k": Gate("dsp.hf_4k8k", MEDIAN, ">=", -8.0, "presence band lost: muffled speech"),
    "dsp.pause_pumping": Gate("dsp.pause_pumping_db", MEDIAN, "<=", 12.0, "pause floor switches on and off between words", "soft"),
    "dsp.hf_8k16k": Gate("dsp.hf_8k16k", MEDIAN, ">=", -12.0, "air band lost: sibilance eaten"),
    "dsp.lkr": Gate("dsp.lkr", MEDIAN, "<=", 0.3, "musical noise"),
    "dsp.hum": Gate("dsp.hum_excess_db", MEDIAN, "<=", 3.0, "hum must not appear"),
    "dsp.clicks": Gate("dsp.clicks_per_s", TAIL, "<=", 0.5, "clicks must not appear"),
    "dsp.dropouts": Gate("dsp.dropouts", TAIL, "<=", 1.0, "no new holes in the programme"),
    "dsp.whistle": Gate("dsp.whistle_db", MEDIAN, "<=", 3.0, "line whistle must not appear"),
    "file.lra": Gate("file.lra", MEDIAN, ">=", -3.0, "dynamics squashed"),
    "file.lufs": Gate("file.lufs", MEDIAN, "abs<=", 1.5, "level moved", SOFT),
    "speech.utmos": Gate("speech.utmos", MEDIAN, ">=", -0.3, "naturalness", SOFT),
    "mos.audiobox_pq": Gate("mos.audiobox_pq", MEDIAN, ">=", -0.3, "production quality", SOFT),
    "mos.audiobox_pc": Gate("mos.audiobox_pc", MEDIAN, ">=", -0.5, "scene simplified: background stripped"),
    "mos.dnsmos_gap": Gate("mos.dnsmos_gap", MEDIAN, ">=", -0.7, "SIG under BAK: the speech paid for the quiet", SOFT),
    "speech.hallucinated": Gate("speech.hallucinated", "tail", "<=", 0.0, "Whisper invents text where the output is silent", SOFT),
    "dsp.output_silent": Gate("dsp.output_silent", "tail", "<=", 0.0, "a window with programme fell silent"),
    "file.sync_drift": Gate("file.sync_drift_ms", "median", "<=", 40.0, "the sync drifts a PAL video frame along the file"),
    "file.sync_offset": Gate("file.sync_offset_ms", "median", "<=", 40.0, "the output runs a PAL video frame off the source"),
    "file.sync_unmatched": Gate("file.sync_unmatched", "median", "<=", 0.0, "the output no longer follows the source"),
    # Listener flags; starting values from the Tata listening set, re-derived by calibration.
    "listener.hiss": Gate("dsp.gap_air_db", "median", "<=", -20.0, "air left in the inter-word gaps (display-only: R4 decides)", SOFT),
    "listener.dead_air": Gate("dsp.gap_air_db", "median", ">=", -25.0, "the gaps emptied: silent pauses", FLAG),
    "listener.pause_collapse": Gate("dsp.pause_depth_db", MEDIAN, "<=", 34.0, "the pauses dropped far under the speech", FLAG),
    # Thin and dull are two flags with different bounds: a de-esser lowers the top of every 's'
    # (cathar reads -100..-380 Hz on Tele7abc and the listener did not object), the neural
    # stage empties the body under it (APL +370..+620 Hz: "distortion of spoken 's'").
    "listener.sibilance_thin": Gate("dsp.sib_centroid_hz", MEDIAN, "<=", 300.0, "the 's' lost its body: thin, lisping", FLAG),
    "listener.sibilance_dull": Gate("dsp.sib_centroid_hz", MEDIAN, ">=", -600.0, "the 's' lost its top: dull", FLAG),
    "listener.attack": Gate("dsp.attack_db", TAIL, ">=", -4.0, "attacks softened: transients smeared", FLAG),
    "listener.bright": Gate("dsp.balance_top_db", "median", "<=", 1.0, "presence or air lifted: bright, thin", FLAG, False),
    "listener.dull": Gate("dsp.balance_top_db", "median", ">=", -1.0, "presence or air lost: dull, underwater", FLAG, False),
}


# Gates ear v3 demoted to display-only: a gates file still sets their threshold, never their
# severity, so a stored v2 calibration cannot make them count again (see the module docstring).
DISPLAY_ONLY = frozenset({"listener.hiss"})


def load_gates(path, base=GATES):
    """`base` with the entries of a JSON file merged over it: `{name: {metric, stat, op, threshold, note?, severity?}}`.

    A `DISPLAY_ONLY` gate the base defines keeps the base's severity whatever the file says.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    merged = dict(base)
    for name, fields in raw.items():
        merged[name] = _merged_gate(base.get(name), _file_fields(name, fields, base))
    return merged


def _file_fields(name, fields, base):
    """A file entry's fields; a display-only gate the base defines loses the entry's `severity`."""
    if name in DISPLAY_ONLY and name in base:
        fields = dict(fields)
        fields.pop("severity", None)
    return fields


def _merged_gate(gate, fields):
    """`gate` (or a new one) with a calibration entry's known fields over it; an entry naming its `source` is calibrated."""
    known = {key: value for key, value in fields.items() if key in GATE_FIELDS}
    if "source" in fields:
        known.setdefault("calibrated", True)
    return Gate(**{**asdict(gate), **known}) if gate else Gate(**known)


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


def _status(value, gate, threshold):
    """skipped when the value or the threshold is unknown, else passed / failed."""
    if value is None or threshold is None:
        return "skipped"
    return "passed" if _passes(value, gate.op, threshold) else "failed"


def evaluate_gates(aggregate, gates=GATES, speaker_floor=None):
    """Verdicts for every gate: passed / failed / skipped (metric absent or floor unknown)."""
    verdicts = []
    for name, gate in gates.items():
        value, threshold = _read(aggregate, gate), _threshold(gate, speaker_floor)
        status = _status(value, gate, threshold)
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
                "calibrated": gate.calibrated,
            }
        )
    return verdicts


def hard_failures(verdicts):
    """Names of the hard gates that failed."""
    return _failed(verdicts, HARD)


def flag_failures(verdicts):
    """Names of the listener flags that failed."""
    return _failed(verdicts, FLAG)


def _failed(verdicts, severity):
    return [verdict["gate"] for verdict in verdicts if verdict["status"] == "failed" and verdict["severity"] == severity]
