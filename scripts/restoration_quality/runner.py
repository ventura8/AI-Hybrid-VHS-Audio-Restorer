"""Orchestration: load a pair, window it, run each metric family, aggregate, gate.

Families run in the order dsp -> stems -> speech -> mos with one model resident at a
time. A family whose model or library is missing is recorded as unavailable with the
reason; it never aborts the run, so `--metrics dsp` needs neither torch nor a download.
"""

import gc
import json
from pathlib import Path

import numpy as np

from scripts.restoration_quality import audio_io, dsp_metrics, pause_metrics, sibilance, transient_metrics
from scripts.restoration_quality.gates import GATES, evaluate_gates, flag_failures, hard_failures
from scripts.restoration_quality.scorecard import METRICS, ScoreCard, WindowRow, aggregate

ALL_FAMILIES = ("dsp", "stems", "speech", "mos")
SCHEMA = 1
MOS_TARGET_LUFS = -23.0
PEAK_GUARD = 0.99
UNAVAILABLE_ERRORS = (ImportError, SystemExit, RuntimeError, OSError, ValueError, AttributeError, TypeError, KeyError)


class ModelRegistry:
    """Lazy model store: `get(name)` loads on first use, `release()` frees the GPU between families."""

    def __init__(self, device="cuda", models_dir=None):
        self.device = device
        self.models_dir = Path(models_dir) if models_dir else None
        self._loaded = {}

    def get(self, name, loader):
        if name not in self._loaded:
            self._loaded[name] = loader(self.device, self.models_dir)
        return self._loaded[name]

    def release(self):
        self._loaded.clear()
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass


class Pair:
    """The aligned, gain-matched source/output pair and its windows, shared by every family."""

    def __init__(self, source_wav, output_wav, cache_dir, seconds, hop):
        self.cache_dir = Path(cache_dir)
        # Path-based consumers (the stem separator, the trade metric) must see DC-free audio too.
        self.source_wav = audio_io.dc_free_copy(source_wav, self.cache_dir)
        self.output_wav = audio_io.dc_free_copy(output_wav, self.cache_dir)
        source_audio, self.rate = audio_io.load_audio(self.source_wav)
        output_audio, output_rate = audio_io.load_audio(self.output_wav)
        if output_rate != self.rate:
            output_audio = audio_io._resample(audio_io.to_mono(output_audio), output_rate, self.rate)[:, None]
        self.raw_source, self.raw_output = audio_io.to_mono(source_audio), audio_io.to_mono(output_audio)
        self.source, self.output, self.lag = audio_io.align_pair(self.raw_source, self.raw_output)
        self.windows = audio_io.windows(len(self.source), self.rate, seconds, hop)
        self.routes = [audio_io.route_window(self.source[w.slice_of(self.rate)], self.rate) for w in self.windows]
        # The output is routed too: a window with programme whose output reads as silence is dead air.
        self.output_routes = [audio_io.route_window(self.output[w.slice_of(self.rate)], self.rate) for w in self.windows]
        self.source_key, self.output_key = audio_io.file_key(self.source_wav), audio_io.file_key(self.output_wav)
        self._resampled = {}

    def at(self, side, rate):
        """`side` ("source" | "output") resampled to `rate` once and cached on disk and in memory."""
        if (side, rate) not in self._resampled:
            mono, key = (self.source, self.source_key) if side == "source" else (self.output, self.output_key)
            self._resampled[(side, rate)] = audio_io.resample_cached(mono, self.rate, rate, self.cache_dir, f"{key}_{self.lag}")
        return self._resampled[(side, rate)]

    def mos_gain(self):
        """One gain that brings the source to -23 LUFS, applied to both sides before the MOS models."""
        lufs, _lra = dsp_metrics.loudness(self.source, self.rate)
        gain = 10.0 ** ((MOS_TARGET_LUFS - lufs) / 20.0) if np.isfinite(lufs) else 1.0
        peak = max(float(np.abs(self.source).max()), float(np.abs(self.output).max()), 1e-9)
        return min(gain, PEAK_GUARD / peak)


def _rows_for(pair):
    output_routes = getattr(pair, "output_routes", None) or pair.routes
    return [
        WindowRow(w.index, w.start_s, w.end_s, route, output_route=out) for w, route, out in zip(pair.windows, pair.routes, output_routes)
    ]


def _dsp_window(pair, row):
    """Every DSP guardrail on one window; paired readings go on the output side with a zero source."""
    sl = row_slice(pair, row)
    src, out = pair.source[sl], pair.output[sl]
    row.source["dsp.residual_noise_db"], row.output["dsp.residual_noise_db"] = 0.0, dsp_metrics.residual_noise_db(src, out)
    row.source["dsp.lkr"], row.output["dsp.lkr"] = 0.0, dsp_metrics.lkr_musical_noise(src, out, pair.rate)
    row.source["dsp.dropouts"], row.output["dsp.dropouts"] = 0.0, float(dsp_metrics.dropout_count(src, out, pair.rate))
    for name, (src_ratio, out_ratio) in dsp_metrics.loud_band_ratios_db(src, out, pair.rate).items():
        row.source[f"dsp.{name}"], row.output[f"dsp.{name}"] = src_ratio, out_ratio
    for name, (src_value, out_value) in dsp_metrics.pause_dynamics(src, out, pair.rate).items():
        row.source[f"dsp.{name}"], row.output[f"dsp.{name}"] = src_value, out_value
    tilt = row.output.get("dsp.pause_tilt_db")
    row.source["dsp.pause_tilt_abs_db"], row.output["dsp.pause_tilt_abs_db"] = 0.0, (None if tilt is None else abs(tilt))
    _both(row, "dsp.clicks_per_s", dsp_metrics.click_density, src, out, pair.rate)
    _both(row, "dsp.whistle_db", dsp_metrics.whistle_line_db, src, out, pair.rate)
    _both(row, "dsp.hum_excess_db", dsp_metrics.hum_excess_db, src, out, pair.rate)


def _both(row, name, function, src, out, rate):
    row.source[name], row.output[name] = function(src, rate), function(out, rate)


def _listener_window(pair, row):
    """The listener readings on one window: the pauses as the ear hears them, and a window that fell silent."""
    sl = row_slice(pair, row)
    src, out = pair.source[sl], pair.output[sl]
    readings = {**pause_metrics.pause_readings(src, out, pair.rate), **sibilance.sib_readings(src, out, pair.rate)}
    if row.route in ("music", "mixed"):
        readings.update(transient_metrics.transient_readings(src, out, pair.rate))
    for name, (src_value, out_value) in readings.items():
        row.source[f"dsp.{name}"], row.output[f"dsp.{name}"] = src_value, out_value
    _abs_delta(row, "dsp.sib_centroid_hz", "dsp.sib_centroid_abs_hz")
    silent = row.output_route == "silence" and row.route != "silence"
    row.source["dsp.output_silent"], row.output["dsp.output_silent"] = 0.0, 1.0 if silent else 0.0


def _abs_delta(row, name, abs_name):
    """The rankable form of a signed paired reading: |output - source| on the output side, zero on the source side."""
    src_value, out_value = row.source.get(name), row.output.get(name)
    missing = src_value is None or out_value is None
    row.source[abs_name], row.output[abs_name] = 0.0, (None if missing else abs(out_value - src_value))


def row_slice(pair, row):
    return slice(int(round(row.start_s * pair.rate)), int(round(row.end_s * pair.rate)))


def _dsp_family(pair, card, _registry):
    for row in card.rows:
        _dsp_window(pair, row)
        _listener_window(pair, row)
    _zimtohrli(pair, card)
    src_lufs, src_lra = dsp_metrics.loudness(pair.raw_source, pair.rate)
    out_lufs, out_lra = dsp_metrics.loudness(pair.raw_output, pair.rate)
    card.file["file.lufs"] = {"source": src_lufs, "output": out_lufs, "delta": out_lufs - src_lufs}
    card.file["file.lra"] = {"source": src_lra, "output": out_lra, "delta": out_lra - src_lra}
    trade = dsp_metrics.trade(pair.source_wav, pair.output_wav) or {}
    for key, value in trade.items():
        card.file[f"file.{key}"] = {"source": 0.0, "output": value, "delta": value}


_UNAVAILABLE_SAID = set()


def _zimtohrli(pair, card):
    """The 48 kHz psychoacoustic distance on the loud frames; a missing binding costs only this reading, said once."""
    from scripts.restoration_quality import judges

    try:
        judges.score_zimtohrli(pair, card)
    except UNAVAILABLE_ERRORS as exc:
        if "zimtohrli" not in _UNAVAILABLE_SAID:
            _UNAVAILABLE_SAID.add("zimtohrli")
            print(f"dsp.zimtohrli_loud unavailable: {type(exc).__name__}: {exc}")


def _stems_family(pair, card, registry):
    from scripts.restoration_quality import stem_metrics

    stem_metrics.score(pair, card, registry)


def _speech_family(pair, card, registry):
    from scripts.restoration_quality import speech_models

    speech_models.score(pair, card, registry)


def _mos_family(pair, card, registry):
    from scripts.restoration_quality import mos_models

    mos_models.score(pair, card, registry)


FAMILIES = {"dsp": _dsp_family, "stems": _stems_family, "speech": _speech_family, "mos": _mos_family}


def _run_family(name, pair, card, registry):
    """Runs one family, recording success or the reason it could not run."""
    try:
        FAMILIES[name](pair, card, registry)
        card.families[name] = "ok"
    except UNAVAILABLE_ERRORS as exc:
        card.families[name] = f"unavailable: {type(exc).__name__}: {exc}"
    finally:
        registry.release()


def _file_aggregate(card):
    """File-level readings in the same shape as window aggregates so gates can read them."""
    for name, entry in card.file.items():
        card.aggregate[name] = {
            side: {"median": float(entry[side]), "tail": float(entry[side]), "n": 1}
            for side in ("source", "output", "delta")
            if entry.get(side) is not None
        }


def score_pair(
    source_wav,
    output_wav,
    *,
    windows=15.0,
    hop=7.5,
    families=ALL_FAMILIES,
    registry=None,
    cache_dir="experiments/quality_cache",
    gates=GATES,
    language="ro",
):
    """Scores `output_wav` against `source_wav`; returns a ScoreCard with rows, aggregates and verdicts."""
    registry = registry or ModelRegistry()
    registry.language = language
    pair = Pair(source_wav, output_wav, cache_dir, windows, hop)
    card = ScoreCard(rows=_rows_for(pair))
    card.lag = pair.lag
    for name in (family for family in ALL_FAMILIES if family in families):
        _run_family(name, pair, card, registry)
    card.aggregate = aggregate(card.rows, METRICS)
    _file_aggregate(card)
    card.verdicts = evaluate_gates(card.aggregate, gates, card.speaker_floor)
    card.passed = not hard_failures(card.verdicts)
    return card, pair


def card_to_dict(card, path):
    """A JSON-ready view of one card."""
    return {
        "path": str(path),
        "lag_samples": getattr(card, "lag", 0),
        "families": card.families,
        "rows": [
            {
                "window": r.window,
                "start_s": r.start_s,
                "end_s": r.end_s,
                "route": r.route,
                "output_route": r.output_route,
                "source": r.source,
                "output": r.output,
                "delta": r.delta(),
            }
            for r in card.rows
        ],
        "file": card.file,
        "aggregate": card.aggregate,
        "speaker_floor": card.speaker_floor,
        "verdicts": card.verdicts,
        "hard_failures": hard_failures(card.verdicts),
        "listener_flags": flag_failures(card.verdicts),
        "passed": getattr(card, "passed", None),
    }


def score_variants(source, outputs, *, cache_dir="experiments/quality_cache", gates=GATES, **options):
    """The JSON document for one source and several labelled outputs; cards and pairs are returned for the listening set."""
    cache_dir = Path(cache_dir)
    source_wav = audio_io.extract_wav(source, cache_dir)
    result = {
        "schema": SCHEMA,
        "source": {"path": str(source), "wav": str(source_wav), "key": audio_io.file_key(source_wav)},
        "variants": {},
    }
    cards = {}
    for label, path in outputs.items():
        output_wav = audio_io.extract_wav(path, cache_dir)
        card, pair = score_pair(source_wav, output_wav, cache_dir=cache_dir, gates=gates, **options)
        result["variants"][label] = card_to_dict(card, path)
        cards[label] = (card, pair)
    first = next(iter(cards.values()), None)
    result["windows"] = {
        "seconds": options.get("windows", 15.0),
        "hop": options.get("hop", 7.5),
        "count": len(first[0].rows) if first else 0,
    }
    result["metrics"] = {name: {"family": s.family, "better": s.better, "unit": s.unit} for name, s in METRICS.items()}
    result["gates"] = {
        name: {"metric": g.metric, "stat": g.stat, "op": g.op, "threshold": g.threshold, "severity": g.severity}
        for name, g in gates.items()
    }
    return result, cards


def aggregates_for_tuning(result, label):
    """Flat `{metric.side.stat: value}` for one variant, the shape a sweep ranks on."""
    variant = result["variants"][label]
    flat = {}
    for name, entry in variant["aggregate"].items():
        flat.update(_flat_entry(name, entry))
    flat["gates.hard_failures"] = float(len(variant["hard_failures"]))
    flat["gates.listener_flags"] = float(len(variant.get("listener_flags", [])))
    flat["gates.passed"] = 1.0 if variant["passed"] else 0.0
    return flat


def _flat_entry(name, entry):
    return {f"{name}.{side}.{stat}": entry[side][stat] for side in ("output", "delta") if entry.get(side) for stat in ("median", "tail")}


def write_result(result, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
