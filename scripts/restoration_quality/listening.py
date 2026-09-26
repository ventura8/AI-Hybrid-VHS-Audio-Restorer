"""The listening set: the windows each metric family found worst, cut from source and every output.

The user is the final judge; the harness only points at where to listen. Picks are
made per metric on the delta (output \u2212 source) in the metric's bad direction, and the
same window is rendered from the source and from every scored output so the ear
compares like with like.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.restoration_quality.scorecard import METRICS

DEFAULT_PICK_METRICS = ("mos.sigmos_col", "mos.sigmos_disc", "speech.cer", "dsp.hf_8k16k", "dsp.lkr", "dsp.residual_noise_db")


@dataclass(frozen=True)
class Pick:
    metric: str
    label: str
    window: int
    start_s: float
    end_s: float
    value: float


def _badness(metric, value):
    """Larger = worse, whatever the metric's direction."""
    return -value if METRICS[metric].better == "higher" else value


def select_worst(result, metric, count=3):
    """The `count` windows, across all outputs, where `metric`'s delta is worst."""
    candidates = []
    for label, variant in result["variants"].items():
        for row in variant["rows"]:
            value = row["delta"].get(metric)
            if value is not None:
                candidates.append(Pick(metric, label, row["window"], row["start_s"], row["end_s"], float(value)))
    candidates.sort(key=lambda pick: _badness(metric, pick.value), reverse=True)
    return candidates[:count]


def _write_window(audio, rate, start_s, end_s, path):
    begin, end = int(round(start_s * rate)), int(round(end_s * rate))
    sf.write(str(path), np.asarray(audio[begin:end], dtype=np.float32), rate, subtype="FLOAT")


def render_listening_set(result, cards, out_dir, metrics=DEFAULT_PICK_METRICS, count=3):
    """Writes `<out_dir>/<metric>_<rank>_<source|label>.wav` for every pick and an index.md; returns the index path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Listening set",
        "",
        "Each pick: the window where one output moved a metric furthest the wrong way, cut from the source and from every output.",
        "",
    ]
    for metric in metrics:
        for rank, pick in enumerate(select_worst(result, metric, count), start=1):
            lines.append(_render_pick(pick, rank, cards, out_dir))
    index = out_dir / "index.md"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def render_from_files(picks, source_wav, outputs, out_dir, prefix=""):
    """Cuts the picks' windows from `source_wav` and every output file (aligned to the source), for a run scored earlier.

    `outputs` is `{label: wav_or_video}`; returns the index lines written.
    """
    from scripts.restoration_quality import audio_io

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source, rate = audio_io.load_audio(source_wav)
    source = audio_io.to_mono(source)
    aligned = {}
    for label, path in outputs.items():
        output, _rate = audio_io.load_audio(audio_io.extract_wav(path, out_dir / "cache"))
        aligned[label] = audio_io.align_pair(source, audio_io.to_mono(output))
    lines = []
    for rank, pick in enumerate(picks, start=1):
        stem = f"{prefix}{pick.metric.replace('.', '_')}_{rank}"
        src, _out, _lag = next(iter(aligned.values()))
        _write_window(src, rate, pick.start_s, pick.end_s, out_dir / f"{stem}_source.wav")
        for label, (_src, out, _lag) in aligned.items():
            _write_window(out, rate, pick.start_s, pick.end_s, out_dir / f"{stem}_{label}.wav")
        lines.append(f"- `{stem}` — {pick.metric} delta {pick.value:+.3f} on **{pick.label}**, {pick.start_s:.1f}-{pick.end_s:.1f} s")
    return lines


def _render_pick(pick, rank, cards, out_dir):
    """Cuts one pick from the source and every output; returns its index line."""
    stem = f"{pick.metric.replace('.', '_')}_{rank}"
    _card, pair = cards[pick.label]
    _write_window(pair.source, pair.rate, pick.start_s, pick.end_s, out_dir / f"{stem}_source.wav")
    for label, (_other_card, other) in cards.items():
        _write_window(other.output, other.rate, pick.start_s, pick.end_s, out_dir / f"{stem}_{label}.wav")
    span = f"window {pick.window} ({pick.start_s:.1f}-{pick.end_s:.1f} s)"
    return f"- `{stem}` \u2014 {pick.metric} delta {pick.value:+.3f} on **{pick.label}**, {span}"
