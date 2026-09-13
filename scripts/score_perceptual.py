#!/usr/bin/env python3
"""Reads DNSMOS P.835 on the source and each mode's restoration of every clip: a perceptual cross-check.

Every ranking on this branch rests on the trade metric, and the trade metric is blind to
what it does not measure -- a stage can move its two numbers the right way and sound worse.
DNSMOS is a reference-free estimate of how a listener would rate a clip on the P.835 scales
(speech quality SIG, background BAK, overall OVRL) plus a P.808 overall MOS, from models
trained on crowd ratings of noise suppressors. It is speech-trained and reads at 16 kHz, so
on a tape corpus that carries music and archive material it is a cross-check and not a gate:
a change that the trade metric and the defect gates approve and that DNSMOS reads clearly
worse is a change to listen to before it ships.

The feature pipeline mirrors the reference `dnsmos_local.py`: 16 kHz, 9.01 s windows at a
one-second hop averaged over the clip, the raw outputs mapped through the published
polynomials, the P.808 model fed a 120-band log-mel spectrogram.
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_dnsmos import MODELS, TARGET_DIR
from scripts.measure_hum import _restored_path
from scripts.measure_tradeoff import _extract

SAMPLE_RATE = 16000
WINDOW_S = 9.01
HOP_S = 1.0
MEL_BANDS = 120
MEL_FRAME = 320
MEL_HOP = 160
# The published mappings from raw network outputs to MOS, non-personalised.
POLY_OVR = (-0.06766283, 1.11546468, 0.04602535)
POLY_SIG = (-0.08397278, 1.22083953, 0.0052439)
POLY_BAK = (-0.13166888, 1.60915514, -0.39604546)
SCALES = ("SIG", "BAK", "OVRL", "P808")


def _sessions():
    """The two ONNX sessions, on the CPU: the models are small and the corpus is not the bottleneck."""
    import onnxruntime

    paths = [TARGET_DIR / name for name in MODELS]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"DNSMOS models missing ({', '.join(m.name for m in missing)}); run scripts/download_dnsmos.py first")
    providers = ["CPUExecutionProvider"]
    return onnxruntime.InferenceSession(str(paths[0]), providers=providers), onnxruntime.InferenceSession(
        str(paths[1]), providers=providers
    )


def _mono_16k(path):
    """The file as mono at 16 kHz, resampled the way the reference does."""
    import librosa
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return librosa.resample(mono, orig_sr=rate, target_sr=SAMPLE_RATE) if rate != SAMPLE_RATE else mono


def _log_mel(segment):
    """The P.808 model's input: 120-band log-mel, normalised as the reference does."""
    import librosa

    mel = librosa.feature.melspectrogram(y=segment, sr=SAMPLE_RATE, n_fft=MEL_FRAME + 1, hop_length=MEL_HOP, n_mels=MEL_BANDS)
    return ((librosa.power_to_db(mel, ref=np.max) + 40.0) / 40.0).T.astype(np.float32)[np.newaxis]


def _windows(audio):
    """The clip's 9.01 s windows at a one-second hop; a clip shorter than one window is repeated to fill it.

    An empty clip has no windows: doubling nothing never fills a window, and a source the
    metric refuses as degenerate can extract to nothing.
    """
    length = int(WINDOW_S * SAMPLE_RATE)
    if len(audio) == 0:
        return []
    while len(audio) < length:
        audio = np.append(audio, audio)
    hops = int(np.floor(len(audio) / SAMPLE_RATE) - WINDOW_S) + 1
    return [audio[int(index * HOP_S * SAMPLE_RATE) : int(index * HOP_S * SAMPLE_RATE) + length] for index in range(hops)]


def score_audio(audio, sessions):
    """The clip's mean SIG, BAK, OVRL and P808 over its windows."""
    primary, p808 = sessions
    scores = []
    windows = _windows(audio.astype(np.float32))
    if not windows:
        return None
    for segment in windows:
        raw_sig, raw_bak, raw_ovr = primary.run(None, {"input_1": segment[np.newaxis]})[0][0]
        mos_808 = float(p808.run(None, {"input_1": _log_mel(segment[:-MEL_HOP])})[0][0][0])
        scores.append((np.polyval(POLY_SIG, raw_sig), np.polyval(POLY_BAK, raw_bak), np.polyval(POLY_OVR, raw_ovr), mos_808))
    return dict(zip(SCALES, (round(float(v), 3) for v in np.mean(scores, axis=0))))


def score_file(path, sessions):
    """DNSMOS for one audio file."""
    return score_audio(_mono_16k(path), sessions)


def _extracted(video_path, target):
    try:
        return _extract(video_path, target)
    except subprocess.TimeoutExpired:
        return None


def _score_clip(clip, record, args, sessions, temp_dir, results):
    """Scores the source and every mode's restoration of one clip."""
    source_wav = temp_dir / f"{clip.stem}_src.wav"
    if _extracted(clip, source_wav) is None:
        return
    scored = score_file(source_wav, sessions)
    source_wav.unlink(missing_ok=True)
    if scored is None:
        return
    results["source"].append({"identifier": record["identifier"], **scored})
    for mode in args.modes:
        restored = next((p for root in args.work_dirs if (p := _restored_path(root, clip, mode))), None)
        restored_wav = temp_dir / f"{clip.stem}_{mode}.wav"
        if restored is None or _extracted(restored, restored_wav) is None:
            continue
        scored = score_file(restored_wav, sessions)
        restored_wav.unlink(missing_ok=True)
        if scored is not None:
            results[mode].append({"identifier": record["identifier"], **scored})


def _report(results):
    """Medians per configuration and the paired deltas against the source."""
    source = {row["identifier"]: row for row in results["source"]}
    print(f"\n{'configuration':<20}" + "".join(f"{scale:>8}" for scale in SCALES) + "   paired delta vs source (OVRL, n)")
    for name, rows in results.items():
        if not rows:
            continue
        medians = "".join(f"{statistics.median(r[scale] for r in rows):>8.2f}" for scale in SCALES)
        paired = [r["OVRL"] - source[r["identifier"]]["OVRL"] for r in rows if r["identifier"] in source]
        delta = f"{statistics.median(paired):+.2f} ({len(paired)})" if paired and name != "source" else ""
        print(f"{name:<20}{medians}   {delta}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("experiments/ia_corpus_1000"))
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--work-dirs", nargs="+", type=Path, required=True, help="Benchmark work dirs holding restored outputs")
    parser.add_argument("--modes", nargs="+", default=["cathar", "auto_pure_linear"])
    parser.add_argument("--limit", type=int, default=0, help="Clips to score; 0 takes the whole catalog")
    parser.add_argument("--report", type=Path, default=Path("experiments/perceptual.json"))
    return parser.parse_args()


def main():
    """Scores the catalog and writes the report."""
    args = _parse_args()
    catalog_path = args.catalog or (args.corpus_dir / "catalog_1000.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if args.limit:
        catalog = catalog[: args.limit]
    sessions = _sessions()
    results = {"source": [], **{mode: [] for mode in args.modes}}
    with tempfile.TemporaryDirectory(prefix="dnsmos_") as temp:
        for record in catalog:
            clip = args.corpus_dir / record["file"]
            if clip.exists():
                _score_clip(clip, record, args, sessions, Path(temp), results)
    _report(results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
