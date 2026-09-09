#!/usr/bin/env python3
"""Separates noise removal from programme damage on real tapes, without a clean reference.

Every corpus metric used on this branch collapses both halves of the trade into one number.
An attenuation ratio rises whether a mode removed the defect or removed the program with it,
which is why it ranked a stage that cut 6.7-10.7 dB of speech energy as an improvement. The
paired fixtures separate the two correctly, but they are synthetic speech, so a reader can
reasonably ask whether the finding transfers to real tape.

This measures both halves directly on real captures, using the source itself to decide where
to look:

- **Noise removed** is the level drop in the frames the source says are quiet. Those frames
  are mostly noise, so a drop there is noise going away. Higher is better.
- **Programme deviation** is how far the restored spectrum moves in the frames the source
  says are loud, inside the 300-3400 Hz band where speech lives. Those frames are dominated
  by content, so movement there is the restoration altering the programme. Lower is better.

Both are computed after matching gain on the loud frames, so the final loudness
normalisation cannot flatter or penalise either number.

The two are reported separately and never combined. Which one matters more is a judgement,
and collapsing them into a single score is precisely the mistake this replaces.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.hardware import get_gpu_name
from modules.utils import FFMPEG_BIN
from scripts.ia_benchmark_common import _run_mode_restoration
from scripts.score_reference import _align

FRAME = 1024
QUIET_PERCENTILE = 20.0
LOUD_PERCENTILE = 70.0
SPEECH_BAND_HZ = (300.0, 3400.0)
# The metric separates quiet frames from loud ones and reads noise from the first and
# programme from the second. A source whose 20th and 70th percentile frame levels sit within
# this many dB of each other has no such separation to read -- 20 corpus clips have a crest
# factor near 0 dB, a saturated or constant-level track rather than audio -- and on those
# both figures measure nothing: "deviation" reads 17.86 dB against 0.41 everywhere else,
# identically for both modes. They are refused rather than reported.
MIN_DYNAMIC_SPREAD_DB = 3.0


def _extract(video_path, target):
    """Pulls float audio out of a container."""
    subprocess.run(
        [FFMPEG_BIN, "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_f32le", "-ar", "44100", str(target)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=300,
    )
    return target if target.exists() else None


def _mono(path):
    """Reads a file as float32 mono."""
    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return samples.mean(axis=1), rate


def _frames(signal_data):
    """Reshapes into non-overlapping analysis frames."""
    count = len(signal_data) // FRAME
    return signal_data[: count * FRAME].reshape(count, FRAME) if count else None


def _band_energy(block, rate):
    """Energy inside the speech band for one block of frames."""
    spectrum = np.abs(np.fft.rfft(block, axis=1))
    freqs = np.fft.rfftfreq(FRAME, 1.0 / rate)
    band = (freqs >= SPEECH_BAND_HZ[0]) & (freqs <= SPEECH_BAND_HZ[1])
    return float(np.sum(spectrum[:, band] ** 2)) + 1e-20


def measure(source_path, restored_path):
    """Returns noise removed and programme deviation, both in dB."""
    source, rate = _mono(source_path)
    restored, _rate = _mono(restored_path)
    source, restored, _lag = _align(source, restored)
    source_frames, restored_frames = _frames(source), _frames(restored)
    if source_frames is None or restored_frames is None or len(source_frames) < 8:
        return None

    count = min(len(source_frames), len(restored_frames))
    source_frames, restored_frames = source_frames[:count], restored_frames[:count]
    level = np.sqrt(np.mean(source_frames**2, axis=1))
    quiet_cut, loud_cut = np.percentile(level, QUIET_PERCENTILE), np.percentile(level, LOUD_PERCENTILE)
    if 20.0 * np.log10((loud_cut + 1e-12) / (quiet_cut + 1e-12)) < MIN_DYNAMIC_SPREAD_DB:
        return None
    quiet = level <= quiet_cut
    loud = level >= loud_cut
    if not quiet.any() or not loud.any():
        return None

    # Gain-match on the loud frames so loudness normalisation cannot move either number.
    source_loud = float(np.sqrt(np.mean(source_frames[loud] ** 2))) + 1e-12
    restored_loud = float(np.sqrt(np.mean(restored_frames[loud] ** 2))) + 1e-12
    restored_frames = restored_frames * (source_loud / restored_loud)

    source_quiet = float(np.sqrt(np.mean(source_frames[quiet] ** 2))) + 1e-12
    restored_quiet = float(np.sqrt(np.mean(restored_frames[quiet] ** 2))) + 1e-12
    noise_removed_db = 20.0 * np.log10(source_quiet / restored_quiet)

    deviation_db = abs(10.0 * np.log10(_band_energy(restored_frames[loud], rate) / _band_energy(source_frames[loud], rate)))
    return {"noise_removed_db": round(float(noise_removed_db), 3), "programme_deviation_db": round(float(deviation_db), 3)}


def _existing_output(work_root, clip, mode):
    """The restored container a previous run left in the work dir, or None."""
    work = work_root / clip.stem / mode
    found = sorted(work.glob("*_Cleaned.*")) if work.is_dir() else []
    return found[0] if found else None


def _measure_pair(clip, restored_video, mode, temp_dir):
    """Extracts both tracks, measures them, and removes the extracted WAVs whatever happened.

    A corpus is thousands of clips and each extraction is 2.6 MB; kept until the run's end
    they filled the temporary directory. An extraction that times out skips this clip.
    """
    source_wav = temp_dir / f"{clip.stem}_src.wav"
    restored_wav = temp_dir / f"{clip.stem}_{mode}.wav"
    try:
        if _extract(clip, source_wav) is None or _extract(restored_video, restored_wav) is None:
            return None
        return measure(source_wav, restored_wav)
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"  extraction timed out: {clip.name} ({mode})\n")
        return None
    finally:
        source_wav.unlink(missing_ok=True)
        restored_wav.unlink(missing_ok=True)


def _restore_and_measure(clip, mode, work_root, gpu_name, temp_dir, rescore=False):
    """Runs one mode over one clip and measures the trade."""
    work = work_root / clip.stem / mode
    work.mkdir(parents=True, exist_ok=True)
    restored_video = _existing_output(work_root, clip, mode) if rescore else _run_mode_restoration(clip, mode, work, gpu_name)
    if restored_video is None:
        return None
    return _measure_pair(clip, restored_video, mode, temp_dir)


def _summarise(rows, label):
    """Prints medians for one configuration."""
    removed = [r["noise_removed_db"] for r in rows]
    deviation = [r["programme_deviation_db"] for r in rows]
    print(f"{label:<26}{np.median(removed):>16.2f}{np.median(deviation):>22.2f}   n={len(rows)}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("experiments/ia_corpus_1000"))
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--modes", nargs="+", default=["cathar", "auto_pure_linear"])
    parser.add_argument("--label", default="auto_pure_linear", help="Name for the configuration under test")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--work-dir", type=Path, default=Path("experiments/tradeoff_work"))
    parser.add_argument("--report", type=Path, default=Path("experiments/tradeoff.json"))
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Measure restored outputs already in --work-dir instead of restoring; a metric change re-reads the same audio",
    )
    return parser.parse_args()


def main():
    """Measures the trade for each requested mode over real captures."""
    args = _parse_args()
    catalog_path = args.catalog or (args.corpus_dir / "catalog_1000.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    gpu_name = None if args.rescore else get_gpu_name()
    # The configuration under test is auto_pure_linear; --label names its row and its key
    # in the report, so a sweep's variants can be told apart.
    labels = {mode: (args.label if mode == "auto_pure_linear" else mode) for mode in args.modes}
    if len(set(labels.values())) != len(labels):
        raise SystemExit(f"--label {args.label!r} collides with a requested mode; the rows would be merged into one")
    results = {labels[mode]: [] for mode in args.modes}

    with tempfile.TemporaryDirectory(prefix="tradeoff_") as temp:
        temp_dir = Path(temp)
        seen = 0
        for record in catalog:
            if seen >= args.limit:
                break
            clip = args.corpus_dir / record["file"]
            if not clip.exists():
                continue
            seen += 1
            for mode in args.modes:
                row = _restore_and_measure(clip, mode, args.work_dir, gpu_name, temp_dir, rescore=args.rescore)
                if row:
                    row["identifier"] = record["identifier"]
                    results[labels[mode]].append(row)
            sys.stdout.write(f"  [{seen}/{args.limit}] {record['identifier'][:44]}\n")
            sys.stdout.flush()

    print(f"\n{'configuration':<26}{'noise removed dB':>16}{'programme deviation dB':>22}")
    print(f"{'':<26}{'(higher better)':>16}{'(lower better)':>22}")
    for label, rows in results.items():
        if rows:
            _summarise(rows, label)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
