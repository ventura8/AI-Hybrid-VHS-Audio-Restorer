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
  are mostly noise, so a drop there is noise going away. Higher is better. It is read twice:
  `noise_removed_db` on the quiet frames' pooled energy, and `noise_removed_median_db` as
  the median of the per-frame drops. The pooled figure follows the loudest few frames: on
  SOTI the hottest 1 % of the shipped APL output's quiet frames hold 34 % of their energy
  (222 of its 3739 quiet frames stand 10 dB or more over the source, cathar's 2; the
  backlog traces them to the repair stage's pop refills, unconfirmed), and the pooled
  reading says APL removed -5.17 dB where the median says +3.00 (cathar +9.61 pooled,
  +10.40 median; measured 2026-10-11).
- **Programme deviation** is how far the restored spectrum moves in the frames the source
  says are loud, inside the 300-3400 Hz band where speech lives. Those frames are dominated
  by content, so movement there is the restoration altering the programme. Lower is better.

Both are computed after matching gain on the loud frames, so the final loudness
normalisation cannot flatter or penalise either number, and on a pair put on one timeline
by `scripts/pair_alignment.py` (`--align-method`; `xcorr_head` reproduces reports written
before 2026-10-11, and every report row names the method).

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
from scripts import pair_alignment
from scripts.cli_paths import confined_path, existing_path_arg, path_arg
from scripts.ia_benchmark_common import _run_mode_restoration

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
    result = subprocess.run(
        [FFMPEG_BIN, "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_f32le", "-ar", "44100", str(target)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        # A failed decode can leave a partial file, which would then be measured as audio.
        Path(target).unlink(missing_ok=True)
        return None
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


def measure(source_path, restored_path, align_method=None, align_prior_s=None, with_alignment=False):
    """Returns noise removed (pooled and per-frame median) and programme deviation, all in dB.

    `align_method` and `align_prior_s` (seconds) reach `pair_alignment.align`; with
    `with_alignment` the row also carries the alignment's record under "alignment".
    """
    source, rate = _mono(source_path)
    restored, _rate = _mono(restored_path)
    prior = None if align_prior_s is None else int(round(align_prior_s * rate))
    source, restored, alignment = pair_alignment.align(source, restored, rate, align_method, prior)
    row = trade_readings(source, restored, rate)
    if row is not None and with_alignment:
        row["alignment"] = alignment.as_dict()
    return row


def common_frames(source, restored):
    """Both signals' analysis frames cut to the same count, or None when either has none or the source fewer than eight."""
    source_frames, restored_frames = _frames(source), _frames(restored)
    if source_frames is None or restored_frames is None or len(source_frames) < 8:
        return None
    count = min(len(source_frames), len(restored_frames))
    return source_frames[:count], restored_frames[:count]


def quiet_and_loud(source, restored):
    """`(source frames, restored frames, quiet, loud)` of an aligned pair, None when it has no quiet-to-loud spread to read."""
    frames = common_frames(source, restored)
    if frames is None:
        return None
    level = np.sqrt(np.mean(frames[0] ** 2, axis=1))
    quiet_cut, loud_cut = np.percentile(level, QUIET_PERCENTILE), np.percentile(level, LOUD_PERCENTILE)
    if 20.0 * np.log10((loud_cut + 1e-12) / (quiet_cut + 1e-12)) < MIN_DYNAMIC_SPREAD_DB:
        return None
    quiet, loud = level <= quiet_cut, level >= loud_cut
    return (*frames, quiet, loud) if quiet.any() and loud.any() else None


def trade_readings(source, restored, rate):
    """The trade readings of an aligned pair at `rate`, None when the source has no quiet-to-loud spread."""
    split = quiet_and_loud(source, restored)
    if split is None:
        return None
    source_frames, restored_frames, quiet, loud = split
    # Gain-match on the loud frames so loudness normalisation cannot move either number.
    source_loud = float(np.sqrt(np.mean(source_frames[loud] ** 2))) + 1e-12
    restored_loud = float(np.sqrt(np.mean(restored_frames[loud] ** 2))) + 1e-12
    restored_frames = restored_frames * (source_loud / restored_loud)

    source_quiet = float(np.sqrt(np.mean(source_frames[quiet] ** 2))) + 1e-12
    restored_quiet = float(np.sqrt(np.mean(restored_frames[quiet] ** 2))) + 1e-12
    noise_removed_db = 20.0 * np.log10(source_quiet / restored_quiet)

    deviation_db = abs(10.0 * np.log10(_band_energy(restored_frames[loud], rate) / _band_energy(source_frames[loud], rate)))
    return {
        "noise_removed_db": round(float(noise_removed_db), 3),
        "noise_removed_median_db": round(median_drop_db(source_frames[quiet], restored_frames[quiet]), 3),
        "programme_deviation_db": round(float(deviation_db), 3),
    }


def median_drop_db(source_frames, restored_frames):
    """The median over frames of each frame's level drop in dB: a few hot frames cannot own it as they own the pooled energy."""
    source_rms = np.sqrt(np.mean(np.asarray(source_frames, dtype=np.float64) ** 2, axis=1)) + 1e-12
    restored_rms = np.sqrt(np.mean(np.asarray(restored_frames, dtype=np.float64) ** 2, axis=1)) + 1e-12
    return float(np.median(20.0 * np.log10(source_rms / restored_rms)))


def _existing_output(work_root, clip, mode):
    """The restored container a previous run left in the work dir, or None."""
    work = work_root / clip.stem / mode
    found = sorted(work.glob("*_Cleaned.*")) if work.is_dir() else []
    return found[0] if found else None


def _measure_pair(clip, restored_video, mode, temp_dir, align_method=None):
    """Extracts both tracks, measures them, and removes the extracted WAVs whatever happened.

    A corpus is thousands of clips and each extraction is 2.6 MB; kept until the run's end
    they filled the temporary directory. An extraction that times out skips this clip.
    """
    source_wav = temp_dir / f"{clip.stem}_src.wav"
    restored_wav = temp_dir / f"{clip.stem}_{mode}.wav"
    try:
        if _extract(clip, source_wav) is None or _extract(restored_video, restored_wav) is None:
            return None
        return measure(source_wav, restored_wav, align_method=align_method, with_alignment=True)
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"  extraction timed out: {clip.name} ({mode})\n")
        return None
    finally:
        source_wav.unlink(missing_ok=True)
        restored_wav.unlink(missing_ok=True)


def _restore_and_measure(clip, mode, work_root, gpu_name, temp_dir, rescore=False, align_method=None):
    """Runs one mode over one clip and measures the trade."""
    work = work_root / clip.stem / mode
    work.mkdir(parents=True, exist_ok=True)
    restored_video = _existing_output(work_root, clip, mode) if rescore else _run_mode_restoration(clip, mode, work, gpu_name)
    if restored_video is None:
        return None
    return _measure_pair(clip, restored_video, mode, temp_dir, align_method)


def _summarise(rows, label):
    """Prints medians for one configuration."""
    removed = [r["noise_removed_db"] for r in rows]
    removed_median = [r["noise_removed_median_db"] for r in rows]
    deviation = [r["programme_deviation_db"] for r in rows]
    print(f"{label:<26}{np.median(removed):>16.2f}{np.median(removed_median):>18.2f}{np.median(deviation):>22.2f}   n={len(rows)}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=existing_path_arg, default=Path("experiments/ia_corpus_1000"))
    parser.add_argument("--catalog", type=existing_path_arg, default=None)
    parser.add_argument("--modes", nargs="+", default=["cathar", "auto_pure_linear"])
    parser.add_argument("--label", default="auto_pure_linear", help="Name for the configuration under test")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--work-dir", type=path_arg, default=Path("experiments/tradeoff_work"))
    parser.add_argument("--report", type=path_arg, default=Path("experiments/tradeoff.json"))
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Measure restored outputs already in --work-dir instead of restoring; a metric change re-reads the same audio",
    )
    parser.add_argument(
        "--align-method",
        choices=pair_alignment.ALIGN_METHODS,
        default=None,
        help="Aligner (default: AI_RESTORE_ALIGN_METHOD, else gcc_phat_spans); xcorr_head reproduces reports before 2026-10-11",
    )
    return parser.parse_args()


def _present_clips(catalog, corpus_dir, limit):
    """The first `limit` catalog records whose clip exists, with the clip path."""
    seen = 0
    for record in catalog:
        if seen >= limit:
            return
        clip = corpus_dir / record["file"]
        if clip.exists():
            seen += 1
            yield record, clip


def _measure_clip_modes(clip, record, args, labels, results, context):
    """Restores and measures one clip in every requested mode, appending the rows."""
    gpu_name, temp_dir = context
    for mode in args.modes:
        row = _restore_and_measure(clip, mode, args.work_dir, gpu_name, temp_dir, rescore=args.rescore, align_method=args.align_method)
        if row:
            row["identifier"] = record["identifier"]
            results[labels[mode]].append(row)


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
        for seen, (record, clip) in enumerate(_present_clips(catalog, args.corpus_dir, args.limit), start=1):
            _measure_clip_modes(clip, record, args, labels, results, (gpu_name, temp_dir))
            sys.stdout.write(f"  [{seen}/{args.limit}] {record['identifier'][:44]}\n")
            sys.stdout.flush()

    print(f"\n{'configuration':<26}{'noise removed dB':>16}{'median per frame':>18}{'programme deviation dB':>22}")
    print(f"{'':<26}{'(higher better)':>16}{'(higher better)':>18}{'(lower better)':>22}")
    for label, rows in results.items():
        if rows:
            _summarise(rows, label)
    report_path = confined_path(args.report, "--report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
