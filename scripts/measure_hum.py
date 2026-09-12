#!/usr/bin/env python3
"""Measures mains hum removal on real tapes, separating the hum taken out from the speech
taken with it.

The claim that `cathar` leads on mains hum is carried from v1.2.0 and has never been
measured on the current build. It was also measured then with a ratio at the fundamental
alone, which saturates on removal and cannot tell a stage that cancelled the hum from one
that cut everything near it -- the same failure that withdrew every other ratio metric on
this branch.

Two figures, kept apart, on the clips that carry hum:

- **Hum removed** is the drop in harmonic excess -- energy at the mains fundamental and its
  harmonics above the local spectral floor -- from source to restored. Higher is better.
- **Low-band deviation** is how far the restored spectrum moves in the loud frames inside
  50-400 Hz, the range speech shares with hum. This is where dehum damages programme, and
  a stage that scores well on the first figure and badly on this one is removing voice.

Both are reference-free, because real tapes have no clean version, and both are gain-matched
on loud frames so loudness normalisation cannot move either number.
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
import scipy.signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.measure_tradeoff import LOUD_PERCENTILE, _extract, _frames, _mono
from scripts.score_reference import _align

HARMONICS = 8
HUM_BAND_HZ = (50.0, 400.0)
# A tape "carries hum" when its harmonic excess clears this. Set from the distribution over
# the corpus so the subset is the tapes where the defect is plainly present, not marginal.
HUM_PRESENT_DB = 6.0


def _mains_hz(record):
    """50 Hz for PAL regions, 60 Hz for NTSC, from the catalogue's file path."""
    return 60.0 if "america" in str(record.get("file", "")) else 50.0


def hum_excess_db(signal_data, rate, mains_hz):
    """How far the mains harmonics stand above their own spectral neighbourhood, in dB.

    A ratio rather than an absolute level, so it reads the same on a quiet tape and a loud
    one: each harmonic's peak against the median of its neighbourhood, summed as energy
    across harmonics. Clean audio sits near 0 dB; a humming tape reads well above it.
    """
    if len(signal_data) < 16384:
        return 0.0
    freqs, psd = scipy.signal.welch(signal_data, rate, nperseg=16384)
    peaks, floors = 0.0, 0.0
    for k in range(1, HARMONICS + 1):
        target = mains_hz * k
        if target >= freqs[-1]:
            break
        index = int(np.argmin(np.abs(freqs - target)))
        width = 30
        left = psd[max(index - width, 0) : max(index - 2, 0)]
        right = psd[index + 3 : index + 3 + width]
        floors += float(np.median(np.concatenate((left, right)))) + 1e-20
        peaks += float(np.max(psd[max(index - 1, 0) : index + 2])) + 1e-20
    return 10.0 * np.log10(peaks / floors)


LOW_FRAME = 8192


def _speech_band_mask(rate, mains_hz):
    """Bins inside 50-400 Hz that are not a mains harmonic.

    Removing hum necessarily drops energy in the band it occupies, so a plain band-energy
    comparison scored a perfect dehum identically to one that had cut the speech out with
    it. Leaving the harmonic bins out of the measure is what separates the two: only energy
    that was never hum can move this figure.
    """
    freqs = np.fft.rfftfreq(LOW_FRAME, 1.0 / rate)
    mask = (freqs >= HUM_BAND_HZ[0]) & (freqs <= HUM_BAND_HZ[1])
    guard = 2 * (rate / LOW_FRAME)
    for k in range(1, HARMONICS + 1):
        mask &= np.abs(freqs - mains_hz * k) > guard
    return mask


def _low_band_deviation_db(source, restored, rate, mains_hz):
    """Movement of the non-hum speech energy in 50-400 Hz across the loud frames."""
    count = min(len(source), len(restored)) // LOW_FRAME
    if count < 4:
        return None
    src = source[: count * LOW_FRAME].reshape(count, LOW_FRAME)
    rst = restored[: count * LOW_FRAME].reshape(count, LOW_FRAME)
    level = np.sqrt(np.mean(src**2, axis=1))
    loud = level >= np.percentile(level, LOUD_PERCENTILE)
    if not loud.any():
        return None
    mask = _speech_band_mask(rate, mains_hz)
    window = np.hanning(LOW_FRAME)
    src_energy = float(np.sum(np.abs(np.fft.rfft(src[loud] * window, axis=1))[:, mask] ** 2)) + 1e-20
    rst_energy = float(np.sum(np.abs(np.fft.rfft(rst[loud] * window, axis=1))[:, mask] ** 2)) + 1e-20
    return abs(10.0 * np.log10(rst_energy / src_energy))


def measure(source_path, restored_path, mains_hz):
    """Returns hum removed and non-hum low-band deviation, both in dB."""
    source, rate = _mono(source_path)
    restored, _rate = _mono(restored_path)
    source, restored, _lag = _align(source, restored)
    source_frames, restored_frames = _frames(source), _frames(restored)
    if source_frames is None or restored_frames is None or len(source_frames) < 8:
        return None
    count = min(len(source_frames), len(restored_frames))
    level = np.sqrt(np.mean(source_frames[:count] ** 2, axis=1))
    loud = level >= np.percentile(level, LOUD_PERCENTILE)
    if not loud.any():
        return None
    gain = (float(np.sqrt(np.mean(source_frames[:count][loud] ** 2))) + 1e-12) / (
        float(np.sqrt(np.mean(restored_frames[:count][loud] ** 2))) + 1e-12
    )
    restored = restored * gain

    excess_before = hum_excess_db(source, rate, mains_hz)
    excess_after = hum_excess_db(restored, rate, mains_hz)
    deviation = _low_band_deviation_db(source, restored, rate, mains_hz)
    if deviation is None:
        return None
    return {
        "hum_excess_source_db": round(float(excess_before), 3),
        "hum_removed_db": round(float(excess_before - excess_after), 3),
        "low_band_deviation_db": round(float(deviation), 3),
    }


def _restored_path(work_root, clip, mode):
    """Locates the restored container a previous benchmark run left behind."""
    candidates = list((work_root / clip.stem / mode).glob("*_Cleaned.*")) if (work_root / clip.stem / mode).is_dir() else []
    return candidates[0] if candidates else None


def _extracted(video_path, target):
    """The extracted WAV, or None when ffmpeg fails or runs out of time on this file."""
    try:
        return _extract(video_path, target)
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"  extraction timed out: {video_path.name}\n")
        return None


def _measure_clip(clip, source_wav, temp_dir, record, mains, args, results):
    """Extracts one clip, reads its hum, and measures every mode's restoration of it when it carries hum.

    Returns the source's harmonic excess, or None when the clip could not be extracted. A
    clip whose extraction times out is skipped, and a mode whose restored output times out
    is skipped for that clip alone. Each restored WAV is removed as soon as it has been
    measured.
    """
    if _extracted(clip, source_wav) is None:
        return None
    source, rate = _mono(source_wav)
    excess = float(hum_excess_db(source, rate, mains))
    if excess < HUM_PRESENT_DB:
        return excess
    for mode in args.modes:
        restored = next((p for root in args.work_dirs if (p := _restored_path(root, clip, mode))), None)
        if restored is None:
            continue
        restored_wav = temp_dir / f"{clip.stem}_{mode}.wav"
        try:
            row = measure(source_wav, restored_wav, mains) if _extracted(restored, restored_wav) else None
        finally:
            restored_wav.unlink(missing_ok=True)
        if row:
            row["identifier"] = record["identifier"]
            results[mode].append(row)
    return excess


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("experiments/ia_corpus_1000"))
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument(
        "--work-dirs", nargs="+", type=Path, required=True, help="Benchmark work dirs holding restored outputs, searched in order"
    )
    parser.add_argument("--modes", nargs="+", default=["cathar", "auto_pure_linear"])
    parser.add_argument("--report", type=Path, default=Path("experiments/hum.json"))
    return parser.parse_args()


def main():
    """Measures hum removal for each mode over the clips that carry hum."""
    args = _parse_args()
    catalog_path = args.catalog or (args.corpus_dir / "catalog_1000.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    results = {mode: [] for mode in args.modes}
    sources = []

    with tempfile.TemporaryDirectory(prefix="hum_") as temp:
        temp_dir = Path(temp)
        for record in catalog:
            clip = args.corpus_dir / record["file"]
            if not clip.exists():
                continue
            mains = _mains_hz(record)
            source_wav = temp_dir / f"{clip.stem}_src.wav"
            try:
                excess = _measure_clip(clip, source_wav, temp_dir, record, mains, args, results)
            finally:
                # The extracted WAVs are 2.6 MB a clip and a corpus has thousands; each is
                # read once and goes before the next clip's arrive.
                source_wav.unlink(missing_ok=True)
            if excess is None:
                continue
            sources.append({"identifier": record["identifier"], "hum_excess_source_db": round(excess, 3), "mains_hz": mains})
            if excess >= HUM_PRESENT_DB:
                sys.stdout.write(f"  hum {excess:6.2f} dB  {record['identifier'][:48]}\n")
                sys.stdout.flush()

    carrying = [s for s in sources if s["hum_excess_source_db"] >= HUM_PRESENT_DB]
    print(f"\n{len(carrying)} of {len(sources)} clips carry hum above {HUM_PRESENT_DB:.0f} dB of harmonic excess\n")
    print(f"{'mode':<20}{'hum removed dB':>16}{'low-band dev dB':>17}{'n':>5}")
    print(f"{'':<20}{'(higher better)':>16}{'(lower better)':>17}")
    for mode, rows in results.items():
        if rows:
            removed = np.median([r["hum_removed_db"] for r in rows])
            deviation = np.median([r["low_band_deviation_db"] for r in rows])
            print(f"{mode:<20}{removed:>16.2f}{deviation:>17.2f}{len(rows):>5}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"sources": sources, "results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
