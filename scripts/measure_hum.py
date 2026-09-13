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

from scripts.measure_tradeoff import LOUD_PERCENTILE, MIN_DYNAMIC_SPREAD_DB, QUIET_PERCENTILE, _extract, _frames, _mono
from scripts.score_reference import _align

HARMONICS = 8
HUM_BAND_HZ = (50.0, 400.0)
# A tape "carries hum" when its harmonic excess clears this. Set from the distribution over
# the corpus so the subset is the tapes where the defect is plainly present, not marginal.
HUM_PRESENT_DB = 6.0
MAINS_CANDIDATES_HZ = (50.0, 60.0)
# The rumble reading uses the same two-figure shape: energy taken out of the quiet frames
# below the cutoff, against movement of the loud frames in the band speech shares with it.
RUMBLE_CUTOFF_HZ = 100.0


def _mains_hz(record):
    """50 Hz for PAL regions, 60 Hz for NTSC, from the catalogue's file path."""
    return 60.0 if "america" in str(record.get("file", "")) else 50.0


def _mains_by_evidence(signal_data, rate):
    """Whichever of 50 or 60 Hz the recording's harmonics support.

    The region's nominal frequency is the wrong one on 10 of the 48 corpus tapes that carry
    hum; a stage that removes hum at the right frequency there would go unread at the
    region's. This is the rule the mode's own detector uses.
    """
    return max(MAINS_CANDIDATES_HZ, key=lambda hz: hum_excess_db(signal_data, rate, hz))


def _harmonic_peaks_and_floors(signal_data, rate, mains_hz):
    """Summed peak power at the mains harmonics and summed median power of their neighbourhoods, or None when too short."""
    if len(signal_data) < 16384:
        return None
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
    return peaks, floors


def hum_excess_db(signal_data, rate, mains_hz):
    """How far the mains harmonics stand above their own spectral neighbourhood, in dB.

    A ratio rather than an absolute level, so it reads the same on a quiet tape and a loud
    one: each harmonic's peak against the median of its neighbourhood, summed as energy
    across harmonics. Clean audio sits near 0 dB; a humming tape reads well above it.
    """
    pair = _harmonic_peaks_and_floors(signal_data, rate, mains_hz)
    if pair is None:
        return 0.0
    peaks, floors = pair
    return 10.0 * np.log10(peaks / floors)


def hum_line_level_db(signal_data, rate, mains_hz):
    """The summed peak power at the mains harmonics on its own, in dB.

    The excess reading is blind to a chain that lowers the floor around a line it left
    behind: the line then stands out more although it is no louder. Read on gain-matched
    audio, this is whether the lines themselves got quieter.
    """
    pair = _harmonic_peaks_and_floors(signal_data, rate, mains_hz)
    return None if pair is None else 10.0 * np.log10(pair[0])


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


def _rumble_db(signal_data, rate):
    """Level of the quietest fifth of frames below the rumble cutoff, in dB.

    Quiet frames are where rumble stands alone -- a motor does not stop for a pause -- so a
    drop there is the rumble going, where a drop in loud frames could be the bass of the
    programme.
    """
    frames = _frames(signal_data)
    if frames is None or len(frames) < 8:
        return None
    level = np.sqrt(np.mean(frames**2, axis=1))
    quiet = frames[level <= np.percentile(level, 100.0 - LOUD_PERCENTILE)]
    spectrum = np.abs(np.fft.rfft(quiet * np.hanning(frames.shape[1]), axis=1)) ** 2
    freqs = np.fft.rfftfreq(frames.shape[1], 1.0 / rate)
    return float(10.0 * np.log10(np.mean(spectrum[:, freqs <= RUMBLE_CUTOFF_HZ]) + 1e-20))


def _gain_matched(source, restored):
    """The restored signal scaled to the source's level on the loud frames, or None when too short."""
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
    return restored * gain


def measure(source_path, restored_path, mains_hz, band="hum"):
    """Returns the defect removed and the non-defect low-band deviation, both in dB.

    `band` selects the reading: "hum" is the harmonic excess at the mains series, "rumble"
    the level below the cutoff in the quiet frames. The deviation figure is the same for
    both: movement of the loud frames' speech energy in 50-400 Hz, harmonics excluded.
    """
    source, rate = _mono(source_path)
    restored, _rate = _mono(restored_path)
    source, restored, _lag = _align(source, restored)
    restored = _gain_matched(source, restored)
    if restored is None:
        return None
    reader = hum_excess_db if band == "hum" else lambda data, r, _hz: _rumble_db(data, r)
    before, after = reader(source, rate, mains_hz), reader(restored, rate, mains_hz)
    deviation = _low_band_deviation_db(source, restored, rate, mains_hz)
    if deviation is None or before is None or after is None:
        return None
    row = {
        "hum_excess_source_db": round(float(before), 3),
        "hum_removed_db": round(float(before - after), 3),
        "low_band_deviation_db": round(float(deviation), 3),
    }
    if band == "hum":
        levels = hum_line_level_db(source, rate, mains_hz), hum_line_level_db(restored, rate, mains_hz)
        row["hum_line_drop_db"] = None if None in levels else round(float(levels[0] - levels[1]), 3)
    return row


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


def _readable(source):
    """Whether the source has the quiet-to-loud spread the readings rest on, on the trade metric's own rule.

    The low-band deviation and the line level are read on gain-matched loud frames; on a
    saturated or constant-level track those frames are the noise, both modes read tens of
    dB of movement identically, and the figure is the metric failing rather than a
    restoration. Such a source is refused here as it is there.
    """
    frames = _frames(source)
    if frames is None or len(frames) < 8:
        return False
    level = np.sqrt(np.mean(frames**2, axis=1))
    quiet_cut, loud_cut = np.percentile(level, QUIET_PERCENTILE), np.percentile(level, LOUD_PERCENTILE)
    return 20.0 * np.log10((loud_cut + 1e-12) / (quiet_cut + 1e-12)) >= MIN_DYNAMIC_SPREAD_DB


def _measure_clip(clip, source_wav, temp_dir, record, mains, args, results):
    """Extracts one clip, reads its hum, and measures every mode's restoration of it when it carries hum.

    Returns the source's harmonic excess, or None when the clip could not be extracted. A
    clip whose extraction times out is skipped, a source without the quiet-to-loud spread
    the readings need is refused (listed under "refused"), and a mode whose restored output
    times out is skipped for that clip alone. Each restored WAV is removed as soon as it
    has been measured.
    """
    if _extracted(clip, source_wav) is None:
        return None
    source, rate = _mono(source_wav)
    if mains is None:
        mains = _mains_by_evidence(source, rate)
    excess = float(hum_excess_db(source, rate, mains))
    if excess < HUM_PRESENT_DB and args.band == "hum":
        return excess
    if not _readable(source):
        results["refused"].append(record["identifier"])
        return excess
    for mode in args.modes:
        restored = next((p for root in args.work_dirs if (p := _restored_path(root, clip, mode))), None)
        if restored is None:
            continue
        restored_wav = temp_dir / f"{clip.stem}_{mode}.wav"
        try:
            row = measure(source_wav, restored_wav, mains, args.band) if _extracted(restored, restored_wav) else None
        finally:
            restored_wav.unlink(missing_ok=True)
        if row:
            row["identifier"] = record["identifier"]
            row["mains_hz"] = mains
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
    parser.add_argument(
        "--mains",
        choices=["region", "auto"],
        default="region",
        help="Read hum at the catalogue region's frequency, or at whichever of 50/60 Hz the recording's harmonics support",
    )
    parser.add_argument(
        "--band",
        choices=["hum", "rumble"],
        default="hum",
        help="Read mains hum (harmonic excess, hum-carrying clips only) or rumble (quiet-frame level below 100 Hz, every clip)",
    )
    return parser.parse_args()


def main():
    """Measures hum removal for each mode over the clips that carry hum."""
    args = _parse_args()
    catalog_path = args.catalog or (args.corpus_dir / "catalog_1000.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    results = {mode: [] for mode in args.modes}
    results["refused"] = []
    sources = []

    with tempfile.TemporaryDirectory(prefix="hum_") as temp:
        temp_dir = Path(temp)
        for record in catalog:
            clip = args.corpus_dir / record["file"]
            if not clip.exists():
                continue
            mains = _mains_hz(record) if args.mains == "region" else None
            source_wav = temp_dir / f"{clip.stem}_src.wav"
            try:
                excess = _measure_clip(clip, source_wav, temp_dir, record, mains, args, results)
            finally:
                # The extracted WAVs are 2.6 MB a clip and a corpus has thousands; each is
                # read once and goes before the next clip's arrive.
                source_wav.unlink(missing_ok=True)
            if excess is None:
                continue
            sources.append({"identifier": record["identifier"], "hum_excess_source_db": round(excess, 3)})
            if excess >= HUM_PRESENT_DB:
                sys.stdout.write(f"  hum {excess:6.2f} dB  {record['identifier'][:48]}\n")
                sys.stdout.flush()

    carrying = [s for s in sources if s["hum_excess_source_db"] >= HUM_PRESENT_DB]
    print(f"\n{len(carrying)} of {len(sources)} clips carry hum above {HUM_PRESENT_DB:.0f} dB of harmonic excess\n")
    removed_label = "hum removed dB" if args.band == "hum" else "rumble removed dB"
    print(f"{'mode':<20}{removed_label:>18}{'low-band dev dB':>17}{'n':>5}")
    print(f"{'':<20}{'(higher better)':>16}{'(lower better)':>17}")
    refused = results.pop("refused")
    for mode, rows in results.items():
        if rows:
            removed = np.median([r["hum_removed_db"] for r in rows])
            deviation = np.median([r["low_band_deviation_db"] for r in rows])
            print(f"{mode:<20}{removed:>18.2f}{deviation:>17.2f}{len(rows):>5}")
    if refused:
        print(f"\n{len(refused)} carrying sources refused: under {MIN_DYNAMIC_SPREAD_DB:.0f} dB of quiet-to-loud spread")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"sources": sources, "results": results, "refused": refused}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
