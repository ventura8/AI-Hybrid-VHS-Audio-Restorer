#!/usr/bin/env python3
"""Builds a supervised dataset for learning the per-bin blend between original and denoised.

Spectral subtraction removes noise and takes some programme with it. Blending its output
back toward the original per frequency bin can recover that content, and an oracle that
picks the ideal weight from the clean reference is worth 1.7-1.9 dB of log-spectral
distance over the current chain -- several times any parameter tuning tried on this branch.

A hand-designed weighting was tried first and lost, twice: a percentile noise floor is
exceeded by 90% of frames so everything reads as programme, and deriving the noise from the
denoiser's own decision restores noise along with the content. The oracle shows the
structure is right and only the weighting function was wrong, which is a supervised
learning problem: the ideal weight is computable from the clean reference, and every
feature it depends on is observable at inference.

This script emits (features, ideal weight, magnitudes) per time-frequency bin so a small
model can learn that function.

The subtraction it runs is the shipping chain's own -- `modules.spectral_denoise._subtract`,
with the tonality-gated factor and the probe length that mode resolves -- because the
first weights were fitted on a subtraction at factor 1.8 with a 0.75 s probe and then
shipped behind one at 3.0 with 2.5 s. The features describe how much each bin lost, and
a model that learned what a gentle subtraction's losses look like was being asked about a
strong one's. Fixtures the margin gate would skip are skipped here too, since the blend
never sees them.
"""

import argparse
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import scipy.signal
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import spectral_denoise
from modules.blend_weights import FEATURE_NAMES, FRAME, feature_stack

EPS = 1e-10


def _stft(samples, rate):
    """Returns the complex STFT used throughout the blend."""
    _f, _t, spectrum = scipy.signal.stft(samples, fs=rate, nperseg=FRAME)
    return spectrum


def _denoise(source_wav, work_dir):
    """Runs the shipping chain's subtraction, gate included, so features match inference."""
    if spectral_denoise.should_apply(source_wav) is None:
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    return spectral_denoise._subtract(source_wav, work_dir, total_duration=None)


def _ideal_weight(magnitude_original, magnitude_denoised, magnitude_clean):
    """The weight that would place the blend exactly on the clean magnitude."""
    spread = magnitude_original - magnitude_denoised
    safe = np.where(np.abs(spread) < 1e-12, 1.0, spread)
    weight = np.where(np.abs(spread) < 1e-12, 0.0, (magnitude_original - magnitude_clean) / safe)
    return np.clip(weight, 0.0, 1.0)


def _pair_arrays(clean_path, degraded_path, denoised_path):
    """Returns (features, ideal weight, |X|, |D|, |C|) flattened over every bin, or None.

    The three files are one fixture before damage, after it, and after subtraction, so
    they have to agree on rate and channel count; a fixture whose files do not is skipped
    rather than paired bin against bin at different rates. The first channel is the one
    trained on: the stereo classes carry the same programme on both, skewed or lowered on
    the right, and the blend runs per channel at inference.
    """
    clean, rate = sf.read(str(clean_path), dtype="float32", always_2d=True)
    original, rate_original = sf.read(str(degraded_path), dtype="float32", always_2d=True)
    denoised, rate_denoised = sf.read(str(denoised_path), dtype="float32", always_2d=True)
    if not rate == rate_original == rate_denoised or not clean.shape[1] == original.shape[1] == denoised.shape[1]:
        sys.stderr.write(f"  skipping {degraded_path.name}: rate or channel count differs between its files\n")
        return None
    length = min(len(clean), len(original), len(denoised))

    spec_clean = _stft(clean[:length, 0], rate)
    spec_original = _stft(original[:length, 0], rate)
    spec_denoised = _stft(denoised[:length, 0], rate)
    frames = min(spec_clean.shape[1], spec_original.shape[1], spec_denoised.shape[1])

    magnitude_clean = np.abs(spec_clean[:, :frames])
    magnitude_original = np.abs(spec_original[:, :frames])
    magnitude_denoised = np.abs(spec_denoised[:, :frames])

    features = feature_stack(magnitude_original, magnitude_denoised)
    weight = _ideal_weight(magnitude_original, magnitude_denoised, magnitude_clean)
    return (
        features.reshape(-1, features.shape[-1]),
        weight.reshape(-1),
        magnitude_original.reshape(-1),
        magnitude_denoised.reshape(-1),
        magnitude_clean.reshape(-1),
    )


# Faults that move the audio in time. The ideal weight is computed bin against bin, and a
# reference that no longer lines up with its degraded signal has no ideal weight.
TIMING_DEFECTS = ("flutter", "scrapeflutter", "drift")


# Fixture names are "<recording><segment index>_<variant>", and both the recording name and
# the variant contain underscores, so splitting on one is wrong: it grouped every real clip
# beginning with the same first word into a single "recording".
KNOWN_VARIANTS = (
    "real_hiss_m08",
    "real_hiss_m14",
    "real_hiss_m20",
    "real_hum_m14",
    "real_hum_m22",
    "real_hum_loud",
    "real_combo_m12",
    "quiet_combo",
    "quiet_hiss",
    "hiss_only",
    "hum_only",
    "rumble_only",
    "whistle_only",
    "combo",
)


def _recording_of(fixture_name):
    """Strips the defect variant and segment index, leaving the source recording.

    Two naming schemes exist. The reference sets end in one of the variants above; the
    calibrated sets read "<language>_<voice><segment>_<class>_m<margin>[_hum]", where the
    class and margin carry underscores and digits of their own, so the recording is what
    precedes the two-digit segment index. Either way, every fixture cut from one voice
    lands on one side of the split.
    """
    stem = str(fixture_name)
    for variant in KNOWN_VARIANTS:
        suffix = f"_{variant}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)].rstrip("0123456789")
    match = re.match(r"^(.*?[^0-9])[0-9]{2}(?:_|$)", stem)
    return match.group(1) if match else stem.rstrip("0123456789")


def _moves_in_time(record):
    """Whether a fixture carries a fault that shifts its audio against the reference."""
    return any(defect in TIMING_DEFECTS for defect in record.get("defects", ()))


def _subsample(parts, stride, seed=0):
    """Thins each fixture's bins; a small model needs variety, not every frame."""
    if stride <= 1:
        return parts
    count = len(parts[1])
    rng = np.random.default_rng(seed)
    keep = rng.choice(count, size=max(1, count // stride), replace=False)
    return tuple(part[keep] for part in parts)


def _collect(records, fixtures_dir, work_root, limit, stride=1):
    """Denoises each fixture and accumulates its bins."""
    features, weights, originals, denoiseds, cleans, sources = [], [], [], [], [], []
    names = []
    for index, record in enumerate(records[:limit] if limit else records):
        clean_path = fixtures_dir / record["clean"]
        degraded_path = fixtures_dir / record["degraded"]
        if not clean_path.exists() or not degraded_path.exists() or _moves_in_time(record):
            continue
        denoised = _denoise(degraded_path, work_root / record["name"])
        if denoised is None or not Path(denoised).exists():
            continue
        paired = _pair_arrays(clean_path, degraded_path, Path(denoised))
        if paired is None:
            continue
        parts = _subsample(paired, stride, seed=index)
        features.append(parts[0])
        weights.append(parts[1])
        originals.append(parts[2])
        denoiseds.append(parts[3])
        cleans.append(parts[4])
        # The source id is the fixture's position among the fixtures that produced data, which
        # is what indexes `names` on the training side; the manifest index would drift past
        # every record skipped above.
        sources.append(np.full(len(parts[1]), len(names), dtype=np.int32))
        names.append(record["name"])
        sys.stdout.write(f"  [{index + 1}] {record['name']}: {len(parts[1]):,} bins\n")
        sys.stdout.flush()
    if not features:
        raise SystemExit("No fixture produced usable data.")
    return (
        np.concatenate(features).astype(np.float32),
        np.concatenate(weights).astype(np.float32),
        np.concatenate(originals).astype(np.float32),
        np.concatenate(denoiseds).astype(np.float32),
        np.concatenate(cleans).astype(np.float32),
        np.concatenate(sources),
        names,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=Path("artifacts/realistic-fixtures"))
    parser.add_argument("--output", type=Path, default=Path("experiments/blend_dataset.npz"))
    parser.add_argument("--work-dir", type=Path, default=Path("experiments/blend_dataset_work"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stride", type=int, default=8, help="Keep 1 bin in N")
    parser.add_argument(
        "--reference",
        choices=("clean", "target"),
        default="clean",
        help="Fit against the clean reference, or against the target a set provides (what a restoration should give back)",
    )
    return parser.parse_args()


def main():
    """Writes the training dataset."""
    args = _parse_args()
    manifest = json.loads((args.fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    all_records = []
    for language, records in manifest["languages"].items():
        for record in records:
            record = dict(record)
            # Qualify by language: every language writes the same "mid_clean.wav", so the
            # unqualified names collide and a split by recording sees one recording for the
            # whole corpus -- which emptied the training side entirely.
            record["name"] = f"{language}_{record['name']}"
            if args.reference not in record:
                raise SystemExit(f"{record['name']} has no '{args.reference}' file; this set cannot be fitted against that reference")
            record["clean"] = f"{language}/{record[args.reference]}"
            record["degraded"] = f"{language}/{record['degraded']}"
            all_records.append(record)

    features, weights, originals, denoiseds, cleans, sources, names = _collect(
        all_records, args.fixtures_dir, args.work_dir, args.limit, args.stride
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        weights=weights,
        magnitude_original=originals,
        magnitude_denoised=denoiseds,
        magnitude_clean=cleans,
        source=sources,
        fixture_names=np.array(names),
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"\n{len(weights):,} bins from {len(set(sources.tolist()))} fixtures -> {args.output}")
    print(
        f"ideal weight: mean {weights.mean():.3f}  median {np.median(weights):.3f}  "
        f"fraction at 0 {np.mean(weights <= 0.001):.3f}  at 1 {np.mean(weights >= 0.999):.3f}"
    )


if __name__ == "__main__":
    main()
