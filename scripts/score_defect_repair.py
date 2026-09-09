#!/usr/bin/env python3
"""Scores impulsive defect repair against a clean reference, separating fix from collateral.

The spectral metrics on this branch cannot see physical tape damage. A click is a handful
of samples, a dropout a fiftieth of a second; log-spectral distance and band energy average
them away, so a stage that repairs impulsive damage and a stage that does nothing score the
same. That is why `auto_pure_linear` repairing none of it has never shown up as a cost.

The measurement mirrors the one that settled the noise trade on real tapes, and for the same
reason: a single score cannot distinguish removing the defect from removing the programme
with it. Here both halves are available exactly, because the degraded signal says precisely
where the damage is:

- **Repaired** is how far the error against the clean reference falls *inside* the damaged
  samples. Higher is better.
- **Collateral** is how far the error against the clean reference rises *outside* them.
  Lower is better, and negative means the stage improved the undamaged audio too.

Both are computed after gain matching, so the final loudness normalisation cannot flatter
or penalise either number.

Azimuth is measured separately, as an inter-channel delay rather than a sample error: the
defect is a phase skew between channels, which a mono error metric cannot express.
"""

import argparse
import json
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

from scripts.score_reference import _align, _match_gain

EPS = 1e-12
# Fraction of samples treated as damaged. The injected defects are sparse by construction:
# one impulse every half second, and dropouts a twentieth of a second every seventeen, which
# together sit well under one percent of a fixture.
DEFECT_FRACTION = 0.005
# Samples either side of a damaged one that are also treated as damaged. A declicker
# interpolates across a neighbourhood rather than replacing one sample, and scoring the
# repair of a ringing edge as collateral damage would penalise the correct behaviour.
DILATION = 32
# Samples counted as undamaged. The gap between this and the defect region is deliberate:
# it leaves the ambiguous edge out of both halves rather than assigning it to whichever
# flatters the result.
CLEAN_FRACTION = 0.95


def _read(path):
    """Reads a file as float32, preserving channels."""
    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return samples, rate


def _mono(samples):
    """Collapses to mono for the sample-error metrics."""
    return samples.mean(axis=1)


def _dilate(mask, width):
    """Widens a boolean mask by `width` samples in both directions."""
    if width <= 0:
        return mask
    widened = mask.copy()
    for shift in range(1, width + 1):
        widened[shift:] |= mask[:-shift]
        widened[:-shift] |= mask[shift:]
    return widened


def _regions(clean, degraded):
    """Returns the damaged and undamaged sample masks, from where the defect actually is.

    Selected by rank rather than by a quantile threshold. These defects are sparse enough
    that the 99.5th percentile of the error is exactly zero -- six impulses in three seconds
    is six samples in a hundred and thirty thousand -- and a `>= 0` comparison then marks
    every sample as damaged and leaves nothing to measure collateral against.
    """
    error = np.abs(degraded - clean)
    count = min(max(1, int(len(error) * DEFECT_FRACTION)), len(error) - 1)
    # A floor relative to the programme, so float round-trip noise is not mistaken for
    # damage on a fixture whose defect is not impulsive at all.
    floor = 1e-5 * (_rms(clean) + EPS)
    ranked = np.zeros(len(error), dtype=bool)
    ranked[np.argpartition(error, -count)[-count:]] = True
    damaged = _dilate(ranked & (error > floor), DILATION)
    undamaged = (error <= np.quantile(error, CLEAN_FRACTION)) & ~damaged
    return damaged, undamaged


def _rms(values):
    """Root mean square, guarded against an empty selection."""
    return float(np.sqrt(np.mean(values**2))) + EPS if values.size else EPS


def _spans(mask, minimum=64):
    """Returns the contiguous (start, stop) runs of a boolean mask."""
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    bounds = np.concatenate(([0], edges + 1, [len(mask)]))
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if mask[a] and b - a >= minimum]


def _spectral_distance_db(reference, estimate, spans):
    """Mean log-spectral distance over the given spans.

    A sample-difference metric is the wrong tool for a stage that *reconstructs* audio
    rather than subtracting from it. Autoregressive gap filling restores the right signal
    but not the same samples, and filling a gap with uncorrelated audio of the correct
    energy scores about 3 dB worse than leaving the gap silent -- so the sample metric
    ranks a good reconstruction below doing nothing. This is the same bias that made
    SI-SDR favour mask-based processing over synthesis earlier on this branch.
    """
    distances = []
    for start, stop in spans:
        window = np.hanning(stop - start)
        ref = 20.0 * np.log10(np.abs(np.fft.rfft(reference[start:stop] * window)) + EPS)
        est = 20.0 * np.log10(np.abs(np.fft.rfft(estimate[start:stop] * window)) + EPS)
        distances.append(float(np.sqrt(np.mean((ref - est) ** 2))))
    return float(np.mean(distances)) if distances else None


def _azimuth_skew_samples(samples):
    """Estimates the inter-channel delay in samples, or None for mono audio.

    FFT correlation over a bounded span: a direct correlation of a whole fixture is
    quadratic and does not finish.
    """
    if samples.shape[1] < 2:
        return None
    limit = min(len(samples), 200000)
    left = samples[:limit, 0] - samples[:limit, 0].mean()
    right = samples[:limit, 1] - samples[:limit, 1].mean()
    if not np.any(left) or not np.any(right):
        return 0
    correlation = scipy.signal.correlate(left, right, mode="full", method="fft")
    return int(np.argmax(np.abs(correlation)) - (len(left) - 1))


def score_repair(clean_path, degraded_path, restored_path):
    """Returns repair and collateral figures for one restored fixture."""
    clean_stereo, _rate = _read(clean_path)
    degraded_stereo, _r2 = _read(degraded_path)
    restored_stereo, _r3 = _read(restored_path)

    clean, degraded, restored = _mono(clean_stereo), _mono(degraded_stereo), _mono(restored_stereo)
    # The degraded signal has to follow the same trim as the reference, or the three stop
    # sharing a time base and the defect mask points at the wrong samples. _align trims the
    # reference from the front only when the lag is negative.
    clean, restored, lag = _align(clean, restored)
    if lag < 0:
        degraded = degraded[-lag:]
    length = min(len(clean), len(degraded), len(restored))
    clean, degraded, restored = clean[:length], degraded[:length], restored[:length]
    restored = _match_gain(clean, restored)

    damaged, undamaged = _regions(clean, degraded)
    if not damaged.any() or not undamaged.any():
        return None

    before_damaged, after_damaged = _rms((degraded - clean)[damaged]), _rms((restored - clean)[damaged])
    after_clean = _rms((restored - clean)[undamaged])

    # Collateral is measured against the programme, not against the error that was there
    # before. Outside a click the prior error is exactly zero, so a ratio to it explodes:
    # it scored a stage that did nothing at all as 150 dB of damage. That is the same
    # failure that made the old attenuation ratios unreadable, and it is avoided the same
    # way -- by dividing by something that cannot be zero. Read it as the level of the
    # damage the stage introduced, relative to programme; more negative is better.
    spans = _spans(damaged)
    before_spectral = _spectral_distance_db(clean, degraded, spans)
    after_spectral = _spectral_distance_db(clean, restored, spans)

    skew_clean = _azimuth_skew_samples(clean_stereo)
    skew_restored = _azimuth_skew_samples(restored_stereo)
    return {
        "repaired_db": round(20.0 * np.log10(before_damaged / after_damaged), 3),
        "spectral_repaired_db": None if before_spectral is None else round(before_spectral - after_spectral, 3),
        "collateral_db": round(20.0 * np.log10(after_clean / _rms(clean)), 3),
        "damaged_samples": int(damaged.sum()),
        "azimuth_skew_samples": None if skew_restored is None else abs(skew_restored - (skew_clean or 0)),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=Path("artifacts/reference-fixtures"))
    parser.add_argument("--restored-dir", type=Path, required=True, help="Directory of restored WAVs named <fixture>_vhs.wav")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main():
    """Scores every fixture that has a restored counterpart."""
    args = _parse_args()
    manifest = json.loads((args.fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for language, records in manifest["languages"].items():
        for record in records:
            restored = args.restored_dir / record["degraded"]
            if not restored.is_file():
                continue
            language_dir = args.fixtures_dir / language
            scored = score_repair(language_dir / record["clean"], language_dir / record["degraded"], restored)
            if scored:
                scored.update({"name": record["name"], "language": language, "defects": record["defects"]})
                rows.append(scored)

    print(f"{'fixture':<26}{'repaired dB':>13}{'collateral dB':>15}{'skew':>7}")
    for row in sorted(rows, key=lambda r: r["name"]):
        skew = "-" if row["azimuth_skew_samples"] is None else str(row["azimuth_skew_samples"])
        print(f"{row['name']:<26}{row['repaired_db']:>13.2f}{row['collateral_db']:>15.2f}{skew:>7}")
    if rows:
        repaired = np.median([r["repaired_db"] for r in rows])
        collateral = np.median([r["collateral_db"] for r in rows])
        print(f"\nmedian repaired {repaired:.2f} dB, collateral {collateral:.2f} dB")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
