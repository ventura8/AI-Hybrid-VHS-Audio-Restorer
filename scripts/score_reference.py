#!/usr/bin/env python3
"""Scores restoration modes against a known clean reference.

Every quality judgement in this project so far has used reference-free metrics on real
tapes, and they repeatedly pointed the wrong way: attenuation is a ratio against a 0.05
clamp, so removing a tone completely produces an enormous number that dominates any
average, and most clips carry no defect to remove at all. Changes that improved those
numbers did not improve the audio, and two of them had to be reverted.

With `scripts/make_reference_fixtures.py` the clean signal is known, so a restoration can
be scored on how close it lands to the truth instead.

## Why the metrics here are magnitude-domain

The first version of this script used SI-SDR and segmental SNR, and they ranked `cathar`
below the untouched degraded input. That was an artefact of the metric, not a finding.
Measured on the same fixture, after alignment and band limiting:

| mode | waveform correlation | envelope correlation |
| :--- | ---: | ---: |
| `cathar` | 0.57 | 0.95 |
| `auto_pure_linear` | 0.93 | 0.996 |

`auto_pure_linear` applies a magnitude mask to the original signal, so it preserves phase
by construction. `cathar` performs spectral subtraction and band replication, which
rebuilds fine structure and moves phase even when the result is closer to the truth by
ear. A waveform metric therefore rewards the mask-based mode for doing nothing to phase
and punishes the synthesis-based mode for working -- and since this harness exists to test
whether `auto_pure_linear` can beat `cathar`, that bias would have manufactured the answer.

The primary metrics are therefore phase-insensitive:

- **Log-spectral distance**, floored 80 dB below peak so near-silent bins cannot dominate.
  It penalises leftover defect and over-suppression alike: gouging a band to kill a tone
  moves the spectrum away from the reference just as leaving the tone does.
- **Defect-band energy error**, in dB against the reference's energy in the band the
  defect occupies. Zero is perfect, positive means defect remains, negative means the
  restoration cut below the truth.
- **Residual level in reference-silent regions**, measured where the truth says silence.

SI-SDR is still reported, labelled as a diagnostic, because it is informative when
comparing two variants of the *same* architecture -- for example two candidate
`auto_pure_linear` chains, where the phase bias applies equally to both.

The degraded input is always scored alongside the modes. A mode that cannot beat its own
input on a defect has restored nothing, which is easy to miss when only modes are compared.
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
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import scipy.signal
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.hardware import get_gpu_name
from modules.utils import FFMPEG_BIN
from scripts.ia_benchmark_common import _run_mode_restoration
from scripts.run_hardware_validation import _make_video_fixture

# The pipeline time-aligns its output against the source, so a restored track can sit a
# few hundred samples off the reference; measured lags run 209-219 samples. Sample-aligned
# metrics collapse under even a small shift, so the lag is removed before scoring.
MAX_ALIGN_LAG = 4096
STFT_FRAME = 1024

# Score broadband metrics only where the reference carries information. The clean
# references are band-limited to ~11 kHz by their 22.05 kHz origin, so above that the
# truth is digital silence and any content reads as pure error -- which drove
# log-spectral distance to a physically meaningless 77 dB before this bound existed.
SCORE_BAND_HZ = 10000.0
_BAND_ORDER = 8

# Near-silent bins otherwise dominate log-spectral distance: an unfloored log puts them at
# -180 dB, so a mode that correctly removes noise scores worse than one that leaves it.
LSD_FLOOR_DB = 80.0

# Where each injected defect lives, from scripts/audio_matrix/vhs_defects.py: hum is
# 50 Hz plus its 100 Hz harmonic, rumble 60 and 75 Hz, whistle 15,625 Hz, hiss broadband.
# Defect-band energy is measured on the full-band signal, since the whistle sits above the
# broadband scoring bound.
DEFECT_BANDS = {
    "hum_only": ((45.0, 105.0),),
    "rumble_only": ((55.0, 80.0),),
    "whistle_only": ((15500.0, 15750.0),),
    "hiss_only": ((2000.0, 10000.0),),
    "combo": ((45.0, 105.0), (15500.0, 15750.0), (2000.0, 10000.0)),
    # On quiet fixtures the programme no longer masks the hiss, so the 2-10 kHz band
    # becomes a usable read on broadband noise rather than a measure of speech energy.
    "quiet_hiss": ((2000.0, 10000.0),),
    "quiet_combo": ((45.0, 105.0), (55.0, 80.0), (2000.0, 10000.0)),
    # Realistic fixtures: real tape noise, and a hum series running to eight harmonics
    # rather than the two the synthetic injector produces, so the hum band runs to 400 Hz.
    "real_hiss_m08": ((2000.0, 10000.0),),
    "real_hiss_m14": ((2000.0, 10000.0),),
    "real_hiss_m20": ((2000.0, 10000.0),),
    "real_hum_m14": ((45.0, 405.0),),
    "real_hum_m22": ((45.0, 405.0),),
    "real_hum_loud": ((45.0, 405.0),),
    "real_combo_m12": ((45.0, 405.0), (55.0, 80.0), (2000.0, 10000.0)),
}
# Hum needs a metric of its own. A wide defect band around the harmonic series is
# dominated by speech, which sits in exactly the same range -- the band reads near 0 dB on
# a fixture that plainly carries hum -- and log-spectral distance under-weights a narrow
# tone because it is one bin among hundreds. Neither can see the thing a listener hears
# most easily. This measures the harmonics themselves, in bins narrow enough to separate
# them from the programme around them.
HUM_BASE_HZ = 50.0
HUM_HARMONICS = 8
HUM_BIN_HALF_WIDTH_HZ = 3.0
HUM_NPERSEG = 32768

METRIC_KEYS = ("lsd_db", "defect_band_db", "hum_excess_db", "residual_noise_db", "si_sdr_db")


def _to_mono(samples):
    """Downmixes to mono; the defects under test are not stereo-specific."""
    return samples.mean(axis=1) if samples.ndim > 1 else samples


def _read_mono(path):
    """Reads a WAV as float32 mono."""
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return _to_mono(samples), sample_rate


def _align(reference, estimate):
    """Removes the residual time offset the sync stage can leave behind."""
    span = min(len(reference), len(estimate), 10 * STFT_FRAME * 8)
    correlation = scipy.signal.correlate(estimate[:span], reference[:span], mode="full")
    lag = int(np.argmax(np.abs(correlation))) - (span - 1)
    lag = max(-MAX_ALIGN_LAG, min(MAX_ALIGN_LAG, lag))
    if lag > 0:
        estimate = estimate[lag:]
    elif lag < 0:
        reference = reference[-lag:]
    length = min(len(reference), len(estimate))
    return reference[:length], estimate[:length], lag


def _match_gain(reference, estimate):
    """Rescales the estimate onto the reference so every metric ignores level."""
    scale = float(np.dot(estimate, reference)) / (float(np.dot(estimate, estimate)) + 1e-12)
    return estimate * scale


def _band_limit(signal_data, sample_rate, cutoff_hz):
    """Low-passes to the band the reference can actually vouch for."""
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2.0:
        return signal_data
    sos = scipy.signal.butter(_BAND_ORDER, cutoff_hz, btype="low", fs=sample_rate, output="sos")
    return scipy.signal.sosfiltfilt(sos, signal_data).astype(np.float32)


def _si_sdr_db(reference, estimate):
    """Scale-invariant SDR. Phase-sensitive: a diagnostic, not a cross-architecture score."""
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    scale = float(np.dot(estimate, reference)) / (float(np.dot(reference, reference)) + 1e-12)
    target = scale * reference
    residual = estimate - target
    return float(10.0 * np.log10((float(np.sum(target**2)) + 1e-12) / (float(np.sum(residual**2)) + 1e-12)))


def _log_spectra(signal_data):
    """Log-magnitude STFT with a relative floor, so silence cannot dominate the distance."""
    _f, _t, spectrum = scipy.signal.stft(signal_data, nperseg=STFT_FRAME, noverlap=STFT_FRAME // 2)
    magnitude = np.abs(spectrum)
    floor = float(np.max(magnitude)) * (10.0 ** (-LSD_FLOOR_DB / 20.0))
    return 20.0 * np.log10(np.maximum(magnitude, floor + 1e-12))


def _lsd_db(reference, estimate):
    """Log-spectral distance; rises for leftover defect and for over-suppression alike."""
    ref_db = _log_spectra(reference)
    est_db = _log_spectra(estimate)
    frames = min(ref_db.shape[1], est_db.shape[1])
    return float(np.sqrt(np.mean((ref_db[:, :frames] - est_db[:, :frames]) ** 2)))


def _band_energy(signal_data, sample_rate, low_hz, high_hz):
    """Total spectral energy inside one band."""
    freqs, psd = scipy.signal.welch(signal_data, sample_rate, nperseg=8192)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sum(psd[band])) + 1e-20


def _defect_band_db(reference, estimate, sample_rate, bands):
    """Energy error in the defect's own band: 0 perfect, + leftover defect, - over-cut."""
    if not bands:
        return None
    errors = [
        10.0 * np.log10(_band_energy(estimate, sample_rate, low, high) / _band_energy(reference, sample_rate, low, high))
        for low, high in bands
    ]
    return float(np.mean(errors))


def _hum_excess_db(reference, estimate, sample_rate, base_hz=HUM_BASE_HZ):
    """Energy at the mains harmonics above the reference, in dB. 0 is perfect.

    Positive means hum remains; negative means the harmonics were cut below the truth,
    which takes programme content with them since speech fundamentals share this range.
    """
    nperseg = min(HUM_NPERSEG, len(reference))
    if nperseg < 4096:
        return None
    freqs, ref_psd = scipy.signal.welch(reference, sample_rate, nperseg=nperseg)
    _f, est_psd = scipy.signal.welch(estimate, sample_rate, nperseg=nperseg)
    ref_total, est_total = 0.0, 0.0
    for harmonic in range(1, HUM_HARMONICS + 1):
        centre = base_hz * harmonic
        band = (freqs >= centre - HUM_BIN_HALF_WIDTH_HZ) & (freqs <= centre + HUM_BIN_HALF_WIDTH_HZ)
        if not np.any(band):
            continue
        ref_total += float(np.sum(ref_psd[band]))
        est_total += float(np.sum(est_psd[band]))
    if ref_total <= 0.0:
        return None
    return float(10.0 * np.log10((est_total + 1e-20) / (ref_total + 1e-20)))


def _residual_noise_db(reference, estimate, frame=1024):
    """Level left where the reference is silent -- what should have been removed."""
    count = min(len(reference), len(estimate)) // frame
    if count == 0:
        return 0.0
    ref = reference[: count * frame].reshape(count, frame)
    est = estimate[: count * frame].reshape(count, frame)
    ref_rms = np.sqrt(np.mean(ref**2, axis=1))
    quiet = ref_rms <= max(float(np.percentile(ref_rms, 10)), 1e-6)
    if not np.any(quiet):
        return 0.0
    return float(20.0 * np.log10(float(np.sqrt(np.mean(est[quiet] ** 2))) + 1e-9))


def score_pair(clean_path, candidate_path, variant=None, band_hz=SCORE_BAND_HZ):
    """Returns every full-reference metric for one restored candidate."""
    reference, sample_rate = _read_mono(clean_path)
    estimate, _rate = _read_mono(candidate_path)
    reference, estimate, lag = _align(reference, estimate)
    if len(reference) < STFT_FRAME:
        return None
    estimate = _match_gain(reference, estimate)
    # Defect-band energy uses the full-band signals: the whistle sits above the broadband
    # scoring bound and would otherwise be filtered away before it could be measured.
    band_error = _defect_band_db(reference, estimate, sample_rate, DEFECT_BANDS.get(variant, ()))
    limited_reference = _band_limit(reference, sample_rate, band_hz)
    limited_estimate = _band_limit(estimate, sample_rate, band_hz)
    hum_excess = _hum_excess_db(reference, estimate, sample_rate)
    return {
        "lsd_db": round(_lsd_db(limited_reference, limited_estimate), 3),
        "defect_band_db": None if band_error is None else round(band_error, 3),
        "hum_excess_db": None if hum_excess is None else round(hum_excess, 3),
        "residual_noise_db": round(_residual_noise_db(limited_reference, limited_estimate), 3),
        "si_sdr_db": round(_si_sdr_db(limited_reference, limited_estimate), 3),
        "align_lag_samples": lag,
    }


def _extract_audio(video_path, wav_path):
    """Pulls the restored track back out of the muxed container."""
    command = [FFMPEG_BIN, "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_f32le", "-ar", "44100", str(wav_path)]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=300)
    return wav_path if wav_path.exists() else None


def _restore(degraded_wav, mode, work_dir, gpu_name):
    """Runs one mode over one fixture and returns the restored audio path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    video_path = work_dir / f"{degraded_wav.stem}.mkv"
    if not video_path.exists():
        try:
            _make_video_fixture(degraded_wav, video_path)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"    could not mux {degraded_wav.name}: {exc}")
            return None
    restored_video = _run_mode_restoration(video_path, mode, work_dir, gpu_name)
    if restored_video is None:
        return None
    return _extract_audio(restored_video, work_dir / f"{degraded_wav.stem}_{mode}.wav")


def _fixture_variant(name):
    """Returns the defect family a fixture name encodes."""
    return name.split("_", 1)[1] if "_" in name else name


def _score_fixture(record, language_dir, work_root, modes, gpu_name):
    """Scores the degraded input and every mode for one fixture."""
    clean_path = language_dir / record["clean"]
    degraded_path = language_dir / record["degraded"]
    if not clean_path.exists() or not degraded_path.exists():
        return None
    variant = _fixture_variant(record["name"])
    baseline = score_pair(clean_path, degraded_path, variant)
    if baseline is None:
        return None
    row = {"fixture": record["name"], "variant": variant, "scores": {"degraded_input": baseline}}
    for mode in modes:
        restored = _restore(degraded_path, mode, work_root / mode / record["name"], gpu_name)
        row["scores"][mode] = score_pair(clean_path, restored, variant) if restored else None
    return row


def _summarise(rows, modes):
    """Aggregates per-variant medians for the degraded input and every mode."""
    summary = {}
    for variant in sorted({r["variant"] for r in rows}):
        subset = [r for r in rows if r["variant"] == variant]
        entry = {"clips": len(subset)}
        for label in ["degraded_input"] + list(modes):
            scored = [r["scores"][label] for r in subset if r["scores"].get(label)]
            if not scored:
                continue
            entry[label] = {}
            for metric in METRIC_KEYS:
                values = [s[metric] for s in scored if s.get(metric) is not None]
                entry[label][metric] = round(statistics.median(values), 3) if values else None
        summary[variant] = entry
    return summary


def _format_metric(value):
    """Formats one metric cell, tolerating metrics that do not apply."""
    return f"{value:>12.2f}" if isinstance(value, (int, float)) else f"{'n/a':>12}"


def _print_summary(summary, modes):
    """Prints the per-variant comparison table."""
    for variant, entry in summary.items():
        print(f"\n=== {variant}   n={entry['clips']} ===")
        print(f"{'candidate':<20}{'LSD dB':>11}{'defect dB':>11}{'hum dB':>9}{'resid dB':>11}{'SI-SDR*':>10}")
        for label in ["degraded_input"] + list(modes):
            values = entry.get(label)
            if not values:
                print(f"{label:<20}{'(failed)':>12}")
                continue
            print(
                f"{label:<20}"
                + "".join(f"{values.get(k):>11.2f}" if isinstance(values.get(k), (int, float)) else f"{'n/a':>11}" for k in METRIC_KEYS)
            )
    print("\n  LSD and defect-band energy are the decision metrics (lower |value| is better).")
    print("  * SI-SDR is phase-sensitive and favours mask-based processing; diagnostic only.")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=Path("artifacts/reference-fixtures"))
    parser.add_argument("--modes", nargs="+", default=["cathar", "auto_pure_linear"])
    parser.add_argument("--variants", nargs="+", default=None, help="Limit to these defect families")
    parser.add_argument("--limit", type=int, default=None, help="Score at most this many fixtures per variant")
    parser.add_argument("--report", type=Path, default=Path("experiments/reference_scores.json"))
    parser.add_argument("--work-dir", type=Path, default=None, help="Keep restored audio here instead of a temp dir")
    return parser.parse_args()


def _select(records, variants, limit):
    """Filters the manifest down to the requested variants and per-variant cap."""
    chosen, seen = [], {}
    for record in records:
        variant = _fixture_variant(record["name"])
        if variants and variant not in variants:
            continue
        if limit and seen.get(variant, 0) >= limit:
            continue
        seen[variant] = seen.get(variant, 0) + 1
        chosen.append(record)
    return chosen


def main():
    """Scores every mode against the paired reference corpus."""
    args = _parse_args()
    manifest_path = args.fixtures_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"No manifest at {manifest_path}; run scripts/make_reference_fixtures.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gpu_name = get_gpu_name()

    temp_dir = None if args.work_dir else tempfile.TemporaryDirectory(prefix="score_ref_")
    work_root = args.work_dir if args.work_dir else Path(temp_dir.name)
    rows = []
    try:
        for language, records in manifest["languages"].items():
            language_dir = args.fixtures_dir / language
            selected = _select(records, args.variants, args.limit)
            for index, record in enumerate(selected, start=1):
                row = _score_fixture(record, language_dir, work_root / language, args.modes, gpu_name)
                if row:
                    rows.append(row)
                sys.stdout.write(f"  [{language} {index}/{len(selected)}] {record['name']}\n")
                sys.stdout.flush()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    if not rows:
        raise SystemExit("No fixture could be scored.")
    summary = _summarise(rows, args.modes)
    _print_summary(summary, args.modes)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "fixtures": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
