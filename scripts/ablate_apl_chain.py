#!/usr/bin/env python3
"""Ablates candidate restoration stages against the paired reference corpus.

`auto_pure_linear` loses to `cathar` on broadband noise, mains hum and rumble. Rather than
assume which replacement stage closes each gap -- the assumption-first approach that cost
two reverted commits in v1.2.0 -- this runs every candidate over fixtures whose clean
signal is known and reports what each one actually buys, per defect family.

Stages are applied directly to the degraded fixture rather than through the full pipeline.
That deliberately omits pre-conditioning, sync and the final loudness normalisation, so
the numbers here are not end-to-end quality; they are a like-for-like comparison of the
stages themselves, which is what selecting between them requires. The winning chain is
then wired into the real pipeline and re-scored end to end with `scripts/score_reference.py`.

Every candidate is reused from the shipping code rather than reimplemented, so what is
measured is what would ship.
"""

import argparse
import json
import shutil
import statistics
import subprocess
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

from modules import cathar as cathar_mod
from modules import filters as filters_mod
from modules import processing as processing_mod
from modules.utils import CATHAR_BIN, FFMPEG_BIN
from scripts.ia_benchmark_common import _compute_noise_floor_db, _split_channels
from scripts.score_reference import _fixture_variant, score_pair

# The fixtures inject a 50 Hz mains tone with its 100 Hz harmonic, and rumble at 60/75 Hz.
FIXTURE_MAINS_HZ = 50.0


def _run_ffmpeg_filter(source, target, filter_expression):
    """Applies one FFmpeg filter chain, preserving 32-bit float."""
    command = [FFMPEG_BIN, "-y", "-i", str(source), "-af", filter_expression, "-c:a", "pcm_f32le", str(target)]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=600)
    return target if target.exists() else None


def _measure_noise_floor_db(path):
    """Returns the degraded input's noise floor, used to aim afftdn at the real level."""
    data, rate = sf.read(str(path), dtype="float32")
    mono, _left, _right = _split_channels(data)
    rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(data**2))) + 1e-9)
    return _compute_noise_floor_db(mono, rate, rms_db)


# Every stage tried so far applies one strength to the whole spectrum, which forces a
# compromise: strong enough to clear hiss from the top octaves is too strong for the
# 300-3400 Hz where speech lives, and gentle enough for speech leaves the hiss. That
# compromise is the fidelity-versus-removal frontier measured across this branch.
#
# This escapes it by refusing to choose globally. The noise floor is estimated per
# frequency bin from the quietest frames of the input, and each bin is then blended
# between the denoised result and the original according to how far that bin's energy
# sits above its own noise floor: bins that are mostly noise take the denoised magnitude,
# bins carrying programme keep the original. The original's phase is kept throughout,
# which is what a magnitude mask should do and what spectral subtraction does not.
BLEND_FRAME = 2048


def _blend_by_local_snr(original_wav, denoised_wav, target):
    """Takes the denoised magnitude only where a bin is dominated by its own noise floor.

    The noise estimate comes from the denoiser itself -- what it removed, |X| - |D| -- not
    from a percentile of the input. A percentile floor was tried first and failed for a
    reason worth recording: the 10th percentile is by definition exceeded by 90% of frames,
    so even a bin containing nothing but noise reads as programme, and the blend kept the
    original almost everywhere (energy-weighted weight 0.015). Deriving the noise from the
    denoiser's own decision is self-calibrating and needs no bias correction.
    """
    original, rate = sf.read(str(original_wav), dtype="float32", always_2d=True)
    denoised, _rate = sf.read(str(denoised_wav), dtype="float32", always_2d=True)
    channels = min(original.shape[1], denoised.shape[1])
    length = min(len(original), len(denoised))
    blended = np.zeros((length, channels), dtype=np.float32)

    for channel in range(channels):
        _f, _t, spec_original = scipy.signal.stft(original[:length, channel], fs=rate, nperseg=BLEND_FRAME)
        _f2, _t2, spec_denoised = scipy.signal.stft(denoised[:length, channel], fs=rate, nperseg=BLEND_FRAME)
        frames = min(spec_original.shape[1], spec_denoised.shape[1])
        magnitude_original = np.abs(spec_original[:, :frames])
        magnitude_denoised = np.abs(spec_denoised[:, :frames])

        removed = np.maximum(magnitude_original - magnitude_denoised, 0.0)
        # Signal-to-removed ratio. Small where the denoiser judged the bin to be mostly
        # noise, large where it left the bin alone.
        ratio = magnitude_denoised**2 / (removed**2 + 1e-20)
        weight = 1.0 / (1.0 + ratio)
        magnitude = weight * magnitude_denoised + (1.0 - weight) * magnitude_original

        spectrum = magnitude * np.exp(1j * np.angle(spec_original[:, :frames]))
        _t3, wave = scipy.signal.istft(spectrum, fs=rate, nperseg=BLEND_FRAME)
        usable = min(length, len(wave))
        blended[:usable, channel] = wave[:usable].astype(np.float32)

    sf.write(str(target), blended, rate, subtype="FLOAT")
    return target


def _stage_learned_blend(source, work_dir, context):
    """Blends toward the original using the weight learned from paired ground truth.

    The hand-designed weightings lost because they were guesses. This one is fitted to the
    ideal weight the clean reference implies; the single-voice fit that captured 61% of the
    oracle's headroom was superseded by the six-language weights the mode ships, and both
    chain names now run those.
    """
    from modules.blend_weights import apply_blend

    original = context.get("original")
    if original is None:
        return source
    return apply_blend(original, source, work_dir / f"learned_{Path(source).name}")


def _stage_snr_blend(source, work_dir, context):
    """Blends the chain's current result back toward the original, per bin, by local SNR."""
    original = context.get("original")
    if original is None:
        return source
    return _blend_by_local_snr(original, source, work_dir / f"blended_{Path(source).name}")


def _stage_apl_notches(source, work_dir, _context):
    """Today's auto_pure_linear tonal chain: pre-denoise harmonics plus residual cleanup."""
    strategy = {"profile": {"notch_hz": FIXTURE_MAINS_HZ, "crt_notch_hz": 15625.0}}
    stages = [
        filters_mod.build_pre_denoise_surgical_filter(strategy),
        filters_mod.build_post_denoise_cleanup_filter(strategy),
    ]
    expression = ",".join(s for s in stages if s)
    if not expression:
        return source
    return _run_ffmpeg_filter(source, work_dir / f"notched_{source.name}", expression)


def _stage_afftdn(source, work_dir, context):
    """FFmpeg spectral denoiser aimed at the measured floor, with noise tracking on."""
    noise_floor = max(-80.0, min(-20.0, context["noise_floor_db"]))
    expression = f"afftdn=nr=12:nf={noise_floor:.1f}:tn=1"
    return _run_ffmpeg_filter(source, work_dir / f"afftdn_{source.name}", expression)


def _stage_anlmdn(source, work_dir, _context):
    """FFmpeg non-local-means denoiser, a different trade-off to spectral subtraction."""
    return _run_ffmpeg_filter(source, work_dir / f"anlmdn_{source.name}", "anlmdn")


def _stage_cathar_dehum(source, work_dir, _context):
    """Cathar adaptive mains dehum across 8 harmonics."""
    return cathar_mod._cathar_dehum_step(source, work_dir, freq=FIXTURE_MAINS_HZ)


def _stage_cathar_dewind(source, work_dir, _context):
    """Cathar low-frequency rumble removal."""
    return cathar_mod._cathar_dewind_step(source, work_dir)


def _cathar_denoise_with_alpha(source, work_dir, alpha):
    """Cathar spectral subtraction at a chosen over-subtraction factor."""
    noiseprint = cathar_mod._cathar_noiseprint_step(source, work_dir)
    return cathar_mod._cathar_denoise_step(source, work_dir, alpha=alpha, noiseprint_path=noiseprint)


def _stage_cathar_denoise(source, work_dir, _context):
    """Cathar spectral subtraction at the factor `cathar` itself ships (alpha 2.5)."""
    return _cathar_denoise_with_alpha(source, work_dir, cathar_mod.CATHAR_ALPHA)


def _stage_cathar_denoise_gentle(source, work_dir, _context):
    """A gentler over-subtraction factor.

    `cathar` overshoots on every defect family measured so far -- band energy lands 3.7 to
    11.7 dB *below* the clean reference, meaning it removes real programme content along
    with the defect. A lower alpha is the obvious lever, and it is how auto_pure_linear
    could beat cathar rather than merely match it: remove the defect without cutting past
    the truth.
    """
    return _cathar_denoise_with_alpha(source, work_dir, 1.2)


def _stage_cathar_denoise_long_print(source, work_dir, _context):
    """Same gentle factor, but a noise profile learned from a longer quiet window.

    The shipped profile is 0.75 s. A longer window averages more of the noise and should
    describe it more accurately, which is the cheap way to subtract more of it without
    raising the over-subtraction factor and cutting into programme.
    """
    noiseprint = cathar_mod._cathar_noiseprint_step(source, work_dir, duration_s=2.0)
    return cathar_mod._cathar_denoise_step(source, work_dir, alpha=1.8, noiseprint_path=noiseprint)


def _stage_cathar_denoise_twice(source, work_dir, _context):
    """Two gentle passes rather than one, each re-learning the profile it has left.

    Two subtractions at 1.3 remove more in total than one at 1.8 if the second profile is
    learned from what the first left behind, and each pass individually cuts less deeply.
    """
    first_dir = work_dir / "pass1"
    first_dir.mkdir(parents=True, exist_ok=True)
    first_print = cathar_mod._cathar_noiseprint_step(source, first_dir)
    first = cathar_mod._cathar_denoise_step(source, first_dir, alpha=1.3, noiseprint_path=first_print)
    if first is None:
        return None
    second_dir = work_dir / "pass2"
    second_dir.mkdir(parents=True, exist_ok=True)
    second_print = cathar_mod._cathar_noiseprint_step(first, second_dir)
    return cathar_mod._cathar_denoise_step(first, second_dir, alpha=1.3, noiseprint_path=second_print)


def _stage_cathar_denoise_soft(source, work_dir, _context):
    """Midpoint between the shipped factor and the gentle one."""
    return _cathar_denoise_with_alpha(source, work_dir, 1.8)


def _neural_stage(source, work_dir, model):
    """Runs the UVR separator, which needs a directory of its own per invocation."""
    target_dir = work_dir / f"neural_{model.split('.')[0]}"
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        return processing_mod._denoise_full_audio_step(source, target_dir, denoise_model=model)
    except Exception as exc:  # a failed candidate must not abort the sweep
        sys.stderr.write(f"    [warn] neural stage {model} failed: {exc}\n")
        return None


# Mel-Roformer denoisers, reported at SDR 27.99 against the VR-architecture models this
# project ships. Available through audio-separator already, so evaluating them costs no new
# dependency -- unlike DeepFilterNet, whose deepfilterlib needs a Rust toolchain to build.
ROFORMER_DENOISE = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
ROFORMER_DENOISE_AGGRESSIVE = "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt"


def _stage_roformer(source, work_dir, _context):
    """Mel-Roformer denoiser, the strongest the separator offers."""
    return _neural_stage(source, work_dir, ROFORMER_DENOISE)


def _stage_roformer_aggressive(source, work_dir, _context):
    """The aggressive Mel-Roformer variant, to see where it lands on the frontier."""
    return _neural_stage(source, work_dir, ROFORMER_DENOISE_AGGRESSIVE)


def _stage_uvr_full(source, work_dir, _context):
    """The deep UVR-DeNoise model."""
    return _neural_stage(source, work_dir, "UVR-DeNoise.pth")


def _stage_uvr_lite(source, work_dir, _context):
    """The transparent UVR-DeNoise-Lite model auto_pure_linear usually selects."""
    return _neural_stage(source, work_dir, "UVR-DeNoise-Lite.pth")


STAGES = {
    "snr_blend": _stage_snr_blend,
    "learned_blend": _stage_learned_blend,
    "learned_blend_multi": _stage_learned_blend,
    "apl_notches": _stage_apl_notches,
    "afftdn": _stage_afftdn,
    "anlmdn": _stage_anlmdn,
    "cathar_dehum": _stage_cathar_dehum,
    "cathar_dewind": _stage_cathar_dewind,
    "cathar_denoise": _stage_cathar_denoise,
    "cathar_denoise_gentle": _stage_cathar_denoise_gentle,
    "cathar_denoise_soft": _stage_cathar_denoise_soft,
    "cathar_denoise_long_print": _stage_cathar_denoise_long_print,
    "cathar_denoise_twice": _stage_cathar_denoise_twice,
    "uvr_full": _stage_uvr_full,
    "roformer": _stage_roformer,
    "roformer_aggr": _stage_roformer_aggressive,
    "uvr_lite": _stage_uvr_lite,
}

# Ordered candidate chains. `apl_today` approximates the shipping tonal chain plus its
# neural stage; `cathar_core` approximates cathar's deterministic cleanup.
CHAINS = {
    "baseline": [],
    "apl_today": ["apl_notches", "uvr_lite"],
    "uvr_full": ["uvr_full"],
    "afftdn": ["afftdn"],
    "anlmdn": ["anlmdn"],
    "cathar_denoise": ["cathar_denoise"],
    "cathar_dehum": ["cathar_dehum"],
    "cathar_dewind": ["cathar_dewind"],
    "cathar_core": ["cathar_dehum", "cathar_dewind", "cathar_denoise"],
    "cathar_core_uvr": ["cathar_dehum", "cathar_dewind", "cathar_denoise", "uvr_lite"],
    "notches_afftdn": ["apl_notches", "afftdn"],
    "notches_cathar_denoise": ["apl_notches", "cathar_denoise"],
    "cathar_denoise_gentle": ["cathar_denoise_gentle"],
    "cathar_denoise_soft": ["cathar_denoise_soft"],
    "notches_gentle": ["apl_notches", "cathar_denoise_gentle"],
    "dehum_dewind_gentle": ["cathar_dehum", "cathar_dewind", "cathar_denoise_gentle"],
    # The candidate for auto_pure_linear: deterministic spectral subtraction at the
    # gentler factor, then the neural stage that measurably wins on healthy-level hiss.
    "soft_uvr_full": ["cathar_denoise_soft", "uvr_full"],
    "soft_uvr_lite": ["cathar_denoise_soft", "uvr_lite"],
    "notches_soft_uvr": ["apl_notches", "cathar_denoise_soft", "uvr_full"],
    "notches_soft": ["apl_notches", "cathar_denoise_soft"],
    "soft_learned": ["cathar_denoise_soft", "learned_blend"],
    "soft_multi_uvr": ["cathar_denoise_soft", "learned_blend_multi", "uvr_full"],
    "soft_learned_uvr": ["cathar_denoise_soft", "learned_blend", "uvr_full"],
    "soft_blend": ["cathar_denoise_soft", "snr_blend"],
    "soft_blend_uvr": ["cathar_denoise_soft", "snr_blend", "uvr_full"],
    "aggressive_blend": ["cathar_denoise", "snr_blend"],
    "aggressive_blend_uvr": ["cathar_denoise", "snr_blend", "uvr_full"],
    "roformer": ["roformer"],
    "roformer_aggr": ["roformer_aggr"],
    "soft_roformer": ["cathar_denoise_soft", "roformer"],
    "notches_soft_roformer": ["apl_notches", "cathar_denoise_soft", "roformer"],
    "long_print": ["cathar_denoise_long_print"],
    "twice": ["cathar_denoise_twice"],
    "long_print_uvr": ["cathar_denoise_long_print", "uvr_full"],
    "twice_uvr": ["cathar_denoise_twice", "uvr_full"],
    # Hum-focused orderings. The question is whether dehum underperforms because it runs
    # after the pre-conditioning notch has already flattened the fundamental its adaptive
    # tracker locks onto, or simply because one pass leaves harmonics behind.
    "dehum_only": ["cathar_dehum"],
    "dehum_then_soft": ["cathar_dehum", "cathar_denoise_soft"],
    "soft_then_dehum": ["cathar_denoise_soft", "cathar_dehum"],
    "dehum_soft_dehum": ["cathar_dehum", "cathar_denoise_soft", "cathar_dehum"],
    "notched_then_dehum": ["apl_notches", "cathar_dehum"],
}


def _run_chain(degraded, chain, work_dir, context):
    """Applies a chain in order, returning the final audio or None if a stage failed."""
    if not chain:
        return degraded
    work_dir.mkdir(parents=True, exist_ok=True)
    current = degraded
    for stage_name in chain:
        try:
            produced = STAGES[stage_name](current, work_dir, context)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"    {stage_name} failed: {exc}")
            return None
        if produced is None or not Path(produced).exists():
            return None
        current = Path(produced)
    return current


def _score_fixture(record, language_dir, work_root, chains):
    """Scores every candidate chain for one fixture."""
    clean = language_dir / record["clean"]
    degraded = language_dir / record["degraded"]
    if not clean.exists() or not degraded.exists():
        return None
    variant = _fixture_variant(record["name"])
    context = {"noise_floor_db": _measure_noise_floor_db(degraded), "original": degraded}
    row = {"fixture": record["name"], "variant": variant, "scores": {}}
    for chain_name in chains:
        work_dir = work_root / record["name"] / chain_name
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        produced = _run_chain(degraded, CHAINS[chain_name], work_dir, context)
        row["scores"][chain_name] = score_pair(clean, produced, variant) if produced else None
    return row


def _summarise(rows, chains):
    """Medians per defect family for every chain."""
    summary = {}
    for variant in sorted({r["variant"] for r in rows}):
        subset = [r for r in rows if r["variant"] == variant]
        entry = {"clips": len(subset)}
        for chain_name in chains:
            scored = [r["scores"][chain_name] for r in subset if r["scores"].get(chain_name)]
            if not scored:
                continue
            entry[chain_name] = {}
            for metric in ("lsd_db", "defect_band_db", "hum_excess_db", "residual_noise_db"):
                values = [s[metric] for s in scored if s.get(metric) is not None]
                entry[chain_name][metric] = round(statistics.median(values), 3) if values else None
        summary[variant] = entry
    return summary


def _cell(value):
    """Formats one metric cell."""
    return f"{value:>12.2f}" if isinstance(value, (int, float)) else f"{'n/a':>12}"


def _print_summary(summary, chains):
    """Prints per-variant tables, ranked by the decision metric."""
    for variant, entry in summary.items():
        print(f"\n=== {variant}   n={entry['clips']} ===")
        print(f"{'chain':<26}{'LSD dB':>11}{'defect dB':>11}{'hum dB':>9}{'resid dB':>11}")
        ranked = sorted(
            (c for c in chains if entry.get(c)),
            key=lambda c: (entry[c]["lsd_db"] if entry[c]["lsd_db"] is not None else 1e9),
        )
        for chain_name in ranked:
            values = entry[chain_name]
            marker = "  <-- baseline" if chain_name == "baseline" else ""
            cells = "".join(
                f"{values.get(k):>11.2f}" if isinstance(values.get(k), (int, float)) else f"{'n/a':>11}"
                for k in ("lsd_db", "defect_band_db", "hum_excess_db", "residual_noise_db")
            )
            print(f"{chain_name:<26}" + cells + marker)
    print("\n  Ranked by log-spectral distance (lower is better).")
    print("  defect dB: 0 is perfect, positive leaves the defect, negative cuts below the truth.")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=Path("artifacts/reference-fixtures"))
    parser.add_argument("--variants", nargs="+", default=["quiet_hiss", "quiet_combo", "hum_only", "rumble_only", "hiss_only"])
    parser.add_argument("--chains", nargs="+", default=list(CHAINS), choices=list(CHAINS))
    parser.add_argument("--limit", type=int, default=2, help="Fixtures per defect family")
    parser.add_argument("--work-dir", type=Path, default=Path("experiments/ablation_work"))
    parser.add_argument("--report", type=Path, default=Path("experiments/apl_ablation.json"))
    return parser.parse_args()


def main():
    """Runs the ablation sweep and writes its report."""
    args = _parse_args()
    if CATHAR_BIN is None:
        sys.stderr.write("[warn] Cathar binary not found; its candidate stages will be skipped.\n")
    manifest = json.loads((args.fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for language, records in manifest["languages"].items():
        language_dir = args.fixtures_dir / language
        seen = {}
        for record in records:
            variant = _fixture_variant(record["name"])
            if variant not in args.variants or seen.get(variant, 0) >= args.limit:
                continue
            seen[variant] = seen.get(variant, 0) + 1
            row = _score_fixture(record, language_dir, args.work_dir / language, args.chains)
            if row:
                rows.append(row)
            sys.stdout.write(f"  scored {record['name']}\n")
            sys.stdout.flush()

    if not rows:
        raise SystemExit("No fixture was scored.")
    summary = _summarise(rows, args.chains)
    _print_summary(summary, args.chains)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "fixtures": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
