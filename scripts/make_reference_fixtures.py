#!/usr/bin/env python3
"""Builds paired clean/degraded fixtures with one defect isolated per variant.

Restoration quality has so far been judged with reference-free metrics on real tapes,
which proved unreliable: CRT and mains attenuation are ratios of the form
`original / max(restored, 0.05)`, so a fully removed tone saturates the clamp and a
handful of clips dominate any average, and most clips in the real corpus carry no defect
at all. Tuning against those numbers rewarded changes that moved the metric without
improving the audio.

A paired reference removes that whole class of error: the clean signal is known, so a
restoration can be scored on how close it lands to the truth rather than on how far it
moved a ratio.

`scripts/audio_matrix/cli.py` already produces such pairs, but it needs Piper to
synthesise speech, and the fixtures currently on disk were written by an older revision
at 22.05 kHz -- a rate at which the 15,625 Hz line whistle cannot exist. This script
derives fresh pairs from the clean references already present: it resamples them to the
44.1 kHz the pipeline runs at (reusing `_resample_fixture`), then injects one defect
family at a time so each restoration stage can be scored against the defect it targets.

Limitation worth stating: the clean source is band-limited to ~11 kHz by its original
22.05 kHz rate, so there is no programme content near the whistle and little above it.
Absolute scores are therefore optimistic in the top octave. Both modes face the identical
signal, so mode-to-mode comparison stays fair, which is what this corpus is for.
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
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audio_matrix.cli import FIXTURE_SAMPLE_RATE, _resample_fixture
from scripts.audio_matrix.vhs_defects import apply_vhs_defects
from scripts.realistic_defects import build_noise_bank, inject_mains_hum, inject_tape_noise

# One defect family per variant, so a stage that targets hum can be scored on hum alone
# rather than through the confound of four other defects. `combo` is the realistic case.
#
# The `quiet_*` variants exist because the dominant real-world gap only appears in that
# regime. On the 174-clip corpus `auto_pure_linear` left the noise floor no better than it
# found it on 82 clips, and the worst cases were all captures around -45 to -60 dBFS,
# where UVR-DeNoise removes almost nothing and the final loudness normalisation then
# amplifies the untouched hiss. Fixtures recorded at a healthy level cannot reproduce that,
# so a harness built only from them would tune the wrong thing.
#
# -21 dB is derived, not picked: the injected hiss sits at 20*log10(0.005) = -46 dBFS, and
# the real failing captures had roughly 8 dB between programme and noise floor (one worst
# case measured -60.6 dBFS RMS against a -68.8 dBFS floor). Putting the programme at about
# -38 dBFS reproduces that margin. A deeper cut was tried first and produced -5.7 dB SNR,
# noise louder than programme, which is harsher than anything in the real corpus.
DEFECT_VARIANTS = {
    "hiss_only": {"defects": ["hiss"], "gain_db": 0.0},
    "hum_only": {"defects": ["hum"], "gain_db": 0.0},
    "rumble_only": {"defects": ["rumble"], "gain_db": 0.0},
    "whistle_only": {"defects": ["whistle"], "gain_db": 0.0},
    "combo": {"defects": ["hiss", "hum", "rumble", "whistle", "azimuth"], "gain_db": 0.0},
    "quiet_hiss": {"defects": ["hiss"], "gain_db": -21.0},
    "quiet_combo": {"defects": ["hiss", "hum", "rumble"], "gain_db": -21.0},
    # Physical tape damage. auto_pure_linear repairs none of this today, and none of it was
    # measurable before these fixtures existed: every earlier variant carried only additive
    # noise and tones, so a stage that repairs impulsive damage and a stage that does
    # nothing scored identically.
    "clicks_only": {"defects": ["clicks"], "gain_db": 0.0},
    "dropout_only": {"defects": ["dropout"], "gain_db": 0.0},
    "clip_only": {"defects": ["clip"], "gain_db": 0.0},
    "azimuth_only": {"defects": ["azimuth"], "gain_db": 0.0},
    "physical_combo": {"defects": ["clicks", "dropout", "clip"], "gain_db": 0.0},
    # Damage on top of the noise the mode already handles, which is the realistic case and
    # the one where a repair stage can undo the denoising that precedes it.
    "damaged_tape": {"defects": ["hiss", "hum", "clicks", "dropout"], "gain_db": 0.0},
}

# Fixtures built from real tape noise and a drifting harmonic hum series, at margins that
# straddle the band where measurement on 174 real clips showed the mode failing. The
# synthetic variants above sit at 14-29 dB margin and never exercised 12-24 dB at all.
REALISTIC_VARIANTS = {
    "real_hiss_m08": {"margin_db": 8.0, "hum": False},
    "real_hiss_m14": {"margin_db": 14.0, "hum": False},
    "real_hiss_m20": {"margin_db": 20.0, "hum": False},
    "real_hum_m14": {"margin_db": 14.0, "hum": True},
    "real_hum_m22": {"margin_db": 22.0, "hum": True},
    # A tape whose hum genuinely dominates. At the default level the harmonics sit only
    # ~1 dB above the speech that shares 50-400 Hz, which is too weak to tell a stage that
    # removes hum from one that removes the programme with it.
    "real_hum_loud": {"margin_db": 18.0, "hum": True, "hum_level": 0.12},
    "real_combo_m12": {"margin_db": 12.0, "hum": True, "rumble": True},
}

# Only `drift` is excluded, and only because it genuinely warps the time base: it resamples
# through np.interp, so a sample-aligned comparison against the clean reference stops being
# meaningful. Robustness against it stays with the hardware-validation fixtures.
#
# `dropout` used to be excluded alongside it on the assumption that it moves samples too.
# It does not -- `apply_vhs_defects` zeroes a span in place every two seconds and leaves
# the time base intact, so it is sample-aligned and scoreable like any other additive
# defect. Excluding it meant
# the one defect class that needs inpainting could not be measured at all.
EXCLUDED_DEFECTS = ("drift",)


def _load_clean(path):
    """Reads a clean reference and resamples it to the rate the pipeline runs at."""
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return _resample_fixture(samples, sample_rate)


def _segments(samples, segment_frames, max_segments):
    """Splits a reference into independent segments to widen a thin fixture set."""
    if segment_frames <= 0 or len(samples) <= segment_frames:
        return [samples]
    starts = list(range(0, len(samples) - segment_frames + 1, segment_frames))[:max_segments]
    # Named bound: black insists on spaces around a computed slice colon and flake8's E203
    # rejects them, so the arithmetic moves out of the subscript.
    segments = []
    for start in starts:
        stop = start + segment_frames
        segments.append(samples[start:stop])
    return segments


def _write_pair(out_dir, name, clean, defects, gain_db=0.0):
    """Writes one clean/degraded pair and returns its sidecar record.

    `gain_db` attenuates the programme before the defects go in. The injected amplitudes
    are absolute constants, so a quieter programme yields a genuinely worse signal-to-noise
    ratio rather than a quieter copy of the same problem -- which is the regime real
    captures fail in.
    """
    scaled = (clean * (10.0 ** (gain_db / 20.0))).astype(np.float32) if gain_db else clean
    clean_path = out_dir / f"{name}_clean.wav"
    degraded_path = out_dir / f"{name}_vhs.wav"
    # FLOAT throughout: soundfile defaults .wav to PCM_16, and quantising a reference
    # would put a noise floor into the very thing the scores are measured against.
    sf.write(str(clean_path), scaled, FIXTURE_SAMPLE_RATE, subtype="FLOAT")
    sf.write(str(degraded_path), apply_vhs_defects(scaled, FIXTURE_SAMPLE_RATE, defects), FIXTURE_SAMPLE_RATE, subtype="FLOAT")
    return {
        "name": name,
        "defects": list(defects),
        "gain_db": gain_db,
        "sample_rate": FIXTURE_SAMPLE_RATE,
        "frames": int(len(scaled)),
        "clean": clean_path.name,
        "degraded": degraded_path.name,
    }


def _build_language(clean_paths, out_dir, segment_frames, max_segments):
    """Generates every defect variant for one language directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for clean_path in clean_paths:
        samples = _load_clean(clean_path)
        for index, segment in enumerate(_segments(samples, segment_frames, max_segments)):
            for variant, spec in DEFECT_VARIANTS.items():
                name = f"{clean_path.stem.replace('_clean', '')}{index:02d}_{variant}"
                records.append(_write_pair(out_dir, name, segment, spec["defects"], spec["gain_db"]))
    return records


def _realistic_degraded(clean_mono, spec, noise, rng):
    """Applies real tape noise, a drifting hum series, and optional rumble."""
    degraded = clean_mono
    if spec.get("hum"):
        degraded = inject_mains_hum(degraded, FIXTURE_SAMPLE_RATE, level=spec.get("hum_level", 0.02), rng=rng)
    if spec.get("rumble"):
        degraded = apply_vhs_defects(degraded, FIXTURE_SAMPLE_RATE, ["rumble"]).mean(axis=1)
    return inject_tape_noise(degraded, noise, spec["margin_db"], rng=rng)


def _write_realistic_pair(out_dir, name, clean, spec, noise, rng):
    """Writes one realistic clean/degraded pair and returns its manifest record."""
    mono = clean.mean(axis=1) if clean.ndim > 1 else clean
    degraded = _realistic_degraded(mono, spec, noise, rng)
    clean_path = out_dir / f"{name}_clean.wav"
    degraded_path = out_dir / f"{name}_vhs.wav"
    sf.write(str(clean_path), mono, FIXTURE_SAMPLE_RATE, subtype="FLOAT")
    sf.write(str(degraded_path), degraded, FIXTURE_SAMPLE_RATE, subtype="FLOAT")
    return {
        "name": name,
        "defects": ["tape_noise"] + (["hum"] if spec.get("hum") else []) + (["rumble"] if spec.get("rumble") else []),
        "margin_db": spec["margin_db"],
        "sample_rate": FIXTURE_SAMPLE_RATE,
        "frames": int(len(mono)),
        "clean": clean_path.name,
        "degraded": degraded_path.name,
    }


def _build_realistic(clean_paths, out_dir, segment_frames, max_segments, noise_bank, seed=4242):
    """Generates every realistic variant for one language directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    for clean_path in clean_paths:
        samples = _load_clean(clean_path)
        for index, segment in enumerate(_segments(samples, segment_frames, max_segments)):
            for variant, spec in REALISTIC_VARIANTS.items():
                noise = noise_bank[(index + len(records)) % len(noise_bank)]
                name = f"{clean_path.stem.replace('_clean', '')}{index:02d}_{variant}"
                records.append(_write_realistic_pair(out_dir, name, segment, spec, noise, rng))
    return records


def _discover_languages(fixtures_dir):
    """Returns {language: [clean reference paths]} from a generated fixture tree."""
    languages = {}
    for language_dir in sorted(p for p in fixtures_dir.iterdir() if p.is_dir()):
        cleans = sorted(language_dir.glob("*_clean.wav"))
        if cleans:
            languages[language_dir.name] = cleans
    return languages


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=Path("artifacts/audio-matrix"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reference-fixtures"))
    parser.add_argument("--segment-seconds", type=float, default=8.0, help="0 keeps each reference whole")
    parser.add_argument("--max-segments", type=int, default=6, help="Cap segments taken from one reference")
    parser.add_argument("--realistic", action="store_true", help="Build fixtures from real tape noise and drifting hum")
    parser.add_argument("--seed", type=int, default=4242, help="Vary to build an independent evaluation set")
    parser.add_argument("--corpus-dir", type=Path, default=Path("experiments/ia_corpus_1000"))
    parser.add_argument("--catalog", type=Path, default=None, help="Defaults to <corpus-dir>/catalog_1000.json")
    return parser.parse_args()


def _load_noise_bank(args):
    """Samples real tape noise from the corpus, which is what makes these fixtures honest."""
    catalog_path = args.catalog or (args.corpus_dir / "catalog_1000.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    bank = build_noise_bank(args.corpus_dir, catalog, args.output_dir / "_noise_probe")
    if not bank:
        raise SystemExit(f"No usable tape noise found under {args.corpus_dir}")
    sys.stdout.write(f"  sampled real tape noise from {len(bank)} captures" + chr(10))
    return bank


def main():
    """Writes the paired fixture corpus and its manifest."""
    args = _parse_args()
    if not args.fixtures_dir.is_dir():
        raise SystemExit(f"No fixtures at {args.fixtures_dir}; generate them first.")
    languages = _discover_languages(args.fixtures_dir)
    if not languages:
        raise SystemExit(f"No *_clean.wav references found under {args.fixtures_dir}")

    segment_frames = int(args.segment_seconds * FIXTURE_SAMPLE_RATE)
    noise_bank = _load_noise_bank(args) if args.realistic else None
    manifest = {"sample_rate": FIXTURE_SAMPLE_RATE, "excluded_defects": list(EXCLUDED_DEFECTS), "languages": {}}
    for language, clean_paths in languages.items():
        if args.realistic:
            records = _build_realistic(clean_paths, args.output_dir / language, segment_frames, args.max_segments, noise_bank, args.seed)
        else:
            records = _build_language(clean_paths, args.output_dir / language, segment_frames, args.max_segments)
        manifest["languages"][language] = records
        sys.stdout.write(f"  {language}: {len(records)} paired fixtures from {len(clean_paths)} references\n")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total = sum(len(v) for v in manifest["languages"].values())
    sys.stdout.write(f"\nWrote {total} paired fixtures and {manifest_path}\n")


if __name__ == "__main__":
    main()
