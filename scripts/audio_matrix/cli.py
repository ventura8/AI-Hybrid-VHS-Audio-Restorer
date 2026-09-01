"""Generate pinned Piper fixtures and their ground-truth metadata."""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import numpy as np

from scripts.audio_matrix.longform import repeat_to_duration
from scripts.audio_matrix.manifest import load_languages, load_manifest
from scripts.audio_matrix.piper import ensure_voice, synthesize
from scripts.audio_matrix.vhs_defects import apply_vhs_defects

FIXTURE_SAMPLE_RATE = 44100


def _resample_fixture(samples, source_rate):
    """Return samples resampled to the fixed hardware-validation rate."""
    if source_rate == FIXTURE_SAMPLE_RATE:
        return samples
    target_frames = round(len(samples) * FIXTURE_SAMPLE_RATE / source_rate)
    positions = np.linspace(0, len(samples) - 1, target_frames)
    source_positions = np.arange(len(samples))
    return np.column_stack([np.interp(positions, source_positions, samples[:, channel]) for channel in range(samples.shape[1])]).astype(
        np.float32
    )


def generate_fixture(entry, voice, output_dir, piper_python, voices_dir):
    """Generate clean and defected WAVs plus an audit-friendly sidecar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    language_dir = output_dir / voice.language
    language_dir.mkdir(parents=True, exist_ok=True)
    clean_path = language_dir / f"{entry.name}_clean.wav"
    model = ensure_voice(voice.voice, voice.md5, voices_dir, piper_python)
    synthesize(voice.text, model, clean_path, piper_python)
    samples, sample_rate = sf.read(clean_path, dtype="float32", always_2d=True)
    samples = _resample_fixture(samples, sample_rate)
    sample_rate = FIXTURE_SAMPLE_RATE
    if len(samples) / sample_rate > entry.duration_seconds:
        raise ValueError(f"Fixture text exceeds the {entry.duration_seconds}-second {entry.name} profile.")
    timeline = repeat_to_duration(samples, sample_rate, entry.duration_seconds)
    sf.write(clean_path, timeline, sample_rate, subtype="FLOAT")
    degraded_path = language_dir / f"{entry.name}_vhs.wav"
    sf.write(degraded_path, apply_vhs_defects(timeline, sample_rate, entry.defects), sample_rate, subtype="FLOAT")
    sidecar = {
        "fixture": entry.name,
        "language": voice.language,
        "voice": voice.voice,
        "md5": voice.md5,
        "duration_seconds": entry.duration_seconds,
        "sample_rate": sample_rate,
        "defects": entry.defects,
    }
    (language_dir / f"{entry.name}.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    return degraded_path


def select_languages(languages, requested):
    """Return requested language entries with a useful unknown-code error."""
    if not requested or "all" in requested:
        return languages
    unknown = sorted(set(requested).difference(languages))
    if unknown:
        available = ", ".join(sorted(languages))
        raise ValueError(f"Unknown language code(s): {', '.join(unknown)}. Available: {available}")
    return {key: languages[key] for key in requested}


def main(argv=None):
    """Run the matrix generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("short", "mid", "longform", "core", "all"))
    parser.add_argument("--piper-python", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/audio-matrix"))
    parser.add_argument("--voices-dir", type=Path, default=Path("artifacts/piper-voices"))
    parser.add_argument("--language", action="append", dest="languages", default=[])
    args = parser.parse_args(argv)
    manifest = load_manifest()
    languages = load_languages()
    selected = select_languages(languages, args.languages)
    names = ("short", "mid") if args.profile == "core" else tuple(manifest) if args.profile == "all" else (args.profile,)
    for language in selected.values():
        for name in names:
            print(generate_fixture(manifest[name], language, args.output_dir, args.piper_python, args.voices_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
