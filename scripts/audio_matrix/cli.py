"""Generate pinned Piper fixtures and their ground-truth metadata."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from scripts.audio_matrix.longform import repeat_to_duration
from scripts.audio_matrix.manifest import load_languages, load_manifest
from scripts.audio_matrix.piper import ensure_voice, synthesize
from scripts.audio_matrix.vhs_defects import apply_vhs_defects

FIXTURE_SAMPLE_RATE = 44100


def _resample_fixture(samples, source_rate):
    """Return samples resampled to the fixed hardware-validation rate."""
    if source_rate == FIXTURE_SAMPLE_RATE:
        return samples
    channels = [
        librosa.resample(samples[:, channel], orig_sr=source_rate, target_sr=FIXTURE_SAMPLE_RATE) for channel in range(samples.shape[1])
    ]
    return np.column_stack(channels).astype(np.float32)


def _synthesize_clean_voice(voice, output_dir, piper_python, voices_dir):
    """Ensure voice model and synthesize clean audio once for a language."""
    language_dir = output_dir / voice.language
    language_dir.mkdir(parents=True, exist_ok=True)
    clean_path = language_dir / "clean.wav"
    model = ensure_voice(voice.voice, voice.md5, voices_dir, piper_python)
    synthesize(voice.text, model, clean_path, piper_python)
    samples, sample_rate = sf.read(clean_path, dtype="float32", always_2d=True)
    return _resample_fixture(samples, sample_rate)


def _prepare_fixture_timeline(samples, sample_rate, duration_seconds):
    """Return trimmed or repeated audio matching the profile duration."""
    target_len = int(duration_seconds * sample_rate)
    if len(samples) >= target_len:
        return samples[:target_len]
    return repeat_to_duration(samples, sample_rate, duration_seconds)


def _write_fixture_sidecar(language_dir, entry, voice, clean_path, degraded_path):
    """Write sidecar metadata for a generated fixture."""
    sidecar = {
        "fixture": entry.name,
        "language": voice.language,
        "voice": voice.voice,
        "md5": voice.md5,
        "duration_seconds": entry.duration_seconds,
        "sample_rate": FIXTURE_SAMPLE_RATE,
        "defects": entry.defects,
        "clean_sha256": hashlib.sha256(clean_path.read_bytes()).hexdigest(),
        "vhs_sha256": hashlib.sha256(degraded_path.read_bytes()).hexdigest(),
    }
    (language_dir / f"{entry.name}.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def generate_fixture(entry, voice, output_dir, piper_python, voices_dir, clean_samples=None):
    """Generate clean and defected WAVs plus an audit-friendly sidecar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    language_dir = output_dir / voice.language
    language_dir.mkdir(parents=True, exist_ok=True)
    if clean_samples is None:
        clean_samples = _synthesize_clean_voice(voice, output_dir, piper_python, voices_dir)
    timeline = _prepare_fixture_timeline(clean_samples, FIXTURE_SAMPLE_RATE, entry.duration_seconds)
    clean_path = language_dir / f"{entry.name}_clean.wav"
    sf.write(clean_path, timeline, FIXTURE_SAMPLE_RATE, subtype="FLOAT")
    degraded_path = language_dir / f"{entry.name}_vhs.wav"
    sf.write(degraded_path, apply_vhs_defects(timeline, FIXTURE_SAMPLE_RATE, entry.defects), FIXTURE_SAMPLE_RATE, subtype="FLOAT")
    _write_fixture_sidecar(language_dir, entry, voice, clean_path, degraded_path)
    return degraded_path


def select_languages(languages, requested):
    """Return requested language entries with a useful unknown-code error."""
    unknown = sorted(set(requested).difference(languages).difference({"all"}))
    if unknown:
        available = ", ".join(sorted(languages))
        raise ValueError(f"Unknown language code(s): {', '.join(unknown)}. Available: {available}")
    if not requested or "all" in requested:
        return languages
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
        clean_samples = _synthesize_clean_voice(language, args.output_dir, args.piper_python, args.voices_dir)
        for name in names:
            print(generate_fixture(manifest[name], language, args.output_dir, args.piper_python, args.voices_dir, clean_samples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
