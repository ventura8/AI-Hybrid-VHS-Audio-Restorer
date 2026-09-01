"""Audio-matrix manifest loading and validation."""

import json
from dataclasses import dataclass
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name("manifest.json")
LANGUAGES_PATH = Path(__file__).with_name("languages.json")


@dataclass(frozen=True)
class MatrixEntry:
    """One reproducible audio fixture definition."""

    name: str
    duration_seconds: int
    text: str
    defects: tuple[str, ...]


@dataclass(frozen=True)
class VoiceEntry:
    """A language-native Piper voice and deterministic validation utterance."""

    language: str
    voice: str
    md5: str
    text: str


def load_manifest(path=MANIFEST_PATH):
    """Load the checked-in fixture definitions."""
    raw_entries = json.loads(Path(path).read_text(encoding="utf-8"))["fixtures"]
    return {entry["name"]: MatrixEntry(**{**entry, "defects": tuple(entry["defects"])}) for entry in raw_entries}


def load_voice_pin(path=MANIFEST_PATH):
    """Return the voice identity and checksum pin from a manifest, or None."""
    return json.loads(Path(path).read_text(encoding="utf-8")).get("voice")


def load_languages(path=LANGUAGES_PATH):
    """Load every language with a Piper voice in the checked-in catalog."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["voices"]
    return {entry["language"]: VoiceEntry(**entry) for entry in raw}
