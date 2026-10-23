"""The bundled RNNoise models match their SHA-256 pins byte for byte, and Git never rewrites them.

The models are plain ASCII, so a Windows checkout with core.autocrlf true wrote them with CRLF
line endings: every bundled model missed its pin (cb.rnnn read 51bb92b9... against f1357c4e...),
and the first arnndn_speech run deleted it and downloaded it again. `.gitattributes` marks
`*.rnnn` binary; a checkout made before that rule needs the models checked out again.
"""

import hashlib
import re
from pathlib import Path

import pytest

from modules import filters

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "models" / "arnndn"
BUNDLED = sorted(path.name for path in MODEL_DIR.glob("*.rnnn"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_some_models_are_bundled_and_every_one_is_pinned():
    """The model folder holds the default model, and no bundled model goes unpinned."""
    assert filters.ARNNDN_MODEL_SHA256.keys() >= set(BUNDLED) >= {"cb.rnnn"}


@pytest.mark.parametrize("name", BUNDLED)
def test_a_bundled_model_matches_its_pin(name):
    """A CRLF checkout fails here; check the models out again after `.gitattributes` (`git checkout -- models/arnndn`)."""
    assert _sha256(MODEL_DIR / name) == filters.ARNNDN_MODEL_SHA256[name]


def test_git_treats_the_models_as_binary():
    """`.gitattributes` keeps Git from converting the models' line endings on any platform."""
    rules = (REPO / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "*.rnnn binary" in rules


@pytest.mark.parametrize("name", BUNDLED)
def test_the_provenance_readme_names_the_same_pin(name):
    """models/arnndn/README.md records each bundled model's SHA-256; it agrees with the code's pin."""
    readme = (MODEL_DIR / "README.md").read_text(encoding="utf-8")
    pins = dict(re.findall(r"\*\*`([a-z]+\.rnnn)`\*\*.*?SHA-256: `([0-9a-f]{64})`", readme, flags=re.S))
    assert pins.get(name) == filters.ARNNDN_MODEL_SHA256[name]
