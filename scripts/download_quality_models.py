#!/usr/bin/env python3
"""Fetch the output-quality models into the model store, pinned to a revision and a hash.

    download_quality_models.py [--set all|mos|speech] [--check] [--models-dir models]

Each model is fetched from a pinned upstream revision (a Hugging Face commit or a GitHub
commit) and every file is hashed after download. The first fetch records the observed
sha256 of each file in `<models-dir>/<name>/MANIFEST.json`; a later run that hashes
differently is refused, the way `scripts/download_dnsmos.py` refuses a mismatched file.
The weights themselves are not tracked; this script is, and pins what it fetched.

UTMOS is loaded through torch.hub from a pinned release tag; `torch.hub.set_dir` keeps
its checkpoint under the model store too.

Licences: SIGMOS, Whisper, WavLM, UTMOS - MIT; Audiobox Aesthetics - CC-BY-4.0.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Pinned 2026-09-20. Revisions are the upstream commits then current; a file that does not
# hash to what the first fetch recorded is not the file the calibration was written against.
PINS = {
    "sigmos": {
        "set": "mos",
        "kind": "github",
        # The .onnx is stored in Git LFS: raw.githubusercontent.com serves a pointer, github.com/.../raw/ the file.
        "base": "https://github.com/microsoft/SIG-Challenge/raw/d3640faa8d36b9398000e01f70ea60b28b243167/ICASSP2024/sigmos/",
        "files": ["sigmos.py", "model-sigmos_1697718653_41d092e8-epo-200.onnx"],
        "sha256": {"model-sigmos_1697718653_41d092e8-epo-200.onnx": "f939dcc1945055a435565b4369e27dafd0f87df3cea4e2ff6eb81225e52cc53b"},
        "license": "MIT (microsoft/SIG-Challenge)",
        "source": "https://github.com/microsoft/SIG-Challenge/tree/main/ICASSP2024/sigmos",
        "size_mb": 40,
    },
    "whisper-large-v3-turbo": {
        "set": "speech",
        "kind": "hf",
        "repo": "openai/whisper-large-v3-turbo",
        "revision": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        "files": [
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "normalizer.json",
            "added_tokens.json",
            "special_tokens_map.json",
            "model.safetensors",
        ],
        "license": "MIT (openai/whisper)",
        "source": "https://huggingface.co/openai/whisper-large-v3-turbo",
        "size_mb": 1620,
    },
    "wavlm-base-plus-sv": {
        "set": "speech",
        "kind": "hf",
        "repo": "microsoft/wavlm-base-plus-sv",
        "revision": "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e",
        "files": ["config.json", "preprocessor_config.json", "pytorch_model.bin"],
        "license": "MIT (microsoft/unilm WavLM)",
        "source": "https://huggingface.co/microsoft/wavlm-base-plus-sv",
        "size_mb": 405,
    },
    "audiobox-aesthetics": {
        "set": "mos",
        "kind": "hf",
        "repo": "facebook/audiobox-aesthetics",
        "revision": "9b1dd8e5df9af7216e836a98974fe3b82c56ded6",
        "files": ["config.json", "checkpoint.pt"],
        "license": "CC-BY-4.0 (facebook/audiobox-aesthetics)",
        "source": "https://huggingface.co/facebook/audiobox-aesthetics",
        "size_mb": 416,
    },
    "utmos": {
        "set": "speech",
        "kind": "torchhub",
        "repo": "tarepan/SpeechMOS:v1.2.0",
        "entry": "utmos22_strong",
        "license": "MIT (tarepan/SpeechMOS, sarulab-speech/UTMOS22)",
        "source": "https://github.com/tarepan/SpeechMOS",
        "size_mb": 380,
    },
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_path(target_dir):
    return target_dir / "MANIFEST.json"


def _load_manifest(target_dir):
    path = _manifest_path(target_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _verify(target_dir, name, observed, pinned=None):
    """Refuses a file whose hash differs from the pinned one, or from the one recorded on first fetch; records a new one."""
    manifest = _load_manifest(target_dir)
    expected = pinned or manifest.get(name)
    if expected and expected != observed:
        Path(target_dir / name).unlink(missing_ok=True)
        raise SystemExit(f"{target_dir.name}/{name}: sha256 {observed} does not match the recorded {expected}; refusing it")
    manifest[name] = observed
    _manifest_path(target_dir).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fetch_github(pin, target_dir):
    for name in pin["files"]:
        target = target_dir / name
        if not target.exists():
            print(f"  fetching {name}")
            urllib.request.urlretrieve(pin["base"] + name, target)
        _verify(target_dir, name, _sha256(target), pin.get("sha256", {}).get(name))


def _fetch_hf(pin, target_dir):
    from huggingface_hub import hf_hub_download

    for name in pin["files"]:
        target = target_dir / name
        if not target.exists():
            print(f"  fetching {name}")
            hf_hub_download(pin["repo"], name, revision=pin["revision"], local_dir=str(target_dir))
        _verify(target_dir, name, _sha256(target))


def _fetch_torchhub(pin, target_dir):
    import torch

    torch.hub.set_dir(str(target_dir))
    torch.hub.load(pin["repo"], pin["entry"], trust_repo=True)
    for checkpoint in sorted((target_dir / "checkpoints").glob("*")):
        _verify(target_dir, f"checkpoints/{checkpoint.name}", _sha256(checkpoint))


FETCHERS = {"github": _fetch_github, "hf": _fetch_hf, "torchhub": _fetch_torchhub}


def _write_readme(name, pin, target_dir):
    lines = [
        f"# {name}",
        "",
        f"Fetched by `scripts/download_quality_models.py` from {pin['source']}",
        f"(revision `{pin.get('revision', pin.get('base', pin.get('repo', '')))}`).",
        "",
        f"Licence: {pin['license']}.",
        "",
        "`MANIFEST.json` holds the sha256 of every file as observed on first fetch; a later",
        "fetch that hashes differently is refused.",
        "",
    ]
    (target_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def fetch(name, pin, models_dir=MODELS_DIR):
    """Fetches one pinned model into `<models_dir>/<name>` and writes its README."""
    target_dir = models_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"{name} ({pin['size_mb']} MB, {pin['license']})")
    FETCHERS[pin["kind"]](pin, target_dir)
    _write_readme(name, pin, target_dir)


def _mismatched(target_dir, manifest, present):
    """Files whose current hash differs from the recorded one."""
    return [f for f in present if manifest.get(f) not in (None, _sha256(target_dir / f))]


def _split_present(target_dir, files):
    """(missing, present) among `files` under `target_dir`."""
    missing = [f for f in files if not (target_dir / f).exists()]
    return missing, [f for f in files if f not in missing]


def check(name, pin, models_dir=MODELS_DIR):
    """True when every file of the model is present and hashes as recorded."""
    target_dir = models_dir / name
    manifest = _load_manifest(target_dir)
    missing, present = _split_present(target_dir, pin.get("files") or list(manifest))
    bad = _mismatched(target_dir, manifest, present)
    if missing or bad:
        print(f"{name}: missing {missing or '-'}; mismatched {bad or '-'}")
    return not (missing or bad)


def _selected(which):
    return {name: pin for name, pin in PINS.items() if which == "all" or pin["set"] == which}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="all", choices=("all", "mos", "speech"))
    parser.add_argument("--check", action="store_true", help="hash what is present, download nothing")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)
    selected = _selected(args.set)
    if args.check:
        return _check_all(selected, args.models_dir)
    for name, pin in selected.items():
        fetch(name, pin, args.models_dir)
    return 0


def _check_all(selected, models_dir):
    ok = all([check(name, pin, models_dir) for name, pin in selected.items()])
    print("all present" if ok else "run scripts/download_quality_models.py to fetch the missing models")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
