#!/usr/bin/env python3
"""Fetches the DNSMOS P.835 models into models/dnsmos, checksum-pinned, with their provenance.

DNSMOS is Microsoft's non-intrusive speech quality estimator from the Deep Noise
Suppression Challenge: two ONNX networks that read 16 kHz audio and return the three
P.835 scores (speech quality SIG, background BAK, overall OVRL) and a P.808 overall MOS.
`scripts/score_perceptual.py` runs them as a cross-check on the restored corpus -- a
reference-free perceptual reading beside the trade metric, never a gate on its own, since
the models are speech-trained and read at 16 kHz.

Source: https://github.com/microsoft/DNS-Challenge (DNSMOS/DNSMOS/), licensed CC BY 4.0.
The files are not tracked; this script is, and pins what it fetched.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/"
TARGET_DIR = Path(__file__).resolve().parent.parent / "models" / "dnsmos"
# Fetched 2026-09-12 from the commit then at master; a file that does not hash to this is
# not the file the cross-check was written against and is refused.
MODELS = {
    "sig_bak_ovr.onnx": "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
    "model_v8.onnx": "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91",
}
README = """# DNSMOS P.835 models: provenance and licensing

The two ONNX files in this directory are the DNSMOS P.835 estimator from Microsoft's
Deep Noise Suppression Challenge repository, fetched by `scripts/download_dnsmos.py`:

- `sig_bak_ovr.onnx` -- SIG, BAK and OVRL (ITU-T P.835), 16 kHz waveform input
- `model_v8.onnx` -- P.808 overall MOS, 120-band log-mel input

Upstream: https://github.com/microsoft/DNS-Challenge, path `DNSMOS/DNSMOS/`.
License: Creative Commons Attribution 4.0 International (CC BY 4.0), which requires
attribution to Microsoft and the DNS Challenge authors when the files are shared.

Reference: Chandan K. A. Reddy, Vishak Gopal, Ross Cutler, "DNSMOS P.835: A
Non-Intrusive Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors",
ICASSP 2022.

SHA-256 as fetched:
"""


def _sha256(path):
    """The file's SHA-256, read in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name, expected, target_dir=TARGET_DIR):
    """Downloads one model unless a file with the pinned hash is already there; refuses a mismatch."""
    target = target_dir / name
    if target.is_file() and _sha256(target) == expected:
        print(f"  {name}: present")
        return target
    target_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(BASE_URL + name, target)
    actual = _sha256(target)
    if actual != expected:
        target.unlink(missing_ok=True)
        raise SystemExit(f"{name}: fetched {actual}, expected {expected}; refusing to keep it")
    print(f"  {name}: fetched, {target.stat().st_size} bytes")
    return target


def main():
    """Fetches every model and writes the provenance note beside them."""
    for name, expected in MODELS.items():
        fetch(name, expected)
    lines = README + "".join(f"- `{name}`: `{digest}`\n" for name, digest in MODELS.items())
    (TARGET_DIR / "README.md").write_text(lines, encoding="utf-8")
    print(f"wrote {TARGET_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
