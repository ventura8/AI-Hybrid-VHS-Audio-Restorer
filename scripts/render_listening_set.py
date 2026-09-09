#!/usr/bin/env python3
"""Renders before/after audio from real tapes so a fidelity trade can be judged by ear.

Every metric argument on this branch eventually needed a reality check, and the remaining
question -- whether restoring content that subtraction over-cut is worth removing less noise
-- is a judgement about how the mode should sound. Numbers cannot settle it.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.hardware import get_gpu_name
from modules.utils import FFMPEG_BIN
from scripts.ia_benchmark_common import _run_mode_restoration


def _extract(video, target):
    """Pulls the audio out of a container as 16-bit PCM for easy playback; None when ffmpeg fails or times out."""
    try:
        subprocess.run(
            [FFMPEG_BIN, "-y", "-i", str(video), "-vn", "-acodec", "pcm_s16le", "-ar", "44100", str(target)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"  extraction timed out: {Path(video).name}\n")
        return None
    return target if target.exists() else None


def main():
    """Renders one clip per acoustic situation for each requested mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("experiments/listen"))
    parser.add_argument("--tag", required=True, help="Label distinguishing this configuration")
    parser.add_argument("--skip-reference", action="store_true", help="Skip source and cathar renders")
    args = parser.parse_args()

    picks = json.loads(Path("experiments/listen_picks.json").read_text(encoding="utf-8"))
    corpus = Path("experiments/ia_corpus_1000")
    args.out.mkdir(parents=True, exist_ok=True)
    gpu = get_gpu_name()

    for label, pick in picks.items():
        clip = corpus / pick["file"]
        if not clip.exists():
            continue
        # Work directory per configuration: _run_mode_restoration returns an existing
        # output rather than recomputing, so sharing one directory between two
        # configurations silently renders the second as a copy of the first.
        work = args.out / f"_work_{label}_{args.tag}"
        work.mkdir(parents=True, exist_ok=True)
        if not args.skip_reference:
            _extract(clip, args.out / f"{label}__00_source.wav")
            restored = _run_mode_restoration(clip, "cathar", work, gpu)
            if restored:
                _extract(restored, args.out / f"{label}__01_cathar.wav")
        restored = _run_mode_restoration(clip, "auto_pure_linear", work, gpu)
        if restored:
            _extract(restored, args.out / f"{label}__02_apl_{args.tag}.wav")
        sys.stdout.write(f"  rendered {label} ({args.tag})\n")


if __name__ == "__main__":
    main()
