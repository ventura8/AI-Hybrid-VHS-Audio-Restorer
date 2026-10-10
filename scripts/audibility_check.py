#!/usr/bin/env python3
"""Is a candidate render audibly different from the incumbent? (ear v3, plan 1.2)

    audibility_check.py INCUMBENT.wav CANDIDATE.wav [--json out.json]

Prints, and with --json writes, `{identical, lag, polarity, diff_db, changed_share,
audible_frac, nmr_max, event_frames, audible, channel_mismatch, nonfinite,
length_mismatch_s, window_audible_frac, window_event_frames, incumbent_sha256,
candidate_sha256}`:

- identical: the decoded PCM hashes match and every sample is finite (the NMR is skipped;
  the self-driving loop decides inertness with its own exact hash, not this one, but its
  audibility tie goes through this check and keeps this field);
- lag, polarity: how the candidate was aligned to the incumbent (samples, sign);
- diff_db: the difference level against the incumbent's loud frames; changed_share: the
  share of samples that moved by more than -120 dBFS;
- audible_frac: the largest per-15 s-window share of frames whose noise-to-mask ratio
  exceeds 0 dB; nmr_max: the largest NMR; event_frames: the largest per-window count of
  frames at +6 dB or more (a click, a dropout);
- audible: some window reaches 5% audible frames or 2 event frames, some frame reaches
  +12 dB (a click near a frame boundary lands in one frame only), or the pair cannot be
  vouched for (else a tie);
- channel_mismatch: two different multichannel counts, compared as mono downmixes (a mono
  file against a stereo one is compared on every channel); nonfinite: NaN or inf samples
  in either file, read as silence; length_mismatch_s: what the alignment leaves
  uncompared at both ends (the |lag| head dropped from the late file plus the tail one
  file holds past the other), so past 50 ms the pair reads audible (a render cut short,
  empty or missing its start is no tie, and neither is a delay over 50 ms).

One-sided: "inaudible" is trusted, "audible" proves nothing about better. Inputs are any
file soundfile reads (the loop's candidates are WAVs); both must share a sample rate. The
model and its calibration on the stored Tata files are in `scripts/restoration_quality/auditory.py`.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.cli_paths import confined_path, existing_path_arg, path_arg
    from scripts.restoration_quality import auditory
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.cli_paths import confined_path, existing_path_arg, path_arg
    from scripts.restoration_quality import auditory


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("incumbent", type=existing_path_arg, help="the incumbent render (A)")
    parser.add_argument("candidate", type=existing_path_arg, help="the candidate render (B)")
    parser.add_argument("--json", type=path_arg, default=None, help="also write the readings to this JSON file")
    return parser.parse_args(argv)


def _write_json(result, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    """Compares the two files, prints the readings as JSON and returns them."""
    args = _parse_args(argv)
    incumbent = confined_path(args.incumbent, "incumbent", must_exist=True)
    candidate = confined_path(args.candidate, "candidate", must_exist=True)
    try:
        result = auditory.compare_files(incumbent, candidate)
    except ValueError as error:
        raise SystemExit(f"audibility_check: {error}") from error
    if args.json is not None:
        _write_json(result, confined_path(args.json, "--json"))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
