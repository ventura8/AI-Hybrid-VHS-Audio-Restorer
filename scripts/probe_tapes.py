#!/usr/bin/env python3
"""Probe durations and stream parameters across video captures in a directory or file."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure modules package can be resolved
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import EXTS
from modules.utils import FFPROBE_BIN


def probe_single_tape(tape_path):
    """Extracts duration and primary audio stream metadata."""
    try:
        out = subprocess.check_output(
            [
                FFPROBE_BIN,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,channels,channel_layout,sample_rate,bit_rate:format=duration",
                "-of",
                "json",
                str(tape_path),
            ],
            timeout=10,
        )

        info = json.loads(out.decode("utf-8"))
        fmt = info.get("format", {})
        duration_sec = float(fmt.get("duration", 0))
        streams = info.get("streams", [])
        audio_stream = streams[0] if streams else None
        return {
            "path": tape_path,
            "duration_sec": duration_sec,
            "duration_min": duration_sec / 60.0,
            "audio": audio_stream,
        }
    except Exception as e:
        return {"path": tape_path, "error": str(e)}


def _format_stream_desc(audio):
    """Formats audio stream metadata summary."""
    if not audio:
        return "NO AUDIO STREAM"
    codec = audio.get("codec_name", "unknown")
    channels = audio.get("channels", 0)
    layout = audio.get("channel_layout", "unknown")
    sr = audio.get("sample_rate", "unknown")
    return f"{codec} | {channels}ch ({layout}) | {sr}Hz"


def _evaluate_single_probe(tape, max_duration_min):
    """Probes single tape and prints summary status."""
    info = probe_single_tape(tape)
    if "error" in info:
        print(f"{tape.name:<45} | ERROR: {info['error']}")
        return None

    dur_min = info["duration_min"]
    stream_desc = _format_stream_desc(info["audio"])
    status = f"OK (<={max_duration_min:.0f}m)" if dur_min <= max_duration_min else f"EXCEED (>{max_duration_min:.0f}m)"
    print(f"{tape.name:<45} | {dur_min:>5.1f}m | {status:<15} | {stream_desc}")

    return info if dur_min <= max_duration_min and info["audio"] else None


def _filter_candidate_files(all_files, max_duration_min):
    """Collects and displays all valid probe candidates."""
    valid_candidates = []
    for tape in all_files:
        candidate = _evaluate_single_probe(tape, max_duration_min)
        if candidate:
            valid_candidates.append(candidate)
    return valid_candidates


def _is_valid_capture(path, valid_exts):
    return path.is_file() and path.suffix.lower() in valid_exts


def _scan_directory_captures(directory):
    valid_exts = {e.lower() for e in EXTS}
    return sorted([p for p in directory.iterdir() if _is_valid_capture(p, valid_exts)])


def _resolve_files_to_probe(input_target):
    """Resolves input file or folder into a list of file paths."""
    target = Path(input_target)
    if target.is_file():
        return [target]
    if target.is_dir():
        return _scan_directory_captures(target)
    return []


def probe_directory(input_target, max_duration_min=80.0):
    """Scans directory or file and prints summary of captures matching duration constraints."""
    print("=" * 80)
    print(f"PROBING CAPTURES IN: {input_target}")
    print("=" * 80)

    all_files = _resolve_files_to_probe(input_target)
    if not all_files:
        print(f"No video files found at '{input_target}'", file=sys.stderr)
        return []

    valid_candidates = _filter_candidate_files(all_files, max_duration_min)

    print("=" * 80)
    print(f"Total files: {len(all_files)} | Eligible candidates (<= {max_duration_min:.0f}m): {len(valid_candidates)}")
    print("=" * 80)
    return valid_candidates


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory or video file of captures to probe")
    parser.add_argument("--max-duration", type=float, default=80.0, help="Maximum eligible duration in minutes")
    return parser.parse_args()


def main():
    args = _parse_args()
    probe_directory(args.input, max_duration_min=args.max_duration)


if __name__ == "__main__":
    main()
