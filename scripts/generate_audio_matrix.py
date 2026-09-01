"""Command-line entry point for deterministic Piper audio matrix generation."""

import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
main = import_module("scripts.audio_matrix.cli").main

if __name__ == "__main__":
    raise SystemExit(main())
