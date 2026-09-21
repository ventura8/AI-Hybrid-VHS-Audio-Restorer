"""Opt-in tests that load the output-quality models (Whisper, WavLM, UTMOS, SIGMOS, Audiobox).

Gated at collection, like tests/hardware: these modules import the ml dependency group
and need the weights `scripts/download_quality_models.py` fetches, neither of which CI
has. Set AI_RESTORE_QUALITY_MODELS=1 to run them.
"""

import os

QUALITY_MODEL_TESTS_ENABLED = os.environ.get("AI_RESTORE_QUALITY_MODELS") == "1"

collect_ignore_glob = [] if QUALITY_MODEL_TESTS_ENABLED else ["test_*.py"]
