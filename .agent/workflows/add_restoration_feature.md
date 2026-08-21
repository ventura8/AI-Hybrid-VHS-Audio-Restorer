# Workflow: Add Audio Restoration Feature

Use this workflow when implementing a new audio filter, AI separation model,
enhancement algorithm, or processing mode.

## Step 1: Update Configuration & Schema

1. Add any new mode names or tuning parameters to `config.yaml`.
1. Update `modules/config.py` to parse, validate, and export the new settings.
1. Update `tests/unit/test_config.py` with parameterized test cases.

## Step 2: Implement Filter or Engine Logic

1. For native FFmpeg DSP filters: implement graph builders in
   `modules/filters.py`.
1. For AI models: add model resolution and inference hooks in
   `modules/processing.py`.
1. Ensure atomic file handling via `.tmp.wav` / `.tmp.mp4` with validity
   checks (`is_valid_audio` / `is_valid_video`).
1. Ensure 32-bit float PCM audio pipeline integrity (`pcm_f32le`).

## Step 3: Update UI and File Scanning

1. Add output file suffixes (e.g. `*_Cleaned.*`) to `_is_cleaned_output()` in
   `modules/ui.py` to prevent recursive re-processing.
1. Update active model descriptions in `_get_active_models_label()`.

## Step 4: Add Comprehensive Tests

1. Create dedicated unit tests in `tests/unit/` targeting all new functions.
1. Add end-to-end integration smoke tests in
   `tests/integration/test_end_to_end.py`.
1. Keep test cyclomatic complexity $\\le 5$ (Grade A).

## Step 5: Update Documentation

1. Update `README.md`, `docs/configuration.md`, and `docs/pipeline_logic.md`.
1. Ensure all Markdown prose lines are wrapped to $\\le 80$ characters.

## Step 6: Verify Full Quality Pipeline

```bash
# On Linux / macOS:
./run_pipeline_locally.sh

# On Windows:
./run_pipeline_locally.ps1
```
