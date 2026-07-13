# SKILL: Audio Pipeline Implementation

Use this skill when editing processing, sync, hardware, and utility modules.

## Objective

Maintain restoration quality and pipeline stability while improving code.

## Domain Constraints

- Keep 32-bit float audio handling for intermediate and final processing.
- Preserve process-mode behavior contract:
  - `hybrid`: separation + vocal enhancement + background denoise + sync + mix.
  - `denoise_only`: full-audio denoise + sync + remux.
- Preserve lossless-background strategy in `hybrid` separation flow.
- Maintain resume safety checks (valid file checks, not existence-only checks).
- Keep sequential high-level stage execution to avoid unstable mixed output.
- Preserve mode-specific output naming:
  - `*_Hybrid_Cleaned.<ext>` for `hybrid`
  - `*_Denoised_Cleaned.<ext>` for `denoise_only`
- Keep input scanning exclusions aligned with both cleaned output suffixes.

## Sync Constraints

- Support both shift and DTW alignment paths.
- Preserve DTW performance and reliability behavior.
- Avoid introducing regressions in pathing and temp file handling.

## Hardware Constraints

- Retain NVIDIA prioritization logic.
- Preserve CUDA DLL/path injection behavior for local Windows execution.
- Keep adaptive batch sizing tied to VRAM profiles.

## Validation

- Run full local pipeline after changes.
- Ensure coverage remains above threshold and badge is regenerated.
