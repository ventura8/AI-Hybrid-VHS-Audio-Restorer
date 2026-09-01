---
name: hardware-validation
description: >-
  Generate deterministic Piper speech fixtures and audit or validate local
  CPU, CUDA, ONNX Runtime, and DirectML restoration hardware.
---

# Hardware Validation

Use this skill when validating the restoration pipeline on a real machine or
when preparing reproducible synthetic audio fixtures. It is opt-in: do not
download voice models, generate long-form audio, or start model inference
without the user's request.

## Audit first

Run `poetry run python scripts/audit_hardware.py` before choosing validation
workloads. Treat the resulting JSON as the authoritative local capability
record: GPU detection alone does not prove that PyTorch or ONNX Runtime can
execute on it.

On Windows, also run `scripts/audit_hardware.ps1`; it reports the driver view
from `nvidia-smi` when that executable is present.

## Generate fixtures

The checked-in language catalog includes every Piper language represented by the
reference voice matrix. Each voice has an upstream MD5 pin. Generate every
language with zero noise and noise-width scales:

```powershell
poetry run python scripts/generate_audio_matrix.py core --language all
```

Piper must run only from `tools/piper-tts/.venv`, which is provisioned by the
project installer. This prevents Piper's CPU-only ONNX Runtime dependency from
overwriting the main CUDA/TensorRT runtime. Rerun `install_dependencies.ps1` in
an existing Windows checkout to provision the isolated runtime.

Use `short` for a fast accelerator smoke check, `mid` for quality validation,
and `longform` only when sustained VRAM and thermal validation is intended.
Fixtures and ground-truth sidecars are written beneath `artifacts/audio-matrix`.

## Validate safely

Preview the selected fixtures and all ten canonical modes without inference:

```powershell
poetry run python scripts/run_hardware_validation.py core
```

Pytest checks under `tests/hardware/` serve as metadata, planning, and dry-run
verification checks rather than physical GPU validation; they inspect catalog
integrity, monkeypatch device settings, and build dry-run reports without model
inference. Physical execution tests are strictly opt-in and skipped unless
the operator explicitly sets `AI_RESTORE_HARDWARE_TESTS=1`. Add `--execute` to
`run_hardware_validation.py` only when actual inference and runner report
validation on physical hardware are intended.
