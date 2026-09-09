# Workflow: Run Hardware Validation

Use this playbook to audit a machine and produce deterministic, defected Piper
fixtures without making the standard test suite depend on a GPU or voice model.

## 1. Audit the host

```powershell
poetry run python scripts/audit_hardware.py
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\audit_hardware.ps1
```

Record the JSON report with the benchmark result. Confirm the intended
PyTorch device and ONNX execution provider, not only the GPU name.

## 2. Pin a Piper voice and generate fixtures

The language catalog supplies one checksum-pinned Piper voice per language. Run:

```powershell
poetry run python scripts/generate_audio_matrix.py core --language all
```

The generator invokes Piper through `tools/piper-tts/.venv`, not through the
application environment. Run `install_dependencies.ps1` once if that isolated
runtime is absent; it deliberately contains Piper's CPU ONNX Runtime while the
application retains CUDA/TensorRT ONNX Runtime support.

Use `longform` only for a separately approved sustained run, since it creates
a five-minute fixture by default.

## 3. Plan or run validation

```powershell
poetry run python scripts/run_hardware_validation.py core
poetry run python scripts/run_hardware_validation.py short --execute --mode vhs_native
$env:AI_RESTORE_HARDWARE_TESTS = "1"
poetry run pytest tests/hardware -v
```

The runner writes `artifacts/hardware-validation.json`. Preserve it with the
machine's driver and model details when comparing RTF or memory results.

## 4. Finish with normal quality checks

Run the canonical PowerShell pipeline before declaring code changes complete.
Hardware validation is supplementary; it must not require physical hardware in
ordinary CI.
