# Hardware Optimization

- **Auto-Detection**: The `get_optimal_settings()` function detects CPU cores,
  GPU VRAM, and acceleration backends across Linux, macOS, and Windows.
- **Profiles**:
  - `EXTREME` (24GB+ VRAM): Batch size 32. **Parallel Processing Enabled**.
  - `HIGH` (15GB–\<24GB VRAM): Batch size 8.
  - `MID` (10GB–\<15GB VRAM): Batch size 4.
  - `LOW` (\<10GB VRAM): Batch size 1.
  - `MPS` (Apple Silicon Unified Memory): Batch size selected dynamically from
    unified-memory size by `_detect_mps_backend()` / `_apply_vram_profile()`
    (32 / 8 / 4 / 1 on the same thresholds as the VRAM profiles), Metal
    Performance Shaders acceleration.
- **OOM Resiliency**: `attempt_run_with_retry` and `attempt_cpu_run_with_retry`
  dynamically reduce batch sizes/threads on failure.
- **Dynamic Linker Injection**: Linux (`LD_LIBRARY_PATH`) and Windows dynamic
  linkers inject CUDNN/CUBLAS libraries from in-project virtualenvs on
  NVIDIA/CUDA platforms. On macOS, Apple Silicon MPS uses native Metal frameworks
  and does not require CUDA dynamic library injection.
