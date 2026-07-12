# Configuration

- **config.yaml**: Manages global settings like audio mix levels (`vocal_mix_volume`,
  `background_mix_volume`), sync behavior, process mode, and file extensions.
- **Defaults**: If `config.yaml` is missing, the script defaults to neutral mix levels (1.0),
  `process_mode: denoise_only`, and standard video extensions (`.mp4`, `.mkv`, `.avi`, `.mov`).

## Process Mode

- `hybrid`:
	- Separation + vocal enhancement + background denoise + sync + final mix.
	- Output suffix: `*_Hybrid_Cleaned`.
- `denoise_only`:
	- Full-audio denoise + sync + final remux.
	- No separation or vocal enhancement.
	- Output suffix: `*_Denoised_Cleaned`.
