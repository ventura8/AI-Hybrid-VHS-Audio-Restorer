"""Denoise Only restoration mode (full-mix broadband denoising)."""

from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseRestorationMode


class DenoiseOnlyMode(BaseRestorationMode):
    """Full audio track denoising with UVR-DeNoise and preconditioning (*_Denoised_Cleaned)."""

    mode_name = "denoise_only"
    display_name = "Full-Audio Denoise"

    def execute(
        self,
        work_dir: Path,
        original_wav: Path,
        video_path: Path,
        final_output_video: Path,
        video_dur: float,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Full audio track denoising with UVR-DeNoise and preconditioning."""
        from .. import processing

        clean_wav, strategy = processing._resolve_preconditioned_audio(work_dir, original_wav, video_dur, self.mode_name, strategy)
        denoise_model = self.resolve_strategy_val(strategy, "denoise_model", None)

        def denoise_step(in_wav, out_dir, total_duration=None):
            return processing._denoise_and_polish_full_audio_step(
                in_wav, out_dir, total_duration=total_duration, denoise_model=denoise_model, strategy=strategy, apply_air=False
            )

        processing._process_single_track_pipeline(
            work_dir,
            clean_wav,
            video_path,
            final_output_video,
            video_dur,
            denoise_step,
            "denoised_full_audio",
            self.display_name,
            sync_method=self.resolve_strategy_val(strategy, "sync_method", None),
            ref_wav=original_wav,
        )
