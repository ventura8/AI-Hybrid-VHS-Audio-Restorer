"""Auto Pure Linear restoration mode.

Executes pure full-mix restoration without stem separation using pre-denoise surgical
bandreject notching, adaptive UVR-DeNoise neural inference, and linear air polish.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseRestorationMode


class AutoPureLinearMode(BaseRestorationMode):
    """Full-Mix Pure Speech & Ambient Denoising Engine (*_PureLinear_Cleaned)."""

    mode_name = "auto_pure_linear"
    display_name = "Preconditioned Full-Audio"

    def execute(
        self,
        work_dir: Path,
        original_wav: Path,
        video_path: Path,
        final_output_video: Path,
        video_dur: float,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Runs pure full-mix restoration without stem separation."""
        from .. import processing

        clean_wav, strategy = processing._resolve_preconditioned_audio(work_dir, original_wav, video_dur, self.mode_name, strategy)
        denoise_model = self.resolve_strategy_val(strategy, "denoise_model", None)

        def denoise_step(in_wav, out_dir, total_duration=None):
            return processing._denoise_and_polish_full_audio_step(
                in_wav, out_dir, total_duration=total_duration, denoise_model=denoise_model, strategy=strategy, apply_air=True
            )

        processing._process_single_track_pipeline(
            work_dir,
            clean_wav,
            video_path,
            final_output_video,
            video_dur,
            denoise_step,
            "denoised_preconditioned_audio",
            self.display_name,
            sync_method=self.resolve_strategy_val(strategy, "sync_method", None),
            ref_wav=original_wav,
        )
