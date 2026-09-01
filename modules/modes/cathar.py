"""Cathar DSP pure-Rust restoration mode tailored for VHS captures."""

from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseRestorationMode


class CatharMode(BaseRestorationMode):
    """Pure-Rust Cathar DSP restoration pipeline (*_Cathar_Cleaned)."""

    mode_name = "cathar"
    display_name = "Cathar DSP"

    def execute(
        self,
        work_dir: Path,
        original_wav: Path,
        video_path: Path,
        final_output_video: Path,
        video_dur: float,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Executes pure-Rust Cathar DSP restoration pipeline tailored for VHS captures."""
        from .. import processing
        from ..filters import filter_cathar_vhs_pipeline

        clean_wav, strategy = processing._resolve_preconditioned_audio(work_dir, original_wav, video_dur, self.mode_name, strategy)

        def cathar_step(in_wav, out_dir, total_duration=None):
            raw_restored = filter_cathar_vhs_pipeline(in_wav, out_dir, total_duration=total_duration, strategy=strategy)
            return processing._polish_full_audio_step(
                raw_restored, out_dir, total_duration=total_duration, strategy=strategy, apply_air=False
            )

        processing._process_single_track_pipeline(
            work_dir,
            clean_wav,
            video_path,
            final_output_video,
            video_dur,
            cathar_step,
            "cathar_restored",
            self.display_name,
            ref_wav=original_wav,
            sync_method=self.resolve_strategy_val(strategy, "sync_method", None),
        )
