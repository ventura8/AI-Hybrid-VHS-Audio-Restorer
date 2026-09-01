"""Auto Pure 2-stem restoration mode (speech/ambient denoising without vocoder synthesis)."""

from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseRestorationMode


class AutoPureMode(BaseRestorationMode):
    """4-Pass Cascaded Pure Restoration (*_Pure_Cleaned)."""

    mode_name = "auto_pure"
    display_name = "Pure Speech & Ambient"

    def execute(
        self,
        work_dir: Path,
        original_wav: Path,
        video_path: Path,
        final_output_video: Path,
        video_dur: float,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """4-Pass Cascaded Pure Restoration: Pre-Scan -> Pre-Conditioning -> Separation & Denoise -> Mix."""
        from .. import processing

        clean_wav, strategy = processing._resolve_preconditioned_audio(work_dir, original_wav, video_dur, self.mode_name, strategy)
        processing._execute_pure_restoration(
            work_dir, clean_wav, original_wav, video_path, final_output_video, video_dur, strategy=strategy
        )
