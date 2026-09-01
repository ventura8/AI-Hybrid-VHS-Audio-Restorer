"""MultiPass cascaded restoration mode."""

from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseRestorationMode


class MultiPassMode(BaseRestorationMode):
    """4-Pass Cascaded Restoration (*_MultiPass_Cleaned)."""

    mode_name = "multipass_auto"
    display_name = "MultiPass Cascaded"

    def execute(
        self,
        work_dir: Path,
        original_wav: Path,
        video_path: Path,
        final_output_video: Path,
        video_dur: float,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Executes 4-Pass Cascaded Restoration: Pre-Scan -> Pre-Conditioning -> AI Separation -> Polish & Sync."""
        from .. import processing

        clean_wav, strategy = processing._resolve_preconditioned_audio(work_dir, original_wav, video_dur, self.mode_name, strategy)
        processing._execute_hybrid_restoration(
            work_dir, clean_wav, original_wav, video_path, final_output_video, video_dur, strategy=strategy
        )
