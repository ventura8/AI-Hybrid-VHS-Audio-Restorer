"""Hybrid 2-stem vocal/background separation and enhancement restoration mode."""

from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseRestorationMode


class HybridMode(BaseRestorationMode):
    """Full 2-stem vocal/background separation and enhancement (*_Hybrid_Cleaned)."""

    mode_name = "hybrid"
    display_name = "Hybrid 2-Stem"

    def execute(
        self,
        work_dir: Path,
        original_wav: Path,
        video_path: Path,
        final_output_video: Path,
        video_dur: float,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Executes 2-stem AI separation, speech enhancement, background denoising, sync, and mixing."""
        from .. import processing

        clean_wav, strategy = processing._resolve_preconditioned_audio(work_dir, original_wav, video_dur, self.mode_name, strategy)
        processing._execute_hybrid_restoration(
            work_dir, clean_wav, original_wav, video_path, final_output_video, video_dur, strategy=strategy
        )
