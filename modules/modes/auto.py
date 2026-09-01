"""Automatic acoustic-profile-driven restoration mode."""

from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseRestorationMode


class AutoMode(BaseRestorationMode):
    """Scan the source and dispatch the selected restoration strategy."""

    mode_name = "auto"
    display_name = "Automatic"

    def execute(
        self,
        work_dir: Path,
        original_wav: Path,
        video_path: Path,
        final_output_video: Path,
        video_dur: float,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Execute the automatic scanner and strategy-selected pipeline."""
        del strategy
        from .. import processing

        processing._process_auto_mode(work_dir, original_wav, video_path, final_output_video, video_dur)
