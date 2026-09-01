"""Native FFmpeg DSP restoration modes (static and auto-tuned chains)."""

from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseRestorationMode


class FFmpegNativeMode(BaseRestorationMode):
    """FFmpeg native DSP filter chain restoration (*_FFmpeg_Cleaned)."""

    mode_name = "ffmpeg_native"
    display_name = "FFmpeg Native"

    def execute(
        self,
        work_dir: Path,
        original_wav: Path,
        video_path: Path,
        final_output_video: Path,
        video_dur: float,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """FFmpeg native DSP filter chain restoration."""
        from .. import processing

        processing._process_single_track_pipeline(
            work_dir,
            original_wav,
            video_path,
            final_output_video,
            video_dur,
            processing._filter_vhs_native_step,
            "ffmpeg_native_audio",
            self.display_name,
            sync_method=self.resolve_strategy_val(strategy, "sync_method", None),
        )


class AutoFFmpegNativeMode(BaseRestorationMode):
    """Intelligent adaptive FFmpeg native DSP restoration (*_AutoFFmpeg_Cleaned)."""

    mode_name = "auto_ffmpeg_native"
    display_name = "Auto-Tuned FFmpeg Native"

    def execute(
        self,
        work_dir: Path,
        original_wav: Path,
        video_path: Path,
        final_output_video: Path,
        video_dur: float,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Intelligent adaptive FFmpeg native DSP restoration with auto-tuned acoustic scan."""
        from .. import processing

        processing._process_single_track_pipeline(
            work_dir,
            original_wav,
            video_path,
            final_output_video,
            video_dur,
            processing._filter_auto_vhs_native_step,
            "auto_ffmpeg_native_audio",
            self.display_name,
            sync_method=self.resolve_strategy_val(strategy, "sync_method", None),
        )
