"""ARNNDN RNNoise recurrent neural network speech restoration mode."""

from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseRestorationMode


class ArnndnSpeechMode(BaseRestorationMode):
    """FFmpeg RNNoise recurrent neural network speech restoration (*_Speech_Cleaned)."""

    mode_name = "arnndn_speech"
    display_name = "ARNNDN Speech"

    def execute(
        self,
        work_dir: Path,
        original_wav: Path,
        video_path: Path,
        final_output_video: Path,
        video_dur: float,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """FFmpeg RNNoise recurrent neural network speech restoration."""
        from .. import processing

        clean_wav, strategy = processing._resolve_preconditioned_audio(work_dir, original_wav, video_dur, self.mode_name, strategy)
        step_func = processing._bind_step_model(
            processing._filter_arnndn_step, "model_name", self.resolve_strategy_val(strategy, "arnndn_model", None)
        )
        processing._process_single_track_pipeline(
            work_dir,
            clean_wav,
            video_path,
            final_output_video,
            video_dur,
            step_func,
            "arnndn_speech_audio",
            self.display_name,
            sync_method=self.resolve_strategy_val(strategy, "sync_method", None),
            ref_wav=original_wav,
        )
