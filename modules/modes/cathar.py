"""Cathar DSP pure-Rust restoration mode tailored for VHS captures."""

from pathlib import Path
from typing import Any, Dict, Optional

from .. import reference_suppress
from ..config import CATHAR_REFERENCE_RELEASE_DB, CATHAR_REFERENCE_SUPPRESS
from .base import BaseRestorationMode


def reference_suppressed(source_wav, restored_wav, audio_dir):
    """cathar's output gained by the source's SNR when `cathar_reference_suppress` is on; as it came otherwise."""
    if not CATHAR_REFERENCE_SUPPRESS:
        return restored_wav
    settings = reference_suppress.with_release(reference_suppress.CATHAR_SETTINGS, CATHAR_REFERENCE_RELEASE_DB)
    return reference_suppress.suppress(source_wav, restored_wav, audio_dir, settings)


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
        from ..cathar import filter_cathar_vhs_pipeline, music_expander_depth_db

        clean_wav, strategy = processing._resolve_preconditioned_audio(work_dir, original_wav, video_dur, self.mode_name, strategy)

        def cathar_step(in_wav, out_dir, total_duration=None):
            from .. import pause_floor

            raw_restored = filter_cathar_vhs_pipeline(in_wav, out_dir, total_duration=total_duration, strategy=strategy)
            # The source's SNR on cathar's output (cathar_reference_suppress, off): it only removes.
            raw_restored = reference_suppressed(in_wav, raw_restored, out_dir)
            # The expander's depth follows the material: the shared depth on speech, the music profile's on music.
            polished = processing._polish_full_audio_step(
                raw_restored,
                out_dir,
                total_duration=total_duration,
                strategy=strategy,
                apply_air=False,
                depth_db=music_expander_depth_db(strategy),
            )
            # After the expander, so it cannot take the fill back out (enable_pause_floor).
            return pause_floor.apply_when_needed(in_wav, polished, out_dir, strategy=strategy)

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
