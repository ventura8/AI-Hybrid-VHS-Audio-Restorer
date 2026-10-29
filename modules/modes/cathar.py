"""Cathar DSP pure-Rust restoration mode tailored for VHS captures."""

from pathlib import Path
from typing import Any, Dict, Optional

from .. import reference_suppress
from ..config import (
    CATHAR_PAUSE_FLOOR_FILL_DB,
    CATHAR_PAUSE_FLOOR_HF_CUT_DB,
    CATHAR_REFERENCE_RELEASE_DB,
    CATHAR_REFERENCE_SUPPRESS,
    CATHAR_SUPPRESS_GAIN_FLOOR_DB,
    CATHAR_SUPPRESS_NOISE_BIAS,
)
from .base import BaseRestorationMode


def reference_suppressed(source_wav, restored_wav, audio_dir):
    """cathar's output gained by the source's SNR when `cathar_reference_suppress` is on (the default since 2026-10-25);
    as it came otherwise."""
    if not CATHAR_REFERENCE_SUPPRESS:
        return restored_wav
    strength = {"noise_bias": CATHAR_SUPPRESS_NOISE_BIAS, "gain_floor_db": CATHAR_SUPPRESS_GAIN_FLOOR_DB}
    settings = reference_suppress.with_release({**reference_suppress.CATHAR_SETTINGS, **strength}, CATHAR_REFERENCE_RELEASE_DB)
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
        from ..cathar import filter_cathar_vhs_pipeline, polish_depth_db, polish_timing_s

        clean_wav, strategy = processing._resolve_preconditioned_audio(work_dir, original_wav, video_dur, self.mode_name, strategy)

        def cathar_step(in_wav, out_dir, total_duration=None):
            from .. import pause_floor

            raw_restored = filter_cathar_vhs_pipeline(in_wav, out_dir, total_duration=total_duration, strategy=strategy)
            # The source's SNR on cathar's output (cathar_reference_suppress, on): it only removes.
            raw_restored = reference_suppressed(in_wav, raw_restored, out_dir)
            # The expander's depth follows the material (cathar's on speech, the music profile's on music); its timing is cathar's.
            polished = processing._polish_full_audio_step(
                raw_restored,
                out_dir,
                total_duration=total_duration,
                strategy=strategy,
                apply_air=False,
                depth_db=polish_depth_db(strategy),
                timing_s=polish_timing_s(),
            )
            # After the expander, so it cannot take the fill back out (enable_pause_floor); cathar's own fill depth and shelf cut.
            suppressed = reference_suppress.is_output(raw_restored)
            return pause_floor.apply_when_needed(
                in_wav,
                polished,
                out_dir,
                strategy=strategy,
                suppressed=suppressed,
                cut_db=CATHAR_PAUSE_FLOOR_HF_CUT_DB,
                fill_db=CATHAR_PAUSE_FLOOR_FILL_DB,
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
