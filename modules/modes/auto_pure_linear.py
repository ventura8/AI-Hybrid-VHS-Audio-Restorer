"""Auto Pure Linear restoration mode.

Executes pure full-mix restoration without stem separation using pre-denoise surgical
bandreject notching, adaptive UVR-DeNoise neural inference, and linear air polish.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from ..config import APL_ENABLE_HUM_CANCEL, APL_EXPANDER_DEPTH_DB, APL_USE_DEEPFILTERNET, APL_USE_RESEMBLE_DENOISE
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
                in_wav,
                out_dir,
                total_duration=total_duration,
                denoise_model=denoise_model,
                strategy=strategy,
                apply_air=True,
                # Opted in here rather than inside the shared step, so denoise_only keeps
                # the behaviour it shipped with.
                spectral_denoise=True,
                physical_repair=True,
                deepfilternet=APL_USE_DEEPFILTERNET,
                # The mode's own stages read their own apl_enable_* switch; opting in here
                # is what lets a sweep of that switch reach the chain. The hum canceller's
                # switch is passed through, because the surgical filter leaves the mains
                # harmonics to the canceller only when the canceller will run.
                hum_cancel=APL_ENABLE_HUM_CANCEL,
                plosive_tamer=True,
                tone_cancel=True,
                resemble_denoise=APL_USE_RESEMBLE_DENOISE,
                # The mode's own expander depth (apl_expander_depth_db); denoise_only keeps the shared one.
                expander_depth_db=APL_EXPANDER_DEPTH_DB,
                # The listener-round stages read their own switches (apl_enable_sibilant_guard,
                # enable_pause_floor, both on since the listener round); opting in here lets a
                # sweep of those switches reach the chain.
                sibilant_guard=True,
                pause_floor=True,
            )

        from .. import apl_stems

        # Music (held partials at or above apl_music_persistence_min) takes the stem path when it is
        # switched on: the chain on the vocal stem, the music through the notches; a separator
        # failure falls back to the single-track path below.
        if apl_stems.wanted(strategy) and apl_stems.execute(
            work_dir, clean_wav, original_wav, video_path, final_output_video, video_dur, strategy, denoise_step
        ):
            return
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
