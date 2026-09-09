"""The deterministic stages `auto_pure_linear` runs ahead of its neural denoiser.

Each stage is opted into by the mode, runs in a fixed order, and returns its input path
unchanged when it has nothing to do -- which is how the runner tells a stage that skipped
from one that changed the audio. The stage logs its own reason; the runner logs the
summary, and reports which stages changed the audio so the caller can follow it (the
deep separator partners the subtraction only when the subtraction actually ran).
"""

from . import hum_cancel as _hum_cancel
from . import physical_repair as _physical_repair
from . import plosive_tamer as _plosive_tamer
from . import spectral_denoise as _spectral_denoise
from . import tone_cancel as _tone_cancel
from .config import ADAPTIVE_DENOISE_THRESHOLD_DB, DEFAULT_DENOISE_MODEL, DENOISE_MODEL
from .utils import log_msg


def profile_noise_floor_db(strategy):
    """Returns the profiled noise floor in dB, or None when it is absent or unusable."""
    raw_nf = (strategy or {}).get("profile", {}).get("noise_floor_db")
    if raw_nf is None:
        return None
    try:
        return float(raw_nf)
    except (ValueError, TypeError):
        return None


def resolve_adaptive_denoise_model(strategy, default_model):
    """Picks the lighter UVR-DeNoise model on clean recordings to prevent over-processing."""
    nf_val = profile_noise_floor_db(strategy)
    if nf_val is None or nf_val >= ADAPTIVE_DENOISE_THRESHOLD_DB:
        return default_model
    effective_model = DENOISE_MODEL if default_model is None else default_model
    log_msg(
        f"    [Adaptive Denoise] Quiet source ({nf_val:.1f} dB); "
        f"overriding {effective_model} with {DEFAULT_DENOISE_MODEL} to preserve transients."
    )
    return DEFAULT_DENOISE_MODEL


def stage_plan(
    audio_dir, total_duration, strategy, physical_repair, spectral_denoise, hum_cancel=False, plosive_tamer=False, tone_cancel=False
):
    """The opted-in stages in chain order: (name, wanted, callable taking the current WAV).

    Physical repair runs ahead of subtraction, not after it. The noise profile is learned
    from the quietest stretch of the capture, and on a tape with dropouts that stretch is a
    dropout -- so an unrepaired hole would be learned as the noise floor and the subtraction
    would have nothing to remove. Hum cancellation sits between the two for the same reason:
    with the hum gone before the probe, the profile is hiss and the subtraction spends its
    floor on hiss.
    """
    return (
        (
            "physical_repair",
            physical_repair,
            lambda wav: _physical_repair.apply_when_needed(wav, audio_dir, strategy=strategy, total_duration=total_duration),
        ),
        ("hum_cancel", hum_cancel, lambda wav: _hum_cancel.apply_when_needed(wav, audio_dir, strategy=strategy)),
        ("tone_cancel", tone_cancel, lambda wav: _tone_cancel.apply_when_needed(wav, audio_dir, strategy=strategy)),
        ("plosive_tamer", plosive_tamer, lambda wav: _plosive_tamer.apply_when_needed(wav, audio_dir, strategy=strategy)),
        (
            "tonal_cleanup",
            spectral_denoise,
            lambda wav: _spectral_denoise.apply_tonal_cleanup(wav, audio_dir, strategy=strategy, total_duration=total_duration),
        ),
        (
            "spectral_denoise",
            spectral_denoise,
            lambda wav: _spectral_denoise.apply_when_needed(wav, audio_dir, total_duration=total_duration),
        ),
    )


def _names(outcomes, changed):
    """The stages whose outcome matches, or "none"."""
    return ", ".join(name for name, outcome in outcomes if outcome is changed) or "none"


def _log_outcomes(outcomes):
    """One line naming the stages that changed the audio and the ones that skipped."""
    log_msg(f"    [Chain] applied: {_names(outcomes, True)}; skipped: {_names(outcomes, False)}")


def run(source_wav, plan):
    """Runs each wanted stage in turn; returns the final path and the names of the stages that changed the audio."""
    current, outcomes = source_wav, []
    for name, wanted, stage in plan:
        if wanted:
            produced = stage(current)
            outcomes.append((name, produced != current))
            current = produced
    _log_outcomes(outcomes)
    return current, [name for name, changed in outcomes if changed]
