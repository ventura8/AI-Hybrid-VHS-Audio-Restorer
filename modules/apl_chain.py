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
from .utils import log_msg


def stage_plan(audio_dir, total_duration, strategy, physical_repair, spectral_denoise, hum_cancel=False, plosive_tamer=False):
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
