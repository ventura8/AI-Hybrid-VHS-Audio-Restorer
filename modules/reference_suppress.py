"""The reference-SNR suppressor as a stage: the restored audio gained by the source's SNR, for either engine.

`spectral_suppress.ReferenceSuppressor` reads the MMSE-LSA gain on the pre-conditioned source,
against the source's whole-file quantile print, and applies it to a restored file: the restoring
stage's mask and the source's SNR multiply, so it can only remove. Two keys reach it, both off:

- `apl_reference_suppress`, after `auto_pure_linear`'s neural stage and sibilant guard
  (`modules/post_neural_suppress.py`), at `apl_suppress_noise_bias`,
  `apl_suppress_gain_floor_db` and `apl_suppress_dd_alpha`;
- `cathar_reference_suppress`, on cathar's output before the polish (`modules/modes/cathar.py`),
  at the same estimator's shipped settings (`CATHAR_SETTINGS`).

`apl_reference_release_db` and `cathar_reference_release_db` (0, off) ease the gain toward 1
where the restored audio stands that far over the source's pause print, programme the restoring
stage kept (`spectral_suppress.RELEASE_SPAN_DB`, `spectral_suppress.pause_print`).

On CPU replay on the four Tata tapes with the per-frame pause fill shelved at 2 kHz by 30 dB
it took the hiss under speech to inaudible on both engines (`docs/configuration.md`,
"Reference-SNR Suppressor"). The output's name carries the settings, so a changed setting is
never reused stale; a source whose print learned the programme (`spectral_suppress.MIN_DEPTH_DB`)
or that cannot be read leaves the restored file as it came.
"""

import hashlib
from pathlib import Path

from . import spectral_suppress
from .stage_names import stage_output
from .utils import is_valid_audio, log_msg

STAGE = "reference_suppress"
# cathar's settings: the estimator's own (auto_pure_linear's apl_suppress_* defaults).
CATHAR_SETTINGS = {"noise_bias": 1.0, "gain_floor_db": -20.0, "dd_alpha": 0.96}


def with_release(settings, release_db):
    """`settings` plus the programme release (`spectral_suppress.RELEASE_SPAN_DB`) when it is on; as they are at 0, so
    an output made with the release off keeps its name and its bytes."""
    return {**settings, "release_db": float(release_db)} if release_db > 0.0 else dict(settings)


def is_output(path):
    """Whether `path` is this stage's output: the suppressor ran (`suppress` hands its input back when it could not)."""
    path = Path(path)
    return path.parent.name == STAGE and path.name.startswith("referenced_")


def target_for(restored_wav, audio_dir, settings):
    """Where the suppressed audio goes: its settings in the name."""
    text = ",".join(f"{key}={value:g}" for key, value in sorted(settings.items()))
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return stage_output(Path(audio_dir) / STAGE, f"referenced_{fingerprint}", restored_wav)


def suppress(source_wav, restored_wav, audio_dir, settings):
    """The restored audio gained by the source's SNR; the restored input when it could not run."""
    target = target_for(restored_wav, audio_dir, settings)
    if is_valid_audio(target):
        log_msg(f"    [Reference Suppress] Reusing {target.name}.")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    log_msg("    [Reference Suppress] MMSE-LSA gain read on the source's SNR, applied to the restored audio...")
    result = spectral_suppress.suppress_by_reference_or_none(source_wav, restored_wav, target, **settings)
    if result is None:
        log_msg("    [Reference Suppress] Skipped: the restored audio goes on as it came.")
        return Path(restored_wav)
    return result
