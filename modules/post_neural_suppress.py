"""After the neural denoiser: a stationary-noise suppressor where it passed the hiss, and the air shelf's gate.

The Mel-RoFormer denoiser is the only stage of `auto_pure_linear` that removes hiss, and it is
nearly an identity on programme-dominated material (music, adverts: -1.2 dB under the
programme on 123 hunt clips) while it passes the floor on some hiss-dominated tapes. On the
diagnosis's hunt (2026-10-11, `modules/hiss_floor.py`) 10 of 78 hiss-dominated clips kept the
3-16 kHz floor within 6 dB of the source on the band median, and 43 kept it there in the
3-4.8 kHz bands' energy, where the ear is most sensitive and the RoFormer weakest.

`apl_post_neural_suppress` runs the mode's own MMSE log-spectral suppressor
(`modules/spectral_suppress.py`, decision-directed, its noise learned from the quietest
`apl_noiseprint_duration_s` of the post-neural audio, at `apl_suppress_noise_bias`,
`apl_suppress_gain_floor_db` and `apl_suppress_dd_alpha`) on the audio after the sibilant
guard, only when one reading of that audio against the pre-conditioned source passes both
halves of the gate:

- the source's 8-16 kHz band is hiss-dominated: its stationarity (median minus p10 of the
  frame energy) is at or under `apl_post_neural_suppress_floor_gap_db` (3 dB);
- the neural stage left a residual: the restored floor under the programme stands within
  `apl_post_neural_suppress_min_residual_db` (6 dB) of the source's, on the 3-16 kHz band
  median or on the 3-4.8 kHz bands' energy.

Ungated, the prototype (P2 on the 235-clip hunt) read like cathar on removal but ate music's
highs (agatha's loud-frame 8-16 kHz -2.0 to -14.2 dB) and left musical noise (lkr +0.42 at the
median, up to 1.0). On the 10 clips the prototype's band-median gate admitted (this gate opens
on 9 of them), followed by the prototype's per-frame shelved pause fill, the floor under the
programme went from -1.8 to -15.7 dB under the source's (medians) for loud-frame 8-16 kHz
from -0.9 to -3.5 dB and lkr from 0.01 to 0.39. Where the RoFormer emptied the floor the gate
stays shut and the stage costs one reading.

Followed by today's pause floor keeper instead, the stage makes the pauses worse: the floor's
absolute target sees a deficit once the suppressor has lowered the pauses' room tone and
refills them with the source's hiss. On the 48 replayed clips the gate opens on, the pause
hiss rose on 19 of the 27 with true pauses, by 29-81 dB on 7 (ending between -16 and +0.3 dB
against the source), and the floor under the programme rose on 6 (by up to 19 dB). Judge this
stage only with the per-frame pause fill.

`apl_reference_suppress` (2026-10-17) reads the same estimator's gain on the pre-conditioned
source and applies it to the post-neural audio (`spectral_suppress.ReferenceSuppressor`): the
neural stage's mask and the source's SNR multiply, so it can only remove. Its noise is the
source's whole-file quantile print, held at its level; the quietest 4 s stretch the gated
suppressor learns from holds speech on a speech tape (13-25 dB over the Tata tapes' true pauses
below 8 kHz). It needs no gate: where the neural stage already emptied a cell there is nothing
left to lower, and where the source holds programme the gain stays near one. With both keys on,
the gated suppressor runs first and this one on its output.

`linear_air_gate` 'hiss_dominated' skips `auto_pure_linear`'s air shelf on the same verdict:
the +1 dB shelf at 7.5 kHz lifted that residual 0.3-0.9 dB under the programme.

Every decision is logged with its numbers, and written to the event log
(`AI_RESTORE_EVENT_LOG`) as `post_neural_suppress__...json`. With both keys at their
defaults nothing is read and the audio is untouched.
"""

import hashlib
from pathlib import Path

from . import event_log, hiss_floor, spectral_suppress
from .config import (
    APL_NOISEPRINT_DURATION_S,
    APL_POST_NEURAL_SUPPRESS,
    APL_POST_NEURAL_SUPPRESS_FLOOR_GAP_DB,
    APL_POST_NEURAL_SUPPRESS_MIN_RESIDUAL_DB,
    APL_REFERENCE_SUPPRESS,
    APL_SUPPRESS_DD_ALPHA,
    APL_SUPPRESS_GAIN_FLOOR_DB,
    APL_SUPPRESS_NOISE_BIAS,
    ENABLE_LINEAR_AIR,
    LINEAR_AIR_GAIN_DB,
    LINEAR_AIR_GATE,
)
from .utils import is_valid_audio, log_msg

STAGE = "post_neural_suppress"
AIR_GATE_HISS = "hiss_dominated"
STAGE_FAILURES = (OSError, RuntimeError, ValueError, MemoryError)


def _settings():
    """The suppressor's settings: the mode's own suppressor keys and probe length."""
    return {
        "noise_bias": APL_SUPPRESS_NOISE_BIAS,
        "gain_floor_db": APL_SUPPRESS_GAIN_FLOOR_DB,
        "dd_alpha": APL_SUPPRESS_DD_ALPHA,
        "probe_s": APL_NOISEPRINT_DURATION_S,
    }


def _thresholds():
    """The gate's settings, carried by every decision record."""
    return {"floor_gap_db": APL_POST_NEURAL_SUPPRESS_FLOOR_GAP_DB, "min_residual_db": APL_POST_NEURAL_SUPPRESS_MIN_RESIDUAL_DB}


def _residual_db(reading):
    """The larger of the two residual readings (band median, presence energy), or None when neither could be read."""
    values = [value for value in (reading.get("median_delta_db"), reading.get("presence_delta_db")) if value is not None]
    return max(values, default=None)


def _hiss_dominated(reading):
    """The gate's first half: the source's 8-16 kHz stationarity at or under its threshold."""
    gap = reading.get("floor_gap_db")
    return gap is not None and gap <= APL_POST_NEURAL_SUPPRESS_FLOOR_GAP_DB


def _residual_left(reading):
    """The gate's second half: the restored floor within the residual threshold of the source's."""
    residual = _residual_db(reading)
    return residual is not None and residual > -APL_POST_NEURAL_SUPPRESS_MIN_RESIDUAL_DB


def _shut_because(reading):
    """Why the gate stays shut on a reading, or None when it opens."""
    if "reason" in reading:
        return f"no reading ({reading['reason']})"
    if not _hiss_dominated(reading):
        return "the source's 8-16 kHz band is not hiss-dominated"
    if not _residual_left(reading):
        return "the neural stage took the floor under the programme past the residual"
    return None


def _number(value):
    """A reading for the log line: one decimal, or a dash."""
    return "-" if value is None else f"{value:+.1f}"


def _summary(reading):
    """The gate's numbers as one log phrase."""
    if "reason" in reading:
        return reading["reason"]
    return (
        f"8-16 kHz floor gap {_number(reading.get('floor_gap_db'))} dB, residual under the programme"
        f" {_number(reading.get('median_delta_db'))} dB (band median), {_number(reading.get('presence_delta_db'))} dB (3-4.8 kHz)"
    )


def decide(source_wav, restored_wav):
    """The gate on one reading of the restored audio against the source: `{"open": bool, "why": str, "reading": dict}`."""
    try:
        reading = hiss_floor.read(source_wav, restored_wav)
    except STAGE_FAILURES as exc:
        reading = {"reason": f"failed: {exc}"}
    shut = _shut_because(reading)
    why = shut or "hiss-dominated source with a residual under the programme"
    log_msg(f"    [Post-Neural Hiss Gate] {'Open' if shut is None else 'Shut'}: {why}; {_summary(reading)}.")
    return {"open": shut is None, "why": why, "reading": reading}


def _target_for(restored_wav, audio_dir):
    """Where the suppressed audio goes: its settings in the name, so a changed setting is never reused stale."""
    settings = ",".join(f"{key}={value:g}" for key, value in _settings().items())
    fingerprint = hashlib.sha256(settings.encode("utf-8")).hexdigest()[:12]
    output_dir = Path(audio_dir) / STAGE
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"suppressed_{fingerprint}_{Path(restored_wav).name}"


def suppress(restored_wav, audio_dir):
    """Runs the suppressor on the post-neural audio; returns its output, or the input when it could not run."""
    target = _target_for(restored_wav, audio_dir)
    if is_valid_audio(target):
        log_msg(f"    [Post-Neural Suppress] Reusing {target.name}.")
        return target
    log_msg("    [Post-Neural Suppress] MMSE-LSA suppression of the hiss the neural stage passed...")
    result = spectral_suppress.suppress_or_none(restored_wav, target, **_settings())
    if result is None:
        log_msg("    [Post-Neural Suppress] Skipped: no quiet stretch to learn the noise from, or the suppressor failed.")
        return Path(restored_wav)
    return result


def _reference_settings():
    """The reference suppressor's settings: the mode's suppressor keys (it learns no probe)."""
    settings = _settings()
    del settings["probe_s"]
    return settings


def _reference_target(restored_wav, audio_dir):
    """Where the reference-suppressed audio goes: its settings in the name, so a changed setting is never reused stale."""
    settings = ",".join(f"{key}={value:g}" for key, value in _reference_settings().items())
    fingerprint = hashlib.sha256(settings.encode("utf-8")).hexdigest()[:12]
    return Path(audio_dir) / STAGE / f"referenced_{fingerprint}_{Path(restored_wav).name}"


def suppress_by_reference(source_wav, restored_wav, audio_dir):
    """The restored audio gained by the source's SNR (`apl_reference_suppress`); the input when it could not run."""
    target = _reference_target(restored_wav, audio_dir)
    if is_valid_audio(target):
        log_msg(f"    [Reference Suppress] Reusing {target.name}.")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    log_msg("    [Reference Suppress] MMSE-LSA gain read on the source's SNR, applied to the post-neural audio...")
    result = spectral_suppress.suppress_by_reference_or_none(source_wav, restored_wav, target, **_reference_settings())
    if result is None:
        log_msg("    [Reference Suppress] Skipped: the restored audio goes on as it came.")
        return Path(restored_wav)
    return result


def _record(restored_wav, verdict, suppressed, air, referenced=False):
    """The decision and its numbers in the event log, when the log is on."""
    record = {**_thresholds(), "open": verdict["open"], "why": verdict["why"], "suppressed": suppressed, "air_shelf": air}
    event_log.write(STAGE, restored_wav, {**record, "referenced": referenced, "reading": verdict["reading"]})


def _shelf_on():
    """Whether the polish builds an air shelf at all: switched on at a gain above zero, the rule `filters` applies."""
    return ENABLE_LINEAR_AIR and LINEAR_AIR_GAIN_DB > 0.0


def _wanted(stages):
    """`(suppress, air gate, reference)`: the stages the caller opted in to whose keys are on (the air gate only with a
    shelf to skip; the reference suppressor through the `post_neural_suppress` opt-in).

    A shelf the caller asked for but the configuration switched off (`enable_linear_air` false
    or `linear_air_gain_db` at 0) leaves the gate nothing to skip, so it reads nothing.
    """
    opted = bool(stages.get("post_neural_suppress"))
    return opted and APL_POST_NEURAL_SUPPRESS, _air_gate_wanted(stages), opted and APL_REFERENCE_SUPPRESS


def _air_gate_wanted(stages):
    """Whether the air gate reads: opted in, with a shelf the polish will build, on the 'hiss_dominated' key."""
    return bool(stages.get("air_gate") and stages.get("apply_air") and LINEAR_AIR_GATE == AIR_GATE_HISS and _shelf_on())


def _air_after(apply_air, gate_air, verdict):
    """Whether the polish keeps its air shelf after the gate."""
    gated = gate_air and verdict["open"]
    if gated:
        log_msg("    [Air Gate] Skipping the air shelf: it would lift the hiss the neural stage left.")
    return apply_air and not gated


def apply_when_needed(source_wav, restored_wav, audio_dir, stages):
    """`(audio, apply_air)` after the opted-in stages: the suppressor and the air gate, both decided on one reading.

    `source_wav` is the pre-conditioned audio the chain started from; `stages` the caller's
    post-neural switches (`apply_air`, `post_neural_suppress`, `air_gate`). A caller opts in
    to each stage; its key (`apl_post_neural_suppress`, `apl_reference_suppress`, which the
    `post_neural_suppress` opt-in also reaches, `linear_air_gate`) then decides. With none
    wanted, nothing is read and both values come back as they went in.
    """
    apply_air = bool(stages.get("apply_air", False))
    wanted = _wanted(stages)
    if not any(wanted):
        return restored_wav, apply_air
    wants_suppress, wants_air_gate, wants_reference = wanted
    verdict = decide(source_wav, restored_wav)
    gated = suppress(restored_wav, audio_dir) if wants_suppress and verdict["open"] else Path(restored_wav)
    output = suppress_by_reference(source_wav, gated, audio_dir) if wants_reference else gated
    air = _air_after(apply_air, wants_air_gate, verdict)
    _record(restored_wav, verdict, gated != Path(restored_wav), air, referenced=wants_reference)
    return output, air
