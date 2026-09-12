"""Physical tape damage repair for the modes that opt in.

`auto_pure_linear` removed noise well and repaired no physical damage at all: clicks,
dropouts, saturation and azimuth skew all survived it. Closing that by running the stages
`cathar` runs was the one approach already known to fail here, because the deterministic
tonal cleanup was adopted that way and measured 6.7-10.7 dB *below* a clean reference.

So each stage was measured on its own against paired fixtures, on the defect it targets and
on material carrying none of it. Four earned their place and three did not:

    defect     stage       sample dB   spectral dB   verdict
    clicks     decrackle       +10.00        +64.35   adopted
    pops       depop            +5.50   (pop spans)   adopted on the calibrated set, see impulse_repair
    dropout    inpaint          -2.00       +115.71   adopted
    clip       declip          +10.03         +0.98   adopted
    azimuth    azimuth         +32.42         +6.50   adopted
    clicks     declick          +1.02         +0.00   rejected, decrackle dominates it
    spikes     repair           -8.26        +68.74   rejected, wrecks undamaged audio
    plosives   deplosive        -0.71         -0.05   rejected, damages undamaged audio

`repair` and `deplosive` are rejected on the controls rather than on their targets: measured
on fixtures carrying no physical damage at all, `repair` scores -15.89 dB and lifts injected
error to -4.68 dB against the programme, and `deplosive` -5.60 and -15.61. Both make clean
material worse, which is the same failure as the tonal cleanup.

Two measurement notes worth keeping, because both inverted a verdict:

- A sample-difference metric ranks `inpaint` *below doing nothing*. Autoregressive gap
  filling restores the right signal but not the same samples, and filling a gap with
  uncorrelated audio of the correct energy scores about 3 dB worse than leaving it silent.
  The spectral reading is the valid one for a stage that reconstructs rather than subtracts.
- Every stage is gated on its own defect because applying them blanket-fashion is harmful,
  not merely wasteful: `decrackle` scores -11.09 dB on material whose only defect is azimuth
  skew.

`declip` is the one stage with a real runtime cost, and gating is what makes it affordable.
Measured on an 8 s fixture, `decrackle`, `azimuth` and `inpaint` take 0.02-0.03 s between
them while `declip` takes 5.29 s -- roughly 0.66x the audio duration, so it about doubles a
restoration whenever it runs. It only runs on material the clipping detector flags, and
across 25 real captures it flagged none, so the cost lands on genuinely saturated tape and
nowhere else. The hardware-validation fixtures do trigger it, which is why the real-time
factors recorded for those fixtures are a worst case rather than a typical one.
"""

import functools
from pathlib import Path

import numpy as np

from .config import (
    APL_DEPOP_THRESHOLD,
    APL_ENABLE_PHYSICAL_REPAIR,
    APL_MUTE_MAX_MS,
    APL_MUTE_MIN_MS,
    APL_MUTE_SILENCE_DB,
)
from .filters import _read_audio_for_analysis
from .utils import log_msg

# Inter-channel skew worth correcting. Below this the correction moves the image by less
# than the detector's own resolution.
AZIMUTH_MIN_MS = 0.05
# What a stage can fail with without that being the restoration's failure: the CLI absent or
# an output it cannot write (OSError), a Cathar step that reports its own error or an audio
# file the wrapper rejects (RuntimeError), a capture a stage's arithmetic refuses (ValueError),
# a capture the host cannot hold for a stage that works in memory (MemoryError).
STAGE_FAILURES = (OSError, RuntimeError, ValueError, MemoryError)


def _profile_value(strategy, key, default):
    """Reads one value from a strategy profile, defensively.

    Strategy shapes vary across callers and a missing profile must not decide a stage.
    """
    if not isinstance(strategy, dict):
        return default
    profile = strategy.get("profile")
    if not isinstance(profile, dict):
        return default
    return _typed_like(profile.get(key, default), default)


def _typed_like(value, default):
    """`value` when it is the default's kind of thing, else the default.

    A float setting accepts an int, since a profile may carry 1 for 1.0; a bool is never a
    number here, since True would otherwise read as a one-millisecond skew.
    """
    if isinstance(value, bool) and not isinstance(default, bool):
        return default
    accepted = (int, float) if isinstance(default, float) else type(default)
    return value if isinstance(value, accepted) else default


def _quiet_runs(quiet):
    """Returns the contiguous quiet spans as (start, stop) pairs.

    Found from the mask's boundaries rather than by walking the signal: a tape is millions
    of samples and this runs before every restoration in the mode.
    """
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    bounds = np.concatenate(([0], edges + 1, [len(quiet)]))
    return [(start, stop) for start, stop in zip(bounds[:-1], bounds[1:]) if quiet[start]]


def _is_dropout(mono, span, limits, guard, loud):
    """Whether one quiet span is a hole punched through running audio.

    Depth and length alone cannot tell a dropout from a speech pause: synthesised speech
    carries genuine digital silence between phrases, and a depth-only test found fourteen of
    them in every clean reference. A dropout is bounded by programme on both sides, where a
    pause is approached and left through a decay -- so what separates them is loud audio
    immediately either side.
    """
    start, stop = span
    minimum, maximum = limits
    if not minimum <= stop - start <= maximum:
        return False
    lead = max(start - guard, 0)
    tail = stop + guard
    before = np.abs(mono[lead:start])
    after = np.abs(mono[stop:tail])
    return bool(before.size and after.size and before.max() > loud and after.max() > loud)


def _analysis_mono(wav_path):
    """Mono samples and rate, or (None, 0) when the audio cannot be measured."""
    samples, rate = _read_audio_for_analysis(wav_path)
    if samples is None or rate <= 0 or len(samples) < rate // 10:
        return None, 0
    return (samples.mean(axis=1) if samples.ndim > 1 else samples), rate


def _count_dropouts(mono, quiet, limits, guard, loud):
    """Counts the quiet runs that qualify as dropouts."""
    return sum(1 for span in _quiet_runs(quiet) if _is_dropout(mono, span, limits, guard, loud))


def _span_limits(rate, min_ms):
    """Sample bounds a quiet run must fall between to be a dropout, plus the edge guard."""
    minimum = max(int(rate * min_ms / 1000.0), 1)
    maximum = max(int(rate * APL_MUTE_MAX_MS / 1000.0), minimum + 1)
    return (minimum, maximum), max(int(rate * 0.01), 1)


def detect_mute_spans(wav_path, silence_db=None, min_ms=None):
    """Counts near-silent spans that look like dropouts rather than pauses.

    Measured about the signal's median rather than zero: a digitiser's DC bias puts a
    dropout at the bias level, not at silence, and on a worn-tape fixture carrying both
    faults every dropout went uncounted until the bias was taken out first.
    """
    mono, rate = _analysis_mono(wav_path)
    if mono is None:
        return 0
    mono = mono - float(np.median(mono))
    depth = APL_MUTE_SILENCE_DB if silence_db is None else silence_db
    reference = float(np.sqrt(np.mean(mono**2))) + 1e-12
    quiet = np.abs(mono) < reference * (10.0 ** (depth / 20.0))
    limits, guard = _span_limits(rate, APL_MUTE_MIN_MS if min_ms is None else min_ms)
    return _count_dropouts(mono, quiet, limits, guard, reference * 0.1)


def _stage_plan(source_wav, strategy):
    """Decides which repair stages this capture actually needs."""
    clicks = bool(_profile_value(strategy, "has_clicks", False))
    return {
        "depop": clicks and APL_DEPOP_THRESHOLD > 0.0,
        "decrackle": clicks,
        "declip": bool(_profile_value(strategy, "has_clipping", False)),
        "azimuth": abs(float(_profile_value(strategy, "azimuth_delay_ms", 0.0))) >= AZIMUTH_MIN_MS,
        "inpaint": detect_mute_spans(source_wav) > 0,
    }


def _steps():
    """The adopted stages, imported lazily so the CLI wrapper stays an optional dependency."""
    from . import cathar, impulse_repair

    return {
        "depop": functools.partial(impulse_repair.depop, threshold=APL_DEPOP_THRESHOLD),
        "decrackle": cathar._cathar_decrackle_step,
        "declip": cathar._cathar_declip_step,
        "azimuth": cathar._cathar_azimuth_step,
        "inpaint": cathar._cathar_inpaint_step,
    }


def _run_one(step, name, current, output_dir, total_duration):
    """Runs one stage, returning its output or the unchanged input on any failure."""
    try:
        produced = step(current, output_dir, total_duration=total_duration)
    except STAGE_FAILURES as exc:
        log_msg(f"    [Repair] {name} bypassed: {exc}")
        return current
    if produced and Path(produced).is_file():
        log_msg(f"    [Repair] {name} applied.")
        return Path(produced)
    return current


def _run_stages(current, output_dir, plan, total_duration):
    """Applies the selected stages in turn, keeping the last good result."""
    steps = _steps()
    for name, wanted in plan.items():
        if wanted:
            current = _run_one(steps[name], name, current, output_dir, total_duration)
    return current


def _plan_summary(plan):
    """Names the stages this capture will run, for the log."""
    return ", ".join(name for name, wanted in plan.items() if wanted)


def _cli_available():
    """Whether the Cathar CLI can run here; a host without it degrades to no repair."""
    from . import cathar

    try:
        cathar._require_cathar_binary()
        return True
    except OSError as exc:
        log_msg(f"    [Repair] Skipped, Cathar CLI unavailable: {exc}")
        return False


def apply_when_needed(source_wav, audio_dir, strategy=None, total_duration=None):
    """Repairs physical tape damage, running only the stages the capture needs.

    Returns the source untouched when nothing is detected, when the stage is switched off,
    or when the Cathar CLI is unavailable, so a host that never provisioned it can still run
    the mode.
    """
    if not APL_ENABLE_PHYSICAL_REPAIR or not _cli_available():
        return source_wav
    source_wav = Path(source_wav)
    plan = _stage_plan(source_wav, strategy)
    if not any(plan.values()):
        return source_wav

    output_dir = Path(audio_dir) / "physical_repair"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_msg(f"    [Repair] Damage detected: {_plan_summary(plan)}.")
    return _run_stages(source_wav, output_dir, plan, total_duration)
