"""The pause floor keeper: a restored pause keeps its own, quieter texture instead of collapsing.

The listener called the `auto_pure_linear` outputs "silent in pauses". The harness read it as
the inter-word gaps' air falling 20 dB under the source's (`dsp.gap_air_db` -27 against -6)
and the deep pauses sinking 36-41 dB further under the speech than they were: the mask-based
denoiser leaves near-digital silence, and the polish expander pushes whatever is left a
further 7-10 dB down. Nothing in either engine put a floor back.

This stage does. In the frames the SOURCE calls quiet (at or under its `quiet_percentile`
level), an attenuated copy of the source is added under the restored audio so the pause sits
`fill_db` under the source's own pause level -- the tape's real texture, quieter, never
louder than the source, and never touched where the source carries programme. Only the
deficit is filled: where the restored audio already sits at or above the target, the gain is
zero and the samples pass through to the bit. The reference is the pre-conditioned audio the
engine started from, so the fill carries no hum or line whistle the notches already took.

With `AI_RESTORE_EVENT_LOG=<dir>` set, the stage writes a record into that folder
(`modules/event_log.py`), in seconds on the restored file's timeline, with the fill depth,
the quiet percentile and how many frames were filled or why none were. Two sets of spans:
`events_s` is the pauses found, the runs of 20 ms frames the stage calls quiet on the
source (the smoothed quiet weight over 0.5); `filled_s` is the frames the fill lifts (a
gain above zero), which can also take the frame just outside each pause and the frames
around a lone quiet one, where the three-frame smoothing leaves a weight of 1/3. The
per-sample gain reaches up to one frame (20 ms) past each end of a `filled_s` run (the
interpolation between frame centres and the 20 ms smoothing) and is exactly zero beyond.
Switched off, the record says so (`skipped`), and a stage that raised records
`failed: <error>`. Unset, nothing is written and the spans are not even computed.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from . import event_log
from .config import ENABLE_PAUSE_FLOOR, PAUSE_FLOOR_FILL_DB, PAUSE_FLOOR_QUIET_PERCENTILE
from .hygiene import atomic_target
from .impulse_repair import _spans
from .utils import is_valid_audio, log_msg

FRAME_S = 0.02
# A pause's frames sit within this many dB of the quietest ones; the speech 20 dB and more above.
QUIET_MARGIN_DB = 10.0
MAX_LAG_FRAMES = 10
MIN_QUIET_FRAMES = 50
MAX_LENGTH_DIFF_S = 0.5
MIN_DEFICIT_DB = 0.5
DEFICIT_SHARE = 0.95
SMOOTH_FRAMES = 3
BLOCK_SAMPLES = 1 << 20
STAGE_FAILURES = (OSError, RuntimeError, ValueError, MemoryError)
STAGE = "pause_floor"


def frame_levels(wav_path, frame):
    """RMS of every whole `frame`-sample frame of the channel-mean, streamed; and the file's rate, channels, length."""
    with sf.SoundFile(str(wav_path)) as handle:
        levels, carry = [], np.zeros(0)
        for block in handle.blocks(blocksize=BLOCK_SAMPLES, dtype="float32", always_2d=True):
            mono = np.concatenate((carry, block.mean(axis=1).astype(np.float64)))
            usable = (len(mono) // frame) * frame
            levels.append(np.sqrt(np.mean(mono[:usable].reshape(-1, frame) ** 2, axis=1)))
            carry = mono[usable:]
        return np.concatenate(levels), handle.samplerate, handle.channels, handle.frames


def envelope_lag(reference, restored):
    """Frames by which `restored` trails `reference`, from the envelopes' cross-correlation within +-MAX_LAG_FRAMES."""
    count = min(len(reference), len(restored))
    a = np.log10(reference[:count] + 1e-9)
    b = np.log10(restored[:count] + 1e-9)
    a, b = a - a.mean(), b - b.mean()
    lags = list(range(-MAX_LAG_FRAMES, MAX_LAG_FRAMES + 1))
    return lags[int(np.argmax([_overlap_score(a, b, lag, count) for lag in lags]))]


def _overlap_score(a, b, lag, count):
    """Dot product of the two envelopes with `b` shifted by `lag` frames, over their overlap."""
    a_start, b_start = max(0, -lag), max(0, lag)
    a_stop, b_stop = count - max(0, lag), count - max(0, -lag)
    return float(np.dot(a[a_start:a_stop], b[b_start:b_stop]))


def quiet_weight(reference, percentile):
    """1 on the frames within QUIET_MARGIN_DB of the source's quiet level, 0 elsewhere, smoothed over three frames.

    The quiet level is the `percentile` frame level; the margin takes the whole pause cluster
    with it (a pause's frames sit within a few dB of each other, the speech 20 dB above), so
    the fill covers pauses evenly instead of the scattered quietest frames alone.
    """
    quiet = (reference <= np.percentile(reference, percentile) * 10.0 ** (QUIET_MARGIN_DB / 20.0)).astype(np.float64)
    kernel = np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES
    return np.convolve(quiet, kernel, mode="same")


def fill_gains(reference, restored, weight, fill_db):
    """Per-frame gain on the source that lifts the restored pause to the target, never above the source itself."""
    floor = np.percentile(reference[weight > 0.5], 50.0) if (weight > 0.5).any() else 0.0
    target = floor * 10.0 ** (-fill_db / 20.0)
    deficit = np.sqrt(np.maximum(target**2 - restored**2, 0.0))
    return weight * np.clip(deficit / (reference + 1e-12), 0.0, 1.0)


def gain_curve(frame_gain, length, frame, rate):
    """Per-sample gain from the per-frame gains, 20 ms Hann-smoothed, snapped to exactly zero away from the fill."""
    centres = (np.arange(len(frame_gain)) + 0.5) * frame
    gain = np.interp(np.arange(length), centres, frame_gain)
    ramp = max(int(FRAME_S * rate), 1)
    kernel = np.hanning(ramp + 2)[1:-1]
    padded = np.pad(gain, ramp, mode="edge")
    inner = slice(ramp, ramp + length)
    smoothed = np.convolve(padded, kernel / kernel.sum(), mode="same")[inner]
    smoothed[np.abs(smoothed) < 1e-9] = 0.0
    return smoothed


def fill_file(reference_wav, restored_wav, target_wav, gain):
    """Writes `restored + gain * reference` over the overlap, the restored tail beyond it copied through."""
    with (
        atomic_target(target_wav) as partial,
        sf.SoundFile(str(reference_wav)) as reference,
        sf.SoundFile(str(restored_wav)) as restored,
        sf.SoundFile(str(partial), "w", samplerate=restored.samplerate, channels=restored.channels, subtype="FLOAT") as out,
    ):
        for start in range(0, restored.frames, BLOCK_SAMPLES):
            block = restored.read(BLOCK_SAMPLES, dtype="float32", always_2d=True)
            stop = start + len(block)
            fill = _reference_block(reference, start, stop, gain)
            out.write((block.astype(np.float64) + fill).astype(np.float32))
    return Path(target_wav)


def _reference_block(reference, start, stop, gain):
    """The gained reference samples for [start, stop), zero beyond the gain curve or the reference."""
    fill = np.zeros((stop - start, reference.channels), dtype=np.float64)
    usable = min(stop, len(gain), reference.frames) - start
    if usable <= 0:
        return fill
    reference.seek(start)
    block = reference.read(usable, dtype="float32", always_2d=True).astype(np.float64)
    lo, hi = start, start + usable
    fill[:usable] = block * gain[lo:hi][:, None]
    return fill


def _deficit_db(reference, restored, weight, fill_db):
    """How far the restored pauses sit under the target, in dB, on the quiet frames."""
    quiet = weight > 0.5
    if not quiet.any():
        return np.zeros(0)
    target = np.percentile(reference[quiet], 50.0) * 10.0 ** (-fill_db / 20.0)
    return 20.0 * np.log10((target + 1e-12) / (restored[quiet] + 1e-12))


def _shape_mismatch(reference_meta, restored_meta):
    """Why the two files cannot be paired, or None."""
    (rate, channels, frames), (rate2, channels2, frames2) = reference_meta, restored_meta
    if rate != rate2 or channels != channels2:
        return "reference and restored audio differ in rate or channels"
    if abs(frames - frames2) > MAX_LENGTH_DIFF_S * rate:
        return "reference and restored audio differ in length"
    return None


def _nothing_to_fill(reference, restored, weight):
    """Why the envelopes give the stage nothing to do, or None."""
    if int((weight > 0.5).sum()) < MIN_QUIET_FRAMES:
        return "too few quiet frames"
    deficit = _deficit_db(reference, restored, weight, PAUSE_FLOOR_FILL_DB)
    if float(np.mean(deficit < MIN_DEFICIT_DB)) >= DEFICIT_SHARE:
        return "nothing to fill"
    return None


def _aligned(reference, restored):
    """The two envelopes on one timeline, trimmed to the overlap after the lag is taken out."""
    lag = envelope_lag(reference, restored)
    ref_start, res_start = max(-lag, 0), max(lag, 0)
    count = min(len(reference) - ref_start, len(restored) - res_start)
    ref_stop, res_stop = ref_start + count, res_start + count
    return reference[ref_start:ref_stop], restored[res_start:res_stop], lag


def _target_for(restored_wav, audio_dir):
    output_dir = Path(audio_dir) / "pause_floor"
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{PAUSE_FLOOR_FILL_DB:g}db_p{PAUSE_FLOOR_QUIET_PERCENTILE:g}"
    return output_dir / f"paused_{tag}_{Path(restored_wav).name}"


def pause_spans(mask, lag):
    """The runs of a per-frame mask on the aligned envelopes, as frame spans on the restored file's timeline."""
    offset = max(lag, 0)
    return [(start + offset, stop + offset) for start, stop in _spans(mask)]


def _found(weight, gains, lag):
    """The frame spans the event log records: the pauses found and the frames filled; both empty while the log is off."""
    if not event_log.enabled():
        return [], []
    filled = [] if gains is None else pause_spans(gains > 0.0, lag)
    return pause_spans(weight > 0.5, lag), filled


def _restored_gain(reference, restored, frame, rate, frames):
    """The per-sample gain on the reference for the restored file's timeline, how many frames it fills, and the spans found.

    Without a gain the second value is why there is none. The spans are the pauses found
    and the frames filled, as frame spans on the restored timeline, read only while the
    event log is on.
    """
    ref_env, res_env, lag = _aligned(reference, restored)
    weight = quiet_weight(ref_env, PAUSE_FLOOR_QUIET_PERCENTILE)
    reason = _nothing_to_fill(ref_env, res_env, weight)
    if reason is not None:
        return None, reason, _found(weight, None, lag)
    gains = fill_gains(ref_env, res_env, weight, PAUSE_FLOOR_FILL_DB)
    frame_gain = np.concatenate((np.zeros(max(lag, 0)), gains))
    return gain_curve(frame_gain, frames, frame, rate), int((gains > 0).sum()), _found(weight, gains, lag)


def _thresholds():
    """The settings every record of the stage carries."""
    return {"fill_db": PAUSE_FLOOR_FILL_DB, "quiet_percentile": PAUSE_FLOOR_QUIET_PERCENTILE}


def _log_pauses(restored_wav, found, frame, rate, outcome):
    """Writes the pauses found and the frames filled, in seconds, into the event log when it is on.

    `outcome` is what the stage did: the number of frames filled, or the reason it filled
    none.
    """
    pauses, filled = found
    details = {**_thresholds(), "outcome": outcome, "filled_s": event_log.seconds(_in_samples(filled, frame), rate)}
    event_log.write_spans(STAGE, restored_wav, _in_samples(pauses, frame), rate, **details)


def _in_samples(spans, frame):
    """Frame spans as sample spans."""
    return [(start * frame, stop * frame) for start, stop in spans]


def keep_floor(reference_wav, restored_wav, audio_dir):
    """Runs the stage; returns the new path, or `restored_wav` with the reason logged."""
    frame = max(1, int(FRAME_S * sf.info(str(restored_wav)).samplerate))
    reference, *reference_meta = frame_levels(reference_wav, frame)
    restored, rate, channels, frames = frame_levels(restored_wav, frame)
    reason = _shape_mismatch(tuple(reference_meta), (rate, channels, frames))
    gain, filled, found = (None, reason, ([], [])) if reason else _restored_gain(reference, restored, frame, rate, frames)
    _log_pauses(restored_wav, found, frame, rate, filled)
    if gain is None:
        log_msg(f"    [Pause Floor] Skipped: {filled}.")
        return restored_wav
    target = _target_for(restored_wav, audio_dir)
    if is_valid_audio(target):
        log_msg(f"    [Pause Floor] Reusing {target.name}.")
        return target
    log_msg(f"    [Pause Floor] Filling {filled} quiet frames to {PAUSE_FLOOR_FILL_DB:g} dB under the source's pauses.")
    return fill_file(reference_wav, restored_wav, target, gain)


def apply_when_needed(reference_wav, restored_wav, audio_dir, strategy=None):
    """Keeps a floor under the restored pauses; returns the new path, or the restored audio untouched.

    `strategy` is accepted for the chain's uniform stage signature and unused.
    """
    del strategy
    if not ENABLE_PAUSE_FLOOR or PAUSE_FLOOR_FILL_DB <= 0.0:
        event_log.write_skip(STAGE, restored_wav, "switched off", **_thresholds())
        return restored_wav
    try:
        return keep_floor(reference_wav, restored_wav, audio_dir)
    except STAGE_FAILURES as exc:
        log_msg(f"    [Pause Floor] Skipped after failure: {exc}")
        event_log.write_skip(STAGE, restored_wav, f"failed: {exc}", **_thresholds())
        return restored_wav
