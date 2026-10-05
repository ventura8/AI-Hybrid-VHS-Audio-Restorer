"""Event-gated sibilant guard for the full-mix chain.

The listener heard `auto_pure_linear` create distortion of a spoken 's': its neural denoise
stage takes the 1-4 kHz body from under each fricative and leaves a thin one. The harness
reads the same thing as `dsp.sib_centroid_hz` +370..+620 Hz against the reference on
Tele7abc, where cathar reads -100..-380 Hz and drew no objection.

The guard finds the fricative events on the audio the neural stage was given -- hops well
above the recording's floor whose energy sits mostly above the guard frequency and whose
zero-crossing rate is noise-like, in runs a fricative's length -- and inside those events
only puts a share of the reference's high band back into the restored audio. Outside the
events the output is the restored audio to the bit. Tonal material skips the stage: a
cymbal or a violin's upper partials read as fricatives on every one of those terms.
"""

from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from .config import APL_ENABLE_SIBILANT_GUARD, APL_SIBILANT_GUARD_HZ, APL_SIBILANT_MIX
from .hygiene import atomic_target
from .impulse_repair import _spans
from .utils import is_valid_audio, log_msg

HOP = 256
LEVEL_FLOOR = 1e-12
# A fricative lasts between these bounds; runs closer than the merge gap are one event.
MIN_MS = 20.0
MAX_MS = 400.0
MERGE_MS = 20.0
RAMP_MS = 5.0
# A hop qualifies when it stands this far over the recording's floor -- the floor being
# the level the quietest hops share -- with most of its energy above the guard frequency
# and a zero-crossing rate a voiced sound never reaches.
LEVEL_ABOVE_FLOOR_DB = 12.0
FLOOR_PERCENTILE = 15.0
HF_SHARE_MIN = 0.5
ZCR_MIN = 0.12
# Blocks are read with this much either side so the zero-phase high-pass never lets its
# transient reach the samples written out.
PAD_S = 0.5
BLOCK_SAMPLES = 1 << 20
STAGE_FAILURES = (OSError, RuntimeError, ValueError, MemoryError)


def _high_pass(guard_hz, rate):
    """The guard band's high-pass, fourth order, as sections."""
    return scipy.signal.butter(4, guard_hz, btype="high", fs=rate, output="sos")


def _hop_readings(mono, high):
    """Per-hop level in dB, share of energy above the guard frequency, and zero-crossing rate of whole hops."""
    count = len(mono) // HOP
    hops = mono[: count * HOP].reshape(count, HOP)
    high_hops = high[: count * HOP].reshape(count, HOP)
    energy = np.sum(hops**2, axis=1)
    level_db = 10.0 * np.log10(energy / HOP + LEVEL_FLOOR)
    hf_share = np.sum(high_hops**2, axis=1) / (energy + LEVEL_FLOOR)
    zcr = np.sum(np.diff(np.signbit(hops), axis=1), axis=1) / HOP
    return level_db, hf_share, zcr


def hop_features(reference_wav, guard_hz):
    """Per-hop level, high-band share and zero-crossing rate across the recording, and the sample rate.

    Streamed with the high-pass's state carried across blocks; the mono downmix is what is read.
    """
    with sf.SoundFile(str(reference_wav)) as handle:
        high = _high_pass(guard_hz, handle.samplerate)
        state = scipy.signal.sosfilt_zi(high) * 0.0
        levels, shares, rates, carry = [], [], [], np.zeros(0)
        for block in handle.blocks(blocksize=BLOCK_SAMPLES, dtype="float32", always_2d=True):
            mono = np.concatenate((carry, block.mean(axis=1).astype(np.float64)))
            usable = (len(mono) // HOP) * HOP
            high_part, state = scipy.signal.sosfilt(high, mono[:usable], zi=state)
            level_db, hf_share, zcr = _hop_readings(mono[:usable], high_part)
            levels.append(level_db)
            shares.append(hf_share)
            rates.append(zcr)
            carry = mono[usable:]
        return np.concatenate(levels), np.concatenate(shares), np.concatenate(rates), handle.samplerate


def _merged(spans, gap_hops):
    """Joins runs whose gap is under the merge span: a fricative's dip is not two fricatives."""
    merged = []
    for start, stop in spans:
        if merged and start - merged[-1][1] < gap_hops:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return merged


def detect_frames(level_db, hf_share, zcr, rate):
    """The fricative events of a set of hop readings as (start, stop) sample spans."""
    if level_db.size == 0:
        return []
    floor = np.percentile(level_db, FLOOR_PERCENTILE)
    candidates = (level_db >= floor + LEVEL_ABOVE_FLOOR_DB) & (hf_share >= HF_SHARE_MIN) & (zcr >= ZCR_MIN)
    hops_per_ms = rate / 1000.0 / HOP
    runs = _merged(_spans(candidates), MERGE_MS * hops_per_ms)
    kept = [(start, stop) for start, stop in runs if MIN_MS * hops_per_ms <= stop - start <= MAX_MS * hops_per_ms]
    return [(start * HOP, stop * HOP) for start, stop in kept]


def detect_events(reference_wav, guard_hz=APL_SIBILANT_GUARD_HZ):
    """The fricative events of a recording as (start, stop) sample spans, and the sample rate."""
    level_db, hf_share, zcr, rate = hop_features(reference_wav, guard_hz)
    return detect_frames(level_db, hf_share, zcr, rate), rate


def gain_curve(events, length, rate, mix):
    """Per-sample weight on the reference's high band: the mix inside the events, zero outside, ramped at the edges."""
    weight = np.zeros(length, dtype=np.float64)
    for start, stop in events:
        weight[start:stop] = mix
    ramp = max(int(RAMP_MS * rate / 1000.0), 1)
    kernel = np.hanning(ramp + 2)[1:-1]
    padded = np.pad(weight, ramp, mode="edge")
    inner = slice(ramp, ramp + length)
    smoothed = np.convolve(padded, kernel / kernel.sum(), mode="same")[inner]
    # The kernel's tails leave rounding dust; snapped, the weight is exactly zero away from
    # the events and the output there is the restored audio to the bit.
    smoothed[np.abs(smoothed) < 1e-9] = 0.0
    return smoothed


def _matched(reference, restored):
    """The rate, channel count and common length of two recordings that can be blended; raises when they cannot."""
    if reference.channels != restored.channels or reference.samplerate != restored.samplerate:
        raise ValueError(
            f"reference {reference.channels} ch @ {reference.samplerate} Hz "
            f"and restored {restored.channels} ch @ {restored.samplerate} Hz do not match"
        )
    return restored.samplerate, restored.channels, min(reference.frames, restored.frames)


def _padded_block(handle, low, high):
    """A block of a recording read at a position, as float64 channels in columns."""
    handle.seek(low)
    return handle.read(high - low, dtype="float32", always_2d=True).astype(np.float64)


def guard_file(reference_wav, restored_wav, target_wav, events, mix, guard_hz):
    """Writes the restored recording with the reference's high band blended back inside the events; returns the target.

    The file is published only once complete, so an interrupted run leaves no fragment.
    """
    with sf.SoundFile(str(reference_wav)) as reference, sf.SoundFile(str(restored_wav)) as restored:
        rate, channels, length = _matched(reference, restored)
        high = _high_pass(guard_hz, rate)
        weight = gain_curve(events, length, rate, mix)
        pad = int(PAD_S * rate)
        with (
            atomic_target(target_wav) as partial,
            sf.SoundFile(str(partial), "w", samplerate=rate, channels=channels, subtype="FLOAT") as out,
        ):
            for start in range(0, length, BLOCK_SAMPLES):
                stop = min(start + BLOCK_SAMPLES, length)
                low, top = max(0, start - pad), min(length, stop + pad)
                inner = slice(start - low, stop - low)
                source_high = scipy.signal.sosfiltfilt(high, _padded_block(reference, low, top), axis=0)[inner]
                block = _padded_block(restored, low, top)
                restored_high = scipy.signal.sosfiltfilt(high, block, axis=0)[inner]
                out.write((block[inner] + weight[start:stop, None] * (source_high - restored_high)).astype(np.float32))
    return Path(target_wav)


def _skip_reason(reference_wav):
    """Why the stage should not run on this recording, or None.

    No tonal skip, unlike the plosive tamer: the Tata tapes read tonal (flatness 0.022) and
    they are where the listener heard the 's' distorted; the fricative detector's own rules
    (high-band share, zero-crossing rate, length) keep a cymbal or a held note out.
    """
    del reference_wav
    if not APL_ENABLE_SIBILANT_GUARD or APL_SIBILANT_MIX <= 0.0:
        return "switched off"
    return None


def _guarded(reference_wav, restored_wav, audio_dir):
    """The guarded file and the number of events it covers; the restored input and zero when there is nothing to guard."""
    events, _rate = detect_events(reference_wav)
    if not events:
        return restored_wav, 0
    output_dir = Path(audio_dir) / "sibilant_guard"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"guarded_{Path(restored_wav).name}"
    if is_valid_audio(target):
        return target, len(events)
    return guard_file(reference_wav, restored_wav, target, events, APL_SIBILANT_MIX, APL_SIBILANT_GUARD_HZ), len(events)


def apply_when_needed(reference_wav, restored_wav, audio_dir, strategy=None):
    """Guards the fricatives of a restored recording against its reference; returns the new path, or the restored input untouched.

    `strategy` is accepted for the chain's uniform stage signature and unused: the events
    are read from the reference.
    """
    del strategy
    reason = _skip_reason(reference_wav)
    if reason is not None:
        if reason != "switched off":
            log_msg(f"    [Sibilant Guard] Skipped: {reason}.")
        return restored_wav
    try:
        produced, count = _guarded(reference_wav, restored_wav, audio_dir)
    except STAGE_FAILURES as exc:
        log_msg(f"    [Sibilant Guard] Skipped after failure: {exc}")
        return restored_wav
    if count == 0:
        log_msg("    [Sibilant Guard] Skipped: no sibilants found.")
        return restored_wav
    log_msg(f"    [Sibilant Guard] Guarded {count} sibilants.")
    return produced
