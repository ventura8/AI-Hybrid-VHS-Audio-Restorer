"""Event-gated plosive control for the full-mix chain.

cathar's `deplosive` was measured and rejected for `auto_pure_linear`: it scores -5.60 dB on
fixtures carrying no plosives at all, because it works on the whole file and a whole-file
process pays on every frame. This one finds the events and touches nothing else.

A plosive is an air blast on the microphone: a burst under 150 Hz with a fast attack, tens
of milliseconds long, well above the low band's own running level and above the mid band
that carries the consonant. Each is found on those terms -- excess over the low band's
running median, low band over mid band, the peak reached within a few hops, a length a
plosive has, isolation from its neighbours, and a density that rules out a kick drum -- and
repaired as a downward expander on the low band alone: the low-passed signal is taken out
in the proportion by which it exceeds the level it had just before the event, so a male
fundamental under the blast keeps its level where a high-pass would take it, and outside
the events the output is the input to the bit. Tonal material skips the stage outright:
bass is what its low band is made of.
"""

from pathlib import Path

import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf

from .config import APL_ENABLE_PLOSIVE_TAMER, APL_PLOSIVE_EXCESS_DB, APL_TONAL_FLATNESS_MAX
from .impulse_repair import _spans
from .spectral_denoise import estimate_tonality
from .utils import log_msg

HOP = 256
LEVEL_FLOOR = 1e-12
CROSSOVER_HZ = 150.0
MID_BAND_HZ = (300.0, 3400.0)
# The low band's running level is read over this long, so a burst stands out against
# what the band held just before it rather than against the whole recording.
BASELINE_S = 1.0
# A plosive reaches its peak within this many hops of its start (17 ms at 44.1 kHz), lasts
# between these bounds, and stands alone: two inside the isolation span are something else.
ATTACK_HOPS = 3
# The low band's rise over its baseline must lead the mid band's by this much: a blast is
# in the low band, a voice onset is in both.
LOW_BAND_LEAD_DB = 6.0
MIN_MS = 10.0
MAX_MS = 120.0
ISOLATION_MS = 250.0
# More events than this in a ten-second window is rhythm -- a kick drum is twenty -- and
# the whole window is left alone.
DENSITY_WINDOW_S = 10.0
DENSITY_MAX = 6
RAMP_MS = 5.0
# Blocks are read with this much either side so the zero-phase low-pass never lets its
# transient reach the samples written out.
PAD_S = 0.5
BLOCK_SAMPLES = 1 << 20
STAGE_FAILURES = (OSError, RuntimeError, ValueError, MemoryError)


def _band_filters(rate):
    """The low band's low-pass and the mid band's band-pass, fourth order each, as sections.

    Fourth order rather than second so a blast's skirt above the crossover does not read
    as mid-band energy of its own: the lead of the low band over the mid band is the
    detector's discriminator, and the analysis bands have to keep the two apart.
    """
    low = scipy.signal.butter(4, CROSSOVER_HZ, btype="low", fs=rate, output="sos")
    mid = scipy.signal.butter(4, MID_BAND_HZ, btype="band", fs=rate, output="sos")
    return low, mid


def _hop_levels_db(samples):
    """The RMS of every whole hop of a signal, in dB, floored at -120 dB so digital silence reads as a floor and not as a void."""
    count = len(samples) // HOP
    hops = samples[: count * HOP].reshape(count, HOP)
    return 10.0 * np.log10(np.mean(hops**2, axis=1) + LEVEL_FLOOR)


def band_levels(source_wav):
    """Per-hop levels of the low band and the mid band across the recording, in dB, and the sample rate.

    Streamed with the filters' state carried across blocks; the mono downmix is what is read.
    """
    with sf.SoundFile(str(source_wav)) as handle:
        low, mid = _band_filters(handle.samplerate)
        low_state, mid_state = scipy.signal.sosfilt_zi(low) * 0.0, scipy.signal.sosfilt_zi(mid) * 0.0
        low_db, mid_db, carry = [], [], np.zeros(0)
        for block in handle.blocks(blocksize=BLOCK_SAMPLES, dtype="float32", always_2d=True):
            mono = np.concatenate((carry, block.mean(axis=1).astype(np.float64)))
            usable = (len(mono) // HOP) * HOP
            low_part, low_state = scipy.signal.sosfilt(low, mono[:usable], zi=low_state)
            mid_part, mid_state = scipy.signal.sosfilt(mid, mono[:usable], zi=mid_state)
            low_db.append(_hop_levels_db(low_part))
            mid_db.append(_hop_levels_db(mid_part))
            carry = mono[usable:]
        return np.concatenate(low_db), np.concatenate(mid_db), handle.samplerate


def baseline_db(low_db, rate):
    """The low band's running median level, over the baseline span."""
    size = max(3, int(round(BASELINE_S * rate / HOP)) | 1)
    return scipy.ndimage.median_filter(low_db, size=size, mode="nearest")


def _qualifies(low_rise, start, stop, rate):
    """Whether a run of candidate hops has a plosive's attack and length."""
    length_ms = (stop - start) * HOP * 1000.0 / rate
    if not MIN_MS <= length_ms <= MAX_MS:
        return False
    return int(np.argmax(low_rise[start:stop])) <= ATTACK_HOPS


def _isolated(events, rate):
    """The events with no other event inside the isolation span either side."""
    gap = ISOLATION_MS * rate / 1000.0
    starts = np.array([start for start, _stop in events], dtype=np.float64)
    stops = np.array([stop for _start, stop in events], dtype=np.float64)
    gaps = starts[1:] - stops[:-1]
    near = (np.concatenate(([np.inf], gaps)) < gap) | (np.concatenate((gaps, [np.inf])) < gap)
    return [event for event, crowded in zip(events, near) if not crowded]


def _unrhythmic(events, rate):
    """The events that are not part of a run dense enough to be rhythm."""
    half = DENSITY_WINDOW_S * rate / 2.0
    starts = np.array([start for start, _stop in events], dtype=np.float64)
    kept = []
    for start, stop in events:
        around = int(np.sum(np.abs(starts - start) <= half))
        if around <= DENSITY_MAX:
            kept.append((start, stop))
    return kept


def _candidate_events(low_db, mid_db, baseline, excess_db, rate):
    """Runs of hops where the low band stands over its baseline and leads the mid band's own rise.

    The lead is what tells a blast from the onset of a low voice, and what bounds the
    event: a voice onset lifts both bands together and reads no lead, a plosive lifts the
    low band alone for as long as it lasts, and a thump lasts too long to be one.
    """
    low_rise = low_db - baseline
    lead = low_rise - (mid_db - baseline_db(mid_db, rate))
    candidates = (low_rise >= excess_db) & (lead >= LOW_BAND_LEAD_DB)
    runs = [(start, stop) for start, stop in _spans(candidates) if _qualifies(low_rise, start, stop, rate)]
    return [(start * HOP, _released(low_rise, start, stop, excess_db, rate) * HOP) for start, stop in runs]


def _released(low_rise, start, stop, excess_db, rate):
    """Where an event ends: the run extended through the blast's decaying tail, while the low band stays half the excess up.

    A blast peaks at once and decays over tens of milliseconds; the run that triggered it
    covers the peak, and the tail it leaves is still the blast.
    """
    limit = min(len(low_rise), start + int(MAX_MS * rate / 1000.0 / HOP))
    while stop < limit and low_rise[stop] >= excess_db / 2.0:
        stop += 1
    return stop


def detect_events(source_wav, excess_db=APL_PLOSIVE_EXCESS_DB):
    """The plosive events of a recording as (start, stop) sample spans, and the per-hop low-band levels.

    Returns (events, low_db, baseline_db, rate). A threshold of zero finds nothing.
    """
    low_db, mid_db, rate = band_levels(source_wav)
    baseline = baseline_db(low_db, rate)
    if excess_db <= 0.0:
        return [], low_db, baseline, rate
    events = _candidate_events(low_db, mid_db, baseline, excess_db, rate)
    return _unrhythmic(_isolated(events, rate), rate), low_db, baseline, rate


def gain_curve(events, low_db, baseline, length, rate):
    """Per-sample gain on the low band: one outside the events, the excess taken out inside, ramped at the edges."""
    hop_gain = np.ones(len(low_db), dtype=np.float64)
    for start, stop in events:
        first, last = start // HOP, stop // HOP
        target = baseline[first]
        hop_gain[first:last] = np.minimum(1.0, 10.0 ** ((target - low_db[first:last]) / 20.0))
    centres = (np.arange(len(low_db)) + 0.5) * HOP
    gain = np.interp(np.arange(length), centres, hop_gain)
    ramp = max(int(RAMP_MS * rate / 1000.0), 1)
    kernel = np.hanning(ramp + 2)[1:-1]
    padded = np.pad(gain, ramp, mode="edge")
    inner = slice(ramp, ramp + length)
    smoothed = np.convolve(padded, kernel / kernel.sum(), mode="same")[inner]
    # The kernel's weights sum to one only to rounding; snapped, the gain is exactly one
    # away from the events and the output there is the input to the bit.
    smoothed[np.abs(smoothed - 1.0) < 1e-9] = 1.0
    return smoothed


def _low_band(samples, rate):
    """The low band of a block, zero-phase, per channel."""
    low, _mid = _band_filters(rate)
    return scipy.signal.sosfiltfilt(low, samples, axis=0)


def tame_file(source_wav, target_wav, events, low_db, baseline):
    """Writes the recording with each event's low-band excess taken out; returns the target."""
    with (
        sf.SoundFile(str(source_wav)) as source,
        sf.SoundFile(str(target_wav), "w", samplerate=source.samplerate, channels=source.channels, subtype="FLOAT") as out,
    ):
        rate, length = source.samplerate, source.frames
        gain = gain_curve(events, low_db, baseline, length, rate)
        pad = int(PAD_S * rate)
        for start in range(0, length, BLOCK_SAMPLES):
            stop = min(start + BLOCK_SAMPLES, length)
            low, high = max(0, start - pad), min(length, stop + pad)
            source.seek(low)
            block = source.read(high - low, dtype="float32", always_2d=True).astype(np.float64)
            reduction = (1.0 - gain[start:stop])[:, None]
            inner = slice(start - low, stop - low)
            out.write((block[inner] - reduction * _low_band(block, rate)[inner]).astype(np.float32))
    return Path(target_wav)


def _skip_reason(source_wav):
    """Why the stage should not run on this recording, or None."""
    if not APL_ENABLE_PLOSIVE_TAMER or APL_PLOSIVE_EXCESS_DB <= 0.0:
        return "switched off"
    tonality = estimate_tonality(source_wav)
    if tonality is None:
        return "unreadable"
    if tonality < APL_TONAL_FLATNESS_MAX:
        return "tonal material"
    return None


def apply_when_needed(source_wav, audio_dir, strategy=None):
    """Tames the plosives a recording carries; returns the new path, or the input untouched.

    `strategy` is accepted for the chain's uniform stage signature and unused: the events
    are read from the recording.
    """
    del strategy
    reason = _skip_reason(source_wav)
    if reason is not None:
        if reason != "switched off":
            log_msg(f"    [Plosive Tamer] Skipped: {reason}.")
        return source_wav
    try:
        events, low_db, baseline, _rate = detect_events(source_wav)
        if not events:
            log_msg("    [Plosive Tamer] Skipped: no plosives found.")
            return source_wav
        output_dir = Path(audio_dir) / "plosive_tamer"
        output_dir.mkdir(parents=True, exist_ok=True)
        produced = tame_file(source_wav, output_dir / f"tamed_{Path(source_wav).name}", events, low_db, baseline)
    except STAGE_FAILURES as exc:
        log_msg(f"    [Plosive Tamer] Skipped after failure: {exc}")
        return source_wav
    log_msg(f"    [Plosive Tamer] Tamed {len(events)} plosives.")
    return produced
