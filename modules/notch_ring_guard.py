"""The ring guard for the pre-denoise surgical notches: a notch never hands on more energy than it was given.

`auto_pure_linear` notches the third to fifth mains harmonics ahead of its chain with Q-30
bandrejects (`filters.build_pre_denoise_surgical_filter`). A notch that narrow is a resonator
in reverse: in steady state it takes the line out, but when the audio feeding it stops, its
band-pass half keeps ringing at the notch frequency for tens of milliseconds at the level of
whatever sat in its band. Speech that holds a partial near 150, 200 or 250 Hz and then stops
dead -- a stop closure, a word cut short -- hands the gap that follows a tone burst instead of
silence. On the SOTI interview (its chain re-rendered on CPU bit for bit against the stage
cache of the shipped render, 2026-10-11) the notch lifted 332 of the 3,740 quietest 23 ms
frames by 10 dB or more at 100-1000 Hz, by up to +28 dB: 46 ms holes inside speech at 111.15 s,
100.98 s and 48.81 s, each filled with 237-258 Hz ringing at -27 dB where the pre-conditioned
audio sat at -50 to -57 dB. The finished output kept them: 323 of its 331 lifted frames are
the notch's (`cathar`'s, which has no such notch, lifts 4). The pop repair and `decrackle`
that follow the notch lift none of those frames.

The guard compares the notch's output with its input frame by frame (512 samples on a
256-sample hop) and caps each output frame at TOLERANCE_DB over its input's energy, the gain
interpolated between frame centres. Where the notch only removes energy the gain is exactly 1
and the samples pass through to the bit; where the output carries more than the input -- the
ringing tail in a gap -- it is brought down to the gap's own level. The first and last frame
of a gap keep part of the ringing, their gain interpolated towards the frames that straddle
the gap's edges: on a synthetic 46 ms gap after a 252 Hz partial the gap's interior returns
to within the tolerance of its own level, while the whole gap, edges included, sits only 9 dB
under the ringing (the gap itself is 61 dB under it). On SOTI the notch's output
then lifts 3 quiet frames instead of 332 (none on 10 s cuts around the three holes, where the
notch alone lifts 12, 19 and 24), its quiet frames sit 0.80 dB over the source's on average
instead of 5.83, and 77% of the samples are untouched. The cap is broadband, so in a gap whose
floor is hiss the ringing may still hold up to a fifth of the frame's energy and stand over
its own narrow band: at 200-300 Hz, against the notch's input, 127 of SOTI's 3,740 quietest
frames still rise 10 dB or more (491 without the guard, up to +24 dB against +40). On
Vaccin, whose quiet frames the notch does not lift, it caps 2.7% of the frames and no
reading moves. On the ten calibrated hum
fixtures (speech at 9 and 15 dB margins, five languages) the lines at 150-250 Hz stay as far
under the clean reference as the notch put them (-8.81 against -8.79 dB, -8.56 against -8.57)
and the error against the clean reference falls by 0.04 dB.

Off by default (`apl_surgical_ring_guard`): it changes the audio wherever a notch rang, so
it waits for a listening session. Like `apl_surgical_mains_notch` it governs the surgical
notch `denoise_only` shares.
"""

from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

from .config import APL_SURGICAL_RING_GUARD
from .hygiene import atomic_target
from .utils import is_valid_audio, log_msg

# Energy is read on 256-sample sub-frames; a frame is two of them (512 samples, 11.6 ms at
# 44.1 kHz) on a one-sub-frame hop, short enough to see a 46 ms gap whole.
SUBFRAME = 256
BLOCK_SAMPLES = SUBFRAME * 4096
# Energy floor in the ratio, far under any audio: a silent input frame with a ringing output
# gets a gain near zero, a frame silent on both sides keeps a gain of 1.
ENERGY_FLOOR = 1e-20
# How far a frame may rise over its input before the guard acts, in dB: a notch shifts the
# phase of what sits beside it, which moves a little energy across a frame's edge without
# ringing. At 0 dB the guard capped 40% of SOTI's frames and 27-28% of the hum fixtures',
# at 1 dB 18.6% and 3.7-3.8%, for 3 lifted quiet frames instead of 1; at 3 dB 12.
TOLERANCE_DB = 1.0
STAGE_FAILURES = (OSError, RuntimeError, ValueError, MemoryError)
PREFIX = "ringguarded_"


def subframe_energy(wav_path):
    """Sum of squares of every SUBFRAME of each channel, streamed: shape (sub-frames, channels); the last one may be short."""
    with sf.SoundFile(str(wav_path)) as handle:
        energies = []
        for block in handle.blocks(blocksize=BLOCK_SAMPLES, dtype="float32", always_2d=True):
            squared = block.astype(np.float64) ** 2
            pad = (-len(squared)) % SUBFRAME
            squared = np.pad(squared, ((0, pad), (0, 0)))
            energies.append(squared.reshape((-1, SUBFRAME, squared.shape[1])).sum(axis=1))
        return np.concatenate(energies), handle.frames, handle.channels


def frame_gains(source_energy, filtered_energy):
    """Per frame (two sub-frames), the gain that caps the filtered frame at the source frame's energy, never above 1."""
    source = source_energy[:-1] + source_energy[1:] if len(source_energy) > 1 else source_energy
    filtered = filtered_energy[:-1] + filtered_energy[1:] if len(filtered_energy) > 1 else filtered_energy
    allowed = 10.0 ** (TOLERANCE_DB / 10.0) * source + ENERGY_FLOOR
    return np.minimum(1.0, np.sqrt(allowed / (filtered + ENERGY_FLOOR)))


def sample_gains(gains, start, count):
    """Per-sample gains for samples [start, start + count), interpolated between frame centres (one SUBFRAME in)."""
    centres = (np.arange(len(gains)) + 1.0) * SUBFRAME
    positions = np.arange(start, start + count, dtype=np.float64)
    return np.column_stack([np.interp(positions, centres, gains[:, channel]) for channel in range(gains.shape[1])])


def write_guarded(filtered_wav, target_wav, gains):
    """Writes the filtered audio times its per-sample gains, streamed."""
    with (
        atomic_target(target_wav) as partial,
        sf.SoundFile(str(filtered_wav)) as filtered,
        sf.SoundFile(str(partial), "w", samplerate=filtered.samplerate, channels=filtered.channels, subtype="FLOAT") as out,
    ):
        for start in range(0, filtered.frames, BLOCK_SAMPLES):
            block = filtered.read(BLOCK_SAMPLES, dtype="float32", always_2d=True)
            out.write((block.astype(np.float64) * sample_gains(gains, start, len(block))).astype(np.float32))
    return Path(target_wav)


def guard(source_wav, filtered_wav, target_wav):
    """Caps every frame of `filtered_wav` at `source_wav`'s energy into `target_wav`; returns (target, capped frames, frames).

    Raises ValueError when the two files differ in length or channel count, since then the
    frames do not line up.
    """
    source_energy, source_frames, source_channels = subframe_energy(source_wav)
    filtered_energy, filtered_frames, filtered_channels = subframe_energy(filtered_wav)
    if (source_frames, source_channels) != (filtered_frames, filtered_channels):
        raise ValueError(f"the notch's output does not line up with its input ({filtered_frames} vs {source_frames} frames)")
    gains = frame_gains(source_energy, filtered_energy)
    capped = int(np.count_nonzero(np.any(gains < 1.0, axis=1)))
    return write_guarded(filtered_wav, target_wav, gains), capped, len(gains)


def _target_for(filtered_wav):
    """The guarded file beside the notch's output, named after it."""
    filtered_wav = Path(filtered_wav)
    return filtered_wav.with_name(f"{PREFIX}{filtered_wav.name}")


def _wanted(source_wav, filtered_wav, enabled):
    """Whether the guard runs: switched on, soundfile importable, and a notch that actually rendered a file of its own."""
    return bool(enabled) and sf is not None and Path(filtered_wav) != Path(source_wav)


def apply_when_enabled(source_wav, filtered_wav, enabled=None):
    """The notch's output with its ringing capped, when the guard is on and the notch ran; else `filtered_wav`.

    A failure leaves the notch's output as it was, logged at WARNING.
    """
    enabled = APL_SURGICAL_RING_GUARD if enabled is None else enabled
    if not _wanted(source_wav, filtered_wav, enabled):
        return filtered_wav
    target = _target_for(filtered_wav)
    if is_valid_audio(target):
        return target
    try:
        produced, capped, frames = guard(source_wav, filtered_wav, target)
    except STAGE_FAILURES as exc:
        log_msg(f"    [Surgical Pre-Denoise DSP] Ring guard skipped: {exc}", level="WARNING")
        return filtered_wav
    log_msg(
        f"    [Surgical Pre-Denoise DSP] Ring guard: capped {capped} of {frames} frames at {TOLERANCE_DB:g} dB over their input's energy."
    )
    return produced
