"""A complementary linear-phase two-band crossover that recombines two renderings of one file.

cathar denoises the same recording twice, once tuned for the band below the crossover and
once for the band above it, and the two renderings have to be put back together as one
file that is the original wherever both left it alone. A pair of independently designed
filters cannot promise that: their sum ripples through the transition and the seam is
audible. This crossover is complementary by construction -- the high-pass is a unit
impulse minus the low-pass, so the two sum to a pure delay -- and linear-phase, so the
delay is the same at every frequency and can be removed exactly. Recombining a file with
itself returns the file to floating-point rounding.
"""

from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from .hygiene import atomic_target
from .utils import is_valid_audio

# Odd, so the centre tap is a sample and the sum of the bands is a whole-sample delay.
TAPS = 511
BLOCK_SAMPLES = 1 << 20


def crossover(rate, hz, taps=TAPS):
    """The low-pass and high-pass kernels of the crossover at `hz`; their sum is a unit impulse at the centre tap.

    Raises ValueError when `hz` is not strictly between zero and Nyquist.
    """
    if not 0.0 < hz < rate / 2.0:
        raise ValueError(f"crossover must lie strictly between 0 and {rate / 2.0} Hz, got {hz}")
    h_lo = scipy.signal.firwin(taps, hz, fs=rate)
    h_hi = -h_lo
    h_hi[(taps - 1) // 2] += 1.0
    return h_lo, h_hi


def _check_matching(low, high):
    """Both renderings must share a rate and a channel count; their lengths may differ."""
    if low.samplerate != high.samplerate:
        raise ValueError(f"sample rates differ: {low.samplerate} vs {high.samplerate}")
    if low.channels != high.channels:
        raise ValueError(f"channel counts differ: {low.channels} vs {high.channels}")


def _band_sum(low, high, h_lo, h_hi):
    """Yields the filtered sum of the two renderings block by block, then the filters' flush.

    The filter state is carried per input and per channel, so the blocking leaves no trace
    in the output. The flush is `delay` zeros through both filters: it carries the tail
    the group delay holds back, so the output is as long as the input.
    """
    delay = (len(h_lo) - 1) // 2
    lo_state = np.zeros((len(h_lo) - 1, low.channels))
    hi_state = np.zeros((len(h_hi) - 1, high.channels))
    length = min(low.frames, high.frames)
    for start in range(0, length, BLOCK_SAMPLES):
        count = min(BLOCK_SAMPLES, length - start)
        lo_block = low.read(count, dtype="float64", always_2d=True)
        hi_block = high.read(count, dtype="float64", always_2d=True)
        lo_part, lo_state = scipy.signal.lfilter(h_lo, 1.0, lo_block, axis=0, zi=lo_state)
        hi_part, hi_state = scipy.signal.lfilter(h_hi, 1.0, hi_block, axis=0, zi=hi_state)
        yield lo_part + hi_part
    zeros = np.zeros((delay, low.channels))
    lo_tail, _lo = scipy.signal.lfilter(h_lo, 1.0, zeros, axis=0, zi=lo_state)
    hi_tail, _hi = scipy.signal.lfilter(h_hi, 1.0, zeros, axis=0, zi=hi_state)
    yield lo_tail + hi_tail


def _write_advanced(out, blocks, delay):
    """Writes the blocks with the first `delay` samples dropped, which takes the group delay out."""
    skip = delay
    for block in blocks:
        head = min(skip, len(block))
        out.write(block[head:].astype(np.float32))
        skip -= head


def recombine(low_wav, high_wav, target_wav, hz, taps=TAPS):
    """Writes the band below `hz` of `low_wav` plus the band above it of `high_wav` to `target_wav`; returns the target.

    The two renderings must share a rate and a channel count. Their lengths may differ
    by a few samples, as two denoise passes over one file can leave them: the shorter
    length is used and the output has exactly that length, the crossover's group delay
    removed. A target that already exists and reads as valid audio is returned as it is,
    the way every cathar step is cached. The file is published only once complete.
    """
    target = Path(target_wav)
    if is_valid_audio(target):
        return target
    with sf.SoundFile(str(low_wav)) as low, sf.SoundFile(str(high_wav)) as high:
        _check_matching(low, high)
        h_lo, h_hi = crossover(low.samplerate, hz, taps)
        with (
            atomic_target(target) as partial,
            sf.SoundFile(str(partial), "w", samplerate=low.samplerate, channels=low.channels, subtype="FLOAT") as out,
        ):
            _write_advanced(out, _band_sum(low, high, h_lo, h_hi), (taps - 1) // 2)
    return target
