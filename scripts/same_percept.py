"""Same-percept transforms: changes no listener can hear, the tie bands of ear v4's hiss readings.

Ear v4 ranks the residual hiss one-sided with dead zone 0, so its tie band (the grids' `tie`)
is what decides which differences count. It must come from transforms that change nothing a
listener hears, never from Session 0's benign pairs, which differed grossly (residual noise
16.4 dB, noise removed 19.9 dB):

- `gain_db`: +-0.1 dB, under the 0.5-1 dB intensity JND;
- `fractional_delay`: 0.37 samples, an FFT phase ramp over the signal padded by 8192 samples
  each side (a delay no alignment can absorb exactly);
- `soxr_round_trip`: a round trip through another rate with `audio_io.resample` (librosa's
  soxr_hq). Not scipy's `resample_poly`: its images at -50..-60 dB land in the noise cells of
  a deep residual, which the hiss readings then count as hiss;
- `fresh_floor`: a fresh realisation of the stationary floor: per 30 s block (1 s overlaps,
  linear crossfade), an STFT at 1024 / 768, the envelope as a 5 x 5 mean of |Z|^2, and in the
  noise-led cells only (envelope at most 2x the block's per-bin P20 of it) Z replaced by
  sqrt(envelope) x the STFT of seeded white noise normalised to unit mean power per bin. Every
  other cell is kept. Exact in the pauses, approximate under the programme. A steady tone's
  envelope sits at its own P20, so hum is redrawn as noise too: the shipped outputs hold
  little of it, both engines remove it on purpose;
- `fresh_like`: the same realisation over every cell, for a noise track on its own (the
  calibration's `fresh_noise_linear`, where the noise is known apart from the programme).

`scripts/reward_noise_floor.py --transforms hiss` scores them on the shipped outputs; the p95
of their deviations, rounded up to 0.05, is each hiss entry's `tie`.
"""

import numpy as np
import scipy.ndimage
import scipy.signal

from scripts.restoration_quality import audio_io

STFT_FRAME = 1024
STFT_OVERLAP = 768
ENVELOPE_CELLS = 5
NOISE_LED_FACTOR = 2.0
NOISE_LED_PERCENTILE = 20.0
BLOCK_S = 30.0
OVERLAP_S = 1.0
DELAY_PAD = 8192
EPS = 1e-30


def gain_db(mono, db):
    """`mono` scaled by `db` decibels."""
    return np.asarray(mono, dtype=np.float64) * 10.0 ** (db / 20.0)


def fractional_delay(mono, delay):
    """`mono` delayed by `delay` samples (fractional) with an FFT phase ramp, cut back to its length."""
    padded = np.pad(np.asarray(mono, dtype=np.float64), (DELAY_PAD, DELAY_PAD))
    spectrum = np.fft.rfft(padded) * np.exp(-2j * np.pi * np.fft.rfftfreq(len(padded)) * delay)
    return np.fft.irfft(spectrum, len(padded))[DELAY_PAD:][: len(mono)]


def soxr_round_trip(mono, rate, via):
    """`mono` resampled to `via` and back with soxr (`audio_io.resample`), cut or padded to its length."""
    there = audio_io.resample(np.asarray(mono, dtype=np.float32), int(rate), int(via))
    back = np.asarray(audio_io.resample(there, int(via), int(rate)), dtype=np.float64)[: len(mono)]
    return np.pad(back, (0, len(mono) - len(back)))


def _stft(mono, rate):
    return scipy.signal.stft(mono, rate, nperseg=STFT_FRAME, noverlap=STFT_OVERLAP)[2]


def _istft(spectrum, rate, length):
    return scipy.signal.istft(spectrum, rate, nperseg=STFT_FRAME, noverlap=STFT_OVERLAP)[1][:length]


def envelope(spectrum):
    """The 5 x 5 (bins x frames) mean of |Z|^2: the stationary power each cell's noise is drawn around.

    Held at 0 or above: the running mean's rounding leaves tiny negative values beside digital silence (a NaN under the
    square root broke the first jitter run).
    """
    return np.maximum(scipy.ndimage.uniform_filter(np.abs(spectrum) ** 2, size=ENVELOPE_CELLS, mode="nearest"), 0.0)


def unit_white(length, rate, rng):
    """The STFT of `length` samples of white noise from `rng`, each bin normalised to unit mean power."""
    spectrum = _stft(rng.standard_normal(length), rate)
    return spectrum / np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=1, keepdims=True) + EPS)


def fresh_like(noise, rate, seed):
    """A fresh realisation of a noise track: every cell's draw replaced around the same envelope."""
    noise = np.asarray(noise, dtype=np.float64)
    spectrum = _stft(noise, rate)
    fresh = np.sqrt(envelope(spectrum)) * unit_white(len(noise), rate, np.random.default_rng(seed))
    return _istft(fresh, rate, len(noise))


def fresh_block(mono, rate, rng):
    """One block with its noise-led cells (envelope at most 2x its per-bin P20) redrawn around the envelope; the rest kept."""
    spectrum = _stft(np.asarray(mono, dtype=np.float64), rate)
    power = envelope(spectrum)
    led = power <= NOISE_LED_FACTOR * np.percentile(power, NOISE_LED_PERCENTILE, axis=1, keepdims=True)
    fresh = np.where(led, np.sqrt(power) * unit_white(len(mono), rate, rng), spectrum)
    return _istft(fresh, rate, len(mono))


def block_starts(length, size, overlap):
    """The starts of `size`-sample blocks overlapping by `overlap`; the last one reaches the end (a block steps size - overlap)."""
    return list(range(0, max(1, length - overlap), size - overlap))


def crossfade_weight(length, overlap, first, last):
    """A block's weight: linear ramps over `overlap` samples at each inner end, 1 elsewhere."""
    weight = np.ones(length)
    ramp = np.linspace(0.0, 1.0, min(overlap, length) + 2)[1:-1]
    if not first:
        weight[: len(ramp)] = ramp
    if not last:
        tail = length - len(ramp)
        weight[tail:] = ramp[::-1]
    return weight


def fresh_floor(mono, rate, seed):
    """A fresh realisation of `mono`'s stationary floor, per 30 s block with 1 s linear crossfades (module docstring)."""
    mono = np.asarray(mono, dtype=np.float64)
    size, overlap = int(BLOCK_S * rate), int(OVERLAP_S * rate)
    if len(mono) <= size:
        return fresh_block(mono, rate, np.random.default_rng(seed))
    out, total = np.zeros(len(mono)), np.zeros(len(mono))
    starts = block_starts(len(mono), size, overlap)
    for index, start in enumerate(starts):
        block = mono[start:][:size]
        weight = crossfade_weight(len(block), overlap, index == 0, index == len(starts) - 1)
        stop = start + len(block)
        out[start:stop] += fresh_block(block, rate, np.random.default_rng((seed, index))) * weight
        total[start:stop] += weight
    return out / np.maximum(total, EPS)
