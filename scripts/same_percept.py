"""Same-percept transforms: changes no listener can hear, one set for the calibration and the hiss ties.

Every same-percept transform the harness uses lives here (the merge of ear v4 and the loop
judging, 2026-10-12, put the calibration's set beside the hiss ear's); three consumers read
different subsets, each named where it is declared:

- the calibration's sensitivity-check floors (`quality_degradations.SAME_PERCEPT`, on the
  speech target, every reading but the hiss ones): `fresh_noise` (`fresh_realisation` of the
  fixture's recorded noise, or two `floor_noise` draws on a real excerpt), `allpass` at 200 Hz
  and 2 kHz, `fractional_delay` by `FRACTIONAL_DELAY_SAMPLES`, `gain_db` by +-`GAIN_STEP_DB` and
  `tpdf_dither`. They floor the effect-versus-floor checks only, never a gate;
- the hiss readings' calibration floor (`quality_degradations.LINEAR_BENIGN`, on the linear
  capture, the `dsp.hiss_*` readings alone): identity, requantisation, `soxr_round_trip`,
  `gain_db` by +`GAIN_STEP_DB`, `fractional_delay` and `fresh_noise_linear` (an exact redraw of
  the capture's synthetic tape hiss, `degradations_hiss`);
- the hiss grids' ties (`reward_noise_floor.HISS_TRANSFORMS`, on the shipped outputs):
  `gain_db` by +-`GAIN_STEP_DB`, `fractional_delay`, `soxr_round_trip`, requantisation and
  `fresh_floor` from three seeds; the jitter floors the loop's other readings take
  (`--jitter-floors`) come from the reference repeat, `shift_1` and requantisation.

`fresh_like` stays for `check_hiss_fixtures`' fixture probe, whose recorded numbers it made. The
names follow what is redrawn: `fresh_noise*` a noise kept apart from the programme
(`fresh_realisation`, `fresh_like`, `fresh_noise_linear`), `fresh_floor` an output's floor in
place, where no noise track exists.

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
  linear crossfade), an STFT at 1024 / 768 (`fresh_block`), the envelope E as a 5 x 5 mean of
  |Z|^2. The noise-led cells (`noise_led`) have an envelope within 3x its bin's P20 over the
  block and no more than 2x the median envelope of the 21 bins around it: a steady tone sits
  at its own P20 like the floor, so without the spectral-peak test a held note in a music
  tape's pauses was redrawn as a band of noise, the pause reading (which leaves tones out)
  read it as hiss and Gaudeamus5's pause median moved +1.6..+2.1 dB. A noise-led cell mixes
  power-complementarily, sqrt(1 - s) Z + sqrt(s E) W, with W the STFT of seeded white noise
  (one draw over the whole file, unit mean power per bin) and s its share: 0 with half or
  less of its 5 x 5 neighbourhood noise-led, 1 from 90 %, linear between. Every other cell is
  kept: a spectral subtractor's musical-noise islands and a held tone stand over 3x the P20.
  The first build tested 2x the P20 (1.5x a stationary floor's mean, its envelope's relative
  sd being about 0.3) and replaced cells outright: about 5 % of a pure noise's cells failed,
  stayed behind as islands of the old draw and lost their energy in the inverse STFT (a white
  noise came back 0.32 dB down and correlated 0.26 with the old draw, so every fresh pause
  reading read low), and each block drew its own noise (up to 3 dB lost in the crossfades).
  This one keeps a white noise within 0.04 dB at correlation 0.07, and each level of a 20 dB
  gated noise within 0.05 dB. Deciding on 9 x 9 means instead redrew the islands too and
  moved cathar renders' programme reading -0.3..-0.8 dB, a change of texture, not of draw;
- `fresh_like`: the same realisation over every cell, for a noise track on its own;
- `fresh_realisation` (the calibration's, backlog E2 part c): a recorded noise track's
  whole-signal magnitude spectrum with new phases (`resynthesised_noise`), under the track's own
  `NOISE_ENVELOPE_S` level trajectory (`noise_envelope`), so the spectrum and the level wander
  both stay: realistic-v2's recorded noise wanders 8.5-8.7 dB p5-p95 over 100 ms frames on en
  and fr, and a plain phase surrogate flattened that to 1.6-1.9 dB, a steadier hiss;
  `floor_noise` shapes a draw like an excerpt's quietest frames, at their power;
- `allpass` (a 2nd-order RBJ all-pass: every magnitude kept, the phase turned) and
  `tpdf_dither` (one 16-bit LSB of TPDF dither, not re-quantised).

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
NOISE_LED_FACTOR = 3.0
NOISE_LED_PERCENTILE = 20.0
SHARE_FROM = 0.5
SHARE_FULL = 0.9
PEAK_CONTEXT_BINS = 21
PEAK_FACTOR = 2.0
BLOCK_S = 30.0
OVERLAP_S = 1.0
DELAY_PAD = 8192
EPS = 1e-30
# The transforms' sizes, one value for every consumer: a 0.37-sample delay and a tenth of a decibel.
FRACTIONAL_DELAY_SAMPLES = 0.37
GAIN_STEP_DB = 0.1
# A resample round trip passes through the first of these that is not the audio's own rate.
ROUND_TRIP_RATES = (48000, 44100)
ALLPASS_Q = 0.7071
# The frame an excerpt's own noise floor is read on when it carries no recorded noise.
NOISE_FRAME = 1024
# The frame a recorded noise's level trajectory is read on: a fresh realisation keeps the wander slower than this.
NOISE_ENVELOPE_S = 0.05


def gain_db(mono, db):
    """`mono` scaled by `db` decibels."""
    return np.asarray(mono, dtype=np.float64) * 10.0 ** (db / 20.0)


def fractional_delay(mono, delay):
    """`mono` delayed by `delay` samples (fractional) with an FFT phase ramp, cut back to its length."""
    padded = np.pad(np.asarray(mono, dtype=np.float64), (DELAY_PAD, DELAY_PAD))
    spectrum = np.fft.rfft(padded) * np.exp(-2j * np.pi * np.fft.rfftfreq(len(padded)) * delay)
    return np.fft.irfft(spectrum, len(padded))[DELAY_PAD:][: len(mono)]


def round_trip_rate(rate):
    """The rate a resample round trip passes through: 48 kHz, or 44.1 kHz for audio already at 48 kHz."""
    return next(via for via in ROUND_TRIP_RATES if via != int(rate))


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


def unit_spectrum(noise, rate):
    """The STFT of the white noise `noise`, each bin normalised to unit mean power."""
    spectrum = _stft(np.asarray(noise, dtype=np.float64), rate)
    return spectrum / np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=1, keepdims=True) + EPS)


def unit_white(length, rate, rng):
    """The STFT of `length` samples of white noise from `rng`, each bin normalised to unit mean power."""
    return unit_spectrum(rng.standard_normal(length), rate)


def fresh_like(noise, rate, seed):
    """A fresh realisation of a noise track: every cell's draw replaced around the same envelope."""
    noise = np.asarray(noise, dtype=np.float64)
    spectrum = _stft(noise, rate)
    fresh = np.sqrt(envelope(spectrum)) * unit_white(len(noise), rate, np.random.default_rng(seed))
    return _istft(fresh, rate, len(noise))


def noise_led(power):
    """Cells whose envelope sits within 3x its bin's P20 over the block and within 2x the median of the 21 bins around it.

    The second test keeps a spectral peak out: a held note sits at its own P20 like the floor does, but stands over the
    bins around it in every frame.
    """
    steady = power <= NOISE_LED_FACTOR * np.percentile(power, NOISE_LED_PERCENTILE, axis=1, keepdims=True)
    local = scipy.ndimage.median_filter(power, size=(PEAK_CONTEXT_BINS, 1), mode="nearest")
    return steady & (power <= PEAK_FACTOR * local)


def redraw_share(led):
    """Each noise-led cell's share of fresh noise: 0 to 1 as its 5 x 5 neighbourhood goes 50 to 90 % noise-led; 0 kept."""
    near = scipy.ndimage.uniform_filter(led.astype(np.float64), size=ENVELOPE_CELLS, mode="nearest")
    return np.where(led, np.clip((near - SHARE_FROM) / (SHARE_FULL - SHARE_FROM), 0.0, 1.0), 0.0)


def fresh_block(mono, rate, noise):
    """One block with its noise-led cells redrawn from `noise` (as long as the block) around the envelope; the rest kept.

    The share mixes power-complementarily, sqrt(1 - s) Z + sqrt(s E) W, so a noise-led cell beside a kept one keeps its
    power through the inverse STFT instead of losing the old draw's coherence with its neighbours.
    """
    spectrum = _stft(np.asarray(mono, dtype=np.float64), rate)
    power = envelope(spectrum)
    share = redraw_share(noise_led(power))
    fresh = np.sqrt(1.0 - share) * spectrum + np.sqrt(share * power) * unit_spectrum(noise, rate)
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
    """A fresh realisation of `mono`'s stationary floor, per 30 s block with 1 s linear crossfades (module docstring).

    One white noise is drawn over the whole file and each block takes its own stretch of it, so two blocks' redraws are
    the same noise in their overlap: independent draws per block lost up to 3 dB of floor at the crossfade's middle.
    """
    mono = np.asarray(mono, dtype=np.float64)
    noise = np.random.default_rng(seed).standard_normal(len(mono))
    size, overlap = int(BLOCK_S * rate), int(OVERLAP_S * rate)
    if len(mono) <= size:
        return fresh_block(mono, rate, noise)
    out, total = np.zeros(len(mono)), np.zeros(len(mono))
    starts = block_starts(len(mono), size, overlap)
    for index, start in enumerate(starts):
        stop = start + len(mono[start:][:size])
        weight = crossfade_weight(stop - start, overlap, index == 0, index == len(starts) - 1)
        out[start:stop] += fresh_block(mono[start:stop], rate, noise[start:stop]) * weight
        total[start:stop] += weight
    return out / np.maximum(total, EPS)


# --- the calibration's realisations and filters (`quality_degradations.SAME_PERCEPT`) ---------


def resynthesised_noise(noise, rng):
    """The noise's whole-signal magnitude spectrum with new phases (a phase surrogate): its power spectrum exactly, and a
    steady level. A frame-wise STFT re-synthesis loses about 6 dB to the incoherent overlap-add, so the whole signal is
    used; `fresh_realisation` puts a recorded noise's level wander back."""
    spectrum = np.fft.rfft(np.asarray(noise, dtype=np.float64))
    phases = rng.uniform(-np.pi, np.pi, len(spectrum))
    phases[[0, -1]] = 0.0
    return np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), len(noise))


def noise_envelope(noise, rate):
    """`noise`'s level trajectory: the RMS of `NOISE_ENVELOPE_S` frames, interpolated to every sample, never 0."""
    data = np.asarray(noise, dtype=np.float64)
    size = max(int(rate * NOISE_ENVELOPE_S), 1)
    count = max(len(data) // size, 1)
    rms = np.sqrt(np.mean(np.resize(data, count * size).reshape(count, size) ** 2, axis=1))
    rms = np.maximum(rms, max(float(np.max(rms)) * 1e-6, 1e-12))
    return np.interp(np.arange(len(data)), (np.arange(count) + 0.5) * size, rms)


def fresh_realisation(noise, rate, rng):
    """A fresh realisation of a recorded noise: the phase surrogate of the noise with its level trajectory divided out,
    under that same trajectory (`noise_envelope`). The recorded tape noise is not steady: realistic-v2 wobbles it +-3 dB
    at 0.3-1 Hz on top of the tape's own wander (8.5-8.7 dB p5-p95 over 100 ms frames on en and fr), and a plain
    surrogate held it within 1.6-1.9 dB, a different hiss."""
    envelope = noise_envelope(noise, rate)
    return resynthesised_noise(np.asarray(noise, dtype=np.float64) / envelope, rng) * envelope


def floor_noise(mono, rng):
    """Noise with the spectrum of `mono`'s quietest tenth of `NOISE_FRAME` frames, at their power (an excerpt's own hiss)."""
    data = np.asarray(mono, dtype=np.float64)
    frames = data[: len(data) // NOISE_FRAME * NOISE_FRAME].reshape(-1, NOISE_FRAME)
    power = np.mean(frames**2, axis=1)
    quiet = frames[power <= np.percentile(power, 10.0)]
    psd = np.mean(np.abs(np.fft.rfft(quiet * np.hanning(NOISE_FRAME), axis=1)) ** 2, axis=0)
    shape = np.sqrt(np.interp(np.fft.rfftfreq(len(data)), np.fft.rfftfreq(NOISE_FRAME), psd))
    noise = np.fft.irfft(np.fft.rfft(rng.standard_normal(len(data))) * shape, len(data))
    return noise * np.sqrt(np.mean(quiet**2) / max(float(np.mean(noise**2)), 1e-30))


def allpass(mono, rate, frequency_hz, q=None):
    """A 2nd-order all-pass biquad (RBJ): every magnitude kept, the phase turned around `frequency_hz`."""
    w0 = 2.0 * np.pi * frequency_hz / rate
    alpha = np.sin(w0) / (2.0 * (q or ALLPASS_Q))
    b = [1.0 - alpha, -2.0 * np.cos(w0), 1.0 + alpha]
    a = [1.0 + alpha, -2.0 * np.cos(w0), 1.0 - alpha]
    return scipy.signal.lfilter(b, a, np.asarray(mono, dtype=np.float64)).astype(np.float32)


def tpdf_dither(mono, rng):
    """TPDF dither of one 16-bit LSB added, without re-quantising (a requantisation rounds as well)."""
    dither = (rng.random(len(mono)) + rng.random(len(mono)) - 1.0) / 32768.0
    return (np.asarray(mono, dtype=np.float64) + dither).astype(np.float32)
