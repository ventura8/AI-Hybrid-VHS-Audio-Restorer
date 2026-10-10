"""The ear v3 calibration degradations (plan 1.3): one controlled failure or source condition each.

`scripts/quality_degradations.py` holds the table that says which reading must move, which
must not, and on which base material; this module holds the generators. Every generator takes
a mono float signal and returns a float32 array of the same length.

- `air_shelf`: the app's own presence shelf. FFmpeg 8.0.1's `treble=g=G:f=7500` (the filter
  `modules/filters._build_linear_air_filter` writes; default width 0.5, width type q, 2 poles)
  is the RBJ cookbook high shelf at Q = 1/sqrt(2), filtered causally: `scipy.signal.lfilter`
  on `treble_coefficients` matches FFmpeg's own output to 9e-8 on white noise at 0.1 RMS
  (44.1 and 48 kHz, g = +2 / -1 / +0.5; measured 2026-10-09). The literal reading of the
  defaults, Q = 0.5, misses by 0.014. At +2 dB the shelf gives +0.12 dB at 4 kHz, +1.0 at
  7.5 kHz, +1.85 at 12 kHz and +1.98 at 16 kHz (48 kHz). The corner defaults to the shipped
  7500 Hz (`linear_air_freq_hz`'s default, a knob since 2026-10-09); `freq_hz` takes another,
  and `quality_degradations`' `air_corner` row sweeps round A1's 9000 / 7500 / 6000 Hz.
- `spectral_tilt`: a zero-phase FFT gain of `slope` dB per octave above 1 kHz, flat below.
- `lf_shelf` and `lf_highpass`: the low end R11 (`lf_metrics`) reads, as the magnitude of FFmpeg's
  own filters applied zero-phase (`fft_filter`). `lf_shelf` is `bass=g=G:f=100`, the RBJ cookbook
  low shelf at Q = 1/sqrt(2) (`bass_coefficients`, the mirror of `treble_coefficients`);
  `lf_highpass` is `highpass=f=F` (2 poles, Q 1/sqrt(2)), a 2nd-order Butterworth.
  `scipy.signal.lfilter` on those coefficients matches FFmpeg 8.0.1 to 4e-5 (shelf, g = -6 /
  -1.5 / +3 / +6 dB) and 8e-5 (high pass at 40 / 60 / 80 Hz) on white noise at 0.1 RMS, 44.1 and
  48 kHz (measured 2026-10-10; FFmpeg filters float input in single precision, which at a corner
  this low costs that much; Q = 0.5 misses the shelf by 5e-3). Run causally, the high pass also
  turned the bass's phase, and on the bass-led music beds the runner's alignment follows the
  bass: `align_pair` read lags of -7 to -10 samples at 40-60 Hz and +101 at 80 Hz (fr), and the
  misaligned pair moved `dsp.lkr` 0.20-0.26 at 80 Hz, a second change where the table wants one;
  zero-phase, every lag reads 0 and lkr 0. The corners are round C3's
  `cathar_music_dewind_cutoff` values.
- The pause residuals: the source's own pauses replaced by a transformed copy of the source,
  speech left bit-identical. A residual the runner's least-squares gain match could absorb (a
  uniform gain over the whole file) would read nothing, so the transform reaches the pauses
  only: `pause_weight` is exactly 0 over speech (`quality_degradations.speech_mask`) and
  within 40 ms of it, then ramps to 1 over 10 ms. The residual is the source at -12 dB
  (`RESIDUAL_ATTEN_DB`), then: as is (`scaled_residual`, the comfort-noise hypothesis, which
  R4 must read as attenuation with an unchanged shape); tilted up above 2 kHz
  (`hiss_residual`, what the user called hiss on cathar alpha 2 / 0.7.3: highs left over a
  floor that lost its lows); low-passed (`dull_residual`, the APL "dead air" side); or broken
  into islands (`island_residual`: a 256-point STFT keeping a random share of its cells,
  each kept cell carrying the dropped cells' power, the spectral-subtraction texture without
  most of its level change; the overlap-add of incoherent frames still loses 1-2.5 dB, R4
  reading 13.0 / 13.4 / 14.4 dB attenuation at 30 / 10 / 3 % against 11.9 for the plain copy).
- `compander_mistrack`: a VHS Hi-Fi 2:1 compander's expander off by up to `error_db`. The
  detector is the compander's own: the sidechain pre-emphasis of the hifi-decode constants
  (240 / 24 us, +20 dB above 6.6 kHz) and a power follower with a 6.5 ms attack and a 70 ms
  release on 1 ms blocks. An ideal 2:1 / 1:2 pair whose expander is off by a fixed level
  returns the input times a constant (both are power laws), which the gain match removes; so
  the error is modelled as an expansion-ratio error: the gain is 0 dB at the window's loud
  detector level (p95) and `-error_db` at its quiet level (p10), linear in dB between. The
  floor drops in the pauses and rises with every syllable: the breathing R6 and the pause
  readings are meant to hear. A model, not a circuit simulation.
- `loudnorm_ride`: FFmpeg loudnorm's dynamic mode as a slow level rider: the 3 s short-term
  level every 100 ms, `depth` of its distance from the median removed (0 = linear mode, 1 =
  flattened), held through stretches 20 dB under the median, capped at +-12 dB, smoothed over
  1 s. R7 (`file.gain_ride_*`) must read it.
- `phasey_resynth`: the "robotic" voice. A 1024 / 256 Hann STFT whose phases each move by
  `amount` times a uniform draw in (-pi, pi), resynthesised by overlap-add. Each frame keeps
  its magnitudes, but the overlap-add of incoherent frames does not: at full randomisation R1
  reads a tilt of -0.83 / -1.15 dB/oct (en / fr speech fixtures); `dsp.dropouts` read 56-59
  holes until the count was re-centred on the programme frames' median drop (en now 1: the
  gain match had scaled the incoherent output down as a whole); at 15-40 % R1 moves under
  0.15. Phasiness can pass for a timbre change.
- `band_limited`: the linear-track source condition, an 8th-order Butterworth low pass run
  forward and back (the `underwater` filter, steep enough for R0's +-1/6 octave rule;
  `source_profile` reads a gentle filter over hiss higher, by design).
- `treble_dropouts`: Wallace spacing loss, 54.6 d / lambda dB with lambda = v / f, at the PAL
  SP linear speed (23.39 mm/s): a 1 um lift costs 18.7 dB at 8 kHz and 2.3 dB at 1 kHz.
  Events of 50-300 ms, 0.4 per second, 5 ms edges, placed by the caller's generator.
- `sibilant_islands`: the round-1 "distorted s" hypothesis R2's texture reading stands for,
  the sibilance tests' recipe: a share of the 's' frames' 4-12 kHz cells of a 256-point STFT
  zeroed at random, the rest raised to keep the power (asked for by the R2 owner).
- `speed_drift`: the output running a fraction of a percent slow, a speed error the sync
  stage missed: a polyphase resample to `1 + percent / 100` times the length.
- `holes`: the `dropouts` degradation's head-contact loss, built for the count that reads it
  (`dsp_metrics.dropout_count`, `file.dropouts`): `count` holes of 30-50 ms with 0.5 ms
  edges, each opening 50 ms of 10 ms frames the count calls programme, 250 ms apart, drawn
  from their own seed so a level's holes are the first ones of every higher level's. The v2
  generator (`realistic_defects.inject_dropouts`, the fixtures' injector, unchanged) drew a
  count per file at 5-50 ms anywhere from the shared generator, each level its own: under
  25 ms a hole cannot fill two frames and half of them fell under the programme cut, so the
  whole pair read 1 / 2 / 2 on Vaccin and 1 / 0 / 3 on SOTI at 3 / 6 / 12 (calibration v3).
  These read 3 / 6 / 12 on en and on every tape cut. They draw nothing from the caller's
  generator, which shifted the shared stream of every later rng-driven case once.
- `window_fricatives`: the 's' frames each 15 s runner window finds on its own percentiles,
  joined; what `quality_degradations.fricative_ramp` weights, so the R2 degradations reach
  the frames R2 reads (the whole-file mask reached 0.80-0.82 of them on Tele7abc).
"""

from fractions import Fraction

import numpy as np
import scipy.ndimage
import scipy.signal

from scripts.restoration_quality import audio_io, dsp_metrics, sibilance

# The shipped shelf corner (linear_air_freq_hz's default); air_shelf's freq_hz takes another.
AIR_SHELF_HZ = 7500.0
TREBLE_Q = 1.0 / np.sqrt(2.0)
# FFmpeg's `bass` default corner, and `highpass`'s two poles.
LF_SHELF_HZ = 100.0
LF_HIGHPASS_ORDER = 2
TILT_PIVOT_HZ = 1000.0
# The pause residuals.
PAUSE_HANGOVER_S = 0.04
PAUSE_RAMP_S = 0.01
RESIDUAL_ATTEN_DB = -12.0
RESIDUAL_TILT_FROM_HZ = 2000.0
RESIDUAL_LOWPASS_ORDER = 8
ISLAND_FRAME = 256
# The Hi-Fi compander (hifi-decode constants).
COMPANDER_ATTACK_S = 0.0065
COMPANDER_RELEASE_S = 0.070
SIDECHAIN_TAU_S = (240e-6, 24e-6)
COMPANDER_BLOCK_S = 0.001
COMPANDER_SPAN_PERCENTILES = (10.0, 95.0)
# The loudnorm rider.
RIDE_WINDOW_S = 3.0
RIDE_HOP_S = 0.1
RIDE_GATE_DB = 20.0
RIDE_MAX_DB = 12.0
RIDE_SMOOTH_S = 1.0
# Phase resynthesis.
PHASE_FRAME = 1024
PHASE_HOP = 256
# The linear track.
BAND_LIMIT_ORDER = 8
# Treble dropouts (Wallace spacing loss).
LINEAR_SP_SPEED_M_S = 0.02339
WALLACE_DB = 54.6
DROPOUT_RATE_PER_S = 0.4
DROPOUT_S = (0.05, 0.3)
DROPOUT_RAMP_S = 0.005
# Islands in the 's' (R2's texture reading).
SIB_ISLAND_FRAME = 256
SIB_ISLAND_BAND_HZ = (4000.0, 12000.0)
# A speed error the sync stage missed.
DRIFT_DENOMINATOR = 100000
# The `dropouts` holes: long enough for the count's two 10 ms frames whatever their phase, from
# their own seed (nested levels), 250 ms apart so no two runs merge, with the fixtures' 0.5 ms
# edges (`realistic_defects.DROPOUT_EDGE_S`).
HOLE_SEED = 20261009
HOLE_MS = (30.0, 50.0)
HOLE_SPACING_S = 0.25
HOLE_EDGE_S = 0.0005
# The runner's windows (`runner.score_pair`'s defaults): R2 finds the 's' frames per window.
RUNNER_WINDOW_S = 15.0
RUNNER_HOP_S = 7.5
EPS = 1e-20


def _float(mono):
    return np.asarray(mono, dtype=np.float64)


def _padded(data, frame):
    """`data` zero-padded to at least one STFT frame (scipy shortens the frame of a shorter input, and its inverse fails)."""
    return np.pad(data, (0, max(0, frame - len(data))))


def _fitted(out, length):
    """`out` cut or zero-padded to `length` samples (an inverse STFT may come back a few samples off)."""
    return np.pad(out[:length], (0, max(0, length - len(out))))


def treble_coefficients(gain_db, rate, freq_hz=AIR_SHELF_HZ, q=TREBLE_Q):
    """`(b, a)` of FFmpeg's `treble=g=gain_db:f=freq_hz` (RBJ high shelf), normalised so `a[0]` is 1."""
    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * np.pi * freq_hz / rate
    cosine, shelf = np.cos(omega), 2.0 * np.sqrt(amplitude) * np.sin(omega) / (2.0 * q)
    up, down = amplitude + 1.0, amplitude - 1.0
    b = amplitude * np.array([up + down * cosine + shelf, -2.0 * (down + up * cosine), up + down * cosine - shelf])
    a = np.array([up - down * cosine + shelf, 2.0 * (down - up * cosine), up - down * cosine - shelf])
    return b / a[0], a / a[0]


def air_shelf(mono, rate, gain_db, freq_hz=AIR_SHELF_HZ):
    """`mono` through the app's presence shelf at `gain_db` (positive lifts the top, negative cuts it)."""
    b, a = treble_coefficients(gain_db, rate, freq_hz)
    return scipy.signal.lfilter(b, a, _float(mono)).astype(np.float32)


def bass_coefficients(gain_db, rate, freq_hz=LF_SHELF_HZ, q=TREBLE_Q):
    """`(b, a)` of FFmpeg's `bass=g=gain_db:f=freq_hz` (RBJ low shelf), normalised so `a[0]` is 1."""
    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * np.pi * freq_hz / rate
    cosine, shelf = np.cos(omega), 2.0 * np.sqrt(amplitude) * np.sin(omega) / (2.0 * q)
    up, down = amplitude + 1.0, amplitude - 1.0
    b = amplitude * np.array([up - down * cosine + shelf, 2.0 * (down - up * cosine), up - down * cosine - shelf])
    a = np.array([up + down * cosine + shelf, -2.0 * (down + up * cosine), up + down * cosine - shelf])
    return b / a[0], a / a[0]


def response_db(b, a, freqs, rate):
    """The magnitude response in dB of the filter `(b, a)` at `freqs` (Hz)."""
    _w, response = scipy.signal.freqz(b, a, worN=np.asarray(freqs, dtype=np.float64), fs=rate)
    return 20.0 * np.log10(np.maximum(np.abs(response), EPS))


def lf_shelf(mono, rate, gain_db, freq_hz=LF_SHELF_HZ):
    """`mono` through the magnitude of FFmpeg's `bass` shelf at `gain_db`, zero phase (negative thins the low end)."""
    b, a = bass_coefficients(gain_db, rate, freq_hz)
    return fft_filter(mono, rate, lambda freqs: response_db(b, a, freqs, rate))


def lf_highpass(mono, rate, cutoff_hz):
    """`mono` through the magnitude of FFmpeg's `highpass=f=cutoff_hz` (2nd-order Butterworth), zero phase."""
    b, a = scipy.signal.butter(LF_HIGHPASS_ORDER, cutoff_hz, btype="highpass", fs=rate)
    return fft_filter(mono, rate, lambda freqs: response_db(b, a, freqs, rate))


def tilt_gain_db(freqs, slope_db_oct, pivot_hz=TILT_PIVOT_HZ):
    """`slope_db_oct` dB per octave above `pivot_hz`, 0 dB below it."""
    return slope_db_oct * np.log2(np.maximum(np.asarray(freqs, dtype=np.float64), pivot_hz) / pivot_hz)


def fft_filter(mono, rate, gain_db):
    """`mono` through the zero-phase gain `gain_db(freqs)` (dB), applied on one FFT of the whole signal."""
    data = _float(mono)
    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1.0 / rate)
    return np.fft.irfft(spectrum * 10.0 ** (gain_db(freqs) / 20.0), n=len(data)).astype(np.float32)


def spectral_tilt(mono, rate, slope_db_oct, pivot_hz=TILT_PIVOT_HZ):
    """`mono` tilted by `slope_db_oct` per octave above `pivot_hz` (zero phase)."""
    return fft_filter(mono, rate, lambda freqs: tilt_gain_db(freqs, slope_db_oct, pivot_hz))


def pause_weight(speech, rate):
    """1 deep in the pauses, exactly 0 over `speech` (a per-sample boolean) and 40 ms either side, then a 10 ms linear ramp."""
    speech = np.asarray(speech, dtype=bool)
    if not speech.any():
        return np.ones(len(speech))
    distance = scipy.ndimage.distance_transform_edt(~speech) - PAUSE_HANGOVER_S * rate
    return np.clip(distance / max(1.0, PAUSE_RAMP_S * rate), 0.0, 1.0)


def in_pauses(mono, weight, residual):
    """`mono` with `residual` faded in where `weight` is 1; bit-identical where it is 0."""
    data = _float(mono)
    return np.where(weight > 0.0, data + weight * (_float(residual) - data), data).astype(np.float32)


def scaled_residual(mono, atten_db=RESIDUAL_ATTEN_DB):
    """The comfort-noise residual: `mono` at `atten_db`."""
    return _float(mono) * 10.0 ** (atten_db / 20.0)


def hiss_residual(mono, rate, slope_db_oct, atten_db=RESIDUAL_ATTEN_DB):
    """The hissy residual: `mono` at `atten_db`, tilted up by `slope_db_oct` per octave above 2 kHz."""
    return spectral_tilt(scaled_residual(mono, atten_db), rate, slope_db_oct, RESIDUAL_TILT_FROM_HZ).astype(np.float64)


def dull_residual(mono, rate, cutoff_hz, atten_db=RESIDUAL_ATTEN_DB):
    """The dead residual: `mono` at `atten_db` through an 8th-order low pass at `cutoff_hz` (zero phase)."""
    sos = scipy.signal.butter(RESIDUAL_LOWPASS_ORDER, cutoff_hz, btype="lowpass", fs=rate, output="sos")
    return scipy.signal.sosfiltfilt(sos, scaled_residual(mono, atten_db))


def island_residual(mono, rate, keep_share, rng, atten_db=RESIDUAL_ATTEN_DB):
    """The musical-noise residual: `mono` at `atten_db` with a random `keep_share` of its STFT cells kept, each at 1/share power."""
    data = scaled_residual(mono, atten_db)
    _f, _t, spec = scipy.signal.stft(_padded(data, ISLAND_FRAME), fs=rate, nperseg=ISLAND_FRAME, noverlap=ISLAND_FRAME // 2)
    kept = (rng.random(spec.shape) < keep_share) / np.sqrt(keep_share)
    _t, out = scipy.signal.istft(spec * kept, fs=rate, nperseg=ISLAND_FRAME, noverlap=ISLAND_FRAME // 2)
    return _fitted(out, len(data))


def follow(power, rate, attack_s, release_s):
    """One-pole attack / release follower over `power` sampled at `rate`."""
    attack, release = np.exp(-1.0 / (attack_s * rate)), np.exp(-1.0 / (release_s * rate))
    out = np.empty(len(power), dtype=np.float64)
    state = float(power[0]) if len(power) else 0.0
    for index, value in enumerate(power):
        coefficient = attack if value > state else release
        state = coefficient * state + (1.0 - coefficient) * value
        out[index] = state
    return out


def sidechain_level_db(mono, rate):
    """`(block centres in samples, detector level in dB)`: pre-emphasised power on 1 ms blocks through the compander's follower."""
    zeros, poles = SIDECHAIN_TAU_S
    b, a = scipy.signal.bilinear([zeros, 1.0], [poles, 1.0], fs=rate)
    emphasised = scipy.signal.lfilter(b, a, _float(mono))
    block = max(1, int(COMPANDER_BLOCK_S * rate))
    count = max(1, len(emphasised) // block)
    power = np.mean(np.resize(emphasised, count * block).reshape(count, block) ** 2, axis=1)
    level = 10.0 * np.log10(follow(power, rate / block, COMPANDER_ATTACK_S, COMPANDER_RELEASE_S) + EPS)
    return (np.arange(count) + 0.5) * block, level


def compander_mistrack(mono, rate, error_db):
    """`mono` through an expander whose gain is off by up to `error_db`: 0 dB at the loud detector level, -error_db at the quiet one."""
    centres, level = sidechain_level_db(mono, rate)
    quiet, loud = np.percentile(level, COMPANDER_SPAN_PERCENTILES)
    position = np.clip((level - loud) / max(loud - quiet, 1e-6), -1.0, 0.0)
    gain_db = np.interp(np.arange(len(mono)), centres, error_db * position)
    return (_float(mono) * 10.0 ** (gain_db / 20.0)).astype(np.float32)


def short_term_level_db(mono, rate):
    """`(hop centres in samples, level in dB)`: the 3 s mean-square level every 100 ms."""
    data = _float(mono)
    window, hop = max(1, int(RIDE_WINDOW_S * rate)), max(1, int(RIDE_HOP_S * rate))
    cumulative = np.concatenate([[0.0], np.cumsum(data**2)])
    starts = np.arange(0, max(1, len(data) - window + 1), hop)
    stops = np.minimum(starts + window, len(data))
    power = (cumulative[stops] - cumulative[starts]) / np.maximum(stops - starts, 1)
    return (starts + stops) / 2.0, 10.0 * np.log10(power + EPS)


def ride_gain_db(level, depth):
    """The rider's gain per hop: `depth` of the level's distance from the median taken out, held through quiet stretches."""
    median = float(np.median(level))
    live = level > median - RIDE_GATE_DB
    gain = np.clip(depth * (median - level), -RIDE_MAX_DB, RIDE_MAX_DB)
    hops = np.arange(len(level))
    held = np.interp(hops, hops[live], gain[live])
    size = max(1, int(round(RIDE_SMOOTH_S / RIDE_HOP_S)))
    return scipy.ndimage.uniform_filter1d(held, size, mode="nearest")


def loudnorm_ride(mono, rate, depth):
    """`mono` under a slow level rider that removes `depth` of the 3 s level's swing (loudnorm's dynamic mode)."""
    centres, level = short_term_level_db(mono, rate)
    gain_db = np.interp(np.arange(len(mono)), centres, ride_gain_db(level, depth))
    return (_float(mono) * 10.0 ** (gain_db / 20.0)).astype(np.float32)


def phasey_resynth(mono, rate, amount, rng):
    """The robotic voice: every STFT phase moved by `amount` x U(-pi, pi), magnitudes kept, overlap-added back."""
    data = _float(mono)
    _f, _t, spec = scipy.signal.stft(_padded(data, PHASE_FRAME), fs=rate, nperseg=PHASE_FRAME, noverlap=PHASE_FRAME - PHASE_HOP)
    turned = spec * np.exp(1j * amount * rng.uniform(-np.pi, np.pi, spec.shape))
    _t, out = scipy.signal.istft(turned, fs=rate, nperseg=PHASE_FRAME, noverlap=PHASE_FRAME - PHASE_HOP)
    return _fitted(out, len(data)).astype(np.float32)


def band_limited(mono, rate, cutoff_hz):
    """The linear-track source: an 8th-order Butterworth low pass at `cutoff_hz`, forward and back."""
    sos = scipy.signal.butter(BAND_LIMIT_ORDER, min(cutoff_hz, 0.49 * rate), btype="lowpass", fs=rate, output="sos")
    return scipy.signal.sosfiltfilt(sos, _float(mono)).astype(np.float32)


def spacing_loss_db(freqs, spacing_um, speed_m_s=LINEAR_SP_SPEED_M_S):
    """Wallace spacing loss in dB (negative): 54.6 d / lambda with lambda = speed / f."""
    wavelength_um = speed_m_s * 1e6 / np.maximum(np.asarray(freqs, dtype=np.float64), 1e-9)
    return -WALLACE_DB * spacing_um / wavelength_um


def dropout_weight(length, rate, rng):
    """1 inside 50-300 ms events (0.4 per second, 5 ms edges), 0 elsewhere."""
    count = max(1, int(round(DROPOUT_RATE_PER_S * length / rate)))
    gate = np.zeros(length, dtype=np.float64)
    for _event in range(count):
        span = int(rng.uniform(*DROPOUT_S) * rate)
        start = int(rng.integers(0, max(1, length - span)))
        gate[start:][:span] = 1.0
    ramp = max(1, int(DROPOUT_RAMP_S * rate))
    return scipy.ndimage.uniform_filter1d(gate, ramp, mode="nearest")


def treble_dropouts(mono, rate, spacing_um, rng):
    """`mono` with the treble lost to a `spacing_um` head lift during short events (Wallace spacing loss)."""
    data = _float(mono)
    lifted = fft_filter(data, rate, lambda freqs: spacing_loss_db(freqs, spacing_um)).astype(np.float64)
    weight = dropout_weight(len(data), rate, rng)
    return (data + weight * (lifted - data)).astype(np.float32)


def sibilant_islands(mono, rate, share, fricative, rng):
    """A fluctuating mask on the 's': `share` of its 4-12 kHz STFT cells zeroed, the rest raised to keep the power.

    `fricative` is a per-sample weight of the 's' (`quality_degradations.fricative_ramp`); a
    256-point frame changes when the weight at its centre is over one half.
    """
    data = _float(mono)
    freqs, times, spec = scipy.signal.stft(_padded(data, SIB_ISLAND_FRAME), fs=rate, nperseg=SIB_ISLAND_FRAME)
    inside = np.asarray(fricative)[np.clip((times * rate).astype(int), 0, len(data) - 1)] > 0.5
    band = (freqs >= SIB_ISLAND_BAND_HZ[0]) & (freqs < SIB_ISLAND_BAND_HZ[1])
    kept = (rng.random(spec.shape) >= share) / np.sqrt(1.0 - share)
    _t, out = scipy.signal.istft(spec * np.where(band[:, None] & inside[None, :], kept, 1.0), fs=rate, nperseg=SIB_ISLAND_FRAME)
    return _fitted(out, len(data)).astype(np.float32)


def speed_drift(mono, percent):
    """`mono` running `percent` slow: resampled (polyphase) to `1 + percent / 100` times its length, so it drifts late."""
    ratio = Fraction(1.0 + percent / 100.0).limit_denominator(DRIFT_DENOMINATOR)
    return scipy.signal.resample_poly(_float(mono), ratio.numerator, ratio.denominator).astype(np.float32)


def spaced_starts(order, frame, count, spacing):
    """The first `count` frame starts in `order` that lie at least `spacing` samples apart."""
    starts = []
    for index in order:
        start = int(index) * frame
        if all(abs(start - other) >= spacing for other in starts):
            starts.append(start)
        if len(starts) == count:
            break
    return starts


def holes(mono, rate, count):
    """`mono` with `count` holes the dropout count resolves (module docstring); level by level, the holes are nested."""
    out = np.asarray(mono, dtype=np.float32).copy()
    for start, length in hole_spans(mono, rate, count, np.random.default_rng(HOLE_SEED)):
        _hole(out, start, length, max(int(HOLE_EDGE_S * rate), 1))
    return out


def hole_spans(mono, rate, count, rng):
    """The first `count` holes `rng` draws: `(start, length)` in samples, each opening a run of programme frames."""
    frame = int(dsp_metrics.DROPOUT_FRAME_S * rate)
    starts = spaced_starts(rng.permutation(programme_starts(mono, frame)), frame, count, HOLE_SPACING_S * rate)
    return [(start, int(rng.uniform(*HOLE_MS) * rate / 1000.0)) for start in starts]


def programme_starts(mono, frame):
    """The frames that open `HOLE_MS[1]` of frames the dropout count calls programme (over the whole signal)."""
    programme = dsp_metrics.programme_frames(dsp_metrics.frame_levels(mono, frame))
    run = int(np.ceil(HOLE_MS[1] / 1000.0 / dsp_metrics.DROPOUT_FRAME_S))
    return np.flatnonzero(np.convolve(programme.astype(np.int64), np.ones(run, dtype=np.int64), mode="valid") == run)


def _hole(out, start, length, edge):
    """Zeroes `length` samples of `out` from `start` (in place), faded over `edge` samples either side.

    A hole opens a run of programme frames inside the signal (`programme_starts`), so its span is whole.
    """
    ramp = np.linspace(1.0, 0.0, edge, dtype=np.float32)
    span = out[start:][:length]
    span[:edge] *= ramp
    span[edge:-edge] = 0.0
    span[-edge:] *= ramp[::-1]


def window_fricatives(mono, rate):
    """Per sample, whether any 15 s runner window (7.5 s hop) calls it fricative on its own percentiles.

    R2 classifies each window's frames on that window's floor and gap hiss; a mask from
    whole-file percentiles missed them on a tape whose floor moves: on Tele7abc the 15 %
    islands reached 18 of window 1's 55 read fricative frames (coverage 0.80-0.82 over the cut,
    SOTI 0.97, Vaccin 0.87), the union 0.97-1.00.
    """
    mask = np.zeros(len(mono), dtype=bool)
    for window in audio_io.windows(len(mono), rate, RUNNER_WINDOW_S, RUNNER_HOP_S):
        span = window.slice_of(rate)
        mask[span] |= sibilance.fricative_mask(mono[span], rate)
    return mask
