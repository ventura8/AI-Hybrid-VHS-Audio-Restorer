"""Ear v4's calibration material and degradations: hiss changed under the programme or in the pauses alone.

The material (fixtures only, `linear_capture`): a linear-track capture built from a realistic-v2
speech fixture: its clean programme through an 8th-order 4.5 kHz low pass (the Tata tapes'
programme band ends at 4.5-5 kHz), two 1.2 s gaps spliced in at a third and two thirds of its
length (10 ms fades) so R4 finds true pauses and the hiss anchor validates, and stationary tape
hiss running on through the gaps (`tape_hiss`: white noise plus a first-order 2 kHz high-passed
copy) at the fixture's own recorded noise's 3-10 kHz level. Not that recorded noise itself
(`vhs - clean`), though the spec named it: it is rumble-heavy (-58 dB per hertz at 100-500 Hz
against -102 at 10-16 kHz on en mid00_speech_m12) at 15 dB under the programme, and on the
gapped capture R4's true-pause mask found no frame at all on en and fr, a stationary
realisation of the same noise none either, so every hiss case read None; the hiss anchor's cap,
built for HF-flat tape hiss, also held its 1-4 kHz bands 7-19 dB under the noise there. The
tape hiss gives 2 s of true pauses, a validated anchor and source margins of +32 / +37 dB
(programme / pauses, en). The source of every case is that hissy capture, so a
restoration-like output moves the readings down and the levels are ordered by increasing
removal (`pause_residual_scaled` does the same).

`pause_weight_true` is 1 on exactly the frames `pause_metrics.true_pause_mask` keeps, with 5 ms
linear ramps placed on the active side, so "the pauses untouched" holds on the mask the pause
reading uses. With p that weight, n the noise and P the programme:

- `hiss_under_programme`: P + n (p + (1 - p) 10^(-L/20)): hiss taken L dB down under the
  programme only, the pauses keeping the tape's hiss sample for sample;
- `hiss_in_pauses_only`: P + n (p 10^(-L/20) + (1 - p) 10^(-20/20)): programme time cleaned to
  -20 dB, the pause hiss varied; `dead_pauses` is the same at 30-60 dB (dead air);
- `hiss_hf_tilt`: P + the noise at -24 dB tilted up L dB per octave above 2 kHz everywhere;
- `low_floor_kept`: P + n at -30 dB + a hiss-free room floor (seeded pink noise through a
  4th-order 1 kHz low pass, zero phase) whose 100-1000 Hz power is the source's true-pause
  100-1000 Hz power x 10^(L/10): a quiet room floor must not read as hiss;
- `hiss_islands`: P + the noise at -12 dB broken into STFT islands (a guard proof);
- `hiss_removed_dulled`: P low-passed at L Hz + n at -30 dB: hiss removed by dulling (a guard
  proof);
- `soft_programme_cut`: P with the STFT cells (2048 / 1024, Hann) the capture holds 6-18 dB over
  the hiss's own mean per bin, from 1 kHz up, cut L dB, + n at -30 dB: an SNR-dependent
  suppressor that takes the soft programme with the hiss (the soft-programme guard's proof).

The builders draw from their own seeds, never from the calibration's shared generator: the
benign cases are built after the degradations, and a draw taken here would move every
existing benign floor.
"""

import numpy as np
import scipy.ndimage
import scipy.signal

from scripts import degradations_v3 as v3
from scripts import same_percept
from scripts.restoration_quality import pause_metrics

LINEAR_LOWPASS_HZ = 4500.0
LOWPASS_ORDER = 8
GAP_S = 1.2
GAP_AT = (1.0 / 3.0, 2.0 / 3.0)
FADE_S = 0.01
EDGE_RAMP_S = 0.005
PROGRAMME_CLEANED_DB = 20.0
HISS_LEFT_DB = 30.0
TILT_ATTEN_DB = -24.0
ISLAND_ATTEN_DB = -12.0
ROOM_CUTOFF_HZ = 1000.0
ROOM_ORDER = 4
ROOM_BAND_HZ = (100.0, 1000.0)
ROOM_SEED = 20261012
ISLAND_SEED = 20261013
FRESH_SEED = 20261014
TAPE_HISS_SEED = 20261015
TAPE_HISS_CORNER_HZ = 2000.0
TAPE_HISS_BAND_HZ = (3000.0, 10000.0)
LINEAR_SUFFIX = "_linear"
SOFT_FRAME = 2048
SOFT_BAND_DB = (6.0, 18.0)
SOFT_LO_HZ = 1000.0
SOFT_EPS = 1e-30


def lowpass(mono, rate, cutoff_hz, order=LOWPASS_ORDER):
    """`mono` through a Butterworth low pass at `cutoff_hz`, zero phase."""
    sos = scipy.signal.butter(order, cutoff_hz, btype="lowpass", fs=rate, output="sos")
    return scipy.signal.sosfiltfilt(sos, np.asarray(mono, dtype=np.float64))


def faded(piece, fade, head, tail):
    """A copy of `piece` faded in over `fade` samples at its head and out at its tail, where asked."""
    piece = np.asarray(piece, dtype=np.float64).copy()
    ramp = np.linspace(0.0, 1.0, min(fade, len(piece)))
    if head:
        piece[: len(ramp)] *= ramp
    if tail:
        start = len(piece) - len(ramp)
        piece[start:] *= ramp[::-1]
    return piece


def with_gaps(mono, rate):
    """`mono` with two 1.2 s zero gaps spliced in at a third and two thirds of its length, faded over 10 ms either side."""
    mono = np.asarray(mono, dtype=np.float64)
    pieces = np.split(mono, [int(at * len(mono)) for at in GAP_AT])
    fade, gap = int(FADE_S * rate), np.zeros(int(GAP_S * rate))
    faded_pieces = [faded(piece, fade, index > 0, index < len(pieces) - 1) for index, piece in enumerate(pieces)]
    return np.concatenate([faded_pieces[0], gap, faded_pieces[1], gap, faded_pieces[2]])


def band_level(mono, rate, band_hz=TAPE_HISS_BAND_HZ):
    """The mean power spectral density of `mono` over `band_hz` (Welch, 4096-point segments)."""
    freqs, density = scipy.signal.welch(np.asarray(mono, dtype=np.float64), rate, nperseg=4096)
    return float(density[(freqs >= band_hz[0]) & (freqs < band_hz[1])].mean())


def tape_hiss(noise, rate, count, seed=TAPE_HISS_SEED):
    """`count` samples of stationary tape hiss (white plus a first-order 2 kHz high-passed copy) at `noise`'s 3-10 kHz level."""
    white = np.random.default_rng(seed).standard_normal(count)
    sos = scipy.signal.butter(1, TAPE_HISS_CORNER_HZ, btype="highpass", fs=rate, output="sos")
    hiss = 0.5 * white + scipy.signal.sosfilt(sos, white)
    return hiss * np.sqrt(band_level(noise, rate) / band_level(hiss, rate))


def linear_capture(clean, noise, rate):
    """`(programme, hiss, capture)`: the low-passed, gapped programme, tape hiss at the fixture noise's level, their sum."""
    programme = with_gaps(lowpass(clean, rate, LINEAR_LOWPASS_HZ), rate)
    hiss = tape_hiss(noise, rate, len(programme))
    return programme, hiss, programme + hiss


def pause_weight_true(source, rate):
    """Per sample: 1 on the true-pause mask's frames, 0 elsewhere, with 5 ms linear ramps on the active side."""
    frame = max(1, int(pause_metrics.FRAME_S * rate))
    mask, _runs = pause_metrics.true_pause_mask(np.asarray(source, dtype=np.float64), rate)
    inside = np.zeros(len(source), dtype=bool)
    inside[: len(mask) * frame] = np.repeat(mask, frame)
    if not inside.any():
        return np.zeros(len(source))
    distance = scipy.ndimage.distance_transform_edt(~inside)
    return np.clip(1.0 - distance / max(1.0, EDGE_RAMP_S * rate), 0.0, 1.0)


def _db(level):
    """The amplitude factor of a level `level` dB down."""
    return 10.0 ** (-level / 20.0)


def _parts(materials):
    return materials["prog_linear"], materials["noise_linear"], materials["vhs_linear"]


def under_programme(level, _base, rate, materials, _rng):
    """`hiss_under_programme`: the hiss L dB down under the programme only."""
    programme, noise, source = _parts(materials)
    weight = pause_weight_true(source, rate)
    return source, programme + noise * (weight + (1.0 - weight) * _db(level))


def in_pauses_only(level, _base, rate, materials, _rng):
    """`hiss_in_pauses_only` and `dead_pauses`: the programme time 20 dB down, the pause hiss L dB down."""
    programme, noise, source = _parts(materials)
    weight = pause_weight_true(source, rate)
    return source, programme + noise * (weight * _db(level) + (1.0 - weight) * _db(PROGRAMME_CLEANED_DB))


def hf_tilt(level, _base, rate, materials, _rng):
    """`hiss_hf_tilt`: the noise at -24 dB tilted up L dB per octave above 2 kHz, everywhere."""
    programme, noise, source = _parts(materials)
    return source, programme + v3.hiss_residual(noise, rate, level, atten_db=TILT_ATTEN_DB)


def band_power(mono, rate, band_hz):
    """Per sample, the power of `mono` in `band_hz` (a 4th-order band pass, zero phase)."""
    sos = scipy.signal.butter(ROOM_ORDER, band_hz, btype="bandpass", fs=rate, output="sos")
    return scipy.signal.sosfiltfilt(sos, np.asarray(mono, dtype=np.float64)) ** 2


def room_floor(source, rate, level):
    """Seeded pink noise through a 1 kHz low pass, at the source's true-pause 100-1000 Hz power x 10^(L/10)."""
    white = np.fft.rfft(np.random.default_rng(ROOM_SEED).standard_normal(len(source)))
    pink = np.fft.irfft(white / np.sqrt(np.maximum(np.arange(len(white)), 1.0)), len(source))
    room = lowpass(pink, rate, ROOM_CUTOFF_HZ, ROOM_ORDER)
    paused = pause_weight_true(source, rate) >= 1.0
    target = band_power(source, rate, ROOM_BAND_HZ)[paused].mean() * 10.0 ** (level / 10.0)
    return room * np.sqrt(target / band_power(room, rate, ROOM_BAND_HZ).mean())


def low_floor(level, _base, rate, materials, _rng):
    """`low_floor_kept`: the hiss at -30 dB and a hiss-free room floor L dB re the source's own pause floor under 1 kHz."""
    programme, noise, source = _parts(materials)
    return source, programme + noise * _db(HISS_LEFT_DB) + room_floor(source, rate, level)


def islands(level, _base, rate, materials, _rng):
    """`hiss_islands`: the noise at -12 dB with a share L of its STFT cells kept (its own seed per level)."""
    programme, noise, source = _parts(materials)
    rng = np.random.default_rng((ISLAND_SEED, int(round(level * 1000))))
    return source, programme + v3.island_residual(noise, rate, level, rng, atten_db=ISLAND_ATTEN_DB)


def dulled(level, _base, rate, materials, _rng):
    """`hiss_removed_dulled`: the programme low-passed at L Hz and the hiss 30 dB down."""
    programme, noise, source = _parts(materials)
    return source, lowpass(programme, rate, level) + noise * _db(HISS_LEFT_DB)


def _spectrum(mono, rate):
    return scipy.signal.stft(np.asarray(mono, dtype=np.float64), rate, nperseg=SOFT_FRAME, noverlap=SOFT_FRAME // 2)[2]


def soft_mask(source, noise, rate, level):
    """Per STFT cell, the amplitude gain L dB down where the source stands 6-18 dB over the noise's own mean in its bin
    (bins of 1 kHz and up), 1 elsewhere."""
    hiss = np.mean(np.abs(_spectrum(noise, rate)) ** 2, axis=1)
    snr = 10.0 * np.log10(np.abs(_spectrum(source, rate)) ** 2 / np.maximum(hiss, SOFT_EPS)[:, np.newaxis] + SOFT_EPS)
    high = np.fft.rfftfreq(SOFT_FRAME, 1.0 / rate) >= SOFT_LO_HZ
    cells = (snr >= SOFT_BAND_DB[0]) & (snr < SOFT_BAND_DB[1]) & high[:, np.newaxis]
    return np.where(cells, _db(level), 1.0)


def soft_cut(level, _base, rate, materials, _rng):
    """`soft_programme_cut`: the programme with its soft cells (`soft_mask` on the capture) cut L dB, the hiss 30 dB down."""
    programme, noise, source = _parts(materials)
    spectrum = _spectrum(programme, rate) * soft_mask(source, noise, rate, level)
    cut = scipy.signal.istft(spectrum, rate, nperseg=SOFT_FRAME, noverlap=SOFT_FRAME // 2)[1][: len(programme)]
    return source, np.pad(cut, (0, len(programme) - len(cut))) + noise * _db(HISS_LEFT_DB)


BUILDERS = {
    "hiss_under_programme": under_programme,
    "hiss_in_pauses_only": in_pauses_only,
    "hiss_hf_tilt": hf_tilt,
    "low_floor_kept": low_floor,
    "dead_pauses": in_pauses_only,
    "hiss_islands": islands,
    "hiss_removed_dulled": dulled,
    "soft_programme_cut": soft_cut,
}


def fresh_noise_linear(_base, rate, materials):
    """The programme with a fresh realisation of its tape hiss: `tape_hiss` from another seed at the same 3-10 kHz level.

    Exact, where `same_percept.fresh_like` (redrawing each cell around a 5 x 5 mean of the old one) carried the old
    draw's local fluctuation into the new one and moved `dsp.hiss_island_kurt` 0.22 on a noise no listener tells apart.
    """
    noise = materials["noise_linear"]
    return materials["prog_linear"] + tape_hiss(noise, rate, len(noise), FRESH_SEED)


# The same-percept transforms on the linear capture, from `scripts/same_percept.py` (one set for every consumer), and
# the exact redraw of its hiss; `quality_degradations.LINEAR_BENIGN` adds `requantise_linear` on the shared generator.
LINEAR_BENIGN = {
    "identity_linear": lambda base, _rate, _materials: np.asarray(base, dtype=np.float64).copy(),
    "resample_linear": lambda base, rate, _materials: same_percept.soxr_round_trip(base, rate, same_percept.round_trip_rate(rate)),
    "gain_0_1_linear": lambda base, _rate, _materials: same_percept.gain_db(base, same_percept.GAIN_STEP_DB),
    "delay_0_37_linear": lambda base, _rate, _materials: same_percept.fractional_delay(base, same_percept.FRACTIONAL_DELAY_SAMPLES),
    "fresh_noise_linear": fresh_noise_linear,
}
