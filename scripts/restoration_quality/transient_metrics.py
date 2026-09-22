"""Transient readings on music: whether the drum hits kept their edge.

On the Gaudeamus non-speech windows measured during planning, the attack rise of the
percussive part (its 10 ms RMS envelope climbing out of the pre-onset floor) sat under the
source by 4.3..4.6 dB for cathar, 2.0 dB for APL and 1.5 dB for the APL plateau, and the
onset-strength curves correlated with the source's at 0.65..0.74 (cathar) against
0.83..0.86 (APL): cathar rounds the hits off, APL keeps more of them. Both engines also sit
15..20 dB under the source in those windows after the global gain match, so an absolute
attack level would read the gain, not the shape. Every reading here is therefore level-free:
the rise in dB relative to the floor just before the onset, a Pearson correlation, and the
percussive-to-harmonic energy ratio in dB.

The split is a median-filter HPSS on a 2048/512 STFT: the harmonic estimate is the
magnitude median-filtered along time, the percussive estimate the same along frequency,
combined as power (p = 2) soft masks. The onsets are found on the SOURCE and reused on the
output at the same positions, like every paired reading in the harness.
"""

from typing import NamedTuple

import numpy as np
import scipy.ndimage
import scipy.signal

STFT_FRAME = 2048
STFT_HOP = 512
MEDIAN_K = 17
ONSET_REFRACTORY_S = 0.05
PRE_S = 0.03
POST_S = 0.05
ENVELOPE_S = 0.01
MIN_ONSETS = 8
NAMES = ("attack_db", "onset_corr", "percussive_share_db")


class _Side(NamedTuple):
    strength: np.ndarray
    env_db: np.ndarray
    share_db: float


def _stft(mono, rate):
    """Complex STFT, shape (bins, frames); frame k is centred on sample k * STFT_HOP."""
    _freqs, _times, spec = scipy.signal.stft(mono, fs=rate, nperseg=STFT_FRAME, noverlap=STFT_FRAME - STFT_HOP)
    return spec


def _istft(spec, rate, length):
    """Time signal of `spec`, trimmed or zero-padded to `length` samples."""
    _times, signal = scipy.signal.istft(spec, fs=rate, nperseg=STFT_FRAME, noverlap=STFT_FRAME - STFT_HOP)
    out = np.zeros(length)
    kept = min(length, len(signal))
    out[:kept] = signal[:kept]
    return out


def hpss(mono, rate):
    """`(harmonic, percussive)` float64 signals of `len(mono)`: median-filter HPSS with power soft masks."""
    x = np.asarray(mono, dtype=np.float64)
    spec = _stft(x, rate)
    magnitude = np.abs(spec)
    harmonic = scipy.ndimage.median_filter(magnitude, size=(1, MEDIAN_K), mode="nearest") ** 2
    percussive = scipy.ndimage.median_filter(magnitude, size=(MEDIAN_K, 1), mode="nearest") ** 2
    mask = harmonic / (harmonic + percussive + 1e-20)
    return _istft(spec * mask, rate, len(x)), _istft(spec * (1.0 - mask), rate, len(x))


def _power_frames(mono, rate):
    """STFT power of `mono`, shape (frames, bins)."""
    return np.abs(_stft(mono, rate)).T ** 2


def onset_strength(percussive_power):
    """Half-wave-rectified frame-to-frame rise of the log magnitude, summed over bins; the first frame reads 0."""
    log_magnitude = np.log10(np.asarray(percussive_power, dtype=np.float64) + 1e-12)
    rise = np.maximum(np.diff(log_magnitude, axis=0), 0.0).sum(axis=1)
    return np.concatenate([[0.0], rise])


def onset_peaks(strength, rate, hop=STFT_HOP):
    """Frame indices of the local maxima above mean + std, the first of any cluster closer than ONSET_REFRACTORY_S."""
    inner = strength[1:-1]
    above = inner > strength.mean() + strength.std()
    candidates = np.flatnonzero(above & (inner > strength[:-2]) & (inner >= strength[2:])) + 1
    refractory = max(1, int(np.ceil(ONSET_REFRACTORY_S * rate / hop)))
    kept = []
    for frame in candidates:
        if not kept or frame - kept[-1] >= refractory:
            kept.append(int(frame))
    return np.asarray(kept, dtype=int)


def onset_samples(mono, rate):
    """Sample positions of the onsets of `mono` (for the calibration's transient-smear degradation)."""
    _harmonic, percussive = hpss(mono, rate)
    return onset_peaks(onset_strength(_power_frames(percussive, rate)), rate) * STFT_HOP


def envelope_db(percussive, rate):
    """Sliding 10 ms RMS envelope of the percussive part, per sample, in dB."""
    size = max(1, int(ENVELOPE_S * rate))
    rms = np.sqrt(scipy.ndimage.uniform_filter1d(np.asarray(percussive, dtype=np.float64) ** 2, size, mode="nearest"))
    return 20.0 * np.log10(rms + 1e-9)


def attack_rise_db(env_db, onsets, rate, hop):
    """Per onset (STFT frame indices, `hop` samples apart): the envelope's peak in the 50 ms after minus its floor in the 30 ms before."""
    pre, post = int(PRE_S * rate), int(POST_S * rate)
    rises = []
    for centre in np.asarray(onsets, dtype=int) * hop:
        lo, hi = centre - pre, centre + post
        if lo >= 0 and hi <= len(env_db):
            rises.append(float(env_db[centre:hi].max() - env_db[lo:centre].min()))
    return np.asarray(rises)


def _pearson(a, b):
    """Pearson correlation, 0.0 when either side has no variance."""
    a, b = a - a.mean(), b - b.mean()
    norm = np.sqrt((a * a).sum() * (b * b).sum())
    return 0.0 if norm == 0.0 else float((a * b).sum() / norm)


def _side(mono, rate):
    """The onset-strength curve, the percussive envelope and the percussive share of one side."""
    harmonic, percussive = hpss(mono, rate)
    share = 10.0 * np.log10((np.sum(percussive**2) + 1e-20) / (np.sum(harmonic**2) + 1e-20))
    return _Side(onset_strength(_power_frames(percussive, rate)), envelope_db(percussive, rate), float(share))


def _median(values):
    return None if len(values) == 0 else float(np.median(values))


def transient_readings(source, output, rate):
    """`{name: (source, output)}` on the source's onsets; `onset_corr` is (0.0, correlation). Nones when there is too little."""
    length = min(len(source), len(output))
    if length < 4 * STFT_FRAME:
        return {name: (None, None) for name in NAMES}
    src, out = _side(source[:length], rate), _side(output[:length], rate)
    onsets = onset_peaks(src.strength, rate)
    if len(onsets) < MIN_ONSETS:
        return {name: (None, None) for name in NAMES}
    return {
        "attack_db": (
            _median(attack_rise_db(src.env_db, onsets, rate, STFT_HOP)),
            _median(attack_rise_db(out.env_db, onsets, rate, STFT_HOP)),
        ),
        "onset_corr": (0.0, _pearson(src.strength, out.strength)),
        "percussive_share_db": (src.share_db, out.share_db),
    }
