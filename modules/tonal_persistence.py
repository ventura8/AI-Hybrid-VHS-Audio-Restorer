"""Tonal persistence: the reading that tells music from speech on a VHS capture.

The scanner's band ratios cannot: its speech ratio reads 1.0 and its music ratio 0.02-0.25
on every Internet Archive music clip in the tuning set, because a 1991 tape's music sits in
the speech band with the hiss over it. What music has and speech has not is held partials:
the share of prominent spectral peaks (100-4000 Hz, 12 dB over their neighbourhood, 93 ms
frames) still present in each of the next eight frames. Measured as the median over 15 s
windows of a whole file:

- music, with or without singing (11 archive clips): 0.064-0.225
- speech over a music bed (three of the identity clips, Vaccin): 0.007-0.033
- dry dialogue (Tele7abc, SOTI, Campanie, Spitalul, two identity clips): 0.000-0.003

Singing is told from instrumental by syllabic modulation, the envelope's power density at
3-8 Hz over 0.5-3 Hz; a steady envelope reads 0. The output-quality harness routes its
windows on the same two readings (`scripts/restoration_quality/audio_io.py`).
"""

import numpy as np

PERSISTENCE_NPERSEG = 4096
PERSISTENCE_HOLD_FRAMES = 8
PERSISTENCE_PROMINENCE_DB = 12.0
PERSISTENCE_BAND_HZ = (100.0, 4000.0)
SYLLABIC_MIN_DEPTH = 0.05
SYLLABIC_FRAME_S = 0.02
WINDOW_S = 15.0
SILENCE_DBFS = -50.0


def tonal_persistence(mono, rate):
    """Share of prominent spectral peaks (100-4000 Hz) still present in each of the next eight frames."""
    import scipy.signal

    if len(mono) < PERSISTENCE_NPERSEG:
        return 0.0
    freqs, _t, z = scipy.signal.stft(
        np.asarray(mono, dtype=np.float64), fs=rate, nperseg=PERSISTENCE_NPERSEG, noverlap=PERSISTENCE_NPERSEG * 3 // 4
    )
    band = (freqs >= PERSISTENCE_BAND_HZ[0]) & (freqs <= PERSISTENCE_BAND_HZ[1])
    magnitude = 20.0 * np.log10(np.abs(z[band]) + 1e-9)
    peaks = np.zeros(magnitude.shape, dtype=bool)
    for i in range(magnitude.shape[1]):
        idx, _props = scipy.signal.find_peaks(magnitude[:, i], prominence=PERSISTENCE_PROMINENCE_DB)
        peaks[idx, i] = True
    return _held_share(peaks, PERSISTENCE_HOLD_FRAMES)


def _held_share(peaks, hold):
    """Of the peaks in frames that have `hold` successors, the share present in every successor."""
    if peaks.shape[1] <= hold:
        return 0.0
    total = int(peaks[:, :-hold].sum())
    if total == 0:
        return 0.0
    held = sum(int(_held_after(peaks, i, hold).sum()) for i in range(peaks.shape[1] - hold))
    return held / total


def _held_after(peaks, i, hold):
    """Peaks of frame `i` present in each of the `hold` frames after it."""
    lo = i + 1
    hi = lo + hold
    return peaks[:, i] & peaks[:, lo:hi].all(axis=1)


def syllabic_modulation(mono, rate):
    """Envelope energy at 3-8 Hz over 0.5-3 Hz (20 ms envelope): the syllable rate of speech and singing."""
    import scipy.signal

    frame = max(1, int(SYLLABIC_FRAME_S * rate))
    count = len(mono) // frame
    if count < 64:
        return 0.0
    envelope = np.sqrt(np.mean(np.asarray(mono[: count * frame], dtype=np.float64).reshape(count, frame) ** 2, axis=1))
    if envelope.std() < SYLLABIC_MIN_DEPTH * (envelope.mean() + 1e-12):
        return 0.0
    envelope -= envelope.mean()
    f, power = scipy.signal.welch(envelope, fs=1.0 / (frame / rate), nperseg=min(256, count))
    syllabic = power[(f >= 3.0) & (f <= 8.0)].mean()
    slow = power[(f >= 0.5) & (f < 3.0)].mean()
    return float(syllabic / (slow + 1e-12))


def median_persistence(mono, rate, window_s=WINDOW_S):
    """The median tonal persistence over the file's 15 s windows that carry programme, 0.0 when none does.

    A whole file is one reading because the engine's stages run on the whole file; windows
    under -50 dBFS are left out so a tape's leader and pauses do not pull a music tape's
    reading down to speech.
    """
    mono = np.asarray(mono, dtype=np.float32)
    step = max(1, int(window_s * rate))
    readings = [tonal_persistence(_window(mono, start, step), rate) for start in _programme_offsets(mono, step)]
    return float(np.median(readings)) if readings else 0.0


def _window(mono, start, step):
    """The `step` samples from `start`, fewer at the end of the file."""
    stop = start + step
    return mono[start:stop]


def _programme_offsets(mono, step):
    """Window start samples whose window is at least a second long and above the silence floor."""
    offsets = []
    for start in range(0, len(mono), step):
        chunk = np.asarray(_window(mono, start, step), dtype=np.float64)
        if len(chunk) < step // WINDOW_S:
            continue
        if 20.0 * np.log10(float(np.sqrt(np.mean(chunk**2))) + 1e-12) >= SILENCE_DBFS:
            offsets.append(start)
    return offsets
