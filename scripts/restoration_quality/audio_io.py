"""Audio loading, alignment, cached resampling and windowing shared by every metric family.

Only numpy/scipy/soundfile are imported at module level; librosa is pulled in
inside the resampler so the pure-DSP path (and the unit tests) never need it.
"""

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from modules.auto_scanner import _compute_chunk_spectrum, _music_ratio_from_spectrum, _speech_ratio_from_spectrum

# The router's readings, shared with the app (re-exported for the tests and the calibration).
from modules.tonal_persistence import syllabic_modulation, tonal_persistence  # noqa: F401
from modules.utils import FFMPEG_BIN
from scripts import pair_alignment
from scripts.score_reference import _match_gain

NATIVE_RATE = 44100
SPEECH_RATE = 16000
SIGMOS_RATE = 48000
MIN_TAIL_WINDOW_S = 5.0
SILENCE_DBFS = -50.0
# The polyphase fallback's anti-alias window when librosa (soxr) is not installed (`_resample`).
FALLBACK_WINDOW = ("kaiser", 16.0)
SPEECH_RATIO_MIN = 0.25
MUSIC_RATIO_MIN = 0.20
# The scanner's music ratio reads 0.03-0.06 on the music of a 1991 VHS (Gaudeamus), so the
# router also reads tonal persistence (`modules.tonal_persistence`): the share of spectral
# peaks (100-4000 Hz, 12 dB prominent, 93 ms frames) that hold for the next eight frames.
# Instrumental music 0.02 and up, speech over a music bed 0.005-0.011, dry speech
# 0.000-0.001; speech is told from singing by its syllabic modulation (envelope power
# density at 3-8 Hz over 0.5-3 Hz, 0.18-1.2 against 0.13-0.14 on the instrumental; a steady
# envelope reads 0).
PERSISTENCE_MUSIC = 0.015
PERSISTENCE_MIXED = 0.004
SYLLABIC_MAX_MUSIC = 0.2
# A broken capture (three Internet Archive clips: a +0.49 DC offset with 5 Hz harmonics
# under it) carries almost all its power below 80 Hz; its steady harmonics look like held
# tones to the persistence reading, so such a window is not programme at all.
INFRASONIC_HZ = 80.0
INFRASONIC_SHARE_MAX = 0.9


@dataclass(frozen=True)
class Window:
    """One scoring window on the aligned timeline, in samples of the given rate."""

    index: int
    start_s: float
    end_s: float

    def slice_of(self, rate):
        """The sample slice for this window at `rate`."""
        return slice(int(round(self.start_s * rate)), int(round(self.end_s * rate)))


def file_key(path):
    """A short stable key for a file: sha256 of its resolved path, size and mtime.

    Hashing the bytes of a 30 GB ProRes capture would cost minutes per run; the path plus
    size and modification time identifies the same file just as well for a cache.
    """
    path = Path(path).resolve()
    stat = path.stat()
    digest = hashlib.sha256(f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")).hexdigest()
    return digest[:16]


def extract_wav(source, cache_dir):
    """The source as a 44.1 kHz float WAV: itself when it already is one, else ffmpeg into the cache."""
    source = Path(source)
    if source.suffix.lower() == ".wav":
        return source
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{file_key(source)}_{source.stem[:40]}.wav"
    if target.exists():
        return target
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(NATIVE_RATE),
        str(target),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
    return target


# A capture can carry a DC offset the app's 2 Hz blocker removes; three Internet Archive
# music clips sat at +0.49 with the programme 23-27 dB below it, so every comparison
# against the raw source read the blocker as destruction. The harness blocks DC itself.
DC_SHARE_MAX = 0.01


def load_audio(path):
    """The file as float32 `(frames, channels)` plus its rate, with any DC offset removed per channel."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return remove_dc(audio), int(rate)


def remove_dc(audio):
    """`audio` minus each channel's mean (a constant offset is what a broken capture carries)."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    return (audio - audio.mean(axis=0, keepdims=True)).astype(np.float32)


def dc_share(path):
    """The share of a file's power that is DC, read before any blocking."""
    audio, _rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.size == 0:
        return 0.0
    mono = audio.mean(axis=1)
    return float(np.mean(mono) ** 2 / (np.mean(mono**2) + 1e-20))


def dc_free_copy(path, cache_dir):
    """`path` itself when its DC share is negligible, else a DC-blocked WAV copy under `cache_dir`."""
    path = Path(path)
    if dc_share(path) < DC_SHARE_MAX:
        return path
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{file_key(path)}_dcfree.wav"
    if not target.exists():
        audio, rate = load_audio(path)
        sf.write(str(target), audio, rate, subtype="FLOAT")
    return target


def to_mono(audio):
    """Channel mean, the same downmix for source and output so the pair stays comparable."""
    return np.asarray(audio, dtype=np.float32).mean(axis=1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)


def align_pair(source, output, rate=NATIVE_RATE, method=None, prior_s=None):
    """Source and output on one timeline, the output gain-matched to the source; returns `(source, output, lag)`.

    The sync stage leaves lags of a few hundred samples; every paired metric would read
    that lag as damage. Alignment (`scripts/pair_alignment.py`) and gain matching are the
    primitives the reference scorer uses (`scripts/score_reference`); `aligned_pair` also
    returns the alignment's record (method, status, polarity).
    """
    aligned_source, aligned_output, alignment = aligned_pair(source, output, rate, method, prior_s)
    return aligned_source, aligned_output, alignment.applied_lag


def aligned_pair(source, output, rate=NATIVE_RATE, method=None, prior_s=None):
    """`(source, output, Alignment)` on one timeline, the output gain-matched to the source.

    `method` names the aligner (None: `AI_RESTORE_ALIGN_METHOD` or the default,
    `gcc_phat_spans`); `prior_s` is an expected lag in seconds (the mux's is
    `pair_alignment.MUX_DELAY_S`), which breaks ties and is where an abstaining aligner
    trims the pair (0 without one).
    """
    prior = None if prior_s is None else int(round(prior_s * rate))
    source, output = np.asarray(source, dtype=np.float64), np.asarray(output, dtype=np.float64)
    aligned_source, aligned_output, alignment = pair_alignment.align(source, output, rate, method, prior)
    matched = _match_gain(aligned_source, aligned_output)
    return aligned_source.astype(np.float32), matched.astype(np.float32), alignment


def resample_cached(mono, rate, target_rate, cache_dir, key):
    """`mono` at `target_rate`, resampled once with soxr and kept as a .npy under the cache."""
    if rate == target_rate:
        return np.asarray(mono, dtype=np.float32)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{key}_{target_rate}.npy"
    if cached.exists():
        return np.load(cached)
    resampled = _resample(np.asarray(mono, dtype=np.float32), rate, target_rate)
    np.save(cached, resampled)
    return resampled


def resample(mono, rate, target_rate):
    """`mono` at `target_rate` (`_resample`): the public name the same-percept transforms call."""
    return _resample(mono, rate, target_rate)


def _resample(mono, rate, target_rate):
    """soxr through librosa when it is installed, scipy's polyphase resampler otherwise."""
    try:
        import librosa

        return librosa.resample(mono, orig_sr=rate, target_sr=target_rate, res_type="soxr_hq").astype(np.float32)
    except ImportError:
        from math import gcd

        from scipy.signal import resample_poly

        # A Kaiser window of beta 16 instead of resample_poly's default (beta 5): its images sit near
        # -60 dB, where they land in the hiss readings' noise cells; a CI without the ml group (no
        # librosa) read a soxr-less round trip at -58 dB against the -60 dB its test asks of soxr.
        factor = gcd(rate, target_rate)
        return resample_poly(mono, target_rate // factor, rate // factor, window=FALLBACK_WINDOW).astype(np.float32)


def windows(n_samples, rate, seconds=15.0, hop=7.5):
    """Windows over `n_samples`; a trailing partial window is kept when it is at least 5 s."""
    total_s = n_samples / float(rate)
    out = []
    start = 0.0
    while start < total_s:
        end = min(start + seconds, total_s)
        if end - start >= min(seconds, MIN_TAIL_WINDOW_S):
            out.append(Window(len(out), start, end))
        start += hop
    return out


def level_matched(reference, estimate):
    """`estimate` scaled to `reference`'s RMS over the pair: the level a window's silence is judged at.

    The pair's own gain match is a least-squares projection (`score_reference._match_gain`),
    which falls toward 0 when the output keeps little of the source's waveform: a quiet,
    noise-dominated tape whose noise the restoration removed. On the IA hunt (2026-10-12) it
    scaled outputs playing at -20 dBFS mean under the -50 dBFS silence line, and 8 cathar and
    4 auto_pure_linear clips failed `dsp.output_silent` on audio that was never silent.
    """
    estimate = np.asarray(estimate, dtype=np.float32)
    ref = float(np.sqrt(np.mean(np.square(np.asarray(reference, dtype=np.float64))))) if len(reference) else 0.0
    est = float(np.sqrt(np.mean(np.square(estimate.astype(np.float64))))) if len(estimate) else 0.0
    return estimate if est <= 0.0 else estimate * np.float32(ref / est)


def output_routes(source, output, windows_, rate):
    """Each window's route on the output, read at the source's level (`level_matched`), not the projection's."""
    levelled = level_matched(source, output)
    return [route_window(levelled[w.slice_of(rate)], rate) for w in windows_]


def route_window(mono, rate):
    """speech | music | mixed | silence: the scanner's band ratios, then tonal persistence where they read nothing."""
    rms = float(np.sqrt(np.mean(np.asarray(mono, dtype=np.float64) ** 2)) + 1e-12)
    if 20.0 * np.log10(rms) < SILENCE_DBFS or infrasonic_share(mono, rate) > INFRASONIC_SHARE_MAX:
        return "silence"
    power, freqs = _compute_chunk_spectrum(np.asarray(mono, dtype=np.float32), rate)
    speech = _speech_ratio_from_spectrum(power, freqs)
    music = _music_ratio_from_spectrum(power)
    persistence = tonal_persistence(mono, rate)
    if persistence >= PERSISTENCE_MUSIC and syllabic_modulation(mono, rate) < SYLLABIC_MAX_MUSIC:
        return "music"
    if music >= MUSIC_RATIO_MIN:
        return "music" if speech < SPEECH_RATIO_MIN else "mixed"
    if persistence >= PERSISTENCE_MIXED:
        return "mixed"
    return "speech" if speech >= SPEECH_RATIO_MIN else "mixed"


def infrasonic_share(mono, rate):
    """Share of the window's power below 80 Hz (DC removed)."""
    import scipy.signal

    x = np.asarray(mono, dtype=np.float64)
    x = x - x.mean()
    if len(x) < 1024:
        return 0.0
    freqs, power = scipy.signal.welch(x, rate, nperseg=min(8192, len(x)))
    return float(power[freqs < INFRASONIC_HZ].sum() / (power.sum() + 1e-20))
