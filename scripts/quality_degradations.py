"""Controlled, single-factor degradations that mimic the ways a restoration fails.

Each generator takes a mono float32 signal (a realistic-v2 `_target` or `_clean` fixture)
and returns the damaged version, at a level the calibration sweeps. The table at the
bottom names, for every failure, which metric must move and which way, so
`scripts/calibrate_quality_metrics.py` can prove each metric responds before the harness
is trusted to rank restorations.
"""

import contextlib
from dataclasses import dataclass

import numpy as np
import scipy.signal

from scripts import realistic_defects as defects
from scripts.restoration_quality import audio_io

WHISTLE_HZ = 15625.0
QUIET_SHARE = 0.2
COMPRESS_ATTACK_S = 0.01
COMPRESS_RELEASE_S = 0.1
MUTE_SPACING_S = 1.0


@contextlib.contextmanager
def _patched(module, **constants):
    """Temporarily rewrites module constants the injectors read at call time."""
    saved = {name: getattr(module, name) for name in constants}
    for name, value in constants.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def hiss(clean, noise, margin_db, rng):
    """Real tape noise under the programme at `margin_db` (from `inject_tape_noise`)."""
    return defects.inject_tape_noise(clean, noise, margin_db, rng=rng).astype(np.float32)


def hum(clean, rate, level, rng):
    return defects.inject_mains_hum(clean, rate, level=level, rng=rng).astype(np.float32)


def lowpass(clean, rate, cutoff_hz):
    """The 'underwater' voice: an 8th-order Butterworth lowpass."""
    sos = scipy.signal.butter(8, cutoff_hz, btype="lowpass", fs=rate, output="sos")
    return scipy.signal.sosfiltfilt(sos, clean).astype(np.float32)


def _stft(mono, rate):
    return scipy.signal.stft(mono, fs=rate, nperseg=1024, noverlap=768)


def _istft(spec, rate, length):
    _t, out = scipy.signal.istft(spec, fs=rate, nperseg=1024, noverlap=768)
    return out[:length].astype(np.float32)


def spectral_oversubtract(vhs, rate, alpha, floor=0.0):
    """Musical noise: subtraction at `alpha` times a noise PSD learned from the quietest frames, no floor."""
    freqs, times, spec = _stft(np.asarray(vhs, dtype=np.float64), rate)
    power = np.abs(spec) ** 2
    frame_level = power.sum(axis=0)
    quiet = frame_level <= np.percentile(frame_level, QUIET_SHARE * 100.0)
    noise_psd = power[:, quiet].mean(axis=1, keepdims=True)
    cleaned = np.maximum(power - alpha * noise_psd, floor * power)
    return _istft(np.sqrt(cleaned) * np.exp(1j * np.angle(spec)), rate, len(vhs))


def random_bin_gate(clean, rate, keep_probability, rng):
    """Chopped spectrum: every time-frequency bin survives with probability `keep_probability`."""
    _f, _t, spec = _stft(np.asarray(clean, dtype=np.float64), rate)
    return _istft(spec * (rng.random(spec.shape) < keep_probability), rate, len(clean))


def griffin_lim_resynth(clean, rate, iterations):
    """The robotic voice: magnitude-only resynthesis with few Griffin-Lim iterations."""
    import librosa

    magnitude = np.abs(librosa.stft(np.asarray(clean, dtype=np.float32), n_fft=1024, hop_length=256))
    out = librosa.griffinlim(magnitude, n_iter=iterations, hop_length=256, n_fft=1024, length=len(clean))
    return out.astype(np.float32)


def _loudest_frames(mono, rate, frame_s=0.1):
    frame = int(frame_s * rate)
    count = len(mono) // frame
    level = np.sqrt(np.mean(np.asarray(mono[: count * frame]).reshape(count, frame) ** 2, axis=1))
    return frame, np.argsort(level)[::-1]


def _spaced_starts(order, frame, count, spacing):
    """The first `count` frame starts in `order` that lie at least `spacing` samples apart."""
    starts = []
    for index in order:
        start = int(index) * frame
        if all(abs(start - other) >= spacing for other in starts):
            starts.append(start)
        if len(starts) == count:
            break
    return starts


def mute_segment(clean, rate, seconds, count):
    """Words removed: `count` spans of `seconds` zeroed at the loudest, mutually distant frames."""
    frame, order = _loudest_frames(clean, rate)
    out = np.asarray(clean, dtype=np.float32).copy()
    starts = _spaced_starts(order, frame, count, MUTE_SPACING_S * rate)
    for start in starts:
        out[start:][: int(seconds * rate)] = 0.0
    return out, [start / rate for start in starts]


def splice_from(clean, donor, rate, seconds):
    """Words replaced: the loudest `seconds` of `clean` overwritten with the loudest `seconds` of `donor`."""
    span = int(seconds * rate)
    frame, order = _loudest_frames(clean, rate)
    start = int(order[0]) * frame
    _frame, donor_order = _loudest_frames(donor, rate)
    donor_start = int(donor_order[0]) * frame
    out = np.asarray(clean, dtype=np.float32).copy()
    piece = np.asarray(donor[donor_start:][:span], dtype=np.float32)
    out[start:][: len(piece)] = piece
    return out


def music_attenuated(voice, music, gain_db):
    """Background stripped: the mix as the source, the same mix with the music `gain_db` lower as the output."""
    length = min(len(voice), len(music))
    voice, music = np.asarray(voice[:length], dtype=np.float32), np.asarray(music[:length], dtype=np.float32)
    return voice + music, voice + music * np.float32(10.0 ** (gain_db / 20.0))


def crackle(clean, rate, per_second, rng):
    with _patched(defects, CRACKLE_PER_SECOND=float(per_second)):
        return defects.inject_crackle(clean, rate, rng).astype(np.float32)


def dropouts(clean, rate, count, rng):
    with _patched(defects, DROPOUT_COUNT=(int(count), int(count) + 1)):
        return defects.inject_dropouts(clean, rate, rng).astype(np.float32)


def whistle(clean, rate, level_db, rng):
    with _patched(defects, WHISTLE_DB=float(level_db)):
        return defects.inject_line_whistle(clean, rate, rng, line_rate_hz=WHISTLE_HZ).astype(np.float32)


def gain(mono, db):
    return (np.asarray(mono, dtype=np.float32) * np.float32(10.0 ** (db / 20.0))).astype(np.float32)


def compress(mono, rate, ratio, threshold_db=-45.0):
    """A feed-forward RMS compressor: squashes the loudness range without moving the timbre."""
    data = np.asarray(mono, dtype=np.float64)
    envelope = _follow(np.abs(data), rate)
    level_db = 20.0 * np.log10(envelope + 1e-9)
    over = np.maximum(level_db - threshold_db, 0.0)
    reduction_db = over - over / ratio
    return (data * 10.0 ** (-reduction_db / 20.0)).astype(np.float32)


def _follow(rectified, rate):
    """Attack/release envelope follower."""
    attack = np.exp(-1.0 / (COMPRESS_ATTACK_S * rate))
    release = np.exp(-1.0 / (COMPRESS_RELEASE_S * rate))
    out = np.empty_like(rectified)
    state = 0.0
    for index, value in enumerate(rectified):
        coefficient = attack if value > state else release
        state = coefficient * state + (1.0 - coefficient) * value
        out[index] = state
    return out


def shift(mono, rate, ms):
    """A benign head delay: the alignment must absorb it."""
    return np.concatenate([np.zeros(int(ms * rate / 1000.0), dtype=np.float32), np.asarray(mono, dtype=np.float32)])


def benign_requantise(mono, rng):
    """16-bit with TPDF dither: audible to nothing, a change to every sample."""
    dither = (rng.random(len(mono)) + rng.random(len(mono)) - 1.0) / 32768.0
    return (np.round((np.asarray(mono, dtype=np.float64) + dither) * 32767.0) / 32767.0).astype(np.float32)


def benign_resample_roundtrip(mono, rate):
    up = audio_io._resample(np.asarray(mono, dtype=np.float32), rate, 48000)
    return audio_io._resample(up, 48000, rate)[: len(mono)]


@dataclass(frozen=True)
class Expectation:
    """`metric` must move `direction` ("up" | "down") on this failure; `blind` pairs are reported, never asserted."""

    metric: str
    direction: str
    blind: bool = False


@dataclass(frozen=True)
class Degradation:
    """A failure mode: its base signal, its levels (mild to severe) and what must move."""

    base: str
    levels: tuple
    expects: tuple


DEGRADATIONS = {
    "hiss": Degradation(
        "speech",
        (40.0, 30.0, 20.0),
        (Expectation("dsp.residual_noise_db", "up"), Expectation("mos.sigmos_noise", "down"), Expectation("mos.dnsmos_bak", "down", True)),
    ),
    "hum": Degradation(
        "speech", (0.003, 0.01, 0.03), (Expectation("dsp.hum_excess_db", "up"), Expectation("mos.sigmos_noise", "down", True))
    ),
    "underwater": Degradation(
        "speech",
        (8000.0, 6000.0, 4000.0),
        (
            Expectation("dsp.hf_4k8k", "down"),
            Expectation("dsp.hf_8k16k", "down"),
            Expectation("mos.sigmos_col", "down"),
            Expectation("speech.utmos", "down"),
            Expectation("mos.audiobox_pq", "down", True),
        ),
    ),
    "musical_noise": Degradation("vhs", (2.0, 4.0, 8.0), (Expectation("dsp.lkr", "up"), Expectation("mos.sigmos_disc", "down", True))),
    "robotic": Degradation(
        "speech",
        (32, 8, 2),
        (Expectation("speech.utmos", "down"), Expectation("mos.sigmos_col", "down"), Expectation("speech.speaker_cos", "down")),
    ),
    "words_muted": Degradation(
        "speech",
        (1, 2, 3),
        (
            Expectation("speech.cer", "up"),
            Expectation("speech.wer", "up"),
            Expectation("dsp.dropouts", "up"),
            Expectation("mos.sigmos_disc", "down", True),
        ),
    ),
    "words_spliced": Degradation(
        "speech", (0.5, 1.0, 2.0), (Expectation("speech.cer", "up"), Expectation("speech.speaker_cos", "down", True))
    ),
    "background_stripped": Degradation(
        "music",
        (-6.0, -12.0, -24.0),
        (
            Expectation("stems.octave_ratio_db", "down"),
            Expectation("stems.si_sdr_db", "down", True),
            Expectation("mos.audiobox_pc", "down", True),
        ),
    ),
    "clicks": Degradation(
        "speech", (2.0, 8.0, 20.0), (Expectation("dsp.clicks_per_s", "up"), Expectation("mos.sigmos_disc", "down", True))
    ),
    "dropouts": Degradation("speech", (3, 6, 12), (Expectation("dsp.dropouts", "up"), Expectation("mos.sigmos_disc", "down", True))),
    "whistle": Degradation("speech", (-50.0, -40.0, -30.0), (Expectation("dsp.whistle_db", "up"),)),
    "gain": Degradation("speech", (-3.0, -6.0, -12.0), (Expectation("file.lufs", "down"),)),
    "compression": Degradation("speech", (4.0, 8.0, 20.0), (Expectation("file.lra", "down", True),)),
}

BENIGN = ("identity", "requantise", "resample", "shift_5ms", "shift_30ms", "shift_60ms")


def apply(name, level, base, rate, materials, rng):
    """The (source, output) pair for one degradation at one level."""
    if name == "hiss":
        return base, hiss(base, materials["noise"], level, rng)
    if name == "hum":
        return base, hum(base, rate, level, rng)
    if name == "underwater":
        return base, lowpass(base, rate, level)
    if name == "musical_noise":
        return materials["vhs"], spectral_oversubtract(materials["vhs"], rate, level)
    return _apply_rest(name, level, base, rate, materials, rng)


def _apply_rest(name, level, base, rate, materials, rng):
    if name == "robotic":
        return base, griffin_lim_resynth(base, rate, level)
    if name == "words_muted":
        return base, mute_segment(base, rate, 0.3, level)[0]
    if name == "words_spliced":
        return base, splice_from(base, materials["donor"], rate, level)
    if name == "background_stripped":
        return music_attenuated(materials["voice"], materials["music"], level)
    return _apply_defects(name, level, base, rate, rng)


def _apply_defects(name, level, base, rate, rng):
    if name == "clicks":
        return base, crackle(base, rate, level, rng)
    if name == "dropouts":
        return base, dropouts(base, rate, level, rng)
    if name == "whistle":
        return base, whistle(base, rate, level, rng)
    if name == "gain":
        return base, gain(base, level)
    return base, compress(base, rate, level)


def apply_benign(name, base, rate, rng):
    """The (source, output) pair for one benign transform: every metric must read nothing."""
    if name == "identity":
        return base, np.asarray(base, dtype=np.float32).copy()
    if name == "requantise":
        return base, benign_requantise(base, rng)
    if name == "resample":
        return base, benign_resample_roundtrip(base, rate)
    return base, shift(base, rate, float(name.split("_")[1].rstrip("ms")))
