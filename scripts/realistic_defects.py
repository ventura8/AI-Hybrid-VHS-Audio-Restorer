#!/usr/bin/env python3
"""Defect injectors that behave like real tape rather than like clean arithmetic.

The synthetic injector in `scripts/audio_matrix/vhs_defects.py` is deliberately simple and
deterministic, which is right for hardware-validation fixtures but wrong for choosing
restoration stages. Twice now it pointed the opposite way to the real corpus, and the
corpus was right both times:

- Its hum is a pure 50 Hz sine plus one harmonic. Narrow fixed notches remove that
  perfectly, so `auto_pure_linear` looked adequate on hum while attenuating real hum 2.74x
  against `cathar`'s 17.98x. Real mains hum on tape drifts with capstan speed and runs to
  eight or more harmonics.
- Its hiss is white Gaussian noise, which is easier to subtract than tape noise: real hiss
  is spectrally shaped, rises towards the top octaves, and is not stationary.
- Its fixtures sit at one programme level, so nothing exercised the 12-24 dB
  signal-to-noise band where the mode was measurably failing on real material.

These injectors fix all three. Tape noise is sampled from the quietest windows of the real
Internet Archive corpus rather than generated, so what gets subtracted is the real thing.

The physical damage and transport faults below follow the same rule. The shipped injector
puts one click every half second at a fixed height, one dropout every two seconds, and
clips at a fixed gain; a worn tape pops at random with a bright decaying tail, loses head
contact for tens of milliseconds at a time with a real edge, saturates in the recorder's
input stage, wows and flutters with the transport, and sits under a line whistle from the
capture monitor. Each injector here is drawn, not stepped, and each is written to trip the
detector the chain gates its stage on -- because a class whose defect the scanner cannot
see is not exercising the stage at all.

Nothing here changes the shipped injector: hardware-validation fixtures keep the exact
behaviour their tests pin.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.utils import FFMPEG_BIN

TARGET_RATE = 44100

# Mains hum decays across its harmonic series rather than stopping at the second, and the
# whole series drifts together with tape speed. 0.2% is within the wow-and-flutter that
# VHS linear audio specifies, and it is what defeats a fixed narrow notch.
HUM_HARMONICS = 8
HUM_ROLLOFF = 0.8
HUM_DRIFT_FRACTION = 0.002
HUM_DRIFT_RATE_HZ = 0.13


def inject_mains_hum(mono, sample_rate, base_hz=50.0, level=0.02, rng=None):
    """Adds a drifting harmonic hum series, the way a tape carries it."""
    rng = rng or np.random.default_rng(20260909)
    time = np.arange(len(mono), dtype=np.float64) / sample_rate
    # A shared slow drift across every harmonic: integrating the instantaneous frequency
    # keeps the harmonics phase-locked to one another, as a real capstan would.
    drift = 1.0 + HUM_DRIFT_FRACTION * np.sin(2.0 * np.pi * HUM_DRIFT_RATE_HZ * time)
    phase = 2.0 * np.pi * np.cumsum(base_hz * drift) / sample_rate
    hum = np.zeros(len(mono), dtype=np.float64)
    for harmonic in range(1, HUM_HARMONICS + 1):
        amplitude = level / (harmonic**HUM_ROLLOFF)
        hum += amplitude * np.sin(harmonic * phase + rng.uniform(0.0, 2.0 * np.pi))
    return (mono + hum).astype(np.float32)


# What makes a quiet window noise rather than quiet programme. The quietest window of a
# capture is often a fade, a dropout, a muted section or speech at low level: of the first
# bank sampled by level alone, eight of twelve windows had their frame levels spread over
# 8-30 dB, and one was digital silence at -83 dBFS. Selecting for steadiness alone then
# admitted test tones -- a -6 dBFS window with a crest factor of 0.2 dB is a sine, not
# hiss. Noise is stationary, noise-like and quiet, so a window qualifies on all three: the
# spread between its quiet and loud frames, a crest factor Gaussian noise reaches, and a
# level a tape's floor can plausibly sit at. The quietest qualifying window is taken.
NOISE_FRAME = 2048
NOISE_MAX_SPREAD_DB = 6.0
# Gaussian noise over four seconds reaches a crest factor of 13-16 dB; a window at 21-32 dB
# holds a click, and a fixture with crackle in its noise is not a controlled fixture. The
# crackle class injects its own.
NOISE_CREST_DB = (9.0, 18.0)
NOISE_LEVEL_DBFS = (-70.0, -20.0)
# The captures' tones come out of the bank too. The line whine is the capture monitor's,
# not the tape's, and the mains hum is a fault of its own; each has a class that injects
# a controlled one, and a bank that carried them put hum and a whine into every class.
# Half-widths of the notches, in Hz.
NOISE_WHINE_NOTCH_HZ = 40.0
NOISE_HUM_NOTCH_HZ = 3.0
NOISE_HUM_HARMONICS = 8


def _frame_levels_db(samples):
    """Frame RMS levels in dB, at the frame the scanner's floor estimate uses."""
    count = len(samples) // NOISE_FRAME
    frames = samples[: count * NOISE_FRAME].reshape(count, NOISE_FRAME)
    return 20.0 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)


def _is_stationary_noise(window):
    """Whether a window is steady, noise-like and quiet enough to be tape noise."""
    levels = _frame_levels_db(window)
    if len(levels) < 4:
        return False
    spread = float(np.percentile(levels, 90) - np.percentile(levels, 10))
    rms = float(np.sqrt(np.mean(window**2))) + 1e-9
    level = 20.0 * np.log10(rms)
    crest = 20.0 * np.log10(float(np.max(np.abs(window))) / rms + 1e-9)
    steady = spread <= NOISE_MAX_SPREAD_DB
    noise_like = NOISE_CREST_DB[0] <= crest <= NOISE_CREST_DB[1]
    return steady and noise_like and NOISE_LEVEL_DBFS[0] <= level <= NOISE_LEVEL_DBFS[1]


def _without_tones(window, rate):
    """Removes the line whines and the mains series from a noise window, leaving the hiss."""
    spectrum = np.fft.rfft(window.astype(np.float64))
    freqs = np.fft.rfftfreq(len(window), 1.0 / rate)
    for line_rate in WHISTLE_LINE_RATES_HZ:
        spectrum[np.abs(freqs - line_rate) <= NOISE_WHINE_NOTCH_HZ] = 0.0
    for mains in (50.0, 60.0):
        for harmonic in range(1, NOISE_HUM_HARMONICS + 1):
            spectrum[np.abs(freqs - mains * harmonic) <= NOISE_HUM_NOTCH_HZ] = 0.0
    return np.fft.irfft(spectrum, n=len(window)).astype(np.float32)


def _quietest_window(mono, window):
    """Returns the quietest stationary window of a recording, which is its noise, or None."""
    if len(mono) <= window:
        return mono if _is_stationary_noise(mono) else None
    hop = max(window // 2, 1)
    best_start, best_rms = None, None
    for start in range(0, len(mono) - window + 1, hop):
        stop = start + window
        candidate = mono[start:stop]
        rms = float(np.sqrt(np.mean(candidate**2)))
        if (best_rms is None or rms < best_rms) and _is_stationary_noise(candidate):
            best_start, best_rms = start, rms
    if best_start is None:
        return None
    stop = best_start + window
    return mono[best_start:stop]


def _extract_mono(video_path, work_wav):
    """Pulls 44.1 kHz float audio out of a corpus clip."""
    command = [FFMPEG_BIN, "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_f32le", "-ar", str(TARGET_RATE), str(work_wav)]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=120)
    except subprocess.TimeoutExpired:
        # A stalled decode is one capture lost to the bank, not the whole bank.
        return None
    if not work_wav.exists():
        return None
    samples, _rate = sf.read(str(work_wav), dtype="float32", always_2d=True)
    return samples.mean(axis=1)


def build_noise_bank(corpus_dir, catalog, work_dir, count=24, seconds=4.0):
    """Collects real tape noise from the quietest stationary window of each of several captures.

    Sampled rather than generated, because the point is to subtract the noise that actually
    defeats the neural stage on real tapes instead of a Gaussian stand-in. Four seconds
    rather than two so an eight-second fixture repeats its noise once, not three times.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    window = int(seconds * TARGET_RATE)
    bank = []
    # Walked in a fixed shuffled order rather than catalog order, so a bank of 24 drawn from
    # a catalog of 1,300 samples the whole corpus and not its first two searches.
    order = np.random.default_rng(0).permutation(len(catalog))
    for index in order:
        record = catalog[int(index)]
        if len(bank) >= count:
            break
        clip = corpus_dir / record["file"]
        if not clip.exists():
            continue
        probe = work_dir / f"probe_{index}.wav"
        try:
            mono = _extract_mono(clip, probe)
        finally:
            # The decoded capture is read once; kept, a bank of 48 windows left 250 of them.
            probe.unlink(missing_ok=True)
        if mono is None or len(mono) <= window:
            continue
        noise = _quietest_window(mono, window)
        if noise is not None:
            bank.append(_without_tones(noise, TARGET_RATE))
    return bank


# How far a tape's noise floor wanders over a second or two: head contact, tape-path friction
# and the recorder's own gain all move it. Across the quiet frames of 102 real clips the
# per-band level spread (10th to 90th percentile) is 11-14 dB in every band; the bank's
# windows, chosen for steadiness, gave the fixtures 8-10, and a profile learned in one
# window fitted the rest of the fixture better than it ever fits a tape.
NOISE_WOBBLE_DB = 3.0
NOISE_WOBBLE_HZ = (0.3, 1.0)


def _wobble(length, rate, rng, depth_db):
    """A slow gain wander of +/- depth_db: a random phase sum of two low-frequency sines."""
    time = np.arange(length, dtype=np.float64) / rate
    wander = 0.5 * (
        np.sin(2.0 * np.pi * rng.uniform(*NOISE_WOBBLE_HZ) * time + rng.uniform(0, 6.28))
        + np.sin(2.0 * np.pi * rng.uniform(*NOISE_WOBBLE_HZ) * time + rng.uniform(0, 6.28))
    )
    return 10.0 ** (depth_db * wander / 20.0)


def inject_tape_noise(mono, noise, target_margin_db, rng=None, wobble_db=0.0):
    """Adds real tape noise scaled to put the programme a chosen margin above it.

    Targeting the margin rather than a fixed amplitude is what lets a fixture land in the
    12-24 dB band, where measurement on 174 real clips showed the mode failing while every
    synthetic fixture sat comfortably outside it. With `wobble_db` the floor wanders slowly
    about that margin, as a tape's does; the reference fixtures pass nothing and keep their
    exact behaviour.
    """
    rng = rng or np.random.default_rng(7)
    if len(noise) < len(mono):
        repeats = int(np.ceil(len(mono) / len(noise)))
        noise = np.tile(noise, repeats)
    offset = int(rng.integers(0, max(1, len(noise) - len(mono) + 1)))
    stop = offset + len(mono)
    noise = noise[offset:stop]
    if wobble_db > 0.0:
        noise = noise * _wobble(len(mono), TARGET_RATE, rng, wobble_db)

    programme_rms = float(np.sqrt(np.mean(mono**2))) + 1e-12
    noise_rms = float(np.sqrt(np.mean(noise**2))) + 1e-12
    wanted_noise_rms = programme_rms / (10.0 ** (target_margin_db / 20.0))
    return (mono + noise * (wanted_noise_rms / noise_rms)).astype(np.float32)


# --- physical damage and transport faults -------------------------------------------------

CRACKLE_PER_SECOND = 8.0
CRACKLE_LEVEL = (0.15, 0.5)
CRACKLE_TAIL = 64
# Oxide dropouts last 1-50 ms (docs/vhs_audio_defects_research.md 2.7), and the inpainter
# reconstructs up to 50 ms; a first draft at 20-120 ms put most of them past its reach and
# the stage measured as doing nothing. The edge is head contact going, not a fade.
DROPOUT_COUNT = (3, 6)
DROPOUT_MS = (5.0, 50.0)
DROPOUT_EDGE_S = 0.0005
CLIP_CEILING = 0.985
CLIP_DRIVE = 2.0
WHISTLE_DB = -35.0
WHISTLE_LINE_RATES_HZ = (15625.0, 15734.0)
# The scanner reads rumble as the share of power under 70 Hz in the opening 0.37 s alone,
# and raises the highpass once that passes 0.1. Against a speech onset that frame holds,
# -8 dB reads 0.05-0.08 and nothing is seen; this is a transport bad enough to be.
RUMBLE_DB = -5.0
RUMBLE_LOWPASS_HZ = 50.0
RUMBLE_MOTOR_HZ = 30.0
# The scanner calls for warped alignment past 0.3% RMS speed deviation; the linear-track
# specification is 0.2-0.3%, so this is a transport past its specification, as a worn
# deck's is. The 8 Hz flutter averages out inside the scanner's 0.37 s frame and is
# carried for realism rather than detection.
WOW_HZ, WOW_DEPTH = 0.6, 0.005
FLUTTER_HZ, FLUTTER_DEPTH = 8.0, 0.001
AZIMUTH_DELAY_MS = (0.15, 0.35)
IMBALANCE_DB = (-6.0, -3.0)
DC_OFFSET = 0.03
# Hi-Fi head switching: a burst at every field, 50 or 60 a second, worse with tracking.
FIELD_RATES_HZ = (50.0, 60.0)
HEAD_SWITCH_DB = -28.0
HEAD_SWITCH_BURST_S = 0.0004
# Video crosstalk into the audio carrier: a buzz at the field rate and its harmonics, high
# in the band, rising and falling with picture brightness.
CROSSTALK_DB = -30.0
CROSSTALK_BAND_HZ = (2000.0, 8000.0)
CROSSTALK_HARMONICS = 160
# The camcorder: its plastic housing rings between 1.5 and 3.5 kHz, its microphone takes
# air blasts under 150 Hz from close consonants, and handling reaches it as thumps.
# The scanner notches a housing resonance only when it stands 20x over its band's median
# and its half-power hump spans 50-400 Hz -- speech formants live here and every tape has
# a loudest bin -- so the class carries a ring worth notching, +16 dB, broad enough that
# the hump it raises under the voice's harmonics is what the scanner measures rather than
# one boosted harmonic: at a Q of 12 the measured width was 32 Hz and one voice in five
# was read.
RESONANCE_HZ = (1500.0, 3500.0)
RESONANCE_GAIN_DB = 16.0
RESONANCE_Q = 5.0
PLOSIVE_COUNT = (4, 8)
PLOSIVE_MS = (30.0, 80.0)
PLOSIVE_LOWPASS_HZ = 150.0
PLOSIVE_DB = 6.0
HANDLING_COUNT = (3, 6)
HANDLING_MS = (50.0, 200.0)
HANDLING_BAND_HZ = (30.0, 300.0)
HANDLING_DB = -10.0
# What the literature adds to the catalog (IASA TC-04 5.4, the BAVC glossary, the LoC
# sticky-shed study, VHS Hi-Fi service notes). Print-through: the layer above prints a pre-echo
# one wrap ahead; near the hub a wrap is 3.5 s. Squeal: stick-slip of a hydrolysed binder, a
# tone that wanders and modulates the programme. Modulation noise: noise the recording itself
# makes, present only where there is signal. Breathing: the Hi-Fi compander's noise floor
# following the programme with a lag when the carrier is weak. Track switching: a deck
# losing the Hi-Fi carrier and falling back to the linear track, and coming back. Undecoded
# Dolby B: the linear stereo track's encoding left in, quiet passages brighter and hissier.
# Edge damage: the linear track lives on the tape's edge, and a creased edge is a slow
# level wobble with the top going with it. Head clog: the top of the band comes and goes.
# Codec: the archive file is lossy, and so is every real tape this branch measured.
PRINT_THROUGH_DB = -35.0
PRINT_THROUGH_WRAP_S = 3.5
SQUEAL_HZ = (1800.0, 3800.0)
SQUEAL_DB = -22.0
SQUEAL_WANDER_HZ = 3.0
SQUEAL_AM_HZ = 27.0
MODULATION_NOISE_DB = -28.0
BREATHING_DB = 10.0
BREATHING_RELEASE_S = 0.4
TRACK_SWITCH_S = (1.5, 3.0)
TRACK_SWITCH_LOWPASS_HZ = 4000.0
TRACK_SWITCH_NOISE_DB = 12.0
DOLBY_B_SHELF_HZ = 1500.0
DOLBY_B_BOOST_DB = 8.0
EDGE_WOBBLE_HZ = (0.3, 2.0)
EDGE_WOBBLE_DB = 5.0
EDGE_LOWPASS_HZ = (3000.0, 9000.0)
CLOG_LOWPASS_HZ = (1500.0, 8000.0)
CLOG_PERIOD_S = 2.5
CODEC_BITRATE = "96k"
GHOST_DB = -24.0
GHOST_BANDWIDTH_HZ = 3500.0
# Scrape flutter: the tape vibrating against a head, a speed modulation above 100 Hz that
# is heard as roughness or a noise skirt rather than as pitch (AV Artifact Atlas, "Audio
# Scrape Flutter"; the wow-and-flutter measurement standards put it above 100 Hz).
SCRAPE_HZ = (150.0, 300.0)
SCRAPE_DEPTH = 0.0005
# An EMI buzz -- a switching supply, a fluorescent fitting, a ground loop with a sharp
# waveform -- is a mains series that does not roll off the way a transformer hum does:
# harmonics to several kHz at nearly even strength. An eight-harmonic dehum leaves most of it.
BUZZ_HARMONICS = 80
BUZZ_ROLLOFF = 0.3
BUZZ_LEVEL = 0.03


def inject_crackle(mono, rate, rng):
    """Surface crackle: sparse impulses with a short, bright, noisy tail, the way a worn tape pops."""
    out = mono.astype(np.float32).copy()
    peak = max(float(np.max(np.abs(mono))), 1e-9)
    count = int(CRACKLE_PER_SECOND * len(mono) / rate)
    for position in rng.integers(0, max(1, len(mono) - CRACKLE_TAIL), count):
        amplitude = float(rng.uniform(*CRACKLE_LEVEL)) * peak * float(rng.choice([-1.0, 1.0]))
        tail = np.exp(-np.arange(CRACKLE_TAIL) / float(rng.uniform(4.0, 12.0))) * rng.normal(1.0, 0.3, CRACKLE_TAIL)
        out[position : position + CRACKLE_TAIL] += (amplitude * tail).astype(np.float32)
    return out


def inject_dropouts(mono, rate, rng):
    """Head-contact loss: brief silences with a 0.5 ms edge, never in the opening half second."""
    out = mono.astype(np.float32).copy()
    edge = max(int(DROPOUT_EDGE_S * rate), 1)
    ramp = np.linspace(1.0, 0.0, edge, dtype=np.float32)
    margin = rate // 2
    for _ in range(int(rng.integers(*DROPOUT_COUNT))):
        length = int(rng.uniform(*DROPOUT_MS) * rate / 1000.0)
        if len(mono) <= 2 * margin + length:
            break
        start = int(rng.integers(margin, len(mono) - margin - length))
        out[start : start + edge] *= ramp
        out[start + edge : start + length - edge] = 0.0
        out[start + length - edge : start + length] *= ramp[::-1]
    return out


def inject_clipping(mono):
    """The recorder's input stage driven into hard clipping at the ceiling the scanner looks for."""
    peak = max(float(np.max(np.abs(mono))), 1e-9)
    return np.clip(mono / peak * CLIP_DRIVE * CLIP_CEILING, -CLIP_CEILING, CLIP_CEILING).astype(np.float32)


def inject_line_whistle(mono, rate, rng, line_rate_hz=None):
    """A CRT line whistle from the capture monitor, gently amplitude-modulated."""
    line_rate_hz = line_rate_hz or float(rng.choice(WHISTLE_LINE_RATES_HZ))
    time = np.arange(len(mono), dtype=np.float64) / rate
    rms = float(np.sqrt(np.mean(mono**2))) + 1e-12
    level = rms * (10.0 ** (WHISTLE_DB / 20.0)) * np.sqrt(2.0)
    tone = level * (1.0 + 0.1 * np.sin(2.0 * np.pi * 2.0 * time)) * np.sin(2.0 * np.pi * line_rate_hz * time + rng.uniform(0, 6.28))
    return (mono + tone).astype(np.float32)


def inject_rumble(mono, rate, rng):
    """Transport rumble: low-passed noise under a motor tone, well below the speech band."""
    b, a = scipy.signal.butter(4, RUMBLE_LOWPASS_HZ / (rate / 2), "low")
    rumble = scipy.signal.lfilter(b, a, rng.normal(0.0, 1.0, len(mono)))
    time = np.arange(len(mono), dtype=np.float64) / rate
    rumble = rumble / (np.sqrt(np.mean(rumble**2)) + 1e-12) + 0.5 * np.sin(2.0 * np.pi * RUMBLE_MOTOR_HZ * time)
    rms = float(np.sqrt(np.mean(mono**2))) + 1e-12
    return (mono + rumble * rms * (10.0 ** (RUMBLE_DB / 20.0)) / (np.sqrt(np.mean(rumble**2)) + 1e-12)).astype(np.float32)


def inject_wow_and_flutter(mono, rate):
    """Tape-speed modulation: a slow wow and a faster flutter, applied as a time warp."""
    time = np.arange(len(mono), dtype=np.float64) / rate
    speed = 1.0 + WOW_DEPTH * np.sin(2.0 * np.pi * WOW_HZ * time) + FLUTTER_DEPTH * np.sin(2.0 * np.pi * FLUTTER_HZ * time)
    position = np.cumsum(speed) / rate
    return np.interp(position, time, mono).astype(np.float32)


def inject_dc_offset(mono):
    """A digitiser's DC bias."""
    return (mono + DC_OFFSET).astype(np.float32)


def _delayed(mono, rate, delay_ms):
    """A fractional-sample delay by linear interpolation."""
    time = np.arange(len(mono), dtype=np.float64)
    return np.interp(time - delay_ms * rate / 1000.0, time, mono, left=0.0).astype(np.float32)


def assemble_stereo(mono, rate, rng, azimuth=False, imbalance=False):
    """Two channels from one: azimuth skew delays the right channel, imbalance lowers it."""
    right = _delayed(mono, rate, float(rng.uniform(*AZIMUTH_DELAY_MS))) if azimuth else mono.astype(np.float32).copy()
    if imbalance:
        right = right * (10.0 ** (float(rng.uniform(*IMBALANCE_DB)) / 20.0))
    return np.column_stack((mono.astype(np.float32), right.astype(np.float32)))


def _level_of(mono):
    """RMS, guarded."""
    return float(np.sqrt(np.mean(mono**2))) + 1e-12


def inject_head_switching(mono, rate, rng, field_hz=None):
    """Hi-Fi head-switching buzz: a broadband burst at every field, jittered and slowly modulated."""
    field_hz = field_hz or float(rng.choice(FIELD_RATES_HZ))
    out = mono.astype(np.float32).copy()
    burst = max(int(HEAD_SWITCH_BURST_S * rate), 4)
    level = _level_of(mono) * (10.0 ** (HEAD_SWITCH_DB / 20.0)) * np.sqrt(rate / (field_hz * burst))
    period = rate / field_hz
    time = np.arange(len(mono)) / rate
    modulation = 1.0 + 0.4 * np.sin(2.0 * np.pi * 0.3 * time + rng.uniform(0, 6.28))
    for start in np.arange(0.0, len(mono) - burst, period):
        position = min(max(int(start + rng.uniform(-2, 2)), 0), len(mono) - burst)
        out[position : position + burst] += (level * modulation[position] * rng.normal(0.0, 1.0, burst)).astype(np.float32)
    return out


def inject_video_crosstalk(mono, rate, rng, field_hz=None):
    """The video signal in the audio: a buzz at the field rate, high in the band, following brightness."""
    field_hz = field_hz or float(rng.choice(FIELD_RATES_HZ))
    time = np.arange(len(mono), dtype=np.float64) / rate
    buzz = np.zeros(len(mono))
    for harmonic in range(1, CROSSTALK_HARMONICS + 1):
        frequency = field_hz * harmonic
        if CROSSTALK_BAND_HZ[0] <= frequency <= CROSSTALK_BAND_HZ[1]:
            buzz += np.sin(2.0 * np.pi * frequency * time + rng.uniform(0, 6.28)) / harmonic
    b, a = scipy.signal.butter(2, 1.5 / (rate / 2), "low")
    brightness = scipy.signal.lfilter(b, a, rng.normal(0.0, 1.0, len(mono)))
    brightness = 0.5 + 0.5 * (brightness - brightness.min()) / (np.ptp(brightness) + 1e-12)
    buzz *= brightness
    return (mono + buzz * _level_of(mono) * (10.0 ** (CROSSTALK_DB / 20.0)) / _level_of(buzz)).astype(np.float32)


def inject_enclosure_resonance(mono, rate, rng):
    """The camcorder body ringing: a peak between 1.5 and 3.5 kHz on everything it records."""
    centre = float(rng.uniform(*RESONANCE_HZ))
    gain = 10.0 ** (RESONANCE_GAIN_DB / 20.0)
    w0 = 2.0 * np.pi * centre / rate
    alpha = np.sin(w0) / (2.0 * RESONANCE_Q)
    b = [1.0 + alpha * gain, -2.0 * np.cos(w0), 1.0 - alpha * gain]
    a = [1.0 + alpha / gain, -2.0 * np.cos(w0), 1.0 - alpha / gain]
    return scipy.signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], mono).astype(np.float32)


def _bursts(mono, rate, rng, count, length_ms, band_hz, level_db):
    """Band-limited noise bursts with a fast attack and a decaying tail, at random positions."""
    out = mono.astype(np.float32).copy()
    b, a = scipy.signal.butter(2, [band_hz[0] / (rate / 2), band_hz[1] / (rate / 2)], "band")
    level = _level_of(mono) * (10.0 ** (level_db / 20.0))
    for _ in range(int(rng.integers(*count))):
        length = int(rng.uniform(*length_ms) * rate / 1000.0)
        if len(mono) <= length + rate // 4:
            break
        start = int(rng.integers(rate // 8, len(mono) - length - rate // 8))
        burst = scipy.signal.lfilter(b, a, rng.normal(0.0, 1.0, length))
        burst *= np.exp(-np.arange(length) / (0.3 * length)) * level / (_level_of(burst) + 1e-12)
        out[start : start + length] += burst.astype(np.float32)
    return out


def inject_plosives(mono, rate, rng):
    """Air blasts on the microphone from close consonants: loud, short, and under 150 Hz."""
    return _bursts(mono, rate, rng, PLOSIVE_COUNT, PLOSIVE_MS, (10.0, PLOSIVE_LOWPASS_HZ), PLOSIVE_DB)


def inject_handling_noise(mono, rate, rng):
    """Thumps and rubs through the camcorder body: low, longer, and well under the speech."""
    return _bursts(mono, rate, rng, HANDLING_COUNT, HANDLING_MS, HANDLING_BAND_HZ, HANDLING_DB)


def _envelope(mono, rate, seconds):
    """A one-pole level follower with the given release, on the block scale of the gain control."""
    block = max(int(0.001 * rate), 1)
    count = len(mono) // block
    peaks = np.max(np.abs(mono[: count * block]).reshape(count, block), axis=1).astype(np.float64)
    coefficient = np.exp(-0.001 / seconds)
    followed = np.zeros(count)
    level = 0.0
    for index, peak in enumerate(peaks):
        level = max(peak, coefficient * level)
        followed[index] = level
    return np.interp(np.arange(len(mono)), (np.arange(count) + 0.5) * block, followed)


def inject_print_through(mono, rate):
    """The layer wound over this one prints a faint copy: a pre-echo one wrap ahead of the programme.

    On a take shorter than a wrap the echo lands half the take ahead instead, so a short
    fixture still carries the fault rather than silently none of it.
    """
    wrap = int(PRINT_THROUGH_WRAP_S * rate)
    shift = wrap if len(mono) > wrap else max(len(mono) // 2, 1)
    echo = np.zeros_like(mono)
    echo[:-shift] = mono[shift:]
    return (mono + echo * (10.0 ** (PRINT_THROUGH_DB / 20.0))).astype(np.float32)


def inject_squeal(mono, rate, rng):
    """Sticky-shed squeal: a wandering tone from stick-slip at the heads, modulating what it rides on."""
    time = np.arange(len(mono), dtype=np.float64) / rate
    centre = float(rng.uniform(*SQUEAL_HZ))
    wander = centre * (1.0 + 0.03 * np.sin(2.0 * np.pi * SQUEAL_WANDER_HZ * time + rng.uniform(0, 6.28)))
    tone = np.sin(2.0 * np.pi * np.cumsum(wander) / rate) * (0.7 + 0.3 * np.sin(2.0 * np.pi * SQUEAL_AM_HZ * time))
    level = _level_of(mono) * (10.0 ** (SQUEAL_DB / 20.0)) * np.sqrt(2.0)
    modulated = mono * (1.0 - 0.15 * (0.5 + 0.5 * np.sin(2.0 * np.pi * SQUEAL_AM_HZ * time)))
    return (modulated + level * tone).astype(np.float32)


def inject_modulation_noise(mono, rate, rng):
    """Noise the recording process makes: present only with signal, scaled by its level."""
    envelope = _envelope(mono, rate, 0.02)
    noise = rng.normal(0.0, 1.0, len(mono))
    return (mono + noise * envelope * (10.0 ** (MODULATION_NOISE_DB / 20.0))).astype(np.float32)


def inject_breathing(mono, rate, noise, rng):
    """A Hi-Fi compander mistracking: the noise floor swells after the programme, with a lag."""
    envelope = _envelope(mono, rate, BREATHING_RELEASE_S)
    envelope = envelope / (np.max(envelope) + 1e-12)
    if len(noise) < len(mono):
        noise = np.tile(noise, int(np.ceil(len(mono) / len(noise))))
    offset = int(rng.integers(0, max(1, len(noise) - len(mono) + 1)))
    breath = noise[offset : offset + len(mono)] / (_level_of(noise) + 1e-12)
    gain = _level_of(mono) * (10.0 ** ((BREATHING_DB - 40.0) / 20.0))
    return (mono + breath * gain * (1.0 + (10.0 ** (BREATHING_DB / 20.0) - 1.0) * envelope)).astype(np.float32)


def inject_track_switch(mono, rate, noise, rng):
    """The deck loses the Hi-Fi carrier and falls back to the linear track for a stretch, then returns."""
    length = int(rng.uniform(*TRACK_SWITCH_S) * rate)
    if len(mono) <= length + rate:
        return mono.astype(np.float32)
    start = int(rng.integers(rate // 2, len(mono) - length - rate // 2))
    b, a = scipy.signal.butter(4, TRACK_SWITCH_LOWPASS_HZ / (rate / 2), "low")
    linear = scipy.signal.lfilter(b, a, mono[start : start + length])
    if len(noise) < length:
        noise = np.tile(noise, int(np.ceil(length / len(noise))))
    hiss = noise[:length] / (_level_of(noise) + 1e-12) * _level_of(mono) * (10.0 ** ((TRACK_SWITCH_NOISE_DB - 30.0) / 20.0))
    out = mono.astype(np.float32).copy()
    out[start : start + length] = (linear + hiss).astype(np.float32)
    return out


def inject_undecoded_dolby_b(mono, rate):
    """Dolby B encoding left in: quiet passages come back with their top end and hiss boosted."""
    b, a = scipy.signal.butter(2, DOLBY_B_SHELF_HZ / (rate / 2), "high")
    top = scipy.signal.lfilter(b, a, mono)
    envelope = _envelope(mono, rate, 0.1)
    quietness = 1.0 - np.clip(envelope / (np.percentile(envelope, 95) + 1e-12), 0.0, 1.0)
    return (mono + top * (10.0 ** (DOLBY_B_BOOST_DB / 20.0) - 1.0) * quietness).astype(np.float32)


def inject_edge_damage(mono, rate, rng):
    """A creased tape edge under the linear head: the level wobbles slowly and the top goes with it."""
    time = np.arange(len(mono), dtype=np.float64) / rate
    wobble = np.sin(2.0 * np.pi * float(rng.uniform(*EDGE_WOBBLE_HZ)) * time + rng.uniform(0, 6.28))
    gain = 10.0 ** (EDGE_WOBBLE_DB * (wobble - 1.0) / 2.0 / 20.0)
    dull = scipy.signal.lfilter(*scipy.signal.butter(2, EDGE_LOWPASS_HZ[0] / (rate / 2), "low"), mono)
    bright = scipy.signal.lfilter(*scipy.signal.butter(2, EDGE_LOWPASS_HZ[1] / (rate / 2), "low"), mono)
    mix = 0.5 + 0.5 * wobble
    return ((bright * mix + dull * (1.0 - mix)) * gain).astype(np.float32)


def inject_head_clog(mono, rate, rng):
    """A clogging head: the top of the band comes and goes as debris passes the gap."""
    time = np.arange(len(mono), dtype=np.float64) / rate
    phase = 0.5 + 0.5 * np.sin(2.0 * np.pi * time / CLOG_PERIOD_S + rng.uniform(0, 6.28))
    dull = scipy.signal.lfilter(*scipy.signal.butter(4, CLOG_LOWPASS_HZ[0] / (rate / 2), "low"), mono)
    bright = scipy.signal.lfilter(*scipy.signal.butter(2, CLOG_LOWPASS_HZ[1] / (rate / 2), "low"), mono)
    return (bright * phase + dull * (1.0 - phase)).astype(np.float32)


def inject_codec(mono, rate, work_dir):
    """The archive's lossy codec, at the rate the corpus files carry."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    source, coded, decoded = work_dir / "codec_in.wav", work_dir / "codec.m4a", work_dir / "codec_out.wav"
    sf.write(str(source), mono.astype(np.float32), rate, subtype="FLOAT")
    encode = [FFMPEG_BIN, "-y", "-i", str(source), "-c:a", "aac", "-b:a", CODEC_BITRATE, str(coded)]
    decode = [FFMPEG_BIN, "-y", "-i", str(coded), "-acodec", "pcm_f32le", "-ar", str(rate), str(decoded)]
    try:
        for command in (encode, decode):
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=120)
    except subprocess.TimeoutExpired:
        return mono.astype(np.float32)
    if not decoded.exists():
        return mono.astype(np.float32)
    out, _rate = sf.read(str(decoded), dtype="float32", always_2d=True)
    out = out.mean(axis=1)
    if len(out) < len(mono):
        out = np.pad(out, (0, len(mono) - len(out)))
    return out[: len(mono)].astype(np.float32)


def inject_ghost_recording(mono, rate, other):
    """An earlier recording not fully erased: another voice, band-limited, faint, everywhere."""
    if len(other) < len(mono):
        other = np.tile(other, int(np.ceil(len(mono) / len(other))))
    ghost = scipy.signal.lfilter(*scipy.signal.butter(4, GHOST_BANDWIDTH_HZ / (rate / 2), "low"), other[: len(mono)])
    return (mono + ghost * _level_of(mono) * (10.0 ** (GHOST_DB / 20.0)) / (_level_of(ghost) + 1e-12)).astype(np.float32)


def inject_scrape_flutter(mono, rate, rng):
    """Scrape flutter: a fast, shallow speed modulation, heard as roughness rather than pitch."""
    time = np.arange(len(mono), dtype=np.float64) / rate
    speed = 1.0 + SCRAPE_DEPTH * np.sin(2.0 * np.pi * float(rng.uniform(*SCRAPE_HZ)) * time + rng.uniform(0, 6.28))
    return np.interp(np.cumsum(speed) / rate, time, mono).astype(np.float32)


def inject_emi_buzz(mono, rate, rng, base_hz=None):
    """A mains-rate buzz with a flat harmonic series reaching several kHz, drifting like the hum does."""
    base_hz = base_hz or float(rng.choice((50.0, 60.0)))
    time = np.arange(len(mono), dtype=np.float64) / rate
    drift = 1.0 + HUM_DRIFT_FRACTION * np.sin(2.0 * np.pi * HUM_DRIFT_RATE_HZ * time)
    phase = 2.0 * np.pi * np.cumsum(base_hz * drift) / rate
    buzz = np.zeros(len(mono), dtype=np.float64)
    for harmonic in range(1, BUZZ_HARMONICS + 1):
        buzz += np.sin(harmonic * phase + rng.uniform(0.0, 2.0 * np.pi)) / (harmonic**BUZZ_ROLLOFF)
    return (mono + buzz * BUZZ_LEVEL / (np.max(np.abs(buzz)) + 1e-12)).astype(np.float32)
