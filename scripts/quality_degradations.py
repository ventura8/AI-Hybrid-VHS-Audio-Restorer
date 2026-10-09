"""Controlled, single-factor degradations that mimic the ways a restoration fails.

Each generator takes a mono float32 signal (a realistic-v2 `_target` or `_clean` fixture)
and returns the damaged version, at a level the calibration sweeps. The table at the
bottom names, for every failure, which metric must move and which way, so
`scripts/calibrate_quality_metrics.py` can prove each metric responds before the harness
is trusted to rank restorations.

Ear v3 (plan 1.3) adds the failures the v2 harness could not see and pre-registers, per
degradation, which readings must move and which must NOT (`Expectation.direction` "flat",
with a tolerance or an absolute twin, and "match" for a source condition). The generators
live in `scripts/degradations_v3.py`; their measured effect on the realistic-v2 fixtures
(read with the reading functions directly, 2026-10-09; the calibration re-measures it):

- `air_shelf_*` (the app's own `treble=g:f=7500` at +-0.5 / 1 / 2 dB): on the en speech
  fixture R2's absolute 's' level reads +0.10 / +0.20 / +0.40 (fr +0.60 / +1.21 at 1 / 2
  dB), the net level reading +0.08 / +0.16 / +0.25, the texture under 0.05. R1 reads no air
  on any fixture the way the runner reads it, clipped to R0's programme band: R1's layout
  gets its first air band under a 5292 Hz band (`balance_metrics.readable_from_hz`), and R0
  reads 1.1-5.0 kHz on the five speech targets (the Italian one 5040 Hz) and 1414 Hz on the
  five music beds, and reads the beds right: their programme ends under 1.6 kHz (the 1.6
  kHz band sits 84-94 dB under the loudest, every band from 2 kHz up 106+ dB). Unclipped,
  R1 read the bed's air +0.48 / +0.95 / +1.91 (en): the shelf's own gain on residue 135-138
  dB under the loudest band. Chosen 2026-10-09 (review finding 12, of asserting R1 air
  where a band carries it or demoting the music entry's check to blind): both entries
  assert R1 air, non-blind. On every fixture the check reports "unscored", reason "clipped"
  (`calibration_checks.why_unscored`), and the run names it; with `--excerpts` the speech
  entry runs on the tape cuts (`on_tape`) and asserts where R0's band reaches 5292 Hz: of
  the Tata tapes Vaccin (7127 Hz; SOTI 5040 and Tele7abc 4490 Hz clip it). Read the way the
  runner reads it (R0 clip, mute guard) on the whole cached Vaccin source, this shelf reads
  R1 air +0.135 / +0.27 / +0.54 dB and the cut the mirror, on 30 of 61 windows, with the
  speech benign set's floor at 0.0001 dB: the check passes there. A failing assertion makes
  the run exit 1. (The user's own air shelf read +0.43 / +0.17 / -0.05 for +2 / +1 / off on
  Vaccin, `scripts/tune_grids/tata_v3.yaml`.)
- `spectral_tilt_*` (+-0.5 / 1 / 2 dB/oct above 1 kHz): R1's tilt reads 0.50 / 1.00 / 1.99
  on en speech (de 0.51 / 1.02 / 2.05), presence 0.53 / 1.07 / 2.13, R2's absolute level
  1.07 / 2.13 / 4.27 against the net 0.21 / 0.35 / 0.69.
- `pause_residual_*` (the source's own pauses replaced by a transformed copy at -12 dB,
  speech untouched): on the fr fixture the scaled copy reads attenuation 5.9 / 11.9 / 17.9
  dB with every shape reading 0.00; the +2 / 4 / 6 dB/oct hiss residual slope 1.06 / 2.12 /
  3.17; the low-passed residual (6 / 4 / 3 kHz) HF excess -18.2 / -38.8 / -48.0 on es; the
  islands (30 / 10 / 3 % of cells kept) island kurtosis 0.12 / 0.22 / 0.48 and modulation
  5.3 / 6.2 / 9.6. Lower cut-offs pin the attenuation at R4's 60 dB floor and flatten the
  slope (1 kHz: -7.6 against -12.3 at 2 kHz), and a keep share of 50 % reads the island
  kurtosis negative (-0.09). The en and de fixtures hold no 200 ms non-speech run R4 reads.
- `hifi_compander_mistrack` (3 / 6 / 10 dB): pause depth +2.0 / +4.2 / +7.2 dB, pumping
  +0.30 / +0.70 / +1.33, R4 attenuation +2.4 / +4.9 / +8.4 (es); R1's tilt moves too (es
  +0.30 / +0.61 / +1.02 dB/oct: the sidechain pre-emphasis gives the 's' frames more gain),
  so it is reported, not asserted either way.
- `loudnorm_ride` (30 / 60 / 90 % of the 3 s level swing removed): R7's p95 ride 0.32 /
  0.65 / 0.99 LU (en), 0.35 / 0.70 / 1.06 (fr), sync drift under 0.03 ms. R1 moves on en
  (tilt +0.18 / +0.36 / +0.53: a 15 s window holds frames the rider treats differently), so
  only R2's level and the sync are asserted flat.
- `phasey_resynth` (STFT phases moved by 15 / 40 / 100 % of +-pi): magnitudes are kept per
  frame, yet the overlap-add of incoherent frames reads as dulling on R1 at full
  randomisation (tilt -0.83 en, -1.15 fr) and opens 56-59 holes in `dsp.dropouts`; at
  15-40 % R1 moves under 0.15. Phasiness can pass for a timbre change; the learned judges
  (vetoes) and the speaker cosine carry it.
- `linear_bandwidth` (source low-passed at 10 / 8 / 5 kHz, output an oracle that removed the
  hiss by 20 dB): R1 moves under 0.03 dB. R0 reads the fixtures' own programme band at
  0.7-4.5 kHz (speech and music beds alike), below every cut-off, so on them the check only
  confirms a cut cannot raise the reading; the Tata excerpts (`--excerpts`; PAL linear, about
  8 kHz) carry the real test at 5 kHz, their output the cut capture itself.
- `treble_dropouts` (d = 0.5 / 1 / 2 um): the broadband dropout count stays 0 except one
  window at 2 um on fr (the treble of an 's' gone), R1 reads 0.00 (the events are sparse
  for a window median), R2's texture rises on fr (0.08 / 0.18 / 0.52). R9's flicker reading,
  which these are for, is not built yet (P1).
- `sibilant_islands` and `sync_drift` (the R2 and R7 owners' requests) and the R2 readings on
  the v2 `sibilants_*` pair: measured beside their table entries.
- `--excerpts` in the calibration runs `on_tape()`, the degradations a real excerpt can carry.
"""

import contextlib
from dataclasses import dataclass

import numpy as np
import scipy.ndimage
import scipy.signal

from scripts import degradations_v3 as v3
from scripts import realistic_defects as defects
from scripts.restoration_quality import audio_io, sibilance

WHISTLE_HZ = 15625.0
QUIET_SHARE = 0.2
SIGMOS_NOISE = "mos.sigmos_noise"
SIGMOS_DISC = "mos.sigmos_disc"
COMPRESS_ATTACK_S = 0.01
COMPRESS_RELEASE_S = 0.1
MUTE_SPACING_S = 1.0
# The listener degradations: the polish expander's timing (`compand=attacks=0.04:decays=0.18`),
# its speech test, the 's' bands, and the smear's pre-onset / envelope windows.
GATE_ATTACK_S = 0.04
GATE_RELEASE_S = 0.18
SPEECH_FRAME_S = 0.02
SPEECH_ABOVE_FLOOR = 4.0
UNITY_GAIN_SNAP = 1e-3
SIB_RAMP_S = 0.005
# What the neural stage takes from under an 's': the 1-4 kHz body (APL's fricative centroid
# rose +370..+620 Hz on Tele7abc with the body ratio +14 dB). The synthetic voices' fricatives
# carry little body and a low centroid, so on them the body ratio is the asserted reading
# and the centroid's direction is reported blind (it moved by a few hertz).
SIB_BODY_HZ = (1000.0, 4000.0)
SIB_TOP_HZ = (6000.0, 12000.0)
SMEAR_PRE_S = 0.03
SMEAR_ENVELOPE_S = 0.01


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
    _f, _t, spec = _stft(np.asarray(vhs, dtype=np.float64), rate)
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


def griffin_lim_resynth(clean, iterations):
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


def _follow(rectified, rate, attack_s=COMPRESS_ATTACK_S, release_s=COMPRESS_RELEASE_S):
    """Attack/release envelope follower (one pole each way)."""
    attack = np.exp(-1.0 / (attack_s * rate))
    release = np.exp(-1.0 / (release_s * rate))
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


# --- the listener degradations: what the user heard on the Tata tapes, one at a time ---------


def _frame_level(mono, rate, frame_s):
    """Per-sample RMS level of `mono` on non-overlapping frames (the trailing partial frame reads as the last full one)."""
    frame = max(1, int(frame_s * rate))
    count = len(mono) // frame
    level = np.sqrt(np.mean(np.asarray(mono[: count * frame], dtype=np.float64).reshape(count, frame) ** 2, axis=1))
    per_sample = np.repeat(level, frame)
    return np.concatenate([per_sample, np.full(len(mono) - len(per_sample), level[-1] if count else 0.0)])


def speech_mask(mono, rate):
    """Speech = the 20 ms frames more than four times the p10 frame level (a percentile cut would land inside the pause cluster)."""
    level = _frame_level(mono, rate, SPEECH_FRAME_S)
    return level > SPEECH_ABOVE_FLOOR * np.percentile(level, 10.0)


def expander_gain(mono, rate):
    """The polish expander's gain curve on `mono`: its speech mask through a 40 ms attack / 180 ms release follower.

    The follower only nears unity, so its last 0.1 % (under 0.01 dB) is snapped to 1.0: the
    speech stays bit-identical to the base and only the pauses and the word edges move.
    """
    gain = _follow(speech_mask(mono, rate).astype(np.float64), rate, GATE_ATTACK_S, GATE_RELEASE_S)
    gain[gain > 1.0 - UNITY_GAIN_SNAP] = 1.0
    return gain


def gated_pauses(mono, rate, floor_db):
    """Dead pauses: the expander's gain drops the gaps `floor_db` under the speech, and the word edges pump with it."""
    gain = expander_gain(mono, rate)
    multiplier = gain + (1.0 - gain) * 10.0 ** (floor_db / 20.0)
    return (np.asarray(mono, dtype=np.float64) * multiplier).astype(np.float32)


def hiss_in_pauses(mono, noise, margin_db, rate, rng):
    """Hiss left in the gaps: the tape noise `hiss` adds at `margin_db`, kept only where the expander's gain is closed."""
    added = hiss(mono, noise, margin_db, rng).astype(np.float64) - np.asarray(mono, dtype=np.float64)
    return (np.asarray(mono, dtype=np.float64) + added * (1.0 - expander_gain(mono, rate))).astype(np.float32)


def _bandpass(mono, rate, band):
    """A 4th-order Butterworth band pass, zero phase; the top edge stays under Nyquist."""
    sos = scipy.signal.butter(4, [band[0], min(band[1], 0.49 * rate)], btype="bandpass", fs=rate, output="sos")
    return scipy.signal.sosfiltfilt(sos, np.asarray(mono, dtype=np.float64))


def fricative_ramp(mono, rate):
    """The fricative frames of `mono` as a 0..1 weight with 5 ms edges; exactly 0 away from the 's' bursts."""
    size = max(1, int(SIB_RAMP_S * rate))
    return np.convolve(sibilance.fricative_mask(mono, rate).astype(np.float64), np.ones(size) / size, mode="same")


def _band_lowered(mono, rate, band, depth_db, weight):
    """`mono` with its `band` lowered by `depth_db` where `weight` is 1; bit-identical where it is 0."""
    removed = (1.0 - 10.0 ** (-depth_db / 20.0)) * _bandpass(mono, rate, band) * weight
    return (np.asarray(mono, dtype=np.float64) - removed).astype(np.float32)


def sibilants_thinned(mono, rate, depth_db):
    """The thin 's': the 1-4 kHz body under every fricative lowered by `depth_db` (what APL's neural stage does)."""
    return _band_lowered(mono, rate, SIB_BODY_HZ, depth_db, fricative_ramp(mono, rate))


def sibilants_dulled(mono, rate, depth_db):
    """The dull 's': the 6-12 kHz top of every fricative lowered by `depth_db` (a de-esser biting too hard)."""
    return _band_lowered(mono, rate, SIB_TOP_HZ, depth_db, fricative_ramp(mono, rate))


def _smear_gain(data, rate, onset, span):
    """A raised-cosine rise over `span` samples after `onset`, from the pre-onset level to unity.

    The start is the RMS of the 30 ms before the onset over the peak 10 ms RMS inside the span
    (capped at 1): a hit out of silence is all but removed at its onset and grows back over
    the span; a hit riding on a bed is only rounded off.
    """
    lo = max(0, onset - int(SMEAR_PRE_S * rate))
    before, after = data[lo:onset], data[onset:][:span]
    size = max(1, int(SMEAR_ENVELOPE_S * rate))
    post_peak = np.sqrt(scipy.ndimage.uniform_filter1d(after**2, size, mode="nearest").max())
    start = min(1.0, np.sqrt(np.mean(before**2)) / (post_peak + 1e-12)) if len(before) else 1.0
    return start + (1.0 - start) * 0.5 * (1.0 - np.cos(np.pi * np.arange(len(after)) / span))


def transient_smear(mono, rate, smear_ms):
    """Softened attacks: the `smear_ms` after every onset of `mono` rise from the pre-onset level instead of hitting at once."""
    from scripts.restoration_quality import transient_metrics

    data = np.asarray(mono, dtype=np.float64)
    span = max(1, int(smear_ms * rate / 1000.0))
    gain = np.ones(len(data))
    for onset in (int(o) for o in transient_metrics.onset_samples(data, rate)):
        stop = min(len(data), onset + span)
        gain[onset:stop] = np.minimum(gain[onset:stop], _smear_gain(data, rate, onset, span))
    return (data * gain).astype(np.float32)


@dataclass(frozen=True)
class Expectation:
    """What `metric` must do on this failure; `blind` pairs are reported, never asserted.

    `direction` is "up" or "down" (it must move that way), "flat" (it must not move: within
    `tolerance` or three benign floors, or strictly under the move of its absolute `twin`),
    or "match" (a source reading within `tolerance` octaves of the level the condition built).
    """

    metric: str
    direction: str
    blind: bool = False
    tolerance: float | None = None
    twin: str | None = None


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
        (Expectation("dsp.residual_noise_db", "up"), Expectation(SIGMOS_NOISE, "down"), Expectation("mos.dnsmos_bak", "down", True)),
    ),
    "hum": Degradation("speech", (0.003, 0.01, 0.03), (Expectation("dsp.hum_excess_db", "up"), Expectation(SIGMOS_NOISE, "down", True))),
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
    # Ear v3: R4's island and modulation readings are reported here too. On the fr fixture they
    # do not order the levels (island kurtosis 0.61 / 0.52 / 0.02, modulation 8.7 / 7.4 / 15.2):
    # at alpha 8 the pauses empty to R4's 60 dB floor. The pause_residual islands carry R4.
    "musical_noise": Degradation(
        "vhs",
        (2.0, 4.0, 8.0),
        (
            Expectation("dsp.lkr", "up"),
            Expectation(SIGMOS_DISC, "down", True),
            Expectation("dsp.gap_island_kurt", "up", True),
            Expectation("dsp.gap_mod_dist_db", "up", True),
        ),
    ),
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
            Expectation(SIGMOS_DISC, "down", True),
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
    "clicks": Degradation("speech", (2.0, 8.0, 20.0), (Expectation("dsp.clicks_per_s", "up"), Expectation(SIGMOS_DISC, "down", True))),
    "dropouts": Degradation("speech", (3, 6, 12), (Expectation("dsp.dropouts", "up"), Expectation(SIGMOS_DISC, "down", True))),
    "whistle": Degradation("speech", (-50.0, -40.0, -30.0), (Expectation("dsp.whistle_db", "up"),)),
    "gain": Degradation("speech", (-3.0, -6.0, -12.0), (Expectation("file.lufs", "down"),)),
    "compression": Degradation("speech", (4.0, 8.0, 20.0), (Expectation("file.lra", "down", True),)),
    # The listener degradations (v2): each reproduces one thing the user heard on the Tata tapes.
    "gated_pauses": Degradation(
        "speech",
        (-15.0, -30.0, -45.0),
        (
            Expectation("dsp.pause_depth_db", "up"),
            Expectation("dsp.pause_pumping_db", "up"),
            Expectation("dsp.gap_air_db", "down"),
            Expectation(SIGMOS_DISC, "down", True),
        ),
    ),
    "hiss_in_pauses": Degradation(
        "speech",
        (30.0, 20.0, 10.0),
        (Expectation("dsp.gap_air_db", "up"), Expectation("dsp.residual_noise_db", "up"), Expectation(SIGMOS_NOISE, "down", True)),
    ),
    # The centroid is blind here: the synthetic fricatives carry too little body for removing
    # it to move them (a few hertz), while on tape the same removal read +370..+620 Hz.
    # Ear v3 (R2): a thinned 's' keeps its top, so the absolute level barely moves (en fixture
    # -0.12 / -0.20 / -0.29 dB, fr and it under 0.03) while the body ratio carries it (+2.5 / +4.8
    # / +8.6); a dulled 's' loses its top on both (fr -2.9 / -5.7 / -10.7, en -0.6 / -1.0 / -1.5).
    "sibilants_thinned": Degradation(
        "speech",
        (3.0, 6.0, 12.0),
        (
            Expectation("dsp.sib_centroid_hz", "up", True),
            Expectation("dsp.sib_body_db", "up"),
            Expectation("dsp.sib_abs_level_db", "flat", twin="dsp.sib_body_db"),
        ),
    ),
    "sibilants_dulled": Degradation(
        "speech",
        (3.0, 6.0, 12.0),
        (
            Expectation("dsp.sib_centroid_hz", "down"),
            Expectation("dsp.sib_body_db", "down"),
            Expectation("dsp.hf_8k16k", "down", True),
            Expectation("dsp.sib_abs_level_db", "down"),
        ),
    ),
    "transient_smear": Degradation(
        "music",
        (20.0, 50.0, 120.0),
        (
            Expectation("dsp.attack_db", "down"),
            Expectation("dsp.onset_corr", "down", True),
            Expectation("stems.envelope_corr", "down", True),
        ),
    ),
}

# --- ear v3 (plan 1.3): the degradations the v2 harness could not see -------------------------

SIB_ABS = "dsp.sib_abs_level_db"
SIB_NET = "dsp.sib_level_db"
TILT = "dsp.balance_tilt_db_oct"
# A static filter leaves the texture of the 's' alone: under 0.05 dB on 44 speech fixtures.
TEXTURE_TOLERANCE_DB = 0.05
# A reading the transform cannot touch (speech left bit-identical, or a level kept).
UNTOUCHED_DB = 0.05
SHAPE_TOLERANCE_DB = 0.1
# The plan's benign bound for R1 under oracle hiss removal.
BALANCE_BENIGN_DB = 0.3
# R0's bandwidth rule resolves 1/6 octave on a steep cut (source_profile).
BANDWIDTH_OCTAVES = 1.0 / 6.0
# One window may lose an 's' to a 2 um lift; the hard gate reads a tail of 1.
DROPOUT_COUNT_TOLERANCE = 1.0
# The sync reading on a pair with no time shift: a full phase randomisation smears the envelope
# peak by up to 0.8 ms (en fixture), the compander by 0.08; the sync gates sit at 40 ms.
SYNC_TOLERANCE_MS = 2.0
SYNC_FLAT = Expectation("file.sync_drift_ms", "flat", tolerance=SYNC_TOLERANCE_MS)
# A static filter makes no musical noise: the shelf reads lkr under 1e-4 on every speech and music fixture measured.
NO_ISLANDS = Expectation("dsp.lkr", "flat", tolerance=UNTOUCHED_DB)
AIR_LEVELS = (0.5, 1.0, 2.0)
TILT_LEVELS = (0.5, 1.0, 2.0)
ORACLE_REMOVED_DB = -20.0


def _negated(levels):
    return tuple(-level for level in levels)


def _air_speech(direction):
    """The shelf on speech: R2's absolute level reads it, the net level lags it, the texture stays, R1's air reads it.

    R1's air asserts wherever R0's programme band leaves it a band (5292 Hz up): on the
    `--excerpts` cuts (Vaccin 7127 Hz), not on a fixture, where it reports "clipped".
    """
    return (
        Expectation(SIB_ABS, direction),
        Expectation(SIB_NET, "flat", twin=SIB_ABS),
        Expectation("dsp.sib_texture_db", "flat", tolerance=TEXTURE_TOLERANCE_DB),
        NO_ISLANDS,
        Expectation("dsp.balance_top_db", direction, True),
        Expectation("dsp.balance_air_db", direction),
    )


def _air_music(direction):
    """The shelf on the music bed: R1's air must read it, and no islands appear.

    The realistic-v2 beds end under 1.6 kHz, so under the runner's R0 clip the air check reads
    nothing there and reports why ("clipped"), as the speech entry's does on the fixtures; it
    asserts on a bed whose band holds air.
    """
    return (Expectation("dsp.balance_air_db", direction), NO_ISLANDS, Expectation(TILT, direction, True))


def _tilt(direction, opposite):
    """A tilt above 1 kHz: R1's tilt and presence and R2's absolute level read it, the net level lags."""
    return (
        Expectation(TILT, direction),
        Expectation("dsp.balance_presence_db", direction),
        Expectation(SIB_ABS, direction),
        Expectation(SIB_NET, "flat", twin=SIB_ABS),
        Expectation("dsp.balance_body_db", opposite, True),
    )


def _speech_untouched():
    """The pause residuals leave the speech bit-identical: R1 and R2 must read nothing."""
    return (Expectation(TILT, "flat", tolerance=UNTOUCHED_DB), Expectation(SIB_ABS, "flat", tolerance=UNTOUCHED_DB))


def _shape_kept():
    """The scaled residual keeps the source floor's shape and texture: R4's shape readings must read nothing."""
    names = ("dsp.gap_slope_db_oct", "dsp.gap_spread_db", "dsp.gap_lsd_db", "dsp.gap_mod_dist_db", "dsp.gap_island_kurt")
    return tuple(Expectation(name, "flat", tolerance=SHAPE_TOLERANCE_DB) for name in names)


def _residual(moving, kurt_tolerance=UNTOUCHED_DB):
    """A shaped pause residual: what must move, the island texture kept, the speech untouched."""
    return (*moving, Expectation("dsp.gap_island_kurt", "flat", tolerance=kurt_tolerance), *_speech_untouched())


V3_DEGRADATIONS = {
    "air_shelf_boost": Degradation("speech", AIR_LEVELS, _air_speech("up")),
    "air_shelf_cut": Degradation("speech", _negated(AIR_LEVELS), _air_speech("down")),
    "air_shelf_boost_music": Degradation("music", AIR_LEVELS, _air_music("up")),
    "air_shelf_cut_music": Degradation("music", _negated(AIR_LEVELS), _air_music("down")),
    "spectral_tilt_up": Degradation("speech", TILT_LEVELS, _tilt("up", "down")),
    "spectral_tilt_down": Degradation("speech", _negated(TILT_LEVELS), _tilt("down", "up")),
    "pause_residual_scaled": Degradation(
        "vhs", (-6.0, -12.0, -18.0), (Expectation("dsp.gap_atten_db", "up"), *_shape_kept(), *_speech_untouched())
    ),
    "pause_residual_hiss": Degradation(
        "vhs",
        (2.0, 4.0, 6.0),
        _residual(
            (Expectation("dsp.gap_slope_db_oct", "up"), Expectation("dsp.gap_hf_excess_db", "up"), Expectation("dsp.gap_lsd_db", "up"))
        ),
    ),
    "pause_residual_dull": Degradation(
        "vhs",
        (6000.0, 4000.0, 3000.0),
        _residual(
            (
                Expectation("dsp.gap_hf_excess_db", "down"),
                Expectation("dsp.gap_slope_db_oct", "down"),
                Expectation("dsp.gap_lsd_db", "up"),
            ),
            SHAPE_TOLERANCE_DB,
        ),
    ),
    "pause_residual_islands": Degradation(
        "vhs",
        (0.3, 0.1, 0.03),
        (Expectation("dsp.gap_island_kurt", "up"), Expectation("dsp.gap_mod_dist_db", "up"), *_speech_untouched()),
    ),
    "hifi_compander_mistrack": Degradation(
        "vhs",
        (3.0, 6.0, 10.0),
        (
            Expectation("dsp.pause_depth_db", "up"),
            Expectation("dsp.pause_pumping_db", "up"),
            Expectation("dsp.gap_atten_db", "up"),
            Expectation("dsp.gap_mod_dist_db", "up", True),
            Expectation(TILT, "up", True),
            SYNC_FLAT,
        ),
    ),
    "loudnorm_ride": Degradation(
        "speech",
        (0.3, 0.6, 0.9),
        (
            Expectation("file.gain_ride_lu", "up"),
            Expectation("file.gain_ride_std_lu", "up"),
            Expectation("file.lra", "down", True),
            SYNC_FLAT,
            Expectation(SIB_ABS, "flat", tolerance=UNTOUCHED_DB),
        ),
    ),
    "phasey_resynth": Degradation(
        "speech",
        (0.15, 0.4, 1.0),
        (
            Expectation("speech.speaker_cos", "down"),
            Expectation("dsp.zimtohrli_loud", "up"),
            Expectation("mos.sigmos_col", "down", True),
            Expectation("speech.utmos", "down", True),
            Expectation("dsp.lkr", "up", True),
            Expectation(TILT, "down", True),
            SYNC_FLAT,
        ),
    ),
    "linear_bandwidth": Degradation(
        "vhs",
        (10000.0, 8000.0, 5000.0),
        (
            Expectation("meta.prog_bandwidth_hz", "match", tolerance=BANDWIDTH_OCTAVES),
            Expectation(TILT, "flat", tolerance=BALANCE_BENIGN_DB),
            Expectation("dsp.balance_presence_db", "flat", tolerance=BALANCE_BENIGN_DB),
            Expectation("dsp.balance_air_db", "flat", tolerance=BALANCE_BENIGN_DB),
        ),
    ),
    "treble_dropouts": Degradation(
        "speech",
        (0.5, 1.0, 2.0),
        (
            Expectation("dsp.dropouts", "flat", tolerance=DROPOUT_COUNT_TOLERANCE),
            Expectation(TILT, "flat", tolerance=SHAPE_TOLERANCE_DB),
            Expectation("dsp.hf_flicker_db", "up", True),
            Expectation("dsp.sib_texture_db", "up", True),
        ),
    ),
    # The round-1 "distorted s" hypothesis: a fluctuating mask on the 's' (15 / 30 / 50 % of its
    # 4-12 kHz cells). Texture fr 0.13 / 0.26 / 0.68 (mid01 0.23 / 0.53 / 0.90); en does not order
    # (0.09 / -0.14 / -0.02). The overlap-add loses level (abs -0.1..-2.4 dB), so that is reported.
    "sibilant_islands": Degradation(
        "speech",
        (0.15, 0.3, 0.5),
        (
            Expectation("dsp.sib_texture_db", "up"),
            Expectation(SIB_ABS, "down", True),
            Expectation(TILT, "flat", tolerance=UNTOUCHED_DB),
        ),
    ),
    # A speed error the sync stage missed (the file_metrics sync gate, 40 ms): 0.05 / 0.1 / 0.2 %
    # slow reads drift 5.1 / 10.1 / 19.9 ms and offset 6.3 / 12.5 / 24.8 on 15 s fixtures. R1 reads
    # it as dulling (tilt -0.1 / -0.4 / -0.6 dB/oct en and fr): one lag per 15 s window cannot
    # follow 30 ms of drift, and the fast-moving highs decorrelate first.
    "sync_drift": Degradation(
        "speech",
        (0.05, 0.1, 0.2),
        (
            Expectation("file.sync_drift_ms", "up"),
            Expectation("file.sync_offset_ms", "up", True),
            Expectation("file.gain_ride_lu", "flat", tolerance=SHAPE_TOLERANCE_DB),
            Expectation("file.sync_unmatched", "flat", tolerance=0.0),
            Expectation(TILT, "down", True),
        ),
    ),
}
DEGRADATIONS.update(V3_DEGRADATIONS)
# Planned readings a v3 degradation pre-registers before they exist (R9's flicker is P1).
UNBUILT_READINGS = ("dsp.hf_flicker_db",)


LISTENER = ("gated_pauses", "hiss_in_pauses", "sibilants_thinned", "sibilants_dulled", "transient_smear")
# `identity_music` gives the stem and transient readings a benign floor of their own.
BENIGN = ("identity", "requantise", "resample", "shift_5ms", "shift_30ms", "shift_60ms", "identity_music")
# A shift is benign to every lag-aligned reading, not to the raw pair's: `file_metrics` reads the
# sync on the raw pair, so its lags and offset read the shift itself (5.0 / 30.0 / 60.0 ms), the
# fault the 40 ms `file.sync_offset` gate stands for. The drift (one constant lag) stays benign.
SHIFT_PREFIX = "shift_"
SHIFT_READINGS = ("file.sync_offset_ms", "file.sync_lag_start_ms", "file.sync_lag_middle_ms", "file.sync_lag_end_ms")


def benign_changes(name):
    """The readings a benign transform moves by construction, kept out of the benign floor: a shift's raw-pair lags."""
    return SHIFT_READINGS if name.startswith(SHIFT_PREFIX) else ()


# Degradations a real tape excerpt cannot carry: they need the fixture's own recorded noise, a
# second voice or a separate music bed, which only the fixtures have (`linear_bandwidth` runs
# on a tape with the cut capture as its output: R0's reading of the cut is what it tests there).
NEEDS_REFERENCE = ("hiss", "hiss_in_pauses", "words_spliced", "background_stripped")
TAPE_BASES = ("speech", "vhs")


def on_tape():
    """The degradations a real excerpt (the Tata tapes) can carry: speech or capture based, no reference needed."""
    return tuple(name for name, spec in DEGRADATIONS.items() if spec.base in TAPE_BASES and name not in NEEDS_REFERENCE)


def on_tape_benign():
    """The benign transforms of the speech set, for a real excerpt."""
    return tuple(name for name in BENIGN if benign_base(name) == "speech")


def benign_base(name):
    """The material a benign case starts from: the music bed for the `*_music` cases, the speech target otherwise."""
    return "music" if name.endswith("_music") else "speech"


def _pause_pair(base, rate, residual):
    """`base` and `base` with `residual` in its pauses (speech bit-identical)."""
    return base, v3.in_pauses(base, v3.pause_weight(speech_mask(base, rate), rate), residual)


def _oracle(materials):
    """The fixture's clean programme with its own noise 20 dB down; a real excerpt has no reference, so its capture stands."""
    if "clean" not in materials or "noise" not in materials:
        return materials["vhs"]
    return np.asarray(materials["clean"], dtype=np.float64) + np.asarray(materials["noise"], dtype=np.float64) * 10.0 ** (
        ORACLE_REMOVED_DB / 20.0
    )


def _band_limited_pair(cutoff_hz, rate, materials):
    """The linear-track source condition: the hissy capture and an oracle that took the hiss 20 dB down, both low-passed.

    On a real excerpt the output is the cut capture itself: there only R0's reading of the
    cut (the "match" check) says anything, the balance checks read an identity.
    """
    return v3.band_limited(materials["vhs"], rate, cutoff_hz), v3.band_limited(_oracle(materials), rate, cutoff_hz)


def _air(level, base, rate, _materials, _rng):
    return base, v3.air_shelf(base, rate, level)


def _tilted(level, base, rate, _materials, _rng):
    return base, v3.spectral_tilt(base, rate, level)


def _scaled(level, base, rate, _materials, _rng):
    return _pause_pair(base, rate, v3.scaled_residual(base, level))


def _hissy(level, base, rate, _materials, _rng):
    return _pause_pair(base, rate, v3.hiss_residual(base, rate, level))


def _dull(level, base, rate, _materials, _rng):
    return _pause_pair(base, rate, v3.dull_residual(base, rate, level))


def _islands(level, base, rate, _materials, rng):
    return _pause_pair(base, rate, v3.island_residual(base, rate, level, rng))


def _mistracked(level, base, rate, _materials, _rng):
    return base, v3.compander_mistrack(base, rate, level)


def _ridden(level, base, rate, _materials, _rng):
    return base, v3.loudnorm_ride(base, rate, level)


def _phasey(level, base, rate, _materials, rng):
    return base, v3.phasey_resynth(base, rate, level, rng)


def _bandwidth(level, _base, rate, materials, _rng):
    return _band_limited_pair(level, rate, materials)


def _lifted(level, base, rate, _materials, rng):
    return base, v3.treble_dropouts(base, rate, level, rng)


def _sib_islands(level, base, rate, _materials, rng):
    return base, v3.sibilant_islands(base, rate, level, fricative_ramp(base, rate), rng)


def _drifting(level, base, _rate, _materials, _rng):
    """The slow output cut to the source's length, as the mux keeps it beside the picture."""
    return base, v3.speed_drift(base, level)[: len(base)]


# Builder per ear v3 family; a degradation's name starts with its family (`air_shelf_cut_music`).
V3_BUILDERS = {
    "air_shelf": _air,
    "spectral_tilt": _tilted,
    "pause_residual_scaled": _scaled,
    "pause_residual_hiss": _hissy,
    "pause_residual_dull": _dull,
    "pause_residual_islands": _islands,
    "hifi_compander_mistrack": _mistracked,
    "loudnorm_ride": _ridden,
    "phasey_resynth": _phasey,
    "linear_bandwidth": _bandwidth,
    "treble_dropouts": _lifted,
    "sibilant_islands": _sib_islands,
    "sync_drift": _drifting,
}


def v3_builder(name):
    """The builder of an ear v3 degradation: the family its name starts with."""
    return next(builder for family, builder in V3_BUILDERS.items() if name.startswith(family))


def _hissed(level, base, _rate, materials, rng):
    return base, hiss(base, materials["noise"], level, rng)


def _hummed(level, base, rate, _materials, rng):
    return base, hum(base, rate, level, rng)


def _underwater(level, base, rate, _materials, _rng):
    return base, lowpass(base, rate, level)


V1_BUILDERS = {"hiss": _hissed, "hum": _hummed, "underwater": _underwater}


def apply(name, level, base, rate, materials, rng):
    """The (source, output) pair for one degradation at one level."""
    if name in V3_DEGRADATIONS:
        return v3_builder(name)(level, base, rate, materials, rng)
    if name in LISTENER:
        return _apply_listener(name, level, base, rate, materials, rng)
    if name in V1_BUILDERS:
        return V1_BUILDERS[name](level, base, rate, materials, rng)
    return _apply_rest(name, level, base, rate, materials, rng)


def _apply_listener(name, level, base, rate, materials, rng):
    if name == "gated_pauses":
        return base, gated_pauses(base, rate, level)
    if name == "hiss_in_pauses":
        return base, hiss_in_pauses(base, materials["noise"], level, rate, rng)
    if name == "sibilants_thinned":
        return base, sibilants_thinned(base, rate, level)
    if name == "sibilants_dulled":
        return base, sibilants_dulled(base, rate, level)
    return base, transient_smear(base, rate, level)


def _apply_rest(name, level, base, rate, materials, rng):
    if name == "musical_noise":
        return materials["vhs"], spectral_oversubtract(materials["vhs"], rate, level)
    if name == "robotic":
        return base, griffin_lim_resynth(base, level)
    if name == "words_muted":
        return base, mute_segment(base, rate, 0.3, level)[0]
    if name == "words_spliced":
        return base, splice_from(base, materials["donor"], rate, level)
    return _apply_defects(name, level, base, rate, materials, rng)


def _apply_defects(name, level, base, rate, materials, rng):
    if name == "background_stripped":
        return music_attenuated(materials["voice"], materials["music"], level)
    if name == "clicks":
        return base, crackle(base, rate, level, rng)
    if name == "dropouts":
        return base, dropouts(base, rate, level, rng)
    if name == "whistle":
        return base, whistle(base, rate, level, rng)
    return _apply_level(name, level, base, rate)


def _apply_level(name, level, base, rate):
    if name == "gain":
        return base, gain(base, level)
    return base, compress(base, rate, level)


def apply_benign(name, base, rate, rng):
    """The (source, output) pair for one benign transform: every metric must read nothing."""
    if name.startswith("identity"):
        return base, np.asarray(base, dtype=np.float32).copy()
    if name == "requantise":
        return base, benign_requantise(base, rng)
    if name == "resample":
        return base, benign_resample_roundtrip(base, rate)
    return base, shift(base, rate, float(name.split("_")[1].rstrip("ms")))
