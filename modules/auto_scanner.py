"""AI acoustic intelligence scanner and adaptive restoration strategy selector.

Analyzes input audio characteristics across speech formants, environmental
textures (birds, cars, ambient soundscapes), musical harmonics, and analog
tape defects to automatically deploy the optimal restoration pipeline.
"""

import importlib.util
import os
import shutil
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    import soundfile as sf
except ImportError:
    sf = None

from .config import (
    APL_NOISEPRINT_DURATION_S,
    APL_TONAL_FLATNESS_MAX,
    AUTO_CATHAR_FLATNESS_MAX,
    AUTO_CATHAR_PROBE_SIMILARITY,
    AUTO_CATHAR_TONAL,
    BACKGROUND_MIX_VOL,
    ENHANCE_NFE,
    ENHANCE_TAU,
    MAX_ENHANCE_NFE,
    SYNC_METHOD,
    VOCAL_MIX_VOL,
)
from .filters import (
    _detect_analog_clipping,
    _detect_click_density,
    _detect_crt_flyback_notch,
    _detect_dc_offset_bias,
    _detect_enclosure_resonance_notch,
    _detect_low_frequency_rumble,
    _detect_mains_buzz_notch,
    _detect_stereo_azimuth_skew,
    _detect_stereo_balance_imbalance,
    _estimate_noise_floor_and_reduction,
    _read_stereo_audio_for_analysis,
)
from .utils import CATHAR_BIN, log_msg

# Upper bound on sliding temporal windows, mirroring the flutter detector cap.
MAX_TEMPORAL_WINDOWS = 60

# PAL and NTSC horizontal line rates. The line whine is recorded onto the tape at
# a fixed physical frequency, so its playback deviation measures tape speed
# directly. Musical pitch moves the dominant spectral peak but never this
# reference, which is what makes content immune from being read as drift.
SPEED_REFERENCE_HZ = (15625.0, 15734.0)
DRIFT_FRAME_LEN = 16384
DRIFT_MAX_FRAMES = 120
DRIFT_MIN_FRAMES = 8
# Reference peaks must clear the median of their own band by this factor. Measured
# in-band on real captures: a tape carrying the whine reads 9.5x (10th pct 5.3x),
# a tape without it 1.1x, white noise 2.8x (90th pct 3.2x), a bare tone 1.4x.
DRIFT_REF_PROMINENCE = 4.0
# Relative RMS speed deviation worth paying for DTW alignment. Standard VHS linear
# audio specifies roughly 0.2-0.3% wow/flutter; a clean measured capture sits at
# 0.0006%, so this trips only on genuinely audible pitch instability.
DRIFT_RATIO_THRESHOLD = 0.003


def _compute_band_energy_ratio(fft_power, freqs, low_hz, high_hz, total_power):
    """Calculates relative spectral power within a specific frequency band."""
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    band_sum = float(np.sum(fft_power[band_mask]))
    return band_sum / (total_power + 1e-9)


def _compute_chunk_spectrum(mono_signal, sr):
    """Computes a single FFT power spectrum and frequency axis for an analysis chunk."""
    chunk = mono_signal[: min(len(mono_signal), sr * 30)]
    fft_power = np.abs(np.fft.rfft(chunk)) ** 2
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)
    return fft_power, freqs


def _speech_ratio_from_spectrum(fft_power, freqs):
    """Derives vocal formant band presence from a precomputed power spectrum."""
    total_power = float(np.sum(fft_power) + 1e-9)
    vocal_ratio = _compute_band_energy_ratio(fft_power, freqs, 300.0, 3400.0, total_power)
    return round(float(np.clip(vocal_ratio * 1.6, 0.0, 1.0)), 2)


def _music_ratio_from_spectrum(fft_power):
    """Derives tonal/harmonic coherence from a precomputed power spectrum."""
    fft_mag = np.sqrt(fft_power)
    total_mag = float(np.sum(fft_mag) + 1e-9)

    # Ratio of top tonal peak energies vs broadband mean energy
    top_peaks = np.sort(fft_mag)[-50:]
    tonal_energy = float(np.sum(top_peaks)) / total_mag
    return round(float(np.clip(tonal_energy * 3.5, 0.0, 1.0)), 2)


def _ambient_ratio_from_spectrum(fft_power, freqs):
    """Derives high-frequency ambient texture energy from a precomputed spectrum."""
    total_power = float(np.sum(fft_power) + 1e-9)
    ambient_ratio = _compute_band_energy_ratio(fft_power, freqs, 4000.0, 16000.0, total_power)
    return round(float(np.clip(ambient_ratio * 4.0, 0.0, 1.0)), 2)


def _estimate_speech_presence_ratio(mono_signal, sr):
    """Estimates active speech presence ratio based on vocal formant band dynamics."""
    if len(mono_signal) < 8192:
        return 0.5

    fft_power, freqs = _compute_chunk_spectrum(mono_signal, sr)
    return _speech_ratio_from_spectrum(fft_power, freqs)


def _estimate_music_harmonic_ratio(mono_signal, sr):
    """Estimates musical harmonic content and tonal coherence."""
    if len(mono_signal) < 8192:
        return 0.2

    fft_power, _ = _compute_chunk_spectrum(mono_signal, sr)
    return _music_ratio_from_spectrum(fft_power)


def _estimate_ambient_texture_ratio(mono_signal, sr):
    """Detects high-frequency ambient textures such as bird chirps and room acoustics."""
    if len(mono_signal) < 8192:
        return 0.1

    fft_power, freqs = _compute_chunk_spectrum(mono_signal, sr)
    return _ambient_ratio_from_spectrum(fft_power, freqs)


def _estimate_tonality(mono_signal, sr):
    """Median spectral flatness in 100-5000 Hz, or None when the signal is too short to frame.

    The same measure the restoration chain gates its subtraction factor on, so the scanner
    reports the number the engines will act on rather than a second opinion of it.
    """
    try:
        from .spectral_denoise import TONALITY_FRAME, tonality_of_signal
    except ImportError:  # pragma: no cover - scipy is a hard dependency of the chain
        return None
    if len(mono_signal) < TONALITY_FRAME * 2:
        return None
    return round(tonality_of_signal(mono_signal, sr), 4)


# The trade metric's frame and percentiles: the quietest fifth of the frames is where a
# noise probe is learned, the loudest three tenths is the programme.
PROBE_FRAME = 1024
PROBE_LOUD_PERCENTILE = 70.0
PROBE_BAND_HZ = (300.0, 3400.0)


def _probe_programme_similarity(mono_signal, sr):
    """How much auto_pure_linear's noise probe resembles the programme, as a correlation.

    The chain learns its noise profile from the quietest 4 s and subtracts it at a factor
    tuned for speech. On a tape with true silence that window is noise and the correlation
    is low; on sustained music with no pauses it is programme, the speech-band shape of the
    probe follows the loud frames', and the subtraction takes programme with the noise.
    None when the recording is too short to hold the probe and a loud stretch beside it.
    """
    from .cathar import _evaluate_quiet_probes

    probe_len = int(APL_NOISEPRINT_DURATION_S * sr)
    frames = len(mono_signal) // PROBE_FRAME
    if frames < 8 or len(mono_signal) <= probe_len * 2:
        return None
    start = _evaluate_quiet_probes(mono_signal, probe_len)
    end = start + probe_len
    loud_shape = _speech_band_shape(_loudest_frames(mono_signal, frames), sr)
    probe_shape = _speech_band_shape(mono_signal[start:end], sr)
    if not (np.any(loud_shape) and np.any(probe_shape)):
        return None
    return round(float(np.corrcoef(loud_shape, probe_shape)[0, 1]), 3)


def _loudest_frames(mono_signal, frames):
    """The frames at or above the loud percentile, joined: the programme the metric reads."""
    blocks = mono_signal[: frames * PROBE_FRAME].reshape(frames, PROBE_FRAME)
    levels = 20.0 * np.log10(np.sqrt(np.mean(blocks**2, axis=1)) + 1e-9)
    return blocks[levels >= np.percentile(levels, PROBE_LOUD_PERCENTILE)].reshape(-1)


def _speech_band_shape(segment, sr):
    """The segment's speech-band log spectrum with its mean removed."""
    import scipy.signal

    freqs, psd = scipy.signal.welch(segment, sr, nperseg=4096)
    band = (freqs >= PROBE_BAND_HZ[0]) & (freqs <= PROBE_BAND_HZ[1])
    spectrum = 10.0 * np.log10(psd[band] + 1e-20)
    return spectrum - spectrum.mean()


def _refine_peak_bin(spectrum, peak_idx):
    """Refines a spectral peak to sub-bin precision by parabolic interpolation."""
    if peak_idx <= 0 or peak_idx >= len(spectrum) - 1:
        return float(peak_idx)

    left = float(spectrum[peak_idx - 1])
    center = float(spectrum[peak_idx])
    right = float(spectrum[peak_idx + 1])
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-12:
        return float(peak_idx)
    return float(peak_idx) + 0.5 * (left - right) / denom


def _reference_band_indices(freqs, ref_hz, bin_width):
    """Selects the spectral bins bracketing a speed-reference frequency."""
    half_width = max(ref_hz * 0.02, 4.0 * bin_width)
    return np.flatnonzero((freqs >= ref_hz - half_width) & (freqs <= ref_hz + half_width))


def _sample_reference_frequency(frame, band_idx, bin_width):
    """Measures one frame's reference frequency and how far it stands above its band."""
    spectrum = np.abs(np.fft.rfft(frame))
    band = spectrum[band_idx]
    peak = int(band_idx[int(np.argmax(band))])
    band_median = float(np.median(band))
    prominence = float(spectrum[peak]) / band_median if band_median > 0.0 else 0.0
    return _refine_peak_bin(spectrum, peak) * bin_width, prominence


def _track_speed_reference(mono_signal, sr, ref_hz):
    """Samples a fixed-frequency speed reference evenly across the whole recording."""
    available = (len(mono_signal) - DRIFT_FRAME_LEN) // DRIFT_FRAME_LEN
    if available < DRIFT_MIN_FRAMES:
        return []

    count = min(available, DRIFT_MAX_FRAMES)
    starts = np.linspace(0, len(mono_signal) - DRIFT_FRAME_LEN, count).astype(int)
    bin_width = float(sr) / DRIFT_FRAME_LEN
    band_idx = _reference_band_indices(np.fft.rfftfreq(DRIFT_FRAME_LEN, 1.0 / sr), ref_hz, bin_width)

    observed = []
    for start in starts:
        end = start + DRIFT_FRAME_LEN
        hz, prominence = _sample_reference_frequency(mono_signal[start:end], band_idx, bin_width)
        if prominence > DRIFT_REF_PROMINENCE:
            observed.append(hz)
    return observed


def _relative_deviation(observed_hz):
    """Relative RMS deviation of a tracked reference, i.e. the wow/flutter figure."""
    if len(observed_hz) < DRIFT_MIN_FRAMES:
        return 0.0

    mean_hz = float(np.mean(observed_hz))
    if mean_hz <= 0.0:
        return 0.0
    return float(np.std(observed_hz)) / mean_hz


def _best_fitting_reference(tracks):
    """Picks the tracked reference whose mean sits nearest its nominal line rate.

    The PAL and NTSC rates are 109 Hz apart while each search band spans +/-2%, so
    both bands lock onto the same peak and frame counts cannot say which standard
    a tape uses. Closeness of the tracked mean to nominal can.
    """
    usable = [(ref_hz, track) for ref_hz, track in tracks if len(track) >= DRIFT_MIN_FRAMES]
    if not usable:
        return []
    return min(usable, key=lambda pair: abs(float(np.mean(pair[1])) / pair[0] - 1.0))[1]


def _measure_tape_speed_deviation(mono_signal, sr):
    """Measures speed deviation using whichever line-rate reference fits best."""
    tracks = [(ref_hz, _track_speed_reference(mono_signal, sr, ref_hz)) for ref_hz in SPEED_REFERENCE_HZ]
    return _relative_deviation(_best_fitting_reference(tracks))


def _detect_flutter_or_pitch_drift(mono_signal, sr):
    """Detects tape wow/flutter speed instability to recommend DTW over linear shift.

    Speed is measured against the PAL/NTSC horizontal line whine recorded on the
    tape rather than against the dominant spectral peak, which tracks musical
    pitch and therefore cannot distinguish content from drift. Recordings with no
    usable reference are reported stable, keeping the fast, artifact-free 'shift'
    alignment as the default.
    """
    if np is None or sr < 32000 or len(mono_signal) < (sr * 4):
        return False
    return bool(_measure_tape_speed_deviation(mono_signal, sr) > DRIFT_RATIO_THRESHOLD)


# Rhythm, not spectrum, is what separates music from conversation. Calibrated on a
# labelled corpus (21 speech tapes, 27 music-video slices from two VHS captures)
# using this exact 30 s measurement window: speech tops out at 0.406, so 0.45
# leaves every speech tape on the separation path while flagging 19/27 music.
# The shipped tonal-peak ratio scored at chance (47% held-out error) because sung
# vocals occupy the same band as speech. A spectral-flatness feature looked far
# better in training (3% error) but collapsed to 68% on a held-out tape - it was
# fitting capture characteristics, not musicality.
ONSET_FRAME_LEN = 2048
ONSET_HOP = 1024
ONSET_MIN_LAG = 4
ONSET_MAX_LAG = 120
MUSIC_PERIODICITY_THRESHOLD = 0.45


def _spectral_flux_envelope(mono_signal, sr):
    """Builds the positive spectral flux envelope used to find onsets."""
    chunk = mono_signal[: min(len(mono_signal), sr * 30)]
    count = (len(chunk) - ONSET_FRAME_LEN) // ONSET_HOP
    if count < ONSET_MAX_LAG:
        return None

    window = np.hanning(ONSET_FRAME_LEN)
    frames = []
    for idx in range(count):
        start = idx * ONSET_HOP
        end = start + ONSET_FRAME_LEN
        frames.append(chunk[start:end] * window)
    magnitudes = np.abs(np.fft.rfft(np.array(frames), axis=1))
    return np.maximum(0.0, np.diff(magnitudes, axis=0)).sum(axis=1)


def _estimate_onset_periodicity(mono_signal, sr):
    """Measures how regularly onsets repeat, i.e. whether the audio has a beat.

    Args:
        mono_signal (numpy.ndarray): Mono audio waveform samples.
        sr (int): Sampling rate in Hz.

    Returns:
        float: Normalised autocorrelation peak of the onset envelope, 0.0 when
            the recording is too short to measure.
    """
    flux = _spectral_flux_envelope(mono_signal, sr)
    if flux is None:
        return 0.0

    centred = flux - float(np.mean(flux))
    tail = len(centred) - 1
    correlation = np.correlate(centred, centred, mode="full")[tail:]
    if correlation[0] <= 0.0:
        return 0.0
    return round(float(np.max(correlation[ONSET_MIN_LAG:ONSET_MAX_LAG] / correlation[0])), 3)


def _pick_optimal_denoise_model(noise_floor_db):
    """Selects between transparent DeNoise-Lite and deep DeNoise for heavy noise."""
    if noise_floor_db >= -35.0:
        return "UVR-DeNoise.pth"
    return "UVR-DeNoise-Lite.pth"


def _pick_optimal_vocals_model(speech_ratio, ambient_ratio):
    """Selects BS-Roformer vs crowd-tuned model based on acoustic scene."""
    if ambient_ratio > 0.40 and speech_ratio > 0.30:
        return "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"
    return "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def _pick_optimal_arnndn_model(rumble_hz, noise_floor_db):
    """Selects specialized RNNoise model based on tape and vehicle noise profile."""
    if rumble_hz >= 60:
        return "sh.rnnn"
    if noise_floor_db >= -35.0:
        return "bd.rnnn"
    return "cb.rnnn"


def _classify_temporal_window(chunk, sr):
    """Classifies acoustic activity inside a single temporal window from one FFT."""
    if len(chunk) < 8192:
        return True, True, False

    fft_power, freqs = _compute_chunk_spectrum(chunk, sr)
    is_speech = _speech_ratio_from_spectrum(fft_power, freqs) >= 0.25
    is_music = _music_ratio_from_spectrum(fft_power) >= 0.20
    is_ambient = _ambient_ratio_from_spectrum(fft_power, freqs) >= 0.25
    return is_speech, is_music, is_ambient


def _scan_temporal_scene_windows(mono_signal, sr, window_sec=5.0, hop_sec=2.5):
    """Pass 1 (Micro): Evaluates sliding temporal windows to map dynamic soundscapes."""
    if len(mono_signal) < int(sr * window_sec) or np is None:
        return {"window_count": 1, "has_dialogue": True, "has_music": False, "has_ambient": False, "dialogue_ratio": 1.0}

    win_len = int(sr * window_sec)
    hop_len = int(sr * hop_sec)
    num_windows = max(1, min((len(mono_signal) - win_len) // hop_len + 1, MAX_TEMPORAL_WINDOWS))

    dialogue, music, ambient = 0, 0, 0
    for idx in range(num_windows):
        start = idx * hop_len
        end = start + win_len
        chunk = mono_signal[start:end]
        s, m, a = _classify_temporal_window(chunk, sr)
        dialogue += int(s)
        music += int(m)
        ambient += int(a)

    return {
        "window_count": num_windows,
        "has_dialogue": dialogue > 0,
        "has_music": music > 0,
        "has_ambient": ambient > 0,
        "dialogue_ratio": round(float(dialogue) / float(num_windows), 2),
    }


def _build_default_strategy():
    """Builds fallback restoration strategy when analysis cannot be performed."""
    mode, reason = _select_strategy_mode(0.5, 0.3, 0.2, -45.0)
    return {
        "mode": mode,
        "reason": f"Audio could not be profiled; {reason}",
        "vocals_model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "denoise_model": "UVR-DeNoise-Lite.pth",
        "arnndn_model": "cb.rnnn",
        "enable_preconditioning": True,
        "precondition_filters": {"highpass_hz": 80, "notch_hz": 0.0, "enable_adeclick": True},
        "enhance_nfe": min(int(ENHANCE_NFE), MAX_ENHANCE_NFE),
        "enhance_tau": ENHANCE_TAU,
        "vocal_mix_vol": VOCAL_MIX_VOL,
        "bg_mix_vol": BACKGROUND_MIX_VOL,
        "sync_method": SYNC_METHOD,
        "profile": {
            "speech_ratio": 0.5,
            "music_ratio": 0.3,
            "ambient_ratio": 0.2,
            "noise_floor_db": -45.0,
            "has_drift": False,
        },
    }


def _is_dialogue_present(speech_ratio, has_dialogue):
    """Checks if speech/dialogue is present either globally or in temporal windows."""
    return speech_ratio >= 0.20 or bool(has_dialogue)


def _is_pure_music_or_ambience(speech_ratio, music_ratio, ambient_ratio):
    """Checks if audio is purely musical or ambient without dialogue."""
    return speech_ratio < 0.15 and (music_ratio >= 0.15 or ambient_ratio >= 0.15)


def _is_rhythmic_music(onset_periodicity):
    """Checks for the sustained regular beat that marks music rather than speech."""
    return onset_periodicity >= MUSIC_PERIODICITY_THRESHOLD


# Every engine `auto` could dispatch, measured on the 136 readable clips of the real-tape
# corpus with scripts/measure_tradeoff.py: median noise removed / programme deviation in
# dB, then how many clips the engine wins on both halves against auto_pure_linear and how
# many it loses on both. The classes are the scanner's own, read on the same clips.
#
#   class            auto_pure_linear  cathar           multipass_auto   denoise_only     auto_ffmpeg_native
#   dialogue (90)    10.10/0.29        6.16/0.31  7/42  -0.25/0.14 2/19  0.33/0.13  4/16  0.76/0.06  3/10
#   rhythmic (45)    11.63/0.36        5.89/0.66  1/28   0.27/0.13 1/9   0.73/0.23  1/12  0.62/0.09  1/5
#   all (136)        10.10/0.31        6.02/0.44  8/70   0.11/0.14 3/28  0.35/0.16  5/28  0.72/0.06  4/15
#
# The three engines the scanner used to run -- the stem engine on dialogue, denoise_only on
# music, auto_ffmpeg_native on tape noise -- leave the noise on the tape. cathar removes it,
# and is beaten on both halves by auto_pure_linear on every class, on 81 sixty-second
# excerpts of 21 local tapes as on the corpus (10.12/0.07 against 6.27/0.11; 44 clips to 4).
#
# What separates the two is the noise probe. auto_pure_linear learns its profile from the
# quietest 4 s and subtracts at a factor tuned for speech, then runs a neural stage; cathar
# learns from the quietest 0.75 s at a gentler factor. Where the tape has true silence the
# 4 s window is noise and auto_pure_linear is ahead on both halves. Where sustained tonal
# programme never pauses, the window is programme, and the subtraction shaves it. Of some
# thirty readings tried on both corpora (experiments/auto_engines_v140/separate.py), that
# one -- tonal, the quietest 4 s shaped like the loud frames, no sustained beat -- is the
# only condition under which cathar deviates less on most clips: 10 of 14 on the corpus
# (0.21 dB against 0.54) and 9 of 15 locally (0.15 against 0.19), each time for 2-3 dB
# less noise removed, and wins-both a wash (4/4 and 2/4). No reading hands cathar a class
# it wins outright; a leave-one-out search for one routes 16 clips and loses 9 of them.
# So auto_pure_linear runs everywhere, cathar takes that tonal condition as a fidelity
# preference (auto_cathar_tonal), and cathar runs wherever the neural denoiser cannot: it
# is the one full-chain engine that needs no model.
ENGINE_EVIDENCE = {
    "rhythmic music": "11.63/0.36 dB against cathar 5.89/0.66 on 45 rhythmic clips",
    "dialogue": "10.10/0.29 dB against cathar 6.16/0.31 on 90 dialogue clips",
    "music or ambience": "10.10/0.31 dB against cathar 6.02/0.44 over 136 clips",
    "tape noise": "10.10/0.31 dB against cathar 6.02/0.44 over 136 clips",
}
TONAL_PROBE_EVIDENCE = "cathar deviates less on 10 of 14 such corpus clips and 9 of 15 local, for 2-3 dB less removal"
# No sustained beat: rhythmic music reads its transients as pauses the probe can use.
TONAL_PERIODICITY_MAX = 0.3
NEURAL_ENGINE = "auto_pure_linear"
DETERMINISTIC_ENGINE = "cathar"
LAST_RESORT_ENGINE = "auto_ffmpeg_native"


def _classify_material(speech_ratio, music_ratio, ambient_ratio, has_dialogue, onset_periodicity):
    """Names what the tape carries: the class the engine choice is reasoned about."""
    # Checked before the dialogue gate: sung vocals trip every speech test, so a
    # music video would otherwise always be read as dialogue.
    if _is_rhythmic_music(onset_periodicity):
        return "rhythmic music", "Sustained rhythmic music"
    if _is_dialogue_present(speech_ratio, has_dialogue):
        return "dialogue", "Dialogue / speech over background acoustics"
    if _is_pure_music_or_ambience(speech_ratio, music_ratio, ambient_ratio):
        return "music or ambience", "Non-vocal music / environmental ambience"
    return "tape noise", "Analog tape noise dominant with no dialogue"


def _neural_denoiser_available():
    """Whether the UVR denoiser can run at all: audio-separator is installed."""
    return importlib.util.find_spec("audio_separator") is not None


def _cathar_available():
    """Whether the cathar binary resolves to a runnable executable.

    A name on PATH resolves through `shutil.which`; a direct path must be a file and, off
    Windows, executable. A directory or an empty setting reads as not installed, so `auto`
    falls back rather than dispatching to a chain that cannot start.
    """
    if not CATHAR_BIN:
        return False
    if shutil.which(CATHAR_BIN) is not None:
        return True
    binary = Path(CATHAR_BIN)
    return binary.is_file() and (os.name == "nt" or os.access(binary, os.X_OK))


def _probe_carries_programme(tonality, probe_similarity, onset_periodicity):
    """Whether the tape is sustained tonal programme with no silence for the 4 s probe."""
    if not AUTO_CATHAR_TONAL or None in (tonality, probe_similarity):
        return False
    tonal = tonality < AUTO_CATHAR_FLATNESS_MAX
    probe_is_programme = probe_similarity >= AUTO_CATHAR_PROBE_SIMILARITY
    return tonal and probe_is_programme and onset_periodicity < TONAL_PERIODICITY_MAX


def _select_strategy_mode(
    speech_ratio, music_ratio, ambient_ratio, nf_db, has_dialogue=False, onset_periodicity=0.0, tonality=None, probe_similarity=None
):
    """Picks the engine for the material, and says which evidence the pick rests on."""
    del nf_db
    material, description = _classify_material(speech_ratio, music_ratio, ambient_ratio, has_dialogue, onset_periodicity)
    cathar = _cathar_available()
    if _neural_denoiser_available():
        if cathar and _probe_carries_programme(tonality, probe_similarity, onset_periodicity):
            return DETERMINISTIC_ENGINE, (
                f"{description}, sustained tonal programme with no silence for the noise probe "
                f"(flatness {tonality:.4f}, probe similarity {probe_similarity:.2f}); {TONAL_PROBE_EVIDENCE}"
            )
        return NEURAL_ENGINE, f"{description}; {NEURAL_ENGINE} leads {DETERMINISTIC_ENGINE}, {ENGINE_EVIDENCE[material]}"
    if cathar:
        return DETERMINISTIC_ENGINE, f"{description}; the neural denoiser is not installed, so the deterministic engine runs"
    return LAST_RESORT_ENGINE, f"{description}; neither the neural denoiser nor {DETERMINISTIC_ENGINE} is installed"


def _tune_adaptive_enhance_tau(nf_db):
    """Dynamically tunes diffusion temperature tau based on noise floor."""
    if nf_db >= -35.0:
        return 0.40
    if nf_db >= -45.0:
        return 0.30
    return 0.25


def evaluate_restoration_strategy(profile):
    """Evaluates acoustic profile and outputs optimal mode, models, and parameters."""
    speech = profile.get("speech_ratio", 0.5)
    music = profile.get("music_ratio", 0.3)
    ambient = profile.get("ambient_ratio", 0.2)
    nf_db = profile.get("noise_floor_db", -45.0)
    rumble = profile.get("highpass_hz", 80)
    has_dialogue = profile.get("temporal_profile", {}).get("has_dialogue", speech >= 0.20)
    periodicity = profile.get("onset_periodicity", 0.0)

    mode, reason = _select_strategy_mode(
        speech, music, ambient, nf_db, has_dialogue, periodicity, profile.get("tonality"), profile.get("probe_similarity")
    )
    sync = "dtw" if profile.get("has_drift", False) else "shift"
    tau = _tune_adaptive_enhance_tau(nf_db)

    return {
        "mode": mode,
        "reason": reason,
        "vocals_model": _pick_optimal_vocals_model(speech, ambient),
        "denoise_model": _pick_optimal_denoise_model(nf_db),
        "arnndn_model": _pick_optimal_arnndn_model(rumble, nf_db),
        "enable_preconditioning": True,
        "precondition_filters": {
            "highpass_hz": rumble,
            "notch_hz": profile.get("notch_hz", 0.0),
            "enable_adeclick": profile.get("has_clicks", True),
            "enable_adeclip": profile.get("has_clipping", False),
            "azimuth_delay_ms": profile.get("azimuth_delay_ms", 0.0),
            "enable_dc_block": profile.get("has_dc_offset", False),
            "balance_db": profile.get("balance_db", 0.0),
            "crt_notch_hz": profile.get("crt_notch_hz", 0.0),
            "resonance_hz": profile.get("resonance_hz", 0.0),
        },
        "enhance_nfe": min(int(ENHANCE_NFE), MAX_ENHANCE_NFE),
        "enhance_tau": tau,
        "vocal_mix_vol": VOCAL_MIX_VOL,
        "bg_mix_vol": BACKGROUND_MIX_VOL,
        "sync_method": sync,
        "profile": profile,
    }


def _extract_profile_from_signal(mono_signal, sr, stereo_signal=None):
    """Extracts all acoustic metrics from audio signal."""
    speech_ratio = _estimate_speech_presence_ratio(mono_signal, sr)
    music_ratio = _estimate_music_harmonic_ratio(mono_signal, sr)
    ambient_ratio = _estimate_ambient_texture_ratio(mono_signal, sr)
    onset_periodicity = _estimate_onset_periodicity(mono_signal, sr)
    tonality = _estimate_tonality(mono_signal, sr)
    probe_similarity = _probe_programme_similarity(mono_signal, sr)
    nf_db, nr_db = _estimate_noise_floor_and_reduction(mono_signal)
    crt_notch = _detect_crt_flyback_notch(mono_signal, sr)
    notch = _detect_mains_buzz_notch(mono_signal, sr, crt_notch)
    hp_freq = _detect_low_frequency_rumble(mono_signal, sr)
    clicks = _detect_click_density(mono_signal)
    has_drift = _detect_flutter_or_pitch_drift(mono_signal, sr)
    has_clip = _detect_analog_clipping(mono_signal)
    azimuth_ms = _detect_stereo_azimuth_skew(stereo_signal, sr) if stereo_signal is not None else 0.0
    has_dc = _detect_dc_offset_bias(mono_signal)
    bal_db = _detect_stereo_balance_imbalance(stereo_signal) if stereo_signal is not None else 0.0
    res_notch = _detect_enclosure_resonance_notch(mono_signal, sr)
    temporal_map = _scan_temporal_scene_windows(mono_signal, sr)

    return {
        "speech_ratio": speech_ratio,
        "music_ratio": music_ratio,
        "ambient_ratio": ambient_ratio,
        "onset_periodicity": onset_periodicity,
        "tonality": tonality,
        "probe_similarity": probe_similarity,
        "noise_floor_db": nf_db,
        "reduction_db": nr_db,
        "notch_hz": notch,
        "highpass_hz": hp_freq,
        "has_clicks": clicks,
        "has_clipping": has_clip,
        "azimuth_delay_ms": azimuth_ms,
        "has_dc_offset": has_dc,
        "balance_db": bal_db,
        "crt_notch_hz": crt_notch,
        "resonance_hz": res_notch,
        "has_drift": has_drift,
        "temporal_profile": temporal_map,
    }


def _log_selected_mode(strategy, executed_mode):
    """Reports the scanner's choice, and whether the caller will actually run it.

    Only `auto` dispatches on this field. `auto_pure` and `multipass_auto` name one
    pipeline and run it regardless, so announcing a target mode they will not honour
    is misleading.
    """
    chosen = strategy["mode"]
    if executed_mode is None or executed_mode == chosen:
        log_msg(f"  [AI Auto-Decision] Target Mode: '{chosen}'")
    else:
        log_msg(f"  [AI Auto-Decision] Best-fit Mode: '{chosen}' (advisory; running '{executed_mode}')")
    material, _separator, verdict = strategy["reason"].partition("; ")
    log_msg(f"    - Rationale       : {material}")
    if verdict:
        log_msg(f"    - Verdict         : {verdict}")


def _describe_rhythm(onset_periodicity):
    """Reads the onset periodicity against the music threshold."""
    verdict = "sustained beat" if _is_rhythmic_music(onset_periodicity) else "no sustained beat"
    return f"onset periodicity {onset_periodicity:.3f} ({verdict})"


def _describe_tonality(tonality):
    """Reads the spectral flatness against the chain's tonal gate."""
    if tonality is None:
        return "not measured"
    verdict = "tonal programme" if tonality < APL_TONAL_FLATNESS_MAX else "broadband programme"
    return f"spectral flatness {tonality:.4f} ({verdict})"


def _describe_probe(probe_similarity):
    """Reads what auto_pure_linear's 4 s noise probe would learn from."""
    if probe_similarity is None:
        return "not measured"
    if probe_similarity >= AUTO_CATHAR_PROBE_SIMILARITY:
        return f"quietest {APL_NOISEPRINT_DURATION_S:g} s carries the programme (similarity {probe_similarity:.2f})"
    return f"quietest {APL_NOISEPRINT_DURATION_S:g} s is noise (similarity {probe_similarity:.2f})"


def _describe_hum(notch_hz):
    """Reads the mains notch the pre-conditioning will apply."""
    return f"{notch_hz:.2f} Hz notch" if notch_hz else "none detected"


def _describe_engines():
    """Names the engines the decision chose between and what each is for."""
    neural = "installed" if _neural_denoiser_available() else "not installed"
    cathar = "installed" if _cathar_available() else "not installed"
    return (
        f"{NEURAL_ENGINE} (default; leads every measured class; neural denoiser {neural}), "
        f"{DETERMINISTIC_ENGINE} (deterministic DSP; sustained tonal programme with no silence, "
        f"or no neural denoiser; {cathar})"
    )


def _log_strategy_decision(strategy, executed_mode=None):
    """Prints formatted auto-scanner diagnosis and strategy decision."""
    prof = strategy["profile"]
    log_msg("  [AI Auto-Scanner] Acoustic Profile Analysis:")
    log_msg(f"    - Speech Presence : {prof.get('speech_ratio', 0.0) * 100:.1f}%")
    log_msg(f"    - Music / Harmony : {prof.get('music_ratio', 0.0) * 100:.1f}%")
    log_msg(f"    - Ambient Textures: {prof.get('ambient_ratio', 0.0) * 100:.1f}%")
    log_msg(f"    - Rhythm          : {_describe_rhythm(prof.get('onset_periodicity', 0.0))}")
    log_msg(f"    - Tonality        : {_describe_tonality(prof.get('tonality'))}")
    log_msg(f"    - Noise Probe     : {_describe_probe(prof.get('probe_similarity'))}")
    log_msg(f"    - Tape Noise Floor: {prof.get('noise_floor_db', -45.0):.1f} dB")
    log_msg(f"    - Mains Hum       : {_describe_hum(prof.get('notch_hz', 0.0))}")
    _log_selected_mode(strategy, executed_mode)
    log_msg(f"    - Engines         : {_describe_engines()}")
    log_msg(f"    - Settings        : Enhance NFE={strategy['enhance_nfe']}, Sync={strategy['sync_method']}")
    log_msg(f"    - Models          : Vocals={strategy.get('vocals_model')}, DeNoise={strategy.get('denoise_model')}")


def scan_and_decide_restoration_strategy(wav_path, executed_mode=None):
    """Scans input audio file and selects best restoration strategy.

    Args:
        wav_path (pathlib.Path): Extracted audio to profile.
        executed_mode (str, optional): The pipeline the caller will actually run.
            Pass it whenever the caller ignores the selected mode, so the log does
            not announce a mode that will not be used.

    Returns:
        dict: Restoration strategy.
    """
    raw_audio, sr = _read_stereo_audio_for_analysis(wav_path)
    if raw_audio is None or sr is None or np is None:
        strategy = _build_default_strategy()
        _log_strategy_decision(strategy, executed_mode)
        return strategy

    mono_signal = np.mean(raw_audio, axis=1) if raw_audio.ndim > 1 else raw_audio
    profile = _extract_profile_from_signal(mono_signal, sr, stereo_signal=raw_audio)
    strategy = evaluate_restoration_strategy(profile)
    _log_strategy_decision(strategy, executed_mode)
    return strategy
