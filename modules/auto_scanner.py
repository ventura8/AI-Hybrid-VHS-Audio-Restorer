"""AI acoustic intelligence scanner and adaptive restoration strategy selector.

Analyzes input audio characteristics across speech formants, environmental
textures (birds, cars, ambient soundscapes), musical harmonics, and analog
tape defects to automatically deploy the optimal restoration pipeline.
"""

try:
    import numpy as np
except ImportError:
    np = None

try:
    import soundfile as sf
except ImportError:
    sf = None

from .config import (
    BACKGROUND_MIX_VOL,
    ENHANCE_NFE,
    MAX_ENHANCE_NFE,
    ENHANCE_TAU,
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
from .utils import log_msg

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
    return {
        "mode": "hybrid",
        "reason": "Default multi-source hybrid restoration.",
        "vocals_model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "denoise_model": "UVR-DeNoise-Lite.pth",
        "arnndn_model": "cb.rnnn",
        "enable_preconditioning": True,
        "precondition_filters": {"highpass_hz": 60, "notch_hz": 0.0, "enable_adeclick": True},
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


def _select_strategy_mode(speech_ratio, music_ratio, ambient_ratio, nf_db, has_dialogue=False, onset_periodicity=0.0):
    """Categorizes acoustic profile into optimal restoration engine mode."""
    del nf_db
    # Checked before the dialogue gate: sung vocals trip every speech test, so a
    # music video would otherwise always be routed to stem separation.
    if _is_rhythmic_music(onset_periodicity):
        return "denoise_only", "Sustained rhythmic music; preserving instruments without stem separation."

    if _is_dialogue_present(speech_ratio, has_dialogue):
        return "hybrid", "Dialogue / speech detected with background acoustics."

    if _is_pure_music_or_ambience(speech_ratio, music_ratio, ambient_ratio):
        return "denoise_only", "Non-vocal music / environmental ambience (preserves instruments & textures)."

    return "auto_ffmpeg_native", "Analog tape noise dominant with no dialogue detected."


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
    rumble = profile.get("highpass_hz", 60)
    has_dialogue = profile.get("temporal_profile", {}).get("has_dialogue", speech >= 0.20)
    periodicity = profile.get("onset_periodicity", 0.0)

    mode, reason = _select_strategy_mode(speech, music, ambient, nf_db, has_dialogue, periodicity)
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
    log_msg(f"    - Rationale       : {strategy['reason']}")


def _log_strategy_decision(strategy, executed_mode=None):
    """Prints formatted auto-scanner diagnosis and strategy decision."""
    prof = strategy["profile"]
    log_msg("  [AI Auto-Scanner] Acoustic Profile Analysis:")
    log_msg(f"    - Speech Presence : {prof.get('speech_ratio', 0.0) * 100:.1f}%")
    log_msg(f"    - Music / Harmony : {prof.get('music_ratio', 0.0) * 100:.1f}%")
    log_msg(f"    - Ambient Textures: {prof.get('ambient_ratio', 0.0) * 100:.1f}%")
    log_msg(f"    - Tape Noise Floor: {prof.get('noise_floor_db', -45.0):.1f} dB")
    _log_selected_mode(strategy, executed_mode)
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
