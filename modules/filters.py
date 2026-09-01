"""Native FFmpeg DSP and ARNNDN neural speech filtering module.

Provides filter graph generation, model path resolution, character escaping,
and robust process execution for VHS native DSP restoration and pre-conditioning:
- Stage 1: Hardware DC Offset Bias Removal (2 Hz sub-audible highpass filter).
- Stage 2: Stereo Channel Balance Auto-Leveling (Left/Right channel RMS balancing).
- Stage 3: Stereo Azimuth Delay & Phase Alignment (inter-channel cross-correlation delay).
- Stage 4: Analog Preamp De-Clipping (peak clipping threshold reconstruction via adeclip).
- Stage 5: Impulsive Pop/Click Suppression (adeclick) and Motor Rumble Highpass (45-75 Hz).
- Stage 6: Fundamental & Harmonic Mains Buzz Rejection (50/60/100/120 Hz notch filters).
- Stage 7: CRT Horizontal Flyback Line Whine Notch (15.625 kHz PAL / 15.734 kHz NTSC).
- Stage 8: Camcorder Plastic Enclosure Acoustic Resonance Notching (1.5 kHz - 3.5 kHz).
- Stage 9: Fast Fourier Spectral Denoising (afftdn) or Neural RNNoise (arnndn).
"""

import hashlib
import http.client
import urllib.parse
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    import soundfile as sf
except ImportError:
    sf = None

from .cathar import (
    _cathar_analog_repair_pass,
    _cathar_azimuth_step,
    _cathar_clean_transients,
    _cathar_declick_step,
    _cathar_declip_step,
    _cathar_decrackle_step,
    _cathar_deesser_step,
    _cathar_dehum_step,
    _cathar_denoise_step,
    _cathar_deplosive_step,
    _cathar_dereverb_step,
    _cathar_dewind_step,
    _cathar_dewow_step,
    _cathar_enhance_step,
    _cathar_inpaint_step,
    _cathar_mono_below_step,
    _cathar_noiseprint_step,
    _cathar_polish_pass,
    _cathar_precondition_pass,
    _cathar_repair_pass,
    _cathar_repair_step,
    _promote_cathar_tmp,
    _run_cathar_step,
    filter_cathar_vhs_pipeline,
)
from .config import (
    AFFTDN_NF,
    AFFTDN_NR,
    AFFTDN_TN,
    ARNNDN_ENABLE_ADECLICK,
    ARNNDN_HIGHPASS_FREQ,
    ARNNDN_MODEL,
    ENABLE_ADECLICK,
    ENABLE_DYNAMIC_EXPANDER,
    ENABLE_LINEAR_AIR,
    HIGHPASS_FREQ,
    LINEAR_AIR_GAIN_DB,
    NOTCH_FREQ,
)
from .hardware import CPU_THREADS
from .utils import (
    FFMPEG_BIN,
    MODELS_DIR,
    attempt_cpu_run_with_retry,
    ffmpeg_has_filter,
    is_valid_audio,
    log_msg,
)

__all__ = [
    "_cathar_analog_repair_pass",
    "_cathar_azimuth_step",
    "_cathar_clean_transients",
    "_cathar_declick_step",
    "_cathar_declip_step",
    "_cathar_decrackle_step",
    "_cathar_deesser_step",
    "_cathar_dehum_step",
    "_cathar_denoise_step",
    "_cathar_deplosive_step",
    "_cathar_dereverb_step",
    "_cathar_dewind_step",
    "_cathar_dewow_step",
    "_cathar_enhance_step",
    "_cathar_inpaint_step",
    "_cathar_mono_below_step",
    "_cathar_noiseprint_step",
    "_cathar_polish_pass",
    "_cathar_precondition_pass",
    "_cathar_repair_pass",
    "_cathar_repair_step",
    "_promote_cathar_tmp",
    "_run_cathar_step",
    "build_post_denoise_cleanup_filter",
    "build_pre_denoise_surgical_filter",
    "build_full_audio_polish_filter",
    "filter_cathar_vhs_pipeline",
]

ARNNDN_REMOTE_HOST = "raw.githubusercontent.com"
ARNNDN_REMOTE_PATH_PREFIX = "/GregorR/rnnoise-models/master"

ARNNDN_MODEL_SUBDIR_MAP = {
    "cb.rnnn": "conjoined-burgers-2018-08-28/cb.rnnn",
    "bd.rnnn": "beguiling-drafter-2018-08-30/bd.rnnn",
    "lq.rnnn": "leavened-quisling-2018-08-31/lq.rnnn",
    "mp.rnnn": "marathon-prescription-2018-08-29/mp.rnnn",
    "sh.rnnn": "somnolent-hogwash-2018-09-01/sh.rnnn",
}

ARNNDN_MODEL_SHA256 = {
    "cb.rnnn": "f1357c4e5be9dee8467bead486dfced2d75b640c26ad0b594fa7f102322371d9",
    "bd.rnnn": "ae3f7411e1e6a884f839a4a145c394408398f09854dbc1216ee02faafc98a17b",
    "lq.rnnn": "1957528b752799fddf06270bc5469af7cf54c3badc358544ae2abed730943ff9",
    "mp.rnnn": "4e84a448a4baf937992aaf4d10c8258007ec5d24219b6647dfd5fb4b563ad231",
    "sh.rnnn": "70bb6685eb0c2a1d18e2918dca3fbfbd39317010b1802eb1b6ea73a92f3fdec0",
}


# Mains hum is often dominated by its second harmonic, so the detector can pick
# 100/120 Hz. Notching only that leaves the fundamental untouched, so map back.
MAINS_FUNDAMENTAL_BY_HARMONIC = {100.0: 50.0, 120.0: 60.0}
MAINS_FUNDAMENTALS = (50.0, 59.94, 60.0)
AZIMUTH_SAMPLE_RATE = 44100


def _append_mains_notches(filters, notch_freq):
    """Appends mains notches covering both the fundamental and its second harmonic.

    Args:
        filters (list): Accumulator list of FFmpeg filter graph expressions.
        notch_freq (float): Detected mains peak, either a fundamental or a harmonic.
    """
    if notch_freq <= 0:
        return

    fundamental = MAINS_FUNDAMENTAL_BY_HARMONIC.get(notch_freq, notch_freq)
    filters.append(f"bandreject=f={fundamental}:width_type=q:w=15")
    if fundamental in MAINS_FUNDAMENTALS:
        harmonic = round(fundamental * 2.0, 2)
        filters.append(f"bandreject=f={harmonic}:width_type=q:w=15")


def _append_notch_filters(filters, notch_freq, crt_notch=0.0, resonance_freq=0.0):
    """Appends fundamental, harmonic, CRT flyback, and resonance notch filters.

    Args:
        filters (list): Accumulator list of FFmpeg filter graph expressions.
        notch_freq (float): Fundamental mains buzz frequency (50 Hz / 60 Hz).
        crt_notch (float): CRT horizontal line whistle (15625 Hz / 15734 Hz).
        resonance_freq (float): Enclosure acoustic resonance peak frequency (Hz).
    """
    _append_mains_notches(filters, notch_freq)
    if crt_notch > 0:
        filters.append(f"bandreject=f={crt_notch}:width_type=q:w=30")
    if resonance_freq > 0:
        filters.append(f"bandreject=f={resonance_freq}:width_type=q:w=12")


# A genuine channel level mismatch is a few dB. Anything beyond this indicates a
# dead or unused channel (measured across a real 21-tape library: values cluster at
# 0 dB or 40-53 dB, with nothing in between), where attenuating the surviving
# channel by that much would destroy the audio.
BALANCE_MIN_DB = 0.5
BALANCE_MAX_DB = 12.0
# Beyond this the quiet side is not a level mismatch but a dead channel. Measured
# on a real 21-tape library: 11 tapes sat at 39-53 dB and one at 30 dB, and in
# every case the quiet side held only noise or bleed, 29-59 dB below the live one.
DEAD_CHANNEL_DB = 25.0


def _append_dead_channel_collapse(filters, balance_db):
    """Mirrors the surviving channel across both sides of a one-sided recording.

    Half of a real 21-tape library carried audio on a single channel. Leaving that
    as-is makes playback one-sided and hands the stem separator a silent channel;
    mirroring the live side centres it and gives separation a full signal.

    Args:
        filters (list): Accumulator list of FFmpeg filter graph expressions.
        balance_db (float): Left vs Right channel volume disparity in dB.
    """
    source = "c0" if balance_db > 0.0 else "c1"
    filters.append(f"pan=stereo|c0={source}|c1={source}")


def _append_level_match(filters, balance_db):
    """Attenuates the louder channel to level a modest stereo imbalance.

    Args:
        filters (list): Accumulator list of FFmpeg filter graph expressions.
        balance_db (float): Left vs Right channel volume disparity in dB.
    """
    gain = round(10.0 ** (-abs(balance_db) / 20.0), 3)
    if balance_db > 0.0:
        filters.append(f"pan=stereo|c0={gain}*c0|c1=c1")
    else:
        filters.append(f"pan=stereo|c0=c0|c1={gain}*c1")


def _append_balance_correction(filters, balance_db):
    """Levels a modest imbalance, or centres a recording with a dead channel.

    balance_db is 20*log10(left_rms / right_rms), so a positive value means the
    left channel is the louder one and is the side that must come down. Imbalances
    between BALANCE_MAX_DB and DEAD_CHANNEL_DB are left alone: too wide to level
    without audible pumping, too narrow to be confident the quiet side is empty.

    Args:
        filters (list): Accumulator list of FFmpeg filter graph expressions.
        balance_db (float): Left vs Right channel volume disparity in dB.
    """
    magnitude = abs(balance_db)
    if magnitude > DEAD_CHANNEL_DB:
        _append_dead_channel_collapse(filters, balance_db)
    elif BALANCE_MIN_DB < magnitude <= BALANCE_MAX_DB:
        _append_level_match(filters, balance_db)


def _append_dc_and_balance(filters, enable_dc_block, balance_db):
    """Appends DC blocking highpass and stereo channel balance leveling.

    Args:
        filters (list): Accumulator list of FFmpeg filter graph expressions.
        enable_dc_block (bool): Whether hardware DC voltage bias was detected.
        balance_db (float): Left vs Right channel volume disparity in dB.
    """
    if enable_dc_block:
        filters.append("highpass=f=2")
    _append_balance_correction(filters, balance_db)


def _append_azimuth_and_declip(filters, enable_adeclip, azimuth_delay_ms):
    """Appends de-clipping and inter-channel azimuth phase alignment.

    Args:
        filters (list): Accumulator list of FFmpeg filter graph expressions.
        enable_adeclip (bool): Whether flat-topped peak clipping was detected.
        azimuth_delay_ms (float): Calculated inter-channel skew delay in ms.
    """
    if enable_adeclip:
        filters.append("adeclip")
    if azimuth_delay_ms > 0:
        delay_samples = round(azimuth_delay_ms * AZIMUTH_SAMPLE_RATE / 1000)
        filters.append(f"adelay={delay_samples}S|0")
    elif azimuth_delay_ms < 0:
        delay_samples = round(abs(azimuth_delay_ms) * AZIMUTH_SAMPLE_RATE / 1000)
        filters.append(f"adelay=0|{delay_samples}S")


def _build_vhs_native_filter_string(
    nr=AFFTDN_NR,
    nf=AFFTDN_NF,
    tn=AFFTDN_TN,
    highpass_freq=HIGHPASS_FREQ,
    enable_adeclick=ENABLE_ADECLICK,
    notch_freq=NOTCH_FREQ,
    enable_adeclip=False,
    azimuth_delay_ms=0.0,
    enable_dc_block=False,
    balance_db=0.0,
    crt_notch=0.0,
    resonance_freq=0.0,
):
    """Builds composite FFmpeg audio filter string for VHS DSP restoration.

    Args:
        nr (float): Noise reduction strength in dB for afftdn.
        nf (float): Baseline noise floor in dB for afftdn.
        tn (bool): Whether noise tracking is enabled.
        highpass_freq (int): Motor rumble cutoff frequency in Hz.
        enable_adeclick (bool): Whether to attach impulsive click filter.
        notch_freq (float): Mains hum fundamental frequency.
        enable_adeclip (bool): Whether to attach peak declipper.
        azimuth_delay_ms (float): Stereo azimuth skew correction in ms.
        enable_dc_block (bool): Whether to attach 2 Hz sub-audible DC blocker.
        balance_db (float): Left vs Right channel balance adjustment.
        crt_notch (float): CRT line whistle frequency in Hz.
        resonance_freq (float): Enclosure acoustic resonance in Hz.

    Returns:
        str: Comma-separated FFmpeg filter graph chain.
    """
    pre = _build_precondition_filter_string(
        highpass_freq,
        enable_adeclick,
        notch_freq,
        enable_adeclip,
        azimuth_delay_ms,
        enable_dc_block,
        balance_db,
        crt_notch,
        resonance_freq,
    )
    afftdn_filter = f"afftdn=nr={nr}:nf={nf}:tn={1 if tn else 0}"
    return f"{pre},{afftdn_filter}" if pre != "anull" else afftdn_filter


def _build_precondition_filter_string(
    highpass_freq=60,
    enable_adeclick=True,
    notch_freq=0.0,
    enable_adeclip=False,
    azimuth_delay_ms=0.0,
    enable_dc_block=False,
    balance_db=0.0,
    crt_notch=0.0,
    resonance_freq=0.0,
):
    """Builds non-destructive pre-conditioning filter string.

    Args:
        highpass_freq (int): Motor rumble cutoff in Hz.
        enable_adeclick (bool): Whether adeclick is active.
        notch_freq (float): Mains notch frequency in Hz.
        enable_adeclip (bool): Whether adeclip is active.
        azimuth_delay_ms (float): Azimuth skew delay in ms.
        enable_dc_block (bool): Whether DC blocker is active.
        balance_db (float): Stereo channel balance difference in dB.
        crt_notch (float): CRT line whistle frequency in Hz.
        resonance_freq (float): Enclosure resonance frequency in Hz.

    Returns:
        str: Formatted FFmpeg filter string or 'anull'.
    """
    filters = []
    _append_dc_and_balance(filters, enable_dc_block, balance_db)
    _append_azimuth_and_declip(filters, enable_adeclip, azimuth_delay_ms)
    if highpass_freq > 0:
        filters.append(f"highpass=f={highpass_freq}")
    if enable_adeclick:
        filters.append("adeclick")
    _append_notch_filters(filters, notch_freq, crt_notch, resonance_freq)
    return ",".join(filters) if filters else "anull"


def _validate_arnndn_file_integrity(model_path, model_name):
    """Verifies that an existing RNNoise model file matches its known SHA-256 digest.

    If the file is corrupt or incomplete, it is deleted so a fresh copy can be downloaded.
    """
    expected_hash = ARNNDN_MODEL_SHA256.get(model_name)
    if not expected_hash:
        return True

    try:
        data = model_path.read_bytes()
        if len(data) < 100 or hashlib.sha256(data).hexdigest() != expected_hash:
            log_msg(f"  [Warning] Corrupt ARNNDN model '{model_name}' detected. Deleting to re-download...", is_error=True)
            model_path.unlink()
            return False
        return True
    except Exception as exc:
        log_msg(f"  [Warning] Failed to verify ARNNDN model '{model_name}': {exc}", is_error=True)
        return False


def _search_model_in_dirs(model_name, search_dirs):
    """Searches candidate folders for an RNNoise model file and verifies its integrity."""
    for directory in search_dirs:
        model_file = directory / model_name
        if model_file.is_file():
            if _validate_arnndn_file_integrity(model_file, model_name):
                return model_file.resolve()
    return None


def _get_remote_model_url_path(model_name):
    """Resolves repository relative path for a known or specified model filename."""
    subpath = ARNNDN_MODEL_SUBDIR_MAP.get(model_name, model_name)
    quoted = urllib.parse.quote(subpath)
    return f"{ARNNDN_REMOTE_PATH_PREFIX}/{quoted}"


def _get_pinned_arnndn_digest(model_name):
    """Returns the pinned digest for a supported ARNNDN model."""
    expected_hash = ARNNDN_MODEL_SHA256.get(model_name)
    if expected_hash is None:
        raise ValueError(f"ARNNDN model '{model_name}' has no pinned SHA-256 digest.")
    return expected_hash


def _fetch_remote_model_bytes(model_name):
    """Fetches raw bytes for a designated model over HTTPS."""
    expected_hash = _get_pinned_arnndn_digest(model_name)
    conn = http.client.HTTPSConnection(ARNNDN_REMOTE_HOST, timeout=30)
    try:
        conn.request("GET", _get_remote_model_url_path(model_name), headers={"User-Agent": "AI-Hybrid-VHS-Audio-Restorer/1.2.0"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP response error {resp.status}: {resp.reason}")
        data = resp.read()
        if len(data) < 100:
            raise ValueError(f"Downloaded model content for {model_name} is too small or invalid.")
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"Downloaded model content for {model_name} failed SHA-256 verification.")
        return data
    finally:
        conn.close()


def _download_arnndn_model(model_name, target_path):
    """Downloads an RNNoise (.rnnn) model file automatically from remote repository."""
    log_msg(f"  [System] ARNNDN model '{model_name}' not found locally. Downloading...")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_suffix(f"{target_path.suffix}.tmp")

    try:
        data = _fetch_remote_model_bytes(model_name)
        with open(temp_target, "wb") as f:
            f.write(data)
        if target_path.exists():
            target_path.unlink()
        temp_target.rename(target_path)
        log_msg(f"  [System] Successfully downloaded ARNNDN model: {target_path.name}")
        return target_path.resolve()
    except Exception as exc:
        if temp_target.exists():
            temp_target.unlink()
        log_msg(f"  [Warning] Failed to auto-download ARNNDN model '{model_name}': {exc}", is_error=True)
        return None


def _is_direct_rnnn_name(model_name):
    """Validates if target is a simple .rnnn filename without directories."""
    return model_name.endswith(".rnnn") and "/" not in model_name and "\\" not in model_name


def _try_auto_download_arnndn(model_name):
    """Attempts auto-downloading an .rnnn model if the model name is a plain filename."""
    if not _is_direct_rnnn_name(model_name) or model_name not in ARNNDN_MODEL_SHA256:
        return None
    default_target = MODELS_DIR / "arnndn" / model_name
    downloaded = _download_arnndn_model(model_name, default_target)
    if downloaded and downloaded.is_file():
        return downloaded
    return None


def _resolve_arnndn_model_path(model_name=ARNNDN_MODEL):
    """Resolves ARNNDN model file across candidate paths, auto-downloading if missing or corrupt."""
    candidate = Path(model_name)
    if candidate.is_file() and _validate_arnndn_file_integrity(candidate, candidate.name):
        return candidate.resolve()

    search_dirs = [Path("models") / "arnndn", Path("models"), MODELS_DIR / "arnndn", MODELS_DIR]
    found = _search_model_in_dirs(model_name, search_dirs)
    if found is not None:
        return found

    downloaded = _try_auto_download_arnndn(model_name)
    return downloaded if downloaded is not None else candidate.resolve()


def _escape_ffmpeg_filter_path(file_path):
    """Escapes path characters (like colons in Windows drive paths) for FFmpeg filter syntax."""
    return str(Path(file_path).resolve().as_posix()).replace("\\", r"\\").replace("'", r"\'").replace(":", r"\:")


def _build_arnndn_filter_string(
    model_path,
    highpass_freq=ARNNDN_HIGHPASS_FREQ,
    enable_adeclick=ARNNDN_ENABLE_ADECLICK,
):
    """Constructs FFmpeg filter expression with ARNNDN speech denoiser."""
    filters = []
    if highpass_freq > 0:
        filters.append(f"highpass=f={highpass_freq}")
    if enable_adeclick:
        filters.append("adeclick")
    escaped = _escape_ffmpeg_filter_path(model_path)
    filters.append(f"arnndn=m='{escaped}'")
    return ",".join(filters)


def _build_filter_audio_cmd(input_wav, output_tmp_wav, filter_str, threads):
    """Generates FFmpeg command array to apply audio filter graph to 32-bit PCM."""
    return [
        FFMPEG_BIN,
        "-stats",
        "-hide_banner",
        "-threads",
        str(threads),
        "-i",
        str(input_wav),
        "-af",
        filter_str,
        "-acodec",
        "pcm_f32le",
        "-ar",
        "44100",
        "-y",
        str(output_tmp_wav),
    ]


def _run_ffmpeg_filter_step(original_wav, output_wav, filter_str, step_label, task_desc, error_desc, total_duration=None):
    """Executes FFmpeg filter chain with retry and atomic output promotion."""
    log_msg(f"  [Step 2/4] Applying {step_label}...")
    tmp_wav = output_wav.with_suffix(".tmp.wav")

    def build_cmd(threads):
        return _build_filter_audio_cmd(original_wav, tmp_wav, filter_str, threads)

    attempt_cpu_run_with_retry(build_cmd, CPU_THREADS, description=task_desc, total_duration=total_duration)

    if is_valid_audio(tmp_wav):
        if output_wav.exists():
            output_wav.unlink()
        tmp_wav.rename(output_wav)
        return output_wav

    if tmp_wav.exists():
        tmp_wav.unlink()
    raise RuntimeError(f"{error_desc}: Output audio is invalid or empty.")


def _filter_vhs_native_step(original_wav, filtered_audio_dir, total_duration=None):
    """Applies FFmpeg native VHS filter chain (highpass, adeclick, afftdn, notch)."""
    output_wav = filtered_audio_dir / f"vhs_filtered_{original_wav.name}"
    if is_valid_audio(output_wav):
        log_msg("  [Step 2/4] Skipping Native VHS Filtering (exists)")
        return output_wav
    return _run_ffmpeg_filter_step(
        original_wav,
        output_wav,
        _build_vhs_native_filter_string(),
        "Native VHS Filtering",
        "Filtering VHS Audio",
        "Native VHS filtering failed",
        total_duration=total_duration,
    )


def _filter_arnndn_step(original_wav, filtered_audio_dir, total_duration=None, model_name=None):
    """Applies FFmpeg Recurrent Neural Network (RNNoise) speech denoiser."""
    output_wav = filtered_audio_dir / f"arnndn_filtered_{original_wav.name}"
    if is_valid_audio(output_wav):
        log_msg("  [Step 2/4] Skipping ARNNDN Speech Filtering (exists)")
        return output_wav

    selected_model = ARNNDN_MODEL if model_name is None else model_name
    if not ffmpeg_has_filter("arnndn"):
        raise RuntimeError("The available FFmpeg executable does not support the 'arnndn' filter.")
    model_path = _resolve_arnndn_model_path(selected_model)
    if not model_path.is_file():
        raise FileNotFoundError(f"ARNNDN model file not found: {selected_model}. Place it in models/arnndn/ or models/.")

    filter_str = _build_arnndn_filter_string(str(model_path))
    return _run_ffmpeg_filter_step(
        original_wav,
        output_wav,
        filter_str,
        f"ARNNDN Speech Denoise ({model_path.name})",
        "Filtering ARNNDN Speech",
        "ARNNDN Speech filtering failed",
        total_duration=total_duration,
    )


def _pick_reduction_db(noise_floor_db):
    """Maps measured noise floor dB to recommended reduction strength."""
    if noise_floor_db < -60.0:
        return 8.0
    if noise_floor_db < -45.0:
        return 11.0
    if noise_floor_db < -35.0:
        return 13.5
    return 16.0


def _estimate_noise_floor_and_reduction(signal_data):
    """Estimates baseline noise floor (dB) and adaptive reduction depth."""
    frame_size = 2048
    hop = 1024
    num_frames = max(1, (len(signal_data) - frame_size) // hop)
    if num_frames < 2 or np is None:
        return -45.0, 12.0

    count = min(num_frames, 100)
    # Spread the probes over the whole recording: reading only the opening frames
    # made the estimate swing by a median of 9.8 dB (max 23.6 dB) on real tapes.
    starts = np.linspace(0, len(signal_data) - frame_size, count).astype(int)
    frames = [signal_data[slice(int(s), int(s) + frame_size)] for s in starts]
    rms_vals = [float(np.sqrt(np.mean(f**2) + 1e-12)) for f in frames]
    p10_rms = float(np.percentile(rms_vals, 10))
    noise_floor_db = float(np.clip(20.0 * np.log10(p10_rms + 1e-6), -80.0, -20.0))
    nr_db = _pick_reduction_db(noise_floor_db)
    return round(noise_floor_db, 1), round(nr_db, 1)


def _compute_peak_ratio(fft_mag, freqs, target_freq):
    """Calculates peak-to-surrounding ratio at a target harmonic frequency."""
    idx = int(np.argmin(np.abs(freqs - target_freq)))
    if not (2 <= idx < len(fft_mag) - 2):
        return 0.0
    surrounding = float(fft_mag[idx - 2] + fft_mag[idx + 2]) / 2.0 + 1e-9
    return float(fft_mag[idx]) / surrounding


# A recording's video line rate fixes its mains frequency: PAL regions run 50 Hz
# mains and a 15625 Hz line rate, NTSC regions 60 Hz and 15734 Hz. When the
# flyback whine identifies the standard, the opposite family must be excluded so
# a stray tone cannot be mistaken for hum and notched out of the bass.
MAINS_CANDIDATES_BY_LINE_RATE = {15625.0: (50.0, 100.0), 15734.0: (60.0, 120.0)}
MAINS_CANDIDATES_DEFAULT = (50.0, 60.0, 100.0, 120.0)


def _detect_mains_buzz_notch(signal_data, sr, line_rate_hz=0.0):
    """Detects 50Hz (PAL), 59.94/60Hz (NTSC), or harmonic buzz peaks.

    Args:
        signal_data (numpy.ndarray): Mono audio waveform samples.
        sr (int): Sampling rate in Hz.
        line_rate_hz (float): Detected CRT flyback rate; narrows the candidate
            set to the mains family that physically matches the video standard.

    Returns:
        float: Detected notch frequency in Hz (0.0 if no buzz detected).
    """
    if len(signal_data) < 8192 or np is None:
        return 0.0

    chunk = signal_data[:32768]
    fft_mag = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)

    candidates = MAINS_CANDIDATES_BY_LINE_RATE.get(line_rate_hz, MAINS_CANDIDATES_DEFAULT)
    best_target, max_ratio = 0.0, 1.8
    for tf in candidates:
        ratio = _compute_peak_ratio(fft_mag, freqs, tf)
        if ratio > max_ratio:
            max_ratio = ratio
            best_target = tf
    return best_target


def _pick_rumble_highpass(rumble_ratio):
    """Maps measured low-frequency rumble power ratio to highpass cutoff frequency.

    Args:
        rumble_ratio (float): Relative low-frequency acoustic power ratio.

    Returns:
        int: Cutoff frequency in Hz (0, 45, 60, or 75).
    """
    if rumble_ratio > 0.3:
        return 75
    if rumble_ratio > 0.1:
        return 60
    return 45 if rumble_ratio > 0.02 else 0


def _detect_low_frequency_rumble(signal_data, sr):
    """Calculates mechanical motor rumble power below 75 Hz.

    Args:
        signal_data (numpy.ndarray): Mono audio waveform samples.
        sr (int): Sampling rate in Hz.

    Returns:
        int: Recommended highpass cutoff frequency in Hz.
    """
    if len(signal_data) < 4096 or np is None:
        return 60

    chunk = signal_data[:16384]
    fft_power = np.abs(np.fft.rfft(chunk)) ** 2
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)

    rumble_band = fft_power[(freqs >= 10.0) & (freqs <= 70.0)]
    total_power = float(np.sum(fft_power) + 1e-9)
    return _pick_rumble_highpass(float(np.sum(rumble_band)) / total_power)


def _detect_click_density(signal_data):
    """Detects whether audio has high-frequency impulsive pop/click bursts.

    Args:
        signal_data (numpy.ndarray): Mono audio waveform samples.

    Returns:
        bool: True if impulsive click bursts exceed threshold.
    """
    if len(signal_data) < 4096 or np is None:
        return True
    d2 = np.abs(np.diff(np.abs(np.diff(signal_data))))
    mean_d2 = float(np.mean(d2) + 1e-9)
    return (float(np.sum(d2 > (mean_d2 * 10.0))) / float(len(signal_data))) > 0.0001


def _detect_analog_clipping(signal_data, threshold=0.985, min_count=20):
    """Detects whether audio contains clipped analog peaks.

    Args:
        signal_data (numpy.ndarray): Audio waveform samples.
        threshold (float): Absolute amplitude threshold for flat tops.
        min_count (int): Minimum count of saturated samples.

    Returns:
        bool: True if clipping is detected.
    """
    if len(signal_data) < 1024 or np is None:
        return False
    clipped_mask = np.abs(signal_data) >= threshold
    return int(np.sum(clipped_mask)) >= min_count


def _is_valid_stereo(stereo_audio):
    """Validates stereo audio dimensions for phase analysis.

    Args:
        stereo_audio (numpy.ndarray): Multichannel audio array.

    Returns:
        bool: True if array has at least 2 channels.
    """
    return stereo_audio is not None and getattr(stereo_audio, "ndim", 0) >= 2 and stereo_audio.shape[1] >= 2


# A cross-correlation lag only means something for channels that share content.
# Measured on real tapes: true stereo pairs correlate 0.98-1.00 while dead-channel
# pairs correlate 0.01-0.08, so a lag read off the latter is pure noise.
AZIMUTH_MIN_CORRELATION = 0.3


def _channel_correlation(left, right):
    """Absolute Pearson correlation between two channels; 0.0 if either is silent."""
    left_centred = left - np.mean(left)
    right_centred = right - np.mean(right)
    denom = float(np.sqrt(np.sum(left_centred**2) * np.sum(right_centred**2)))
    if denom <= 0.0:
        return 0.0
    return abs(float(np.sum(left_centred * right_centred) / denom))


def _detect_stereo_azimuth_skew(stereo_audio, sr, max_lag_samples=44):
    """Detects sub-millisecond inter-channel time delay (azimuth phase skew).

    Args:
        stereo_audio (numpy.ndarray): Stereo 2-channel audio array.
        sr (int): Sampling rate in Hz.
        max_lag_samples (int): Maximum lag window in samples (~1 ms).

    Returns:
        float: Time delay in milliseconds.
    """
    if not _is_valid_stereo(stereo_audio) or np is None:
        return 0.0
    left = stereo_audio[:32768, 0]
    right = stereo_audio[:32768, 1]
    if _channel_correlation(left, right) < AZIMUTH_MIN_CORRELATION:
        return 0.0

    corr = np.correlate(left - np.mean(left), right - np.mean(right), mode="full")
    lag = int(np.argmax(corr)) - (len(right) - 1)
    if not (0 < abs(lag) <= max_lag_samples):
        return 0.0
    return round(float(lag) / float(sr) * 1000.0, 2)


def _analysis_block_offsets(total_frames, block_frames):
    """Returns four evenly spaced read offsets for a long audio stream."""
    return [round((total_frames - block_frames) * index / 3) for index in range(4)]


def _read_analysis_block(wav_path, start_frame, block_frames):
    """Reads one analysis block from a known frame offset."""
    return sf.read(str(wav_path), dtype="float32", start=start_frame, frames=block_frames)[0]


def _read_evenly_spaced_analysis_blocks(wav_path, audio_info, block_frames):
    """Reads and concatenates evenly distributed analysis blocks."""
    offsets = _analysis_block_offsets(audio_info.frames, block_frames)
    blocks = [_read_analysis_block(wav_path, offset, block_frames) for offset in offsets]
    return np.concatenate(blocks)


def _read_analysis_audio(wav_path, audio_info):
    """Reads either the full short stream or representative blocks from a long one."""
    block_frames = min(audio_info.frames, audio_info.samplerate * 60)
    # Four non-overlapping blocks only fit when the stream is at least 4x a block;
    # below that, read a single contiguous block instead of concatenating overlaps.
    if audio_info.frames < block_frames * 4:
        return _read_analysis_block(wav_path, 0, min(audio_info.frames, block_frames))
    return _read_evenly_spaced_analysis_blocks(wav_path, audio_info, block_frames)


def _read_stereo_audio_for_analysis(wav_path):
    """Safely reads raw multichannel audio for azimuth phase analysis.

    Args:
        wav_path (pathlib.Path): Path to WAV audio file.

    Returns:
        tuple: (data_array, sample_rate) or (None, None).
    """
    if sf is None or np is None:
        return None, None
    try:
        audio_info = sf.info(str(wav_path))
        return _read_analysis_audio(wav_path, audio_info), audio_info.samplerate
    except Exception:
        return None, None


def _read_audio_for_analysis(wav_path):
    """Safely reads audio and downmixes to mono signal for analysis.

    Args:
        wav_path (pathlib.Path): Path to WAV audio file.

    Returns:
        tuple: (mono_signal_array, sample_rate) or (None, None).
    """
    data, sr = _read_stereo_audio_for_analysis(wav_path)
    if data is None or sr is None or np is None:
        return None, None
    mono_signal = np.mean(data, axis=1) if data.ndim > 1 else data
    return mono_signal, sr


def _analyze_vhs_audio_profile(wav_path):
    """Scans and extracts acoustic noise parameters from audio file.

    Args:
        wav_path (pathlib.Path): Path to input WAV file.

    Returns:
        dict: Extracted acoustic noise parameters.
    """
    stereo_data, sr = _read_stereo_audio_for_analysis(wav_path)
    if stereo_data is None or sr is None:
        return {
            "nr": 12.0,
            "nf": -45.0,
            "tn": True,
            "highpass": 60,
            "adeclick": True,
            "notch": 0.0,
            "clipping": False,
            "azimuth": 0.0,
            "dc_block": False,
            "balance": 0.0,
            "crt_notch": 0.0,
            "resonance": 0.0,
        }

    mono_signal = np.mean(stereo_data, axis=1) if stereo_data.ndim > 1 else stereo_data
    nf_db, nr_db = _estimate_noise_floor_and_reduction(mono_signal)
    crt = _detect_crt_flyback_notch(mono_signal, sr)
    return {
        "nr": nr_db,
        "nf": nf_db,
        "tn": True,
        "highpass": _detect_low_frequency_rumble(mono_signal, sr),
        "adeclick": _detect_click_density(mono_signal),
        "notch": _detect_mains_buzz_notch(mono_signal, sr, crt),
        "clipping": _detect_analog_clipping(mono_signal),
        "azimuth": _detect_stereo_azimuth_skew(stereo_data, sr),
        "dc_block": _detect_dc_offset_bias(mono_signal),
        "balance": _detect_stereo_balance_imbalance(stereo_data),
        "crt_notch": crt,
        "resonance": _detect_enclosure_resonance_notch(mono_signal, sr),
    }


def _build_auto_vhs_native_filter_string(wav_path):
    """Scans tape acoustic profile and generates fine-tuned DSP filter string.

    Args:
        wav_path (pathlib.Path): Path to input audio track.

    Returns:
        str: Comma-separated FFmpeg filter graph chain.
    """
    profile = _analyze_vhs_audio_profile(wav_path)
    log_msg(
        f"    [Auto-Scan] Profile: Noise Floor={profile['nf']:.1f}dB | "
        f"Reduction={profile['nr']:.1f}dB | Highpass={profile['highpass']}Hz | "
        f"Declick={profile['adeclick']} | Notch={profile['notch']}Hz"
    )
    return _build_vhs_native_filter_string(
        nr=profile["nr"],
        nf=profile["nf"],
        tn=profile["tn"],
        highpass_freq=profile["highpass"],
        enable_adeclick=profile["adeclick"],
        notch_freq=profile["notch"],
        enable_adeclip=profile.get("clipping", False),
        azimuth_delay_ms=profile.get("azimuth", 0.0),
        enable_dc_block=profile.get("dc_block", False),
        balance_db=profile.get("balance", 0.0),
        crt_notch=profile.get("crt_notch", 0.0),
        resonance_freq=profile.get("resonance", 0.0),
    )


def _filter_auto_vhs_native_step(original_wav, filtered_audio_dir, total_duration=None):
    """Performs acoustic scanning and executes auto-tuned native VHS DSP filtering.

    Args:
        original_wav (pathlib.Path): Path to source audio WAV file.
        filtered_audio_dir (pathlib.Path): Destination directory for filtered file.
        total_duration (float, optional): Total duration in seconds for progress bar.

    Returns:
        pathlib.Path: Path to filtered audio WAV file.
    """
    output_wav = filtered_audio_dir / f"auto_vhs_filtered_{original_wav.name}"
    if is_valid_audio(output_wav):
        log_msg("  [Step 2/4] Skipping Auto VHS Filtering (exists)")
        return output_wav
    filter_str = _build_auto_vhs_native_filter_string(original_wav)
    return _run_ffmpeg_filter_step(
        original_wav,
        output_wav,
        filter_str,
        "Auto-Tuned VHS Native Filtering",
        "Auto-Filtering VHS Audio",
        "Auto-Tuned VHS filtering failed",
        total_duration=total_duration,
    )


def _is_short_for_crt(signal_data, sr):
    """Validates signal length and sample rate for high-frequency CRT flyback scanning.

    Args:
        signal_data (numpy.ndarray): Mono audio waveform samples.
        sr (int): Sampling rate in Hz.

    Returns:
        bool: True if audio is too short or sample rate too low.
    """
    return len(signal_data) < 16384 or sr < 32000 or np is None


# The two line rates are only 109 Hz apart, so the whine must be found by
# searching the band that spans both and then classified by which nominal it is
# nearest. Probing each nominal bin exactly fails whenever the tape runs slightly
# off speed: the strongest whine in a real 21-tape library sat at 15633 Hz and was
# being reported as NTSC. Prominence separation is wide - tapes carrying the whine
# measure 1.9-85.7x the band median, tapes without it 1.0-1.5x.
CRT_SEARCH_LOW_HZ = 15450.0
CRT_SEARCH_HIGH_HZ = 15900.0
CRT_MIN_PROMINENCE = 1.7
CRT_LINE_RATES_HZ = (15625.0, 15734.0)


def _detect_crt_flyback_notch(signal_data, sr):
    """Detects the 15625 Hz (PAL) or 15734 Hz (NTSC) CRT line whistle.

    Args:
        signal_data (numpy.ndarray): Mono audio waveform samples.
        sr (int): Sampling rate in Hz.

    Returns:
        float: Nominal line rate of the detected standard (0.0 if not detected).
    """
    if _is_short_for_crt(signal_data, sr):
        return 0.0

    chunk = signal_data[:32768]
    fft_mag = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)
    band = np.flatnonzero((freqs >= CRT_SEARCH_LOW_HZ) & (freqs <= CRT_SEARCH_HIGH_HZ))
    if len(band) == 0:
        return 0.0

    peak = int(band[int(np.argmax(fft_mag[band]))])
    prominence = float(fft_mag[peak]) / (float(np.median(fft_mag[band])) + 1e-12)
    if prominence < CRT_MIN_PROMINENCE:
        return 0.0
    return min(CRT_LINE_RATES_HZ, key=lambda nominal: abs(float(freqs[peak]) - nominal))


def _detect_dc_offset_bias(signal_data, threshold=0.003):
    """Detects whether audio has hardware digitizer DC voltage bias.

    Args:
        signal_data (numpy.ndarray): Mono audio waveform samples.
        threshold (float): Minimum absolute DC mean value to flag.

    Returns:
        bool: True if DC offset bias is detected.
    """
    if len(signal_data) < 1024 or np is None:
        return False
    return abs(float(np.mean(signal_data[:32768]))) >= threshold


def _detect_stereo_balance_imbalance(stereo_audio):
    """Calculates Left vs Right channel volume disparity in dB.

    Args:
        stereo_audio (numpy.ndarray): Multichannel audio array.

    Returns:
        float: Relative Left vs Right gain difference in dB.
    """
    if not _is_valid_stereo(stereo_audio) or np is None:
        return 0.0
    l_rms = float(np.sqrt(np.mean(stereo_audio[:32768, 0] ** 2) + 1e-12))
    r_rms = float(np.sqrt(np.mean(stereo_audio[:32768, 1] ** 2) + 1e-12))
    diff_db = 20.0 * np.log10(l_rms / r_rms)
    return round(float(diff_db), 2) if abs(diff_db) >= 1.0 else 0.0


# A resonance is a high-contrast bump of moderate width in the smoothed spectral
# envelope. Measured references: a Q=10..50 resonator reads contrast 26-529 at
# 65-215 Hz wide, a pure tone 23000 at 22 Hz, a voice harmonic 192 at 32 Hz, and
# real speech formants 2.4-16 at 22-580 Hz. Gating on contrast alone would notch
# voice; gating on width alone would notch formants.
RESONANCE_SEGMENT = 4096
RESONANCE_MAX_SEGMENTS = 120
RESONANCE_MIN_WIDTH_HZ = 50.0
RESONANCE_MAX_WIDTH_HZ = 400.0
RESONANCE_MIN_CONTRAST = 20.0


def _averaged_power_spectrum(signal_data, sr):
    """Averages overlapping windowed periodograms into a smooth spectral envelope.

    A single FFT of noise-excited audio fluctuates bin to bin, which hides the
    envelope shape entirely; averaging exposes it.
    """
    seg = RESONANCE_SEGMENT
    if len(signal_data) < seg * 2 or np is None:
        return None, None

    count = min((len(signal_data) - seg) // (seg // 2), RESONANCE_MAX_SEGMENTS)
    starts = np.linspace(0, len(signal_data) - seg, max(count, 1)).astype(int)
    window = np.hanning(seg)
    total = np.zeros(seg // 2 + 1)
    for start in starts:
        end = start + seg
        total += np.abs(np.fft.rfft(signal_data[start:end] * window)) ** 2
    return total / len(starts), np.fft.rfftfreq(seg, 1.0 / sr)


def _envelope_peak_width_hz(band_power, band_freqs, peak_idx):
    """Width in Hz over which the envelope stays above half the peak power."""
    half = float(band_power[peak_idx]) / 2.0
    low = peak_idx
    while low > 0 and band_power[low] > half:
        low -= 1
    high = peak_idx
    while high < len(band_power) - 1 and band_power[high] > half:
        high += 1
    return float(band_freqs[high] - band_freqs[low])


def _is_resonance_shaped(contrast, width_hz):
    """Separates a housing resonance from a formant hump or a harmonic spike."""
    if contrast < RESONANCE_MIN_CONTRAST:
        return False
    return RESONANCE_MIN_WIDTH_HZ <= width_hz <= RESONANCE_MAX_WIDTH_HZ


def _detect_enclosure_resonance_notch(signal_data, sr):
    """Detects a sharp acoustic housing resonance between 1.5 kHz and 3.5 kHz.

    Speech formants peak in this same band, so the loudest bin is not evidence of
    a resonance: across a real 21-tape library every single tape produced one, and
    notching it carves ~167 Hz out of the intelligibility region.

    Args:
        signal_data (numpy.ndarray): Mono audio waveform samples.
        sr (int): Sampling rate in Hz.

    Returns:
        float: Resonance center frequency in Hz (0.0 if not detected).
    """
    power, freqs = _averaged_power_spectrum(signal_data, sr)
    if power is None:
        return 0.0

    mask = (freqs >= 1500.0) & (freqs <= 3500.0)
    band, band_freqs = power[mask], freqs[mask]
    if len(band) == 0:
        return 0.0

    peak_idx = int(np.argmax(band))
    contrast = float(band[peak_idx]) / (float(np.median(band)) + 1e-12)
    if not _is_resonance_shaped(contrast, _envelope_peak_width_hz(band, band_freqs, peak_idx)):
        return 0.0
    return round(float(band_freqs[peak_idx]), 1)


def _filter_precondition_step(original_wav, output_wav, precond_config, total_duration=None):
    """Pass 2: Non-destructive analog pre-conditioning DSP (adeclick + highpass + notch)."""
    if is_valid_audio(output_wav):
        log_msg("  [Pass 2/4] Skipping Analog Pre-Conditioning (exists)")
        return output_wav
    filter_str = _build_precondition_filter_string(
        highpass_freq=int(precond_config.get("highpass_hz", 80)),
        enable_adeclick=bool(precond_config.get("enable_adeclick", True)),
        notch_freq=float(precond_config.get("notch_hz", 0.0)),
        enable_adeclip=bool(precond_config.get("enable_adeclip", False)),
        azimuth_delay_ms=float(precond_config.get("azimuth_delay_ms", 0.0)),
        enable_dc_block=bool(precond_config.get("enable_dc_block", False)),
        balance_db=float(precond_config.get("balance_db", 0.0)),
        crt_notch=float(precond_config.get("crt_notch_hz", 0.0)),
        resonance_freq=float(precond_config.get("resonance_hz", 0.0)),
    )
    return _run_ffmpeg_filter_step(
        original_wav=original_wav,
        output_wav=output_wav,
        filter_str=filter_str,
        step_label="Pre-Conditioning DSP",
        task_desc="Pre-conditioning DSP",
        error_desc="Pre-conditioning DSP failed",
        total_duration=total_duration,
    )


def _build_full_audio_expander_filter(noise_floor_db=None):
    """Constructs an adaptive downward dynamic expander curve based on noise floor."""
    if noise_floor_db is None:
        return "compand=attacks=0.04:decays=0.18:points=-90/-100|-65/-72|-45/-45|0/0"
    knee = max(-60.0, min(-35.0, float(noise_floor_db) + 4.0))
    mid = round((knee - 90.0) / 2.0, 1)
    return f"compand=attacks=0.04:decays=0.18:points=-90/-100|{mid:.1f}/{mid - 7.0:.1f}|{knee:.1f}/{knee:.1f}|0/0"


def _build_linear_air_filter(gain_db=LINEAR_AIR_GAIN_DB):
    """Constructs a gentle high-shelf presence curve compensating for tape head loss."""
    if not ENABLE_LINEAR_AIR or gain_db <= 0.0:
        return None
    return f"treble=g={gain_db:.1f}:f=7500"


def _append_linear_air_stage(stages, apply_air):
    """Appends high-shelf presence filter if air enhancement is requested."""
    if apply_air:
        air = _build_linear_air_filter()
        if air:
            stages.append(air)


def _append_expander_stage(stages, strategy):
    """Appends adaptive downward dynamic expander stage if enabled."""
    if ENABLE_DYNAMIC_EXPANDER:
        noise_floor_db = (strategy or {}).get("profile", {}).get("noise_floor_db")
        stages.append(_build_full_audio_expander_filter(noise_floor_db))


def build_full_audio_polish_filter(strategy=None, apply_air=False):
    """Assembles the chained polish filter with optional linear air and dynamic expander."""
    stages = []
    _append_linear_air_stage(stages, apply_air)
    _append_expander_stage(stages, strategy)
    return ",".join(stages) if stages else None


def _append_pre_denoise_harmonics(stages, notch_hz):
    """Appends mains hum fundamental and 4 harmonics with sharp Q=30 notches.

    Pre-cleans tonal mains noise before neural denoising so the neural model
    does not over-process or distort programme content trying to cancel hum.
    """
    if not notch_hz or notch_hz <= 0:
        return
    fundamental = MAINS_FUNDAMENTAL_BY_HARMONIC.get(notch_hz, notch_hz)
    for harmonic_idx in range(1, 6):
        freq = round(fundamental * harmonic_idx, 2)
        stages.append(f"bandreject=f={freq}:width_type=q:w=30")


def _append_pre_denoise_crt(stages, crt_hz):
    """Appends high-Q notch filter for CRT flyback line whistle before neural denoising."""
    if crt_hz and crt_hz > 0:
        stages.append(f"bandreject=f={crt_hz}:width_type=q:w=50")


def _append_pre_denoise_rumble(stages, highpass_hz):
    """Appends highpass rumble cutoff if configured."""
    if highpass_hz and highpass_hz > 0:
        stages.append(f"highpass=f={highpass_hz}")


def _get_strategy_freq(strategy, key):
    """Retrieves a frequency value from profile or precondition_filters."""
    if not isinstance(strategy, dict):
        return 0.0
    val = strategy.get("profile", {}).get(key)
    if val is None:
        val = strategy.get("precondition_filters", {}).get(key, 0.0)
    return float(val)


def _extract_notch_and_crt(strategy):
    """Extracts detected mains notch and CRT line whistle frequencies from strategy."""
    notch_hz = _get_strategy_freq(strategy, "notch_hz")
    crt_hz = _get_strategy_freq(strategy, "crt_notch_hz")
    return notch_hz, crt_hz


def build_pre_denoise_surgical_filter(strategy=None):
    """Builds surgical DSP filter graph executed before neural denoising in auto_pure_linear.

    Eliminates tonal interference (mains hum harmonics, CRT line whistle, rumble)
    before handing the audio to UVR-DeNoise. This prevents neural model over-processing.
    """
    stages = []
    notch_hz, crt_hz = _extract_notch_and_crt(strategy)
    rumble_hz = int(_get_strategy_freq(strategy, "highpass_hz"))

    _append_pre_denoise_rumble(stages, rumble_hz)
    _append_pre_denoise_harmonics(stages, notch_hz)
    _append_pre_denoise_crt(stages, crt_hz)
    return ",".join(stages) if stages else None


def _append_post_denoise_mains(stages, notch_hz):
    """Appends a narrow fundamental mains notch to catch post-denoising residual tone."""
    if notch_hz and notch_hz > 0:
        fundamental = MAINS_FUNDAMENTAL_BY_HARMONIC.get(notch_hz, notch_hz)
        stages.append(f"bandreject=f={fundamental}:width_type=q:w=40")


def _append_post_denoise_crt(stages, crt_hz):
    """Appends ultra-narrow notch for residual CRT flyback whistle that survived denoising."""
    if crt_hz and crt_hz > 0:
        stages.append(f"bandreject=f={crt_hz}:width_type=q:w=80")


def build_post_denoise_cleanup_filter(strategy=None):
    """Builds surgical residual cleanup filter applied between neural denoising and polish.

    Uses high-Q surgical notches to catch any lingering mains fundamental or CRT whistle.
    """
    stages = []
    notch_hz, crt_hz = _extract_notch_and_crt(strategy)
    _append_post_denoise_mains(stages, notch_hz)
    _append_post_denoise_crt(stages, crt_hz)
    return ",".join(stages) if stages else None
