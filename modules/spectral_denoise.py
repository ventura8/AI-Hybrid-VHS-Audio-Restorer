"""Noise-profile spectral subtraction for the full-mix restoration chain.

UVR-DeNoise works on material recorded at a healthy level and does essentially nothing on
quiet captures. Measured against a clean reference at a 8.9 dB programme-to-noise margin it
left log-spectral distance at 18.0 dB against the degraded input's 18.2 dB -- no change at
all -- while spectral subtraction reached 7.6 dB on the same fixtures.

That regime is not an edge case. On a 20-clip sample of the Internet Archive corpus the
median margin is 7.4 dB, which is why roughly half of those tapes (82 of 174) came out of
`auto_pure_linear` with the noise floor no better than it went in: the neural stage removed
nothing and the final loudness normalisation then amplified the untouched hiss.

This module supplies the missing stage, and applies it only where it is the better tool.
"""

from pathlib import Path

import numpy as np
import scipy.signal

from .blend_weights import apply_blend
from .config import (
    APL_ENABLE_DEHUM,
    APL_ENABLE_LEARNED_BLEND,
    APL_ENABLE_SPECTRAL_DENOISE,
    APL_ENABLE_TONAL_CLEANUP,
    APL_HUM_MIN_EXCESS_DB,
    APL_NOISEPRINT_DURATION_S,
    APL_SPECTRAL_ALPHA,
    APL_SPECTRAL_ALPHA_TONAL,
    APL_SPECTRAL_MARGIN_DB,
    APL_TONAL_FLATNESS_MAX,
)
from .filters import _read_audio_for_analysis, estimate_snr_margin_db
from .utils import log_msg

# When subtraction has run, the deep separator is the better partner for it. The light
# model is chosen elsewhere to stay transparent on material that still has its dynamics,
# but measured on realistic fixtures the deep one leaves a background 15 dB quieter after
# subtraction (-62.29 dB against -47.14 on hum) at equal or better spectral fidelity.
#
# The separator still earns its place after subtraction, which was worth confirming rather
# than assuming: choosing between the deep and the lite model moves both metrics by 0.01,
# so the choice does not matter much, but skipping the stage outright costs 1.05 dB of
# noise removal and returns nothing in fidelity (deviation 0.32 to 0.31).
DEEP_DENOISE_MODEL = "UVR-DeNoise.pth"

# Mains detection for the modes that opt in. The shared scanner reads only the first 32768
# samples and tests a single bin at the fundamental, and it misses hum that is plainly
# there: on 25 corpus clips carrying genuine mains hum it fired on 14. Deliberately not a
# change to the shared detector -- `notch_hz` feeds the pre-conditioning graph every mode
# is built from, and widening it there would move already-released modes, cathar above all.
#
# This one reads the whole recording and weighs eight harmonics as energy against their own
# neighbourhood, which is what real hum looks like and a single bin does not. It decides
# between 50 and 60 Hz by that evidence rather than by region, because on the 48 corpus
# tapes that carry hum the region's nominal frequency was the wrong one for 10.
MAINS_CANDIDATES_HZ = (50.0, 60.0)
MAINS_HARMONICS = 8
MAINS_DETECT_MIN_SAMPLES = 16384


def _harmonic_excess_db(mono_signal, sample_rate, mains_hz):
    """How far the mains harmonics stand above their own spectral neighbourhood, in dB.

    Summed as energy across harmonics and expressed as a ratio, so it reads the same on a
    quiet tape and a loud one: clean audio sits near 0 dB, a humming tape well above it.
    Real hum runs to eight harmonics, and a single-bin test at the fundamental is what the
    previous detector used and what let it miss hum on 11 of 25 tapes that plainly had it.
    """
    if len(mono_signal) < MAINS_DETECT_MIN_SAMPLES:
        return 0.0
    freqs, psd = scipy.signal.welch(mono_signal, sample_rate, nperseg=MAINS_DETECT_MIN_SAMPLES)
    peaks, floors = 0.0, 0.0
    for harmonic in range(1, MAINS_HARMONICS + 1):
        target = mains_hz * harmonic
        if target >= freqs[-1]:
            break
        index = int(np.argmin(np.abs(freqs - target)))
        left_lo, left_hi = max(index - 30, 0), max(index - 2, 0)
        right_lo, right_hi = index + 3, index + 33
        peak_lo, peak_hi = max(index - 1, 0), index + 2
        floors += float(np.median(np.concatenate((psd[left_lo:left_hi], psd[right_lo:right_hi])))) + 1e-20
        peaks += float(np.max(psd[peak_lo:peak_hi])) + 1e-20
    return float(10.0 * np.log10(peaks / floors))


def _scannable_mono(wav_path):
    """Returns mono samples long enough to scan, or (None, None)."""
    mono_signal, sample_rate = _read_audio_for_analysis(wav_path)
    if mono_signal is None or sample_rate is None or len(mono_signal) < MAINS_DETECT_MIN_SAMPLES:
        return None, None
    return mono_signal, sample_rate


def detect_mains_hz(wav_path):
    """Returns the mains frequency carrying hum across the recording, 0.0 for none, None when unreadable.

    50 or 60 Hz, chosen by which one the harmonics support rather than by region: on the 48
    corpus tapes that carry hum, the region's nominal frequency was the wrong one for 10.
    Tapes cross regions, and a dehum at the wrong frequency does nothing at all.

    Constrained to those two candidates deliberately. A free peak search was tried and
    locked onto programme content on 39 of 48 tapes, notching bass rather than hum; real
    mains is stable to well under half a hertz, so the freedom bought nothing but errors.
    """
    mono_signal, sample_rate = _scannable_mono(wav_path)
    if sample_rate is None:
        return None
    best_hz, best_excess = 0.0, 0.0
    for candidate in MAINS_CANDIDATES_HZ:
        excess = _harmonic_excess_db(mono_signal, sample_rate, candidate)
        if excess > best_excess:
            best_hz, best_excess = candidate, excess
    return best_hz if best_excess >= APL_HUM_MIN_EXCESS_DB else 0.0


def should_apply(source_wav):
    """Returns the measured margin when subtraction is the better tool, else None.

    Above the threshold the neural stage already handles the material and this stage costs
    a little fidelity: on healthy fixtures it moved log-spectral distance 0.04-1.0 dB in
    the wrong direction. Below it, the neural stage contributes nothing.
    """
    if not APL_ENABLE_SPECTRAL_DENOISE:
        return None
    margin_db = estimate_snr_margin_db(source_wav)
    if margin_db is None or margin_db >= APL_SPECTRAL_MARGIN_DB:
        return None
    return margin_db


def _cathar_available():
    """Returns whether the Cathar CLI can be invoked."""
    from .cathar import _require_cathar_binary

    try:
        _require_cathar_binary()
        return True
    except FileNotFoundError:
        return False


TONALITY_BAND_HZ = (100.0, 5000.0)
TONALITY_FRAME = 4096


def estimate_tonality(wav_path):
    """Median per-frame spectral flatness in the speech band, or None when unreadable.

    Flatness is the geometric mean of the power spectrum over its arithmetic mean: near 1
    for noise, near 0 for a signal made of sustained peaks. Music sits low, and it is the
    material on which subtraction at the speech-tuned factor takes programme with the noise.
    """
    mono_signal, sample_rate = _scannable_mono(wav_path)
    if sample_rate is None:
        return None
    freqs, _times, spectrum = scipy.signal.stft(mono_signal, sample_rate, nperseg=TONALITY_FRAME)
    band = (freqs >= TONALITY_BAND_HZ[0]) & (freqs <= TONALITY_BAND_HZ[1])
    power = np.abs(spectrum[band]) ** 2 + 1e-20
    flatness = np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0)
    return float(np.median(flatness))


def _alpha_for(source_wav):
    """The subtraction factor this material can take.

    3.0 was set on real tape and is right for it in aggregate. Split by tonality it is not:
    on the most tonal third of the corpus the mode deviated 0.49 dB against cathar's 0.32,
    where on the noisiest third it was 0.32 against 0.56. Sustained tones sit near the
    noise profile and a speech-tuned factor subtracts them. At 2.0 on that third the
    deviation is 0.33 -- level with cathar -- and removal stays ahead at 7.97 against 7.00.
    The blend was suspected first and is not it: without it the tonal deviation is worse.
    """
    tonality = estimate_tonality(source_wav)
    if tonality is not None and tonality < APL_TONAL_FLATNESS_MAX:
        log_msg(f"    [Spectral Denoise] Tonal material (flatness {tonality:.3f}); subtracting at {APL_SPECTRAL_ALPHA_TONAL}.")
        return APL_SPECTRAL_ALPHA_TONAL
    return APL_SPECTRAL_ALPHA


def _subtract(source_wav, output_dir, total_duration):
    """Learns a noise profile from a quiet window and subtracts it."""
    from .cathar import _cathar_denoise_step, _cathar_noiseprint_step

    # Both values are passed explicitly: cathar shipped in v1.2.0 with a 0.75 s probe and
    # an alpha of its own, and those are shared settings this mode must not move.
    noiseprint = _cathar_noiseprint_step(source_wav, output_dir, duration_s=APL_NOISEPRINT_DURATION_S)
    return _cathar_denoise_step(
        source_wav,
        output_dir,
        alpha=_alpha_for(source_wav),
        noiseprint_path=noiseprint,
        total_duration=total_duration,
    )


def apply_when_needed(source_wav, audio_dir, total_duration=None):
    """Subtracts a learned noise profile when the neural denoiser cannot cope.

    The over-subtraction factor is deliberately gentler than the one `cathar` ships. At its
    2.5 the band energy lands 3.7-11.7 dB below the clean reference, taking programme
    content with the defect; 1.8 scored better on every defect family tested.

    Returns the input untouched when the Cathar CLI is unavailable, so this mode keeps
    working on a host that never provisioned it.
    """
    margin_db = should_apply(source_wav)
    if margin_db is None:
        return source_wav
    if not _cathar_available():
        log_msg("    [Spectral Denoise] Cathar CLI unavailable; leaving the neural stage to work alone.")
        return source_wav

    log_msg(f"    [Spectral Denoise] Programme sits {margin_db:.1f} dB above its noise floor; subtracting a learned profile.")
    output_dir = audio_dir / "spectral_denoised"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        subtracted = _subtract(source_wav, output_dir, total_duration)
    except (OSError, RuntimeError) as exc:
        log_msg(f"    [Spectral Denoise] Skipped after failure: {exc}")
        return source_wav
    return _repair_over_subtraction(source_wav, subtracted, output_dir)


def _repair_over_subtraction(source_wav, subtracted, output_dir):
    """Blends the subtracted result back toward its input where too much was removed.

    Subtraction cannot avoid over-cutting with one global strength: the band it clears of
    noise is also where the programme lives. The blend decides per frequency bin instead,
    using weights fitted against clean references, and repairs that damage rather than
    trading it for something else.
    """
    if not APL_ENABLE_LEARNED_BLEND or subtracted is None or subtracted == source_wav:
        return subtracted
    blended = apply_blend(source_wav, subtracted, output_dir / f"blended_{Path(subtracted).name}")
    if blended != subtracted:
        log_msg("    [Spectral Denoise] Repaired over-subtraction with the fitted per-bin blend.")
    return blended


def _profile_value(strategy, key):
    """Reads one scanned profile value, tolerating a missing or partial strategy."""
    if not isinstance(strategy, dict):
        return 0.0
    value = strategy.get("profile", {}).get(key)
    if value is None:
        value = strategy.get("precondition_filters", {}).get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dehum(source_wav, output_dir, notch_hz, total_duration):
    """Cancels mains hum across eight harmonics with adaptive tracking."""
    from .cathar import _cathar_dehum_step

    return _cathar_dehum_step(source_wav, output_dir, freq=notch_hz, total_duration=total_duration)


def _dewind(source_wav, output_dir, total_duration):
    """Removes low-frequency mechanical rumble."""
    from .cathar import _cathar_dewind_step

    return _cathar_dewind_step(source_wav, output_dir, total_duration=total_duration)


def _resolve_notch_hz(strategy, source_wav):
    """Returns the frequency to dehum at, decided by the recording rather than the scanner.

    The scanned value used to win. It is the region's nominal frequency when the scanner
    fires at all, and on 10 of the 48 corpus tapes that carry hum that is the wrong one --
    tapes cross regions -- so a dehum at it does nothing. When the recording cannot be read
    or scanned the scanned value is all there is, and it is used; a recording read and found
    free of hum is 0.0, and the scanned value does not override that.
    """
    if source_wav is None:
        return _profile_value(strategy, "notch_hz")
    detected = detect_mains_hz(source_wav)
    return _profile_value(strategy, "notch_hz") if detected is None else detected


def _tonal_targets(strategy, source_wav=None):
    """Returns the frequencies to clean, or None when nothing detected is switched on.

    Each stage is gated on its own defect having been found, because both cut real content
    when applied to material that does not have it. Dehum and dewind are switched
    separately: dehum measured as a gain on real tape and dewind did not.
    """
    notch_hz = _resolve_notch_hz(strategy, source_wav) if APL_ENABLE_DEHUM else 0.0
    highpass_hz = _profile_value(strategy, "highpass_hz") if APL_ENABLE_TONAL_CLEANUP else 0
    if notch_hz <= 0 and highpass_hz < 60:
        return None
    return notch_hz, highpass_hz


def _run_tonal_stages(source_wav, output_dir, targets, total_duration):
    """Applies whichever tonal stages the scan called for, in order."""
    notch_hz, highpass_hz = targets
    current = source_wav
    if notch_hz > 0:
        current = _dehum(current, output_dir, notch_hz, total_duration)
    if highpass_hz >= 60:
        current = _dewind(current, output_dir, total_duration)
    return current


def apply_tonal_cleanup(source_wav, audio_dir, strategy=None, total_duration=None):
    """Removes detected mains hum and rumble, for callers that switch either on.

    Both are off by default because neither earned a place end to end. Dehum is the
    instructive one: alone, at the frequency the harmonics support, it removes a median
    2.07 dB of hum on the 48 corpus tapes that carry it at 0.25 dB of speech-band movement.
    In the chain that becomes +0.66 dB on 20 of 48 tapes with the speech band moving 1.52
    to 2.15 dB, because the 2.5 s noise profile already captures stationary hum and the
    stage mostly relocates that removal. The frequency detection is kept correct regardless:
    on 10 of those 48 tapes the region's nominal mains frequency is the wrong one.

    Missing CLI or a failed stage leaves the audio untouched rather than failing the
    restoration.
    """
    targets = _tonal_targets(strategy, source_wav)
    if targets is None or not _cathar_available():
        return source_wav
    output_dir = audio_dir / "tonal_cleanup"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _run_tonal_stages(source_wav, output_dir, targets, total_duration)
    except (OSError, RuntimeError) as exc:
        log_msg(f"    [Tonal Cleanup] Skipped after failure: {exc}")
        return source_wav
