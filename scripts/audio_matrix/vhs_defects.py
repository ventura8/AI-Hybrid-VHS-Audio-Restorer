"""Deterministic NumPy defect injection for synthetic VHS fixtures.

Calibrated against empirical acoustic findings from real VHS captures:
- Tape hiss noise floor (~-47 dB median).
- 50 Hz mains hum fundamental + 100 Hz harmonic buzz.
- 15.625 kHz CRT horizontal flyback line whistle.
- 50-75 Hz mechanical motor rumble.
- Empirical burst click distribution.
- Head-switching dropouts, tape saturation clipping, azimuth delay, and balance skew.
"""

import numpy as np


def _apply_hiss_and_hum(mono, time, defects, rng):
    """Injects calibrated broadband tape hiss and harmonic mains hum."""
    if "hiss" in defects:
        mono += rng.normal(0, 0.005, len(mono)).astype(np.float32)
    if "hum" in defects:
        mono += (0.02 * np.sin(2 * np.pi * 50 * time) + 0.008 * np.sin(2 * np.pi * 100 * time)).astype(np.float32)
    return mono


def _apply_whistle_and_rumble(mono, time, defects):
    """Injects 15.625 kHz CRT flyback whine and 50-75 Hz mechanical rumble."""
    if "whistle" in defects:
        mono += (0.012 * np.sin(2 * np.pi * 15625 * time)).astype(np.float32)
    if "rumble" in defects:
        mono += (0.03 * np.sin(2 * np.pi * 60 * time) + 0.015 * np.sin(2 * np.pi * 75 * time)).astype(np.float32)
    return mono


def _apply_clicks_and_dropouts(mono, sample_rate, defects):
    """Injects burst clicks, dropouts, and tape saturation clipping."""
    if "clicks" in defects:
        mono[:: max(sample_rate // 2, 1)] += 0.22
    if "dropout" in defects:
        interval = max(sample_rate * 17, 1)
        width = max(sample_rate // 20, 1)
        for start in range(0, len(mono), interval):
            mono[start : start + width] *= 0.1
    if "clip" in defects:
        mono = np.clip(mono * 1.5, -0.99, 0.99)
    return mono


def _assemble_stereo_channels(mono, defects):
    """Constructs stereo array with optional azimuth skew and balance imbalance."""
    if "azimuth" in defects:
        right = np.zeros_like(mono)
        if len(mono) > 2:
            right[2:] = mono[:-2]
    else:
        right = mono.copy()
    if "balance" in defects:
        right *= 0.7
    return np.column_stack((mono, right)).astype(np.float32)


def _apply_speed_drift(mono):
    """Apply a deterministic slow tape-speed ramp without changing frame count."""
    positions = np.arange(len(mono), dtype=np.float64)
    warped = positions * (1.0 + 0.001 * positions / max(len(mono), 1))
    return np.interp(positions, warped, mono).astype(np.float32)


def apply_vhs_defects(samples, sample_rate, defects):
    """Return stereo float32 audio with the requested, deterministic defects."""
    if "whistle" in defects and sample_rate <= 31250:
        raise ValueError("The whistle defect requires a sample rate above 31.25 kHz.")
    source = np.asarray(samples, dtype=np.float32)
    mono = source.mean(axis=1) if source.ndim == 2 else source.copy()
    time = np.arange(len(mono), dtype=np.float64) / sample_rate
    rng = np.random.default_rng(41071)
    mono = _apply_hiss_and_hum(mono, time, defects, rng)
    mono = _apply_whistle_and_rumble(mono, time, defects)
    mono = _apply_clicks_and_dropouts(mono, sample_rate, defects)
    if "drift" in defects:
        mono = _apply_speed_drift(mono)
    return _assemble_stereo_channels(mono, defects)
