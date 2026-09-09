"""CPU-only tests for the VHS defect injector that builds validation fixtures.

The hardware suite only checks manifest metadata and dry-run planning, so a regression in
this module -- non-deterministic output, a wrong shape or dtype, or a defect that stops being
injected -- would sail through while quietly producing invalid fixtures. Every fixture the
hardware validation measures comes out of here, so its ground truth has to be pinned.
"""

import numpy as np
import pytest

from scripts.audio_matrix.vhs_defects import apply_vhs_defects

SAMPLE_RATE = 44100


def _speech_like(seconds=1.0, freq=180.0):
    """A voiced tone standing in for the Piper speech the fixtures are built from."""
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False, dtype=np.float32)
    return (0.25 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _band_energy(mono, low_hz, high_hz):
    """Energy within a frequency band, used to assert a defect actually landed."""
    spectrum = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / SAMPLE_RATE)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sum(spectrum[band] ** 2))


def test_output_is_stereo_float32():
    """Fixtures must be stereo float32 whatever the mono input was."""
    out = apply_vhs_defects(_speech_like(), SAMPLE_RATE, ["hiss"])
    assert out.dtype == np.float32
    assert out.ndim == 2 and out.shape[1] == 2


def test_injection_is_deterministic():
    """The same input and defects must produce byte-identical audio on every run."""
    source = _speech_like()
    first = apply_vhs_defects(source, SAMPLE_RATE, ["hiss", "hum", "rumble"])
    second = apply_vhs_defects(source, SAMPLE_RATE, ["hiss", "hum", "rumble"])
    assert np.array_equal(first, second)


def test_hum_lands_at_mains_frequency():
    """The hum defect adds energy around 50 Hz, which is what the benchmarks measure."""
    clean = _speech_like()
    dirty = apply_vhs_defects(clean, SAMPLE_RATE, ["hum"])[:, 0]
    assert _band_energy(dirty, 45.0, 55.0) > _band_energy(clean, 45.0, 55.0) * 10


def test_whistle_lands_near_the_crt_line_frequency():
    """The whistle defect adds energy near 15.625 kHz, the PAL CRT line rate."""
    clean = _speech_like()
    dirty = apply_vhs_defects(clean, SAMPLE_RATE, ["whistle"])[:, 0]
    assert _band_energy(dirty, 15500.0, 15750.0) > _band_energy(clean, 15500.0, 15750.0) * 10


def test_hiss_raises_the_noise_floor():
    """Hiss must raise broadband energy; a silent 'defect' would fake a clean fixture."""
    clean = _speech_like()
    dirty = apply_vhs_defects(clean, SAMPLE_RATE, ["hiss"])[:, 0]
    assert float(np.sqrt(np.mean(dirty**2))) > float(np.sqrt(np.mean(clean**2)))


def test_azimuth_offsets_the_channels():
    """Azimuth skew is a stereo defect: the two channels must stop being identical."""
    out = apply_vhs_defects(_speech_like(), SAMPLE_RATE, ["azimuth"])
    assert not np.array_equal(out[:, 0], out[:, 1])


def test_unknown_defect_is_rejected():
    """An unsupported name fails loudly rather than silently producing a clean fixture."""
    with pytest.raises(ValueError, match="Unknown VHS defect"):
        apply_vhs_defects(_speech_like(), SAMPLE_RATE, ["not_a_defect"])


def test_whistle_requires_a_sample_rate_that_can_represent_it():
    """15.625 kHz cannot be represented below a 31.25 kHz sample rate; that must raise."""
    with pytest.raises(ValueError, match="whistle defect requires"):
        apply_vhs_defects(_speech_like(), 22050, ["whistle"])
