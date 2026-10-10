"""R2 matches the gain around each 's', so a slow level rider is no change to the 's'; the texture needs 30 frames.

Split from `test_restoration_quality_sibilance.py` (its voice and readers are shared) to keep
each file's maintainability index at grade A.
"""

import numpy as np
import pytest

from scripts import degradations_v3 as v3
from scripts.restoration_quality import sibilance
from scripts.restoration_quality.dsp_metrics import framed_psd
from tests.unit.test_restoration_quality_sibilance import RATE, _r2, _voice_with_esses


def _ridden_source(seed=7):
    """3 s of vowels with 's' bursts, then 3 s of vowels alone 10 dB louder: a rider lifts the first half, lowers the second."""
    voice, _gate = _voice_with_esses(seconds=6.0, seed=seed)
    t = np.arange(len(voice)) / RATE
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2 * np.pi, 30)
    vowel = sum(np.sin(2 * np.pi * 160.0 * k * t + phases[k]) / k for k in range(1, 30))
    ramp = np.hanning(int(0.01 * RATE))
    gate = np.convolve(((t % 0.6) < 0.3).astype(np.float64), ramp / ramp.sum(), mode="same")
    loud = 0.1 * 10.0 ** (10.0 / 20.0) * vowel * gate + 1e-4 * rng.standard_normal(len(t))
    half = len(voice) // 2
    return np.concatenate([voice[:half], loud[half:]]).astype(np.float32)


def _window_gain_level(source, output):
    """R2's absolute level with the one window gain the reading subtracted before the local match."""
    frame = int(sibilance.FRAME_S * RATE)
    count = len(source) // frame
    freqs, src_power, src_level = framed_psd(source[: count * frame], RATE, frame)
    _freqs, out_power, _level = framed_psd(output[: count * frame], RATE, frame)
    fricative, _plain = sibilance._classes((src_power, freqs), 20.0 * np.log10(src_level + 1e-9))
    bins = (freqs >= sibilance.SIB_BAND_HZ[0]) & (freqs < sibilance.SIB_BAND_HZ[1])
    gain = sibilance._programme_gain_db(source, output, RATE)
    return sibilance.abs_level_db(src_power[fricative][:, bins], out_power[fricative][:, bins], gain)


@pytest.mark.parametrize("depth", [0.3, 0.9])
def test_a_slow_level_rider_leaves_the_absolute_level_alone(depth):
    """loudnorm's dynamic mode as a rider: the local match reads no change; the window gain read the ride (2.3 / 6.9 dB)."""
    source = _ridden_source()
    output = v3.loudnorm_ride(source, RATE, depth)
    level, _texture = _r2(source, output)
    assert abs(level) < 0.05
    assert abs(_window_gain_level(source, output)) > 0.1


def test_on_a_static_gain_every_local_gain_is_the_window_gain():
    """One gain for the window: every 's' frame's gain is R1's window gain within 0.01 dB."""
    source = _voice_with_esses()[0]
    output = (0.5 * source).astype(np.float32)
    gains = sibilance.frame_gains_db(source, output, RATE, np.linspace(0.2, 5.8, 20))
    assert np.allclose(gains, sibilance._programme_gain_db(source, output, RATE), atol=0.01)


def test_a_frame_with_too_few_cells_near_takes_the_window_gain():
    """Past the window's end no programme cell is within 0.5 s: the frame takes the window's gain."""
    source = _voice_with_esses()[0]
    output = (0.5 * source).astype(np.float32)
    window = sibilance._programme_gain_db(source, output, RATE)
    assert sibilance.frame_gains_db(source, output, RATE, np.array([60.0]))[0] == window


def test_no_window_gain_leaves_no_local_gains():
    """Where R1 refuses the window there is no gain to match, locally or at all."""
    quiet = (_voice_with_esses()[0] * 1e-3).astype(np.float32)
    assert sibilance.frame_gains_db(quiet, quiet, RATE, np.array([1.0])) is None


def test_the_texture_needs_thirty_fricative_frames_the_level_ten():
    """1.2 s of the voice holds 24 fricative frames: the level reads, the texture does not; 1.8 s (36) reads both."""
    short = _voice_with_esses(seconds=1.2)[0]
    level, texture = _r2(short, (0.5 * short).astype(np.float32))
    assert level is not None
    assert texture is None
    longer = _voice_with_esses(seconds=1.8)[0]
    assert _r2(longer, (0.5 * longer).astype(np.float32))[1] is not None


def test_fricative_times_are_the_frame_centres():
    """Frames 0 and 3 of 10 ms centre at 5 ms and 35 ms."""
    assert np.allclose(sibilance.fricative_times(np.array([True, False, False, True]), RATE), [0.005, 0.035])
