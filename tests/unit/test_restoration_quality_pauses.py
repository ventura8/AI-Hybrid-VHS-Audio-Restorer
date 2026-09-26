"""The pause readings hear dead air, hiss left in the gaps, and a window that fell silent."""

import numpy as np
import scipy.signal

from scripts.restoration_quality import pause_metrics
from scripts.restoration_quality.scorecard import WindowRow

RATE = 44100


def _voice(seconds=8.0, seed=3, hiss=3e-3):
    """Bright harmonic bursts (0.5 s on, 0.5 s off) over a steady hiss: loud frames, gaps and deep pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    phases = rng.uniform(0.0, 2 * np.pi, 60)
    voice = sum(np.sin(2 * np.pi * 150.0 * k * t + phases[k]) / np.sqrt(k) for k in range(1, 60))
    ramp = np.hanning(int(0.05 * RATE))
    gate = np.convolve(((t % 1.0) < 0.5).astype(np.float64), ramp / ramp.sum(), mode="same")
    return (0.05 * voice * gate + hiss * rng.standard_normal(len(t))).astype(np.float32)


def _smooth_mask(mask, attack_s, release_s):
    """A one-pole follower: fast to open, slow to close, like the polish expander."""
    out = np.zeros(len(mask))
    a_up, a_down = np.exp(-1.0 / (attack_s * RATE)), np.exp(-1.0 / (release_s * RATE))
    level = 0.0
    for i, target in enumerate(mask):
        coeff = a_up if target > level else a_down
        level = coeff * level + (1.0 - coeff) * target
        out[i] = level
    return out


def _speech_mask(source):
    frame = int(0.02 * RATE)
    level = np.repeat(np.sqrt(np.mean(source[: len(source) // frame * frame].reshape(-1, frame) ** 2, axis=1)), frame)
    # Speech sits far above the hiss; a percentile cut would land inside the pause cluster.
    return np.concatenate([level, np.full(len(source) - len(level), level[-1])]) > 4.0 * np.percentile(level, 10)


def _gated(source, floor_db=-30.0):
    gain = _smooth_mask(_speech_mask(source).astype(np.float64), 0.04, 0.18)
    return (source * (gain + (1.0 - gain) * 10 ** (floor_db / 20.0))).astype(np.float32)


def _hissed(source, level=1.5e-2, seed=9):
    noise = np.random.default_rng(seed).standard_normal(len(source))
    sos = scipy.signal.butter(4, 3000.0, btype="highpass", fs=RATE, output="sos")
    quiet = 1.0 - _smooth_mask(_speech_mask(source).astype(np.float64), 0.04, 0.18)
    return (source + level * scipy.signal.sosfiltfilt(sos, noise) * quiet).astype(np.float32)


def test_identity_reads_zero_delta_and_the_source_keeps_its_air():
    source = _voice()
    readings = pause_metrics.pause_readings(source, source, RATE)
    for name in pause_metrics.NAMES:
        src, out = readings[name]
        assert abs(src - out) < 1e-9
    assert -30.0 < readings["gap_air_db"][0] < 0.0


def test_gated_pauses_read_as_dead_air_and_deeper_pauses():
    source = _voice()
    readings = pause_metrics.pause_readings(source, _gated(source), RATE)
    air_src, air_out = readings["gap_air_db"]
    depth_src, depth_out = readings["pause_depth_db"]
    assert air_out < air_src - 10.0
    # The 180 ms release reaches the floor only late in each 0.5 s pause, so the deep frames sink by ~10 dB.
    assert depth_out > depth_src + 8.0


def test_hiss_in_the_gaps_reads_as_more_air():
    source = _voice()
    air_src, air_out = pause_metrics.pause_readings(source, _hissed(source), RATE)["gap_air_db"]
    assert air_out > air_src + 6.0


def test_too_little_material_reads_none():
    short = _voice(seconds=0.5)
    assert pause_metrics.pause_readings(short, short, RATE) == {name: (None, None) for name in pause_metrics.NAMES}
    flat = np.full(6 * RATE, 0.01, dtype=np.float32)
    assert pause_metrics.pause_readings(flat, flat, RATE)["gap_air_db"] == (None, None)


def test_level_classes_are_disjoint():
    deep, gap, loud = pause_metrics.level_classes(np.linspace(-80.0, -10.0, 200))
    assert not (deep & gap).any()
    assert not (gap & loud).any()


def test_level_classes_each_hold_enough_frames():
    deep, gap, loud = pause_metrics.level_classes(np.linspace(-80.0, -10.0, 200))
    assert deep.sum() >= 20
    assert gap.sum() >= 20
    assert loud.sum() >= 20


def test_runner_flags_a_window_whose_output_fell_silent():
    from scripts.restoration_quality import runner

    class _Pair:
        rate = RATE
        source = _voice(seconds=2.0)
        output = np.zeros(len(source), dtype=np.float32)

    row = WindowRow(0, 0.0, 2.0, "speech", output_route="silence")
    runner._listener_window(_Pair(), row)
    assert row.output["dsp.output_silent"] == 1.0
    assert row.output["dsp.gap_air_db"] is None
    kept = WindowRow(0, 0.0, 2.0, "speech", output_route="speech")
    runner._listener_window(_Pair(), kept)
    assert kept.output["dsp.output_silent"] == 0.0
