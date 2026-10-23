"""The output's windows are routed at the source's level, so a collapsed projection gain is not read as silence."""

import numpy as np

from scripts.restoration_quality import audio_io

RATE = 16000


def _speechlike(seconds, seed=1, level=0.1):
    """Noise bursts at a syllable rate: routed as programme at a normal level."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    envelope = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 4.0 * t))
    return (level * envelope * rng.standard_normal(t.size)).astype(np.float32)


def test_a_level_matched_copy_takes_the_reference_rms():
    reference = _speechlike(2.0, level=0.2)
    estimate = reference * np.float32(1e-4)
    matched = audio_io.level_matched(reference, estimate)
    assert np.isclose(np.sqrt(np.mean(matched.astype(np.float64) ** 2)), np.sqrt(np.mean(reference.astype(np.float64) ** 2)), rtol=1e-4)


def test_an_all_zero_or_empty_estimate_stays_as_it_is():
    reference = _speechlike(1.0)
    assert not audio_io.level_matched(reference, np.zeros(RATE, dtype=np.float32)).any()
    assert audio_io.level_matched(reference, np.zeros(0, dtype=np.float32)).size == 0


def test_an_output_the_projection_scaled_down_still_reads_as_programme():
    source = _speechlike(15.0, seed=2)
    output = source * np.float32(1e-4)
    windows = audio_io.windows(len(source), RATE, 15.0, 7.5)
    assert audio_io.route_window(output[windows[0].slice_of(RATE)], RATE) == "silence"
    assert "silence" not in audio_io.output_routes(source, output, windows, RATE)


def test_a_window_the_output_really_dropped_still_reads_as_silence():
    source = _speechlike(30.0, seed=3)
    output = source.copy()
    windows = audio_io.windows(len(source), RATE, 15.0, 7.5)
    output[windows[-1].slice_of(RATE)] = 0.0
    assert audio_io.output_routes(source, output, windows, RATE)[-1] == "silence"
