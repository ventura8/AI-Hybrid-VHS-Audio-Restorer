"""The `dropouts` holes are nested, resolvable and on programme; `file.dropouts` counts them where window medians go blind."""

import functools
import types

import numpy as np
import pytest

from scripts import degradations_v3 as v3
from scripts import quality_degradations as deg
from scripts.restoration_quality import audio_io, dsp_metrics, runner
from tests.unit.test_restoration_quality_file_metrics import RATE, _speech

LEVELS = deg.DEGRADATIONS["dropouts"].levels
FRAME = int(dsp_metrics.DROPOUT_FRAME_S * RATE)


@functools.lru_cache(maxsize=None)
def _minute():
    return _speech(60.0)


@functools.lru_cache(maxsize=None)
def _holed(level):
    return deg.dropouts(_minute(), RATE, level)


def _zeros(out):
    return set(np.flatnonzero(out == 0.0).tolist())


def test_the_levels_are_nested():
    """Every sample a level zeroes, each higher level zeroes too: its holes are the first ones of the next level's."""
    zeros = [_zeros(_holed(level)) - _zeros(_minute()) for level in LEVELS]
    assert zeros[0] < zeros[1] < zeros[2]


def _spans():
    return v3.hole_spans(_minute(), RATE, LEVELS[-1], np.random.default_rng(v3.HOLE_SEED))


def _on_programme(span, programme):
    """Whether a hole starts on a frame edge and every 10 ms frame it covers carries programme."""
    start, length = span
    first, last = start // FRAME, (start + length - 1) // FRAME
    return start % FRAME == 0 and bool(programme[first:][: last - first + 1].all())


def test_every_hole_lasts_30_to_50_ms_and_sits_on_programme_frames():
    """Each hole is two 10 ms frames or more whatever its phase, and every frame it covers carries programme."""
    programme = dsp_metrics.programme_frames(dsp_metrics.frame_levels(_minute(), FRAME))
    spans = _spans()
    assert len(spans) == LEVELS[-1]
    assert all(0.030 * RATE <= length <= 0.050 * RATE for _start, length in spans)
    assert all(_on_programme(span, programme) for span in spans)


def test_the_holes_lie_a_quarter_second_apart():
    """No two holes sit within 250 ms, so no two runs merge."""
    starts = sorted(start for start, _length in _spans())
    assert min(np.diff(starts)) >= v3.HOLE_SPACING_S * RATE


@pytest.mark.parametrize("level", LEVELS)
def test_the_whole_pair_counts_exactly_the_level(level):
    """3, 6 and 12 holes read 3, 6 and 12 over the whole pair."""
    assert dsp_metrics.dropout_count(_minute(), _holed(level), RATE) == level


def test_a_dropouts_case_never_draws_from_the_shared_generator():
    """The holes come from their own seed: the case leaves the shared stream where it found it."""
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state
    deg.apply("dropouts", LEVELS[0], _minute(), RATE, {}, rng)
    assert rng.bit_generator.state == state


def test_file_dropouts_counts_sparse_holes_the_window_median_misses():
    """Three holes in 300 s: the whole pair reads 3, the median over the 15 s windows reads 0."""
    source = _speech(300.0)
    output = deg.dropouts(source, RATE, 3)
    entry = runner.dropout_entry(types.SimpleNamespace(source=source, output=output, rate=RATE))
    assert entry == {"file.dropouts": {"source": 0.0, "output": 3.0, "delta": 3.0}}
    per_window = [
        dsp_metrics.dropout_count(source[w.slice_of(RATE)], output[w.slice_of(RATE)], RATE) for w in audio_io.windows(len(source), RATE)
    ]
    assert np.median(per_window) == 0.0
