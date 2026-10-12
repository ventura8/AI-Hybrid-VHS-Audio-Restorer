"""The `sync_holds` calibration case: nested DTW glitches R12 resolves one by one, and a table row that asserts them."""

import numpy as np

from scripts import degradations_v3 as v3
from scripts import quality_degradations as deg
from scripts.restoration_quality import timebase_metrics as tb
from tests.unit.test_restoration_quality_timebase import RATE, _programme


def test_the_glitches_are_nested_spaced_and_fit_the_signal():
    """A level's glitch starts are the first ones of every higher level's, 1 s apart, each with room to catch up."""
    base = _programme()
    starts = {count: v3.sync_hold_starts(base, RATE, count) for count in (1, 3, 6)}
    assert starts[3][:1] == starts[1] and starts[6][:3] == starts[3]
    ordered = sorted(starts[6])
    assert np.min(np.diff(ordered)) >= 0.99 * RATE
    assert ordered[-1] + int((v3.SYNC_CATCH_UP_S + 2 * v3.SYNC_FRAME_S) * RATE) <= len(base)


def test_each_hold_freezes_one_frame_on_a_loud_sample():
    """Every sample of a held frame is the value the hold started on, the loudest of its first 10 ms."""
    base = _programme()
    output = v3.sync_holds(base, RATE, 3)
    frame = int(round(v3.SYNC_FRAME_S * RATE))
    around = int(0.01 * RATE)
    for start in v3.sync_hold_starts(base, RATE, 3):
        held = output[start:][:frame]
        assert np.all(held == held[0]) and held[0] == base[start]
        first = start - around
        nearby = base[first:][: 2 * around].astype(np.float64)
        assert abs(held[0]) > 0.5 * np.sqrt(np.mean(nearby**2))
    assert len(output) == len(base)


def test_the_time_base_reads_every_glitch_and_nothing_else():
    """R12 reads holds 1 / 3 / 6 and as many plateaus; the output outside the glitches is the source itself."""
    base = _programme()
    for count in (1, 3, 6):
        source, output = deg.apply("sync_holds", count, base, RATE, {"speech": base}, np.random.default_rng(3))
        readings = tb.timebase_readings(source, output, RATE)
        assert (readings["holds"], readings["plateaus"]) == (float(count), float(count))
        assert readings["rate_jumps"] >= count
    untouched = v3.sync_holds(base, RATE, 1)[: v3.sync_hold_starts(base, RATE, 1)[0]]
    assert np.array_equal(untouched, base[: len(untouched)])


def test_a_glitch_in_silence_changes_nothing_and_a_short_signal_has_no_room():
    """Held digital silence is still silence; a signal too short for a glitch and its catch-up gets none."""
    silence = np.zeros(RATE * 3, dtype=np.float32)
    assert np.array_equal(v3.sync_holds(silence, RATE, 3), silence)
    short = _programme(seconds=0.3)
    assert v3.sync_hold_starts(short, RATE, 3) == []
    assert np.array_equal(v3.sync_holds(short, RATE, 3), short)


def test_the_table_asserts_the_holds_and_the_jumps_and_holds_the_sync_under_its_gate():
    """The row asserts R12's two gated readings up, keeps the 3-anchor sync under 40 ms and unmatched at 0."""
    spec = deg.DEGRADATIONS["sync_holds"]
    asserted = {(e.metric, e.direction, e.tolerance) for e in spec.expects if not e.blind}
    assert asserted == {
        (deg.TIMEBASE_HOLDS, "up", None),
        (deg.TIMEBASE_JUMPS, "up", None),
        ("file.sync_drift_ms", "flat", deg.SYNC_GATE_MS),
        ("file.sync_unmatched", "flat", 0.0),
    }
    assert deg.SYNC_GATE_MS == 40.0


def test_the_row_is_built_by_its_own_family_and_runs_on_a_tape():
    """`sync_holds` has its own builder and a real excerpt can carry it (it needs no clean reference)."""
    assert deg.v3_builder("sync_holds") is deg.V3_BUILDERS["sync_holds"]
    assert "sync_holds" in deg.on_tape()


def test_the_drift_row_reads_the_speed_and_no_time_base_event():
    """A speed error the sync missed moves R12's speed reading and none of its gated counts."""
    expects = {(e.metric, e.direction, e.tolerance) for e in deg.DEGRADATIONS["sync_drift"].expects}
    assert {(deg.TIMEBASE_SPEED, "up", None), (deg.TIMEBASE_HOLDS, "flat", 0.0), (deg.TIMEBASE_JUMPS, "flat", 0.0)} <= expects
