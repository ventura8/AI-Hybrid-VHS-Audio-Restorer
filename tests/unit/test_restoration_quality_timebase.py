"""R12, the time base: DTW-like holds and steps inserted into the output read; a pure lag, a gain, a mute and a speed error do not."""

import numpy as np
import pytest
import scipy.signal

from scripts.restoration_quality import dsp_metrics
from scripts.restoration_quality import timebase_metrics as tb

RATE = 22050
FRAME_S = 0.04
EVENTS = ((3.0, 0.3), (7.0, 0.5), (11.0, 0.3))


def _programme(seconds=15.0, rate=RATE, seed=3):
    """Speech-like programme with no periodic stretch: a gliding harmonic voice (random phases) over band noise, syllabic envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    f0 = 150.0 + 50.0 * np.sin(2 * np.pi * 0.7 * t) + 20.0 * np.sin(2 * np.pi * 3.1 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    voice = sum(np.sin(k * phase + rng.uniform(0, 2 * np.pi)) / k for k in range(1, 16))
    sos = scipy.signal.butter(4, [200.0, min(6000.0, 0.45 * rate)], btype="bandpass", fs=rate, output="sos")
    noise = scipy.signal.sosfiltfilt(sos, rng.standard_normal(len(t)))
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 2 * np.pi))
    mixed = envelope * (0.5 * voice / np.std(voice) + 0.5 * noise / np.std(noise))
    return (0.1 * mixed).astype(np.float32)


def _warped(mono, rate, events):
    """`mono` read through a DTW-like map: each `(at_s, gap_s)` holds one 40 ms frame at `at_s`, and plays one at 2x `gap_s` later.

    A `gap_s` of None never catches up: the output stays one frame late (a step).
    """
    speed = np.ones(len(mono))
    frame = int(FRAME_S * rate)
    for at_s, gap_s in events:
        start = int(at_s * rate)
        speed[start:][:frame] = 0.0
        if gap_s is not None:
            catch = int((at_s + gap_s) * rate)
            speed[catch:][:frame] = 2.0
    position = np.concatenate(([0.0], np.cumsum(speed)[:-1]))
    return np.interp(position, np.arange(len(mono)), mono).astype(np.float32)


def _read(source, output, rate=RATE):
    return tb.timebase_readings(source, output, rate)


@pytest.fixture(scope="module", name="programme")
def fixture_programme():
    """One 15 s programme at 22.05 kHz, shared by the tests that read it unchanged."""
    return _programme()


def test_identity_reads_nothing_and_tracks_the_programme(programme):
    """The source against itself: no hold, plateau or jump, no speed error, and nearly every programme frame tracked."""
    readings = _read(programme, programme.copy())
    assert (readings["holds"], readings["holds_per_min"], readings["plateaus"], readings["rate_jumps"]) == (0.0, 0.0, 0.0, 0.0)
    assert readings["speed_dev_pct"] < 0.01
    assert readings["tracked_share"] > 0.9


@pytest.mark.parametrize("rate", [16000, 22050, 44100, 48000])
def test_dtw_holds_and_their_catch_up_read_at_any_rate(rate):
    """Three DTW glitches (a frame held, a frame at 2x later): three holds, three 40 ms plateaus, a jump in and out of each."""
    source = _programme(rate=rate)
    readings = _read(source, _warped(source, rate, EVENTS), rate)
    assert readings["holds"] == 3.0
    assert readings["holds_per_min"] == pytest.approx(3.0 / 0.25, rel=1e-3)
    assert readings["plateaus"] == 3.0
    assert readings["rate_jumps"] == 6.0


def test_the_holds_sit_where_they_were_inserted(programme):
    """Each hold starts within 10 ms of its frame and lasts 20-45 ms."""
    found = tb.holds(programme, _warped(programme, RATE, EVENTS), RATE)
    assert len(found) == len(EVENTS)
    for (start, length), (at_s, _gap) in zip(found, EVENTS):
        assert abs(start - at_s) <= 0.01
        assert 0.02 <= length <= 0.045


def test_a_step_late_reads_one_hold_and_one_jump(programme):
    """A frame held and never caught up: the output plays one frame late from there on, a jump and no plateau to come back from."""
    readings = _read(programme, _warped(programme, RATE, [(7.0, None)]))
    assert (readings["holds"], readings["plateaus"], readings["rate_jumps"]) == (1.0, 0.0, 1.0)


def test_a_step_early_reads_one_jump_and_no_hold(programme):
    """A frame played at 2x with no hold to match: the output skips 40 ms ahead, a jump with nothing frozen."""
    speed = np.ones(len(programme))
    start = int(7.0 * RATE)
    speed[start:][: int(FRAME_S * RATE)] = 2.0
    position = np.minimum(np.concatenate(([0.0], np.cumsum(speed)[:-1])), len(programme) - 1)
    output = np.interp(position, np.arange(len(programme)), programme).astype(np.float32)
    readings = _read(programme, output)
    assert (readings["holds"], readings["plateaus"], readings["rate_jumps"]) == (0.0, 0.0, 1.0)


def test_a_two_frame_hold_reads_an_80_ms_plateau(programme):
    """Two frames held back to back, caught up 0.4 s later in two 2x frames: one hold and a plateau at two frames."""
    output = _warped(programme, RATE, [(5.0, 0.4), (5.04, 0.4)])
    readings = _read(programme, output)
    assert (readings["holds"], readings["plateaus"]) == (1.0, 1.0)
    lag = tb.lag_track(programme, output, RATE)
    sure = tb.confident(lag)
    plateau = (lag["times_s"] > 5.15) & (lag["times_s"] < 5.35) & sure
    assert np.median(lag["lag_ms"][plateau]) == pytest.approx(80.0, abs=1.0)


@pytest.mark.parametrize("delay", [1, 123, int(0.003 * RATE)])
def test_a_pure_lag_reads_nothing(programme, delay):
    """An output late by a constant (the runner aligns it first; the reading must not need that): no event, no speed error."""
    output = np.concatenate([np.zeros(delay, dtype=np.float32), programme[:-delay]])
    readings = _read(programme, output)
    assert (readings["holds"], readings["plateaus"], readings["rate_jumps"]) == (0.0, 0.0, 0.0)
    assert readings["speed_dev_pct"] < 0.01


@pytest.mark.parametrize("gain", [0.25, 2.0])
def test_a_gain_change_reads_nothing(programme, gain):
    """A static gain moves no reading."""
    readings = _read(programme, (gain * programme).astype(np.float32))
    assert (readings["holds"], readings["plateaus"], readings["rate_jumps"]) == (0.0, 0.0, 0.0)


def test_a_slow_gain_ride_reads_nothing(programme):
    """A +-6 dB ride at 0.5 Hz (loudnorm's dynamic mode at its worst) is level, not time."""
    t = np.arange(len(programme)) / RATE
    ride = 10.0 ** (6.0 * np.sin(2 * np.pi * 0.5 * t) / 20.0)
    readings = _read(programme, (programme * ride).astype(np.float32))
    assert (readings["holds"], readings["plateaus"], readings["rate_jumps"]) == (0.0, 0.0, 0.0)


def test_a_muted_hole_is_a_dropout_not_a_hold(programme):
    """A 40 ms mute holds no value: `file.dropouts` counts it, the time base does not."""
    output = programme.copy()
    start = int(6.0 * RATE)
    output[start:][: int(FRAME_S * RATE)] = 0.0
    readings = _read(programme, output)
    assert (readings["holds"], readings["plateaus"], readings["rate_jumps"]) == (0.0, 0.0, 0.0)
    assert dsp_metrics.dropout_count(programme, output, RATE) == 1


def test_digital_silence_holding_the_loaders_dc_residue_is_no_hold(programme):
    """An output emptied to digital zero, less the loader's DC removal (-100 dBFS), where the source plays: no hold."""
    output = programme.copy()
    start = int(4.0 * RATE)
    output[start:][: int(0.1 * RATE)] = -1e-5
    assert _read(programme, output)["holds"] == 0.0


@pytest.mark.parametrize("error", [0.003, -0.002])
def test_a_constant_speed_error_reads_itself(programme, error):
    """The output played 0.3 % slow (0.2 % fast): the speed reading recovers it within 10 %, and no event reads."""
    position = np.arange(len(programme)) * (1.0 - error)
    output = np.interp(position, np.arange(len(programme)), programme).astype(np.float32)
    readings = _read(programme, output)
    assert readings["speed_dev_pct"] == pytest.approx(abs(error) * 100.0, rel=0.1)
    assert (readings["holds"], readings["plateaus"], readings["rate_jumps"]) == (0.0, 0.0, 0.0)


def test_a_slow_sinusoidal_speed_error_reads_its_depth(programme):
    """0.3 % at 0.2 Hz in the output only reads its depth times sin(pi f) / (pi f) (0.935) within 10 %."""
    t = np.arange(len(programme)) / RATE
    depth, rate_hz = 0.003, 0.2
    position = (t + depth / (2 * np.pi * rate_hz) * (1.0 - np.cos(2 * np.pi * rate_hz * t))) * RATE
    output = np.interp(position, np.arange(len(programme)), programme).astype(np.float32)
    expected = depth * 100.0 * np.sin(np.pi * rate_hz) / (np.pi * rate_hz)
    assert _read(programme, output)["speed_dev_pct"] == pytest.approx(expected, rel=0.1)


def test_a_periodic_programme_is_not_tracked():
    """A steady harmonic tone correlates as well one period away: no frame is confident, so the track readings are None."""
    t = np.arange(int(5.0 * RATE)) / RATE
    tone = (0.1 * sum(np.sin(2 * np.pi * 100.0 * k * t + k) / k for k in range(1, 10))).astype(np.float32)
    readings = _read(tone, tone.copy())
    assert readings["plateaus"] is None and readings["tracked_share"] is None
    assert readings["holds"] == 0.0


def test_a_pair_shorter_than_one_search_reads_holds_only():
    """Under 260 ms there is no track; the hold count is still read."""
    short = _programme(seconds=0.2)
    readings = _read(short, short.copy())
    assert readings["holds"] == 0.0
    assert readings["holds_per_min"] == 0.0
    assert all(readings[name] is None for name in ("plateaus", "rate_jumps", "speed_dev_pct", "tracked_share"))
    assert tb.lag_track(short, short, RATE)["lag_ms"].size == 0


def test_an_empty_pair_reads_no_hold_and_no_rate():
    """Nothing to read: no hold, no per-minute rate."""
    empty = np.zeros(0, dtype=np.float32)
    readings = _read(empty, empty)
    assert (readings["holds"], readings["holds_per_min"]) == (0.0, None)
    assert tb.dipped_frames(np.zeros(0), np.zeros(0)).size == 0
    assert tb.programme({"energy": np.zeros(0)}).size == 0


JUMP_TIMES = np.arange(45) * 0.01


@pytest.mark.parametrize("change_ms, expected", [(40.0, 1), (-79.0, 1), (96.0, 0), (20.0, 0)])
def test_only_a_held_change_of_whole_frames_is_a_jump(change_ms, expected):
    """A held change of 40 or 80 ms (either sign) is one jump; 96 ms (a repeat of the programme) or 20 ms is none."""
    assert tb.count_jumps(JUMP_TIMES, np.where(np.arange(45) >= 20, change_ms, 0.0)) == expected


def test_two_stray_frames_or_a_lone_frame_are_no_jump():
    """Two frames at a wrong whole-frame lag are no jump in and out (the five-frame median); one frame has no pair."""
    stray = np.zeros(45)
    stray[20:22] = 40.0
    assert tb.count_jumps(JUMP_TIMES, stray) == 0
    assert tb.count_jumps(JUMP_TIMES[:1], stray[:1]) == 0


def test_frames_far_apart_are_no_jump():
    """Two confident stretches more than 200 ms apart are not compared."""
    times = np.concatenate([np.arange(10) * 0.01, 1.0 + np.arange(10) * 0.01])
    lags = np.concatenate([np.zeros(10), np.full(10, 40.0)])
    assert tb.count_jumps(times, lags) == 0


@pytest.mark.parametrize(
    "steps, frames, expected",
    [
        ([0, 1, 1, 1, 0, -1, -1, 2], [0, 1, 2, 3, 4, 5, 6, 7], 2),
        ([1, 1], [0, 51], 0),
        ([1, 1], [0, 50], 1),
        ([1, 0, 1], [0, 1, 2], 0),
        ([], [], 0),
    ],
)
def test_plateaus_need_two_frames_on_one_step(steps, frames, expected):
    """A run on one whole step counts once from its second frame; over 0.5 s unread, or a change of step, breaks it."""
    assert tb.count_plateaus(steps, frames) == expected


def test_the_step_index_takes_one_or_two_frames_within_2_ms():
    """40 and 80 ms (either sign) within 2 ms are steps; 0, 20, 43 and 120 ms are not."""
    deviation = np.array([0.0, 20.0, 39.0, -41.5, 43.0, 80.5, -79.0, 120.0])
    assert tb.step_index(deviation).tolist() == [0, 0, 1, -1, 0, 2, -2, 0]


def test_the_local_lag_needs_a_read_block():
    """Under ten confident frames in every 0.5 s block there is no local lag and no speed reading."""
    times = np.array([0.1, 0.6, 1.1, 1.6])
    lags = np.zeros(4)
    assert tb.local_lag(times, lags) is None
    assert tb.speed_dev_pct(times, lags) is None


def test_the_rolling_median_skips_unread_blocks():
    """The running median ignores NaN blocks, and a stretch of NaN only stays NaN."""
    smooth = tb.rolling_nanmedian(np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan]), 3)
    assert smooth[:3].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(smooth[5])


def test_the_decimation_and_the_track_sizes_follow_the_rate():
    """44.1 and 48 kHz are read every 4th sample, 22.05 kHz every 2nd, 16 kHz as they are; the window is 60 ms at any rate."""
    assert [tb.decimation(rate) for rate in (16000, 22050, 44100, 48000, 96000)] == [1, 2, 4, 4, 8]
    assert tb.track_sizes(11025.0) == (662, 110, 1102)
    assert tb.track_sizes(16000.0) == (960, 160, 1600)


def test_the_card_entries_are_paired_against_a_zero_source(programme):
    """Every R12 reading lands as `file.timebase_*` with a zero source side and the reading as its delta."""
    entries = tb.timebase_entries(programme, _warped(programme, RATE, EVENTS[:1]), RATE)
    assert set(entries) == {f"file.timebase_{name}" for name in tb.READINGS}
    assert entries["file.timebase_holds"] == {"source": 0.0, "output": 1.0, "delta": 1.0}
    assert all(entry["source"] == 0.0 and entry["delta"] == entry["output"] for entry in entries.values())
