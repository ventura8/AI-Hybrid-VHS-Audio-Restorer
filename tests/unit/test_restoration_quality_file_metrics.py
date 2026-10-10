"""The file-level readings hear a level riding under the speech, not a denoiser emptying the pauses or a late output."""

import functools

import numpy as np
import pyloudnorm

from scripts.restoration_quality import file_metrics

RATE = 16000
PAL_FRAME_MS = 40.0


def _speech(seconds, rate=RATE, seed=5, hiss=0.0):
    """Harmonic syllables of random length and level with random gaps: an envelope that never repeats."""
    rng = np.random.default_rng(seed)
    count = int(seconds * rate)
    t = np.arange(count) / rate
    gate = np.zeros(count)
    position = 0
    while position < count:
        length = int(rng.uniform(0.08, 0.4) * rate)
        gate[slice(position, position + length)] = rng.uniform(0.4, 1.0)
        position += length + int(rng.uniform(0.05, 0.6) * rate)
    ramp = np.hanning(int(0.02 * rate))
    gate = np.convolve(gate, ramp / ramp.sum(), mode="same")
    phase = 2 * np.pi * np.cumsum(140.0 * (1.0 + 0.08 * np.sin(2 * np.pi * 0.7 * t))) / rate
    voice = sum(np.sin(k * phase + rng.uniform(0, 6)) / k for k in range(1, 20))
    return (0.1 * voice * gate + hiss * rng.standard_normal(count)).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _minute():
    return _speech(60.0)


@functools.lru_cache(maxsize=None)
def _long():
    return _speech(100.0)


@functools.lru_cache(maxsize=None)
def _paused():
    """96 s of speech in 8 s turns with 8 s pauses under a 1e-2 hiss only ~11 LU below it: the source, the hiss
    removed, and the pauses expanded 12 dB down."""
    clean = _speech(96.0).astype(np.float64)
    speaking = (np.arange(len(clean)) / RATE) % 16.0 < 8.0
    clean = clean * speaking
    source = clean + 1e-2 * np.random.default_rng(11).standard_normal(len(clean))
    gain = np.convolve(np.where(speaking, 1.0, 10.0 ** (-12.0 / 20.0)), np.ones(800) / 800, mode="same")
    return source.astype(np.float32), clean.astype(np.float32), (source * gain).astype(np.float32)


def _rider(source, depth_db=3.0, rate_hz=0.2):
    t = np.arange(len(source)) / RATE
    return (source * 10.0 ** (depth_db * np.sin(2 * np.pi * rate_hz * t) / 20.0)).astype(np.float32)


def _delayed(source, ms):
    shift = int(round(ms * RATE / 1000.0))
    if shift >= 0:
        return np.concatenate([np.zeros(shift, dtype=np.float32), source])[: len(source)]
    return np.concatenate([source[-shift:], np.zeros(-shift, dtype=np.float32)])


def _sine(seconds, rate, freq=997.0):
    return np.sin(2 * np.pi * freq * np.arange(int(seconds * rate)) / rate)


def _p95(source, output, lag_samples=0):
    return file_metrics.gain_ride(source, output, RATE, lag_samples)["p95_abs_lu"]


# Short-term loudness and the floor -------------------------------------------------------


def test_a_full_scale_sine_reads_minus_3_lufs_like_pyloudnorm():
    """A 0 dBFS 997 Hz sine is -3.01 LUFS in BS.1770, and pyloudnorm's own meter agrees."""
    sine = _sine(4.0, 44100)
    levels = file_metrics.short_term_loudness(sine, 44100)
    assert len(levels) == 2
    assert np.allclose(levels, -3.01, atol=0.05)
    assert abs(levels[0] - pyloudnorm.Meter(44100).integrated_loudness(sine)) < 0.05


def test_chunked_k_weighting_equals_one_pass(monkeypatch):
    """Carrying the filter state across chunks gives one pass's energies: 61 s splits at the default 60 s and at 7 s."""
    noise = np.random.default_rng(3).standard_normal(int(61.5 * RATE)) * 0.1
    default = file_metrics.hop_energies(noise, RATE, RATE)
    monkeypatch.setattr(file_metrics, "CHUNK_S", 7.0)
    small = file_metrics.hop_energies(noise, RATE, RATE)
    monkeypatch.setattr(file_metrics, "CHUNK_S", 1000.0)
    whole = file_metrics.hop_energies(noise, RATE, RATE)
    assert len(whole) == 61
    assert np.allclose(default, whole, rtol=1e-10)
    assert np.allclose(small, whole, rtol=1e-10)


def test_shorter_than_one_window_reads_no_levels():
    """Under 3 s there is no short-term window at all."""
    assert len(file_metrics.short_term_loudness(_sine(2.5, RATE), RATE)) == 0


def test_the_floor_is_the_level_between_words():
    """The pauses set the floor (the 10th percentile of 100 ms blocks), whatever the speech above them does."""
    source = _paused()[0]
    hiss = (1e-2 * np.random.default_rng(11).standard_normal(len(source))).astype(np.float32)
    floor = file_metrics.source_floor(source, RATE)
    assert abs(floor - file_metrics.source_floor(hiss, RATE)) < 0.5
    assert np.max(file_metrics.short_term_loudness(source, RATE)) - floor < 12.0


def test_the_floor_is_held_at_the_absolute_gate():
    """Digital silence, NaN blocks or no blocks at all hold the floor at -70 LUFS."""
    assert file_metrics.source_floor(np.zeros(5 * RATE, dtype=np.float32), RATE) == file_metrics.ABSOLUTE_GATE_LUFS
    assert file_metrics.noise_floor(np.full(5, np.nan), 1600) == file_metrics.ABSOLUTE_GATE_LUFS
    assert file_metrics.noise_floor(np.zeros(0), 1600) == file_metrics.ABSOLUTE_GATE_LUFS


def test_the_gate_keeps_programme_and_drops_pauses_and_silence():
    """Windows more than 10 LU under the programme, or under -70 LUFS, are not read."""
    levels = np.array([-20.0, -21.0, -19.0, -35.0, -90.0, -np.inf])
    active = file_metrics.speech_active(levels)
    assert active.tolist() == [True, True, True, False, False, False]
    assert not file_metrics.speech_active(np.full(4, -80.0)).any()


def test_the_gate_also_drops_windows_within_10_lu_of_the_floor():
    """A window inside the programme gate but under 10 LU over the noise floor is mostly pause: not read."""
    levels = np.array([-20.0, -24.0, -29.0])
    assert file_metrics.speech_active(levels).all()
    assert file_metrics.speech_active(levels, floor=-35.0).tolist() == [True, True, False]


# Gain ride -------------------------------------------------------------------------------


def test_identity_reads_no_ride():
    """The source against itself rides nowhere."""
    ride = file_metrics.gain_ride(_minute(), _minute(), RATE)
    assert ride["windows"] > 50
    assert ride["std_lu"] == 0.0
    assert ride["p95_abs_lu"] == 0.0
    assert ride["deriv_p10_p90"] == 0.0


def test_a_static_gain_is_the_offset_not_a_ride():
    """One gain for the whole file is the mastering stage's job: it is the offset, the ride stays 0."""
    ride = file_metrics.gain_ride(_minute(), 0.5 * _minute(), RATE)
    assert abs(ride["offset_lu"] + 6.02) < 0.01
    assert ride["p95_abs_lu"] < 1e-6
    assert ride["std_lu"] < 1e-6


def test_a_slow_gain_rider_reads_above_one_and_a_half_lu():
    """A +-3 dB rider at 0.2 Hz, a dynamic loudnorm's kind of move, reads above 1.5 LU p95."""
    ride = file_metrics.gain_ride(_minute(), _rider(_minute()), RATE)
    assert ride["p95_abs_lu"] > 1.5
    assert ride["std_lu"] > 0.8
    assert ride["deriv_p10_p90"] > 2.0


def test_hiss_removed_under_continuous_speech_is_not_a_ride():
    """With the floor far under the speech every window is read, and a denoiser's change to them is no ride."""
    ride = file_metrics.gain_ride(_speech(60.0, hiss=3e-3), _minute(), RATE)
    assert ride["windows"] > 50
    assert ride["p95_abs_lu"] < 0.2


def test_emptied_long_pauses_are_not_a_ride():
    """Hiss removed from 8 s pauses only ~11 LU under the speech, or the pauses expanded 12 dB down, rides nowhere."""
    source, clean, expanded = _paused()
    assert file_metrics.gain_ride(source, clean, RATE)["windows"] >= file_metrics.MIN_RIDE_WINDOWS
    assert _p95(source, clean) < 0.3
    assert _p95(source, expanded) < 0.3


def test_without_the_floor_gate_the_long_pauses_read_as_a_ride(monkeypatch):
    """The programme gate alone keeps every pause window here: the floor gate is what keeps them out."""
    monkeypatch.setattr(file_metrics, "FLOOR_MARGIN_LU", -1e9)
    source, clean, expanded = _paused()
    assert _p95(source, clean) > 10.0
    assert _p95(source, expanded) > 5.0


def test_a_late_output_rides_only_until_its_lag_is_taken_out():
    """A 300 ms late output reads as a ride on the raw pair; taking the lag out by timing alone leaves none, either way round."""
    shift = int(0.3 * RATE)
    assert _p95(_minute(), _delayed(_minute(), 300.0)) > 0.5
    assert _p95(_minute(), _delayed(_minute(), 300.0), shift) < 0.1
    assert _p95(_minute(), _delayed(_minute(), -300.0), -shift) < 0.1


def test_a_short_file_reads_no_ride():
    """A file under one window reads None with no windows."""
    ride = file_metrics.gain_ride(_speech(2.0), _speech(2.0), RATE)
    assert ride["windows"] == 0
    assert all(ride[name] is None for name in (*file_metrics.RIDE_NAMES, "offset_lu"))


def test_too_few_programme_windows_read_no_ride():
    """Fewer than five programme windows are too few to call a spread."""
    ride = file_metrics.gain_ride(_speech(6.0), _speech(6.0), RATE)
    assert 0 < ride["windows"] < file_metrics.MIN_RIDE_WINDOWS
    assert ride["p95_abs_lu"] is None


def test_silence_has_no_programme_windows():
    """Digital silence has no window above the absolute gate."""
    silence = np.zeros(10 * RATE, dtype=np.float32)
    assert file_metrics.gain_ride(silence, silence, RATE)["windows"] == 0


def test_steps_are_read_only_between_neighbouring_windows():
    """Steps 1, 2 and 0.5 spread 1.8 - 0.6 = 1.2 LU; the 6 LU step across a gated-out window would make it 4.15."""
    assert abs(file_metrics.step_spread(np.array([0.0, 1.0, 3.0, np.nan, 9.0, 9.5])) - 1.2) < 1e-12
    assert abs(file_metrics.step_spread(np.array([0.0, 1.0, 3.0, 9.0, 9.5])) - 4.15) < 1e-12
    assert file_metrics.step_spread(np.array([0.0, 1.0, np.nan, 5.0])) is None


# Card entries ----------------------------------------------------------------------------


def test_file_entries_carry_every_reading_on_the_output_side():
    """The card entries put each reading on the output side over a zero source, like the trade metrics."""
    entries = file_metrics.file_entries(_long(), _rider(_long()), RATE)
    assert set(entries) >= {"file.gain_ride_lu", "file.sync_drift_ms", "file.sync_offset_ms", "file.sync_unmatched"}
    assert entries["file.gain_ride_lu"]["output"] > 1.5
    assert entries["file.sync_unmatched"] == {"source": 0.0, "output": 0.0, "delta": 0.0}
    assert entries["file.sync_drift_ms"]["delta"] == entries["file.sync_drift_ms"]["output"] < 1.0


def test_file_entries_take_the_sync_lag_out_of_the_ride():
    """A 300 ms late output is an offset in the sync entries and no ride in the gain entries."""
    entries = file_metrics.file_entries(_long(), _delayed(_long(), 300.0), RATE)
    assert abs(entries["file.sync_offset_ms"]["output"] - 300.0) < 0.5
    assert entries["file.gain_ride_lu"]["output"] < 0.1


def test_file_entries_of_a_tiny_file_are_none():
    """A file too short for either reading leaves None in its entries and nothing unmatched."""
    entries = file_metrics.file_entries(_speech(2.0), _speech(2.0), RATE)
    assert entries["file.gain_ride_lu"]["output"] is None
    assert entries["file.sync_drift_ms"]["delta"] is None
    assert entries["file.sync_unmatched"]["output"] == 0.0
