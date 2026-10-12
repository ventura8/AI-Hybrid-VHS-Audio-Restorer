"""The hiss-floor reading: bands, programme frames, the floor under the programme, the level match and the stationarity."""

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

from modules import hiss_floor

RATE = 44100


def _bursts(seconds, seed=1):
    """A speech-like programme: harmonic bursts of 0.3 s with 0.2 s gaps, its partials reaching 6 kHz."""
    length = int(seconds * RATE)
    t = np.arange(length) / RATE
    tone = sum(np.sin(2 * np.pi * 150.0 * k * t + k) / k for k in range(1, 40))
    gate = (t % 0.5) < 0.3
    amplitude = np.random.default_rng(seed).uniform(0.6, 1.0, size=int(seconds / 0.5) + 1)[(t // 0.5).astype(int)]
    return 0.2 * tone / np.max(np.abs(tone)) * gate * amplitude


def _hiss(seconds, seed=2, level=3e-3):
    return np.random.default_rng(seed).normal(0.0, level, int(seconds * RATE))


def _write(path, mono, channels=2, rate=RATE):
    sf.write(str(path), np.stack([mono] * channels, axis=1).astype(np.float32), rate, subtype="FLOAT")
    return path


def _pair(tmp_path, residual, seconds=6.0, level=1.0):
    """A source (programme plus hiss) and a restored file keeping `residual` of the hiss, at `level` of the programme."""
    programme, hiss = _bursts(seconds), _hiss(seconds)
    source = _write(tmp_path / "source.wav", programme + hiss)
    restored = _write(tmp_path / "restored.wav", level * (programme + residual * hiss))
    return source, restored


def test_the_bands_are_the_diagnosis_erb_bands():
    edges = hiss_floor.erb_band_edges(3000.0, 16000.0)
    expected = [3000, 3367, 3776, 4231, 4738, 5302, 5931, 6631, 7410, 8278, 9245, 10322, 11521, 12857, 14344, 16000]
    assert np.round(edges).astype(int).tolist() == expected


def test_a_restored_file_keeping_a_tenth_of_the_hiss_reads_twenty_db_under_the_source(tmp_path):
    reading = hiss_floor.read(*_pair(tmp_path, residual=0.1))
    residual = (reading["median_delta_db"], reading["presence_delta_db"])
    assert residual == pytest.approx((-20.0, -20.0), abs=1.5)
    match = (reading["floor_gap_db"] < 4.0, abs(reading["programme_gain_db"]) < 0.5, reading["programme_correlation"] > 0.9)
    assert match == (True, True, True)
    assert (0.0 < reading["programme_share"] < 1.0, len(reading["band_delta_db"]), len(reading["band_edges_hz"])) == (True, 15, 16)


def test_a_level_drop_across_the_stage_is_matched_out(tmp_path):
    reading = hiss_floor.read(*_pair(tmp_path, residual=0.5, level=0.3))
    assert reading["programme_gain_db"] == pytest.approx(20 * np.log10(1 / 0.3), abs=0.3)
    assert reading["median_delta_db"] == pytest.approx(20 * np.log10(0.5), abs=1.5)


def test_an_unrelated_restored_file_gives_no_reading(tmp_path):
    source = _write(tmp_path / "source.wav", _bursts(4.0) + _hiss(4.0))
    restored = _write(tmp_path / "restored.wav", _hiss(4.0, seed=9, level=0.05))
    assert "does not follow" in hiss_floor.read(source, restored)["reason"]


def test_a_file_under_ten_frames_and_a_rate_mismatch_give_no_reading(tmp_path):
    short = _write(tmp_path / "short.wav", _hiss(0.1))
    assert hiss_floor.read(short, short) == {"reason": "too short to read"}
    other = _write(tmp_path / "other.wav", _hiss(1.0), rate=48000)
    assert "rate differs" in hiss_floor.read(short, other)["reason"]


def test_a_programme_too_sparse_for_the_floor_gives_no_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(hiss_floor, "programme_frames", lambda vad, rate, frame: np.zeros(len(vad), dtype=bool))
    assert hiss_floor.read(*_pair(tmp_path, residual=0.5))["reason"].startswith("too little programme")


def test_a_band_limited_source_has_no_stationarity_reading_and_programme_moves_it(tmp_path):
    sos = scipy.signal.butter(10, 3000.0, btype="low", fs=RATE, output="sos")
    dull = scipy.signal.sosfiltfilt(sos, _bursts(4.0) + _hiss(4.0))
    reading = hiss_floor.read(_write(tmp_path / "dull.wav", dull), _write(tmp_path / "dull_out.wav", dull))
    assert reading["floor_gap_db"] is None
    bright = _bursts(4.0) + _hiss(4.0, level=1e-4)
    sos_hp = scipy.signal.butter(4, 8000.0, btype="high", fs=RATE, output="sos")
    music = bright + 4.0 * scipy.signal.sosfilt(sos_hp, _bursts(4.0, seed=5) * _hiss(4.0, seed=6, level=1.0))
    moving = hiss_floor.read(_write(tmp_path / "music.wav", music), _write(tmp_path / "music_out.wav", music))
    assert moving["floor_gap_db"] > 3.0


def test_band_deltas_without_a_readable_band_read_nothing():
    layout = hiss_floor._layout(RATE, 882)
    count = layout["hiss"].shape[1]
    deltas = hiss_floor._band_deltas(np.full(2 * count, 1e-30), {"layout": layout})
    assert deltas["median_delta_db"] is None
    assert deltas["presence_delta_db"] is None
    assert deltas["band_delta_db"] == [None] * count


def test_a_silent_output_matches_nothing():
    stats = {"level": np.ones(20), "cross": np.zeros(20), "out_power": np.zeros(20), "src_power": np.ones(20)}
    assert hiss_floor.programme_match(stats) == (0.0, 0.0)


def test_the_programme_frames_follow_a_burst_and_its_hangover():
    vad = np.full((100, 4), 1e-6)
    vad[40:50, 1] = 1.0
    speech = hiss_floor.programme_frames(vad, RATE, 882)
    assert speech[40:50].all()
    assert not speech[:30].any()
    assert speech[36:40].any()
    assert speech[50:54].any()


def test_the_floor_of_a_long_programme_is_read_on_a_spread_of_frames(monkeypatch):
    monkeypatch.setattr(hiss_floor, "MAX_FLOOR_POINTS", 50)
    bands = np.random.default_rng(3).exponential(1.0, size=(5000, 2))
    floor = hiss_floor.programme_floor(bands, np.ones(5000, dtype=bool))
    assert floor.shape == (2,)
    assert np.all(floor < np.median(bands, axis=0))


def test_isolated_programme_frames_leave_no_window_to_read():
    speech = np.zeros(400, dtype=bool)
    speech[::50] = True
    assert hiss_floor.programme_floor(np.ones((400, 2)), speech) is None
