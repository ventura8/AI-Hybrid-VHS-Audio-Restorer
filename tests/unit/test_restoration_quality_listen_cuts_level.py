"""Listening cuts at one speech level: the target, the ceiling, the pause loop, the catch controls and the practice pair."""

import numpy as np
import pytest

from scripts.restoration_quality import listen_cuts
from tests.unit.test_restoration_quality_listen_cuts import RATE, _quieter_delayed_copy, _voice, _wav


def test_every_cut_lands_on_the_target_speech_level(tmp_path):
    """Both cuts read the target level on the reference's speech-active frames, whatever level they came in at."""
    cuts, info = _quieter_delayed_copy(tmp_path)
    levels = listen_cuts.active_levels_db(cuts, "ref")
    assert info["target_dbfs"] == listen_cuts.TARGET_ACTIVE_DBFS
    assert levels["ref"] == pytest.approx(listen_cuts.TARGET_ACTIVE_DBFS, abs=0.05)
    assert levels["out"] == pytest.approx(listen_cuts.TARGET_ACTIVE_DBFS, abs=0.05)


def test_a_cut_that_would_clip_at_the_target_is_unlevelled(tmp_path):
    """A quiet voice with a click 30 dB over it: at -23 dBFS the click passes the ceiling; the block names where it would fit."""
    source, _gate = _voice()
    clicked = 0.1 * source
    clicked[5 * RATE] = 0.5
    paths = {"ref": _wav(tmp_path / "ref.wav", source), "clicky": _wav(tmp_path / "clicky.wav", clicked)}
    _cuts, info = listen_cuts.prepare_cuts(paths, 2.0, 8.0, tmp_path / "work")
    assert info["unlevelled"] == ["clicky"]
    assert info["peaks_dbfs"]["clicky"] > listen_cuts.PEAK_MAX_DBFS
    assert info["fits_at_dbfs"] == pytest.approx(-23.0 + listen_cuts.PEAK_MAX_DBFS - info["peaks_dbfs"]["clicky"])


def test_the_peak_reads_in_dbfs():
    """Half scale is -6.02 dBFS; silence reads the floor."""
    assert listen_cuts.peak_dbfs(np.full((4, 2), -0.5, dtype=np.float32)) == pytest.approx(-6.02, abs=0.01)
    assert listen_cuts.peak_dbfs(np.zeros((4, 2), dtype=np.float32)) == pytest.approx(-240.0)
    assert listen_cuts.unlevelled_labels({"a": -0.1, "b": -0.2, "c": -3.0}) == ["a"]


def test_the_pause_loop_sits_in_the_quietest_stretch():
    """A loud cut with 2.5 s of near-silence from 3 s: the 2 s loop starts inside that gap."""
    loud = (0.2 * np.random.default_rng(4).standard_normal((8 * RATE, 2))).astype(np.float32)
    loud[slice(3 * RATE, int(5.5 * RATE))] *= 0.001
    start, end = listen_cuts.quietest_span(loud)
    assert 3.0 <= start <= 3.5
    assert end - start == pytest.approx(listen_cuts.LOOP_SECONDS)


def test_a_cut_shorter_than_the_loop_loops_whole():
    """Under 2 s of audio the loop is the whole cut."""
    short = np.ones((RATE // 2, 2), dtype=np.float32) * 0.1
    assert listen_cuts.quietest_span(short) == [0.0, pytest.approx(0.5)]


def _tone_level_db(audio, freq_hz):
    """The level of one frequency in the second half of channel 0 (past the filter's start-up)."""
    half = slice(len(audio) // 2, None)
    t = np.arange(len(audio))[half] / RATE
    return 20.0 * np.log10(abs(np.mean(audio[half, 0] * np.exp(-2j * np.pi * freq_hz * t))))


def test_the_high_shelf_lifts_the_air_and_leaves_the_voice():
    """+6 dB at 7.5 kHz: a 15 kHz tone gains about 6 dB, a 200 Hz tone nothing."""
    t = np.arange(2 * RATE) / RATE
    for freq, gain in ((15000.0, 6.0), (200.0, 0.0)):
        tone = np.repeat((0.1 * np.sin(2 * np.pi * freq * t))[:, None], 2, axis=1).astype(np.float32)
        lifted = listen_cuts.high_shelf(tone, 6.0, 7500.0)
        assert _tone_level_db(lifted, freq) - _tone_level_db(tone, freq) == pytest.approx(gain, abs=0.3)


def test_the_air_catch_differs_in_timbre_not_in_speech_level():
    """The air catch reads the reference's speech level on its frames, yet differs from it; the raw catch is half of each."""
    source, _gate = _voice(4.0)
    cut = np.repeat(source[:, None], 2, axis=1)
    raw = (0.05 * np.random.default_rng(9).standard_normal(cut.shape)).astype(np.float32)
    stimuli, gains = listen_cuts.catch_cuts({"ref": cut, "raw": raw}, "ref", "raw")
    levels = listen_cuts.active_levels_db({"ref": cut, "air": stimuli[listen_cuts.CATCH_AIR]}, "ref")
    assert levels["air"] == pytest.approx(levels["ref"], abs=0.05)
    differs = float(np.max(np.abs(stimuli[listen_cuts.CATCH_AIR] - cut))) > 0.01
    half = bool(np.allclose(stimuli[listen_cuts.CATCH_RAW], 0.5 * cut + 0.5 * raw))
    assert (differs, half, gains[listen_cuts.CATCH_AIR] < 0.0, gains[listen_cuts.CATCH_RAW]) == (True, True, True, 0.0)


def test_the_air_catch_is_held_under_the_ceiling_and_needs_no_raw_tape():
    """A reference already at the ceiling: the shelved catch is scaled back under it; without a raw tape there is no raw catch."""
    loud = np.repeat(np.sign(np.sin(np.arange(RATE) / 3.0))[:, None], 2, axis=1).astype(np.float32) * 0.97
    stimuli, gains = listen_cuts.catch_cuts({"ref": loud}, "ref")
    assert set(stimuli) == {listen_cuts.CATCH_AIR}
    assert np.max(np.abs(stimuli[listen_cuts.CATCH_AIR])) <= listen_cuts.PEAK_MAX + 1e-6
    assert set(gains) == {listen_cuts.CATCH_AIR}


def test_the_familiarisation_pair_exaggerates_the_difference_under_the_ceiling():
    """The exaggerated side is the first plus twice the difference; one common gain keeps both under the ceiling."""
    first = np.full((100, 2), 0.2, dtype=np.float32)
    second = np.full((100, 2), 0.7, dtype=np.float32)
    quiet = listen_cuts.familiar_cuts(first, second * 0.5, 2.0)
    assert np.allclose(quiet[listen_cuts.FAMILIAR_B], 0.2 + 2.0 * (0.35 - 0.2))
    loud = listen_cuts.familiar_cuts(first, second, 2.0)
    assert np.max(np.abs(loud[listen_cuts.FAMILIAR_B])) == pytest.approx(listen_cuts.PEAK_MAX)
    assert loud[listen_cuts.FAMILIAR_A][0, 0] == pytest.approx(0.2 * listen_cuts.PEAK_MAX / 1.2)
