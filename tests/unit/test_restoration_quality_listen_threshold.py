"""The absolute-threshold run: tones under the hard cap, QUEST convergence for a simulated observer, the map and the level."""

import numpy as np
import pytest

from scripts.restoration_quality import auditory, listen_modes, listen_protocol, listen_threshold, playback_calibration

LEVEL = -23.0
INTERVALS = listen_threshold.INTERVALS


def _wrong(key):
    """An interval other than `key`."""
    return INTERVALS[(INTERVALS.index(key) + 1) % len(INTERVALS)]


def _observe(mode, true_75, rng):
    """Answers every trial as a listener whose 75%-correct point is `true_75[hz]` dBFS (slope, guess and lapse as modelled)."""
    while not mode.finished():
        trial = mode.trial(len(mode.answers))
        threshold = true_75[trial["hz"]] - listen_threshold.OFFSET_75_DB
        right = rng.random() < float(listen_threshold.p_right(trial["x"], threshold))
        mode.record(trial, trial["key"] if right else _wrong(trial["key"]))
    return mode


def _shifted_truth(frequencies, shift_db):
    """Thresholds `shift_db` above where the check's 65 dB SPL assumption puts them."""
    return {hz: listen_threshold.prior_threshold_dbfs(hz, LEVEL) + shift_db for hz in frequencies}


def _entry(hz, offset, censored):
    return {"hz": hz, "threshold_dbfs": -80.0, "spl_offset_db": offset, "censored": censored}


# ----------------------------------------------------------------------------- the tones and the cap


def test_the_cap_is_minus_33_dbfs_or_10_db_under_the_speech():
    """-33 dBFS at the default speech level; 10 dB under a quieter one."""
    assert listen_threshold.cap_dbfs(-23.0) == -33.0
    assert listen_threshold.cap_dbfs(-30.0) == -40.0


def _tone_at_minus_60():
    """A 1 kHz tone asked for at -60 dBFS under a -33 dBFS cap."""
    return listen_threshold.tone_clip(1000.0, -60.0, -33.0)


def test_a_tone_is_one_second_of_float32_on_both_channels():
    """The tone lasts one second, in float32, and plays the same on both channels."""
    clip = _tone_at_minus_60()
    assert clip.shape == (listen_threshold.RATE, 2)
    assert clip.dtype == np.float32
    assert np.array_equal(clip[:, 0], clip[:, 1])


def test_a_tone_plays_at_its_rms_level_in_the_middle_of_its_second():
    """A -60 dBFS tone reads -60 dBFS RMS over its flat part."""
    rate = listen_threshold.RATE
    middle = _tone_at_minus_60()[slice(int(0.45 * rate), int(0.55 * rate)), 0]
    assert 20.0 * np.log10(np.sqrt(np.mean(middle.astype(np.float64) ** 2))) == pytest.approx(-60.0, abs=0.05)


def test_a_tone_starts_and_ends_silent():
    """The first 0.3 s and the last 0.3 s of the tone's second are silent."""
    clip = _tone_at_minus_60()
    rate = listen_threshold.RATE
    assert not clip[slice(0, int(0.3 * rate))].any()
    assert not clip[slice(int(0.7 * rate), None)].any()


def test_no_tone_passes_the_cap_whatever_is_asked():
    """Asked for 0 dBFS, the tone stays at the cap: its peak is the cap's RMS plus 3 dB, no more."""
    clip = listen_threshold.tone_clip(4000.0, 0.0, -33.0)
    assert np.max(np.abs(clip)) <= np.sqrt(2.0) * 10.0 ** (-33.0 / 20.0) + 1e-7


def _guessing_block(seed):
    """A listener who hears nothing, answering at random, through the protocol with its practice and catch trials."""
    rng = np.random.default_rng(seed)
    mode = listen_threshold.ThresholdMode(60, rng, LEVEL)
    protocol = listen_protocol.Protocol(mode, rng, familiar=5, catches=8, catch_labels=(listen_threshold.CATCH_TONE,))
    while not protocol.finished():
        trial = protocol.trial()
        protocol.take(trial, str(rng.choice(INTERVALS)))
    return mode, protocol


def test_a_listener_who_hears_nothing_is_never_played_above_the_cap():
    """The posterior climbs, the tones do not: every scored level stays at or under the cap."""
    mode, _protocol = _guessing_block(1)
    assert max(answer["x"] for answer in mode.answers) <= mode.cap


def test_a_listener_who_hears_nothing_misses_the_catch_tones():
    """The catch tones play at the cap, which a guessing listener misses two times in three: the block reads disengaged."""
    _mode, protocol = _guessing_block(2)
    context = protocol.context()
    assert context["catch"]["missed"] >= 3
    assert context["block_flags"] == [listen_protocol.DISENGAGED]


def test_tones_out_of_reach_everywhere_place_no_level():
    """Every threshold far above the cap: most frequencies censored, and none of the censored ones in the map."""
    mode = listen_threshold.ThresholdMode(60, np.random.default_rng(3), LEVEL)
    body = _observe(mode, dict.fromkeys(listen_threshold.FREQUENCIES_HZ, -5.0), np.random.default_rng(4)).results()[0]
    censored = [entry["censored"] for entry in body["thresholds"]]
    assert sum(censored) >= 4
    assert all(body["sl_map"][f"{entry['hz']:g}"] is None for entry in body["thresholds"] if entry["censored"])


def test_no_level_is_placed_when_every_frequency_is_censored():
    """Every entry censored: no full-scale SPL, no loud-frame level, no residuals, and the summary says so."""
    entries = [_entry(1000.0, 90.0, True), _entry(2000.0, 91.0, True)]
    assert listen_threshold.placed_level(entries, LEVEL) == (None, None)
    assert listen_threshold.run_context(entries, None)["residuals_db"] == {}
    assert listen_threshold.sl_map(entries) == {"1000": None, "2000": None}
    assert "above the cap" in listen_threshold.threshold_text(entries[0])


def test_the_level_is_the_median_of_the_uncensored_offsets():
    """Offsets 88, 90 and 95 with one censored at 40: the full-scale SPL is 90, the loud frames 90 - 23 + 1.5."""
    entries = [_entry(1000.0, 88.0, False), _entry(2000.0, 90.0, False), _entry(4000.0, 95.0, False), _entry(8000.0, 40.0, True)]
    full_scale, loud = listen_threshold.placed_level(entries, LEVEL)
    assert (full_scale, loud) == (90.0, 90.0 + LEVEL + listen_threshold.LOUD_OVER_ACTIVE_DB)
    assert listen_threshold.run_context(entries, full_scale)["residuals_db"] == {"1000": -2.0, "2000": 0.0, "4000": 5.0}


def test_a_trial_plays_the_tone_in_its_key_and_silence_in_the_others():
    """Three intervals: the tone in the one the key names, silence in the other two; A, B, C and "can't tell" to choose from."""
    mode = listen_threshold.ThresholdMode(5, np.random.default_rng(2), LEVEL)
    trial = mode.tone_trial(0, -50.0, "B")
    sounding = {name: bool(clip.any()) for name, clip in trial["play"]}
    assert (sounding, trial["choices"]) == ({"A": False, "B": True, "C": False}, ["A", "B", "C", listen_modes.CANT_TELL])
    assert (trial["hz"], trial["x"], trial["stimuli"], bool(listen_threshold.silent_clip().any())) == (1000.0, -50.0, ["1000Hz"], False)


def test_the_keys_cover_all_three_intervals():
    """The scored trials put the tone in each of the three intervals; the protocol draws practice and catch keys from them too."""
    mode = listen_threshold.ThresholdMode(30, np.random.default_rng(5), LEVEL)
    assert set(mode.slots) == set(INTERVALS)
    assert mode.keys == INTERVALS
    protocol = listen_protocol.Protocol(mode, np.random.default_rng(6), familiar=5)
    keys = {protocol.trial()["key"] for _index in range(30)}
    assert keys == set(INTERVALS)


def test_practice_and_catch_trials_play_at_the_cap():
    """Practice cycles through the frequencies at the cap; a catch plays the cap where the threshold sits lowest so far."""
    mode = listen_threshold.ThresholdMode(10, np.random.default_rng(3), LEVEL)
    practice = [mode.familiar_trial(index, "A") for index in range(5)]
    assert ([trial["hz"] for trial in practice], {trial["x"] for trial in practice}) == (list(listen_threshold.FREQUENCIES_HZ), {mode.cap})
    catch = mode.catch_trial(listen_threshold.CATCH_TONE, "B")
    assert (catch["x"], catch["hz"]) == (mode.cap, 4000.0)


def test_the_frequencies_interleave_in_fresh_rounds():
    """Each round of five trials holds every frequency once."""
    order = listen_threshold.interleaved(5, 12, np.random.default_rng(4))
    assert len(order) == 12
    assert sorted(order[:5]) == sorted(order[5:10]) == [0, 1, 2, 3, 4]


def test_the_prior_is_the_checks_own_assumption():
    """Speech at -23 dBFS playing at 65 dB SPL puts the 1 kHz threshold at Terhardt's 1 kHz value minus 86.5 dB."""
    expected = float(auditory.threshold_in_quiet_db(1000.0)) - (65.0 + 23.0 - 1.5)
    assert listen_threshold.prior_threshold_dbfs(1000.0, -23.0) == pytest.approx(expected)


def test_the_psychometric_function_reads_75_percent_at_the_reported_point():
    """The 75% point lies OFFSET_75_DB from the Weibull parameter; far below it the listener guesses one in three."""
    assert float(listen_threshold.p_right(-60.0 + listen_threshold.OFFSET_75_DB, -60.0)) == pytest.approx(0.75)
    assert float(listen_threshold.p_right(-100.0, -60.0)) == pytest.approx(1.0 / 3.0, abs=1e-3)


# ----------------------------------------------------------------------------- convergence for a simulated observer


@pytest.mark.parametrize("shift_db", [-15.0, 0.0, 15.0])
def test_quest_finds_a_simulated_observer(shift_db):
    """12 trials at 1 kHz, 25 simulated listeners: the median error stays under 2.5 dB, and most 90% bands hold the truth."""
    errors, inside = [], 0
    for seed in range(25):
        rng = np.random.default_rng(100 + seed)
        mode = listen_threshold.ThresholdMode(12, np.random.default_rng(seed), LEVEL, frequencies=(1000.0,))
        truth = _shifted_truth((1000.0,), shift_db)[1000.0]
        entry = _observe(mode, {1000.0: truth}, rng).threshold(0)
        errors.append(entry["threshold_dbfs"] - truth)
        inside += entry["band_dbfs"][0] <= truth <= entry["band_dbfs"][1]
    assert np.median(np.abs(errors)) < 2.5
    assert inside >= 18


def test_a_full_run_places_the_playback_level_of_a_simulated_listener():
    """Thresholds 10 dB above the assumption mean the speech plays 10 dB under 65 dB SPL: the run reads 55 within 3 dB."""
    readings = []
    for seed in range(5):
        mode = listen_threshold.ThresholdMode(60, np.random.default_rng(seed), LEVEL)
        _observe(mode, _shifted_truth(listen_threshold.FREQUENCIES_HZ, 10.0), np.random.default_rng(50 + seed))
        readings.append(mode.results()[0]["loud_frames_spl"])
    assert np.median(readings) == pytest.approx(55.0, abs=3.0)


def test_a_frequency_not_heard_under_the_cap_is_censored_and_left_out():
    """12.5 kHz out of reach: censored, null in the map, out of the level, which the other four still place."""
    truth = _shifted_truth(listen_threshold.FREQUENCIES_HZ, 0.0)
    truth[12500.0] = -5.0
    mode = listen_threshold.ThresholdMode(60, np.random.default_rng(7), LEVEL)
    body = _observe(mode, truth, np.random.default_rng(8)).results()[0]
    censored = [entry["hz"] for entry in body["thresholds"] if entry["censored"]]
    mapped = (body["sl_map"]["12500"], body["sl_map"]["1000"] is None, "12500" in body["context"]["residuals_db"])
    assert (censored, mapped) == ([12500.0], (None, False, False))
    assert (body["loud_frames_spl"], "12.5 kHz above the cap" in mode.summary()) == (pytest.approx(65.0, abs=4.0), True)


def test_censoring_reads_the_answers_at_the_cap():
    """Under the cap and heard: not censored; 2 of 4 right at the cap: censored; above the cap: censored; 2 trials are too few."""
    heard = [[-40.0, True], [-33.0, True], [-33.0, True], [-33.0, True]]
    guessed = [[-33.0, True], [-33.0, False], [-33.0, True], [-33.0, False]]
    assert not listen_threshold.censored(-45.0, heard, -33.0)
    assert listen_threshold.censored(-45.0, guessed, -33.0)
    assert listen_threshold.censored(-30.0, [], -33.0)
    assert not listen_threshold.censored(-45.0, guessed[:2], -33.0)


def test_cant_tell_is_kept_apart_from_the_posterior():
    """A "can't tell" leaves the frequency's posterior as it was and is listed apart from the responses."""
    mode = listen_threshold.ThresholdMode(5, np.random.default_rng(9), LEVEL, frequencies=(2000.0,))
    trial = mode.trial(0)
    before = mode.log_posterior[0].copy()
    mode.record(trial, listen_modes.CANT_TELL)
    assert np.array_equal(mode.log_posterior[0], before)
    entry = mode.threshold(0)
    assert (entry["cant_tell"], entry["responses"]) == ([trial["x"]], [])


def test_the_run_body_makes_a_valid_calibration_record():
    """The body plus the fields the tool adds passes the calibration schema; the map and the level agree with the thresholds."""
    mode = listen_threshold.ThresholdMode(60, np.random.default_rng(10), LEVEL)
    _observe(mode, _shifted_truth(listen_threshold.FREQUENCIES_HZ, 0.0), np.random.default_rng(11))
    body = mode.results()[0]
    record = {"id": "r1", "date": "2026-10-12", "round": "session1", "listener": "user", "playback": {"device": "hp", "volume": "40"}}
    record.update(body)
    assert (playback_calibration.validate(record), body["sl_map"]["1000"]) == ([], body["thresholds"][0]["threshold_dbfs"])
    full_scale, loud = listen_threshold.placed_level(body["thresholds"], LEVEL)
    assert (body["spl_full_scale_db"], body["loud_frames_spl"]) == (full_scale, loud)
    assert "dB SPL (+-7-10 dB)" in mode.summary()
