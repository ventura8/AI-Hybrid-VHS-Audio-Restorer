"""The ear v3 degradation table: what each failure must move and must not, and that every entry builds its pair."""

import functools

import numpy as np
import pytest

from scripts import degradations_v3 as v3
from scripts import quality_degradations as deg
from scripts.restoration_quality import scorecard
from tests.unit.test_quality_degradations import RATE
from tests.unit.test_quality_degradations_listener import _speech

MOVING = ("up", "down", "match")


@functools.lru_cache(maxsize=None)
def _swelling_voice():
    """The listener tests' vowels and 's' bursts, their level swelling +-3 dB every 4 s so the loudnorm rider has a swing to ride."""
    speech = _speech()
    t = np.arange(len(speech)) / RATE
    return (speech * 10.0 ** (3.0 * np.sin(2 * np.pi * t / 4.0) / 20.0)).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _materials():
    voice = _swelling_voice()
    noise = (1e-3 * np.random.default_rng(5).standard_normal(len(voice))).astype(np.float32)
    return {"speech": voice, "clean": voice, "vhs": voice + noise, "noise": noise, "voice": voice, "music": np.roll(voice, 5000)}


def _expectations(direction=None):
    return [(name, e) for name, spec in deg.V3_DEGRADATIONS.items() for e in spec.expects if direction in (None, e.direction)]


def test_every_v3_degradation_sits_in_the_calibration_table():
    """Every v3 degradation sits in the calibration table."""
    assert set(deg.V3_DEGRADATIONS) <= set(deg.DEGRADATIONS)
    assert all(len(spec.levels) == 3 and spec.base in _materials() for spec in deg.V3_DEGRADATIONS.values())


@pytest.mark.parametrize("name", sorted(deg.V3_DEGRADATIONS))
def test_each_v3_degradation_declares_what_must_move_and_what_must_not(name):
    """An asserted move (or one pre-registered on a reading not built yet, R9) and an asserted must-not-move."""
    expects = deg.V3_DEGRADATIONS[name].expects
    assert any(e.direction in MOVING and (not e.blind or e.metric in deg.UNBUILT_READINGS) for e in expects), name
    assert any(e.direction == "flat" and not e.blind for e in expects), name


def test_every_v3_expectation_names_a_registered_or_planned_reading():
    """Every v3 expectation names a registered or planned reading."""
    named = {e.metric for _name, e in _expectations()} | {e.twin for _name, e in _expectations() if e.twin}
    assert named <= set(scorecard.METRICS) | set(deg.UNBUILT_READINGS)


def test_flat_and_match_expectations_carry_their_allowance():
    """Flat and match expectations carry their allowance."""
    assert all(e.tolerance is not None or e.twin for _name, e in _expectations("flat"))
    assert all(e.tolerance and e.tolerance > 0 for _name, e in _expectations("match"))


def test_the_net_sibilance_reading_is_held_against_its_absolute_twin_on_the_shelf_and_the_tilt():
    """The net sibilance reading is held against its absolute twin on the shelf and the tilt."""
    twins = {name: e.twin for name, e in _expectations("flat") if e.metric == deg.SIB_NET}
    assert set(twins) == {"air_shelf_boost", "air_shelf_cut", "air_corner", "spectral_tilt_up", "spectral_tilt_down"}
    assert set(twins.values()) == {deg.SIB_ABS}


def test_the_cut_entries_mirror_the_boosts():
    """The cut entries mirror the boosts."""
    assert deg.DEGRADATIONS["air_shelf_cut"].levels == tuple(-level for level in deg.AIR_LEVELS)
    assert deg.DEGRADATIONS["spectral_tilt_down"].levels == tuple(-level for level in deg.TILT_LEVELS)


@pytest.mark.parametrize("name", sorted(deg.V3_DEGRADATIONS))
def test_each_v3_degradation_builds_a_pair_of_equal_length(name):
    """Each v3 degradation builds a pair of equal length."""
    spec = deg.DEGRADATIONS[name]
    materials = _materials()
    source, output = deg.apply(name, spec.levels[-1], materials[spec.base], RATE, materials, np.random.default_rng(3))
    assert len(source) == len(output) > 0
    assert not np.array_equal(source, output)


@pytest.mark.parametrize("family", sorted(deg.V3_BUILDERS))
def test_each_family_builds_its_own_degradations(family):
    """Each family builds its own degradations."""
    names = [name for name in deg.V3_DEGRADATIONS if name.startswith(family)]
    assert names
    assert all(deg.v3_builder(name) is deg.V3_BUILDERS[family] for name in names)


def test_the_pause_residual_pairs_leave_the_speech_bit_identical():
    """The pause residual pairs leave the speech bit identical."""
    materials = _materials()
    source, output = deg.apply("pause_residual_hiss", 6.0, materials["vhs"], RATE, materials, np.random.default_rng(3))
    speech = deg.speech_mask(source, RATE)
    assert np.array_equal(output[speech], source[speech])
    assert np.any(output[~speech] != source[~speech])


def test_the_linear_bandwidth_pair_limits_the_capture_and_the_oracle_alike():
    """The source is the hissy capture, the output the oracle with the hiss 20 dB down, both under the same low pass."""
    materials = _materials()
    source, output = deg.apply("linear_bandwidth", 5000.0, materials["vhs"], RATE, materials, np.random.default_rng(3))
    removed = 0.9 * v3.band_limited(materials["noise"], RATE, 5000.0).astype(np.float64)
    assert np.allclose(source.astype(np.float64) - output, removed, atol=1e-6)


def test_a_real_excerpt_carries_the_degradations_that_need_no_reference():
    """The tape set keeps the v3 speech and capture degradations and drops what needs the fixtures' clean reference."""
    on_tape = set(deg.on_tape())
    assert {"pause_residual_hiss", "air_shelf_boost", "hifi_compander_mistrack", "musical_noise"} <= on_tape
    assert not on_tape & {"hiss", "hiss_in_pauses", "air_shelf_boost_music", "background_stripped"}
    assert "linear_bandwidth" in on_tape


def test_a_real_excerpt_keeps_the_speech_identity_and_drops_the_music_one():
    """The benign tape set keeps the identity pair and drops the music identity, which starts from the fixtures' music bed."""
    benign = deg.on_tape_benign()
    assert "identity_music" not in benign
    assert "identity" in benign


def test_only_the_shifts_change_readings_by_construction():
    """A shift moves the raw-pair sync lags and offset (not the drift); every other benign case moves nothing."""
    assert set(deg.benign_changes("shift_30ms")) == {
        "file.sync_offset_ms",
        "file.sync_lag_start_ms",
        "file.sync_lag_middle_ms",
        "file.sync_lag_end_ms",
    }
    assert set(deg.SHIFT_READINGS) <= set(scorecard.METRICS)
    assert all(not deg.benign_changes(name) for name in deg.BENIGN if not name.startswith(deg.SHIFT_PREFIX))


def test_on_a_real_excerpt_the_cut_capture_is_its_own_output():
    """No clean reference: the linear-bandwidth pair is the cut capture twice, and only R0's reading of it counts."""
    excerpt = {"vhs": _materials()["vhs"], "speech": _materials()["vhs"]}
    source, output = deg.apply("linear_bandwidth", 5000.0, excerpt["vhs"], RATE, excerpt, np.random.default_rng(3))
    assert np.array_equal(source, output)


def _asserted_on(metric):
    """The v3 degradations that assert a move of `metric` (not blind)."""
    return {name for name, e in _expectations() if e.metric == metric and not e.blind and e.direction in MOVING}


def test_both_air_shelf_entries_assert_r1_air():
    """Finding 12, option 1: R1's air is non-blind on speech and music; where R0's band leaves it no band it reports "clipped"."""
    shelves = {"air_shelf_boost", "air_shelf_cut", "air_shelf_boost_music", "air_shelf_cut_music", "air_corner"}
    assert _asserted_on("dsp.balance_air_db") == shelves
    assert {"air_shelf_boost", "air_shelf_cut", "air_corner"} <= set(deg.on_tape())


def test_the_corner_row_sweeps_round_a1s_corners_at_its_top_gain():
    """`linear_air_freq_hz` is a knob: the shelf at +1.5 dB with its corner at 9000 / 7500 / 6000 Hz, mild to severe."""
    spec, materials = deg.DEGRADATIONS["air_corner"], _materials()
    assert (spec.levels, deg.AIR_CORNER_GAIN_DB) == ((9000.0, 7500.0, 6000.0), 1.5)
    built = [deg.apply("air_corner", corner, materials["speech"], RATE, materials, None)[1] for corner in spec.levels]
    assert all(np.array_equal(out, v3.air_shelf(materials["speech"], RATE, 1.5, corner)) for out, corner in zip(built, spec.levels))


def test_the_corner_row_asks_the_readings_round_a1_is_judged_by_to_follow_the_corner():
    """R2's absolute level and both HF bands must rise as the corner drops; the net level stays under its twin."""
    moving = {e.metric for e in deg.DEGRADATIONS["air_corner"].expects if e.direction == "up" and not e.blind}
    assert moving == {deg.SIB_ABS, "dsp.hf_4k8k", "dsp.hf_8k16k", "dsp.balance_air_db"}
