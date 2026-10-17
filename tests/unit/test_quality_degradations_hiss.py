"""Ear v4's hiss degradations: the linear capture, the true-pause weight, the regions each builder keeps, the benign set."""

from functools import lru_cache

import numpy as np
import pytest

from scripts import degradations_hiss as hd
from scripts import quality_degradations as deg
from tests.unit.test_restoration_quality_hiss_floor import RATE, voice

LINEAR_KEYS = ("prog_linear", "noise_linear", "vhs_linear")


@lru_cache(maxsize=None)
def _noise():
    return 3e-4 * np.random.default_rng(12).standard_normal(int(9 * RATE))


@lru_cache(maxsize=None)
def _materials():
    """The linear capture of 12 s of the synthetic voice over a 9 s noise track (resized through the gaps)."""
    return dict(zip(LINEAR_KEYS, hd.linear_capture(voice(12.0), _noise(), RATE)))


def _weight():
    return hd.pause_weight_true(_materials()["vhs_linear"], RATE)


def test_the_linear_capture_holds_two_gaps_and_continuous_noise():
    """The programme is silent for 1.2 s at a third and two thirds; the noise runs on, the capture is their sum."""
    programme, noise, capture = (_materials()[key] for key in LINEAR_KEYS)
    third, gap = int(12.0 * RATE / 3.0), int(hd.GAP_S * RATE)
    assert len(programme) == len(noise) == int(12.0 * RATE) + 2 * gap
    assert not programme[third:][:gap].any()
    assert np.array_equal(capture, programme + noise)


def test_the_tape_hiss_sits_at_the_recorded_noises_level_and_leans_to_the_highs():
    """The capture's hiss matches the fixture noise's 3-10 kHz level and stands higher above 4 kHz than under 1 kHz."""
    hiss = _materials()["noise_linear"]
    assert hd.band_level(hiss, RATE) == pytest.approx(hd.band_level(_noise(), RATE), rel=1e-6)
    assert hd.band_level(hiss, RATE, (4000.0, 10000.0)) > 2.0 * hd.band_level(hiss, RATE, (200.0, 1000.0))
    assert not np.array_equal(hd.tape_hiss(_noise(), RATE, RATE, seed=1), hd.tape_hiss(_noise(), RATE, RATE, seed=2))


def test_the_true_pause_weight_is_1_on_the_mask_and_ramps_on_the_active_side():
    """Inside the mask the weight is exactly 1; it never moves faster than a 5 ms ramp, so it falls to 0 within 5 ms."""
    weight = _weight()
    assert np.max(np.abs(np.diff(weight))) <= 1.0 / (hd.EDGE_RAMP_S * RATE) + 1e-9
    assert weight.max() == 1.0
    assert hd.pause_weight_true(np.zeros(RATE), RATE).max() == 0.0


def test_hiss_under_programme_leaves_the_pauses_bit_identical():
    """Where the weight is 1 the output is the capture itself; under the programme the noise is L dB down."""
    source, output = deg.apply("hiss_under_programme", 12.0, None, RATE, _materials(), None)
    weight = _weight()
    assert np.array_equal(output[weight == 1.0], source[weight == 1.0])
    programme, noise = _materials()["prog_linear"], _materials()["noise_linear"]
    assert output[weight == 0.0] == pytest.approx((programme + noise * 10.0 ** (-12.0 / 20.0))[weight == 0.0])


def test_hiss_in_pauses_only_cleans_the_programme_time_to_20_db():
    """Where the weight is 0 the noise is 20 dB down whatever the level; in the pauses it is L dB down."""
    programme, noise = _materials()["prog_linear"], _materials()["noise_linear"]
    _source, mild = deg.apply("hiss_in_pauses_only", 6.0, None, RATE, _materials(), None)
    _source, dead = deg.apply("dead_pauses", 60.0, None, RATE, _materials(), None)
    weight = _weight()
    assert np.array_equal(mild[weight == 0.0], dead[weight == 0.0])
    assert dead[weight == 1.0] == pytest.approx((programme + noise * 1e-3)[weight == 1.0])


def test_the_room_floor_sits_at_its_level_under_1_khz():
    """The room's 100-1000 Hz power is the source's true-pause power there x 10^(L/10)."""
    source = _materials()["vhs_linear"]
    room = hd.room_floor(source, RATE, -6.0)
    paused = _weight() >= 1.0
    target = hd.band_power(source, RATE, hd.ROOM_BAND_HZ)[paused].mean() * 10.0 ** (-0.6)
    assert hd.band_power(room, RATE, hd.ROOM_BAND_HZ).mean() == pytest.approx(target, rel=1e-6)


@pytest.mark.parametrize("name", ["hiss_hf_tilt", "low_floor_kept", "hiss_islands", "hiss_removed_dulled"])
def test_every_hiss_builder_returns_the_capture_as_its_source(name):
    """The source of every case is the hissy capture; the output keeps its length."""
    source, output = deg.apply(name, deg.DEGRADATIONS[name].levels[0], None, RATE, _materials(), None)
    assert source is _materials()["vhs_linear"]
    assert len(output) == len(source)


def test_the_hiss_degradations_need_the_fixtures_noise():
    """They run on fixtures only and never on a tape cut."""
    assert set(deg.HISS_DEGRADATIONS) <= set(deg.NEEDS_REFERENCE)
    assert not set(deg.HISS_DEGRADATIONS) & set(deg.on_tape())


def test_the_linear_benign_cases_start_from_the_capture_and_floor_only_the_hiss_readings():
    """`*_linear` cases read the linear capture and give the hiss readings their floor; the others keep theirs."""
    assert [deg.benign_base(name) for name in ("fresh_noise_linear", "identity_music", "requantise")] == ["vhs_linear", "music", "speech"]
    assert (deg.benign_only("delay_0_37_linear"), deg.benign_only("requantise")) == (("dsp.hiss_",), None)
    assert not set(deg.LINEAR_BENIGN) & set(deg.on_tape_benign())


def test_the_fresh_noise_case_keeps_the_programme_and_redraws_the_noise():
    """`fresh_noise_linear`: the programme plus a fresh realisation of its noise, no shared draw."""
    capture = _materials()["vhs_linear"]
    _source, fresh = deg.apply_benign("fresh_noise_linear", capture, RATE, None, _materials())
    residual = fresh - _materials()["prog_linear"]
    assert abs(np.corrcoef(residual, _materials()["noise_linear"])[0, 1]) < 0.1
    level = 10.0 * np.log10(hd.band_level(residual, RATE) / hd.band_level(_materials()["noise_linear"], RATE))
    assert abs(level) < 0.2


@pytest.mark.parametrize("name", ["identity_linear", "requantise_linear", "resample_linear", "gain_0_1_linear", "delay_0_37_linear"])
def test_every_same_percept_case_stays_near_the_capture(name):
    """The other linear cases change the capture by under -30 dB (0.1 dB of gain is -38.8 dB, the 0.37-sample delay -36)."""
    capture = _materials()["vhs_linear"]
    _source, output = deg.apply_benign(name, capture, RATE, np.random.default_rng(3), _materials())
    error = np.mean((np.asarray(output[: len(capture)], dtype=np.float64)[2000:-2000] - capture[2000:-2000]) ** 2)
    assert 10.0 * np.log10(error / np.mean(capture**2) + 1e-30) < -30.0
