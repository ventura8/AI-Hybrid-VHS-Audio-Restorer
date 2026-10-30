"""The azimuth scan's polarity check: an inverted pair is flagged, and refused only behind its switch.

A pair with one channel's polarity inverted passes the |r| gate, and the cross-correlation's
maximum then sits half a period of the programme's strongest partial away from zero: a 1 kHz
tone reads -0.50 ms, which the pre-conditioning would apply as an azimuth delay.
"""

from unittest.mock import patch

import numpy as np
import pytest

from modules import filters

RATE = 44100


def _tone(freq_hz=1000.0, seconds=0.3):
    """A steady tone, the programme whose inverted copy fakes the clearest skew."""
    return 0.3 * np.sin(2 * np.pi * freq_hz * np.arange(int(RATE * seconds)) / RATE)


def _pair(left, right):
    """Two channels as a stereo array."""
    return np.stack([left, right], axis=1)


def test_the_signed_correlation_keeps_the_sign_the_gate_drops():
    """+1 in phase, -1 inverted; the gate's |r| reads both as 1."""
    tone = _tone()
    assert filters.signed_channel_correlation(tone, tone) == pytest.approx(1.0)
    assert filters.signed_channel_correlation(tone, -tone) == pytest.approx(-1.0)
    assert filters._channel_correlation(tone, -tone) == pytest.approx(1.0)
    assert filters.signed_channel_correlation(tone, np.zeros_like(tone)) == 0.0


def test_an_inverted_pair_still_reads_todays_skew_by_default_and_is_flagged():
    """Off by default: the same value as before the check, so every output keeps its bytes, and a line in the log."""
    with patch("modules.filters.AZIMUTH_REJECT_INVERTED_PAIR", False), patch("modules.filters.log_msg") as log:
        skew = filters._detect_stereo_azimuth_skew(_pair(_tone(), -_tone()), RATE)
    assert skew == pytest.approx(-0.5)
    assert "polarity-inverted (r=-1.00)" in log.call_args[0][0]
    assert "unreliable" in log.call_args[0][0]


def test_the_switch_refuses_a_skew_from_an_inverted_pair():
    """On, the inverted 2 kHz pair reads no skew instead of -0.25 ms."""
    with patch("modules.filters.AZIMUTH_REJECT_INVERTED_PAIR", True), patch("modules.filters.log_msg") as log:
        skew = filters._detect_stereo_azimuth_skew(_pair(_tone(2000.0), -_tone(2000.0)), RATE)
    assert skew == 0.0
    assert "no azimuth delay read from it" in log.call_args[0][0]


@pytest.mark.parametrize("refuse", [False, True])
def test_an_in_phase_pair_reads_its_skew_whatever_the_switch(refuse):
    """A real 5-sample skew between in-phase channels is read the same with the switch on or off, and nothing is flagged."""
    tone = _tone(440.0)
    pair = _pair(tone, np.roll(tone, 5))
    with patch("modules.filters.AZIMUTH_REJECT_INVERTED_PAIR", refuse), patch("modules.filters.log_msg") as log:
        skew = filters._detect_stereo_azimuth_skew(pair, RATE)
    assert skew == pytest.approx(round(-5 / RATE * 1000.0, 2))
    log.assert_not_called()


def test_unrelated_channels_are_not_read_or_flagged():
    """Independent noise fails the gate before the sign is looked at."""
    rng = np.random.default_rng(3)
    with patch("modules.filters.log_msg") as log:
        assert filters._azimuth_pair_usable(rng.standard_normal(8192), rng.standard_normal(8192)) is False
    log.assert_not_called()


@pytest.mark.parametrize("refuse", [False, True])
def test_a_nan_correlation_passes_the_gate_as_before_and_is_not_called_inverted(refuse):
    """A channel holding NaN samples reads r = NaN, which no comparison catches: the gate lets it through as it always did."""
    tone = _tone()
    broken = tone.copy()
    broken[100] = np.nan
    with patch("modules.filters.AZIMUTH_REJECT_INVERTED_PAIR", refuse), patch("modules.filters.log_msg") as log:
        assert filters._azimuth_pair_usable(tone, broken) is True
    log.assert_not_called()


@pytest.mark.parametrize(
    "correlation, refuse, usable",
    [(-0.3, False, True), (-0.3, True, False), (-0.29, True, False), (0.29, True, False), (0.3, True, True)],
)
def test_the_inverted_branch_starts_where_the_gate_does(correlation, refuse, usable):
    """r = -0.3 is the first inverted value the |r| gate passes, read by default and refused behind the switch; under 0.3 is noise."""
    with (
        patch("modules.filters.signed_channel_correlation", return_value=correlation),
        patch("modules.filters.AZIMUTH_REJECT_INVERTED_PAIR", refuse),
        patch("modules.filters.log_msg"),
    ):
        assert filters._azimuth_pair_usable(None, None) is usable
