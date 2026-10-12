"""scanner_mains_vote: the scan's mains notch from the head's single-bin vote ('head') or that vote confirmed ('confirmed').

'head' (the default) reads the first 32768 samples, each candidate's bin against the bins two either side,
as every release has. The readings pinned below were recorded on these fixtures with HEAD e429683's
modules/filters.py: the default key must return them exactly, without reading the whole signal, so every
mode's pre-conditioning graph, cathar's dehum family and auto_pure_linear's notches keep their bytes.
'confirmed' keeps the head's notch only where the whole analysed signal's eight harmonics of its family stand
6 dB over their neighbourhood (spectral_denoise.harmonic_excess_db): a tone in the head alone is dropped.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import modules.config as cfg
from modules import auto_scanner, filters, spectral_denoise
from scripts import tune_restoration as tune

RATE = 44100
KEY = "scanner_mains_vote"
VOTE_CONSTANT = "modules.filters.SCANNER_MAINS_VOTE"
HEAD = "head"
CONFIRMED = "confirmed"
PAL_LINE_HZ = 15625.0
PAL = {"highpass_hz": 80, "crt_notch_hz": PAL_LINE_HZ}
PAL_TAIL = ",bandreject=f=15625.0:width_type=q:w=30"
LOG = "modules.mains_vote.log_msg"
READING = "harmonic_excess_db"
SERIES_50 = "series_50"
SERIES_60 = "series_60"
SERIES_50_PAL = "series_50_pal"
SERIES_60_PAL = "series_60_pal"
BURST = "head_burst"
SECOND = "second_harmonic"
MODE = {"process_mode": "cathar"}


def _series(mains_hz, seed=1):
    """Ten seconds of an eight-harmonic mains series (1/k amplitudes from 0.02) in white noise at 0.01."""
    rng = np.random.default_rng(seed)
    t = np.arange(10 * RATE) / RATE
    hum = sum((0.02 / k) * np.sin(2 * np.pi * mains_hz * k * t + k) for k in range(1, 9))
    return (hum + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def _head_burst():
    """Ten seconds of white noise at 0.1 with a 0.01 tone at 50 Hz in its first 32768 samples alone."""
    rng = np.random.default_rng(2)
    t = np.arange(10 * RATE) / RATE
    signal = 0.1 * rng.standard_normal(len(t))
    signal[:32768] += 0.01 * np.sin(2 * np.pi * 50.0 * t[:32768])
    return signal.astype(np.float32)


def _second_harmonic():
    """Ten seconds of a 100 Hz tone at 0.03 over white noise at 0.005: the head votes for the harmonic."""
    rng = np.random.default_rng(3)
    t = np.arange(10 * RATE) / RATE
    return (0.005 * rng.standard_normal(len(t)) + 0.03 * np.sin(2 * np.pi * 100.0 * t)).astype(np.float32)


FIXTURES = {
    SERIES_50: lambda: (_series(50.0), 0.0),
    SERIES_60: lambda: (_series(60.0), 0.0),
    SERIES_50_PAL: lambda: (_series(50.0), PAL_LINE_HZ),
    SERIES_60_PAL: lambda: (_series(60.0), PAL_LINE_HZ),
    BURST: lambda: (_head_burst(), 0.0),
    SECOND: lambda: (_second_harmonic(), 0.0),
}
# HEAD e429683: the head vote on each fixture.
HEAD_READINGS = {
    SERIES_50: 50.0,
    SERIES_60: 120.0,
    SERIES_50_PAL: 50.0,
    SERIES_60_PAL: 0.0,
    BURST: 50.0,
    SECOND: 100.0,
}
# The confirmed vote on each fixture: the head burst reads 0.85 dB at 50 Hz over the whole signal and loses its notch.
CONFIRMED_READINGS = {**HEAD_READINGS, BURST: 0.0}


def _vote(name, vote):
    """`_detect_mains_buzz_notch` on fixture `name` with `scanner_mains_vote` set to `vote`, the scan's log silenced."""
    signal, line_rate_hz = FIXTURES[name]()
    with patch(VOTE_CONSTANT, vote), patch(LOG):
        return filters._detect_mains_buzz_notch(signal, RATE, line_rate_hz)


def test_the_key_defaults_to_head():
    """The resolved config holds every release's vote."""
    assert cfg.CONFIG[KEY] == HEAD


def test_the_module_constant_defaults_to_head():
    """config's constant holds every release's vote."""
    assert cfg.SCANNER_MAINS_VOTE == HEAD


def test_the_scans_copy_defaults_to_head():
    """The scan's copy of the constant holds every release's vote."""
    assert filters.SCANNER_MAINS_VOTE == HEAD


def _loaded(user):
    """The configuration load_config resolves from a config.yaml holding `user`, and the warning printer."""
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = user
    with (
        patch("modules.config.Path.exists", return_value=True),
        patch.object(cfg, "yaml", mock_yaml),
        patch("builtins.open", MagicMock()),
        patch("builtins.print") as warn,
    ):
        conf, _source = cfg.load_config()
    return conf, warn


@pytest.mark.parametrize(
    "user, expected",
    [({}, HEAD), ({KEY: " Confirmed "}, CONFIRMED), ({KEY: "whole"}, HEAD), ({KEY: 1}, HEAD)],
)
def test_the_key_is_a_choice(user, expected):
    """An older config.yaml gets 'head'; case and spaces are forgiven; an unknown value or a non-string keeps 'head'."""
    conf, _warn = _loaded({**MODE, **user})
    assert conf[KEY] == expected


@pytest.mark.parametrize("user, warned", [({}, False), ({KEY: CONFIRMED}, False), ({KEY: "whole"}, True)])
def test_only_an_unknown_value_warns(user, warned):
    """A missing or known value is quiet; an unknown one is named in a warning."""
    _conf, warn = _loaded({**MODE, **user})
    assert warn.called is warned


def test_the_repository_config_ships_head():
    """config.yaml ships 'head'."""
    text = (Path(cfg.__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
    assert f'\n{KEY}: "head"\n' in text


def test_the_tuning_driver_knows_the_key():
    """A tuning grid may set the key to either value."""
    assert tune.known_config_fields()[KEY][0] == {HEAD, CONFIRMED}


def test_the_tuning_driver_refuses_an_unknown_value():
    """An unknown value is a problem in a grid."""
    fields = tune.known_config_fields()
    assert tune._problem(KEY, "whole", fields[KEY])


@pytest.mark.parametrize("name", sorted(HEAD_READINGS))
def test_the_default_vote_is_heads(name):
    """The default key votes on each fixture as HEAD e429683 did."""
    assert _vote(name, HEAD) == pytest.approx(HEAD_READINGS[name], abs=1e-12)


@pytest.mark.parametrize("name", sorted(HEAD_READINGS))
def test_the_default_vote_never_reads_the_whole_signal(name):
    """'head' never calls the whole-signal reading, so it costs the scan nothing."""
    with patch.object(spectral_denoise, READING) as reading:
        _vote(name, HEAD)
    assert reading.called is False


def test_the_default_keeps_a_head_false_fire_in_the_graph():
    """On the head burst the default scan still notches 50 and 100 Hz in the pre-conditioning, as every release has."""
    graph = filters._precondition_filter_from_config({**PAL, "notch_hz": _vote(BURST, HEAD)})
    assert graph == "highpass=f=80,adeclick,bandreject=f=50.0:width_type=q:w=15,bandreject=f=100.0:width_type=q:w=15" + PAL_TAIL


@pytest.mark.parametrize("name", sorted(CONFIRMED_READINGS))
def test_the_confirmed_vote(name):
    """'confirmed' keeps every notch the whole signal carries, the head's own value included, and drops the burst's."""
    assert _vote(name, CONFIRMED) == pytest.approx(CONFIRMED_READINGS[name], abs=1e-12)


def test_the_confirmed_vote_leaves_the_graph_without_mains_notches():
    """With the burst's notch dropped the pre-conditioning keeps only its high pass, declick and CRT notch."""
    graph = filters._precondition_filter_from_config({**PAL, "notch_hz": _vote(BURST, CONFIRMED)})
    assert graph == "highpass=f=80,adeclick" + PAL_TAIL


def test_a_dropped_notch_is_logged():
    """The scan names the head's frequency, the excess it read and the gate."""
    signal, line_rate_hz = FIXTURES[BURST]()
    with patch(VOTE_CONSTANT, CONFIRMED), patch(LOG) as log:
        filters._detect_mains_buzz_notch(signal, RATE, line_rate_hz)
    assert "the head's 50 Hz not confirmed (0.85 dB of harmonic excess at 50 Hz, under 6)" in log.call_args.args[0]


def test_a_harmonic_is_confirmed_at_its_family():
    """The head's 100 Hz is read at 50 Hz, the family it belongs to."""
    signal, line_rate_hz = FIXTURES[SECOND]()
    with patch(VOTE_CONSTANT, CONFIRMED), patch.object(spectral_denoise, READING, return_value=9.0) as reading:
        filters._detect_mains_buzz_notch(signal, RATE, line_rate_hz)
    assert reading.call_args.args[2] == pytest.approx(50.0)


@pytest.mark.parametrize("excess_db, expected", [(6.0, 100.0), (5.99, 0.0)])
def test_the_gate_is_inclusive_at_six_db(excess_db, expected):
    """An excess at the gate keeps the head's notch; just under it drops the notch."""
    signal, line_rate_hz = FIXTURES[SECOND]()
    with (
        patch(VOTE_CONSTANT, CONFIRMED),
        patch.object(spectral_denoise, READING, return_value=excess_db),
        patch(LOG),
    ):
        notch = filters._detect_mains_buzz_notch(signal, RATE, line_rate_hz)
    assert notch == pytest.approx(expected)


def test_no_head_notch_reads_nothing_more():
    """Where the head found no hum, 'confirmed' adds none and reads nothing."""
    with patch.object(spectral_denoise, READING) as reading:
        _vote(SERIES_60_PAL, CONFIRMED)
    assert reading.called is False


def test_a_signal_too_short_for_the_reading_keeps_no_notch():
    """Under the reading's 16384-sample frame the excess is 0 dB, so a head notch on 8192-16383 samples is dropped."""
    t = np.arange(12000) / RATE
    signal = (0.5 * np.sin(2 * np.pi * 50.0 * t)).astype(np.float32)
    with patch(VOTE_CONSTANT, CONFIRMED), patch(LOG):
        notch = filters._detect_mains_buzz_notch(signal, RATE)
    assert notch == pytest.approx(0.0)


def test_the_scanner_profile_carries_the_confirmed_vote():
    """The strategy's notch_hz, which every mode's pre-conditioning and cathar's dehum read, is the confirmed one."""
    signal, _line = FIXTURES[BURST]()
    with patch(VOTE_CONSTANT, CONFIRMED), patch(LOG):
        profile = auto_scanner._extract_profile_from_signal(signal, RATE)
    assert profile["notch_hz"] == pytest.approx(0.0)


def test_the_whole_signal_reading_is_detect_mains_hzs():
    """The confirmation reads what auto_pure_linear's detector votes on: the public reading it now calls."""
    signal, _line = FIXTURES[SERIES_50]()
    with patch.object(spectral_denoise, "_scannable_mono", return_value=(signal, RATE)):
        detected = spectral_denoise.detect_mains_hz("unused.wav")
    assert detected == pytest.approx(50.0)
