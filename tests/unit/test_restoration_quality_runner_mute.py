"""R1's mute guard in the runner: a window whose source holds an analog mute reads no balance, decided on the source."""

import types

import numpy as np
import pytest

from scripts.restoration_quality import audio_io, balance_metrics, runner
from scripts.restoration_quality.scorecard import WindowRow
from tests.unit.test_restoration_quality_runner_v3 import PROFILE, _score_short, _speech_with_pauses
from tests.unit.test_restoration_quality_sibilance import RATE

HISS = 1e-2
MUTE_DBFS = -90.0


def _mute(seconds, seed=5, dbfs=MUTE_DBFS):
    """A capture floor 50 dB under the hiss (at the default `dbfs`): what a VCR leaves on blank tape."""
    return (10.0 ** (dbfs / 20.0) * np.random.default_rng(seed).standard_normal(int(seconds * RATE))).astype(np.float32)


def _muted(mono, start_s, seconds, dbfs=MUTE_DBFS):
    """`mono` with `seconds` from `start_s` replaced by the mute at `dbfs`."""
    muted = np.array(mono, dtype=np.float32)
    first = int(start_s * RATE)
    muted[slice(first, first + int(seconds * RATE))] = _mute(seconds, dbfs=dbfs)
    return muted


def _restored(seconds=15.0):
    """`(source, output)`: speech over a -40 dBFS hiss, and the hiss taken 20 dB down (an oracle denoiser)."""
    voice, noise = _speech_with_pauses(seconds, hiss=HISS)
    return voice + noise, voice + 0.1 * noise


def _pair(source, routes=None):
    """The fields `mute_floor_db` reads: the source, its rate, its 15 s windows and their routes."""
    windows = audio_io.windows(len(source), RATE)
    return types.SimpleNamespace(source=source, rate=RATE, windows=windows, routes=routes or ["speech"] * len(windows))


def test_a_window_holding_a_mute_reads_no_balance_where_r1_would_read_hiss_removal_as_dulling():
    """2 s of mute in 15 s: unguarded, the oracle reads about -19 dB of air; guarded, every R1 reading is None."""
    source, output = _restored()
    floor = runner.quiet_level_db(source, RATE)
    muted = _muted(source, 13.0, 2.0), _muted(output, 13.0, 2.0)
    assert runner.top_db(*(runner.balance_window(*muted, RATE, None)[name] for name in ("balance_presence_db", "balance_air_db"))) < -10.0
    assert set(runner.balance_window(*muted, RATE, None, floor).values()) == {None}
    readings = runner.balance_window(source, output, RATE, None, floor)
    assert set(readings) == set(balance_metrics.READINGS)
    assert abs(readings["balance_tilt_db_oct"]) < 0.3


def test_the_skipped_window_keeps_its_other_readings_and_a_zero_source_side():
    """Only R1 and its presence-or-air reading go None; the source side stays the identity."""
    source, output = _restored()
    pair = types.SimpleNamespace(source=_muted(source, 13.0, 2.0), output=_muted(output, 13.0, 2.0), rate=RATE, profile={})
    row = WindowRow(0, 0.0, 15.0, "speech")
    runner._dsp_window(pair, row, runner.quiet_level_db(source, RATE))
    assert row.output["dsp.balance_top_db"] is None and row.source["dsp.balance_top_db"] == 0.0
    assert all(row.output[f"dsp.{name}"] is None for name in balance_metrics.READINGS)
    assert row.output["dsp.residual_noise_db"] is not None


def test_the_floor_is_the_median_window_p10_so_a_mute_moves_only_the_windows_it_fills():
    """52.5 s (seven windows) with 2 s of mute at 48.5 s: the two windows over it read it, the floor stays at the hiss."""
    source = _muted(_restored(52.5)[0], 48.5, 2.0)
    pair = _pair(source)
    floor = runner.mute_floor_db(pair)
    assert len(pair.windows) == runner.MUTE_MEDIAN_WINDOWS and floor == runner.median_floor_db(pair, pair.windows)
    assert abs(floor - runner.quiet_level_db(source[: 15 * RATE], RATE)) < 1.0
    assert [runner.holds_mute(source[w.slice_of(RATE)], RATE, floor) for w in pair.windows] == [False] * 5 + [True] * 2


def _blank_then_speech():
    """60 s of blank tape before 52.5 s of speech, and the routes that call the seven blank windows silence."""
    source = np.concatenate([_mute(60.0), _restored(52.5)[0]])
    return source, ["silence" if w.end_s <= 60.0 else "speech" for w in audio_io.windows(len(source), RATE)]


def test_windows_routed_as_silence_stay_out_of_the_floor_unless_every_window_is_one():
    """Counted, the eight blank windows would make the mute the median; with nothing but silence, every window counts."""
    source, routes = _blank_then_speech()
    hiss = runner.quiet_level_db(source[slice(60 * RATE, None)], RATE)
    assert abs(runner.mute_floor_db(_pair(source, routes)) - hiss) < 1.0
    assert runner.mute_floor_db(_pair(source)) < hiss - 30.0
    assert runner.mute_floor_db(_pair(source, ["silence"] * len(routes))) == runner.mute_floor_db(_pair(source))


def _r1_over_windows(source, output, routes=None):
    """R1 on each window of a pair, guarded by the floor `mute_floor_db` reads on that pair's own source."""
    pair = _pair(source, routes)
    floor = runner.mute_floor_db(pair)
    return [runner.balance_window(source[w.slice_of(RATE)], output[w.slice_of(RATE)], RATE, None, floor) for w in pair.windows]


def _muted_pair(seconds, start_s, length_s=2.0, dbfs=MUTE_DBFS):
    """The oracle restoration of `seconds` of speech with a `length_s` mute at `dbfs` from `start_s` on both sides."""
    source, output = _restored(seconds)
    return _muted(source, start_s, length_s, dbfs), _muted(output, start_s, length_s, dbfs)


def _tilts(readings):
    """Per window: None where R1 read nothing (the guard skipped it), else the tilt R1 read."""
    return [None if set(read.values()) == {None} else read["balance_tilt_db_oct"] for read in readings]


def _skips_only(readings, skipped):
    """Whether R1 read nothing on the windows `skipped` names and read the oracle as no change on every other one."""
    tilts = _tilts(readings)
    return all((tilt is None) if index in skipped else (tilt is not None and abs(tilt) < 0.3) for index, tilt in enumerate(tilts))


def test_a_15_s_pair_reads_its_floor_past_a_mute_that_fills_both_of_its_windows():
    """Both windows hold the last 7.5 s: a median of their p10s was the mute, and R1 read air -19 dB in both."""
    source, output = _muted_pair(15.0, 13.0)
    pair = _pair(source)
    assert len(pair.windows) == 2 and runner.mute_floor_db(pair) == runner.span_floor_db(pair, pair.windows)
    assert runner.median_floor_db(pair, pair.windows) < runner.mute_floor_db(pair) - 30.0
    assert abs(runner.mute_floor_db(pair) - runner.quiet_level_db(_restored()[0], RATE)) < 1.0
    assert [set(readings.values()) for readings in _r1_over_windows(source, output)] == [{None}, {None}]


@pytest.mark.parametrize(("seconds", "dbfs"), [(40.0, -90.0), (40.0, -65.0), (45.0, -65.0)])
def test_a_4_s_mute_in_three_of_five_or_six_windows_still_skips_them(seconds, dbfs):
    """13-17 s sits in windows 0-2 (air -19 dB unguarded): it was the median of five, half of six's; the p25 is the hiss."""
    source, output = _muted_pair(seconds, 13.0, 4.0, dbfs)
    pair = _pair(source)
    assert len(pair.windows) < runner.MUTE_MEDIAN_WINDOWS
    assert runner.median_floor_db(pair, pair.windows) < runner.mute_floor_db(pair) - 10.0
    assert abs(runner.mute_floor_db(pair) - runner.quiet_level_db(_restored()[0], RATE)) < 1.0
    assert _skips_only(_r1_over_windows(source, output), {0, 1, 2})


def _blank_lead_in(blank_s, speech_s, dbfs=MUTE_DBFS):
    """`blank_s` of blank tape before the oracle restoration of `speech_s` of speech, and the routes the runner gives it."""
    source, output = _restored(speech_s)
    lead = _mute(blank_s, dbfs=dbfs)
    source, output = np.concatenate([lead, source]), np.concatenate([lead, output])
    return source, output, [audio_io.route_window(source[w.slice_of(RATE)], RATE) for w in audio_io.windows(len(source), RATE)]


@pytest.mark.parametrize(
    ("blank_s", "speech_s", "skipped", "dbfs"),
    [
        (4.0, 11.0, {0}, MUTE_DBFS),
        (8.0, 15.0, {0}, MUTE_DBFS),
        (10.0, 20.0, {0, 1}, MUTE_DBFS),
        (4.0, 11.0, {0}, -65.0),
        (10.0, 20.0, {0, 1}, -65.0),
    ],
)
def test_a_blank_lead_in_skips_the_windows_it_fills_though_it_owns_the_span_p25(blank_s, speech_s, skipped, dbfs):
    """The blank part of the opening window counts in the span: from a quarter of it the p25 is the blank (R1 air -19 dB).

    At -65 dBFS over half of an even window count the mean of the two middle p10s sat half-way (-52.6 dBFS) and missed it;
    the higher middle p10 keeps the floor on the hiss.
    """
    source, output, routes = _blank_lead_in(blank_s, speech_s, dbfs)
    pair = _pair(source, routes)
    assert runner.span_floor_db(pair, runner.programme_windows(pair)) < runner.mute_floor_db(pair) - 20.0
    assert _skips_only(_r1_over_windows(source, output, routes), skipped)


def test_a_long_blank_before_a_short_programme_skips_the_window_half_over_it():
    """60 s blank + 15 s speech: seven windows route as silence; of the three programme windows, 52.5 s is half blank."""
    source, output, routes = _blank_lead_in(60.0, 15.0)
    assert routes.count("silence") == 7
    pair = _pair(source, routes)
    assert [w.start_s for w in runner.programme_windows(pair)] == [52.5, 60.0, 67.5]
    tilts = _tilts(_r1_over_windows(source, output, routes))
    assert tilts[7] is None and all(abs(tilt) < 0.3 for tilt in tilts[8:])


def test_a_20_s_pair_skips_the_two_windows_over_its_mute_and_reads_the_third():
    """17-19 s fills 16 % of 7.5-20 s and 40 % of 15-20 s; 0-15 s holds none of it and reads the oracle as no change."""
    source, output = _muted_pair(20.0, 17.0)
    first, *muted = _r1_over_windows(source, output)
    assert all(set(readings.values()) == {None} for readings in muted) and len(muted) == 2
    assert abs(first["balance_tilt_db_oct"]) < 0.3


def test_a_short_pair_without_a_mute_skips_nothing():
    """The p25 sits on the programme's own pauses: the hiss the oracle removed is no mute."""
    source, output = _restored(15.0)
    assert all(abs(readings["balance_tilt_db_oct"]) < 0.3 for readings in _r1_over_windows(source, output))


def test_digital_silence_is_no_mute():
    """Capture padding sits under R1's 90 dB live range: it is not counted, and R1's own trim handles it."""
    source, _output = _restored()
    padded = np.array(source)
    padded[: 8 * RATE] = 0.0
    assert not runner.holds_mute(padded, RATE, runner.quiet_level_db(source, RATE))


def test_no_floor_or_no_frame_skips_nothing():
    """Without a floor every window reads R1; a window shorter than one 20 ms frame has no level and no floor."""
    source, _output = _restored()
    assert not runner.holds_mute(_muted(source, 13.0, 2.0), RATE, None)
    assert runner.quiet_level_db(source[:100], RATE) is None
    assert not runner.holds_mute(source[:100], RATE, runner.quiet_level_db(source, RATE))
    assert runner.mute_floor_db(types.SimpleNamespace(source=source[:100], rate=RATE, windows=[], routes=[])) is None


def test_the_dsp_family_hands_r1_the_sources_floor(tmp_path, monkeypatch):
    """One floor per pair, read on the source before the windows, reaches the guard on every window."""
    floors = []
    monkeypatch.setattr(runner, "mute_floor_db", lambda _pair: -123.0)
    monkeypatch.setattr(runner, "holds_mute", lambda _mono, _rate, floor: floors.append(floor) or False)
    _score_short(tmp_path, monkeypatch, PROFILE)
    assert floors == [-123.0]


def test_the_upper_median_takes_the_higher_middle_p10_of_an_even_count():
    """Below seven windows the floor reads the higher of the two middle p10s; from seven up the plain median."""
    source, _output, routes = _blank_lead_in(10.0, 20.0, -65.0)
    pair = _pair(source, routes)
    windows = runner.programme_windows(pair)
    assert len(windows) % 2 == 0
    assert runner.median_floor_db(pair, windows, upper=True) > runner.median_floor_db(pair, windows) + 5.0
