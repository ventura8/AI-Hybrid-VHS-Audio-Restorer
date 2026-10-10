"""R11 in the runner: read on every window against the source's own mains, paired against a zero source, skipped over a mute."""

import types

import pytest

from scripts.restoration_quality import audio_io, balance_metrics, lf_metrics, runner, scorecard, source_profile
from scripts.restoration_quality.scorecard import WindowRow
from tests.unit.test_restoration_quality_runner_v3 import PROFILE, _recorder, _registry, _speech_with_pauses, _write_pair
from tests.unit.test_restoration_quality_sibilance import RATE


def _scored(tmp_path, monkeypatch, profile, route="music"):
    """`score_pair` (dsp) on 6 s with R0 replaced by `profile`, every window routed as `route`; returns the card and the mains seen."""
    calls = []
    monkeypatch.setattr(source_profile, "capture_profile", lambda _audio, _rate: profile)
    monkeypatch.setattr(audio_io, "route_window", lambda _mono, _rate: route)
    monkeypatch.setattr(lf_metrics, "lf_programme_db", _recorder(calls, lf_metrics.lf_programme_db, 2))
    source, output = _write_pair(tmp_path, seconds=6.0)
    card, _pair = runner.score_pair(source, output, families=("dsp",), registry=_registry(), cache_dir=tmp_path / "cache")
    return card, calls


@pytest.mark.parametrize("route", ["music", "mixed", "speech"])
def test_r11_lands_on_every_programme_route_paired_against_a_zero_source(tmp_path, monkeypatch, route):
    """The output side carries R11, the source side the identity; the aggregate a grid ranks has it on every programme route."""
    card, _calls = _scored(tmp_path, monkeypatch, PROFILE, route)
    row = card.rows[0]
    assert row.source[runner.LF_READING] == 0.0
    assert isinstance(row.output[runner.LF_READING], float)
    assert runner.LF_READING in card.aggregate


def test_r11_reads_the_sources_own_mains_and_leaves_nothing_out_where_r0_names_none(tmp_path, monkeypatch):
    """R0's mains leaves the band (60 Hz here); a profile that names none leaves the whole band in (no 50 Hz fallback)."""
    _card, calls = _scored(tmp_path, monkeypatch, PROFILE)
    assert set(calls) == {(60.0,)}
    _card, calls = _scored(tmp_path, monkeypatch, {**PROFILE, "mains_hz": None})
    assert set(calls) == {(None,)}


def _window(monkeypatch, muted):
    """`_dsp_window` on 6 s of speech and its oracle restoration with the mute guard forced to `muted`; the row and R11's calls."""
    voice, noise = _speech_with_pauses(6.0)
    pair = types.SimpleNamespace(source=voice + noise, output=voice + 0.1 * noise, rate=RATE, profile={"mains_hz": 50.0})
    guards, calls = [], []
    monkeypatch.setattr(runner, "holds_mute", lambda _mono, _rate, floor: guards.append(floor) or muted)
    monkeypatch.setattr(lf_metrics, "lf_programme_db", _recorder(calls, lf_metrics.lf_programme_db, 2))
    row = WindowRow(0, 0.0, 6.0, "music")
    runner._dsp_window(pair, row, -60.0)
    assert guards == [-60.0]
    return row, calls


def test_a_window_the_mute_guard_skips_reads_r11_none_beside_r1(monkeypatch):
    """The production path: one mute decision per window skips R1 and R11 together; the source side stays the identity."""
    row, calls = _window(monkeypatch, True)
    assert row.output[runner.LF_READING] is None
    assert row.source[runner.LF_READING] == 0.0
    assert all(row.output[f"dsp.{name}"] is None for name in balance_metrics.READINGS)
    assert not calls


def test_a_window_the_mute_guard_keeps_reads_r11_with_the_profiles_mains(monkeypatch):
    """Not muted: R11 reads a float against R0's mains, paired against a zero source."""
    row, calls = _window(monkeypatch, False)
    assert isinstance(row.output[runner.LF_READING], float)
    assert row.source[runner.LF_READING] == 0.0
    assert calls == [(50.0,)]


def test_r11_is_a_two_sided_reading_with_a_neutral_target():
    """A thinner bass and a low end left are both wrong to a listener: the tail points away from 0 either way."""
    spec = scorecard.METRICS[runner.LF_READING]
    assert (spec.family, spec.better, spec.unit, spec.target) == ("dsp", scorecard.TWO_SIDED, "dB", 0.0)
    assert spec.routes == scorecard.PROGRAMME_ROUTES
    assert runner.LF_READING == f"dsp.{lf_metrics.READING}"
