"""Ear v3 wiring in the runner: the capture profile cached per source, R1/R2/R4 per window, R7 and sync per file."""

import json
import types

import numpy as np
import pytest
import soundfile as sf

from scripts.restoration_quality import audio_io, dsp_metrics, pause_metrics, runner, sibilance, source_profile
from scripts.restoration_quality.scorecard import ScoreCard
from tests.unit.test_restoration_quality_sibilance import RATE, _voice_with_esses

PROFILE = {"prog_bandwidth_hz": 4490.0, "brickwall_hz": 15400.0, "mains_hz": 60.0, "line_hz": 15734.0, "channel_state": {"channels": 2}}


def _registry():
    """No models: the dsp family needs none, and the real registry's release imports torch."""
    return types.SimpleNamespace(language="ro", release=lambda: None)


def _speech_with_pauses(seconds=16.0, hiss=3e-3, seed=7):
    """`_voice_with_esses` (vowels, 's' bursts) silenced to its hiss for 0.6 s in every 2 s: true pauses for R4."""
    voice, _ess = _voice_with_esses(seconds, seed)
    t = np.arange(len(voice)) / RATE
    noise = hiss * np.random.default_rng(seed + 1).standard_normal(len(voice))
    return (voice * ((t % 2.0) < 1.4)).astype(np.float32), noise.astype(np.float32)


def _write_pair(folder, seconds=16.0):
    """A stereo source (programme + hiss) and an output with the hiss 20 dB down: a restoration."""
    voice, noise = _speech_with_pauses(seconds)
    source, output = folder / "source.wav", folder / "output.wav"
    sf.write(str(source), np.stack([voice + noise, 0.8 * (voice + noise)], axis=1), RATE, subtype="FLOAT")
    sf.write(str(output), np.stack([voice + 0.1 * noise, 0.8 * (voice + 0.1 * noise)], axis=1), RATE, subtype="FLOAT")
    return source, output


@pytest.fixture(scope="module", name="scored")
def fixture_scored(tmp_path_factory):
    """One dsp-family run of `score_variants` with every window routed as speech."""
    folder = tmp_path_factory.mktemp("v3")
    source, output = _write_pair(folder)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(audio_io, "route_window", lambda _mono, _rate: "speech")
        result, cards = runner.score_variants(source, {"v": output}, families=("dsp",), registry=_registry(), cache_dir=folder / "cache")
    return result, cards["v"][0], folder


def test_the_file_level_readings_land_on_the_card(scored):
    """R7, the sync and the whole-pair holes are read once per pair; the capture profile lands as source-side `meta.*` readings."""
    _result, card, _folder = scored
    assert {"file.gain_ride_lu", "file.sync_drift_ms", "file.sync_unmatched", "file.dropouts"} <= set(card.file)
    assert card.file["file.dropouts"] == {"source": 0.0, "output": 0.0, "delta": 0.0}
    assert card.file["file.sync_unmatched"]["output"] == 0.0
    assert set(card.file["meta.prog_bandwidth_hz"]) == {"source"}


def test_the_window_readings_aggregate_on_the_speech_route(scored):
    """R1 (with the presence-or-air reading), R2 and R4 reach the aggregate the grids rank."""
    _result, card, _folder = scored
    expected = {"dsp.balance_tilt_db_oct", "dsp.balance_top_db", "dsp.sib_abs_level_db", "dsp.sib_texture_db", "dsp.gap_atten_db"}
    assert expected <= set(card.aggregate)
    assert card.aggregate["dsp.gap_atten_db"]["output"]["median"] > 10.0
    assert abs(card.aggregate["dsp.balance_tilt_db_oct"]["output"]["median"]) < 0.3


def test_the_balance_readings_are_paired_against_a_zero_source(scored):
    """Every R1 reading sits on the output side; the source side is the identity."""
    _result, card, _folder = scored
    row = card.rows[0]
    assert row.source["dsp.balance_top_db"] == row.source["dsp.balance_body_db"] == 0.0
    assert {"dsp.balance_tilt_db_oct", "dsp.balance_air_db", "dsp.balance_presence_db"} <= set(row.output)


def test_the_card_keeps_the_profile_and_the_vad_that_ran(scored):
    """The run notes say which capture profile and which pause VAD the readings stand on."""
    result, card, _folder = scored
    assert card.meta["pause_vad"] in ("dsp", "silero")
    assert card.meta["source_profile"]["channel_state"]["channels"] == 2
    assert result["source"]["profile"] == card.meta["source_profile"]


def test_the_document_marks_uncalibrated_gates_and_two_sided_targets(scored):
    """A starting threshold says it is one, and a two-sided reading carries its neutral target."""
    result, _card, _folder = scored
    assert result["gates"]["listener.bright"]["calibrated"] is False
    assert result["gates"]["dsp.hf_4k8k"]["calibrated"] is True
    assert result["metrics"]["dsp.balance_air_db"]["target"] == 0.0
    assert "target" not in result["metrics"]["dsp.hf_4k8k"]


def test_the_profile_is_cached_beside_the_source_features(scored):
    """One JSON per source under `<cache>/profile`, named by the source key and the profile code's hash."""
    _result, _card, folder = scored
    cached = list((folder / "cache" / runner.PROFILE_DIR).glob("*.json"))
    assert len(cached) == 1
    assert cached[0].stem.endswith(runner.profile_code_hash())


def test_the_tuning_view_flattens_the_new_readings(scored):
    """The flat view a grid ranks carries R1/R2/R4/R7, and the counts' worst window and p95."""
    result, _card, _folder = scored
    flat = runner.aggregates_for_tuning(result, "v")
    assert "dsp.sib_texture_db.output.median" in flat
    assert "file.gain_ride_lu.output.median" in flat
    assert {"dsp.clicks_per_s.output.max", "dsp.clicks_per_s.output.p95"} <= set(flat)


def _recorder(calls, real, signals):
    """`real`, recording the arguments after its `signals` audio arguments and its rate."""

    first = signals + 1

    def record(*args, **kwargs):
        calls.append(args[first:] + tuple(kwargs.values()))
        return real(*args, **kwargs)

    return record


def _score_short(tmp_path, monkeypatch, profile, route="speech"):
    """`score_pair` (dsp) on 6 s with R0 replaced by `profile`, every window routed as `route`; returns the calls seen."""
    calls = {"hum": [], "whistle": [], "sib": [], "gap": []}
    monkeypatch.setattr(source_profile, "capture_profile", lambda _audio, _rate: profile)
    monkeypatch.setattr(audio_io, "route_window", lambda _mono, _rate: route)
    monkeypatch.setattr(dsp_metrics, "hum_excess_db", _recorder(calls["hum"], dsp_metrics.hum_excess_db, 1))
    monkeypatch.setattr(dsp_metrics, "whistle_line_db", _recorder(calls["whistle"], dsp_metrics.whistle_line_db, 1))
    monkeypatch.setattr(sibilance, "sib_readings", _recorder(calls["sib"], sibilance.sib_readings, 2))
    monkeypatch.setattr(pause_metrics, "gap_residual_readings", _recorder(calls["gap"], pause_metrics.gap_residual_readings, 2))
    source, output = _write_pair(tmp_path, seconds=6.0)
    runner.score_pair(source, output, families=("dsp",), registry=_registry(), cache_dir=tmp_path / "cache")
    return calls


def test_hum_and_whistle_read_the_sources_own_mains_and_line(tmp_path, monkeypatch):
    """R0 replaces the 50 Hz / 15625 Hz assumptions on both sides of every window."""
    calls = _score_short(tmp_path, monkeypatch, PROFILE)
    assert set(calls["hum"]) == {(60.0,)}
    assert set(calls["whistle"]) == {(15734.0,)}


def test_without_a_mains_or_line_hum_and_whistle_fall_back_to_pal(tmp_path, monkeypatch):
    """A profile that found neither reads 50 Hz and 15625 Hz, as every release before."""
    calls = _score_short(tmp_path, monkeypatch, {**PROFILE, "mains_hz": None, "line_hz": None})
    assert set(calls["hum"]) == {(runner.DEFAULT_MAINS_HZ,)}
    assert set(calls["whistle"]) == {(dsp_metrics.WHISTLE_HZ,)}


def test_the_s_and_the_pauses_are_clipped_to_the_brickwall_not_the_programme_band(tmp_path, monkeypatch):
    """R2 and R4 read up to the codec cut: the voiced band misses the 's' and the hiss in the pauses."""
    calls = _score_short(tmp_path, monkeypatch, PROFILE)
    assert set(calls["sib"]) == {(15400.0,)}
    assert set(calls["gap"]) == {(15400.0,)}


@pytest.mark.parametrize(("route", "gap_calls"), [("music", 0), ("mixed", 1), ("silence", 0)])
def test_the_pause_residual_is_read_on_speech_and_mixed_windows_only(tmp_path, monkeypatch, route, gap_calls):
    """A music window has no true pauses to judge; the 's' is read on every window (its own detector decides)."""
    calls = _score_short(tmp_path, monkeypatch, PROFILE, route)
    assert len(calls["gap"]) == gap_calls
    assert len(calls["sib"]) == 1


def _counted_profile(monkeypatch, calls):
    def fake(audio, rate):
        calls.append((audio.shape, rate))
        return {**PROFILE, "channel_state": {"dual_mono": np.bool_(True)}, "line_ppm": np.float64(7.0)}

    monkeypatch.setattr(source_profile, "capture_profile", fake)


def _profile_pair(tmp_path):
    wav = tmp_path / "s.wav"
    sf.write(str(wav), np.zeros((RATE, 2), dtype=np.float32), RATE, subtype="FLOAT")
    return types.SimpleNamespace(cache_dir=tmp_path / "cache", source_key="abc", source_wav=wav, profile=None)


def test_the_profile_is_read_on_both_channels_once_and_then_served_from_the_cache(tmp_path, monkeypatch):
    """The first read sees the raw stereo file; the second never decodes it; numpy values survive the JSON."""
    calls = []
    _counted_profile(monkeypatch, calls)
    pair = _profile_pair(tmp_path)
    first, second = runner.load_profile(pair), runner.load_profile(pair)
    assert calls == [((RATE, 2), RATE)]
    assert second == {**PROFILE, "channel_state": {"dual_mono": True}, "line_ppm": 7.0}
    assert first["mains_hz"] == second["mains_hz"] == 60.0


def test_a_profile_from_other_code_or_a_broken_file_is_read_again(tmp_path, monkeypatch):
    """The cache name carries the code hash; an unreadable or non-object file reads as missing."""
    path = runner.profile_path(tmp_path, "abc")
    monkeypatch.setattr(runner, "profile_code_hash", lambda: "other")
    assert runner.profile_path(tmp_path, "abc") != path
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert runner.read_profile(path) is None
    path.write_text("[1, 2]", encoding="utf-8")
    assert runner.read_profile(path) is None


def test_the_profile_cache_key_covers_the_code_the_profile_calls(monkeypatch):
    """A change in the hum or level-class helpers the profile reads with re-reads it too."""
    before = runner.profile_code_hash()
    monkeypatch.setattr(runner, "PROFILE_CODE", runner.PROFILE_CODE[:1])
    assert runner.profile_code_hash() != before
    assert {module.__name__.rsplit(".", 1)[-1] for module in runner.PROFILE_CODE} == {"source_profile"}


def test_writing_a_profile_leaves_no_temporary_file(tmp_path):
    """The write goes through a temporary file that is renamed over the target."""
    path = tmp_path / "profile" / "k.json"
    runner.write_profile(path, {"mains_hz": np.float64(50.0), "dual_mono": np.bool_(False)})
    assert json.loads(path.read_text(encoding="utf-8")) == {"mains_hz": 50.0, "dual_mono": False}
    assert [entry.name for entry in path.parent.iterdir()] == ["k.json"]


@pytest.mark.parametrize(("presence", "air", "top"), [(0.2, -0.5, -0.5), (0.4, None, 0.4), (None, None, None), (-0.3, 0.3, -0.3)])
def test_the_top_reading_is_presence_or_air_whichever_moved_further(presence, air, top):
    """The bright / dull flags read one number: the larger move, presence on a tie or without air."""
    assert runner.top_db(presence, air) == top


def test_profile_entries_keep_the_numbers_on_the_source_side():
    """The channel state and missing fields stay out of the `meta.*` readings."""
    entries = runner.profile_entries({**PROFILE, "line_ppm": None})
    assert entries["meta.mains_hz"] == {"source": 60.0}
    assert "meta.line_ppm" not in entries
    assert not any("channel" in name for name in entries)


def test_card_to_dict_carries_the_run_notes():
    """A card's meta (profile, VAD) is part of its JSON view."""
    assert runner.card_to_dict(ScoreCard(meta={"pause_vad": "dsp"}), "x")["meta"] == {"pause_vad": "dsp"}
