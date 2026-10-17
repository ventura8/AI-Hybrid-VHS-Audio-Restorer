"""Ear v4 wiring in the runner: the hiss readings per window, the cached anchor, its meta entries, the speech-only pause
depth and the trade skipped on a rate mismatch."""

import itertools
import types

import numpy as np
import pytest
import soundfile as sf

from scripts.restoration_quality import audio_io, dsp_metrics, hiss_floor, runner
from tests.unit.test_restoration_quality_hiss_floor import RATE, capture

HISS_FIELDS = ("dsp.hiss_prog_margin_db", "dsp.hiss_prog_excess_db", "dsp.hiss_pause_margin_db", "dsp.hiss_room_delta_db")
ANCHOR_META, HOW_ENTRY, PROGRAMME_HOW, SOURCE = "hiss_anchor", "meta.hiss_anchor_how", "programme_how", "source"


def _registry():
    """No models: the dsp family needs none, and the real registry's release imports torch."""
    return types.SimpleNamespace(language="ro", release=lambda: None)


def _write_pair(folder):
    """The synthetic capture as the source and the same with its hiss 12 dB down as the output."""
    programme, noise, source = capture()
    paths = folder / "source.wav", folder / "output.wav"
    sf.write(str(paths[0]), source.astype(np.float32), RATE, subtype="FLOAT")
    sf.write(str(paths[1]), (programme + noise * 10.0 ** (-12.0 / 20.0)).astype(np.float32), RATE, subtype="FLOAT")
    return paths


@pytest.fixture(scope="module", name="scored")
def fixture_scored(tmp_path_factory):
    """One dsp-family run with the four source windows routed speech, mixed, silence, speech (the output's alike)."""
    folder = tmp_path_factory.mktemp("hiss")
    source, output = _write_pair(folder)
    routes = itertools.cycle(("speech", "mixed", "silence", "speech"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(audio_io, "route_window", lambda _mono, _rate: next(routes))
        result, cards = runner.score_variants(source, {"v": output}, families=("dsp",), registry=_registry(), cache_dir=folder / "cache")
    card, pair = cards["v"]
    return result, card, pair, folder


def test_every_programme_window_carries_the_hiss_readings(scored):
    """Each window routed as speech or mixed holds the paired hiss readings; the one routed as silence holds none."""
    _result, card, _pair, _folder = scored
    read = {row.route: all(name in row.output for name in HISS_FIELDS) for row in card.rows}
    assert read == {"speech": True, "mixed": True, "silence": False}


def test_the_hiss_readings_reach_the_aggregate(scored):
    """The output with its hiss 12 dB down reads its pause margin more than 6 dB under the source's."""
    _result, card, _pair, _folder = scored
    margin = card.aggregate["dsp.hiss_pause_margin_db"]
    assert margin["output"]["median"] < margin["source"]["median"] - 6.0


def test_pause_depth_on_speech_windows_reads_only_them(scored):
    """`dsp.pause_depth_speech_db` aggregates the two speech windows; `dsp.pause_depth_db` the speech and mixed ones."""
    _result, card, _pair, _folder = scored
    counts = (card.aggregate["dsp.pause_depth_speech_db"]["output"]["n"], card.aggregate["dsp.pause_depth_db"]["output"]["n"])
    assert counts == (2, 3)


def test_the_anchor_lands_in_the_card(scored):
    """The anchor's coverage is a source-side `meta.*` entry; the run notes say how it was read and at which level."""
    _result, card, _pair, _folder = scored
    assert set(card.file["meta.hiss_anchor_pause_s"]) == set(card.file["meta.hiss_anchor_validated"]) == {SOURCE}
    assert card.meta[ANCHOR_META]["l_crit_db"] == hiss_floor.L_CRIT_DB
    assert card.meta[ANCHOR_META]["how"] == hiss_floor.PAUSES


def test_the_anchor_path_lands_in_the_card(scored):
    """The synthetic capture validates from its pauses: path code 1 and `programme_how` "pauses"."""
    _result, card, _pair, _folder = scored
    assert card.file[HOW_ENTRY] == {SOURCE: runner.ANCHOR_HOW_CODES[hiss_floor.PAUSES]}
    assert card.meta[ANCHOR_META][PROGRAMME_HOW] == hiss_floor.PAUSES


def test_each_programme_path_has_its_code():
    """No validated anchor reads 0, the pause anchor 1, the stationary anchor 2."""
    second = hiss_floor.Anchor(np.array([1e-8]), np.array([1000.0]), 0.0, hiss_floor.STATIONARY, True)
    cases = [
        hiss_floor.Anchor(None, np.zeros(0), 0.0, hiss_floor.NO_FLOOR, False),
        hiss_floor.Anchor(np.array([1e-8]), np.array([1000.0]), 2.0, hiss_floor.PAUSES, True),
        hiss_floor.Anchor(np.array([1e-8]), np.array([1000.0]), 0.0, hiss_floor.ALL_FRAMES, False, second),
    ]
    codes = [runner.anchor_entries(anchor)[HOW_ENTRY][SOURCE] for anchor in cases]
    assert codes == pytest.approx([0.0, 1.0, 2.0])
    assert runner.anchor_meta(cases[2])[PROGRAMME_HOW] == hiss_floor.STATIONARY


def test_the_anchor_is_cached_and_served_by_the_code_hash(scored, monkeypatch):
    """Once written, the anchor is read back without reading the source again; a broken file is read afresh."""
    _result, _card, pair, _folder = scored
    path = runner.hiss_anchor_path(pair.cache_dir, pair.source_key)
    monkeypatch.setattr(hiss_floor, "source_anchor", lambda *_args: pytest.fail("the cached anchor was not served"))
    assert runner.load_hiss_anchor(pair, {}).pause_s == hiss_floor.Anchor.from_dict(runner.read_profile(path)).pause_s
    assert path.name.endswith(f"_hiss_{runner.hiss_code_hash()}.json")


def test_a_broken_anchor_file_is_read_afresh(scored, monkeypatch):
    """A cached file that is not an anchor is replaced by a fresh reading."""
    _result, _card, pair, _folder = scored
    path = runner.hiss_anchor_path(pair.cache_dir, pair.source_key)
    path.write_text('{"level": null}', encoding="utf-8")
    fresh = hiss_floor.Anchor(None, np.zeros(0), 0.0, hiss_floor.NO_FLOOR, False)
    monkeypatch.setattr(hiss_floor, "source_anchor", lambda *_args: fresh)
    assert runner.load_hiss_anchor(pair, {}).how == hiss_floor.NO_FLOOR


def test_the_trade_is_skipped_when_the_rates_differ(monkeypatch):
    """A 44.1 kHz source against a 48 kHz output records why instead of a wrong `file.noise_removed_db`."""
    monkeypatch.setattr(dsp_metrics, "trade", lambda *_paths: {"noise_removed_db": 5.0})
    card = types.SimpleNamespace(meta={})
    mismatched = types.SimpleNamespace(rate=44100, output_rate=48000, source_wav="s", output_wav="o")
    matched = types.SimpleNamespace(rate=44100, output_rate=44100, source_wav="s", output_wav="o")
    assert (runner.trade_entries(mismatched, card), card.meta["trade_skipped"]) == ({}, "sample rates differ")
    assert runner.trade_entries(matched, card)["file.noise_removed_db"]["output"] == 5.0


def _refused(*_paths):
    """`os.replace` as Windows answers it while another scorer replaces the same file."""
    raise PermissionError(5, "Access is denied")


def test_a_profile_another_scorer_is_writing_is_left_to_it(tmp_path, monkeypatch):
    """Two scorers caching the same anchor at once: the loser drops its temporary file; with no file there it raises."""
    path = tmp_path / "profile" / "key_hiss_hash.json"
    runner.write_profile(path, {"how": "pauses"})
    monkeypatch.setattr(runner.os, "replace", _refused)
    runner.write_profile(path, {"how": "pauses"})
    assert sorted(item.name for item in path.parent.iterdir()) == [path.name]
    with pytest.raises(PermissionError):
        runner.write_profile(tmp_path / "other" / "missing.json", {"how": "pauses"})
