"""The pause floor keeper's per-band deficit (`pause_floor_deficit` 'bands'): each band of a shelved fill to its own target.

The fixture is the hiss-free tests' tape-like voice (room tone and hiss under tone bursts). Its
restored copy keeps the room tone in the deep pauses 11 dB under the source while the highs sit
40 dB down, the way the Mel-RoFormer and the expander left the IA hunt's Pavarotti documentary:
the low band meets a 12 dB target, so a deficit read on the shaped audio as a whole fills nothing
and the highs stay emptied.
"""

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

import modules.config as cfg
import modules.pause_floor as pause_floor
import scripts.tune_restoration as tr
from modules import event_log
from tests.unit.test_pause_floor import RATE, _gated, _on, _write
from tests.unit.test_pause_floor_hissfree import HIGH_BAND, LOW_BAND, _band_db, _pause_mask, _settings, _tape_voice


def _room_kept(source):
    """The gated copy with the source's band under 2 kHz put back 11 dB down in the deep pauses."""
    low = scipy.signal.sosfiltfilt(scipy.signal.butter(4, 2000.0, fs=RATE, output="sos"), source.astype(np.float64), axis=0)
    mask = _pause_mask(source)[:, None]
    return (_gated(source).astype(np.float64) + 10.0 ** (-11.0 / 20.0) * low * mask).astype(np.float32)


def _keep(tmp_path, deficit, shape="shelf", restored_audio=None):
    """The keeper (12 dB, p15, 'frame', 2 kHz, cut 18) over the room-kept copy; returns (produced, restored)."""
    source = _tape_voice()
    tmp_path.mkdir(parents=True, exist_ok=True)
    reference = _write(tmp_path / "ref.wav", source)
    restored = _write(tmp_path / "res.wav", _room_kept(source) if restored_audio is None else restored_audio)
    with ExitStack() as stack:
        for context in (*_on(), *_settings(target="frame", shape=shape, hz=2000.0, cut=18.0)):
            stack.enter_context(context)
        stack.enter_context(patch("modules.pause_floor.PAUSE_FLOOR_DEFICIT", deficit))
        return pause_floor.apply_when_needed(reference, restored, tmp_path), restored


def _pause_bands_db(path):
    """The (low, high) band levels of a file in the deep pauses, in dB against the source's own."""
    source = _tape_voice()
    audio = sf.read(str(path), dtype="float64", always_2d=True)[0]
    mask = _pause_mask(source)
    return tuple(_band_db(audio, band, mask) - _band_db(source, band, mask) for band in (LOW_BAND, HIGH_BAND))


def test_a_broadband_deficit_leaves_the_highs_empty_where_the_room_tone_meets_its_target(tmp_path):
    produced, _restored = _keep(tmp_path, "broadband")
    assert _pause_bands_db(produced)[1] < -38.0


def test_per_band_the_highs_rise_to_the_fill_plus_the_cut_and_the_room_tone_stays(tmp_path):
    produced, restored = _keep(tmp_path, "bands")
    low, high = _pause_bands_db(produced)
    assert produced != restored
    assert produced.name.startswith("paused_12db_p15_frame_shelf2000c18_bands_")
    assert abs(low - _pause_bands_db(restored)[0]) < 0.5
    assert abs(high + 30.0) < 2.5


def test_per_band_both_bands_rise_when_both_were_emptied(tmp_path):
    produced, _restored = _keep(tmp_path, "bands", restored_audio=_gated(_tape_voice()))
    low, high = _pause_bands_db(produced)
    assert abs(low + 12.0) < 1.5
    assert abs(high + 30.0) < 2.5


@pytest.mark.parametrize("shape", ["flat", "lowpass"])
def test_bands_change_nothing_without_a_shelf(tmp_path, shape):
    broadband, _restored = _keep(tmp_path / "a", "broadband", shape=shape)
    bands, _restored = _keep(tmp_path / "b", "bands", shape=shape)
    assert broadband.name == bands.name
    assert np.array_equal(sf.read(str(broadband))[0], sf.read(str(bands))[0])


def test_per_band_nothing_is_filled_where_both_bands_meet_their_targets(tmp_path):
    source = _tape_voice()
    produced, restored = _keep(tmp_path, "bands", restored_audio=source)
    assert produced == restored


def test_the_record_names_the_per_band_deficit_only_on_a_shelf(tmp_path):
    shelf = pause_floor.Profile("frame", "shelf", 12.0, 2000.0, 18.0, "bands")
    flat = shelf._replace(shape="flat")
    with patch("modules.pause_floor.PAUSE_FLOOR_SUPPRESSED_ONLY", False):
        assert pause_floor._thresholds(shelf)["deficit"] == "bands"
        assert "deficit" not in pause_floor._thresholds(flat)
        assert "deficit" not in pause_floor._thresholds(shelf._replace(deficit="broadband"))
    del tmp_path


def test_the_log_line_says_each_band_has_its_own_target():
    shelf = pause_floor.Profile("frame", "shelf", 12.0, 2000.0, 18.0, "bands")
    assert pause_floor._note(0.0, shelf).endswith(", each band to its own target")
    assert not pause_floor._note(0.0, shelf._replace(deficit="broadband")).endswith("own target")


def test_the_spans_record_the_frames_either_band_lifts(tmp_path, monkeypatch):
    monkeypatch.setenv(event_log.ENV_VAR, str(tmp_path / "events"))
    produced, _restored = _keep(tmp_path / "work", "bands")
    record = json.loads((tmp_path / "events" / event_log.record_name(pause_floor.STAGE, Path("res.wav"))).read_text(encoding="utf-8"))
    assert record["deficit"] == "bands"
    assert record["outcome"] > 0
    assert record["filled_s"]
    assert produced.exists()


def test_the_band_shapers_split_at_the_corner_and_keep_the_fill_s_cut():
    shaper = pause_floor.shaper_for("shelf", RATE, 2000.0, 18.0)
    fills, bands = pause_floor.band_shapers(pause_floor.Shaping("src.wav", shaper, shaper))
    assert [(s.wet, s.dry) for s in fills] == [(1.0, 0.0), (0.0, shaper.dry)]
    assert [(s.wet, s.dry) for s in bands] == [(1.0, 0.0), (0.0, 1.0)]
    shelved_fills, _bands = pause_floor.band_shapers(pause_floor.Shaping("shelved.wav", None, shaper))
    assert shelved_fills[1].dry == 1.0


def test_the_bands_add_up_to_the_shaped_audio():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((4096, 2))
    shaper = pause_floor.shaper_for("shelf", RATE, 2000.0, 18.0)
    fills, _bands = pause_floor.band_shapers(pause_floor.Shaping("src.wav", shaper, shaper))
    split = pause_floor.shaped(data, fills[0]) + pause_floor.shaped(data, fills[1])
    assert np.allclose(split, pause_floor.shaped(data, shaper), atol=1e-12)


def test_a_full_wet_shaper_is_the_one_every_release_ran():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((2048, 1))
    shaper = pause_floor.shaper_for("shelf", RATE, 2000.0, 12.0)
    low = scipy.signal.sosfiltfilt(shaper.sos, data, axis=0, padlen=min(3 * (2 * len(shaper.sos) + 1), len(data) - 1))
    assert np.array_equal(pause_floor.shaped(data, shaper), low + shaper.dry * (data - low))


def test_two_parts_at_half_gain_write_one_part_at_full(tmp_path):
    source = _tape_voice(seconds=2.0)
    reference = _write(tmp_path / "ref.wav", source)
    restored = _write(tmp_path / "res.wav", np.zeros_like(source))
    gain = np.full(len(source), 0.5)
    halves = pause_floor.fill_file_parts(reference, restored, tmp_path / "halves.wav", [(gain, None), (gain, None)])
    whole = pause_floor.fill_file(reference, restored, tmp_path / "whole.wav", gain * 2.0)
    assert np.allclose(sf.read(str(halves))[0], sf.read(str(whole))[0], atol=1e-6)


def test_the_profile_carries_the_configured_deficit_and_the_release_fill_stays_broadband():
    with patch("modules.pause_floor.PAUSE_FLOOR_DEFICIT", "bands"):
        assert pause_floor.configured().deficit == "bands"
    assert pause_floor.release_profile().deficit == "broadband"


@pytest.mark.parametrize(("raw", "expected"), [("bands", "bands"), ("BROADBAND", "broadband"), ("odd", "broadband"), (None, "broadband")])
def test_the_setting_is_normalised_to_a_known_deficit(raw, expected):
    assert cfg._normalize_pause_floor_deficit(raw) == expected


def test_the_tuner_and_the_config_know_the_setting():
    assert cfg.PAUSE_FLOOR_DEFICIT == "broadband"
    assert tr.validate_overrides({"pause_floor_deficit": "bands"}) == []
    assert tr.validate_overrides({"pause_floor_deficit": "everywhere"})
