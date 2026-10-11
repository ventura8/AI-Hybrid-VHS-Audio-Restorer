"""The pause floor keeper's 'frame' target and shaped fill: levelled to each source frame, hiss-free, the body kept.

The fixture is a tape-like voice: tone bursts on the first half of each second over a floor of
low room tone (100-800 Hz, about -50 dBFS) and white hiss (about -56 dBFS). The restored copy
has its pauses pushed 40 dB down, the way the Mel-RoFormer leaves them.
"""

import functools
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.ndimage
import scipy.signal
import soundfile as sf

import modules.config as cfg
import modules.pause_floor as pause_floor
import scripts.tune_restoration as tr
from modules import event_log
from tests.unit.test_pause_floor import RATE, _gated, _on, _speech_mask, _write

LOW_BAND, HIGH_BAND = (150.0, 700.0), (6000.0, 12000.0)


@functools.lru_cache(maxsize=None)
def _tape_voice(seconds=8.0, seed=5):
    """Tone bursts over room tone and hiss, stereo float32. Cached: read, never written."""
    rng = np.random.default_rng(seed)
    count = int(seconds * RATE)
    t = np.arange(count) / RATE
    ramp = np.hanning(int(0.02 * RATE))
    burst = np.convolve(((t % 1.0) < 0.5).astype(np.float64), ramp / ramp.sum(), mode="same")
    room = scipy.signal.sosfilt(scipy.signal.butter(4, [100.0, 800.0], btype="bandpass", fs=RATE, output="sos"), rng.standard_normal(count))
    mono = 0.1 * np.sin(2 * np.pi * 220.0 * t) * burst + 3e-3 * room / np.sqrt(np.mean(room**2)) + 1.5e-3 * rng.standard_normal(count)
    return np.repeat(mono[:, None], 2, axis=1).astype(np.float32)


def _pause_mask(source):
    """The deep pauses: off the speech mask widened by 60 ms, so the bursts' ramps stay out."""
    return ~scipy.ndimage.binary_dilation(_speech_mask(source), iterations=int(0.06 * RATE))


def _band_db(audio, band, mask):
    """RMS in dB of the channel-mean's band over the masked samples."""
    sos = scipy.signal.butter(4, list(band), btype="bandpass", fs=RATE, output="sos")
    filtered = scipy.signal.sosfiltfilt(sos, np.asarray(audio, dtype=np.float64).mean(axis=1))
    return float(20.0 * np.log10(np.sqrt(np.mean(filtered[: len(mask)][mask[: len(filtered)]] ** 2)) + 1e-12))


def _settings(target="cluster", shape="flat", hz=3000.0, cut=18.0):
    return (
        patch("modules.pause_floor.PAUSE_FLOOR_TARGET", target),
        patch("modules.pause_floor.PAUSE_FLOOR_SHAPE", shape),
        patch("modules.pause_floor.PAUSE_FLOOR_SHELF_HZ", hz),
        patch("modules.pause_floor.PAUSE_FLOOR_HF_CUT_DB", cut),
    )


def _keep(reference, restored, work, **settings):
    """The keeper on (12 dB, p15) at the given target and shape; returns the path it hands on."""
    with ExitStack() as stack:
        for context in (*_on(), *_settings(**settings)):
            stack.enter_context(context)
        return pause_floor.apply_when_needed(reference, restored, work)


def _added(produced, restored):
    """What the stage added: the produced samples minus the restored ones."""
    return sf.read(str(produced), dtype="float64", always_2d=True)[0] - sf.read(str(restored), dtype="float64", always_2d=True)[0]


def _pair(tmp_path, restored_audio, source=None):
    source = _tape_voice() if source is None else source
    tmp_path.mkdir(parents=True, exist_ok=True)
    return _write(tmp_path / "ref.wav", source), _write(tmp_path / "res.wav", restored_audio)


def _fill_bands_db(tmp_path, **settings):
    """The fill's (low, high) band levels in the deep pauses, in dB against the source's own."""
    source = _tape_voice()
    reference, restored = _pair(tmp_path, _gated(source))
    added = _added(_keep(reference, restored, tmp_path, **settings), restored)
    mask = _pause_mask(source)
    return tuple(_band_db(added, band, mask) - _band_db(source, band, mask) for band in (LOW_BAND, HIGH_BAND))


def test_the_flat_frame_fill_is_the_source_twelve_db_down_in_every_band(tmp_path):
    low, high = _fill_bands_db(tmp_path, target="frame")
    assert abs(low + 12.0) < 1.5
    assert abs(high + 12.0) < 1.5


def test_the_shelf_keeps_the_room_tone_and_cuts_the_hiss_by_its_depth(tmp_path):
    low, high = _fill_bands_db(tmp_path, target="frame", shape="shelf")
    assert abs(low + 12.0) < 1.5
    assert abs(high + 30.0) < 2.5


def test_the_low_pass_keeps_the_room_tone_and_drops_the_hiss(tmp_path):
    low, high = _fill_bands_db(tmp_path, target="frame", shape="lowpass")
    assert abs(low + 12.0) < 1.5
    assert high < -45.0


def _quiet_db(path_or_audio, mask):
    audio = sf.read(str(path_or_audio), dtype="float64", always_2d=True)[0] if isinstance(path_or_audio, Path) else path_or_audio
    quiet = np.asarray(audio, dtype=np.float64)[: len(mask)][mask[: len(audio)]].mean(axis=1)
    return float(20.0 * np.log10(np.sqrt(np.mean(quiet**2)) + 1e-12))


def test_the_frame_target_follows_the_programme_level_the_separator_left(tmp_path):
    """The programme 10 dB down across the neural stage: 'frame' fills 22 dB under the source, 'cluster' still 12."""
    source = _tape_voice()
    mask = _pause_mask(source)
    reference, restored = _pair(tmp_path, (_gated(source) * 0.316).astype(np.float32))
    source_db = _quiet_db(source, mask)
    frame = _quiet_db(_keep(reference, restored, tmp_path / "frame", target="frame"), mask)
    cluster = _quiet_db(_keep(reference, restored, tmp_path / "cluster"), mask)
    assert abs(frame - (source_db - 22.0)) < 1.5
    assert abs(cluster - (source_db - 12.0)) < 1.5


def test_the_frame_target_leaves_a_floor_the_restoration_kept(tmp_path):
    """The programme 10 dB down and the pauses 16 dB down: 'frame' finds them at target, 'cluster' fills on top."""
    source = _tape_voice()
    speech = _speech_mask(source)
    gain = np.concatenate((np.where(speech, 0.316, 0.158), np.full(len(source) - len(speech), 0.316)))
    reference, restored = _pair(tmp_path, (source * gain[:, None]).astype(np.float32))
    assert _keep(reference, restored, tmp_path / "frame", target="frame") == restored
    assert _keep(reference, restored, tmp_path / "cluster") != restored


def test_the_shaped_fill_keeps_the_body_where_a_hiss_burst_stands_in_the_pause(tmp_path):
    """The restored pauses hold the source's hiss above 3 kHz and nothing under it.

    Read broadband, the hiss meets the frame target and the flat fill finds nothing to fill:
    the pause keeps its hiss and loses its body. The shaped fill reads the deficit through the
    shelf, where the hiss counts 18 dB down, and puts the room tone back 12 dB under the source.
    """
    source = _tape_voice()
    hiss = scipy.signal.sosfiltfilt(scipy.signal.butter(8, 3000.0, btype="highpass", fs=RATE, output="sos"), source, axis=0)
    speech = _speech_mask(source)
    speech = np.concatenate((speech, np.ones(len(source) - len(speech), dtype=bool)))
    reference, restored = _pair(tmp_path, np.where(speech[:, None], source, hiss).astype(np.float32))
    mask = _pause_mask(source)
    assert _keep(reference, restored, tmp_path / "flat", target="frame") == restored
    shelved = sf.read(str(_keep(reference, restored, tmp_path / "shelf", target="frame", shape="shelf")), always_2d=True)[0]
    assert abs(_band_db(shelved, LOW_BAND, mask) - _band_db(source, LOW_BAND, mask) + 12.0) < 2.0


def test_the_programme_gain_and_the_frame_target():
    reference = np.concatenate((np.full(70, 0.001), np.full(30, 0.1)))
    restored = np.concatenate((np.zeros(70), np.full(30, 0.03)))
    assert pause_floor.programme_gain(reference, restored) == pytest.approx(0.3)
    assert np.allclose(pause_floor.frame_target(reference, 0.3, 12.0), 0.3 * 10 ** (-12.0 / 20.0) * reference)


def test_the_deficit_gain_fills_only_what_is_missing_and_never_beyond_the_source():
    fill, weight = np.full(4, 0.01), np.ones(4)
    gains = pause_floor.deficit_gains(fill, np.array([0.0, 0.005, 0.02, 0.0]), weight, np.array([0.004, 0.004, 0.004, 0.05]))
    assert np.allclose(gains, [0.4, 0.0, 0.0, 1.0])


def test_the_frame_target_fills_one_frame_short_and_the_cluster_keeps_its_share_rule():
    """'frame' fills as soon as one quiet frame is 0.5 dB short of its target; 'cluster' at a 5 % share, as it always has."""
    met, one = np.r_[np.full(10, 0.4), np.zeros(990)], np.r_[3.0, np.zeros(999)]
    share, over = np.r_[np.full(50, 3.0), np.zeros(950)], np.r_[np.full(51, 3.0), np.zeros(949)]
    with _settings(target="frame")[0]:
        assert (pause_floor._targets_met(met), pause_floor._targets_met(one)) == (True, False)
    assert (pause_floor._targets_met(one), pause_floor._targets_met(share), pause_floor._targets_met(over)) == (True, True, False)


def test_flat_and_unknown_shapes_have_no_filter_and_a_short_block_still_shapes():
    shelf = pause_floor.shaper_for("shelf", RATE, 3000.0, 18.0)
    assert (pause_floor.shaper_for("flat", RATE, 3000.0, 18.0), pause_floor.shaper_for("odd", RATE, 3000.0, 18.0)) == (None, None)
    assert pause_floor.shaped(np.ones((5, 2)), shelf).shape == (5, 2)


def test_the_shelf_and_the_low_pass_filters():
    shelf = pause_floor.shaper_for("shelf", RATE, 3000.0, 18.0)
    assert (shelf.dry, len(shelf.sos)) == (pytest.approx(10 ** (-18.0 / 20.0)), 2)
    assert (pause_floor.shaper_for("lowpass", RATE, 2000.0, 18.0).dry, len(pause_floor.shaper_for("lowpass", RATE, 2000.0, 18.0).sos)) == (
        0.0,
        1,
    )
    assert pause_floor.shaper_for("shelf", 8000, 20000.0, 18.0) is not None


def test_shaping_block_by_block_matches_the_whole_file(tmp_path):
    source = _tape_voice(seconds=3.0)
    path = _write(tmp_path / "ref.wav", source)
    shaper = pause_floor.shaper_for("shelf", RATE, 3000.0, 18.0)
    whole = pause_floor.shaped(source.astype(np.float64), shaper)
    frame = int(0.02 * RATE)
    expected = np.sqrt(np.mean(whole.mean(axis=1)[: len(whole) // frame * frame].reshape(-1, frame) ** 2, axis=1))
    with patch("modules.pause_floor.BLOCK_SAMPLES", 8192):
        streamed = pause_floor.shaped_frame_levels(path, frame, shaper)
    assert np.allclose(streamed, expected, rtol=1e-9, atol=1e-12)


def test_the_shaped_fill_does_not_depend_on_the_block_size(tmp_path):
    source = _tape_voice(seconds=4.0)
    reference, restored = _pair(tmp_path, _gated(source), source)
    first = _keep(reference, restored, tmp_path / "a", target="frame", shape="shelf")
    with patch("modules.pause_floor.BLOCK_SAMPLES", 8192):
        second = _keep(reference, restored, tmp_path / "b", target="frame", shape="shelf")
    assert np.abs(sf.read(str(first))[0] - sf.read(str(second))[0]).max() < 1e-6


@pytest.mark.parametrize(
    ("settings", "name"),
    [
        ({}, "paused_12db_p15_res.wav"),
        ({"target": "frame"}, "paused_12db_p15_frame_res.wav"),
        ({"target": "frame", "shape": "shelf"}, "paused_12db_p15_frame_shelf3000c18_res.wav"),
        ({"shape": "lowpass", "hz": 2000.0}, "paused_12db_p15_lowpass2000_res.wav"),
    ],
)
def test_the_output_names_a_non_default_fill(tmp_path, settings, name):
    """A render at other settings never reuses another's file; the default keeps today's name."""
    with ExitStack() as stack:
        for context in _settings(**settings):
            stack.enter_context(context)
        assert pause_floor._target_for(tmp_path / "res.wav", tmp_path).name == name


def _record(tmp_path, monkeypatch, **settings):
    monkeypatch.setenv("AI_RESTORE_EVENT_LOG", str(tmp_path / "events"))
    reference, restored = _pair(tmp_path, _gated(_tape_voice()))
    with patch("modules.pause_floor.log_msg") as log:
        _keep(reference, restored, tmp_path / "work", **settings)
    record = json.loads((tmp_path / "events" / event_log.record_name(pause_floor.STAGE, Path("res.wav"))).read_text(encoding="utf-8"))
    return record, log.call_args.args[0]


def test_the_event_log_and_the_log_line_name_the_target_and_the_shape(tmp_path, monkeypatch):
    record, line = _record(tmp_path / "frame", monkeypatch, target="frame", shape="shelf")
    assert (record["target"], record["shape"], record["shelf_hz"], record["hf_cut_db"]) == ("frame", "shelf", 3000.0, 18.0)
    assert (abs(record["programme_gain_db"]) < 0.5, record["outcome"] > 0) == (True, True)
    assert ("each source frame at the restored programme's level (" in line, "the hiss cut 18 dB above 3000 Hz" in line) == (True, True)


def test_the_default_record_and_log_line_keep_their_fields(tmp_path, monkeypatch):
    record, line = _record(tmp_path / "default", monkeypatch)
    assert (record["target"], record["shape"], record["programme_gain_db"], "shelf_hz" in record) == ("cluster", "flat", None, False)
    assert line.endswith("dB under the source's pauses.")


def _loaded(user):
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


def test_a_config_without_the_keys_resolves_todays_fill_quietly():
    conf, warn = _loaded({"process_mode": "auto_pure_linear"})
    keys = ("pause_floor_target", "pause_floor_shape", "pause_floor_shelf_hz", "pause_floor_hf_cut_db")
    assert tuple(conf[key] for key in keys) == ("cluster", "flat", 3000.0, 18.0)
    warn.assert_not_called()


@pytest.mark.parametrize(
    ("key", "raw", "expected", "warned"),
    [
        ("pause_floor_target", " Frame ", "frame", False),
        ("pause_floor_target", "median", "cluster", True),
        ("pause_floor_target", None, "cluster", True),
        ("pause_floor_shape", " Shelf ", "shelf", False),
        ("pause_floor_shape", "lowpass", "lowpass", False),
        ("pause_floor_shape", "hiss", "flat", True),
        ("pause_floor_shelf_hz", 2000, 2000.0, False),
        ("pause_floor_shelf_hz", 100.0, 3000.0, True),
        ("pause_floor_hf_cut_db", 24.0, 24.0, False),
        ("pause_floor_hf_cut_db", 61.0, 18.0, True),
    ],
)
def test_the_new_keys_are_checked_like_the_others(key, raw, expected, warned):
    """Choices forgive case and spaces; numbers keep their bounds; anything else warns and keeps the default."""
    conf, warn = _loaded({key: raw})
    assert conf[key] == expected
    assert warn.called is warned


def test_a_tuning_grid_can_set_the_new_keys():
    fields = tr.known_config_fields()
    good = {"pause_floor_target": "frame", "pause_floor_shape": "shelf", "pause_floor_shelf_hz": 2500.0, "pause_floor_hf_cut_db": 24.0}
    assert tr.validate_overrides(good, fields) == []
    assert len(tr.validate_overrides({"pause_floor_target": "median", "pause_floor_shape": "hiss"}, fields)) == 2
