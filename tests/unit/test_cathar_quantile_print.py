"""cathar's whole-file quantile print in the chain (`cathar_noiseprint_method`).

At the default ('window') the chain must issue exactly HEAD 053ba30's commands on the four signals of the depth guard's
tests (whose recorded lines are HEAD 57bfdb8's, unchanged since), slice and noiseprint runs included, and never read the
quantile module. On 'quantile' the window pick is not made: no slice, no `cathar noiseprint`, and the denoise subtracts
`noise_<stem>_quantile_k32_q0.05.np.json`, written from the whole stage input. Material the quantile print cannot be
learned from goes back to the window pick, guards and all. The music profile learns no print, and auto_pure_linear's
subtraction never reads the key.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

import modules.cathar as cathar
import modules.noiseprint_quantile as noiseprint_quantile
import modules.spectral_denoise as spectral_denoise
from modules import config as cfg
from scripts import tune_restoration
from tests.unit.test_cathar_print_depth_guard import DENOISE, HEAD, PRINT, SIGNALS, WORK, _denoise, _fake_tool, _keys, _run

QUANTILE_NAME = "quantile_k32_q0.05.np.json"
LOG = "log_msg"
QUANTILE = "quantile"
WINDOW = "window"
DENSE = "dense_clip"
KEY = "cathar_noiseprint_method"
UNREAD = "quantile read"
QUANTILE_PRINT = "quantile_print"
UTF8 = "utf-8"


def _quantile_denoise(signal):
    return DENOISE + ["--noiseprint", f"{WORK}/noise_{signal}_{QUANTILE_NAME}"]


def _without_denoise(calls):
    return [call for call in calls if call[:2] != ["cathar", "denoise"]]


def _window_learning(calls):
    """The window pick's external runs: ffmpeg's slice (from `-y`) and cathar's noiseprint."""
    return [call for call in calls if call[0] in ("-y", "noiseprint")]


@pytest.mark.parametrize("signal", sorted(SIGNALS))
def test_the_default_config_issues_heads_commands(tmp_path, signal):
    """'window': HEAD's slice, noiseprint and stage lines on every signal; the quantile module is never read."""
    with patch.object(noiseprint_quantile, QUANTILE_PRINT, side_effect=AssertionError(UNREAD)) as quantile:
        assert _run(tmp_path, signal) == HEAD[signal]
    quantile.assert_not_called()


@pytest.mark.parametrize("signal", sorted(SIGNALS))
def test_window_issues_heads_commands(tmp_path, signal):
    assert _run(tmp_path, signal, noiseprint_method=WINDOW) == HEAD[signal]


@pytest.mark.parametrize("signal", sorted(SIGNALS))
def test_quantile_learns_no_window_and_subtracts_the_quantile_print(tmp_path, signal):
    """'quantile': no slice and no `cathar noiseprint`; the denoise takes the quantile print; every stage line is HEAD's."""
    with patch.object(noiseprint_quantile, LOG):
        calls = _run(tmp_path, signal, noiseprint_method=QUANTILE)
    assert _denoise(calls) == _quantile_denoise(signal)
    assert _window_learning(calls) == []
    assert _without_denoise(calls) == [call for call in _without_denoise(HEAD[signal]) if call[0] not in ("-y", "noiseprint")]


@pytest.mark.parametrize("signal", ["paused_clip", "dense_tape"])
def test_the_quantile_print_is_the_modules_print_of_the_stage_input(tmp_path, signal):
    with patch.object(noiseprint_quantile, LOG):
        _run(tmp_path, signal, noiseprint_method=QUANTILE)
        expected, _acc = noiseprint_quantile.learn(tmp_path / f"{signal}.wav")
    written = json.loads((tmp_path / "work" / f"noise_{signal}_{QUANTILE_NAME}").read_text(encoding=UTF8))
    assert np.allclose(written, expected, rtol=1e-12)


def test_the_quantile_print_logs_one_line(tmp_path):
    log = MagicMock()
    with patch.object(noiseprint_quantile, LOG, log):
        _run(tmp_path, DENSE, noiseprint_method=QUANTILE)
    lines = [call.args[0] for call in log.call_args_list]
    assert len(lines) == 1
    assert lines[0].startswith(f"{noiseprint_quantile.LOG_PREFIX} per-bin 5% quantile of the 1.02 s moving mean over ")


def test_material_it_cannot_learn_from_goes_back_to_the_window_pick(tmp_path):
    """No live window (the quantile module returns None): HEAD's window pick, slice and noiseprint included."""
    with patch.object(noiseprint_quantile, QUANTILE_PRINT, return_value=None) as quantile:
        calls = _run(tmp_path, DENSE, noiseprint_method=QUANTILE)
    assert calls == HEAD[DENSE]
    quantile.assert_called_once()


def test_a_digital_mute_stage_goes_back_to_the_window_pick_with_a_warning(tmp_path):
    """A stage input that is all digital silence: the quantile print logs a WARNING and the window pick learns the print."""
    wav = tmp_path / "mute.wav"
    sf.write(str(wav), np.zeros((3 * 16000, 2), dtype=np.float32), 16000, subtype="FLOAT")
    log = MagicMock()
    with (
        patch("modules.cathar.run_command_with_progress", side_effect=_fake_tool),
        patch("modules.cathar.log_msg"),
        patch.object(noiseprint_quantile, LOG, log),
        _keys(noiseprint_method=QUANTILE),
    ):
        result = cathar._learned_print(wav, tmp_path, 3.0)
    assert result == tmp_path / "noise_mute_single_0.75s.np.json"
    assert log.call_args.kwargs == {"level": "WARNING"}


def test_the_window_guards_do_not_touch_the_quantile_print(tmp_path):
    """The depth guard measures windows; the quantile print has none, so at 20 dB it is still what the denoise gets."""
    with (
        patch.object(noiseprint_quantile, LOG),
        patch("modules.noiseprint_depth.guard_print", side_effect=AssertionError("guarded")),
        patch("modules.noiseprint_guard.guarded_quiet_probe", side_effect=AssertionError("mute-guarded")),
    ):
        calls = _run(tmp_path, DENSE, noiseprint_method=QUANTILE, noiseprint_min_depth_db=20.0, noiseprint_mute_guard=True)
    assert _denoise(calls) == _quantile_denoise(DENSE)


def test_the_split_band_passes_get_the_quantile_print(tmp_path):
    with patch("modules.split_band.recombine", side_effect=lambda low, high, out, hz: out), patch.object(noiseprint_quantile, LOG):
        calls = _run(tmp_path, DENSE, noiseprint_method=QUANTILE, split_band_hz=4000)
    prints = [call[-1] for call in calls if call[:2] == ["cathar", "denoise"]]
    assert prints == [f"{WORK}/noise_dense_clip_{QUANTILE_NAME}"] * 2


def test_the_music_profile_learns_no_print_on_quantile(tmp_path):
    """Music keeps `cathar_music_enable_noiseprint` (false): no print of either kind, the denoise runs without one."""
    wav = SIGNALS[DENSE][0](tmp_path / "music.wav")
    denoise = MagicMock(return_value=wav)
    with (
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._run_cathar_step", side_effect=lambda cmd, input_wav, *a, **k: input_wav),
        patch("modules.cathar._cathar_denoise_step", denoise),
        patch("modules.cathar.log_msg"),
        patch.object(noiseprint_quantile, QUANTILE_PRINT, side_effect=AssertionError(UNREAD)),
        _keys(noiseprint_method=QUANTILE),
    ):
        cathar.filter_cathar_vhs_pipeline(wav, tmp_path, total_duration=10.0, strategy={"profile": {"tonal_persistence": 0.2}})
    assert denoise.call_args.kwargs["noiseprint_path"] is None


def test_the_key_never_reaches_auto_pure_linears_subtraction(tmp_path):
    """APL learns its print through the window step whatever the key says."""
    print_json = tmp_path / "noise.np.json"
    print_json.write_text(json.dumps(PRINT), encoding=UTF8)
    with (
        patch.object(cathar, "CATHAR_NOISEPRINT_METHOD", QUANTILE),
        patch.object(spectral_denoise, "APL_USE_NATIVE_SUPPRESS", False),
        patch("modules.spectral_denoise._material", return_value=(1.8, 4.0)),
        patch("modules.spectral_denoise._cathar_available", return_value=True),
        patch("modules.cathar._cathar_noiseprint_step", return_value=print_json),
        patch("modules.cathar._cathar_denoise_step", return_value=tmp_path / "out.wav") as denoise,
        patch.object(noiseprint_quantile, QUANTILE_PRINT, side_effect=AssertionError(UNREAD)),
    ):
        spectral_denoise._subtract(tmp_path / "in.wav", tmp_path, 10.0)
    assert denoise.call_args.kwargs["noiseprint_path"] == print_json


def test_the_method_defaults_to_window_everywhere():
    methods = [cfg.CONFIG[KEY], cfg.CATHAR_NOISEPRINT_METHOD, cathar.CATHAR_NOISEPRINT_METHOD]
    assert methods == [WINDOW] * 3


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


def test_a_config_without_the_key_takes_the_window_quietly():
    conf, warn = _loaded({"process_mode": "cathar"})
    assert conf[KEY] == WINDOW
    warn.assert_not_called()


@pytest.mark.parametrize(
    ("raw", "expected", "warned"),
    [
        (" Quantile ", QUANTILE, False),
        (WINDOW, WINDOW, False),
        ("median", WINDOW, True),
        (None, WINDOW, True),
        (3, WINDOW, True),
    ],
)
def test_the_method_is_a_choice(raw, expected, warned):
    conf, warn = _loaded({KEY: raw})
    assert conf[KEY] == expected
    assert warn.called is warned


def test_the_tuning_driver_reads_the_method_as_a_choice():
    assert tune_restoration.known_config_fields()[KEY] == ({WINDOW, QUANTILE}, None, None)


def test_the_repository_config_takes_the_window():
    """config.yaml names the key at its default, so the shipped configuration keeps every output's bytes."""
    text = (Path(cfg.__file__).resolve().parents[1] / "config.yaml").read_text(encoding=UTF8)
    assert f'\n{KEY}: "{WINDOW}"\n' in text.replace("\r\n", "\n")
