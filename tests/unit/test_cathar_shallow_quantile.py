"""The depth guard's 'quantile' action (`cathar_noiseprint_shallow_action: quantile`): a shallow print replaced by the quantile print.

At the defaults (the guard off, 'scale') and with 'quantile' while the guard is off (0 dB), the
chain must issue exactly the depth guard's recorded HEAD 57bfdb8 commands on its four signals
and never read the quantile module. With the key at 20 dB a deep print (a clip or a tape with
hiss pauses) keeps those commands too. A shallow print (programme on a clip or a tape with no
pause) is still learned by the window pick, measured, and replaced in the denoise by the
whole-file quantile print, `noise_<stage>_quantile_k32_q0.05.np.json`, written beside it; the
guard logs the reading and the replacement. Material the quantile print cannot be learned from
falls back to 'scale' with a WARNING, and so does the guard after `cathar_noiseprint_method:
quantile` has already failed on the same material, without learning it a second time.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

import modules.cathar as cathar
import modules.noiseprint_depth as npd
import modules.noiseprint_quantile as nq
from modules import config as cfg
from scripts import tune_restoration
from tests.unit.test_cathar_print_depth_guard import DENOISE, HEAD, PRINT, SIGNALS, WORK, _denoise, _run

QUANTILE = "quantile"
GUARD = npd.LOG_PREFIX
LOG = "log_msg"
QUANTILE_PRINT = "quantile_print"
UNREAD = "quantile read"
UTF8 = "utf-8"
WARNING = {"level": "WARNING"}
DENSE_CLIP = "dense_clip"
SHALLOW = {"noiseprint_min_depth_db": 20.0, "noiseprint_shallow_action": QUANTILE}
# The guard's reading of the dense clip's window (the depth guard's tests, same seeded signal).
CLIP_READING = "the print's 1 window(s) sit 4.8 dB under the loud frames (-28.7 / -23.9 dBFS)"
FALLBACK_LINE = f"{GUARD} no quantile print could be learned from the material; falling back to 'scale'"
RATE = 8000
FRAME = npd.LOUD_FRAME


def _quantile_name(signal):
    return f"noise_{signal}_quantile_k32_q0.05.np.json"


def _logged(tmp_path, signal, **keys):
    """One run of the chain: (its commands, the guard's log, the quantile module's log)."""
    guard_log, quantile_log = MagicMock(), MagicMock()
    with patch.object(nq, LOG, quantile_log):
        calls = _run(tmp_path, signal, logs=guard_log, **keys)
    return calls, guard_log, quantile_log


def _guard_lines(log):
    return [call.args[0] for call in log.call_args_list if call.args[0].startswith(GUARD)]


def _without_denoise(calls):
    return [call for call in calls if call[:2] != ["cathar", "denoise"]]


@pytest.mark.parametrize("signal", sorted(SIGNALS))
@pytest.mark.parametrize("keys", [{}, {"noiseprint_shallow_action": QUANTILE}])
def test_the_guard_off_issues_heads_commands_and_reads_no_quantile_print(tmp_path, signal, keys):
    """The defaults, and 'quantile' with the key at 0: HEAD's slice, noiseprint and stage lines; the quantile module unread."""
    with patch.object(nq, QUANTILE_PRINT, side_effect=AssertionError(UNREAD)) as quantile:
        assert _run(tmp_path, signal, **keys) == HEAD[signal]
    quantile.assert_not_called()


@pytest.mark.parametrize("signal", ["paused_clip", "paused_tape"])
def test_a_deep_print_keeps_heads_commands_on_quantile(tmp_path, signal):
    """A print 20 dB or more under the loud frames is handed on untouched; no quantile print is learned."""
    with patch.object(nq, QUANTILE_PRINT, side_effect=AssertionError(UNREAD)) as quantile:
        assert _run(tmp_path, signal, **SHALLOW) == HEAD[signal]
    quantile.assert_not_called()


@pytest.mark.parametrize("signal", [DENSE_CLIP, "dense_tape"])
def test_a_shallow_print_is_replaced_by_the_quantile_print(tmp_path, signal):
    """Programme in the print: the denoise subtracts the quantile print; the window pick still learns (and the guard measures)."""
    calls, _guard, _quantile = _logged(tmp_path, signal, **SHALLOW)
    assert _denoise(calls) == DENOISE + ["--noiseprint", f"{WORK}/{_quantile_name(signal)}"]
    assert _without_denoise(calls) == _without_denoise(HEAD[signal])


@pytest.mark.parametrize("signal", [DENSE_CLIP, "dense_tape"])
def test_the_replacement_is_the_quantile_modules_print_of_the_stage_input(tmp_path, signal):
    """The file the denoise gets is the quantile module's print of the stage input, the same one the method would write."""
    _logged(tmp_path, signal, **SHALLOW)
    with patch.object(nq, LOG):
        expected, _acc = nq.learn(tmp_path / f"{signal}.wav")
    written = json.loads((tmp_path / "work" / _quantile_name(signal)).read_text(encoding=UTF8))
    assert np.allclose(written, expected, rtol=1e-12)


def test_the_replacement_writes_no_scaled_print(tmp_path):
    _logged(tmp_path, DENSE_CLIP, **SHALLOW)
    assert not (tmp_path / "work" / "noise_dense_clip_single_0.75s_depth20db.np.json").exists()


def test_the_replacement_is_logged_with_the_reading(tmp_path):
    """One guard line: the window's depth and levels, and the quantile print that replaces the learned one."""
    _calls, guard, _quantile = _logged(tmp_path, DENSE_CLIP, **SHALLOW)
    assert _guard_lines(guard) == [
        f"{GUARD} {CLIP_READING}: the print is programme; replaced by the whole-file quantile print {_quantile_name(DENSE_CLIP)}"
    ]


def test_the_quantile_module_logs_its_own_line(tmp_path):
    _calls, _guard, quantile = _logged(tmp_path, DENSE_CLIP, **SHALLOW)
    lines = [call.args[0] for call in quantile.call_args_list]
    assert len(lines) == 1
    assert lines[0].startswith(f"{nq.LOG_PREFIX} per-bin 5% quantile of the 1.02 s moving mean over ")


@pytest.mark.parametrize(("signal", "mode"), [(DENSE_CLIP, "single_0.75s"), ("dense_tape", "stitched_4.5s")])
def test_material_without_a_quantile_print_falls_back_to_scale(tmp_path, signal, mode):
    """Too few live windows for the quantile print: the denoise subtracts the print scaled to 20 dB, as on 'scale'."""
    with patch.object(nq, "MIN_WINDOWS", 10**9):
        calls, _guard, _quantile = _logged(tmp_path, signal, **SHALLOW)
    assert _denoise(calls) == DENOISE + ["--noiseprint", f"{WORK}/noise_{signal}_{mode}_depth20db.np.json"]
    assert not (tmp_path / "work" / _quantile_name(signal)).exists()


def test_the_fallback_is_logged_as_a_warning_then_scaled(tmp_path):
    """The guard's WARNING names the fallback, then the scale is logged as 'scale' logs it."""
    with patch.object(nq, "MIN_WINDOWS", 10**9):
        _calls, guard, _quantile = _logged(tmp_path, DENSE_CLIP, **SHALLOW)
    assert _guard_lines(guard) == [
        FALLBACK_LINE,
        f"{GUARD} {CLIP_READING}: the print is programme; scaled by -15.2 dB into noise_dense_clip_single_0.75s_depth20db.np.json",
    ]
    assert next(call for call in guard.call_args_list if call.args[0] == FALLBACK_LINE).kwargs == WARNING


def test_the_quantile_modules_warning_says_the_guard_scales(tmp_path):
    with patch.object(nq, "MIN_WINDOWS", 10**9):
        _calls, _guard, quantile = _logged(tmp_path, DENSE_CLIP, **SHALLOW)
    assert quantile.call_args.args[0].endswith(f"fewer than {10**9}; {npd.QUANTILE_FALLBACK}")
    assert quantile.call_args.kwargs == WARNING


def test_a_failed_quantile_method_is_not_learned_again_by_the_guard(tmp_path):
    """'quantile' as the method failed, so the window pick learns: the guard then scales, without a second quantile read."""
    with patch.object(nq, QUANTILE_PRINT, return_value=None) as quantile:
        calls = _run(tmp_path, DENSE_CLIP, noiseprint_method=QUANTILE, **SHALLOW)
    assert _denoise(calls) == DENOISE + ["--noiseprint", f"{WORK}/noise_dense_clip_single_0.75s_depth20db.np.json"]
    quantile.assert_called_once()


@pytest.mark.parametrize(
    ("method", "action", "expected"),
    [("window", QUANTILE, QUANTILE), ("window", "scale", "scale"), ("window", "drop", "drop"), (QUANTILE, QUANTILE, "scale")],
)
def test_the_guards_action(method, action, expected):
    assert npd.shallow_action(action, method) == expected


def test_the_split_band_passes_get_the_quantile_print(tmp_path):
    with patch("modules.split_band.recombine", side_effect=lambda low, high, out, hz: out), patch.object(nq, LOG):
        calls = _run(tmp_path, DENSE_CLIP, split_band_hz=4000, **SHALLOW)
    prints = [call[-1] for call in calls if call[:2] == ["cathar", "denoise"]]
    assert prints == [f"{WORK}/{_quantile_name(DENSE_CLIP)}"] * 2


def test_the_action_defaults_to_scale():
    assert (cfg.CONFIG["cathar_noiseprint_shallow_action"], cathar.CATHAR_NOISEPRINT_SHALLOW_ACTION) == ("scale", "scale")


def test_quantile_is_a_valid_action_for_the_config_and_the_guard():
    assert QUANTILE in cfg.VALID_CATHAR_NOISEPRINT_SHALLOW_ACTIONS
    assert QUANTILE in npd.SHALLOW_ACTIONS


@pytest.mark.parametrize(("raw", "warned"), [("quantile", False), (" Quantile ", False)])
def test_the_config_reads_quantile(raw, warned):
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = {"cathar_noiseprint_shallow_action": raw}
    with (
        patch("modules.config.Path.exists", return_value=True),
        patch.object(cfg, "yaml", mock_yaml),
        patch("builtins.open", MagicMock()),
        patch("builtins.print") as warn,
    ):
        conf, _source = cfg.load_config()
    assert conf["cathar_noiseprint_shallow_action"] == QUANTILE
    assert warn.called is warned


def test_the_tuning_driver_offers_quantile():
    assert tune_restoration.known_config_fields()["cathar_noiseprint_shallow_action"] == ({"scale", "drop", QUANTILE}, None, None)


# The guard on its own (modules/noiseprint_depth.py): a live noise signal and the step signal of its unit tests.


def _noise_steps(path, live=True):
    """300 frames of 1024 samples at levels 1/300 .. 1 on the mono mix: noise (live) or constant frames (all held: mutes)."""
    levels = np.repeat(np.arange(1, 301) / 300.0, FRAME)
    mono = levels * np.random.default_rng(9).normal(0.0, 1.0, len(levels)) if live else levels
    sf.write(str(path), np.stack([mono, mono], axis=1).astype(np.float32), RATE, subtype="FLOAT")
    return path


def _guarded(tmp_path, live=True, values=None, start_frame=199, action=QUANTILE):
    print_json = tmp_path / "noise_x_single_0.75s.np.json"
    print_json.write_text(json.dumps(PRINT if values is None else values), encoding=UTF8)
    wav = _noise_steps(tmp_path / "x.wav", live)
    guard_log, quantile_log = MagicMock(), MagicMock()
    with patch.object(npd, LOG, guard_log), patch.object(nq, LOG, quantile_log):
        result = npd.guard_print(print_json, wav, lambda: ([start_frame * FRAME], FRAME), 20.0, action)
    return result, print_json, wav, guard_log


def test_the_guard_replaces_a_shallow_print_with_the_quantile_print_beside_it(tmp_path):
    result, print_json, wav, _log = _guarded(tmp_path)
    assert result == print_json.parent / nq.print_name(wav)


def test_the_guards_replacement_is_the_quantile_print(tmp_path):
    result, _print_json, wav, _log = _guarded(tmp_path)
    with patch.object(nq, LOG):
        expected, _acc = nq.learn(wav)
    assert np.allclose(json.loads(result.read_text(encoding=UTF8)), expected, rtol=1e-12)


def test_the_guard_keeps_a_deep_print_on_quantile(tmp_path):
    with patch.object(nq, QUANTILE_PRINT, side_effect=AssertionError(UNREAD)):
        result, print_json, _wav, _log = _guarded(tmp_path, start_frame=9)
    assert result is print_json


def test_mute_material_falls_back_to_the_scaled_print(tmp_path):
    """Constant frames are all digital mutes to the quantile print: no live window, so the print is scaled as on 'scale'."""
    result, print_json, _wav, log = _guarded(tmp_path, live=False)
    assert result == npd.scaled_print_path(print_json, 20.0)
    assert log.call_args_list[0].kwargs == WARNING


def test_a_fallback_that_cannot_scale_keeps_the_learned_print(tmp_path):
    """No quantile print and a print in a format the guard does not know: the learned print goes on, with a WARNING."""
    result, print_json, _wav, log = _guarded(tmp_path, live=False, values={"bins": PRINT})
    assert result is print_json
    assert log.call_args.kwargs == WARNING


def test_an_existing_quantile_print_is_reused_by_the_guard(tmp_path):
    """A resumed run: the quantile print already beside the learned one is what the denoise gets, not learned again."""
    first, _print_json, _wav, _log = _guarded(tmp_path)
    with patch.object(nq, "learn", side_effect=AssertionError("learned again")):
        second, _print_json, _wav, _log = _guarded(tmp_path)
    assert second == first
