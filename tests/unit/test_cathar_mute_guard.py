"""cathar's noise-print mute guard (`cathar_noiseprint_mute_guard`): a digital mute is not tape noise.

The picks below were recorded from HEAD bb5fa2a, before the guard existed, on the synthetic
signals this file builds. With the key off (its default) both picks, the slice cathar learns
from and every command line must stay exactly those; with it on, a window that is a digital mute
(mostly held samples, or under -70 dBFS and 30 dB under the 23 ms frame p20) is never learned
from, and when every window is one the denoise runs without `--noiseprint`.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from modules import cathar
from modules import config as cfg
from scripts import tune_restoration

RATE = 16000
TAPE_RATE = 8000

# HEAD bb5fa2a's single-window starts (seconds), and with the guard on.
HEAD_SINGLE = {
    "zero": 0.9486875,
    "dither": 0.237125,
    "held": 0.0,
    "all_dither": 4.506375,
    "all_zero": 0.0,
    "none": 8.775625,
    "low_floor": 8.775625,
}
GUARDED_SINGLE = {
    "zero": 1.185875,
    "dither": 0.9486875,
    "held": 1.185875,
    "all_dither": None,
    "all_zero": None,
    "none": 8.775625,
    "low_floor": 8.775625,
}
# HEAD bb5fa2a's stitched pick on the tape (8 windows of 0.75 s), and with the guard on.
HEAD_STITCHED = "    [Cathar] Noise print stitched from 8 pauses at 2.0, 9.8, 10.5, 12.0, 18.8, 25.5, 26.2, 27.8 s"
GUARDED_STITCHED = "    [Cathar] Noise print stitched from 8 pauses at 2.0, 4.2, 9.8, 12.5, 14.2, 18.8, 22.0, 39.0 s"
GUARD = "    [Cathar] Noise-print mute guard:"
# The tape's two mutes: a -110 dBFS dither at 10-13 s and exact zeros at 25.5-28 s.
TAPE_MUTES_S = ((10.0, 13.0), (25.5, 28.0))
MUTES = {
    "zero": lambda rng, m: np.zeros(m),
    "dither": lambda rng, m: rng.normal(0.0, 10 ** (-110 / 20), m),
    "held": lambda rng, m: np.full(m, 2e-4),
    "all_dither": lambda rng, m: rng.normal(0.0, 10 ** (-110 / 20), m),
    "all_zero": lambda rng, m: np.zeros(m),
}


def _clip(kind, hiss_dbfs=-60.0, seconds=10.0, mute_s=1.5, seed=11):
    """Hiss with 1 s tone bursts every 2.5 s; its first 1.5 s (or, for an all_ kind, all of it) replaced by a mute."""
    rng = np.random.default_rng(seed)
    count = int(seconds * RATE)
    time = np.arange(count) / RATE
    mono = rng.normal(0.0, 10 ** (hiss_dbfs / 20), count) + 0.1 * np.sin(2 * np.pi * 440 * time) * ((time % 2.5) < 1.0)
    mute = count if kind.startswith("all_") else int(mute_s * RATE)
    if kind in MUTES:
        mono[:mute] = MUTES[kind](rng, mute)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _write_clip(tmp_path, kind):
    path = tmp_path / f"clip_{kind}.wav"
    hiss = -85.0 if kind == "low_floor" else -60.0
    sf.write(str(path), _clip(kind, hiss_dbfs=hiss), RATE, subtype="FLOAT")
    return path


def _write_tape(tmp_path):
    """40 s of hiss at -40 dBFS with 1.5 s bursts every 5 s, carrying the two mutes of TAPE_MUTES_S."""
    rng = np.random.default_rng(7)
    count = int(40.0 * TAPE_RATE)
    time = np.arange(count) / TAPE_RATE
    mono = rng.normal(0.0, 0.01, count) + np.sin(2 * np.pi * 440 * time) * 0.5 * ((time % 5.0) < 1.5)
    (a, b), (c, d) = ((int(start * TAPE_RATE), int(stop * TAPE_RATE)) for start, stop in TAPE_MUTES_S)
    mono[a:b] = rng.normal(0.0, 10 ** (-110 / 20), b - a)
    mono[c:d] = 0.0
    path = tmp_path / "tape.wav"
    sf.write(str(path), np.stack([mono, mono], axis=1).astype(np.float32), TAPE_RATE, subtype="FLOAT")
    return path


@contextmanager
def _logs():
    """cathar's log and the guard's, as one ordered list of messages."""
    log = MagicMock()
    with patch("modules.cathar.log_msg", log), patch("modules.noiseprint_guard.log_msg", log):
        messages = []
        yield messages
    messages.extend(call.args[0] for call in log.call_args_list)


@pytest.mark.parametrize("kind", sorted(HEAD_SINGLE))
def test_off_the_single_window_pick_is_heads(tmp_path, kind):
    """Key off: the quiet-window search returns HEAD's start on every signal, mutes included."""
    assert cathar._find_quiet_window(_write_clip(tmp_path, kind), 0.75) == HEAD_SINGLE[kind]


@pytest.mark.parametrize("kind", sorted(GUARDED_SINGLE))
def test_on_the_single_window_pick_skips_every_mute(tmp_path, kind):
    """Key on: exact zeros, a -110 dBFS dither and a held value are never picked; tape at -60 or -85 dBFS keeps its window."""
    with _logs():
        assert cathar._find_quiet_window(_write_clip(tmp_path, kind), 0.75, mute_guard=True) == GUARDED_SINGLE[kind]


def test_on_the_single_window_log_names_each_rejected_window_and_the_pick(tmp_path):
    """The rejected windows, each with its level and reason, then the window learned from with its level and the p20."""
    with _logs() as messages:
        cathar._find_quiet_window(_write_clip(tmp_path, "dither"), 0.75, mute_guard=True)
    assert len(messages) == 5
    assert messages[0].startswith(f"{GUARD} rejected the window at 0.00 s (-109.") and messages[0].endswith("dB under the frame p20")
    assert messages[-1].startswith(f"{GUARD} learning from the window at 0.95 s (-65.8 dBFS, frame p20 -60.")


def test_on_material_no_longer_than_the_window_keeps_the_start(tmp_path):
    """Material no longer than the window: the start, as without the guard."""
    path = tmp_path / "short.wav"
    sf.write(str(path), np.zeros((RATE // 2, 2), dtype=np.float32), RATE, subtype="FLOAT")
    assert cathar._find_quiet_window(path, 0.75, mute_guard=True) == 0.0


def _stitched_messages(tmp_path, guard):
    with _logs() as messages:
        assert cathar._extract_stitched_noiseprint(_write_tape(tmp_path), tmp_path / "probe.wav", 6.0, mute_guard=guard) is True
    return messages


def _picked_starts(message):
    return [float(s) for s in message.split(" at ")[1].rstrip(" s").split(", ")]


def test_off_the_stitched_pick_is_heads(tmp_path):
    """Key off: the stitched print takes HEAD's eight pauses, five of them inside the two mutes."""
    assert _stitched_messages(tmp_path, False) == [HEAD_STITCHED]


def test_on_the_stitched_pick_takes_no_pause_inside_a_mute(tmp_path):
    """Key on: no pause lies wholly inside a mute (two straddle an edge, at tape level)."""
    messages = _stitched_messages(tmp_path, True)
    assert messages[-1] == GUARDED_STITCHED
    starts = _picked_starts(GUARDED_STITCHED)
    assert not any(begin <= start and start + 0.75 <= end for start in starts for begin, end in TAPE_MUTES_S)


def test_on_the_stitched_log_gives_each_mute_one_line_then_the_pauses_level(tmp_path):
    """Each mute is one logged run with its quietest window and reason, then the pauses' levels and the p20."""
    messages = _stitched_messages(tmp_path, True)
    first, second, pauses = messages[0], messages[1], messages[2]
    assert first.startswith(f"{GUARD} rejected 10 window(s) from 10.00 to 13.00 s") and "dB under the frame p20" in first
    assert second.startswith(f"{GUARD} rejected 10 window(s) from 25.25 to 28.25 s") and "of its samples hold one value" in second
    assert pauses.startswith(f"{GUARD} pauses at ") and "frame p20 -40." in pauses


def test_on_a_stitched_pick_with_nothing_but_mute_drops_the_print(tmp_path):
    """Every window a mute: no probe is written and the print is bypassed."""
    path = tmp_path / "silent.wav"
    sf.write(str(path), np.zeros((40 * TAPE_RATE, 2), dtype=np.float32), TAPE_RATE, subtype="FLOAT")
    with _logs() as messages:
        assert cathar._extract_stitched_noiseprint(path, tmp_path / "probe.wav", 6.0, mute_guard=True) is False
    assert messages[-1] == f"{GUARD} every window is a digital mute (no frame above the digital floor); denoising without a learned print"
    assert not (tmp_path / "probe.wav").exists()


def _fake_tool(cmd, description=None, total_duration=None):
    """ffmpeg's slice and cathar's noiseprint, by the files they write."""
    del description, total_duration
    if "noiseprint" in cmd:
        Path(cmd[cmd.index("-o") + 1]).write_text("{}", encoding="utf-8")
        return
    sf.write(str(cmd[-1]), np.zeros((RATE, 2), dtype=np.float32), RATE, subtype="FLOAT")


def _pipeline_commands(work, wav, guard, total_duration=10.0):
    """Every external command of the cathar chain on `wav`: the slice, the noiseprint and each stage's line."""
    work.mkdir(parents=True, exist_ok=True)
    calls = []

    def stage(cmd, input_wav, *_args, **_kwargs):
        calls.append(["cathar", *map(str, cmd)])
        return input_wav

    def tool(cmd, **kwargs):
        calls.append([str(part) for part in cmd[1:]])
        _fake_tool(cmd, **kwargs)

    with (
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._run_cathar_step", side_effect=stage),
        patch("modules.cathar.run_command_with_progress", side_effect=tool),
        _logs(),
        patch.object(cathar, "CATHAR_NOISEPRINT_MUTE_GUARD", guard),
    ):
        cathar.filter_cathar_vhs_pipeline(wav, work, total_duration=total_duration)
    return calls


def _denoise(calls):
    return next(call for call in calls if call[:2] == ["cathar", "denoise"])


def _slice_start(calls):
    return next(call[call.index("-ss") + 1] for call in calls if "-ss" in call)


def _stages(calls):
    """The stage command lines other than the denoise, whose print path the guard may change."""
    return [call for call in calls if call[0] == "cathar" and call[1] != "denoise"]


def test_off_the_chain_slices_heads_window_and_denoises_with_its_print(tmp_path):
    """Key off on a clip opening on a -110 dBFS mute: HEAD's slice (0.237 s, inside the mute) and its print name."""
    calls = _pipeline_commands(tmp_path / "work", _write_clip(tmp_path, "dither"), False)
    expected = ["cathar", "denoise", "--alpha", "1.0", "--beta", "0.02", "--noiseprint"]
    assert (_slice_start(calls), _denoise(calls)) == (
        "0.237",
        expected + [str(tmp_path / "work" / "noise_clip_dither_single_0.75s.np.json")],
    )


def test_on_the_chain_slices_tape_and_names_the_print_by_the_guard(tmp_path):
    """Key on: the slice moves out of the mute and the print carries the guard in its name; every other stage line is unchanged."""
    wav = _write_clip(tmp_path, "dither")
    calls = _pipeline_commands(tmp_path / "on", wav, True)
    assert _slice_start(calls) == "0.949"
    assert _denoise(calls)[-1] == str(tmp_path / "on" / "noise_clip_dither_single_0.75s_muteguard.np.json")
    assert _stages(calls) == _stages(_pipeline_commands(tmp_path / "off", wav, False))


def test_on_a_clip_that_is_all_mute_the_chain_denoises_without_a_print(tmp_path):
    """Key on, every window a mute: no slice, no noiseprint run, and `denoise` estimates the noise itself."""
    calls = _pipeline_commands(tmp_path / "work", _write_clip(tmp_path, "all_dither"), True)
    assert not any("-ss" in call or "noiseprint" in call for call in calls)
    assert _denoise(calls) == ["cathar", "denoise", "--alpha", "1.0", "--beta", "0.02"]


def test_off_a_clip_that_is_all_mute_still_learns_heads_print(tmp_path):
    """Key off, the same clip: HEAD slices at 4.506 s and learns a print from the dither."""
    calls = _pipeline_commands(tmp_path / "work", _write_clip(tmp_path, "all_dither"), False)
    assert (_slice_start(calls), "--noiseprint" in _denoise(calls)) == ("4.506", True)


def test_on_a_tape_the_stitched_print_carries_the_guard_in_its_name(tmp_path):
    """A tape of 120 s or more takes the stitched path; with the key on its print is the guarded one."""
    calls = _pipeline_commands(tmp_path / "work", _write_tape(tmp_path), True, total_duration=600.0)
    assert _denoise(calls)[-1] == str(tmp_path / "work" / "noise_tape_stitched_4.5s_muteguard.np.json")


def test_prints_are_named_apart_by_the_guard_and_the_extractor_follows_it(tmp_path):
    """An unguarded name is HEAD's; the guarded extractor is the plain one with the guard bound."""
    assert cathar._noiseprint_name(tmp_path / "x.wav", 0.75, False) == "noise_x_single_0.75s.np.json"
    assert cathar._noiseprint_name(tmp_path / "x.wav", 4.5, True, True) == "noise_x_stitched_4.5s_muteguard.np.json"
    assert cathar._noiseprint_extractor(False, False) is cathar._extract_noiseprint_slice
    assert getattr(cathar._noiseprint_extractor(True, True), "keywords", None) == {"mute_guard": True}


def test_the_quiet_search_on_material_shorter_than_its_window_takes_the_start():
    """HEAD's search, unchanged: no window fits, so the start."""
    assert cathar._evaluate_quiet_probes(np.zeros(10, dtype=np.float32), 20) == 0


def test_the_key_defaults_off_everywhere():
    """The typed defaults, the resolved config, its constant and cathar's copy all hold the guard off."""
    defaults = cfg._typed_config_defaults()["cathar_noiseprint_mute_guard"]
    resolved = (cfg.CONFIG["cathar_noiseprint_mute_guard"], cfg.CATHAR_NOISEPRINT_MUTE_GUARD, cathar.CATHAR_NOISEPRINT_MUTE_GUARD)
    assert (defaults, *resolved) == (False, False, False, False)


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


@pytest.mark.parametrize(
    ("raw", "expected", "warns"), [(True, True, False), ("on", True, False), ("false", False, False), ("maybe", False, True)]
)
def test_the_loader_reads_the_key_as_a_switch(raw, expected, warns):
    """Quoted switches resolve by content; anything else warns and keeps the guard off."""
    conf, warn = _loaded({"cathar_noiseprint_mute_guard": raw})
    assert (conf["cathar_noiseprint_mute_guard"], warn.called) == (expected, warns)


def test_a_config_without_the_key_keeps_the_guard_off_quietly():
    """An older config.yaml, without the key, keeps the guard off and warns about nothing."""
    conf, warn = _loaded({"process_mode": "cathar"})
    assert (conf["cathar_noiseprint_mute_guard"], warn.called) == (False, False)


def test_the_tuning_driver_knows_the_key_as_a_switch():
    """A tuning grid may set the key: the driver reads it as a switch."""
    assert tune_restoration.known_config_fields()["cathar_noiseprint_mute_guard"] == (bool, None, None)
