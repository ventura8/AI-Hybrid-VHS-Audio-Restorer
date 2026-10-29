"""cathar's noise-print depth guard in the chain (`cathar_noiseprint_min_depth_db`, `cathar_noiseprint_shallow_action`).

The command lines below were recorded from HEAD 57bfdb8, before the guard existed, on the
synthetic signals this file builds, the work directory written as `<WORK>`. At the default
(0) the chain must issue exactly those, slice and noiseprint runs included, and never measure
anything; with the key at 20 dB a print whose windows sit 20 dB or more under the loud frames
(a clip with hiss pauses, a tape with pauses) must still issue them, so its output keeps its
bytes. A print shallower than the key (programme on both clips and tapes with no pause) is
scaled into `<print>_depth20db.np.json` or, on 'drop', left out of the denoise. The keys
never reach auto_pure_linear, which learns its print through the same step.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

import modules.cathar as cathar
import modules.noiseprint_depth as noiseprint_depth
import modules.spectral_denoise as spectral_denoise
from modules import config as cfg
from scripts import tune_restoration

CLIP_RATE = 16000
TAPE_RATE = 8000
PRINT = [0.25 * (1 + k % 7) for k in range(1025)]
WORK = "<WORK>"
GUARD = noiseprint_depth.LOG_PREFIX
STAGES = [
    ["cathar", "dewind", "--cutoff", "80"],
    ["cathar", "azimuth", "--max-ms", "5.0", "--method", "gcc-phat"],
    ["cathar", "stereo", "--mono-below", "100"],
    ["cathar", "declick", "--method", "ar", "--threshold", "8.0"],
    ["cathar", "decrackle", "--sensitivity", "6"],
    ["cathar", "inpaint", "--max-gap-ms", "50", "--iterations", "3"],
    ["cathar", "deplosive", "-s", "4"],
    ["cathar", "declip", "--method", "spade", "--threshold", "0.95"],
    ["cathar", "dehum", "--freq", "50", "--harmonics", "8", "--adaptive"],
    ["cathar", "repair", "-s", "2"],
]
DEESSER = ["cathar", "deesser", "--bands", "3", "-f", "4000", "--threshold=12.0"]
DENOISE = ["cathar", "denoise", "--alpha", "1.0", "--beta", "0.02"]


def _learned(name, mode, slice_at=None):
    """HEAD's runs that learn the print: ffmpeg's 0.75 s slice from `slice_at` (single-window clips), then cathar's noiseprint."""
    probe = f"{WORK}/silence_probe_{name}.wav"
    cut = [["-y", "-ss", slice_at, "-i", f"<TMP>/{name}.wav", "-t", "0.750", "-c:a", "pcm_f32le", probe]] if slice_at else []
    return cut + [["noiseprint", probe, "-o", f"{WORK}/noise_{name}_{mode}.np.json", "--no-banner"]]


def _polish(name, mode):
    """HEAD's denoise with the learned print, then the de-esser."""
    return [DENOISE + ["--noiseprint", f"{WORK}/noise_{name}_{mode}.np.json"], DEESSER]


# HEAD 57bfdb8's external commands for each signal (main checkout, 2026-10-12), and its stitched picks.
HEAD = {
    "paused_clip": STAGES + _learned("paused_clip", "single_0.75s", "8.776") + _polish("paused_clip", "single_0.75s"),
    "dense_clip": STAGES + _learned("dense_clip", "single_0.75s", "1.660") + _polish("dense_clip", "single_0.75s"),
    "paused_tape": STAGES + _learned("paused_tape", "stitched_4.5s") + _polish("paused_tape", "stitched_4.5s"),
    "dense_tape": STAGES + _learned("dense_tape", "stitched_4.5s") + _polish("dense_tape", "stitched_4.5s"),
}
# The guard's readings on the seeded signals: the stitched pauses 31 dB down, the clip's window 4.8 dB.
DEEP_TAPE_READING = "the print's 6 window(s) sit 31.1 dB under the loud frames (-40.1 / -9.0 dBFS), at least 20 dB"
SHALLOW_CLIP_READING = (
    "the print's 1 window(s) sit 4.8 dB under the loud frames (-28.7 / -23.9 dBFS): the print is programme; "
    "scaled by -15.2 dB into noise_dense_clip_single_0.75s_depth20db.np.json"
)
HEAD_STITCHED = {
    "paused_tape": "    [Cathar] Noise print stitched from 6 pauses at 2.0, 4.2, 8.5, 12.2, 18.8, 26.5 s",
    "dense_tape": "    [Cathar] Noise print stitched from 6 pauses at 16.0, 24.2, 25.0, 33.0, 33.8, 34.5 s",
}


def _paused_clip(path):
    """10 s of hiss at -60 dBFS with 1 s tone bursts every 2.5 s: its quietest 0.75 s is hiss, about 37 dB down."""
    rng = np.random.default_rng(11)
    time = np.arange(10 * CLIP_RATE) / CLIP_RATE
    mono = rng.normal(0.0, 1e-3, len(time)) + 0.1 * np.sin(2 * np.pi * 440 * time) * ((time % 2.5) < 1.0)
    sf.write(str(path), np.stack([mono, mono], axis=1).astype(np.float32), CLIP_RATE, subtype="FLOAT")
    return path


def _dense_clip(path):
    """10 s of programme-like noise that never pauses: its quietest 0.75 s sits a few dB under the loud frames."""
    rng = np.random.default_rng(5)
    time = np.arange(10 * CLIP_RATE) / CLIP_RATE
    mono = rng.normal(0.0, 0.05, len(time)) * (1.0 + 0.3 * np.sin(2 * np.pi * 0.37 * time))
    sf.write(str(path), np.stack([mono, mono], axis=1).astype(np.float32), CLIP_RATE, subtype="FLOAT")
    return path


def _paused_tape(path):
    """40 s of hiss at -40 dBFS with 1.5 s bursts every 5 s: six stitched pauses, about 31 dB down."""
    rng = np.random.default_rng(7)
    time = np.arange(40 * TAPE_RATE) / TAPE_RATE
    mono = rng.normal(0.0, 0.01, len(time)) + 0.5 * np.sin(2 * np.pi * 440 * time) * ((time % 5.0) < 1.5)
    sf.write(str(path), np.stack([mono, mono], axis=1).astype(np.float32), TAPE_RATE, subtype="FLOAT")
    return path


def _dense_tape(path):
    """40 s of slowly swelling noise with no pause: the stitched pauses are programme."""
    rng = np.random.default_rng(3)
    time = np.arange(40 * TAPE_RATE) / TAPE_RATE
    mono = rng.normal(0.0, 0.05, len(time)) * (1.0 + 0.3 * np.sin(2 * np.pi * 0.11 * time))
    sf.write(str(path), np.stack([mono, mono], axis=1).astype(np.float32), TAPE_RATE, subtype="FLOAT")
    return path


SIGNALS = {
    "paused_clip": (_paused_clip, 10.0),
    "dense_clip": (_dense_clip, 10.0),
    "paused_tape": (_paused_tape, 600.0),
    "dense_tape": (_dense_tape, 600.0),
}


def _fake_tool(cmd, description=None, total_duration=None):
    """ffmpeg's slice and cathar's noiseprint, by the files they write: the print is a list of 1025 magnitudes."""
    del description, total_duration
    if "noiseprint" in cmd:
        Path(cmd[cmd.index("-o") + 1]).write_text(json.dumps(PRINT), encoding="utf-8")
        return
    sf.write(str(cmd[-1]), np.zeros((CLIP_RATE, 2), dtype=np.float32), CLIP_RATE, subtype="FLOAT")


@contextmanager
def _keys(**keys):
    """cathar's copies of the given keys (CATHAR_<NAME>) patched for the block; none patched keeps the repository's config."""
    with patch.multiple(cathar, **{f"CATHAR_{name.upper()}": value for name, value in keys.items()}) if keys else _nothing():
        yield


@contextmanager
def _nothing():
    yield


def _run(tmp_path, signal, logs=None, **keys):
    """Every external command of the cathar chain on one signal, its paths written as `<WORK>`."""
    make, total = SIGNALS[signal]
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    wav = make(tmp_path / f"{signal}.wav")
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
        patch("modules.cathar.log_msg", logs or MagicMock()),
        patch.object(noiseprint_depth, "log_msg", logs or MagicMock()),
        _keys(**keys),
    ):
        cathar.filter_cathar_vhs_pipeline(wav, work, total_duration=total)
    return [[_placed(part, work, tmp_path) for part in call] for call in calls]


def _placed(part, work, tmp_path):
    """A command argument with the work and temporary directories as placeholders and `/` as the separator, on any OS."""
    return part.replace(str(work), WORK).replace(str(tmp_path), "<TMP>").replace("\\", "/")


def _denoise(calls):
    return next(call for call in calls if call[:2] == ["cathar", "denoise"])


@pytest.mark.parametrize("signal", sorted(SIGNALS))
def test_the_default_config_issues_heads_commands(tmp_path, signal):
    """Key at its default (0): HEAD 57bfdb8's slice, noiseprint and stage lines on every signal, deep print or shallow."""
    assert _run(tmp_path, signal) == HEAD[signal]


@pytest.mark.parametrize("signal", sorted(HEAD_STITCHED))
@pytest.mark.parametrize("depth", [0.0, 20.0])
def test_the_stitched_pick_is_heads_with_the_guard_off_or_on(tmp_path, signal, depth):
    """The stitched extractor, now split from its pick, takes HEAD's six pauses; the guard's own pick logs nothing."""
    log = MagicMock()
    _run(tmp_path, signal, logs=log, noiseprint_min_depth_db=depth)
    assert [call.args[0] for call in log.call_args_list if "stitched from" in call.args[0]] == [HEAD_STITCHED[signal]]


@pytest.mark.parametrize("signal", sorted(SIGNALS))
def test_the_default_config_never_measures_the_print(tmp_path, signal):
    """Key at 0: the guard is not called at all, so nothing is read and nothing is written beyond HEAD's files."""
    with patch("modules.noiseprint_depth.guard_print", side_effect=AssertionError("measured")) as guard:
        _run(tmp_path, signal)
    guard.assert_not_called()


@pytest.mark.parametrize(
    ("signal", "action"), [("paused_clip", "scale"), ("paused_clip", "drop"), ("paused_tape", "scale"), ("paused_tape", "drop")]
)
def test_a_deep_print_keeps_heads_commands_with_the_key_at_20(tmp_path, signal, action):
    """A print 20 dB or more under the loud frames (hiss pauses, single window or stitched) is handed on untouched."""
    assert _run(tmp_path, signal, noiseprint_min_depth_db=20.0, noiseprint_shallow_action=action) == HEAD[signal]


def _guard_lines(tmp_path, signal, **keys):
    """The guard's log lines over one run of the chain."""
    log = MagicMock()
    _run(tmp_path, signal, logs=log, **keys)
    return [call.args[0] for call in log.call_args_list if call.args[0].startswith(GUARD)]


def test_a_deep_print_logs_its_depth_and_that_it_is_kept(tmp_path):
    """One line: the six stitched pauses' depth and levels, and that the learned print goes on."""
    assert _guard_lines(tmp_path, "paused_tape", noiseprint_min_depth_db=20.0) == [
        f"{GUARD} {DEEP_TAPE_READING}; the learned print is kept"
    ]


def test_a_shallow_print_logs_its_depth_and_the_scaled_file(tmp_path):
    """One line: the window's depth and levels, the gain and the scaled print's name."""
    assert _guard_lines(tmp_path, "dense_clip", noiseprint_min_depth_db=20.0) == [f"{GUARD} {SHALLOW_CLIP_READING}"]


@pytest.mark.parametrize(("signal", "mode"), [("dense_clip", "single_0.75s"), ("dense_tape", "stitched_4.5s")])
def test_a_shallow_print_is_scaled_and_the_denoise_takes_the_scaled_one(tmp_path, signal, mode):
    """Programme in the print (no pause): the denoise subtracts `<print>_depth20db.np.json`; every other line is HEAD's."""
    calls = _run(tmp_path, signal, noiseprint_min_depth_db=20.0)
    assert _denoise(calls) == DENOISE + ["--noiseprint", f"{WORK}/noise_{signal}_{mode}_depth20db.np.json"]
    assert [call for call in calls if call[:2] != ["cathar", "denoise"]] == [
        call for call in HEAD[signal] if call[:2] != ["cathar", "denoise"]
    ]


@pytest.mark.parametrize(("signal", "mode"), [("dense_clip", "single_0.75s"), ("dense_tape", "stitched_4.5s")])
def test_the_scaled_print_sits_at_the_floor(tmp_path, signal, mode):
    """The scaled print is the learned one times 10^(-(20 - depth)/20), the depth measured on the windows the pick took."""
    stitched = mode.startswith("stitched")
    _run(tmp_path, signal, noiseprint_min_depth_db=20.0)
    wav = tmp_path / f"{signal}.wav"
    depth = noiseprint_depth.print_depth_db(wav, cathar._probe_windows(wav, 4.5 if stitched else 0.75, stitched, False))[0]
    scaled = json.loads((tmp_path / "work" / f"noise_{signal}_{mode}_depth20db.np.json").read_text(encoding="utf-8"))
    assert np.allclose(scaled, np.asarray(PRINT) * 10.0 ** ((depth - 20.0) / 20.0), rtol=1e-9)
    assert 2.0 < depth < 10.0


@pytest.mark.parametrize("signal", ["dense_clip", "dense_tape"])
def test_a_shallow_print_is_dropped_on_drop(tmp_path, signal):
    """'drop': the denoise runs without --noiseprint, as the music profile does; the print is still learned (HEAD's runs)."""
    calls = _run(tmp_path, signal, noiseprint_min_depth_db=20.0, noiseprint_shallow_action="drop")
    assert _denoise(calls) == DENOISE
    assert any(call[0] == "noiseprint" for call in calls)


def test_the_guard_measures_the_window_the_single_pick_took(tmp_path):
    """The single-window path's window is the quiet search's own (start in samples, 0.75 s long)."""
    wav = _dense_clip(tmp_path / "dense.wav")
    start_s = cathar._find_quiet_window(wav, 0.75)
    assert cathar._probe_windows(wav, 0.75, False, False) == ([int(round(start_s * CLIP_RATE))], int(0.75 * CLIP_RATE))


def test_the_guard_measures_the_pauses_the_stitched_pick_took(tmp_path):
    """The stitched path's windows are the stitched extractor's own six pauses."""
    wav = _paused_tape(tmp_path / "tape.wav")
    picked, window, _rate = cathar._stitched_picks(wav, 4.5)
    assert cathar._probe_windows(wav, 4.5, True, False) == (picked, window)


@pytest.mark.parametrize(
    ("seconds", "rate", "total", "mode"), [(10, CLIP_RATE, 10.0, "single_0.75s"), (40, TAPE_RATE, 600.0, "stitched_4.5s")]
)
def test_a_mono_source_is_guarded_on_both_paths(tmp_path, seconds, rate, total, mode):
    """A one-channel stage with no pause: both picks read its one channel, and the denoise gets the scaled print."""
    wav = tmp_path / "mono.wav"
    sf.write(str(wav), np.random.default_rng(5).normal(0.0, 0.05, seconds * rate).astype(np.float32), rate, subtype="FLOAT")
    with (
        patch("modules.cathar.run_command_with_progress", side_effect=_fake_tool),
        patch("modules.cathar.log_msg"),
        patch.object(noiseprint_depth, "log_msg"),
        _keys(noiseprint_min_depth_db=20.0),
    ):
        result = cathar._learned_print(wav, tmp_path, total)
    assert result == tmp_path / f"noise_mono_{mode}_depth20db.np.json"


def test_the_guard_follows_the_mute_guard_pick(tmp_path):
    """With the mute guard on, the windows measured are the guarded pick's (None from the single search: no window)."""
    wav = _dense_clip(tmp_path / "dense.wav")
    with patch("modules.cathar._find_quiet_window", return_value=None) as search:
        assert cathar._probe_windows(wav, 0.75, False, True) == ([], int(0.75 * CLIP_RATE))
    assert search.call_args.kwargs == {"mute_guard": True}


def test_no_learned_print_means_no_guard(tmp_path):
    """The noiseprint step found nothing to learn from: the guard is not asked, the denoise runs without a print."""
    with (
        patch("modules.cathar._cathar_noiseprint_step", return_value=None),
        patch("modules.noiseprint_depth.guard_print") as guard,
        patch.object(cathar, "CATHAR_NOISEPRINT_MIN_DEPTH_DB", 20.0),
    ):
        assert cathar._learned_print(tmp_path / "x.wav", tmp_path, 10.0) is None
    guard.assert_not_called()


def test_the_split_band_passes_get_the_guarded_print(tmp_path):
    """With split-band subtraction both passes subtract the scaled print."""
    with patch("modules.split_band.recombine", side_effect=lambda low, high, out, hz: out):
        calls = _run(tmp_path, "dense_clip", noiseprint_min_depth_db=20.0, split_band_hz=4000)
    prints = [call[-1] for call in calls if call[:2] == ["cathar", "denoise"]]
    assert prints == [f"{WORK}/noise_dense_clip_single_0.75s_depth20db.np.json"] * 2


def test_the_depth_defaults_to_off_everywhere():
    """The typed default, the resolved config, its constant and cathar's copy: 0 dB, the guard off."""
    depths = [
        cfg._typed_config_defaults()["cathar_noiseprint_min_depth_db"],
        cfg.CONFIG["cathar_noiseprint_min_depth_db"],
        cfg.CATHAR_NOISEPRINT_MIN_DEPTH_DB,
        cathar.CATHAR_NOISEPRINT_MIN_DEPTH_DB,
    ]
    assert depths == pytest.approx([0.0] * 4)


def test_the_shallow_action_defaults_to_scale_everywhere():
    assert (cfg.CONFIG["cathar_noiseprint_shallow_action"], cathar.CATHAR_NOISEPRINT_SHALLOW_ACTION) == ("scale", "scale")


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


def test_a_config_without_the_keys_keeps_the_guard_off_quietly():
    conf, warn = _loaded({"process_mode": "cathar"})
    assert conf["cathar_noiseprint_min_depth_db"] == pytest.approx(0.0)
    assert conf["cathar_noiseprint_shallow_action"] == "scale"
    warn.assert_not_called()


@pytest.mark.parametrize(
    ("raw", "expected", "warned"),
    [
        (20, 20.0, False),
        ("17.5", 17.5, False),
        (0, 0.0, False),
        (-1.0, 0.0, True),
        (61.0, 0.0, True),
        (True, 0.0, True),
        ("deep", 0.0, True),
    ],
)
def test_the_depth_is_a_bounded_number_of_db(raw, expected, warned):
    """0 to 60 dB, quoted or not; a bool, a negative, anything over 60 or a word warns and keeps the guard off."""
    conf, warn = _loaded({"cathar_noiseprint_min_depth_db": raw})
    assert conf["cathar_noiseprint_min_depth_db"] == pytest.approx(expected)
    assert warn.called is warned


@pytest.mark.parametrize(
    ("raw", "expected", "warned"), [(" Drop ", "drop", False), ("scale", "scale", False), ("halve", "scale", True), (None, "scale", True)]
)
def test_the_shallow_action_is_a_choice(raw, expected, warned):
    conf, warn = _loaded({"cathar_noiseprint_shallow_action": raw})
    assert conf["cathar_noiseprint_shallow_action"] == expected
    assert warn.called is warned


def test_the_tuning_driver_reads_the_depth_as_a_float_from_0_to_60():
    kind, low, high = tune_restoration.known_config_fields()["cathar_noiseprint_min_depth_db"]
    assert kind is float
    assert [low, high] == pytest.approx([0.0, 60.0])


def test_the_tuning_driver_reads_the_shallow_action_as_a_choice():
    assert tune_restoration.known_config_fields()["cathar_noiseprint_shallow_action"] == ({"scale", "drop", "quantile"}, None, None)


def test_the_keys_never_reach_auto_pure_linears_subtraction(tmp_path):
    """APL learns its print through the same step, never through the guard: at 20 dB its denoise keeps the learned print."""
    print_json = tmp_path / "noise.np.json"
    print_json.write_text(json.dumps(PRINT), encoding="utf-8")
    with (
        patch.object(cathar, "CATHAR_NOISEPRINT_MIN_DEPTH_DB", 20.0),
        patch.object(spectral_denoise, "APL_USE_NATIVE_SUPPRESS", False),
        patch("modules.spectral_denoise._material", return_value=(1.8, 4.0)),
        patch("modules.spectral_denoise._cathar_available", return_value=True),
        patch("modules.cathar._cathar_noiseprint_step", return_value=print_json),
        patch("modules.cathar._cathar_denoise_step", return_value=tmp_path / "out.wav") as denoise,
        patch("modules.noiseprint_depth.guard_print", side_effect=AssertionError("guarded")),
    ):
        spectral_denoise._subtract(tmp_path / "in.wav", tmp_path, 10.0)
    assert denoise.call_args.kwargs["noiseprint_path"] == print_json
