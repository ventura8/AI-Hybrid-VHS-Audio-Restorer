"""The mux's limiter compensation and rate keys, and the plan reaching both mux commands.

Defaults must render every graph and command string for string as before: the limiter
without latency compensation, 44.1 kHz, no timestamp options when the source is unread.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import modules.config as cfg
import modules.processing as processing
from modules import mastering
from modules.mux_format import MuxPlan, SourceAudio
from scripts import tune_restoration as tune

LIMITER = "alimiter=limit=0.891:level=disabled"
TARGET = "I=-16.0:TP=-1.0:LRA=20.0"
PLAN = MuxPlan(SourceAudio(48000, 0.225578), 48000)


@pytest.fixture
def loudnorm_on(monkeypatch):
    monkeypatch.setattr(cfg, "ENABLE_LOUDNORM", True)
    monkeypatch.setattr(cfg, "LOUDNORM_TARGET_LRA", 20.0)


@pytest.mark.parametrize(("compensate", "expected"), [(False, LIMITER), (True, f"{LIMITER}:latency=1")])
def test_the_limiter_compensates_its_delay_only_when_asked(monkeypatch, compensate, expected):
    monkeypatch.setattr(cfg, "MUX_LIMITER_LATENCY_COMPENSATION", compensate)
    assert mastering._limiter_filter() == expected


def test_the_default_limiter_is_todays_filter():
    assert cfg.MUX_LIMITER_LATENCY_COMPENSATION is False
    assert mastering._limiter_filter() == mastering.LOUDNORM_TRUE_PEAK_LIMITER


@pytest.mark.parametrize(("compensate", "suffix"), [(False, ""), (True, ":latency=1")])
def test_both_mux_graphs_carry_the_limiter_setting(monkeypatch, loudnorm_on, compensate, suffix):
    monkeypatch.setattr(cfg, "MUX_LIMITER_LATENCY_COMPENSATION", compensate)
    single = mastering._build_single_audio_filter_expression(TARGET)
    mix = mastering._build_mix_filter_expression(loudnorm_args=TARGET)
    assert single == f"[1:a]loudnorm={TARGET},aresample=44100,{LIMITER}{suffix}[mastered]"
    assert mix.endswith(f",loudnorm={TARGET},aresample=44100,{LIMITER}{suffix}[mixed]")


def test_the_graphs_resample_to_the_mux_rate_before_the_limiter(loudnorm_on):
    single = mastering._build_single_audio_filter_expression(TARGET, sample_rate=48000)
    mix = mastering._build_mix_filter_expression(None, None, TARGET, sample_rate=48000)
    assert single == f"[1:a]loudnorm={TARGET},aresample=48000,{LIMITER}[mastered]"
    assert mix.endswith(f"aresample=48000,{LIMITER}[mixed]")


def test_the_fallback_record_renders_at_the_mux_rate(tmp_path, monkeypatch, loudnorm_on):
    monkeypatch.setenv("AI_RESTORE_EVENT_LOG", str(tmp_path / "events"))
    monkeypatch.setattr(cfg, "LOUDNORM_LINEAR_FALLBACK", "gain_limiter")
    track = tmp_path / ".temp_work_Tape" / "cathar" / "processed.wav"
    track.parent.mkdir(parents=True)
    measured = {"input_i": "-20.00", "input_tp": "-3.00", "input_lra": "6.00", "input_thresh": "-30.00", "target_offset": "0.10"}
    with (
        patch("modules.mastering._run_loudness_analysis", side_effect=[measured, None]) as run,
        patch("modules.mastering.log_msg"),
    ):
        applied = mastering._resolve_single_track_loudnorm_args("v.mpg", track, sample_rate=48000)
    assert applied == mastering.LinearGain(4.0)
    assert run.call_args_list[1].args[1].startswith(f"[1:a]volume=4.00dB,aresample=48000,{LIMITER},loudnorm=")


def test_the_mix_measurement_passes_the_rate_on(loudnorm_on):
    with patch("modules.mastering._resolve_measured_loudnorm", return_value=None) as resolve:
        mastering._resolve_loudnorm_args("v.mpg", "voc.wav", "bg.wav", None, None, total_duration=9.0, sample_rate=48000)
    assert resolve.call_args.kwargs == {"total_duration": 9.0, "sample_rate": 48000}


def test_the_single_track_command_places_and_rates_the_track():
    cmd = processing._build_single_audio_mux_command(
        Path("v.mpg"), Path("a.wav"), Path("o.mpg"), ["-c:a", "mp2"], 4, "[1:a]x[mastered]", PLAN
    )
    copyts, shift = cmd.index("-copyts"), cmd.index("-itsoffset")
    assert (cmd[copyts + 1], cmd[copyts + 2]) == ("-i", "v.mpg")
    assert (cmd[shift + 1], cmd[shift + 2], cmd[shift + 3]) == ("0.225578", "-i", "a.wav")
    assert cmd[cmd.index("-ar:a:0") + 1] == "48000"


def test_with_loudnorm_off_the_track_is_still_rated_by_the_plan():
    """No mastering graph: the WAV is mapped directly and the output option sets the container's rate."""
    cmd = processing._build_single_audio_mux_command(Path("v.mpg"), Path("a.wav"), Path("o.mpg"), ["-c:a", "mp2"], 4, None, PLAN)
    assert cmd[cmd.index("-map", cmd.index("0:v:0")) + 1] == "1:a:0"
    assert cmd[cmd.index("-ar:a:0") + 1] == "48000"


def test_the_mix_command_shifts_both_stems():
    cmd = processing._final_mix_output_command(
        Path("v.mpg"), Path("v.wav"), Path("b.wav"), Path("o.mpg"), ["-c:a", "mp2"], 4, "graph", PLAN
    )
    shifted = [cmd[i + 3] for i, arg in enumerate(cmd) if arg == "-itsoffset"]
    assert shifted == ["v.wav", "b.wav"]
    assert cmd.count("-copyts") == 1


@pytest.mark.parametrize(("archival", "shift"), [(False, "-0.330000"), (True, "0.000000")])
def test_both_commands_tell_an_mpeg_plan_whether_the_archival_track_is_mapped(monkeypatch, archival, shift):
    """Audio leading the video by 330 ms in MPEG-PS: no -copyts, and the shift follows the streams the command maps."""
    monkeypatch.setattr(processing, "PRESERVE_ORIGINAL_AUDIO_TRACK", archival)
    plan = MuxPlan(SourceAudio(48000, 0.21, 0.54, True, 0.21), 48000)
    single = processing._build_single_audio_mux_command(Path("v.mpg"), Path("a.wav"), Path("o.mpg"), ["-c:a", "mp2"], 4, None, plan)
    mix = processing._final_mix_output_command(Path("v.mpg"), Path("v.wav"), Path("b.wav"), Path("o.mpg"), ["-c:a", "mp2"], 4, "g", plan)
    for cmd in (single, mix):
        assert "-copyts" not in cmd
        assert {cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-itsoffset"} == {shift}


def test_without_a_plan_both_commands_are_todays():
    single = processing._build_single_audio_mux_command(Path("v.mp4"), Path("a.wav"), Path("o.mp4"), ["-c:a", "aac"], 4, None)
    mix = processing._final_mix_output_command(Path("v.mp4"), Path("v.wav"), Path("b.wav"), Path("o.mp4"), ["-c:a", "aac"], 4, "g")
    for cmd in (single, mix):
        assert not {"-copyts", "-itsoffset", "-ar:a:0"} & set(cmd)


def _mux_with_plan(tmp_path, step):
    """Runs one mux step with the plan, the measurement and the runner mocked; returns (plan_for mock, resolver kwargs, command)."""
    video, voc, bg = tmp_path / "tape.mpg", tmp_path / "voc.wav", tmp_path / "bg.wav"
    for path in (video, voc, bg):
        path.write_text("x")
    runner = MagicMock()
    with (
        patch.object(processing._mux_format, "plan_for", return_value=PLAN) as plan_for,
        patch.object(processing, "_resolve_single_track_loudnorm_args", return_value=None) as single,
        patch.object(processing, "_resolve_loudnorm_args", return_value=None) as mix,
        patch.object(processing, "attempt_cpu_run_with_retry", runner),
        patch.object(processing, "_promote_valid_output"),
        patch.object(processing, "is_valid_video", return_value=False),
    ):
        step(video, voc, bg, tmp_path / "tape_Out.mpg")
    resolver = single if single.called else mix
    return plan_for, resolver.call_args.kwargs, runner.call_args.args[0](2)


def test_the_single_track_step_probes_once_and_hands_the_plan_on(tmp_path):
    plan_for, kwargs, cmd = _mux_with_plan(tmp_path, lambda v, voc, _bg, out: processing._final_mux_single_audio_step(v, voc, out, 9.0))
    plan_for.assert_called_once_with(tmp_path / "tape.mpg", ".mpg")
    assert kwargs["sample_rate"] == 48000
    assert "-itsoffset" in cmd and "-copyts" in cmd


def test_the_mix_step_probes_once_and_hands_the_plan_on(tmp_path):
    plan_for, kwargs, cmd = _mux_with_plan(tmp_path, lambda v, voc, bg, out: processing._final_mix_step(v, voc, bg, out, 9.0))
    plan_for.assert_called_once_with(tmp_path / "tape.mpg", ".mpg")
    assert kwargs["sample_rate"] == 48000
    assert cmd.count("-itsoffset") == 2


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


def test_an_older_config_resolves_todays_mux_quietly():
    conf, warn = _loaded({"process_mode": "cathar"})
    assert (conf["mux_limiter_latency_compensation"], conf["mux_sample_rate"]) == (False, "pipeline")
    warn.assert_not_called()


@pytest.mark.parametrize(
    ("raw", "expected", "warned"),
    [(" Source ", "source", False), ("pipeline", "pipeline", False), ("48000", "pipeline", True), (None, "pipeline", True)],
)
def test_the_mux_rate_is_a_choice(raw, expected, warned):
    conf, warn = _loaded({"mux_sample_rate": raw})
    assert conf["mux_sample_rate"] == expected
    assert warn.called is warned


@pytest.mark.parametrize(("raw", "expected"), [(True, True), ("yes", True), ("off", False)])
def test_the_limiter_compensation_is_a_switch(raw, expected):
    conf, _warn = _loaded({"mux_limiter_latency_compensation": raw})
    assert conf["mux_limiter_latency_compensation"] is expected


def test_the_repository_config_keeps_both_defaults():
    text = (Path(cfg.__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
    assert "\nmux_limiter_latency_compensation: false\n" in text
    assert '\nmux_sample_rate: "pipeline"\n' in text
    assert (cfg.MUX_LIMITER_LATENCY_COMPENSATION, cfg.MUX_SAMPLE_RATE) == (False, "pipeline")


def test_the_tuning_driver_knows_both_keys():
    fields = tune.known_config_fields()
    assert fields["mux_limiter_latency_compensation"][0] is bool
    assert fields["mux_sample_rate"][0] == {"pipeline", "source"}
    assert tune._problem("mux_sample_rate", "48000", fields["mux_sample_rate"])
