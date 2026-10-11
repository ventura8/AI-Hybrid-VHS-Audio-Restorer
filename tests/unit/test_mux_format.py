"""The mux plan (modules/mux_format.py): the source audio's start time and the rate each container keeps."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from modules import mux_format
from modules.mux_format import MuxPlan, SourceAudio

AT_48K_LATE = SourceAudio(48000, 0.300000)


def _probe_json(*streams):
    return json.dumps({"streams": list(streams)})


def _stream(rate="48000", start="0.300000", channels=2, default=0):
    return {"sample_rate": rate, "start_time": start, "channels": channels, "disposition": {"default": default}}


def test_the_default_plan_is_todays_command():
    plan = MuxPlan()
    assert (plan.global_args(), plan.restored_input("a.wav"), plan.output_args()) == ([], ["-i", "a.wav"], [])


def test_a_known_start_keeps_input_timestamps_and_shifts_the_restored_input():
    plan = MuxPlan(AT_48K_LATE, 44100)
    assert plan.global_args() == ["-copyts"]
    assert plan.restored_input("a.wav") == ["-itsoffset", "0.300000", "-i", "a.wav"]


def test_a_negative_start_shifts_back():
    assert MuxPlan(SourceAudio(48000, -0.021333)).restored_input("a.wav")[:2] == ["-itsoffset", "-0.021333"]


def test_an_unread_start_leaves_the_input_alone():
    plan = MuxPlan(SourceAudio(48000, None), 48000)
    assert (plan.global_args(), plan.restored_input("a.wav")) == ([], ["-i", "a.wav"])


def test_a_rate_other_than_the_pipelines_is_set_on_the_restored_stream():
    assert MuxPlan(AT_48K_LATE, 48000).output_args() == ["-ar:a:0", "48000"]


def test_the_probe_reads_rate_and_start():
    assert mux_format.parse_probe(_probe_json(_stream())) == AT_48K_LATE


def test_no_audio_stream_reads_as_no_source():
    assert mux_format.parse_probe(_probe_json()) is None
    assert mux_format.parse_probe("{}") is None


@pytest.mark.parametrize("start", ["N/A", None, "nan", "inf"])
def test_an_unreadable_start_is_none(start):
    assert mux_format.parse_probe(_probe_json(_stream(start=start))).start_time is None


def test_the_pick_follows_ffmpeg_most_channels_first_on_a_tie():
    picked = mux_format.parse_probe(
        _probe_json(_stream(rate="32000", channels=2), _stream(rate="48000", channels=6), _stream(rate="44100", channels=6))
    )
    assert picked.sample_rate == 48000


def test_a_default_stream_outranks_any_channel_count():
    picked = mux_format.parse_probe(_probe_json(_stream(rate="48000", channels=6), _stream(rate="32000", channels=1, default=1)))
    assert picked.sample_rate == 32000


def test_probe_source_audio_asks_ffprobe_for_the_audio_streams(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout=_probe_json(_stream()), returncode=0)

    monkeypatch.setattr(mux_format.subprocess, "run", fake_run)
    assert mux_format.probe_source_audio("tape.mpg") == AT_48K_LATE
    cmd, kwargs = calls[0]
    assert cmd[0] == mux_format.FFPROBE_BIN and cmd[-1] == "tape.mpg" and "a" in cmd
    assert (kwargs["encoding"], kwargs["errors"]) == ("utf-8", "replace")


@pytest.mark.parametrize(
    "outcome",
    [OSError("no ffprobe"), subprocess.TimeoutExpired("ffprobe", 30), SimpleNamespace(stdout="not json"), SimpleNamespace(stdout=None)],
)
def test_an_unreadable_probe_is_no_source(monkeypatch, outcome):
    def fake_run(_cmd, **_kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(mux_format.subprocess, "run", fake_run)
    assert mux_format.probe_source_audio("tape.mpg") is None


@pytest.mark.parametrize(
    ("suffix", "rate", "expected"),
    [
        (".mpg", 48000, 48000),
        (".MPEG", 44100, 44100),
        (".mpg", 32000, 32000),
        (".mpg", 22050, 22050),
        (".mpg", 96000, 48000),
        (".mpg", None, 44100),
        (".mp4", 48000, 44100),
        (".mkv", 48000, 44100),
        (".avi", 32000, 44100),
    ],
)
def test_the_pipeline_setting_keeps_44k1_but_in_mpeg_ps(monkeypatch, suffix, rate, expected):
    monkeypatch.setattr(mux_format.config, "MUX_SAMPLE_RATE", "pipeline")
    assert mux_format.encoded_sample_rate(suffix, SourceAudio(rate, 0.0)) == expected


@pytest.mark.parametrize(
    ("suffix", "rate", "expected"),
    [
        (".mp4", 48000, 48000),
        (".ts", 96000, 96000),
        (".mp4", 44056, 44100),
        (".mkv", 44056, 44056),
        (".mov", 48000, 48000),
        (".avi", 32000, 32000),
        (".mpg", 96000, 48000),
        (".mp4", None, 44100),
    ],
)
def test_the_source_setting_keeps_every_rate_its_codec_encodes(monkeypatch, suffix, rate, expected):
    monkeypatch.setattr(mux_format.config, "MUX_SAMPLE_RATE", "source")
    assert mux_format.encoded_sample_rate(suffix, SourceAudio(rate, 0.0)) == expected


def test_no_source_keeps_the_pipeline_rate(monkeypatch):
    monkeypatch.setattr(mux_format.config, "MUX_SAMPLE_RATE", "source")
    assert mux_format.encoded_sample_rate(".mpg", None) == 44100


def test_plan_for_combines_the_probe_and_the_container(monkeypatch):
    monkeypatch.setattr(mux_format, "probe_source_audio", lambda path: AT_48K_LATE)
    monkeypatch.setattr(mux_format.config, "MUX_SAMPLE_RATE", "pipeline")
    assert mux_format.plan_for("tape.mpg", ".mpg") == MuxPlan(AT_48K_LATE, 48000)
    assert mux_format.plan_for("tape.mp4", ".mp4") == MuxPlan(AT_48K_LATE, 44100)
