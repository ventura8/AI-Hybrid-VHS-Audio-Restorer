"""The mux plan (modules/mux_format.py): the source audio's start time and the rate each container keeps."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from modules import mux_format
from modules.mux_format import MuxPlan, SourceAudio

AT_48K_LATE = SourceAudio(48000, 0.300000, first_audio_start=0.300000)


def _probe_json(*streams, format_name="mov,mp4,m4a,3gp,3g2,mj2"):
    return json.dumps({"streams": list(streams), "format": {"format_name": format_name}})


def _stream(rate="48000", start="0.300000", channels=2, default=0):
    return {"codec_type": "audio", "sample_rate": rate, "start_time": start, "channels": channels, "disposition": {"default": default}}


def _video(start="0.000000"):
    return {"codec_type": "video", "start_time": start, "disposition": {"default": 1}}


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
    assert mux_format.parse_probe(_probe_json(_video())) is None
    assert mux_format.parse_probe("{}") is None


@pytest.mark.parametrize(("format_name", "rebases"), [("mpeg", True), ("mpegts", True), ("matroska,webm", False), ("avi", False)])
def test_the_probe_reads_the_first_video_start_and_the_demuxer(format_name, rebases):
    probe = _probe_json(_video("0.540000"), _stream(start="0.649978"), _video("9.000000"), format_name=format_name)
    assert mux_format.parse_probe(probe) == SourceAudio(48000, 0.649978, 0.54, rebases, 0.649978)


def test_the_probe_keeps_the_first_audio_start_for_the_archival_track():
    """The archival track maps the first audio stream, which need not be the one the extraction picks."""
    probe = _probe_json(_stream(rate="32000", start="0.400000", channels=1), _stream(start="0.500000", channels=6), format_name="mpeg")
    assert mux_format.parse_probe(probe) == SourceAudio(48000, 0.5, None, True, 0.4)


def test_a_probe_without_a_format_does_not_rebase():
    assert mux_format.parse_probe(json.dumps({"streams": [_stream()]})).rebases_video is False


def test_mpeg_ps_and_ts_place_the_track_against_the_video_without_copyts():
    """Their demuxer corrects timestamp jumps only without -copyts; ffmpeg starts the input at the copied video."""
    plan = MuxPlan(SourceAudio(48000, 0.649978, 0.54, True, 0.649978), 48000)
    assert plan.global_args() == []
    assert plan.restored_input("a.wav") == ["-itsoffset", "0.109978", "-i", "a.wav"]


@pytest.mark.parametrize(("audio_start", "shift"), [(0.21, "0.000000"), (0.649978, "0.109978")])
def test_with_the_archival_track_an_mpeg_input_starts_at_its_earliest_mapped_stream(audio_start, shift):
    """Audio leading the video moves the input's start to the audio; audio after it leaves the video's."""
    plan = MuxPlan(SourceAudio(48000, audio_start, 0.54, True, audio_start), 48000)
    assert plan.restored_input("a.wav", archival=True) == ["-itsoffset", shift, "-i", "a.wav"]


@pytest.mark.parametrize(
    ("source", "archival"),
    [
        (SourceAudio(48000, 0.649978, None, True, 0.649978), False),
        (SourceAudio(48000, None, 0.54, True, None), False),
        (SourceAudio(48000, 0.649978, 0.54, True, None), True),
    ],
)
def test_mpeg_ps_and_ts_with_an_unread_start_run_todays_command(source, archival):
    plan = MuxPlan(source, 48000)
    assert (plan.global_args(), plan.restored_input("a.wav", archival)) == ([], ["-i", "a.wav"])


@pytest.mark.parametrize("archival", [False, True])
def test_other_containers_keep_copyts_whatever_the_video_start(archival):
    plan = MuxPlan(SourceAudio(48000, 0.3, 0.1, False, 0.0))
    assert (plan.global_args(), plan.restored_input("a.wav", archival)[:2]) == (["-copyts"], ["-itsoffset", "0.300000"])


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


def _probed(monkeypatch):
    """`probe_source_audio("tape.mpg")` against a stand-in ffprobe: its result, and the command and keywords it ran."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout=_probe_json(_stream()), returncode=0)

    monkeypatch.setattr(mux_format.subprocess, "run", fake_run)
    result = mux_format.probe_source_audio("tape.mpg")
    cmd, kwargs = calls[0]
    return result, cmd, kwargs


def test_probe_source_audio_asks_ffprobe_for_the_audio_streams(monkeypatch):
    result, cmd, kwargs = _probed(monkeypatch)
    assert result == AT_48K_LATE
    assert cmd[0] == mux_format.FFPROBE_BIN
    assert cmd[-1] == "tape.mpg"
    assert (kwargs["encoding"], kwargs["errors"]) == ("utf-8", "replace")


def test_probe_source_audio_asks_for_the_stream_kind_its_start_and_the_container(monkeypatch):
    _result, cmd, _kwargs = _probed(monkeypatch)
    entries = cmd[cmd.index("-show_entries") + 1]
    assert "codec_type" in entries
    assert "start_time" in entries
    assert "format=format_name" in entries


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
