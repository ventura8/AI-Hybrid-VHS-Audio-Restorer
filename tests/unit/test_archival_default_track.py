"""With the original track kept for the archive, the restored track is the one a player starts on.

ffmpeg carries the source stream's default flag over to the archival copy and leaves the
restored track (a filter output) unflagged, so before 2026-10-17 MP4 and MOV opened on the
unrestored tape audio. The mux now flags the restored track default and clears the archival one.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import modules.config as cfg
import modules.processing as processing
from modules import mastering, utils

RATE = 48000
AUDIO_ARGS = ["-c:a", "aac", "-b:a", "320k"]
DISPOSITION_FLAG = "-disposition:a:0"
# The containers whose muxer stores a default flag, and those that store none (a player takes the first stream).
FLAGGED_CONTAINERS = {
    ".mp4": (["-c:v", "mpeg4"], ["-c:a", "aac", "-b:a", "192k"]),
    ".mov": (["-c:v", "mpeg4"], ["-c:a", "aac", "-b:a", "192k"]),
    ".mkv": (["-c:v", "mpeg4"], ["-c:a", "aac", "-b:a", "192k"]),
}


def _paths(tmp_path):
    return tmp_path / "video.mp4", tmp_path / "a1.wav", tmp_path / "a2.wav", tmp_path / "out.mp4"


def _pairs(command):
    return list(zip(command, command[1:]))


def test_the_archival_arguments_flag_the_restored_track_default_and_clear_the_archival_one():
    """The helper maps the source's first audio stream and ends with both disposition flags."""
    args = mastering._archival_track_args(".mp4", AUDIO_ARGS)
    assert args[:2] == ["-map", "0:a:0?"]
    assert args[-4:] == list(mastering.ARCHIVAL_DISPOSITION_ARGS)


def test_the_disposition_names_the_restored_stream_first():
    """Audio stream 0 (the restored track) is flagged default, stream 1 (the archival copy) cleared."""
    pairs = _pairs(list(mastering.ARCHIVAL_DISPOSITION_ARGS))
    assert (DISPOSITION_FLAG, "default") in pairs
    assert ("-disposition:a:1", "0") in pairs


def test_the_single_track_mux_flags_the_restored_track_when_the_archive_is_kept(tmp_path):
    """The single-track mux command carries the disposition flags with preservation on."""
    video, audio, _unused, out = _paths(tmp_path)
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(processing, "PRESERVE_ORIGINAL_AUDIO_TRACK", True)
        command = processing._build_single_audio_mux_command(video, audio, out, AUDIO_ARGS, 4)
    assert (DISPOSITION_FLAG, "default") in _pairs(command)


def test_the_two_stem_mix_flags_the_restored_track_when_the_archive_is_kept(tmp_path):
    """The two-stem mix command carries the disposition flags with preservation on."""
    video, vocals, background, out = _paths(tmp_path)
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(processing, "PRESERVE_ORIGINAL_AUDIO_TRACK", True)
        command = processing._final_mix_output_command(video, vocals, background, out, AUDIO_ARGS, 4)
    assert (DISPOSITION_FLAG, "default") in _pairs(command)


def test_without_the_archive_no_disposition_is_written(tmp_path):
    """Preservation off (the default) leaves the single-track command as it was: no disposition flag."""
    video, audio, _unused, out = _paths(tmp_path)
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(processing, "PRESERVE_ORIGINAL_AUDIO_TRACK", False)
        command = processing._build_single_audio_mux_command(video, audio, out, AUDIO_ARGS, 4)
    assert DISPOSITION_FLAG not in command


def _available(binary):
    return shutil.which(binary) is not None or Path(binary).is_file()


needs_ffmpeg = pytest.mark.skipif(
    not (_available(utils.FFMPEG_BIN) and _available(utils.FFPROBE_BIN)), reason="needs the ffmpeg and ffprobe binaries"
)


def _source(folder, suffix):
    """A 2 s synthetic source: test-pattern video and a 440 Hz tone, its audio flagged default by the muxer."""
    video_args, audio_args = FLAGGED_CONTAINERS[suffix]
    t = np.arange(int(2.0 * RATE)) / RATE
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    wav = folder / "tone.wav"
    sf.write(str(wav), np.stack([tone, tone], axis=1).astype(np.float32), RATE, subtype="FLOAT")
    source = folder / f"source{suffix}"
    lavfi = ["-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=2"]
    command = [utils.FFMPEG_BIN, "-v", "error", "-y", *lavfi, "-i", str(wav), "-map", "0:v", "-map", "1:a", *video_args, *audio_args]
    subprocess.run([*command, str(source)], check=True, capture_output=True)
    return source


def _audio_defaults(path):
    """The default flag of each audio stream, in output order."""
    out = subprocess.run(
        [utils.FFPROBE_BIN, "-v", "error", "-select_streams", "a", "-show_entries", "stream_disposition=default", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        encoding="utf-8",
    ).stdout
    return [stream["disposition"]["default"] for stream in json.loads(out)["streams"]]


@needs_ffmpeg
@pytest.mark.parametrize("suffix", sorted(FLAGGED_CONTAINERS))
def test_a_real_mux_with_the_archive_starts_on_the_restored_track(tmp_path, monkeypatch, suffix):
    """Through the app's own single-track mux: two audio streams, the restored first and default, the archival copy not."""
    monkeypatch.setattr(cfg, "ENABLE_LOUDNORM", False)
    monkeypatch.setattr(processing, "PRESERVE_ORIGINAL_AUDIO_TRACK", True)
    monkeypatch.delenv("AI_RESTORE_EVENT_LOG", raising=False)
    source = _source(tmp_path, suffix)
    restored = tmp_path / "restored.wav"
    processing._extract_audio_step(source, restored, total_duration=2.0)
    final = tmp_path / f"source_Restored{suffix}"
    processing._final_mux_single_audio_step(source, restored, final, total_duration=2.0)
    assert _audio_defaults(final) == [1, 0]
