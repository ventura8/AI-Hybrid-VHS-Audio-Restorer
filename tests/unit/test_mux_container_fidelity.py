"""The mux on real files: the restored track keeps the source's rate and its place against the video.

Each case builds a 2 s synthetic source with the installed ffmpeg (320x240 video and 48 kHz
stereo audio whose first sample sits 300 ms after the first video frame, with a click 0.8 s
into the audio), extracts its audio the way the app does, hands that WAV to the mux as the
restored track, and finds the click in the output against the first video frame (each
stream's start time plus the click's sample index in it). AVI cannot carry a start offset
(ffmpeg's muxer writes none), so its source starts with its video and the check holds it there.

On bb5fa2a the click landed 273.7-295.0 ms early in mp4, mov, mkv and mpg (the offset was
dropped) and every output was 44.1 kHz. The limiter's 219-sample lookahead (4.97 ms at
44.1 kHz, 239 samples at 48 kHz) is still in the default render; with
`mux_limiter_latency_compensation` the click sits where the source's did.
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
from modules import utils

RATE = 48000
OFFSET_S = 0.3
CLICK_S = 0.8
TOLERANCE_MS = 1.0
CONTAINERS = {
    ".mp4": (["-c:v", "mpeg4"], ["-c:a", "aac", "-b:a", "192k"]),
    ".mpg": (["-c:v", "mpeg2video"], ["-c:a", "mp2", "-b:a", "384k"]),
    ".mov": (["-c:v", "mpeg4"], ["-c:a", "aac", "-b:a", "192k"]),
    ".mkv": (["-c:v", "mpeg4"], ["-c:a", "aac", "-b:a", "192k"]),
    ".avi": (["-c:v", "mpeg4"], ["-c:a", "pcm_s16le"]),
}
# alimiter's lookahead at the rate each output is rendered at (modules/mastering.py).
LIMITER_DELAY_MS = {44100: 1000 * 219 / 44100, 48000: 1000 * 239 / 48000}


def _available(binary):
    return shutil.which(binary) is not None or Path(binary).is_file()


pytestmark = pytest.mark.skipif(
    not (_available(utils.FFMPEG_BIN) and _available(utils.FFPROBE_BIN)), reason="needs the ffmpeg and ffprobe binaries"
)


def _run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def _source(folder, suffix):
    """The synthetic source: video, then 48 kHz audio shifted 300 ms later with a click 0.8 s into it."""
    t = np.arange(int(2.0 * RATE)) / RATE
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    tone[int(CLICK_S * RATE)] = 0.9
    wav = folder / "tone.wav"
    sf.write(str(wav), np.stack([tone, tone], axis=1).astype(np.float32), RATE, subtype="FLOAT")
    video_args, audio_args = CONTAINERS[suffix]
    source = folder / f"source{suffix}"
    lavfi = ["-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2.4"]
    shifted = ["-itsoffset", str(OFFSET_S), "-i", str(wav)]
    _run([utils.FFMPEG_BIN, "-v", "error", "-y", *lavfi, *shifted, "-map", "0:v", "-map", "1:a", *video_args, *audio_args, str(source)])
    return source


def _streams(path):
    out = subprocess.run(
        [utils.FFPROBE_BIN, "-v", "error", "-show_entries", "stream=index,codec_type,sample_rate,start_time", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        encoding="utf-8",
    ).stdout
    streams = json.loads(out)["streams"]
    return next(s for s in streams if s["codec_type"] == "video"), next(s for s in streams if s["codec_type"] == "audio")


def _click_ms(path):
    """(the click's time against the first video frame in ms, the audio's rate)."""
    video, audio = _streams(path)
    decoded = path.with_suffix(".decoded.wav")
    _run([utils.FFMPEG_BIN, "-v", "error", "-y", "-i", str(path), "-map", f"0:{audio['index']}", "-c:a", "pcm_f32le", str(decoded)])
    data, rate = sf.read(str(decoded), dtype="float64", always_2d=True)
    at = float(audio["start_time"]) + int(np.argmax(np.abs(data[:, 0]))) / rate
    return 1000 * (at - float(video["start_time"])), int(audio["sample_rate"])


def _restored(tmp_path, suffix, monkeypatch, rate_setting, compensate):
    """(source click ms, output click ms, output rate) for one container through the single-track mux."""
    monkeypatch.setattr(cfg, "ENABLE_LOUDNORM", True)
    monkeypatch.setattr(cfg, "MUX_SAMPLE_RATE", rate_setting)
    monkeypatch.setattr(cfg, "MUX_LIMITER_LATENCY_COMPENSATION", compensate)
    monkeypatch.setattr(processing, "PRESERVE_ORIGINAL_AUDIO_TRACK", False)
    monkeypatch.delenv("AI_RESTORE_EVENT_LOG", raising=False)
    source = _source(tmp_path, suffix)
    original = tmp_path / "original.wav"
    processing._extract_audio_step(source, original, total_duration=2.0)
    final = tmp_path / f"source_Restored{suffix}"
    processing._final_mux_single_audio_step(source, original, final, total_duration=2.0)
    source_click, _source_rate = _click_ms(source)
    output_click, output_rate = _click_ms(final)
    return source_click, output_click, output_rate


@pytest.mark.parametrize("suffix", sorted(CONTAINERS))
def test_by_default_the_track_keeps_its_place_but_for_the_limiter(tmp_path, monkeypatch, suffix):
    """Defaults: MPEG-PS keeps 48 kHz, the rest 44.1 kHz; the click is late by the limiter's lookahead alone."""
    source_click, output_click, rate = _restored(tmp_path, suffix, monkeypatch, "pipeline", False)
    assert rate == (48000 if suffix == ".mpg" else 44100)
    assert output_click - source_click == pytest.approx(LIMITER_DELAY_MS[rate], abs=TOLERANCE_MS)


@pytest.mark.parametrize("suffix", sorted(CONTAINERS))
def test_with_both_keys_the_track_keeps_its_rate_and_its_place(tmp_path, monkeypatch, suffix):
    """mux_sample_rate source and the limiter's compensation: 48 kHz everywhere, the click within 1 ms of the source's."""
    source_click, output_click, rate = _restored(tmp_path, suffix, monkeypatch, "source", True)
    assert rate == RATE
    assert output_click == pytest.approx(source_click, abs=TOLERANCE_MS)


def test_the_source_fixture_carries_its_offset(tmp_path):
    """The check measures something: an MP4 source's click sits 300 ms later than in its audio file."""
    click, rate = _click_ms(_source(tmp_path, ".mp4"))
    assert rate == RATE
    assert click == pytest.approx(1000 * (OFFSET_S + CLICK_S), abs=TOLERANCE_MS)
