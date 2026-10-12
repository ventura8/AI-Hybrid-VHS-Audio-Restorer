"""What the final mux keeps of the source's audio: where it starts against the video, and its sample rate.

Every stage runs at 44.1 kHz (processing.PIPELINE_SAMPLE_RATE): the extraction resamples the
source to it, and the restored WAV's first sample is the source audio's first decoded sample.
The mux used to hand that WAV to ffmpeg as a stream starting at zero and to encode it at
44.1 kHz in every container. Measured on bb5fa2a (2026-10-11, ffmpeg 9.0.1) through the
single-track mux on 3 s synthetic sources (320x240 video, 48 kHz stereo audio starting 300 ms
after the video, 300 ms before it, or with it, a click 1.000 s into the audio), the click
landed against the first video frame:

- mp4 and mov with the audio 300 ms late: 273.7 ms early (the source's start, 278.667 ms
  once AAC priming is counted, was dropped); mkv: 295.0 ms early; mpg (MPEG-PS, MP2): 285.0
  ms early, and 315.0 ms late with the audio 300 ms early, and 15.0 ms late with no offset
  (MP2's 481-sample decoder delay, 10.02 ms at 48 kHz, re-based to zero). Every case was
  also 4.97 ms late from the limiter (`mux_limiter_latency_compensation`, modules/mastering.py).
  An AVI carries no start offset (ffmpeg's muxer writes none), so its audio starts with its
  video either way.
- Every output was 44.1 kHz, MP2 in MPEG-PS included, which DVD-Video allows only at 48 kHz.
  The capture E:/Test/2015_12_17_10_56_3.mpg (MP2 at 48 kHz, audio 5.578 ms after the video)
  came out of the 16 Sep renders as 44.1 kHz MP2 with its audio 10.911 ms before the video.

Placing the restored WAV at the source audio's own start time (`-itsoffset` on that input)
with ffmpeg's input timestamps kept (`-copyts`) puts the click within 0.002 ms of the
source's in all five containers and all three offsets. Without `-copyts` ffmpeg re-bases an
MPEG-PS or MPEG-TS input to the earliest stream the command maps from it, while an MP4, MOV or
matroska input goes to its first stream of any kind, so no single offset fits every container
that way. With the source's start at zero (every cathar identity clip) the shift is zero and
the decoded audio is unchanged.

MPEG-PS and MPEG-TS take no `-copyts`: their demuxer corrects a timestamp jump (two captures
joined byte-wise, a recorder's pause, a broadcast splice) only while timestamps are not
copied. Measured (2026-10-12, ffmpeg 9.0.1) on sources of two 6 s halves joined byte-wise,
each with a flash frame at 1 s and 4 s and a click 120 ms after each, read the way a player
built on ffmpeg plays them: with `-copyts` the second half's copied video timestamps piled up
at the jump (all near 6.46 s), and its two clicks moved against their flashes by -254 and
-620 ms in MPEG-PS, +84 and +83 ms in MPEG-TS with MPEG-2 video, +124 and +123 ms with H.264
(the WAV mapped directly); through the full single-track mux with the limiter compensated,
+336 and -620 ms in MPEG-PS, +84 and +83 ms in MPEG-TS.

Without `-copyts` ffmpeg re-bases such an input to the earliest stream the command maps from
it, not to the file's first stream: the copied video alone, or the video and the first audio
stream when the archival track is kept. The restored input therefore goes to the audio's
start against that earliest start (`-itsoffset <audio start - mapped start>`). That held all
four joined clicks within 0.01 ms of the source's; on continuous sources with the audio
300 ms late, 330 ms early or with the video (the MP2 or AAC priming its only lead) it sat
within 0.005 ms, with and without the archival track, where placing it against the video alone
missed by the audio's lead (330.00 and 10.00 ms in MPEG-PS, 341.00 and 21.00 ms in MPEG-TS)
once the archival track was kept. A real MPEG-PS capture cut whose audio led its video by
554.5 ms sat +4.978 ms from the source (the limiter), -0.001 ms with the limiter compensated.
An unreadable start leaves these containers on the command every earlier release ran. A
source cut byte-wise mid-GOP read the same under both placements.

The rate: an MPEG-PS output keeps its source's rate when MP2 can encode it (32, 44.1 and
48 kHz and the half rates; any other goes to 48 kHz, an unreadable one stays at 44.1 kHz);
the cathar identity clips are MP4 and never take this path. `mux_sample_rate: source`
extends it to every container whose codec takes the rate, which changes the bytes of every
48 kHz source (three of the five identity clips), so its default, 'pipeline', keeps 44.1 kHz.
"""

import json
import math
import subprocess
from dataclasses import dataclass

from . import config
from .mastering import PIPELINE_SAMPLE_RATE, _get_audio_encoding_args
from .utils import FFPROBE_BIN

MUX_RATE_PIPELINE = "pipeline"
MUX_RATE_SOURCE = "source"
MPEG_PS_SUFFIXES = frozenset({".mpg", ".mpeg"})
MPEG_PS_DEFAULT_RATE = 48000
# What each lossy codec of AUDIO_CODEC_ARGS_BY_EXT encodes (ffmpeg's encoders); PCM takes any rate.
MP2_SAMPLE_RATES = frozenset({16000, 22050, 24000, 32000, 44100, 48000})
AAC_SAMPLE_RATES = frozenset({7350, 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000})
CODEC_SAMPLE_RATES = {"mp2": MP2_SAMPLE_RATES, "aac": AAC_SAMPLE_RATES}
PROBE_TIMEOUT_S = 30
# ffmpeg's own pick of an input's audio when no -map names one (the extraction runs that way):
# the highest channel count, a default-disposition stream outranking any count, the first on a tie.
DEFAULT_DISPOSITION_WEIGHT = 5_000_000
# ffprobe's demuxer names for MPEG-PS and MPEG-TS: timestamp jumps corrected only without -copyts,
# and the input re-based to the earliest stream the command maps (the module docstring has the measurements).
REBASING_DEMUXERS = frozenset({"mpeg", "mpegts"})


@dataclass(frozen=True)
class SourceAudio:
    """The source audio stream the extraction decodes: its rate in Hz and start time in seconds, each None when unreadable.

    `video_start` is the first video stream's start (the one the mux copies), `first_audio_start`
    the first audio stream's (the one the archival track maps), and `rebases_video` is True for
    an MPEG-PS or MPEG-TS source (REBASING_DEMUXERS).
    """

    sample_rate: int | None = None
    start_time: float | None = None
    video_start: float | None = None
    rebases_video: bool = False
    first_audio_start: float | None = None


def _mapped_start(source, archival):
    """Where ffmpeg starts an MPEG-PS/TS input without -copyts: its earliest mapped stream, None when one is unreadable."""
    starts = (source.video_start, source.first_audio_start) if archival else (source.video_start,)
    return None if None in starts else min(starts)


def _rebased_shift(source, archival):
    """The audio's start against the input's mapped start, or None when either is unreadable."""
    mapped = _mapped_start(source, archival)
    if source.start_time is None or mapped is None:
        return None
    return source.start_time - mapped


@dataclass(frozen=True)
class MuxPlan:
    """How the mux places and encodes the restored track; the default plan is the command every release has run."""

    source: SourceAudio | None = None
    sample_rate: int = PIPELINE_SAMPLE_RATE

    def _shift(self, archival):
        """Where the restored input starts: the audio's own start, or in MPEG-PS/TS its start against the mapped start."""
        if self.source is None:
            return None
        return _rebased_shift(self.source, archival) if self.source.rebases_video else self.source.start_time

    def global_args(self):
        """ffmpeg keeps every input's timestamps, so the copied video and the shifted track keep the source's relation.

        Never in MPEG-PS or MPEG-TS, whose demuxer corrects timestamp jumps only without it.
        """
        copies = self.source is not None and not self.source.rebases_video and self.source.start_time is not None
        return ["-copyts"] if copies else []

    def restored_input(self, path, archival=False):
        """A restored WAV as an ffmpeg input, starting where the source's audio started.

        `archival` is True when the command also maps the source's first audio stream (the
        archival track), which moves an MPEG-PS/TS input's start when that stream leads the video.
        """
        start = self._shift(archival)
        shift = [] if start is None else ["-itsoffset", f"{start:.6f}"]
        return [*shift, "-i", str(path)]

    def output_args(self):
        """The restored stream's rate when it is not the pipeline's; the mastering graph resamples first, this covers loudnorm off."""
        return [] if self.sample_rate == PIPELINE_SAMPLE_RATE else ["-ar:a:0", str(self.sample_rate)]


def _number(raw, cast):
    """`raw` as a finite number of type `cast`, or None ('N/A', missing, NaN)."""
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _pick_score(stream):
    """ffmpeg's ranking of an audio stream for an input mapped without -map."""
    default = _number((stream.get("disposition") or {}).get("default"), int) or 0
    return (_number(stream.get("channels"), int) or 0) + DEFAULT_DISPOSITION_WEIGHT * default


def _streams_of(streams, codec_type):
    """The probed streams of one codec type, in the source's order."""
    return [stream for stream in streams if stream.get("codec_type") == codec_type]


def _rebases_video(probe):
    """True when ffprobe names the source's demuxer MPEG-PS or MPEG-TS."""
    names = str((probe.get("format") or {}).get("format_name") or "").split(",")
    return bool(REBASING_DEMUXERS.intersection(names))


def parse_probe(text):
    """The picked audio stream from ffprobe's JSON, or None when the source has no audio stream."""
    probe = json.loads(text)
    streams = probe.get("streams") or []
    audio = _streams_of(streams, "audio")
    if not audio:
        return None
    picked = max(audio, key=_pick_score)
    video = next(iter(_streams_of(streams, "video")), {})
    return SourceAudio(
        _number(picked.get("sample_rate"), int),
        _number(picked.get("start_time"), float),
        _number(video.get("start_time"), float),
        _rebases_video(probe),
        _number(audio[0].get("start_time"), float),
    )


def probe_source_audio(video_path):
    """The source audio's rate and start time, or None when ffprobe cannot read them (the mux then runs as before)."""
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,sample_rate,channels,start_time:stream_disposition=default:format=format_name",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=PROBE_TIMEOUT_S
        )
        return parse_probe(completed.stdout or "{}")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _keeps_source_rate(output_suffix):
    """MPEG-PS always keeps the source's rate; every other container only with `mux_sample_rate: source`."""
    return output_suffix.lower() in MPEG_PS_SUFFIXES or getattr(config, "MUX_SAMPLE_RATE") == MUX_RATE_SOURCE


def _encodable(rate, output_suffix):
    """True when the container's codec encodes `rate`."""
    allowed = CODEC_SAMPLE_RATES.get(_get_audio_encoding_args(output_suffix)[1])
    return allowed is None or rate in allowed


def _unencodable_rate(output_suffix):
    """The rate for a source rate the codec cannot take: DVD's 48 kHz for MPEG-PS, the pipeline's elsewhere."""
    return MPEG_PS_DEFAULT_RATE if output_suffix.lower() in MPEG_PS_SUFFIXES else PIPELINE_SAMPLE_RATE


def encoded_sample_rate(output_suffix, source):
    """The rate the restored track is encoded at in a container with this suffix."""
    rate = None if source is None else source.sample_rate
    if rate is None or not _keeps_source_rate(output_suffix):
        return PIPELINE_SAMPLE_RATE
    return rate if _encodable(rate, output_suffix) else _unencodable_rate(output_suffix)


def plan_for(video_path, output_suffix):
    """The mux plan for a source: its audio's start time and the rate its container keeps."""
    source = probe_source_audio(video_path)
    return MuxPlan(source, encoded_sample_rate(output_suffix, source))
