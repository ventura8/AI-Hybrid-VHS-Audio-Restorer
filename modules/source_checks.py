"""Sources that cannot be restored for a reason no retry changes: a logged skip, never a crash or a retry loop.

Hunt v3 (2026-10-11, 240 IA clips) held four of them. everest-the-mountain-at-the-millenium-
commerical has no audio stream: the extraction failed six times on fewer and fewer threads
before the task failed with the last ffmpeg command line as its message. Pooh, Powerpuff and
Rocky decode to nothing (Pooh's AAC packets all sit before its edit list's start, skip_samples
1939727), so ffmpeg exits 0 with a 92-byte WAV header and no samples, and the task failed as
"Output audio is invalid, empty, or too small". Each now ends at the extraction with a
`[Skip]` line naming the reason, and the restoration returns False.
"""

try:
    import soundfile as sf
except ImportError:
    sf = None

from .utils import _probe_stream_types

NO_AUDIO_STREAM = "the source has no audio stream; nothing to restore"
EMPTY_DECODE = "its audio stream decodes to no samples (an edit list or a cut that leaves nothing to play); nothing to restore"


class SourceSkipped(RuntimeError):
    """A source no retry can restore; the task logs a skip with this message and returns False."""


def require_audio_stream(video_path):
    """Raises SourceSkipped when ffprobe reads the source and finds no audio stream; an unreadable probe lets the extraction decide."""
    types = _probe_stream_types(video_path)
    if types and "audio" not in types:
        raise SourceSkipped(NO_AUDIO_STREAM)


def _frames(wav_path):
    """The WAV's sample-frame count, or None when it cannot be read as audio at all."""
    if sf is None:
        return None
    try:
        return sf.info(str(wav_path)).frames
    except (RuntimeError, OSError):
        # soundfile raises LibsndfileError (a RuntimeError) for a missing or unreadable file.
        return None


def reject_empty_decode(wav_path):
    """Removes an extraction that holds a readable header and no samples, and raises SourceSkipped for it."""
    if _frames(wav_path) != 0:
        return
    wav_path.unlink(missing_ok=True)
    raise SourceSkipped(EMPTY_DECODE)
