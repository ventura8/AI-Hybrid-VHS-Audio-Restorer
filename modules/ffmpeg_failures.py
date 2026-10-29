"""ffmpeg and ffprobe failures no retry can change, read from the tail of the failed command's output.

The CPU retry (utils.attempt_cpu_run_with_retry) halves the thread count after a failure, for
a run that ran out of memory. A failure the command itself decides retries to the same end:
in hunt v3 (2026-10-11) the extraction of everest-the-mountain-at-the-millenium-commerical,
a clip with no audio stream, failed with "Output file does not contain any stream" six times
(threads 32 down to 1) before the task gave up. A failed command whose output carries one of
these messages is raised at once. The messages are ffmpeg 9.0.1's for: an output with no
stream, a -map matching nothing, an input that is missing or unreadable, a filter, encoder or
option that does not exist, a filter option out of range. Anything else, an out-of-memory
abort included, keeps the retry.

An unreadable input is read from ffmpeg's "Error opening input" line, never from the bare
"Invalid data found when processing input": ffmpeg 9.0.1 prints that phrase for every broken
frame of a damaged MP2 capture ("Error submitting packet to decoder: Invalid data found when
processing input", on a run that exits 0), so a long capture that then ran out of memory
would have lost its retry. A memory message keeps the retry whatever else the tail holds
("Cannot allocate memory"; "Not enough space", the Windows C runtime's text for ENOMEM).
"""

DETERMINISTIC_MARKERS = (
    "does not contain any stream",
    "matches no streams",
    "no such file or directory",
    "error opening input",
    "no such filter",
    "filter not found",
    "encoder not found",
    "unknown encoder",
    "unrecognized option",
    "option not found",
    "error applying option",
)
MEMORY_MARKERS = ("cannot allocate memory", "not enough memory", "not enough space", "out of memory")


def _output_text(error):
    """The failed command's captured output as text; bytes are decoded leniently, a missing output reads as empty."""
    output = getattr(error, "output", None) or ""
    return output.decode("utf-8", "replace") if isinstance(output, bytes) else str(output)


def _mentions(text, markers):
    """True when the lower-cased `text` holds any of `markers`."""
    return any(marker in text for marker in markers)


def is_deterministic_failure(error):
    """True when a CalledProcessError's output names a failure that fewer threads cannot change and no lack of memory."""
    text = _output_text(error).lower()
    return _mentions(text, DETERMINISTIC_MARKERS) and not _mentions(text, MEMORY_MARKERS)
