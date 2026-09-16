"""Duration-based chunking for the UVR neural denoiser.

The VR separator holds several full-length copies of the track in host memory once
inference is done, so its peak scales with duration and not with any batch setting. A
three-hour capture asks for one more 3.6 GiB array than a 62 GB machine has left and the
restoration fails outright, while a 2h11m capture on the same machine finishes.

The separator's working set was measured at 48 GB for a 94-minute chunk, about 30 GB per
hour of audio, so the chunk length is sized from the machine: half its memory at that rate,
between five minutes and two hours, unless `neural_chunk_seconds` fixes it. A 62 GB
desktop gets 57-minute chunks, a 30 GB laptop 28-minute ones. Captures up to the chunk
length are denoised whole, exactly as before. Longer ones are cut into equal chunks no
longer than that, each overlapping the next, denoised one at a time and rejoined with a
linear crossfade over the overlap. Both sides of a seam are the same audio denoised
twice, so they are correlated and a linear fade sums to unity; an equal-power fade would
leave a +3 dB bump at every seam (see `enhance_chunking`).

The model normalises its spectrogram by the file's maximum before inference, so a chunk on
its own is scaled by its own loudest moment rather than the whole file's, and the network
is not scale-invariant: measured on an 88-minute tape, that alone left the body of a chunk
33 dB below the signal from the whole-file result. Every chunk is therefore given the
file's loudest stride-aligned block as a lead-in, which the join drops; the chunk then sees
the same maximum the whole file did, and its output is the whole file's to within float
rounding. A chunk that already contains that block needs no anchor. Each chunk is also
cut with one stride of the true audio before and after it, dropped at the join, so its
edges are denoised in real context rather than against the model's reflection padding;
both sides of every seam are then the whole file's own samples, and the crossfade of two
identical signals is that signal. Measured on the 88-minute tape, the join is identical
to the whole-file result on 99.9% of samples with the rest at float rounding.

The separator also peak-limits every file it writes. Applied per chunk that would step the
level at a seam, so chunks are written with the limit at full scale, the highest the
library allows, and the whole-file rule -- scale to 0.9 when the peak is above it -- is
applied once to the joined result. A chunk that still came back at full scale was limited
by the separator on its own; it is denoised again at half scale, which the model is
indifferent to since it normalises its input, and the join doubles it back, so the rule
sees the chunk's true level. Only a chunk that limits pays for that: the retry costs the
fidelity of half-precision inference on a rescaled input, about -57 dB.

Nothing here holds the file in memory: the split reads ranges, the join streams blocks and
keeps only one overlap in hand.
"""

import math
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from .config import NEURAL_CHUNK_SECONDS
from .hygiene import atomic_target
from .utils import is_valid_audio, log_msg

# Long enough that a chunk's edge, where the model has no context, is faded to nothing before
# it matters; the model's own window is a few hundred milliseconds.
CHUNK_OVERLAP_SECONDS = 2.0
# The separator's host memory per hour of audio, measured on a 94-minute chunk (48 GB), and
# the share of the machine a chunk may claim; the rest is the other stages and the user.
HOST_GB_PER_HOUR = 32.0
HOST_SHARE = 0.5
# Measured in containers on an 88-minute tape: fifteen-minute chunks pass at 16 GB and are
# killed at 12 GB, so the floor is low enough for the rule to size a 12 GB machine below it.
MIN_CHUNK_SECONDS = 5 * 60
MAX_CHUNK_SECONDS = 2 * 3600
# What a machine is assumed to have when its memory cannot be read.
FALLBACK_HOST_GB = 32.0
BLOCK_FRAMES = 1 << 16
# The separator's normalization threshold on a whole file (see processing._build_separator)
# and the one chunks are written with.
WHOLE_FILE_PEAK = 0.9
CHUNK_PEAK = 1.0
CLEAN_STEM_TAG = "(No Noise)"
# The input scale a peak-limited chunk is denoised again at, undone by the join.
RETRY_GAIN = 0.5
# The separator trims its output to a multiple of its hop, so a chunk comes back a few
# hundred frames short of what it was given; that is padded with silence at the join. A
# shortfall past this is not the trim, it is a broken chunk.
MAX_SHORTFALL_SECONDS = 1.0
# Every chunk starts on a multiple of the model's patch stride, so its frames and patches
# fall on the grid the whole file's would and the recurrent layer sees the same windows.
# The shipped denoise models stride 192 spectrogram frames over a 1024-sample hop
# (UVR-DeNoise-Lite) or a 480-sample hop across four bands (UVR-DeNoise); this is the
# common multiple, 66.9 s at 44.1 kHz. Another model's grid is simply not matched.
PATCH_FRAMES = 192 * 15360


# Where a container's memory limit is published (cgroup v2, then v1). psutil reports the
# host's memory inside a container, and a chunk sized to the host is killed by the limit.
CGROUP_LIMIT_FILES = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")


def cgroup_memory_gb():
    """The memory limit a container runs under in GB, or None when there is none."""
    for path in CGROUP_LIMIT_FILES:
        try:
            value = Path(path).read_text(encoding="ascii").strip()
        except OSError:
            continue
        if value.isdigit() and int(value) < 2**60:
            return int(value) / 2**30
    return None


def host_memory_gb():
    """The memory available to this process in GB: the machine's, capped by a container limit."""
    try:
        import psutil

        total = psutil.virtual_memory().total / 2**30
    except (ImportError, OSError):
        total = FALLBACK_HOST_GB
    limit = cgroup_memory_gb()
    return total if limit is None else min(total, limit)


def chunk_seconds_for_host(total_gb=None):
    """The chunk length this machine can denoise in one pass, in whole minutes."""
    total_gb = host_memory_gb() if total_gb is None else total_gb
    seconds = total_gb * HOST_SHARE / HOST_GB_PER_HOUR * 3600
    return float(int(min(max(seconds, MIN_CHUNK_SECONDS), MAX_CHUNK_SECONDS)) // 60 * 60)


def resolved_chunk_seconds():
    """The configured chunk length, or the one sized for this machine when the setting is 0."""
    return NEURAL_CHUNK_SECONDS if NEURAL_CHUNK_SECONDS > 0 else chunk_seconds_for_host()


def plan(frames, rate, chunk_seconds=None, overlap_seconds=CHUNK_OVERLAP_SECONDS):
    """Frame ranges of equal chunks no longer than `chunk_seconds`, each overlapping the next.

    Returns an empty list when the file fits in one chunk or when the chunk length would
    not clear twice the overlap.
    """
    chunk_seconds = resolved_chunk_seconds() if chunk_seconds is None else chunk_seconds
    limit, overlap = int(chunk_seconds * rate), int(overlap_seconds * rate)
    if limit <= 2 * overlap or frames <= limit:
        return []
    core = math.ceil(frames / math.ceil(frames / (limit - overlap)))
    core = -(-core // PATCH_FRAMES) * PATCH_FRAMES
    return [(start, min(frames, start + core + overlap)) for start in range(0, frames, core)]


def duration_needs_chunking(source_wav, chunk_seconds=None):
    """Whether the file is longer than one chunk; an unreadable file is denoised whole as before."""
    try:
        info = sf.info(str(source_wav))
    except (OSError, RuntimeError):
        return False
    return bool(plan(info.frames, info.samplerate, chunk_seconds))


# The analysis the loudest block is found with: the shipped models' own frame, read a frame
# of context either side of a block so no frame is judged against zero padding.
ANCHOR_NFFT = 2048


def _block_peak(source, start, frames):
    """The largest spectrogram magnitude among the frames centred inside the block at `start`, mono."""
    nfft = min(ANCHOR_NFFT, frames) & ~1
    low = max(0, start - nfft)
    source.seek(low)
    samples = source.read(min(source.frames, start + frames + nfft) - low, dtype="float32", always_2d=True)
    _f, centres, spectrum = scipy.signal.stft(samples.mean(axis=1), fs=1.0, nperseg=nfft, noverlap=nfft // 2, boundary=None)
    inside = (low + centres >= start) & (low + centres < start + frames)
    return float(np.abs(spectrum[:, inside]).max(initial=0.0))


def anchor_range(source, stride=None):
    """The stride-long, stride-aligned range holding the file's loudest moment.

    A chunk that does not contain it is given it as a lead-in, so the model normalises the
    chunk by the maximum the whole file has. The range is one stride long so the chunk
    behind it still starts on the patch grid.
    """
    stride = PATCH_FRAMES if stride is None else stride
    starts = range(0, source.frames, stride)
    loudest = max(starts, key=lambda start: _block_peak(source, start, stride))
    return min(loudest, max(0, source.frames - stride)), min(source.frames, loudest + stride)


# The anchor block and the context that follows it are not adjacent in the file, so the
# junction between them is a discontinuity, and a click there would raise the maximum the
# chunk is normalised by. Both sides are faded over this many frames; the chunk's own
# region is a full stride away and untouched.
JUNCTION_FADE_FRAMES = 1024


def _copy_range(source, start, stop, out, fade_in=False, fade_out=False):
    """Streams frames [start, stop) of an open file into an open output, optionally faded at either end."""
    source.seek(start)
    remaining = stop - start
    position = 0
    while remaining > 0:
        block = source.read(min(BLOCK_FRAMES, remaining), dtype="float32", always_2d=True)
        block = _faded(block, position, stop - start, fade_in, fade_out)
        out.write(block)
        position += len(block)
        remaining -= len(block)


def _faded(block, position, total, fade_in, fade_out):
    """The block with a raised-cosine ramp applied where it covers a faded end of the range."""
    index = np.arange(position, position + len(block), dtype=np.float32)
    gain = np.ones(len(block), dtype=np.float32)
    if fade_in:
        gain *= np.clip(index / JUNCTION_FADE_FRAMES, 0.0, 1.0)
    if fade_out:
        gain *= np.clip((total - 1 - index) / JUNCTION_FADE_FRAMES, 0.0, 1.0)
    return block * (0.5 - 0.5 * np.cos(np.pi * gain))[:, None]


def _margins(anchor, start, stop, frames, stride):
    """(lead, tail) frames around a chunk: the anchor when needed plus true context either side.

    The context is one stride so the chunk still starts on the patch grid; it stops at the
    file's ends, which the whole file shares.
    """
    before, after = min(stride, start), min(stride, frames - stop)
    return _anchor_frames(anchor, start, stop) + before, after


def _write_chunk(source, anchor, start, stop, target, stride):
    """Writes one chunk with its anchor and context; returns the (lead, tail) frames the join drops."""
    lead, tail = _margins(anchor, start, stop, source.frames, stride)
    with (
        atomic_target(target) as partial,
        sf.SoundFile(str(partial), "w", samplerate=source.samplerate, channels=source.channels, subtype="FLOAT") as out,
    ):
        anchored = bool(_anchor_frames(anchor, start, stop))
        before = lead - _anchor_frames(anchor, start, stop)
        if anchored:
            _copy_range(source, anchor[0], anchor[1], out, fade_out=True)
        # The fade-in belongs to the context, never to the chunk's own opening samples: the
        # first chunk has no context before it, so there the anchor alone is faded.
        _copy_range(source, start - before, stop + tail, out, fade_in=anchored and before > 0)
    return lead, tail


def _anchor_frames(anchor, start, stop):
    """Anchor frames a chunk carries: none when it already contains the anchor block."""
    return 0 if start <= anchor[0] and anchor[1] <= stop else anchor[1] - anchor[0]


def split(source_wav, chunk_dir, ranges):
    """Writes one WAV per range under `chunk_dir`, reusing a chunk a previous run completed.

    Returns the chunk paths and, per chunk, the (lead, tail) frames the join drops.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    paths = [chunk_dir / f"chunk_{index:04d}.wav" for index in range(len(ranges))]
    stride = PATCH_FRAMES
    with sf.SoundFile(str(source_wav)) as source:
        anchor = anchor_range(source, stride)
        for path, (start, stop) in zip(paths, ranges):
            if not is_valid_audio(path):
                _write_chunk(source, anchor, start, stop, path, stride)
        margins = [_margins(anchor, start, stop, source.frames, stride) for start, stop in ranges]
    return paths, margins


def clean_output_in(out_dir):
    """The separator's clean stem in a chunk's output directory, or None."""
    candidates = sorted(out_dir.glob("*.wav"), key=lambda path: path.name.lower())
    clean = [path for path in candidates if CLEAN_STEM_TAG in path.name and is_valid_audio(path)]
    return clean[0] if clean else None


def _output_peak(path):
    """The largest sample magnitude of a WAV, streamed."""
    peak = 0.0
    with sf.SoundFile(str(path)) as handle:
        for block in handle.blocks(blocksize=BLOCK_FRAMES, dtype="float32", always_2d=True):
            peak = max(peak, float(np.abs(block).max(initial=0.0)))
    return peak


def _scaled_copy(source_wav, target, gain):
    """Writes the WAV scaled by `gain`, atomically."""
    with (
        sf.SoundFile(str(source_wav)) as source,
        atomic_target(target) as partial,
        sf.SoundFile(str(partial), "w", samplerate=source.samplerate, channels=source.channels, subtype="FLOAT") as out,
    ):
        for block in source.blocks(blocksize=BLOCK_FRAMES, dtype="float32", always_2d=True):
            out.write(block * gain)


def _finished(out_dir):
    """A chunk's finished clean stem in `out_dir`, or None."""
    return clean_output_in(out_dir) if out_dir.is_dir() else None


def _denoise_or_reuse(chunk_path, out_dir, label, denoise_chunk):
    """The chunk's clean stem, from a previous run when it left one."""
    produced = _finished(out_dir)
    if produced is None:
        log_msg(f"    [Neural] Denoising chunk {label}...")
        produced = denoise_chunk(chunk_path, out_dir)
    return produced


def _denoise_one(chunk_path, chunk_dir, index, count, denoise_chunk):
    """One chunk's clean stem and the gain the join applies: 1, or 1/RETRY_GAIN after a retry.

    A chunk the separator peak-limited is denoised again at half scale from a scaled copy
    of the input; a retry a previous run finished is reused ahead of the first attempt.
    """
    retry_dir = chunk_dir / f"out_{index:04d}_retry"
    retried = _finished(retry_dir)
    if retried is not None:
        return retried, 1.0 / RETRY_GAIN
    produced = _denoise_or_reuse(chunk_path, chunk_dir / f"out_{index:04d}", f"{index + 1}/{count}", denoise_chunk)
    if _output_peak(produced) < CHUNK_PEAK - 1e-6:
        return produced, 1.0
    log_msg(f"    [Neural] Chunk {index + 1} was peak-limited by the separator; denoising it again at half scale.")
    scaled = chunk_dir / f"chunk_{index:04d}_retry.wav"
    if not is_valid_audio(scaled):
        _scaled_copy(chunk_path, scaled, RETRY_GAIN)
    return _denoise_or_reuse(scaled, retry_dir, f"{index + 1}/{count} (retry)", denoise_chunk), 1.0 / RETRY_GAIN


def _denoise_chunks(chunk_paths, chunk_dir, denoise_chunk):
    """Runs the denoiser over each chunk; returns the outputs in order and the gain the join applies to each."""
    results = [_denoise_one(path, chunk_dir, index, len(chunk_paths), denoise_chunk) for index, path in enumerate(chunk_paths)]
    return [produced for produced, _gain in results], [gain for _produced, gain in results]


class _PaddedChunk:
    """Reads a chunk output as if it were exactly `frames` long, silence past its real end.

    Interior chunks lose their last few hundred frames to the separator's trim inside the
    overlap, where the fade has already taken them to nothing; the last chunk is padded so
    the join has the source's exact length.
    """

    def __init__(self, handle, frames, index, margins=(0, 0), gain=1.0):
        lead, tail = margins
        self.gain = gain
        # Beyond the range is the tail context, less the separator's trim; short of it is a
        # shortfall the reader pads, or a broken chunk when it exceeds the trim.
        extra = handle.frames - lead - frames
        if extra > tail or -extra > MAX_SHORTFALL_SECONDS * handle.samplerate:
            raise RuntimeError(f"chunk {index} came back with {handle.frames - lead - tail} frames, expected {frames}")
        handle.seek(lead)
        self.handle, self.frames, self.channels = handle, frames, handle.channels

    def read(self, frames):
        block = self.handle.read(frames, dtype="float32", always_2d=True) * self.gain
        if len(block) < frames:
            block = np.concatenate([block, np.zeros((frames - len(block), self.channels), dtype=np.float32)])
        return block


def _blend(carry, head):
    """Linear crossfade from the previous chunk's tail into the next chunk's head."""
    fade = np.linspace(0.0, 1.0, len(head), dtype=np.float32)[:, None]
    return carry * (1.0 - fade) + head * fade


def _stream(chunk, frames, out):
    """Copies `frames` frames from the padded chunk to the output; returns the peak seen."""
    peak = 0.0
    while frames > 0:
        block = chunk.read(min(BLOCK_FRAMES, frames))
        out.write(block)
        peak = max(peak, float(np.abs(block).max(initial=0.0)))
        frames -= len(block)
    return peak


def _overlaps(ranges):
    """Overlap in frames between each chunk and the one before it (0 for the first), then a trailing 0."""
    return [0] + [ranges[index - 1][1] - ranges[index][0] for index in range(1, len(ranges))] + [0]


def _join_chunk(chunk, out, carry, overlap_in, overlap_out):
    """Writes one chunk: its head blended into the carried tail, its middle, and hands back its tail."""
    peak = 0.0
    if carry is not None:
        blended = _blend(carry, chunk.read(overlap_in))
        out.write(blended)
        peak = float(np.abs(blended).max(initial=0.0))
    peak = max(peak, _stream(chunk, chunk.frames - overlap_in - overlap_out, out))
    tail = chunk.read(overlap_out) if overlap_out else None
    return peak, tail


def _write_joined(outputs, ranges, margins, gains, partial):
    """Streams the crossfaded join into `partial`; returns the per-chunk peaks."""
    overlaps = _overlaps(ranges)
    peaks, carry = [], None
    with (
        sf.SoundFile(str(outputs[0])) as first,
        sf.SoundFile(str(partial), "w", samplerate=first.samplerate, channels=first.channels, subtype="FLOAT") as out,
    ):
        for index, path in enumerate(outputs):
            with sf.SoundFile(str(path)) as handle:
                chunk = _PaddedChunk(handle, ranges[index][1] - ranges[index][0], index, margins[index], gains[index])
                peak, carry = _join_chunk(chunk, out, carry, overlaps[index], overlaps[index + 1])
                peaks.append(peak)
    return peaks


def _rescale(path, factor):
    """Rewrites a WAV scaled by `factor`, in place and atomically.

    The source is closed before the partial is published over it: Windows refuses to
    replace a file that is still open.
    """
    with atomic_target(path) as partial:
        with (
            sf.SoundFile(str(path)) as source,
            sf.SoundFile(str(partial), "w", samplerate=source.samplerate, channels=source.channels, subtype="FLOAT") as out,
        ):
            for block in source.blocks(blocksize=BLOCK_FRAMES, dtype="float32", always_2d=True):
                out.write(block * factor)


def join(outputs, ranges, target, margins=None, gains=None):
    """Crossfades the denoised chunks into `target`, dropping each chunk's margins, and applies the peak rule once."""
    margins = margins or [(0, 0)] * len(outputs)
    gains = gains or [1.0] * len(outputs)
    with atomic_target(target) as partial:
        peaks = _write_joined(outputs, ranges, margins, gains, partial)
        if max(peaks) > WHOLE_FILE_PEAK:
            _rescale(partial, WHOLE_FILE_PEAK / max(peaks))
    return target


def joined_name(input_wav, chunk_path, chunk_output):
    """The name the separator would have given the whole file: its stem plus the chunk output's suffix."""
    suffix_at = len(Path(chunk_path).stem)
    return Path(input_wav).stem + Path(chunk_output).name[suffix_at:]


def run(input_wav, output_dir, denoise_chunk, chunk_seconds=None):
    """Denoises a long file in overlapping chunks; returns the joined clean track.

    `denoise_chunk(chunk_wav, out_dir)` runs the separator on one chunk into `out_dir` and
    returns its clean stem. Every chunk output is kept until the join is published, so an
    interrupted run resumes from the chunks it finished. The chunk directory is named after
    the plan, so a changed setting never reuses chunks cut to another grid.
    """
    input_wav, output_dir = Path(input_wav), Path(output_dir)
    with sf.SoundFile(str(input_wav)) as source:
        ranges = plan(source.frames, source.samplerate, chunk_seconds)
    chunk_dir = output_dir / f"chunks_{len(ranges)}x{ranges[0][1]}"
    log_msg(f"    [Neural] Denoising in {len(ranges)} overlapping chunks: the track is longer than one pass holds.")
    chunk_paths, margins = split(input_wav, chunk_dir, ranges)
    outputs, gains = _denoise_chunks(chunk_paths, chunk_dir, denoise_chunk)
    return join(outputs, ranges, output_dir / joined_name(input_wav, chunk_paths[0], outputs[0]), margins, gains)
