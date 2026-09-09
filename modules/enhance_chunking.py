"""Duration-based chunking for Resemble-Enhance.

Resemble-Enhance holds the whole clip on the GPU, so peak VRAM scales with duration rather
than with any batch setting: an 8 GB card enhances an 8-second clip and dies on a 45-second
one with "CUDA out of memory". Splitting the input keeps the model at full quality -- nfe is
untouched -- instead of trading accuracy for memory.
"""

import shutil

import numpy as np
import soundfile as sf

from .utils import log_msg

# Long enough to cover a spoken word either side of a seam, short enough to stay negligible
# against the shortest chunk.
CHUNK_OVERLAP_SEC = 0.5


def chunk_seconds_for_vram(vram_gb):
    """Returns the chunk length a card of this size can enhance, or 0 to process in one pass."""
    if vram_gb <= 0:
        return 0.0
    if vram_gb < 10.0:
        return 15.0
    if vram_gb < 24.0:
        return 45.0
    return 0.0


def split_for_enhance(source_wav, target_dir, chunk_seconds, overlap_seconds=CHUNK_OVERLAP_SEC):
    """Writes overlapping chunks named so Resemble-Enhance emits them in sortable order."""
    samples, sample_rate = sf.read(str(source_wav), dtype="float32", always_2d=True)
    chunk_len = int(chunk_seconds * sample_rate)
    overlap_len = int(overlap_seconds * sample_rate)
    step = max(chunk_len - overlap_len, 1)

    written = []
    for index, start in enumerate(range(0, len(samples), step)):
        end = start + chunk_len
        block = samples[start:end]
        if len(block) == 0:
            break
        path = target_dir / f"chunk_{index:04d}_{source_wav.name}"
        # subtype is explicit: soundfile defaults .wav to PCM_16, so chunking silently
        # quantised a 32-bit float pipeline down to 16-bit -- and only on the machines small
        # enough to need chunking, which is exactly where it would go unnoticed.
        sf.write(str(path), block, sample_rate, subtype="FLOAT")
        written.append(path)
        if end >= len(samples):
            break
    return written, sample_rate


def join_enhanced_chunks(chunk_paths, destination, overlap_seconds=CHUNK_OVERLAP_SEC):
    """Concatenates enhanced chunks, crossfading the overlap so seams are inaudible."""
    joined = None
    sample_rate = None
    for path in chunk_paths:
        block, rate = sf.read(str(path), dtype="float32", always_2d=True)
        sample_rate = rate if sample_rate is None else sample_rate
        if joined is None:
            joined = block
            continue
        overlap_len = min(int(overlap_seconds * sample_rate), len(joined), len(block))
        if overlap_len <= 0:
            joined = np.concatenate([joined, block], axis=0)
            continue
        # Linear crossfade, deliberately not equal-power. Both sides of the overlap are the same
        # source audio enhanced twice, so they are correlated and sum coherently: an equal-power
        # fade peaks at cos+sin = 1.41x and leaves a +3 dB bump at every seam. Measured on a
        # 220 Hz tone, peak error was 0.124 equal-power against 0.00003 linear.
        fade = np.linspace(0.0, 1.0, overlap_len, dtype=np.float32)[:, None]
        head, tail = joined[:-overlap_len], joined[-overlap_len:]
        blended = tail * (1.0 - fade) + block[:overlap_len] * fade
        joined = np.concatenate([head, blended, block[overlap_len:]], axis=0)

    sf.write(str(destination), joined, sample_rate, subtype="FLOAT")
    return destination


def handle_enhance_output(enhanced_vocals_dir, vocals_wav):
    """Verifies that Resemble-Enhance generated output, providing fallback if empty."""
    candidates_enhanced = list(enhanced_vocals_dir.glob("*.wav"))
    if not candidates_enhanced:
        log_msg("    [Warning] Resemble-Enhance did not produce output. Using raw vocals.", is_error=True)
        fb_path = enhanced_vocals_dir / f"fallback_{vocals_wav.name}"
        shutil.copy(vocals_wav, fb_path)
        return fb_path

    return candidates_enhanced[0]


def prepare_enhance_input(vocals_wav, enhance_input_dir, total_duration):
    """Fills the enhancement input directory, chunking when the GPU cannot hold the whole clip.

    Resemble-Enhance processes every file in the directory, so chunking is just a matter of
    writing more than one -- how it is invoked does not change.
    """
    from . import hardware as _hardware

    chunk_seconds = chunk_seconds_for_vram(float(_hardware.get_optimal_settings().get("gpu_vram_gb", 0.0)))
    if not (chunk_seconds and total_duration and total_duration > chunk_seconds):
        shutil.copy(vocals_wav, enhance_input_dir / vocals_wav.name)
        return []
    chunk_paths, _ = split_for_enhance(vocals_wav, enhance_input_dir, chunk_seconds)
    log_msg(f"    [Enhance] Split into {len(chunk_paths)} x {chunk_seconds:.0f}s chunks to fit available VRAM.")
    return chunk_paths


def collect_enhance_result(enhanced_vocals_tmp_dir, vocals_wav, chunk_paths):
    """Returns the enhanced audio, rejoining chunks when the input was split."""
    if not chunk_paths:
        return handle_enhance_output(enhanced_vocals_tmp_dir, vocals_wav)
    # Sorted by the zero-padded index the split wrote, not by whatever order glob returns, so
    # the timeline is rebuilt in the order it was cut.
    enhanced_chunks = sorted(enhanced_vocals_tmp_dir.glob("chunk_*.wav"))
    if len(enhanced_chunks) != len(chunk_paths):
        log_msg(
            f"    [Warning] Enhancement returned {len(enhanced_chunks)} of {len(chunk_paths)} chunks; using raw vocals.",
            is_error=True,
        )
        # The raw vocals, explicitly. handle_enhance_output() returns the first *.wav it finds,
        # which here is chunk_0000 -- a single 15-second fragment standing in for the whole
        # take, while the log claims the raw vocals were used.
        fallback = enhanced_vocals_tmp_dir / f"fallback_{vocals_wav.name}"
        shutil.copy(vocals_wav, fallback)
        return fallback
    destination = enhanced_vocals_tmp_dir / f"joined_{vocals_wav.name}"
    return join_enhanced_chunks(enhanced_chunks, destination)
