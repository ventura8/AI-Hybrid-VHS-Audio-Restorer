"""Unit tests for duration-based Resemble-Enhance chunking."""

from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from modules import enhance_chunking

SAMPLE_RATE = 44100


def _tone(seconds, freq=220.0):
    """Builds a continuous tone, where any seam discontinuity is obvious."""
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False, dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * freq * t)).reshape(-1, 1)


@pytest.mark.parametrize(
    "vram_gb, expected",
    [(0.0, 0.0), (7.66, 15.0), (8.0, 15.0), (12.0, 45.0), (23.9, 45.0), (24.0, 0.0), (31.8, 0.0)],
)
def test_chunk_seconds_for_vram(vram_gb, expected):
    """Chunk length follows the card: small cards split, large cards run in one pass."""
    assert enhance_chunking.chunk_seconds_for_vram(vram_gb) == expected


def _split_and_join(tmp_path, duration, chunk):
    """Runs one full split/join round trip and returns the chunks and both audio buffers."""
    source = tmp_path / "vocals.wav"
    sf.write(str(source), _tone(duration), SAMPLE_RATE)
    chunk_dir = tmp_path / "in"
    chunk_dir.mkdir()

    chunks, rate = enhance_chunking.split_for_enhance(source, chunk_dir, chunk)
    joined_path = enhance_chunking.join_enhanced_chunks(chunks, tmp_path / "joined.wav")
    original, _ = sf.read(str(source), dtype="float32", always_2d=True)
    joined, _ = sf.read(str(joined_path), dtype="float32", always_2d=True)
    return chunks, rate, joined_path, original, joined


@pytest.mark.parametrize("duration, chunk", [(47.0, 15.0), (120.0, 45.0)])
def test_split_and_join_reconstructs_the_original(tmp_path, duration, chunk):
    """A split followed by a join returns the original audio, sample count included."""
    _chunks, _rate, _path, original, joined = _split_and_join(tmp_path, duration, chunk)
    assert len(joined) == len(original)
    # Linear crossfade over correlated content is transparent; equal-power would peak near 0.12.
    assert float(np.max(np.abs(original - joined))) < 0.001


@pytest.mark.parametrize("duration, chunk", [(47.0, 15.0), (120.0, 45.0)])
def test_split_produces_multiple_chunks_at_the_source_rate(tmp_path, duration, chunk):
    """Audio longer than one chunk splits, and the reported rate is the source's."""
    chunks, rate, _path, _original, _joined = _split_and_join(tmp_path, duration, chunk)
    assert rate == SAMPLE_RATE
    assert len(chunks) > 1


@pytest.mark.parametrize("duration, chunk", [(47.0, 15.0), (120.0, 45.0)])
def test_chunking_round_trip_stays_32_bit_float(tmp_path, duration, chunk):
    """soundfile defaults .wav to PCM_16; the pipeline is 32-bit float end to end, and
    chunking must not quantise it on the low-VRAM machines that are the only ones to
    take this path."""
    chunks, _rate, joined_path, _original, _joined = _split_and_join(tmp_path, duration, chunk)
    assert all(sf.info(str(path)).subtype == "FLOAT" for path in chunks)
    assert sf.info(str(joined_path)).subtype == "FLOAT"


def test_split_emits_a_single_chunk_for_short_audio(tmp_path):
    """Audio shorter than one chunk still produces exactly one file."""
    source = tmp_path / "short.wav"
    sf.write(str(source), _tone(5.0), SAMPLE_RATE)
    chunk_dir = tmp_path / "in"
    chunk_dir.mkdir()

    chunks, _ = enhance_chunking.split_for_enhance(source, chunk_dir, 15.0)
    assert len(chunks) == 1


def test_prepare_enhance_input_copies_whole_file_on_a_large_card(tmp_path):
    """A card that needs no chunking receives the clip as a single file."""
    source = tmp_path / "vocals.wav"
    sf.write(str(source), _tone(60.0), SAMPLE_RATE)
    target = tmp_path / "in"
    target.mkdir()

    with patch("modules.hardware.get_optimal_settings", return_value={"gpu_vram_gb": 32.0}):
        chunks = enhance_chunking.prepare_enhance_input(source, target, 60.0)

    assert chunks == []
    assert (target / source.name).exists()


def test_prepare_enhance_input_chunks_on_a_small_card(tmp_path):
    """A small card splits the clip rather than trying to enhance it whole."""
    source = tmp_path / "vocals.wav"
    sf.write(str(source), _tone(60.0), SAMPLE_RATE)
    target = tmp_path / "in"
    target.mkdir()

    with patch("modules.hardware.get_optimal_settings", return_value={"gpu_vram_gb": 8.0}):
        chunks = enhance_chunking.prepare_enhance_input(source, target, 60.0)

    assert len(chunks) > 1


def test_collect_enhance_result_joins_chunks(tmp_path):
    """Enhanced chunks are rejoined in the order they were cut."""
    source = tmp_path / "vocals.wav"
    sf.write(str(source), _tone(40.0), SAMPLE_RATE)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    chunks, _ = enhance_chunking.split_for_enhance(source, out_dir, 15.0)

    result = enhance_chunking.collect_enhance_result(out_dir, source, chunks)
    assert result.name.startswith("joined_")


def test_collect_enhance_result_falls_back_on_missing_chunks(tmp_path):
    """A short delivery falls back to the raw vocals rather than joining a gapped timeline."""
    source = tmp_path / "vocals.wav"
    sf.write(str(source), _tone(40.0), SAMPLE_RATE)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    chunks, _ = enhance_chunking.split_for_enhance(source, out_dir, 15.0)
    chunks[-1].unlink()

    result = enhance_chunking.collect_enhance_result(out_dir, source, chunks)
    assert not result.name.startswith("joined_")
    # The whole take, not the first surviving chunk: a partial delivery must not quietly
    # substitute a 15-second fragment for the full vocals.
    restored, _ = sf.read(str(result), dtype="float32", always_2d=True)
    original, _ = sf.read(str(source), dtype="float32", always_2d=True)
    assert len(restored) == len(original)
