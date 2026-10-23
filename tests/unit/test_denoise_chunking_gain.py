"""The chunk writer's input gain: an unscaled range keeps its samples bit for bit, a scaled one is the float32 product."""

from unittest.mock import MagicMock

import numpy as np
import soundfile as sf

from modules import denoise_chunking as chunking

RATE = 100
# Signed zeros, a full-scale peak and ordinary values: a gain of exactly 1 must leave every bit pattern alone.
DATA = np.array([[0.25, -0.0], [0.0, -0.7], [1.0, 0.1], [-1.0, 0.333]] * 8, dtype=np.float32)


def _source(tmp_path):
    path = tmp_path / "source.wav"
    sf.write(str(path), DATA, RATE, subtype="FLOAT")
    return path


def _copied(tmp_path, **kwargs):
    """The samples `_copy_range` writes for the whole source, read back from a FLOAT file."""
    target = tmp_path / "copy.wav"
    with (
        sf.SoundFile(str(_source(tmp_path))) as source,
        sf.SoundFile(str(target), "w", samplerate=RATE, channels=2, subtype="FLOAT") as out,
    ):
        chunking._copy_range(source, (0, len(DATA)), out, **kwargs)
    return sf.read(str(target), dtype="float32", always_2d=True)[0]


def test_an_unscaled_range_is_copied_bit_for_bit(tmp_path):
    assert _copied(tmp_path).tobytes() == DATA.tobytes()


def test_a_scaled_range_is_the_float32_product(tmp_path):
    gain = np.float32(0.6)
    assert _copied(tmp_path, gain=gain).tobytes() == (DATA * gain).tobytes()


def test_the_default_gain_keeps_the_blocks_float32(tmp_path):
    out = MagicMock()
    with sf.SoundFile(str(_source(tmp_path))) as source:
        chunking._copy_range(source, (0, len(DATA)), out)
    assert out.write.call_args.args[0].dtype == np.float32
