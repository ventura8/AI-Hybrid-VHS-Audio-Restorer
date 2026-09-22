"""The pause floor keeper fills the deficit under a restored pause and touches nothing else."""

import functools
from unittest.mock import patch

import numpy as np
import soundfile as sf

import modules.pause_floor as pause_floor

RATE = 44100


@functools.lru_cache(maxsize=None)
def _voice(seconds=8.0, seed=3, channels=2):
    """Tone bursts (0.5 s on / 0.5 s off) over a -50 dBFS hiss; the pauses are the hiss alone. Cached: read, never written."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    burst = ((t % 1.0) < 0.5).astype(np.float64)
    ramp = np.hanning(int(0.02 * RATE))
    burst = np.convolve(burst, ramp / ramp.sum(), mode="same")
    mono = 0.1 * np.sin(2 * np.pi * 220.0 * t) * burst + 3e-3 * rng.standard_normal(len(t))
    return np.repeat(mono[:, None], channels, axis=1).astype(np.float32)


def _speech_mask(audio):
    frame = int(0.02 * RATE)
    level = np.sqrt(np.mean(audio.mean(axis=1)[: len(audio) // frame * frame].reshape(-1, frame) ** 2, axis=1))
    return np.repeat(level > 4.0 * np.percentile(level, 10), frame)


def _gated(audio, floor_db=-40.0):
    """The restored file: the pauses pushed down by `floor_db`, the bursts untouched."""
    mask = _speech_mask(audio)
    gain = np.where(mask, 1.0, 10 ** (floor_db / 20.0))
    gain = np.concatenate([gain, np.ones(len(audio) - len(gain))])
    return (audio * gain[:, None]).astype(np.float32)


def _write(path, audio, rate=RATE):
    sf.write(str(path), audio, rate, subtype="FLOAT")
    return path


def _quiet_level_db(path, mask):
    """Level of the deep pauses: the speech mask widened by 60 ms so the bursts' ramps stay out."""
    import scipy.ndimage

    audio, _rate = sf.read(str(path), dtype="float32", always_2d=True)
    wide = scipy.ndimage.binary_dilation(mask, iterations=int(0.06 * RATE))
    quiet = audio[: len(wide)][~wide].mean(axis=1)
    return 20.0 * np.log10(np.sqrt(np.mean(quiet**2)) + 1e-12)


def _on(fill_db=12.0):
    return (
        patch("modules.pause_floor.ENABLE_PAUSE_FLOOR", True),
        patch("modules.pause_floor.PAUSE_FLOOR_FILL_DB", fill_db),
        patch("modules.pause_floor.PAUSE_FLOOR_QUIET_PERCENTILE", 15.0),
    )


def _filled(tmp_path):
    """The gated voice run through the keeper: (reference, restored, produced, speech mask)."""
    source = _voice()
    reference = _write(tmp_path / "ref.wav", source)
    restored = _write(tmp_path / "res.wav", _gated(source))
    with _on()[0], _on()[1], _on()[2]:
        produced = pause_floor.apply_when_needed(reference, restored, tmp_path)
    return reference, restored, produced, _speech_mask(source)


def test_the_fill_brings_the_pauses_to_the_target(tmp_path):
    reference, restored, produced, mask = _filled(tmp_path)
    assert produced != restored
    assert produced.name.startswith("paused_12db_p15_")
    target_db = _quiet_level_db(reference, mask) - 12.0
    assert abs(_quiet_level_db(produced, mask) - target_db) < 1.5


def test_the_fill_leaves_the_bursts_bit_exact(tmp_path):
    import scipy.ndimage

    _reference, restored, produced, mask = _filled(tmp_path)
    out, _rate = sf.read(str(produced), dtype="float32", always_2d=True)
    res, _rate = sf.read(str(restored), dtype="float32", always_2d=True)
    # Bit-exact away from the pauses: the fill's ramps may touch a burst's fading tail, never its body.
    core = scipy.ndimage.binary_erosion(mask, iterations=int(0.06 * RATE))
    assert core.sum() > 0.3 * len(mask)
    assert np.array_equal(out[: len(core)][core], res[: len(core)][core])
    assert len(out) == len(res)


def test_the_fill_never_exceeds_the_source_and_skips_when_there_is_nothing_to_fill(tmp_path):
    source = _voice()
    reference = _write(tmp_path / "ref.wav", source)
    restored = _write(tmp_path / "res.wav", source)
    with _on()[0], _on()[1], _on()[2]:
        assert pause_floor.apply_when_needed(reference, restored, tmp_path) == restored
    gains = pause_floor.fill_gains(np.full(100, 0.01), np.zeros(100), np.ones(100), 0.0)
    assert gains.max() <= 1.0
    assert gains.min() >= 0.0


def _nothing_written(tmp_path):
    """No paused_* file under the stage dir (a missing dir globs to nothing)."""
    return not list((tmp_path / "pause_floor").glob("paused_*"))


def test_switched_off_returns_the_restored_audio_and_writes_nothing(tmp_path):
    source = _voice()
    reference = _write(tmp_path / "ref.wav", source)
    restored = _write(tmp_path / "res.wav", _gated(source))
    assert pause_floor.apply_when_needed(reference, restored, tmp_path) == restored
    assert _nothing_written(tmp_path)


def test_mismatched_inputs_return_the_restored_audio_and_write_nothing(tmp_path):
    source = _voice()
    reference = _write(tmp_path / "ref.wav", source)
    mono = _write(tmp_path / "mono.wav", _gated(source)[:, :1])
    short = _write(tmp_path / "short.wav", _gated(source)[: 3 * RATE])
    with _on()[0], _on()[1], _on()[2]:
        assert pause_floor.apply_when_needed(reference, mono, tmp_path) == mono
        assert pause_floor.apply_when_needed(reference, short, tmp_path) == short
    assert _nothing_written(tmp_path)


def test_a_failing_fill_returns_the_restored_audio_and_writes_nothing(tmp_path):
    source = _voice()
    reference = _write(tmp_path / "ref.wav", source)
    restored = _write(tmp_path / "res.wav", _gated(source))
    with _on()[0], _on()[1], _on()[2], patch("modules.pause_floor.fill_file", side_effect=OSError("disk")):
        assert pause_floor.apply_when_needed(reference, restored, tmp_path) == restored
    assert _nothing_written(tmp_path)


def test_block_size_does_not_change_the_result_and_the_file_is_reused(tmp_path):
    source = _voice(seconds=4.0)
    reference = _write(tmp_path / "ref.wav", source)
    restored = _write(tmp_path / "res.wav", _gated(source))
    with _on()[0], _on()[1], _on()[2]:
        first = pause_floor.apply_when_needed(reference, restored, tmp_path / "a")
        with patch("modules.pause_floor.BLOCK_SAMPLES", 8192):
            second = pause_floor.apply_when_needed(reference, restored, tmp_path / "b")
        stamp = first.stat().st_mtime_ns
        again = pause_floor.apply_when_needed(reference, restored, tmp_path / "a")
    a, _r = sf.read(str(first), dtype="float32")
    b, _r = sf.read(str(second), dtype="float32")
    assert np.abs(a - b).max() < 1e-6
    assert again == first
    assert first.stat().st_mtime_ns == stamp


def _shifted_pair():
    env = np.abs(np.random.default_rng(1).standard_normal(400)) + 0.01
    return env, np.concatenate([np.full(3, 0.01), env[:-3]])


def test_envelope_lag_finds_a_shift_and_alignment_trims_to_the_overlap():
    env, shifted = _shifted_pair()
    assert pause_floor.envelope_lag(env, shifted) == 3
    ref, res, lag = pause_floor._aligned(env, shifted)
    assert lag == 3
    assert len(ref) == len(res) == 397
    assert np.allclose(ref, res)


def test_alignment_reads_the_opposite_shift_as_a_negative_lag():
    env, shifted = _shifted_pair()
    ref, res, lag = pause_floor._aligned(shifted, env)
    assert lag == -3
    assert len(ref) == len(res) == 397
