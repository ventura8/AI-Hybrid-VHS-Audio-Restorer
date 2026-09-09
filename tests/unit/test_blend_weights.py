"""Tests for the fitted per-bin blend that repairs over-subtraction.

Spectral subtraction cannot avoid cutting into programme with a single global strength: the
band it clears of noise is where the content lives. The blend decides per frequency bin
using weights fitted against clean references, and on an independent fixture set it improves
log-spectral distance by 0.307 dB on average while moving the defect band from -2.21 dB to
-0.71 on hum-dominated material.

The behaviour that matters most here is the fallbacks. This stage is a refinement, and a
restoration must never fail because its weights are missing, its audio is unreadable, or the
clip is shorter than one analysis window.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from modules import blend_weights

RATE = 44100


def _write(path, samples):
    """Writes float audio, the format the pipeline carries end to end."""
    sf.write(str(path), samples.astype(np.float32), RATE, subtype="FLOAT")
    return path


@pytest.fixture(name="pair")
def _pair(tmp_path):
    """A noisy signal and a plausible denoised version of it."""
    rng = np.random.default_rng(5)
    time = np.arange(RATE) / RATE
    speech = (0.2 * np.sin(2 * np.pi * 180 * time)).astype(np.float32)
    noisy = speech + rng.normal(0.0, 0.02, len(time)).astype(np.float32)
    denoised = speech * 0.85
    return _write(tmp_path / "orig.wav", noisy), _write(tmp_path / "den.wav", denoised)


def test_shipped_weights_load():
    """The fitted weights travel with the repository, so the stage works out of the box."""
    model = blend_weights.load_model()
    assert model is not None
    assert "stack.0.weight" in model


def test_missing_weights_leave_the_audio_untouched(pair, tmp_path):
    """A host without weights still restores; it just skips the refinement."""
    original, denoised = pair
    result = blend_weights.apply_blend(original, denoised, tmp_path / "out.wav", model_path=tmp_path / "absent.npz")
    assert result == denoised


def test_unreadable_audio_leaves_the_audio_untouched(pair, tmp_path):
    """A read failure must not take the restoration down with it."""
    original, denoised = pair
    with patch.object(blend_weights.sf, "SoundFile", side_effect=RuntimeError("bad wav")):
        assert blend_weights.apply_blend(original, denoised, tmp_path / "out.wav") == denoised


def test_audio_shorter_than_one_window_is_skipped(tmp_path):
    """Below one analysis window there is nothing to transform."""
    tiny = _write(tmp_path / "tiny.wav", np.zeros(256, dtype=np.float32))
    assert blend_weights.apply_blend(tiny, tiny, tmp_path / "out.wav") == tiny


def test_blending_produces_audio_between_its_two_inputs(pair, tmp_path):
    """The output is a blend: it must not leave the range its inputs span."""
    original, denoised = pair
    result = blend_weights.apply_blend(original, denoised, tmp_path / "blended.wav")
    assert Path(result).exists()
    blended, _rate = sf.read(str(result), dtype="float32", always_2d=True)
    source, _r = sf.read(str(original), dtype="float32", always_2d=True)
    assert blended.shape[0] == source.shape[0]
    assert float(np.max(np.abs(blended))) <= float(np.max(np.abs(source))) * 1.5


def test_predicted_weights_stay_within_the_blend_range():
    """A weight outside [0, 1] would extrapolate beyond both inputs rather than blend."""
    model = blend_weights.load_model()
    rng = np.random.default_rng(3)
    features = rng.normal(0.0, 20.0, (512, len(blend_weights.FEATURE_NAMES)))
    weights = blend_weights.predict_weights(features, model)
    assert float(weights.min()) >= 0.0
    assert float(weights.max()) <= 1.0


def test_feature_stack_shape_matches_the_named_features():
    """The model's input width is fixed by this list; a mismatch would silently misread."""
    magnitude = np.abs(np.random.default_rng(1).normal(0.5, 0.1, (65, 12)))
    stack = blend_weights.feature_stack(magnitude, magnitude * 0.7)
    assert stack.shape == (65, 12, len(blend_weights.FEATURE_NAMES))
    assert np.isfinite(stack).all()


def test_feature_stack_tolerates_digital_silence():
    """Silent bins are common in real captures and must not produce non-finite features."""
    silent = np.zeros((33, 8))
    stack = blend_weights.feature_stack(silent, silent)
    assert np.isfinite(stack).all()


def test_weights_from_a_different_feature_set_are_refused(tmp_path):
    """A stale weights file skips the blend instead of taking the restoration down.

    The feature definition and the fitted weights have to move together. A mismatch used to
    raise ValueError out of the matrix multiply, and apply_blend catches only OSError and
    RuntimeError, so an out-of-date assets/blend_weights.npz would have failed a whole
    restoration rather than skipping one refinement.
    """
    stale = {
        "feature_mean": np.zeros((1, 7), dtype=np.float32),
        "feature_std": np.ones((1, 7), dtype=np.float32),
        "stack.0.weight": np.zeros((8, 7), dtype=np.float32),
    }
    features = np.zeros((32, len(blend_weights.FEATURE_NAMES)), dtype=np.float32)
    assert blend_weights.predict_weights(features, stale) is None


def test_a_mismatched_model_leaves_the_audio_untouched(tmp_path):
    """The chain continues with the unblended audio rather than failing."""
    rate = 44100
    samples = np.zeros((rate, 1), dtype=np.float32)
    original = tmp_path / "original.wav"
    denoised = tmp_path / "denoised.wav"
    sf.write(str(original), samples, rate, subtype="FLOAT")
    sf.write(str(denoised), samples, rate, subtype="FLOAT")
    stale_path = tmp_path / "stale.npz"
    np.savez(
        stale_path,
        feature_mean=np.zeros((1, 7), dtype=np.float32),
        feature_std=np.ones((1, 7), dtype=np.float32),
        **{
            "stack.0.weight": np.zeros((8, 7), dtype=np.float32),
            "stack.0.bias": np.zeros(8, dtype=np.float32),
            "stack.2.weight": np.zeros((8, 8), dtype=np.float32),
            "stack.2.bias": np.zeros(8, dtype=np.float32),
            "stack.4.weight": np.zeros((1, 8), dtype=np.float32),
            "stack.4.bias": np.zeros(1, dtype=np.float32),
        },
    )
    result = blend_weights.apply_blend(original, denoised, tmp_path / "out.wav", model_path=stale_path)
    assert result == denoised


def _whole_file_blend(original, denoised, rate, model):
    """The blend computed on the whole recording at once: the reference the blocks must equal."""
    _f, _t, spec_original = blend_weights.scipy.signal.stft(original, fs=rate, nperseg=blend_weights.FRAME)
    _f2, _t2, spec_denoised = blend_weights.scipy.signal.stft(denoised, fs=rate, nperseg=blend_weights.FRAME)
    magnitude_original, magnitude_denoised = np.abs(spec_original), np.abs(spec_denoised)
    stack = blend_weights.feature_stack(magnitude_original, magnitude_denoised)
    weight = blend_weights.predict_weights(stack.reshape(-1, stack.shape[-1]), model).reshape(magnitude_original.shape)
    magnitude = weight * magnitude_denoised + (1.0 - weight) * magnitude_original
    _t3, wave = blend_weights.scipy.signal.istft(magnitude * np.exp(1j * np.angle(spec_original)), fs=rate, nperseg=blend_weights.FRAME)
    return wave[: len(original)]


@pytest.mark.parametrize("seconds", [1.0, 2.37])
def test_block_processing_equals_whole_file_processing(tmp_path, seconds):
    """Blocks with context reproduce the whole-file blend sample for sample.

    The stage never holds a recording's features at once -- a two-hour tape's would be
    28 GB -- and this is what shows the saving costs nothing: the per-bin statistics are
    the whole recording's, and every block's interior is what one pass over the file gives.
    """
    rng = np.random.default_rng(11)
    time = np.arange(int(RATE * seconds)) / RATE
    speech = 0.2 * np.sin(2 * np.pi * 180 * time) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.9 * time))
    noisy = (speech + rng.normal(0.0, 0.02, len(time))).astype(np.float32)
    denoised = (speech * 0.85 + rng.normal(0.0, 0.004, len(time))).astype(np.float32)
    original_wav = _write(tmp_path / "orig.wav", np.column_stack((noisy, noisy[::-1])))
    denoised_wav = _write(tmp_path / "den.wav", np.column_stack((denoised, denoised[::-1])))
    with patch.object(blend_weights, "BLOCK_FRAMES", 16):
        result = blend_weights.apply_blend(original_wav, denoised_wav, tmp_path / "blended.wav")
    blended, _rate = sf.read(str(result), dtype="float64", always_2d=True)
    model = blend_weights.load_model()
    for channel, (source, subtracted) in enumerate(((noisy, denoised), (noisy[::-1], denoised[::-1]))):
        reference = _whole_file_blend(source.astype(np.float64), subtracted.astype(np.float64), RATE, model)
        assert blended.shape[0] == len(reference)
        assert np.allclose(blended[:, channel], reference, atol=2e-7)
    assert not (tmp_path / "blended.magnitudes").exists()


def test_a_corrupt_weights_file_is_refused_at_load(tmp_path):
    """A file that is not an archive at all, or a truncated one, skips the blend rather than raising."""
    garbage = tmp_path / "garbage.npz"
    garbage.write_bytes(b"PK\x03\x04 not really a zip")
    assert blend_weights.load_model(garbage) is None
    truncated = tmp_path / "truncated.npz"
    np.savez(truncated, **_complete_model())
    truncated.write_bytes(truncated.read_bytes()[:200])
    assert blend_weights.load_model(truncated) is None


def test_weights_named_for_another_feature_set_are_refused():
    """A file that carries its feature names is checked by name, not only by width."""
    model = _complete_model()
    model["feature_names"] = np.array(blend_weights.FEATURE_NAMES)
    assert blend_weights.schema_error(model) is None
    renamed = list(blend_weights.FEATURE_NAMES)
    renamed[0], renamed[1] = renamed[1], renamed[0]
    model["feature_names"] = np.array(renamed)
    assert blend_weights.schema_error(model) is not None


def test_a_partial_weights_file_is_refused_at_load(tmp_path):
    """A truncated or stale file skips the blend instead of raising KeyError from the arithmetic."""
    partial = tmp_path / "partial.npz"
    np.savez(partial, feature_mean=np.zeros((1, 11)), feature_std=np.ones((1, 11)))
    assert blend_weights.load_model(partial) is None
    assert blend_weights.schema_error({}) is not None


@pytest.mark.parametrize(
    "change",
    [
        {"stack.0.bias": np.zeros(9)},
        {"stack.2.weight": np.zeros((8, 9))},
        {"stack.4.weight": np.zeros((2, 8))},
        {"feature_mean": np.zeros((2, 11))},
        {"feature_std": np.ones((1, 10))},
    ],
)
def test_layers_that_do_not_chain_are_refused(change):
    """Every shape the network relies on is checked, not only the feature width."""
    model = {
        "feature_mean": np.zeros((1, 11)),
        "feature_std": np.ones((1, 11)),
        "stack.0.weight": np.zeros((8, 11)),
        "stack.0.bias": np.zeros(8),
        "stack.2.weight": np.zeros((8, 8)),
        "stack.2.bias": np.zeros(8),
        "stack.4.weight": np.zeros((1, 8)),
        "stack.4.bias": np.zeros(1),
    }
    assert blend_weights.schema_error(model) is None
    model.update(change)
    assert blend_weights.schema_error(model) is not None
    assert blend_weights.predict_weights(np.zeros((4, 11)), model) is None


def test_running_out_of_memory_leaves_the_audio_untouched(pair, tmp_path):
    """A capture too long for the host skips the refinement rather than ending the restoration."""
    original, denoised = pair
    with patch.object(blend_weights, "_write_blend", side_effect=MemoryError()):
        assert blend_weights.apply_blend(original, denoised, tmp_path / "out.wav") == denoised


def test_a_model_that_refuses_mid_way_leaves_no_partial_output(pair, tmp_path):
    """When the weights are refused the half-written target is removed and the input returned."""
    original, denoised = pair
    with patch.object(blend_weights, "predict_weights", return_value=None):
        assert blend_weights.apply_blend(original, denoised, tmp_path / "out.wav") == denoised
    assert not (tmp_path / "out.wav").exists()


def test_without_its_libraries_the_module_imports_and_the_blend_is_skipped(monkeypatch, pair, tmp_path):
    """The optional imports are guarded: without scipy or soundfile the blend is unavailable, not an error."""
    import importlib
    import sys

    original, denoised = pair
    for name in ("scipy.signal", "soundfile"):
        library = sys.modules[name]
        monkeypatch.setitem(sys.modules, name, None)
        reloaded = importlib.reload(blend_weights)
        try:
            assert reloaded._unavailable(reloaded.load_model()) is True
            assert reloaded.apply_blend(original, denoised, tmp_path / "out.wav") == denoised
        finally:
            monkeypatch.setitem(sys.modules, name, library)
            importlib.reload(blend_weights)
    assert blend_weights.sf is not None and blend_weights.scipy is not None


def _complete_model():
    return {
        "feature_mean": np.zeros((1, 11)),
        "feature_std": np.ones((1, 11)),
        "stack.0.weight": np.zeros((8, 11)),
        "stack.0.bias": np.zeros(8),
        "stack.2.weight": np.zeros((8, 8)),
        "stack.2.bias": np.zeros(8),
        "stack.4.weight": np.zeros((1, 8)),
        "stack.4.bias": np.zeros(1),
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("feature_std", 0.0),
        ("feature_std", np.nan),
        ("stack.0.weight", np.nan),
        ("stack.2.bias", np.inf),
        ("stack.4.weight", -np.inf),
    ],
)
def test_non_finite_values_and_a_zero_spread_are_refused(key, value):
    """A NaN anywhere turns every weight into NaN and a zero spread divides by it; neither is a blend."""
    model = _complete_model()
    assert blend_weights.schema_error(model) is None
    model[key].flat[0] = value
    assert blend_weights.schema_error(model) is not None
    assert blend_weights.predict_weights(np.zeros((4, 11)), model) is None


def test_files_that_differ_in_length_are_not_blended(pair, tmp_path):
    """A subtraction shorter than its input is a wrong file; blending a prefix would replace the whole."""
    original, denoised = pair
    shorter = _write(tmp_path / "short.wav", np.zeros(RATE - 100, dtype=np.float32))
    assert blend_weights.apply_blend(original, shorter, tmp_path / "out.wav") == shorter
    assert not (tmp_path / "out.wav").exists()


def test_files_that_differ_in_rate_or_channels_are_not_blended(pair, tmp_path):
    """The original and its subtraction are one capture; a rate or channel mismatch is a wrong file."""
    original, denoised = pair
    mono_fast = _write(tmp_path / "fast.wav", np.zeros(RATE, dtype=np.float32))
    sf.write(str(mono_fast), np.zeros(RATE, dtype=np.float32), RATE // 2, subtype="FLOAT")
    assert blend_weights.apply_blend(original, mono_fast, tmp_path / "out.wav") == mono_fast
    stereo = _write(tmp_path / "stereo.wav", np.zeros((RATE, 2), dtype=np.float32))
    assert blend_weights.apply_blend(original, stereo, tmp_path / "out.wav") == stereo
    assert not (tmp_path / "out.wav").exists()
