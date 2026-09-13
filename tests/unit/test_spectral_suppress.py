"""The mode's own noise suppressor: it takes stationary noise to the floor, passes programme, and reconstructs exactly.

Every signal is synthetic and seeded: hiss under a voice with pauses, a clean tone, a
noise floor that steps up part way through.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from modules import spectral_denoise, spectral_suppress
from modules.blend_weights import FRAME, HOP

RATE = 44100
SECONDS = 8.0
SETTINGS = {"noise_bias": 1.0, "gain_floor_db": -20.0, "dd_alpha": 0.96, "probe_s": 2.5}


def _time():
    return np.arange(int(RATE * SECONDS)) / RATE


def _pauses():
    """Speech-like phrasing with one pause longer than the noise probe, so the probe is noise alone."""
    t = _time()
    gate = (np.sin(2 * np.pi * 0.4 * t) > -0.2).astype(float)
    quiet_from, quiet_to = int(2.0 * RATE), int(5.2 * RATE)
    gate[quiet_from:quiet_to] = 0.0
    return gate


def _voice(seed=1):
    t, rng = _time(), np.random.default_rng(seed)
    phase = 2 * np.pi * np.cumsum(137.0 * (1.0 + 0.06 * np.sin(2 * np.pi * 5.0 * t))) / RATE
    partials = sum((0.3 / k) * np.sin(k * phase + rng.uniform(0, 6)) for k in range(1, 12))
    return partials * (0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t)) ** 2 * _pauses()


def _feed(tracker, value, count):
    """Feeds one value to the tracker `count` times and returns the last level."""
    level = None
    for _ in range(count):
        level = tracker.update(value)
    return level


def _hiss(seed=2, level=0.02):
    return np.random.default_rng(seed).normal(0.0, level, int(RATE * SECONDS))


def _write(path, channels):
    sf.write(str(path), np.stack(channels, axis=1).astype(np.float32), RATE, subtype="FLOAT")
    return path


def _db(samples):
    return 20.0 * np.log10(np.sqrt(np.mean(samples**2)) + 1e-12)


@pytest.fixture(name="noisy")
def noisy_fixture(tmp_path):
    return _write(tmp_path / "noisy.wav", [_voice() + _hiss(), _voice(seed=3) + _hiss(seed=4)])


def _run(source, target, **overrides):
    return spectral_suppress.suppress_file(source, target, **{**SETTINGS, **overrides})


def test_stationary_noise_is_taken_toward_the_floor_and_the_voice_kept(noisy, tmp_path):
    """In the pauses the hiss falls by well over ten decibels; the voiced stretches keep their level."""
    restored = sf.read(str(_run(noisy, tmp_path / "out.wav")), dtype="float32")[0]
    original = sf.read(str(noisy), dtype="float32")[0]
    paused = np.abs(_voice()) < 1e-6
    assert _db(original[paused, 0]) - _db(restored[paused, 0]) > 12.0
    assert abs(_db(original[~paused, 0]) - _db(restored[~paused, 0])) < 1.5


def test_a_clean_tone_passes_within_a_fraction_of_a_decibel(tmp_path):
    """Programme far above the floor is not touched."""
    t = _time()
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t) * _pauses()
    source = _write(tmp_path / "tone.wav", [tone + _hiss(level=0.0005), tone + _hiss(seed=5, level=0.0005)])
    restored = sf.read(str(_run(source, tmp_path / "out.wav")), dtype="float32")[0]
    voiced = np.abs(tone) > 0.0
    assert abs(_db(restored[voiced, 0]) - _db(tone[voiced])) < 0.5


def test_the_gain_floor_is_honoured(noisy, tmp_path):
    """No bin, and so no stretch, falls further than the floor allows."""
    restored = sf.read(str(_run(noisy, tmp_path / "out.wav", gain_floor_db=-6.0)), dtype="float32")[0]
    original = sf.read(str(noisy), dtype="float32")[0]
    paused = np.abs(_voice()) < 1e-6
    assert _db(original[paused, 0]) - _db(restored[paused, 0]) < 6.5


def test_the_noise_bias_scales_the_estimate(noisy, tmp_path):
    """Raising the noise estimate removes more."""
    plain = sf.read(str(_run(noisy, tmp_path / "plain.wav")), dtype="float32")[0]
    biased = sf.read(str(_run(noisy, tmp_path / "biased.wav", noise_bias=2.0)), dtype="float32")[0]
    voiced = np.abs(_voice()) > 0.0
    assert _db(biased[voiced, 0]) < _db(plain[voiced, 0])


def test_the_level_tracker_follows_a_rising_floor_and_holds_its_band():
    """A floor that steps up is followed inside three seconds; silence cannot drop it past the band."""
    tracker = spectral_suppress.LevelTracker()
    assert _feed(tracker, 1.0, 200) == pytest.approx(spectral_suppress.MS_BIAS)
    assert _feed(tracker, 3.0, 200) == pytest.approx(min(3.0 * spectral_suppress.MS_BIAS, spectral_suppress.LEVEL_CEILING))
    assert _feed(tracker, 0.0, 200) == pytest.approx(spectral_suppress.LEVEL_FLOOR)


def test_the_tracker_falls_at_once_and_rises_only_after_its_window():
    """A minimum is taken the frame it appears; a rise waits for the old minimum to leave the ring."""
    tracker = spectral_suppress.LevelTracker()
    _feed(tracker, 2.0, 100)
    assert tracker.update(0.7) == pytest.approx(0.7 * spectral_suppress.MS_BIAS)
    span = (spectral_suppress.MS_SUBWINDOWS + 1) * spectral_suppress.MS_SUBWINDOW_FRAMES
    assert _feed(tracker, 2.0, 1) == pytest.approx(0.7 * spectral_suppress.MS_BIAS)
    assert _feed(tracker, 2.0, span) == pytest.approx(2.0 * spectral_suppress.MS_BIAS)


def test_the_exponential_integral_matches_the_library_across_its_range():
    """The plain-arithmetic E1 agrees with scipy's compiled one to a part in a million, from the guard to the tail."""
    import scipy.special

    x = np.logspace(-10, 2.5, 400)
    reference = getattr(scipy.special, "exp1")(x)
    ours = spectral_suppress.exponential_integral(x)
    assert np.allclose(ours, reference, rtol=2e-6, atol=1e-12)


def test_the_lsa_gain_is_finite_at_the_extremes_and_wiener_for_large_nu():
    """Zero, tiny and huge ratios give a finite gain; a large ratio gives the Wiener gain."""
    gamma = np.array([0.0, 1e-12, 1.0, 1e6])
    xi = np.array([1e-3, 1e-3, 1.0, 1e6])
    gain = spectral_suppress.lsa_gain(gamma, xi)
    assert np.all(np.isfinite(gain))
    assert gain[3] == pytest.approx(1e6 / (1.0 + 1e6))
    assert spectral_suppress.lsa_gain(np.array([50.0]), np.array([50.0]))[0] == pytest.approx(50.0 / 51.0, abs=1e-3)


def test_one_gain_serves_both_channels(tmp_path):
    """Anti-phase channels are gained by the mean of their powers, so the image is kept."""
    t = _time()
    tone = 0.3 * np.sin(2 * np.pi * 1200.0 * t) * _pauses()
    source = _write(tmp_path / "anti.wav", [tone + _hiss(level=0.002), -tone + _hiss(seed=5, level=0.002)])
    restored = sf.read(str(_run(source, tmp_path / "out.wav")), dtype="float32")[0]
    voiced = np.abs(tone) > 0.0
    assert abs(_db(restored[voiced, 0]) - _db(restored[voiced, 1])) < 0.2
    assert abs(_db(restored[voiced, 0]) - _db(tone[voiced])) < 0.5


def test_the_output_length_rate_and_channels_match_the_input(noisy, tmp_path):
    """What the blend needs of its inputs: the same capture, frame for frame."""
    with sf.SoundFile(str(_run(noisy, tmp_path / "out.wav"))) as out, sf.SoundFile(str(noisy)) as source:
        assert (out.frames, out.samplerate, out.channels) == (source.frames, source.samplerate, source.channels)


def test_the_probe_is_the_quietest_stretch_on_cathars_grid(tmp_path):
    """A recording with an obvious quiet stretch has its probe taken there, as cathar takes its noise print."""
    from modules import cathar

    t = _time()
    loud = 0.3 * np.sin(2 * np.pi * 300.0 * t)
    quiet_from, quiet_to = int(2.0 * RATE), int(5.0 * RATE)
    loud[quiet_from:quiet_to] = 0.0
    source = _write(tmp_path / "quiet.wav", [loud + _hiss(level=0.01), loud + _hiss(seed=5, level=0.01)])
    start, count = spectral_suppress.quietest_probe(source, 2.5)
    assert count == int(2.5 * RATE)
    grid = set(np.linspace(0, len(t) - count, spectral_suppress.PROBE_POSITIONS, dtype=int).tolist())
    assert start in grid and int(cathar._find_quiet_window(source, 2.5) * RATE) in grid
    assert 2.0 * RATE <= start <= 2.5 * RATE


def test_a_recording_shorter_than_the_probe_has_no_profile(tmp_path):
    source = _write(tmp_path / "short.wav", [_hiss()[: RATE * 2]])
    assert spectral_suppress.quietest_probe(source, 2.5) is None
    assert spectral_suppress.noise_profile(source, 2.5) is None
    assert spectral_suppress.suppress_file(source, tmp_path / "out.wav", **SETTINGS) is None


def test_reconstruction_is_exact_at_unit_gain(noisy, tmp_path, monkeypatch):
    """The square-root windows add up to one: with the gain held at one the file comes back as it went in."""
    monkeypatch.setattr(spectral_suppress.Suppressor, "_gain", lambda self, power: np.ones_like(power))
    restored = sf.read(str(_run(noisy, tmp_path / "out.wav")), dtype="float32")[0]
    original = sf.read(str(noisy), dtype="float32")[0]
    assert np.allclose(restored, original, atol=1e-6)


def test_block_processing_matches_whole_file_processing(noisy, tmp_path, monkeypatch):
    """The state carries across blocks and the grid is absolute, so the block size cannot change the result."""
    whole = sf.read(str(_run(noisy, tmp_path / "whole.wav")), dtype="float32")[0]
    monkeypatch.setattr(spectral_suppress, "BLOCK_FRAMES", 7)
    blocked = sf.read(str(_run(noisy, tmp_path / "blocked.wav")), dtype="float32")[0]
    assert np.allclose(whole, blocked, atol=1e-6)


def test_the_output_is_deterministic(noisy, tmp_path):
    first = sf.read(str(_run(noisy, tmp_path / "a.wav")), dtype="float32")[0]
    second = sf.read(str(_run(noisy, tmp_path / "b.wav")), dtype="float32")[0]
    assert np.array_equal(first, second)


def test_a_failure_inside_the_estimator_returns_none_and_logs(noisy, tmp_path):
    with (
        patch.object(spectral_suppress, "suppress_file", side_effect=MemoryError("no room")),
        patch("modules.spectral_suppress.log_msg") as log,
    ):
        assert spectral_suppress.suppress_or_none(noisy, tmp_path / "out.wav", **SETTINGS) is None
    assert "falling back" in log.call_args[0][0]


def test_the_native_engine_is_used_only_when_switched_on(tmp_path):
    """Off, the cathar path runs; on, the native path runs first and the cathar path is not touched."""
    source, produced = tmp_path / "in.wav", tmp_path / "suppressed_in.wav"
    with (
        patch.object(spectral_denoise, "APL_USE_NATIVE_SUPPRESS", False),
        patch.object(spectral_denoise, "_material", return_value=(3.0, 4.0)),
        patch.object(spectral_denoise, "_cathar_available", return_value=True),
        patch("modules.cathar._cathar_noiseprint_step", return_value=tmp_path / "np.json") as noiseprint,
        patch("modules.cathar._cathar_denoise_step", return_value=tmp_path / "den.wav") as denoise,
    ):
        assert spectral_denoise._subtract(source, tmp_path, None) == tmp_path / "den.wav"
    noiseprint.assert_called_once()
    denoise.assert_called_once()
    with (
        patch.object(spectral_denoise, "APL_USE_NATIVE_SUPPRESS", True),
        patch.object(spectral_denoise, "_material", return_value=(2.0, 4.0)),
        patch("modules.spectral_suppress.suppress_or_none", return_value=produced) as native,
        patch("modules.cathar._cathar_denoise_step") as denoise,
    ):
        assert spectral_denoise._subtract(source, tmp_path, None) == produced
    denoise.assert_not_called()
    assert native.call_args.kwargs["noise_bias"] == pytest.approx(2.0 / 3.0 * spectral_denoise.APL_SUPPRESS_NOISE_BIAS)


def test_a_failed_native_engine_falls_back_to_cathar(tmp_path):
    """Nothing from the native path means the cathar path runs, and no cathar means the stage skips."""
    source = tmp_path / "in.wav"
    with (
        patch.object(spectral_denoise, "APL_USE_NATIVE_SUPPRESS", True),
        patch.object(spectral_denoise, "_material", return_value=(3.0, 4.0)),
        patch("modules.spectral_suppress.suppress_or_none", return_value=None),
        patch.object(spectral_denoise, "_cathar_available", return_value=True),
        patch("modules.cathar._cathar_noiseprint_step", return_value=tmp_path / "np.json"),
        patch("modules.cathar._cathar_denoise_step", return_value=tmp_path / "den.wav"),
    ):
        assert spectral_denoise._subtract(source, tmp_path, None) == tmp_path / "den.wav"
    with (
        patch.object(spectral_denoise, "APL_USE_NATIVE_SUPPRESS", True),
        patch.object(spectral_denoise, "_material", return_value=(3.0, 4.0)),
        patch("modules.spectral_suppress.suppress_or_none", return_value=None),
        patch.object(spectral_denoise, "_cathar_available", return_value=False),
    ):
        assert spectral_denoise._subtract(source, tmp_path, None) is None
        assert spectral_denoise._engine_available() is True
    with (
        patch.object(spectral_denoise, "APL_USE_NATIVE_SUPPRESS", False),
        patch.object(spectral_denoise, "_cathar_available", return_value=False),
    ):
        assert spectral_denoise._engine_available() is False


def test_no_engine_leaves_the_neural_stage_to_work_alone(tmp_path):
    """The stage entry point logs and returns the input when every engine came back empty."""
    source = tmp_path / "in.wav"
    with (
        patch.object(spectral_denoise, "should_apply", return_value=10.0),
        patch.object(spectral_denoise, "_engine_available", return_value=True),
        patch.object(spectral_denoise, "_subtract", return_value=None),
        patch("modules.spectral_denoise.log_msg") as log,
    ):
        assert spectral_denoise.apply_when_needed(source, tmp_path) == source
    assert "No engine could run" in log.call_args[0][0]
    assert isinstance(Path(source), Path) and FRAME > HOP
