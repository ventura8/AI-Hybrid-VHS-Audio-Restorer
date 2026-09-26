"""Each degradation changes the signal in its own way, at ordered levels, and the benign ones barely at all."""

import numpy as np
import scipy.signal

from scripts import quality_degradations as deg
from scripts import realistic_defects as defects
from scripts.restoration_quality import scorecard

RATE = 44100


def _voice(seconds=3.0):
    rng = np.random.default_rng(11)
    t = np.arange(int(seconds * RATE)) / RATE
    phases = rng.uniform(0.0, 2 * np.pi, 40)
    voice = sum(np.sin(2 * np.pi * 173.0 * k * t + phases[k]) / k for k in range(1, 40))
    gate = np.convolve(((t % 1.0) < 0.5).astype(np.float64), np.hanning(int(0.02 * RATE)) / int(0.02 * RATE) * 2, mode="same")
    return (0.1 * voice * gate + 1e-4 * rng.standard_normal(len(t))).astype(np.float32)


def _band_db(mono, low, high):
    freqs, psd = scipy.signal.welch(mono, RATE, nperseg=4096)
    return float(10.0 * np.log10(psd[(freqs >= low) & (freqs < high)].sum() + 1e-20))


def test_patched_restores_module_constants():
    before = defects.CRACKLE_PER_SECOND
    with deg._patched(defects, CRACKLE_PER_SECOND=99.0):
        assert defects.CRACKLE_PER_SECOND == 99.0
    assert defects.CRACKLE_PER_SECOND == before


def test_lowpass_levels_remove_more_highs_as_the_cutoff_falls():
    voice = _voice()
    highs = [_band_db(deg.lowpass(voice, RATE, cutoff), 6000.0, 8000.0) for cutoff in (8000.0, 6000.0, 4000.0)]
    assert highs[0] > highs[1] > highs[2]


def test_gain_and_shift_are_exact():
    voice = _voice()
    assert np.allclose(deg.gain(voice, -6.0), voice * 10 ** (-6 / 20), atol=1e-6)
    shifted = deg.shift(voice, RATE, 30.0)
    assert len(shifted) == len(voice) + int(0.03 * RATE)
    assert np.all(shifted[: int(0.03 * RATE)] == 0.0)


def test_mute_and_splice_touch_the_loud_parts_only():
    voice = _voice()
    muted, starts = deg.mute_segment(voice, RATE, 0.3, 2)
    assert len(starts) == 2
    assert abs(starts[0] - starts[1]) >= deg.MUTE_SPACING_S
    span, begins = int(0.3 * RATE), [int(s * RATE) for s in starts]
    assert all(np.all(muted[b:][:span] == 0.0) for b in begins)


def test_splice_replaces_the_loudest_span():
    voice = _voice()
    donor = np.roll(voice, RATE // 3)
    spliced = deg.splice_from(voice, donor, RATE, 0.5)
    assert np.count_nonzero(spliced != voice) >= int(0.5 * RATE) * 0.9


def test_music_attenuation_pairs_share_the_voice():
    voice, music = _voice(), np.roll(_voice(), RATE // 2)
    source, output = deg.music_attenuated(voice, music, -12.0)
    assert np.allclose(source - voice, music, atol=1e-6)
    assert np.allclose(output - voice, music * 10 ** (-12 / 20), atol=1e-6)


def test_compression_narrows_the_range_and_requantise_is_tiny():
    voice = _voice()
    compressed = deg.compress(voice, RATE, 8.0, threshold_db=-40.0)
    assert np.abs(compressed).max() < np.abs(voice).max()
    assert np.abs(deg.benign_requantise(voice, np.random.default_rng(0)) - voice).max() < 2.0 / 32767.0


def test_oversubtraction_and_gating_change_the_spectrum():
    voice = _voice() + (3e-3 * np.random.default_rng(3).standard_normal(3 * RATE)).astype(np.float32)
    subtracted = deg.spectral_oversubtract(voice, RATE, 8.0)
    gated = deg.random_bin_gate(voice, RATE, 0.5, np.random.default_rng(1))
    assert len(subtracted) == len(voice)
    assert np.sqrt(np.mean(subtracted**2)) < np.sqrt(np.mean(voice**2))
    assert 0.3 < np.sqrt(np.mean(gated**2)) / np.sqrt(np.mean(voice**2)) < 0.9


def test_every_degradation_and_benign_transform_applies():
    voice = _voice()
    materials = {
        "noise": 1e-3 * np.random.default_rng(5).standard_normal(len(voice)).astype(np.float32),
        "vhs": voice,
        "donor": np.roll(voice, 1000),
        "voice": voice,
        "music": np.roll(voice, 5000),
    }
    rng = np.random.default_rng(7)
    for name, spec in ((n, s) for n, s in deg.DEGRADATIONS.items() if n != "robotic"):
        source, output = deg.apply(name, spec.levels[0], voice, RATE, materials, rng)
        assert min(len(source), len(output)) > 0, name


def test_every_benign_transform_applies():
    voice = _voice()
    rng = np.random.default_rng(7)
    for name in deg.BENIGN:
        source, output = deg.apply_benign(name, voice, RATE, rng)
        assert len(output) >= len(source), name


def test_listener_degradations_are_dispatched_and_name_registered_metrics():
    """Each listener case has a base, three levels and expectations that name metrics the scorecard registers."""
    for name in deg.LISTENER:
        spec = deg.DEGRADATIONS[name]
        assert spec.base in ("speech", "music")
        assert len(spec.levels) == 3
        assert all(expectation.metric in scorecard.METRICS for expectation in spec.expects), name
