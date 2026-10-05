"""The listener degradations touch only the frames they mean to and read more severe with every level.

The fixtures and the degraded copies are cached per module: every generator returns a fresh array, nobody writes into them.
"""

import functools

import numpy as np
import pytest
import scipy.signal

from scripts import quality_degradations as deg
from scripts.restoration_quality import pause_metrics, sibilance, transient_metrics

RATE = 44100
HIT_S = 0.03
HIT_PERIOD_S = 0.4


def _bandpassed(noise, low, high):
    sos = scipy.signal.butter(4, [low, high], btype="bandpass", fs=RATE, output="sos")
    return scipy.signal.sosfiltfilt(sos, noise)


def _smoothed(gate):
    """A 0/1 gate with 10 ms edges."""
    ramp = np.hanning(int(0.01 * RATE))
    return np.convolve(gate.astype(np.float64), ramp / ramp.sum(), mode="same")


@functools.lru_cache(maxsize=None)
def _speech(seconds=6.0, seed=7):
    """Vowel bursts (0.4 s) and 120 ms 's' bursts over long pauses of faint hiss: loud frames, gaps, deep pauses, fricatives."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    phases = rng.uniform(0.0, 2 * np.pi, 30)
    vowel = sum(np.sin(2 * np.pi * 160.0 * k * t + phases[k]) / k for k in range(1, 30))
    cycle = t % 1.2
    vowel_gate, ess_gate = _smoothed(cycle < 0.4), _smoothed((cycle >= 0.5) & (cycle < 0.62))
    ess = 0.5 * _bandpassed(rng.standard_normal(len(t)), 4000.0, 10000.0) + 0.35 * _bandpassed(rng.standard_normal(len(t)), 1000.0, 4000.0)
    return (0.1 * vowel * vowel_gate + 0.1 * ess * ess_gate + 1e-4 * rng.standard_normal(len(t))).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _music(seconds=8.0, seed=5):
    """A 220 Hz bed over hiss with a train of instant-attack bursts every 0.4 s."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    bed = 0.05 * sum(np.sin(2 * np.pi * 220.0 * k * t) / k for k in range(1, 6)) + 3e-3 * rng.standard_normal(len(t))
    burst = int(HIT_S * RATE)
    decay = np.exp(-np.arange(burst) / (0.008 * RATE))
    hits = np.zeros(len(t))
    for start in np.arange(0.2, seconds - HIT_S, HIT_PERIOD_S):
        lo, settled = int(start * RATE), 2 * burst
        hits[lo:][:burst] = 0.3 * _bandpassed(rng.standard_normal(3 * burst), 1000.0, 6000.0)[settled:] * decay
    return (bed + hits).astype(np.float32)


def _pause_readings(speech, outputs, name):
    return [pause_metrics.pause_readings(speech, out, RATE)[name][1] for out in outputs]


@functools.lru_cache(maxsize=None)
def _expander_gain():
    return deg.expander_gain(_speech(), RATE)


@functools.lru_cache(maxsize=None)
def _gated_outputs():
    """The speech with its pauses gated, one copy per level of the degradation."""
    return tuple(deg.gated_pauses(_speech(), RATE, level) for level in deg.DEGRADATIONS["gated_pauses"].levels)


def test_gated_pauses_touch_only_the_gaps():
    """Where the expander's gain is open the output is the base bit for bit; where it is closed nothing grows."""
    speech, gain = _speech(), _expander_gain()
    assert 0.05 < np.mean(gain == 1.0) < 0.5
    for out in _gated_outputs():
        assert np.array_equal(out[gain == 1.0], speech[gain == 1.0])
        assert np.all(np.abs(out[gain < 0.5]) <= np.abs(speech[gain < 0.5]))


def test_gated_pauses_deepen_with_the_floor():
    """The pauses read deeper and emptier with every level."""
    speech, outputs = _speech(), _gated_outputs()
    depth, air = _pause_readings(speech, outputs, "pause_depth_db"), _pause_readings(speech, outputs, "gap_air_db")
    assert depth[0] < depth[1] < depth[2]
    assert air[0] > air[1] > air[2]


@functools.lru_cache(maxsize=None)
def _hissed_outputs():
    """The speech with hiss left in its pauses, one copy per margin of the degradation."""
    speech = _speech()
    noise = (1e-3 * np.random.default_rng(5).standard_normal(len(speech))).astype(np.float32)
    margins = deg.DEGRADATIONS["hiss_in_pauses"].levels
    return tuple(deg.hiss_in_pauses(speech, noise, margin, RATE, np.random.default_rng(1)) for margin in margins)


def test_hiss_in_pauses_adds_air_to_the_gaps_only():
    """The noise lands only where the gain is closed."""
    speech, gain = _speech(), _expander_gain()
    for out in _hissed_outputs():
        assert np.array_equal(out[gain == 1.0], speech[gain == 1.0])
        assert np.any(out[gain < 0.5] != speech[gain < 0.5])


def test_hiss_in_pauses_adds_more_air_with_less_margin():
    """The gap air rises as the margin shrinks."""
    air = _pause_readings(_speech(), _hissed_outputs(), "gap_air_db")
    assert air[0] < air[1] < air[2]


def _sib_shape(speech, out):
    readings = sibilance.sib_readings(speech, out, RATE)
    return readings["sib_centroid_hz"][1], readings["sib_body_db"][1]


@functools.lru_cache(maxsize=None)
def _fricative_weight():
    return deg.fricative_ramp(_speech(), RATE)


def test_the_fricative_weight_is_zero_away_from_the_ess_bursts_and_one_inside_them():
    weight = _fricative_weight()
    assert 0.7 < np.mean(weight == 0.0) < 0.98
    assert (weight >= 0.999).sum() > 0


@pytest.mark.parametrize("generator", [deg.sibilants_thinned, deg.sibilants_dulled])
def test_thinned_and_dulled_sibilants_change_the_ess_bursts_only(generator):
    """Outside the (5 ms ramped) fricative weight the output is the base bit for bit; inside, nearly every sample moves."""
    speech, weight = _speech(), _fricative_weight()
    inside = weight >= 0.999
    out = generator(speech, RATE, 6.0)
    assert np.array_equal(out[weight == 0.0], speech[weight == 0.0])
    assert np.mean(out[inside] != speech[inside]) > 0.9


def _sib_shapes(generator):
    """(base shape, [shape per level]) of the speech through `generator`; the shape is (centroid, top-over-body)."""
    speech = _speech()
    levels = deg.DEGRADATIONS["sibilants_thinned"].levels
    return _sib_shape(speech, speech), [_sib_shape(speech, generator(speech, RATE, level)) for level in levels]


def test_thinned_sibilants_read_brighter_with_the_depth():
    """The centroid and the top-over-body ratio climb with a thinned 's', level by level."""
    base, thin = _sib_shapes(deg.sibilants_thinned)
    assert base[0] < thin[0][0] < thin[1][0] < thin[2][0]
    assert base[1] < thin[0][1] < thin[1][1] < thin[2][1]


def test_dulled_sibilants_read_duller_with_the_depth():
    """The centroid and the top-over-body ratio fall with a dulled 's', level by level."""
    base, dull = _sib_shapes(deg.sibilants_dulled)
    assert base[0] > dull[0][0] > dull[1][0] > dull[2][0]
    assert base[1] > dull[0][1] > dull[1][1] > dull[2][1]


def _touched(length, onsets, smear_ms):
    """The samples inside the smear span after any onset."""
    mask = np.zeros(length, dtype=bool)
    for onset in onsets:
        mask[onset:][: int(smear_ms * RATE / 1000.0)] = True
    return mask


@functools.lru_cache(maxsize=None)
def _smeared_outputs():
    """(smear_ms, the music with its attacks smeared over that span), one per level of the degradation."""
    return tuple((smear_ms, deg.transient_smear(_music(), RATE, smear_ms)) for smear_ms in deg.DEGRADATIONS["transient_smear"].levels)


def test_transient_smear_touches_the_spans_after_the_onsets_only():
    """Only the span after each onset changes."""
    music = _music()
    onsets = transient_metrics.onset_samples(music, RATE)
    assert len(onsets) >= 8
    for smear_ms, out in _smeared_outputs():
        touched = _touched(len(music), onsets, smear_ms)
        assert np.array_equal(out[~touched], music[~touched])
        assert np.mean(out[touched] != music[touched]) > 0.99


def test_transient_smear_softens_the_attack_with_the_span():
    """The attack reading falls as the span grows."""
    music = _music()
    attacks = [transient_metrics.transient_readings(music, out, RATE)["attack_db"][1] for _smear_ms, out in _smeared_outputs()]
    assert transient_metrics.transient_readings(music, music, RATE)["attack_db"][1] > attacks[0] > attacks[1] > attacks[2]


def test_benign_music_identity_is_exact_and_starts_from_the_music_bed():
    """The music identity case copies the bed exactly and is routed to the music fixture."""
    music = _music(seconds=2.0)
    source, output = deg.apply_benign("identity_music", music, RATE, np.random.default_rng(0))
    assert source is music
    assert output is not music
    assert np.array_equal(output, music)


def test_benign_bases_route_the_music_identity_to_the_music_bed():
    assert deg.benign_base("identity_music") == "music"
    assert deg.benign_base("identity") == "speech"
    assert deg.benign_base("shift_5ms") == "speech"
