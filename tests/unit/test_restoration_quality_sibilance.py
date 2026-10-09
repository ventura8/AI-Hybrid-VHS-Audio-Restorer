"""The sibilance readings find the 's' bursts and read a thinned or dulled 's' the right way."""

import numpy as np
import scipy.signal

from scripts.restoration_quality import sibilance

RATE = 44100


def _bandpass(noise, low, high):
    sos = scipy.signal.butter(4, [low, high], btype="bandpass", fs=RATE, output="sos")
    return scipy.signal.sosfiltfilt(sos, noise)


def _voice_with_esses(seconds=6.0, seed=7):
    """Vowel bursts (harmonic buzz, 0.3 s) alternating with 120 ms 's' bursts (1-12 kHz noise) over a faint hiss."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    phases = rng.uniform(0.0, 2 * np.pi, 30)
    vowel = sum(np.sin(2 * np.pi * 160.0 * k * t + phases[k]) / k for k in range(1, 30))
    vowel_gate = ((t % 0.6) < 0.3).astype(np.float64)
    ess_gate = ((t % 0.6) >= 0.36) & ((t % 0.6) < 0.48)
    ramp = np.hanning(int(0.01 * RATE))
    vowel_gate = np.convolve(vowel_gate, ramp / ramp.sum(), mode="same")
    ess_gate = np.convolve(ess_gate.astype(np.float64), ramp / ramp.sum(), mode="same")
    ess = 0.5 * _bandpass(rng.standard_normal(len(t)), 4000.0, 10000.0) + 0.35 * _bandpass(rng.standard_normal(len(t)), 1000.0, 4000.0)
    return (0.1 * vowel * vowel_gate + 0.1 * ess * ess_gate + 1e-4 * rng.standard_normal(len(t))).astype(np.float32), ess_gate > 0.5


def _thinned(source, ess_gate, depth_db=12.0):
    """The 1-4 kHz body under the 's' removed inside the bursts only."""
    body = _bandpass(np.asarray(source, dtype=np.float64), 1000.0, 4000.0)
    return (source - (1.0 - 10 ** (-depth_db / 20.0)) * body * ess_gate).astype(np.float32)


def _dulled(source, ess_gate, depth_db=12.0):
    """The 6-12 kHz top of the 's' attenuated inside the bursts only."""
    top = _bandpass(np.asarray(source, dtype=np.float64), 6000.0, 12000.0)
    return (source - (1.0 - 10 ** (-depth_db / 20.0)) * top * ess_gate).astype(np.float32)


def test_the_detector_finds_the_ess_bursts_and_not_the_vowels():
    """Most detected frames fall in the 's' bursts, most of the bursts are found, and silence has none."""
    source, ess_gate = _voice_with_esses()
    mask = sibilance.fricative_mask(source, RATE)
    hit = mask & ess_gate
    assert hit.sum() > 0.6 * mask.sum()
    assert mask.sum() > 0.4 * ess_gate.sum()
    assert sibilance.fricative_mask(np.zeros(100, dtype=np.float32), RATE).sum() == 0


def test_identity_reads_the_same_shape_on_both_sides():
    """The source against itself reads the same v2 shape on both sides, with an 's' centroid over 3.5 kHz."""
    source, _gate = _voice_with_esses()
    readings = sibilance.sib_readings(source, source, RATE)
    for name in sibilance.NAMES:
        src, out = readings[name]
        assert abs(src - out) < 1e-6
    assert readings["sib_centroid_hz"][0] > sibilance.CENTROID_MIN_HZ


def test_a_thinned_ess_moves_the_centroid_up_and_the_body_ratio_up():
    """The 1-4 kHz body lowered under the 's' raises its centroid and its 4-12 kHz over 1-4 kHz ratio."""
    source, ess_gate = _voice_with_esses()
    readings = sibilance.sib_readings(source, _thinned(source, ess_gate), RATE)
    centroid_src, centroid_out = readings["sib_centroid_hz"]
    body_src, body_out = readings["sib_body_db"]
    assert centroid_out - centroid_src > 300.0
    assert body_out - body_src > 3.0


def test_a_dulled_ess_moves_the_centroid_down():
    """The 6-12 kHz top lowered on the 's' lowers its centroid and its 4-12 kHz over 1-4 kHz ratio."""
    source, ess_gate = _voice_with_esses()
    readings = sibilance.sib_readings(source, _dulled(source, ess_gate), RATE)
    centroid_src, centroid_out = readings["sib_centroid_hz"]
    body_src, body_out = readings["sib_body_db"]
    assert centroid_out - centroid_src < -300.0
    assert body_out - body_src < -3.0


def test_too_few_fricatives_reads_none():
    """A window too short, or one with no fricative at all, reads every name unread."""
    short, _gate = _voice_with_esses(seconds=0.3)
    assert sibilance.sib_readings(short, short, RATE) == {name: (None, None) for name in sibilance.READINGS}
    t = np.arange(4 * RATE) / RATE
    tone = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    assert sibilance.sib_readings(tone, tone, RATE)["sib_centroid_hz"] == (None, None)


def _shelved(source, gain_db=2.0, ramp_hz=(3000.0, 3500.0)):
    """A flat `gain_db` lift above 3.5 kHz (rising over 3-3.5 kHz) on the whole file: a shelf that lifts every frame alike."""
    spectrum = np.fft.rfft(np.asarray(source, dtype=np.float64))
    freqs = np.fft.rfftfreq(len(source), 1.0 / RATE)
    ramp = np.clip((freqs - ramp_hz[0]) / (ramp_hz[1] - ramp_hz[0]), 0.0, 1.0)
    return np.fft.irfft(spectrum * 10.0 ** (gain_db * ramp / 20.0), len(source)).astype(np.float32)


def _stft_inside(source, ess_gate, nperseg):
    """`(freqs, STFT, frames whose centre falls inside an 's' burst)`."""
    freqs, times, spec = scipy.signal.stft(np.asarray(source, dtype=np.float64), fs=RATE, nperseg=nperseg)
    return freqs, spec, ess_gate[np.clip((times * RATE).astype(int), 0, len(source) - 1)]


def _resynth(spec, nperseg, length):
    _times, mono = scipy.signal.istft(spec, fs=RATE, nperseg=nperseg)
    return mono[:length].astype(np.float32)


def _islands(source, ess_gate, share=0.5, seed=3):
    """A fluctuating mask on the 's': `share` of its 4-12 kHz STFT cells zeroed at random, the rest raised to keep the power."""
    rng = np.random.default_rng(seed)
    freqs, spec, inside = _stft_inside(source, ess_gate, 256)
    cells = ((freqs >= 4000.0) & (freqs < 12000.0))[:, np.newaxis] & inside[np.newaxis, :]
    kept = rng.random(spec.shape) >= share
    return _resynth(spec * np.where(cells, kept / np.sqrt(1.0 - share), 1.0), 256, len(source))


def _random_phase(source, ess_gate, seed=3):
    """The 's' resynthesised from its STFT magnitudes with random phases."""
    rng = np.random.default_rng(seed)
    _freqs, spec, inside = _stft_inside(source, ess_gate, 512)
    phase = np.exp(1j * rng.uniform(0.0, 2 * np.pi, spec.shape))
    spec[:, inside] = np.abs(spec[:, inside]) * phase[:, inside]
    return _resynth(spec, 512, len(source))


def _r2(source, output, bandwidth_hz=None):
    """`(sib_abs_level_db, sib_texture_db)` on the output side."""
    readings = sibilance.sib_readings(source, output, RATE, bandwidth_hz)
    return readings["sib_abs_level_db"][1], readings["sib_texture_db"][1]


def test_identity_reads_no_absolute_change_and_no_texture_change():
    """The R2 readings are paired: 0.0 on the source side and exactly no change on the output side."""
    source, _gate = _voice_with_esses()
    readings = sibilance.sib_readings(source, source, RATE)
    assert readings["sib_abs_level_db"] == (0.0, 0.0)
    assert readings["sib_texture_db"] == (0.0, 0.0)


def test_a_broadband_gain_cancels_in_the_programme_gain_match():
    """+3 dB on every sample is matched away on the 300-3000 Hz programme cells; the texture is level-blind."""
    source, _gate = _voice_with_esses()
    level, texture = _r2(source, (source * 10 ** (3.0 / 20.0)).astype(np.float32))
    assert abs(level) < 0.05
    assert abs(texture) < 0.01


def test_a_shelf_over_every_frame_moves_the_absolute_level_and_not_the_net_level():
    """The air-shelf blind spot: +2 dB above 3.5 kHz on every frame cancels in the net level reading, not in this one."""
    source, _gate = _voice_with_esses()
    shelved = _shelved(source)
    level, texture = _r2(source, shelved)
    net_src, net_out = sibilance.sib_readings(source, shelved, RATE)["sib_level_db"]
    assert abs(level - 2.0) < 0.1
    assert abs(net_out - net_src) < 0.1
    assert abs(texture) < 0.05


def test_a_dulled_ess_reads_its_top_lost_and_a_thinned_one_keeps_it():
    """The thinned 's' loses its 1-4 kHz body, not its top: only the body filter's skirt above 4 kHz leaves."""
    source, ess_gate = _voice_with_esses()
    dulled, _texture = _r2(source, _dulled(source, ess_gate))
    thinned, _texture = _r2(source, _thinned(source, ess_gate))
    assert dulled < -3.0
    assert -1.0 < thinned <= 0.0


def test_static_filters_on_the_ess_leave_the_texture_alone():
    """A filter held steady over each burst shifts every bin by a constant: no roughness change."""
    source, ess_gate = _voice_with_esses()
    _level, dulled = _r2(source, _dulled(source, ess_gate))
    _level, thinned = _r2(source, _thinned(source, ess_gate))
    assert abs(dulled) < 0.1
    assert abs(thinned) < 0.1


def test_islands_in_the_ess_move_the_texture_more_the_more_cells_they_take():
    """A fluctuating mask on the 's' (random zeroed cells, power kept) reads rougher, the more so the more cells it takes."""
    source, ess_gate = _voice_with_esses()
    level, half = _r2(source, _islands(source, ess_gate, share=0.5))
    _level, third = _r2(source, _islands(source, ess_gate, share=0.3))
    assert half > 0.5
    assert 0.25 < third < half
    assert abs(level) < 2.0


def test_random_phase_on_the_ess_moves_the_texture():
    """The 's' resynthesised with random STFT phases reads rougher."""
    source, ess_gate = _voice_with_esses()
    _level, texture = _r2(source, _random_phase(source, ess_gate))
    assert texture > 0.1


def test_the_bandwidth_caps_the_absolute_band():
    """Read to 5 kHz, the dulled 's' (6-12 kHz lowered) keeps its band; under 4 kHz nothing is left to read."""
    source, ess_gate = _voice_with_esses()
    level, _texture = _r2(source, _dulled(source, ess_gate), bandwidth_hz=5000.0)
    narrow = sibilance.sib_readings(source, source, RATE, bandwidth_hz=3000.0)
    assert abs(level) < 0.5
    assert narrow["sib_abs_level_db"] == narrow["sib_texture_db"] == (None, None)
    assert narrow["sib_body_db"][0] is not None


def test_a_silent_output_has_no_texture_and_no_gain_leaves_no_level():
    """A silent output band has no texture to read; without a programme gain there is no absolute level."""
    source, _gate = _voice_with_esses()
    readings = sibilance.sib_readings(source, np.zeros_like(source), RATE)
    band = np.ones((12, 8))
    assert readings["sib_texture_db"] == (None, None)
    assert sibilance.abs_level_db(band, band, None) is None


def test_a_zero_or_nan_bandwidth_leaves_no_band_to_read():
    """Only None means "no cap": 0 Hz (no programme band) and NaN leave no bin, so neither R2 reading is read."""
    source, _gate = _voice_with_esses()
    assert _r2(source, source, bandwidth_hz=0.0) == (None, None)
    assert _r2(source, source, bandwidth_hz=float("nan")) == (None, None)
