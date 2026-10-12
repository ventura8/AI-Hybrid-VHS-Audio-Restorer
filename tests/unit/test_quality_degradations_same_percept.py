"""The same-percept transforms (backlog E2 part c): what they change, their seeding, and the one floor they feed."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

from scripts import calibrate_quality_metrics as cal
from scripts import calibration_checks as checks
from scripts import quality_degradations as deg
from scripts import same_percept

RATE = 16000


def _voice(seconds=2.0):
    t = np.arange(int(seconds * RATE)) / RATE
    envelope = (np.sin(2 * np.pi * 2.0 * t) > 0).astype(np.float64)
    return (0.2 * envelope * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _noise(seconds=2.0, seed=1):
    return (0.01 * np.random.default_rng(seed).standard_normal(int(seconds * RATE))).astype(np.float32)


def _band_db(signal):
    """Mean power per octave band 125 Hz - 8 kHz, dB."""
    freqs, psd = scipy.signal.welch(signal, fs=RATE, nperseg=1024)
    edges = [125 * 2**k for k in range(7)]
    return np.array([10 * np.log10(np.mean(psd[(freqs >= lo) & (freqs < 2 * lo)])) for lo in edges])


def test_the_same_percept_transforms_join_the_benign_set_on_speech_and_on_tape():
    assert set(deg.SAME_PERCEPT) <= set(deg.BENIGN)
    assert set(deg.SAME_PERCEPT) <= set(deg.on_tape_benign())
    assert all(deg.benign_base(name) == "speech" and not deg.benign_changes(name) for name in deg.SAME_PERCEPT)
    old = deg.BENIGN[: deg.BENIGN.index("identity_music") + 1]
    assert deg.BENIGN == (*old, *deg.SAME_PERCEPT, *deg.LINEAR_BENIGN)


def test_each_case_is_seeded_from_its_language_name_and_level():
    draw = [
        deg.benign_rng(*key).random()
        for key in (("en", "fresh_noise"), ("en", "fresh_noise"), ("fr", "fresh_noise"), ("en", "tpdf_dither"))
    ]
    assert draw[0] == draw[1]
    assert len(set(draw[1:])) == 3
    assert deg.benign_rng("en", "x", 1).random() != deg.benign_rng("en", "x", 2).random()


def test_fresh_noise_keeps_the_recorded_noise_on_one_side_and_its_spectrum_on_the_other():
    voice, noise = _voice(), _noise()
    source, output = deg.fresh_noise(voice, RATE, deg.benign_rng("en", "fresh_noise"), noise)
    assert np.allclose(source, voice + noise, atol=1e-7)
    fresh = output.astype(np.float64) - voice
    assert np.max(np.abs(_band_db(fresh) - _band_db(noise))) < 1.0
    assert abs(np.corrcoef(fresh, noise)[0, 1]) < 0.2


def _wobbling_noise(seconds=4.0, seed=2):
    """White noise under a +-6 dB wobble at 0.5 Hz: a tape floor that wanders, as realistic-v2's does."""
    t = np.arange(int(seconds * RATE)) / RATE
    return (_noise(seconds, seed) * 10 ** (6.0 * np.sin(2 * np.pi * 0.5 * t) / 20)).astype(np.float32)


def _frame_db(signal, seconds=0.1):
    size = int(RATE * seconds)
    frames = np.asarray(signal[: len(signal) // size * size], dtype=np.float64).reshape(-1, size)
    return 10 * np.log10(np.mean(frames**2, axis=1))


def test_fresh_noise_keeps_the_recorded_noises_level_wander_frame_by_frame():
    """The fresh realisation follows the wobble within a decibel or so and is new noise; a plain phase surrogate is steady."""
    noise = _wobbling_noise()
    fresh = same_percept.fresh_realisation(noise, RATE, deg.benign_rng("en", "fresh_noise"))
    assert np.max(np.abs(_frame_db(fresh) - _frame_db(noise))) < 1.5
    assert abs(np.corrcoef(fresh, noise)[0, 1]) < 0.2
    assert np.max(np.abs(_band_db(fresh) - _band_db(noise))) < 1.0
    steady = same_percept.resynthesised_noise(noise, deg.benign_rng("en", "fresh_noise"))
    assert np.ptp(_frame_db(steady)) < 0.5 * np.ptp(_frame_db(noise))


def test_a_silent_or_short_recorded_noise_gives_a_finite_realisation():
    for noise in (np.zeros(RATE), _noise(0.01)):
        envelope = same_percept.noise_envelope(noise, RATE)
        assert envelope.shape == noise.shape
        assert np.all(envelope > 0.0)
        assert np.all(np.isfinite(same_percept.fresh_realisation(noise, RATE, deg.benign_rng("en", "fresh_noise"))))


def test_without_a_recorded_noise_two_realisations_of_the_excerpts_own_floor_are_added():
    voice = _voice() + _noise(seed=4)
    source, output = deg.fresh_noise(voice, RATE, deg.benign_rng("ro-cut", "fresh_noise"))
    added = [np.asarray(side, dtype=np.float64) - voice for side in (source, output)]
    assert abs(10 * np.log10(np.mean(added[0] ** 2) / np.mean(added[1] ** 2))) < 0.5
    assert abs(np.corrcoef(*added)[0, 1]) < 0.2
    assert len(source) == len(output) == len(voice)


@pytest.mark.parametrize("frequency", [200.0, 2000.0])
def test_the_all_pass_filters_keep_every_magnitude_and_turn_the_phase(frequency):
    noise = _noise(4.0).astype(np.float64)
    filtered = same_percept.allpass(noise, RATE, frequency).astype(np.float64)
    assert np.max(np.abs(_band_db(filtered) - _band_db(noise))) < 0.3
    assert not np.allclose(filtered, noise, atol=1e-3)


def test_the_fractional_delay_moves_a_tone_by_037_samples():
    t = np.arange(4 * RATE)
    tone = np.sin(2 * np.pi * 440.0 * t / RATE)
    delayed = same_percept.fractional_delay(tone, same_percept.FRACTIONAL_DELAY_SAMPLES).astype(np.float64)
    expected = np.sin(2 * np.pi * 440.0 * (t - 0.37) / RATE)
    middle = slice(RATE, 3 * RATE)
    assert np.max(np.abs(delayed[middle] - expected[middle])) < 1e-3
    assert len(delayed) == len(tone)


def test_the_gain_cases_move_the_level_a_tenth_of_a_decibel_each_way():
    voice = _voice()
    for name, db in deg.BENIGN_GAIN_DB.items():
        _source, output = deg.apply_benign(name, voice, RATE, None)
        ratio = np.max(np.abs(output)) / np.max(np.abs(voice))
        assert 20 * np.log10(ratio) == pytest.approx(db, abs=1e-4)


def test_tpdf_dither_adds_at_most_one_lsb_without_rounding():
    voice = _voice()
    _source, output = deg.apply_benign("tpdf_dither", voice, RATE, deg.benign_rng("en", "tpdf_dither"))
    added = output.astype(np.float64) - voice
    assert np.max(np.abs(added)) <= 1.0 / 32768.0 + 1e-7
    assert np.std(added) > 0.0
    assert not np.allclose(output * 32767.0, np.round(output * 32767.0), atol=1e-3)


@pytest.mark.parametrize("name", deg.SAME_PERCEPT)
def test_every_same_percept_case_applies_with_and_without_the_fixtures_noise(name):
    for materials in ({"noise": _noise()}, None):
        source, output = deg.apply_benign(name, _voice(), RATE, deg.benign_rng("en", name), materials)
        assert len(source) == len(output) == len(_voice())


def _materials():
    """The speech, music and recorded noise, and ear v4's linear capture the `*_linear` cases start from."""
    voice, noise = _voice(1.0), _noise(1.0)
    linear = dict(zip(cal.LINEAR_KEYS, deg.linear_capture(voice, noise, RATE)))
    return {"speech": voice, "music": voice, "noise": noise, "rate": RATE, **linear}


def _outputs(tmp_path, seed):
    cases = cal._benign_cases(_materials(), "en", tmp_path, np.random.default_rng(seed))
    return {case.name: sf.read(str(case.output), dtype="float32")[0] for case in cases}


def test_the_new_cases_draw_from_their_own_generators_and_the_old_ones_from_the_shared_stream(tmp_path):
    """Two runs on different shared streams: the same-percept outputs repeat exactly, requantisation does not."""
    first, second = _outputs(tmp_path / "a", 1), _outputs(tmp_path / "b", 2)
    assert all(np.array_equal(first[name], second[name]) for name in deg.SAME_PERCEPT)
    assert not np.array_equal(first["requantise"], second["requantise"])


def test_the_old_cases_are_bit_identical_to_the_set_before_the_new_ones(tmp_path):
    """Requantisation draws the same samples from the shared stream as when it was the only benign case drawing."""
    outputs = _outputs(tmp_path, 7)
    _source, alone = deg.apply_benign("requantise", _materials()["speech"], RATE, np.random.default_rng(7))
    assert np.array_equal(outputs["requantise"], alone)


def _case(name):
    return cal.Case("benign", name, None, "en", "s.wav", "o.wav")


HF_EXCESS = "dsp.gap_hf_excess_db"
SUITE_READINGS = {
    f"en/benign_{name}_None": {HF_EXCESS: value}
    for name, value in [("identity", 0.0), ("requantise", 0.004), ("fresh_noise", 0.67), ("allpass_2khz", 0.1)]
}


def _run_checks(seen):
    def run(_cases, _readings, floor):
        seen["floor"] = floor
        return []

    return run


@pytest.fixture(name="suite")
def _suite(tmp_path, monkeypatch):
    """The synthetic suite on four benign cases, half of them same-percept, with the floor its checks were given."""
    cases = [_case("identity"), _case("requantise"), _case("fresh_noise"), _case("allpass_2khz")]
    seen = {}
    monkeypatch.setattr(cal, "build_cases", lambda *_args: cases)
    monkeypatch.setattr(cal, "build_tape_cases", lambda *_args: [])
    monkeypatch.setattr(cal, "score_cases", lambda *_args: {})
    monkeypatch.setattr(cal, "readings_of", lambda _scores, _route=None: SUITE_READINGS)
    monkeypatch.setattr(cal, "family_status", lambda _scores: {})
    monkeypatch.setattr(cal.checks_mod, "run_checks", _run_checks(seen))
    suite = cal.synthetic(SimpleNamespace(fixtures="x", languages=["en"], out=tmp_path, excerpts=None, oracle_max=0), ("dsp",), None)
    return suite, seen


def test_the_sensitivity_checks_read_the_same_percept_floor(suite):
    """A fresh hiss realisation reads 0.67 dB of HF excess: the checks' floor takes it in."""
    found, seen = suite
    assert found["check_floor"][HF_EXCESS]["floor"] > 0.5
    assert seen["floor"] is found["check_floor"]


def test_the_gates_and_routes_keep_the_old_benign_floor(suite):
    found, _seen = suite
    assert found["floor"][HF_EXCESS]["floor"] == pytest.approx(0.002)
    assert found["route_floors"]["speech"][HF_EXCESS]["floor"] == pytest.approx(0.002)


def test_the_floor_can_leave_benign_cases_out_by_name():
    cases = [_case("identity"), _case("fresh_noise")]
    readings = {"en/benign_identity_None": {"m": 0.0}, "en/benign_fresh_noise_None": {"m": 1.0}}
    assert checks.noise_floor(cases, readings, ("fresh_noise",)) == {"m": {"centre": 0.0, "floor": 0.0}}
    assert checks.noise_floor(cases, readings)["m"]["floor"] == pytest.approx(0.5)
