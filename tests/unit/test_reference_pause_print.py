"""The programme release's reference, the pause print (2026-10-21): a coder's dropouts cannot pull it down.

Vaccin's MP3 coder drops 10-16 kHz in its speech frames. The whole-file print's 5th percentile
read those dropouts and sat 11.7 dB under the hiss its pauses hold, so the residual hiss stood over
the print and the release gave it back (the APL v9 loop's round 4). The pause print reads the
pauses, where the coder keeps the hiss.
"""

from unittest.mock import patch

import numpy as np
import scipy.signal
import soundfile as sf

from modules import spectral_suppress
from modules.blend_weights import FRAME
from tests.unit.test_reference_release import BINS, INTERIOR, _frames
from tests.unit.test_reference_suppress import SETTINGS

RATE = 44100
HIGH = 10000.0


def _coded(tmp_path, seconds=12.0, seed=11):
    """A coded interview: white hiss throughout, a loud low tone in the first 0.7 s of each second (the speech), and
    in those frames everything over HIGH dropped, as an MP3 coder drops it."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    speech = (t % 1.0) < 0.7
    hiss = 3e-3 * rng.standard_normal(len(t))
    low, high = (scipy.signal.butter(8, HIGH, btype=kind, fs=RATE, output="sos") for kind in ("low", "high"))
    hiss_low, hiss_high = scipy.signal.sosfilt(low, hiss), scipy.signal.sosfilt(high, hiss)
    mono = hiss_low + np.where(speech, 0.0, 1.0) * hiss_high + np.where(speech, 0.2, 0.0) * np.sin(2 * np.pi * 300.0 * t)
    path = tmp_path / "coded.wav"
    sf.write(str(path), np.repeat(mono[:, None], 2, axis=1).astype(np.float32), RATE, subtype="FLOAT")
    return path


def _band_mean_db(spectrum, lo, hi):
    freqs = np.fft.rfftfreq(FRAME, 1.0 / RATE)
    band = (freqs >= lo) & (freqs < hi)
    return float(10.0 * np.log10(np.mean(spectrum[band])))


def test_the_pause_print_stands_at_the_hiss_the_coder_s_dropouts_pull_the_print_under(tmp_path):
    source = _coded(tmp_path)
    noise, _depth = spectral_suppress.reference_noise(source)
    paused = spectral_suppress.pause_print(source, noise)
    assert _band_mean_db(noise, 12000.0, 16000.0) < _band_mean_db(paused, 12000.0, 16000.0) - 10.0
    assert abs(_band_mean_db(paused, 1000.0, 8000.0) - _band_mean_db(noise, 1000.0, 8000.0)) < 3.0
    assert np.all(paused >= noise)


def test_without_enough_pause_the_release_reads_against_the_print(tmp_path):
    source = _coded(tmp_path, seconds=12.0)
    noise, _depth = spectral_suppress.reference_noise(source)
    with patch.object(spectral_suppress, "PAUSE_MIN_WINDOWS", 10**9):
        assert spectral_suppress.pause_print(source, noise) is noise
    silent = tmp_path / "silent.wav"
    sf.write(str(silent), np.zeros((RATE, 2), dtype=np.float32), RATE, subtype="FLOAT")
    assert spectral_suppress.pause_print(silent, noise) is noise


def test_residual_hiss_over_the_print_but_under_the_pause_print_is_not_released():
    noise = np.ones(BINS)
    suppressor = spectral_suppress.ReferenceSuppressor(noise, 1.0, -20.0, 0.96, (3.0, np.full(BINS, 100.0)))
    restored = _frames(30.0)
    out = suppressor.apply_pair(_frames(1.5), restored)
    whole = spectral_suppress.ReferenceSuppressor(noise, 1.0, -20.0, 0.96).apply_pair(_frames(1.5), restored)
    assert np.allclose(out[-1, 0, INTERIOR], whole[-1, 0, INTERIOR])


def test_the_pause_print_is_learned_only_when_the_release_is_on(tmp_path):
    source = _coded(tmp_path, seconds=6.0)
    with (
        patch.object(spectral_suppress, "MIN_DEPTH_DB", 0.0),
        patch.object(spectral_suppress, "pause_print", wraps=spectral_suppress.pause_print) as learned,
    ):
        spectral_suppress.suppress_by_reference(source, source, tmp_path / "off.wav", **SETTINGS)
        assert learned.call_count == 0
        spectral_suppress.suppress_by_reference(source, source, tmp_path / "on.wav", **SETTINGS, release_db=3.0)
        assert learned.call_count == 1
