"""R2's programme gain match is R1's own, and clearing hiss under the programme band moves neither R2 reading.

Split from `test_restoration_quality_sibilance.py` (its voice and readers are shared) to keep
each file's maintainability index at grade A.
"""

import numpy as np
import pytest
import scipy.signal

from scripts.restoration_quality import auditory, balance_metrics, sibilance
from tests.unit.test_restoration_quality_sibilance import RATE, _r2, _voice_with_esses


def _lowpass(signal, cutoff_hz=8000.0):
    """An 8th-order Butterworth low-pass run forward and back: band-limited (PAL linear) programme."""
    sos = scipy.signal.butter(8, cutoff_hz, btype="lowpass", fs=RATE, output="sos")
    return scipy.signal.sosfiltfilt(sos, np.asarray(signal, dtype=np.float64)).astype(np.float32)


def _hissy_pair(hiss_dbfs, seed=7):
    """`(source, output)`: the 8 kHz band-limited voice plus white hiss at `hiss_dbfs` RMS, and that voice cleared by an oracle."""
    clean = _lowpass(_voice_with_esses(seed=seed)[0])
    hiss = 10.0 ** (hiss_dbfs / 20.0) * np.random.default_rng(100 + seed).standard_normal(len(clean))
    return (clean + hiss).astype(np.float32), clean


def _recorded_gains(monkeypatch):
    """Every `auditory.programme_gain_db` result in call order: R1 and R2 both match the gain through it."""
    gains = []
    original = auditory.programme_gain_db

    def recording(*args):
        gains.append(original(*args))
        return gains[-1]

    monkeypatch.setattr(auditory, "programme_gain_db", recording)
    return gains


def test_the_gain_match_is_r1s_own_on_a_hissy_pair(monkeypatch):
    """R2 subtracts the very gain R1 subtracts; with R1's 90 dB term alone it read 0.04 dB lower on this pair."""
    source, output = _hissy_pair(-45.0)
    gains = _recorded_gains(monkeypatch)
    balance_metrics.balance_readings(source, output, RATE)
    r1_calls = len(gains)
    sibilance.sib_readings(source, output, RATE)
    assert gains[r1_calls - 1] is not None
    assert len(gains) == r1_calls + 1
    assert abs(gains[-1] - gains[r1_calls - 1]) < 1e-9


def _quiet(source):
    """Both sides 60 dB down: under R1's -70 dBFS silence level."""
    quiet = (source * 1e-3).astype(np.float32)
    return quiet, quiet


def _padded(source):
    """1.1 s of voice, then digital silence: fricatives enough for this module, fewer than R1's 100 live STFT frames."""
    padded = np.zeros_like(source)
    padded[: RATE * 11 // 10] = source[: RATE * 11 // 10]
    return padded, padded


def _sunk(source):
    """A loud source and its output 61 dB down: a gain under R1's -60 dB floor."""
    loud = (source * 100.0).astype(np.float32)
    return loud, (loud * 10.0 ** (-61.0 / 20.0)).astype(np.float32)


@pytest.mark.parametrize("make_pair", [_quiet, _padded, _sunk])
def test_no_absolute_level_wherever_r1_reads_no_gain(make_pair):
    """Where R1 refuses the window, R2 has no gain to subtract; the texture, which needs none, still reads.

    Except on the padded pair: its 1.1 s of voice hold about 24 fricative frames, under the
    texture's 30 (`sibilance.TEXTURE_MIN_FRAMES`).
    """
    source, output = make_pair(_voice_with_esses()[0])
    level, texture = _r2(source, output)
    assert balance_metrics.band_profile(source, output, RATE) is None
    assert level is None
    assert (texture is None) == (make_pair is _padded)


def test_clearing_hiss_under_the_band_moves_neither_reading():
    """The benign floor: the band-limited voice with -50 dBFS hiss against the oracle-cleared voice, read to its 8 kHz band."""
    source, output = _hissy_pair(-50.0)
    level, texture = _r2(source, output, bandwidth_hz=8000.0)
    assert abs(level) < 0.05
    assert abs(texture) < 0.05


def test_without_the_cap_the_hiss_above_the_band_moves_the_texture():
    """Read to 12 kHz, the 8-12 kHz bins carry hiss alone: clearing it reads as smoothing, as far as the 15 % islands."""
    source, output = _hissy_pair(-50.0)
    _level, texture = _r2(source, output)
    assert texture < -0.1
