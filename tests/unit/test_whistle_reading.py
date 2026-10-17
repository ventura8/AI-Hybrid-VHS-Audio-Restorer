"""dsp.whistle_db made oracle-proof (2026-10-12): a floor under the neighbourhood and nothing under 0.

Before, the reading was the core's peak over its neighbourhood's median, whatever either held: a
capture whose tape noise was notched at the line read -40 dB, a clean programme with nothing at
15.6 kHz read the maximum of its near-empty bins (+2..+5 dB), and the oracle check failed on 31 %
of the realistic-v2 pairs. Now the neighbourhood is read no lower than a sine
`WHISTLE_FLOOR_DB` (80 dB) under the loud frames would put in one bin, and a hole reads 0.
"""

import numpy as np
import pytest
import scipy.signal

from scripts.restoration_quality import dsp_metrics as dsp

RATE = 44100
SECONDS = 6.0
# A line on an exact Welch bin near the PAL rate: no scalloping between the planted and the read level.
LINE_HZ = round(dsp.WHISTLE_HZ * dsp.WHISTLE_NPERSEG / RATE) * RATE / dsp.WHISTLE_NPERSEG
HOLE_HZ = 50.0


def _programme(seed=1):
    """Speech-like bursts low-passed at 6 kHz: loud frames, and nothing of its own at the line."""
    rng = np.random.default_rng(seed)
    count = int(SECONDS * RATE)
    gate = (np.arange(count) / RATE % 1.0) < 0.6
    sos = scipy.signal.butter(8, 6000.0, fs=RATE, output="sos")
    return 0.1 * scipy.signal.sosfilt(sos, rng.standard_normal(count) * gate)


def _hiss(level, seed=2):
    """White tape hiss at `level` RMS."""
    return level * np.random.default_rng(seed).standard_normal(int(SECONDS * RATE))


def _notched(signal):
    """`signal` with both line rates cut out, as realistic-v2 builds its tape noise (`realistic_defects._without_tones`)."""
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / RATE)
    for line in (15625.0, 15734.0):
        spectrum[np.abs(freqs - line) <= HOLE_HZ] = 0.0
    return np.fft.irfft(spectrum, len(signal))


def _line(level_db_re_loud, programme):
    """A steady line `level_db_re_loud` dB under the programme's loud frames (mean square)."""
    levels = dsp.frame_levels(programme)
    loud = np.mean(levels[levels >= np.percentile(levels, dsp.LOUD_PERCENTILE)] ** 2)
    amplitude = np.sqrt(2.0 * loud * 10.0 ** (level_db_re_loud / 10.0))
    return amplitude * np.sin(2.0 * np.pi * LINE_HZ * np.arange(len(programme)) / RATE)


def _ratio_db(mono):
    """The reading before 2026-10-12: the core's peak over the neighbourhood's median, unfloored."""
    freqs, psd = scipy.signal.welch(mono, RATE, nperseg=dsp.WHISTLE_NPERSEG)
    core = np.abs(freqs - dsp.WHISTLE_HZ) <= dsp.WHISTLE_CORE_HZ
    around = (np.abs(freqs - dsp.WHISTLE_HZ) <= dsp.WHISTLE_NEIGHBOURHOOD_HZ) & ~core
    return float(10.0 * np.log10((psd[core].max() + 1e-20) / (np.median(psd[around]) + 1e-20)))


def test_a_hole_at_the_line_reads_zero_not_a_negative_whistle():
    """realistic-v2's notched tape noise read about -40 dB before; a hole is not a line."""
    source = _programme() + _notched(_hiss(0.01))
    assert _ratio_db(source) < -20.0
    assert dsp.whistle_line_db(source, RATE) == 0.0


def test_a_band_with_nothing_audible_in_it_reads_zero():
    """A clean programme whose band holds only residue 110 dB under its loud frames: the old ratio read it as a whistle."""
    programme = _programme()
    clean = programme + _hiss(1e-7) + _line(-110.0, programme)
    assert _ratio_db(clean) > 3.0
    assert dsp.whistle_line_db(clean, RATE) == 0.0


def test_a_line_over_hiss_above_the_floor_reads_as_before():
    """Where the hiss sits over the floor the reading is the unfloored ratio, unchanged."""
    mono = _programme() + _hiss(0.01) + _line(-50.0, _programme())
    assert dsp.whistle_line_db(mono, RATE) == pytest.approx(_ratio_db(mono), abs=1e-9)


def test_a_line_on_an_empty_band_reads_its_level_over_the_floor():
    """A line 50 dB under the loud frames over nothing reads 30 dB: its level over the 80 dB floor."""
    programme = _programme()
    reading = dsp.whistle_line_db(programme + _line(-50.0, programme), RATE)
    assert reading == pytest.approx(dsp.WHISTLE_FLOOR_DB - 50.0, abs=0.5)


def test_the_oracle_pair_passes_the_gate():
    """A notched-hiss capture against its clean programme: the delta the gate reads stays under 3 dB."""
    programme = _programme()
    source = programme + _notched(_hiss(0.01))
    assert dsp.whistle_line_db(programme, RATE) - dsp.whistle_line_db(source, RATE) <= 3.0


def test_a_line_the_hiss_hid_and_the_denoise_exposes_still_fails_the_gate():
    """A line 62 dB under the loud frames that the source's hiss hides (it reads what the hiss alone reads), the hiss
    removed and the line kept: the reading rises past the 3 dB gate."""
    programme = _programme()
    line = _line(-62.0, programme)
    hiss = _hiss(0.004)
    assert dsp.whistle_line_db(programme + hiss + line, RATE) == pytest.approx(dsp.whistle_line_db(programme + hiss, RATE), abs=0.5)
    assert dsp.whistle_line_db(programme + line, RATE) - dsp.whistle_line_db(programme + hiss + line, RATE) > 3.0


def test_digital_silence_reads_zero():
    assert dsp.whistle_line_db(np.zeros(2 * dsp.WHISTLE_NPERSEG), RATE) == 0.0


@pytest.mark.parametrize("length, rate", [(dsp.WHISTLE_NPERSEG - 1, RATE), (2 * dsp.WHISTLE_NPERSEG, 22050)])
def test_too_short_or_a_line_over_nyquist_reads_none(length, rate):
    assert dsp.whistle_line_db(np.ones(length), rate) is None


def test_the_floor_is_80_db_under_the_loud_frames_as_a_sine_in_one_bin():
    """A sine at the loud frames' level puts 10^8 times the floor density in its bin."""
    programme = _programme()
    levels = dsp.frame_levels(programme)
    loud = np.mean(levels[levels >= np.percentile(levels, dsp.LOUD_PERCENTILE)] ** 2)
    sine_density = loud / (dsp.WHISTLE_ENBW_BINS * RATE / dsp.WHISTLE_NPERSEG)
    assert 10.0 * np.log10(sine_density / dsp.whistle_floor_density(programme, RATE)) == pytest.approx(dsp.WHISTLE_FLOOR_DB)
