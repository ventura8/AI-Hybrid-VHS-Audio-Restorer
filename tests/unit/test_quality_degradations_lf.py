"""R11's calibration entries: FFmpeg's `bass` shelf and `highpass` as the generators, and the low end read in order on each."""

import numpy as np
import pytest
import scipy.signal

from scripts import degradations_v3 as v3
from scripts import quality_degradations as deg
from scripts.restoration_quality import audio_io, lf_metrics
from tests.unit.test_restoration_quality_lf import RATE, _source

LF_ENTRIES = ("lf_shelf_boost_music", "lf_shelf_cut_music", "lf_highpass_music", "lf_highpass")


def _response_db(b, a, freqs):
    _w, response = scipy.signal.freqz(b, a, worN=np.asarray(freqs, dtype=np.float64), fs=RATE)
    return 20.0 * np.log10(np.abs(response))


@pytest.mark.parametrize("gain_db", [-6.0, -3.0, 1.5, 6.0])
def test_the_bass_shelf_is_the_rbj_low_shelf_ffmpeg_runs(gain_db):
    """Its full gain at DC, half of it in dB at the 100 Hz corner, nothing at Nyquist: the mirror of the treble shelf."""
    b, a = v3.bass_coefficients(gain_db, RATE)
    low, corner, top = _response_db(b, a, [1e-3, v3.LF_SHELF_HZ, RATE / 2.0 - 1.0])
    assert abs(low - gain_db) < 1e-3
    assert abs(corner - gain_db / 2.0) < 1e-6
    assert abs(top) < 1e-3
    assert a[0] == 1.0


def _spectrum_ratio_db(source, output, freqs_hz):
    """The output's FFT over the source's, in dB, at the bins nearest `freqs_hz`."""
    freqs = np.fft.rfftfreq(len(source), 1.0 / RATE)
    ratio = np.abs(np.fft.rfft(output.astype(np.float64))) / np.abs(np.fft.rfft(source.astype(np.float64)))
    return 20.0 * np.log10(ratio[np.searchsorted(freqs, freqs_hz)])


def test_the_bass_shelf_applies_ffmpegs_magnitude_with_no_phase_and_keeps_the_length():
    """`lf_shelf` gives every bin the shelf's own gain, float32 out, the same length, and no lag for the alignment to find."""
    source = _source()
    out = v3.lf_shelf(source, RATE, -3.0)
    b, a = v3.bass_coefficients(-3.0, RATE)
    probes = np.array([45.0, 90.0, 135.0, 450.0])
    assert out.dtype == np.float32
    assert len(out) == len(source)
    assert np.allclose(_spectrum_ratio_db(source, out, probes), _response_db(b, a, probes), atol=1e-3)
    assert audio_io.align_pair(source, out)[2] == 0


@pytest.mark.parametrize("corner_hz", deg.LF_HIGHPASS_HZ)
def test_the_high_pass_is_ffmpegs_two_pole_butterworth(corner_hz):
    """3 dB down at its corner, 12 dB per octave below it, flat above: FFmpeg's `highpass=f=` at Q 1/sqrt(2)."""
    b, a = scipy.signal.butter(v3.LF_HIGHPASS_ORDER, corner_hz, btype="highpass", fs=RATE)
    response = _response_db(b, a, [corner_hz, corner_hz / 2.0, corner_hz * 20.0])
    assert np.allclose(response, [-3.0103, -12.3, 0.0], atol=0.6)
    assert abs(response[0] + 3.0103) < 0.01


@pytest.mark.parametrize("corner_hz", deg.LF_HIGHPASS_HZ)
def test_the_high_pass_applies_its_magnitude_with_no_phase(corner_hz):
    """`lf_highpass` gives every bin the filter's own gain and leaves the alignment no lag to find."""
    b, a = scipy.signal.butter(v3.LF_HIGHPASS_ORDER, corner_hz, btype="highpass", fs=RATE)
    source = _source()
    out = v3.lf_highpass(source, RATE, corner_hz)
    probes = np.array([45.0, 90.0, 180.0])
    assert np.allclose(_spectrum_ratio_db(source, out, probes), _response_db(b, a, probes), atol=1e-3)
    assert audio_io.align_pair(source, out)[2] == 0


def test_the_low_end_entries_sweep_the_shelf_both_ways_and_round_c3s_corners():
    """The cut mirrors the boost (-6 / -3 dB and +3 among the levels), the high pass runs 40 / 60 / 80 Hz, mild to severe."""
    levels = {name: deg.DEGRADATIONS[name].levels for name in LF_ENTRIES}
    assert levels == {
        "lf_shelf_boost_music": (1.5, 3.0, 6.0),
        "lf_shelf_cut_music": (-1.5, -3.0, -6.0),
        "lf_highpass_music": (40.0, 60.0, 80.0),
        "lf_highpass": (40.0, 60.0, 80.0),
    }
    assert [deg.DEGRADATIONS[name].base for name in LF_ENTRIES] == ["music", "music", "music", "speech"]


@pytest.mark.parametrize("name", LF_ENTRIES)
def test_each_low_end_entry_asserts_r11_and_holds_the_top_and_the_texture(name):
    """R11 asserted (never blind), R1's body reported, the tilt above 1 kHz and the musical-noise reading held flat."""
    expects = {e.metric: (e.direction, e.blind) for e in deg.DEGRADATIONS[name].expects}
    direction = "up" if "boost" in name else "down"
    held = {"dsp.balance_body_db": (direction, True), deg.TILT: ("flat", False), deg.LKR: ("flat", False)}
    assert expects == {deg.LF_PROGRAMME: (direction, False), **held}


@pytest.mark.parametrize("name", LF_ENTRIES)
def test_r11_reads_each_entry_in_order_of_its_levels(name):
    """On the R11 tests' bass-led programme, each entry's three levels read strictly in order, the right way."""
    source = _source()
    spec = deg.DEGRADATIONS[name]
    materials = {"music": source, "speech": source}
    readings = []
    for level in spec.levels:
        base, output = deg.apply(name, level, materials[spec.base], RATE, materials, np.random.default_rng(3))
        readings.append(lf_metrics.lf_programme_db(base, output, RATE))
    signed = readings if "boost" in name else [-value for value in readings]
    assert 0.0 < signed[0] < signed[1] < signed[2]


def test_only_the_speech_high_pass_runs_on_a_tape_excerpt():
    """A tape cut is speech: the music-bed entries need the fixtures' bed, the speech high pass needs nothing."""
    on_tape = set(deg.on_tape())
    assert "lf_highpass" in on_tape
    assert not on_tape & {"lf_shelf_boost_music", "lf_shelf_cut_music", "lf_highpass_music"}
