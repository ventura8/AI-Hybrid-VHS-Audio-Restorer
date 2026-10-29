"""Ear v4's soft-programme guard: the programme standing only 6-18 dB over the hiss, kept (`hiss_prog_kept_soft_db`)."""

import types
from functools import lru_cache

import numpy as np
import pytest
import scipy.signal

from scripts.restoration_quality import hiss_programme
from tests.unit.test_restoration_quality_hiss_floor import RATE, capture, context_for

WINDOW = slice(0, 15 * RATE)
SOFT, KEPT_HF = "hiss_prog_kept_soft_db", "hiss_prog_kept_hf_db"
STFT = {"nperseg": 2048, "noverlap": 1024}


def read(output, name=SOFT):
    """The output side of one programme reading: the capture's first 15 s against `output`'s."""
    source = capture()[2][WINDOW]
    out = np.asarray(output, dtype=np.float64)[WINDOW]
    return hiss_programme.programme_readings(source, out, RATE, context_for(source, out))[name][1]


def _lowpass(mono, hz):
    return scipy.signal.sosfiltfilt(scipy.signal.butter(8, hz, "low", fs=RATE, output="sos"), mono)


def soft_cut(cut_db):
    """The programme alone (a perfect denoiser) with the cells the source holds 6-18 dB over the hiss, from 1 kHz up, cut
    `cut_db` (an STFT mask read on the source against the hiss's true mean per bin): an SNR-dependent suppressor."""
    programme, noise, source = capture()
    hiss = np.mean(np.abs(scipy.signal.stft(noise, RATE, **STFT)[2]) ** 2, axis=1)
    snr = 10.0 * np.log10(np.abs(scipy.signal.stft(source, RATE, **STFT)[2]) ** 2 / hiss[:, np.newaxis] + 1e-30)
    cells = (snr >= 6.0) & (snr < 18.0) & (np.fft.rfftfreq(2048, 1.0 / RATE) >= 1000.0)[:, np.newaxis]
    spectrum = scipy.signal.stft(programme, RATE, **STFT)[2]
    return scipy.signal.istft(np.where(cells, spectrum * 10.0 ** (-cut_db / 20.0), spectrum), RATE, **STFT)[1][: len(programme)]


@lru_cache(maxsize=None)
def ladder():
    """The soft reading of the identity, the hiss 6 / 12 / 20 dB down, the programme alone, and that cut 3 and 6 dB."""
    programme, noise, source = capture()
    oracle = [programme + noise * 10.0 ** (-gain / 20.0) for gain in (6.0, 12.0, 20.0)]
    return [read(output) for output in (source, *oracle, programme, soft_cut(3.0), soft_cut(6.0))]


def test_the_identity_keeps_every_soft_cell():
    """An output equal to its source reads 0 exactly."""
    assert ladder()[0] == 0.0


def test_a_perfect_denoiser_reads_only_the_hiss_share_of_the_soft_cells():
    """Removing the hiss alone takes a cell 6 dB over it about 1 dB down and one 12 dB over it 0.3 dB: the ladder reads
    -0.27 / -0.39 / -0.45 and the programme alone -0.49, each step no higher than the one before."""
    readings = ladder()[:5]
    assert all(later <= earlier + 0.05 for earlier, later in zip(readings, readings[1:]))
    assert -1.0 < readings[-1] < -0.2


def test_cutting_the_soft_programme_reads_the_cut():
    """The perfect denoiser with its soft cells cut 3 / 6 dB reads -1.90 / -3.05 against its -0.49; the 18 dB cells of the
    kept-HF guard barely see it (-0.08 / -0.11 against -0.02)."""
    perfect, cut3, cut6 = ladder()[4:]
    assert cut3 <= perfect - 1.0
    assert cut6 <= cut3 - 0.8
    assert read(soft_cut(6.0), KEPT_HF) > -0.5


def test_a_low_pass_trips_the_soft_guard_too():
    """Everything low-passed at 4 kHz reads the soft programme -6 dB: the cells above the cut are gone."""
    assert read(_lowpass(capture()[2], 4000.0)) <= -3.0


def _one_band(bins):
    """A stand-in layout: `bins` bins of one band centred at 2 kHz, a flat shape; `bins_of` reads its `kindex`."""
    layout = types.SimpleNamespace(
        kindex=np.zeros(bins, dtype=int), shape=np.ones(bins), nbins=np.array([float(bins)]), centres=np.array([2000.0]), nb=1
    )
    layout.bins_of = lambda band_mask: band_mask[layout.kindex]
    return layout


def test_hiss_alone_is_never_soft_programme():
    """Cells of hiss alone (exponential power at the floor's mean): 1.9 % stand 6 dB over it, but their 3 x 3 neighbourhood
    almost never does, so none is taken for soft programme; a contiguous stretch 10 dB over the floor is."""
    power = np.random.default_rng(4).exponential(1.0, size=(2000, 400))
    power[1000:1010, 100:120] *= 10.0
    layout, floor, active = _one_band(400), np.ones((2000, 1)), np.ones(2000, dtype=bool)
    cells = hiss_programme.soft_cells(power, floor, layout, active)
    assert np.mean(power >= 10.0**0.6) == pytest.approx(0.0187, abs=0.002)
    assert cells[:990].mean() < 1e-4
    assert cells[1001:1009, 101:119].mean() > 0.5


def test_the_floor_follows_the_band_shape():
    """A band's floor is spread over its bins by the source's shape (a roll-off inside the band is not read as programme);
    a bin out of every band, or an unread floor, takes no soft cell."""
    layout = _one_band(4)
    layout.kindex = np.array([0, 0, 0, -1])
    layout.shape = np.array([1.0, 1.0, 4.0, 9.0])
    layout.nbins = np.array([3.0])
    floor = np.array([[2.0], [np.nan]])
    shaped = hiss_programme.shaped_floor(floor, layout)
    assert shaped[0].tolist() == pytest.approx([1.0, 1.0, 4.0, np.inf])
    assert np.isinf(shaped[1]).all()


def test_too_few_soft_cells_read_nothing():
    """A band needs 20 soft cells; with none the reading is None."""
    layout = _one_band(4)
    assert hiss_programme.kept_db(np.zeros((5, 4)), np.zeros((5, 4), dtype=bool), layout) is None
    assert hiss_programme.kept_db(np.full((5, 4), -2.0), np.ones((5, 4), dtype=bool), layout) == -2.0
