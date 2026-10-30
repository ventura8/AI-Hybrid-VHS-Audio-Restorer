"""The kept-HF guard leaves out coded bands, whose bins do not spread like noise in the source's quiet frames (2026-10-21)."""

import numpy as np
import pytest

from scripts.restoration_quality import hiss_floor
from scripts.restoration_quality import hiss_programme as hp

RATE = 44100
CODED_HZ = (9000.0, 13000.0)
PROGRAMME_HZ = (4000.0, 5000.0)


def _noise(seconds=6.0, seed=3):
    return np.random.default_rng(seed).standard_normal(int(seconds * RATE)) * 1e-3


def _thinned(power, freqs, share=0.6, first=0, seed=5):
    """`power` with `share` of the bins in CODED_HZ dropped 40 dB in every frame from `first` on: a coder's sparse HF."""
    out = power.copy()
    sel = np.flatnonzero((freqs >= CODED_HZ[0]) & (freqs < CODED_HZ[1]))
    rng = np.random.default_rng(seed)
    for frame in range(first, len(out)):
        out[frame, rng.choice(sel, int(share * len(sel)), replace=False)] *= 1e-4
    return out


@pytest.fixture(name="white")
def _white():
    freqs, power = hiss_floor.stft_power(_noise(), RATE)
    return freqs, power, np.ones(len(power), dtype=bool)


def _inside(layout, span):
    return (layout.centres > span[0] + 600.0) & (layout.centres < span[1] - 600.0)


def test_white_noise_holds_no_coded_band(white):
    freqs, power, active = white
    assert not hp.coded_bands(power, hp.Layout(freqs, power, RATE, None), active).any()


def test_a_band_a_coder_thinned_is_coded_and_the_bands_below_it_are_not(white):
    freqs, power, active = white
    thinned = _thinned(power, freqs)
    layout = hp.Layout(freqs, thinned, RATE, None)
    coded = hp.coded_bands(thinned, layout, active)
    assert (bool(coded[_inside(layout, CODED_HZ)].all()), bool(coded[layout.centres < 8000.0].any())) == (True, False)


def _kept_case(white):
    """A source whose HF is coded, under a floor 30 dB too low there, and loud programme at 4-5 kHz kept whole; the output
    removes 30 dB of the coded band's hiss."""
    freqs, power, active = white
    src = _thinned(power, freqs)
    src[: len(src) // 2, (freqs >= PROGRAMME_HZ[0]) & (freqs < PROGRAMME_HZ[1])] *= 1e4
    out = src.copy()
    out[:, (freqs >= CODED_HZ[0]) & (freqs < CODED_HZ[1])] *= 1e-3
    layout = hp.Layout(freqs, src, RATE, None)
    floor = np.tile([power[:, layout.kindex == band].mean() for band in range(layout.nb)], (len(src), 1))
    floor[:, _inside(layout, CODED_HZ)] *= 1e-3
    return (src, out), floor, layout, active


def test_hiss_removed_from_a_coded_band_is_not_read_as_lost_highs(white):
    powers, floor, layout, active = _kept_case(white)
    assert abs(hp.kept_readings(powers, floor, layout, active)["kept_hf_db"]) < 0.5


def test_the_coded_bands_are_out_of_the_loud_floor(white):
    powers, floor, layout, active = _kept_case(white)
    coded = hp.coded_bands(powers[0], layout, active)
    loud = hp.loud_floor(powers[0], floor, layout, active)
    assert (bool(np.isnan(loud[:, coded]).all()), bool(np.isfinite(loud[:, ~coded & (layout.nbins > 0)]).all())) == (True, True)


def test_without_the_coded_test_and_the_quiet_bound_the_same_case_reads_dull(white, monkeypatch):
    powers, floor, layout, active = _kept_case(white)
    monkeypatch.setattr(hp, "coded_bands", lambda *_args: np.zeros(layout.nb, dtype=bool))
    monkeypatch.setattr(hp, "quiet_bound", lambda _src, floor, *_args: floor)
    assert hp.kept_readings(powers, floor, layout, active)["kept_hf_db"] < -10.0


def test_the_soft_reading_keeps_the_coded_bands(white, monkeypatch):
    powers, floor, layout, active = _kept_case(white)
    soft = hp.kept_readings(powers, floor, layout, active)["kept_soft_db"]
    monkeypatch.setattr(hp, "coded_bands", lambda *_args: np.zeros(layout.nb, dtype=bool))
    assert hp.kept_readings(powers, floor, layout, active)["kept_soft_db"] == soft


def _low_floor_case(white):
    """A clean source (no coded band) whose floor reads its hiss 15 dB low, as a low anchor does; the output removes
    20 dB of the hiss above 6 kHz, keeping the loud 4-5 kHz programme."""
    freqs, power, active = white
    src = power.copy()
    src[: len(src) // 2, (freqs >= PROGRAMME_HZ[0]) & (freqs < PROGRAMME_HZ[1])] *= 1e4
    out = src.copy()
    out[:, freqs >= 6000.0] *= 1e-2
    layout = hp.Layout(freqs, src, RATE, None)
    floor = np.tile([power[:, layout.kindex == band].mean() * 10**-1.5 for band in range(layout.nb)], (len(src), 1))
    return (src, out), floor, layout, active


def test_a_floor_read_low_is_held_at_the_band_s_quiet_level(white):
    powers, floor, layout, active = _low_floor_case(white)
    assert abs(hp.kept_readings(powers, floor, layout, active)["kept_hf_db"]) < 0.5


def test_the_quiet_bound_leaves_a_true_floor_alone(white):
    powers, floor, layout, active = _low_floor_case(white)
    true_floor = floor * 10**1.5
    assert np.allclose(hp.quiet_bound(powers[0], true_floor, layout, active), true_floor, equal_nan=True)


def test_with_few_active_frames_every_frame_counts():
    means = np.arange(20, dtype=float)
    active = np.zeros(20, dtype=bool)
    active[:3] = True
    assert hp.quiet_frames(means, active).sum() == 6
