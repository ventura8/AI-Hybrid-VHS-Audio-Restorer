"""The loud-frame high-band ratios abstain on a band the source's programme does not reach (hiss is not air)."""

import numpy as np

from scripts.restoration_quality import dsp_metrics

RATE = 44100


def _pair(seconds=4.0, seed=5):
    """A voice band-limited at 3.5 kHz over white hiss, and an output with the hiss removed."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    voice = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((220.0, 440.0, 880.0, 1760.0, 3300.0), start=1))
    voice *= 0.2 * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 3.0 * t)))
    hiss = 0.01 * rng.standard_normal(n)
    return (voice + hiss).astype(np.float32), voice.astype(np.float32)


def test_a_band_above_the_programme_abstains():
    source, output = _pair()
    bands = dsp_metrics.programme_band_ratios_db(source, output, RATE, 3500.0)
    assert bands["hf_8k16k"] == (None, None)
    assert bands["hf_4k8k"] == (None, None)


def test_a_band_the_programme_reaches_still_reads():
    source, output = _pair()
    bands = dsp_metrics.programme_band_ratios_db(source, output, RATE, 6000.0)
    assert bands["hf_8k16k"] == (None, None)
    assert bands["hf_4k8k"] == dsp_metrics.loud_band_ratios_db(source, output, RATE)["hf_4k8k"]


def test_without_a_profile_every_band_reads_as_before():
    source, output = _pair()
    assert dsp_metrics.programme_band_ratios_db(source, output, RATE, None) == dsp_metrics.loud_band_ratios_db(source, output, RATE)
