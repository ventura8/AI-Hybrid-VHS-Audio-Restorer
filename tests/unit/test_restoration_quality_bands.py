"""Loud-frame band ratios and route-aware aggregation."""

import numpy as np
import scipy.signal

from scripts.restoration_quality import dsp_metrics as dsp
from scripts.restoration_quality import scorecard as sc

RATE = 44100


def _bright_voice(seconds=4.0):
    rng = np.random.default_rng(9)
    t = np.arange(int(seconds * RATE)) / RATE
    phases = rng.uniform(0.0, 2 * np.pi, 60)
    voice = sum(np.sin(2 * np.pi * 173.0 * k * t + phases[k]) / np.sqrt(k) for k in range(1, 60))
    gate = ((t % 1.0) < 0.5).astype(np.float64)
    return (0.1 * voice * gate + 2e-3 * rng.standard_normal(len(t))).astype(np.float32)


def test_loud_band_ratio_ignores_denoised_pauses_but_reads_muffled_speech():
    source = _bright_voice()
    gate = (np.arange(len(source)) / RATE % 1.0) < 0.5
    denoised = np.where(gate, source, np.float32(0.0)).astype(np.float32)
    sos = scipy.signal.butter(8, 3000.0, btype="lowpass", fs=RATE, output="sos")
    muffled = scipy.signal.sosfiltfilt(sos, source).astype(np.float32)
    clean_pauses = dsp.loud_band_ratios_db(source, denoised, RATE)["hf_4k8k"]
    lost_highs = dsp.loud_band_ratios_db(source, muffled, RATE)["hf_4k8k"]
    assert abs(clean_pauses[1] - clean_pauses[0]) < 1.0
    assert lost_highs[1] < lost_highs[0] - 10.0


def test_loud_band_ratio_on_a_short_signal_is_undefined():
    short = np.zeros(100, dtype=np.float32)
    assert dsp.loud_band_ratios_db(short, short, RATE) == {"hf_4k8k": (None, None), "hf_8k16k": (None, None)}


def test_aggregate_only_uses_rows_on_the_metric_routes():
    rows = [
        sc.WindowRow(0, 0.0, 15.0, "speech", {"dsp.hf_4k8k": 0.0}, {"dsp.hf_4k8k": -1.0}),
        sc.WindowRow(1, 7.5, 22.5, "silence", {"dsp.hf_4k8k": 0.0}, {"dsp.hf_4k8k": -30.0}),
        sc.WindowRow(2, 15.0, 30.0, "music", {"dsp.hf_4k8k": 0.0}, {"dsp.hf_4k8k": -20.0}),
    ]
    result = sc.aggregate(rows)
    assert result["dsp.hf_4k8k"]["output"]["n"] == 1
    assert result["dsp.hf_4k8k"]["delta"]["median"] == -1.0


def test_octave_ratio_ignores_the_crt_line():
    from scripts.restoration_quality import stem_metrics

    rate = 44100
    t = np.arange(2 * rate) / rate
    rng = np.random.default_rng(3)
    music = (0.05 * rng.standard_normal(len(t))).astype(np.float32)
    with_line = music + (0.5 * np.sin(2 * np.pi * 15625.0 * t)).astype(np.float32)
    # Removing the line from a capture is restoration: the worst octave must not read it as loss.
    assert stem_metrics.octave_ratio_db(with_line, music, rate) > -1.0
