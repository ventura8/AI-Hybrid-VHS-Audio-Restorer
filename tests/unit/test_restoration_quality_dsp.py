"""The DSP guardrails move the right way on controlled damage to a synthetic voice."""

import numpy as np
import pytest
import scipy.signal

from scripts.restoration_quality import dsp_metrics as dsp

RATE = 44100


def _voice(seconds=4.0, seed=3):
    """A harmonic buzz that speaks in 0.5 s bursts over a faint hiss: loud frames and true pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    phases = rng.uniform(0.0, 2 * np.pi, 40)
    voice = sum(np.sin(2 * np.pi * 173.0 * k * t + phases[k]) / k for k in range(1, 40))
    ramp = np.hanning(int(0.02 * RATE))
    gate = np.convolve(((t % 1.0) < 0.5).astype(np.float64), ramp / ramp.sum(), mode="same")
    return (0.1 * voice * gate + 1e-4 * rng.standard_normal(len(t))).astype(np.float32)


def _lowpass(mono, cutoff_hz):
    sos = scipy.signal.butter(8, cutoff_hz, btype="lowpass", fs=RATE, output="sos")
    return scipy.signal.sosfiltfilt(sos, mono).astype(np.float32)


def test_residual_noise_reads_added_hiss_and_nothing_on_identity():
    source = _voice()
    hissy = source + np.float32(3e-3) * np.random.default_rng(1).standard_normal(len(source)).astype(np.float32)
    assert dsp.residual_noise_db(source, hissy) > 6.0
    assert abs(dsp.residual_noise_db(source, source)) < 1e-6
    assert dsp.residual_noise_db(np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32)) is None


def test_hf_ratio_drops_when_the_highs_are_filtered():
    source = _voice()
    muffled = _lowpass(source, 3000.0)
    band = dsp.HF_BANDS_HZ["hf_4k8k"]
    assert dsp.hf_ratio_db(muffled, RATE, band) < dsp.hf_ratio_db(source, RATE, band) - 10.0


def test_lkr_rises_on_spectral_gating_and_is_flat_on_identity():
    source = _voice(seconds=6.0)
    freqs, _times, spec = scipy.signal.stft(source, fs=RATE, nperseg=dsp.STFT_FRAME, noverlap=dsp.STFT_FRAME - dsp.STFT_HOP)
    rng = np.random.default_rng(5)
    mask = rng.random(spec.shape) > 0.6
    _t, gated = scipy.signal.istft(spec * mask, fs=RATE, nperseg=dsp.STFT_FRAME, noverlap=dsp.STFT_FRAME - dsp.STFT_HOP)
    gated = gated[: len(source)].astype(np.float32)
    assert dsp.lkr_musical_noise(source, gated, RATE) > 0.3
    assert abs(dsp.lkr_musical_noise(source, source, RATE)) < 1e-6
    assert dsp.lkr_musical_noise(source[:2000], source[:2000], RATE) is None


def test_click_density_counts_injected_impulses():
    source = _voice()
    clicked = source.copy()
    positions = np.linspace(RATE // 10, len(source) - RATE // 10, 20).astype(int)
    clicked[positions] += 0.5
    assert dsp.click_density(clicked, RATE) >= 4.0
    assert dsp.click_density(source, RATE) < 1.0


def test_click_density_ignores_inaudible_residue_in_a_gated_pause():
    rng = np.random.default_rng(7)
    floor = (1e-6 * rng.standard_normal(4 * RATE)).astype(np.float32)
    positions = np.linspace(RATE // 10, len(floor) - RATE // 10, 20).astype(int)
    residue, clicks = floor.copy(), floor.copy()
    residue[positions] += 5e-5
    clicks[positions] += 0.05
    assert dsp.click_density(residue, RATE) == 0.0
    assert dsp.click_density(clicks, RATE) >= 4.0


def _pauses():
    rng = np.random.default_rng(1)
    t = np.arange(6 * RATE) / RATE
    speech = np.where((t % 2.0) < 1.0, 0.3 * np.sin(2 * np.pi * 220 * t), 0.0)
    return t, speech, 0.01 * rng.standard_normal(6 * RATE)


def test_pause_dynamics_read_a_gated_floor_as_pumping_and_identity_as_nothing():
    t, speech, noise = _pauses()
    source = speech + noise
    gated = speech + noise * np.where((t * 8) % 1.0 < 0.5, 1.0, 0.03)
    same = dsp.pause_dynamics(source, source, RATE)
    pumped = dsp.pause_dynamics(source, gated, RATE)
    assert same["pause_pumping_db"][0] == same["pause_pumping_db"][1]
    assert same["pause_tilt_db"][1] == 0.0
    assert pumped["pause_pumping_db"][1] > pumped["pause_pumping_db"][0] + 8.0


def test_pause_dynamics_tilt_is_positive_when_the_pauses_keep_rumble_and_lose_air():
    _t, speech, noise = _pauses()
    sos = scipy.signal.butter(4, 1000, btype="lowpass", fs=RATE, output="sos")
    dull = speech + scipy.signal.sosfiltfilt(sos, noise)
    assert dsp.pause_dynamics(speech + noise, dull, RATE)["pause_tilt_db"][1] > 10.0


def test_hum_excess_reads_a_mains_series():
    source = (1e-3 * np.random.default_rng(4).standard_normal(4 * RATE)).astype(np.float32)
    t = np.arange(len(source)) / RATE
    hum = sum(0.01 * np.sin(2 * np.pi * 50.0 * k * t) / k for k in range(1, 9)).astype(np.float32)
    assert dsp.hum_excess_db(source + hum, RATE) > dsp.hum_excess_db(source, RATE) + 6.0


def test_dropout_count_finds_a_hole_in_the_programme_only():
    source = _voice()
    hole, inside_burst, inside_pause = int(0.3 * RATE), int(0.1 * RATE), int(0.6 * RATE)
    holed = source.copy()
    holed[inside_burst:][:hole] = 0.0
    assert dsp.dropout_count(source, holed, RATE) == 1
    assert dsp.dropout_count(source, source, RATE) == 0
    paused = source.copy()
    paused[inside_pause:][:hole] = 0.0
    assert dsp.dropout_count(source, paused, RATE) == 0


def _holed(mono, start_s=0.2, ms=40.0):
    """`mono` with one hole of `ms` inside the first burst."""
    out = mono.copy()
    start, length = int(start_s * RATE), int(ms * RATE / 1000.0)
    out[start:][:length] = 0.0
    return out


def test_dropout_count_reads_a_hole_not_a_level_offset():
    """An output 20 dB down everywhere (a gain match gone wrong) holds no hole; the same with one 40 ms hole holds one."""
    source = _voice()
    quieter = (source * np.float32(0.1)).astype(np.float32)
    assert dsp.dropout_count(source, quieter, RATE) == 0
    assert dsp.dropout_count(source, _holed(quieter), RATE) == 1


def _hissing_bursts(seconds=4.0, seed=3):
    """White-noise bursts (0.5 s every 1 s) over a faint hiss: programme whose power sits above 1 kHz, as an 's' does."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    gate = ((t % 1.0) < 0.5).astype(np.float64)
    return (0.1 * gate * rng.standard_normal(len(t)) + 1e-4 * rng.standard_normal(len(t))).astype(np.float32)


def test_dropout_count_skips_a_treble_only_loss(monkeypatch):
    """100 ms with everything above 1 kHz 20 dB down and the 100-1000 Hz body kept is a treble dropout, not a hole.

    Those frames fall 13 dB broadband: with the second band moved up to where the treble went, they are one hole.
    """
    source = _hissing_bursts()
    freqs = np.fft.rfftfreq(len(source), 1.0 / RATE)
    dull = np.fft.irfft(np.fft.rfft(source) * np.where(freqs > 1000.0, 0.1, 1.0), n=len(source)).astype(np.float32)
    lifted = source.copy()
    span = slice(int(0.1 * RATE), int(0.2 * RATE))
    lifted[span] = dull[span]
    assert dsp.dropout_count(source, lifted, RATE) == 0
    assert dsp.dropout_count(source, _holed(lifted, start_s=1.2), RATE) == 1
    monkeypatch.setattr(dsp, "DROPOUT_BODY_HZ", (2000.0, 8000.0))
    assert dsp.dropout_count(source, lifted, RATE) == 1


def test_chunked_levels_equal_one_pass(monkeypatch):
    """Read 1 s at a time the frame levels are one pass's: broadband exactly, the body band within 0.01 dB."""
    source, frame = _voice(), int(dsp.DROPOUT_FRAME_S * RATE)
    sos = scipy.signal.butter(4, dsp.DROPOUT_BODY_HZ, btype="bandpass", fs=RATE, output="sos")
    whole = dsp.chunk_levels(source, RATE, frame, sos)
    monkeypatch.setattr(dsp, "DROPOUT_CHUNK_S", 1.0)
    assert np.array_equal(dsp.chunk_levels(source, RATE, frame), dsp.frame_levels(source, frame))
    chunked = dsp.chunk_levels(source, RATE, frame, sos)
    loud = whole > 1e-3
    assert np.max(np.abs(20.0 * np.log10(chunked[loud] / whole[loud]))) < 0.01


def test_without_programme_frames_the_drop_is_not_recentred():
    """No programme frame leaves the drops as they are."""
    drop = np.array([-3.0, -12.0])
    assert np.array_equal(dsp.recentred(drop, np.zeros(2, dtype=bool)), drop)
    assert np.array_equal(dsp.recentred(drop, np.array([True, False])), [0.0, -9.0])


def test_whistle_line_reads_the_crt_tone():
    source = _voice()
    t = np.arange(len(source)) / RATE
    whistling = source + (0.003 * np.sin(2 * np.pi * dsp.WHISTLE_HZ * t)).astype(np.float32)
    assert dsp.whistle_line_db(whistling, RATE) > dsp.whistle_line_db(source, RATE) + 6.0
    assert dsp.whistle_line_db(source[:1000], RATE) is None


def test_loudness_tracks_a_gain_change_and_keeps_the_range():
    source = _voice(seconds=8.0)
    lufs, lra = dsp.loudness(source, RATE)
    quieter_lufs, quieter_lra = dsp.loudness(source * np.float32(0.5), RATE)
    assert quieter_lufs == pytest.approx(lufs - 6.02, abs=0.3)
    assert quieter_lra == pytest.approx(lra, abs=0.5)
    assert dsp.loudness(source[: RATE // 2], RATE)[1] == 0.0


def test_run_counter():
    assert dsp._count_runs(np.array([True, True, False, True, True, True, False]), 2) == 2
    assert dsp._count_runs(np.array([True, False, True]), 2) == 0


def test_event_counter_and_empty_frames():
    assert dsp._count_events(np.zeros(10, dtype=bool), 3) == 0
    assert dsp._count_events(np.array([0, 1, 1, 0, 0, 0, 0, 1, 0, 0], dtype=bool), 3) == 2
    assert len(dsp.frame_levels(np.zeros(10, dtype=np.float32))) == 0
