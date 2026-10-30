"""Ear v4, the pause reading's sung runs: held partials told from hiss by the odd and the even frames of a run."""

import numpy as np
import pytest

from scripts.restoration_quality import dsp_metrics, hiss_floor, pause_metrics, pause_partials, runner
from tests.unit.test_restoration_quality_hiss_floor import RATE, hiss

SCALE_HZ = (262.0, 294.0, 330.0, 349.0, 392.0, 440.0)
NOTE_S = 0.75
UNDER_HISS_DB = 3.0
FREQS = np.fft.rfftfreq(pause_partials.FRAME, 1.0 / RATE)
BAND = (FREQS >= pause_partials.BAND_HZ[0]) & (FREQS < pause_partials.BAND_HZ[1])
NO_CONTROL = np.zeros(len(FREQS))


def note(count, f0, glide_st=0.0, seed=0, first=1):
    """A harmonic note: partials `first`.. up to 4 kHz falling 1 / k, gliding `glide_st` semitones over its length."""
    t = np.arange(count) / RATE
    rise = 2.0 ** (glide_st * t / max(t[-1], 1e-9) / 12.0)
    phase = 2.0 * np.pi * np.cumsum(f0 * rise) / RATE
    rng = np.random.default_rng(seed)
    return sum(np.sin(k * phase + rng.uniform(0.0, 2.0 * np.pi)) / k for k in range(first, int(4000.0 // f0) + 1))


def melody(count, seed=0):
    """Notes of 0.75 s drawn from a scale, partials 2.. (the soft fundamental of a choir left out)."""
    rng = np.random.default_rng(seed)
    size = int(NOTE_S * RATE)
    out = np.zeros(count)
    for start in range(0, count, size):
        piece = slice(start, min(start + size, count))
        out[piece] = note(piece.stop - piece.start, SCALE_HZ[int(rng.integers(len(SCALE_HZ)))], seed=start, first=2)
    return out


def band_power(signal):
    """Mean power over 250-4000 Hz on the run test's frames."""
    return float(pause_partials.fine_power(signal).mean(axis=0)[BAND].sum())


def at_band_snr(signal, noise, snr_db):
    """`signal` scaled to stand `snr_db` against `noise` over 250-4000 Hz."""
    return signal * np.sqrt(band_power(noise) / band_power(signal) * 10.0 ** (snr_db / 10.0))


def _vad_band(signal, band):
    freqs, power, _level = dsp_metrics.framed_psd(np.asarray(signal, dtype=np.float64), RATE, max(1, int(pause_metrics.FRAME_S * RATE)))
    return float(power[:, (freqs >= band[0]) & (freqs < band[1])].sum(axis=1).mean())


def under_the_vad(signal, noise):
    """`signal` held 3 dB under `noise` in every band of the DSP VAD, so the pauses it sits in stay pauses."""
    gains = [np.sqrt(_vad_band(noise, band) / max(_vad_band(signal, band), 1e-30)) for band in pause_metrics.VAD_BANDS_HZ]
    return signal * min(gains) * 10.0 ** (-UNDER_HISS_DB / 20.0)


def tape_hiss(seconds, seed):
    """The synthetic capture's tape-like hiss, `seconds` long."""
    return np.asarray(hiss(int(seconds * RATE), seed=seed), dtype=np.float64)


def comb(count, scale=1.0):
    """Mains hum: 50 Hz harmonics to 3 kHz falling 1 / k, in phase (the hiss_pause hum test's)."""
    t = np.arange(count) / RATE
    return scale * sum(2e-3 / k * np.sin(2.0 * np.pi * 50.0 * k * t) for k in range(1, 61))


def r_of(segment, control=NO_CONTROL):
    """The run test's r over the whole band, no tonal bin left out."""
    return pause_partials.split_correlation(pause_partials.fine_power(segment), BAND, control)


def test_hiss_alone_holds_no_partials():
    """Tape hiss of 0.25-2 s reads r around 0 and never 0.3, the null's sd near 0.1 whatever the length."""
    found = np.array([r_of(tape_hiss(seconds, seed)) for seconds in (0.25, 0.5, 1.0, 2.0) for seed in range(30, 50)])
    assert found.max() < pause_partials.HELD_R
    assert (abs(float(np.median(found))) < 0.05, 0.05 < float(np.std(found)) < 0.15) == (True, True)


def test_a_held_note_under_the_hiss_holds_partials():
    """A note 6 dB under the hiss over 250-4000 Hz, held 1 s: r over 0.6 on every draw."""
    found = [r_of((noise := tape_hiss(1.0, seed)) + at_band_snr(note(len(noise), 220.0, seed=seed), noise, -6.0)) for seed in range(10)]
    assert min(found) > 0.6


def test_a_slow_glide_holds_and_a_fast_wide_one_does_not():
    """Both halves span the run, so a semitone's glide over 1 s holds (r over 0.5); a fifth in 0.5 s smears every partial
    over more bins than the whitening keeps and reads as noise (r under 0.3)."""
    slow, fast = [], []
    for seed in range(10):
        noise = tape_hiss(1.0, seed)
        slow.append(r_of(noise + at_band_snr(note(len(noise), 220.0, glide_st=1.0, seed=seed), noise, -6.0)))
        short = noise[: len(noise) // 2]
        fast.append(r_of(short + at_band_snr(note(len(short), 220.0, glide_st=7.0, seed=seed), short, -6.0)))
    assert min(slow) > 0.5
    assert max(fast) < pause_partials.HELD_R


def test_the_file_spectrum_regresses_a_stationary_comb_out():
    """Hum 20-35 dB over the hiss in the fine bins reads r near 0.8 alone; the file's pause spectrum regressed out, under
    0.3, while a note 6 dB under the hum and the hiss together still holds (r over 0.8)."""
    _freqs, power = hiss_floor.stft_power(tape_hiss(30.0, 99) + comb(30 * RATE), RATE)
    control = pause_partials.file_control(power.mean(axis=0), len(FREQS))
    runs = [tape_hiss(1.0, seed) + comb(RATE) for seed in range(8)]
    assert min(r_of(run) for run in runs) > 0.6
    assert max(r_of(run, control) for run in runs) < pause_partials.HELD_R
    assert min(r_of(run + at_band_snr(note(RATE, 220.0, seed=seed), run, -6.0), control) for seed, run in enumerate(runs)) > 0.8


def test_a_fitted_control_gains_nothing_on_hiss():
    """Hiss runs read against a hummy file's control: the fit per half adds no shared pattern (r stays under 0.3)."""
    _freqs, power = hiss_floor.stft_power(tape_hiss(30.0, 98) + comb(30 * RATE), RATE)
    control = pause_partials.file_control(power.mean(axis=0), len(FREQS))
    found = np.array([r_of(tape_hiss(seconds, seed), control) for seconds in (0.5, 1.0) for seed in range(60, 75)])
    assert (bool(found.max() < pause_partials.HELD_R), abs(float(np.median(found))) < 0.05) == (True, True)


def _pauses(count, spans):
    """A mask and runs that are True on `spans` (20 ms frames), nothing else."""
    mask = np.zeros(count, dtype=bool)
    for start, stop in spans:
        mask[start:stop] = True
    return mask, mask.copy()


def _with_song(noise, spans, song_spans):
    """`noise` with a melody under the VAD in `song_spans` (20 ms frame spans of `spans`)."""
    out = noise.copy()
    frame = int(pause_metrics.FRAME_S * RATE)
    for start, stop in song_spans:
        piece = slice(start * frame, stop * frame)
        out[piece] += at_band_snr(melody(stop * frame - start * frame, seed=start), noise[piece], -6.0)
    return out


SPANS = [(50, 100), (200, 240), (400, 460)]


def test_a_sung_run_leaves_the_mask_alone():
    """Of three runs one holds a melody: only its frames leave (40 of 150, under half)."""
    noise = tape_hiss(10.0, 7)
    source = _with_song(noise, SPANS, [SPANS[1]])
    pauses = _pauses(len(source) // int(pause_metrics.FRAME_S * RATE), SPANS)
    kept = pause_partials.unsung(source, RATE, pauses)
    left_out = (int(pauses[0].sum() - kept.sum()), bool(kept[200:240].any()), bool(kept[50:100].all()), bool(kept[400:460].all()))
    assert left_out == (40, False, True, True)


def test_a_sung_passage_keeps_no_pause():
    """Two of the three runs sung (100 of 150 frames): the window is a sung passage and keeps none."""
    noise = tape_hiss(10.0, 8)
    source = _with_song(noise, SPANS, SPANS[1:])
    pauses = _pauses(len(source) // int(pause_metrics.FRAME_S * RATE), SPANS)
    assert not pause_partials.unsung(source, RATE, pauses).any()


def test_short_masks_and_short_runs_are_not_judged():
    """A mask under 25 frames comes back as it is; a run under four 2048-point frames, or under four bins, reads None."""
    noise = tape_hiss(4.0, 9)
    small = _pauses(len(noise) // 882, [(10, 30)])
    assert np.array_equal(pause_partials.unsung(noise, RATE, small), small[0])
    short = _pauses(len(noise) // 882, [(10, 18), (50, 90)])
    found = pause_partials.run_correlations(noise, RATE, short)
    assert (found[0][2] is None, found[1][2] is None) == (True, False)
    assert pause_partials.split_correlation(pause_partials.fine_power(noise), np.zeros(len(FREQS), dtype=bool), NO_CONTROL) is None
    assert not pause_partials.held(None)


def test_the_windows_tones_leave_the_judged_bins():
    """A steady 1 kHz tone in the pauses: the fine bins near it (tonal_bins' +-2 bins of 50 Hz, then +-25 Hz) leave."""
    noise = tape_hiss(4.0, 10)
    t = np.arange(len(noise)) / RATE
    source = noise + 0.01 * np.sin(2.0 * np.pi * 1000.0 * t)
    mask = np.ones(len(noise) // 882, dtype=bool)
    keep = pause_partials.judged_bins(source, RATE, mask)
    assert not keep[(FREQS > 890.0) & (FREQS < 1110.0)].any()
    bands = (bool(keep[(FREQS > 1500.0) & (FREQS < 3000.0)].all()), bool(keep[FREQS < 250.0].any()), bool(keep[FREQS >= 4000.0].any()))
    assert bands == (True, False, False)


def test_no_control_and_a_constant_one_fit_nothing():
    """No control, or one of the wrong length, is zeros; a constant control leaves the values centred, no more."""
    assert not pause_partials.file_control(None, 5).any()
    assert not pause_partials.file_control(np.ones(4), 5).any()
    values = np.array([1.0, 2.0, 4.0])
    assert np.allclose(pause_partials.residual(values, np.ones(3)), values - values.mean())


def test_constant_vectors_and_empty_tone_lists():
    """A constant vector correlates 0; no tone marks no bin; a tone marks the bins within its reach."""
    assert pause_partials.correlation(np.ones(5), np.arange(5.0)) == 0.0
    assert not pause_partials.near(FREQS, [], 25.0).any()
    assert pause_partials.near(np.array([100.0, 200.0]), [190.0], 25.0).tolist() == [False, True]


def test_the_anchors_pause_spectrum_feeds_the_runner_hash():
    """The module is part of the hiss code: a change to it re-reads every cached anchor."""
    assert pause_partials in runner.HISS_CODE


@pytest.mark.parametrize("seed", [0, 1])
def test_a_melody_under_the_vad_keeps_the_pauses_pauses(seed):
    """The fixtures' melody held 3 dB under the hiss in every VAD band leaves R4's speech decision where it was."""
    noise = tape_hiss(6.0, 20 + seed)
    song = under_the_vad(melody(len(noise), seed=seed), noise)
    for band in pause_metrics.VAD_BANDS_HZ:
        assert _vad_band(song, band) <= _vad_band(noise, band) * 10.0 ** (-UNDER_HISS_DB / 10.0) * 1.0001
