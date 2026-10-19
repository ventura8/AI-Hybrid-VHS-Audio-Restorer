"""Ear v4, reading 2: the hiss-weighted pause floor over the threshold in quiet, its room guards and its estimator."""

import dataclasses

import numpy as np
import pytest
import scipy.signal

from scripts.restoration_quality import hiss_floor, hiss_pause, pause_metrics
from tests.unit.test_restoration_quality_hiss_floor import PROG_BANDWIDTH_HZ, RATE, anchor_of, capture, context_for, hiss
from tests.unit.test_restoration_quality_pause_partials import melody, under_the_vad

WINDOW = slice(0, 15 * RATE)
MARGIN = "hiss_pause_margin_db"


def read(output, source=None, anchor=None):
    """The pause readings of one window: the capture's first 15 s against `output`'s (`anchor`: the capture's)."""
    source = capture()[2] if source is None else source
    src, out = source[WINDOW], np.asarray(output, dtype=np.float64)[WINDOW]
    return hiss_pause.pause_readings(src, out, RATE, context_for(src, out, anchor))


def out_of(readings, name=MARGIN):
    """The output side of one reading."""
    return readings[name][1]


def _room(count, level, seed=4):
    """Seeded pink noise through a 4th-order 1 kHz low pass (zero phase): a hiss-free room floor."""
    white = np.fft.rfft(np.random.default_rng(seed).standard_normal(count))
    pink = np.fft.irfft(white / np.sqrt(np.maximum(np.arange(len(white)), 1.0)), count)
    room = scipy.signal.sosfiltfilt(scipy.signal.butter(4, 1000.0, "low", fs=RATE, output="sos"), pink)
    return room / np.std(room) * level


def _hum(count, mains_hz=50.0, harmonics=60):
    """Mains hum with harmonics to 3 kHz, falling 1 / k."""
    t = np.arange(count) / RATE
    return sum(2e-3 / k * np.sin(2 * np.pi * mains_hz * k * t) for k in range(1, harmonics + 1))


def test_a_scaled_residual_moves_the_margin_a_db_per_db():
    """The capture's own hiss scaled down by 6, 12 and 20 dB reads the margin that much lower (within 0.3 dB)."""
    programme, noise, source = capture()
    base = out_of(read(source))
    for gain in (6.0, 12.0, 20.0):
        assert out_of(read(programme + noise * 10.0 ** (-gain / 20.0))) == pytest.approx(base - gain, abs=0.3)


def test_an_empty_pause_reads_the_cap_never_none():
    """The programme alone empties the pauses: margin L_crit - 140 = -65, onset 140, sone 0, room delta far under 0."""
    readings = read(capture()[0])
    assert out_of(readings) == hiss_floor.L_CRIT_DB - hiss_pause.ONSET_CAP == -65.0
    assert (out_of(readings, "hiss_pause_onset_db"), out_of(readings, "hiss_pause_sone")) == (140.0, 0.0)
    assert out_of(readings, "hiss_pause_excess_db") == 0.0
    assert out_of(readings, "hiss_room_delta_db") <= -54.0


def test_hum_leaves_both_sides_before_the_reading():
    """Hum harmonics to 3 kHz on both sides move the margin under 0.5 dB; hum removed on the output reads the same.

    The hummy source is read with its own anchor, as the runner reads it: the anchor's pause spectrum carries the comb,
    which the sung-run test regresses out (`pause_partials`), so no hiss-only pause leaves the mask.
    """
    programme, noise, source = capture()
    hum = _hum(len(source))
    hummy = hiss_floor.source_anchor((source + hum).astype(np.float32), RATE, PROG_BANDWIDTH_HZ, None)
    output = programme + noise * 0.25
    plain = out_of(read(output))
    assert out_of(read(output + hum, source=source + hum, anchor=hummy)) == pytest.approx(plain, abs=0.5)
    assert out_of(read(output, source=source + hum, anchor=hummy)) == pytest.approx(plain, abs=0.5)


def test_a_quiet_room_floor_is_not_hiss_and_the_room_guard_sees_it():
    """A pink floor under 1 kHz added to the output moves the margin under 0.5 dB, while the room delta rises."""
    programme, noise, _source = capture()
    output = programme + noise * 10.0 ** (-12.0 / 20.0)
    roomy = output + _room(len(output), 2e-4)
    plain, with_room = read(output), read(roomy)
    assert out_of(with_room) == pytest.approx(out_of(plain), abs=0.5)
    assert out_of(with_room, "hiss_room_delta_db") > out_of(plain, "hiss_room_delta_db") + 3.0
    assert out_of(with_room, "hiss_room_onset_db") < out_of(plain, "hiss_room_onset_db")


def test_the_block_median_of_white_noise_is_bias_corrected_to_its_mean():
    """Wilson-Hilferty with k = bins / 1.5 x 10 brings the median of 200 ms blocks within 0.3 dB of the band mean."""
    frame = int(hiss_pause.FRAME_S * RATE)
    noise = np.random.default_rng(8).standard_normal(frame * 400) * 0.01
    freqs, power = hiss_floor.ms_frames(noise, RATE, frame)
    edges, index, centres = hiss_pause.band_table(freqs, RATE, None)
    frames, kept = hiss_pause.band_frame_power(power, np.ones(len(freqs), dtype=bool), index, len(centres))
    rows, per_row = hiss_pause.block_rows(frames, [(0, 400)])
    floor = hiss_pause.stationary_floor(rows, kept, per_row)
    error = 10.0 * np.log10(floor / np.nanmean(frames, axis=0))
    assert np.nanmax(np.abs(error[centres >= 1000.0])) < 0.3
    assert len(edges) == len(centres) + 1


def test_the_hiss_weight_and_the_loudness_below_threshold():
    """w(f) reads 0.01 at 2 kHz, 0.10 at 2.5 kHz, 0.5 at 3.15 kHz and 0.91 at 4 kHz; a level under the threshold is 0 sone."""
    weights = hiss_pause.hiss_weight([2000.0, 2500.0, 3150.0, 4000.0])
    assert weights == pytest.approx([0.0125, 0.098, 0.5, 0.909], abs=0.002)
    assert hiss_pause.specific_loudness(-10.0, 0.0) == 0.0
    assert hiss_pause.specific_loudness(40.0, 0.0) > 0.0
    assert hiss_pause.wilson_hilferty_median_ratio(0.01) == pytest.approx(0.05**3)


def test_a_window_without_true_pauses_reads_nothing():
    """A held tone has no pause (no contrast), a short or broken window none either: every name None."""
    tone = 0.1 * np.sin(2 * np.pi * 440.0 * np.arange(15 * RATE) / RATE)
    assert all(pair == (None, None) for pair in read(tone, source=tone).values())
    short = np.ones(100)
    assert hiss_pause.pause_readings(short, short, RATE, context_for(short, short)) == hiss_floor.none_pairs(hiss_pause.PAUSE_NAMES)
    broken = capture()[2].copy()
    broken[10] = np.inf
    assert all(pair == (None, None) for pair in read(broken).values())


def test_the_readings_are_paired_and_the_onset_is_level_free():
    """Identity reads both sides alike; the onset is L_crit minus the margin; the coverage sits on both sides."""
    readings = read(capture()[2])
    assert readings[MARGIN][0] == readings[MARGIN][1] > 0.0
    assert out_of(readings, "hiss_pause_onset_db") == pytest.approx(hiss_floor.L_CRIT_DB - out_of(readings))
    assert readings["hiss_room_delta_db"] == (0.0, 0.0)
    assert readings["hiss_pause_s"][0] == readings["hiss_pause_s"][1] > 1.0


def test_the_band_edge_cases():
    """No room band reads None; no 5-frame block falls back to the frames; a band with no kept bin is NaN."""
    assert hiss_pause.room_level(np.ones(2), np.array([3000.0, 5000.0]), 0.0) is None
    assert hiss_pause.block_means(np.ones((4, 2)), [(0, 2), (2, 4)]) is None
    rows, per_row = hiss_pause.block_rows(np.ones((4, 2)), [(0, 2)])
    floor = hiss_pause.stationary_floor(rows, np.array([3.0, 3.0]), per_row)
    assert (per_row, floor.tolist()) == (1.0, pytest.approx([1.0 / hiss_pause.wilson_hilferty_median_ratio(2.0)] * 2))


def test_a_pause_filtered_near_to_nothing_reads_the_floor_not_below_it():
    """The capture's hiss 600 dB down (a band-filtered pause's 1e-30 residue) reads the -65 floor, never far under it."""
    programme, noise, _source = capture()
    assert out_of(read(programme + noise * 1e-30)) == hiss_floor.L_CRIT_DB - hiss_pause.ONSET_CAP


def test_the_margin_never_reads_under_its_floor_and_bands_are_smoothed():
    """A band at -1900 dB reads the -65 floor; one band 30 dB over its neighbours reads the power mean of five bands."""
    assert hiss_pause.hiss_margin(np.array([-1900.0]), np.array([3000.0])) == hiss_floor.L_CRIT_DB - hiss_pause.ONSET_CAP
    smooth = hiss_pause.smoothed(np.array([0.0, 0.0, 30.0, 0.0, 0.0, np.nan]))
    assert smooth[2] == pytest.approx(10.0 * np.log10((1.0 + 1.0 + 1000.0 + 1.0 + 1.0) / 5.0))
    assert np.isnan(smooth[5])
    assert smooth[4] == pytest.approx(10.0 * np.log10((1000.0 + 1.0 + 1.0) / 3.0))


def test_a_band_without_a_kept_bin_and_a_window_without_a_room_band():
    """The band reads NaN and counts no bin; without a 100-1500 Hz band the room delta reads None."""
    power, kept = hiss_pause.band_frame_power(np.ones((2, 3)), np.array([True, False, False]), np.array([0, 1, -1]), 2)
    assert (kept.tolist(), bool(np.isnan(power[:, 1]).all())) == ([1.0, 0.0], True)
    assert hiss_pause.room_delta((None, np.array([3000.0]), np.ones(1), np.ones(1)), context_for(np.ones(10), np.ones(10))) is None


def _hiss_pair(count):
    """`(4 s of the capture's hiss, its anchor read from its true band means, a 1.3-2.8 kHz noise in its first half)`."""
    noise = np.asarray(hiss(count))
    freqs, power = hiss_floor.stft_power(noise, RATE)
    index, centres = hiss_floor.band_layout(freqs, RATE)
    anchor = hiss_floor.Anchor(hiss_floor.band_means(power, index, len(centres)), centres, 4.0, hiss_floor.PAUSES, True)
    sos = scipy.signal.butter(4, [1300.0, 2800.0], "bandpass", fs=RATE, output="sos")
    programme = scipy.signal.sosfiltfilt(sos, np.random.default_rng(9).standard_normal(count)) * 2e-3
    return noise, anchor, programme * (np.arange(count) < count // 2)


def test_pause_blocks_holding_programme_leave_both_sides():
    """Soft programme 10-14 dB over the hiss in half the pause blocks: with the anchor those bands read the hiss alone."""
    count = 4 * RATE
    noise, anchor, programme = _hiss_pair(count)
    mask = np.ones(count // int(hiss_pause.FRAME_S * RATE), dtype=bool)
    plain = hiss_pause.pause_bands((noise, noise), RATE, mask, anchor)[2]
    for chosen, low, high in ((anchor, -1.0, 1.0), (None, 6.0, 20.0)):
        _edges, centres, src_floor, out_floor = hiss_pause.pause_bands((noise + programme, noise + programme), RATE, mask, chosen)
        rise = 10.0 * np.log10(src_floor / plain)[(centres > 2000.0) & (centres < 2800.0)]
        assert np.all((rise > low) & (rise < high))
        assert np.array_equal(src_floor, out_floor, equal_nan=True)


def test_a_window_whose_every_hiss_band_held_programme_reads_nothing():
    """Every block 20 dB over the anchor from 1 kHz up: no band is read, so the pause reading abstains."""
    count = 2 * RATE
    noise, anchor, _programme = _hiss_pair(count)
    mask = np.ones(count // int(hiss_pause.FRAME_S * RATE), dtype=bool)
    assert not hiss_pause.hiss_read(hiss_pause.pause_bands((noise * 10.0, noise * 10.0), RATE, mask, anchor))
    assert hiss_pause.hiss_read(hiss_pause.pause_bands((noise, noise), RATE, mask, anchor))


def test_the_anchor_becomes_band_power():
    """A flat anchor scales by 2048 / frame per bin and by the band's bins; under two read bands, or none, there is none."""
    anchor = hiss_floor.Anchor(np.array([1.0, 1.0, np.nan]), np.array([500.0, 4000.0, 8000.0]), 2.0, hiss_floor.PAUSES, True)
    expected = hiss_pause.anchor_band_power(anchor, np.array([800.0, 2000.0, 6000.0]), np.array([2.0, 4.0, 8.0]), 1024)
    assert expected.tolist() == pytest.approx([4.0, 8.0, 16.0])
    single = hiss_floor.Anchor(np.array([1.0]), np.array([500.0]), 0.0, hiss_floor.PAUSES, True)
    assert [hiss_pause.anchor_band_power(found, np.ones(1), np.ones(1), 1) for found in (single, None)] == [None, None]


def test_blocks_3_db_over_the_anchor_are_not_silent_from_1_khz_up():
    """Only bands of 1 kHz and up are judged; a block 3 dB or more over the anchor's band power is programme there."""
    rows = np.array([[100.0, 15.0, 33.0], [100.0, 17.0, 31.0]])
    centres = np.array([800.0, 2000.0, 6000.0])
    silent = hiss_pause.silent_rows(rows, np.array([4.0, 8.0, 16.0]), centres)
    assert silent.tolist() == [[True, True, False], [True, False, True]]
    assert hiss_pause.silent_rows(rows, None, centres).all()


def _floor_rise(monkeypatch, programme_db):
    """The 2.0-2.5 kHz source floor's rise over the hiss alone, dB, with `_hiss_pair`'s programme 16.5 dB down.

    There the programme lifts half the blocks 3.7-4.3 dB over the anchor, between the two silent tests.
    """
    monkeypatch.setattr(hiss_pause, "PROGRAMME_DB", programme_db)
    count = 4 * RATE
    noise, anchor, programme = _hiss_pair(count)
    mask = np.ones(count // int(hiss_pause.FRAME_S * RATE), dtype=bool)
    plain = hiss_pause.pause_bands((noise, noise), RATE, mask, anchor)[2]
    soft = noise + programme * 0.15
    _edges, centres, src_floor, _out_floor = hiss_pause.pause_bands((soft, soft), RATE, mask, anchor)
    return 10.0 * np.log10(src_floor / plain)[(centres > 2000.0) & (centres < 2500.0)]


def test_soft_programme_3_to_6_db_over_the_hiss_leaves_the_floor(monkeypatch):
    """Soft programme lifting half the blocks about 4 dB over the hiss (soft singing on a choir tape): it lifted the floor
    1.8-2.4 dB under the 6 dB test; at 3 dB the floor reads the hiss alone."""
    assert np.all(_floor_rise(monkeypatch, 6.0) > 1.5)
    assert np.all(np.abs(_floor_rise(monkeypatch, 3.0)) < 0.5)


def test_the_reading_judges_blocks_from_1_khz_at_3_db_and_reads_the_margin_from_2_khz():
    """The silent test runs at 3 dB from 1 kHz up (6 dB let Gaudeamus5's soft singing through); the margin starts at 2 kHz."""
    assert hiss_pause.PROGRAMME_DB == 3.0
    assert hiss_pause.SILENT_FROM_HZ == 1000.0 < hiss_pause.HISS_FROM_HZ == 2000.0


def _room_floor(count, cutoff_hz, seed=11):
    """Seeded pink noise through a 4th-order low pass at `cutoff_hz`, zero phase: a hiss-free room floor (unit rms)."""
    white = np.fft.rfft(np.random.default_rng(seed).standard_normal(count))
    pink = np.fft.irfft(white / np.sqrt(np.maximum(np.arange(len(white)), 1.0)), count)
    room = scipy.signal.sosfiltfilt(scipy.signal.butter(4, cutoff_hz, "low", fs=RATE, output="sos"), pink)
    return room / np.std(room)


def _room_level(source, room):
    """The gain that puts `room`'s 100-1000 Hz power at the source's own (the room floor as loud as the tape there)."""
    sos = scipy.signal.butter(4, [100.0, 1000.0], "bandpass", fs=RATE, output="sos")
    return float(np.sqrt(np.mean(scipy.signal.sosfiltfilt(sos, source) ** 2) / np.mean(scipy.signal.sosfiltfilt(sos, room) ** 2)))


def test_room_tone_under_2_khz_is_not_hiss_even_as_loud_as_the_tape():
    """The user allows room tone mostly under 2-3 kHz: the hiss removed and a pink floor low-passed at 2 kHz put in its
    place, as loud as the tape's own hiss at 100-1000 Hz, reads no audible pause hiss, while that hiss reads audible
    and still does 6 dB down. The room guard sees the floor kept."""
    programme, noise, source = capture()
    room = _room_floor(len(source), 2000.0)
    room = room * _room_level(noise[WINDOW], room[WINDOW])
    readings = read(programme + room)
    assert out_of(readings) <= 0.0
    assert out_of(readings, "hiss_room_delta_db") > -6.0
    assert out_of(read(source)) > out_of(read(programme + noise * 0.5)) > 0.0


def test_the_weight_comes_before_the_smoothing():
    """A floor that stops at 2.2 kHz: weighted first, its power does not reach the 2.5-2.7 kHz bands at their full weight."""
    centres = auditory_centres()
    rel = np.where(centres < 2200.0, -40.0, np.nan)
    rel[centres >= 2200.0] = -120.0
    weighted_first = hiss_pause.hiss_margin(rel, centres)
    late = (
        hiss_floor.L_CRIT_DB
        + hiss_pause.smoothed(rel)
        + 10.0 * np.log10(hiss_pause.hiss_weight(centres))
        - hiss_pause.quiet_threshold(centres)
    )
    assert weighted_first < float(np.nanmax(np.where(centres >= hiss_pause.HISS_FROM_HZ, late, np.nan))) - 3.0


def auditory_centres():
    """The pause reading's band centres at 44.1 kHz with no brickwall."""
    return hiss_pause.band_table(np.fft.rfftfreq(882, 1.0 / RATE), RATE, None)[2]


def test_pauses_standing_far_over_the_anchor_everywhere_read_nothing():
    """An anchor 30 dB under the capture's hiss makes every pause block programme from 1 kHz up: every name None."""
    src = capture()[2][WINDOW]
    low = dataclasses.replace(anchor_of(), level=anchor_of().level * 1e-3)
    assert hiss_pause.pause_readings(src, src, RATE, context_for(src, src, low)) == hiss_floor.none_pairs(hiss_pause.PAUSE_NAMES)


def _sung_capture(spans=None):
    """The capture with a melody 3 dB under its hiss in every VAD band: over its first 15 s, or over `spans` (20 ms frames)."""
    programme, noise, source = capture()
    frame = int(hiss_pause.FRAME_S * RATE)
    sung = source.copy()
    for start, stop in spans or [(0, 15 * RATE // frame)]:
        piece = slice(start * frame, stop * frame)
        sung[piece] += under_the_vad(melody(stop * frame - start * frame, seed=start), noise[piece])
    anchor = hiss_floor.source_anchor(sung.astype(np.float32), RATE, PROG_BANDWIDTH_HZ, None)
    return programme, noise, sung, anchor


def test_a_window_whose_pauses_are_sung_reads_nothing():
    """Soft singing under the VAD through the window's pauses: R4's mask and the 3 dB test still read them (`hiss_read`),
    the sung-run test leaves most runs out, and the window reads None."""
    programme, noise, sung, anchor = _sung_capture()
    src, out = sung[WINDOW], (programme + noise * 0.25)[WINDOW]
    mask = pause_metrics.true_pause_mask(src, RATE)[0]
    assert mask.sum() >= pause_metrics.MIN_PAUSE_FRAMES
    assert hiss_pause.hiss_read(hiss_pause.pause_bands((src, out), RATE, mask, anchor))
    assert not hiss_pause.unsung_mask(src, RATE, anchor).any()
    assert out_of(read(out, source=sung, anchor=anchor)) is None


def _longest_pause_run(source):
    """The `(start, stop)` 20 ms span of the longest VAD run of `source` holding true-pause frames."""
    mask, runs = pause_metrics.true_pause_mask(source, RATE)
    spans = [span for span in pause_metrics.run_spans(runs) if mask[slice(*span)].any()]
    return max(spans, key=lambda span: span[1] - span[0])


def test_one_sung_pause_leaves_only_its_run():
    """A melody in the longest pause run alone: that run's frames leave the mask, the others stay and are read."""
    longest = _longest_pause_run(capture()[2][WINDOW])
    _programme, _noise, sung, anchor = _sung_capture([longest])
    sung_mask = pause_metrics.true_pause_mask(sung[WINDOW], RATE)[0]
    kept = hiss_pause.unsung_mask(sung[WINDOW], RATE, anchor)
    inside = np.zeros(len(kept), dtype=bool)
    inside[slice(*longest)] = True
    assert sung_mask[inside].any()
    assert not kept[inside].any()
    assert np.array_equal(kept[~inside], sung_mask[~inside])
    assert out_of(read(sung, source=sung, anchor=anchor)) is not None
