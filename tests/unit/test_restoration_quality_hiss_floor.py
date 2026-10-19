"""Ear v4's shared hiss floor: the STFT scaling, the white-noise calibration, the source anchor and its validation."""

import dataclasses
from functools import lru_cache

import numpy as np
import pytest
import scipy.signal

from scripts.restoration_quality import auditory, hiss_floor

RATE = 44100
# The hiss level that puts the synthetic capture's programme margin near the realistic-v2 fixtures' (+9 dB at 75 dB SPL).
HISS_LEVEL = 2e-4
PROG_BANDWIDTH_HZ = 4500.0


@lru_cache(maxsize=None)
def voice(seconds=30.0, seed=1, lowpass_hz=PROG_BANDWIDTH_HZ):
    """Speech-like bursts: a gliding harmonic voice (partials at 1 / k^1.5), short gaps and long pauses, low-passed."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * RATE))
    position = 0.3
    while position < seconds - 0.6:
        length = rng.uniform(0.2, 0.5)
        first, last = int(position * RATE), int((position + length) * RATE)
        t = np.arange(last - first) / RATE
        phase = 2.0 * np.pi * np.cumsum(rng.uniform(110.0, 180.0) * (1.0 + 0.2 * (t / length - 0.5))) / RATE
        burst = sum(np.sin(k * phase + rng.uniform(0.0, 2.0 * np.pi)) / k**1.5 for k in range(1, 40))
        out[first:last] += burst * np.hanning(last - first) * 0.3
        position += length + rng.choice([0.06, 0.1, 0.45, 0.8])
    sos = scipy.signal.butter(8, lowpass_hz, "low", fs=RATE, output="sos")
    return scipy.signal.sosfiltfilt(sos, out)


@lru_cache(maxsize=None)
def hiss(count, seed=2, level=HISS_LEVEL):
    """Tape-like hiss: white noise plus a first-order 2 kHz high-passed copy, so it leans to the highs."""
    white = np.random.default_rng(seed).standard_normal(count)
    sos = scipy.signal.butter(1, 2000.0, "high", fs=RATE, output="sos")
    return (0.5 * white + scipy.signal.sosfilt(sos, white)) * level


def capture(seconds=30.0):
    """`(programme, hiss, source)` of the synthetic capture."""
    programme = voice(seconds)
    noise = hiss(len(programme))
    return programme, noise, programme + noise


@lru_cache(maxsize=None)
def anchor_of(seconds=30.0, prog_bandwidth_hz=PROG_BANDWIDTH_HZ):
    """The synthetic capture's anchor."""
    return hiss_floor.source_anchor(capture(seconds)[2].astype(np.float32), RATE, prog_bandwidth_hz, None)


def context_for(source, output, anchor=None):
    """A `HissContext` for one window pair, the SPL anchors taken on the pair itself."""
    anchor = anchor_of() if anchor is None else anchor
    return hiss_floor.HissContext(anchor, None, hiss_floor.spl_offset_db(source), hiss_floor.loud_db(source), hiss_floor.loud_db(output))


def _true_band_means(noise):
    freqs, power = hiss_floor.stft_power(noise, RATE)
    index, centres = hiss_floor.band_layout(freqs, RATE)
    return hiss_floor.band_means(power, index, len(centres)), centres


def test_a_white_band_sums_to_its_mean_square():
    """The STFT scaling: the bins of white noise sum, frame by frame, to its mean square (within 0.1 dB)."""
    noise = np.random.default_rng(5).standard_normal(5 * RATE) * 0.01
    _freqs, power = hiss_floor.stft_power(noise, RATE)
    assert abs(10.0 * np.log10(power[2:-2].sum(axis=1).mean() / np.mean(noise**2))) < 0.1


def test_the_white_calibration_reproduces_the_true_mean_on_another_seed():
    """C_b maps the block floor onto the band means of a white noise it was not measured on.

    From 1 kHz up the median band reads within 0.5 dB (0.17 here) and every band within 1 dB: the 1-2.5 kHz bands hold
    8-25 bins, so the calibration's own 15 s draw and this one scatter by up to 0.74 dB there.
    """
    noise = np.random.default_rng(99).standard_normal(10 * RATE) * 0.02
    freqs, power = hiss_floor.stft_power(noise, RATE)
    index, centres = hiss_floor.band_layout(freqs, RATE)
    valley = hiss_floor.valley_levels(power, index, len(centres), hiss_floor.band_shape(power))
    floor = hiss_floor.block_floor(valley, np.ones(len(power), dtype=bool), RATE, hiss_floor.white_calibration(RATE))
    error = 10.0 * np.log10(np.median(floor, axis=0) / hiss_floor.band_means(power, index, len(centres)))
    assert np.median(np.abs(error[centres >= 1000.0])) < 0.5
    assert np.max(np.abs(error[centres >= 1000.0])) < 1.0


def test_the_anchor_reads_the_hiss_under_the_pauses_within_one_db():
    """Gapped speech-like bursts low-passed at 4.5 kHz over shaped hiss: the anchor's error is at most 1 dB from 1 kHz up."""
    anchor = anchor_of()
    truth, centres = _true_band_means(capture()[1])
    error = 10.0 * np.log10(anchor.level / truth)
    assert (anchor.how, anchor.validated) == (hiss_floor.PAUSES, True)
    assert anchor.pause_s > hiss_floor.MIN_ANCHOR_S
    assert np.nanmax(np.abs(error[centres >= 1000.0])) <= 1.0


def _muted():
    """The capture with three 2 s mutes to digital zero (a muted break): they outweigh its 7 s of natural pauses at the P10."""
    muted = capture()[2].copy()
    for start_s in (5.0, 15.0, 25.0):
        first, last = int(start_s * RATE), int((start_s + 2.0) * RATE)
        muted[first:last] = 0.0
    return muted


def test_digital_silence_in_the_pauses_falls_back_to_every_live_frame():
    """Pauses muted to digital zero hold no floor (NaN under the dither), so the anchor reads every live frame instead."""
    anchor = hiss_floor.source_anchor(_muted().astype(np.float32), RATE, PROG_BANDWIDTH_HZ, None)
    assert (anchor.how, anchor.validated) == (hiss_floor.ALL_FRAMES, False)
    assert np.isfinite(anchor.level).any()


CAP_CENTRES = np.array([500.0, 1000.0, 2000.0, 5000.0, 6000.0, 8000.0])
CAP_ANCHOR = np.array([1e-6, np.nan, 1e-8, 1e-9, 2e-9, 3e-9])


def test_hiss_cap_holds_the_bands_under_the_programme_at_the_hiss_level_plus_3_db():
    """Room tone under the programme band reads at most the median anchor above it (2e-9) plus 3 dB; NaN stays NaN."""
    capped = hiss_floor.hiss_cap(CAP_ANCHOR, CAP_CENTRES, 4500.0, 16000.0)
    held = 2e-9 * 10.0**0.3
    assert capped[[0, 2]] == pytest.approx([held, held])
    assert np.isnan(capped[1])
    assert capped[3:] == pytest.approx(CAP_ANCHOR[3:])


def test_hiss_cap_without_two_reference_bands_leaves_the_anchor():
    """A programme band reaching the top leaves under two reference bands: the anchor is returned as it is."""
    assert hiss_floor.hiss_cap(CAP_ANCHOR, CAP_CENTRES, 15000.0, 16000.0) is CAP_ANCHOR
    assert not hiss_floor.reference_bands(None, CAP_CENTRES, 4500.0, 16000.0).any()


def test_an_anchor_validates_only_from_pauses_with_a_programme_band_of_2_khz():
    """A narrow programme band or none leaves the anchor unvalidated; the level is kept."""
    narrow = hiss_floor.source_anchor(capture()[2].astype(np.float32), RATE, 1500.0, None)
    unknown = hiss_floor.source_anchor(capture()[2].astype(np.float32), RATE, None, None)
    assert (anchor_of().validated, narrow.validated, unknown.validated) == (True, False, False)
    assert np.isfinite(unknown.level).any()


def test_validation_needs_the_pauses_the_band_and_two_reference_bands():
    """`anchor_validated` asks for all three."""
    cases = [
        (hiss_floor.PAUSES, 4500.0, 2),
        (hiss_floor.ALL_FRAMES, 4500.0, 2),
        (hiss_floor.PAUSES, 1999.0, 2),
        (hiss_floor.PAUSES, 4500.0, 1),
    ]
    assert [hiss_floor.anchor_validated(*case) for case in cases] == [True, False, False, False]


def test_an_anchor_survives_its_dict_round_trip():
    """NaN is stored as null and comes back as NaN."""
    anchor = hiss_floor.Anchor(np.array([1e-8, np.nan]), np.array([1000.0, 2000.0]), 2.5, hiss_floor.PAUSES, True)
    back = hiss_floor.Anchor.from_dict(anchor.to_dict())
    assert anchor.to_dict()["level"] == [1e-8, None]
    assert np.isnan(back.level[1])
    assert (float(back.level[0]), back.pause_s, back.how, back.validated) == (1e-8, 2.5, hiss_floor.PAUSES, True)


def _second(validated):
    """A stationary anchor with one band read."""
    return hiss_floor.Anchor(np.array([2e-8]), np.array([1000.0]), 0.0, hiss_floor.STATIONARY, validated)


def test_the_programme_reading_takes_the_second_anchor_only_when_it_alone_validated():
    """A validated pause anchor stays; an unvalidated one gives way to a validated second anchor, not to an unvalidated one."""
    first = hiss_floor.Anchor(np.array([1e-8]), np.array([1000.0]), 2.5, hiss_floor.PAUSES, True, _second(True))
    unvalidated = dataclasses.replace(first, validated=False)
    neither = dataclasses.replace(unvalidated, programme=_second(False))
    alone = dataclasses.replace(unvalidated, programme=None)
    assert [anchor.for_programme().how for anchor in (first, unvalidated, neither, alone)] == [hiss_floor.PAUSES, hiss_floor.STATIONARY] + [
        hiss_floor.PAUSES
    ] * 2
    assert [anchor.programme_how() for anchor in (first, unvalidated, neither)] == [
        hiss_floor.PAUSES,
        hiss_floor.STATIONARY,
        hiss_floor.NO_ANCHOR,
    ]


def test_the_second_anchor_survives_the_dict_round_trip():
    """The nested stationary anchor is written and read back with its level and its verdict."""
    first = hiss_floor.Anchor(np.array([1e-8]), np.array([1000.0]), 0.0, hiss_floor.ALL_FRAMES, False, _second(True))
    back = hiss_floor.Anchor.from_dict(first.to_dict())
    assert (back.programme.how, back.programme.validated, float(back.programme.level[0])) == (hiss_floor.STATIONARY, True, 2e-8)


def test_an_anchor_without_a_level_stays_without_one():
    """`no_floor` round-trips with no level."""
    empty = hiss_floor.Anchor.from_dict(hiss_floor.Anchor(None, np.zeros(0), 0.0, hiss_floor.NO_FLOOR, False).to_dict())
    assert (empty.level, empty.how) == (None, hiss_floor.NO_FLOOR)
    assert empty.pause_spectrum is None


def test_the_anchor_carries_the_pause_spectrum_of_its_pause_frames():
    """The capture's anchor holds the mean power of its pause frames on the 2048-point grid, which survives the round trip
    and stands at the hiss's own level (its 2-8 kHz mean within 1 dB of the hiss alone's)."""
    anchor = anchor_of()
    freqs, power = hiss_floor.stft_power(hiss(len(capture()[2])), RATE)
    band = (freqs >= 2000.0) & (freqs < 8000.0)
    assert anchor.pause_spectrum.shape == (hiss_floor.FRAME // 2 + 1,)
    assert abs(10.0 * np.log10(anchor.pause_spectrum[band].mean() / power[:, band].mean())) < 1.0
    back = hiss_floor.Anchor.from_dict(anchor.to_dict())
    assert np.allclose(back.pause_spectrum, anchor.pause_spectrum)


def test_a_source_without_pause_frames_has_no_pause_spectrum():
    """Steady hiss alone holds no true pause (its loud frames stand under 6 dB over its quiet ones): no pause spectrum."""
    anchor = hiss_floor.source_anchor(np.asarray(hiss(20 * RATE), dtype=np.float32), RATE, PROG_BANDWIDTH_HZ, None)
    assert (anchor.pause_spectrum, anchor.how) == (None, hiss_floor.ALL_FRAMES)


def test_the_spl_offset_puts_the_loud_frames_at_the_critical_level():
    """A sine at 0.1 peak (-23 dBFS mean square) plays its loud frames at 75 dB SPL: the offset is 75 - (-23)."""
    sine = 0.1 * np.sin(2 * np.pi * 1000.0 * np.arange(RATE) / RATE)
    assert hiss_floor.L_CRIT_DB == auditory.PLAYBACK_SPL_LOUD_FRAMES + 10.0 == 75.0
    assert hiss_floor.spl_offset_db(sine) == pytest.approx(75.0 - 10.0 * np.log10(0.005), abs=0.05)


def test_a_source_too_short_or_silent_has_no_floor():
    """Under four frames there is no segment to read; digital silence everywhere leaves no band over the dither."""
    short = hiss_floor.source_anchor(np.zeros(1000, dtype=np.float32), RATE, 4500.0, None)
    silent = hiss_floor.source_anchor(np.zeros(5 * RATE, dtype=np.float32), RATE, 4500.0, None)
    assert (short.how, short.level, silent.how, silent.level) == (hiss_floor.NO_FLOOR, None, hiss_floor.NO_FLOOR, None)


def test_the_chunks_and_the_frame_map():
    """A tail under half a chunk joins the one before; the frame map reads False past the mask's end."""
    assert hiss_floor.chunk_bounds(240, 100) == [(0, 100), (100, 240)]
    assert hiss_floor.chunk_bounds(260, 100) == [(0, 100), (100, 200), (200, 260)]
    assert hiss_floor.frame_map(RATE, 4, np.array([True, False]), start=0).tolist() == [True, False, False, False]
    assert len(hiss_floor.chunked_pause_mask(np.zeros(100), RATE)) == 0


def test_the_column_statistic_the_excess_and_the_empty_pairs():
    """A NaN column reads NaN without a warning; the excess is one-sided; an unread reading is a pair of Nones."""
    stat = hiss_floor.column_stat(np.array([[1.0, np.nan], [3.0, np.nan]]), np.median)
    assert stat[0] == 2.0
    assert np.isnan(stat[1])
    assert (hiss_floor.excess(-2.0), hiss_floor.excess(1.5), hiss_floor.excess(None)) == (0.0, 1.5, None)
    assert hiss_floor.none_pairs(("a",)) == {"a": (None, None)}


def test_the_tonal_exclusion_needs_twelve_quiet_frames_and_the_layout_its_top():
    """Under 12 quiet frames no tone is looked for; the bands end at min(16 kHz, Nyquist, the brickwall)."""
    index = np.arange(10)
    assert hiss_floor.tonal_exclusion(np.ones((10, 10)), index) is index
    assert (hiss_floor.layout_top(RATE, 18000.0), hiss_floor.layout_top(16000, None)) == (hiss_floor.TOP_HZ, 8000.0)


def test_the_mask_is_read_per_chunk_and_keeps_every_frame():
    """A source longer than one chunk gets one mask frame per 20 ms, whatever the chunking."""
    source = np.tile(capture()[2], 3)[: int(70 * RATE)]
    mask = hiss_floor.chunked_pause_mask(source, RATE)
    assert len(mask) == len(source) // int(0.02 * RATE)
    assert mask.any()


def test_an_unvalidated_copy_keeps_its_level():
    """`dataclasses.replace` is how a test turns validation off; the level is the same array."""
    anchor = anchor_of()
    assert dataclasses.replace(anchor, validated=False).level is anchor.level
