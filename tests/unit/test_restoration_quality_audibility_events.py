"""The audibility verdict beyond the 5% share: sparse events (clicks, dropouts), broken (non-finite) renders, other lengths.

A click or a dropout changes too few frames for the 5% share of design 5.2, yet a listener
hears it at once, wherever it falls in the frame grid; a NaN from a broken render and a
render cut short at either end must never read as a tie. The speech-like builders are those of
`test_restoration_quality_audibility`.
"""

import numpy as np
import pytest

from scripts.restoration_quality import auditory
from tests.unit.test_restoration_quality_audibility import RATE, _speech, _with_noise

# A hop boundary of the frame grid inside a pause of `_speech` (1.138 s, 0.538 s into a 0.6 s cycle).
HOP_BOUNDARY = 49 * auditory.NMR_HOP


def _clicked(voice, count=4):
    """`voice` with `count` 1 ms clicks at 0.3 in its pauses (0.54 s into every other 0.6 s cycle)."""
    clicked = voice.copy()
    for index in range(count):
        start = int((0.54 + 1.2 * index + 0.6) * RATE)
        end = start + 44
        clicked[start:end] = 0.3
    return clicked


def test_four_clicks_in_the_pauses_read_audible_by_the_event_clause():
    """Too few frames for the 5% share, yet each click clears the masked threshold by far more than +6 dB."""
    voice = _speech()
    result = auditory.audibility(voice, _clicked(voice), RATE)
    assert result["audible_frac"] < auditory.TIE_AUDIBLE_FRAC
    assert result["event_frames"] >= auditory.EVENT_MIN_FRAMES
    assert result["audible"] is True


@pytest.mark.parametrize(("offset", "width", "level"), [(0, 44, 0.3), (0, 1, 0.5), (-32, 44, 0.03)])
def test_a_click_on_a_hop_boundary_reads_audible_from_its_one_event_frame(offset, width, level):
    """On the boundary one Hann frame holds the click and the other almost nothing: one event frame, past +12 dB."""
    voice = _speech()
    clicked, start = voice.copy(), HOP_BOUNDARY + offset
    end = start + width
    clicked[start:end] = level
    result = auditory.audibility(voice, clicked, RATE)
    assert (result["lag"], result["event_frames"]) == (0, 1)
    assert result["audible_frac"] < auditory.TIE_AUDIBLE_FRAC
    assert result["nmr_max"] >= auditory.SINGLE_EVENT_NMR_DB
    assert result["audible"] is True


def test_a_50_ms_dropout_in_a_vowel_reads_audible_by_the_event_clause():
    """A muted 50 ms in the middle of a vowel touches about three frames, every one far over the threshold."""
    voice = _speech()
    dropped, start = voice.copy(), int(1.25 * RATE)
    end = start + int(0.05 * RATE)
    dropped[start:end] = 0.0
    result = auditory.audibility(voice, dropped, RATE)
    assert result["audible_frac"] < auditory.TIE_AUDIBLE_FRAC
    assert result["window_event_frames"][0] >= auditory.EVENT_MIN_FRAMES
    assert result["audible"] is True


def test_changes_near_the_threshold_hold_no_event_frame():
    """A coherent +1 dB caps the NMR under +6 dB and -90 dBFS noise stays under the threshold: neither is an event."""
    voice = _speech()
    for candidate in (voice * np.float32(10.0 ** (1.0 / 20.0)), _with_noise(voice, -90.0)):
        result = auditory.audibility(voice, candidate, RATE)
        assert result["event_frames"] == 0
        assert result["nmr_max"] < auditory.EVENT_NMR_DB


def test_a_nan_holed_candidate_reads_audible_and_counts_the_holes():
    """A broken render is never a tie: the hole reads as a dropout at lag 0 and the non-finite samples are counted."""
    voice = _speech()
    holed = voice.copy()
    hole = slice(RATE, RATE + 1000)
    holed[hole] = np.nan
    result = auditory.audibility(voice, holed, RATE)
    assert (result["lag"], result["nonfinite"]) == (0, 1000)
    assert result["diff_db"] > auditory.DB_FLOOR
    assert result["audible"] is True


def test_an_all_nan_candidate_is_not_read_as_identical():
    """Every sample non-finite: the candidate reads as silence against the voice, audible, every sample counted."""
    voice = _speech()
    result = auditory.audibility(voice, np.full_like(voice, np.nan), RATE)
    assert result["nonfinite"] == len(voice)
    assert result["audible"] is True


def _length_cases(voice):
    """Candidates the common span alone read as a tie: cut to half, to 1 s, to nothing, and 10 s of loud noise appended."""
    tail = (0.3 * np.random.default_rng(2).standard_normal(10 * RATE)).astype(np.float32)
    return [voice[: len(voice) // 2], voice[:RATE], voice[:0], np.concatenate([voice, tail])]


def test_a_render_cut_short_empty_or_extended_reads_audible_and_reports_the_span():
    """The NMR compares nothing past the common span, so the unmatched span decides: the pair cannot be vouched for."""
    voice = _speech()
    results = [auditory.audibility(voice, candidate, RATE) for candidate in _length_cases(voice)]
    assert [result["length_mismatch_s"] for result in results] == [3.0, 5.0, 6.0, 10.0]
    assert [result["nmr_max"] for result in results] == [auditory.DB_FLOOR] * 4
    assert [result["audible"] for result in results] == [True] * 4


def test_a_length_difference_within_the_tolerance_is_reported_and_still_a_tie():
    """100 samples (2.3 ms) off the tail and a 1-sample lag (a sample at each end) are within 50 ms: reported, still a tie."""
    voice = _speech()
    trimmed = auditory.audibility(voice, voice[:-100], RATE)
    assert trimmed["length_mismatch_s"] == 100 / RATE
    assert trimmed["audible"] is False
    shifted = auditory.audibility(voice, np.concatenate([np.zeros(1, dtype=np.float32), voice[:-1]]), RATE)
    assert (shifted["lag"], shifted["length_mismatch_s"], shifted["audible"]) == (1, 2 / RATE, False)


def _head_cases(voice, head=4000):
    """The voice missing its first `head` samples (91 ms), and the voice behind `head` samples of noise at 0.3."""
    burst = (0.3 * np.random.default_rng(4).standard_normal(head)).astype(np.float32)
    return [voice[head:], np.concatenate([burst, voice])]


def test_a_render_missing_its_head_or_with_a_burst_in_front_reads_audible():
    """The lag aligns the common span perfectly (nmr_max -200), but the 91 ms the alignment drops in front is uncompared."""
    voice = _speech()
    results = [auditory.audibility(voice, candidate, RATE) for candidate in _head_cases(voice)]
    assert [result["length_mismatch_s"] for result in results] == pytest.approx([4000 / RATE] * 2)
    readings = [(result["lag"], result["nmr_max"], result["audible"]) for result in results]
    assert readings == [(-4000, auditory.DB_FLOOR, True), (4000, auditory.DB_FLOOR, True)]


def test_a_delay_reads_its_uncompared_ends_against_the_tolerance():
    """Silence in front counts its own length (57 ms: audible, 23 ms: a tie); an equal-length 23 ms shift counts twice."""
    voice = _speech()
    late = [np.concatenate([np.zeros(delay, dtype=np.float32), voice]) for delay in (2500, 1000)]
    results = [auditory.audibility(voice, candidate, RATE) for candidate in late]
    shifted = auditory.audibility(voice, late[1][: len(voice)], RATE)
    assert [(result["lag"], result["audible"]) for result in results] == [(2500, True), (1000, False)]
    assert (shifted["lag"], shifted["length_mismatch_s"], shifted["audible"]) == (1000, 2000 / RATE, False)
