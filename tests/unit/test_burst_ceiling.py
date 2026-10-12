"""The burst ceiling: hiss bursts in the gaps capped under the source's floor, programme and untouched samples kept.

The fixture is a 6 s "tape": five voiced words (harmonics of 180 Hz to 3.6 kHz) over white hiss
at -50 dBFS, and a "restored" copy that kept the words, lost the hiss and kept one 60 ms burst
of the source's own hiss after each word, as the Mel-RoFormer leaves them near speech.
"""

import json

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

from modules import burst_ceiling, event_log, hiss_floor, utils

RATE = 44100
SECONDS = 6.0
WORD_S = 0.4
PERIOD_S = 1.2
FIRST_S = 0.3
BURST_S = 0.06
HISS_DB = -50.0
DEPTH_DB = 30.0
GAPS = "gaps"
EDGES = "edges"
OPTED_IN = {"burst_ceiling": True}
EVENTS = "events"
AUDIO = "audio"
RESTORED = "restored.wav"
REFERENCE = "ref.wav"


def _word_starts():
    return np.arange(FIRST_S, SECONDS - 0.5, PERIOD_S)


def _voice(count):
    t = np.arange(count) / RATE
    tone = sum(np.sin(2.0 * np.pi * 180.0 * k * t) / k for k in range(1, 21)) * 0.05
    envelope = np.zeros(count)
    for start in _word_starts():
        lo, hi = int(start * RATE), int((start + WORD_S) * RATE)
        envelope[lo:hi] = np.hanning(hi - lo)
    return tone * envelope


def _hiss(count, seed=7):
    return np.random.default_rng(seed).standard_normal(count) * 10.0 ** (HISS_DB / 20.0)


def _burst_spans(offset_s):
    spans = []
    for start in _word_starts():
        lo = int((start + WORD_S + offset_s) * RATE)
        spans.append((lo, lo + int(BURST_S * RATE)))
    return spans


def _write(path, mono, rate=RATE):
    sf.write(str(path), np.stack([mono, mono], axis=1), rate, subtype="FLOAT")
    return path


def _tape(tmp_path, offset_s=0.25, bursts=True):
    """(reference, restored, restored samples, burst spans) for bursts `offset_s` after each word."""
    count = int(SECONDS * RATE)
    voice, hiss = _voice(count), _hiss(count)
    restored = voice.copy()
    spans = _burst_spans(offset_s)
    for lo, hi in spans if bursts else []:
        restored[lo:hi] += hiss[lo:hi]
    reference = _write(tmp_path / "reference.wav", voice + hiss)
    return reference, _write(tmp_path / RESTORED, restored), restored, spans


def _hiss_band_db(samples, spans):
    """3.5-15.5 kHz energy of the 256-sample STFT frames centred within 10 ms of the spans."""
    freqs, times, spectrum = scipy.signal.stft(samples, fs=RATE, nperseg=256, noverlap=192)
    power = (np.abs(spectrum[(freqs >= 3500.0) & (freqs < 15500.0)]) ** 2).sum(axis=0)
    keep = np.zeros(len(times), dtype=bool)
    for lo, hi in spans:
        keep |= (times >= lo / RATE - 0.01) & (times <= hi / RATE + 0.01)
    return 10.0 * np.log10(power[keep].sum() + 1e-30)


def _record(tmp_path):
    """The stage's one event-log record in the test folder's event log."""
    return json.loads(next((tmp_path / EVENTS).glob("burst_ceiling__*.json")).read_text(encoding="utf-8"))


def _capped_samples(tmp_path, scope, offset_s=0.25):
    reference, restored_wav, restored, spans = _tape(tmp_path, offset_s)
    the_plan = burst_ceiling.plan(reference, restored_wav, scope, DEPTH_DB)
    out = burst_ceiling.write_capped(restored_wav, tmp_path / f"capped_{scope}.wav", the_plan)
    return sf.read(str(out))[0][:, 0], restored, spans


@pytest.fixture(name="scoped")
def scoped_fixture(monkeypatch, tmp_path):
    """The stage's keys set to a scope, the session log in the test folder, no event log."""
    monkeypatch.setattr(utils, "LOG_FILE", tmp_path / "session_log.txt")
    monkeypatch.delenv(event_log.ENV_VAR, raising=False)

    def scope(name, depth=DEPTH_DB):
        monkeypatch.setattr(burst_ceiling, "APL_BURST_CEILING", name)
        monkeypatch.setattr(burst_ceiling, "APL_BURST_CEILING_DEPTH_DB", depth)

    return scope


def test_a_burst_in_a_gap_is_held_to_the_depth_under_the_source_floor(tmp_path):
    out, restored, spans = _capped_samples(tmp_path, GAPS)
    assert _hiss_band_db(out, spans) - _hiss_band_db(restored, spans) < -DEPTH_DB


def test_the_words_keep_their_samples_exactly(tmp_path):
    out, restored, _spans = _capped_samples(tmp_path, GAPS)
    middle = int((FIRST_S + WORD_S / 2.0) * RATE)
    window = slice(middle - 2000, middle + 2000)
    assert np.array_equal(out[window].astype(np.float32), restored[window].astype(np.float32))


def test_gaps_leaves_a_burst_inside_the_detector_s_hangover_and_edges_reaches_it(tmp_path):
    gaps, restored, spans = _capped_samples(tmp_path, GAPS, offset_s=0.0)
    edges, _restored, _spans = _capped_samples(tmp_path, EDGES, offset_s=0.0)
    drops = [_hiss_band_db(out, spans) - _hiss_band_db(restored, spans) for out in (gaps, edges)]
    assert drops[0] > -6.0
    assert drops[1] < -DEPTH_DB


def test_a_restored_file_that_kept_the_hiss_is_left_alone(tmp_path):
    reference, _restored_wav, _restored, _spans = _tape(tmp_path)
    reason = burst_ceiling.plan(reference, reference, GAPS, DEPTH_DB)
    assert reason == "the neural stage left every band's gaps within 30 dB of the source's floor"


def test_a_file_with_no_burst_has_nothing_over_the_ceiling(tmp_path):
    reference, restored_wav, _restored, _spans = _tape(tmp_path, bursts=False)
    assert burst_ceiling.plan(reference, restored_wav, GAPS, DEPTH_DB) == "nothing stands over the ceiling"


def test_an_unrelated_file_is_not_matched(tmp_path):
    reference, _restored_wav, _restored, _spans = _tape(tmp_path)
    other = _write(tmp_path / "other.wav", _hiss(int(SECONDS * RATE), seed=3) * 100.0)
    assert burst_ceiling.plan(reference, other, GAPS, DEPTH_DB) == "the restored programme does not follow the source's"


def test_a_tape_of_speech_alone_has_too_few_gaps(tmp_path, monkeypatch):
    reference, restored_wav, _restored, _spans = _tape(tmp_path)
    monkeypatch.setattr(hiss_floor, "programme_frames", lambda vad, _rate, _frame: np.ones(len(vad), dtype=bool))
    assert burst_ceiling.plan(reference, restored_wav, GAPS, DEPTH_DB) == "too few gaps to read the restored floor in"


def test_pairs_that_cannot_be_read_together_say_why(tmp_path):
    reference, _restored_wav, restored, _spans = _tape(tmp_path)
    slow = _write(tmp_path / "slow.wav", restored[::2], rate=RATE // 2)
    short = _write(tmp_path / "short.wav", restored[:-RATE])
    tiny = _write(tmp_path / "tiny.wav", restored[:2000])
    reasons = [burst_ceiling.plan(reference, path, GAPS, DEPTH_DB) for path in (slow, short)]
    reasons.append(burst_ceiling.plan(tiny, tiny, GAPS, DEPTH_DB))
    expected = ["the restored audio's rate differs from the source's", "reference and restored audio differ in length", "too short to read"]
    assert reasons == expected


def test_the_ceiling_keeps_programme_lets_its_neighbours_keep_their_source_and_caps_hiss():
    """Band 3 carries programme 20 dB over the floor; bands 2 and 4 keep up to their own source level, bands 0 and 6 hiss alone."""
    floor = np.ones((5, 7))
    src = np.ones((5, 7))
    src[:, 3] = 100.0
    ceiling, share = burst_ceiling.ceiling_db(src, floor, DEPTH_DB)
    restored_db = 10.0 * np.log10(50.0)
    cut = burst_ceiling.cut_db(np.full((5, 7), 50.0), ceiling, share)
    assert np.allclose(share[:, 2:5], 1.0)
    assert np.allclose(cut[:, 3], 0.0)
    assert np.allclose(cut[:, [2, 4]], burst_ceiling.HEADROOM_DB - restored_db)
    assert np.allclose(cut[:, [0, 6]], -DEPTH_DB - restored_db)


def _cell_share(level_db):
    """The share of band 3 standing level_db over a floor its six neighbours sit at."""
    envelope = np.ones((1, 7))
    envelope[0, 3] = 10.0 ** (level_db / 10.0)
    return float(burst_ceiling.programme_share(envelope, np.ones((1, 7)))[0, 3])


def test_a_cell_alone_counts_as_programme_from_ten_decibels_surely_from_thirteen():
    assert np.allclose([_cell_share(level) for level in (9.0, 11.5, 13.0, 20.0)], [0.0, 0.5, 1.0, 1.0])


def test_a_neighbourhood_counts_as_programme_from_six_decibels():
    shares = [
        float(burst_ceiling.programme_share(np.full((1, 3), 10.0 ** (level / 10.0)), np.ones((1, 3)))[0, 1]) for level in (5.0, 7.5, 9.0)
    ]
    assert np.allclose(shares, [0.0, 0.5, 1.0])


def test_the_envelope_is_closed_one_dip_filled_and_edges_kept():
    closed = burst_ceiling.closed_envelope(np.array([1, 1, 9, 1, 9, 9, 9, 1, 1], dtype=float)[:, None])
    assert closed[:, 0].tolist() == [1, 1, 9, 9, 9, 9, 9, 1, 1]


def test_a_hiss_cell_takes_the_deepest_cut_beside_it_and_a_programme_cell_keeps_its_own():
    restored = np.array([1e-9, 1.0, 1e-9, 1e-9])[:, None]
    ceiling = np.full((4, 1), -DEPTH_DB)
    share = np.array([0.0, 0.0, 0.0, 0.5])[:, None]
    cut = burst_ceiling.cut_db(restored, ceiling, share)[:, 0]
    assert np.allclose(cut, [-30.0, -30.0, -30.0, 0.0])


def test_the_scope_weight_covers_the_gaps_or_the_edges_too():
    speech = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
    block_s = burst_ceiling.EDGE_REACH_S / 2.0
    gaps = burst_ceiling.scope_weight(speech, GAPS, block_s)
    edges = burst_ceiling.scope_weight(speech, EDGES, block_s)
    assert np.allclose(gaps, [0, 0, 0, 1 / 3, 2 / 3, 2 / 3, 1 / 3, 0, 0, 0])
    assert np.allclose(edges, [0, 1 / 3, 2 / 3, 1, 1, 1, 1, 2 / 3, 1 / 3, 0])


def test_the_background_needs_enough_gap():
    restored, floor = np.ones((20, 2)), np.ones((20, 2))
    speech = np.ones(20, dtype=bool)
    speech[:3] = False
    assert burst_ceiling.gap_background_db(restored, floor, speech, 0.05) is None
    assert np.allclose(burst_ceiling.gap_background_db(restored, floor, speech, 0.1), 0.0)


def test_a_band_acts_only_over_dither_and_where_the_gaps_were_emptied():
    stft = burst_ceiling.stft_for(RATE)
    bands = len(stft.edges) - 1
    floor = np.full((10, bands), 1e-2)
    floor[:, 0] = 1e-20
    background = np.full(bands, -40.0)
    background[1] = -10.0
    active = burst_ceiling.active_bands(background, floor, stft, DEPTH_DB)
    assert active.tolist() == [False, False] + [True] * (bands - 2)


def test_gains_reach_no_bin_outside_three_to_sixteen_khz():
    stft = burst_ceiling.stft_for(RATE)
    gains = burst_ceiling.bin_gains(np.full((2, len(stft.edges) - 1), -20.0), stft)
    assert np.count_nonzero(gains[:, stft.band < 0]) == 0
    assert np.allclose(gains[:, stft.band >= 0], 10.0 ** (-1.0) - 1.0)


def _constant_plan(count, cut):
    stft = burst_ceiling.stft_for(RATE)
    blocks = -(-count // burst_ceiling.block_samples(stft))
    bands = len(stft.edges) - 1
    return burst_ceiling.Plan(np.full((blocks, bands), cut), stft, RATE, np.ones(bands, dtype=bool), np.zeros(bands), 0.0)


def _band_change_db(before, after, lo, hi):
    freqs = np.fft.rfftfreq(len(before), 1.0 / RATE)
    keep = (freqs >= lo) & (freqs < hi)
    return 10.0 * np.log10(np.sum(np.abs(np.fft.rfft(after)[keep]) ** 2) / np.sum(np.abs(np.fft.rfft(before)[keep]) ** 2))


def test_a_constant_cut_lowers_the_band_by_exactly_that_and_nothing_else(tmp_path):
    noise = _hiss(3 * RATE, seed=1) * 300.0
    source = _write(tmp_path / "noise.wav", noise)
    out = sf.read(str(burst_ceiling.write_capped(source, tmp_path / "out.wav", _constant_plan(len(noise), -12.0))))[0][:, 0]
    changes = [round(_band_change_db(noise, out, lo, hi), 2) for lo, hi in ((500.0, 2500.0), (3500.0, 15500.0), (16500.0, 20000.0))]
    assert changes == [0.0, -12.0, 0.0]


def test_the_written_audio_does_not_depend_on_the_block_size(tmp_path, monkeypatch):
    noise = _hiss(2 * RATE, seed=2) * 300.0
    source = _write(tmp_path / "noise.wav", noise)
    plan = _constant_plan(len(noise), 0.0)
    plan.cut_db[30:40] = -25.0
    whole = sf.read(str(burst_ceiling.write_capped(source, tmp_path / "whole.wav", plan)))[0]
    monkeypatch.setattr(burst_ceiling, "BLOCK_SAMPLES", 3001)
    pieces = sf.read(str(burst_ceiling.write_capped(source, tmp_path / "pieces.wav", plan)))[0]
    assert np.array_equal(whole, pieces)


def test_frames_over_a_span_are_every_window_that_reaches_it():
    stft = burst_ceiling.stft_for(RATE)
    centres = burst_ceiling.frames_over(10000, 10001, stft)
    half = stft.size // 2
    assert (centres.min() - half < 10001, centres.max() + half > 10000, len(centres)) == (True, True, stft.size // stft.hop)


def test_the_capped_spans_are_the_blocks_that_lose_over_a_decibel():
    plan = _constant_plan(10 * 1024, 0.0)
    plan.cut_db[2:4, 5] = -3.0
    plan.cut_db[6, 5] = -0.5
    size = burst_ceiling.block_samples(plan.stft)
    assert burst_ceiling.capped_spans(plan) == [(2 * size, 4 * size)]


def test_cap_writes_once_reuses_after_and_logs_each(tmp_path, scoped):
    scoped(GAPS)
    reference, restored_wav, _restored, _spans = _tape(tmp_path)
    first = burst_ceiling.cap(reference, restored_wav, tmp_path / AUDIO)
    again = burst_ceiling.cap(reference, restored_wav, tmp_path / AUDIO)
    log = (tmp_path / "session_log.txt").read_text(encoding="utf-8")
    assert (first, first.name) == (again, "capped_gaps30_restored.wav")
    assert (first.parent.name, "[Burst Ceiling] Capping" in log, "[Burst Ceiling] Reusing" in log) == ("burst_ceiling", True, True)


def test_cap_records_the_capped_blocks_in_the_event_log(tmp_path, scoped, monkeypatch):
    scoped(EDGES, depth=24.0)
    monkeypatch.setenv(event_log.ENV_VAR, str(tmp_path / EVENTS))
    reference, restored_wav, _restored, spans = _tape(tmp_path)
    burst_ceiling.cap(reference, restored_wav, tmp_path / AUDIO)
    record = _record(tmp_path)
    settings = (record["scope"], record["depth_db"], record["programme_db"], record["programme_cell_db"], record["headroom_db"])
    assert settings == (EDGES, 24.0, burst_ceiling.PROGRAMME_DB, burst_ceiling.PROGRAMME_CELL_DB, burst_ceiling.HEADROOM_DB)
    assert record["count"] >= len(spans)
    assert len(record["active_bands_hz"]) == len(record["background_db"])


def test_a_skip_returns_the_input_and_records_why(tmp_path, scoped, monkeypatch):
    scoped(GAPS)
    monkeypatch.setenv(event_log.ENV_VAR, str(tmp_path / EVENTS))
    reference, restored_wav, _restored, _spans = _tape(tmp_path, bursts=False)
    assert burst_ceiling.cap(reference, restored_wav, tmp_path / AUDIO) == restored_wav
    record = _record(tmp_path)
    assert record["skipped"] == "nothing stands over the ceiling"


def test_switched_off_or_not_opted_in_nothing_is_read(tmp_path, scoped, monkeypatch):
    monkeypatch.setattr(burst_ceiling, "plan", lambda *_args: pytest.fail("read"))
    restored = tmp_path / RESTORED
    scoped("none")
    off = burst_ceiling.apply_when_needed(tmp_path / REFERENCE, restored, tmp_path, OPTED_IN)
    scoped(GAPS)
    not_opted = burst_ceiling.apply_when_needed(tmp_path / REFERENCE, restored, tmp_path, {})
    assert (off is restored, not_opted is restored) == (True, True)


def test_a_failure_leaves_the_audio_and_records_it(tmp_path, scoped, monkeypatch):
    scoped(GAPS)
    monkeypatch.setenv(event_log.ENV_VAR, str(tmp_path / EVENTS))
    restored = tmp_path / RESTORED
    monkeypatch.setattr(burst_ceiling, "plan", lambda *_args: (_ for _ in ()).throw(OSError("disk")))
    assert burst_ceiling.apply_when_needed(tmp_path / REFERENCE, restored, tmp_path, OPTED_IN) is restored
    record = _record(tmp_path)
    assert record["skipped"] == "failed: disk"


def test_an_opted_in_scope_runs_the_stage(tmp_path, scoped):
    scoped(GAPS)
    reference, restored_wav, _restored, _spans = _tape(tmp_path)
    out = burst_ceiling.apply_when_needed(reference, restored_wav, tmp_path / AUDIO, OPTED_IN)
    assert out.name == "capped_gaps30_restored.wav"
