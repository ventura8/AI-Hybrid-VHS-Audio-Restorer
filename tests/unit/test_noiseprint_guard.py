"""The mute guard's readings (`modules/noiseprint_guard.py`): what makes a candidate window a digital mute.

cathar's own picks through the guard, with the key on and off, are tested in
`test_cathar_mute_guard.py`; these are the pieces on their own.
"""

from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from modules import noiseprint_guard as guard

RATE = 8000


@pytest.mark.parametrize(
    ("level_db", "held", "p20_db", "rejected"),
    [
        (-60.0, 0.9, -40.0, True),
        (-65.0, 0.0, None, False),
        (-110.0, 0.0, None, True),
        (-80.0, 0.0, -55.0, False),
        (-90.0, 0.0, -55.0, True),
        (-82.7, 0.0, -81.8, False),
    ],
)
def test_a_window_is_a_mute_when_held_or_under_both_floors(level_db, held, p20_db, rejected):
    """Held samples at any level; under -70 dBFS only when 30 dB under the p20 or with no tape frame; an analog floor stays."""
    assert (guard.mute_reason(level_db, held, p20_db) is not None) is rejected


def test_held_samples_read_one_and_hiss_reads_none():
    """Exact zeros and a single sample read held; Gaussian hiss never repeats a value."""
    hiss = np.random.default_rng(1).normal(0.0, 1e-3, 4000).astype(np.float32)
    shares = (guard.held_share(np.zeros(4000, dtype=np.float32)), guard.held_share(hiss), guard.held_share(np.zeros(1)))
    assert shares == (1.0, 0.0, 1.0)


def test_digital_silence_has_a_finite_level_and_leaves_the_p20_undefined():
    """Exact zeros read -240 dBFS; frames all under the digital floor give no p20, frames above it give theirs."""
    assert guard.db(0.0) == pytest.approx(-240.0)
    assert guard.tape_p20_db(np.array([-120.0, -95.0])) is None
    assert guard.tape_p20_db(np.array([-120.0, -60.0, -50.0, -40.0, -30.0, -20.0])) == pytest.approx(-52.0)


def test_the_frame_is_23_ms_at_any_rate():
    """1024 samples at 44.1 kHz, the IA evidence's frame, scaled to the rate."""
    assert (guard.mute_frame(44100), guard.mute_frame(48000), guard.mute_frame(8000)) == (1024, 1115, 186)


def test_runs_of_rejected_windows_are_grouped_and_nothing_rejected_is_no_run():
    """Consecutive rejected windows form one run; no rejection, no run."""
    assert not guard.rejected_runs(np.array([False, False]))
    assert guard.rejected_runs(np.array([True, True, False, True])) == [(0, 1), (3, 3)]


def test_mute_evidence_on_material_shorter_than_a_window_has_no_windows(tmp_path):
    """Fewer hops than a window: no window to judge, as cathar's levels have none, and the p20 still read."""
    path = tmp_path / "short.wav"
    sf.write(str(path), np.full((RATE // 2, 2), 0.01, dtype=np.float32), RATE, subtype="FLOAT")
    held, p20_db = guard.mute_evidence(path, RATE // 4, 3, RATE)
    assert (len(held), round(p20_db, 2)) == (0, -40.0)


def test_mute_evidence_lays_its_windows_out_like_the_levels(tmp_path):
    """One share per window at the levels' hop: a held first second (four hops), then hiss."""
    mono = np.random.default_rng(2).normal(0.0, 0.01, 4 * RATE).astype(np.float32)
    mono[:RATE] = 0.0
    path = tmp_path / "tape.wav"
    sf.write(str(path), np.stack([mono, mono], axis=1), RATE, subtype="FLOAT")
    held, _p20_db = guard.mute_evidence(path, RATE // 4, 3, RATE)
    assert len(held) == 16 - 3 + 1
    assert (held[0], held[1], round(held[2], 3), held[-1]) == (1.0, 1.0, 0.667, 0.0)


def test_the_guarded_probe_logs_times_from_the_files_start():
    """`offset` places the searched block in the file, so logged times are file times; exact zeros read -120 dBFS, as cathar reads them."""
    mono = np.random.default_rng(3).normal(0.0, 0.01, 4 * RATE).astype(np.float32)
    mono[:RATE] = 0.0
    with patch("modules.noiseprint_guard.log_msg") as log:
        start = guard.guarded_quiet_probe(mono, RATE // 2, RATE, offset=10 * RATE)
    assert start >= RATE // 2
    assert log.call_args_list[0].args[0].startswith("    [Cathar] Noise-print mute guard: rejected the window at 10.00 s (-120.0 dBFS)")


def test_the_guarded_pause_pick_hands_only_kept_windows_to_cathars_pick(tmp_path):
    """The pick sees only the windows that are not mutes (the third still holds two hops of zeros), and its result is returned."""
    mono = np.random.default_rng(4).normal(0.0, 0.01, 4 * RATE).astype(np.float32)
    mono[:RATE] = 0.0
    path = tmp_path / "tape.wav"
    sf.write(str(path), np.stack([mono, mono], axis=1), RATE, subtype="FLOAT")
    hop = RATE // 4
    starts = np.arange(14) * hop
    levels = np.where(starts < RATE - 2 * hop, 0.0, 0.01)
    seen = {}

    def pick(kept_levels, kept_starts, window, count):
        seen.update(starts=kept_starts.tolist(), levels=kept_levels.tolist(), window=window, count=count)
        return [int(kept_starts[0])]

    with patch("modules.noiseprint_guard.log_msg"):
        picked = guard.pick_guarded_pauses(path, (levels, starts, hop, 3), 2, RATE, pick)
    assert picked == [3 * hop]
    assert (seen["starts"][0], seen["window"], seen["count"]) == (3 * hop, 3 * hop, 2)
