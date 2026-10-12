"""cathar's noise-print depth guard (`modules/noiseprint_depth.py`): the depth measure, the scaled print and the drop.

The depth is the print's window(s) under the material's loud frames, both on the mono mix: the
loud frames are the p90 of the 1024-sample frame RMS, streamed in blocks, and the windows' level
is the RMS over all of them together. The signals below have frames of known RMS, so every
number is checked against arithmetic done here, not against the module.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from modules import noiseprint_depth as npd

RATE = 8000
FRAME = npd.LOUD_FRAME
# 300 frames of 1024 samples: more than one streamed block (240 frames), so the stream is tested across a seam.
FRAME_LEVELS = np.arange(1, 301) / 300.0
PRINT = [0.25 * (1 + k % 7) for k in range(1025)]
GUARD = npd.LOG_PREFIX


def _steps(path, levels=FRAME_LEVELS, left_gain=2.0):
    """Frame k holds the constant level `levels[k]` on the mono mix: the left channel at left_gain x, the right at the rest."""
    mono = np.repeat(levels, FRAME)
    left = mono * left_gain
    sf.write(str(path), np.stack([left, 2.0 * mono - left], axis=1).astype(np.float32), RATE, subtype="FLOAT")
    return path


def _print(tmp_path, values=None):
    path = tmp_path / "noise_x_single_0.75s.np.json"
    path.write_text(json.dumps(PRINT if values is None else values), encoding="utf-8")
    return path


def _guarded(tmp_path, windows, min_depth_db, action="scale", values=None):
    """guard_print on the step signal with a synthetic print: (result, the print it was given, the log's messages)."""
    print_json = _print(tmp_path, values)
    log = MagicMock()
    with patch("modules.noiseprint_depth.log_msg", log):
        result = npd.guard_print(print_json, _steps(tmp_path / "steps.wav"), lambda: windows, min_depth_db, action)
    return result, print_json, [call.args[0] for call in log.call_args_list], log


def _depth_of(frames):
    """The depth of windows covering these whole frames of the step signal, by arithmetic."""
    loud = np.percentile(FRAME_LEVELS, 90.0)
    probe = np.sqrt(np.mean(FRAME_LEVELS[list(frames)] ** 2))
    return 20.0 * np.log10(loud / probe)


def test_the_loud_level_is_the_p90_of_the_frame_rms_across_blocks(tmp_path):
    """300 frames at 1/300 .. 1: the p90 of their RMS, read over two streamed blocks of the mono mix."""
    assert npd.loud_level(_steps(tmp_path / "steps.wav")) == pytest.approx(np.percentile(FRAME_LEVELS, 90.0), rel=1e-6)


def test_the_loud_level_reads_the_mono_mix_not_the_left_channel(tmp_path):
    """The left channel alone is twice as loud; the mix is what the pick reads, so the level is the mix's."""
    assert npd.loud_level(_steps(tmp_path / "steps.wav", left_gain=1.0)) == pytest.approx(npd.loud_level(_steps(tmp_path / "b.wav")))


def test_the_loud_level_drops_a_partial_last_frame(tmp_path):
    """A tail shorter than one frame is not a frame (as the window levels drop theirs)."""
    path = tmp_path / "tail.wav"
    mono = np.concatenate([np.full(FRAME, 0.5), np.full(FRAME // 2, 0.9)])
    sf.write(str(path), np.stack([mono, mono], axis=1).astype(np.float32), RATE, subtype="FLOAT")
    assert npd.loud_level(path) == pytest.approx(0.5)


def test_material_shorter_than_a_frame_has_no_loud_level(tmp_path):
    path = tmp_path / "short.wav"
    sf.write(str(path), np.full((100, 2), 0.5, dtype=np.float32), RATE, subtype="FLOAT")
    assert npd.loud_level(path) == pytest.approx(0.0)


def test_the_windows_level_is_the_rms_over_every_window_together(tmp_path):
    """Two one-frame windows at frames 0 and 2: the RMS of both, not the mean of their two RMS."""
    level = npd.windows_level(_steps(tmp_path / "steps.wav"), [0, 2 * FRAME], FRAME)
    assert level == pytest.approx(np.sqrt((FRAME_LEVELS[0] ** 2 + FRAME_LEVELS[2] ** 2) / 2.0), rel=1e-6)


def test_no_window_is_refused():
    with pytest.raises(ValueError, match="no probe window"):
        npd.windows_level(Path("unused.wav"), [], FRAME)


def test_a_window_past_the_end_reads_nothing_and_measures_silence(tmp_path):
    """A start past the end of the file reads no sample; the level is 0, so the print reads as deep, never as programme."""
    assert npd.windows_level(_steps(tmp_path / "steps.wav"), [10**7], FRAME) == pytest.approx(0.0)


def test_the_depth_is_the_windows_under_the_loud_frames(tmp_path):
    """Frames 9 and 29 against the p90: the depth, and both levels in dBFS."""
    depth, loud_db, probe_db = npd.print_depth_db(_steps(tmp_path / "steps.wav"), ([9 * FRAME, 29 * FRAME], FRAME))
    assert depth == pytest.approx(_depth_of([9, 29]), abs=1e-4)
    assert loud_db - probe_db == pytest.approx(depth)


def test_db_floors_digital_silence():
    assert npd.db(0.0) == pytest.approx(-240.0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("noise_x_single_0.75s.np.json", "noise_x_single_0.75s_depth20db.np.json"),
        ("noise_x_stitched_4.5s_muteguard.np.json", "noise_x_stitched_4.5s_muteguard_depth20db.np.json"),
        ("odd.json", "odd_depth20db.np.json"),
    ],
)
def test_the_scaled_print_is_named_beside_the_learned_one_by_its_depth(tmp_path, name, expected):
    assert npd.scaled_print_path(tmp_path / name, 20.0) == tmp_path / expected


def test_a_fractional_depth_keeps_its_digits_in_the_name(tmp_path):
    assert npd.scaled_print_path(tmp_path / "noise_x.np.json", 17.5).name == "noise_x_depth17.5db.np.json"


def test_scaling_multiplies_every_magnitude(tmp_path):
    """-6 dB: each of the 1025 magnitudes times 10^(-6/20), written as a JSON list."""
    out = npd.scale_print(_print(tmp_path), -6.0, tmp_path / "scaled.np.json")
    scaled = json.loads(out.read_text(encoding="utf-8"))
    assert np.allclose(scaled, np.asarray(PRINT) * 10.0 ** (-6.0 / 20.0), rtol=1e-12)


def test_scaling_leaves_no_temporary_file(tmp_path):
    npd.scale_print(_print(tmp_path), -6.0, tmp_path / "scaled.np.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["noise_x_single_0.75s.np.json", "scaled.np.json"]


@pytest.mark.parametrize("values", [{"bins": PRINT}, ["0.5", 1.0], None])
def test_scaling_refuses_a_print_that_is_not_a_list_of_numbers(tmp_path, values):
    path = tmp_path / "odd.np.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="not a list of magnitudes"):
        npd.scale_print(path, -6.0, tmp_path / "scaled.np.json")


def test_a_deep_print_is_handed_on_untouched(tmp_path):
    """Frames 9 and 29 sit 21.6 dB down: at 20 dB the learned print itself goes to the denoise, and nothing is written."""
    result, print_json, messages, _log = _guarded(tmp_path, ([9 * FRAME, 29 * FRAME], FRAME), 20.0)
    assert result is print_json
    assert not npd.scaled_print_path(print_json, 20.0).exists()
    assert messages == [
        f"{GUARD} the print's 2 window(s) sit {_depth_of([9, 29]):.1f} dB under the loud frames (-22.6 / -0.9 dBFS), "
        "at least 20 dB; the learned print is kept"
    ]


def test_a_shallow_print_is_scaled_to_the_floor(tmp_path):
    """Frame 199 sits 2.6 dB down; at 20 dB its print is scaled by -(20 - 2.6) dB into the named sibling."""
    result, print_json, _messages, _log = _guarded(tmp_path, ([199 * FRAME], FRAME), 20.0)
    gain = _depth_of([199]) - 20.0
    assert result == npd.scaled_print_path(print_json, 20.0)
    assert np.allclose(json.loads(result.read_text(encoding="utf-8")), np.asarray(PRINT) * 10.0 ** (gain / 20.0), rtol=1e-6)


def test_the_scaled_print_leaves_the_learned_one_as_it_was(tmp_path):
    _result, print_json, _messages, _log = _guarded(tmp_path, ([199 * FRAME], FRAME), 20.0)
    assert np.array_equal(json.loads(print_json.read_text(encoding="utf-8")), PRINT)


def test_the_scale_is_logged_with_its_depth_gain_and_file(tmp_path):
    _result, _print_json, messages, _log = _guarded(tmp_path, ([199 * FRAME], FRAME), 20.0)
    assert messages == [
        f"{GUARD} the print's 1 window(s) sit 2.6 dB under the loud frames (-3.5 / -0.9 dBFS): the print is programme; "
        "scaled by -17.4 dB into noise_x_single_0.75s_depth20db.np.json"
    ]


def test_a_shallow_print_is_dropped_on_drop(tmp_path):
    """'drop': no print at all, so the denoise runs without --noiseprint; nothing is written."""
    result, print_json, messages, _log = _guarded(tmp_path, ([199 * FRAME], FRAME), 20.0, action="drop")
    assert result is None
    assert not npd.scaled_print_path(print_json, 20.0).exists()
    assert messages[0].endswith("under 20 dB: the print is programme; denoising without a learned print")


def test_a_print_exactly_at_the_floor_is_kept(tmp_path):
    """At the floor is deep enough: only a print shallower than it is programme."""
    depth = npd.print_depth_db(_steps(tmp_path / "steps.wav"), ([199 * FRAME], FRAME))[0]
    result, print_json, _messages, _log = _guarded(tmp_path, ([199 * FRAME], FRAME), depth)
    assert result is print_json


def test_an_unmeasurable_depth_keeps_the_learned_print_with_a_warning(tmp_path):
    """No window to measure (or an unreadable file): the guard falls back to the learned print."""
    result, print_json, messages, log = _guarded(tmp_path, ([], FRAME), 20.0)
    assert result is print_json
    assert messages == [f"{GUARD} could not measure the print's depth (no probe window to measure); the learned print is kept"]
    assert log.call_args.kwargs == {"level": "WARNING"}


def test_an_unreadable_material_keeps_the_learned_print(tmp_path):
    print_json = _print(tmp_path)
    with patch("modules.noiseprint_depth.log_msg"):
        assert npd.guard_print(print_json, tmp_path / "missing.wav", lambda: ([0], FRAME), 20.0, "scale") is print_json


def test_a_print_that_cannot_be_scaled_is_kept_with_a_warning(tmp_path):
    """A print in a format the guard does not know is not guessed at: the learned print goes on, with a WARNING."""
    result, print_json, messages, log = _guarded(tmp_path, ([199 * FRAME], FRAME), 20.0, values={"bins": PRINT})
    assert result is print_json
    assert "but the print could not be scaled" in messages[0]
    assert log.call_args.kwargs == {"level": "WARNING"}


def _unreadable_pick():
    raise RuntimeError("Error opening the stage file")


def test_a_pick_that_cannot_be_read_keeps_the_learned_print(tmp_path):
    """The windows are picked inside the guard: a soundfile error there keeps the learned print, as an unreadable depth does."""
    print_json = _print(tmp_path)
    with patch("modules.noiseprint_depth.log_msg"):
        assert npd.guard_print(print_json, tmp_path / "x.wav", _unreadable_pick, 20.0, "scale") is print_json
