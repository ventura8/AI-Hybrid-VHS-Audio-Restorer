"""The ring guard caps a notch's ringing in the gaps after speech and leaves the rest of its output to the bit.

What has to hold: a frame the notch only took energy from keeps a gain of exactly 1, a frame
where the notch rings into a gap is brought down to the gap's level (within the tolerance), the
hum the notch removed from a steady line stays removed, the guard is off by default and inert
when the notch did not run, and a file it cannot line up is handed on untouched.
"""

from unittest.mock import patch

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

from modules import config, notch_ring_guard

RATE = 44100
TOLERANCE = 10.0 ** (notch_ring_guard.TOLERANCE_DB / 10.0)


def _notch(audio, freq=250.0, q=30.0):
    """A Q-30 notch like the surgical stage's bandreject, applied causally."""
    b, a = scipy.signal.iirnotch(freq, q, RATE)
    return scipy.signal.lfilter(b, a, audio, axis=0)


def _speech_with_gap(seconds=1.0, gap=(0.5, 0.546)):
    """A 252 Hz partial under a little hiss that stops dead for 46 ms, the way a stop closure does.

    Returns the audio and the gap's (whole, inner) slices: the inner one leaves out the frame
    either side, whose gain is interpolated towards the frames that straddle the gap's edges.
    """
    rng = np.random.default_rng(7)
    t = np.arange(int(seconds * RATE)) / RATE
    voice = 0.3 * np.sin(2 * np.pi * 252.0 * t)
    start, stop = int(gap[0] * RATE), int(gap[1] * RATE)
    voice[start:stop] = 0.0
    hiss = 1e-4 * rng.standard_normal(len(t))
    frame = 2 * notch_ring_guard.SUBFRAME
    return (voice + hiss).astype(np.float32), (slice(start, stop), slice(start + frame, stop - frame))


def _write(path, audio):
    sf.write(str(path), audio, RATE, subtype="FLOAT")
    return path


def _pair(tmp_path, audio):
    """The notch's input and output on disk, stereo, as the surgical step leaves them."""
    stereo = np.column_stack((audio, audio)).astype(np.float32)
    source = _write(tmp_path / "pre.wav", stereo)
    filtered = _write(tmp_path / "surgical_pre.wav", _notch(stereo.astype(np.float64)).astype(np.float32))
    return source, filtered


def _level_db(samples):
    return 20.0 * np.log10(np.sqrt(np.mean(np.square(samples, dtype=np.float64))) + 1e-12)


def test_the_guard_is_off_by_default():
    assert config.CONFIG["apl_surgical_ring_guard"] is False
    assert config.APL_SURGICAL_RING_GUARD is False


def test_a_frame_the_notch_only_thinned_keeps_a_gain_of_exactly_one():
    source = np.array([[4.0], [4.0], [4.0]])
    filtered = np.array([[1.0], [3.0], [4.0]])
    assert np.array_equal(notch_ring_guard.frame_gains(source, filtered), np.ones((2, 1)))


def test_a_ringing_frame_is_capped_at_its_input_energy_within_the_tolerance():
    source = np.array([[1.0], [0.0], [0.0]])
    filtered = np.array([[1.0], [3.0], [8.0]])
    gains = notch_ring_guard.frame_gains(source, filtered)
    capped = (source[:-1] + source[1:]) * TOLERANCE
    assert np.allclose(gains[:, 0] ** 2 * (filtered[:-1, 0] + filtered[1:, 0]), capped[:, 0], rtol=1e-9, atol=1e-15)


def test_a_file_of_one_subframe_still_gets_a_gain():
    assert notch_ring_guard.frame_gains(np.array([[1.0]]), np.array([[4.0]])).shape == (1, 1)


def _interpolated_curve():
    """Two frames, the first passing and the second shut, spread over three sub-frames of samples."""
    return notch_ring_guard.sample_gains(np.array([[1.0], [0.0]]), 0, 3 * notch_ring_guard.SUBFRAME)


def test_the_gain_is_interpolated_between_frame_centres():
    curve = _interpolated_curve()
    centre = notch_ring_guard.SUBFRAME
    assert curve[centre, 0] == 1.0
    assert curve[2 * centre, 0] == 0.0
    assert 0.0 < curve[centre + centre // 2, 0] < 1.0


def test_the_gain_holds_its_end_values_beyond_the_outer_centres():
    curve = _interpolated_curve()
    assert curve[0, 0] == 1.0
    assert curve[-1, 0] == 0.0


def test_the_ringing_in_a_gap_is_taken_down_to_the_gap(tmp_path):
    """Inside the gap the output sits at the gap's level within the tolerance; across all of it, well under the ringing."""
    audio, (whole, inner) = _speech_with_gap()
    source, filtered = _pair(tmp_path, audio)
    produced, capped, frames = notch_ring_guard.guard(source, filtered, tmp_path / "out.wav")
    before, _ = sf.read(str(source), dtype="float32", always_2d=True)
    rang, _ = sf.read(str(filtered), dtype="float32", always_2d=True)
    guarded, _ = sf.read(str(produced), dtype="float32", always_2d=True)
    assert _level_db(rang[inner, 0]) > _level_db(before[inner, 0]) + 40.0
    assert _level_db(guarded[inner, 0]) <= _level_db(before[inner, 0]) + notch_ring_guard.TOLERANCE_DB + 0.5
    assert _level_db(guarded[whole, 0]) < _level_db(rang[whole, 0]) - 6.0
    assert 0 < capped < frames


def test_the_samples_the_notch_did_not_ring_on_pass_to_the_bit(tmp_path):
    audio, _gap = _speech_with_gap()
    source, filtered = _pair(tmp_path, audio)
    produced, _capped, _frames = notch_ring_guard.guard(source, filtered, tmp_path / "out.wav")
    rang, _ = sf.read(str(filtered), dtype="float32", always_2d=True)
    guarded, _ = sf.read(str(produced), dtype="float32", always_2d=True)
    steady = slice(int(0.1 * RATE), int(0.45 * RATE))
    assert np.array_equal(guarded[steady], rang[steady])


def test_the_hum_a_notch_takes_from_a_steady_line_stays_taken(tmp_path):
    rng = np.random.default_rng(11)
    t = np.arange(RATE) / RATE
    audio = (0.05 * np.sin(2 * np.pi * 250.0 * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    source, filtered = _pair(tmp_path, audio)
    produced, _capped, _frames = notch_ring_guard.guard(source, filtered, tmp_path / "out.wav")
    guarded, _ = sf.read(str(produced), dtype="float32", always_2d=True)
    rang, _ = sf.read(str(filtered), dtype="float32", always_2d=True)
    tail = slice(RATE // 2, None)
    line = np.abs(np.fft.rfft(guarded[tail, 0]))[125]
    assert line <= np.abs(np.fft.rfft(rang[tail, 0]))[125] * 1.01
    assert line < np.abs(np.fft.rfft(audio[tail]))[125] / 10.0


def test_a_long_file_is_streamed_in_blocks(tmp_path):
    audio, _gap = _speech_with_gap(seconds=1.0)
    source, filtered = _pair(tmp_path, audio)
    with patch.object(notch_ring_guard, "BLOCK_SAMPLES", notch_ring_guard.SUBFRAME * 16):
        produced, _c, frames = notch_ring_guard.guard(source, filtered, tmp_path / "blocks.wav")
    whole, _c2, whole_frames = notch_ring_guard.guard(source, filtered, tmp_path / "whole.wav")
    a, _ = sf.read(str(produced), dtype="float32")
    b, _ = sf.read(str(whole), dtype="float32")
    assert frames == whole_frames
    assert np.array_equal(a, b)


def test_files_that_do_not_line_up_are_refused(tmp_path):
    audio, _gap = _speech_with_gap()
    source = _write(tmp_path / "pre.wav", audio)
    filtered = _write(tmp_path / "surgical_pre.wav", audio[:-100])
    with pytest.raises(ValueError, match="does not line up"):
        notch_ring_guard.guard(source, filtered, tmp_path / "out.wav")


def test_off_the_notch_output_is_handed_on(tmp_path):
    audio, _gap = _speech_with_gap()
    source, filtered = _pair(tmp_path, audio)
    assert notch_ring_guard.apply_when_enabled(source, filtered) == filtered
    assert notch_ring_guard.apply_when_enabled(source, filtered, enabled=False) == filtered
    assert not (tmp_path / "ringguarded_surgical_pre.wav").exists()


def test_without_a_notch_or_soundfile_the_input_path_is_handed_on(tmp_path):
    audio, _gap = _speech_with_gap()
    source, filtered = _pair(tmp_path, audio)
    assert notch_ring_guard.apply_when_enabled(source, source, enabled=True) == source
    with patch.object(notch_ring_guard, "sf", None):
        assert notch_ring_guard.apply_when_enabled(source, filtered, enabled=True) == filtered
    assert not (tmp_path / "ringguarded_surgical_pre.wav").exists()


def test_on_it_writes_the_guarded_file_beside_the_notch_and_reuses_it(tmp_path):
    audio, _gap = _speech_with_gap()
    source, filtered = _pair(tmp_path, audio)
    with patch("modules.notch_ring_guard.log_msg") as log:
        produced = notch_ring_guard.apply_when_enabled(source, filtered, enabled=True)
    assert produced == tmp_path / "ringguarded_surgical_pre.wav"
    assert "Ring guard: capped" in log.call_args[0][0]
    with patch("modules.notch_ring_guard.guard") as guard, patch("modules.notch_ring_guard.log_msg"):
        assert notch_ring_guard.apply_when_enabled(source, filtered, enabled=True) == produced
    guard.assert_not_called()


def test_the_config_switch_turns_it_on(tmp_path):
    audio, _gap = _speech_with_gap()
    source, filtered = _pair(tmp_path, audio)
    with patch.object(notch_ring_guard, "APL_SURGICAL_RING_GUARD", True), patch("modules.notch_ring_guard.log_msg"):
        assert notch_ring_guard.apply_when_enabled(source, filtered).name == "ringguarded_surgical_pre.wav"


def test_a_failure_hands_on_the_notch_output_with_a_warning(tmp_path):
    audio, _gap = _speech_with_gap()
    source = _write(tmp_path / "pre.wav", audio)
    filtered = _write(tmp_path / "surgical_pre.wav", audio[:-100])
    with patch("modules.notch_ring_guard.log_msg") as log:
        assert notch_ring_guard.apply_when_enabled(source, filtered, enabled=True) == filtered
    assert log.call_args[1]["level"] == "WARNING"
    assert not (tmp_path / "ringguarded_surgical_pre.wav").exists()


def test_the_chain_reads_the_guarded_notch_output(tmp_path):
    """processing._neural_output hands the chain whatever the guard returns, with the notch's input as its reference."""
    import modules.processing as processing

    surgical, guarded = tmp_path / "s.wav", tmp_path / "ringguarded_s.wav"
    with (
        patch("modules.processing._resolve_adaptive_denoise_model", return_value="m"),
        patch("modules.processing._pre_denoise_surgical_step", return_value=surgical),
        patch("modules.processing._notch_ring_guard.apply_when_enabled", return_value=guarded) as ring,
        patch("modules.processing._apl_chain.stage_plan", return_value=[]),
        patch("modules.processing._apl_chain.run", side_effect=lambda wav, plan: (wav, [])) as chain,
        patch("modules.processing._neural_stage", return_value=("m", False)),
        patch("modules.processing._without_stale_neural_output", side_effect=lambda d: d),
    ):
        flags = dict.fromkeys(processing.NEURAL_FLAGS, False)
        pair = processing._neural_output(tmp_path / "pre.wav", tmp_path, 10.0, None, {"profile": {}}, flags)
    assert ring.call_args.args == (tmp_path / "pre.wav", surgical)
    assert chain.call_args.args[0] == guarded
    assert pair == (guarded, guarded)
