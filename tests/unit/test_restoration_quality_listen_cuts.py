"""Listening cuts: aligned (lag and polarity), speech-level matched within bounds, picked where outputs differ audibly."""

import io
import tempfile

import numpy as np
import pytest
import soundfile as sf

from scripts.restoration_quality import auditory, listen_cuts

# The rate the listening page serves: a WAV at it is read in place, anything else is decoded through ffmpeg.
RATE = listen_cuts.RATE


def _voice(seconds=12.0, seed=3, hiss=3e-3, rate=RATE):
    """Bright harmonic bursts (0.5 s on, 0.5 s off) over a steady hiss: loud frames, gaps and deep pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    phases = rng.uniform(0.0, 2 * np.pi, 60)
    voice = sum(np.sin(2 * np.pi * 150.0 * k * t + phases[k]) / np.sqrt(k) for k in range(1, 60))
    ramp = np.hanning(int(0.05 * rate))
    gate = np.convolve(((t % 1.0) < 0.5).astype(np.float64), ramp / ramp.sum(), mode="same")
    return (0.05 * voice * gate + hiss * rng.standard_normal(len(t))).astype(np.float32), gate


def _wav(path, audio, rate=RATE):
    sf.write(str(path), audio, rate, subtype="FLOAT")
    return path


def _delayed(audio, samples):
    return np.concatenate([np.zeros(samples, dtype=np.float32), audio[: len(audio) - samples]])


def _fake_ffmpeg(calls):
    """Stands in for ffmpeg: writes the input's audio (mono when asked) to the target, at the input's own rate."""

    def run(command, **_kwargs):
        calls.append(command)
        if "-ac" in command:
            audio, rate = sf.read(command[command.index("-i") + 1], dtype="float32", always_2d=True)
            _wav(command[-1], audio.mean(axis=1), rate)
            return
        _wav(command[-1], np.full((RATE // 2, 2), 0.25, dtype=np.float32))

    return run


def test_read_segment_reads_a_window_of_a_wav(tmp_path):
    """A WAV at the session rate is read straight from the requested frames."""
    audio, _gate = _voice(3.0)
    segment = listen_cuts.read_segment(_wav(tmp_path / "a.wav", audio), 1.0, 1.0, tmp_path / "work")
    assert segment.shape == (RATE, 1)
    assert np.array_equal(segment[:, 0], audio[slice(RATE, 2 * RATE)])


def test_read_segment_decodes_other_files_once_through_ffmpeg(tmp_path, monkeypatch):
    """A video (or another rate) goes through ffmpeg into the work directory, once."""
    calls = []
    monkeypatch.setattr(listen_cuts.subprocess, "run", _fake_ffmpeg(calls))
    video = tmp_path / "tape.mov"
    video.write_bytes(b"not decoded here")
    first = listen_cuts.read_segment(video, 2.0, 0.5, tmp_path / "work")
    listen_cuts.read_segment(video, 2.0, 0.5, tmp_path / "work")
    assert first.shape == (RATE // 2, 2)
    assert len(calls) == 1
    assert calls[0][calls[0].index("-ss") + 1] == "2.000"
    assert calls[0][calls[0].index("-ar") + 1] == str(RATE)


def test_ffmpeg_command_refuses_a_negative_start(tmp_path):
    """Numbers reach the command line checked."""
    with pytest.raises(SystemExit, match="start"):
        listen_cuts.ffmpeg_command(tmp_path / "a.mov", -1.0, 1.0, tmp_path / "out.wav")


def test_window_at_pads_where_the_file_does_not_reach():
    """A window that starts before the file or runs past it is zero-padded, never shifted."""
    audio = np.arange(1, 6, dtype=np.float32)[:, None]
    assert listen_cuts.window_at(audio, -2, 4)[:, 0].tolist() == [0.0, 0.0, 1.0, 2.0]
    assert listen_cuts.window_at(audio, 3, 4)[:, 0].tolist() == [4.0, 5.0, 0.0, 0.0]
    assert not listen_cuts.window_at(audio, 9, 2).any()


def _quieter_delayed_copy(tmp_path):
    """The voice and a copy of it 12 dB down and 300 samples late, prepared as 8 s cuts from 2 s."""
    # 0.25 in amplitude is -12.04 dB: the gain the speech-level match has to give back.
    source, _gate = _voice()
    paths = {"ref": _wav(tmp_path / "ref.wav", source), "out": _wav(tmp_path / "out.wav", 0.25 * _delayed(source, 300))}
    return listen_cuts.prepare_cuts(paths, 2.0, 8.0, tmp_path / "work")


def test_prepare_cuts_aligns_and_matches_the_speech_level(tmp_path):
    """A quieter, delayed copy comes back aligned, 12 dB up, as stereo cuts of the requested length."""
    cuts, info = _quieter_delayed_copy(tmp_path)
    assert cuts["ref"].shape == cuts["out"].shape == (8 * RATE, 2)
    assert info["gains_db"]["out"] == pytest.approx(12.04, abs=0.05)
    assert np.corrcoef(cuts["ref"][:, 0], cuts["out"][:, 0])[0, 1] > 0.999


def test_a_level_matched_copy_is_neither_clamped_nor_silent(tmp_path):
    """The 12 dB the quieter copy needs lies inside the gain bounds, and both cuts carry speech."""
    _cuts, info = _quieter_delayed_copy(tmp_path)
    assert not info["clamped"]
    assert not info["silent"]


def test_an_inverted_copy_is_turned_back_and_blends_without_cancelling(tmp_path):
    """A polarity-inverted, delayed copy is aligned and re-inverted: half of each mixed plays at full level, not as a null."""
    source, _gate = _voice()
    paths = {"ref": _wav(tmp_path / "ref.wav", source), "inv": _wav(tmp_path / "inv.wav", -_delayed(source, 300))}
    cuts, _info = listen_cuts.prepare_cuts(paths, 2.0, 8.0, tmp_path / "work")
    blend = 0.5 * cuts["ref"] + 0.5 * cuts["inv"]
    assert np.corrcoef(cuts["ref"][:, 0], cuts["inv"][:, 0])[0, 1] > 0.999
    assert np.sqrt(np.mean(blend**2)) == pytest.approx(np.sqrt(np.mean(cuts["ref"] ** 2)), rel=0.01)


def test_the_level_match_ignores_the_pauses(tmp_path):
    """An output that empties the pauses keeps its speech level: no gain, though its whole-cut RMS dropped."""
    source, gate = _voice()
    gated = (source * (gate > 0.5)).astype(np.float32)
    paths = {"ref": _wav(tmp_path / "ref.wav", source), "gated": _wav(tmp_path / "gated.wav", gated)}
    cuts, info = listen_cuts.prepare_cuts(paths, 0.0, 8.0, tmp_path / "work")
    assert info["gains_db"]["gated"] == pytest.approx(0.0, abs=0.1)
    assert np.sqrt(np.mean(cuts["gated"] ** 2)) < np.sqrt(np.mean(cuts["ref"] ** 2))


def test_gains_are_bounded_and_silence_is_named():
    """A gain beyond +-20 dB is held at the bound and named; a level under the silence floor marks the cut silent."""
    gains, clamped = listen_cuts.matching_gains_db({"ref": -20.0, "quiet": -45.0, "loud": 5.0, "near": -30.0}, "ref")
    assert gains == {"ref": 0.0, "quiet": 20.0, "loud": -20.0, "near": 10.0}
    assert clamped == ["loud", "quiet"]
    assert listen_cuts.silent_labels({"ref": -20.0, "gone": -240.0, "edge": -50.0}) == ["gone"]


def test_a_start_past_the_end_reads_as_silent(tmp_path):
    """Cuts taken past the end of both files are all padding: both are reported silent."""
    source, _gate = _voice(3.0)
    paths = {"a": _wav(tmp_path / "a.wav", source), "b": _wav(tmp_path / "b.wav", source)}
    _cuts, info = listen_cuts.prepare_cuts(paths, 30.0, 8.0, tmp_path / "work")
    assert info["silent"] == ["a", "b"]


def test_prepare_cuts_leaves_nothing_in_the_temp_directory(tmp_path, monkeypatch):
    """Without a work directory the decoded segments go to a temporary directory that is removed."""
    monkeypatch.setattr(listen_cuts.subprocess, "run", _fake_ffmpeg([]))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    video = tmp_path / "tape.mov"
    video.write_bytes(b"not decoded here")
    cuts, _info = listen_cuts.prepare_cuts({"a": video, "b": video}, 0.0, 0.2)
    assert cuts["a"].shape == (int(0.2 * RATE), 2)
    assert not list((tmp_path / "tmp").iterdir())


def test_finish_cuts_share_one_gain_under_the_peak_ceiling():
    """The loudest cut is brought under the ceiling and the others keep their ratio to it; the edges fade in and out."""
    cuts = {"a": np.full((RATE, 2), 0.5, dtype=np.float32), "b": np.full((RATE, 2), 0.25, dtype=np.float32)}
    finished = listen_cuts.finish_cuts(cuts, {"a": 4.0, "b": 4.0})
    assert np.max(np.abs(finished["a"])) == pytest.approx(listen_cuts.PEAK_MAX, rel=1e-6)
    assert finished["b"][RATE // 2, 0] == pytest.approx(finished["a"][RATE // 2, 0] / 2.0)
    assert finished["a"][0, 0] == 0.0
    assert finished["a"][-1, 0] == 0.0


def test_speech_active_falls_back_to_every_frame():
    """Frames all under the silence floor: every frame counts rather than none."""
    assert listen_cuts.speech_active(np.full(40, -90.0)).all()


def test_best_window_goes_where_the_difference_is_audible():
    """A 12 dB drop over 12-16 s: only windows touching it have audible frames, and the one inside it wins."""
    first, _gate = _voice(24.0)
    second = first.copy()
    second[slice(12 * RATE, 16 * RATE)] *= 0.25
    _spans, scores = listen_cuts.window_scores(first, second, 4.0, RATE)
    assert listen_cuts.best_window(first, second, 4.0, RATE) == pytest.approx(12.0)
    assert scores[6] == pytest.approx(1.0)
    assert scores[0] == 0.0


def test_best_window_reads_through_a_delay_and_an_inversion():
    """The candidate 300 samples late and inverted: the pick still lands on the changed stretch."""
    first, _gate = _voice(24.0)
    second = first.copy()
    second[slice(12 * RATE, 16 * RATE)] *= 0.25
    assert listen_cuts.best_window(first, -_delayed(second, 300), 4.0, RATE) == pytest.approx(12.0)


def test_inaudible_differences_fall_back_to_the_spectral_proxy():
    """No audible frame anywhere: the windows are ranked by the third-octave disagreement instead."""
    first, _gate = _voice(24.0)
    second = (first * 0.999).astype(np.float32)
    second[slice(12 * RATE, 16 * RATE)] = first[slice(12 * RATE, 16 * RATE)] * 0.99
    _spans, scores = listen_cuts.window_scores(first, second, 4.0, RATE)
    assert max(listen_cuts.audible_shares(first, second, _spans, RATE)) == 0.0
    assert scores[6] > 5.0 * scores[0] > 0.0
    assert listen_cuts.best_window(first, second, 4.0, RATE) == pytest.approx(12.0)


def test_the_proxy_reaches_16_khz_and_ignores_digital_silence():
    """The third-octave bands reach the 12.8-16 kHz air region; silence disagrees with nothing."""
    assert listen_cuts.THIRD_OCTAVES_HZ[-1] >= 16000.0 > listen_cuts.THIRD_OCTAVES_HZ[-2]
    assert listen_cuts.disagreement_db(np.zeros(RATE), np.zeros(RATE)) == 0.0


def test_pick_start_decodes_both_files_whole_and_cleans_up(tmp_path, monkeypatch):
    """Without a chosen start the cut lands where the two files differ, and the temporary decodes are gone afterwards."""
    calls = []
    monkeypatch.setattr(listen_cuts.subprocess, "run", _fake_ffmpeg(calls))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    first, _gate = _voice(24.0, rate=listen_cuts.PICK_RATE)
    second = first.copy()
    second[slice(4 * listen_cuts.PICK_RATE, 8 * listen_cuts.PICK_RATE)] *= 0.2
    pair = (_wav(tmp_path / "a.wav", first, listen_cuts.PICK_RATE), _wav(tmp_path / "b.wav", second, listen_cuts.PICK_RATE))
    assert listen_cuts.pick_start(*pair, 4.0) == pytest.approx(4.0)
    assert len(calls) == 2
    assert calls[0][calls[0].index("-ar") + 1] == str(listen_cuts.PICK_RATE)
    assert not list((tmp_path / "tmp").iterdir())


def test_wav_bytes_and_the_cut_hash():
    """The served cut is a 24-bit WAV of the same samples; the cut hash is the audibility check's PCM hash."""
    audio = (0.1 * np.sin(np.arange(RATE) / 10.0)).astype(np.float32)
    decoded, rate = sf.read(io.BytesIO(listen_cuts.wav_bytes(audio)), dtype="float32")
    assert rate == RATE
    assert np.max(np.abs(decoded - audio)) < 1e-6
    assert listen_cuts.cut_sha256(audio) == auditory.pcm_sha256(audio, RATE) != listen_cuts.cut_sha256(audio * 0.5)
