"""The tone canceller: persistent lines that are not the mains series are found and taken; notes, squeals and rings are not."""

from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from modules import tone_cancel

RATE = 44100
SECONDS = 8.0


def _time():
    return np.arange(int(RATE * SECONDS)) / RATE


def _voice(seed=1):
    """Speech-like partials with a moving pitch, in phrases with pauses, over hiss."""
    t, rng = _time(), np.random.default_rng(seed)
    phase = 2 * np.pi * np.cumsum(180.0 * (1.0 + 0.06 * np.sin(2 * np.pi * 5.0 * t))) / RATE
    partials = sum((0.2 / k**0.7) * np.sin(k * phase + rng.uniform(0, 6)) for k in range(1, 14))
    return partials * (0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t)) ** 2 * (np.sin(2 * np.pi * 0.4 * t) > -0.2) + rng.normal(0.0, 0.004, len(t))


def _tone(hz, level=0.01, wander_hz=0.0, wander_rate=0.0):
    t = _time()
    instantaneous = hz + wander_hz * np.sin(2 * np.pi * wander_rate * t)
    return level * np.sin(2 * np.pi * np.cumsum(instantaneous) / RATE)


def _write(path, channels):
    sf.write(str(path), np.stack(channels, axis=1).astype(np.float32), RATE, subtype="FLOAT")
    return path


def _level_at(signal, hz, width=8.0):
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / RATE)
    return 20.0 * np.log10(np.max(spectrum[np.abs(freqs - hz) <= width]) + 1e-12)


def test_a_steady_whine_is_detected_and_cancelled(tmp_path):
    """A 3.1 kHz whine 22 dB under the voice is found within a hertz and taken 15 dB down."""
    voice = _voice()
    whine = _tone(3100.0, level=0.008)
    source = _write(tmp_path / "whine.wav", [voice + whine, _voice(seed=3) + whine])
    mono = (voice + whine).astype(np.float64)
    lines = tone_cancel.detect_lines(mono, RATE)
    assert len(lines) == 1 and abs(lines[0][0] - 3100.0) < 1.0
    restored = sf.read(str(tone_cancel.cancel_tones(source, tmp_path / "out.wav", lines)), dtype="float32")[0]
    assert _level_at(voice + whine, 3100.0) - _level_at(restored[:, 0], 3100.0) > 15.0


def test_a_flutter_wandering_whine_above_8_khz_is_cancelled(tmp_path):
    """A recorded line whistle wandering 16 Hz at 3 Hz is inside the wide band above 8 kHz."""
    voice = _voice()
    whine = _tone(15734.0 - 400.0, level=0.006, wander_hz=16.0, wander_rate=3.0)
    source = _write(tmp_path / "flutter.wav", [voice + whine, _voice(seed=3) + whine])
    lines = tone_cancel.detect_lines((voice + whine).astype(np.float64), RATE)
    assert len(lines) == 1 and abs(lines[0][0] - 15334.0) < 5.0
    assert tone_cancel.bandwidths_for([hz for hz, _floor in lines]) == [tone_cancel.HIGH_BANDWIDTH_HZ]
    restored = sf.read(str(tone_cancel.cancel_tones(source, tmp_path / "out.wav", lines)), dtype="float32")[0]
    assert _level_at(voice + whine, 15334.0, width=30.0) - _level_at(restored[:, 0], 15334.0, width=30.0) > 12.0


def test_mains_harmonics_and_the_crt_line_are_left_to_their_own_stages():
    """A 300 Hz line with mains detected at 50 Hz, and a 15625 Hz whistle, are not this stage's."""
    voice = _voice()
    mono = (voice + _tone(300.0, level=0.02) + _tone(15625.0, level=0.01)).astype(np.float64)
    assert tone_cancel.detect_lines(mono, RATE, mains_hz=50.0) == []
    lines = tone_cancel.detect_lines(mono, RATE, mains_hz=0.0)
    assert [round(hz) for hz, _floor in lines] == [300]


def test_a_squeal_and_a_resonance_are_not_detected():
    """The sticky-shed squeal wanders too far to be a line; the enclosure ring is no line at all."""
    voice = _voice()
    squeal = _tone(2500.0, level=0.01, wander_hz=75.0, wander_rate=3.0)
    assert tone_cancel.detect_lines((voice + squeal).astype(np.float64), RATE) == []
    import scipy.signal

    sos = scipy.signal.iirpeak(2000.0, 5.0, fs=RATE)
    rung = scipy.signal.sosfilt(scipy.signal.tf2sos(*sos), voice) * 6.0 + voice
    assert tone_cancel.detect_lines(rung.astype(np.float64), RATE) == []


def test_a_note_absent_from_the_quiet_frames_is_not_a_defect():
    """A tone present only where the programme is loud is a note, and the quiet frames say so."""
    t = _time()
    voice = _voice()
    loud = np.abs(voice) > 0.02
    note = _tone(1500.0, level=0.03) * loud
    assert tone_cancel.detect_lines((voice + note).astype(np.float64), RATE) == []
    assert t.size == voice.size


def test_lines_are_merged_and_capped():
    """Bins closer than the merge span are one line; more lines than the cap keep the strongest."""
    voice = _voice()
    many = sum(_tone(700.0 + 137.0 * k, level=0.006) for k in range(20))
    lines = tone_cancel.detect_lines((voice + many).astype(np.float64), RATE)
    assert len(lines) == tone_cancel.MAX_LINES
    excess = np.array([1.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 9.0])
    candidates = np.array([True, True, True, False, False, False, False, False, True, True])
    assert tone_cancel._merged_peaks(candidates, excess) == [1, 9]
    assert tone_cancel._merged_peaks(np.zeros(4, dtype=bool), excess[:4]) == []


def test_the_stage_cancels_through_the_chain_entry_point(tmp_path):
    voice = _voice()
    whine = _tone(3100.0, level=0.008)
    source = _write(tmp_path / "whine.wav", [voice + whine, _voice(seed=3) + whine])
    with patch.object(tone_cancel, "APL_ENABLE_TONE_CANCEL", True), patch("modules.tone_cancel.log_msg") as log:
        produced = tone_cancel.apply_when_needed(source, tmp_path / "work")
    assert produced == tmp_path / "work" / "tone_cancel" / "tonecancel_whine.wav"
    assert "Cancelled 1 lines: 3100 Hz" in log.call_args[0][0]


@pytest.mark.parametrize("case", ["disabled", "unreadable", "no_lines"])
def test_disabled_unreadable_or_clean_yields_the_input(tmp_path, case):
    if case == "unreadable":
        source = tmp_path / "rubbish.wav"
        source.write_text("not audio")
    else:
        source = _write(tmp_path / "clean.wav", [_voice(), _voice(seed=3)])
    with patch.object(tone_cancel, "APL_ENABLE_TONE_CANCEL", case != "disabled"), patch("modules.tone_cancel.log_msg"):
        assert tone_cancel.apply_when_needed(source, tmp_path / "work") == source


def test_a_recording_too_short_to_track_yields_the_input(tmp_path):
    source = _write(tmp_path / "clean.wav", [_voice(), _voice(seed=3)])
    with (
        patch.object(tone_cancel, "APL_ENABLE_TONE_CANCEL", True),
        patch.object(tone_cancel, "_plan", return_value=([(3100.0, 1e-9)], None)),
        patch.object(tone_cancel, "cancel_tones", return_value=None),
        patch("modules.tone_cancel.log_msg") as log,
    ):
        assert tone_cancel.apply_when_needed(source, tmp_path / "work") == source
    assert "too short" in log.call_args[0][0]


def test_a_failure_inside_the_stage_leaves_the_audio_usable(tmp_path):
    source = _write(tmp_path / "clean.wav", [_voice(), _voice(seed=3)])
    with (
        patch.object(tone_cancel, "APL_ENABLE_TONE_CANCEL", True),
        patch.object(tone_cancel, "_plan", return_value=([(3100.0, 1e-9)], None)),
        patch.object(tone_cancel, "cancel_tones", side_effect=MemoryError("no room")),
        patch("modules.tone_cancel.log_msg") as log,
    ):
        assert tone_cancel.apply_when_needed(source, tmp_path / "work") == source
    assert "after failure" in log.call_args[0][0]


def test_the_output_is_deterministic(tmp_path):
    voice = _voice()
    whine = _tone(3100.0, level=0.008)
    source = _write(tmp_path / "whine.wav", [voice + whine, voice + whine])
    lines = tone_cancel.detect_lines((voice + whine).astype(np.float64), RATE)
    first = sf.read(str(tone_cancel.cancel_tones(source, tmp_path / "a.wav", lines)), dtype="float32")[0]
    second = sf.read(str(tone_cancel.cancel_tones(source, tmp_path / "b.wav", lines)), dtype="float32")[0]
    assert np.array_equal(first, second)


def test_the_stage_is_wired_after_the_hum_canceller(tmp_path):
    from modules import apl_chain

    plan = apl_chain.stage_plan(tmp_path, None, None, physical_repair=True, spectral_denoise=True, hum_cancel=True, tone_cancel=True)
    names = [name for name, _wanted, _stage in plan]
    assert names.index("tone_cancel") == names.index("hum_cancel") + 1
