"""Keys, windows, routing, alignment and the resample cache."""

from unittest.mock import patch

import numpy as np
import soundfile as sf

from scripts.restoration_quality import audio_io

RATE = 44100


def test_file_key_is_stable_and_changes_with_the_file(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(str(path), np.zeros((RATE, 2), dtype=np.float32), RATE, subtype="FLOAT")
    first = audio_io.file_key(path)
    assert first == audio_io.file_key(path)
    assert len(first) == 16
    sf.write(str(path), np.zeros((2 * RATE, 2), dtype=np.float32), RATE, subtype="FLOAT")
    assert audio_io.file_key(path) != first


def test_extract_wav_returns_a_wav_untouched(tmp_path):
    wav = tmp_path / "x.wav"
    sf.write(str(wav), np.zeros((100, 1), dtype=np.float32), RATE, subtype="FLOAT")
    assert audio_io.extract_wav(wav, tmp_path / "cache") == wav


def test_extract_wav_calls_ffmpeg_for_a_container(tmp_path):
    video = tmp_path / "clip.mov"
    video.write_bytes(b"not really a video")
    with patch("scripts.restoration_quality.audio_io.subprocess.run") as run:
        target = audio_io.extract_wav(video, tmp_path / "cache")
    assert target.parent == tmp_path / "cache"
    assert target.suffix == ".wav"
    assert run.call_args[0][0][-1] == str(target)


def test_windows_keep_a_tail_of_at_least_five_seconds():
    spans = [(w.start_s, w.end_s) for w in audio_io.windows(20 * RATE, RATE)]
    assert spans == [(0.0, 15.0), (7.5, 20.0), (15.0, 20.0)]
    assert audio_io.windows(3 * RATE, RATE) == []
    assert audio_io.windows(125 * RATE, RATE, 15.0, 7.5)[-1].index == 16


def test_short_file_and_slice():
    only = audio_io.windows(7 * RATE, RATE)
    assert len(only) == 1
    assert only[0].end_s == 7.0
    assert audio_io.Window(0, 1.0, 2.0).slice_of(100) == slice(100, 200)


def test_route_window_separates_silence_speech_and_broadband():
    rng = np.random.default_rng(0)
    assert audio_io.route_window(np.zeros(RATE, dtype=np.float32), RATE) == "silence"
    noise = rng.standard_normal(RATE).astype(np.float32) * 0.1
    band = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1.0 / RATE)
    band[(freqs < 300) | (freqs > 3400)] = 0.0
    speechy = np.fft.irfft(band, len(noise)).astype(np.float32)
    assert audio_io.route_window(speechy, RATE) == "speech"
    assert audio_io.route_window(noise, RATE) == "mixed"


def test_align_pair_removes_a_lag_and_matches_gain():
    rng = np.random.default_rng(2)
    source = rng.standard_normal(RATE).astype(np.float32)
    shifted = np.concatenate([np.zeros(200, dtype=np.float32), source * 0.5])
    src, out, lag = audio_io.align_pair(source, shifted)
    assert lag == 200
    assert len(src) == len(out)
    assert np.allclose(src, out, atol=1e-3)


def test_resample_cached_returns_native_rate_untouched(tmp_path):
    mono = np.ones(100, dtype=np.float32)
    assert audio_io.resample_cached(mono, RATE, RATE, tmp_path, "k") is mono


def test_resample_cached_writes_and_reads_the_cache(tmp_path):
    mono = np.ones(100, dtype=np.float32)
    with patch("scripts.restoration_quality.audio_io._resample", return_value=np.ones(36, dtype=np.float32)) as resample:
        first = audio_io.resample_cached(mono, RATE, 16000, tmp_path, "k")
        second = audio_io.resample_cached(mono, RATE, 16000, tmp_path, "k")
    assert resample.call_count == 1
    assert len(first) == len(second) == 36
    assert (tmp_path / "k_16000.npy").exists()


def test_resample_falls_back_to_scipy_without_librosa():
    with patch.dict("sys.modules", {"librosa": None}):
        out = audio_io._resample(np.ones(4410, dtype=np.float32), RATE, 16000)
    assert len(out) == 1600


def _chord(seconds=15.0):
    t = np.arange(int(seconds * RATE)) / RATE
    # A harmonic series: a held tone whose envelope does not beat (a chord's beating reads as syllables).
    return sum(0.2 / k * np.sin(2 * np.pi * 110.0 * k * t) for k in (1, 2, 3, 4, 5)).astype(np.float32)


def _syllables(seconds=15.0):
    rng = np.random.default_rng(5)
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.3 * rng.standard_normal(len(t)) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 4.0 * t)))).astype(np.float32)


def test_tonal_persistence_is_high_on_a_held_chord_and_nil_on_syllabic_noise():
    assert audio_io.tonal_persistence(_chord(), RATE) > audio_io.PERSISTENCE_MUSIC
    assert audio_io.tonal_persistence(_syllables(), RATE) < audio_io.PERSISTENCE_MIXED


def test_syllabic_modulation_reads_the_four_hertz_bursts():
    assert audio_io.syllabic_modulation(_syllables(), RATE) > audio_io.SYLLABIC_MAX_MUSIC
    assert audio_io.syllabic_modulation(_chord(), RATE) < audio_io.SYLLABIC_MAX_MUSIC


def test_route_window_calls_a_held_tone_music_and_syllabic_bursts_not_music():
    assert audio_io.route_window(_chord(), RATE) == "music"
    assert audio_io.route_window(_syllables(), RATE) != "music"


def test_load_audio_removes_a_dc_offset_per_channel(tmp_path):
    import soundfile as sf

    t = np.arange(RATE) / RATE
    stereo = np.stack([0.1 * np.sin(2 * np.pi * 440 * t) + 0.49, 0.1 * np.sin(2 * np.pi * 440 * t) - 0.2], axis=1).astype(np.float32)
    sf.write(str(tmp_path / "dc.wav"), stereo, RATE, subtype="FLOAT")
    audio, rate = audio_io.load_audio(tmp_path / "dc.wav")
    assert rate == RATE
    assert np.abs(audio.mean(axis=0)).max() < 1e-4
    assert abs(float(np.sqrt(np.mean(audio[:, 0] ** 2))) - 0.1 / np.sqrt(2)) < 1e-3


def test_dc_free_copy_returns_the_file_itself_unless_dc_dominates(tmp_path):
    import soundfile as sf

    t = np.arange(RATE) / RATE
    clean = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(tmp_path / "clean.wav"), clean, RATE, subtype="FLOAT")
    sf.write(str(tmp_path / "offset.wav"), clean + 0.49, RATE, subtype="FLOAT")
    assert audio_io.dc_free_copy(tmp_path / "clean.wav", tmp_path / "cache") == tmp_path / "clean.wav"
    copy = audio_io.dc_free_copy(tmp_path / "offset.wav", tmp_path / "cache")
    assert copy.name.endswith("_dcfree.wav") and audio_io.dc_share(copy) < 1e-6


def test_route_window_calls_an_infrasonic_capture_silence():
    t = np.arange(4 * RATE) / RATE
    junk = sum(0.2 / k * np.sin(2 * np.pi * 5.3 * k * t) for k in range(1, 12)).astype(np.float32)
    assert audio_io.infrasonic_share(junk, RATE) > audio_io.INFRASONIC_SHARE_MAX
    assert audio_io.route_window(junk, RATE) == "silence"
    assert audio_io.infrasonic_share(_chord(), RATE) < 0.05
