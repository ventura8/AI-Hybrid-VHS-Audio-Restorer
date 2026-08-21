from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

import modules.auto_scanner


def test_compute_band_energy_ratio():
    """Verify band energy ratio computation."""
    freqs = np.array([100.0, 500.0, 1000.0, 5000.0])
    fft_power = np.array([1.0, 4.0, 4.0, 1.0])
    total_power = float(np.sum(fft_power))

    ratio = modules.auto_scanner._compute_band_energy_ratio(fft_power, freqs, 400.0, 2000.0, total_power)
    assert 0.79 <= ratio <= 0.81


def test_estimate_speech_presence_ratio():
    """Test speech presence estimation on vocal tone vs short signal."""
    sr = 44100
    # Short signal fallback
    assert modules.auto_scanner._estimate_speech_presence_ratio(np.zeros(100, dtype=np.float32), sr) == 0.5

    # Speech formant synthetic signal (1000 Hz tone)
    t = np.arange(sr * 2, dtype=np.float32) / float(sr)
    speech_signal = (0.8 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
    ratio = modules.auto_scanner._estimate_speech_presence_ratio(speech_signal, sr)
    assert ratio > 0.5


def test_estimate_music_harmonic_ratio():
    """Test musical harmonicity on tonal peaks vs short signal."""
    sr = 44100
    assert modules.auto_scanner._estimate_music_harmonic_ratio(np.zeros(100, dtype=np.float32), sr) == 0.2

    # Multi-tonal chord (440Hz, 554Hz, 659Hz)
    t = np.arange(sr * 2, dtype=np.float32) / float(sr)
    chord = (0.3 * np.sin(2.0 * np.pi * 440.0 * t) + 0.3 * np.sin(2.0 * np.pi * 554.0 * t) + 0.3 * np.sin(2.0 * np.pi * 659.0 * t)).astype(
        np.float32
    )
    ratio = modules.auto_scanner._estimate_music_harmonic_ratio(chord, sr)
    assert ratio >= 0.15


def test_estimate_ambient_texture_ratio():
    """Test high frequency ambient texture detection."""
    sr = 44100
    assert modules.auto_scanner._estimate_ambient_texture_ratio(np.zeros(100, dtype=np.float32), sr) == 0.1

    # High frequency bird chirp / texture (8000 Hz)
    t = np.arange(sr * 2, dtype=np.float32) / float(sr)
    birds = (0.6 * np.sin(2.0 * np.pi * 8000.0 * t)).astype(np.float32)
    ratio = modules.auto_scanner._estimate_ambient_texture_ratio(birds, sr)
    assert ratio > 0.3


def test_detect_flutter_or_pitch_drift():
    """Test wow/flutter speed instability detector."""
    sr = 44100
    # Short signal
    assert modules.auto_scanner._detect_flutter_or_pitch_drift(np.zeros(100, dtype=np.float32), sr) is False

    # Modulated pitch (wow & flutter)
    t = np.arange(sr * 5, dtype=np.float32) / float(sr)
    mod_freq = 440.0 + 100.0 * np.sin(2.0 * np.pi * 2.0 * t)
    phase = 2.0 * np.pi * np.cumsum(mod_freq) / float(sr)
    flutter_signal = (0.7 * np.sin(phase)).astype(np.float32)

    # Frame-peak variance for this 2 Hz modulation stays under the detector's
    # threshold, so no drift is reported and 'shift' sync remains selected.
    assert modules.auto_scanner._detect_flutter_or_pitch_drift(flutter_signal, sr) is False


def test_detect_flutter_rejects_low_sample_rate():
    """Low-rate audio cannot contain the VHS line-whine reference band."""
    assert modules.auto_scanner._detect_flutter_or_pitch_drift(np.zeros(120000, dtype=np.float32), 16000) is False


def _speed_reference_signal(sr, depth=0.0, wow_hz=0.5, seconds=8, seed=7):
    """Builds audio carrying a PAL line whine whose speed wobbles by ``depth``.

    The phase is accumulated in float64: float32 loses enough precision over a
    multi-second ramp to jitter the reference tone on its own.
    """
    count = sr * seconds
    t = np.arange(count, dtype=np.float64) / sr
    instantaneous = 15625.0 * (1.0 + depth * np.sin(2.0 * np.pi * wow_hz * t))
    whine = 0.05 * np.sin(2.0 * np.pi * np.cumsum(instantaneous) / sr)
    bed = 0.3 * np.sin(2.0 * np.pi * 440.0 * t) + 0.02 * np.random.default_rng(seed).normal(0, 1, count)
    return (whine + bed).astype(np.float32)


def _melody_signal(sr, seconds=8):
    """Builds tonal audio whose pitch changes constantly but whose speed is steady."""
    t = np.arange(sr * seconds, dtype=np.float64) / sr
    signal = np.zeros_like(t)
    notes = [261.6, 329.6, 392.0, 523.3, 440.0, 349.2, 293.7, 587.3, 220.0, 660.0]
    seg_len = len(t) // len(notes)
    for idx, freq in enumerate(notes):
        chunk = slice(idx * seg_len, (idx + 1) * seg_len)
        signal[chunk] = 0.7 * np.sin(2.0 * np.pi * freq * t[chunk])
    return signal.astype(np.float32)


def test_detect_flutter_ignores_stable_speed_reference():
    """A steady line-rate reference means steady tape speed, so no drift."""
    sr = 44100
    assert modules.auto_scanner._detect_flutter_or_pitch_drift(_speed_reference_signal(sr), sr) is False


def test_detect_flutter_reports_drift_for_unstable_speed_reference():
    """A line-rate reference wobbling by 1% is genuine wow and must select DTW."""
    sr = 44100
    signal = _speed_reference_signal(sr, depth=0.01)
    assert modules.auto_scanner._detect_flutter_or_pitch_drift(signal, sr) is True
    assert modules.auto_scanner._measure_tape_speed_deviation(signal, sr) > 0.005


def test_detect_flutter_ignores_content_pitch_changes():
    """Musical pitch moves the dominant peak but not tape speed, so it is not drift."""
    sr = 44100
    assert modules.auto_scanner._detect_flutter_or_pitch_drift(_melody_signal(sr), sr) is False


def test_detect_flutter_ignores_broadband_noise():
    """Broadband noise carries no speed reference and must not be read as drift."""
    sr = 44100
    noise = (0.3 * np.random.default_rng(3).normal(0, 1, sr * 8)).astype(np.float32)
    assert modules.auto_scanner._detect_flutter_or_pitch_drift(noise, sr) is False


def test_track_speed_reference_needs_enough_frames():
    """Recordings too short to sample the reference yield no measurement."""
    short = np.zeros(40000, dtype=np.float32)
    assert modules.auto_scanner._track_speed_reference(short, 8000, 15625.0) == []
    assert modules.auto_scanner._detect_flutter_or_pitch_drift(short, 8000) is False


def test_best_fitting_reference_prefers_nominal_match():
    """PAL and NTSC bands overlap, so the closest-to-nominal track wins."""
    pal_track = [15625.0] * modules.auto_scanner.DRIFT_MIN_FRAMES
    ntsc_track = [15625.0] * (modules.auto_scanner.DRIFT_MIN_FRAMES * 2)
    tracks = [(15625.0, pal_track), (15734.0, ntsc_track)]

    # The NTSC track is longer, but its mean sits 109 Hz from nominal.
    assert modules.auto_scanner._best_fitting_reference(tracks) == pal_track


def test_best_fitting_reference_without_usable_track():
    """No reference with enough frames means no measurement at all."""
    assert modules.auto_scanner._best_fitting_reference([(15625.0, []), (15734.0, [15625.0])]) == []


@pytest.mark.parametrize(
    ("spectrum", "peak_idx", "expected"),
    [
        ([5.0, 1.0, 1.0], 0, 0.0),
        ([1.0, 1.0, 5.0], 2, 2.0),
        ([1.0, 1.0, 1.0], 1, 1.0),
        ([1.0, 3.0, 1.0], 1, 1.0),
        ([1.0, 3.0, 2.0], 1, 1.0 + 1.0 / 6.0),
    ],
)
def test_refine_peak_bin_interpolation(spectrum, peak_idx, expected):
    """Parabolic refinement handles edges, flat peaks, and asymmetric peaks."""
    assert modules.auto_scanner._refine_peak_bin(np.array(spectrum), peak_idx) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        ([1.0, 2.0], 0.0),
        ([0.0] * 10, 0.0),
        ([100.0] * 10, 0.0),
    ],
)
def test_relative_deviation_guards(observed, expected):
    """Deviation is zero for undersized samples, zero means, and perfectly steady tracks."""
    assert modules.auto_scanner._relative_deviation(observed) == pytest.approx(expected)


@pytest.mark.parametrize(
    "speech,music,ambient,nf,expected_mode",
    [
        (0.5, 0.4, 0.2, -45.0, "hybrid"),
        (0.6, 0.05, 0.05, -60.0, "hybrid"),
        (0.6, 0.05, 0.05, -35.0, "hybrid"),
        (0.05, 0.5, 0.1, -45.0, "denoise_only"),
        (0.05, 0.05, 0.05, -30.0, "auto_ffmpeg_native"),
    ],
)
def test_select_strategy_mode_rules(speech, music, ambient, nf, expected_mode):
    """Verify decision rules for every mode _select_strategy_mode can return."""
    mode, _ = modules.auto_scanner._select_strategy_mode(speech, music, ambient, nf)
    assert mode == expected_mode


def test_evaluate_restoration_strategy():
    """Verify parameter auto-tuning and strategy payload generation."""
    profile = {
        "speech_ratio": 0.6,
        "music_ratio": 0.4,
        "ambient_ratio": 0.3,
        "noise_floor_db": -40.0,
        "has_drift": True,
    }
    strategy = modules.auto_scanner.evaluate_restoration_strategy(profile)
    assert strategy["mode"] == "hybrid"
    assert strategy["sync_method"] == "dtw"
    assert strategy["enhance_nfe"] >= 256
    assert "profile" in strategy


def test_scan_and_decide_restoration_strategy(tmp_path):
    """Verify end-to-end scanner execution on generated audio file."""
    sr = 44100
    t = np.arange(sr * 2, dtype=np.float32) / float(sr)
    # Combined speech (1000Hz) + music chord (440Hz) + noise
    audio = (0.4 * np.sin(2.0 * np.pi * 1000.0 * t) + 0.3 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    wav_file = tmp_path / "scene.wav"
    sf.write(str(wav_file), audio, sr)

    strategy = modules.auto_scanner.scan_and_decide_restoration_strategy(wav_file)
    assert strategy["mode"] in ("hybrid", "arnndn_speech", "denoise_only", "auto_ffmpeg_native")
    assert strategy["enhance_nfe"] >= 256


def test_scan_and_decide_fallback_on_read_failure(tmp_path):
    """Verify fallback to default strategy when audio read fails."""
    wav_file = tmp_path / "corrupt.wav"
    wav_file.write_text("not-a-wav-file")

    with patch("modules.auto_scanner._read_stereo_audio_for_analysis", return_value=(None, None)):
        strategy = modules.auto_scanner.scan_and_decide_restoration_strategy(wav_file)
        assert strategy["mode"] == "hybrid"


def test_pick_optimal_denoise_model():
    """Verify DeNoise vs DeNoise-Lite selection."""
    assert modules.auto_scanner._pick_optimal_denoise_model(-30.0) == "UVR-DeNoise.pth"
    assert modules.auto_scanner._pick_optimal_denoise_model(-45.0) == "UVR-DeNoise-Lite.pth"


def test_pick_optimal_vocals_model():
    """Verify crowd vs standard BS-Roformer selection."""
    assert "crowd" in modules.auto_scanner._pick_optimal_vocals_model(0.6, 0.5)
    assert "bs_roformer" in modules.auto_scanner._pick_optimal_vocals_model(0.4, 0.1)


def test_pick_optimal_arnndn_model():
    """Verify RNNoise model selection for rumble, high noise, and dialogue."""
    assert modules.auto_scanner._pick_optimal_arnndn_model(60, -45.0) == "sh.rnnn"
    assert modules.auto_scanner._pick_optimal_arnndn_model(0, -30.0) == "bd.rnnn"
    assert modules.auto_scanner._pick_optimal_arnndn_model(0, -50.0) == "cb.rnnn"


def test_classify_temporal_window():
    """Verify classification of acoustic activity inside a single temporal window."""
    sr = 44100
    t = np.arange(sr * 2, dtype=np.float32) / float(sr)
    speech_chunk = (0.8 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
    s, m, a = modules.auto_scanner._classify_temporal_window(speech_chunk, sr)
    assert s is True


def test_scan_temporal_scene_windows():
    """Verify temporal window sliding analysis on short and multi-second signals."""
    sr = 44100
    # Short signal fallback
    short_res = modules.auto_scanner._scan_temporal_scene_windows(np.zeros(100, dtype=np.float32), sr)
    assert short_res["window_count"] == 1

    # 6-second synthetic signal
    t = np.arange(sr * 6, dtype=np.float32) / float(sr)
    signal = (0.5 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
    res = modules.auto_scanner._scan_temporal_scene_windows(signal, sr, window_sec=5.0, hop_sec=2.5)
    assert res["window_count"] >= 1
    assert "has_dialogue" in res
    assert "dialogue_ratio" in res


def test_extract_profile_with_precision_metrics():
    """Verify profile extraction includes DC offset, balance, CRT notch, and resonance."""
    sr = 48000
    t = np.arange(sr * 2, dtype=np.float32)
    left = np.sin(2.0 * np.pi * 15625.0 * t / sr) + 0.02
    right = left * 0.5
    stereo = np.column_stack([left, right])
    mono = np.mean(stereo, axis=1)

    profile = modules.auto_scanner._extract_profile_from_signal(mono, sr, stereo_signal=stereo)
    assert profile["has_dc_offset"] is True
    assert profile["balance_db"] > 4.0
    assert profile["crt_notch_hz"] == 15625.0


def test_evaluate_strategy_with_precision_metrics():
    """Verify strategy evaluation propagates DC offset, balance, and CRT notch."""
    profile = {
        "speech_ratio": 0.6,
        "music_ratio": 0.1,
        "ambient_ratio": 0.1,
        "noise_floor_db": -40.0,
        "has_dc_offset": True,
        "balance_db": 5.0,
        "crt_notch_hz": 15625.0,
    }
    strategy = modules.auto_scanner.evaluate_restoration_strategy(profile)
    assert strategy["precondition_filters"]["enable_dc_block"] is True
    assert strategy["precondition_filters"]["balance_db"] == 5.0
    assert strategy["precondition_filters"]["crt_notch_hz"] == 15625.0


def _rhythmic_signal(sr, beat_hz=2.0, seconds=8, seed=5):
    """Builds tonal audio with a steady beat, as music has."""
    count = sr * seconds
    t = np.arange(count, dtype=np.float64) / sr
    envelope = np.exp(-8.0 * (t * beat_hz % 1.0))
    tone = 0.6 * np.sin(2.0 * np.pi * 220.0 * t) + 0.4 * np.sin(2.0 * np.pi * 330.0 * t)
    noise = 0.02 * np.random.default_rng(seed).normal(0, 1, count)
    return (tone * envelope + noise).astype(np.float32)


def _irregular_signal(sr, seconds=8, seed=5):
    """Builds tonal bursts at irregular intervals, as conversation has."""
    rng = np.random.default_rng(seed)
    count = sr * seconds
    t = np.arange(count, dtype=np.float64) / sr
    envelope = np.zeros(count)
    position = 0
    while position < count:
        length = int(sr * rng.uniform(0.15, 0.5))
        end = position + length
        envelope[position:end] = 1.0
        position = end + int(sr * rng.uniform(0.1, 0.6))
    tone = 0.6 * np.sin(2.0 * np.pi * 220.0 * t)
    return (tone * envelope + 0.02 * rng.normal(0, 1, count)).astype(np.float32)


def test_onset_periodicity_separates_a_beat_from_irregular_speech():
    """A steady beat must score above the music threshold; irregular onsets below."""
    sr = 44100
    rhythmic = modules.auto_scanner._estimate_onset_periodicity(_rhythmic_signal(sr), sr)
    irregular = modules.auto_scanner._estimate_onset_periodicity(_irregular_signal(sr), sr)

    assert rhythmic >= modules.auto_scanner.MUSIC_PERIODICITY_THRESHOLD
    assert irregular < modules.auto_scanner.MUSIC_PERIODICITY_THRESHOLD


def test_onset_periodicity_needs_enough_audio():
    """Too short to hold a full lag window means no measurement at all."""
    assert modules.auto_scanner._estimate_onset_periodicity(np.zeros(4096, dtype=np.float32), 44100) == 0.0


def test_onset_periodicity_of_silence_is_zero():
    """Silence has no onsets, so autocorrelation cannot be normalised."""
    assert modules.auto_scanner._estimate_onset_periodicity(np.zeros(44100 * 8, dtype=np.float32), 44100) == 0.0


@pytest.mark.parametrize(
    ("periodicity", "expected_mode"),
    [
        # Rhythm is checked before the dialogue gate: sung vocals trip every
        # speech test, so a music video would otherwise always reach 'hybrid'.
        (0.60, "denoise_only"),
        (0.45, "denoise_only"),
        (0.44, "hybrid"),
        (0.10, "hybrid"),
    ],
)
def test_rhythmic_music_routes_away_from_stem_separation(periodicity, expected_mode):
    """Speech-dominant input still reaches 'hybrid' unless a real beat is present."""
    mode, _ = modules.auto_scanner._select_strategy_mode(1.0, 0.02, 0.04, -45.0, True, periodicity)
    assert mode == expected_mode


def test_strategy_passes_measured_periodicity_into_mode_selection():
    """The profile value must actually reach the decision, not a default."""
    profile = {"speech_ratio": 1.0, "music_ratio": 0.02, "ambient_ratio": 0.04, "onset_periodicity": 0.7}
    assert modules.auto_scanner.evaluate_restoration_strategy(profile)["mode"] == "denoise_only"
