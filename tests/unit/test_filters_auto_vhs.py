from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

import modules.filters


def _rng():
    """Fresh seeded generator so threshold assertions never depend on global RNG state."""
    return np.random.default_rng(20240517)


def test_estimate_noise_floor_and_reduction():
    """Verify noise floor estimation and reduction tier assignments."""
    short_sig = np.zeros(500, dtype=np.float32)
    assert modules.filters._estimate_noise_floor_and_reduction(short_sig) == (-45.0, 12.0)

    clean_sig = _rng().normal(0, 0.0001, 10000).astype(np.float32)
    nf_clean, nr_clean = modules.filters._estimate_noise_floor_and_reduction(clean_sig)
    assert nf_clean < -60.0 and nr_clean == 8.0

    loud_sig = _rng().normal(0, 0.1, 10000).astype(np.float32)
    nf_loud, nr_loud = modules.filters._estimate_noise_floor_and_reduction(loud_sig)
    assert nf_loud > -35.0 and nr_loud == 16.0


def test_detect_mains_buzz_notch():
    """Verify 50Hz and 60Hz hum peak detection."""
    sr = 44100
    n = 32768
    t = np.arange(n, dtype=np.float32) / float(sr)
    hum_50 = (0.5 * np.sin(2.0 * np.pi * 50.0 * t)).astype(np.float32)
    assert modules.filters._detect_mains_buzz_notch(hum_50, sr) == 50.0

    hum_60 = (0.5 * np.sin(2.0 * np.pi * 60.0 * t)).astype(np.float32)
    assert modules.filters._detect_mains_buzz_notch(hum_60, sr) == 60.0

    silence = np.zeros(100, dtype=np.float32)
    assert modules.filters._detect_mains_buzz_notch(silence, sr) == 0.0


def test_detect_low_frequency_rumble():
    """Verify sub-bass motor rumble frequency detection."""
    sr = 44100
    n = 16384
    t = np.arange(n, dtype=np.float32) / float(sr)
    rumble = (0.8 * np.sin(2.0 * np.pi * 35.0 * t)).astype(np.float32)
    assert modules.filters._detect_low_frequency_rumble(rumble, sr) >= 60

    silence = np.zeros(100, dtype=np.float32)
    assert modules.filters._detect_low_frequency_rumble(silence, sr) == 60


def test_detect_click_density():
    """Verify pop/click impulse detection."""
    sig = np.zeros(5000, dtype=np.float32)
    sig[::100] = 1.0
    assert modules.filters._detect_click_density(sig) is True

    smooth = np.linspace(0, 1.0, 5000, dtype=np.float32)
    assert modules.filters._detect_click_density(smooth) is False


def test_analyze_vhs_audio_profile(tmp_path):
    """Verify profile extraction from audio file."""
    wav_path = tmp_path / "test_prof.wav"
    sr = 44100
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    data = (0.3 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    sf.write(str(wav_path), data, sr)

    prof = modules.filters._analyze_vhs_audio_profile(wav_path)
    assert "nr" in prof
    assert "nf" in prof
    assert "highpass" in prof
    assert prof["notch"] == 50.0


@patch("modules.filters.is_valid_audio", return_value=True)
def test_filter_auto_vhs_native_step_skips_when_valid(mock_valid, tmp_path):
    """Step skips filtering if output is already valid."""
    orig = tmp_path / "orig.wav"
    out_dir = tmp_path / "out_dir"
    out_dir.mkdir()
    result = modules.filters._filter_auto_vhs_native_step(orig, out_dir)
    assert result == out_dir / "auto_vhs_filtered_orig.wav"


@patch("modules.filters.attempt_cpu_run_with_retry")
@patch("modules.filters.is_valid_audio")
def test_filter_auto_vhs_native_step_success(mock_valid, mock_retry, tmp_path):
    """Step analyzes audio, runs filter command, and promotes output."""
    mock_valid.side_effect = [False, True]
    orig = tmp_path / "orig.wav"
    orig.write_text("audio")
    out_dir = tmp_path / "out_dir"
    out_dir.mkdir()

    output_wav = out_dir / "auto_vhs_filtered_orig.wav"
    tmp_wav = output_wav.with_suffix(".tmp.wav")

    def fake_retry(cmd_builder, threads, description, total_duration):
        tmp_wav.write_text("filtered")
        return True

    mock_retry.side_effect = fake_retry
    with patch(
        "modules.filters._analyze_vhs_audio_profile",
        return_value={"nr": 12.0, "nf": -45.0, "tn": True, "highpass": 60, "adeclick": True, "notch": 0.0},
    ):
        res = modules.filters._filter_auto_vhs_native_step(orig, out_dir)
        assert res == output_wav
        assert output_wav.exists()


def test_detect_click_density_short_signal():
    """Verify click detector fallback on short signal."""
    assert modules.filters._detect_click_density(np.zeros(100, dtype=np.float32)) is True


def test_detect_low_frequency_rumble_tiers():
    """Verify 45Hz and 0Hz tiers in low frequency rumble detector."""
    sr = 44100
    t = np.arange(16384, dtype=np.float32) / float(sr)
    tone_45 = (0.2 * np.sin(2.0 * np.pi * 35.0 * t) + 0.8 * np.sin(2.0 * np.pi * 500.0 * t)).astype(np.float32)
    assert modules.filters._detect_low_frequency_rumble(tone_45, sr) == 45

    high_tone = (0.8 * np.sin(2.0 * np.pi * 1500.0 * t)).astype(np.float32)
    assert modules.filters._detect_low_frequency_rumble(high_tone, sr) == 0


def test_read_audio_for_analysis_stereo_and_fallbacks(tmp_path):
    """Test stereo downmix and fallback behavior when soundfile fails or is missing."""
    sr = 44100
    t = np.arange(16384, dtype=np.float32) / float(sr)
    high_tone = (0.8 * np.sin(2.0 * np.pi * 1500.0 * t)).astype(np.float32)
    stereo_path = tmp_path / "stereo.wav"
    stereo_data = np.column_stack([high_tone, high_tone])
    sf.write(str(stereo_path), stereo_data, sr)

    prof = modules.filters._analyze_vhs_audio_profile(stereo_path)
    assert prof["highpass"] == 0

    with patch("modules.filters.sf.read", side_effect=RuntimeError("Corrupt audio")):
        assert modules.filters._read_audio_for_analysis(stereo_path) == (None, None)

    with patch("modules.filters.sf", None):
        assert modules.filters._read_audio_for_analysis(stereo_path) == (None, None)
        assert modules.filters._analyze_vhs_audio_profile(stereo_path)["nr"] == 12.0


@pytest.mark.parametrize(
    "notch_freq,expected_count,expected_snippet",
    [
        (0.0, 0, None),
        (50.0, 2, "f=50.0"),
        (59.94, 2, "f=59.94"),
        (60.0, 2, "f=60.0"),
        # A harmonic now also pulls in the fundamental it belongs to.
        (120.0, 2, "f=60.0"),
        (100.0, 2, "f=50.0"),
    ],
)
def test_append_notch_filters(notch_freq, expected_count, expected_snippet):
    """Verify fundamental and harmonic notch generation."""
    filters = []
    modules.filters._append_notch_filters(filters, notch_freq)
    assert len(filters) == expected_count
    if expected_snippet:
        assert expected_snippet in filters[0]


def test_detect_analog_clipping():
    """Verify clipped analog peak detection."""
    clean = np.sin(np.linspace(0, 100, 2048, dtype=np.float32)) * 0.8
    assert not modules.filters._detect_analog_clipping(clean)

    clipped = np.clip(np.sin(np.linspace(0, 100, 2048, dtype=np.float32)) * 1.5, -1.0, 1.0)
    assert modules.filters._detect_analog_clipping(clipped)
    assert not modules.filters._detect_analog_clipping(np.zeros(10, dtype=np.float32))


def test_detect_stereo_azimuth_skew():
    """Verify sub-millisecond inter-channel time delay estimation."""
    sr = 44100
    t = np.arange(4096, dtype=np.float32)
    left = np.sin(2.0 * np.pi * 440.0 * t / sr)
    right = np.roll(left, 5)  # 5 sample delay
    stereo = np.column_stack([left, right])

    skew_ms = modules.filters._detect_stereo_azimuth_skew(stereo, sr)
    assert abs(skew_ms) > 0.0
    assert modules.filters._detect_stereo_azimuth_skew(None, sr) == 0.0
    assert modules.filters._detect_stereo_azimuth_skew(left, sr) == 0.0


def test_build_precondition_filter_string_advanced():
    """Verify composite pre-conditioning filter string with declip and azimuth."""
    s_full = modules.filters._build_precondition_filter_string(
        highpass_freq=60,
        enable_adeclick=True,
        notch_freq=50.0,
        enable_adeclip=True,
        azimuth_delay_ms=0.15,
    )
    assert "adeclip" in s_full
    assert "adelay=7S|0" in s_full
    assert "bandreject=f=50.0" in s_full

    s_neg = modules.filters._build_precondition_filter_string(
        highpass_freq=0,
        enable_adeclick=False,
        notch_freq=0.0,
        enable_adeclip=False,
        azimuth_delay_ms=-0.25,
    )
    assert s_neg == "adelay=0|11S"


def test_detect_crt_flyback_notch():
    """Verify detection of 15.625kHz / 15.734kHz CRT flyback spikes."""
    sr = 48000
    t = np.arange(32768, dtype=np.float32)
    sig = np.sin(2.0 * np.pi * 15625.0 * t / sr) + _rng().normal(0, 0.05, len(t)).astype(np.float32)
    crt_hz = modules.filters._detect_crt_flyback_notch(sig, sr)
    assert crt_hz == 15625.0
    assert modules.filters._detect_crt_flyback_notch(np.zeros(10, dtype=np.float32), sr) == 0.0


def test_detect_dc_offset_bias():
    """Verify detection of hardware digitizer DC voltage bias."""
    clean = np.zeros(2048, dtype=np.float32)
    assert not modules.filters._detect_dc_offset_bias(clean)
    biased = np.ones(2048, dtype=np.float32) * 0.02
    assert modules.filters._detect_dc_offset_bias(biased)
    assert not modules.filters._detect_dc_offset_bias(np.zeros(10, dtype=np.float32))


def test_detect_stereo_balance_imbalance():
    """Verify stereo Left/Right channel volume imbalance calculation."""
    left = np.ones(4096, dtype=np.float32)
    right = np.ones(4096, dtype=np.float32) * 0.5  # ~6 dB difference
    stereo = np.column_stack([left, right])
    diff_db = modules.filters._detect_stereo_balance_imbalance(stereo)
    assert diff_db > 5.0
    assert modules.filters._detect_stereo_balance_imbalance(None) == 0.0


def _resonant_noise(sr, centre_hz, q_factor, seconds=3):
    """Shapes white noise with a resonance envelope, as a housing cavity would."""
    count = sr * seconds
    noise = _rng().normal(0, 0.05, count)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(count, 1.0 / sr)
    bandwidth = centre_hz / q_factor
    envelope = 1.0 + 40.0 / (1.0 + ((freqs - centre_hz) / (bandwidth / 2.0)) ** 2)
    shaped = np.fft.irfft(spectrum * envelope, n=count)
    tone = 0.3 * np.sin(2.0 * np.pi * 440.0 * np.arange(count) / sr)
    return (shaped + tone).astype(np.float32)


def test_detect_enclosure_resonance_notch():
    """A genuine housing resonance is detected by its broad, high-contrast bump."""
    sr = 44100
    res_hz = modules.filters._detect_enclosure_resonance_notch(_resonant_noise(sr, 2500.0, 15.0), sr)
    assert abs(res_hz - 2500.0) < 120.0
    assert modules.filters._detect_enclosure_resonance_notch(np.zeros(10, dtype=np.float32), sr) == 0.0


def test_resonance_detector_ignores_tones_and_speech_formants():
    """Speech formants and harmonics live in the same band and must not be notched."""
    sr = 44100
    t = np.arange(sr * 3, dtype=np.float64) / sr
    noise = _rng().normal(0, 0.05, len(t))

    pure_tone = (0.3 * np.sin(2.0 * np.pi * 2000.0 * t) + noise).astype(np.float32)
    assert modules.filters._detect_enclosure_resonance_notch(pure_tone, sr) == 0.0

    harmonics = sum(0.3 / k * np.sin(2.0 * np.pi * 150.0 * k * t) for k in range(1, 12))
    voice = (harmonics + noise).astype(np.float32)
    assert modules.filters._detect_enclosure_resonance_notch(voice, sr) == 0.0


@pytest.mark.parametrize(
    ("contrast", "width_hz", "expected"),
    [
        (25.0, 200.0, True),
        (19.9, 200.0, False),
        (25.0, 49.0, False),
        (25.0, 401.0, False),
        (25.0, 50.0, True),
        (25.0, 400.0, True),
    ],
)
def test_is_resonance_shaped(contrast, width_hz, expected):
    """Only a high-contrast bump of moderate width counts as a resonance."""
    assert modules.filters._is_resonance_shaped(contrast, width_hz) is expected


@pytest.mark.parametrize(
    ("detected", "expected"),
    [
        (0.0, []),
        (50.0, ["50.0", "100.0"]),
        (60.0, ["60.0", "120.0"]),
        # A harmonic win must still notch the fundamental it belongs to.
        (100.0, ["50.0", "100.0"]),
        (120.0, ["60.0", "120.0"]),
    ],
)
def test_append_mains_notches_covers_the_fundamental(detected, expected):
    """Locking onto the second harmonic must not leave the fundamental un-notched."""
    filters = []
    modules.filters._append_mains_notches(filters, detected)
    assert [x.split(":")[0].replace("bandreject=f=", "") for x in filters] == expected


def test_noise_floor_samples_the_whole_signal():
    """A loud opening must not set the noise floor for an otherwise quiet recording."""
    sr = 44100
    loud = _rng().normal(0, 0.5, sr * 3)
    quiet = _rng().normal(0, 0.002, sr * 30)
    signal = np.concatenate([loud, quiet]).astype(np.float32)

    nf_db, _ = modules.filters._estimate_noise_floor_and_reduction(signal)
    loud_only, _ = modules.filters._estimate_noise_floor_and_reduction(loud.astype(np.float32))
    assert nf_db < loud_only - 10.0


def test_build_precondition_filter_string_full_suite():
    """Verify pre-conditioning string with DC block, balance, and CRT notch."""
    s = modules.filters._build_precondition_filter_string(
        highpass_freq=45,
        enable_adeclick=True,
        notch_freq=50.0,
        enable_adeclip=True,
        azimuth_delay_ms=0.10,
        enable_dc_block=True,
        balance_db=3.0,
        crt_notch=15625.0,
        resonance_freq=2400.0,
    )
    # Exact stage matching: DC-block highpass and the requested rumble cutoff.
    assert {"highpass=f=2", "highpass=f=45"} <= set(s.split(","))
    assert "pan=stereo" in s
    assert "bandreject=f=15625.0" in s
    assert "bandreject=f=2400.0" in s


@pytest.mark.parametrize(
    ("balance_db", "expected"),
    [
        # Positive means left is louder, so left is the side that must come down.
        (6.0, "pan=stereo|c0=0.501*c0|c1=c1"),
        (3.0, "pan=stereo|c0=0.708*c0|c1=c1"),
        (-6.0, "pan=stereo|c0=c0|c1=0.501*c1"),
        (-3.0, "pan=stereo|c0=c0|c1=0.708*c1"),
    ],
)
def test_balance_correction_attenuates_the_louder_channel(balance_db, expected):
    """Levelling must lower the loud side; lowering the quiet side doubles the imbalance."""
    filters = []
    modules.filters._append_balance_correction(filters, balance_db)
    assert filters == [expected]


@pytest.mark.parametrize("balance_db", [0.0, 0.4, -0.4, 12.1, -12.1, 24.9, -24.9])
def test_balance_correction_skips_negligible_and_ambiguous_imbalances(balance_db):
    """Sub-decibel gaps need no fix; the 12-25 dB band is too uncertain to touch."""
    filters = []
    modules.filters._append_balance_correction(filters, balance_db)
    assert filters == []


def test_balance_correction_boundary_is_inclusive():
    """The maximum correctable imbalance is still corrected."""
    filters = []
    modules.filters._append_balance_correction(filters, modules.filters.BALANCE_MAX_DB)
    assert len(filters) == 1


@pytest.mark.parametrize(
    ("balance_db", "expected"),
    [
        # Positive means left is the live side, so left is mirrored to both.
        (49.7, "pan=stereo|c0=c0|c1=c0"),
        (25.1, "pan=stereo|c0=c0|c1=c0"),
        (-29.9, "pan=stereo|c0=c1|c1=c1"),
        (-53.4, "pan=stereo|c0=c1|c1=c1"),
    ],
)
def test_dead_channel_collapses_to_dual_mono(balance_db, expected):
    """A one-sided recording is centred by mirroring whichever channel is alive."""
    filters = []
    modules.filters._append_balance_correction(filters, balance_db)
    assert filters == [expected]


def test_channel_correlation_separates_stereo_from_dead_channel():
    """Correlation is the evidence that both channels share content."""
    sr = 44100
    t = np.arange(sr, dtype=np.float64) / sr
    live = np.sin(2.0 * np.pi * 440.0 * t)
    silence = _rng().normal(0, 1e-5, len(t))

    assert modules.filters._channel_correlation(live, live) == pytest.approx(1.0)
    assert modules.filters._channel_correlation(live, silence) < 0.3
    assert modules.filters._channel_correlation(live, np.zeros_like(live)) == 0.0


def test_azimuth_skew_ignores_uncorrelated_channels():
    """A lag read off a dead channel is noise, so it must not become an adelay."""
    sr = 44100
    t = np.arange(32768, dtype=np.float64) / sr
    live = np.sin(2.0 * np.pi * 440.0 * t)
    dead = _rng().normal(0, 1e-4, len(t))

    one_sided = np.column_stack([live, dead]).astype(np.float32)
    assert modules.filters._detect_stereo_azimuth_skew(one_sided, sr) == 0.0

    # A correlated pair with a real 5-sample offset is still measured.
    correlated = np.column_stack([live, np.roll(live, 5)]).astype(np.float32)
    assert modules.filters._detect_stereo_azimuth_skew(correlated, sr) != 0.0


@pytest.mark.parametrize(
    ("whine_hz", "expected"),
    [
        (15625.0, 15625.0),
        # A tape running slightly fast shifts the whine off nominal; it is still PAL.
        (15633.0, 15625.0),
        (15660.0, 15625.0),
        (15734.0, 15734.0),
        (15700.0, 15734.0),
    ],
)
def test_crt_flyback_classified_by_nearest_line_rate(whine_hz, expected):
    """An off-nominal whine must still resolve to the standard it belongs to."""
    sr = 44100
    t = np.arange(sr, dtype=np.float64) / sr
    signal = (0.2 * np.sin(2.0 * np.pi * whine_hz * t) + _rng().normal(0, 0.01, len(t))).astype(np.float32)
    assert modules.filters._detect_crt_flyback_notch(signal, sr) == expected


def test_crt_flyback_absent_when_no_whine():
    """Tape-like audio with no line whistle must not report a standard.

    The prominence gate is calibrated against capture spectra, which roll off well
    below the line rate; flat full-band noise is not representative of a VHS rip.
    """
    sr = 44100
    noise = _rng().normal(0, 0.05, sr)
    spectrum = np.fft.rfft(noise)
    spectrum[np.fft.rfftfreq(len(noise), 1.0 / sr) > 12000.0] = 0.0
    tape_like = np.fft.irfft(spectrum, n=len(noise)).astype(np.float32)
    assert modules.filters._detect_crt_flyback_notch(tape_like, sr) == 0.0


@pytest.mark.parametrize(
    ("line_rate_hz", "hum_hz", "expected"),
    [
        # PAL rules out the 60 Hz family, so a stray 60 Hz tone is not "hum".
        (15625.0, 60.0, 0.0),
        (15625.0, 50.0, 50.0),
        # NTSC rules out the 50 Hz family.
        (15734.0, 50.0, 0.0),
        (15734.0, 60.0, 60.0),
        # With no line rate known, both families stay in play.
        (0.0, 60.0, 60.0),
        (0.0, 50.0, 50.0),
    ],
)
def test_mains_candidates_constrained_by_line_rate(line_rate_hz, hum_hz, expected):
    """The video standard fixes which mains frequency is physically possible."""
    sr = 44100
    t = np.arange(32768, dtype=np.float64) / sr
    signal = (0.5 * np.sin(2.0 * np.pi * hum_hz * t)).astype(np.float32)
    assert modules.filters._detect_mains_buzz_notch(signal, sr, line_rate_hz) == expected
