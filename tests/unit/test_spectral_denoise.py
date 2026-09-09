"""Tests for the noise-profile spectral subtraction stage.

The stage exists because UVR-DeNoise is inert on quiet captures: measured against a clean
reference at a 8.9 dB programme-to-noise margin it left log-spectral distance unchanged at
18.0 dB, where subtraction reached 7.6 dB. It is gated on that margin because above the
threshold the neural stage is the better tool and subtraction costs a little fidelity.

These tests pin the gate in both directions and the fallbacks, since a host without the
Cathar CLI must still be able to run the mode.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from modules import config, spectral_denoise


@pytest.fixture(name="quiet_wav")
def _quiet_wav(tmp_path):
    """A stand-in path; every test patches the measurement rather than reading audio."""
    wav = tmp_path / "source.wav"
    wav.write_text("audio")
    return wav


def test_disabled_by_configuration_skips_the_stage(quiet_wav):
    """The stage can be switched off wholesale without touching the chain."""
    with patch.object(spectral_denoise, "APL_ENABLE_SPECTRAL_DENOISE", False):
        assert spectral_denoise.should_apply(quiet_wav) is None


def test_pristine_material_skips_the_stage(quiet_wav):
    """Only audio already free of noise is left alone.

    A 20 dB gate was tried first, from synthetic fixtures. On real tape it was holding the
    stage back: raising it improved 6 of 25 captures by 5.61 dB of noise removal at no
    measurable programme cost. The guard now catches only pristine sources, well above the
    51.7 dB margin of the loudest capture in the corpus.
    """
    with patch.object(spectral_denoise, "estimate_snr_margin_db", return_value=72.0):
        assert spectral_denoise.should_apply(quiet_wav) is None


def test_ordinary_tape_material_is_treated(quiet_wav):
    """A healthy-sounding capture still carries tape noise worth subtracting."""
    with patch.object(spectral_denoise, "estimate_snr_margin_db", return_value=29.3):
        assert spectral_denoise.should_apply(quiet_wav) == 29.3


def test_quiet_material_selects_the_stage(quiet_wav):
    """Below the threshold the measured margin is returned so it can be logged."""
    with patch.object(spectral_denoise, "estimate_snr_margin_db", return_value=8.9):
        assert spectral_denoise.should_apply(quiet_wav) == 8.9


def test_unreadable_audio_skips_the_stage(quiet_wav):
    """An unmeasurable input is skipped rather than guessed at."""
    with patch.object(spectral_denoise, "estimate_snr_margin_db", return_value=None):
        assert spectral_denoise.should_apply(quiet_wav) is None


def test_pristine_material_is_returned_untouched(quiet_wav, tmp_path):
    """The chain continues with the original audio when the stage does not apply."""
    with patch.object(spectral_denoise, "estimate_snr_margin_db", return_value=72.0):
        assert spectral_denoise.apply_when_needed(quiet_wav, tmp_path) == quiet_wav


def test_missing_cathar_binary_falls_back_to_the_neural_stage(quiet_wav, tmp_path):
    """A host that never provisioned the CLI must still be able to run this mode."""
    with (
        patch.object(spectral_denoise, "estimate_snr_margin_db", return_value=8.9),
        patch("modules.cathar._require_cathar_binary", side_effect=FileNotFoundError("cathar")),
    ):
        assert spectral_denoise.apply_when_needed(quiet_wav, tmp_path) == quiet_wav


def test_quiet_material_is_subtracted(quiet_wav, tmp_path):
    """A quiet capture is denoised, at the gentler over-subtraction factor."""
    produced = tmp_path / "denoised.wav"
    with (
        patch.object(spectral_denoise, "estimate_snr_margin_db", return_value=8.9),
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._cathar_noiseprint_step", return_value=tmp_path / "noise.json"),
        patch("modules.cathar._cathar_denoise_step", return_value=produced) as mock_denoise,
    ):
        assert spectral_denoise.apply_when_needed(quiet_wav, tmp_path) == produced
    assert mock_denoise.call_args.kwargs["alpha"] == spectral_denoise.APL_SPECTRAL_ALPHA


def test_the_noise_probe_is_this_modes_own_and_leaves_cathar_alone(quiet_wav, tmp_path):
    """The probe duration is passed explicitly, so the shared cathar value cannot move.

    A 0.75 s probe was the largest single thing holding this mode back: 2.5 s removes
    3.39 dB more noise for 0.12 dB more programme deviation on 25 real captures, which is
    what puts the mode ahead of cathar on both halves of the trade. cathar shipped in
    v1.2.0 on 0.75 s and the setting is shared, so passing it explicitly is the only way to
    take the gain without moving a released mode.
    """
    produced = tmp_path / "denoised.wav"
    with (
        patch.object(spectral_denoise, "estimate_snr_margin_db", return_value=8.9),
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._cathar_noiseprint_step", return_value=tmp_path / "noise.json") as mock_noiseprint,
        patch("modules.cathar._cathar_denoise_step", return_value=produced),
    ):
        spectral_denoise.apply_when_needed(quiet_wav, tmp_path)
    assert mock_noiseprint.call_args.kwargs["duration_s"] == spectral_denoise.APL_NOISEPRINT_DURATION_S
    assert spectral_denoise.APL_NOISEPRINT_DURATION_S == 2.5
    assert config.CATHAR_NOISEPRINT_DURATION_S == 0.75


def test_a_failed_subtraction_leaves_the_audio_usable(quiet_wav, tmp_path):
    """A failure here must not take the restoration down with it."""
    with (
        patch.object(spectral_denoise, "estimate_snr_margin_db", return_value=8.9),
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._cathar_noiseprint_step", return_value=None),
        patch("modules.cathar._cathar_denoise_step", side_effect=RuntimeError("cathar exploded")),
    ):
        assert spectral_denoise.apply_when_needed(quiet_wav, tmp_path) == quiet_wav


def test_the_output_directory_is_created(quiet_wav, tmp_path):
    """The stage owns its own subdirectory so artefacts stay separable."""
    with (
        patch.object(spectral_denoise, "estimate_snr_margin_db", return_value=8.9),
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._cathar_noiseprint_step", return_value=None),
        patch("modules.cathar._cathar_denoise_step", return_value=tmp_path / "out.wav"),
    ):
        spectral_denoise.apply_when_needed(quiet_wav, tmp_path)
    assert Path(tmp_path / "spectral_denoised").is_dir()


def test_tonal_cleanup_is_off_by_default(quiet_wav, tmp_path):
    """Neither dehum nor dewind earned a default place end to end.

    Dehum is the instructive case: the stage alone removes a median 2.07 dB of hum on the 48
    corpus tapes that carry it, and in the chain that collapses to +0.66 dB on 20 of 48 with
    the speech band moving 1.52 to 2.15 dB. The two are switched separately so either can be
    enabled on its own.
    """
    strategy = {"profile": {"notch_hz": 50.0, "highpass_hz": 75}}
    assert spectral_denoise.APL_ENABLE_TONAL_CLEANUP is False
    assert spectral_denoise.APL_ENABLE_DEHUM is False
    assert spectral_denoise.apply_tonal_cleanup(quiet_wav, tmp_path, strategy=strategy) == quiet_wav


def test_tonal_cleanup_skips_when_no_defect_was_detected(quiet_wav, tmp_path):
    """Both stages cut real content, so neither runs on material without the defect."""
    strategy = {"profile": {"notch_hz": 0.0, "highpass_hz": 45}}
    with patch.object(spectral_denoise, "APL_ENABLE_TONAL_CLEANUP", True):
        assert spectral_denoise.apply_tonal_cleanup(quiet_wav, tmp_path, strategy=strategy) == quiet_wav


def test_tonal_cleanup_can_be_disabled(quiet_wav, tmp_path):
    """The whole stage is switchable without editing the chain."""
    strategy = {"profile": {"notch_hz": 50.0, "highpass_hz": 75}}
    with patch.object(spectral_denoise, "APL_ENABLE_TONAL_CLEANUP", False):
        assert spectral_denoise.apply_tonal_cleanup(quiet_wav, tmp_path, strategy=strategy) == quiet_wav
    assert spectral_denoise.APL_ENABLE_TONAL_CLEANUP is False


def test_a_malformed_strategy_does_not_crash_the_gate():
    """Strategy shapes vary across callers, so the gate reads defensively."""
    assert spectral_denoise._tonal_targets(None) is None
    assert spectral_denoise._profile_value({"profile": {"notch_hz": "bad"}}, "notch_hz") == 0.0


def test_the_deep_model_partners_the_subtraction_stage():
    """Subtraction and the deep separator are chosen together, not independently.

    The light model is selected elsewhere to stay transparent on material that still has
    its dynamics. Once a noise profile has been subtracted, the deep model leaves a
    background 15 dB quieter at equal spectral fidelity.
    """
    assert spectral_denoise.DEEP_DENOISE_MODEL == "UVR-DeNoise.pth"


def _tone_wav(tmp_path, name, hz, seconds=3.0, level=0.05):
    """Writes speech-like noise carrying a steady tone, as tape hum appears."""
    import numpy as np
    import soundfile as sf

    rate = 44100
    rng = np.random.default_rng(11)
    time = np.arange(int(rate * seconds)) / rate
    signal = rng.normal(0.0, 0.02, len(time)).astype(np.float32)
    if hz:
        signal = signal + (level * np.sin(2 * np.pi * hz * time)).astype(np.float32)
    path = tmp_path / name
    sf.write(str(path), signal, rate, subtype="FLOAT")
    return path


def test_mains_detection_finds_a_hum_the_scanner_would_miss(tmp_path):
    """A steady 50 Hz tone is reported, which is what gates the dehum stage."""
    assert spectral_denoise.detect_mains_hz(_tone_wav(tmp_path, "hum.wav", 50.0)) == 50.0


def test_mains_detection_reports_nothing_on_noise(tmp_path):
    """Broadband noise carries no mains tone, so dehum must not be requested for it."""
    assert spectral_denoise.detect_mains_hz(_tone_wav(tmp_path, "noise.wav", 0.0)) == 0.0


def test_mains_detection_skips_audio_too_short_to_scan(tmp_path):
    """A clip shorter than one analysis window cannot be scanned, which is not the same as carrying no hum."""
    assert spectral_denoise.detect_mains_hz(_tone_wav(tmp_path, "tiny.wav", 50.0, seconds=0.2)) is None


def test_an_unscannable_recording_falls_back_to_the_scanned_notch(tmp_path):
    """Unreadable audio keeps the scanner's value; audio read and found free of hum reads 0.0 regardless of it."""
    strategy = {"profile": {"notch_hz": 60.0}}
    assert spectral_denoise._resolve_notch_hz(strategy, tmp_path / "absent.wav") == 60.0
    assert spectral_denoise._resolve_notch_hz(strategy, _tone_wav(tmp_path, "noise.wav", 0.0)) == 0.0


def test_the_recording_decides_the_frequency_over_the_scan(tmp_path):
    """A 50 Hz hum is dehummed at 50 Hz even when the scanner says 60.

    The scanned value used to win. It is the region's nominal frequency when the scanner
    fires at all, and on 10 of the 48 corpus tapes that carry hum that was the wrong one --
    tapes cross regions -- so a dehum at it did nothing.
    """
    strategy = {"profile": {"notch_hz": 60.0}}
    assert spectral_denoise._resolve_notch_hz(strategy, _tone_wav(tmp_path, "h.wav", 50.0)) == 50.0


def test_the_scanned_value_is_used_only_when_the_recording_cannot_be_read():
    """Without audio to measure, the scanner's value is all there is."""
    assert spectral_denoise._resolve_notch_hz({"profile": {"notch_hz": 60.0}}, None) == 60.0


def test_mains_detection_picks_the_frequency_the_harmonics_support(tmp_path):
    """A 60 Hz hum on a tape that would be assumed 50 Hz is still found at 60."""
    assert spectral_denoise.detect_mains_hz(_tone_wav(tmp_path, "h60.wav", 60.0)) == 60.0


def test_the_wider_detection_fills_in_when_the_scan_found_nothing(tmp_path):
    """Where the shared scanner missed the hum, the opted-in mode still cleans it."""
    strategy = {"profile": {"notch_hz": 0.0}}
    assert spectral_denoise._resolve_notch_hz(strategy, _tone_wav(tmp_path, "h2.wav", 50.0)) == 50.0


def test_tonal_cleanup_dehums_when_mains_was_detected(quiet_wav, tmp_path):
    """Detected mains hum is cancelled across harmonics, at the frequency the recording shows."""
    produced = tmp_path / "dehummed.wav"
    strategy = {"profile": {"notch_hz": 50.0, "highpass_hz": 0}}
    with (
        patch.object(spectral_denoise, "APL_ENABLE_DEHUM", True),
        patch.object(spectral_denoise, "detect_mains_hz", return_value=50.0),
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._cathar_dehum_step", return_value=produced) as mock_dehum,
        patch("modules.cathar._cathar_dewind_step") as mock_dewind,
    ):
        assert spectral_denoise.apply_tonal_cleanup(quiet_wav, tmp_path, strategy=strategy) == produced
    assert mock_dehum.call_args.kwargs["freq"] == 50.0
    mock_dewind.assert_not_called()


def test_tonal_cleanup_dewinds_when_rumble_was_detected(quiet_wav, tmp_path):
    """When switched on, a rumble-driven cutoff triggers the rumble stage."""
    produced = tmp_path / "dewinded.wav"
    strategy = {"profile": {"notch_hz": 0.0, "highpass_hz": 75}}
    with (
        patch.object(spectral_denoise, "APL_ENABLE_TONAL_CLEANUP", True),
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._cathar_dewind_step", return_value=produced) as mock_dewind,
    ):
        assert spectral_denoise.apply_tonal_cleanup(quiet_wav, tmp_path, strategy=strategy) == produced
    mock_dewind.assert_called_once()


def test_tonal_cleanup_survives_a_failed_stage(quiet_wav, tmp_path):
    """A failure leaves the audio usable rather than failing the restoration."""
    strategy = {"profile": {"notch_hz": 50.0, "highpass_hz": 0}}
    with (
        patch.object(spectral_denoise, "APL_ENABLE_TONAL_CLEANUP", True),
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._cathar_dehum_step", side_effect=RuntimeError("cathar exploded")),
    ):
        assert spectral_denoise.apply_tonal_cleanup(quiet_wav, tmp_path, strategy=strategy) == quiet_wav


def _material_wav(tmp_path, name, tonal, seconds=3.0):
    """Writes either sustained tones (music-like) or shaped noise (speech-like)."""
    import numpy as np
    import soundfile as sf

    rate = 44100
    time = np.arange(int(rate * seconds)) / rate
    rng = np.random.default_rng(3)
    if tonal:
        signal = sum(0.1 / k * np.sin(2 * np.pi * 220.0 * k * time) for k in range(1, 8))
    else:
        signal = rng.normal(0.0, 0.1, len(time)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * time))
    path = tmp_path / name
    sf.write(str(path), signal.astype(np.float32), rate, subtype="FLOAT")
    return path


def test_sustained_tones_read_as_tonal_and_noise_does_not(tmp_path):
    """Spectral flatness separates held harmonics from broadband material.

    This is the gate for the gentler subtraction factor: on the most tonal third of the
    corpus the speech-tuned factor deviated 0.49 dB against cathar's 0.32, and sustained
    tones are exactly what a noise profile subtracts as if they were noise.
    """
    tonal = spectral_denoise.estimate_tonality(_material_wav(tmp_path, "tones.wav", tonal=True))
    noisy = spectral_denoise.estimate_tonality(_material_wav(tmp_path, "noise.wav", tonal=False))
    assert tonal < spectral_denoise.APL_TONAL_FLATNESS_MAX < noisy


def test_tonal_material_gets_the_gentler_factor(tmp_path):
    """Music-like audio is subtracted at the tonal factor, everything else at the default."""
    assert spectral_denoise._alpha_for(_material_wav(tmp_path, "t.wav", tonal=True)) == spectral_denoise.APL_SPECTRAL_ALPHA_TONAL
    assert spectral_denoise._alpha_for(_material_wav(tmp_path, "n.wav", tonal=False)) == spectral_denoise.APL_SPECTRAL_ALPHA


def test_unreadable_audio_keeps_the_default_factor(quiet_wav):
    """A recording that cannot be measured is not assumed tonal."""
    assert spectral_denoise.estimate_tonality(quiet_wav) is None
    assert spectral_denoise._alpha_for(quiet_wav) == spectral_denoise.APL_SPECTRAL_ALPHA


def test_the_tonal_factor_is_gentler_than_the_default():
    """The whole point of the gate is to subtract less on material that cannot take more."""
    assert spectral_denoise.APL_SPECTRAL_ALPHA_TONAL < spectral_denoise.APL_SPECTRAL_ALPHA
