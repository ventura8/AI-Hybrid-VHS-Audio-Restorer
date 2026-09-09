"""Opt-in short-fixture hardware smoke checks."""

import pytest

from scripts.audio_matrix.cli import select_languages
from scripts.audio_matrix.manifest import load_languages
from scripts.run_hardware_validation import DEFAULT_MODES, build_dry_run_report, require_nvidia_cuda

EXPECTED_MODES = {
    "auto",
    "multipass_auto",
    "auto_pure",
    "auto_pure_linear",
    "cathar",
    "hybrid",
    "denoise_only",
    "auto_ffmpeg_native",
    "vhs_native",
    "arnndn_speech",
}
EXPECTED_LANGUAGES = {
    "ar",
    "bg",
    "bn",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fr",
    "he",
    "hi",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "ka",
    "kk",
    "ko",
    "lb",
    "lv",
    "ml",
    "mr",
    "ne",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
    "sw",
    "te",
    "tr",
    "uk",
    "ur",
    "vi",
    "zh",
}


def test_short_matrix_covers_all_restoration_modes(tmp_path):
    """The smoke plan must submit every canonical mode."""
    fixture = tmp_path / "short_vhs.wav"
    fixture.touch()
    report = build_dry_run_report([fixture], DEFAULT_MODES)
    assert set(report["modes"]) == EXPECTED_MODES
    assert report["fixtures"] == [str(fixture)]


def test_catalog_includes_every_piper_language_in_the_reference_matrix():
    """Multilingual smoke coverage must not silently regress to English only."""
    langs = load_languages()
    assert set(langs) == EXPECTED_LANGUAGES


def test_language_selection_rejects_unknown_codes():
    """An invalid language selector must not fail later with a bare key error."""
    with pytest.raises(ValueError, match="Unknown language code"):
        select_languages({"en": object()}, ["en_US"])


def test_nvidia_guard_accepts_nvidia_cuda(monkeypatch):
    """Physical execution remains explicitly NVIDIA-only."""
    settings = {"is_nvidia": True, "cpu_only_fallback": False}
    monkeypatch.setattr("modules.hardware.get_optimal_settings", lambda: settings)
    assert require_nvidia_cuda() == settings

    monkeypatch.setattr("modules.hardware.get_optimal_settings", lambda: {"is_nvidia": False, "cpu_only_fallback": False})
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        require_nvidia_cuda()

    monkeypatch.setattr("modules.hardware.get_optimal_settings", lambda: {"is_nvidia": True, "cpu_only_fallback": True})
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        require_nvidia_cuda()
