"""The engine decision of the auto scanner: classes, the tonal preference, the environment gates, the log."""

from unittest.mock import patch

import numpy as np
import pytest

import modules.auto_scanner


@pytest.mark.parametrize(
    "speech,music,ambient,nf,expected_material",
    [
        (0.5, 0.4, 0.2, -45.0, "dialogue"),
        (0.6, 0.05, 0.05, -60.0, "dialogue"),
        (0.6, 0.05, 0.05, -35.0, "dialogue"),
        (0.05, 0.5, 0.1, -45.0, "music or ambience"),
        (0.05, 0.05, 0.05, -30.0, "tape noise"),
    ],
)
def test_select_strategy_mode_runs_the_measured_engine_on_every_class(speech, music, ambient, nf, expected_material):
    """Every acoustic class runs auto_pure_linear, and the reason names the class and the evidence."""
    with patch("modules.auto_scanner._neural_denoiser_available", return_value=True):
        mode, reason = modules.auto_scanner._select_strategy_mode(speech, music, ambient, nf)
    assert mode == "auto_pure_linear"
    assert modules.auto_scanner.ENGINE_EVIDENCE[expected_material] in reason
    assert "cathar" in reason


@pytest.mark.parametrize(
    ("speech", "music", "ambient", "has_dialogue", "periodicity", "expected_material"),
    [
        (1.0, 0.02, 0.04, True, 0.6, "rhythmic music"),
        (0.5, 0.4, 0.2, False, 0.1, "dialogue"),
        (0.1, 0.02, 0.02, True, 0.1, "dialogue"),
        (0.05, 0.5, 0.1, False, 0.1, "music or ambience"),
        (0.05, 0.05, 0.05, False, 0.1, "tape noise"),
    ],
)
def test_material_classes_keep_their_gates(speech, music, ambient, has_dialogue, periodicity, expected_material):
    """The class names follow the same thresholds the engine choice used to dispatch on."""
    material, _description = modules.auto_scanner._classify_material(speech, music, ambient, has_dialogue, periodicity)
    assert material == expected_material


@pytest.mark.parametrize(
    ("tonality", "similarity", "periodicity", "expected_mode"),
    [
        # Tonal, the quietest 4 s shaped like the programme, no beat: cathar's fidelity.
        (0.0281, 0.93, 0.10, "cathar"),
        (0.0399, 0.90, 0.29, "cathar"),
        # Any one condition missing keeps auto_pure_linear.
        (0.04, 0.93, 0.10, "auto_pure_linear"),
        (0.0281, 0.89, 0.10, "auto_pure_linear"),
        (0.0281, 0.93, 0.30, "auto_pure_linear"),
        (None, 0.93, 0.10, "auto_pure_linear"),
        (0.0281, None, 0.10, "auto_pure_linear"),
    ],
)
def test_sustained_tonal_programme_without_silence_runs_cathar(tonality, similarity, periodicity, expected_mode):
    """The one condition both corpora agree cathar deviates less under hands the tape to cathar."""
    with (
        patch("modules.auto_scanner._neural_denoiser_available", return_value=True),
        patch("modules.auto_scanner._cathar_available", return_value=True),
    ):
        mode, reason = modules.auto_scanner._select_strategy_mode(1.0, 0.02, 0.04, -45.0, True, periodicity, tonality, similarity)
    assert mode == expected_mode
    if expected_mode == "cathar":
        assert f"probe similarity {similarity:.2f}" in reason
        assert modules.auto_scanner.TONAL_PROBE_EVIDENCE in reason


def test_tonal_preference_needs_cathar_and_can_be_switched_off():
    """Without the binary the tape stays on auto_pure_linear; auto_cathar_tonal false disables the preference."""
    with (
        patch("modules.auto_scanner._neural_denoiser_available", return_value=True),
        patch("modules.auto_scanner._cathar_available", return_value=False),
    ):
        assert modules.auto_scanner._select_strategy_mode(1.0, 0.02, 0.04, -45.0, True, 0.1, 0.0281, 0.93)[0] == "auto_pure_linear"
    with patch("modules.auto_scanner.AUTO_CATHAR_TONAL", False):
        assert modules.auto_scanner._probe_carries_programme(0.0281, 0.93, 0.1) is False


def test_strategy_passes_tonality_and_probe_similarity_into_mode_selection():
    """Both profile readings reach the decision, and the log names what the probe would learn."""
    profile = {
        "speech_ratio": 1.0,
        "music_ratio": 0.02,
        "ambient_ratio": 0.04,
        "onset_periodicity": 0.1,
        "tonality": 0.0281,
        "probe_similarity": 0.93,
    }
    with (
        patch("modules.auto_scanner._neural_denoiser_available", return_value=True),
        patch("modules.auto_scanner._cathar_available", return_value=True),
    ):
        assert modules.auto_scanner.evaluate_restoration_strategy(profile)["mode"] == "cathar"
    assert modules.auto_scanner._describe_probe(0.93) == "quietest 4 s carries the programme (similarity 0.93)"
    assert modules.auto_scanner._describe_probe(0.61) == "quietest 4 s is noise (similarity 0.61)"
    assert modules.auto_scanner._describe_probe(None) == "not measured"


def _sustained_programme(sr, envelope, seconds=12):
    """Forty partials across the speech band under the given envelope, over a little noise."""
    rng = np.random.default_rng(3)
    t = np.arange(sr * seconds, dtype=np.float64) / sr
    amps, freqs = rng.uniform(0.05, 0.3, 40), rng.uniform(150.0, 3300.0, 40)
    programme = sum(a * np.sin(2.0 * np.pi * f * t) for f, a in zip(freqs, amps)) / 4.0
    return (programme * envelope(t) + 0.01 * rng.normal(0, 1, len(t))).astype(np.float32)


def test_probe_similarity_separates_silence_from_sustained_programme():
    """A pause makes the probe noise (low similarity); a held tone makes it programme (high)."""
    sr = 44100
    paused = modules.auto_scanner._probe_programme_similarity(_sustained_programme(sr, lambda t: (t < 7.0).astype(float)), sr)
    held = modules.auto_scanner._probe_programme_similarity(
        _sustained_programme(sr, lambda t: 0.6 + 0.4 * np.sin(2.0 * np.pi * 0.2 * t)), sr
    )
    assert paused is not None and held is not None
    assert held >= modules.auto_scanner.AUTO_CATHAR_PROBE_SIMILARITY > paused
    assert modules.auto_scanner._probe_programme_similarity(np.zeros(4096, dtype=np.float32), sr) is None


def test_cathar_runs_when_the_neural_denoiser_is_missing():
    """Without audio-separator the deterministic engine is the one full chain that can run."""
    with (
        patch("modules.auto_scanner._neural_denoiser_available", return_value=False),
        patch("modules.auto_scanner._cathar_available", return_value=True),
    ):
        mode, reason = modules.auto_scanner._select_strategy_mode(0.6, 0.05, 0.05, -35.0)
    assert mode == "cathar"
    assert "neural denoiser is not installed" in reason


def test_native_dsp_is_the_last_resort():
    """With neither engine installed the FFmpeg chain still produces an output."""
    with (
        patch("modules.auto_scanner._neural_denoiser_available", return_value=False),
        patch("modules.auto_scanner._cathar_available", return_value=False),
    ):
        mode, reason = modules.auto_scanner._select_strategy_mode(0.6, 0.05, 0.05, -35.0)
    assert mode == "auto_ffmpeg_native"
    assert "neither" in reason


def test_neural_denoiser_availability_reads_the_installed_package():
    """The probe asks importlib for the package rather than importing torch."""
    with patch("modules.auto_scanner.importlib.util.find_spec", return_value=None):
        assert modules.auto_scanner._neural_denoiser_available() is False
    with patch("modules.auto_scanner.importlib.util.find_spec", return_value=object()):
        assert modules.auto_scanner._neural_denoiser_available() is True


def test_cathar_availability_resolves_the_binary(tmp_path):
    """An empty, missing or directory path reads as unavailable; an executable file as available."""
    with patch("modules.auto_scanner.CATHAR_BIN", ""):
        assert modules.auto_scanner._cathar_available() is False
    present = tmp_path / "cathar.exe"
    present.write_bytes(b"")
    present.chmod(0o755)
    with patch("modules.auto_scanner.shutil.which", return_value=None):
        with patch("modules.auto_scanner.CATHAR_BIN", str(tmp_path / "missing-cathar")):
            assert modules.auto_scanner._cathar_available() is False
        with patch("modules.auto_scanner.CATHAR_BIN", str(tmp_path)):
            assert modules.auto_scanner._cathar_available() is False
        with patch("modules.auto_scanner.CATHAR_BIN", str(present)):
            assert modules.auto_scanner._cathar_available() is True


def test_cathar_availability_accepts_a_name_on_path(tmp_path):
    """A bare name that `shutil.which` resolves is installed, whatever the working directory holds."""
    with (
        patch("modules.auto_scanner.shutil.which", return_value=str(tmp_path / "cathar")),
        patch("modules.auto_scanner.CATHAR_BIN", "cathar"),
    ):
        assert modules.auto_scanner._cathar_available() is True


def test_cathar_availability_requires_execute_permission_off_windows(tmp_path):
    """Off Windows a direct path that is a plain file without the execute bit is not installed."""
    plain = tmp_path / "cathar"
    plain.write_bytes(b"")
    with (
        patch("modules.auto_scanner.shutil.which", return_value=None),
        patch("modules.auto_scanner.CATHAR_BIN", str(plain)),
        patch("modules.auto_scanner.os.name", "posix"),
        patch("modules.auto_scanner.os.access", return_value=False),
    ):
        assert modules.auto_scanner._cathar_available() is False


def _logged_decision(strategy, executed_mode=None, cathar_installed=False):
    """Captures the analysis block the scanner logs for a strategy."""
    lines = []
    with (
        patch("modules.auto_scanner.log_msg", side_effect=lambda msg, **_kw: lines.append(msg)),
        patch("modules.auto_scanner._neural_denoiser_available", return_value=True),
        patch("modules.auto_scanner._cathar_available", return_value=cathar_installed),
    ):
        modules.auto_scanner._log_strategy_decision(strategy, executed_mode=executed_mode)
    return "\n".join(lines)


def _logged_strategy(profile, reason):
    """A minimal strategy dict for the log block."""
    return {"mode": "auto_pure_linear", "reason": reason, "enhance_nfe": 64, "sync_method": "shift", "profile": profile}


@pytest.mark.parametrize(
    "expected_line",
    [
        "onset periodicity 0.120 (no sustained beat)",
        "spectral flatness 0.0281 (tonal programme)",
        "Noise Probe     : quietest 4 s carries the programme (similarity 0.93)",
        "Mains Hum       : 50.00 Hz notch",
        "Best-fit Mode: 'auto_pure_linear' (advisory; running 'cathar')",
        "Rationale       : Dialogue",
        "cathar (deterministic DSP; sustained tonal programme with no silence, or no neural denoiser; not installed)",
    ],
)
def test_profile_log_reports_the_engine_evidence(expected_line):
    """The analysis block names rhythm, tonality, hum and both engines beside the pick."""
    profile = {"onset_periodicity": 0.12, "tonality": 0.0281, "probe_similarity": 0.93, "notch_hz": 50.0, "noise_floor_db": -40.0}
    text = _logged_decision(_logged_strategy(profile, "Dialogue"), executed_mode="cathar")
    assert expected_line in text
    assert "Verdict" not in text


@pytest.mark.parametrize(
    "expected_line",
    [
        "(sustained beat)",
        "Tonality        : not measured",
        "Noise Probe     : not measured",
        "Mains Hum       : none detected",
        "Rationale       : Sustained rhythmic music",
        "Verdict         : auto_pure_linear leads cathar",
        "cathar (deterministic DSP; sustained tonal programme with no silence, or no neural denoiser; installed)",
    ],
)
def test_profile_log_reads_unmeasured_values_and_the_verdict(expected_line):
    """Missing measurements read as such, and a reason with a verdict prints it on its own line."""
    profile = {"onset_periodicity": 0.7, "tonality": None, "notch_hz": 0.0}
    reason = "Sustained rhythmic music; auto_pure_linear leads cathar"
    text = _logged_decision(_logged_strategy(profile, reason), cathar_installed=True)
    assert expected_line in text
