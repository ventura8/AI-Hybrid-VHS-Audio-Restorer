"""Unit tests for the final mastering stage: two-pass loudness and stem bit depth."""

from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

import modules.processing

LOUDNORM_ANALYSIS_JSON = (
    '{\n"input_i" : "-37.20",\n"input_tp" : "-19.30",\n"input_lra" : "7.70",\n'
    '"input_thresh" : "-47.40",\n"output_i" : "-16.02",\n"target_offset" : "-0.20"\n}'
)


def test_parse_loudnorm_json_extracts_measurements():
    """The analysis block is parsed out of ffmpeg's surrounding log noise."""
    stderr = f"[Parsed_loudnorm_0 @ 0x1] \n{LOUDNORM_ANALYSIS_JSON}\n"
    measurements = modules.processing._parse_loudnorm_json(stderr)
    assert measurements["input_i"] == "-37.20"
    assert measurements["target_offset"] == "-0.20"


def test_parse_loudnorm_json_with_trailing_stderr():
    """The analysis block is parsed even if trailing content follows the JSON braces."""
    stderr = f"[Parsed_loudnorm_0 @ 0x1] \n{LOUDNORM_ANALYSIS_JSON}\n[out#0/wav @ 0x2] video:0kB audio:100kB other:0kB"
    measurements = modules.processing._parse_loudnorm_json(stderr)
    assert measurements is not None
    assert measurements["input_i"] == "-37.20"
    assert measurements["target_offset"] == "-0.20"


@pytest.mark.parametrize(
    "stderr",
    [
        "no json at all",
        "}{",
        '{"input_i": "-37.2"}',  # incomplete: missing the other measured values
        "{not valid json}",
        (
            '{\n"input_i" : "-inf",\n"input_tp" : "-19.30",\n"input_lra" : "7.70",\n'
            '"input_thresh" : "-47.40",\n"output_i" : "-16.02",\n"target_offset" : "inf"\n}'
        ),
    ],
)
def test_parse_loudnorm_json_rejects_unusable_output(stderr):
    """Anything short of a complete measurement block must fall back to single-pass."""
    assert modules.processing._parse_loudnorm_json(stderr) is None


@pytest.mark.parametrize(
    "fragment",
    [
        "measured_I=-37.20",
        "measured_TP=-19.30",
        "measured_LRA=7.70",
        "measured_thresh=-47.40",
        "offset=-0.20",
        "linear=true",
    ],
)
def test_measured_loudnorm_args_carry_every_value(fragment):
    """Measured values must reach the applied pass, or it cannot hit the target."""
    measurements = modules.processing._parse_loudnorm_json(LOUDNORM_ANALYSIS_JSON)
    assert fragment in modules.processing._measured_loudnorm_args(measurements)


def test_two_pass_expression_applies_measured_args_and_limiter():
    """The applied mix carries the measured pass and ends at the true-peak limiter."""
    measurements = modules.processing._parse_loudnorm_json(LOUDNORM_ANALYSIS_JSON)
    args = modules.processing._measured_loudnorm_args(measurements)

    with patch("modules.processing.ENABLE_LOUDNORM", True):
        expr = modules.processing._build_mix_filter_expression(None, None, args)
    assert f"loudnorm={args}" in expr
    assert expr.endswith(f"{modules.processing.LOUDNORM_TRUE_PEAK_LIMITER}[mixed]")


@pytest.mark.parametrize(
    "resolver",
    [
        lambda: modules.processing._resolve_loudnorm_args("v", "a", "b", None, None),
        lambda: modules.processing._resolve_single_track_loudnorm_args("v", "a"),
    ],
)
def test_resolve_loudnorm_args_skips_measurement_when_disabled(resolver):
    """No loudness stage means no analysis pass should be run at all."""
    with patch("modules.processing.ENABLE_LOUDNORM", False):
        with patch("modules.processing._run_loudness_analysis") as mock_measure:
            assert resolver() is None
            mock_measure.assert_not_called()


@pytest.mark.parametrize(
    "resolver",
    [
        lambda: modules.processing._resolve_loudnorm_args("v", "a", "b", None, None),
        lambda: modules.processing._resolve_single_track_loudnorm_args("v", "a"),
    ],
)
def test_resolve_loudnorm_args_falls_back_when_measurement_fails(resolver):
    """A failed analysis pass must degrade to single-pass, not break the render."""
    with patch("modules.processing.ENABLE_LOUDNORM", True):
        with patch("modules.processing._run_loudness_analysis", return_value=None):
            assert resolver() is None


def test_loudness_analysis_survives_ffmpeg_failure():
    """Analysis is best-effort; a crashed probe must not abort the restoration."""
    with patch("modules.processing.subprocess.run", side_effect=OSError("ffmpeg missing")):
        assert modules.processing._run_loudness_analysis(("v", "a"), "[1:a]anull[mastered]") is None


def test_single_track_modes_get_the_same_mastering_chain():
    """denoise_only and the native modes must not skip loudness and limiting."""
    with patch("modules.processing.ENABLE_LOUDNORM", True):
        expr = modules.processing._build_single_audio_filter_expression()
    assert expr.startswith("[1:a]loudnorm=")
    assert f"aresample={modules.processing.PIPELINE_SAMPLE_RATE}" in expr
    assert expr.endswith(f"{modules.processing.LOUDNORM_TRUE_PEAK_LIMITER}[mastered]")


def test_single_track_mux_routes_through_the_mastering_graph():
    """The mastered label must be mapped, not the raw input stream."""
    expr = "[1:a]anull[mastered]"
    cmd = modules.processing._build_single_audio_mux_command("v.mp4", "a.wav", "o.mp4", ["-c:a", "aac"], 4, expr)
    assert "-filter_complex" in cmd and expr in cmd
    assert "[mastered]" in cmd
    assert "1:a:0" not in cmd


def test_single_track_mux_maps_directly_when_loudness_is_off():
    """With no mastering graph the processed track is mapped as-is."""
    cmd = modules.processing._build_single_audio_mux_command("v.mp4", "a.wav", "o.mp4", ["-c:a", "aac"], 4, None)
    assert "1:a:0" in cmd
    assert "-filter_complex" not in cmd


def test_ensure_float_pcm_upgrades_fixed_point_stems(tmp_path):
    """Separators can emit PCM_16; the chain's contract is 32-bit float throughout."""
    samples = (0.3 * np.sin(2.0 * np.pi * 440.0 * np.arange(4410) / 44100.0)).astype("float32")
    fixed = tmp_path / "stem.wav"
    sf.write(fixed, samples, 44100, subtype="PCM_16")

    assert modules.processing._ensure_float_pcm(fixed) == fixed
    assert sf.info(fixed).subtype == "FLOAT"
    restored, _ = sf.read(fixed, dtype="float32")
    assert np.max(np.abs(restored - samples)) < 1e-4


def test_ensure_float_pcm_leaves_float_and_unreadable_files_alone(tmp_path):
    """Float stems are untouched, and an unreadable file must not raise."""
    already = tmp_path / "float.wav"
    sf.write(already, np.zeros(4410, dtype="float32"), 44100, subtype="FLOAT")
    modules.processing._ensure_float_pcm(already)
    assert sf.info(already).subtype == "FLOAT"

    broken = tmp_path / "broken.wav"
    broken.write_text("not audio", encoding="utf-8")
    assert modules.processing._ensure_float_pcm(broken) == broken


@pytest.mark.parametrize(
    ("foreground", "background"),
    [
        # BS-Roformer, the configured default.
        ("clip_(Vocals)_bs.wav", "clip_(Instrumental)_bs.wav"),
        # Modern MelBand Roformers use lowercase labels.
        ("clip_(vocals)_kim.wav", "clip_(other)_kim.wav"),
        # The crowd model the auto-scanner selects for ambient-heavy scenes.
        ("clip_(crowd)_aufr33.wav", "clip_(other)_aufr33.wav"),
        # Older conventions that must keep working.
        ("clip_(Vocals)_x.wav", "clip_(Background)_x.wav"),
        ("clip_(Vocals)_y.wav", "clip_(No Vocals)_y.wav"),
    ],
)
def test_stem_detection_handles_every_model_naming(tmp_path, foreground, background):
    """Each separator labels stems differently; all must resolve to a usable pair."""
    for name in (foreground, background):
        sf.write(tmp_path / name, np.zeros(1024, dtype="float32"), 44100, subtype="FLOAT")

    vocals, backing, all_wavs = modules.processing._collect_stem_candidates(tmp_path)
    assert [p.name for p in vocals] == [foreground]
    assert [p.name for p in backing] == [background]
    assert len(all_wavs) == 2


def test_no_vocals_label_is_not_mistaken_for_the_speech_stem(tmp_path):
    """'(No Vocals)' is a background label and must never match the foreground."""
    sf.write(tmp_path / "clip_(No Vocals)_x.wav", np.zeros(1024, dtype="float32"), 44100, subtype="FLOAT")
    vocals, backing, _ = modules.processing._collect_stem_candidates(tmp_path)
    assert vocals == []
    assert len(backing) == 1
