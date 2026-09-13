"""The Resemble denoise-only stage: staged, invoked, rejoined, and never a reason to fail the restoration."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import modules.processing as processing
from modules import resemble_denoise


def _wav(path, seconds=1.0):
    sf.write(str(path), np.zeros((int(44100 * seconds), 2), dtype=np.float32), 44100, subtype="FLOAT")
    return path


def _fake_cli(target_name):
    """A stand-in for the package's CLI: writes one output per staged input."""

    def run(command, **_kwargs):
        input_dir, output_dir = Path(command[1]), Path(command[2])
        assert "--denoise_only" in command
        for staged in input_dir.glob("*.wav"):
            _wav(output_dir / staged.name)
        return target_name

    return run


def test_the_denoiser_is_invoked_denoise_only_and_its_output_returned(tmp_path):
    source = _wav(tmp_path / "in.wav")
    with patch("modules.resemble_denoise.run_command_with_progress", side_effect=_fake_cli("ok")) as cli:
        produced = resemble_denoise.denoise(source, tmp_path / "work")
    assert produced == tmp_path / "work" / "resemble_in.wav" and produced.is_file()
    command = cli.call_args.args[0]
    assert Path(command[0]).stem == "resemble-enhance" and command[1:3] == [
        str(tmp_path / "work" / "input"),
        str(tmp_path / "work" / "output"),
    ]
    assert not (tmp_path / "work" / "input").exists()


def test_no_output_means_no_result(tmp_path):
    """A run that produced nothing reads as the stage being absent, not as its input."""
    source = _wav(tmp_path / "in.wav")
    with patch("modules.resemble_denoise.run_command_with_progress", return_value=None):
        assert resemble_denoise.denoise(source, tmp_path / "work") is None


def test_the_stage_falls_back_when_off_or_failing(tmp_path):
    source = _wav(tmp_path / "in.wav")
    fallback = tmp_path / "uvr.wav"
    assert resemble_denoise.denoise_or(source, tmp_path / "work", False, lambda: fallback) == fallback
    with patch("modules.resemble_denoise.denoise", side_effect=RuntimeError("no gpu")), patch("modules.resemble_denoise.log_msg") as log:
        assert resemble_denoise.denoise_or(source, tmp_path / "work", True, lambda: fallback) == fallback
    assert "falling back" in log.call_args[0][0]


def test_the_stage_falls_back_on_no_output_and_returns_its_own_otherwise(tmp_path):
    source = _wav(tmp_path / "in.wav")
    fallback = tmp_path / "uvr.wav"
    with patch("modules.resemble_denoise.denoise", return_value=None):
        assert resemble_denoise.denoise_or(source, tmp_path / "work", True, lambda: fallback) == fallback
    with patch("modules.resemble_denoise.denoise", return_value=tmp_path / "res.wav"), patch("modules.resemble_denoise.log_msg"):
        assert resemble_denoise.denoise_or(source, tmp_path / "work", True, lambda: fallback) == tmp_path / "res.wav"


def test_a_configured_model_name_overrides_the_chains_choice(tmp_path):
    """apl_neural_model names the UVR model outright, whatever the adaptive pick and the subtraction upgrade say."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    models = []
    with (
        patch("modules.processing.APL_NEURAL_MODEL", "some_other_model.ckpt"),
        patch("modules.processing._pre_denoise_surgical_step", return_value=tmp_path / "surg.wav"),
        patch("modules.apl_chain._spectral_denoise.apply_tonal_cleanup", side_effect=lambda wav, *_a, **_k: wav),
        patch("modules.apl_chain._spectral_denoise.apply_when_needed", return_value=tmp_path / "subtracted.wav"),
        patch("modules.processing._denoise_full_audio_step", side_effect=lambda wav, _d, **kw: models.append(kw["denoise_model"]) or wav),
        patch("modules.processing._post_denoise_cleanup_step", side_effect=lambda wav, *_a, **_k: wav),
        patch("modules.processing._polish_full_audio_step", side_effect=lambda wav, *_a, **_k: wav),
        patch("modules.apl_chain.log_msg"),
    ):
        processing._denoise_and_polish_full_audio_step(
            tmp_path / "orig.wav", out_dir, denoise_model="UVR-DeNoise-Lite.pth", spectral_denoise=True
        )
    assert models == ["some_other_model.ckpt"]


def test_the_resemble_stage_sits_between_deepfilternet_and_uvr(tmp_path):
    """Asked for, it runs ahead of the UVR fallback; its own failure reaches UVR."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    seen = []
    with (
        patch("modules.processing._pre_denoise_surgical_step", return_value=tmp_path / "surg.wav"),
        patch("modules.resemble_denoise.denoise", side_effect=lambda wav, d, *_a, **_k: seen.append("resemble") or tmp_path / "res.wav"),
        patch("modules.resemble_denoise.log_msg"),
        patch("modules.processing._denoise_full_audio_step", side_effect=lambda wav, _d, **_k: seen.append("uvr") or wav),
        patch("modules.processing._post_denoise_cleanup_step", side_effect=lambda wav, *_a, **_k: wav),
        patch("modules.processing._polish_full_audio_step", side_effect=lambda wav, *_a, **_k: wav),
        patch("modules.apl_chain.log_msg"),
    ):
        result = processing._denoise_and_polish_full_audio_step(tmp_path / "orig.wav", out_dir, resemble_denoise=True)
    assert result == tmp_path / "res.wav" and seen == ["resemble"]


def test_a_valid_output_is_reused_and_an_invalid_one_replaced(tmp_path):
    """Resumed runs skip the inference when its output stands; a truncated one is redone."""
    source = _wav(tmp_path / "in.wav")
    work = tmp_path / "work"
    work.mkdir()
    existing = _wav(work / "resemble_in.wav")
    with patch("modules.resemble_denoise.run_command_with_progress", side_effect=_fake_cli("ok")) as cli:
        assert resemble_denoise.denoise(source, work) == existing
    cli.assert_not_called()
    existing.write_bytes(b"")
    with patch("modules.resemble_denoise.run_command_with_progress", side_effect=_fake_cli("ok")) as cli:
        produced = resemble_denoise.denoise(source, work)
    cli.assert_called_once()
    assert produced == existing and produced.stat().st_size > 0
