from unittest.mock import MagicMock, patch

import modules.cathar as cathar
import modules.config as cfg


def test_cathar_config_coercion():
    """Verifies that Cathar numeric and boolean settings coerce cleanly."""
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = {
        "process_mode": "cathar",
        "cathar_denoise_method": "wiener",
        "cathar_azimuth_method": "gcc-phat",
        "cathar_enhance_method": "replicate",
        "cathar_alpha": "4.5",
        "cathar_beta": "0.02",
        "cathar_dewind_cutoff": "70",
        "cathar_enable_dewind": "true",
        "cathar_enable_coherent": "false",
        "cathar_enable_inpaint": "true",
        "cathar_inpaint_max_gap_ms": "60",
        "cathar_inpaint_iterations": "4",
        "cathar_enable_enhance": "true",
        "cathar_enable_noiseprint": "true",
        "cathar_noiseprint_duration_s": "0.85",
        "cathar_dehum_adaptive": "true",
        "cathar_dehum_harmonics": "6",
        "cathar_enable_mono_below": "true",
        "cathar_mono_below_hz": "110",
        "cathar_enable_deplosive": "true",
        "cathar_deplosive_strength": "5",
        "cathar_enable_deesser": "true",
        "cathar_deesser_bands": "4",
        "cathar_deesser_freq": "4500",
        "cathar_deesser_threshold": "-22",
        "cathar_enable_dereverb": "true",
        "cathar_dereverb_wpe": "true",
        "cathar_dereverb_strength": "2.5",
        "enable_linear_air": "true",
        "linear_air_gain_db": "2.5",
    }
    with patch.object(cfg, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        conf, _ = cfg.load_config()
    expected = {
        "process_mode": "cathar",
        "cathar_denoise_method": "wiener",
        "cathar_azimuth_method": "gcc-phat",
        "cathar_enhance_method": "replicate",
        "cathar_alpha": 4.5,
        "cathar_beta": 0.02,
        "cathar_dewind_cutoff": 70,
        "cathar_enable_dewind": True,
        "cathar_enable_coherent": False,
        "cathar_enable_inpaint": True,
        "cathar_inpaint_max_gap_ms": 60,
        "cathar_inpaint_iterations": 4,
        "cathar_enable_enhance": True,
        "cathar_enable_noiseprint": True,
        "cathar_noiseprint_duration_s": 0.85,
        "cathar_dehum_adaptive": True,
        "cathar_dehum_harmonics": 6,
        "cathar_enable_mono_below": True,
        "cathar_mono_below_hz": 110,
        "cathar_enable_deplosive": True,
        "cathar_deplosive_strength": 5,
        "cathar_enable_deesser": True,
        "cathar_deesser_bands": 4,
        "cathar_deesser_freq": 4500,
        "cathar_deesser_threshold": -22.0,
        "cathar_enable_dereverb": True,
        "cathar_dereverb_wpe": True,
        "cathar_dereverb_strength": 2.5,
        "enable_linear_air": True,
        "linear_air_gain_db": 2.5,
    }
    subset = {k: conf[k] for k in expected}
    assert subset == expected


def test_cathar_precondition_pass_enabled(tmp_path):
    """Precondition pass executes all 7 stages when enabled."""
    in_wav = tmp_path / "in.wav"
    with (
        patch("modules.cathar._run_cathar_step", side_effect=lambda cmd, i, o, *a, **k: o),
        patch("modules.cathar._is_stereo_audio", return_value=True),
        patch("modules.cathar.CATHAR_ENABLE_DEWIND", True),
        patch("modules.cathar.CATHAR_ENABLE_AZIMUTH", True),
        patch("modules.cathar.CATHAR_ENABLE_MONO_BELOW", True),
        patch("modules.cathar.CATHAR_ENABLE_DECLICK", True),
        patch("modules.cathar.CATHAR_ENABLE_DECRACKLE", True),
        patch("modules.cathar.CATHAR_ENABLE_INPAINT", True),
        patch("modules.cathar.CATHAR_ENABLE_DEPLOSIVE", True),
    ):
        res = cathar._cathar_precondition_pass(in_wav, tmp_path)
    assert "deplosived_inpainted_decrackled_declicked_monobelow_azimuth_dewinded_in.wav" in res.name


def test_cathar_precondition_pass_disabled(tmp_path):
    """Precondition pass bypasses all stages when disabled."""
    in_wav = tmp_path / "in.wav"
    with (
        patch("modules.cathar.CATHAR_ENABLE_DEWIND", False),
        patch("modules.cathar.CATHAR_ENABLE_AZIMUTH", False),
        patch("modules.cathar.CATHAR_ENABLE_MONO_BELOW", False),
        patch("modules.cathar.CATHAR_ENABLE_DECLICK", False),
        patch("modules.cathar.CATHAR_ENABLE_DECRACKLE", False),
        patch("modules.cathar.CATHAR_ENABLE_INPAINT", False),
        patch("modules.cathar.CATHAR_ENABLE_DEPLOSIVE", False),
    ):
        res = cathar._cathar_precondition_pass(in_wav, tmp_path)
    assert res == in_wav


def test_cathar_repair_pass_enabled(tmp_path):
    """Repair pass executes declip, dehum, repair, dewow, and dereverb when enabled."""
    in_wav = tmp_path / "in.wav"
    with (
        patch("modules.cathar._run_cathar_step", side_effect=lambda cmd, i, o, *a, **k: o),
        patch("modules.cathar.CATHAR_ENABLE_DECLIP", True),
        patch("modules.cathar.CATHAR_ENABLE_DEHUM", True),
        patch("modules.cathar.CATHAR_ENABLE_REPAIR", True),
        patch("modules.cathar.CATHAR_ENABLE_DEWOW", True),
        patch("modules.cathar.CATHAR_ENABLE_DEREVERB", True),
        patch("modules.cathar.CATHAR_BIN", "python"),
        patch("modules.cathar.shutil.which", return_value="python"),
    ):
        res = cathar._cathar_repair_pass(in_wav, tmp_path, notch_freq=50.0)
    assert "dereverbed_dewowed_repaired_dehummed_declipped_in.wav" in res.name


def test_cathar_repair_pass_disabled(tmp_path):
    """Repair pass bypasses all stages when disabled."""
    in_wav = tmp_path / "in.wav"
    with (
        patch("modules.cathar.CATHAR_ENABLE_DECLIP", False),
        patch("modules.cathar.CATHAR_ENABLE_DEHUM", False),
        patch("modules.cathar.CATHAR_ENABLE_REPAIR", False),
        patch("modules.cathar.CATHAR_ENABLE_DEWOW", False),
        patch("modules.cathar.CATHAR_ENABLE_DEREVERB", False),
    ):
        res = cathar._cathar_repair_pass(in_wav, tmp_path, notch_freq=50.0)
    assert res == in_wav


def test_cathar_polish_pass(tmp_path):
    """Polish pass applies de-essing and enhancement when enabled."""
    in_wav = tmp_path / "in.wav"
    with (
        patch("modules.cathar._run_cathar_step", side_effect=lambda cmd, i, o, *a, **k: o),
        patch("modules.cathar.CATHAR_ENABLE_DEESSER", True),
        patch("modules.cathar.CATHAR_ENABLE_ENHANCE", True),
    ):
        res = cathar._cathar_polish_pass(in_wav, tmp_path)
    assert "enhanced_deessed_in.wav" in res.name

    with (
        patch("modules.cathar.CATHAR_ENABLE_DEESSER", False),
        patch("modules.cathar.CATHAR_ENABLE_ENHANCE", False),
    ):
        assert cathar._cathar_polish_pass(in_wav, tmp_path) == in_wav


def test_cathar_steps_call_runner(tmp_path):
    """Verifies each cathar step invokes runner with expected output name."""
    in_wav = tmp_path / "in.wav"
    with (
        patch("modules.cathar._run_cathar_step", side_effect=lambda cmd, i, o, *a, **k: o),
        patch("modules.cathar._is_stereo_audio", return_value=True),
    ):
        res_dewind = cathar._cathar_dewind_step(in_wav, tmp_path)
        res_azimuth = cathar._cathar_azimuth_step(in_wav, tmp_path)
        res_mono = cathar._cathar_mono_below_step(in_wav, tmp_path)
        res_declick = cathar._cathar_declick_step(in_wav, tmp_path)
        res_decrackle = cathar._cathar_decrackle_step(in_wav, tmp_path)
        res_inpaint = cathar._cathar_inpaint_step(in_wav, tmp_path)
        res_deplosive = cathar._cathar_deplosive_step(in_wav, tmp_path)
        res_enhance = cathar._cathar_enhance_step(in_wav, tmp_path)
        res_declip = cathar._cathar_declip_step(in_wav, tmp_path)
        res_repair = cathar._cathar_repair_step(in_wav, tmp_path)
        res_dereverb = cathar._cathar_dereverb_step(in_wav, tmp_path)
        res_deesser = cathar._cathar_deesser_step(in_wav, tmp_path)
        res_denoise = cathar._cathar_denoise_step(in_wav, tmp_path)
    names = (
        res_dewind.name,
        res_azimuth.name,
        res_mono.name,
        res_declick.name,
        res_decrackle.name,
        res_inpaint.name,
        res_deplosive.name,
        res_enhance.name,
        res_declip.name,
        res_repair.name,
        res_dereverb.name,
        res_deesser.name,
        res_denoise.name,
    )
    expected = (
        "dewinded_in.wav",
        "azimuth_in.wav",
        "monobelow_in.wav",
        "declicked_in.wav",
        "decrackled_in.wav",
        "inpainted_in.wav",
        "deplosived_in.wav",
        "enhanced_in.wav",
        "declipped_in.wav",
        "repaired_in.wav",
        "dereverbed_in.wav",
        "deessed_in.wav",
        "denoised_in.wav",
    )
    assert names == expected
