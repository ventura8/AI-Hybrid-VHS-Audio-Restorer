from unittest.mock import MagicMock, patch

import pytest

import modules.config


def test_config_paths_prioritize_launch_directory(monkeypatch, tmp_path):
    """A user config beside the launcher takes precedence over bundled settings."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(modules.config.sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)

    launch_config, bundled_config = modules.config._config_paths()

    assert launch_config == tmp_path / "config.yaml"
    assert bundled_config == tmp_path / "bundle" / "config.yaml"


@patch("modules.config.Path.exists", return_value=True)
def test_load_config_fail(mock_exists):
    """Test config loading failure handling."""
    mock_yaml = MagicMock()
    mock_yaml.safe_load.side_effect = Exception("YAML parse error")
    with patch.object(modules.config, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        conf, src = modules.config.load_config()
        assert src == "Defaults"


@pytest.mark.parametrize(
    "valid_mode",
    [
        "auto",
        "multipass_auto",
        "multipass",
        "auto_pure",
        "auto_pure_linear",
        "cathar",
        "cathar_vhs",
        "pure",
        "hybrid",
        "denoise_only",
        "ffmpeg_native",
        "auto_ffmpeg_native",
        "vhs_native",
        "auto_vhs_native",
        "arnndn_speech",
    ],
)
@patch("modules.config.Path.exists", return_value=True)
def test_load_config_normalizes_valid_process_mode(mock_exists, valid_mode):
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = {"process_mode": valid_mode}
    with patch.object(modules.config, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        conf, _ = modules.config.load_config()
        assert conf["process_mode"] == valid_mode


@patch("modules.config.Path.exists", return_value=True)
def test_load_config_fallback_process_mode(mock_exists):
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = {"process_mode": "INVALID_MODE"}
    with patch.object(modules.config, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        conf, _ = modules.config.load_config()
        assert conf["process_mode"] == "auto_pure_linear"


@patch("builtins.print")
def test_normalize_process_mode_warns_for_non_string(mock_print):
    result = modules.config._normalize_process_mode(None)
    assert result == "auto_pure_linear"
    mock_print.assert_called_once()


@patch("builtins.print")
def test_normalize_process_mode_warns_for_unknown_string(mock_print):
    result = modules.config._normalize_process_mode(" invalid_mode ")
    assert result == "auto_pure_linear"
    mock_print.assert_called_once()


@patch("modules.config.Path.exists", return_value=True)
def test_load_config_defaults_when_pyyaml_missing_and_config_exists(mock_exists):
    expected_defaults = {
        "process_mode": "auto_pure_linear",
        "afftdn_nr": 10.0,
        "afftdn_nf": -55.0,
        "afftdn_tn": True,
        "highpass_freq": 80,
        "enable_adeclick": True,
        "notch_freq": 50.0,
        "arnndn_model": "cb.rnnn",
        "arnndn_highpass_freq": 60,
        "arnndn_enable_adeclick": True,
    }
    with patch.object(modules.config, "yaml", None):
        conf, src = modules.config.load_config()
        assert src == "Defaults (PyYAML missing)"
        assert all(conf[k] == v for k, v in expected_defaults.items())


@patch("modules.config.Path.exists", return_value=True)
def test_load_config_non_mapping_yaml(mock_exists):
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = ["item1", "item2"]
    with patch.object(modules.config, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        conf, src = modules.config.load_config()
        assert src == "Defaults (invalid config.yaml)"
        assert conf["process_mode"] == "auto_pure_linear"
        assert conf["arnndn_model"] == "cb.rnnn"


def test_normalize_mix_volume_valid_numeric():
    assert modules.config._normalize_mix_volume(1.5, "vocal_mix_volume") == 1.5
    assert modules.config._normalize_mix_volume(0, "background_mix_volume") == 0.0
    assert modules.config._normalize_mix_volume(10.0, "vocal_mix_volume") == 10.0
    assert modules.config._normalize_mix_volume(" 0.75 ", "vocal_mix_volume") == 0.75


@pytest.mark.parametrize(
    "bad_value",
    [
        "1.0[v];ametadata=file=wipe",
        "1.0,movie=http://attacker",
        -1.0,
        100.0,
        float("nan"),
        float("inf"),
        ["not", "valid"],
    ],
)
@patch("builtins.print")
def test_normalize_mix_volume_invalid_and_malicious(mock_print, bad_value):
    assert modules.config._normalize_mix_volume(bad_value, "vocal_mix_volume") == 1.0
    mock_print.assert_called_once()


@patch("modules.config.Path.exists", return_value=True)
def test_load_config_sanitizes_mix_volumes(mock_exists):
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = {
        "vocal_mix_volume": "1.0;ametadata=mode=print:file=wiped",
        "background_mix_volume": 0.8,
    }
    with patch.object(modules.config, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        conf, _ = modules.config.load_config()
        assert conf["vocal_mix_volume"] == 1.0
        assert conf["background_mix_volume"] == 0.8


@pytest.mark.parametrize(
    "field, bad_value, expected",
    [
        ("afftdn_nr", None, 12.0),
        ("afftdn_nf", "loud", -45.0),
        ("highpass_freq", None, 60),
        ("highpass_freq", float("nan"), 60),
        ("arnndn_highpass_freq", "sixty", 60),
        ("notch_freq", ["x"], 0.0),
        ("cathar_inpaint_iterations", -1, 3),
        ("cathar_dehum_harmonics", -5, 8),
    ],
)
@patch("builtins.print")
def test_coerce_number_rejects_invalid_and_null(mock_print, field, bad_value, expected):
    meta = {name: (cast, min_val) for name, cast, _, min_val in modules.config._NUMERIC_CONFIG_FIELDS}[field]
    caster, min_val = meta
    assert modules.config._coerce_number(bad_value, caster, field, expected, min_val=min_val) == expected
    mock_print.assert_called_once()


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("false", False),
        ("False", False),
        (" 0 ", False),
        ("off", False),
        ("true", True),
        ("yes", True),
        (0, False),
        (1, True),
    ],
)
def test_coerce_bool_parses_string_booleans_by_content(raw_value, expected):
    assert modules.config._coerce_bool(raw_value, "afftdn_tn", True) is expected


@patch("builtins.print")
def test_coerce_bool_falls_back_for_unparseable(mock_print):
    assert modules.config._coerce_bool("maybe", "enable_loudnorm", True) is True
    mock_print.assert_called_once()


@patch("modules.config.Path.exists", return_value=True)
def test_load_config_normalizes_typed_fields(mock_exists):
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = {
        "afftdn_nr": None,
        "highpass_freq": "not-a-number",
        "afftdn_tn": "false",
        "preserve_original_audio_track": "yes",
    }
    with patch.object(modules.config, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        conf, _ = modules.config.load_config()
    assert conf["afftdn_nr"] == 10.0
    assert conf["highpass_freq"] == 80
    assert conf["afftdn_tn"] is False
    assert conf["preserve_original_audio_track"] is True


@pytest.mark.parametrize("extensions", [".mp4", 123, None])
@patch("modules.config.Path.exists", return_value=True)
def test_load_config_rejects_scalar_extensions(mock_exists, extensions):
    """Scalar extension values must not be converted into character sets."""
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = {"extensions": extensions}
    with patch.object(modules.config, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        conf, _ = modules.config.load_config()
    assert conf["extensions"] == [".mp4", ".mkv", ".avi", ".mov", ".mpg", ".mpeg", ".ts", ".m2ts"]
