from unittest.mock import MagicMock, patch

import modules.config


@patch("modules.config.Path.exists", return_value=True)
def test_load_config_fail(mock_exists):
    """Test config loading failure handling."""
    mock_yaml = MagicMock()
    mock_yaml.safe_load.side_effect = Exception("YAML parse error")
    with patch.object(modules.config, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        conf, src = modules.config.load_config()
        assert src == "Defaults"


@patch("modules.config.Path.exists", return_value=True)
def test_load_config_normalizes_valid_process_mode(mock_exists):
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = {"process_mode": "denoise_only"}
    with patch.object(modules.config, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        conf, _ = modules.config.load_config()
        assert conf["process_mode"] == "denoise_only"


@patch("modules.config.Path.exists", return_value=True)
def test_load_config_fallback_process_mode(mock_exists):
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = {"process_mode": "INVALID_MODE"}
    with patch.object(modules.config, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        conf, _ = modules.config.load_config()
        assert conf["process_mode"] == "denoise_only"


@patch("builtins.print")
def test_normalize_process_mode_warns_for_non_string(mock_print):
    result = modules.config._normalize_process_mode(None)
    assert result == "denoise_only"
    mock_print.assert_called_once()


@patch("builtins.print")
def test_normalize_process_mode_warns_for_unknown_string(mock_print):
    result = modules.config._normalize_process_mode(" invalid_mode ")
    assert result == "denoise_only"
    mock_print.assert_called_once()


@patch("modules.config.Path.exists", return_value=True)
def test_load_config_defaults_when_pyyaml_missing_and_config_exists(mock_exists):
    with patch.object(modules.config, "yaml", None):
        conf, src = modules.config.load_config()
        assert src == "Defaults (PyYAML missing)"
        assert conf["process_mode"] == "denoise_only"
