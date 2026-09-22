"""The polish and mastering knobs the tuning loop searches keep their shipped strings at the defaults."""

from unittest.mock import MagicMock, patch

import modules.config as cfg
import modules.filters as filters
import modules.mastering as mastering


def test_expander_defaults_reproduce_the_shipped_curves():
    assert filters._build_full_audio_expander_filter() == "compand=attacks=0.04:decays=0.18:points=-90/-100|-65/-72|-45/-45|0/0"
    assert (
        filters._build_full_audio_expander_filter(-52.0) == "compand=attacks=0.04:decays=0.18:points=-90/-100|-69.0/-76.0|-48.0/-48.0|0/0"
    )


def test_expander_depth_and_knee_offset_move_the_curve():
    shallow = filters._build_full_audio_expander_filter(-52.0, depth_db=4.0, knee_offset_db=0.0)
    assert shallow == "compand=attacks=0.04:decays=0.18:points=-90/-100|-71.0/-75.0|-52.0/-52.0|0/0"
    assert filters._build_full_audio_expander_filter(None, depth_db=12.0).endswith("|-65/-77|-45/-45|0/0")
    with patch("modules.filters.EXPANDER_DEPTH_DB", 3.0), patch("modules.filters.EXPANDER_KNEE_OFFSET_DB", 8.0):
        assert (
            filters._build_full_audio_expander_filter(-52.0)
            == "compand=attacks=0.04:decays=0.18:points=-90/-100|-67.0/-70.0|-44.0/-44.0|0/0"
        )


def test_crt_notch_width_follows_the_config():
    stages = []
    filters._append_notch_filters(stages, 0.0, crt_notch=15625.0)
    assert stages[-1] == "bandreject=f=15625.0:width_type=q:w=30"
    with patch("modules.filters.CRT_NOTCH_Q", 120.0):
        stages = []
        filters._append_notch_filters(stages, 0.0, crt_notch=15625.0)
    assert stages[-1] == "bandreject=f=15625.0:width_type=q:w=120"


def test_loudnorm_target_reads_the_configured_range():
    assert mastering._loudnorm_target_args() == "I=-16.0:TP=-1.0:LRA=11.0"
    with patch.object(cfg, "LOUDNORM_TARGET_LRA", 40.0):
        assert mastering._loudnorm_target_args() == "I=-16.0:TP=-1.0:LRA=40.0"


def _loudness_messages():
    with patch("modules.mastering.log_msg") as log:
        mastering._log_loudness_range({"input_i": "-20.1", "input_lra": "18.4", "input_tp": "-3.0"})
        mastering._log_loudness_range({"input_i": "-20.1", "input_lra": "6.0", "input_tp": "-3.0"})
        mastering._log_loudness_range({"input_lra": "nan"})
    return [call.args[0] for call in log.call_args_list]


def test_a_wide_measured_range_is_logged_as_dynamic_mode():
    messages = _loudness_messages()
    assert "dynamic" in messages[0]
    assert "LRA=18.4" in messages[0]


def test_a_narrow_range_stays_linear_and_an_unreadable_one_is_called_dynamic():
    messages = _loudness_messages()
    assert "-> linear" in messages[1]
    assert "dynamic" in messages[2]


def test_the_new_keys_coerce_and_are_bounded():
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = {
        "loudnorm_target_lra": "20",
        "expander_depth_db": "12.5",
        "expander_knee_offset_db": "-3",
        "crt_notch_q": "500",
    }
    with patch.object(cfg, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        conf, _ = cfg.load_config()
    assert conf["loudnorm_target_lra"] == 20.0
    assert conf["expander_depth_db"] == 12.5
    assert conf["expander_knee_offset_db"] == -3.0
    assert conf["crt_notch_q"] == 30.0
