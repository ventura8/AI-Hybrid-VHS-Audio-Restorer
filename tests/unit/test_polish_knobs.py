"""The polish and mastering knobs the tuning loop searches keep their shipped strings at the defaults."""

from unittest.mock import MagicMock, patch

import modules.config as cfg
import modules.filters as filters
import modules.mastering as mastering


def test_expander_defaults_reproduce_the_shipped_curves():
    """7 dB under a knee 8 dB over the floor: the listener round's curve for cathar on speech."""
    assert filters._build_full_audio_expander_filter() == "compand=attacks=0.04:decays=0.18:points=-90/-100|-65/-72|-45/-45|0/0"
    assert (
        filters._build_full_audio_expander_filter(-52.0) == "compand=attacks=0.04:decays=0.18:points=-90/-100|-67.0/-74.0|-44.0/-44.0|0/0"
    )


def test_the_polish_filter_takes_the_callers_expander_depth():
    """auto_pure_linear's 12 dB and cathar's 4 dB on music reach the curve; None keeps the shared depth."""
    strategy = {"profile": {"noise_floor_db": -52.0}}
    with patch("modules.filters.ENABLE_DYNAMIC_EXPANDER", True), patch("modules.filters.ENABLE_LINEAR_AIR", False):
        shared = filters.build_full_audio_polish_filter(strategy)
        deep = filters.build_full_audio_polish_filter(strategy, depth_db=12.0)
        shallow = filters.build_full_audio_polish_filter(strategy, depth_db=4.0)
    assert shared.endswith("|-67.0/-74.0|-44.0/-44.0|0/0")
    assert deep.endswith("|-67.0/-79.0|-44.0/-44.0|0/0")
    assert shallow.endswith("|-67.0/-71.0|-44.0/-44.0|0/0")


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


def test_a_precondition_config_can_name_its_own_crt_notch_width():
    """cathar's music profile puts its width into the config the graph is built from; absent, the shared Q holds."""
    graph = filters._precondition_filter_from_config({"highpass_hz": 80, "notch_hz": 0.0, "crt_notch_hz": 15625.0, "crt_notch_q": 60.0})
    assert graph.endswith("bandreject=f=15625.0:width_type=q:w=60")
    graph = filters._precondition_filter_from_config({"highpass_hz": 80, "notch_hz": 0.0, "crt_notch_hz": 15625.0})
    assert graph.endswith("bandreject=f=15625.0:width_type=q:w=30")


def test_loudnorm_target_reads_the_configured_range():
    assert mastering._loudnorm_target_args() == "I=-16.0:TP=-1.0:LRA=20.0"
    with patch.object(cfg, "LOUDNORM_TARGET_LRA", 40.0):
        assert mastering._loudnorm_target_args() == "I=-16.0:TP=-1.0:LRA=40.0"


def _loudness_messages():
    """The three verdicts at the broadcast 11 LU target: 18.4 measured is over it, 6.0 under it, nan unreadable.

    The narrow range peaks at -6 dBTP, so the +4.1 dB gain to -16 LUFS leaves it at -1.9 dBTP,
    inside ffmpeg's true-peak half of the linear rule too.
    """
    with patch("modules.mastering.log_msg") as log, patch.object(cfg, "LOUDNORM_TARGET_LRA", 11.0):
        mastering._log_loudness_range({"input_i": "-20.1", "input_lra": "18.4", "input_tp": "-3.0", "input_thresh": "-30.5"})
        mastering._log_loudness_range({"input_i": "-20.1", "input_lra": "6.0", "input_tp": "-6.0", "input_thresh": "-30.5"})
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


def test_the_air_shelf_defaults_to_the_by_ear_choice():
    """+1 dB at 7.5 kHz: the user's pick over +2 dB and off on the Tata tapes (2026-10-08)."""
    assert cfg.LINEAR_AIR_GAIN_DB == 1.0
    with patch("modules.filters.ENABLE_LINEAR_AIR", True):
        assert filters._build_linear_air_filter() == "treble=g=1.0:f=7500"


# The polish graphs every release up to 2026-10-09 built (HEAD 8d57030), on a -52 dBFS floor and on none.
SHIPPED_POLISH = {
    "auto_pure_linear": "treble=g=1.0:f=7500,compand=attacks=0.04:decays=0.18:points=-90/-100|-67.0/-79.0|-44.0/-44.0|0/0",
    "cathar speech, denoise_only": "compand=attacks=0.04:decays=0.18:points=-90/-100|-67.0/-74.0|-44.0/-44.0|0/0",
    "cathar music": "compand=attacks=0.04:decays=0.18:points=-90/-100|-67.0/-71.0|-44.0/-44.0|0/0",
    "no floor": "treble=g=1.0:f=7500,compand=attacks=0.04:decays=0.18:points=-90/-100|-65/-77|-45/-45|0/0",
}


def test_the_default_polish_graphs_are_the_shipped_strings():
    """The repository's config.yaml builds the same bytes as the hard-coded 7500 Hz and 0.04 / 0.18 s did."""
    floor = {"profile": {"noise_floor_db": -52.0}}
    built = {
        "auto_pure_linear": filters.build_full_audio_polish_filter(floor, apply_air=True, depth_db=cfg.APL_EXPANDER_DEPTH_DB),
        "cathar speech, denoise_only": filters.build_full_audio_polish_filter(floor),
        "cathar music": filters.build_full_audio_polish_filter(floor, depth_db=cfg.CATHAR_MUSIC_EXPANDER_DEPTH_DB),
        "no floor": filters.build_full_audio_polish_filter({}, apply_air=True, depth_db=cfg.APL_EXPANDER_DEPTH_DB),
    }
    assert built == SHIPPED_POLISH


def test_the_new_keys_default_to_the_shipped_shelf_corner_and_timing():
    """Unset, and as the repository's config.yaml sets them: 7500 Hz, 0.04 s and 0.18 s."""
    defaults = cfg._typed_config_defaults()
    shipped = (7500.0, 0.04, 0.18)
    assert (defaults["linear_air_freq_hz"], defaults["expander_attack_s"], defaults["expander_decay_s"]) == shipped
    assert (cfg.LINEAR_AIR_FREQ_HZ, cfg.EXPANDER_ATTACK_S, cfg.EXPANDER_DECAY_S) == shipped


def test_the_air_corner_follows_the_config():
    """Round A1's corners; a fractional corner keeps its fraction."""
    built = {}
    for corner in (6000.0, 9000.0, 7500.5):
        with patch("modules.filters.ENABLE_LINEAR_AIR", True), patch("modules.filters.LINEAR_AIR_FREQ_HZ", corner):
            built[corner] = filters._build_linear_air_filter(1.5)
    assert built == {6000.0: "treble=g=1.5:f=6000", 9000.0: "treble=g=1.5:f=9000", 7500.5: "treble=g=1.5:f=7500.5"}


def test_the_expander_timing_follows_the_config():
    """Round A3's attack and decay reach both curves, with and without a scanned floor."""
    with patch("modules.filters.EXPANDER_ATTACK_S", 0.02), patch("modules.filters.EXPANDER_DECAY_S", 0.3):
        unscanned = filters._build_full_audio_expander_filter()
        scanned = filters._build_full_audio_expander_filter(-52.0)
    assert unscanned == "compand=attacks=0.02:decays=0.3:points=-90/-100|-65/-72|-45/-45|0/0"
    assert scanned == "compand=attacks=0.02:decays=0.3:points=-90/-100|-67.0/-74.0|-44.0/-44.0|0/0"


def test_the_corner_and_timing_keys_coerce_and_are_bounded():
    """Quoted numbers are read; zero, a corner past 16 kHz, a decay past 5 s and a word fall back to the shipped value."""
    mock_yaml = MagicMock()
    accepted = {"linear_air_freq_hz": "9000", "expander_attack_s": "0.08", "expander_decay_s": 0.12}
    refused = {"linear_air_freq_hz": 30000, "expander_attack_s": 0, "expander_decay_s": "slow"}
    resolved = []
    for user in (accepted, refused, {"expander_decay_s": 10.0, "expander_attack_s": True}):
        mock_yaml.safe_load.return_value = user
        with patch.object(cfg, "yaml", mock_yaml), patch("builtins.open", MagicMock()), patch("builtins.print"):
            conf, _ = cfg.load_config()
        resolved.append((conf["linear_air_freq_hz"], conf["expander_attack_s"], conf["expander_decay_s"]))
    assert resolved == [(9000.0, 0.08, 0.12), (7500.0, 0.04, 0.18), (7500.0, 0.04, 0.18)]
