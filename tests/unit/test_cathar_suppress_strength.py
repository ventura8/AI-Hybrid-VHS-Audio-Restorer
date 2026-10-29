"""cathar's reference suppressor strength (2026-10-26): `cathar_suppress_noise_bias` and `cathar_suppress_gain_floor_db`.

A higher bias or a deeper floor takes more from the cells where the source held only hiss and leaves the cells where the
programme stood over it near 1. The bias is 1.5 since 2026-10-29 (the corpus-guarded v10 plateau, with cathar's fill at
24 dB; blind 3-0 over the shipped settings, 33 same); the floor keeps the estimator's -20 dB.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

import modules.config as cfg
import modules.modes.cathar as cathar_mode
import modules.reference_suppress as reference_suppress
from scripts import autotune_restoration as at

REPO = Path(__file__).resolve().parents[2]
KEYS = ("cathar_suppress_noise_bias", "cathar_suppress_gain_floor_db")


def test_the_defaults_are_the_v10_plateau_s_bias_and_the_estimator_s_floor_in_both_config_files():
    shipped = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    assert tuple(shipped[key] for key in KEYS) == tuple(cfg.CONFIG[key] for key in KEYS) == (1.5, -20.0)
    assert (cfg.CATHAR_SUPPRESS_NOISE_BIAS, cfg.CATHAR_SUPPRESS_GAIN_FLOOR_DB) == (1.5, reference_suppress.CATHAR_SETTINGS["gain_floor_db"])


def _loaded(user_config):
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = user_config
    with patch.object(cfg, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        return cfg.load_config()[0]


def test_the_keys_coerce_and_fall_back_out_of_bounds():
    coerced = _loaded(dict(zip(KEYS, ("2", "-30"))))
    assert tuple(coerced[key] for key in KEYS) == (2.0, -30.0)
    rejected = _loaded(dict(zip(KEYS, (9.0, 3.0))))
    assert tuple(rejected[key] for key in KEYS) == (1.5, -20.0)


def _settings_seen(tmp_path, **constants):
    seen = []
    with patch.object(cathar_mode.reference_suppress, "suppress", side_effect=lambda *a: seen.append(a[3]) or tmp_path / "out.wav"):
        with patch.multiple(cathar_mode, CATHAR_REFERENCE_SUPPRESS=True, CATHAR_REFERENCE_RELEASE_DB=6.0, **constants):
            cathar_mode.reference_suppressed(tmp_path / "src.wav", tmp_path / "raw.wav", tmp_path)
    return seen[0]


def test_at_the_defaults_the_settings_are_the_estimator_s_with_the_plateau_s_bias(tmp_path):
    settings = _settings_seen(tmp_path)
    plateau = {**reference_suppress.with_release(reference_suppress.CATHAR_SETTINGS, 6.0), "noise_bias": 1.5}
    assert settings == plateau
    estimator = reference_suppress.with_release(reference_suppress.CATHAR_SETTINGS, 6.0)
    assert reference_suppress.target_for(tmp_path / "raw.wav", tmp_path, settings) != reference_suppress.target_for(
        tmp_path / "raw.wav", tmp_path, estimator
    )


def test_the_strength_reaches_the_suppressor_and_names_its_own_output(tmp_path):
    settings = _settings_seen(tmp_path, CATHAR_SUPPRESS_NOISE_BIAS=2.0, CATHAR_SUPPRESS_GAIN_FLOOR_DB=-30.0)
    assert (settings["noise_bias"], settings["gain_floor_db"], settings["dd_alpha"], settings["release_db"]) == (2.0, -30.0, 0.96, 6.0)
    default = reference_suppress.with_release(reference_suppress.CATHAR_SETTINGS, 6.0)
    assert reference_suppress.target_for(tmp_path / "raw.wav", tmp_path, settings) != reference_suppress.target_for(
        tmp_path / "raw.wav", tmp_path, default
    )


def test_the_cathar_loop_tunes_the_strength_and_reads_it_inert_with_the_suppressor_off():
    assert set(KEYS) <= set(at.KNOBS["cathar"])
    assert at.KNOBS["cathar"]["cathar_suppress_noise_bias"][0] == 1.0
    inert = {tuple(sorted(condition.items())): knobs for condition, knobs in at.INERT_WHEN["cathar"]}
    assert set(KEYS) <= set(inert[(("cathar_reference_suppress", False),)])


def test_the_apl_loop_tunes_its_suppressors_noise_bias_dead_only_with_every_suppressor_off():
    """auto_pure_linear's key already existed (the native, post-neural and reference suppressors read it); since the v10
    plateau left Vaccin's hiss under speech audible it is a loop knob, inert where the gain floor is."""
    assert at.KNOBS["apl"]["apl_suppress_noise_bias"] == [1.0, 1.5, 2.0, 3.0, 4.0]
    holding = _apl_conditions_holding("apl_suppress_noise_bias")
    assert holding == _apl_conditions_holding("apl_suppress_gain_floor_db")
    assert {condition["apl_reference_suppress"] for condition in holding} == {False}


def _apl_conditions_holding(knob):
    """The INERT_WHEN conditions of auto_pure_linear under which `knob` is dead."""
    return [condition for condition, knobs in at.INERT_WHEN["apl"] if knob in knobs]
