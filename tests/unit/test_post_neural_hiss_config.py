"""The post-neural hiss keys: defaults that keep every output's bytes, validation, and the tuner's knowledge of them."""

from unittest.mock import MagicMock, patch

import pytest
import yaml

from modules import config as cfg
from modules import neural_level, post_neural_suppress
from scripts import tune_restoration

KEYS = {
    "apl_post_neural_suppress": False,
    "apl_post_neural_suppress_floor_gap_db": 3.0,
    "apl_post_neural_suppress_min_residual_db": 6.0,
    "apl_neural_level_restore": False,
    "linear_air_gate": "none",
}


def _loaded(user):
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = user
    with (
        patch("modules.config.Path.exists", return_value=True),
        patch.object(cfg, "yaml", mock_yaml),
        patch("builtins.open", MagicMock()),
        patch("builtins.print") as warn,
    ):
        conf, _source = cfg.load_config()
    return conf, warn


def test_the_resolved_defaults_and_their_module_copies_keep_every_output_s_bytes():
    assert {key: cfg.CONFIG[key] for key in KEYS} == KEYS
    constants = (cfg.APL_POST_NEURAL_SUPPRESS, cfg.APL_NEURAL_LEVEL_RESTORE, cfg.LINEAR_AIR_GATE)
    thresholds = (cfg.APL_POST_NEURAL_SUPPRESS_FLOOR_GAP_DB, cfg.APL_POST_NEURAL_SUPPRESS_MIN_RESIDUAL_DB)
    assert (constants, thresholds) == ((False, False, "none"), (3.0, 6.0))
    copies = (post_neural_suppress.APL_POST_NEURAL_SUPPRESS, post_neural_suppress.LINEAR_AIR_GATE, neural_level.APL_NEURAL_LEVEL_RESTORE)
    assert copies == (False, "none", False)


def test_the_shipped_config_yaml_names_the_defaults():
    with open(cfg.APP_DIR / "config.yaml", encoding="utf-8") as handle:
        shipped = yaml.safe_load(handle)
    assert {key: shipped[key] for key in KEYS} == KEYS


def test_a_config_without_the_keys_resolves_the_defaults_quietly():
    conf, warn = _loaded({"process_mode": "auto_pure_linear"})
    assert {key: conf[key] for key in KEYS} == KEYS
    warn.assert_not_called()


@pytest.mark.parametrize(
    ("raw", "expected", "warned"),
    [
        (" Hiss_Dominated ", "hiss_dominated", False),
        ("none", "none", False),
        ("always", "none", True),
        (True, "none", True),
        (None, "none", True),
    ],
)
def test_the_air_gate_is_a_choice(raw, expected, warned):
    conf, warn = _loaded({"linear_air_gate": raw})
    assert conf["linear_air_gate"] == expected
    assert warn.called is warned


def test_the_typed_keys_are_validated_like_every_other():
    conf, _warn = _loaded(
        {"apl_post_neural_suppress": "yes", "apl_post_neural_suppress_floor_gap_db": -1.0, "apl_post_neural_suppress_min_residual_db": "9"}
    )
    assert conf["apl_post_neural_suppress"] is True
    assert (conf["apl_post_neural_suppress_floor_gap_db"], conf["apl_post_neural_suppress_min_residual_db"]) == (3.0, 9.0)


def test_the_tuner_knows_the_keys_and_their_choices():
    fields = tune_restoration.known_config_fields()
    assert fields["linear_air_gate"][0] == {"none", "hiss_dominated"}
    sound = {"linear_air_gate": "hiss_dominated", "apl_post_neural_suppress": True, "apl_neural_level_restore": True}
    assert not tune_restoration.validate_overrides({**sound, "apl_post_neural_suppress_min_residual_db": 4.0})
    assert tune_restoration.validate_overrides({"linear_air_gate": "always"})
    assert tune_restoration.validate_overrides({"apl_post_neural_suppress_floor_gap_db": -1.0})
