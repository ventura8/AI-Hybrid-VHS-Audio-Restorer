"""The burst ceiling's keys: defaults that keep every output's bytes, validation, the tuner's and the stage cache's view of them."""

from unittest.mock import MagicMock, patch

import pytest
import yaml

from modules import burst_ceiling
from modules import config as cfg
from modules import stage_cache_key as cache_key
from scripts import tune_restoration

SCOPE_KEY = "apl_burst_ceiling"
DEPTH_KEY = "apl_burst_ceiling_depth_db"
EDGES = "edges"
KEYS = {SCOPE_KEY: "none", DEPTH_KEY: 30.0}


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


def test_the_resolved_defaults_and_the_module_copies_switch_the_stage_off():
    assert {key: cfg.CONFIG[key] for key in KEYS} == KEYS
    resolved = (cfg.APL_BURST_CEILING, cfg.APL_BURST_CEILING_DEPTH_DB)
    copies = (burst_ceiling.APL_BURST_CEILING, burst_ceiling.APL_BURST_CEILING_DEPTH_DB)
    assert (resolved, copies) == (("none", 30.0), ("none", 30.0))


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
        (" Edges ", EDGES, False),
        ("gaps", "gaps", False),
        ("none", "none", False),
        ("all", "none", True),
        (True, "none", True),
        (None, "none", True),
    ],
)
def test_the_scope_is_a_choice(raw, expected, warned):
    conf, warn = _loaded({SCOPE_KEY: raw})
    assert conf[SCOPE_KEY] == expected
    assert warn.called is warned


@pytest.mark.parametrize(("raw", "expected"), [("24", 24.0), (-1.0, 30.0), (200.0, 30.0), (0, 0.0)])
def test_the_depth_is_validated_like_every_typed_key(raw, expected):
    conf, _warn = _loaded({DEPTH_KEY: raw})
    assert conf[DEPTH_KEY] == expected


def test_the_tuner_knows_the_keys_and_their_choices():
    fields = tune_restoration.known_config_fields()
    assert fields[SCOPE_KEY][0] == {"none", "gaps", EDGES}
    assert not tune_restoration.validate_overrides({SCOPE_KEY: EDGES, DEPTH_KEY: 24.0})
    assert tune_restoration.validate_overrides({SCOPE_KEY: "all"})


def test_the_stage_cache_keeps_the_keys_and_the_module_out_of_its_key():
    keys = set(KEYS) <= cache_key.POST_NEURAL_CONFIG_KEYS
    assert (keys, "modules/burst_ceiling.py" in cache_key.POST_CACHE_MODULES) == (True, True)
