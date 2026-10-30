"""cathar's own pause fill depth (`cathar_pause_floor_fill_db`, 2026-10-27).

The cathar v10 loop (calibrated listener, corpus guard) plateaued with the shared `pause_floor_fill_db` at 24 dB,
while APL's guarded loop refused 24 (13 of 42 sentinels), so cathar reads a key of its own, as it already did for
the shelf's cut. 24 dB since 2026-10-29, with `cathar_suppress_noise_bias` 1.5: heard blind against the shipped
settings (session3_cathar_v10) it never lost a pick, 3-0 with 33 same. The shared key keeps 18 for auto_pure_linear.
"""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

import modules.config as cfg
import modules.pause_floor as pause_floor
import modules.processing as processing
from modules.modes import CatharMode
from scripts import autotune_restoration as at
from tests.unit.test_pause_floor import _gated, _on
from tests.unit.test_pause_floor_hissfree import _pair, _settings, _tape_voice

REPO = Path(__file__).resolve().parents[2]
KEY = "cathar_pause_floor_fill_db"


def test_both_config_files_hold_cathar_s_fill_at_the_plateau_and_the_shared_one_where_it_was():
    shipped = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    assert (shipped[KEY], cfg.CONFIG[KEY], cfg.CATHAR_PAUSE_FLOOR_FILL_DB) == (24.0, 24.0, 24.0)
    assert (shipped["pause_floor_fill_db"], cfg.PAUSE_FLOOR_FILL_DB) == (18.0, 18.0)


def _loaded(user_config):
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = user_config
    with patch.object(cfg, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        return cfg.load_config()[0]


def test_the_key_coerces_and_falls_back_out_of_bounds_like_its_shared_twin():
    assert (_loaded({KEY: "18"})[KEY], _loaded({KEY: -3.0})[KEY], _loaded({KEY: 99.0})[KEY]) == (18.0, 24.0, 24.0)


def test_a_suppressed_file_takes_cathar_s_depth_and_an_unsuppressed_one_the_release_fill():
    with patch("modules.pause_floor.PAUSE_FLOOR_SUPPRESSED_ONLY", True), patch("modules.pause_floor.PAUSE_FLOOR_FILL_DB", 18.0):
        assert pause_floor.profile_for(True, fill_db=24.0) == pause_floor.configured()._replace(fill_db=24.0)
        assert pause_floor.profile_for(True, 12.0, 24.0) == pause_floor.configured()._replace(cut_db=12.0, fill_db=24.0)
        assert pause_floor.profile_for(True).fill_db == 18.0
        assert pause_floor.profile_for(False, 18.0, 24.0) == pause_floor.release_profile()


def _kept(tmp_path, **own):
    reference, restored = _pair(tmp_path, _gated(_tape_voice()))
    with ExitStack() as stack:
        for context in (*_on(18.0), *_settings(target="frame", shape="shelf", hz=2000.0, cut=12.0)):
            stack.enter_context(context)
        return pause_floor.apply_when_needed(reference, restored, tmp_path / "work", suppressed=True, **own)


def test_on_real_audio_the_fill_sits_cathar_s_depth_down(tmp_path):
    assert _kept(tmp_path / "own", cut_db=18.0, fill_db=24.0).name == "paused_24db_p15_frame_shelf2000c18_res.wav"
    assert _kept(tmp_path / "shared").name == "paused_18db_p15_frame_shelf2000c12_res.wav"


def test_the_cathar_mode_hands_its_own_depth_to_the_pause_floor(tmp_path):
    captured, handed = {}, []
    with (
        patch.object(processing, "_resolve_preconditioned_audio", return_value=(tmp_path / "clean.wav", {})),
        patch.object(processing, "_process_single_track_pipeline", side_effect=lambda *a, **k: captured.update(step=a[5])),
        patch("modules.cathar.filter_cathar_vhs_pipeline", return_value=tmp_path / "raw.wav"),
        patch("modules.modes.cathar.reference_suppressed", return_value=tmp_path / "ref.wav"),
        patch("modules.modes.cathar.CATHAR_PAUSE_FLOOR_FILL_DB", 24.0),
        patch.object(processing, "_polish_full_audio_step", return_value=tmp_path / "pol.wav"),
        patch.object(pause_floor, "apply_when_needed", side_effect=lambda *_a, **kw: handed.append(kw["fill_db"]) or tmp_path / "out.wav"),
    ):
        CatharMode().execute(tmp_path, tmp_path / "o.wav", tmp_path / "v.mp4", tmp_path / "f.mp4", 10.0, {})
        assert captured["step"](tmp_path / "in.wav", tmp_path) == tmp_path / "out.wav"
    assert handed == [24.0]


def _knobs(engine, incumbent):
    return {knob for knob, _ in at.neighbour_moves(engine, incumbent)}


def test_the_cathar_loop_tunes_its_own_depth_never_the_shared_one():
    """A cathar loop proposing pause_floor_fill_db would render its incumbent again; APL keeps the shared key."""
    assert (KEY in _knobs("cathar", {}), "pause_floor_fill_db" in _knobs("cathar", {})) == (True, False)
    assert ("pause_floor_fill_db" in _knobs("apl", {}), KEY in _knobs("apl", {})) == (True, False)
    off = ({"enable_pause_floor": False}, {"pause_floor_suppressed_only": True, "cathar_reference_suppress": False})
    assert all(KEY in at.inert_knobs("cathar", when) for when in off)
