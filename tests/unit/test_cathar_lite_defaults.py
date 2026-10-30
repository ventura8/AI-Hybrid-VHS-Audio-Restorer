"""cathar's defaults since 2026-10-25: the v9 loop's lighter set (candidate cfd333f9c7).

The reference suppressor with its 6 dB release, the per-frame fill shelved at 2 kHz (cut 18 dB,
fill 18 dB) and a 12 dB expander at 0.08 / 0.12 s, heard blind 36-0 over the shipped settings on
the four Tata tapes (session2_lite_vs_shipped), 6 of 235 IA clips past a v4 guard, realistic-v2
holding. The keys auto_pure_linear and the other modes share (the expander depth and timing, the
shelf's cut) keep their values; cathar reads its own twins, so only cathar's output moves.
Rendered at these defaults, the four tapes match the candidate's audio sample for sample.
"""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

import modules.cathar as cathar
import modules.config as cfg
import modules.filters as filters
import modules.pause_floor as pause_floor
import modules.processing as processing
from modules.modes import CatharMode
from tests.unit.test_pause_floor import _gated, _on
from tests.unit.test_pause_floor_hissfree import _pair, _settings, _tape_voice

REPO = Path(__file__).resolve().parents[2]
CATHAR_OWN = {
    "cathar_reference_suppress": True,
    "cathar_reference_release_db": 6.0,
    "cathar_expander_depth_db": 12.0,
    "cathar_expander_attack_s": 0.08,
    "cathar_expander_decay_s": 0.12,
    "cathar_pause_floor_hf_cut_db": 18.0,
}
# What auto_pure_linear and the modes with no key of their own keep reading.
SHARED = {"expander_depth_db": 7.0, "expander_attack_s": 0.04, "expander_decay_s": 0.18, "pause_floor_hf_cut_db": 12.0}


def test_the_lighter_set_is_cathar_s_default_in_both_config_files():
    shipped = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    for key, value in {**CATHAR_OWN, **SHARED}.items():
        assert (shipped[key], cfg.CONFIG[key]) == (value, value), key


def test_the_constants_carry_the_lighter_set():
    constants = (
        cfg.CATHAR_REFERENCE_SUPPRESS,
        cfg.CATHAR_REFERENCE_RELEASE_DB,
        cfg.CATHAR_EXPANDER_DEPTH_DB,
        cfg.CATHAR_EXPANDER_ATTACK_S,
        cfg.CATHAR_EXPANDER_DECAY_S,
        cfg.CATHAR_PAUSE_FLOOR_HF_CUT_DB,
    )
    shared = (cfg.EXPANDER_DEPTH_DB, cfg.EXPANDER_ATTACK_S, cfg.EXPANDER_DECAY_S, cfg.PAUSE_FLOOR_HF_CUT_DB)
    assert constants == tuple(CATHAR_OWN.values())
    assert shared == tuple(SHARED.values())


def _loaded(user_config):
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = user_config
    with patch.object(cfg, "yaml", mock_yaml), patch("builtins.open", MagicMock()):
        return cfg.load_config()[0]


def test_the_new_keys_coerce_and_fall_back_out_of_bounds_like_their_shared_twins():
    keys = ("cathar_expander_depth_db", "cathar_expander_attack_s", "cathar_expander_decay_s", "cathar_pause_floor_hf_cut_db")
    coerced = _loaded(dict(zip(keys, ("7", "0.04", "0.18", "24"))))
    assert tuple(coerced[key] for key in keys) == (7.0, 0.04, 0.18, 24.0)
    rejected = _loaded(dict(zip(keys, (99.0, 0.0, 9.0, -3.0))))
    assert tuple(rejected[key] for key in keys) == (12.0, 0.08, 0.12, 18.0)


def test_cathar_s_timing_reaches_the_compand_and_none_keeps_the_shared_one():
    strategy = {"profile": {"noise_floor_db": -52.0}}
    with patch("modules.filters.ENABLE_DYNAMIC_EXPANDER", True), patch("modules.filters.ENABLE_LINEAR_AIR", False):
        own = filters.build_full_audio_polish_filter(strategy, depth_db=12.0, timing_s=cathar.polish_timing_s())
        shared = filters.build_full_audio_polish_filter(strategy)
    assert own == "compand=attacks=0.08:decays=0.12:points=-90/-100|-67.0/-79.0|-44.0/-44.0|0/0"
    assert shared == "compand=attacks=0.04:decays=0.18:points=-90/-100|-67.0/-74.0|-44.0/-44.0|0/0"


def test_the_polish_step_names_its_output_by_the_timing(tmp_path):
    """A render at other timing never reuses this one: the stage's name carries the filter's fingerprint."""
    wav = tmp_path / "in.wav"
    seen = []
    with (
        patch("modules.processing.is_valid_audio", side_effect=lambda path: path == wav),
        patch("modules.processing._run_dsp_filter_file", side_effect=lambda _i, out, *_a: seen.append(out.name) or out),
    ):
        processing._polish_full_audio_step(wav, tmp_path, timing_s=(0.08, 0.12))
        processing._polish_full_audio_step(wav, tmp_path)
    assert len(set(seen)) == 2


def test_a_suppressed_file_takes_cathar_s_cut_and_an_unsuppressed_one_the_release_fill():
    with patch("modules.pause_floor.PAUSE_FLOOR_SUPPRESSED_ONLY", True), patch("modules.pause_floor.PAUSE_FLOOR_HF_CUT_DB", 12.0):
        assert pause_floor.profile_for(True, 18.0) == pause_floor.configured()._replace(cut_db=18.0)
        assert pause_floor.profile_for(True).cut_db == 12.0
        assert pause_floor.profile_for(False, 18.0) == pause_floor.release_profile()


def test_on_real_audio_the_fill_is_cut_cathar_s_depth(tmp_path):
    reference, restored = _pair(tmp_path, _gated(_tape_voice()))
    with ExitStack() as stack:
        for context in (*_on(), *_settings(target="frame", shape="shelf", hz=2000.0, cut=12.0)):
            stack.enter_context(context)
        kept = pause_floor.apply_when_needed(reference, restored, tmp_path / "work", suppressed=True, cut_db=18.0)
    assert kept.name == "paused_12db_p15_frame_shelf2000c18_res.wav"


def test_the_cathar_mode_hands_its_own_cut_to_the_pause_floor(tmp_path):
    captured, cuts = {}, []
    with (
        patch.object(processing, "_resolve_preconditioned_audio", return_value=(tmp_path / "clean.wav", {})),
        patch.object(processing, "_process_single_track_pipeline", side_effect=lambda *a, **k: captured.update(step=a[5])),
        patch("modules.cathar.filter_cathar_vhs_pipeline", return_value=tmp_path / "raw.wav"),
        patch("modules.modes.cathar.reference_suppressed", return_value=tmp_path / "ref.wav"),
        patch.object(processing, "_polish_full_audio_step", return_value=tmp_path / "pol.wav"),
        patch.object(pause_floor, "apply_when_needed", side_effect=lambda *_a, **kw: cuts.append(kw["cut_db"]) or tmp_path / "out.wav"),
    ):
        CatharMode().execute(tmp_path, tmp_path / "o.wav", tmp_path / "v.mp4", tmp_path / "f.mp4", 10.0, {})
        assert captured["step"](tmp_path / "in.wav", tmp_path) == tmp_path / "out.wav"
    assert cuts == [18.0]
