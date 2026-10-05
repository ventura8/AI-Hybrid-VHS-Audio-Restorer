"""The listener round's defaults (2026-09-25): what each engine takes on speech and on music, and where it is wired."""

from unittest.mock import patch

import modules.processing as processing
from modules.modes import AutoPureLinearMode, CatharMode


def _music():
    return {"precondition_filters": {"highpass_hz": 80, "crt_notch_hz": 15625.0}, "profile": {"tonal_persistence": 0.2}}


def _speech():
    return {"precondition_filters": {"highpass_hz": 80, "crt_notch_hz": 15625.0}, "profile": {"tonal_persistence": 0.001}}


def test_the_neural_model_follows_the_material_inside_the_chain():
    with (
        patch("modules.processing.APL_NEURAL_MODEL", "speech.ckpt"),
        patch("modules.processing.APL_MUSIC_NEURAL_MODEL", ""),
        patch("modules.apl_stems.APL_MUSIC_PERSISTENCE_MIN", 0.05),
        patch("modules.spectral_denoise.neural_wanted", return_value=True),
    ):
        assert processing._neural_stage(True, "adaptive.pth", "s.wav", _speech()) == ("speech.ckpt", True)
        assert processing._neural_stage(True, "adaptive.pth", "s.wav", _music()) == ("adaptive.pth", True)
        assert processing._neural_stage(True, "adaptive.pth", "s.wav", None) == ("speech.ckpt", True)
        assert processing._neural_stage(False, "adaptive.pth", "s.wav", _music()) == ("adaptive.pth", True)


def test_a_named_music_model_is_taken_on_music():
    with (
        patch("modules.processing.APL_NEURAL_MODEL", "speech.ckpt"),
        patch("modules.processing.APL_MUSIC_NEURAL_MODEL", "music.ckpt"),
        patch("modules.apl_stems.APL_MUSIC_PERSISTENCE_MIN", 0.05),
        patch("modules.spectral_denoise.neural_wanted", return_value=True),
    ):
        assert processing._neural_stage(True, "adaptive.pth", "s.wav", _music())[0] == "music.ckpt"


def test_cathar_on_music_narrows_the_crt_notch_in_the_preconditioning_config():
    with (
        patch("modules.cathar.CATHAR_MUSIC_PROFILE", True),
        patch("modules.cathar.CATHAR_MUSIC_PERSISTENCE_MIN", 0.05),
        patch("modules.cathar.CATHAR_MUSIC_CRT_NOTCH_Q", 60.0),
    ):
        music = processing._precondition_config_for("cathar", _music())
        speech = processing._precondition_config_for("cathar", _speech())
        apl = processing._precondition_config_for("auto_pure_linear", _music())
    assert music == {"highpass_hz": 80, "crt_notch_hz": 15625.0, "crt_notch_q": 60.0}
    assert speech == {"highpass_hz": 80, "crt_notch_hz": 15625.0}
    assert apl == {"highpass_hz": 80, "crt_notch_hz": 15625.0}
    assert "crt_notch_q" not in _music()["precondition_filters"]


def test_the_apl_mode_passes_its_own_expander_depth(tmp_path):
    orig = tmp_path / "orig.wav"
    with (
        patch("modules.processing._resolve_preconditioned_audio", return_value=(orig, {})),
        patch("modules.processing._process_single_track_pipeline") as pipeline,
        patch("modules.processing._denoise_and_polish_full_audio_step", return_value=orig) as denoise,
        patch("modules.modes.auto_pure_linear.APL_EXPANDER_DEPTH_DB", 12.0),
    ):
        AutoPureLinearMode().execute(tmp_path / "work", orig, tmp_path / "in.mp4", tmp_path / "out.mp4", 20.0, strategy={})
        pipeline.call_args[0][5](orig, tmp_path / "work")
    assert denoise.call_args.kwargs["expander_depth_db"] == 12.0


def test_the_cathar_mode_passes_the_music_depth_only_on_music(tmp_path):
    orig = tmp_path / "orig.wav"
    seen = []
    with (
        patch("modules.processing._resolve_preconditioned_audio", side_effect=lambda _w, _o, _d, _m, strategy: (orig, strategy)),
        patch("modules.processing._process_single_track_pipeline") as pipeline,
        patch("modules.cathar.filter_cathar_vhs_pipeline", return_value=orig),
        patch("modules.processing._polish_full_audio_step", side_effect=lambda w, *_a, **kw: seen.append(kw["depth_db"]) or w),
        patch("modules.pause_floor.apply_when_needed", side_effect=lambda _i, w, *_a, **_k: w),
        patch("modules.cathar.CATHAR_MUSIC_PROFILE", True),
        patch("modules.cathar.CATHAR_MUSIC_PERSISTENCE_MIN", 0.05),
        patch("modules.cathar.CATHAR_MUSIC_EXPANDER_DEPTH_DB", 4.0),
    ):
        for strategy in (_music(), _speech()):
            CatharMode().execute(tmp_path / "work", orig, tmp_path / "in.mp4", tmp_path / "out.mp4", 20.0, strategy=strategy)
            pipeline.call_args[0][5](orig, tmp_path / "work")
    assert seen == [4.0, None]
