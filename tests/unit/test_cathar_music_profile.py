"""cathar's music profile: which tapes take it and which stages it changes."""

from unittest.mock import patch

import pytest

import modules.cathar as cathar


def _strategy(persistence):
    return {"precondition_filters": {"notch_hz": 50.0}, "profile": {"tonal_persistence": persistence}}


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (_strategy(0.12), True),
        (_strategy(0.05), True),
        (_strategy(0.033), False),
        (_strategy(None), False),
        ({"precondition_filters": {}}, False),
        (None, False),
    ],
)
def test_music_material_keys_on_the_persistence_floor(strategy, expected):
    with patch("modules.cathar.CATHAR_MUSIC_PROFILE", True), patch("modules.cathar.CATHAR_MUSIC_PERSISTENCE_MIN", 0.05):
        assert cathar._is_music_material(strategy) is expected


def test_music_profile_can_be_switched_off():
    with patch("modules.cathar.CATHAR_MUSIC_PROFILE", False):
        assert cathar._is_music_material(_strategy(0.9)) is False


def test_material_settings_are_the_speech_defaults_or_the_music_profile():
    with (
        patch("modules.cathar.CATHAR_ALPHA", 2.0),
        patch("modules.cathar.CATHAR_ENABLE_NOISEPRINT", True),
        patch("modules.cathar.CATHAR_ENABLE_COHERENT", False),
        patch("modules.cathar.CATHAR_ENABLE_DEPLOSIVE", True),
        patch("modules.cathar.CATHAR_MUSIC_ALPHA", 0.5),
        patch("modules.cathar.CATHAR_MUSIC_ENABLE_NOISEPRINT", False),
        patch("modules.cathar.CATHAR_MUSIC_ENABLE_COHERENT", True),
        patch("modules.cathar.CATHAR_MUSIC_ENABLE_DEPLOSIVE", False),
        patch("modules.cathar.CATHAR_ALPHA_HIGH", 3.0),
        patch("modules.cathar.CATHAR_MUSIC_ALPHA_HIGH", 0.25),
        patch("modules.cathar.CATHAR_ENABLE_DEESSER", True),
        patch("modules.cathar.CATHAR_MUSIC_ENABLE_DEESSER", False),
    ):
        speech = cathar._material_settings(_strategy(0.001))
        music = cathar._material_settings(_strategy(0.2))
    assert speech == {"alpha": 2.0, "alpha_high": 3.0, "noiseprint": True, "coherent": False, "deplosive": True, "deesser": True}
    assert music == {"alpha": 0.5, "alpha_high": 0.25, "noiseprint": False, "coherent": True, "deplosive": False, "deesser": False}


def test_the_music_profile_has_its_own_expander_depth_and_notch_width():
    """On music the polish expander and the CRT notch take the profile's values; on speech the shared ones (None)."""
    with (
        patch("modules.cathar.CATHAR_MUSIC_PROFILE", True),
        patch("modules.cathar.CATHAR_MUSIC_PERSISTENCE_MIN", 0.05),
        patch("modules.cathar.CATHAR_MUSIC_EXPANDER_DEPTH_DB", 4.0),
        patch("modules.cathar.CATHAR_MUSIC_CRT_NOTCH_Q", 60.0),
    ):
        assert cathar.music_expander_depth_db(_strategy(0.2)) == 4.0
        assert cathar.music_crt_notch_q(_strategy(0.2)) == 60.0
        assert cathar.music_expander_depth_db(_strategy(0.001)) is None
        assert cathar.music_crt_notch_q(None) is None


def test_the_deesser_follows_the_material(tmp_path):
    """The polish pass runs the de-esser when the material's switch says so, the speech default when unspecified."""
    wav = tmp_path / "in.wav"
    with (
        patch("modules.cathar._cathar_deesser_step", side_effect=lambda w, *_a, **_k: w.with_name("deessed.wav")) as deess,
        patch("modules.cathar.CATHAR_ENABLE_ENHANCE", False),
        patch("modules.cathar.CATHAR_ENABLE_DEESSER", True),
    ):
        assert cathar._cathar_polish_pass(wav, tmp_path, deesser=False) == wav
        deess.assert_not_called()
        assert cathar._cathar_polish_pass(wav, tmp_path).name == "deessed.wav"
        assert cathar._cathar_polish_pass(wav, tmp_path, deesser=True).name == "deessed.wav"
        assert deess.call_count == 2


def _run_pipeline(tmp_path, strategy):
    original_wav = tmp_path / "orig.wav"
    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)
    with (
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._cathar_precondition_pass", return_value=original_wav) as pre,
        patch("modules.cathar._cathar_repair_pass", return_value=original_wav),
        patch("modules.cathar._cathar_noiseprint_step", return_value=work_dir / "noise.np.json") as noiseprint,
        patch("modules.cathar._cathar_denoise_step", return_value=original_wav) as denoise,
        patch("modules.cathar._cathar_polish_pass", return_value=original_wav),
        patch("modules.cathar.CATHAR_MUSIC_ALPHA", 0.5),
        patch("modules.cathar.CATHAR_MUSIC_ENABLE_NOISEPRINT", False),
        patch("modules.cathar.CATHAR_MUSIC_ENABLE_COHERENT", True),
        patch("modules.cathar.CATHAR_MUSIC_ENABLE_DEPLOSIVE", False),
    ):
        cathar.filter_cathar_vhs_pipeline(original_wav, work_dir, strategy=strategy)
    return pre, noiseprint, denoise


def test_pipeline_runs_the_music_profile_on_music(tmp_path):
    pre, noiseprint, denoise = _run_pipeline(tmp_path, _strategy(0.15))
    assert pre.call_args.kwargs["deplosive"] is False
    noiseprint.assert_not_called()
    assert denoise.call_args.kwargs["alpha"] == 0.5
    assert denoise.call_args.kwargs["coherent"] is True
    assert denoise.call_args.kwargs["noiseprint_path"] is None


def test_pipeline_keeps_the_speech_settings_on_dialogue(tmp_path):
    with (
        patch("modules.cathar.CATHAR_ALPHA", 2.0),
        patch("modules.cathar.CATHAR_ENABLE_NOISEPRINT", True),
        patch("modules.cathar.CATHAR_ENABLE_COHERENT", True),
        patch("modules.cathar.CATHAR_ENABLE_DEPLOSIVE", True),
    ):
        pre, noiseprint, denoise = _run_pipeline(tmp_path, _strategy(0.002))
    assert pre.call_args.kwargs["deplosive"] is True
    noiseprint.assert_called_once()
    assert denoise.call_args.kwargs["alpha"] == 2.0
    assert denoise.call_args.kwargs["coherent"] is True
    assert denoise.call_args.kwargs["noiseprint_path"] == tmp_path / "work" / "noise.np.json"


def test_precondition_pass_takes_the_material_switch_over_the_default(tmp_path):
    in_wav = tmp_path / "in.wav"
    with (
        patch("modules.cathar._run_cathar_step", side_effect=lambda cmd, i, o, *a, **k: o),
        patch("modules.cathar.CATHAR_ENABLE_DEWIND", False),
        patch("modules.cathar.CATHAR_ENABLE_AZIMUTH", False),
        patch("modules.cathar.CATHAR_ENABLE_MONO_BELOW", False),
        patch("modules.cathar.CATHAR_ENABLE_DECLICK", False),
        patch("modules.cathar.CATHAR_ENABLE_DECRACKLE", False),
        patch("modules.cathar.CATHAR_ENABLE_INPAINT", False),
        patch("modules.cathar.CATHAR_ENABLE_DEPLOSIVE", True),
    ):
        assert cathar._cathar_precondition_pass(in_wav, tmp_path, deplosive=False) == in_wav
        assert cathar._cathar_precondition_pass(in_wav, tmp_path).name == "deplosived_in.wav"
        assert cathar._deplosive_wanted(None) is True
