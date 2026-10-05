"""auto_pure_linear's stem path on music: when it runs, what each stem goes through, and its fallback."""

from pathlib import Path
from unittest.mock import patch

import pytest

import modules.apl_stems as apl_stems


def _strategy(persistence):
    return {"profile": {"tonal_persistence": persistence}, "sync_method": "shift"}


@pytest.mark.parametrize(
    ("switch", "persistence", "expected"),
    [(True, 0.2, True), (True, 0.05, True), (True, 0.033, False), (True, None, False), (False, 0.9, False)],
)
def test_wanted_keys_on_the_switch_and_the_persistence_floor(switch, persistence, expected):
    with patch("modules.apl_stems.APL_MUSIC_STEM_PATH", switch), patch("modules.apl_stems.APL_MUSIC_PERSISTENCE_MIN", 0.05):
        assert apl_stems.wanted(_strategy(persistence)) is expected
    assert apl_stems.wanted({}) is False


def test_music_material_is_the_persistence_floor_alone():
    """The engine's music reading does not depend on the stem path's switch."""
    with patch("modules.apl_stems.APL_MUSIC_STEM_PATH", False), patch("modules.apl_stems.APL_MUSIC_PERSISTENCE_MIN", 0.05):
        assert apl_stems.music_material(_strategy(0.2)) is True
        assert apl_stems.music_material(_strategy(0.01)) is False
        assert apl_stems.music_material(None) is False


def _run(tmp_path, floor_db, separator=None, suppressor=None):
    calls = {}

    def denoise_step(in_wav, out_dir, total_duration=None):
        calls["vocals"] = (in_wav, out_dir)
        return out_dir / "restored_vocals.wav"

    with (
        patch(
            "modules.processing._separate_stems_step", side_effect=separator or (lambda *a, **k: (tmp_path / "v.wav", tmp_path / "b.wav"))
        ),
        patch("modules.processing._post_denoise_cleanup_step", side_effect=lambda wav, d, **k: d / f"cleaned_{Path(wav).name}") as cleanup,
        patch("modules.spectral_suppress.suppress_or_none", side_effect=suppressor or (lambda src, target, **s: target)) as suppress,
        patch("modules.processing._align_and_mix_stems") as mix,
        patch("modules.apl_stems.APL_MUSIC_BG_FLOOR_DB", floor_db),
    ):
        delivered = apl_stems.execute(
            tmp_path, tmp_path / "clean.wav", tmp_path / "orig.wav", "in.mp4", "out.mp4", 12.0, _strategy(0.2), denoise_step
        )
    return delivered, calls, cleanup, suppress, mix


def test_the_vocals_take_the_denoise_chain_and_reach_the_mix(tmp_path):
    delivered, calls, _cleanup, _suppress, mix = _run(tmp_path, -10.0)
    assert delivered is True
    assert calls["vocals"] == (tmp_path / "v.wav", tmp_path / "denoised_vocals")
    args = mix.call_args.args
    assert args[2] == tmp_path / "denoised_vocals" / "restored_vocals.wav"
    assert args[4:7] == ("in.mp4", "out.mp4", 12.0)


def test_the_background_takes_the_notches_and_the_suppressor(tmp_path):
    _delivered, _calls, cleanup, suppress, mix = _run(tmp_path, -10.0)
    assert cleanup.call_args.args[0] == tmp_path / "b.wav"
    assert suppress.call_args.kwargs["gain_floor_db"] == -10.0
    assert suppress.call_args.kwargs["noise_bias"] == 1.0
    assert mix.call_args.args[3] == tmp_path / "background" / "suppressed_cleaned_b.wav"


def test_a_zero_floor_passes_the_background_through_the_notches_only(tmp_path):
    _delivered, _calls, _cleanup, suppress, mix = _run(tmp_path, 0.0)
    suppress.assert_not_called()
    assert mix.call_args.args[3] == tmp_path / "background" / "cleaned_b.wav"


def test_a_suppressor_without_a_probe_leaves_the_notched_background(tmp_path):
    _delivered, _calls, _cleanup, _suppress, mix = _run(tmp_path, -10.0, suppressor=lambda src, target, **s: None)
    assert mix.call_args.args[3] == tmp_path / "background" / "cleaned_b.wav"


def test_the_mode_takes_the_stem_path_only_when_wanted(tmp_path):
    from modules.modes.auto_pure_linear import AutoPureLinearMode

    mode = AutoPureLinearMode()
    with (
        patch("modules.processing._resolve_preconditioned_audio", return_value=(tmp_path / "clean.wav", _strategy(0.2))),
        patch("modules.processing._process_single_track_pipeline") as single,
        patch("modules.apl_stems.execute", return_value=True) as stems,
        patch("modules.apl_stems.APL_MUSIC_STEM_PATH", True),
    ):
        mode.execute(tmp_path, tmp_path / "orig.wav", "in.mp4", "out.mp4", 12.0, None)
        stems.assert_called_once()
        single.assert_not_called()
        stems.reset_mock()
        with patch("modules.apl_stems.APL_MUSIC_STEM_PATH", False):
            mode.execute(tmp_path, tmp_path / "orig.wav", "in.mp4", "out.mp4", 12.0, None)
        stems.assert_not_called()
        single.assert_called_once()


def test_a_failing_separator_hands_back_to_the_single_track_path(tmp_path):
    def broken(*args, **kwargs):
        raise Exception("no GPU")

    delivered, calls, _cleanup, _suppress, mix = _run(tmp_path, -10.0, separator=broken)
    assert delivered is False
    assert "vocals" not in calls
    mix.assert_not_called()
