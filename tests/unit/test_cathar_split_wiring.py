"""cathar's split-band subtraction: when it runs and what it renders per band."""

from unittest.mock import patch

import modules.cathar as cathar


def _strategy(persistence=0.001):
    return {"precondition_filters": {"notch_hz": 50.0}, "profile": {"tonal_persistence": persistence}}


def test_split_band_is_wanted_only_with_a_crossover_and_a_factor_to_split():
    with patch("modules.cathar.CATHAR_SPLIT_BAND_HZ", 6000), patch("modules.cathar.CATHAR_DENOISE_METHOD", "spectral"):
        assert cathar._split_band_wanted() is True
    with patch("modules.cathar.CATHAR_SPLIT_BAND_HZ", 0), patch("modules.cathar.CATHAR_DENOISE_METHOD", "spectral"):
        assert cathar._split_band_wanted() is False
    with patch("modules.cathar.CATHAR_SPLIT_BAND_HZ", 6000), patch("modules.cathar.CATHAR_DENOISE_METHOD", "wiener"):
        assert cathar._split_band_wanted() is False


def _run(tmp_path, crossover, strategy=None):
    original = tmp_path / "orig.wav"
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    with (
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._cathar_precondition_pass", return_value=original),
        patch("modules.cathar._cathar_repair_pass", return_value=original),
        patch("modules.cathar._cathar_noiseprint_step", return_value=work / "noise.np.json"),
        patch("modules.cathar._cathar_denoise_step", side_effect=lambda cur, out_dir, **kw: out_dir / f"denoised_{cur.name}") as denoise,
        patch("modules.cathar._cathar_polish_pass", side_effect=lambda cur, *a, **k: cur),
        patch("modules.split_band.recombine", side_effect=lambda low, high, target, hz: target) as recombine,
        patch("modules.cathar.CATHAR_SPLIT_BAND_HZ", crossover),
        patch("modules.cathar.CATHAR_DENOISE_METHOD", "spectral"),
        patch("modules.cathar.CATHAR_ALPHA", 1.5),
        patch("modules.cathar.CATHAR_ALPHA_HIGH", 3.0),
        patch("modules.cathar.CATHAR_MUSIC_ALPHA", 0.5),
        patch("modules.cathar.CATHAR_MUSIC_ALPHA_HIGH", 0.25),
    ):
        result = cathar.filter_cathar_vhs_pipeline(original, work, strategy=strategy or _strategy())
    return result, denoise, recombine, work


def test_a_crossover_renders_the_low_band_in_place_and_the_high_band_in_its_own_dir(tmp_path):
    _result, denoise, _recombine, work = _run(tmp_path, 6000)
    assert denoise.call_count == 2
    low_call, high_call = denoise.call_args_list
    assert (low_call.kwargs["alpha"], low_call.args[1]) == (1.5, work)
    assert (high_call.kwargs["alpha"], high_call.args[1]) == (3.0, work / "split_high")
    assert (work / "split_high").is_dir()


def test_a_crossover_recombines_the_bands_into_a_named_file(tmp_path):
    result, _denoise, recombine, _work = _run(tmp_path, 6000)
    recombine.assert_called_once()
    assert recombine.call_args.args[3] == 6000
    assert result.name == "splitband_6000_orig.wav"


def test_no_crossover_keeps_the_single_pass(tmp_path):
    result, denoise, recombine, work = _run(tmp_path, 0)
    assert denoise.call_count == 1
    recombine.assert_not_called()
    assert result == work / "denoised_orig.wav"


def test_music_takes_its_own_high_factor(tmp_path):
    with patch("modules.cathar.CATHAR_MUSIC_PROFILE", True), patch("modules.cathar.CATHAR_MUSIC_PERSISTENCE_MIN", 0.05):
        _result, denoise, _recombine, _work = _run(tmp_path, 8000, strategy=_strategy(0.2))
    assert [call.kwargs["alpha"] for call in denoise.call_args_list] == [0.5, 0.25]
