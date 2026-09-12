"""The full-audio chain's stage runner: order, opt-in, and what the neural model follows.

The stages ahead of the neural denoiser are opted into by the mode, run in a fixed order,
and each returns its input unchanged when it skips; the runner reads that to say what ran.
"""

from unittest.mock import patch

import modules.processing as processing
from modules import apl_chain


def test_the_stages_run_in_the_documented_order(tmp_path):
    """Physical repair before the tonal stages, and subtraction last, ahead of the neural stage."""
    plan = apl_chain.stage_plan(tmp_path, 10.0, {"profile": {}}, physical_repair=True, spectral_denoise=True)
    names = [name for name, _wanted, _stage in plan]
    assert names == ["physical_repair", "hum_cancel", "tone_cancel", "plosive_tamer", "tonal_cleanup", "spectral_denoise"]
    assert [wanted for _name, wanted, _stage in plan] == [True, False, False, False, True, True]


def test_a_mode_that_opts_into_nothing_runs_nothing(tmp_path):
    """denoise_only keeps the behaviour it shipped with: no optional stage, input returned."""
    plan = apl_chain.stage_plan(tmp_path, None, None, physical_repair=False, spectral_denoise=False)
    source = tmp_path / "in.wav"
    with patch("modules.apl_chain.log_msg") as log:
        assert apl_chain.run(source, plan) == (source, [])
    assert "applied: none; skipped: none" in log.call_args[0][0]


def test_the_runner_reports_which_stages_changed_the_audio(tmp_path):
    """A stage that hands back its input skipped; one that hands back another file applied."""
    source, repaired = tmp_path / "in.wav", tmp_path / "repaired.wav"
    plan = (
        ("physical_repair", True, lambda wav: repaired),
        ("tonal_cleanup", True, lambda wav: wav),
        ("spectral_denoise", False, lambda wav: tmp_path / "never.wav"),
    )
    with patch("modules.apl_chain.log_msg") as log:
        assert apl_chain.run(source, plan) == (repaired, ["physical_repair"])
    assert "applied: physical_repair; skipped: tonal_cleanup" in log.call_args[0][0]


def test_each_stage_receives_the_previous_stage_output(tmp_path):
    """The chain is sequential: every stage sees the file the one before it produced."""
    seen = []

    def stage(name):
        def run(wav):
            seen.append((name, wav))
            return tmp_path / f"{name}.wav"

        return run

    plan = tuple((name, True, stage(name)) for name in ("first", "second"))
    with patch("modules.apl_chain.log_msg"):
        final, applied = apl_chain.run(tmp_path / "in.wav", plan)
    assert seen == [("first", tmp_path / "in.wav"), ("second", tmp_path / "first.wav")]
    assert (final, applied) == (tmp_path / "second.wav", ["first", "second"])


def _chain_patches(tmp_path, models, subtraction):
    """Patches every stage around the runner so only the chain's own decisions are exercised."""
    return (
        patch("modules.processing._pre_denoise_surgical_step", return_value=tmp_path / "surg.wav"),
        patch("modules.apl_chain._physical_repair.apply_when_needed", return_value=tmp_path / "repaired.wav"),
        patch("modules.apl_chain._spectral_denoise.apply_tonal_cleanup", side_effect=lambda wav, *_a, **_k: wav),
        patch("modules.apl_chain._spectral_denoise.apply_when_needed", side_effect=subtraction),
        patch("modules.processing._denoise_full_audio_step", side_effect=lambda wav, _d, **kw: models.append(kw["denoise_model"]) or wav),
        patch("modules.processing._post_denoise_cleanup_step", side_effect=lambda wav, *_a, **_k: wav),
        patch("modules.processing._polish_full_audio_step", side_effect=lambda wav, *_a, **_k: wav),
        patch("modules.apl_chain.log_msg"),
    )


def test_the_deep_model_follows_the_subtraction_stage_not_the_path(tmp_path):
    """A stage before subtraction may change the file; only subtraction having run upgrades the model."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    models = []
    patches = _chain_patches(tmp_path, models, lambda wav, *_a, **_k: wav)
    for active in patches:
        active.start()
    try:
        result = processing._denoise_and_polish_full_audio_step(
            tmp_path / "orig.wav", out_dir, denoise_model="UVR-DeNoise-Lite.pth", physical_repair=True, spectral_denoise=True
        )
    finally:
        for active in patches:
            active.stop()
    assert result == tmp_path / "repaired.wav"
    assert models == ["UVR-DeNoise-Lite.pth"]


def test_subtraction_having_run_upgrades_the_neural_model(tmp_path):
    """When the subtraction stage produced a new file the deep separator partners it."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    models = []
    patches = _chain_patches(tmp_path, models, lambda wav, *_a, **_k: tmp_path / "subtracted.wav")
    for active in patches:
        active.start()
    try:
        result = processing._denoise_and_polish_full_audio_step(
            tmp_path / "orig.wav", out_dir, denoise_model="UVR-DeNoise-Lite.pth", physical_repair=True, spectral_denoise=True
        )
    finally:
        for active in patches:
            active.stop()
    assert result == tmp_path / "subtracted.wav"
    assert models == [processing._spectral_denoise.DEEP_DENOISE_MODEL]
