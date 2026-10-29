"""The stages after the neural denoise: sibilant guard, polish, pause floor; and which modes opt in."""

from pathlib import Path
from unittest.mock import patch

import modules.processing as processing
from modules.modes import AutoPureLinearMode
from modules.modes.denoise_only import DenoiseOnlyMode


def _run(stages, tmp_path):
    original, surgical, cleaned = tmp_path / "orig.wav", tmp_path / "surgical.wav", tmp_path / "cleaned.wav"
    with (
        patch("modules.processing._polish_full_audio_step", side_effect=lambda wav, *a, **k: Path(str(wav) + ".polished")) as polish,
        patch("modules.sibilant_guard.apply_when_needed", side_effect=lambda ref, wav, *a, **k: Path(str(wav) + ".guarded")) as guard,
        patch("modules.pause_floor.apply_when_needed", side_effect=lambda ref, wav, *a, **k: Path(str(wav) + ".floored")) as floor,
    ):
        result = processing._post_neural_stages(original, surgical, cleaned, tmp_path, 10.0, {"profile": {}}, stages)
    return result, polish, guard, floor


def test_all_stages_run_in_order_with_their_references(tmp_path):
    result, polish, guard, floor = _run({"sibilant_guard": True, "pause_floor": True, "apply_air": True}, tmp_path)
    assert result.name == "cleaned.wav.guarded.polished.floored"
    assert guard.call_args.args[0] == tmp_path / "surgical.wav"
    assert floor.call_args.args[0] == tmp_path / "orig.wav"
    assert polish.call_args.kwargs["apply_air"] is True


def test_stages_not_opted_in_are_never_imported_or_called(tmp_path):
    result, polish, guard, floor = _run({}, tmp_path)
    assert result.name == "cleaned.wav.polished"
    guard.assert_not_called()
    floor.assert_not_called()
    assert polish.call_args.kwargs["apply_air"] is False


def test_the_denoise_step_threads_the_switches_through(tmp_path):
    with (
        patch("modules.processing._resolve_adaptive_denoise_model", return_value="m"),
        patch("modules.processing._pre_denoise_surgical_step", return_value=tmp_path / "s.wav"),
        patch("modules.processing._apl_chain.stage_plan", return_value=[]),
        patch("modules.processing._apl_chain.run", return_value=(tmp_path / "s.wav", [])),
        patch("modules.processing._neural_stage", return_value=("m", False)),
        patch("modules.processing._without_stale_neural_output", side_effect=lambda d: d),
        patch("modules.processing._post_denoise_cleanup_step", return_value=tmp_path / "c.wav"),
        patch("modules.processing._post_neural_stages", return_value=tmp_path / "done.wav") as post,
    ):
        out = processing._denoise_and_polish_full_audio_step(tmp_path / "o.wav", tmp_path, sibilant_guard=True, pause_floor=True)
    assert out == tmp_path / "done.wav"
    switches = dict.fromkeys(("apply_air", "post_neural_suppress", "neural_level_restore", "air_gate", "burst_ceiling"), False)
    assert post.call_args.args[-2] == {"sibilant_guard": True, "pause_floor": True, **switches}
    assert post.call_args.args[-1] is None


def test_the_stage_cache_is_handed_the_input_the_stage_folder_and_every_neural_switch(tmp_path):
    """The cache keys on all seven switches and the strategy, and none of the post-neural settings."""
    pair = (tmp_path / "s.wav", tmp_path / "d.wav")
    switches = dict.fromkeys(processing.NEURAL_FLAGS, True)
    with (
        patch("modules.processing._stage_cache.through", return_value=pair) as through,
        patch("modules.processing._post_denoise_cleanup_step", return_value=tmp_path / "c.wav") as cleanup,
        patch("modules.processing._post_neural_stages", return_value=tmp_path / "done.wav"),
    ):
        processing._denoise_and_polish_full_audio_step(
            tmp_path / "o.wav", tmp_path, strategy={"profile": {}}, apply_air=True, expander_depth_db=12.0, **switches
        )
    produce, input_wav, audio_dir, call = through.call_args.args
    handed = (callable(produce), input_wav, audio_dir, call["flags"], call["strategy"], call["stage"])
    assert handed == (True, tmp_path / "o.wav", tmp_path, switches, {"profile": {}}, "full_mix_neural")
    post_neural = {"apply_air", "expander_depth_db", "sibilant_guard", "pause_floor"} & set(call)
    assert (post_neural, cleanup.call_args.args[0]) == (set(), tmp_path / "d.wav")


def test_the_hiss_stages_sit_between_the_guard_and_the_polish(tmp_path):
    stages = {"sibilant_guard": True, "apply_air": True, "post_neural_suppress": True, "air_gate": True}
    with patch(
        "modules.post_neural_suppress.apply_when_needed", side_effect=lambda _src, wav, *_a, **_k: (Path(str(wav) + ".suppressed"), False)
    ) as hiss:
        result, polish, _guard, _floor = _run(stages, tmp_path)
    assert result.name == "cleaned.wav.guarded.suppressed.polished"
    assert hiss.call_args.args == (tmp_path / "orig.wav", tmp_path / "cleaned.wav.guarded", tmp_path, stages)
    assert polish.call_args.kwargs["apply_air"] is False


def test_hiss_stages_not_opted_in_read_nothing(tmp_path):
    """Keys on, but the caller did not opt in: nothing is read, the polish keeps its shelf."""
    with (
        patch("modules.post_neural_suppress.APL_POST_NEURAL_SUPPRESS", True),
        patch("modules.post_neural_suppress.LINEAR_AIR_GATE", "hiss_dominated"),
        patch("modules.hiss_floor.read", side_effect=AssertionError("read")),
    ):
        result, polish, _guard, _floor = _run({"apply_air": True}, tmp_path)
    assert (result.name, polish.call_args.kwargs["apply_air"]) == ("cleaned.wav.polished", True)


def _level_step(tmp_path, **stages):
    """The cleanup's input after the level step, with the neural stage's pair stood in."""
    pair = (tmp_path / "s.wav", tmp_path / "d.wav")
    with (
        patch("modules.processing._stage_cache.through", return_value=pair),
        patch("modules.processing._post_denoise_cleanup_step", return_value=tmp_path / "c.wav") as cleanup,
        patch("modules.processing._post_neural_stages", return_value=tmp_path / "done.wav"),
        patch("modules.neural_level.APL_NEURAL_LEVEL_RESTORE", True),
        patch("modules.neural_level.restore", return_value=tmp_path / "levelled.wav") as level,
    ):
        processing._denoise_and_polish_full_audio_step(tmp_path / "o.wav", tmp_path, **stages)
    return cleanup.call_args.args[0], level.call_args


def test_the_level_step_runs_before_the_cleanup_only_when_opted_in(tmp_path):
    opted, call = _level_step(tmp_path, neural_level_restore=True)
    assert (opted, call.args) == (tmp_path / "levelled.wav", (tmp_path / "s.wav", tmp_path / "d.wav", tmp_path))
    assert _level_step(tmp_path) == (tmp_path / "d.wav", None)


def _mode_kwargs(mode, tmp_path):
    orig = tmp_path / "orig.wav"
    with (
        patch("modules.processing._resolve_preconditioned_audio", return_value=(orig, {})),
        patch("modules.processing._process_single_track_pipeline") as pipeline,
        patch("modules.processing._denoise_and_polish_full_audio_step", return_value=orig) as denoise,
    ):
        mode().execute(tmp_path / "work", orig, tmp_path / "in.mp4", tmp_path / "out.mp4", 20.0, strategy={})
        pipeline.call_args[0][5](orig, tmp_path / "work")
    return denoise.call_args.kwargs


def test_only_auto_pure_linear_opts_in_to_the_hiss_stages(tmp_path):
    hiss = {"post_neural_suppress", "neural_level_restore", "air_gate", "burst_ceiling"}
    apl = _mode_kwargs(AutoPureLinearMode, tmp_path)
    assert {name: apl[name] for name in hiss} == dict.fromkeys(hiss, True)
    assert not hiss & set(_mode_kwargs(DenoiseOnlyMode, tmp_path))


def test_the_burst_ceiling_sits_between_the_polish_and_the_pause_floor(tmp_path):
    stages = {"pause_floor": True, "burst_ceiling": True}
    with patch("modules.burst_ceiling.apply_when_needed", side_effect=lambda _src, wav, *_a, **_k: Path(str(wav) + ".capped")) as ceiling:
        result, _polish, _guard, floor = _run(stages, tmp_path)
    assert result.name == "cleaned.wav.polished.capped.floored"
    assert ceiling.call_args.args == (tmp_path / "orig.wav", tmp_path / "cleaned.wav.polished", tmp_path, stages)
    assert floor.call_args.args[1] == tmp_path / "cleaned.wav.polished.capped"


def test_the_burst_ceiling_not_opted_in_is_never_called(tmp_path):
    with (
        patch("modules.burst_ceiling.APL_BURST_CEILING", "edges"),
        patch("modules.burst_ceiling.apply_when_needed", side_effect=AssertionError("called")),
    ):
        result, _polish, _guard, _floor = _run({"pause_floor": True}, tmp_path)
    assert result.name == "cleaned.wav.polished.floored"
