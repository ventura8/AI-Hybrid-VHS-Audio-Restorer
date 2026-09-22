"""The stages after the neural denoise: sibilant guard, polish, pause floor; and which modes opt in."""

from pathlib import Path
from unittest.mock import patch

import modules.processing as processing


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
    assert post.call_args.args[-1] == {"sibilant_guard": True, "pause_floor": True, "apply_air": False}
