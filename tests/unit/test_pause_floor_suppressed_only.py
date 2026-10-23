"""pause_floor_suppressed_only: the configured fill only where the reference suppressor ran, the release fill elsewhere.

The per-frame shelved fill is the suppressor's pause half. On a file whose print the suppressor
refuses the programme keeps its hiss, and the fill tripped every guard ear v4 tripped on the IA
confirmation's 135 such clips (2026-10-21); with the switch on they take the release fill.
"""

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import modules.pause_floor as pause_floor
import modules.processing as processing
from modules import event_log, reference_suppress
from modules.modes import CatharMode
from tests.unit.test_pause_floor import _gated, _on
from tests.unit.test_pause_floor_hissfree import _loaded, _pair, _settings, _tape_voice

SHELVED = {"target": "frame", "shape": "shelf", "hz": 2000.0}


def _keep(tmp_path, suppressed, switch, **settings):
    """The keeper at `settings` with the switch at `switch`; returns the path it hands on."""
    reference, restored = _pair(tmp_path, _gated(_tape_voice()))
    with ExitStack() as stack:
        for context in (*_on(), *_settings(**settings), patch("modules.pause_floor.PAUSE_FLOOR_SUPPRESSED_ONLY", switch)):
            stack.enter_context(context)
        return pause_floor.apply_when_needed(reference, restored, tmp_path / "work", suppressed=suppressed)


def test_the_release_fill_reaches_only_a_file_the_suppressor_did_not_run_on():
    with patch("modules.pause_floor.PAUSE_FLOOR_SUPPRESSED_ONLY", True), patch("modules.pause_floor.PAUSE_FLOOR_TARGET", "frame"):
        assert (pause_floor.profile_for(False), pause_floor.profile_for(True).target) == (pause_floor.release_profile(), "frame")
    with patch("modules.pause_floor.PAUSE_FLOOR_SUPPRESSED_ONLY", False), patch("modules.pause_floor.PAUSE_FLOOR_TARGET", "frame"):
        assert pause_floor.profile_for(False).target == "frame"


def test_an_unsuppressed_file_gets_the_release_fill_sample_for_sample(tmp_path):
    released = _keep(tmp_path / "released", False, True, **SHELVED)
    default = _keep(tmp_path / "default", False, False)
    assert released.name == default.name == "paused_12db_p15_res.wav"
    assert np.array_equal(sf.read(str(released))[0], sf.read(str(default))[0])


def test_a_suppressed_file_keeps_the_configured_fill(tmp_path):
    kept = _keep(tmp_path / "kept", True, True, **SHELVED)
    assert kept.name == "paused_12db_p15_frame_shelf2000c18_res.wav"


def test_the_record_says_when_the_release_fill_stood_in(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RESTORE_EVENT_LOG", str(tmp_path / "events"))
    _keep(tmp_path, False, True, **SHELVED)
    record = json.loads((tmp_path / "events" / event_log.record_name(pause_floor.STAGE, Path("res.wav"))).read_text(encoding="utf-8"))
    assert (record["target"], record["shape"], record["release_fill"]) == ("cluster", "flat", True)


def test_the_suppressor_s_output_is_told_from_its_input(tmp_path):
    settings = reference_suppress.CATHAR_SETTINGS
    assert reference_suppress.is_output(reference_suppress.target_for(tmp_path / "x.wav", tmp_path, settings))
    assert not reference_suppress.is_output(tmp_path / "x.wav")
    assert not reference_suppress.is_output(tmp_path / reference_suppress.STAGE / "x.wav")


def _floor_calls(tmp_path, suppressor_output):
    """The pause floor's keyword arguments after the post-neural stages, the suppressor handing back `suppressor_output`."""
    with (
        patch("modules.post_neural_suppress.apply_when_needed", return_value=(suppressor_output, False)),
        patch("modules.processing._polish_full_audio_step", side_effect=lambda wav, *a, **k: Path(str(wav) + ".polished")),
        patch("modules.pause_floor.apply_when_needed", side_effect=lambda ref, wav, *a, **k: wav) as floor,
    ):
        processing._post_neural_stages(
            tmp_path / "o.wav", tmp_path / "s.wav", tmp_path / "c.wav", tmp_path, 10.0, None, {"pause_floor": True}
        )
    return floor.call_args.kwargs


def test_auto_pure_linear_tells_the_pause_floor_whether_the_suppressor_ran(tmp_path):
    referenced = reference_suppress.target_for(tmp_path / "c.wav", tmp_path, reference_suppress.CATHAR_SETTINGS)
    assert _floor_calls(tmp_path, referenced)["suppressed"] is True
    assert _floor_calls(tmp_path, tmp_path / "c.wav")["suppressed"] is False


def test_cathar_tells_the_pause_floor_whether_the_suppressor_ran(tmp_path):
    orig = tmp_path / "orig.wav"
    seen = []
    strategy = {"precondition_filters": {"highpass_hz": 80}, "profile": {"tonal_persistence": 0.001}}
    referenced = reference_suppress.target_for(orig, tmp_path, reference_suppress.CATHAR_SETTINGS)
    for handed in (referenced, orig):
        with (
            patch("modules.processing._resolve_preconditioned_audio", side_effect=lambda _w, _o, _d, _m, strategy: (orig, strategy)),
            patch("modules.processing._process_single_track_pipeline") as pipeline,
            patch("modules.cathar.filter_cathar_vhs_pipeline", return_value=orig),
            patch("modules.modes.cathar.reference_suppressed", return_value=handed),
            patch("modules.processing._polish_full_audio_step", side_effect=lambda w, *_a, **_k: w),
            patch("modules.pause_floor.apply_when_needed", side_effect=lambda _i, w, *_a, **k: seen.append(k["suppressed"]) or w),
        ):
            CatharMode().execute(tmp_path / "work", orig, tmp_path / "in.mp4", tmp_path / "out.mp4", 20.0, strategy=strategy)
            pipeline.call_args[0][5](orig, tmp_path / "work")
    assert seen == [True, False]


def test_the_switch_and_the_hiss_shelf_are_settings_the_config_accepts():
    conf, warn = _loaded({"pause_floor_suppressed_only": True, "pause_floor_shape": " Hiss_Shelf "})
    assert (conf["pause_floor_suppressed_only"], conf["pause_floor_shape"]) == (True, "hiss_shelf")
    warn.assert_not_called()
    conf, _warn = _loaded({"process_mode": "auto_pure_linear"})
    assert conf["pause_floor_suppressed_only"] is False
