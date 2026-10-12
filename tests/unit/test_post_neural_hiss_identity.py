"""The post-neural hiss stages keep today's bytes by default, and the stage cache replays when only they change.

The real step, guard, polish and pause floor run on the stage-cache bench's fixture
(`tests/unit/test_stage_cache_replay.py`): the chain and the neural model are counting
stand-ins and ffmpeg's polish is a gain keyed by its filter string. The APL mode's call with
the new opt-ins must render the same samples at the same path as the call without them while
the keys sit at their defaults; with the keys on, the render changes. cathar never reaches
the new modules.
"""

import re
import shutil
from pathlib import Path

import pytest
import soundfile as sf

from modules import config, neural_level, post_neural_suppress, processing, stage_cache, stage_cache_key, utils
from tests.unit.test_stage_cache import hold_the_host_still
from tests.unit.test_stage_cache_reach import _Graph, _sources
from tests.unit.test_stage_cache_replay import INPUT_NAME, RATE, SECONDS, STRATEGY, Fakes, _polish_as_gain, _programme

APL_OPT_INS = {"post_neural_suppress": True, "neural_level_restore": True, "air_gate": True}
NEW_FOLDERS = {"post_neural_suppress", "neural_level"}


@pytest.fixture(name="bench")
def bench_fixture(tmp_path, monkeypatch):
    fakes = Fakes()
    monkeypatch.setattr(utils, "LOG_FILE", tmp_path / "session_log.txt")
    monkeypatch.setattr(processing._apl_chain, "stage_plan", fakes.stage_plan)
    monkeypatch.setattr(processing, "_denoise_full_audio_step", fakes.denoise)
    monkeypatch.setattr(processing, "_run_dsp_filter_file", _polish_as_gain)
    monkeypatch.setattr(stage_cache_key, "binaries", lambda: {"ffmpeg": {"absent": True}, "cathar": {"absent": True}})
    monkeypatch.setattr(stage_cache_key, "torch", None)
    monkeypatch.setattr(stage_cache_key, "gpu_driver", lambda: None)
    monkeypatch.delenv("AI_RESTORE_EVENT_LOG", raising=False)
    monkeypatch.delenv(stage_cache.ENV_VAR, raising=False)
    hold_the_host_still(monkeypatch)
    source = tmp_path / "source.wav"
    sf.write(str(source), _programme(), RATE, subtype="FLOAT")
    return fakes, source, tmp_path


def _render(bench, **opt_ins):
    """One APL render in a fresh work folder: the final WAV relative to its stage folder, its samples' hash, its folders."""
    _fakes, source, base = bench
    work = base / f"run{len(list(base.glob('run*')))}"
    audio_dir = work / "denoised_preconditioned_audio"
    audio_dir.mkdir(parents=True)
    input_wav = Path(shutil.copy2(source, work / INPUT_NAME))
    final = processing._denoise_and_polish_full_audio_step(
        input_wav,
        audio_dir,
        total_duration=SECONDS,
        strategy=STRATEGY,
        spectral_denoise=True,
        physical_repair=True,
        hum_cancel=True,
        plosive_tamer=True,
        tone_cancel=True,
        apply_air=True,
        expander_depth_db=12.0,
        sibilant_guard=True,
        pause_floor=True,
        **opt_ins,
    )
    # The neural stage's folder carries a hash of the work folder's path, one per run.
    folders = {re.sub(r"_[0-9a-f]{12}$", "", path.name) for path in audio_dir.iterdir() if path.is_dir()}
    return final.relative_to(audio_dir), stage_cache_key.input_identity(final)["pcm_sha256"], folders


def _keys_on(monkeypatch):
    monkeypatch.setattr(post_neural_suppress, "APL_POST_NEURAL_SUPPRESS", True)
    monkeypatch.setattr(post_neural_suppress, "LINEAR_AIR_GATE", "hiss_dominated")
    monkeypatch.setattr(neural_level, "APL_NEURAL_LEVEL_RESTORE", True)
    for key, value in (("apl_post_neural_suppress", True), ("linear_air_gate", "hiss_dominated"), ("apl_neural_level_restore", True)):
        monkeypatch.setitem(config.CONFIG, key, value)
    monkeypatch.setattr(post_neural_suppress, "decide", lambda _source, _restored: {"open": True, "why": "forced", "reading": {}})


def test_the_defaults_render_today_s_samples_at_today_s_path_with_the_opt_ins(bench):
    today = _render(bench)
    opted = _render(bench, **APL_OPT_INS)
    assert opted == today
    assert not NEW_FOLDERS & today[2]


def test_with_the_keys_on_the_render_changes(bench, monkeypatch):
    today = _render(bench, **APL_OPT_INS)
    _keys_on(monkeypatch)
    on = _render(bench, **APL_OPT_INS)
    assert on[1] != today[1]
    assert "post_neural_suppress" in on[2]


def test_a_hiss_key_change_replays_the_cached_neural_stage(bench, monkeypatch):
    monkeypatch.setenv(stage_cache.ENV_VAR, str(bench[2] / "cache"))
    before = _render(bench, **APL_OPT_INS)
    _keys_on(monkeypatch)
    hit = _render(bench, **APL_OPT_INS)
    assert (bench[0].chain, bench[0].neural, hit[1] != before[1]) == (1, 1, True)


def test_cathar_never_reaches_the_hiss_stages():
    reached = _Graph(_sources()).reachable([("modes/cathar.py", "CatharMode")])
    assert not {module for module, _function in reached} & {"hiss_floor.py", "post_neural_suppress.py", "neural_level.py"}
