"""The burst ceiling keeps today's bytes by default, changes the render when on, and a change of its keys replays the cached neural stage.

The real step, guard, polish and pause floor run on the stage-cache bench's fixture
(`tests/unit/test_stage_cache_replay.py`), as in `tests/unit/test_post_neural_hiss_identity.py`.
The APL mode's call with the opt-in must render the same samples at the same path as the call
without it while the key sits at 'none'. cathar never reaches the new module.
"""

import numpy as np
import pytest
import soundfile as sf

from modules import burst_ceiling, config, stage_cache
from tests.unit.test_post_neural_hiss_identity import _render, make_bench
from tests.unit.test_stage_cache_reach import _Graph, _sources

OPT_IN = {"burst_ceiling": True}
FOLDER = "burst_ceiling"


@pytest.fixture(name="bench")
def bench_fixture(tmp_path, monkeypatch):
    return make_bench(tmp_path, monkeypatch)


def _constant_plan(_reference, restored, _scope, _depth):
    """A plan that lowers every 3-16 kHz bin of `restored` by 12 dB, whatever the source holds."""
    info = sf.info(str(restored))
    stft = burst_ceiling.stft_for(info.samplerate)
    blocks = -(-info.frames // burst_ceiling.block_samples(stft))
    bands = len(stft.edges) - 1
    return burst_ceiling.Plan(np.full((blocks, bands), -12.0), stft, info.samplerate, np.ones(bands, dtype=bool), np.zeros(bands), 0.0)


def _key_on(monkeypatch):
    """The scope on and a constant plan; `CONFIG` patched too, since the stage cache's key reads it, not the module copy."""
    monkeypatch.setattr(burst_ceiling, "APL_BURST_CEILING", "edges")
    monkeypatch.setattr(burst_ceiling, "APL_BURST_CEILING_DEPTH_DB", 24.0)
    monkeypatch.setitem(config.CONFIG, "apl_burst_ceiling", "edges")
    monkeypatch.setitem(config.CONFIG, "apl_burst_ceiling_depth_db", 24.0)
    monkeypatch.setattr(burst_ceiling, "plan", _constant_plan)


def test_the_default_renders_today_s_samples_at_today_s_path_with_the_opt_in(bench):
    today = _render(bench)
    opted = _render(bench, **OPT_IN)
    assert opted == today
    assert FOLDER not in today[2]


def test_with_the_key_on_the_render_changes(bench, monkeypatch):
    today = _render(bench, **OPT_IN)
    _key_on(monkeypatch)
    on = _render(bench, **OPT_IN)
    assert (on[1] != today[1], FOLDER in on[2]) == (True, True)


def test_a_ceiling_key_change_replays_the_cached_neural_stage(bench, monkeypatch):
    monkeypatch.setenv(stage_cache.ENV_VAR, str(bench[2] / "cache"))
    before = _render(bench, **OPT_IN)
    _key_on(monkeypatch)
    hit = _render(bench, **OPT_IN)
    assert (bench[0].chain, bench[0].neural, hit[1] != before[1]) == (1, 1, True)


def test_cathar_never_reaches_the_ceiling():
    reached = _Graph(_sources()).reachable([("modes/cathar.py", "CatharMode")])
    assert "burst_ceiling.py" not in {module for module, _function in reached}
