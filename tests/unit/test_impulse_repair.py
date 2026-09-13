"""Tests for the pop-removal stage.

What has to hold: pops are found where they are and not where speech merely starts, a found
pop is refilled with something close to what was under it, an undamaged signal comes back
essentially untouched, and the WAV wrapper handles channels, a zero threshold and a file it
cannot read. The stage earned its place on the calibrated crackle class, where it repairs
+5.5 dB inside the pops against cathar's decrackle at +1.7.
"""

import numpy as np
import pytest
import soundfile as sf

from modules import impulse_repair

RATE = 44100


def _voiced(seconds=2.0, seed=4):
    """A harmonic voice with a slow envelope: predictable, the way a short AR model likes it."""
    rng = np.random.default_rng(seed)
    time = np.arange(int(RATE * seconds)) / RATE
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * time)
    voiced = sum(np.sin(2 * np.pi * 140.0 * h * time + rng.uniform(0, 6.28)) / h for h in range(1, 10))
    return (0.2 * envelope * voiced + 0.003 * rng.normal(0.0, 1.0, len(time))).astype(np.float32)


def _with_pops(audio, positions, amplitude=0.4, width=3):
    popped = audio.copy()
    for position in positions:
        end = position + width
        popped[position:end] += amplitude
    return popped


def test_pops_are_detected_where_they_are():
    audio = _voiced()
    positions = [RATE // 4, RATE // 2, RATE]
    mask = impulse_repair.detect_pops(_with_pops(audio, positions))
    for position in positions:
        end = position + 3
        assert mask[position:end].any()


def test_a_clean_voice_raises_almost_nothing():
    """A voice's own onsets are not pops: the residual test must stay quiet on undamaged audio."""
    mask = impulse_repair.detect_pops(_voiced())
    assert mask.mean() < 0.002


def test_a_refilled_pop_is_close_to_what_was_under_it():
    audio = _voiced()
    positions = [RATE // 3, (2 * RATE) // 3]
    repaired, count = impulse_repair.remove_pops(_with_pops(audio, positions))
    assert count >= 2
    for position in positions:
        window = slice(position - 8, position + 12)
        before = np.sqrt(np.mean((_with_pops(audio, positions)[window] - audio[window]) ** 2))
        after = np.sqrt(np.mean((repaired[window] - audio[window]) ** 2))
        assert after < before / 4


def test_undamaged_audio_comes_back_essentially_untouched():
    audio = _voiced()
    repaired, _count = impulse_repair.remove_pops(audio)
    change = np.sqrt(np.mean((repaired - audio) ** 2)) / np.sqrt(np.mean(audio**2))
    assert 20 * np.log10(change + 1e-12) < -40.0


def test_spans_are_merged_and_long_ones_left_alone():
    assert impulse_repair._merged([(10, 12), (15, 18), (40, 41)], 8) == [(10, 18), (40, 41)]
    audio = _voiced()
    damaged = audio.copy()
    start, end = RATE // 2, RATE // 2 + 2000
    damaged[start:end] += 0.4  # far longer than a pop: the dropout stage's business
    repaired, _count = impulse_repair.remove_pops(damaged)
    inner = slice(start + 100, end - 100)
    assert np.allclose(repaired[inner], damaged[inner])


def test_a_cluster_of_transients_is_left_alone():
    """A drum hit's onset sits among other outliers; a pop stands alone. Only the pop is refilled."""
    audio = _voiced()
    hits = audio.copy()
    for offset in range(0, 1200, 60):  # a burst of outliers over 27 ms: a ringing hit, not a pop
        onset = RATE // 2 + offset
        end = onset + 2
        hits[onset:end] += 0.4
    hits = _with_pops(hits, [RATE // 4])
    repaired, _count = impulse_repair.remove_pops(hits)
    burst = slice(RATE // 2, RATE // 2 + 1300)
    pop = slice(RATE // 4, RATE // 4 + 3)
    assert np.allclose(repaired[burst], hits[burst])
    assert np.max(np.abs(repaired[pop] - audio[pop])) < 0.1


def test_interpolation_refuses_a_span_at_the_very_edge():
    """A span without a model's worth of context on both sides cannot be refilled."""
    samples = _voiced(0.2).astype(np.float64)
    assert impulse_repair._interpolate(samples, 1, 6, np.array([1.0, -0.5, 0.1])) is False
    assert impulse_repair._interpolate(samples, len(samples) - 4, len(samples), np.array([1.0, -0.5, 0.1])) is False
    assert impulse_repair._repair_span(samples[:40], 20, 24) is False


def test_the_wav_stage_repairs_every_channel(tmp_path):
    audio = _voiced()
    stereo = np.column_stack((_with_pops(audio, [RATE // 2]), _with_pops(audio, [RATE // 4])))
    source = tmp_path / "in.wav"
    sf.write(str(source), stereo, RATE, subtype="FLOAT")
    produced = impulse_repair.depop(source, tmp_path / "out")
    assert produced is not None and produced.name == "depopped_in.wav"
    repaired, _rate = sf.read(str(produced), dtype="float32", always_2d=True)
    assert repaired.shape == stereo.shape
    for channel, position in ((0, RATE // 2), (1, RATE // 4)):
        end = position + 3
        assert np.max(np.abs(repaired[position:end, channel] - audio[position:end])) < 0.1


def test_a_zero_threshold_switches_the_stage_off(tmp_path):
    assert impulse_repair.depop(tmp_path / "in.wav", tmp_path, threshold=0.0) is None


def test_an_unreadable_file_yields_nothing(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_text("not audio")
    assert impulse_repair.depop(bad, tmp_path / "out") is None


@pytest.mark.parametrize("seconds", [0.001, 0.01])
def test_audio_too_short_for_a_model_is_left_alone(seconds):
    audio = _voiced(seconds)
    repaired, count = impulse_repair.remove_pops(audio)
    assert count == 0 and np.allclose(repaired, audio)


def test_the_stage_is_wired_in_ahead_of_decrackle():
    from modules import physical_repair

    steps = physical_repair._steps()
    assert list(steps)[:2] == ["depop", "decrackle"]
    assert steps["depop"].keywords == {"threshold": physical_repair.APL_DEPOP_THRESHOLD}
    assert steps["depop"].func is impulse_repair.depop


def test_a_capture_the_host_cannot_hold_is_left_unrepaired(tmp_path, monkeypatch):
    """Running out of memory inside the stage skips it rather than ending the restoration."""
    source = tmp_path / "in.wav"
    sf.write(str(source), _voiced(0.5), RATE, subtype="FLOAT")
    monkeypatch.setattr(impulse_repair, "remove_pops", lambda *_a, **_k: (_ for _ in ()).throw(MemoryError()))
    assert impulse_repair.depop(source, tmp_path / "out") is None


def test_an_empty_file_yields_an_empty_repair(tmp_path):
    """A zero-frame WAV is valid audio with nothing in it; the stage must not index into it."""
    assert impulse_repair._spans(np.zeros(0, dtype=bool)) == []
    repaired, count = impulse_repair.remove_pops(np.zeros(0, dtype=np.float32))
    assert count == 0 and repaired.shape == (0,)
    source = tmp_path / "empty.wav"
    sf.write(str(source), np.zeros((0, 1), dtype=np.float32), RATE, subtype="FLOAT")
    produced = impulse_repair.depop(source, tmp_path / "out")
    assert produced is not None and sf.info(str(produced)).frames == 0


def test_without_soundfile_the_module_imports_and_the_stage_yields_nothing(monkeypatch, tmp_path):
    """The optional import is guarded: without soundfile the module loads and depop() is None."""
    import importlib
    import sys

    library = sys.modules["soundfile"]
    monkeypatch.setitem(sys.modules, "soundfile", None)
    reloaded = importlib.reload(impulse_repair)
    try:
        assert reloaded.sf is None
        assert reloaded.depop(tmp_path / "in.wav", tmp_path / "out") is None
    finally:
        monkeypatch.setitem(sys.modules, "soundfile", library)
        importlib.reload(impulse_repair)
    assert impulse_repair.sf is not None
