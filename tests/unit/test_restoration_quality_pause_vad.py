"""The true-pause VAD: the DSP default, the optional silero-vad hook, and the run bookkeeping."""

import importlib
import importlib.machinery
import sys
import types

import numpy as np
import pytest
import scipy.signal

from scripts.restoration_quality import pause_metrics

RATE = 44100
FRAME = int(pause_metrics.FRAME_S * RATE)


def test_the_dsp_vad_hears_a_fricative_that_leaves_the_broadband_level_alone():
    """A 100 ms burst above 5 kHz under a loud hum: the top band hears it although the broadband level moves 0.1 dB."""
    rng = np.random.default_rng(2)
    t = np.arange(4 * RATE) / RATE
    burst = ((t % 1.0) >= 0.5) & ((t % 1.0) < 0.6)
    fricative = 1e-2 * scipy.signal.sosfiltfilt(
        scipy.signal.butter(8, 5000.0, "highpass", fs=RATE, output="sos"), rng.standard_normal(len(t))
    )
    speech = pause_metrics.dsp_vad(
        0.1 * np.sin(2 * np.pi * 150.0 * t) + 1e-3 * rng.standard_normal(len(t)) + fricative * burst, RATE, FRAME
    )
    phase = (np.arange(len(speech)) * pause_metrics.FRAME_S) % 1.0
    assert speech[(phase >= 0.52) & (phase < 0.58)].all()
    assert not speech[phase >= 0.8].any()


def test_the_dsp_vad_runs_at_a_rate_whose_top_band_is_past_nyquist():
    """At 8 kHz the 5-12 kHz band is empty and simply hears nothing."""
    rng = np.random.default_rng(4)
    mono = 1e-3 * rng.standard_normal(4 * 8000)
    mono[8000:12000] += 0.1 * np.sin(2 * np.pi * 300.0 * np.arange(4000) / 8000)
    speech = pause_metrics.dsp_vad(mono, 8000, 160)
    assert len(speech) == 200
    assert speech[50:75].all()
    assert not speech[120:].any()


def test_long_runs_keep_only_runs_of_the_minimum_length():
    """Runs shorter than the minimum are dropped whole; longer ones are kept whole."""
    mask = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
    assert pause_metrics.run_spans(mask) == [(0, 2), (3, 6), (8, 9)]
    assert list(pause_metrics.long_runs(mask, 3)) == [False, False, False, True, True, True, False, False, False]


def _fake_silero(stamps, calls):
    """A stand-in `silero_vad` module that records its calls and returns fixed speech timestamps."""
    module = types.ModuleType(pause_metrics.SILERO_MODULE)
    module.__spec__ = importlib.machinery.ModuleSpec(pause_metrics.SILERO_MODULE, None)
    module.load_silero_vad = lambda: "model"

    def get_speech_timestamps(audio, model, sampling_rate):
        calls.append((len(audio), model, sampling_rate))
        return stamps

    module.get_speech_timestamps = get_speech_timestamps
    return module


def test_the_vad_is_the_dsp_one_unless_silero_is_asked_for(monkeypatch):
    """No name and no environment, or an unknown name: the DSP VAD."""
    monkeypatch.delenv(pause_metrics.VAD_ENV, raising=False)
    assert pause_metrics.resolve_vad() is pause_metrics.dsp_vad
    assert pause_metrics.resolve_vad("webrtc") is pause_metrics.dsp_vad


def test_silero_falls_back_to_the_dsp_vad_when_not_installed(monkeypatch):
    """Asked for by name or by the environment, silero-vad still needs the package; without it the DSP VAD runs, with a warning."""
    absent = types.SimpleNamespace(util=types.SimpleNamespace(find_spec=lambda _name: None), import_module=importlib.import_module)
    monkeypatch.setattr(pause_metrics, "importlib", absent)
    monkeypatch.setenv(pause_metrics.VAD_ENV, "silero")
    with pytest.warns(RuntimeWarning, match="silero-vad"):
        assert pause_metrics.resolve_vad() is pause_metrics.dsp_vad
    with pytest.warns(RuntimeWarning, match="not installed"):
        assert pause_metrics.resolved_vad_name("Silero") == "dsp"


def test_the_silero_hook_runs_at_16_khz_and_maps_its_timestamps_to_frames(monkeypatch):
    """With the package importable and the environment asking for it, silero's speech becomes 20 ms frames."""
    calls = []
    monkeypatch.setitem(sys.modules, pause_metrics.SILERO_MODULE, _fake_silero([{"start": 8000, "end": 16000}], calls))
    monkeypatch.setenv(pause_metrics.VAD_ENV, "silero")
    pause_metrics.silero_model.cache_clear()
    vad = pause_metrics.resolve_vad()
    speech = vad(np.zeros(2 * RATE, dtype=np.float32), RATE, FRAME)
    pause_metrics.silero_model.cache_clear()
    assert vad is pause_metrics.silero_vad
    assert pause_metrics.resolved_vad_name() == "silero"
    assert calls == [(2 * pause_metrics.SILERO_RATE, "model", pause_metrics.SILERO_RATE)]
    assert list(np.flatnonzero(speech)) == list(range(25, 50))


def _short_vad(mono, _rate, frame):
    """A VAD that hears no speech and comes back five frames short (a timestamp rounded down)."""
    return np.zeros(len(mono) // frame - 5, dtype=bool)


def test_a_vad_that_comes_back_short_is_padded_as_speech():
    """The missing frames count as speech, the side that keeps them out of the mask."""
    speech = pause_metrics.vad_frames(_short_vad, (np.zeros(100 * FRAME), RATE, FRAME), 100)
    assert list(np.flatnonzero(speech)) == list(range(95, 100))


def test_the_mask_survives_a_vad_that_comes_back_short():
    """Tone bursts over hiss with a short VAD: no broadcast error, and the padded tail stays out of the runs."""
    rng = np.random.default_rng(6)
    t = np.arange(6 * RATE) / RATE
    source = 0.1 * np.sin(2 * np.pi * 200.0 * t) * ((t % 1.0) < 0.5) + 1e-3 * rng.standard_normal(len(t))
    mask, runs = pause_metrics.true_pause_mask(source, RATE, _short_vad)
    assert len(mask) == len(runs) == len(t) // FRAME
    assert mask.any()
    assert not runs[-5:].any()
