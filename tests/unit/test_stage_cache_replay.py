"""A stage-cache hit renders the same samples as a fresh render, through the real step, guard, polish and pause floor.

The chain and the neural model are counting stand-ins and ffmpeg's polish is a gain keyed by
its filter string; everything between them is the real code: the step, the cache and its
key, the neural-stage choice and folders, the sibilant guard and the pause floor keeper.
Each run copies the input into a work folder of its own, as `process_hybrid_audio` does.
"""

import functools
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

from modules import config, processing, sibilant_guard, stage_cache, stage_cache_key, utils
from tests.unit.test_stage_cache import hold_the_host_still

RATE = 44100
SECONDS = 4.0
VOWEL_S, GAP_S, SIB_S, SIB_OFFSET_S = 0.3, 0.3, 0.12, 0.09
VOWELS = [(int(i * (VOWEL_S + GAP_S) * RATE), int((i * (VOWEL_S + GAP_S) + VOWEL_S) * RATE)) for i in range(5)]
SIBILANTS = [(stop + int(SIB_OFFSET_S * RATE), stop + int((SIB_OFFSET_S + SIB_S) * RATE)) for _start, stop in VOWELS]
STRATEGY = {"profile": {"noise_floor_db": -50, "tonal_persistence": 0}}
INPUT_NAME = "preconditioned_fixture_audio.wav"


def _gate(spans, length, ramp_ms=10.0):
    gate = np.zeros(length)
    for start, stop in spans:
        gate[start:stop] = 1.0
    ramp = int(ramp_ms * RATE / 1000.0)
    return np.convolve(gate, np.hanning(ramp) / np.hanning(ramp).sum(), mode="same")


def _band_noise(band, seed, length):
    sos = scipy.signal.butter(4, band, btype="band", fs=RATE, output="sos")
    return scipy.signal.sosfilt(sos, np.random.default_rng(seed).normal(0.0, 1.0, length))


@functools.lru_cache(maxsize=None)
def _programme():
    """Five vowel bursts, a 5-9 kHz fricative after each, hiss in the pauses and a second of it at the end."""
    length = int(RATE * SECONDS)
    t = np.arange(length) / RATE
    vowel = sum(np.sin(2 * np.pi * 160.0 * k * t) / k for k in range(1, 31))
    vowel *= 0.1 / np.sqrt(np.mean(vowel**2)) * _gate(VOWELS, length)
    sib_gate = _gate(SIBILANTS, length)
    fricative = (0.03 * _band_noise((1000.0, 4000.0), 2, length) + 0.05 * _band_noise((5000.0, 9000.0), 3, length)) * sib_gate
    hiss = np.random.default_rng(4).normal(0.0, 3e-3, length)
    mono = vowel + fricative + hiss
    return np.stack([mono, mono], axis=1).astype(np.float32)


class Fakes:
    """The chain stage and the neural model, counting their runs."""

    def __init__(self):
        self.chain = 0
        self.neural = 0

    def stage_plan(self, audio_dir, *_args, **_kwargs):
        return (("hum_cancel", True, lambda wav: self._chain(wav, audio_dir)),)

    def _chain(self, wav, audio_dir):
        self.chain += 1
        data, rate = sf.read(str(wav), dtype="float32", always_2d=True)
        target = Path(audio_dir) / "hum_cancel" / f"humcancel_{Path(wav).name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(target), data * np.float32(0.9), rate, subtype="FLOAT")
        return target

    def denoise(self, surgical_wav, denoise_sub_dir, denoise_model=None):
        """A low-pass and a gate: the fricatives lose their top, the pauses drop 20 dB."""
        del denoise_model
        self.neural += 1
        data, rate = sf.read(str(surgical_wav), dtype="float32", always_2d=True)
        smooth = np.stack([np.convolve(data[:, ch], np.ones(8) / 8.0, mode="same") for ch in range(data.shape[1])], axis=1)
        level = np.sqrt(np.convolve(np.mean(smooth**2, axis=1), np.ones(441) / 441.0, mode="same"))
        gated = smooth * np.where(level > 0.01, 1.0, 0.1)[:, None]
        target = Path(denoise_sub_dir) / f"{Path(surgical_wav).stem}_(No Noise)_fake.wav"
        sf.write(str(target), gated.astype(np.float32), rate, subtype="FLOAT")
        return target


def _polish_as_gain(input_wav, output_wav, filter_expr, desc, total_duration):
    """ffmpeg's stand-in: a gain the filter string decides, so a different polish is a different output."""
    del desc, total_duration
    gain = 0.5 + int(hashlib.sha256(filter_expr.encode("utf-8")).hexdigest()[:4], 16) / 65535.0 * 0.5
    data, rate = sf.read(str(input_wav), dtype="float32", always_2d=True)
    sf.write(str(output_wav), data * np.float32(gain), rate, subtype="FLOAT")
    return Path(output_wav)


@pytest.fixture(name="bench")
def bench_fixture(tmp_path, monkeypatch):
    """The stand-ins in place, the key's outside world held still, and a source input to copy."""
    fakes = Fakes()
    monkeypatch.setattr(utils, "LOG_FILE", tmp_path / "session_log.txt")
    monkeypatch.setattr(processing._apl_chain, "stage_plan", fakes.stage_plan)
    monkeypatch.setattr(processing, "_denoise_full_audio_step", fakes.denoise)
    monkeypatch.setattr(processing, "_run_dsp_filter_file", _polish_as_gain)
    monkeypatch.setattr(stage_cache_key, "binaries", lambda: {"ffmpeg": {"absent": True}, "cathar": {"absent": True}})
    monkeypatch.setattr(stage_cache_key, "torch", None)
    monkeypatch.setattr(stage_cache_key, "gpu_driver", lambda: None)
    monkeypatch.delenv("AI_RESTORE_EVENT_LOG", raising=False)
    hold_the_host_still(monkeypatch)
    source = tmp_path / "source.wav"
    sf.write(str(source), _programme(), RATE, subtype="FLOAT")
    return fakes, source, tmp_path


def _render(bench, monkeypatch, cache_on):
    """One render in a fresh work folder; returns the final WAV relative to its stage folder and its samples' hash."""
    _fakes, source, base = bench
    if cache_on:
        monkeypatch.setenv(stage_cache.ENV_VAR, str(base / "cache"))
    else:
        monkeypatch.delenv(stage_cache.ENV_VAR, raising=False)
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
    )
    return final.relative_to(audio_dir), stage_cache_key.input_identity(final)["pcm_sha256"]


def _counts(bench):
    """How often the chain stand-in and the neural stand-in ran, and how many entries the cache holds."""
    layout = bench[2] / "cache" / stage_cache.LAYOUT
    return bench[0].chain, bench[0].neural, len(list(layout.iterdir())) if layout.exists() else 0


def test_a_hit_renders_the_same_samples_as_the_miss_and_as_a_fresh_render(bench, monkeypatch):
    miss = _render(bench, monkeypatch, cache_on=True)
    after_miss = _counts(bench)
    hit = _render(bench, monkeypatch, cache_on=True)
    after_hit = _counts(bench)
    fresh = _render(bench, monkeypatch, cache_on=False)
    assert (after_miss, after_hit, _counts(bench)) == ((1, 1, 1), (1, 1, 1), (2, 2, 1))
    assert (hit, fresh[1]) == (miss, miss[1])
    # The replayed references are the ones the guard and the floor read: both wrote their stage.
    assert (miss[0].parts[0], "guarded_" in miss[0].name) == ("pause_floor", True)


def test_a_post_neural_change_replays_and_renders_what_its_fresh_render_does(bench, monkeypatch):
    """A sibilant-guard mix the loop moves: the hit must equal a render without the cache under the same setting."""
    before = _render(bench, monkeypatch, cache_on=True)[1]
    monkeypatch.setattr(sibilant_guard, "APL_SIBILANT_MIX", 0.5)
    monkeypatch.setitem(config.CONFIG, "apl_sibilant_mix", 0.5)
    hit = _render(bench, monkeypatch, cache_on=True)[1]
    after_hit = _counts(bench)
    fresh = _render(bench, monkeypatch, cache_on=False)[1]
    assert (after_hit, hit != before, fresh) == ((1, 1, 1), True, hit)


def test_a_pre_neural_change_misses_and_stores_a_second_entry(bench, monkeypatch):
    _render(bench, monkeypatch, cache_on=True)
    monkeypatch.setitem(config.CONFIG, "apl_tonal_flatness_max", 0.01)
    _render(bench, monkeypatch, cache_on=True)
    assert _counts(bench) == (2, 2, 2)
