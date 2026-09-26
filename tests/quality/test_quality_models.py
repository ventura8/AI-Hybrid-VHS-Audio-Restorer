"""Every model-backed scorer loads from the model store and returns a finite reading on a short clip (SCOREQ and MERT too)."""

import math

import numpy as np
import pytest

from scripts.restoration_quality import mos_models, runner, speech_models

RATE = 16000


def _speechlike(seconds=10.0):
    rng = np.random.default_rng(1)
    t = np.arange(int(seconds * RATE)) / RATE
    voice = sum(np.sin(2 * np.pi * 140.0 * k * t + k) / k for k in range(1, 20))
    gate = ((t % 0.8) < 0.45).astype(np.float64)
    return (0.1 * voice * gate + 1e-3 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture(scope="module")
def registry():
    return runner.ModelRegistry(device="cuda")


def test_sigmos_reads_seven_dimensions(registry):
    estimator = registry.get("sigmos", mos_models.load_sigmos)
    audio48 = np.repeat(_speechlike(), 3)
    scores = mos_models.sigmos_scores(estimator, audio48)
    assert set(scores) == {f"mos.sigmos_{k}" for k in ("col", "disc", "loud", "noise", "reverb", "sig", "ovrl")}
    assert all(math.isfinite(v) for v in scores.values())


def test_dnsmos_reads_four_scales(registry):
    sessions = registry.get("dnsmos", mos_models.load_dnsmos)
    scores = mos_models.dnsmos_scores(sessions, _speechlike())
    assert set(scores) == {"mos.dnsmos_sig", "mos.dnsmos_bak", "mos.dnsmos_ovrl", "mos.dnsmos_p808"}


def test_audiobox_reads_four_axes(registry):
    predictor = registry.get("audiobox", mos_models.load_audiobox)
    scores = mos_models.audiobox_scores(predictor, _speechlike())
    assert set(scores) == {"mos.audiobox_pq", "mos.audiobox_pc", "mos.audiobox_ce", "mos.audiobox_cu"}
    registry.release()


def test_whisper_transcribes_and_reports_confidence(registry):
    model, processor = registry.get("whisper", speech_models.load_whisper)
    whisper = speech_models.Whisper(model, processor, "ro")
    record = whisper.transcribe(_speechlike())
    assert set(record) == {"text", "avg_logprob", "no_speech_prob"}
    assert 0.0 <= record["no_speech_prob"] <= 1.0
    registry.release()


def test_wavlm_embeddings_are_unit_vectors_and_self_similar(registry):
    model, extractor = registry.get("wavlm", speech_models.load_wavlm)
    audio = _speechlike()
    first, second = speech_models.xvector(model, extractor, audio), speech_models.xvector(model, extractor, audio)
    assert abs(np.linalg.norm(first) - 1.0) < 1e-3
    assert float(np.dot(first, second)) > 0.99
    registry.release()


def test_utmos_scores_in_the_mos_range(registry):
    predictor = registry.get("utmos", speech_models.load_utmos)
    score = speech_models.utmos_score(predictor, _speechlike(), "cuda")
    assert 0.5 <= score <= 5.5
    registry.release()


def test_scoreq_reads_a_mos_on_the_gpu_or_cpu(registry):
    session = registry.get("scoreq", mos_models.load_scoreq)
    scores = mos_models.scoreq_scores(session, _speechlike())
    assert set(scores) == {"mos.scoreq_nr"}
    assert 0.5 <= scores["mos.scoreq_nr"] <= 5.5
    registry.release()


def test_mert_hears_the_same_music_as_the_same(registry):
    from scripts.restoration_quality import judges

    model, extractor = registry.get("mert", judges.load_mert)
    audio = _speechlike()
    assert judges.mert_distance(model, extractor, audio, audio) < 1e-3
    assert judges.mert_distance(model, extractor, audio, audio[::-1].copy()) > 1e-3
    registry.release()
