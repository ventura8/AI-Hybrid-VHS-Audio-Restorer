"""Learned quality predictors: SIGMOS (P.804 dimensions), DNSMOS (P.835 / P.808) and Audiobox Aesthetics.

Guardrails, not objectives. SIGMOS is the one that separates what a listener separates:
COL (coloration: muffled, metallic), DISC (discontinuity: chopping, musical noise),
NOISE (residual hiss). DNSMOS stays the cross-check it always was. Audiobox reads the
whole programme, music included; a drop in its PC axis is a scene that got simpler.
Both sides are brought to -23 LUFS with one gain first, so level cannot move a score.
"""

import importlib.util
from pathlib import Path

import numpy as np

from modules.utils import MODELS_DIR
from scripts.restoration_quality import audio_io
from scripts.score_perceptual import WINDOW_S as DNSMOS_MIN_S
from scripts.score_perceptual import _sessions, score_audio

SIGMOS_DIR = "sigmos"
SIGMOS_FILE = "sigmos.py"
AUDIOBOX_DIR = "audiobox-aesthetics"
AUDIOBOX_CHECKPOINT = "checkpoint.pt"
SIGMOS_KEYS = {
    "MOS_COL": "col",
    "MOS_DISC": "disc",
    "MOS_LOUD": "loud",
    "MOS_NOISE": "noise",
    "MOS_REVERB": "reverb",
    "MOS_SIG": "sig",
    "MOS_OVRL": "ovrl",
}
AUDIOBOX_KEYS = {"PQ": "pq", "PC": "pc", "CE": "ce", "CU": "cu"}


def _require(models_dir, name, set_name):
    path = Path(models_dir or MODELS_DIR) / name
    if not path.exists():
        raise SystemExit(f"{name} missing under {path.parent}; run scripts/download_quality_models.py --set {set_name}")
    return path


def load_sigmos(_device, models_dir):
    """Microsoft's SigMOS estimator, imported from the pinned `sigmos.py` beside its ONNX weight (CPU)."""
    path = _require(models_dir, SIGMOS_DIR, "mos")
    spec = importlib.util.spec_from_file_location("sigmos", path / SIGMOS_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SigMOS(model_dir=str(path))


def load_dnsmos(_device, _models_dir):
    return _sessions()


def load_audiobox(device, models_dir):
    from audiobox_aesthetics.infer import AesPredictor

    path = _require(models_dir, AUDIOBOX_DIR, "mos") / AUDIOBOX_CHECKPOINT
    predictor = AesPredictor(checkpoint_pth=str(path), data_col="path")
    predictor.device = device
    predictor.model = predictor.model.to(device)
    return predictor


def sigmos_scores(estimator, audio48k):
    result = estimator.run(np.asarray(audio48k, dtype=np.float32), sr=audio_io.SIGMOS_RATE)
    return {f"mos.sigmos_{short}": float(result[key]) for key, short in SIGMOS_KEYS.items()}


def dnsmos_scores(sessions, audio16k):
    if len(audio16k) < DNSMOS_MIN_S * audio_io.SPEECH_RATE:
        return {}
    result = score_audio(np.asarray(audio16k, dtype=np.float32), sessions) or {}
    return {f"mos.dnsmos_{key.lower()}": float(value) for key, value in result.items()}


def audiobox_scores(predictor, audio16k):
    import torch

    wave = torch.from_numpy(np.asarray(audio16k, dtype=np.float32))[None, :]
    result = predictor.forward([{"path": wave, "sample_rate": audio_io.SPEECH_RATE}])[0]
    return {f"mos.audiobox_{short}": float(result[key]) for key, short in AUDIOBOX_KEYS.items()}


def _window(pair, side, row, rate, gain):
    audio = pair.at(side, rate)
    begin, end = int(round(row.start_s * rate)), int(round(row.end_s * rate))
    return audio[begin:end] * np.float32(gain)


def _apply(pair, card, gain, rate, scorer, routes):
    for row in card.rows:
        if row.route not in routes:
            continue
        row.source.update(scorer(_window(pair, "source", row, rate, gain)))
        row.output.update(scorer(_window(pair, "output", row, rate, gain)))


def score(pair, card, registry):
    """SIGMOS and DNSMOS on speech/mixed windows, Audiobox on every non-silent window."""
    gain = pair.mos_gain()
    speech = ("speech", "mixed")
    sigmos = registry.get("sigmos", load_sigmos)
    _apply(pair, card, gain, audio_io.SIGMOS_RATE, lambda audio: sigmos_scores(sigmos, audio), speech)
    registry.release()
    sessions = registry.get("dnsmos", load_dnsmos)
    _apply(pair, card, gain, audio_io.SPEECH_RATE, lambda audio: dnsmos_scores(sessions, audio), speech)
    registry.release()
    predictor = registry.get("audiobox", load_audiobox)
    _apply(pair, card, gain, audio_io.SPEECH_RATE, lambda audio: audiobox_scores(predictor, audio), ("speech", "mixed", "music"))
