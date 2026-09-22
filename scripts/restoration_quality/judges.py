"""Two pretrained judges that read the pair the way a listener compares it: Zimtohrli and MERT.

Zimtohrli (Google, Apache-2.0) is a psychoacoustic distance at 48 kHz built around the
just-noticeable difference; it is read on the frames where the source is loud, because
that is where a restoration's coloration is audible and where the noise floor is not.
Its Python binding is imported lazily: the PyPI wheel `zimtohrli` is not pinned in
`pyproject.toml` (it is uploaded by a project contributor from a renamed tree, not by
the Google project's own release pipeline, and its compiled extension cannot be checked
against the source), so the reading is recorded as unavailable until the binding is
built from <https://github.com/google/zimtohrli> (`pip install git+...`, module
`pyohrli`) or the wheel is installed on purpose. The distance is the number used, lower
better; `mos_from_signals` is the same distance through a fixed MOS map.

MERT-v1-95M (m-a-p, CC-BY-NC-4.0, research use) is a music encoder; one minus the
cosine of the time-averaged layer-12 states of the source and the output is a distance
in what the model hears as the music, read on the full mix at 24 kHz.
"""

import importlib
from pathlib import Path

import numpy as np

from modules.utils import MODELS_DIR

ZIMTOHRLI_RATE = 48000
ZIMTOHRLI_FRAME = 4096
LOUD_PERCENTILE = 70.0
ZIMTOHRLI_KEY = "dsp.zimtohrli_loud"
ZIMTOHRLI_ROUTES = ("speech", "music", "mixed")
ZIMTOHRLI_MODULES = ("zimtohrli", "pyohrli")
MERT_DIR = "mert-v1-95m"
MERT_RATE = 24000
MERT_LAYER = 12
MERT_LAYERS = 12
MERT_KEY = "stems.mert_dist"
MERT_ROUTES = ("music", "mixed")


def _require(models_dir, name, set_name):
    path = Path(models_dir or MODELS_DIR) / name
    if not path.exists():
        raise SystemExit(f"{name} missing under {path.parent}; run scripts/download_quality_models.py --set {set_name}")
    return path


def zimtohrli_module():
    """The Zimtohrli binding: `zimtohrli` (the PyPI wheel) or `pyohrli` (the module the Google source tree builds)."""
    for name in ZIMTOHRLI_MODULES:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise ImportError("zimtohrli is not installed; build the binding from https://github.com/google/zimtohrli (module pyohrli)")


def loud_frames(source, frame=ZIMTOHRLI_FRAME, percentile=LOUD_PERCENTILE):
    """Which `frame`-sample frames of `source` are loud: RMS at or above the given percentile of the frames."""
    count = len(source) // frame
    if count == 0:
        return np.zeros(0, dtype=bool)
    frames = np.asarray(source[: count * frame], dtype=np.float64).reshape(count, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return rms >= np.percentile(rms, percentile)


def _select(audio, mask, frame=ZIMTOHRLI_FRAME):
    """The masked frames of `audio`, concatenated, as float32."""
    frames = np.asarray(audio[: len(mask) * frame], dtype=np.float32).reshape(len(mask), frame)
    return np.ascontiguousarray(frames[mask].reshape(-1))


def zimtohrli_loud(source, output, rate=ZIMTOHRLI_RATE):
    """Zimtohrli distance between the two sides on the frames where the source is loud; None when no frame is."""
    if rate != ZIMTOHRLI_RATE:
        raise ValueError(f"zimtohrli reads 48 kHz audio, not {rate} Hz")
    module = zimtohrli_module()
    length = min(len(source), len(output))
    mask = loud_frames(source[:length])
    if not mask.any():
        return None
    return float(module.Pyohrli().distance(_select(source, mask), _select(output, mask)))


def _slice(row, rate):
    return slice(int(round(row.start_s * rate)), int(round(row.end_s * rate)))


def score_zimtohrli(pair, card):
    """Fills `dsp.zimtohrli_loud` on every programme row (paired: output side, zero source); ImportError when unavailable."""
    zimtohrli_module()
    source, output = pair.at("source", ZIMTOHRLI_RATE), pair.at("output", ZIMTOHRLI_RATE)
    for row in card.rows:
        if row.route not in ZIMTOHRLI_ROUTES:
            continue
        sl = _slice(row, ZIMTOHRLI_RATE)
        row.source[ZIMTOHRLI_KEY], row.output[ZIMTOHRLI_KEY] = 0.0, zimtohrli_loud(source[sl], output[sl])


def load_mert(device, models_dir):
    """MERT-v1-95M from the pinned store with its two remote-code files, plus its 24 kHz feature extractor."""
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    path = _require(models_dir, MERT_DIR, "stems")
    model = AutoModel.from_pretrained(str(path), trust_remote_code=True).to(device).eval()
    return model, Wav2Vec2FeatureExtractor.from_pretrained(str(path))


def mert_embedding(model, extractor, audio24k):
    """The unit-normalised time mean of the layer-`MERT_LAYER` hidden states of one clip."""
    import torch

    device = getattr(model, "device", "cpu")
    inputs = extractor(np.asarray(audio24k, dtype=np.float32), sampling_rate=MERT_RATE, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.no_grad():
        states = _layer_states(model(**inputs, output_hidden_states=True))
    vector = states[0].float().mean(dim=0).detach().cpu().numpy().astype(np.float64)
    return vector / (np.linalg.norm(vector) + 1e-12)


def _layer_states(outputs):
    """The layer-`MERT_LAYER` states; under transformers 5 the pinned remote code returns no per-layer stack.

    Its `output_hidden_states` no longer reaches the new `HubertEncoder`, so `hidden_states` comes back
    None; layer 12 is the last of the twelve, whose output is `last_hidden_state` (verified equal to a
    forward hook on `encoder.layers[11]` on the pinned weights).
    """
    if outputs.hidden_states is not None:
        return outputs.hidden_states[MERT_LAYER]
    if MERT_LAYER != MERT_LAYERS:
        raise RuntimeError(f"layer {MERT_LAYER} needs the per-layer stack this transformers version does not return")
    return outputs.last_hidden_state


def mert_distance(model, extractor, src24k, out24k):
    """1 - cosine between the MERT embeddings of the two sides: 0 when the model hears the same music."""
    src, out = mert_embedding(model, extractor, src24k), mert_embedding(model, extractor, out24k)
    return float(max(0.0, 1.0 - float(np.dot(src, out))))


def score_mert(pair, card, registry):
    """Fills `stems.mert_dist` on music/mixed rows from the full mix at 24 kHz (paired: output side, zero source)."""
    model, extractor = registry.get("mert", load_mert)
    source, output = pair.at("source", MERT_RATE), pair.at("output", MERT_RATE)
    for row in card.rows:
        if row.route not in MERT_ROUTES:
            continue
        sl = _slice(row, MERT_RATE)
        row.source[MERT_KEY], row.output[MERT_KEY] = 0.0, mert_distance(model, extractor, source[sl], output[sl])
