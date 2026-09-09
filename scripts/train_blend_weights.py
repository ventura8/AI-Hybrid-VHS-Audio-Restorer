#!/usr/bin/env python3
"""Learns the per-bin blend weight between the original and the subtracted signal.

Spectral subtraction takes some programme with the noise. An oracle that picks the ideal
blend weight from the clean reference recovers 1.7-1.9 dB of log-spectral distance over the
shipping chain, which is several times any parameter tuning tried on this branch. Two
hand-designed weightings lost; the oracle shows the structure is right and only the function
was wrong, so it is learned here instead.

Two choices matter more than the architecture:

- The loss is the log-magnitude error of the blended result against the clean reference,
  not the error on the weight itself. Weight error is not what anybody hears, and a given
  error matters far more in a loud bin than a quiet one; optimising the blended magnitude
  in dB optimises the quantity the quality gate actually measures.
- The train/test split is by fixture, never by bin. Bins from one recording are highly
  correlated, so a random split would let the model see the same audio on both sides and
  report a score it cannot reproduce on a new tape.
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import blend_weights
from scripts.build_blend_dataset import _recording_of

EPS = 1e-10

# Silence in the reference is what makes this loss unstable. 13.7% of bins in the
# multi-voice corpus have a clean magnitude of exactly zero -- the digital silence around
# each utterance -- and such a bin asks the model to drive the blend to absolute zero. As
# the blended magnitude approaches the log floor its gradient, 1/x, explodes: training
# collapsed to a loss of 0.000 and produced NaN weights. Flooring both sides 80 dB below
# the loudest bin removes the pathology and matches how log-spectral distance is measured
# in the first place; anything under that floor is inaudible either way.
FLOOR_BELOW_PEAK_DB = 80.0
GRADIENT_CLIP = 5.0


class BlendWeightNet(nn.Module):
    """A small MLP mapping one bin's observable features to its blend weight."""

    def __init__(self, features, hidden=64):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch):
        """Returns a weight in [0, 1] per row."""
        return torch.sigmoid(self.stack(batch)).squeeze(-1)


def _log_magnitude(magnitude, floor):
    """Decibel magnitude, floored so reference silence cannot destabilise the gradient."""
    return 20.0 * torch.log10(torch.clamp(magnitude, min=floor))


def _blend_loss(weight, magnitude_original, magnitude_denoised, magnitude_clean, floor):
    """Log-magnitude error of the blended result, which is what the gate measures."""
    blended = weight * magnitude_denoised + (1.0 - weight) * magnitude_original
    return torch.mean((_log_magnitude(blended, floor) - _log_magnitude(magnitude_clean, floor)) ** 2)


def _split_by_fixture(source, names, holdout_fraction=0.25, seed=0):
    """Splits on recordings, so no tape contributes to both sides.

    Splitting on fixtures would not be enough: several fixtures are different segments and
    different defect variants of the same recording, and those are near-duplicates. A model
    that saw one segment of a tape has effectively seen the others.
    """
    # One lookup per fixture, indexed per bin: the name rule is a regex, and there are
    # millions of bins for a few hundred fixtures.
    recordings = np.array([_recording_of(name) for name in names])[source]
    unique = np.unique(recordings)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    cut = max(1, int(len(unique) * holdout_fraction))
    held = set(unique[:cut].tolist())
    mask = np.isin(recordings, list(held))
    return ~mask, mask, len(unique) - cut, cut


def _standardise(train_features):
    """Feature scaling taken from the training split alone."""
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True) + 1e-6
    return mean, std


def _evaluate(weight, magnitudes, floor):
    """Returns the log-magnitude error for a given weight, in dB."""
    original, denoised, clean = magnitudes
    blended = weight * denoised + (1.0 - weight) * original
    error = 20.0 * np.log10(np.maximum(blended, floor)) - 20.0 * np.log10(np.maximum(clean, floor))
    return float(np.sqrt(np.mean(error**2)))


def _reference_scores(magnitudes, ideal, floor):
    """Scores for the strategies the model has to beat."""
    original, denoised, _clean = magnitudes
    return {
        "keep original (no denoise)": _evaluate(np.zeros_like(original), magnitudes, floor),
        "always denoised (current chain)": _evaluate(np.ones_like(original), magnitudes, floor),
        "oracle (ideal weight)": _evaluate(ideal, magnitudes, floor),
    }


def _baseline_score(model_path, features, magnitudes, floor):
    """The held-out score of an already fitted weights file, on the same split as the retrain.

    Returns None when the file is absent or was fitted on a different feature set, so a
    retrain can always be read against what ships rather than against a number from
    another dataset.
    """
    model = blend_weights.load_model(model_path)
    if model is None:
        return None
    predicted = blend_weights.predict_weights(features, model)
    return None if predicted is None else _evaluate(predicted, magnitudes, floor)


def _train(model, tensors, epochs, batch, learning_rate, device, floor):
    """Runs the optimisation loop and returns the trained model."""
    features, original, denoised, clean = tensors
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    count = len(features)
    for epoch in range(epochs):
        permutation = torch.randperm(count, device=device)
        total, seen = 0.0, 0
        for start in range(0, count, batch):
            index = permutation[start : start + batch]
            optimiser.zero_grad()
            weight = model(features[index])
            loss = _blend_loss(weight, original[index], denoised[index], clean[index], floor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimiser.step()
            total += float(loss) * len(index)
            seen += len(index)
        print(f"  epoch {epoch + 1:2d}/{epochs}  train loss {total / max(seen, 1):8.3f} dB^2")
    return model


def _predict(model, features, mean, std, batch, device):
    """Held-out predictions a batch at a time: the features are scaled on the host and only one
    batch is on the device at once, so a held-out split of millions of bins does not have to fit there."""
    predicted = []
    with torch.no_grad():
        for start in range(0, len(features), batch):
            stop = start + batch
            scaled = np.ascontiguousarray((features[start:stop] - mean) / std)
            predicted.append(model(torch.from_numpy(scaled).float().to(device)).cpu().numpy())
    return np.concatenate(predicted) if predicted else np.zeros(0, dtype=np.float32)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("experiments/blend_dataset.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/blend_weights.npz"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=65536)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--baseline", type=Path, default=Path("assets/blend_weights.npz"), help="Fitted weights to score on the same split")
    return parser.parse_args()


def main():
    """Trains the blend-weight predictor and reports held-out performance."""
    args = _parse_args()
    data = np.load(args.dataset)
    stored = tuple(str(name) for name in data["feature_names"])
    if stored != tuple(blend_weights.FEATURE_NAMES):
        raise SystemExit(
            f"the dataset was built on features {stored}, not the current {blend_weights.FEATURE_NAMES}; rebuild it with "
            "scripts/build_blend_dataset.py before training"
        )
    features = data["features"]
    source = data["source"]
    magnitudes = (data["magnitude_original"], data["magnitude_denoised"], data["magnitude_clean"])
    ideal = data["weights"]

    names = data["fixture_names"]
    train_mask, test_mask, train_fixtures, test_fixtures = _split_by_fixture(source, names)
    print(f"{len(features):,} bins | {train_fixtures} fixtures train, {test_fixtures} held out\n")

    if not train_mask.any() or not test_mask.any():
        raise SystemExit(
            f"Split left {int(train_mask.sum())} training and {int(test_mask.sum())} held-out bins. "
            "Fixture names probably collide across sources, so every bin grouped into one recording."
        )
    mean, std = _standardise(features[train_mask])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    def to_tensor(array):
        return torch.from_numpy(np.ascontiguousarray(array)).float().to(device)

    train_tensors = (
        to_tensor((features[train_mask] - mean) / std),
        to_tensor(magnitudes[0][train_mask]),
        to_tensor(magnitudes[1][train_mask]),
        to_tensor(magnitudes[2][train_mask]),
    )
    # The floor is a training-set constant; the held-out recordings do not get to set it.
    floor = float(np.max(magnitudes[0][train_mask])) * (10.0 ** (-FLOOR_BELOW_PEAK_DB / 20.0))
    print(f"magnitude floor: {floor:.3e} ({FLOOR_BELOW_PEAK_DB:.0f} dB below peak)" + chr(10))
    model = BlendWeightNet(features.shape[1], args.hidden).to(device)
    _train(model, train_tensors, args.epochs, args.batch, args.learning_rate, device, floor)

    model.eval()
    predicted = _predict(model, features[test_mask], mean, std, args.batch, device)

    held = tuple(m[test_mask] for m in magnitudes)
    scores = _reference_scores(held, ideal[test_mask], floor)
    scores["learned blend"] = _evaluate(predicted, held, floor)
    baseline = _baseline_score(args.baseline, features[test_mask], held, floor)
    if baseline is not None:
        scores[f"shipped blend ({args.baseline.name})"] = baseline

    print("\nheld-out log-magnitude error (dB, lower is better):")
    for label, value in sorted(scores.items(), key=lambda kv: kv[1]):
        print(f"  {label:<34} {value:7.3f}")
    baseline = scores["always denoised (current chain)"]
    oracle = scores["oracle (ideal weight)"]
    gain = baseline - scores["learned blend"]
    headroom = baseline - oracle
    share = gain / headroom * 100.0 if headroom > 0 else 0.0
    print(f"\n  gain over the current chain: {gain:+.3f} dB")
    print(f"  that is {share:.0f}% of the {headroom:.3f} dB the oracle shows is available")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    # The feature names travel with the weights, so a later change to the feature set is
    # refused by name and not only by width.
    np.savez(
        args.output,
        feature_mean=mean,
        feature_std=std,
        feature_names=np.array(blend_weights.FEATURE_NAMES),
        hidden=np.array(args.hidden),
        **weights,
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
