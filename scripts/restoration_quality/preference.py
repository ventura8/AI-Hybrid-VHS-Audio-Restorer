"""Bradley-Terry preference head: which reading differences make the listener prefer one clip over another.

The judge the ear plan trains is not an audio model (plan 3.1). It is a regularised
logistic fit on the differences of harness readings between two clips the listener
compared:

    P(a preferred over b) = sigmoid(sum_r w_r * (x_r(a) - x_r(b)) / s_r)

- One record per answer: `(features_a, features_b, outcome)`, features `{reading: value}`
  (flat readings, or the two-sided distances of `reward.distances` when the preference is
  two-sided, as for brightness), outcome 1.0 when `a` won, 0.0 when `b` won and 0.5 for an
  explicit "same". The records are plain tuples, so this module does not depend on the
  verdict ledger's format.
- A reading missing on either side contributes 0: no evidence either way.
- No intercept: swapping the clips flips the probability exactly. Order effects belong to
  the listening protocol (both orders, replicates), not to the head.
- Every difference column is divided by its root-mean-square over the training pairs, so
  the L2 penalty (default 1.0) weighs readings in different units alike and the
  standardised weights compare as importances. The fit is L-BFGS on the exact gradient.
- Validation is leave-one-group-out with the tape as the group: the readings of one tape
  share a source, so a pair from a held-in tape says little about a new tape.
- The active-learning picker (plan 3.1: "pairs where the head is near 50% or readings
  disagree") ranks the open pairs by `|P - 0.5| * (1 - disagreement)`. The disagreement is
  how evenly the readings split on which clip is better, each voting with its standardised
  weight: 0 when they all point one way, 1 when the votes cancel. A pair the head is unsure
  of, or one whose readings pull against each other, comes first; there one answer moves
  the fit most. On a weak head, nearest-0.5 alone also prefers near-identical clips, the
  pairs the plan's principle 6 never shows the listener, so `exclude` (the audibility check
  of `auditory.py`, plan 1.2) drops a pair before it is offered.
- The fit must converge. The loss is strictly convex with L2 > 0, so L-BFGS failing means
  something odd (extreme scales); `fit` raises instead of handing back a silent model the
  falsifier in `check_verdicts.py` would use. `require_convergence=False` keeps the model
  and records `converged` and scipy's `message` on it.

Why a head this small, and not a learned audio judge (research of 2026-10-08,
`grpo_synthesis.md` C1/C2): the listener answers in batches of 10-20 picks per stage and the
calibration bank `known_ordering_v2.json` holds 66 strict pairs from about three tapes,
enough to calibrate the weights of readings that already exist, not to learn new
perception. Open audio LLMs were at chance on "which clip is more degraded" (Qwen2.5-Omni
50.17 %, Audio Flamingo 3 50.50 %, against 94-99 % for humans), their Whisper front end
hears nothing above 8 kHz, they have no Romanian, and a cloud judge would upload family
tapes. The head sets the grids' weights and targets and is the falsifier in
`check_verdicts.py`; it is never the loop objective on its own. numpy/scipy, CPU, no new
dependency.
"""

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

DEFAULT_L2 = 1.0
MIN_SCALE = 1e-12
MIN_VOTES = 1e-12
PROBABILITY_CLIP = 1e-12
SAME = 0.5


@dataclass(frozen=True)
class PreferenceModel:
    """A fitted head: reading names, weights per standardised difference, the scales and the fit's settings."""

    names: tuple
    weights: np.ndarray
    scales: np.ndarray
    l2: float = DEFAULT_L2
    n_pairs: int = 0
    converged: bool = True
    message: str = ""

    def coefficients(self, standardised=False):
        """`{reading: log-odds per unit of difference}`; `standardised=True` gives them per RMS difference (importance)."""
        values = self.weights if standardised else self.weights / self.scales
        return {name: float(value) for name, value in zip(self.names, values)}


@dataclass(frozen=True)
class Validation:
    """Held-out agreement: accuracy on decided pairs ("same" left out), log-loss on every pair, per group and pooled."""

    accuracy: float | None
    log_loss: float
    n_pairs: int
    by_group: dict = field(default_factory=dict)


def sigmoid(z):
    """1 / (1 + exp(-z)), computed without overflow for any z."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=np.float64)))


def _finite(value):
    return isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool) and bool(np.isfinite(value))


def feature_names(records):
    """The sorted union of the reading names of both clips of every record."""
    return tuple(sorted({name for record in records for name in itertools.chain(record[0], record[1])}))


def _value(features, name):
    value = features.get(name)
    return float(value) if _finite(value) else np.nan


def difference_matrix(records, names):
    """`(pairs, readings)` of a - b; a reading missing on either side reads 0."""
    rows = [[_value(record[0], name) - _value(record[1], name) for name in names] for record in records]
    return np.nan_to_num(np.asarray(rows, dtype=np.float64).reshape(len(records), len(names)), nan=0.0)


def outcomes(records):
    """The outcome of every record as floats in [0, 1]; ValueError on anything else."""
    values = np.asarray([float(record[2]) for record in records], dtype=np.float64)
    if not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError("an outcome must lie in [0, 1]: 1 = a preferred, 0 = b preferred, 0.5 = same")
    return values


def _scales(diffs):
    """Root-mean-square of every difference column; a column that never moves keeps scale 1."""
    rms = np.sqrt(np.mean(diffs**2, axis=0))
    return np.where(rms > MIN_SCALE, rms, 1.0)


def _loss_and_gradient(weights, x, y, l2):
    """Logistic loss with soft labels plus the L2 penalty, and its gradient."""
    z = x @ weights
    loss = float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * weights @ weights)
    return loss, x.T @ (sigmoid(z) - y) + l2 * weights


def _check_l2(l2):
    """ValueError unless `l2 > 0`; NaN is refused like zero."""
    if math.isnan(l2) or l2 <= 0.0:
        raise ValueError(f"l2 must be > 0 (an unregularised fit diverges on separable answers): {l2!r}")


def _checked(records, l2, names):
    if not records:
        raise ValueError("no preference records to fit")
    _check_l2(l2)
    names = tuple(names) if names is not None else feature_names(records)
    if not names:
        raise ValueError("the records carry no readings")
    return names


def _converged(result, required):
    """scipy's verdict on the fit; RuntimeError when it failed and convergence is required."""
    if required and not result.success:
        raise RuntimeError(f"the preference fit did not converge: {result.message}")
    return bool(result.success), str(result.message)


def fit(records, *, l2=DEFAULT_L2, names=None, require_convergence=True):
    """A `PreferenceModel` fitted on `[(features_a, features_b, outcome), ...]`; `names` fixes the readings and their order.

    RuntimeError when L-BFGS does not converge, unless `require_convergence=False`: the model
    then carries `converged=False` and scipy's message.
    """
    names = _checked(records, l2, names)
    diffs = difference_matrix(records, names)
    scales = _scales(diffs)
    result = minimize(
        _loss_and_gradient, np.zeros(len(names)), args=(diffs / scales, outcomes(records), float(l2)), jac=True, method="L-BFGS-B"
    )
    converged, message = _converged(result, require_convergence)
    return PreferenceModel(names, np.asarray(result.x, dtype=np.float64), scales, float(l2), len(records), converged, message)


def predict_records(model, records):
    """P(a preferred) for every record (the outcome, if any, is ignored)."""
    diffs = difference_matrix([(record[0], record[1]) for record in records], model.names)
    return sigmoid((diffs / model.scales) @ model.weights)


def predict(model, features_a, features_b):
    """P(a preferred over b)."""
    return float(predict_records(model, [(features_a, features_b)])[0])


def _accuracy(probabilities, values):
    """Share of decided pairs on the right side of 0.5 (a prediction of exactly 0.5 earns half), or None."""
    decided = values != SAME
    if not np.any(decided):
        return None
    signs = np.sign((probabilities[decided] - SAME) * (values[decided] - SAME))
    return float(np.mean(0.5 + 0.5 * signs))


def _log_loss(probabilities, values):
    p = np.clip(probabilities, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    return float(np.mean(-(values * np.log(p) + (1.0 - values) * np.log(1.0 - p))))


def agreement(probabilities, values):
    """`{"accuracy", "log_loss", "n"}` of predictions against outcomes."""
    probabilities, values = np.asarray(probabilities, dtype=np.float64), np.asarray(values, dtype=np.float64)
    return {"accuracy": _accuracy(probabilities, values), "log_loss": _log_loss(probabilities, values), "n": int(len(values))}


def _split(records, groups, label):
    held_in = [record for record, group in zip(records, groups) if group != label]
    held_out = [record for record, group in zip(records, groups) if group == label]
    return held_in, held_out


def _fold(records, groups, label, l2, names):
    """(probabilities, outcomes) for the records of `label`, from a head fitted on every other group."""
    held_in, held_out = _split(records, groups, label)
    model = fit(held_in, l2=l2, names=names)
    return predict_records(model, held_out), outcomes(held_out)


def _labels(records, groups):
    labels = list(dict.fromkeys(groups))
    if len(labels) < 2 or len(groups) != len(records):
        raise ValueError("leave-one-group-out needs one group label per record and at least two groups")
    return labels


def _pooled(folds):
    probabilities = np.concatenate([fold[0] for fold in folds.values()])
    return agreement(probabilities, np.concatenate([fold[1] for fold in folds.values()]))


def leave_one_group_out(records, groups, *, l2=DEFAULT_L2):
    """`Validation` with each group (a tape) held out in turn; `groups` runs parallel to `records`."""
    names = feature_names(records)
    folds = {label: _fold(records, groups, label, l2, names) for label in _labels(records, groups)}
    pooled = _pooled(folds)
    return Validation(pooled["accuracy"], pooled["log_loss"], pooled["n"], {label: agreement(*fold) for label, fold in folds.items()})


def _open_pairs(clips, asked):
    done = {frozenset(pair) for pair in asked}
    return [pair for pair in itertools.combinations(clips, 2) if frozenset(pair) not in done]


def disagreement(model, features_a, features_b):
    """How evenly the head's readings split on which clip is better, in [0, 1].

    Every reading read on both clips votes for the clip it favours with its standardised
    weight (the sign of the difference times the weight): 0 when the votes all point one way
    (or nothing votes), 1 when they cancel.
    """
    votes = np.sign(difference_matrix([(features_a, features_b)], model.names)[0]) * model.weights
    total = float(np.sum(np.abs(votes)))
    return 0.0 if total < MIN_VOTES else 1.0 - abs(float(np.sum(votes))) / total


def _scored(model, clips, pair):
    first, second = pair
    return first, second, predict(model, clips[first], clips[second]), disagreement(model, clips[first], clips[second])


def _priority(item):
    """`|P - 0.5|` shrunk by the readings' disagreement: lower is asked first."""
    return abs(item[2] - SAME) * (1.0 - item[3])


def _never(_first, _second):
    return False


def _offered(scored, exclude):
    return (item[:3] for item in scored if not exclude(item[0], item[1]))


def pick_pairs(model, clips, *, count=1, asked=(), exclude=None):
    """The `count` pairs not yet asked that the head is least sure of or whose readings disagree most, best first.

    `clips` is `{clip_id: features}`; `asked` holds pairs already put to the listener, in
    either order; `exclude(clip_a, clip_b)` returns True for a pair never to offer (the
    audibility check calling it inaudible), and is called only until `count` pairs are kept.
    Returns `[(clip_a, clip_b, P(a preferred))]`; ties keep the clips' order.
    """
    scored = sorted((_scored(model, clips, pair) for pair in _open_pairs(clips, asked)), key=_priority)
    return list(itertools.islice(_offered(scored, exclude or _never), max(0, int(count))))
