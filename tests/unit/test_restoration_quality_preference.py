"""The Bradley-Terry preference head: recovering known weights, symmetry, leave-one-tape-out, and the pair picker."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.restoration_quality import preference as pref

# The listener's hidden weights (log-odds per unit of reading difference) and how far each
# reading spreads between clips: three readings in very different units.
TRUE = {"air": 1.5, "gap": -0.4, "sib": 4.0}
SPREAD = {"air": 1.0, "gap": 5.0, "sib": 0.2}


def _clips(rng, count):
    return [{name: float(rng.normal(0.0, SPREAD[name])) for name in TRUE} for _ in range(count)]


def _answer(rng, clip_a, clip_b):
    """1.0 when the simulated listener prefers `clip_a`, drawn from the Bradley-Terry probability of TRUE."""
    z = sum(weight * (clip_a[name] - clip_b[name]) for name, weight in TRUE.items())
    return float(rng.random() < 1.0 / (1.0 + np.exp(-z)))


def _records(seed=0, clips=80, pairs=3000):
    rng = np.random.default_rng(seed)
    pool = _clips(rng, clips)
    picks = [rng.choice(clips, 2, replace=False) for _ in range(pairs)]
    return [(pool[i], pool[j], _answer(rng, pool[i], pool[j])) for i, j in picks]


def _taped(tapes=("tele7abc", "soti", "vaccin"), pairs=400):
    """Records and their tape labels: each tape has its own clips, the listener's weights are shared."""
    records, groups = [], []
    for index, tape in enumerate(tapes):
        batch = _records(seed=10 + index, clips=30, pairs=pairs)
        records += batch
        groups += [tape] * len(batch)
    return records, groups


def _one_reading_model(weight=1.0):
    return pref.PreferenceModel(names=("air",), weights=np.array([weight]), scales=np.array([1.0]))


def test_the_head_recovers_a_known_weight_vector():
    """3000 simulated answers recover the listener's weights in direction and, within 20 %, in size."""
    model = pref.fit(_records())
    found = np.array([model.coefficients()[name] for name in TRUE])
    truth = np.array(list(TRUE.values()))
    assert found @ truth / (np.linalg.norm(found) * np.linalg.norm(truth)) > 0.998
    assert np.max(np.abs(found - truth) / np.abs(truth)) < 0.2


def test_standardised_coefficients_are_the_raw_ones_times_the_scales():
    """The importance view is the per-unit view times each reading's RMS difference."""
    model = pref.fit(_records(pairs=600))
    raw, standardised = model.coefficients(), model.coefficients(standardised=True)
    assert [standardised[name] for name in model.names] == pytest.approx(list(np.array([raw[n] for n in model.names]) * model.scales))
    assert model.n_pairs == 600
    assert model.names == ("air", "gap", "sib")


def test_a_prediction_flips_with_the_order_and_ignores_missing_readings():
    """No intercept: P(a, b) + P(b, a) = 1, a clip against itself is 0.5, a missing reading is no evidence."""
    model = pref.fit(_records(pairs=600))
    a, b = {"air": 1.0, "gap": 2.0, "sib": 0.1}, {"air": 0.0, "gap": -1.0, "sib": 0.0}
    assert pref.predict(model, a, b) + pref.predict(model, b, a) == pytest.approx(1.0)
    assert pref.predict(model, a, a) == pytest.approx(0.5)
    assert pref.predict(model, {"air": 1.0}, {"gap": None, "sib": float("nan")}) == pref.predict(model, {"air": 1.0}, {})


def test_the_difference_matrix_reads_a_missing_reading_as_no_evidence():
    """The names are the union of both clips' readings; a reading missing on either side reads 0."""
    records = [({"a": 2.0, "b": 1.0}, {"a": 0.5}, 1.0), ({"c": 1.0}, {"a": 1.0, "c": 3.0}, 0.0)]
    names = pref.feature_names(records)
    assert names == ("a", "b", "c")
    assert pref.difference_matrix(records, names).tolist() == [[1.5, 0.0, 0.0], [0.0, 0.0, -2.0]]


def _oracle(records):
    """How well the listener's own weights (TRUE) predict the same answers: the best any head can do."""
    probabilities = [pref.sigmoid(sum(weight * (a[name] - b[name]) for name, weight in TRUE.items())) for a, b, _y in records]
    return pref.agreement(probabilities, [record[2] for record in records])


def _held_out_gap(validation, records, groups, tape):
    """Oracle accuracy minus the held-out head's accuracy on one tape."""
    oracle = _oracle([record for record, group in zip(records, groups) if group == tape])
    return oracle["accuracy"] - validation.by_group[tape]["accuracy"]


def test_leave_one_tape_out_does_as_well_as_the_listeners_own_weights():
    """Held out tape by tape, the head predicts within 3 points of the true weights."""
    records, groups = _taped()
    validation = pref.leave_one_group_out(records, groups)
    assert set(validation.by_group) == {"tele7abc", "soti", "vaccin"}
    assert validation.n_pairs == len(records)
    assert max(_held_out_gap(validation, records, groups, tape) for tape in validation.by_group) < 0.03
    assert validation.log_loss < _oracle(records)["log_loss"] + 0.02


@pytest.mark.parametrize("groups", [["one"] * 3, ["a", "b"]])
def test_leave_one_group_out_needs_a_label_per_record_and_two_groups(groups):
    """One group, or labels that do not match the records, cannot be validated."""
    records = [({"air": 1.0}, {"air": 0.0}, 1.0)] * 3
    with pytest.raises(ValueError):
        pref.leave_one_group_out(records, groups)


@pytest.mark.parametrize(
    "records, l2",
    [
        ([], 1.0),
        ([({"air": 1.0}, {"air": 0.0}, 1.0)], 0.0),
        ([({"air": 1.0}, {"air": 0.0}, 2.0)], 1.0),
        ([({}, {}, 1.0)], 1.0),
    ],
)
def test_the_fit_refuses_no_records_no_regularisation_bad_outcomes_or_no_readings(records, l2):
    """Each unusable input is refused before the fit runs."""
    with pytest.raises(ValueError):
        pref.fit(records, l2=l2)


def test_a_same_answer_counts_in_the_log_loss_but_not_the_accuracy():
    """'Same' answers are left out of the accuracy; a 0.5 prediction earns half a point."""
    assert pref.agreement([0.9, 0.5], [1.0, 0.5])["accuracy"] == 1.0
    assert pref.agreement([0.6], [0.5])["accuracy"] is None
    assert pref.agreement([0.5, 0.9], [1.0, 0.0])["accuracy"] == 0.25
    assert pref.agreement([0.5], [1.0])["log_loss"] == pytest.approx(np.log(2.0))


def test_same_answers_pull_the_fit_towards_indifference():
    """Half 'same', half 'a': the head lands between indifference and the mean answer 0.75."""
    same = [({"air": 1.0}, {"air": 0.0}, 0.5)] * 20
    model = pref.fit(same + [({"air": 1.0}, {"air": 0.0}, 1.0)] * 20)
    assert 0.5 < pref.predict(model, {"air": 1.0}, {"air": 0.0}) < 0.75


def test_the_picker_asks_about_the_pair_the_head_is_least_sure_of():
    """The pair whose predicted preference is nearest 50 % comes first."""
    clips = {"a": {"air": 0.0}, "b": {"air": 3.0}, "c": {"air": 0.1}, "d": {"air": 6.0}}
    picks = pref.pick_pairs(_one_reading_model(), clips)
    assert [pick[:2] for pick in picks] == [("a", "c")]
    assert picks[0][2] == pytest.approx(pref.sigmoid(-0.1))


def test_the_picker_skips_pairs_already_asked_in_either_order():
    """A pair already asked, in either order, is never asked again."""
    clips = {"a": {"air": 0.0}, "b": {"air": 3.0}, "c": {"air": 0.1}, "d": {"air": 6.0}}
    picks = pref.pick_pairs(_one_reading_model(), clips, count=2, asked=[("c", "a")])
    assert [pick[:2] for pick in picks] == [("b", "c"), ("a", "b")]
    assert not pref.pick_pairs(_one_reading_model(), clips, count=0)


def test_the_picker_never_offers_a_pair_the_audibility_check_calls_inaudible():
    """`exclude` drops a pair before it is offered, however unsure the head is about it."""
    clips = {"a": {"air": 0.0}, "b": {"air": 3.0}, "c": {"air": 0.1}, "d": {"air": 6.0}}
    calls = []

    def inaudible(first, second):
        calls.append((first, second))
        return {first, second} == {"a", "c"}

    picks = pref.pick_pairs(_one_reading_model(), clips, count=1, exclude=inaudible)
    assert [pick[:2] for pick in picks] == [("b", "c")]
    assert calls == [("a", "c"), ("b", "c")]


def _two_reading_model():
    return pref.PreferenceModel(names=("air", "gap"), weights=np.array([1.0, 1.0]), scales=np.array([1.0, 1.0]))


# p against q: only air differs. p and q against r: air and gap pull opposite ways.
SPLIT = {"p": {"air": 0.0, "gap": 0.0}, "q": {"air": 0.5, "gap": 0.0}, "r": {"air": 3.0, "gap": -2.0}}


def test_the_disagreement_is_how_evenly_the_readings_split():
    """Readings voting the same way read 0, cancelling votes read 1, a head with no weight reads 0."""
    model = _two_reading_model()
    assert pref.disagreement(model, SPLIT["p"], SPLIT["q"]) == 0.0
    assert pref.disagreement(model, SPLIT["p"], SPLIT["r"]) == 1.0
    assert pref.disagreement(_one_reading_model(0.0), SPLIT["p"], SPLIT["r"]) == 0.0


def test_the_picker_asks_first_where_the_readings_disagree():
    """p-r is the head's surest pair (P 0.27), but its readings cancel, so it comes ahead of p-q (P 0.38)."""
    picks = pref.pick_pairs(_two_reading_model(), SPLIT, count=3)
    assert [pick[:2] for pick in picks] == [("p", "r"), ("q", "r"), ("p", "q")]
    assert picks[0][2] == pytest.approx(pref.sigmoid(-1.0))


def _failed_minimize(*_args, **_kwargs):
    return SimpleNamespace(success=False, message="ABNORMAL_TERMINATION_IN_LNSRCH", x=np.zeros(1))


def test_a_fit_that_does_not_converge_is_refused_unless_asked_to_keep_it(monkeypatch):
    """A failed L-BFGS raises; `require_convergence=False` keeps the model and says why it is suspect."""
    monkeypatch.setattr(pref, "minimize", _failed_minimize)
    records = [({"air": 1.0}, {"air": 0.0}, 1.0)]
    with pytest.raises(RuntimeError, match="ABNORMAL"):
        pref.fit(records)
    model = pref.fit(records, require_convergence=False)
    assert (model.converged, model.message) == (False, "ABNORMAL_TERMINATION_IN_LNSRCH")


def test_a_converged_fit_says_so():
    """The ordinary fit carries scipy's verdict."""
    model = pref.fit([({"air": 1.0}, {"air": 0.0}, 1.0)] * 4)
    assert model.converged
    assert model.message


def test_the_sigmoid_never_overflows():
    """Large arguments saturate to 0 and 1 without an overflow."""
    with np.errstate(over="raise"):
        values = pref.sigmoid([-1000.0, 0.0, 1000.0])
    assert values.tolist() == pytest.approx([0.0, 0.5, 1.0])
