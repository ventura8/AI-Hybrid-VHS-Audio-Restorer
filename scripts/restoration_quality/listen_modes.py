"""The three listening questions `scripts/listen_ab.py` asks, each turned into ledger records.

One listener, short batches (10-30 picks per question), so each mode is chosen for what a small
number of answers can show:

- ABX (24 trials by default): X is A or B at random and the one-sided binomial p says whether the
  difference was heard. 17/24 right is p 0.032 with power 0.77 when the true hit rate is 75%;
  12/16 is p 0.038 but with power 0.63 only, so 16 trials miss a real difference one time in three.
- pair with an explicit "same": which of two outputs sounds better, "same" when they cannot be
  told apart or no choice can be made. Without "same" two indistinguishable outputs give a coin
  flip that reads as a preference. Each pair is played `repeats` times (once by default) in a
  fresh random A/B order, and one pair is played once more at the end of the block with A and B
  swapped: the replicate that measures the listener's own consistency (plan 1.4, critique 12:
  one replicate per session, because every repeat costs listening minutes; repeating every pair
  doubled a 6-variant block to 30 trials). The replicate is a pair with the session's anchor
  when there is one, else a random pair; its trial is marked in the record, and it counts in
  the pair's tally like any other answer, so the counts feed a Bradley-Terry fit.
- blend: the candidate mixed into the incumbent at x in [0, 1], a controlled axis as a
  psychometric fit needs. Each trial plays the reference R (the incumbent), then A and B, one of
  them the blend, and asks which one differs from R (chance 0.5). A QUEST-style posterior over the
  listener's threshold t (Weibull with slope 3.5, guess 0.5, lapse 0.02, a log-uniform prior on
  0.01-2) places the next trial at its median, the first at x = 1 so the listener hears what to
  listen for. The record keeps the x heard 75% of the time and its 90% band; above 1 means even
  the full candidate was not reliably heard.
"""

import itertools
import math

import numpy as np

from scripts.restoration_quality import ledger

ABX_ALPHA = 0.05
QUEST_GRID = np.geomspace(0.01, 2.0, 160)
QUEST_SLOPE = 3.5
QUEST_GUESS = 0.5
QUEST_LAPSE = 0.02
TARGET_P = 0.75
BAND_QUANTILES = (0.05, 0.95)
BLEND_X_MIN = 0.01


def p_correct(x, threshold):
    """Probability of a right answer at blend level `x` for a listener whose threshold is `threshold`."""
    detect = 1.0 - np.exp(-((x / np.asarray(threshold)) ** QUEST_SLOPE))
    return QUEST_GUESS + (1.0 - QUEST_GUESS - QUEST_LAPSE) * detect


def x_at_target(threshold):
    """The blend level heard TARGET_P of the time by a listener with `threshold`."""
    share = (TARGET_P - QUEST_GUESS) / (1.0 - QUEST_GUESS - QUEST_LAPSE)
    return float(threshold * (-math.log(1.0 - share)) ** (1.0 / QUEST_SLOPE))


def chosen(answer):
    """The stimulus label a pair answer picked, or "same"."""
    first, second = answer["stimuli"]
    return {"A": first, "B": second}.get(answer["choice"], ledger.SAME)


class Mode:
    """What every mode shares: its stimuli, its trial count, the question text and the answers so far."""

    kind = ""

    def __init__(self, labels, total):
        self.labels = list(labels)
        self.total = total
        self.answers = []
        self.text = ""

    def finished(self):
        """True once every trial of the block has its answer."""
        return len(self.answers) >= self.total

    def record(self, trial, choice):
        """Keeps one answer with the trial it answered."""
        answer = {"choice": choice, "key": trial["key"], "x": trial.get("x"), "stimuli": trial["stimuli"]}
        self.answers.append({**answer, "replicate": trial.get("replicate", False)})

    def question(self, stimuli=None):
        """The ledger `question` of this block (or of one pair of it)."""
        return {"type": self.kind, "stimuli": list(stimuli or self.labels), "blind": True, "text": self.text}

    def right_answers(self):
        """How many answers matched their trial's key."""
        return sum(1 for answer in self.answers if answer["choice"] == answer["key"])


class AbxMode(Mode):
    """ABX: X is A or B at random; the record holds n_correct of n_trials and the one-sided binomial p."""

    kind = "abx"

    def __init__(self, labels, trials, rng):
        super().__init__(labels, trials)
        self.x_is_a = [bool(flip) for flip in rng.integers(0, 2, trials)]

    def trial(self, index):
        """Trial `index`: A, B and X, with X's identity as the key."""
        first, second = self.labels
        key = "A" if self.x_is_a[index] else "B"
        x_label = first if key == "A" else second
        play = [("A", {first: 1.0}), ("B", {second: 1.0}), ("X", {x_label: 1.0})]
        return {"prompt": "Is X the same as A or as B?", "play": play, "choices": ["A", "B"], "key": key, "stimuli": self.labels}

    def results(self):
        """One abx record body, with every trial's key and answer."""
        n_correct, n_trials = self.right_answers(), len(self.answers)
        p_value = ledger.binomial_p(n_correct, n_trials)
        trials = [[answer["key"], answer["choice"]] for answer in self.answers]
        answer = {"heard": p_value < ABX_ALPHA, "p_value": p_value, "trials": trials}
        return [{"question": self.question(), "answer": answer, "n_trials": n_trials, "n_correct": n_correct, "confidence": 1.0 - p_value}]

    def summary(self):
        """One line for the page and the terminal."""
        result = self.results()[0]
        verdict = "heard" if result["answer"]["heard"] else "not shown to be heard"
        return f"{result['n_correct']}/{result['n_trials']} right, p = {result['answer']['p_value']:.3f}: {verdict}"


class PairMode(Mode):
    """Pairwise preference with "same": every pair `repeats` times in a fresh random order, then one replicate pair."""

    kind = "pair"

    def __init__(self, labels, repeats, rng, anchor=None):
        pairs = [pair for pair in itertools.combinations(labels, 2) for _ in range(repeats)]
        rng.shuffle(pairs)
        order = [pair if flip else pair[::-1] for pair, flip in zip(pairs, rng.integers(0, 2, len(pairs)))]
        self.replicate_of = replicate_index(order, anchor, rng)
        self.order = [*order, order[self.replicate_of][::-1]]
        super().__init__(labels, len(self.order))

    def trial(self, index):
        """Trial `index`: A and B, no right answer; the last trial is the replicate."""
        first, second = self.order[index]
        prompt = "Which one sounds better? Choose 'same' when you cannot tell them apart or cannot choose."
        play = [("A", {first: 1.0}), ("B", {second: 1.0})]
        replicate = index == len(self.order) - 1
        return {
            "prompt": prompt,
            "play": play,
            "choices": ["A", "B", ledger.SAME],
            "key": None,
            "stimuli": [first, second],
            "replicate": replicate,
        }

    def results(self):
        """One pair record body per pair of stimuli: the counts and every trial `[A, B, chosen, replicate]` in the order heard."""
        tallies = {}
        for answer in self.answers:
            pair = tuple(sorted(answer["stimuli"]))
            tally = tallies.setdefault(pair, {"counts": {pair[0]: 0, pair[1]: 0, ledger.SAME: 0}, "trials": []})
            tally["counts"][chosen(answer)] += 1
            tally["trials"].append([*answer["stimuli"], chosen(answer), answer["replicate"]])
        return [self._pair_record(pair, tally) for pair, tally in tallies.items()]

    def _pair_record(self, pair, tally):
        count = sum(tally["counts"].values())
        return {"question": self.question(pair), "answer": tally, "n_trials": count, "n_correct": None, "confidence": None}

    def replicate_note(self):
        """Whether the replicate drew the answer its first presentation drew, once both are in."""
        if len(self.answers) < self.total:
            return ""
        first, again = chosen(self.answers[self.replicate_of]), chosen(self.answers[-1])
        verdict = "the same answer" if first == again else f"{first}, then {again}"
        return f"replicate {' vs '.join(sorted(self.answers[-1]['stimuli']))}: {verdict}"

    def summary(self):
        """One clause per pair: how often each side and "same" were chosen; then the replicate."""
        counts = "; ".join(", ".join(f"{label} {n}" for label, n in result["answer"]["counts"].items()) for result in self.results())
        return " | ".join(filter(None, [counts, self.replicate_note()]))


def replicate_index(order, anchor, rng):
    """The position in `order` of the pair to replicate: a random pair with `anchor`, or any random pair without one."""
    candidates = [index for index, pair in enumerate(order) if anchor is None or anchor in pair]
    return int(rng.choice(candidates or list(range(len(order)))))


class BlendMode(Mode):
    """Blend continuum: which of A and B (one of them the blend at x) differs from the reference R."""

    kind = "blend"

    def __init__(self, labels, trials, rng):
        super().__init__(labels, trials)
        self.slots = [bool(flip) for flip in rng.integers(0, 2, trials)]
        self.log_posterior = np.zeros(len(QUEST_GRID))

    def quantile(self, share):
        """The threshold at `share` of the posterior."""
        posterior = np.exp(self.log_posterior - self.log_posterior.max())
        cdf = np.cumsum(posterior) / posterior.sum()
        return float(QUEST_GRID[min(int(np.searchsorted(cdf, share)), len(QUEST_GRID) - 1)])

    def next_x(self):
        """x = 1 first, then the posterior median kept inside [BLEND_X_MIN, 1]."""
        if not self.answers:
            return 1.0
        return float(np.clip(self.quantile(0.5), BLEND_X_MIN, 1.0))

    def trial(self, index):
        """Trial `index`: R, then A and B with the blend in the slot the key names."""
        incumbent, candidate = self.labels
        x = self.next_x()
        blend, plain = {incumbent: 1.0 - x, candidate: x}, {incumbent: 1.0}
        key = "A" if self.slots[index] else "B"
        pair = [("A", blend), ("B", plain)] if key == "A" else [("A", plain), ("B", blend)]
        prompt = "R is the reference. Which of A and B differs from R?"
        return {"prompt": prompt, "play": [("R", plain), *pair], "choices": ["A", "B"], "key": key, "x": x, "stimuli": self.labels}

    def record(self, trial, choice):
        """Keeps the answer and moves the posterior."""
        super().record(trial, choice)
        right = p_correct(trial["x"], QUEST_GRID)
        self.log_posterior += np.log(right if choice == trial["key"] else 1.0 - right)

    def results(self):
        """One blend record body: the x heard 75% of the time, its 90% band and every response."""
        responses = [[answer["x"], answer["choice"] == answer["key"]] for answer in self.answers]
        threshold = x_at_target(self.quantile(0.5))
        band = [x_at_target(self.quantile(share)) for share in BAND_QUANTILES]
        answer = {"threshold_x": threshold, "band": band, "heard_at_full": threshold <= 1.0, "responses": responses}
        return [
            {
                "question": self.question(),
                "answer": answer,
                "n_trials": len(responses),
                "n_correct": self.right_answers(),
                "confidence": None,
            }
        ]

    def summary(self):
        """One line: the threshold and its band."""
        answer = self.results()[0]["answer"]
        low, high = answer["band"]
        return f"heard 75% of the time at x = {answer['threshold_x']:.2f} (90% band {low:.2f}-{high:.2f})"
