"""What makes a blind block informative: familiarisation, catch trials, block flags and the block's statistical power.

Session 0 (2026-10-09) left three gaps this module closes (backlog item E4, as salvaged):

- No positive control: a block where no difference was heard could not tell "the listener
  heard nothing" from "the listener was not listening". Catch trials are now interleaved among
  the scored ones, one per `CATCH_EVERY` scored trials (at least one), each in its own stretch of
  the block and never first. A catch trial is shaped exactly like the block's scored trials, over
  the first stimulus and a positive control (`listen_cuts.catch_cuts`: +6 dB of air at 7.5 kHz,
  levelled back to the same speech level, or half the raw tape), and it is scored apart: it
  never enters the binomial score, the pair counts or the QUEST posterior. A catch is caught
  when its key is chosen; in a pair (no right answer) when the answer is A or B and not "same".
- Block flags (`ledger.BLOCK_FLAGS`, kept in the records' `context.block_flags`): "no signal"
  when every decided scored answer was one button ("can't tell" answers aside: the page check
  of 2026-10-12 answered one "can't tell" and fifteen A, the same null), "disengaged" when a
  catch trial was missed. A
  one-button block is never "disengaged": Session 0's two all-B ABX blocks came with the
  listener's words "I chose B every time because they all sounded the same", the honest null of
  a forced choice without a "same" button, and the same listener answered "same" 20 times of 21
  in the pair blocks and picked the restoration over the raw tape 3 of 3. A 1-button rule at 80%
  (the backlog's first proposal) would have called that listener disengaged; only a missed
  positive control can.
- Familiarisation: a few trials before the scored ones (`FAMILIAR_TRIALS`), over an exaggerated
  pair (ABX: the difference to the second stimulus doubled; blend: the full candidate; threshold:
  each tone at the cap), answered with feedback and kept out of every score. Feedback inside a
  scored block would teach the listener the answers; here it only teaches what to listen for.
- Power: each record keeps `context.power`. ABX: the critical score at alpha 0.05 and the chance
  of reaching it at a true hit rate of 0.6, 0.75 and 0.9, on the decided trials (Leventhal, JAES
  1986: 16 trials miss a 60%-correct listener 83% of the time, power 0.167). Pair: the same
  binomial on the preferred side's share of one pair's trials (one or two trials per pair reach
  no significance on their own: the counts feed a Bradley-Terry fit). Blend: simulated, the share
  of `BLEND_POWER_RUNS` QUEST blocks of the same length whose 90% band ends under x = 1 (the full
  candidate shown heard) for a listener whose 75% point is x = 0.5.
"""

import math

import numpy as np

from scripts.restoration_quality import ledger, listen_modes

CATCH_EVERY = 8
FAMILIAR_TRIALS = {"abx": 2, "pair": 0, "blend": 2, "threshold": 5}
POWER_ALPHA = 0.05
POWER_HIT_RATES = (0.6, 0.75, 0.9)
BLEND_POWER_X75 = 0.5
BLEND_POWER_RUNS = 200
NO_SIGNAL, DISENGAGED = ledger.BLOCK_FLAGS
FAMILIAR, SCORED, CATCH = "familiar", "scored", "catch"


def default_catches(scored):
    """One catch trial per CATCH_EVERY scored trials, at least one."""
    return max(1, math.ceil(scored / CATCH_EVERY))


def catch_positions(scored, catches, rng):
    """After how many scored trials each catch trial comes: one in each equal stretch of the block, never before the first."""
    count = min(catches, scored)
    edges = np.linspace(0, scored, count + 1).round().astype(int)
    return sorted(int(rng.integers(low, high)) + 1 for low, high in zip(edges[:-1], edges[1:]))


def schedule(scored, familiar, catches, rng):
    """The block's presentations in order: the familiarisation, then the scored trials with the catch trials among them."""
    after = catch_positions(scored, catches, rng)
    order = [FAMILIAR] * familiar
    for done in range(1, scored + 1):
        order.append(SCORED)
        order.extend([CATCH] * after.count(done))
    return order


def caught(trial, choice):
    """A catch is caught when its key is chosen; in a pair (no key) when the answer names a side, not "same"."""
    if trial["key"] is None:
        return choice in ("A", "B")
    return choice == trial["key"]


def one_button(choices):
    """Two or more scored answers whose decided ones ("can't tell" aside) all took one button, or none was decided."""
    return len(choices) >= 2 and len(set(choices) - {listen_modes.CANT_TELL}) <= 1


def block_flags(choices, catches):
    """'no signal' for a one-button block (never 'disengaged' then); 'disengaged' when a catch was missed."""
    if one_button(choices):
        return [NO_SIGNAL]
    return [DISENGAGED] if any(not entry[3] for entry in catches) else []


def feedback_text(key, right):
    """What a familiarisation answer is told."""
    return f"{'Right' if right else 'Not this time'}: the answer was {key}."


def critical_correct(n_trials, alpha=POWER_ALPHA):
    """The fewest right answers of `n_trials` whose one-sided binomial p is under `alpha`; None when no score reaches it."""
    return next((k for k in range(n_trials + 1) if ledger.binomial_p(k, n_trials) < alpha), None)


def binomial_power(n_trials, hit_rate, alpha=POWER_ALPHA):
    """The chance that a listener right `hit_rate` of the time reaches the critical score of `n_trials`."""
    critical = critical_correct(n_trials, alpha)
    if critical is None:
        return 0.0
    tail = range(critical, n_trials + 1)
    return float(sum(math.comb(n_trials, k) * hit_rate**k * (1.0 - hit_rate) ** (n_trials - k) for k in tail))


def binomial_power_record(n_trials, test="one-sided binomial"):
    """`context.power` of a binomial question: the critical score and the power at each POWER_HIT_RATES."""
    rates = {f"{rate:g}": round(binomial_power(n_trials, rate), 4) for rate in POWER_HIT_RATES}
    return {"test": test, "alpha": POWER_ALPHA, "n": n_trials, "critical_correct": critical_correct(n_trials), "at_hit_rate": rates}


def _other(key):
    return "B" if key == "A" else "A"


def simulated_band_high(trials, threshold, rng):
    """The upper end of the 90% band a simulated listener with `threshold` leaves after a blend block of `trials`."""
    mode = listen_modes.BlendMode(["incumbent", "candidate"], trials, rng)
    for index in range(trials):
        trial = mode.trial(index)
        right = rng.random() < float(listen_modes.p_correct(trial["x"], threshold))
        mode.record(trial, trial["key"] if right else _other(trial["key"]))
    return listen_modes.x_at_target(mode.quantile(listen_modes.BAND_QUANTILES[1]))


def blend_power(trials, x75=BLEND_POWER_X75, runs=BLEND_POWER_RUNS, seed=0):
    """The share of `runs` simulated blend blocks of `trials` whose 90% band ends under x = 1, for a listener at `x75`."""
    rng = np.random.default_rng(seed)
    threshold = x75 / listen_modes.x_at_target(1.0)
    return sum(simulated_band_high(trials, threshold, rng) < 1.0 for _ in range(runs)) / float(runs)


def blend_power_record(n_trials, runs=BLEND_POWER_RUNS):
    """`context.power` of a blend block."""
    share = blend_power(n_trials, runs=runs) if n_trials else 0.0
    return {"test": "simulated QUEST", "n": n_trials, "x75": BLEND_POWER_X75, "runs": runs, "band_ends_under_full": share}


_POWER = {
    "abx": binomial_power_record,
    "pair": lambda n_trials: binomial_power_record(n_trials, "one-sided sign test on the preferred side"),
    "blend": blend_power_record,
}


def power_of(result):
    """`context.power` for one record body."""
    return _POWER[result["question"]["type"]](result["n_trials"])


class Protocol:
    """One block run through its mode: familiarisation (with feedback, never scored), the scored trials and the catch trials."""

    def __init__(self, mode, rng, familiar=0, catches=0, catch_labels=(listen_modes.CATCH_AIR,)):
        self.mode = mode
        self.rng = rng
        self.order = schedule(mode.total, familiar, catches, rng)
        labels = [catch_labels[index % len(catch_labels)] for index in range(self.order.count(CATCH))]
        self.catch_labels = [str(label) for label in rng.permutation(labels)] if labels else []
        self.position = 0
        self.familiar = []
        self.catches = []

    def current(self):
        """The kind of the presentation at hand (FAMILIAR, SCORED or CATCH), None once the block is over."""
        return self.order[self.position] if self.position < len(self.order) else None

    def finished(self):
        """True once every presentation has its answer."""
        return self.position >= len(self.order)

    def _key(self):
        return str(self.rng.choice(self.mode.keys))

    def trial(self):
        """The trial at hand, built by the mode, with its `phase` (and its `catch` label on a catch trial)."""
        kind = self.current()
        if kind == FAMILIAR:
            return {**self.mode.familiar_trial(len(self.familiar), self._key()), "phase": kind}
        if kind == CATCH:
            label = self.catch_labels[len(self.catches)]
            return {**self.mode.catch_trial(label, self._key()), "phase": kind, "catch": label}
        return {**self.mode.trial(len(self.mode.answers)), "phase": SCORED}

    def take(self, trial, choice):
        """Keeps one answer where it belongs and moves on; returns the feedback a familiarisation answer gets, else ""."""
        handlers = {FAMILIAR: self._familiar, CATCH: self._catch, SCORED: self._scored}
        feedback = handlers[trial["phase"]](trial, choice)
        self.position += 1
        return feedback

    def _familiar(self, trial, choice):
        right = choice == trial["key"]
        self.familiar.append(right)
        return feedback_text(trial["key"], right)

    def _catch(self, trial, choice):
        self.catches.append([trial["catch"], trial["key"], choice, caught(trial, choice)])
        return ""

    def _scored(self, trial, choice):
        self.mode.record(trial, choice)
        return ""

    def progress(self):
        """`{phase, number, of}` for the page: "practice" through the familiarisation, then "test"."""
        practice = self.order.count(FAMILIAR)
        if self.position < practice:
            return {"phase": "practice", "number": self.position + 1, "of": practice}
        return {"phase": "test", "number": self.position - practice + 1, "of": len(self.order) - practice}

    def context(self):
        """The block's familiarisation, catch trials and flags, as the records' `context` keeps them."""
        missed = sum(1 for entry in self.catches if not entry[3])
        return {
            "familiarisation": {"trials": len(self.familiar), "right": sum(self.familiar)},
            "catch": {"trials": self.catches, "caught": len(self.catches) - missed, "missed": missed},
            "block_flags": block_flags(self.mode.scored_choices(), self.catches),
        }

    def note(self):
        """The summary's clause on the catch trials and the block flags."""
        context = self.context()
        parts = [f"catch trials {context['catch']['caught']}/{len(self.catches)} caught" if self.catches else "no catch trials"]
        return " | ".join(parts + context["block_flags"])
