"""Group reward for candidate settings: two-sided distance per reading, pairwise win rate per family, worst family.

Given the flat readings of a group of candidates rendered from the same material (the
`{metric.side.stat: value}` dicts `runner.aggregates_for_tuning` returns) and a grid
`{reading: {target, dead_zone, scale, weight, family}}`, every candidate gets:

1. per reading, the two-sided distance from what the listener accepted,
   `weight * max(0, |x - target| - dead_zone) / scale` (lower is better). A reading the
   candidate lacks while another compared candidate has it LOSES: it counts as the worst
   distance read in the group plus one scale unit (`fill_missing`). Only a reading nobody
   in the group has is left out;
2. per family, its within-group pairwise win rate: each other candidate is compared on the
   sum of the distances of the family's readings (1 win, 0.5 tie, 0 loss). The tie band is
   the test-retest noise of those readings (`noise_floors`, from
   `scripts/reward_noise_floor.py`), so a difference identical audio can produce is a tie;
3. its reward: the win rate of its WORST family, over every family rated in the group.

Why this shape (ear v3 plan, 3.0; facts as measured in the repository):

- Two-sided, never raw direction. `tune_grids/tata_v2.yaml` ranks
  `dsp.hf_4k8k.delta.median: up` and the HF gates in `gates.py` are one-sided (-8 / -12 dB),
  so a brighter top could only ever rank better. The +2 dB air shelf lifted the APL output
  ~1.6 dB above 4 kHz on every frame; the user picked +1 dB over +2 dB and off by ear
  (round 3, 2026-10-08). Inside the dead zone two candidates read 0 and tie.
- Win rates with a tie band, not ranks or std-normalised scores. Both make an inaudible
  difference as decisive as an audible one: the v3 sibilance loop (mean rank per tape)
  accepted moves whose difference from the v2 plateau sat at -70..-100 dBFS on 0.5-32 % of
  samples, and the reward must call that pair a tie (`check_verdicts.py` acceptance, plan
  3.0). A win rate is bounded, scale-free and turns a sub-noise difference into exactly 0.5.
- The worst family decides (Pref-GRPO style). A candidate cannot buy residual-noise readings
  with timbre or sibilance damage: it is only as good as the family it does worst on.
- An unread reading loses, never drops out. `sibilance.texture_db` reads None when the
  output's 4 kHz+ band is silent, so a candidate that wipes out the fricatives loses that
  reading. While a missing reading was skipped, a probe group (sibilance texture and gap
  residual, the dull candidate's texture None) ranked the dull candidate first at 1.0
  against 0.5 for the fair one: an optimiser (BO or GRPO, plan 3.2) would learn to make
  readings unread. A reading that failed for one candidate only (a learned family's error)
  loses the same way: conservative, the candidate cannot win on what was not measured.
- The tie band follows the dead zone. A reading's share for one side is the most its
  clipped distance can move under a reading change of `floor`:
  `weight * min(floor, max(0, |x - target| + floor - dead_zone)) / scale`. A side deeper
  than `floor` inside the dead zone adds nothing, and the pair takes the larger of the two
  shares. Summing the full `weight * floor / scale` of every reading instead let two
  readings parked inside their dead zones tie away a difference of eight floors on a third
  (probe of 2026-10-09). The v3 grids put dead zones around the accepted target, so
  near-incumbent candidates would tie on whole families and the GRPO advantage
  (win rate - 0.5) would vanish. The shares are still summed linearly across a family's
  readings: conservative, since the median and tail of one metric are correlated.
- Constraints are not reward terms. Hard gates, listener flags, the audibility tie
  (`auditory.py`, plan 1.2), the verdict-reversal guard (a move across a ledger boundary in
  the rejected direction, e.g. air past +2 dB) and the learned-judge median veto (beyond 3x
  their benign floor) arrive as booleans per candidate (`Constraints`). A summed penalty is
  something an optimiser can trade against other terms; a constraint cannot be bought:
    * hard gate, verdict reversal, learned veto -> "excluded": left out of the comparisons
      (it gives nobody a free win) and rewarded 0.0;
    * inaudible -> "tie": left out of the comparisons and rewarded 0.5, exactly the group
      baseline, so it moves nothing; it cannot win and is never shown to the listener;
    * more listener flags than the incumbent -> "capped": compared (its audio is real) but
      its reward is capped at 0.5, so it never gains an advantage;
    * nobody to be compared with, or no grid reading read by anyone -> "unrated": 0.5,
      cannot win.

Two of those booleans come from a grid's own sections, which `autotune_restoration` applies
(the v1 and v2 grids have neither):

- `vetoes:` `{reading: {worse: up|down, floor_multiple}}` (`parse_vetoes`, `vetoed_readings`):
  a learned judge's median on a tape moved the `worse` way from the incumbent's by more than
  `floor_multiple` x its benign floor (`noise_floors` of a `scripts/reward_noise_floor.py`
  report). A reading the incumbent has and the candidate lost vetoes too, for the reason a
  missing reading loses above; one the incumbent lacks vetoes nothing. A veto without a floor
  above 0 is refused: it would veto every move or none. The refusal names the cause. A
  reading the report lacks (`NO_FLOOR`) was not scored: the report reads only the families
  it was run with, dsp alone by default, so it names `--families`. A floor of 0 or below
  (`NO_BENIGN_FLOOR`) was read with no jitter: a discrete reading (CER) measured at
  `--repeats 1` can show none at all, and a limit of 0 would then veto any move the worse
  way, Whisper's own included, so it names `--repeats 2` or more and `resample_roundtrip`.
- `reversals:` `[{key, rejected_above | rejected_below, verdict}]` (`parse_reversals`,
  `reversals_refusing`): a ledger boundary. `rejected_above: 2.0` on `linear_air_gain_db`
  says the user rejected +2 dB for a lower setting (round 3, air +1 over +2), so a move that
  raises the knob and ends at 2.0 or above is refused, before it is rendered; a move back
  towards the accepted side is never refused.

`distances` also gives the per-reading features a preference head (`preference.py`) fits
on when the listener's preference is two-sided. numpy only; nothing here reads a file.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np

NEUTRAL = 0.5
EXCLUDED_REWARD = 0.0
REQUIRED_KEYS = ("target", "family")
# key -> (default, smallest allowed value, whether that value itself is allowed)
NUMBER_KEYS = {"dead_zone": (0.0, 0.0, True), "scale": (1.0, 0.0, False), "weight": (1.0, 0.0, False)}
FIXED_REWARD = {"excluded": EXCLUDED_REWARD, "tie": NEUTRAL}
COMPARED = ("ok", "capped")
HARD_KEY = "gates.hard_failures"
FLAG_KEY = "gates.listener_flags"
# A learned judge's `worse` direction as the sign of a worse move.
WORSE_SIGN = {"up": 1.0, "down": -1.0}
DEFAULT_FLOOR_MULTIPLE = 3.0
# A veto's reading absent from the report (or not finite there): the report reads only the
# families it was run with, and `scripts/reward_noise_floor.py` runs dsp alone by default.
NO_FLOOR = (
    "no benign floor in --noise-floors (run python -m scripts.reward_noise_floor on the shipped outputs with the family "
    "that reads it in --families, speech, mos or stems for a learned judge; the default is dsp only)"
)
# A floor of 0 or below: the reading was read but showed no jitter.
NO_BENIGN_FLOOR = (
    "no benign floor above 0 (run python -m scripts.reward_noise_floor on the shipped outputs with --repeats 2 or more, "
    "or with resample_roundtrip in --transforms, so the learned reading's jitter is measured)"
)
# A ledger boundary's rejected side as the sign of a move towards it.
REJECTED_SIDES = {"rejected_above": 1.0, "rejected_below": -1.0}


@dataclass(frozen=True)
class ReadingSpec:
    """One grid entry: where the listener's accepted value sits, how wide the 'same' band is, and its weight."""

    target: float
    family: str
    dead_zone: float = 0.0
    scale: float = 1.0
    weight: float = 1.0

    def distance(self, value):
        """`weight * max(0, |value - target| - dead_zone) / scale`, or None for a missing or non-finite reading."""
        if not _finite(value):
            return None
        return self.weight * max(0.0, abs(float(value) - self.target) - self.dead_zone) / self.scale

    def noise(self, value, floor):
        """This reading's tie-band share at `value`: the most a reading change of `floor` can move its distance.

        0 when `value` sits deeper than `floor` inside the dead zone, or is missing.
        """
        if not _finite(value):
            return 0.0
        room = abs(float(value) - self.target) + float(floor) - self.dead_zone
        return self.weight * min(float(floor), max(0.0, room)) / self.scale


@dataclass(frozen=True)
class Constraints:
    """Per-candidate constraints, each a boolean the caller measured; none of them is a reward term."""

    hard_gate: bool = False
    listener_flag: bool = False
    inaudible: bool = False
    verdict_reversal: bool = False
    learned_veto: bool = False

    def status(self):
        """One of excluded, tie, capped or ok (see the module docstring)."""
        if any((self.hard_gate, self.verdict_reversal, self.learned_veto)):
            return "excluded"
        if self.inaudible:
            return "tie"
        return "capped" if self.listener_flag else "ok"


@dataclass(frozen=True)
class CandidateReward:
    """One candidate's reward, why, and whether it may win."""

    candidate: str
    reward: float
    status: str
    eligible: bool
    worst_family: str | None = None
    win_rates: dict = field(default_factory=dict)


def _finite(value):
    return isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool) and bool(np.isfinite(value))


def _missing_keys(entry):
    return [key for key in REQUIRED_KEYS if key not in entry]


def _in_range(value, minimum, inclusive):
    if not _finite(value):
        return False
    return value >= minimum if inclusive else value > minimum


def _out_of_range(entry):
    """The first number key outside its range, or None."""
    for key, (default, minimum, inclusive) in NUMBER_KEYS.items():
        if not _in_range(entry.get(key, default), minimum, inclusive):
            return key
    return None


def entry_error(entry):
    """Why a grid entry is unusable, or None."""
    missing = _missing_keys(entry)
    if missing:
        return f"missing {', '.join(missing)}"
    if not _finite(entry["target"]):
        return "target must be a finite number"
    bad = _out_of_range(entry)
    return None if bad is None else f"{bad} out of range: {entry[bad]!r}"


def _parsed_entry(reading, entry):
    if isinstance(entry, ReadingSpec):
        return entry
    error = entry_error(entry)
    if error:
        raise ValueError(f"grid entry {reading}: {error}")
    return _spec(entry)


def parse_grid(grid):
    """`{reading: ReadingSpec}` from `{reading: {target, family, dead_zone?, scale?, weight?}}`; ValueError names a bad entry."""
    return {reading: _parsed_entry(reading, entry) for reading, entry in grid.items()}


def _spec(entry):
    numbers = {key: float(entry.get(key, default)) for key, (default, _minimum, _inclusive) in NUMBER_KEYS.items()}
    return ReadingSpec(target=float(entry["target"]), family=str(entry["family"]), **numbers)


def distances(flat, grid, weighted=True):
    """`{reading: two-sided distance or None}` for one candidate; `weighted=False` drops the weight (preference features)."""
    specs = parse_grid(grid)
    return {reading: _unweighted(spec, weighted).distance(flat.get(reading)) for reading, spec in specs.items()}


def _unweighted(spec, weighted):
    return spec if weighted else ReadingSpec(spec.target, spec.family, spec.dead_zone, spec.scale, 1.0)


def noise_floors(report):
    """`{reading: floor}` from a `scripts/reward_noise_floor.py` report."""
    return {reading: float(entry["floor"]) for reading, entry in report.get("readings", {}).items() if _finite(entry.get("floor"))}


def tolerance(flat, grid, floors=None):
    """`{reading: tie-band share}` of one candidate at its own readings; a reading without a floor adds nothing."""
    floors = floors or {}
    return {reading: spec.noise(flat.get(reading), floors[reading]) for reading, spec in parse_grid(grid).items() if reading in floors}


def readings_by_family(specs):
    """`{family: [reading, ...]}` in grid order."""
    families = {}
    for reading, spec in specs.items():
        families.setdefault(spec.family, []).append(reading)
    return families


def _read(dists, reading):
    return [dist[reading] for dist in dists.values() if dist.get(reading) is not None]


def _penalties(dists, specs):
    """`{reading: worst distance read in the group + one scale unit}` for every reading at least one candidate has."""
    penalties = {}
    for reading, spec in specs.items():
        read = _read(dists, reading)
        if read:
            penalties[reading] = max(read) + spec.weight
    return penalties


def _filled(dist, penalties):
    return {**dist, **{reading: value for reading, value in penalties.items() if dist.get(reading) is None}}


def fill_missing(dists, grid):
    """`{candidate: distances}` with every reading a candidate lacks but another one has set to the group's worst + one scale unit.

    A reading is None when the candidate destroyed what it reads (`sibilance.texture_db` on a
    silent fricative band), so the missing side loses that reading instead of skipping it.
    """
    penalties = _penalties(dists, parse_grid(grid))
    return {candidate: _filled(dist, penalties) for candidate, dist in dists.items()}


def _common(dist_a, dist_b, readings):
    return [reading for reading in readings if dist_a.get(reading) is not None and dist_b.get(reading) is not None]


def _outcome(gap, band):
    """Points for the first candidate when the second one's summed distance exceeds its own by `gap`."""
    if abs(gap) <= band:
        return NEUTRAL
    return 1.0 if gap > 0 else 0.0


def _band(common, band_a, band_b):
    """The pair's tie band: per reading, the larger of the two sides' shares, summed over the readings compared."""
    return float(np.sum([max(band_a.get(reading, 0.0), band_b.get(reading, 0.0)) for reading in common]))


def pair_points(dist_a, dist_b, readings, band_a=None, band_b=None):
    """Points for `a` against `b` on one family (1 / 0.5 / 0), or None when no reading is read on both.

    `band_a` and `band_b` are each side's tie-band shares per reading (`tolerance`).
    """
    common = _common(dist_a, dist_b, readings)
    if not common:
        return None
    gap = float(np.sum([dist_b[reading] - dist_a[reading] for reading in common]))
    return _outcome(gap, _band(common, band_a or {}, band_b or {}))


def _record(points, pair, family, score):
    if score is None:
        return
    first, second = pair
    points[first].setdefault(family, []).append(score)
    points[second].setdefault(family, []).append(1.0 - score)


def _pairwise_points(dists, bands, families):
    points = {candidate: {} for candidate in dists}
    for first, second in itertools.combinations(dists, 2):
        for family, readings in families.items():
            _record(points, (first, second), family, pair_points(dists[first], dists[second], readings, bands[first], bands[second]))
    return points


def family_win_rates(group, grid, floors=None):
    """`{candidate: {family: win rate}}` for the flat readings `{candidate: flat}`; a family nobody has a reading in is absent."""
    specs = parse_grid(grid)
    dists = fill_missing({candidate: distances(flat, specs) for candidate, flat in group.items()}, specs)
    bands = {candidate: tolerance(flat, specs, floors) for candidate, flat in group.items()}
    points = _pairwise_points(dists, bands, readings_by_family(specs))
    return {candidate: {family: float(np.mean(scores)) for family, scores in fams.items()} for candidate, fams in points.items()}


def _as_constraints(value):
    if isinstance(value, Constraints):
        return value
    return Constraints(**(value or {}))


def _statuses(group, constraints):
    constraints = constraints or {}
    return {candidate: _as_constraints(constraints.get(candidate)).status() for candidate in group}


def _worst(rates):
    """(family, win rate) of the worst family; ties go to the family listed first."""
    if not rates:
        return None, None
    family = min(rates, key=rates.get)
    return family, rates[family]


def _reward(candidate, status, rates):
    if status in FIXED_REWARD:
        return CandidateReward(candidate, FIXED_REWARD[status], status, False)
    family, value = _worst(rates)
    if value is None:
        return CandidateReward(candidate, NEUTRAL, "unrated", False)
    value = min(value, NEUTRAL) if status == "capped" else value
    return CandidateReward(candidate, value, status, status == "ok", family, dict(rates))


def group_rewards(group, grid, constraints=None, floors=None):
    """`{candidate: CandidateReward}` for one group scored on the same material.

    `group` is `{candidate: flat readings}`; `constraints` maps a candidate to a `Constraints`
    or a dict of its booleans (absent: unconstrained); `floors` is `noise_floors(report)`.
    """
    specs = parse_grid(grid)
    statuses = _statuses(group, constraints)
    compared = {candidate: group[candidate] or {} for candidate in group if statuses[candidate] in COMPARED}
    rates = family_win_rates(compared, specs, floors)
    return {candidate: _reward(candidate, statuses[candidate], rates.get(candidate, {})) for candidate in group}


def ranked(rewards):
    """The candidates that may win, best reward first (ties keep the group's order)."""
    return sorted((entry for entry in rewards.values() if entry.eligible), key=lambda entry: -entry.reward)


def _count(flat, key):
    value = (flat or {}).get(key, 0.0)
    return float(value) if _finite(value) else 0.0


def gate_constraints(flat, incumbent, *, inaudible=False, verdict_reversal=False, learned_veto=False):
    """`Constraints` for a candidate: the harness gate counts against the incumbent's, plus what the caller measured.

    Mirrors `autotune_restoration._beats`: no more hard failures and no more listener flags
    than the incumbent (`gates.hard_failures`, `gates.listener_flags` in the flat readings).
    """
    return Constraints(
        hard_gate=_count(flat, HARD_KEY) > _count(incumbent, HARD_KEY),
        listener_flag=_count(flat, FLAG_KEY) > _count(incumbent, FLAG_KEY),
        inaudible=bool(inaudible),
        verdict_reversal=bool(verdict_reversal),
        learned_veto=bool(learned_veto),
    )


@dataclass(frozen=True)
class Veto:
    """One learned judge's veto: the direction that is worse and how far a median may move that way."""

    worse: str
    limit: float

    def trips(self, value, incumbent):
        """Whether `value` moved the worse way from `incumbent` by more than the limit, or was lost while `incumbent` was read."""
        if not _finite(incumbent):
            return False
        if not _finite(value):
            return True
        return WORSE_SIGN[self.worse] * (float(value) - float(incumbent)) > self.limit


def _benign_floor(floors, reading):
    """The reading's benign floor when one was measured above 0, else None: a zero limit would veto any worse move."""
    floor = floors.get(reading)
    return float(floor) if _in_range(floor, 0.0, False) else None


def _floor_error(floors, reading):
    """NO_FLOOR when the report lacks the reading (or holds no number for it), NO_BENIGN_FLOOR at 0 or below, else None."""
    if not _finite(floors.get(reading)):
        return NO_FLOOR
    return None if _benign_floor(floors, reading) is not None else NO_BENIGN_FLOOR


def _veto_error(entry, floors, reading):
    if not isinstance(entry, dict) or entry.get("worse") not in WORSE_SIGN:
        return "worse must be up or down"
    if not _in_range(entry.get("floor_multiple", DEFAULT_FLOOR_MULTIPLE), 0.0, False):
        return "floor_multiple must be a positive number"
    return _floor_error(floors, reading)


def _veto(reading, entry, floors):
    error = _veto_error(entry, floors, reading)
    if error:
        raise ValueError(f"veto {reading}: {error}")
    return Veto(entry["worse"], float(entry.get("floor_multiple", DEFAULT_FLOOR_MULTIPLE)) * _benign_floor(floors, reading))


def parse_vetoes(section, floors):
    """`{reading: Veto}` from a grid's `vetoes:` mapping and `noise_floors(report)`; ValueError names a bad entry."""
    if not isinstance(section or {}, dict):
        raise ValueError("vetoes must map a reading to {worse, floor_multiple}")
    return {reading: _veto(reading, entry, floors) for reading, entry in (section or {}).items()}


def vetoed_readings(flat, incumbent, vetoes):
    """The readings whose veto `flat` trips against the incumbent's flat readings on the same material."""
    flat, incumbent = flat or {}, incumbent or {}
    return [reading for reading, veto in vetoes.items() if veto.trips(flat.get(reading), incumbent.get(reading))]


@dataclass(frozen=True)
class Reversal:
    """A ledger boundary: `key` at `bound` and beyond, on the `side` the user rejected (+1 above, -1 below)."""

    key: str
    bound: float
    side: float
    verdict: str = ""

    def refuses(self, value, current):
        """Whether moving the knob from `current` to `value` heads the rejected way and ends at the bound or past it."""
        if not _finite(value):
            return False
        heads = not _finite(current) or self.side * (float(value) - float(current)) > 0
        return heads and self.side * (float(value) - self.bound) >= 0

    def describe(self, value):
        """Why a move to `value` is refused, for the loop's log."""
        where = "above" if self.side > 0 else "below"
        return f"{self.key} {value} is at or {where} {self.bound:g}, rejected by ear ({self.verdict or 'ledger'})"


def _rejected_side(entry):
    sides = [name for name in REJECTED_SIDES if name in entry]
    return sides[0] if len(sides) == 1 and _finite(entry[sides[0]]) else None


def _reversal(index, entry):
    side = _rejected_side(entry) if isinstance(entry, dict) else None
    if side is None or not entry.get("key"):
        raise ValueError(f"reversal {index}: needs a key and one number, rejected_above or rejected_below: {entry!r}")
    return Reversal(str(entry["key"]), float(entry[side]), REJECTED_SIDES[side], str(entry.get("verdict", "")))


def parse_reversals(section):
    """`(Reversal, ...)` from a grid's `reversals:` list; ValueError names a bad entry."""
    if not isinstance(section or [], list):
        raise ValueError("reversals must be a list of {key, rejected_above | rejected_below, verdict}")
    return tuple(_reversal(index, entry) for index, entry in enumerate(section or []))


def reversals_refusing(overrides, incumbent, reversals):
    """The reversals a candidate's settings trip against the incumbent's (a key either leaves out is the app's default)."""
    return [rule for rule in reversals if rule.refuses(overrides.get(rule.key), incumbent.get(rule.key))]
