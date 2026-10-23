"""Judging resolution for the self-driving loop: the tie band, the dead zones' floor, the hiss exemption, family-wise vetoes.

Who owns what in the loop's ranking path (merge of ear v4, 2026-10-12): this module owns the
ranking (each reading's distance, tie band and dead-zone floor, tape wins, the round's label and
distance table) and the family-wise learned veto. `restoration_quality/reward.py` owns the
primitives both branches share: `ReadingSpec` (distance, tie-band share, the grid's `tie`),
`Veto`, `Constraint` and the constraint status. `autotune_guards` owns what is decided outside
the ranking: the reversals before rendering, the audibility tie, the marking of the vetoes (here,
family-wise) and of the grid's `constraints:` (any one tape, explicit limits, no floor), and
`allowed`. `autotune_restoration.judge` composes them: tables, then constraints, then vetoes,
then `qualify` (`_beats` and `allowed`). Nothing is applied twice: a reading's tie forgives once,
in the band, never in its dead zone as well; a constraint never takes a floor and a veto never
reads a grid `tie`. The sweep (`tune_restoration.rank_variants`) is not the loop: it ranks a
whole group in tie buckets (`tune_restoration.tie_buckets`) where this module bands each pair.

`scripts/autotune_restoration.judge` ranks candidates per tape by a weighted mean rank over the
grid's readings. Until 2026-10-11 two candidates tied on a reading only when their two-sided
distances were exactly equal (`tune_restoration._ranks`), and a learned veto fired on any one
tape-reading past its limit. Both are replaced here (backlog E2 parts a and b, E3 part b).

The tie band (E2 b). Each v3 reading spec ranks on `max(0, |x - target| - dead_zone) / scale`.
Two candidates now tie on a reading when their distances differ by no more than the tie band
`reward.py` uses for its win rates: per side `min(floor, max(0, |x - target| + floor -
dead_zone)) / scale` (`reward.ReadingSpec.noise`), the larger side counting, with `floor` the
reading's Session 0 benign-pair floor (`experiments/reward/noise_floor_benign.json`, the p95 of
the deviation between restorations a blind listener could not tell apart), and every dead zone
is bounded from below by that floor. A candidate's rank on a reading is 1 + every candidate
ahead of it beyond the band + half of every tie: with no band that is exactly the average rank
`_ranks` gave, so without a floors file the loop judges as before. Why: the v6 APL speech round
tied 25 renders at one mean rank, and the spread of those renders on each ranked reading
(tilt 0.064 dB/oct, presence 0.051, body 0.024, sib_abs 0.43, LSD 0.15, gain ride 0.03 LU) sat
under the floors (0.85, 0.72, 1.20, 1.75, 1.01, 0.38). Re-judged with narrower zones and no
band, `expander_attack_s` 0.08 won 4 of 4 tapes on moves of 0.002-0.1 dB (E2's refutation):
with exact ranks any difference decides once a zone is narrower than the incumbent's own
distance. A rank among a group is relative, so a candidate tied with the incumbent on every
reading of a tape could still out-rank it through a third candidate, and so could one whose
only move past the band is the worse way (a sub-floor gain on one reading beating a pack the
incumbent ties, a real loss on another): a tape is a win only when the candidate out-ranks the
incumbent and moved past the band the better way on at least one reading there (`better`).
Without floors that adds nothing: out-ranking the incumbent on exact ranks needs some reading
strictly better.

The hiss exemption (the user's rule, 2026-10-11): the ear must be very critical of VHS hiss,
under speech or singing and in the pauses. Readings of hiss or residual noise (`HISS_PREFIXES`:
every `dsp.hiss_*`, `dsp.residual_noise_db`, every `dsp.gap_*`, `dsp.pause_depth_db`,
`dsp.hf_8k16k`, `mos.sigmos_noise`, and on the same rule the other pause readings, the file's
noise removed and DNSMOS's background score) never take a tie band, a dead-zone floor or a veto
limit from the benign pairs: those pairs differed grossly (residual noise floor 16.4 dB, noise
removed 19.9 dB, pause depth 5.8 dB, gap HF excess 4.5 dB) and would blind the ear to hiss.
Their floors come only from scorer repeat jitter and same-percept near-copies
(`experiments/reward/noise_floor.json`: the reference repeat, `shift_1`, `requantise_16`;
`--jitter-floors`), and a report holding a benign pair is refused there. The user's hiss flags
(`listener.hiss_programme`, `listener.hiss_pause`, `gates.USER_RULE_FIXED`) fall under the
same prefixes, so no floor reaches them either.

The hiss ear's ties (ear v4, `tune_grids/tata_v4.yaml`, `music_v4.yaml`). A v4 reading spec may
carry `tie`, in the reading's units: the p95 of same-percept transforms (a fresh realisation of
the same noise, a 0.37-sample delay, +-0.1 dB, a soxr round trip, 16-bit requantisation;
`scripts/reward_noise_floor.py --transforms hiss`, its `tie_floor` rounded up). Its band is the
larger of that tie and the reading's floor (`Floors.band`), with or without a floors file, as
`reward.tolerance` widens it. Such an entry keeps the dead zone its grid sets (`_zone_floor`): the
hiss excess is one-sided, target 0, dead zone 0, so any audible hiss is charged and the most a
tie forgives is the tie itself; a dead zone raised to the jitter floor beside a band of the tie
would forgive the jitter twice. The grids' H1 constraints take the same tie as `max_move`, so a
hiss move inside the tie is a tie in the ranking and passes H1, and one past it the worse way is
behind in the ranking and excluded by H1: the two layers agree. A jitter floor wider than the tie
(none is measured yet) widens the band alone: H1 stays at the tie, so the guard is then the
stricter layer, never the looser one.

Family-wise vetoes (E3 b). In `autotune_v6_music` a learned veto fired on any one clip-reading
at 1x a p95 floor, about 48 tests per candidate (12 clips x 4 readings): in the cathar music
loop 37 of 54 vetoes came from one clip-reading, and in the APL one 15 vetoed candidates won
at least 6 of 12 clips. Now, per veto reading, the paired deltas against the incumbent are
counted per clip as worse or better when they pass the scorer's jitter floor (still within
it), an exact one-sided sign test asks whether worse outnumbers better, Holm corrects across
the grid's veto readings at `ALPHA`, and at least `MOVE_SHARE` of the clips the incumbent read
must have moved the worse way past the veto's own limit (`floor_multiple` x the benign floor,
which the grid keeps; a lost reading counts as such a move). Below `resolvable_clips` clips
(seven at 0.05 over four to six readings) no outcome can reach the Holm threshold, so on the
four speech tapes a reading vetoes only when every clip moved the worse way past the jitter
floor, with the same share past the limit: a rule the formal test cannot back, kept so the
speech loop is not left without a learned guard; the offline re-judge reports its null rate.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.restoration_quality import reward

# Where `python -m scripts.reward_noise_floor` writes the output point's floors (reference repeat, shift_1, requantise_16).
DEFAULT_JITTER_FLOORS = Path("experiments/reward/noise_floor.json")
BENIGN_PAIR = "benign_pair"
# The readings of hiss the user named (2026-10-11): `dsp.hiss_` covers every reading of ear v4 (margins, excesses, the
# guards), `listener.hiss` the flags on them (and the display-only gap-air flag).
USER_HISS_PREFIXES = (
    "dsp.hiss_",
    "listener.hiss",
    "dsp.residual_noise_db.",
    "dsp.gap_",
    "dsp.pause_depth_db.",
    "dsp.hf_8k16k.",
    "mos.sigmos_noise.",
)
# The other readings of the residual noise, on the same rule: pause pumping and tilt, ear v4's speech-route pause depth,
# the file's noise removed, DNSMOS BAK.
SAME_RULE_PREFIXES = ("dsp.pause_", "file.noise_removed_db.", "mos.dnsmos_bak.")
HISS_PREFIXES = USER_HISS_PREFIXES + SAME_RULE_PREFIXES
ALPHA = 0.05
MOVE_SHARE = 1.0 / 3.0
UNMOVED, BANDED, MOVED = "unmoved", "banded", "moved"
STATUS_ORDER = (UNMOVED, BANDED, MOVED)
PLATEAU = "no candidate qualified: plateau"
BELOW_FLOOR = "no candidate qualified: every move below the Session 0 benign-pair floor"
INSIDE_ZONES = "no candidate qualified: every candidate level with the incumbent inside the dead zones"


def is_hiss_reading(key):
    """Whether `key` reads hiss or residual noise: its floors come only from scorer jitter and same-percept near-copies."""
    return str(key).startswith(HISS_PREFIXES)


def finite(value):
    """Whether `value` is a real, finite number (a bool is not one)."""
    return isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool) and bool(np.isfinite(value))


def _number(value):
    return float(value) if finite(value) and value > 0 else 0.0


@dataclass(frozen=True)
class Floors:
    """`benign`: the Session 0 benign-pair floors; `jitter`: scorer repeat jitter and same-percept near-copies."""

    benign: dict = field(default_factory=dict)
    jitter: dict = field(default_factory=dict)

    def floor(self, reading):
        """The tie band's and the dead zone's floor: a hiss reading's jitter floor, else the larger of the two (0 unknown)."""
        jitter = _number(self.jitter.get(reading))
        return jitter if is_hiss_reading(reading) else max(_number(self.benign.get(reading)), jitter)

    def band(self, reading, tie=0.0):
        """The tie band of `reading`: the larger of its floor and the grid's `tie` (ear v4), 0 with neither."""
        return max(self.floor(reading), _number(tie))

    def source(self, reading, tie=0.0):
        """Where `band(reading, tie)` comes from, for the round's log."""
        # `_number` makes every floor 0.0 (unknown) or positive, so `<= 0` is the exact zero test without a float equality.
        if self.band(reading, tie) <= 0.0:
            return "none"
        if _number(tie) >= self.floor(reading):
            return "grid tie"
        if is_hiss_reading(reading) or _number(self.jitter.get(reading)) > _number(self.benign.get(reading)):
            return "jitter"
        return "benign pair"


NO_FLOORS = Floors()


def veto_floors(floors):
    """`{reading: floor}` the learned vetoes are measured in: the benign-pair floor, a hiss reading's jitter floor."""
    merged = {reading: value for reading, value in floors.benign.items() if not is_hiss_reading(reading)}
    merged.update({reading: value for reading, value in floors.jitter.items() if is_hiss_reading(reading)})
    return merged


def jitter_floors(report, where=""):
    """`{reading: floor}` from a scorer-jitter report; a report holding a benign pair is refused (the hiss exemption)."""
    points = set(report.get("operating_points") or ()) | set(report.get("transforms") or ())
    if BENIGN_PAIR in points:
        raise SystemExit(
            f"--jitter-floors {where}: a benign-pair report; hiss readings take floors only from scorer repeat jitter "
            "and same-percept near-copies (python -m scripts.reward_noise_floor SOURCE=OUTPUT ... --repeats 2)"
        )
    return reward.noise_floors(report)


def describe_floors(floors):
    """One line for the console: where the ranking's floors come from."""
    if floors == NO_FLOORS:
        return "none: exact ties, every dead zone as the grid sets it (a grid `tie` still bands its reading)"
    return f"{len(floors.benign)} benign-pair readings, {len(floors.jitter)} jitter readings (hiss readings: jitter only)"


# ----------------------------------------------------------------------------- the banded ranking


@dataclass(frozen=True)
class Entry:
    """One ranking entry on one tape: each candidate's distance (None unread, lower first) and tie-band share."""

    key: str
    weight: float
    distances: dict
    shares: dict

    def band(self, first, second):
        """The pair's tie band: the larger of the two sides' shares."""
        return max(self.shares[first], self.shares[second])

    def points_behind(self, cid, other):
        """1 when `other` is ahead of `cid` beyond the band, 0.5 for a tie, 0 when `cid` is ahead."""
        return points_behind(self.distances[cid], self.distances[other], self.band(cid, other))

    def status(self, cid, other):
        """`UNMOVED` (same distance), `BANDED` (inside the tie band) or `MOVED` against `other`."""
        if self.distances[cid] == self.distances[other]:
            return UNMOVED
        # `points_behind` returns exactly 0, 0.5 or 1, so `isclose` picks out the tie just as an equality would.
        return BANDED if math.isclose(self.points_behind(cid, other), 0.5) else MOVED


def _unread_points(mine, theirs):
    if mine is None and theirs is None:
        return 0.5
    return 1.0 if mine is None else 0.0


def points_behind(mine, theirs, band):
    """1 when `theirs` beats `mine` by more than `band` (lower is better, unread is last), 0.5 for a tie, else 0."""
    if mine is None or theirs is None:
        return _unread_points(mine, theirs)
    if abs(mine - theirs) <= band:
        return 0.5
    return 1.0 if theirs < mine else 0.0


def _signed(value, direction):
    """A v1 / v2 direction entry as a lower-is-better number: such an entry keeps exact ties."""
    if value is None:
        return None
    return -float(value) if direction == "up" else float(value)


def _floored(spec, floor):
    """The spec with its dead zone bounded from below by `floor`, at unit weight (the weight multiplies the rank)."""
    return reward.ReadingSpec(spec.target, spec.family, max(spec.dead_zone, floor), spec.scale, 1.0, spec.tie)


def _zone_floor(spec, floor):
    """What bounds a spec's dead zone from below: its floor, or nothing for an entry with a grid `tie` (module docstring)."""
    return 0.0 if spec.tie > 0.0 else floor


def entry(key, spec, values, floors=NO_FLOORS):
    """The `Entry` of one ranking entry over `{cid: value}`; a bad v3 spec raises ValueError naming `key`.

    The band is `floors.band(key, tie)`: a hiss reading's jitter floor or a grid `tie`, never a benign pair's floor.
    """
    if not isinstance(spec, dict):
        return Entry(key, 1.0, {cid: _signed(v, spec) for cid, v in values.items()}, dict.fromkeys(values, 0.0))
    reading = reward.parse_grid({key: spec})[key]
    band = floors.band(key, reading.tie)
    unit = _floored(reading, _zone_floor(reading, floors.floor(key)))
    return Entry(
        key, reading.weight, {cid: unit.distance(v) for cid, v in values.items()}, {cid: unit.noise(v, band) for cid, v in values.items()}
    )


def tape_entries(flats, ranking, floors=NO_FLOORS):
    """The ranking's entries on one tape (`{cid: flat}`), leaving out those no candidate read: they would tie everyone."""
    entries = []
    for key, spec in ranking.items():
        values = {cid: flat.get(key) for cid, flat in flats.items()}
        if any(value is not None for value in values.values()):
            entries.append(entry(key, spec, values, floors))
    return entries


def entry_ranks(item):
    """`{cid: rank}` on one entry: 1 + every candidate ahead beyond the band + half of every tie."""
    return {cid: 1.0 + sum(item.points_behind(cid, other) for other in item.distances if other != cid) for cid in item.distances}


def _weighted_mean(ranks, weights):
    return float(np.average(ranks, weights=weights)) if ranks else None


def mean_ranks(entries, cids):
    """`{cid: weight-averaged rank over the entries}`, None for every candidate when nothing is ranked."""
    per_entry = [(entry_ranks(item), item.weight) for item in entries]
    weights = [weight for _ranks, weight in per_entry]
    return {cid: _weighted_mean([ranks[cid] for ranks, _weight in per_entry], weights) for cid in cids}


def _worst(statuses):
    return max(statuses, key=STATUS_ORDER.index, default=UNMOVED)


@dataclass(frozen=True)
class TapeTable:
    """One tape judged: ranks, per-reading distances, per candidate its status against the incumbent per reading, and
    `improved`: per candidate the readings it moved past the tie band the better way against the incumbent."""

    ranks: dict
    distances: dict
    status: dict
    improved: dict = field(default_factory=dict)

    def level(self, cid):
        """Whether no reading moved past its tie band against the incumbent on this tape."""
        return _worst(self.status[cid].values()) != MOVED

    def better(self, cid):
        """Whether some reading moved past its tie band the better way against the incumbent on this tape."""
        return bool(self.improved.get(cid))


def tape_table(flats, ranking, incumbent_id, floors=NO_FLOORS):
    """The `TapeTable` of `{cid: flat}` on one tape."""
    entries = tape_entries(flats, ranking, floors)
    return TapeTable(
        ranks=mean_ranks(entries, list(flats)),
        distances={cid: {item.key: item.distances[cid] for item in entries} for cid in flats},
        status={cid: {item.key: item.status(cid, incumbent_id) for item in entries} for cid in flats},
        improved=improved_readings(entries, list(flats), incumbent_id),
    )


def improved_readings(entries, cids, incumbent_id):
    """`{cid: [reading, ...]}`: the entries on which each candidate is ahead of the incumbent beyond the tie band."""
    # `points_behind` returns exactly 0, 0.5 or 1, so `isclose` picks out the full point just as an equality would.
    return {cid: [item.key for item in entries if math.isclose(item.points_behind(incumbent_id, cid), 1.0)] for cid in cids}


def reading_status(tables, cid):
    """`{reading: worst status on any tape}` of one candidate against the incumbent."""
    worst = {}
    for table in tables.values():
        for key, status in table.status[cid].items():
            worst[key] = _worst([worst.get(key, UNMOVED), status])
    return worst


def moved_tapes(tables, cid):
    """`{reading: [slug, ...]}`: the tapes on which each reading moved past its tie band against the incumbent."""
    moved = {}
    for slug, table in tables.items():
        for key, status in table.status[cid].items():
            if status == MOVED:
                moved.setdefault(key, []).append(slug)
    return moved


def tape_wins(tables, cid, incumbent_id):
    """The tapes on which `cid` out-ranks the incumbent and moved past the tie band the better way on some reading."""
    return [slug for slug, table in tables.items() if _ahead(table, cid, incumbent_id) and table.better(cid)]


def _ahead(table, cid, incumbent_id):
    rank, base = table.ranks[cid], table.ranks[incumbent_id]
    return rank is not None and base is not None and rank < base


# ----------------------------------------------------------------------------- the round's label and distances


def _held_back(verdict, base):
    """Whether something other than the ranking kept a candidate out: a reversal, a veto, a grid constraint (ear v4), more
    failures or flags."""
    if verdict.get("refused") or verdict.get("vetoed") or verdict.get("constrained"):
        return True
    return any(verdict.get(key, 0.0) > base.get(key, 0.0) for key in ("failures", "flags"))


def _held(verdicts, incumbent_id):
    """The candidates a reversal, a veto or the gates kept out."""
    base = verdicts.get(incumbent_id, {})
    return [cid for cid, verdict in verdicts.items() if _held_back(verdict, base)]


def _scored(verdicts, incumbent_id):
    return {cid: v for cid, v in verdicts.items() if cid != incumbent_id and v.get("status") is not None}


def unmoved_readings(verdicts, incumbent_id):
    """`(unmoved, banded)`: how many ranked readings no scored candidate moved at all, and how many it moved inside the band."""
    statuses = {}
    for verdict in _scored(verdicts, incumbent_id).values():
        for key, status in verdict["status"].items():
            statuses[key] = _worst([statuses.get(key, UNMOVED), status])
    counted = list(statuses.values())
    return counted.count(UNMOVED), counted.count(BANDED)


def plateau_label(verdicts, incumbent_id, floors=NO_FLOORS):
    """The outcome of a round nobody won: below the floor when every scored candidate is level with the incumbent and
    nothing else held one back; otherwise a plateau, with how many candidates were level, moved and lost, or held back."""
    scored = _scored(verdicts, incumbent_id)
    if not scored:
        return PLATEAU
    held, level = _held(verdicts, incumbent_id), _level(scored)
    if held or len(level) < len(scored):
        lost = len(set(scored) - set(level) - set(held))
        counts = f"{len(level)} of {len(scored)} scored candidates level with the incumbent, {lost} moved and lost, {len(held)} held back"
        return f"{PLATEAU} ({counts})"
    return _below_floor(verdicts, incumbent_id, floors)


def _level(scored):
    """The scored candidates level with the incumbent on every tape and reading."""
    return [cid for cid, verdict in scored.items() if verdict.get("level") is not False]


def _below_floor(verdicts, incumbent_id, floors):
    """The label of a round whose every scored candidate stayed level with the incumbent."""
    unmoved, banded = unmoved_readings(verdicts, incumbent_id)
    head = BELOW_FLOOR if floors.benign else INSIDE_ZONES
    inside = f"; {banded} moved less than their floor" if banded else ""
    return f"{head} ({unmoved} readings unmoved{inside})"


def _cell(distances, moved):
    """Per-tape distances joined by '/', a '*' where the reading moved past its band against the incumbent."""
    return "/".join(("-" if value is None else f"{value:.3g}") + ("*" if slug in moved else "") for slug, value in distances)


def _reading_cells(verdict, keys):
    per_tape = verdict.get("distances") or {}
    moved = verdict.get("moved") or {}
    return [_cell([(slug, flat.get(key)) for slug, flat in per_tape.items()], moved.get(key, ())) for key in keys]


def _ranked_keys(ordered):
    """Every reading some scored candidate carries a distance for, in the order they first appear."""
    keys = {}
    for _cid, verdict in ordered:
        for flat in (verdict.get("distances") or {}).values():
            keys.update(dict.fromkeys(flat))
    return list(keys)


def _reading_specs(ranking):
    """The ranking's v3 entries (`{reading: spec dict}`), the v1 / v2 direction entries left out."""
    return {key: spec for key, spec in (ranking or {}).items() if isinstance(spec, dict)}


def grid_ties(ranking):
    """`{reading: tie}` of the ranking's v3 entries that carry a grid `tie` (ear v4); a bad spec raises ValueError."""
    return {key: spec.tie for key, spec in reward.parse_grid(_reading_specs(ranking)).items() if spec.tie > 0.0}


def _table_head(keys, floors, ties):
    head = ["", "per-reading distances (per tape; * moved past the tie band against the incumbent):", ""]
    head += ["| candidate | change | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    cells = (f"{floors.band(key, ties.get(key, 0.0)):.3g} ({floors.source(key, ties.get(key, 0.0))})" for key in keys)
    return head + ["| floor | | " + " | ".join(cells) + " |"]


def distance_lines(ordered, described, floors=NO_FLOORS, ranking=None):
    """The round's per-reading distance table: one row per scored candidate, the band and its source under each reading
    (the floor, or the grid `tie` of `ranking` where it is wider)."""
    keys = _ranked_keys(ordered)
    if not keys:
        return []
    rows = [(cid, verdict) for cid, verdict in ordered if verdict.get("distances")]
    return _table_head(keys, floors, grid_ties(ranking)) + [
        f"| {cid} | {described(cid)} | " + " | ".join(_reading_cells(verdict, keys)) + " |" for cid, verdict in rows
    ]


# ----------------------------------------------------------------------------- the family-wise learned veto


@dataclass(frozen=True)
class ClipMoves:
    """One veto reading across the clips the incumbent read: worse / better past the jitter floor, worse past the limit."""

    clips: int = 0
    worse: int = 0
    better: int = 0
    past_limit: int = 0


def _signed_move(veto, value, base):
    """The move the worse way (positive is worse), or None when the candidate lost the reading."""
    return reward.WORSE_SIGN[veto.worse] * (float(value) - float(base)) if finite(value) else None


def _clip(veto, value, base, jitter):
    """`(worse, better, past_limit)` of one clip; a lost reading is worse and past the limit."""
    move = _signed_move(veto, value, base)
    if move is None:
        return 1, 0, 1
    return int(move > jitter), int(move < -jitter), int(move > veto.limit)


def clip_moves(tape_scores, incumbent_scores, reading, veto, jitter=0.0):
    """`ClipMoves` of `reading` over every tape the incumbent read it on."""
    counts = [0, 0, 0, 0]
    for slug, flat in tape_scores.items():
        base = (incumbent_scores.get(slug) or {}).get(reading)
        if finite(base):
            worse, better, past = _clip(veto, (flat or {}).get(reading), base, jitter)
            counts = [counts[0] + 1, counts[1] + worse, counts[2] + better, counts[3] + past]
    return ClipMoves(*counts)


def sign_test_p(worse, better):
    """The exact one-sided sign test: P(at least `worse` of `worse + better` fair coin flips land worse); 1 with no move."""
    total = worse + better
    if total == 0:
        return 1.0
    return sum(math.comb(total, k) for k in range(worse, total + 1)) / 2.0**total


def holm(pvalues, alpha=ALPHA):
    """The keys Holm's step-down procedure rejects at family-wise level `alpha`."""
    rejected = set()
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    for index, (key, pvalue) in enumerate(ordered):
        if pvalue > alpha / (len(ordered) - index):
            break
        rejected.add(key)
    return rejected


def resolvable_clips(readings, alpha=ALPHA):
    """The fewest clips on which a sign test can reach Holm's first threshold over `readings` veto readings."""
    return math.ceil(math.log2(max(1, readings) / alpha))


def _enough_moved(moves):
    return moves.clips > 0 and moves.past_limit >= math.ceil(moves.clips * MOVE_SHARE)


def _trips(moves, rejected, readings):
    """Whether one reading's moves veto: enough clips past the limit, and Holm's rejection or, too few clips, unanimity."""
    if not _enough_moved(moves):
        return False
    return rejected if moves.clips >= resolvable_clips(readings) else moves.worse == moves.clips


def _why(reading, moves, pvalue, readings):
    rule = f"sign test p {pvalue:.3g}, Holm over {readings}" if moves.clips >= resolvable_clips(readings) else "unanimous (too few clips)"
    return f"{reading} worse on {moves.worse} of {moves.clips} clips ({moves.better} better), {moves.past_limit} past the limit; {rule}"


def familywise_vetoes(tape_scores, incumbent_scores, vetoes, floors=NO_FLOORS):
    """One line per veto reading the family-wise rule trips for a candidate against the incumbent across the clips."""
    moves = {
        reading: clip_moves(tape_scores, incumbent_scores, reading, veto, _number(floors.jitter.get(reading)))
        for reading, veto in vetoes.items()
    }
    pvalues = {reading: sign_test_p(m.worse, m.better) for reading, m in moves.items()}
    rejected = holm(pvalues)
    return [_why(reading, m, pvalues[reading], len(vetoes)) for reading, m in moves.items() if _trips(m, reading in rejected, len(vetoes))]
