"""The self-driving loop's ear v3 guards (plan 1.5): what a v3 grid adds to `autotune_restoration`'s verdict.

A v3 grid (`tune_grids/tata_v3.yaml`, `music_v3.yaml`) carries three sections the loop
applies; a v1 or v2 grid has none of them (`NO_GUARDS`) and its rounds are judged as before:

- `reversals:` the verdict-reversal guard, one ledger boundary per entry
  (`{key, rejected_above | rejected_below, verdict}`, `reward.parse_reversals`). A move that
  heads the rejected way and ends at the value or past it (`linear_air_gain_db` up to 2.0 or
  above: round 3 heard +1 dB over +2 dB and off) is refused before it is rendered; a move back
  towards the accepted side never is. A key no engine tunes stops the loop as a typo.
- `audibility:` the audibility tie (principle 6). After the hash, each live candidate is
  compared with the incumbent per tape by `auditory.compare_files` (the null test and the
  noise-to-mask ratio at the section's `offset_db`), stopping at the first audible tape; the
  verdict is kept in `cands/<cid>/<slug>.audibility.json` while both files, the offset and
  auditory's verdict rule (`auditory.VERDICT_RULE`, bumped with every change to what reads
  audible) are unchanged, so a tie an older rule called is read again, never reused. A
  candidate no tape tells apart by ear is a tie: not scored, never accepted. A
  pair the check cannot compare (other sample rates) reads audible, so it is scored. The check
  decodes whole files (about 1.3 GB per hour of stereo per copy) while no scorer runs; on
  synthetic stereo it took about 1 s per 2 min of audio.
- `vetoes:` the learned median veto (principle 7). A candidate whose median of a listed learned
  judge moved the `worse` way from the incumbent's, on some tape, by more than
  `floor_multiple` x that reading's benign floor, or lost a reading the incumbent has, cannot
  win: `reward.gate_constraints(learned_veto=...)` must read "ok" beside `_beats`. The floors
  come from a `scripts/reward_noise_floor.py` report of the shipped outputs (`--noise-floors`,
  default `DEFAULT_NOISE_FLOORS`); a veto with no floor, or a floor of 0, stops the loop before
  the first render, so a v3 grid never runs unguarded. The refusal says which: a missing floor
  points to `--families` (the report reads only the families it was run with, dsp by
  default), a zero floor to `--repeats 2` or more or `resample_roundtrip`
  (`reward.NO_FLOOR`, `reward.NO_BENIGN_FLOOR`).

Each refused, tied or vetoed candidate gets a cell in the round's table and a line in
`log.md` with the reason: the boundary and its ledger verdict, or the tape, the reading and
its move.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.restoration_quality import audio_io, auditory, reward

AUDIBILITY_SIDECAR_SUFFIX = ".audibility.json"
# What the audibility sidecar keeps of `auditory.compare_files` (the per-window lists stay out).
AUDIBILITY_FIELDS = ("audible", "identical", "nmr_max", "audible_frac", "event_frames")
# Where `scripts/reward_noise_floor.py` writes the benign floors the learned vetoes are measured in.
DEFAULT_NOISE_FLOORS = Path("experiments/reward/noise_floor.json")
# The verdict cell of a candidate set aside before scoring, by the key its verdict carries.
ASIDE_CELLS = (
    ("inert", "inert (= `{}`)"),
    ("tie", "tie (inaudible on every tape)"),
    ("refused", "refused (verdict reversal)"),
)
# One log line per candidate a guard set aside or vetoed, by the key its verdict carries.
GUARD_FINDINGS = (
    ("tie", "audibility tie: `{cid}` {move} is inaudible from the incumbent on every tape; not scored"),
    ("refused", "verdict reversal: `{cid}` {move} refused, not rendered: {why}"),
    ("vetoed", "learned veto: `{cid}` {move} cannot win: {why}"),
)


@dataclass(frozen=True)
class Guards:
    """What a v3 grid adds to the verdict.

    `vetoes` is `{reading: reward.Veto}` (the `vetoes:` section with its floors), `reversals`
    `(reward.Reversal, ...)` (the `reversals:` section), `audibility_offset_db` the masking
    offset of the audibility tie (the `audibility:` section), None for no tie.
    """

    vetoes: dict = field(default_factory=dict)
    reversals: tuple = ()
    audibility_offset_db: float | None = None


NO_GUARDS = Guards()


def read_json(path):
    """The JSON object stored at `path`, or {} when the file is missing, does not parse or holds anything but an object."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def candidate_wav(out_dir, cid, slug):
    """Where the loop keeps a candidate's audio of one tape."""
    return Path(out_dir) / "cands" / cid / f"{slug}.wav"


# ----------------------------------------------------------------------------- loading the grid


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and bool(np.isfinite(value))


def _audibility_offset(section):
    """The masking offset the grid's `audibility:` mapping asks for (auditory's own by default), or None without the section."""
    if section is None:
        return None
    offset = section.get("offset_db", auditory.MASKING_OFFSET_DB) if isinstance(section, dict) else None
    if not _finite_number(offset):
        raise ValueError(f"audibility must be a mapping with a finite offset_db: {section!r}")
    return float(offset)


def loop_guards(grid, floors=None, knobs=()):
    """The grid's `vetoes:`, `reversals:` and `audibility:` sections, checked; SystemExit names what is wrong.

    `floors` is `reward.noise_floors(report)`: a veto is measured in its reading's benign floor,
    so a grid with a veto and no floor above 0 for it stops the loop. `knobs` are the settings the loop
    tunes; a reversal naming none of them would never fire, so it is refused as a typo.
    """
    try:
        guards = Guards(
            reward.parse_vetoes(grid.get("vetoes"), floors or {}),
            reward.parse_reversals(grid.get("reversals")),
            _audibility_offset(grid.get("audibility")),
        )
    except ValueError as error:
        raise SystemExit(f"grid: {error}") from error
    unknown = sorted({rule.key for rule in guards.reversals} - set(knobs))
    if unknown:
        raise SystemExit(f"grid: reversals name settings no engine tunes: {unknown}")
    return guards


def noise_floors(path):
    """`{reading: floor}` from a `scripts/reward_noise_floor.py` report, or {} without one."""
    return reward.noise_floors(read_json(path)) if path else {}


def describe_guards(guards):
    """One line for the console: what the grid adds to the verdict."""
    if guards == NO_GUARDS:
        return "none (the grid has no vetoes, reversals or audibility section)"
    tie = "off" if guards.audibility_offset_db is None else f"at a {guards.audibility_offset_db:g} dB masking offset"
    return f"{len(guards.vetoes)} learned vetoes, {len(guards.reversals)} verdict reversals, audibility tie {tie}"


# ----------------------------------------------------------------------------- before rendering: the ledger's boundaries


def refusals(everything, incumbent, reversals):
    """`{cid: [why, ...]}` for every candidate a ledger boundary refuses; never the incumbent, which moves nothing."""
    refused = {}
    for cid, overrides in everything.items():
        why = [rule.describe(overrides.get(rule.key)) for rule in reward.reversals_refusing(overrides, incumbent, reversals)]
        if why:
            refused[cid] = why
    return refused


# ----------------------------------------------------------------------------- before scoring: the audibility tie


def _audibility(incumbent_wav, wav, offset_db):
    """`auditory.compare_files`' verdict without its per-window lists; a pair it cannot compare reads audible, so it is scored."""
    try:
        found = auditory.compare_files(incumbent_wav, wav, offset_db)
    except ValueError as error:
        return {"audible": True, "error": str(error)}
    return {key: found[key] for key in AUDIBILITY_FIELDS}


def audibility_key(incumbent_wav, wav, offset_db):
    """What a stored verdict stands on: auditory's verdict rule, both files and the masking offset."""
    return f"{auditory.VERDICT_RULE}|{audio_io.file_key(incumbent_wav)}|{audio_io.file_key(wav)}|{offset_db:g}"


def cached_audible(incumbent_wav, wav, offset_db):
    """Whether `wav` sounds different from `incumbent_wav`, kept beside `wav` while `audibility_key` is unchanged."""
    key = audibility_key(incumbent_wav, wav, offset_db)
    sidecar = wav.with_name(wav.stem + AUDIBILITY_SIDECAR_SUFFIX)
    cached = read_json(sidecar)
    if cached.get("key") != key or not isinstance(cached.get("audible"), bool):
        cached = {"key": key, **_audibility(incumbent_wav, wav, offset_db)}
        sidecar.write_text(json.dumps(cached), encoding="utf-8")
    return cached["audible"]


def inaudible_everywhere(out_dir, incumbent_id, cid, tapes, offset_db):
    """Whether no tape tells the candidate's audio from the incumbent's by ear; stops at the first audible tape."""
    pairs = ((candidate_wav(out_dir, incumbent_id, slug), candidate_wav(out_dir, cid, slug)) for slug in tapes)
    return not any(cached_audible(incumbent_wav, wav, offset_db) for incumbent_wav, wav in pairs)


def find_ties(live, incumbent_id, tapes, out_dir, offset_db):
    """The candidates inaudible from the incumbent on every tape; none when the grid asks for no tie (`offset_db` None)."""
    if offset_db is None:
        return set()
    return {cid for cid in live if cid != incumbent_id and inaudible_everywhere(out_dir, incumbent_id, cid, tapes, offset_db)}


# ----------------------------------------------------------------------------- judging: the learned median veto


def _move(flat, base, reading):
    value = (flat or {}).get(reading)
    return "unread" if value is None else f"{float(value) - float(base[reading]):+.3g}"


def learned_vetoes(tape_scores, incumbent_scores, vetoes):
    """`["<slug> <reading> <move>", ...]` for every tape and learned reading whose median moved past its veto."""
    found = []
    for slug, flat in tape_scores.items():
        base = incumbent_scores.get(slug) or {}
        found += [f"{slug} {reading} {_move(flat, base, reading)}" for reading in reward.vetoed_readings(flat, base, vetoes)]
    return found


def mark_vetoes(verdict, tape_scores, incumbent_scores, vetoes):
    """Adds `vetoed` to a candidate's verdict when a learned judge vetoes it on some tape."""
    vetoed = learned_vetoes(tape_scores, incumbent_scores, vetoes or {})
    if vetoed:
        verdict["vetoed"] = vetoed


def _gate_counts(verdict):
    return {reward.HARD_KEY: verdict.get("failures", 0.0), reward.FLAG_KEY: verdict.get("flags", 0.0)}


def allowed(verdict, base):
    """reward's constraint status is "ok": no more hard failures or listener flags than the incumbent, and no learned veto."""
    constraints = reward.gate_constraints(_gate_counts(verdict), _gate_counts(base), learned_veto=bool(verdict.get("vetoed")))
    return constraints.status() == "ok"


# ----------------------------------------------------------------------------- the log


def aside_cell(verdict):
    """The verdict cell of a candidate set aside before scoring (inert, a tie, refused), or "" for a scored one."""
    return next((text.format(verdict[key]) for key, text in ASIDE_CELLS if verdict.get(key)), "")


def qualifies_cell(verdict):
    """The verdict cell of a scored candidate."""
    if verdict.get("vetoed"):
        return "vetoed"
    return "yes" if verdict["qualifies"] else "-"


def _why(reasons):
    return "; ".join(reasons) if isinstance(reasons, list) else ""


def guard_findings(ordered, described):
    """One line per candidate the audibility tie, a verdict reversal or a learned veto kept from winning, with the reason.

    `ordered` is `[(cid, verdict)]`; `described(cid)` is the candidate's move as the log shows it.
    """
    return [
        text.format(cid=cid, move=described(cid), why=_why(verdict[key]))
        for key, text in GUARD_FINDINGS
        for cid, verdict in ordered
        if verdict.get(key)
    ]
