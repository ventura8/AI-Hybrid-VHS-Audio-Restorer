"""The sensitivity checks of the calibration: what each degradation declares, tested on the scored cases.

Per (degradation, reading) the table in `scripts/quality_degradations.py` declares one of:

- "up" / "down": the reading must point that way at the severest level, order the three
  levels on at least 80 % of the languages, and clear three benign floors at the mildest;
- "flat": the reading must not move. Each level's |median - benign centre| must stay within
  the expectation's tolerance or three benign floors, whichever is larger; with a `twin` it
  must stay strictly under the twin's own move instead. The twin form exists because a
  "must not move" can be false in absolute terms and still be the point: the app's +2 dB
  air shelf at 7.5 kHz moves the net sibilance reading 0.25 dB on the en speech fixture
  where the absolute reading moves 0.40 (over 44 realistic-v2 speech fixtures and six levels
  the net reading moves a median 53 % of the absolute one, at most 96 %), because the plain
  frames' own 4-12 kHz energy sits below the shelf's rise. What R2 adds is that the absolute
  twin reads the shelf in full; the net reading lagging behind it is what is asserted;
- "match": a source reading must read the condition the degradation built, within
  `tolerance` octaves of the level or of the untouched material's own reading when that is
  lower (`base.*`): a 10 kHz low pass cannot lift a programme band that ends at 4 kHz.

A reading missing at some level makes the check "unscored": nothing is asserted on it. A
"match" check is unscored only when no case carries the source reading (the dsp family, which
reads R0, did not run); a case without it while others carry it is a miss, R0 read no band
there. An unscored check says why (`reason`): "clipped" when the runner's clip of R1 to R0's
programme band leaves the reading no band of R1's own layout on any case, the widest band read
included (`balance_metrics.bands_under`: air's first band needs a band of 5292 Hz, presence
2125 Hz, the tilt's third 1408 Hz; the realistic-v2 music beds read 1414 Hz, the speech
targets 1.1-5.0 kHz), "not read" otherwise.

The benign floor leaves out what a benign case changes by construction
(`quality_degradations.benign_changes`): the shifts delay the raw pair, so the raw-pair sync
readings read the shift itself. With them in, shift_5ms / 30ms / 60ms (5.0 / 30.0 / 60.0 ms)
set a 57 ms floor under `file.sync_offset_ms` on realistic-v2 en + fr, and the floor rule
relaxed the 40 ms sync-offset gate to 172 ms.
"""

import numpy as np

from scripts import quality_degradations as deg
from scripts.restoration_quality import balance_metrics

FLOOR_PERCENTILE = 95.0
EFFECT_MULTIPLE = 3.0
MONOTONIC_SHARE = 0.8
BASE_PREFIX = "base."
META_PREFIX = "meta."
PROGRAMME_BAND = "meta.prog_bandwidth_hz"
# The runner reads R1 up to R0's programme band only, so a source whose band leaves a reading no
# band of the runner's layout (`balance_metrics.readable_under`) leaves it unread by design.
R1_PREFIX = "dsp."
CLIPPED = "clipped"
NOT_READ = "not read"


def _cases_of(cases, kind, name=None):
    """The cases of one kind, narrowed to one degradation when `name` is given."""
    return [c for c in cases if c.kind == kind and name in (None, c.name)]


def noise_floor(cases, readings):
    """Per metric: the benign centre (median delta) and floor (p95 |delta - centre|) over every benign case."""
    values = {}
    for case in _cases_of(cases, "benign"):
        for metric, value in _floor_readings(readings.get(case.case_id, {}), deg.benign_changes(case.name)):
            values.setdefault(metric, []).append(value)
    return {metric: _centre_and_floor(v) for metric, v in values.items()}


def _floor_readings(readings, changed=()):
    """The (metric, number) pairs a benign floor is taken over: changes only, not the profile or base readings, nor `changed`.

    `changed` names the readings the benign case moves by construction (a shift's raw-pair sync lag).
    """
    skipped = (BASE_PREFIX, META_PREFIX)
    return [(m, v) for m, v in readings.items() if isinstance(v, (int, float)) and not m.startswith(skipped) and m not in changed]


def _centre_and_floor(values):
    centre = float(np.median(values))
    return {"centre": centre, "floor": float(np.percentile(np.abs(np.asarray(values) - centre), FLOOR_PERCENTILE))}


def _level_values(cases, readings, name, metric):
    """`{level: [reading per language]}` for one degradation and metric, levels in table order."""
    out = {level: [] for level in deg.DEGRADATIONS[name].levels}
    for case in _cases_of(cases, "degradation", name):
        value = readings.get(case.case_id, {}).get(metric)
        if value is not None:
            out[case.level].append(value)
    return out


def _level_medians(per_level):
    """The median reading per level, None where a level has no readings."""
    return [float(np.median(v)) if v else None for v in per_level.values()]


def _signed(direction, value):
    return value if direction == "up" else -value


def _monotonic_share(per_level, direction, centre):
    """Share of languages on which the three levels are strictly ordered the expected way."""
    per_language = list(zip(*per_level.values()))
    if not per_language:
        return 0.0
    ordered = [all(_signed(direction, b - centre) > _signed(direction, a - centre) for a, b in zip(seq, seq[1:])) for seq in per_language]
    return float(np.mean(ordered))


def _status(direction, monotonic, effect_mild):
    return "pass" if direction and monotonic >= MONOTONIC_SHARE and effect_mild >= EFFECT_MULTIPLE else "fail"


def _reference(floor, metric):
    return floor.get(metric, {"centre": 0.0, "floor": 0.0})


def _moving(name, expectation, cases, readings, floor):
    """The up / down check: direction at the severest level, monotonic across levels, mild level over three floors."""
    per_level = _level_values(cases, readings, name, expectation.metric)
    medians = _level_medians(per_level)
    if None in medians:
        return {"status": "unscored"}
    reference = _reference(floor, expectation.metric)
    signed = [_signed(expectation.direction, m - reference["centre"]) for m in medians]
    monotonic = _monotonic_share(per_level, expectation.direction, reference["centre"])
    effect_mild = float(signed[0] / (reference["floor"] + 1e-9))
    direction = bool(signed[-1] > 0)
    verdict = {"direction": direction, "monotonic_share": monotonic, "effect_mild": effect_mild, "medians": medians}
    return {**verdict, "status": _status(direction, monotonic, effect_mild)}


def _moves(name, metric, cases, readings, floor):
    """Per level, |median - benign centre|; None where a level has no reading."""
    centre = _reference(floor, metric)["centre"]
    return [None if m is None else abs(m - centre) for m in _level_medians(_level_values(cases, readings, name, metric))]


def _allowed(expectation, floor, twin_moves):
    """Per level, how far a "flat" reading may move: its tolerance or three floors, or under its twin's move."""
    if expectation.twin:
        return twin_moves
    bound = max(expectation.tolerance or 0.0, EFFECT_MULTIPLE * _reference(floor, expectation.metric)["floor"])
    return [bound] * len(twin_moves)


def _flat(name, expectation, cases, readings, floor):
    """The must-not-move check: every level's move inside the allowance (strictly under the twin's move when it has one)."""
    moves = _moves(name, expectation.metric, cases, readings, floor)
    twin = _moves(name, expectation.twin, cases, readings, floor) if expectation.twin else moves
    allowed = _allowed(expectation, floor, twin)
    if None in moves or None in allowed:
        return {"status": "unscored"}
    return {"moves": moves, "allowed": allowed, "status": "pass" if _inside(moves, allowed, bool(expectation.twin)) else "fail"}


def _inside(moves, allowed, strict):
    """Every move within its allowance (strictly under it against a twin)."""
    return all(move < bound if strict else move <= bound for move, bound in zip(moves, allowed))


def _matches(value, level, base, octaves):
    """Whether a source reading sits within `octaves` of what the condition makes it: the level, or the base's own if lower."""
    if value is None or not value > 0:
        return False
    expected = min(float(level), base) if base else float(level)
    return abs(np.log2(value / expected)) <= octaves


def _match_rows(name, metric, cases, readings):
    """`(source reading, level, base reading)` per case of the degradation."""
    key = BASE_PREFIX + metric.split(".", 1)[1]
    found = [(case.level, readings.get(case.case_id, {})) for case in _cases_of(cases, "degradation", name)]
    return [(read.get(metric), level, read.get(key)) for level, read in found]


def _match(name, expectation, cases, readings, _floor):
    """The source-condition check: the reading matches the level on most languages; unscored when no case carries it."""
    rows = _match_rows(name, expectation.metric, cases, readings)
    values = [value for value, _level, _base in rows]
    if all(value is None for value in values):
        return {"status": "unscored"}
    share = float(np.mean([_matches(*row, expectation.tolerance) for row in rows]))
    return {"match_share": share, "values": values, "status": _match_status(share)}


def _match_status(share):
    return "pass" if share >= MONOTONIC_SHARE else "fail"


CHECKS = {"up": _moving, "down": _moving, "flat": _flat, "match": _match}


def _source_band(name, cases, readings):
    """The widest programme band R0 read on the degradation's sources, None when it read none.

    The widest, not the median: with `--excerpts` the fixture and tape cases of one degradation
    pool under its name, and the median of en (3000 Hz), Tele7abc (4490), SOTI (5040) and Vaccin
    (7127) is 4765 Hz, which called an air reading Vaccin failed to give 'clipped'. 'clipped'
    means no case's band leaves the reading a band of R1's layout.
    """
    bands = [readings.get(case.case_id, {}).get(PROGRAMME_BAND) for case in _cases_of(cases, "degradation", name)]
    found = [band for band in bands if band is not None]
    return float(max(found)) if found else None


def _r1_reading(metric):
    """The R1 reading a metric names (`dsp.balance_air_db` -> `balance_air_db`), None for any other metric."""
    name = metric.removeprefix(R1_PREFIX)
    return name if metric.startswith(R1_PREFIX) and name in balance_metrics.READING_SPANS_HZ else None


def why_unscored(name, metric, cases, readings):
    """Why a check read nothing: `clipped` when R0's widest programme band leaves the reading no band of R1's layout, else `not read`.

    The layout is the runner's (`balance_metrics.bands_under`), not the reading's nominal start:
    under a 5.0-5.29 kHz band air has no band, though its span starts at 5 kHz. `reading_from_hz`
    is the lowest band under which the layout carries the reading.
    """
    band, reading = _source_band(name, cases, readings), _r1_reading(metric)
    if band is None or reading is None or balance_metrics.readable_under(reading, band):
        return {"reason": NOT_READ}
    return {"reason": CLIPPED, "programme_band_hz": band, "reading_from_hz": balance_metrics.readable_from_hz(reading)}


def check_expectation(name, expectation, cases, readings, floor):
    """The declared check for one (degradation, reading); a reading that is not read makes it 'unscored', with the reason."""
    head = {"degradation": name, "metric": expectation.metric, "expect": expectation.direction, "blind": expectation.blind}
    result = CHECKS[expectation.direction](name, expectation, cases, readings, floor)
    if result["status"] == "unscored":
        result = {**result, **why_unscored(name, expectation.metric, cases, readings)}
    return {**head, **result}


def run_checks(cases, readings, floor):
    """Every declared (degradation, reading) check of the table."""
    return [
        check_expectation(name, expectation, cases, readings, floor)
        for name, spec in deg.DEGRADATIONS.items()
        for expectation in spec.expects
    ]
