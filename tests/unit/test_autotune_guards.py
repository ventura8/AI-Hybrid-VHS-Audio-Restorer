"""The loop's ear v3 guards (plan 1.5): the verdict-reversal guard, the audibility tie and the learned median veto."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from scripts import autotune_guards as ag
from scripts import autotune_restoration as at

GRIDS = at.REPO / "scripts" / "tune_grids"
CER = "speech.cer.output.median"
AIR = "linear_air_gain_db"
RANKING = {"m": "down"}
OFFSET = ag.auditory.MASKING_OFFSET_DB


def _guards(name, floors=None):
    return ag.loop_guards(at.tune.load_grid(GRIDS / name), floors, at.ALL_KNOBS)


def _v3_floors(name):
    """A benign floor for every veto the grid lists."""
    return {reading: 0.01 for reading in at.tune.load_grid(GRIDS / name)["vetoes"]}


# ----------------------------------------------------------------------------- loading the grid


def test_a_v2_grid_brings_no_guards():
    """tata_v2 has no vetoes, reversals or audibility section, so its rounds are judged as before."""
    guards = _guards("tata_v2.yaml")
    assert guards == ag.NO_GUARDS
    assert ag.describe_guards(guards).startswith("none")


@pytest.mark.parametrize("name", ["tata_v3.yaml", "music_v3.yaml"])
def test_a_v3_grid_brings_its_vetoes_the_air_reversal_and_the_tie(name):
    """Every veto is measured in three floors; round 3's air verdict is a boundary; the tie uses the calibrated offset."""
    guards = _guards(name, _v3_floors(name))
    assert [veto.limit for veto in guards.vetoes.values()] == pytest.approx([0.03] * len(guards.vetoes))
    assert [(rule.key, rule.bound, rule.side) for rule in guards.reversals] == [(AIR, 2.0, 1.0)]
    assert guards.audibility_offset_db == OFFSET
    assert "1 verdict reversals" in ag.describe_guards(guards)


def test_a_v3_grid_without_its_noise_floors_does_not_start():
    """A veto with no benign floor would veto every move or none, so the loop refuses to run unguarded."""
    with pytest.raises(SystemExit, match="no benign floor"):
        _guards("tata_v3.yaml")


@pytest.mark.parametrize(
    ("grid", "message"),
    [
        ({"reversals": [{"key": "linear_air_gian_db", "rejected_above": 2.0}]}, "no engine tunes"),
        ({"reversals": [{"key": AIR}]}, "reversal 0"),
        ({"audibility": {"offset_db": "loud"}}, "finite offset_db"),
        ({"audibility": 24.0}, "finite offset_db"),
    ],
)
def test_a_guard_section_the_loop_cannot_apply_stops_it(grid, message):
    """A misspelt knob, a boundary without a side and an offset that is not a number each stop the loop."""
    with pytest.raises(SystemExit, match=message):
        ag.loop_guards({"ranking": RANKING, **grid}, knobs=at.ALL_KNOBS)


def test_an_empty_audibility_section_takes_the_calibrated_offset():
    """`audibility: {}` asks for the tie at auditory's own masking offset."""
    guards = ag.loop_guards({"audibility": {}})
    assert guards.audibility_offset_db == OFFSET
    assert ag.describe_guards(guards).endswith(f"audibility tie at a {OFFSET:g} dB masking offset")


def test_the_noise_floors_come_from_a_report_or_nowhere(tmp_path):
    """A reward_noise_floor report gives `{reading: floor}`; no path, a missing file or a list give nothing."""
    report = tmp_path / "noise_floor.json"
    report.write_text(json.dumps({"readings": {CER: {"floor": 0.004}}}), encoding="utf-8")
    assert ag.noise_floors(report) == {CER: 0.004}
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    assert [ag.noise_floors(path) for path in (None, tmp_path / "absent.json", tmp_path / "list.json")] == [{}, {}, {}]


# ----------------------------------------------------------------------------- the verdict-reversal guard


def test_a_move_past_a_ledger_boundary_is_refused_and_a_move_back_is_not():
    """+1 -> +2 dB of air is refused with the ledger verdict named; the incumbent and other moves are not."""
    rules = ag.reward.parse_reversals([{"key": AIR, "rejected_above": 2.0, "verdict": "r3-air-preference"}])
    everything = {"inc": {AIR: 1.0}, "up": {AIR: 2.0}, "other": {AIR: 1.0, "x": 1}}
    refused = ag.refusals(everything, {AIR: 1.0}, rules)
    assert list(refused) == ["up"]
    assert "r3-air-preference" in refused["up"][0]
    assert not ag.refusals({"down": {AIR: 1.0}}, {AIR: 2.0}, rules)


# ----------------------------------------------------------------------------- the learned median veto


def _scores(candidate_cer):
    """The incumbent and one candidate with a better rank on two tapes; the candidate's CER on t2 is `candidate_cer`."""
    incumbent = {"m": 2.0, CER: 0.010}
    candidate = {"t1": {"m": 1.0, CER: 0.010}, "t2": {"m": 1.0, CER: candidate_cer}}
    return {"inc": {"t1": incumbent, "t2": incumbent}, "c1": candidate}


VETOES = {CER: ag.reward.Veto("up", 0.012)}


def test_a_better_ranked_candidate_whose_learned_median_moved_past_its_veto_cannot_win():
    """CER 0.010 -> 0.045 on one tape: under the hard gate (0.05) and better ranked, but 0.035 past a 0.012 veto."""
    verdicts = at.judge(_scores(0.045), "inc", RANKING, VETOES)
    assert not verdicts["c1"]["qualifies"]
    assert verdicts["c1"]["vetoed"] == [f"t2 {CER} +0.035"]
    assert "vetoed" not in verdicts["inc"]


def test_a_move_inside_the_veto_and_a_v2_judge_leave_the_verdict_as_it_was():
    """0.010 -> 0.020 stays within the veto; with no vetoes (a v2 grid) the 0.045 candidate wins as it always did."""
    assert at.judge(_scores(0.020), "inc", RANKING, VETOES)["c1"]["qualifies"]
    plain = at.judge(_scores(0.045), "inc", RANKING)["c1"]
    assert plain["qualifies"]
    assert "vetoed" not in plain


def test_a_learned_reading_the_candidate_lost_vetoes_it():
    """The incumbent read CER on t2 and the candidate did not: it cannot win on what was not measured."""
    verdicts = at.judge(_scores(None), "inc", RANKING, VETOES)
    assert verdicts["c1"]["vetoed"] == [f"t2 {CER} unread"]
    assert not verdicts["c1"]["qualifies"]


def test_more_flags_still_refuse_a_candidate_the_way_the_reward_caps_it():
    """`reward.gate_constraints` reads the same counts `_beats` does: one more flag than the incumbent is not "ok"."""
    assert not ag.allowed({"failures": 0.0, "flags": 1.0}, {"failures": 0.0, "flags": 0.0})
    assert ag.allowed({"failures": 1.0, "flags": 0.0}, {"failures": 1.0, "flags": 0.0})


# ----------------------------------------------------------------------------- the audibility tie


def _voice(seconds=6.0, rate=48000):
    t = np.arange(int(seconds * rate)) / rate
    return (0.2 * np.sin(2 * np.pi * 220.0 * t) * (1.0 + 0.5 * np.sin(2 * np.pi * 3.0 * t))).astype(np.float32)


def _write(out_dir, cid, slug, samples, rate=48000):
    path = ag.candidate_wav(out_dir, cid, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, rate, subtype="FLOAT")


def _tie_set(tmp_path):
    """The incumbent and three candidates on two tapes: -120 dBFS noise everywhere, +3 dB on t2, one at another rate."""
    noise = np.float32(1e-6) * np.random.default_rng(1).standard_normal(_voice().size).astype(np.float32)
    louder = {"t1": np.float32(1.0), "t2": np.float32(1.41)}
    for slug in ("t1", "t2"):
        _write(tmp_path, "inc", slug, _voice())
        _write(tmp_path, "same", slug, _voice() + noise)
        _write(tmp_path, "louder", slug, _voice() * louder[slug])
        _write(tmp_path, "resampled", slug, _voice(), rate=44100)
    return {"t1": "t1.mov", "t2": "t2.mov"}


def test_a_candidate_inaudible_on_every_tape_ties_and_one_audible_on_a_single_tape_does_not(tmp_path):
    """-120 dBFS of noise is a tie; +3 dB on one tape is audible; a pair at another rate cannot be compared and is scored."""
    tapes = _tie_set(tmp_path)
    live = dict.fromkeys(("inc", "same", "louder", "resampled"), {})
    assert ag.find_ties(live, "inc", tapes, tmp_path, OFFSET) == {"same"}
    sidecar = ag.candidate_wav(tmp_path, "resampled", "t1").with_name("t1" + ag.AUDIBILITY_SIDECAR_SUFFIX)
    assert ag.read_json(sidecar)["audible"] is True
    assert "sample rates differ" in ag.read_json(sidecar)["error"]
    assert ag.find_ties(live, "inc", tapes, tmp_path, None) == set()


def test_the_audibility_verdict_is_reused_while_both_files_are_unchanged(tmp_path, monkeypatch):
    """A second look at the same pair reads the sidecar; the first audible tape ends the check."""
    tapes = _tie_set(tmp_path)
    compare = Mock(return_value={"audible": True, "identical": False, "nmr_max": 9.0, "audible_frac": 0.5, "event_frames": 4})
    monkeypatch.setattr(ag.auditory, "compare_files", compare)
    assert [ag.inaudible_everywhere(tmp_path, "inc", "louder", tapes, 24.0) for _ in range(2)] == [False, False]
    assert compare.call_count == 1


def test_a_verdict_an_older_rule_stored_is_read_again(tmp_path, monkeypatch):
    """auditory.VERDICT_RULE is part of the sidecar key: once it changes, a stored tie is compared again, not reused."""
    tapes = _tie_set(tmp_path)
    compare = Mock(return_value={"audible": False, "identical": False, "nmr_max": -9.0, "audible_frac": 0.0, "event_frames": 0})
    monkeypatch.setattr(ag.auditory, "compare_files", compare)
    calls = []
    for rule in (ag.auditory.VERDICT_RULE, ag.auditory.VERDICT_RULE, "next-rule"):
        monkeypatch.setattr(ag.auditory, "VERDICT_RULE", rule)
        assert ag.inaudible_everywhere(tmp_path, "inc", "same", tapes, 24.0)
        calls.append(compare.call_count)
    assert calls == [2, 2, 4]
    sidecar = ag.candidate_wav(tmp_path, "same", "t2").with_name("t2" + ag.AUDIBILITY_SIDECAR_SUFFIX)
    assert ag.read_json(sidecar)["key"].startswith("next-rule|")


# ----------------------------------------------------------------------------- a round with the guards on

GUARD_SECTIONS = {
    "vetoes": {CER: {"worse": "up"}},
    "reversals": [{"key": AIR, "rejected_above": 2.0, "verdict": "r3-air-preference"}],
    "audibility": {"offset_db": 24.0},
}
PROPOSALS = {"c_air": {AIR: 2.0}, "c_tie": {AIR: 1.0, "k": 1}, "c_veto": {AIR: 1.0, "k": 2}, "c_good": {AIR: 1.0, "k": 3}}


@pytest.fixture(name="guarded_round")
def _guarded_round(tmp_path, monkeypatch):
    """A round with a refused, a tied, a vetoed and a winning candidate: what it rendered, what it scored, its state."""
    state = {"incumbent": {AIR: 1.0}, "rounds": [], "candidates": {}}
    flats = {at.candidate_id(state["incumbent"]): {"m": 2.0, CER: 0.01}, "c_veto": {"m": 0.5, CER: 0.5}, "c_good": {"m": 1.0, CER: 0.01}}
    seen = SimpleNamespace(rendered=[], scored=[], state=state, log=tmp_path / at.LOG_MD)
    monkeypatch.setattr(at, "propose", lambda *_args: PROPOSALS)
    monkeypatch.setattr(at, "run_candidate", lambda _engine, overrides, *_args: seen.rendered.append(overrides))
    monkeypatch.setattr(at, "audio_fingerprint", lambda _out, cid, _tapes: (cid,))
    monkeypatch.setattr(ag, "inaudible_everywhere", lambda _out, _inc, cid, _tapes, _offset: cid == "c_tie")
    monkeypatch.setattr(at, "score_round", lambda live, *_args: seen.scored.extend(live) or {cid: {"t1": flats[cid]} for cid in live})
    guards = ag.loop_guards(GUARD_SECTIONS, {CER: 0.01}, at.ALL_KNOBS)
    args = SimpleNamespace(families=("dsp",), gates=None, language="ro", parallel=1, guards=guards)
    seen.moved = at.run_round(args, "apl", {"t1": "t1.mov"}, tmp_path, state, RANKING, 1)
    return seen


def test_a_guarded_round_accepts_the_best_candidate_no_guard_held_back(guarded_round):
    """The vetoed candidate ranks best but cannot win; the next one is accepted."""
    assert guarded_round.moved
    assert guarded_round.state["rounds"][0]["accepted"] == "c_good"
    verdicts = guarded_round.state["rounds"][0]["verdicts"]
    assert [verdicts[cid]["qualifies"] for cid in ("c_air", "c_tie", "c_veto", "c_good")] == [False, False, False, True]


def test_a_refused_move_is_never_rendered_and_a_tie_never_scored(guarded_round):
    """The air move stops before the render, the inaudible one before the scorer."""
    assert {AIR: 2.0} not in guarded_round.rendered
    assert not {"c_air", "c_tie"} & set(guarded_round.scored)
    verdicts = guarded_round.state["rounds"][0]["verdicts"]
    assert [bool(verdicts[cid].get(key)) for cid, key in (("c_air", "refused"), ("c_tie", "tie"), ("c_veto", "vetoed"))] == [True] * 3


GUARD_LOG_LINES = (
    "refused (verdict reversal)",
    "tie (inaudible on every tape)",
    "| 0 | vetoed |",
    "audibility tie: `c_tie`",
    "verdict reversal: `c_air`",
    "r3-air-preference",
    f'learned veto: `c_veto` {{"k": 2}} cannot win: t1 {CER} +0.49',
)


def test_every_guard_says_in_the_log_what_it_held_back_and_why(guarded_round):
    """The table cells and one line per candidate: the boundary and its ledger verdict, the tie, the tape and the move."""
    log = guarded_round.log.read_text(encoding="utf-8")
    assert [line for line in GUARD_LOG_LINES if line not in log] == []


# ----------------------------------------------------------------------------- the command line


def _cli(tmp_path, grid_text, *extra):
    tapes = tmp_path / "tapes.json"
    tapes.write_text(json.dumps({"t1": "t1.mov"}), encoding="utf-8")
    grid = tmp_path / "grid.yaml"
    grid.write_text(grid_text, encoding="utf-8")
    return ["--engine", "apl", "--tapes", str(tapes), "--grid", str(grid), "--out", str(tmp_path / "out"), *extra]


V3_GRID = "ranking:\n  m: down\nvetoes:\n  speech.cer.output.median: {worse: up}\n"


def _loop(monkeypatch):
    rounds = Mock(return_value=False)
    monkeypatch.setattr(at, "run_round", rounds)
    monkeypatch.setattr(at, "seed_defaults", lambda _engine, start, _out: start)
    monkeypatch.setattr(at.tune, "probe_duration", lambda _path: None)
    return rounds


def test_the_command_line_measures_the_vetoes_in_the_noise_floor_report(tmp_path, monkeypatch):
    """--noise-floors feeds reward.noise_floors; every round gets the grid's guards on args."""
    floors = tmp_path / "noise_floor.json"
    floors.write_text(json.dumps({"readings": {CER: {"floor": 0.004}}}), encoding="utf-8")
    rounds = _loop(monkeypatch)
    at.main(_cli(tmp_path, V3_GRID, "--noise-floors", str(floors)))
    assert rounds.call_args.args[0].guards.vetoes == {CER: ag.reward.Veto("up", pytest.approx(0.012))}


def test_the_command_line_refuses_a_v3_grid_without_floors_and_a_missing_report(tmp_path, monkeypatch):
    """No report at all names the missing floor and --families; a report path that does not exist is refused as such."""
    _loop(monkeypatch)
    monkeypatch.setattr(ag, "DEFAULT_NOISE_FLOORS", tmp_path / "absent.json")
    with pytest.raises(SystemExit, match=f"grid: veto {CER}: no benign floor in --noise-floors") as refused:
        at.main(_cli(tmp_path, V3_GRID))
    assert "--families" in str(refused.value) and "--repeats" not in str(refused.value)
    with pytest.raises(SystemExit, match="does not exist"):
        at.main(_cli(tmp_path, V3_GRID, "--noise-floors", str(tmp_path / "absent.json")))


def test_the_command_line_refuses_a_zero_floor_before_it_renders(tmp_path, monkeypatch):
    """A report whose CER floor read 0 (one repeat, no resample) stops the loop by name before the first round."""
    floors = tmp_path / "noise_floor.json"
    floors.write_text(json.dumps({"readings": {CER: {"floor": 0.0}}}), encoding="utf-8")
    rounds = _loop(monkeypatch)
    with pytest.raises(SystemExit, match=f"grid: veto {CER}: no benign floor above 0") as refused:
        at.main(_cli(tmp_path, V3_GRID, "--noise-floors", str(floors)))
    assert "--repeats 2 or more" in str(refused.value) and "resample_roundtrip" in str(refused.value)
    rounds.assert_not_called()
