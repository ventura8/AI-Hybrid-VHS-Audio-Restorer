"""E1 (2026-10-12): the calibration writes no gates file without a known ordering, and checks the gates on oracle pairs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import calibrate_quality_metrics as cal
from tests.unit.test_quality_calibration import _checks_pass, _fake_ordering, _fake_scoring, _names_in, _oracle_fixtures, _run

# The fixtures this module borrows by name from the whole-run tests.
USED_FIXTURES = (_checks_pass, _fake_ordering, _fake_scoring)


@pytest.mark.usefixtures("fake_scoring", "fake_ordering", "checks_pass")
def test_without_a_known_ordering_the_run_writes_its_reports_and_no_gates_file(tmp_path, capsys):
    """The floor rule alone would hand the hand-set guesses on as a calibration (the v3 and v3b files): no gates file."""
    assert _run(tmp_path, ordering=False) == 0
    assert _names_in(tmp_path / "out") >= {"report.json", "report.md"}
    assert not (tmp_path / "out" / "gates.json").exists()
    assert "gates files not written: no --known-ordering" in capsys.readouterr().out


@pytest.mark.usefixtures("fake_scoring", "fake_ordering", "checks_pass")
def test_a_gate_that_fails_the_oracle_pairs_stops_the_gates_files(tmp_path, monkeypatch):
    """A hard gate refusing more than 5% of the oracle pairs refuses denoising itself: no gates file, exit 1."""
    failing = {
        "pairs": 2,
        "gates": {"dsp.whistle": {"failed": 2, "share": 1.0}},
        "by_output": {},
        "failing": ["dsp.whistle"],
        "passed": False,
    }
    monkeypatch.setattr(cal, "oracle_check", lambda *_args: failing)
    assert _run(tmp_path) == 1
    report = json.loads((tmp_path / "out" / "report.json").read_text("utf-8"))
    assert report["refusal"].startswith("oracle: dsp.whistle")
    assert not (tmp_path / "out" / "gates.json").exists()


def test_the_oracle_cases_pair_each_capture_with_both_programmes(tmp_path):
    """`_vhs` against `_target` and `_clean`; a missing programme is skipped; `--oracle-max` spreads the captures."""
    fixtures = _oracle_fixtures(tmp_path)
    (fixtures / "en" / "mid00_speech_m15_clean.wav").unlink()
    cases = cal.build_oracle_cases(fixtures, ["en", "fr"])
    assert [(case.kind, case.name, case.output.name) for case in cases] == [
        ("oracle", "mid00_speech_m15__target", "mid00_speech_m15_target.wav")
    ]
    assert cases[0].case_id == "en/oracle_mid00_speech_m15__target_None"
    assert cal.spread(list(range(10)), 3) == [0, 4, 9]
    assert cal.spread([1, 2], 0) == cal.spread([1, 2], 5) == [1, 2]


def _oracle_doc(lkr):
    """A stored oracle score: its aggregate holds one lkr reading."""
    entry = {"median": lkr, "tail": lkr, "n": 2}
    return {
        "aggregate": {"dsp.lkr": {"source": {"median": 0.0, "tail": 0.0, "n": 2}, "output": entry, "delta": entry}},
        "speaker_floor": None,
    }


def _oracle_cases(kinds):
    return [cal.Case("oracle", f"mid0{index}_x__{kind}", None, "en", Path("s.wav"), Path("o.wav")) for index, kind in enumerate(kinds)]


def test_the_oracle_check_counts_every_hard_gate_failure_per_output():
    """Over 5% of the pairs failing one gate fails the check; the counts split by target and clean."""
    cases = _oracle_cases(["target", "clean", "target"])
    scores = {cases[0].case_id: _oracle_doc(2.0), cases[1].case_id: _oracle_doc(0.1), cases[2].case_id: _oracle_doc(0.2)}
    oracle = cal.oracle_check(cases, scores, cal.GATES)
    assert oracle["pairs"] == 3
    assert oracle["gates"]["dsp.lkr"] == {"failed": 1, "share": pytest.approx(1 / 3)}
    assert oracle["by_output"] == {"target": {"dsp.lkr": 1}, "clean": {}}
    assert (oracle["failing"], oracle["passed"]) == (["dsp.lkr"], False)


def test_no_oracle_pair_cannot_vouch_for_the_gates():
    """Without a single pair the check does not pass, and the refusal says why."""
    oracle = cal.oracle_check([], {}, cal.GATES)
    assert (oracle["pairs"], oracle["passed"]) == (0, False)
    args = SimpleNamespace(known_ordering=Path("k.json"))
    assert cal.refusal_reason(args, oracle).startswith("no oracle pair")
    assert cal.refusal_reason(args, {"pairs": 3, "failing": []}) is None


def test_the_oracle_check_prints_in_one_line():
    """The summary line names every gate's count, or none; no line when the check did not run."""
    assert cal.oracle_line(None) is None
    assert cal.oracle_line(cal.oracle_check([], {}, cal.GATES)) == "oracle pairs: 0; hard failures per gate: none"
    assert cal.refusal_line(None) is None


@pytest.mark.usefixtures("fake_scoring", "fake_ordering", "checks_pass")
def test_a_whole_run_checks_the_oracle_pairs_and_writes_the_gates_when_they_pass(tmp_path):
    """Both programmes of the one fixture capture are checked; nothing fails, the gates files are written, exit 0."""
    assert _run(tmp_path) == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text("utf-8"))
    assert (report["oracle"]["pairs"], report["oracle"]["passed"], report["gates_written"]) == (2, True, True)


def test_a_cached_oracle_score_must_keep_its_aggregate():
    """An oracle pair is gated from its stored aggregate: a score written before it was kept is scored again."""
    doc = {"schema": cal.SCORE_SCHEMA, "requested": ["dsp"]}
    oracle, degradation = _oracle_cases(["clean"])[0], cal.Case("degradation", "hiss", 20.0, "en", Path("s.wav"), Path("o.wav"))
    assert cal.usable(doc, degradation, ("dsp",))
    assert not cal.usable(doc, oracle, ("dsp",))
    assert cal.usable({**doc, "aggregate": {}}, oracle, ("dsp",))
