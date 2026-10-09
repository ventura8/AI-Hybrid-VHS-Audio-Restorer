"""experiments/tata_listen/check_verdicts.py: the ear's flags against the ledger, and how much of the ledger a run asserted."""

import importlib.util
import json
import sys
from pathlib import Path

from scripts.restoration_quality import gates, ledger
from tests.unit.test_restoration_quality_ledger import _base

CHECK_PATH = Path(__file__).resolve().parents[2] / "experiments" / "tata_listen" / "check_verdicts.py"
THIN = "listener.sibilance_thin"
HISS = "listener.hiss"
NOT_ASSERTED = "not asserted (longer rankings, trials, empty flag records)"


def _check_module():
    spec = importlib.util.spec_from_file_location("check_verdicts", CHECK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scores(directory, slug, variants):
    """A minimal scored report: `{label: (listener flags, hard failure count)}`."""
    directory.mkdir(parents=True, exist_ok=True)
    body = {label: {"listener_flags": flags, "hard_failures": ["gate"] * hard} for label, (flags, hard) in variants.items()}
    (directory / f"{slug}.json").write_text(json.dumps({"variants": body}), encoding="utf-8")


def _on_soti(record):
    record["tape"] = "soti"
    return record


def _verdict_ledger(path, *extra):
    """x thin, y accepted with no complaint, z clean of dead air, w heard; y preferred to x; then `extra` records."""
    flags = _base(
        "v-flags",
        "flag",
        {"flagged": {THIN: ["x"]}, "clean": {ledger.ANY_FLAG: ["y"], "listener.dead_air": ["z"]}},
        ("x", "y", "z"),
    )
    flags["context"] = {"heard": ["w"]}
    preference = _base("v-pref", "preference", {"order": [["y"], ["x"]]}, ("y", "x"))
    ledger.append_many([_on_soti(flags), _on_soti(preference), *extra], path)
    return path


def _run(tmp_path, capsys):
    code = _check_module().main(["--ledger", str(tmp_path / "v.jsonl"), "--scores", str(tmp_path / "scores")])
    return code, capsys.readouterr().out


def test_check_verdicts_reproduces_a_synthetic_ledger(tmp_path, capsys):
    """Every verdict holds: exit 0, both records counted as checked, a label no record names printed as unjudged."""
    _verdict_ledger(tmp_path / "v.jsonl")
    _scores(tmp_path / "scores", "soti", {"x": ([THIN], 2), "y": ([], 1), "z": ([THIN], 0), "v": ([], 0), "w": ([], 0)})
    code, out = _run(tmp_path, capsys)
    assert code == 0 and "every checked listener verdict is reproduced" in out
    assert f"2 verdict records checked, 0 {NOT_ASSERTED}, 0 on display-only gates alone, 0 waiting for a report" in out
    assert "soti: v (unjudged): -" in out and "soti: w: -" in out


def test_check_verdicts_names_each_broken_verdict(tmp_path, capsys):
    """A missing flag, a flagged accepted output, a clean flag raised and a reversed preference each get a line."""
    _verdict_ledger(tmp_path / "v.jsonl")
    _scores(tmp_path / "scores", "soti", {"x": ([], 1), "y": (["listener.dead_air"], 3), "z": (["listener.dead_air"], 0)})
    code, out = _run(tmp_path, capsys)
    assert code == 1 and "soti: x should be flagged listener.sibilance_thin, has nothing" in out
    assert "soti: y was accepted by ear but carries ['listener.dead_air']" in out
    assert "soti: z was heard clean of listener.dead_air but carries it" in out
    assert "soti: y fails more hard gates (3) than x (1)" in out


def test_check_verdicts_without_scores_checks_nothing(tmp_path, capsys):
    """No scored report: exit 2 and "nothing checked", never "reproduced"."""
    _verdict_ledger(tmp_path / "v.jsonl")
    code, out = _run(tmp_path, capsys)
    assert code == 2 and "no scores under" in out and "nothing checked" in out
    assert "reproduced" not in out and "2 waiting for a report: v-flags, v-pref" in out


def test_check_verdicts_whose_reports_score_no_judged_label_checks_nothing(tmp_path, capsys):
    """Reports exist but score other labels (or another tape): the records wait, exit 2."""
    _verdict_ledger(tmp_path / "v.jsonl")
    _scores(tmp_path / "scores", "soti", {"other": ([], 0)})
    _scores(tmp_path / "scores", "vaccin", {"x": ([THIN], 0)})
    code, out = _run(tmp_path, capsys)
    assert code == 2 and "0 verdict records checked" in out and "2 waiting for a report" in out


def test_a_two_tier_ranking_is_asserted_and_a_longer_one_is_not(tmp_path, capsys):
    """y over both x and z (two tiers) is one verdict and is checked; a three-tier rank is counted as not asserted."""
    two_tier = _on_soti(_base("v-two", "preference", {"order": [["y"], ["x", "z"]]}, ("x", "y", "z")))
    ranking = _on_soti(_base("v-rank", "preference", {"order": [["z"], ["y"], ["x"]]}, ("x", "y", "z")))
    _verdict_ledger(tmp_path / "v.jsonl", two_tier, ranking)
    _scores(tmp_path / "scores", "soti", {"x": ([THIN], 0), "y": ([], 2), "z": ([], 1)})
    code, out = _run(tmp_path, capsys)
    assert code == 1 and "soti: y fails more hard gates (2) than z (1)" in out and "z fails more" not in out
    assert f"3 verdict records checked, 1 {NOT_ASSERTED}, 0 on display-only gates alone, 0 waiting" in out


def _hiss_record(rid, flagged, clean):
    """A soti flag record judging `listener.hiss` only (plus `clean`'s other gates)."""
    return _on_soti(_base(rid, "flag", {"flagged": {HISS: flagged} if flagged else {}, "clean": clean}, ("x", "y", "a")))


def test_a_flag_on_a_display_only_gate_is_reported_not_asserted(tmp_path, capsys):
    """Hiss is display-only: x flagged hissy carries only its thin flag (a new report), y heard clean of hiss and a,
    accepted with no complaint, still carry it (a stored report); none of the three fails, and the run lists the flags
    it skipped."""
    hiss = _hiss_record("v-hiss", ["x"], {HISS: ["y"], ledger.ANY_FLAG: ["a"]})
    _verdict_ledger(tmp_path / "v.jsonl", hiss)
    variants = {"x": ([THIN], 2), "y": ([HISS], 1), "z": ([], 0), "a": ([HISS], 0)}
    _scores(tmp_path / "scores", "soti", variants)
    code, out = _run(tmp_path, capsys)
    assert code == 0 and "every checked listener verdict is reproduced" in out
    assert f"3 verdict records checked, 0 {NOT_ASSERTED}, 0 on display-only gates alone, 0 waiting for a report" in out
    assert out.rstrip().splitlines()[-2].endswith("; 1 display-only flags not asserted: v-hiss listener.hiss")
    assert "soti: y: listener.hiss" in out and "soti: a: listener.hiss" in out


def test_the_same_report_fails_the_flag_once_its_gate_is_asserted():
    """With nothing display-only (the v2 rule) the hiss verdicts above are checked again and each one breaks."""
    check = _check_module()
    hiss = _hiss_record("v-hiss", ["x"], {HISS: ["y"], ledger.ANY_FLAG: ["a"]})
    report = {"variants": {"x": {"listener_flags": [], "hard_failures": []}}}
    report["variants"].update({label: {"listener_flags": [HISS], "hard_failures": []} for label in ("y", "a")})
    assert check.check_report("soti", report, ledger, [hiss]) == [
        "soti: x should be flagged listener.hiss, has nothing",
        "soti: a was accepted by ear but carries ['listener.hiss']",
        "soti: y was heard clean of listener.hiss but carries it",
    ]
    assert check.check_report("soti", report, ledger, [hiss], frozenset({HISS})) == []


def test_a_flag_record_on_display_only_gates_alone_is_counted_on_its_own(tmp_path, capsys):
    """A record judging only hiss is never checked or waited for, and is not counted with the rankings and trials (an
    empty flag record is); alone in the ledger, nothing is checked (exit 2)."""
    ledger.append_many([_hiss_record("v-hiss", ["x"], {HISS: ["y"]}), _hiss_record("v-empty", [], {})], tmp_path / "v.jsonl")
    _scores(tmp_path / "scores", "soti", {"x": ([HISS], 0), "y": ([], 0)})
    code, out = _run(tmp_path, capsys)
    assert code == 2 and f"0 verdict records checked, 1 {NOT_ASSERTED}, 1 on display-only gates alone, 0 waiting for a report" in out
    assert "1 display-only flags not asserted: v-hiss listener.hiss" in out


def test_the_coverage_tally_files_a_display_only_record_apart_from_the_unasserted_ones():
    """status() calls the hiss-only record display-only alone and the empty one not asserted; with nothing display-only
    the hiss record waits for a report like any flag record."""
    check = _check_module()
    records = [_hiss_record("v-hiss", ["x"], {HISS: ["y"]}), _hiss_record("v-empty", [], {})]
    tally = check.coverage(ledger, records, {}, frozenset({HISS}))
    assert (tally[check.DISPLAY_ONLY_ALONE], tally[check.NOT_ASSERTED], tally[check.WAITING]) == (["v-hiss"], ["v-empty"], [])
    assert check.coverage(ledger, records, {})[check.WAITING] == ["v-hiss"]


def test_the_display_only_gates_are_the_demoted_ones_and_the_soft_ones():
    """listener.hiss and every soft gate (never counted as a flag) are skipped; the listener flags still checked are not."""
    display_only = _check_module().display_only_gates(gates)
    assert HISS in display_only and "file.lufs" in display_only
    assert not display_only & {name for name, gate in gates.GATES.items() if gate.severity == gates.FLAG}


def test_check_verdicts_finds_the_ledger_module_when_run_as_a_script(monkeypatch):
    """Run as a script, the repository is not on the path yet; the loader puts it there."""
    module = _check_module()
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != str(module.REPO)])
    assert module.ledger_module() is ledger
    assert str(module.REPO) in sys.path


def test_the_shipped_ledger_against_the_stored_reports_says_what_it_checked(capsys):
    """On the repository's own ledger and reports the run states its coverage and never claims more than it checked."""
    code = _check_module().main([])
    out = capsys.readouterr().out
    assert code in (0, 1, 2) and "verdict records checked" in out and "waiting for a report" in out
    assert "on display-only gates alone" in out
    assert (code == 2) == ("nothing checked" in out)
    assert "should be flagged listener.hiss" not in out and "clean of listener.hiss" not in out
