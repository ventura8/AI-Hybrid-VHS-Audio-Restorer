"""The calibration report: every check type renders, the gates files are written, and the previous round is read back."""

import json

from scripts import calibration_report as report_mod

ROW = {"degradation": "d", "metric": "m", "blind": False}


def _report():
    checks = [
        {
            **ROW,
            "expect": "up",
            "status": "pass",
            "direction": True,
            "monotonic_share": 1.0,
            "effect_mild": 4.0,
            "medians": [1.0, 2.0, 3.0],
        },
        {**ROW, "expect": "flat", "status": "fail", "moves": [0.1, 0.2, 0.4], "allowed": [0.3, 0.3, 0.3]},
        {**ROW, "expect": "match", "status": "pass", "match_share": 1.0, "values": [4000.0, None, 2828.4]},
        {**ROW, "expect": "down", "status": "unscored", "blind": True},
        {**ROW, "expect": "up", "status": "unscored", "reason": "not read"},
        {**ROW, "expect": "up", "status": "unscored", "reason": "clipped", "programme_band_hz": 1414.2, "reading_from_hz": 5000.0},
    ]
    gate = {"threshold": -22.0, "severity": "hard", "source": "known ordering", "n_verdicts": 2, "rounds_agreeing": 2, "rounds": ["1", "4"]}
    record = {"rules": {"ranked_on": 0}, "scores": {"a": None}, "verdict_source": "ledger", "result": {"variants": {}}}
    return {
        "round": "1+2",
        "checks": checks,
        "families": {"dsp": {"ok": 3}, "mos": {"unavailable": 3}},
        "floor": {"m": {"centre": 0.0, "floor": 0.1}},
        "ordering_records": {"tele": record},
        "ordering_after_records": {"tele": record},
        "gates": {"listener.dead_air": gate},
        "skipped_gates": {"mos.sigmos_col": "family mos not scored in this run"},
        "route_gates": {route: {} for route in report_mod.ROUTES},
        "route_skipped": {route: {} for route in report_mod.ROUTES},
        "unmapped_flags": ["listener.dull"],
    }


def _rendered(tmp_path):
    """The markdown report of `_report()`, written with a previous round's gates file."""
    report_mod.write_reports(tmp_path, _report(), {"gates.json": {"listener.dead_air": {"threshold": -22.0}}})
    return (tmp_path / "report.md").read_text("utf-8")


def test_every_check_type_renders(tmp_path):
    """The flat, match and blind check rows render."""
    markdown = _rendered(tmp_path)
    assert "| d | m | flat | fail | | | | 0.100 / 0.300, 0.200 / 0.300, 0.400 / 0.300 |" in markdown
    assert "| d | m | match | pass | | 1.00 | | 4000, , 2828 |" in markdown
    assert "| d | m (blind) | down | unscored | | | | |" in markdown


def test_every_section_renders(tmp_path):
    """The unmapped verdict flags and the per-route derived gates render."""
    markdown = _rendered(tmp_path)
    assert "Verdict flags no gate reads: listener.dull" in markdown
    assert "## Derived gates (mixed)" in markdown


def test_an_unscored_check_says_why_and_a_gate_names_its_rounds(tmp_path):
    """The reason sits in the readings cell; the rounds cell names the ledger rounds behind the count."""
    report_mod.write_reports(tmp_path, _report(), {})
    markdown = (tmp_path / "report.md").read_text("utf-8")
    assert "| d | m | up | unscored | | | | not read |" in markdown
    assert "| d | m | up | unscored | | | | clipped: programme band 1414 Hz, reading from 5000 Hz |" in markdown
    assert "| listener.dead_air | -22.000 | hard | known ordering | 2 | 2 (1, 4) |" in markdown


def test_the_json_report_keeps_the_rules_not_the_scored_results(tmp_path):
    """The json report keeps the rules not the scored results."""
    report_mod.write_reports(tmp_path, _report(), {})
    written = json.loads((tmp_path / "report.json").read_text("utf-8"))
    assert written["ordering"]["tele"] == {"ranked_on": 0, "scores": {"a": None}, "verdict_source": "ledger"}
    assert "ordering_records" not in written


def test_the_previous_rounds_gates_files_are_read_back(tmp_path):
    """The previous rounds gates files are read back."""
    (tmp_path / "gates.json").write_text(json.dumps({"x": {"threshold": 1.0}}), "utf-8")
    (tmp_path / "gates_music.json").write_text(json.dumps({}), "utf-8")
    assert report_mod.previous_gates(tmp_path) == {"gates.json": {"x": {"threshold": 1.0}}, "gates_music.json": {}}
    assert report_mod.gate_files() == ["gates.json", "gates_speech.json", "gates_music.json", "gates_mixed.json"]
