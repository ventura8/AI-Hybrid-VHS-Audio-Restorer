"""The calibration's closing summary: what counts as a pass, and the checks R0's programme band clipped away."""

from scripts import calibrate_quality_metrics as cal

CLIPPED_AIR = {
    "degradation": "air_shelf_boost_music",
    "metric": "dsp.balance_air_db",
    "status": "unscored",
    "blind": False,
    "reason": "clipped",
    "programme_band_hz": 1414.2,
    "reading_from_hz": 5000.0,
}


def test_the_summary_counts_passes_only_and_names_every_clipped_check(tmp_path, capsys):
    """An unscored check is not a pass; a non-blind one R0's band clipped away is named with the band."""
    checks = [CLIPPED_AIR, {"status": "pass", "blind": False}, {"status": "unscored", "blind": False, "reason": "not read"}]
    assert cal.print_summary(tmp_path, checks, {"dsp": {"ok": 1}}, ("dsp",)) == 0
    out = capsys.readouterr().out
    assert "1/3 sensitivity checks pass; 0 non-blind failures; 2 non-blind unscored" in out
    assert "air_shelf_boost_music dsp.balance_air_db (1414 Hz)" in out
    assert "families not scored" not in out


def test_a_non_blind_failure_still_sets_the_exit_code(tmp_path, capsys):
    """Unscored checks inform; only a failed non-blind check fails the run."""
    assert cal.print_summary(tmp_path, [{"status": "fail", "blind": False}], {}, ("dsp",)) == 1
    assert "families not scored (their gates keep the hand-set values): dsp" in capsys.readouterr().out


def test_without_a_clipped_check_there_is_no_clipped_line():
    """A blind clipped check is reported in the table only."""
    assert cal.clipped_line([{**CLIPPED_AIR, "blind": True}]) is None
    assert cal.non_blind_unscored([{**CLIPPED_AIR, "blind": True}]) == []
