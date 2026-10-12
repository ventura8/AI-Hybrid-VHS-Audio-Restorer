"""E1 (2026-10-12): the gates that read the denoising itself, the calibrated flag, floor entries and retired names."""

import json

import pytest

from scripts.restoration_quality import gates as g
from tests.unit.test_restoration_quality_gates import _aggregate, _status


def test_cer_and_audiobox_pc_are_shown_not_counted():
    """Both read the denoising itself on hissy tape (E1): they fail as soft gates until a verdict sets them."""
    aggregate = _aggregate(**{"speech.cer": (0.4, 0.4, 0.0, 0.0), "mos.audiobox_pc": (0.0, 0.0, -1.2, -1.2)})
    verdicts = g.evaluate_gates(aggregate, g.GATES)
    assert _status(verdicts, "speech.cer_median") == _status(verdicts, "mos.audiobox_pc") == "failed"
    assert g.hard_failures(verdicts) == []


def test_a_hand_set_threshold_says_it_is_uncalibrated():
    """Only lkr's value came from a calibration; every other threshold in code is a guess."""
    assert not g.Gate("x", "median", "<=", 1.0).calibrated
    assert not g.GATES["dsp.hum"].calibrated
    assert not g.GATES["file.dropouts"].calibrated
    assert g.GATES["dsp.lkr"].calibrated


def _load(tmp_path, entries):
    path = tmp_path / "gates.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return g.load_gates(path)


@pytest.mark.parametrize(
    ("entry", "calibrated"),
    [
        ({"threshold": 3.0, "source": "floor", "n_verdicts": 0}, False),
        ({"threshold": 3.0, "source": "floor"}, False),
        ({"threshold": 3.0, "source": "floor", "n_verdicts": 0, "calibrated": True}, False),
        ({"threshold": 2.0, "source": "known good bound", "n_verdicts": 4}, True),
        ({"threshold": 2.0, "source": "known ordering", "n_verdicts": 6, "calibrated": False}, False),
    ],
)
def test_a_floor_entry_resting_on_no_verdict_is_never_calibrated(tmp_path, entry, calibrated):
    """A floor threshold is the guess of the day or three benign floors: whatever the file says, it is no calibration."""
    assert _load(tmp_path, {"dsp.hum": entry})["dsp.hum"].calibrated is calibrated


@pytest.mark.parametrize(
    ("name", "entry", "expected"),
    [
        ("dsp.lkr", {"threshold": 0.3, "source": "floor", "n_verdicts": 0}, 1.747),
        ("dsp.lkr", {"threshold": 2.5, "source": "floor", "n_verdicts": 0}, 2.5),
        ("dsp.lkr", {"threshold": 1.2, "source": "known good bound", "n_verdicts": 3}, 1.2),
        ("dsp.hf_4k8k", {"threshold": -6.0, "source": "floor"}, -8.0),
        ("file.lufs", {"threshold": 1.0, "source": "floor", "n_verdicts": 0}, 1.5),
    ],
)
def test_a_floor_entry_never_tightens_the_threshold_in_code(tmp_path, name, entry, expected):
    """The v3b file's lkr 0.3 is the old guess: the looser of the two stands, as re-deriving would give; verdicts still tighten."""
    assert _load(tmp_path, {name: entry})[name].threshold == pytest.approx(expected)


def test_a_floor_entry_keeps_a_relative_threshold_as_written():
    """`floor-0.05` is resolved per tape, so there is nothing to compare."""
    assert g.floor_threshold("floor-0.05", -0.3, ">=") == "floor-0.05"
    assert g.floor_threshold(0.5, True, "<=") == 0.5


def test_a_retired_gate_in_a_stored_file_is_left_out(tmp_path):
    """The v2 file's dropouts bound (57.3) belongs to the old count; the raw octave gate is gone: neither lands on a successor."""
    merged = _load(
        tmp_path,
        {"dsp.dropouts": {"threshold": 57.3, "severity": "hard", "source": "known good bound"}, "stems.octave": {"threshold": -9.0}},
    )
    assert "dsp.dropouts" not in merged and "stems.octave" not in merged
    assert merged["file.dropouts"].threshold == g.DROPOUT_BOUND
    assert set(g.RETIRED.values()) <= set(merged)


@pytest.mark.parametrize(("holes", "status"), [(4.0, "passed"), (5.0, "failed")])
def test_the_dropout_gate_counts_the_holes_over_the_whole_pair(holes, status):
    """`file.dropouts` is one reading per file: the window tail no longer hides a few holes in a long file."""
    verdicts = g.evaluate_gates(_aggregate(**{"file.dropouts": (holes, holes, holes, holes)}), g.GATES)
    assert _status(verdicts, "file.dropouts") == status


def test_the_background_gate_reads_the_programme_octave_not_the_raw_one():
    """The raw octave is displayed only; the gate reads the worst window of the programme-cell octave."""
    aggregate = _aggregate(**{"stems.octave_ratio_db": (-30.0, -38.8, -30.0, -38.8), "stems.octave_programme_db": (-0.5, -1.0, -0.5, -1.0)})
    verdicts = g.evaluate_gates(aggregate, g.GATES)
    assert _status(verdicts, "stems.octave_programme") == "passed"
    assert "stems.octave" not in {v["gate"] for v in verdicts}
