"""R12 in the runner and the gates: the time-base readings land on the card once per pair, and a hold fails a hard, uncalibrated gate."""

import types

import numpy as np
import pytest
import soundfile as sf

from scripts.restoration_quality import audio_io, runner
from scripts.restoration_quality.gates import GATES, HARD, evaluate_gates, hard_failures
from scripts.restoration_quality.scorecard import METRICS
from tests.unit.test_restoration_quality_timebase import EVENTS, RATE, _programme, _warped


def _registry():
    """No models: the dsp family needs none, and the real registry's release imports torch."""
    return types.SimpleNamespace(language="ro", release=lambda: None)


@pytest.fixture(scope="module", name="cards")
def fixture_cards(tmp_path_factory):
    """One dsp-family run over a clean output and one carrying three DTW glitches, every window routed as speech."""
    folder = tmp_path_factory.mktemp("timebase")
    programme = _programme(seconds=16.0)
    paths = {"source": folder / "source.wav", "clean": folder / "clean.wav", "glitched": folder / "glitched.wav"}
    signals = {"source": programme, "clean": programme, "glitched": _warped(programme, RATE, EVENTS)}
    for name, path in paths.items():
        sf.write(str(path), np.stack([signals[name], signals[name]], axis=1), RATE, subtype="FLOAT")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(audio_io, "route_window", lambda _mono, _rate: "speech")
        outputs = {"clean": paths["clean"], "glitched": paths["glitched"]}
        result, cards = runner.score_variants(paths["source"], outputs, families=("dsp",), registry=_registry(), cache_dir=folder / "cache")
    return result, {label: card for label, (card, _pair) in cards.items()}


NAMES = {f"file.timebase_{name}" for name in ("holds", "holds_per_min", "plateaus", "rate_jumps", "speed_dev_pct", "tracked_share")}


@pytest.mark.parametrize("label", ["clean", "glitched"])
def test_the_time_base_readings_land_on_the_card_once_per_pair(cards, label):
    """Every `file.timebase_*` reading is a file entry with a zero source side, and the aggregate carries it for the gates."""
    card = cards[1][label]
    assert NAMES <= set(card.file) & set(card.aggregate)
    assert {card.file[name]["source"] for name in NAMES} == {0.0}


def test_a_glitched_output_fails_both_time_base_gates_and_a_clean_one_passes(cards):
    """Three holds and their catch-up fail `timebase.holds` and `timebase.rate_jumps`; the clean output reads none."""
    _result, by_label = cards
    glitched, clean = by_label["glitched"], by_label["clean"]
    assert glitched.file["file.timebase_holds"]["output"] == 3.0
    assert {"timebase.holds", "timebase.rate_jumps"} <= set(hard_failures(glitched.verdicts))
    assert not {"timebase.holds", "timebase.rate_jumps"} & set(hard_failures(clean.verdicts))
    assert clean.file["file.timebase_plateaus"]["output"] == 0.0


def test_the_document_records_the_time_base_gates_as_uncalibrated(cards):
    """The JSON document's gate registry says the starting thresholds are uncalibrated."""
    result, _by_label = cards
    for name in ("timebase.holds", "timebase.rate_jumps"):
        assert result["gates"][name] == {
            "metric": GATES[name].metric,
            "stat": "median",
            "op": "<=",
            "threshold": 0.0,
            "severity": HARD,
            "calibrated": False,
        }


def test_the_time_base_gates_are_hard_starting_values_on_registered_readings():
    """Both gates are hard, uncalibrated, at none at all, and read registered file metrics."""
    for name in ("timebase.holds", "timebase.rate_jumps"):
        gate = GATES[name]
        assert (gate.severity, gate.calibrated, gate.op, gate.threshold) == (HARD, False, "<=", 0.0)
        assert METRICS[gate.metric].family == "file"


def test_a_time_base_gate_skips_an_unread_track():
    """A track that read no confident frame (None) skips the rate-jump gate; the hold gate still reads its count."""
    aggregate = {"file.timebase_holds": {"output": {"median": 0.0, "tail": 0.0, "n": 1}}}
    gates = {"timebase.holds": GATES["timebase.holds"], "timebase.rate_jumps": GATES["timebase.rate_jumps"]}
    verdicts = evaluate_gates(aggregate, gates)
    assert [(v["gate"], v["status"], v["calibrated"]) for v in verdicts] == [
        ("timebase.holds", "passed", False),
        ("timebase.rate_jumps", "skipped", False),
    ]
