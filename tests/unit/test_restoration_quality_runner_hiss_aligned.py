"""Ear v4 merged with the GCC-PHAT aligner: the hiss readings read the pair the aligner put on one timeline."""

import types

import numpy as np
import pytest
import soundfile as sf

from scripts.restoration_quality import runner
from tests.unit.test_restoration_quality_hiss_floor import RATE, capture

# Ten milliseconds late: a sync stage's lag, which a reading on the unaligned pair would read as hiss moved in time.
LAG = 441
MARGINS = ("dsp.hiss_prog_margin_db", "dsp.hiss_pause_margin_db")


def _registry():
    """No models: the dsp family needs none, and the real registry's release imports torch."""
    return types.SimpleNamespace(language="ro", release=lambda: None)


def _write(path, audio):
    sf.write(str(path), np.asarray(audio, dtype=np.float32), RATE, subtype="FLOAT")
    return path


@pytest.fixture(scope="module", name="cards")
def fixture_cards(tmp_path_factory):
    """The synthetic capture against its hiss 12 dB down, on time and `LAG` samples late."""
    folder = tmp_path_factory.mktemp("aligned")
    programme, noise, source = capture()
    output = programme + noise * 10.0 ** (-12.0 / 20.0)
    late = np.concatenate([np.zeros(LAG), output])[: len(output)]
    outputs = {"on_time": _write(folder / "on_time.wav", output), "late": _write(folder / "late.wav", late)}
    _result, cards = runner.score_variants(
        _write(folder / "source.wav", source), outputs, families=("dsp",), registry=_registry(), cache_dir=folder / "cache"
    )
    return cards


def test_the_aligner_finds_the_late_outputs_lag(cards):
    _card, pair = cards["late"]
    assert pair.alignment.applied_lag == LAG


@pytest.mark.parametrize("key", MARGINS)
def test_the_late_output_reads_the_hiss_of_the_on_time_one(cards, key):
    """Both hiss margins read the late output within 0.1 dB of the on-time one: they stand on the aligned pair."""
    on_time, late = (cards[label][0].aggregate[key]["output"]["median"] for label in ("on_time", "late"))
    assert late == pytest.approx(on_time, abs=0.1)
