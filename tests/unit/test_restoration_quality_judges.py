"""The pretrained judges with their models faked: SCOREQ preprocessing, Zimtohrli's loud-frame selection, MERT's distance."""

import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from scripts.restoration_quality import judges, mos_models
from scripts.restoration_quality.scorecard import ScoreCard, WindowRow


class _FakeSession:
    """An ONNX session that records what it was fed and answers one MOS."""

    def __init__(self):
        self.feeds = []

    def get_inputs(self):
        return [types.SimpleNamespace(name="audio_input")]

    def run(self, _outputs, feed):
        self.feeds.append(feed)
        return [np.array([[3.5]], dtype=np.float32)]


def test_scoreq_pads_to_a_multiple_of_320_and_reads_the_scalar():
    session = _FakeSession()
    scores = mos_models.scoreq_scores(session, np.ones(16001, dtype=np.float64))
    assert scores == {"mos.scoreq_nr": 3.5}
    fed = session.feeds[0]["audio_input"]
    assert fed.shape == (1, 16320) and fed.dtype == np.float32
    assert np.all(fed[0, 16001:] == 0.0) and np.all(fed[0, :16001] == 1.0)


def test_scoreq_leaves_an_exact_multiple_alone_and_skips_empty_audio():
    session = _FakeSession()
    mos_models.scoreq_scores(session, np.zeros(640, dtype=np.float32))
    assert session.feeds[0]["audio_input"].shape == (1, 640)
    assert mos_models.scoreq_scores(session, np.zeros(0)) == {}


def _fake_zimtohrli(calls):
    module = types.ModuleType("zimtohrli")

    class Pyohrli:
        def distance(self, a, b):
            calls.append((np.asarray(a), np.asarray(b)))
            return 0.25

    module.Pyohrli = Pyohrli
    return module


def _loud_then_quiet(frames_loud=10, frames_quiet=10, frame=judges.ZIMTOHRLI_FRAME):
    """Square waves, so every loud frame has the same RMS and the p70 threshold falls between the two halves."""
    loud = np.tile(np.array([0.5, -0.5]), frames_loud * frame // 2)
    quiet = np.tile(np.array([0.001, -0.001]), frames_quiet * frame // 2)
    return np.concatenate([loud, quiet]).astype(np.float32)


def test_zimtohrli_scores_only_the_frames_where_the_source_is_loud():
    calls = []
    source = _loud_then_quiet()
    output = np.arange(len(source), dtype=np.float32)
    with patch.dict(sys.modules, {"zimtohrli": _fake_zimtohrli(calls)}):
        assert judges.zimtohrli_loud(source, output, 48000) == 0.25
    src, out = calls[0]
    assert len(src) == len(out) == 10 * judges.ZIMTOHRLI_FRAME
    assert np.array_equal(src, source[: 10 * judges.ZIMTOHRLI_FRAME])
    assert np.array_equal(out, output[: 10 * judges.ZIMTOHRLI_FRAME])


def test_zimtohrli_is_none_on_too_short_audio_and_refuses_another_rate():
    with patch.dict(sys.modules, {"zimtohrli": _fake_zimtohrli([])}):
        assert judges.zimtohrli_loud(np.zeros(100), np.zeros(100), 48000) is None
    with pytest.raises(ValueError):
        judges.zimtohrli_loud(np.zeros(100), np.zeros(100), 44100)


def test_zimtohrli_missing_is_an_import_error():
    with patch.dict(sys.modules, {"zimtohrli": None, "pyohrli": None}):
        with pytest.raises(ImportError):
            judges.zimtohrli_module()


class _FakePair:
    """A pair whose resampled sides are whatever the test hands it."""

    def __init__(self, source, output):
        self.calls = []
        self._sides = {"source": source, "output": output}

    def at(self, side, rate):
        self.calls.append((side, rate))
        return self._sides[side]


def _card(routes, seconds=1.0):
    return ScoreCard(rows=[WindowRow(i, i * seconds, (i + 1) * seconds, route) for i, route in enumerate(routes)])


def test_score_zimtohrli_writes_programme_rows_at_48k():
    frame = judges.ZIMTOHRLI_FRAME
    one_second = np.tile(_loud_then_quiet(frames_loud=6, frames_quiet=5)[: 48000 // frame * frame], 1)
    audio = np.concatenate([np.pad(one_second, (0, 48000 - len(one_second)))] * 4)
    pair, card = _FakePair(audio, audio), _card(["speech", "silence", "music", "mixed"])
    with patch.dict(sys.modules, {"zimtohrli": _fake_zimtohrli([])}):
        judges.score_zimtohrli(pair, card)
    assert set(pair.calls) == {("source", 48000), ("output", 48000)}
    assert [row.output.get("dsp.zimtohrli_loud") for row in card.rows] == [0.25, None, 0.25, 0.25]
    assert card.rows[0].source["dsp.zimtohrli_loud"] == 0.0 and "dsp.zimtohrli_loud" not in card.rows[1].source


class _FakeMert:
    """A model whose layer-12 states are the input samples spread over time, so the embedding is the input's direction."""

    device = "cpu"

    def __call__(self, input_values, output_hidden_states=False, **_kwargs):
        torch = pytest.importorskip("torch")
        states = [torch.zeros(1, 1, 3)] * 12 + [input_values.reshape(1, -1, 3)]
        return types.SimpleNamespace(hidden_states=states)


def _fake_extractor(audio, sampling_rate, return_tensors):
    torch = pytest.importorskip("torch")
    assert sampling_rate == 24000 and return_tensors == "pt"
    return {"input_values": torch.from_numpy(np.asarray(audio, dtype=np.float32))}


def test_mert_distance_is_zero_for_the_same_music_and_one_for_orthogonal_music():
    pytest.importorskip("torch")
    a, b = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    assert judges.mert_distance(_FakeMert(), _fake_extractor, a, a) == pytest.approx(0.0, abs=1e-6)
    assert judges.mert_distance(_FakeMert(), _fake_extractor, a, b) == pytest.approx(1.0, abs=1e-6)
    assert judges.mert_distance(_FakeMert(), _fake_extractor, a, 2.0 * a) == pytest.approx(0.0, abs=1e-6)


class _Registry:
    def __init__(self, model):
        self.model = model

    def get(self, _name, _loader):
        return self.model, _fake_extractor


def test_score_mert_writes_music_and_mixed_rows_from_the_24k_mix():
    pytest.importorskip("torch")
    source = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), 24000 * 4)
    output = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), 24000 * 4)
    pair, card = _FakePair(source, output), _card(["speech", "music", "mixed", "silence"])
    judges.score_mert(pair, card, _Registry(_FakeMert()))
    assert set(pair.calls) == {("source", 24000), ("output", 24000)}
    assert [round(row.output.get("stems.mert_dist", -1.0), 6) for row in card.rows] == [-1.0, 1.0, 1.0, -1.0]
    assert card.rows[1].source["stems.mert_dist"] == 0.0 and "stems.mert_dist" not in card.rows[0].source


def test_load_mert_and_load_scoreq_name_the_download_set(tmp_path):
    with pytest.raises(SystemExit, match="--set stems"):
        judges.load_mert("cpu", tmp_path)
    with pytest.raises(SystemExit, match="--set mos"):
        mos_models.load_scoreq("cpu", tmp_path)
