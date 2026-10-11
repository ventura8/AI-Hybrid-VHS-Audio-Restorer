"""The aligner inside the harness: the pair's record, the scored JSON, the report line, the CLI flags and every caller's rate."""

import types

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

from scripts import measure_hum, pair_alignment, score_defect_repair, score_reference, validate_restoration
from scripts.restoration_quality import audio_io, dsp_metrics, report, runner, stem_metrics

RATE = 44100
MUX = 219


@pytest.fixture(autouse=True)
def _default_method(monkeypatch):
    """Every test reads the default aligner unless it names one."""
    monkeypatch.delenv(pair_alignment.METHOD_ENV, raising=False)


def _speechlike(seconds, seed=3):
    rng = np.random.default_rng(seed)
    count = int(seconds * RATE)
    sos = scipy.signal.butter(4, [200.0, 4000.0], btype="band", fs=RATE, output="sos")
    gates = np.repeat(rng.uniform(0.0, 1.0, count // 4410 + 1) > 0.4, 4410).astype(float)
    envelope = scipy.signal.lfilter([0.01], [1.0, -0.99], gates[:count])
    return (0.3 * scipy.signal.sosfilt(sos, rng.standard_normal(count)) * envelope + 1e-3 * rng.standard_normal(count)).astype(np.float32)


def _late(signal, lag, gain=1.0):
    return (gain * np.concatenate((np.zeros(lag, dtype=np.float32), signal))).astype(np.float32)


def _write(path, audio, rate=RATE):
    sf.write(str(path), audio, rate, subtype="FLOAT")
    return path


def test_aligned_pair_returns_the_record_and_takes_its_prior_in_seconds():
    source = _speechlike(4.0)
    src, out, alignment = audio_io.aligned_pair(source, _late(source, MUX, 0.5), RATE)
    assert (alignment.lag, alignment.method, src.dtype, out.dtype) == (MUX, pair_alignment.ALIGN_GCC, np.float32, np.float32)
    np.testing.assert_allclose(out, src, atol=1e-4)
    stranger = _speechlike(4.0, seed=9)
    _src, _out, abstained = audio_io.aligned_pair(source, stranger, RATE, prior_s=pair_alignment.MUX_DELAY_S)
    assert (abstained.lag, abstained.applied_lag) == (None, MUX)
    assert audio_io.align_pair(source, _late(source, 7), RATE, pair_alignment.ALIGN_LEGACY)[2] == 7


def _scored_pair(tmp_path):
    source = _speechlike(6.0, seed=4)
    return _write(tmp_path / "s.wav", source), _write(tmp_path / "o.wav", _late(source, MUX, 0.7))


def test_the_scored_variant_carries_its_alignment(tmp_path):
    source_wav, output_wav = _scored_pair(tmp_path)
    card, pair = runner.score_pair(source_wav, output_wav, families=(), cache_dir=tmp_path / "cache")
    record = runner.card_to_dict(card, output_wav)
    assert (record["lag_samples"], record["alignment"]["lag_samples"], record["alignment"]["method"]) == (
        MUX,
        MUX,
        pair_alignment.ALIGN_GCC,
    )
    assert runner.align_options(pair) == (pair_alignment.ALIGN_GCC, None)
    assert runner.ALIGN_METHODS[0] == pair_alignment.DEFAULT_METHOD


def test_the_runner_scores_with_the_aligner_it_is_given(tmp_path):
    source_wav, output_wav = _scored_pair(tmp_path)
    legacy, legacy_pair = runner.score_pair(
        source_wav, output_wav, families=(), cache_dir=tmp_path / "cache", align_method=pair_alignment.ALIGN_LEGACY, align_prior_s=0.005
    )
    assert runner.card_to_dict(legacy, output_wav)["alignment"]["status"] == pair_alignment.STATUS_LEGACY
    assert runner.align_options(legacy_pair) == (pair_alignment.ALIGN_LEGACY, 0.005)


def test_a_card_or_pair_without_an_alignment_reads_none():
    assert runner.alignment_record(types.SimpleNamespace()) is None
    assert runner.align_options(types.SimpleNamespace()) == (None, None)


def test_the_trade_reading_reaches_the_aligner_options(monkeypatch):
    calls = []
    monkeypatch.setattr(dsp_metrics, "measure", lambda *args, **kwargs: calls.append((args, kwargs)) or {})
    dsp_metrics.trade("s.wav", "o.wav", pair_alignment.ALIGN_LEGACY, 0.004)
    assert calls == [(("s.wav", "o.wav"), {"align_method": pair_alignment.ALIGN_LEGACY, "align_prior_s": 0.004})]


def _variant(alignment=None):
    return {"lag_samples": 219, "alignment": alignment}


def test_the_report_states_the_lag_its_method_and_an_abstention():
    ok = pair_alignment.Alignment(pair_alignment.ALIGN_GCC, MUX, MUX, 1, pair_alignment.STATUS_OK).as_dict()
    inverted = {**ok, "polarity": -1}
    abstained = pair_alignment.Alignment(pair_alignment.ALIGN_GCC, None, MUX, None, pair_alignment.STATUS_ABSTAINED, "weak correlation")
    assert report.lag_text(_variant(ok)) == "lag 219 samples (gcc_phat_spans)"
    assert report.lag_text(_variant(inverted)) == "lag 219 samples (gcc_phat_spans), inverted"
    assert report.lag_text(_variant(abstained.as_dict())) == "lag unknown (gcc_phat_spans): weak correlation, trimmed at 219 samples"
    assert report.lag_text(_variant()) == "lag 219 samples"


def test_validate_restoration_passes_the_align_flags(tmp_path, monkeypatch):
    seen = {}

    def score_variants(_source, _outputs, **options):
        seen.update(options)
        return {}, {}

    monkeypatch.setattr(validate_restoration.runner, "score_variants", score_variants)
    monkeypatch.setattr(validate_restoration.report, "render_markdown", lambda *_args: "")
    source = _write(tmp_path / "s.wav", np.zeros(100, dtype=np.float32))
    argv = [
        str(source),
        f"cathar={source}",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--align-method",
        "xcorr_head",
        "--align-prior-ms",
        "5",
    ]
    assert validate_restoration.main(argv) == 0
    assert (seen["align_method"], seen["align_prior_s"]) == ("xcorr_head", 0.005)
    validate_restoration.main(argv[:4])
    assert (seen["align_method"], seen["align_prior_s"]) == (None, None)


def test_the_reference_scorer_records_its_alignment(tmp_path):
    clean = _speechlike(5.0, seed=6)
    reference, estimate, lag = score_reference._align(clean, _late(clean, MUX), RATE)
    assert (lag, len(reference)) == (MUX, len(estimate))
    assert score_reference._align(clean, _late(clean, MUX), method=pair_alignment.ALIGN_LEGACY)[2] == MUX
    scored = score_reference.score_pair(_write(tmp_path / "c.wav", clean), _write(tmp_path / "r.wav", _late(clean, MUX)))
    assert (scored["align_lag_samples"], scored["alignment"]["method"]) == (MUX, pair_alignment.ALIGN_GCC)


def test_the_defect_scorer_aligns_at_the_files_rate(tmp_path):
    rate = 48000
    clean = _speechlike(4.0, seed=7)
    paths = [_write(tmp_path / f"{name}.wav", audio, rate) for name, audio in (("c", clean), ("d", clean), ("r", _late(clean, 240)))]
    prepared = score_defect_repair._prepare(*paths, None)
    assert len(prepared[0]) == len(prepared[2]) == len(clean)


def test_the_hum_reading_aligns_at_the_files_rate(tmp_path):
    source = _speechlike(8.0, seed=8)
    hum = (0.02 * np.sin(2.0 * np.pi * 50.0 * np.arange(len(source)) / RATE)).astype(np.float32)
    row = measure_hum.measure(_write(tmp_path / "s.wav", source + hum), _write(tmp_path / "r.wav", _late(source, MUX)), 50.0)
    assert row["hum_removed_db"] > 6.0
    assert row["low_band_deviation_db"] < 0.2


def test_the_stems_are_aligned_with_the_pairs_method(tmp_path, monkeypatch):
    source = _speechlike(3.0, seed=10)
    stems = {"src": _write(tmp_path / "src.wav", source), "out": _write(tmp_path / "out.wav", _late(source, 11))}
    monkeypatch.setattr(stem_metrics, "separate_once", lambda _sep, _wav, _dir, key: stems[key])
    methods = []
    original = audio_io.align_pair

    def spy(source_mono, output_mono, rate, method=None):
        methods.append(method)
        return original(source_mono, output_mono, rate, method)

    monkeypatch.setattr(stem_metrics.audio_io, "align_pair", spy)
    pair = types.SimpleNamespace(
        cache_dir=tmp_path, source_wav=None, output_wav=None, source_key="src", output_key="out", align_method="xcorr_head"
    )
    registry = types.SimpleNamespace(get=lambda _name, _loader: object())
    src, out, rate = stem_metrics._stems(pair, registry)
    assert (methods, rate, len(src)) == (["xcorr_head"], RATE, len(out))
