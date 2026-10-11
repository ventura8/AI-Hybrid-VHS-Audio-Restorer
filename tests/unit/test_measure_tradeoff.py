"""The trade metric: noise removed read pooled and as a per-frame median, the pair's alignment, and the CLI's plumbing."""

import json
import shutil
import sys

import numpy as np
import soundfile as sf

from scripts import measure_tradeoff as mt
from scripts import pair_alignment as pa

RATE = 44100
FRAME = mt.FRAME


def _frames_signal(count=1200, seed=21):
    """`(source, quiet-ish mask)`: 60 % loud frames, 40 % hiss frames 40 dB under them, frame by frame."""
    rng = np.random.default_rng(seed)
    hiss = rng.uniform(0.0, 1.0, count) < 0.4
    amplitude = np.where(hiss, 0.003, 0.3) * rng.uniform(0.8, 1.2, count)
    source = (rng.standard_normal((count, FRAME)) * amplitude[:, None]).reshape(-1)
    return source, hiss


def _refilled(source, hiss, every=50):
    """Hiss frames 10 dB down, except every `every`-th hiss frame lifted 30 dB (a repair stage's hot refill)."""
    gains = np.ones(len(hiss))
    hiss_index = np.flatnonzero(hiss)
    gains[hiss_index] = 10.0 ** (-10.0 / 20.0)
    gains[hiss_index[::every]] = 10.0 ** (30.0 / 20.0)
    return (source.reshape(len(hiss), FRAME) * gains[:, None]).reshape(-1)


def _write(path, mono):
    sf.write(str(path), np.asarray(mono, dtype=np.float32), RATE, subtype="FLOAT")
    return path


def test_a_few_hot_quiet_frames_own_the_pooled_reading_but_not_the_median(tmp_path, monkeypatch):
    """2 % of the quiet frames 30 dB hot, the rest 10 dB down: pooled reads a rise, the median the 10 dB."""
    monkeypatch.delenv(pa.METHOD_ENV, raising=False)
    source, hiss = _frames_signal()
    row = mt.measure(_write(tmp_path / "s.wav", source), _write(tmp_path / "r.wav", _refilled(source, hiss)))
    assert row["noise_removed_db"] < -5.0
    assert abs(row["noise_removed_median_db"] - 10.0) < 0.5
    assert row["programme_deviation_db"] < 0.1
    assert set(row) == {"noise_removed_db", "noise_removed_median_db", "programme_deviation_db"}


def test_a_uniform_drop_reads_the_same_pooled_and_median(tmp_path):
    source, hiss = _frames_signal(seed=22)
    row = mt.measure(_write(tmp_path / "s.wav", source), _write(tmp_path / "r.wav", _refilled(source, hiss, every=10**9)))
    assert abs(row["noise_removed_db"] - row["noise_removed_median_db"]) < 0.3


def test_the_row_can_carry_the_alignment_it_was_read_on(tmp_path, monkeypatch):
    monkeypatch.delenv(pa.METHOD_ENV, raising=False)
    source, hiss = _frames_signal(seed=23)
    late = np.concatenate((np.zeros(219), _refilled(source, hiss, every=10**9)))
    source_wav, restored_wav = _write(tmp_path / "s.wav", source), _write(tmp_path / "r.wav", late)
    row = mt.measure(source_wav, restored_wav, with_alignment=True)
    assert (row["alignment"]["method"], row["alignment"]["lag_samples"]) == (pa.ALIGN_GCC, 219)
    legacy = mt.measure(source_wav, restored_wav, align_method=pa.ALIGN_LEGACY, with_alignment=True)
    assert legacy["alignment"]["method"] == pa.ALIGN_LEGACY
    prior = mt.measure(source_wav, restored_wav, align_prior_s=pa.MUX_DELAY_S, with_alignment=True)
    assert prior["alignment"]["applied_lag_samples"] == 219


def test_a_source_without_spread_or_frames_is_refused(tmp_path):
    steady = np.random.default_rng(3).standard_normal(40 * FRAME) * 0.1
    assert mt.measure(_write(tmp_path / "s.wav", steady), _write(tmp_path / "r.wav", steady)) is None
    assert mt.quiet_and_loud(np.zeros(4 * FRAME), np.zeros(4 * FRAME)) is None
    assert mt.trade_readings(np.zeros(10), np.zeros(10), RATE) is None


def test_a_non_finite_source_reads_no_quiet_or_loud_frames():
    source, _hiss = _frames_signal(count=40, seed=4)
    source[5] = np.nan
    assert mt.quiet_and_loud(source, source) is None


def test_median_drop_is_per_frame():
    source = np.ones((4, 8))
    restored = np.vstack([np.full(8, 0.1)] * 3 + [np.full(8, 100.0)])
    assert abs(mt.median_drop_db(source, restored) - 20.0) < 1e-6


def _fake_extract(sources):
    """An `_extract` that copies the prepared WAV for each container instead of running ffmpeg."""

    def extract(video, target):
        shutil.copyfile(sources[str(video)], target)
        return target

    return extract


def _clip_and_output(tmp_path):
    source, hiss = _frames_signal(seed=31)
    clip = tmp_path / "corpus" / "clip.mp4"
    output = tmp_path / "work" / "clip" / "cathar" / "clip_Cathar_Cleaned.mp4"
    for path in (clip, output):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"container")
    sources = {str(clip): _write(tmp_path / "src.wav", source), str(output): _write(tmp_path / "out.wav", _refilled(source, hiss))}
    return clip, output, sources


def test_measure_pair_reads_the_extracted_tracks_and_cleans_up(tmp_path, monkeypatch):
    clip, output, sources = _clip_and_output(tmp_path)
    monkeypatch.setattr(mt, "_extract", _fake_extract(sources))
    temp = tmp_path / "temp"
    temp.mkdir()
    row = mt._measure_pair(clip, output, "cathar", temp, align_method=pa.ALIGN_LEGACY)
    assert row["alignment"]["method"] == pa.ALIGN_LEGACY
    assert "noise_removed_median_db" in row
    assert list(temp.iterdir()) == []


def test_rescore_measures_the_output_left_in_the_work_dir(tmp_path, monkeypatch):
    clip, _output, sources = _clip_and_output(tmp_path)
    monkeypatch.setattr(mt, "_extract", _fake_extract(sources))
    temp = tmp_path / "temp"
    temp.mkdir()
    row = mt._restore_and_measure(clip, "cathar", tmp_path / "work", None, temp, rescore=True, align_method=pa.ALIGN_GCC)
    assert row["alignment"]["method"] == pa.ALIGN_GCC
    assert mt._restore_and_measure(clip, "auto_pure_linear", tmp_path / "work", None, temp, rescore=True) is None


def test_the_cli_rescores_with_the_named_aligner_and_prints_the_median(tmp_path, monkeypatch, capsys):
    clip, _output, sources = _clip_and_output(tmp_path)
    monkeypatch.setattr(mt, "_extract", _fake_extract(sources))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([{"file": "clip.mp4", "identifier": "clip"}]), encoding="utf-8")
    report = tmp_path / "tradeoff.json"
    argv = ["measure_tradeoff.py", "--corpus-dir", str(clip.parent), "--catalog", str(catalog), "--modes", "cathar"]
    argv += ["--work-dir", str(tmp_path / "work"), "--report", str(report), "--rescore", "--align-method", pa.ALIGN_LEGACY]
    monkeypatch.setattr(sys, "argv", argv)
    mt.main()
    rows = json.loads(report.read_text(encoding="utf-8"))["cathar"]
    assert rows[0]["alignment"]["method"] == pa.ALIGN_LEGACY
    assert "median per frame" in capsys.readouterr().out
