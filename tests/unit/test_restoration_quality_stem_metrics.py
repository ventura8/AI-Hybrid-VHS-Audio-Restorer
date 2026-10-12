"""The stems family's wiring of the programme-cell octave: a level-true source stem, the mixes on its timeline, every window."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts.restoration_quality import scorecard, stem_metrics

RATE = 44100


def test_the_file_peak_is_the_largest_sample_over_every_channel(tmp_path):
    """Read in blocks, every channel counts."""
    path = tmp_path / "x.wav"
    audio = np.zeros((5000, 2), dtype=np.float32)
    audio[4000, 1] = -0.75
    sf.write(str(path), audio, RATE, subtype="FLOAT")
    assert stem_metrics.file_peak(path) == pytest.approx(0.75)


def _pair(lag=0, rate=RATE):
    """A stand-in pair: the aligned sides as ramps, each side resampled by `at` only when the rate differs."""
    source, output = np.arange(1000, dtype=np.float32), -np.arange(1000, dtype=np.float32)
    calls = []

    def at(side, target):
        calls.append((side, target))
        return source if side == "source" else output

    return SimpleNamespace(lag=lag, rate=rate, at=at, calls=calls)


def test_the_level_true_stem_takes_the_pairs_lag_at_its_own_rate():
    """A late source (negative lag) loses its head on the stem too, the lag rescaled to the stem's rate."""
    stem = np.arange(500, dtype=np.float32)
    pair = _pair(lag=-20, rate=88200)
    trimmed, source, output = stem_metrics.levelled(pair, {"level_true": stem, "rate": RATE})
    assert trimmed[0] == 10.0
    assert source[1] == 1.0
    assert output[1] == -1.0
    assert pair.calls == [("source", RATE), ("output", RATE)]


def test_an_early_source_keeps_its_stem_whole_and_a_capped_stem_reads_nothing():
    """A positive lag trims the output side only; without a level-true stem there is no reading."""
    stem = np.arange(500, dtype=np.float32)
    assert stem_metrics.levelled(_pair(lag=30), {"level_true": stem, "rate": RATE})[0][0] == 0.0
    assert stem_metrics.levelled(_pair(), {"level_true": None, "rate": RATE}) is None
    assert stem_metrics.programme_octave(None, 0, 10, RATE) is None


def test_one_window_of_the_mixes_goes_to_the_reading(monkeypatch):
    """`programme_octave` hands the window's slice of each array to `stem_octave.octave_programme_db`."""
    seen = []
    monkeypatch.setattr(stem_metrics.stem_octave, "octave_programme_db", lambda *args: seen.append(args) or -1.5)
    arrays = tuple(np.arange(100, dtype=np.float32) + offset for offset in (0, 1000, 2000))
    assert stem_metrics.programme_octave(arrays, 10, 20, RATE) == -1.5
    stem, source, output, rate = seen[0]
    assert (stem[0], source[0], output[-1], rate) == (10.0, 1010.0, 2019.0, RATE)


def _separator(stems_dir, gain=1.0):
    """A stand-in separator writing each input's left channel times `gain` as its 'instrumental' stem.

    At 1.0 a music-only clip is all background; at 1e-4 the stem is an interview's residual room, 80 dB down.
    """

    def separate(wav_path, custom_output_names):
        audio, rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        sf.write(str(Path(stems_dir) / f"{custom_output_names['Instrumental']}.wav"), gain * audio[:, :1], rate, subtype="FLOAT")

    return SimpleNamespace(separate=separate)


def _registry(stems_dir, gain=1.0):
    """A model registry that hands out the stand-in separator and releases nothing."""
    return SimpleNamespace(get=lambda _name, _loader: _separator(stems_dir, gain), release=lambda: None)


def _song(seed, seconds=15.0):
    """A moving bed over hiss, mono: a new chord of three harmonic notes every 0.4 s, each with its attack and decay."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.4 * RATE)) / RATE
    envelope = np.minimum(1.0, t / 0.02) * np.exp(-t / 0.25)
    chords = [
        sum(
            0.02 / k * envelope * np.sin(2 * np.pi * root * k * t + rng.uniform(0, 2 * np.pi))
            for root in rng.uniform(110, 440, 3)
            for k in range(1, 12)
        )
        for _chord in range(int(seconds / 0.4) + 1)
    ]
    bed = np.concatenate(chords)[: int(seconds * RATE)]
    return (bed + 10.0 ** (-45.0 / 20.0) * rng.standard_normal(len(bed))).astype(np.float32)


def _scoring_pair(tmp_path):
    """Source and output WAVs, the output the source with its bed 12 dB down, aligned and windowed like the runner's."""
    source = _song(1)
    output = 0.25 * source
    src_wav, out_wav = tmp_path / "src.wav", tmp_path / "out.wav"
    sf.write(str(src_wav), source, RATE, subtype="FLOAT")
    sf.write(str(out_wav), output, RATE, subtype="FLOAT")
    pair = SimpleNamespace(
        cache_dir=tmp_path,
        source_wav=src_wav,
        output_wav=out_wav,
        source_key="s",
        output_key="o",
        rate=RATE,
        lag=0,
        source=source,
        output=4.0 * output,
        at=lambda side, _rate: source if side == "source" else 4.0 * output,
    )
    return pair


def test_score_fills_every_background_window_with_the_programme_octave(tmp_path, monkeypatch):
    """A music-only clip turned down as a whole: the raw readings and the programme octave land on every window; it reads ~0."""
    monkeypatch.setattr(stem_metrics.judges, "score_mert", lambda *_args: None)
    pair = _scoring_pair(tmp_path)
    registry = _registry(tmp_path / "stems")
    card = scorecard.ScoreCard(rows=[scorecard.WindowRow(0, 0.0, 15.0, "music")])
    stem_metrics.score(pair, card, registry)
    row = card.rows[0]
    assert row.output["stems.has_background"] == 1.0
    assert row.output[stem_metrics.PROGRAMME_OCTAVE] == pytest.approx(0.0, abs=0.5)
    assert row.source[stem_metrics.PROGRAMME_OCTAVE] == 0.0
    assert "stems.octave_ratio_db" in row.output


def test_a_stem_the_separator_capped_leaves_the_programme_octave_unread(tmp_path, monkeypatch):
    """A source stem written at the 0.9 peak cannot be brought back to its mix's level: the reading is None, the rest stays."""
    monkeypatch.setattr(stem_metrics.judges, "score_mert", lambda *_args: None)
    monkeypatch.setattr(stem_metrics, "file_peak", lambda path: 0.9 if "instrumental" in Path(path).name else 0.5)
    pair = _scoring_pair(tmp_path)
    registry = _registry(tmp_path / "stems")
    card = scorecard.ScoreCard(rows=[scorecard.WindowRow(0, 0.0, 15.0, "music")])
    stem_metrics.score(pair, card, registry)
    assert card.rows[0].output[stem_metrics.PROGRAMME_OCTAVE] is None
    assert card.rows[0].output["stems.si_sdr_db"] is not None


def test_a_window_without_a_background_is_skipped_and_a_missing_mert_costs_only_its_reading(tmp_path, monkeypatch, capsys):
    """A near-silent window reads no stem reading; MERT failing to load is said and swallowed."""

    def broken(*_args):
        raise ImportError("no transformers")

    monkeypatch.setattr(stem_metrics.judges, "score_mert", broken)
    pair = _scoring_pair(tmp_path)
    registry = _registry(tmp_path / "stems", 1e-4)
    card = scorecard.ScoreCard(rows=[scorecard.WindowRow(0, 0.0, 15.0, "music")])
    stem_metrics.score(pair, card, registry)
    assert card.rows[0].output == {"stems.has_background": 0.0}
    assert "unavailable: ImportError" in capsys.readouterr().out


def test_a_separator_that_writes_nothing_is_an_error(tmp_path):
    """No stem on disk after separating is a broken separator, not an empty background."""
    silent = SimpleNamespace(separate=lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="no instrumental stem"):
        stem_metrics.separate_once(silent, tmp_path / "x.wav", tmp_path, "k")


def test_the_separator_is_the_apps_bs_roformer_writing_the_instrumental_only(tmp_path, monkeypatch):
    """`load_separator` asks audio-separator for the app's model and the instrumental stem alone."""
    made = {}

    def fake_separator(**options):
        made.update(options)
        return SimpleNamespace(load_model=lambda model_filename: made.update(model=model_filename))

    module = SimpleNamespace(Separator=fake_separator)
    monkeypatch.setitem(__import__("sys").modules, "audio_separator", SimpleNamespace(separator=module))
    monkeypatch.setitem(__import__("sys").modules, "audio_separator.separator", module)
    stem_metrics.load_separator(tmp_path, tmp_path / "stems")
    assert (made["model"], made["output_single_stem"], made["model_file_dir"]) == (stem_metrics.STEM_MODEL, "Instrumental", str(tmp_path))


def test_the_raw_readings_skip_what_they_cannot_read():
    """A silent stem has no octave; too short or flat envelopes have no correlation."""
    silent = np.zeros(RATE, dtype=np.float32)
    assert stem_metrics.octave_ratio_db(silent, silent, RATE) is None
    assert stem_metrics.envelope_corr(silent[:100], silent[:100], RATE) is None
    assert stem_metrics.envelope_corr(np.ones(RATE), np.ones(RATE), RATE) is None
