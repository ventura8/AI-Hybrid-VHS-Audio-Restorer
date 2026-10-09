"""The reward noise floor: near-copies of an output, the operating point, the retest deviations, the report and the command line."""

import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts import reward_noise_floor as nf
from scripts.restoration_quality.scorecard import ScoreCard

RATE = 44100
KEY = "dsp.x.output.median"
# What the fake scorer reads per near-copy, above the reference's 1.0.
OFFSETS = {"shift_1": 0.1, "requantise_16": 0.3, "resample_roundtrip": 0.2}


def _voice(seconds=6.0, seed=3, hiss=1e-3, rate=RATE):
    """A harmonic buzz that speaks in 0.5 s bursts over a hiss: loud frames and true pauses (same programme per seed)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    phases = rng.uniform(0.0, 2 * np.pi, 30)
    voice = sum(np.sin(2 * np.pi * 173.0 * k * t + phases[k]) / k for k in range(1, 30))
    gate = ((t % 1.0) < 0.5).astype(np.float64)
    return (0.1 * voice * gate + hiss * rng.standard_normal(len(t))).astype(np.float32)


def _stereo(seconds=6.0, hiss=1e-3, rate=RATE):
    mono = _voice(seconds, hiss=hiss, rate=rate)
    return np.stack([mono, 0.5 * mono], axis=1)


def _source(tmp_path, seconds=6.0, hiss=1e-3):
    path = tmp_path / "source.wav"
    sf.write(str(path), _stereo(seconds, hiss), RATE, subtype="FLOAT")
    return path


def _output(tmp_path, seconds=6.0):
    """The same programme with the hiss taken down to about -70 dBFS: a restored output."""
    path = tmp_path / "output.wav"
    sf.write(str(path), _stereo(seconds, 3e-4), RATE, subtype="FLOAT")
    return path


def _offset(output):
    return next((value for name, value in OFFSETS.items() if Path(output).stem.endswith(name)), 0.0)


def _card(value):
    card = ScoreCard(families={"dsp": "ok"})
    entry = {"median": value, "tail": value, "n": 1}
    card.aggregate = {"dsp.x": {"source": entry, "output": entry, "delta": entry}}
    return card


def _fake_scorer(calls, drift=0.0):
    """A scorer that reads 1.0 on the reference, 1.0 + OFFSETS on each near-copy, plus `drift` per earlier call."""

    def score(source, output, **options):
        value = 1.0 + _offset(output) + drift * len(calls)
        calls.append((source, output, options))
        return _card(value), None

    return score


def _no_op():
    return None


def _registry(device="cpu", models_dir=None):
    """Stands in for `runner.ModelRegistry`: the DSP family needs no model, and nothing imports torch."""
    return SimpleNamespace(device=device, models_dir=models_dir, release=_no_op)


def _measured(tmp_path, plan, scorer, entry=None):
    entry = _source(tmp_path) if entry is None else entry
    return nf.measure([entry], plan, cache_dir=tmp_path / "cache", scorer=scorer, windows=15.0, registry=_registry())


def test_a_one_sample_shift_delays_every_channel_and_keeps_the_programme():
    """The shift adds one silent frame ahead and leaves every sample after it untouched."""
    audio = _stereo(1.0)
    shifted = nf.shift_one_sample(audio, RATE, None)
    assert shifted.shape == (audio.shape[0] + 1, 2)
    assert np.array_equal(shifted[1:], audio)
    assert not shifted[0].any()


def test_requantisation_and_the_resample_round_trip_stay_near_the_source():
    """16-bit with dither moves a sample by under two steps; the round trip keeps length and programme."""
    audio = _stereo(1.0)
    requantised = nf.requantise_16(audio, RATE, np.random.default_rng(1))
    resampled = nf.resample_roundtrip(audio, RATE, None)
    assert np.max(np.abs(requantised - audio)) < 2.0 / 32767.0
    assert resampled.shape == audio.shape
    assert np.sqrt(np.mean((resampled - audio) ** 2)) < 0.05 * np.sqrt(np.mean(audio**2))


def test_the_round_trip_never_passes_through_the_audios_own_rate():
    """At 48 kHz the trip goes through 44.1 kHz, so the copy really differs; elsewhere it goes through 48 kHz."""
    audio = _stereo(1.0, rate=48000)
    resampled = nf.resample_roundtrip(audio, 48000, None)
    assert (nf.round_trip_rate(48000), nf.round_trip_rate(44100)) == (44100, 48000)
    assert np.max(np.abs(resampled - audio)) > 1e-3
    assert np.sqrt(np.mean((resampled - audio) ** 2)) < 0.05 * np.sqrt(np.mean(audio**2))


def test_a_near_copy_is_written_once_and_reused(tmp_path):
    """A second run finds the near-copy by its keyed name and leaves it (and the scorer caches keyed on it) alone."""
    source = _source(tmp_path)
    first = nf.variant_wav(source, "shift_1", tmp_path / "cache")
    stamp = first.stat().st_mtime_ns
    second = nf.variant_wav(source, "shift_1", tmp_path / "cache")
    assert second == first
    assert second.stat().st_mtime_ns == stamp
    assert not list((tmp_path / "cache" / nf.WORK_DIR).glob("*.partial.wav"))


def test_max_seconds_scores_only_the_head_of_the_source(tmp_path):
    """The cut is written once into the cache; 0 means the whole source, unchanged."""
    source = _source(tmp_path)
    cut = nf.prepared_source(source, tmp_path / "cache", 2.0)
    assert sf.info(str(cut)).frames == 2 * RATE
    assert nf.prepared_source(source, tmp_path / "cache", 0.0) == source
    assert nf.prepared_source(source, tmp_path / "cache", 2.0) == cut


def test_a_pair_cuts_the_source_and_the_output_alike(tmp_path):
    """Both sides of a pair are cut to the same head; a bare source is its own output."""
    material = nf.prepared((_source(tmp_path), _output(tmp_path)), tmp_path / "cache", 2.0)
    assert [sf.info(str(path)).frames for path in (material.source_wav, material.output_wav)] == [2 * RATE] * 2
    assert material.operating_point == "output"
    assert nf.prepared(_source(tmp_path), tmp_path / "cache").operating_point == "identity"


def test_the_floor_is_the_largest_retest_deviation_per_transform(tmp_path):
    """Each near-copy's deviation from the reference lands under its own name; the floor is their p95."""
    report = _measured(tmp_path, nf.Plan(("shift_1", "requantise_16")), _fake_scorer([]))
    entry = report["readings"][KEY]
    assert entry["by_transform"] == pytest.approx({"shift_1": 0.1, "requantise_16": 0.3})
    assert (entry["n"], entry["max"]) == (2, pytest.approx(0.3))
    assert 0.1 < entry["floor"] <= 0.3
    assert report["transforms"] == ["reference", "shift_1", "requantise_16"]


def test_an_output_is_perturbed_and_its_source_left_untouched(tmp_path):
    """At the output point every scoring keeps the source; the reference is the output, the near-copies are made of it."""
    calls = []
    source, output = _source(tmp_path, hiss=5e-3), _output(tmp_path)
    report = _measured(tmp_path, nf.Plan(("shift_1", "requantise_16")), _fake_scorer(calls), (source, output))
    assert {call[0] for call in calls} == {source}
    assert [Path(call[1]).name for call in calls][0] == output.name
    assert all(Path(call[1]).name.startswith(nf.audio_io.file_key(output)) for call in calls[1:])
    assert (report["operating_points"], report["sources"][0]["output"]) == (["output"], str(output))


def test_repeats_score_every_pair_again_and_their_jitter_enters_the_floor(tmp_path):
    """Three repeats of two pairs: the reference re-scorings carry the scorer's own jitter."""
    calls = []
    report = _measured(tmp_path, nf.Plan(("shift_1",), repeats=3), _fake_scorer(calls, drift=0.01))
    entry = report["readings"][KEY]
    assert len(calls) == 6
    assert entry["n"] == 5
    assert entry["by_transform"] == pytest.approx({"reference": 0.04, "shift_1": 0.15})


def test_every_repeat_after_the_first_scores_through_its_own_cache(tmp_path):
    """No cached transcript or stem from an earlier repeat can hide the jitter; the report lists the caches."""
    calls = []
    report = _measured(tmp_path, nf.Plan(("shift_1",), repeats=3), _fake_scorer(calls))
    work = tmp_path / "cache" / nf.WORK_DIR
    expected = [tmp_path / "cache", work / "repeat2", work / "repeat3"]
    assert [call[2]["cache_dir"] for call in calls] == [cache for cache in expected for _ in range(2)]
    assert report["repeat_caches"] == [str(cache) for cache in expected]


def _stored_report(tmp_path):
    """The report of two near-copies of a bare source, as it reads back after a round trip through JSON."""
    report = _measured(tmp_path, nf.Plan(("shift_1", "requantise_16")), _fake_scorer([]))
    return json.loads(json.dumps(report, default=str))


def test_the_report_names_its_sources_and_survives_json(tmp_path):
    """Each source records its scorings, the families that scored, its operating point and its output."""
    source = _stored_report(tmp_path)["sources"][0]
    assert source["scorings"] == 3
    assert source["families"] == {"dsp": "ok"}
    assert (source["operating_point"], source["output"]) == ("identity", None)


def test_the_report_names_its_settings_and_survives_json(tmp_path):
    """The model registry stays out of the report; the gate counts are readings like any other."""
    stored = _stored_report(tmp_path)
    assert "registry" not in stored["score_options"]
    assert stored["score_options"]["windows"] == 15.0
    assert stored["readings"]["gates.hard_failures"]["floor"] == 0.0


def test_deviations_skip_readings_missing_or_not_finite_on_either_side():
    """NaN, a boolean and a reading the reference lacks give no deviation."""
    instances = [("reference", {"a": 1.0, "b": 2.0, "g": 1.0}, {}), ("shift_1", {"a": 1.5, "b": float("nan"), "c": 3.0, "g": True}, {})]
    assert nf.deviations(instances) == {"a": [("shift_1", 0.5)]}


def test_the_summary_lists_the_widest_floors_first():
    """The console summary counts readings and pairs, names the operating point, then the widest floors."""
    entry = {"max": 1.0, "n": 2, "by_transform": {}}
    readings = {"narrow": {**entry, "floor": 0.1}, "wide": {**entry, "floor": 0.9}}
    report = {"sources": [{}], "operating_points": ["output"], "transforms": ["reference"], "readings": readings}
    lines = nf.render_summary(report, top=1).splitlines()
    assert lines[0].startswith("2 readings from 1 pair(s) at output")
    assert lines[1:] == ["  wide: floor 0.9 (max 1, n 2)"]


@pytest.fixture(scope="module", name="real_run")
def fixture_real_run(tmp_path_factory):
    """One command-line run through `runner.score_pair` (DSP family) on a SOURCE=OUTPUT pair: exit code, report, console."""
    tmp_path = tmp_path_factory.mktemp("real")
    pair = f"{_source(tmp_path, hiss=5e-3)}={_output(tmp_path)}"
    out = tmp_path / "floor.json"
    argv = [pair, "--out", str(out), "--cache-dir", str(tmp_path / "cache"), "--device", "cpu", "--transforms", "shift_1,requantise_16"]
    console = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(console):
        patch.setattr(nf.runner, "ModelRegistry", _registry)
        code = nf.main(argv)
    return code, json.loads(out.read_text(encoding="utf-8")), console.getvalue()


def _dsp_by_transform(report, name):
    return {key: entry["by_transform"][name] for key, entry in report["readings"].items() if key.startswith("dsp.")}


def test_the_command_line_run_succeeds_and_says_it_wrote_the_report(real_run):
    """End to end: the run exits 0 and its console reports the file it wrote."""
    code, _report, console = real_run
    assert code == 0
    assert "wrote" in console


def test_the_command_line_scores_a_restored_output_against_its_source(real_run):
    """End to end: the pair is read from SOURCE=OUTPUT and the report says it measured the output point."""
    _code, report, _console = real_run
    assert report["operating_points"] == ["output"]
    assert Path(report["sources"][0]["output"]).name == "output.wav"
    assert report["readings"]["gates.hard_failures"]["by_transform"] == {"shift_1": 0.0, "requantise_16": 0.0}


def test_a_one_sample_shift_of_the_output_moves_no_dsp_reading(real_run):
    """The alignment absorbs the shift: every DSP reading stays within 1e-3, far below what requantisation moves."""
    _code, report, _console = real_run
    shifted, requantised = _dsp_by_transform(report, "shift_1"), _dsp_by_transform(report, "requantise_16")
    assert len(shifted) > 20
    assert max(shifted.values()) < 1e-3 < max(requantised.values())


def test_a_pair_argument_splits_where_both_sides_exist(tmp_path):
    """SOURCE=OUTPUT splits at an '=' with a file on both sides; a bare source has no output."""
    source, output = _source(tmp_path), _output(tmp_path)
    assert nf.pair_arg(f"{source}={output}") == (source.resolve(), output.resolve())
    assert nf.pair_arg(str(source)) == (source.resolve(), None)


def test_a_path_holding_an_equals_sign_is_read_whole(tmp_path):
    """An existing file whose name holds '=' is a bare source, not a pair."""
    odd = tmp_path / "take=2.wav"
    sf.write(str(odd), _stereo(0.5), RATE, subtype="FLOAT")
    assert nf.pair_arg(str(odd)) == (odd.resolve(), None)


@pytest.mark.parametrize("make", [lambda root: f"{root / 'source.wav'}={root / 'absent.wav'}", lambda root: f"={root / 'source.wav'}"])
def test_a_pair_without_a_file_on_both_sides_is_refused(tmp_path, make):
    """A missing output, or nothing before the '=', is refused at parse time."""
    _source(tmp_path)
    # Arguments are built ahead of `pytest.raises`, so only the call under test can raise inside it.
    argument = make(tmp_path)
    with pytest.raises(argparse.ArgumentTypeError):
        nf.pair_arg(argument)


def test_the_argument_types_read_lists_and_numbers():
    """`all` names every family; an empty transform list scores only the reference."""
    assert nf.families_arg("all") == nf.runner.ALL_FAMILIES
    assert nf.transforms_arg(" shift_1, resample_roundtrip ") == ("shift_1", "resample_roundtrip")
    assert (nf.transforms_arg(""), nf.seconds_arg("0"), nf.repeats_arg("2")) == ((), 0.0, 2)
    assert nf.percentile_arg("100") == 100.0


@pytest.mark.parametrize(
    "extra",
    [
        ["--transforms", "shift_2"],
        ["--families", "dsp,ears"],
        ["--families", " , "],
        ["--repeats", "0"],
        ["--repeats", "1.5"],
        ["--windows", "-1"],
        ["--hop", "nan"],
        ["--percentile", "101"],
        ["--max-seconds", "-2"],
        ["--language", "romanian"],
        ["--device", "tpu"],
    ],
)
def test_the_command_line_refuses_bad_values(tmp_path, extra):
    """Every value is checked at parse time; argparse exits naming it."""
    argv = [str(_source(tmp_path)), "--out", str(tmp_path / "floor.json"), *extra]
    with pytest.raises(SystemExit):
        nf.parse_args(argv)


def test_a_missing_source_or_an_output_outside_every_root_is_refused(tmp_path, monkeypatch):
    """Inputs must exist; every path must lie in the repository, the temp directory or a listed data root."""
    monkeypatch.delenv("AI_RESTORE_DATA_ROOTS", raising=False)
    missing_source = [str(tmp_path / "absent.wav"), "--out", str(tmp_path / "floor.json")]
    with pytest.raises(SystemExit):
        nf.parse_args(missing_source)
    outside_out = [str(_source(tmp_path)), "--out", str(Path(tmp_path.anchor) / "outside_every_root" / "floor.json")]
    with pytest.raises(SystemExit):
        nf.main(outside_out)
