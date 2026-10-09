"""The calibration script: its cases, the score cache that re-scores a missing family, the per-route readings, and a whole run."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts import calibrate_quality_metrics as cal
from scripts import quality_degradations as deg
from scripts.restoration_quality import scorecard

RATE = 44100
SUBSET = ("underwater", "linear_bandwidth", "air_shelf_boost")
SCORED = []


def _materials(_fixtures, _language):
    """One second of a gated 220 Hz tone over faint noise stands in for every fixture."""
    rng = np.random.default_rng(1)
    voice = (0.1 * np.sin(2 * np.pi * 220.0 * np.arange(RATE) / RATE) * (np.arange(RATE) % 22050 < 11025)).astype(np.float32)
    noise = (1e-3 * rng.standard_normal(RATE)).astype(np.float32)
    keys = ("speech", "clean", "voice", "donor", "music")
    return {**{key: voice for key in keys}, "vhs": voice + noise, "noise": noise, "rate": RATE}


def _card(output_path, families):
    """A fake card: one speech and one music window, its reading set by the output file's name; only dsp scores."""
    value = 0.0 if "benign" in Path(output_path).name else 1.0 + len(Path(output_path).name) / 100.0
    rows = [
        scorecard.WindowRow(0, 0.0, 15.0, "speech", {"dsp.hf_4k8k": 0.0}, {"dsp.hf_4k8k": -value}),
        scorecard.WindowRow(1, 7.5, 22.5, "music", {"dsp.balance_air_db": 0.0}, {"dsp.balance_air_db": value}),
    ]
    status = {family: "ok" if family == "dsp" else "unavailable: ImportError: x" for family in families}
    card = scorecard.ScoreCard(rows=rows, families=status)
    card.aggregate = {**scorecard.aggregate(rows), "meta.prog_bandwidth_hz": {"source": {"median": 4000.0, "tail": 4000.0, "n": 1}}}
    return card


def _score_pair(_source, output, *, families, **_options):
    """The fake scorer: records the output it was asked for and returns its card."""
    SCORED.append(Path(output).name)
    return _card(output, families), None


@pytest.fixture(name="fake_scoring")
def _fake_scoring(monkeypatch):
    """Synthetic materials, three degradations, and a scorer that counts its calls instead of loading any model."""
    SCORED.clear()
    monkeypatch.setattr(cal, "_materials", _materials)
    monkeypatch.setattr(deg, "DEGRADATIONS", {name: deg.DEGRADATIONS[name] for name in SUBSET})
    monkeypatch.setattr(cal.runner, "score_pair", _score_pair)
    monkeypatch.setattr(cal.source_profile, "capture_profile", lambda _audio, _rate: {"prog_bandwidth_hz": 4000.0})
    return SCORED


def _registry():
    return SimpleNamespace(language=None)


def test_a_case_id_names_language_kind_failure_and_level():
    """The id keys the score cache: language, kind, failure and level."""
    case = cal.Case("degradation", "hiss", 20.0, "en", Path("s.wav"), Path("o.wav"))
    assert case.case_id == "en/degradation_hiss_20.0"
    assert case.base is None


def test_card_readings_take_the_delta_medians_and_the_profiles_source_side():
    """A change is read as its delta median, the capture profile as the source's own reading."""
    card = _card("x_output.wav", ("dsp",))
    readings = cal.card_readings(card.aggregate)
    assert readings["dsp.hf_4k8k"] == pytest.approx(-1.12)
    assert readings["meta.prog_bandwidth_hz"] == 4000.0


def test_route_readings_split_the_windows_and_keep_the_whole_file_readings():
    """Each route reads its own windows; the profile rides along on every route."""
    card = _card("x_output.wav", ("dsp",))
    routes = cal.route_readings(card, cal.card_readings(card.aggregate))
    assert "dsp.hf_4k8k" in routes["speech"]
    assert "dsp.hf_4k8k" not in routes["music"]
    assert routes["music"]["meta.prog_bandwidth_hz"] == 4000.0
    assert routes["mixed"] == {"meta.prog_bandwidth_hz": 4000.0}


def test_a_cached_score_covers_only_its_own_families_and_schema():
    """A dsp-only score never stands in for an all-family run, nor a score of another schema."""
    doc = {"schema": cal.SCORE_SCHEMA, "requested": ["dsp", "mos"]}
    assert cal.covers(doc, ("dsp",))
    assert not cal.covers(doc, ("dsp", "speech"))
    assert not cal.covers({"requested": ["dsp"]}, ("dsp",))


# The case filters live in helpers: radon counts a comprehension and every assert as a branch, so a test that
# holds four single assertions (Sonar S9073) keeps its own body free of comprehensions to stay at grade A.
def _of_kind(cases, kind):
    """The cases of one kind (degradation or benign), in build order."""
    return [case for case in cases if case.kind == kind]


@pytest.mark.usefixtures("fake_scoring")
def test_build_cases_writes_every_pair_with_its_base_and_scores_nothing(tmp_path):
    """Three levels per degradation plus the benign set; each degradation case points at its untouched material."""
    cases = cal.build_cases("unused", ["en"], tmp_path)
    degradations = _of_kind(cases, "degradation")
    assert len(degradations) == 3 * len(SUBSET)
    assert len(cases) == len(degradations) + len(deg.BENIGN)
    assert all(case.base.is_file() for case in degradations)
    assert not SCORED


@pytest.mark.usefixtures("fake_scoring")
def test_a_benign_case_has_no_base(tmp_path):
    """A benign transform has no source condition to check."""
    assert {case.base for case in cal.build_cases("unused", ["en"], tmp_path) if case.kind == "benign"} == {None}


@pytest.mark.usefixtures("fake_scoring")
def test_scores_are_cached_and_scored_again_when_a_family_is_added(tmp_path):
    """The second dsp run reuses the cache; asking for mos too scores both cases again."""
    cases = cal.build_cases("unused", ["en"], tmp_path)[:2]
    cal.score_cases(cases, ("dsp",), _registry(), tmp_path, tmp_path / "scores")
    cal.score_cases(cases, ("dsp",), _registry(), tmp_path, tmp_path / "scores")
    scores = cal.score_cases(cases, ("dsp", "mos"), _registry(), tmp_path, tmp_path / "scores")
    assert len(SCORED) == 4
    assert scores[cases[0].case_id]["requested"] == ["dsp", "mos"]


def _named(cases, name):
    """The cases of one degradation or benign transform, in build order."""
    return [case for case in cases if case.name == name]


@pytest.mark.usefixtures("fake_scoring")
def test_a_source_condition_case_reads_its_bases_profile(tmp_path):
    """The base material's own bandwidth rides along as `base.*`, pooled and per route."""
    cases = _named(cal.build_cases("unused", ["en"], tmp_path), "linear_bandwidth")
    scores = cal.score_cases(cases, ("dsp",), _registry(), tmp_path, tmp_path / "scores")
    readings = cal.readings_of(scores)[cases[0].case_id]
    assert readings["base.prog_bandwidth_hz"] == 4000.0
    assert readings["meta.prog_bandwidth_hz"] == 4000.0
    assert cal.readings_of(scores, "music")[cases[0].case_id]["base.prog_bandwidth_hz"] == 4000.0


def test_family_status_counts_ok_and_unavailable():
    """Per family, how many cases scored and why the rest did not."""
    scores = {"a": {"families": {"dsp": "ok", "mos": "unavailable: ImportError: x"}}, "b": {"families": {"dsp": "ok"}}}
    status = cal.family_status(scores)
    assert status == {"dsp": {"ok": 2}, "mos": {"unavailable": 1}}
    assert cal.scored_families(status) == ["dsp"]


def test_the_round_id_is_explicit_or_the_verdict_rounds_or_the_manifest():
    """`--round` wins; else the ledger rounds the verdicts came from; else the manifest."""
    assert cal.round_id("session0", {}) == "session0"
    assert cal.round_id(None, {"t": {"rounds": ["2", "1"]}, "u": {"rounds": ["1"]}}) == "1+2"
    assert cal.round_id(None, {"t": {}}) == "manifest"


def _run(tmp_path, *extra):
    """A whole calibration on the fakes: dsp and mos asked for, mos unavailable, the repository's first manifest."""
    manifest = Path(cal.REPO) / "assets" / "quality_calibration" / "known_ordering.json"
    argv = ["--fixtures", "unused", "--out", str(tmp_path), "--metrics", "dsp,mos", "--known-ordering", str(manifest)]
    return cal.main([*argv, "--no-ledger", *extra])


def _score_variants(_source, outputs, **options):
    """The fake known-ordering scorer: one card per labelled output."""
    variants = {label: _variant_doc(_card(f"{label}_output.wav", options["families"])) for label in outputs}
    return {"schema": 1, "variants": variants}, {}


@pytest.fixture(name="fake_ordering")
def _fake_ordering(monkeypatch):
    """The known-ordering tapes scored by the fake instead of the real tapes."""
    monkeypatch.setattr(cal.cordering.runner, "score_variants", _score_variants)


def _variant_doc(card):
    """One variant as `runner.card_to_dict` records it, the parts the calibration reads."""
    rows = [
        {"window": r.window, "start_s": r.start_s, "end_s": r.end_s, "route": r.route, "source": r.source, "output": r.output}
        for r in card.rows
    ]
    verdicts = cal.cordering.gates_mod.evaluate_gates(card.aggregate)
    return {"rows": rows, "aggregate": card.aggregate, "verdicts": verdicts, "hard_failures": [], "families": card.families}


def _names_in(directory):
    """The names of the files a run left in `directory`."""
    return {path.name for path in directory.iterdir()}


@pytest.mark.usefixtures("fake_scoring", "fake_ordering")
def test_a_whole_run_writes_the_pooled_and_per_route_gates_and_the_report(tmp_path, capsys):
    """Every gates file and both reports are written; a family that did not score is said, with its gates."""
    _run(tmp_path)
    written = _names_in(tmp_path)
    assert {"gates.json", "gates_speech.json", "gates_music.json", "gates_mixed.json", "report.json", "report.md"} <= written
    report = json.loads((tmp_path / "report.json").read_text("utf-8"))
    assert report["round"] == "manifest"
    assert report["skipped_gates"]["mos.sigmos_col"] == "family mos not scored in this run"
    assert "families not scored" in capsys.readouterr().out


@pytest.mark.usefixtures("fake_scoring", "fake_ordering")
def test_a_second_run_reads_the_first_runs_gates_as_the_previous_round(tmp_path):
    """The gates carry the new round; only a verdict-derived threshold counts agreeing rounds."""
    _run(tmp_path, "--round", "1")
    _run(tmp_path, "--round", "2")
    gates = json.loads((tmp_path / "gates.json").read_text("utf-8"))
    assert {entry["round"] for entry in gates.values()} == {"2"}
    assert all(entry["rounds_agreeing"] == 0 for entry in gates.values() if entry["source"] != "known ordering")


def test_the_exit_code_follows_the_non_blind_failures():
    """A blind pair is reported, never asserted."""
    assert cal.non_blind_failures([{"status": "fail", "blind": True}, {"status": "pass", "blind": False}]) == []
    assert len(cal.non_blind_failures([{"status": "fail", "blind": False}])) == 1


def test_the_benign_floor_is_still_reachable_from_the_script():
    """`reward_noise_floor.py` cites the floor as `calibrate_quality_metrics.noise_floor`."""
    assert cal.noise_floor is cal.checks_mod.noise_floor


@pytest.mark.usefixtures("fake_scoring")
def test_a_written_case_is_kept_on_the_next_build(tmp_path):
    """A case already on disk is not written again (the cache keys on its path)."""
    first = cal.build_cases("unused", ["en"], tmp_path)[0]
    sf.write(str(first.output), np.zeros(10, dtype=np.float32), RATE, subtype="FLOAT")
    again = cal.build_cases("unused", ["en"], tmp_path)[0]
    assert again.output == first.output
    assert sf.info(str(again.output)).frames == 10


@pytest.mark.usefixtures("fake_scoring")
def test_a_tape_excerpt_carries_only_the_degradations_it_can(tmp_path):
    """The speech and capture degradations and the speech benign set go in, as Romanian; the cut capture is its own output."""
    excerpt = tmp_path / "Tele7abc_125s.wav"
    sf.write(str(excerpt), _materials(None, None)["vhs"], RATE, subtype="FLOAT")
    cases = cal.build_tape_cases([excerpt], tmp_path)
    assert {case.name for case in cases if case.kind == "degradation"} == set(SUBSET)
    assert {case.name for case in cases if case.kind == "benign"} == set(deg.on_tape_benign())
    assert {cal.whisper_language(case) for case in cases} == {"ro"}
    assert cases[0].language == "ro-Tele7abc_125s"


def test_no_excerpt_adds_no_case(tmp_path):
    """Without `--excerpts` the suite is the fixtures' alone."""
    assert not cal.build_tape_cases(None, tmp_path)


def test_the_default_languages_carry_a_voice_whose_windows_hold_enough_fricatives():
    """en alone left the texture check on one window of 10 fricative frames; fr holds 102."""
    assert cal._parse_args([]).languages == ["en", "fr"] == list(cal.DEFAULT_LANGUAGES)
