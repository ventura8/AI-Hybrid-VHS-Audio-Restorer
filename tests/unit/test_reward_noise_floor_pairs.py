"""The reward noise floor's benign-pair point: parsing, the A-B deviations, the report, one point per report, confinement."""

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts import cli_paths
from scripts import reward_noise_floor as nf
from tests.unit.test_reward_noise_floor import KEY, RATE, _card, _registry

LEDGER_ID = "2026-10-09-pair-tele7abc-cedee2-1"
# What the fake scorer reads per output file name; anything else reads 1.0.
READS = {"b.wav": 1.25, "take=2,b.wav": 1.5}


def _wav(path, seconds=0.5, seed=0):
    rng = np.random.default_rng(seed)
    sf.write(str(path), (0.01 * rng.standard_normal((int(seconds * RATE), 2))).astype(np.float32), RATE, subtype="FLOAT")
    return path


def _trio(root, names=("source.wav", "a.wav", "b.wav"), seconds=0.5):
    """A source and two restorations of it, written into `root`."""
    return [_wav(root / name, seconds, seed) for seed, name in enumerate(names)]


def _fake_scorer(calls, drift=0.0):
    """A scorer that reads READS by the output's name (1.0 otherwise), plus `drift` per earlier call."""

    def score(source, output, **options):
        calls.append((source, output, options))
        return _card(READS.get(Path(output).name, 1.0) + drift * (len(calls) - 1)), None

    return score


def _measured(tmp_path, entries, plan=None, calls=None, drift=0.0):
    scorer = _fake_scorer([] if calls is None else calls, drift)
    return nf.measure(entries, plan or nf.Plan(), cache_dir=tmp_path / "cache", scorer=scorer, windows=15.0, registry=_registry())


def _args(tmp_path, *extra):
    return [*extra, "--out", str(tmp_path / "floor.json"), "--cache-dir", str(tmp_path / "cache")]


def test_a_benign_pair_reads_paths_holding_equals_signs_and_commas(tmp_path):
    """Each path is its own value, so neither '=' nor ',' splits it; the ledger id is kept as given."""
    source, first, second = _trio(tmp_path, ("src=1,a.wav", "take=1,a.wav", "take=2,b.wav"))
    args = nf.parse_args(_args(tmp_path, "--benign-pair", str(source), str(first), str(second), LEDGER_ID))
    assert args.benign_pairs == [nf.BenignPair(source.resolve(), first.resolve(), second.resolve(), LEDGER_ID)]
    assert (args.sources, args.pool) == ([], False)


def test_the_ledger_id_is_optional_and_pairs_repeat(tmp_path):
    """Three values make a pair without a ledger record; the option repeats, one pair per use."""
    source, first, second = (str(path) for path in _trio(tmp_path))
    args = nf.parse_args(_args(tmp_path, "--benign-pair", source, first, second, "--benign-pair", source, second, first, LEDGER_ID))
    assert [(pair.output_a.name, pair.ledger_id) for pair in args.benign_pairs] == [("a.wav", None), ("b.wav", LEDGER_ID)]


@pytest.mark.parametrize(
    "values",
    [
        lambda paths: paths[:2],
        lambda paths: [*paths, LEDGER_ID, "extra"],
        lambda paths: [paths[0], paths[1], paths[1]],
        lambda paths: [paths[0], paths[1], str(Path(paths[2]).with_name("absent.wav"))],
    ],
    ids=["two values", "five values", "one file twice", "a missing output"],
)
def test_a_malformed_benign_pair_is_refused_at_parse_time(tmp_path, values):
    """Three or four values, three existing paths, two different outputs: argparse exits otherwise."""
    paths = [str(path) for path in _trio(tmp_path)]
    argv = _args(tmp_path, "--benign-pair", *values(paths))
    with pytest.raises(SystemExit):
        nf.parse_args(argv)


def test_an_entry_swallowed_as_the_ledger_id_is_refused_with_a_hint(tmp_path, capsys):
    """A SOURCE=OUTPUT written after a pair would be its fourth value: not a record id, so the message says where it goes."""
    source, first, second = (str(path) for path in _trio(tmp_path))
    argv = _args(tmp_path, "--benign-pair", source, first, second, f"{source}={first}")
    with pytest.raises(SystemExit):
        nf.parse_args(argv)
    assert "not a ledger record id" in capsys.readouterr().err


def test_no_entry_at_all_is_refused(tmp_path, capsys):
    """Neither a positional entry nor a benign pair: nothing to measure."""
    argv = _args(tmp_path)
    with pytest.raises(SystemExit):
        nf.parse_args(argv)
    assert "--benign-pair" in capsys.readouterr().err


def test_points_mix_only_when_pooled_explicitly(tmp_path, capsys):
    """An output pair beside a benign pair is refused, naming both points, unless --pool asks for one report."""
    source, first, second = (str(path) for path in _trio(tmp_path))
    mixed = _args(tmp_path, f"{source}={first}", "--benign-pair", source, first, second)
    with pytest.raises(SystemExit):
        nf.parse_args(mixed)
    assert "benign_pair, output" in capsys.readouterr().err
    pooled = nf.parse_args([*mixed, "--pool"])
    assert (len(pooled.sources), len(pooled.benign_pairs), pooled.pool) == (1, 1, True)


def test_every_entry_kind_names_its_operating_point(tmp_path):
    """A bare source is the identity point, a (source, output) pair the output point, a BenignPair its own."""
    source, first, second = _trio(tmp_path)
    entries = [source, (source, None), (source, first), nf.BenignPair(source, first, second)]
    assert [nf.point_of(entry) for entry in entries] == ["identity", "identity", "output", "benign_pair"]
    assert nf.require_one_point(entries[2:], pool=True) == ["benign_pair", "output"]
    with pytest.raises(ValueError, match="--pool"):
        nf.require_one_point(entries[2:])


def test_a_benign_pair_scores_both_outputs_against_the_untouched_source(tmp_path):
    """A is the reference, B follows under `benign_pair`; no near-copy of either is written."""
    calls = []
    source, first, second = _trio(tmp_path)
    report = _measured(tmp_path, [nf.BenignPair(source, first, second, LEDGER_ID)], calls=calls)
    assert [(Path(call[0]), Path(call[1]).name) for call in calls] == [(source, "a.wav"), (source, "b.wav")]
    assert report["readings"][KEY]["by_transform"] == pytest.approx({"benign_pair": 0.25})
    assert not list((tmp_path / "cache").rglob("*_shift_1.wav"))


def test_repeats_add_the_scorers_jitter_beside_the_pairs_difference(tmp_path):
    """Two repeats: A's re-scoring lands under `reference`, both B scorings under `benign_pair`, all against A's first."""
    calls = []
    report = _measured(tmp_path, [nf.BenignPair(*_trio(tmp_path))], nf.Plan(repeats=2), calls, drift=0.01)
    entry = report["readings"][KEY]
    assert (len(calls), entry["n"]) == (4, 3)
    assert entry["by_transform"] == pytest.approx({"benign_pair": 0.28, "reference": 0.02})
    assert entry["max"] == pytest.approx(0.28)
    assert 0.25 < entry["floor"] <= 0.28


def test_the_report_records_the_pair_and_its_ledger_record(tmp_path):
    """sources[] names A, B, their WAVs, the ledger id and the names scored; the report lists the point and survives JSON."""
    source, first, second = _trio(tmp_path)
    report = json.loads(json.dumps(_measured(tmp_path, [nf.BenignPair(source, first, second, LEDGER_ID)]), default=str))
    described = report["sources"][0]
    assert (described["output"], described["output_b"], described["output_b_wav"]) == (str(first), str(second), str(second))
    assert (described["ledger_id"], described["operating_point"], described["transforms"]) == (
        LEDGER_ID,
        "benign_pair",
        ["reference", "benign_pair"],
    )
    assert (report["operating_points"], report["transforms"], report["pool"], report["schema"]) == (
        ["benign_pair"],
        ["reference", "benign_pair"],
        False,
        3,
    )


def test_max_seconds_cuts_the_source_and_both_outputs_alike(tmp_path):
    """Every side of a benign pair is cut to the same head before scoring."""
    material = nf.prepared(nf.BenignPair(*_trio(tmp_path, seconds=1.0)), tmp_path / "cache", 0.5)
    frames = [sf.info(str(path)).frames for path in (material.source_wav, material.output_wav, material.output_b_wav)]
    assert frames == [RATE // 2] * 3
    assert material.operating_point == "benign_pair"


def test_a_pooled_report_keeps_what_each_kind_contributed(tmp_path):
    """Pooled, the floor reads every deviation; by_transform separates the near-copies from the benign pair."""
    source, first, second = _trio(tmp_path)
    entries = [(source, first), nf.BenignPair(source, first, second)]
    report = _measured(tmp_path, entries, nf.Plan(transforms=("shift_1",), pool=True))
    assert report["readings"][KEY]["by_transform"] == pytest.approx({"shift_1": 0.0, "benign_pair": 0.25})
    assert (report["operating_points"], report["transforms"]) == (["benign_pair", "output"], ["reference", "shift_1", "benign_pair"])
    with pytest.raises(ValueError):
        _measured(tmp_path, entries)


def test_the_command_line_writes_a_benign_pair_report(tmp_path, monkeypatch):
    """End to end with the scorer faked: the pair's paths hold '=' and ',', the report and the console name the point."""
    source, first, second = map(str, _trio(tmp_path, ("src.wav", "take=1,a.wav", "take=2,b.wav")))
    monkeypatch.setattr(nf.runner, "ModelRegistry", _registry)
    monkeypatch.setattr(nf.runner, "score_pair", _fake_scorer([]))
    console = io.StringIO()
    with contextlib.redirect_stdout(console):
        code = nf.main(_args(tmp_path, "--benign-pair", source, first, second, LEDGER_ID, "--device", "cpu"))
    report = json.loads((tmp_path / "floor.json").read_text(encoding="utf-8"))
    assert code == 0
    assert "at benign_pair" in console.getvalue()
    assert report["readings"][KEY]["floor"] == pytest.approx(0.5)
    assert report["sources"][0]["ledger_id"] == LEDGER_ID


def test_a_benign_pair_outside_every_data_root_is_refused(tmp_path, monkeypatch):
    """The pair's paths are confined where they are used: a file outside the allowed roots stops the run, named."""
    source, first, second = (str(path) for path in _trio(tmp_path))
    monkeypatch.setattr(cli_paths, "allowed_roots", lambda: (tmp_path / "elsewhere",))
    argv = _args(tmp_path, "--benign-pair", source, first, second)
    with pytest.raises(SystemExit, match="--benign-pair SOURCE"):
        nf.main(argv)
