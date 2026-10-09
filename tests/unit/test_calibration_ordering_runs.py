"""Scoring the known-ordering tapes: the cache that never serves a dsp-only result to an all-family run, the re-gating, the routes."""

import functools
import json
from pathlib import Path

import pytest

from scripts import calibration_ordering as co
from scripts.restoration_quality.gates import FLAG, GATES, Gate

REPO = Path(co.__file__).resolve().parent.parent
LISTEN_GATES = ("listener.dead_air", "listener.pause_collapse", "listener.hiss", "listener.sibilance_thin", "listener.sibilance_dull")


def _row(window, route, air):
    return {
        "window": window,
        "start_s": 7.5 * window,
        "end_s": 7.5 * window + 15.0,
        "route": route,
        "source": {"dsp.gap_air_db": 0.0},
        "output": {"dsp.gap_air_db": air},
    }


def _scored():
    """A fake `runner.score_variants`: two variants, one speech and one music window each, every call counted."""
    calls = []

    def score_variants(source, outputs, **options):
        """The fake `runner.score_variants`: every call recorded, one speech and one music window per variant."""
        calls.append((source, sorted(outputs), tuple(options["families"])))
        variants = {label: _variant_doc(index) for index, label in enumerate(outputs)}
        return {"schema": 1, "variants": variants}, {}

    return score_variants, calls


def _variant_doc(index):
    rows = [_row(0, "speech", -10.0 - 20.0 * index), _row(1, "music", -30.0)]
    aggregate = {
        "dsp.gap_air_db": {"output": {"median": -10.0 - 20.0 * index, "tail": -10.0, "n": 2}},
        "file.lufs": {"delta": {"median": 0.5}},
    }
    return {"rows": rows, "aggregate": aggregate, "verdicts": [], "hard_failures": [], "families": {"dsp": "ok"}, "speaker_floor": None}


def _scoring(families, out_dir):
    return {"families": families, "registry": None, "cache_dir": out_dir, "out_dir": out_dir}


@pytest.fixture(name="manifest")
def _manifest(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    lists = {"flags": {"listener.dead_air": ["dead"]}, "clean": {"listener.dead_air": ["live"]}}
    path.write_text(
        json.dumps({"tele": {"source": "tele.wav", "variants": {"live": "a.wav", "dead": "b.wav"}, "by_ear_rank": {"live": 1}, **lists}}),
        "utf-8",
    )
    monkeypatch.setattr(co, "manifest_inside_repo", Path)
    return path


def test_each_tape_is_scored_once_and_kept(manifest, tmp_path, monkeypatch):
    """Each tape is scored once and kept."""
    score_variants, calls = _scored()
    monkeypatch.setattr(co.runner, "score_variants", score_variants)
    first = co.run_known_ordering(manifest, _scoring(("dsp",), tmp_path), {})
    co.run_known_ordering(manifest, _scoring(("dsp",), tmp_path), {})
    assert len(calls) == 1
    assert json.loads((tmp_path / "ordering" / "tele.json").read_text("utf-8"))["requested"] == ["dsp"]
    assert first["tele"]["verdict_source"] == "manifest"


def test_a_result_scored_without_a_family_now_asked_for_is_scored_again(manifest, tmp_path, monkeypatch):
    """A result scored without a family now asked for is scored again."""
    score_variants, calls = _scored()
    monkeypatch.setattr(co.runner, "score_variants", score_variants)
    co.run_known_ordering(manifest, _scoring(("dsp",), tmp_path), {})
    co.run_known_ordering(manifest, _scoring(("dsp", "mos"), tmp_path), {})
    assert [call[2] for call in calls] == [("dsp",), ("dsp", "mos")]


def test_the_rules_are_read_again_under_the_derived_gates(manifest, tmp_path, monkeypatch):
    """The rules are read again under the derived gates."""
    monkeypatch.setattr(co.runner, "score_variants", _scored()[0])
    ordering = co.run_known_ordering(manifest, _scoring(("dsp",), tmp_path), {})
    strict = {"listener.dead_air": Gate("dsp.gap_air_db", "median", ">=", -25.0, severity=FLAG)}
    assert co.reevaluate_ordering(ordering, strict, {})["tele"]["rules"]["flags_reproduced"] is True
    loose = {"listener.dead_air": Gate("dsp.gap_air_db", "median", ">=", -40.0, severity=FLAG)}
    after = co.reevaluate_ordering(ordering, loose, {})["tele"]
    assert after["rules"]["flags_reproduced"] is False and after["flags"] == {"listener.dead_air": ["dead"]}


def test_a_route_view_reads_only_that_routes_windows_and_keeps_the_file_readings(manifest, tmp_path, monkeypatch):
    """A route view reads only that routes windows and keeps the file readings."""
    monkeypatch.setattr(co.runner, "score_variants", _scored()[0])
    ordering = co.run_known_ordering(manifest, _scoring(("dsp",), tmp_path), {})
    music = co.route_ordering(ordering, "music", GATES, {})["tele"]["result"]["variants"]["dead"]["aggregate"]
    speech = co.route_ordering(ordering, "speech", GATES, {})["tele"]["result"]["variants"]["dead"]["aggregate"]
    assert "dsp.gap_air_db" not in music
    assert speech["dsp.gap_air_db"]["output"]["median"] == -30.0 and music["file.lufs"] == speech["file.lufs"]


def test_a_manifest_outside_the_repository_is_refused(tmp_path):
    """A manifest outside the repository is refused."""
    path = tmp_path / "manifest.json"
    path.write_text("{}", "utf-8")
    with pytest.raises(SystemExit):
        co.manifest_inside_repo(path)
    assert co.manifest_inside_repo(REPO / "assets" / "quality_calibration" / "known_ordering.json").is_file()


@functools.lru_cache(maxsize=None)
def _asset(name):
    return json.loads((REPO / "assets" / "quality_calibration" / name).read_text(encoding="utf-8"))


def _listen():
    listen = _asset("known_ordering_v2.json")["tele7abc_listen"]
    return listen, set(listen["variants"])


def test_v2_manifest_keeps_the_old_tapes():
    """V2 manifest keeps the old tapes."""
    v1, v2 = _asset("known_ordering.json"), _asset("known_ordering_v2.json")
    assert {tape: v2[tape] for tape in v1} == v1


def test_the_listened_tele7abc_has_thirteen_variants_of_a_mov_with_lists_for_every_listen_gate():
    """The listened tele7abc has thirteen variants of a mov with lists for every listen gate."""
    listen, labels = _listen()
    assert len(labels) == 13
    assert listen["source"].endswith(".mov")
    assert set(listen["flags"]) == set(listen["clean"]) == set(LISTEN_GATES)


@pytest.mark.parametrize("gate", LISTEN_GATES)
def test_each_listen_gate_is_a_listener_gate_whose_lists_name_disjoint_listened_labels(gate):
    """A flag, or display-only (`listener.hiss` since ear v3: gap level does not order the verdicts)."""
    listen, labels = _listen()
    assert GATES[gate].severity in (FLAG, "soft")
    assert set(listen["flags"][gate]) | set(listen["clean"][gate]) <= labels
    assert not set(listen["flags"][gate]) & set(listen["clean"][gate])


def test_the_by_ear_ranks_name_listened_labels_and_leave_the_finals_unranked():
    """The by ear ranks name listened labels and leave the finals unranked."""
    listen, labels = _listen()
    assert set(listen["by_ear_rank"]) <= labels
    assert listen["by_ear_rank"]["cathar075__alpha_2_0"] == 1
    assert {"final_apl", "final_cathar"} <= labels - set(listen["by_ear_rank"])


def test_the_listened_tape_reads_its_flags_from_the_ledger():
    """`tele7abc_listen` belongs to the ledger's `tele7abc`; its round-1 flag record names the same flags, and `*`."""
    listen, _labels = _listen()
    verdicts = co.tape_verdicts("tele7abc_listen", listen, co.ledger_records())
    assert verdicts["verdict_source"] == "ledger" and "1" in verdicts["rounds"]
    assert verdicts["flags"]["listener.hiss"] == sorted(listen["flags"]["listener.hiss"])
