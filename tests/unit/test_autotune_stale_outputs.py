"""The loop never collects an output it did not just render: one already beside a tape is moved aside first."""

import pytest

from scripts import autotune_restoration as at

SUFFIX = "_PureLinear_Cleaned"


@pytest.fixture(autouse=True)
def _any_free_space(monkeypatch):
    """The render's free-space floor is off: a CI runner has less than the 40 GB a Tata render needs."""
    monkeypatch.setattr(at.renders, "MIN_FREE_GB", 0.0)


def _tapes(tmp_path):
    folder = tmp_path / "tapes"
    folder.mkdir()
    tapes = {"a": folder / "alpha.mov", "b": folder / "beta.mp4"}
    for tape in tapes.values():
        tape.write_bytes(b"source")
    return {slug: str(path) for slug, path in tapes.items()}


def test_an_output_left_beside_a_tape_is_moved_into_the_run_and_named(tmp_path, capsys):
    tapes = _tapes(tmp_path)
    stale = tmp_path / "tapes" / f"alpha{SUFFIX}.mov"
    stale.write_bytes(b"an earlier candidate's render")
    at._move_stale_outputs("abc123", tapes, SUFFIX, tmp_path / "out")
    moved = tmp_path / "out" / "stale_outputs" / f"abc123__alpha{SUFFIX}.mov"
    assert (stale.exists(), moved.read_bytes()) == (False, b"an earlier candidate's render")
    assert sorted(p.name for p in (tmp_path / "tapes").iterdir()) == ["alpha.mov", "beta.mp4"]
    assert "moved alpha_PureLinear_Cleaned.mov" in capsys.readouterr().out


def test_with_nothing_beside_the_tapes_nothing_moves(tmp_path):
    at._move_stale_outputs("abc123", _tapes(tmp_path), SUFFIX, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_candidate_clears_stale_outputs_before_the_app_renders(tmp_path, monkeypatch):
    tapes = _tapes(tmp_path)
    order = []
    monkeypatch.setattr(at, "_prepare_candidate", lambda cid, cand_dir, *_a: cand_dir.mkdir(parents=True) or {})
    monkeypatch.setattr(at, "_move_stale_outputs", lambda *_a: order.append("clear"))
    monkeypatch.setattr(at, "_restore", lambda *_a: order.append("restore"))
    monkeypatch.setattr(at, "_collect_outputs", lambda *_a: order.append("collect"))
    monkeypatch.setattr(at, "stage_cache_counts", lambda _log: {"hits": 0, "misses": 0})
    (tmp_path / "out" / "cands").mkdir(parents=True)
    at.run_candidate("apl", {"linear_air_gain_db": 0.5}, tapes, tmp_path / "out")
    assert order == ["clear", "restore", "collect"]
