"""Corpus mode of the tuning driver: whole clips from a catalog, no cutting, cathar's probe note instead of a refusal."""

import json
from unittest.mock import patch

from scripts import tune_restoration as tr


def test_slug_fallback_keeps_identifiers_short_and_ascii():
    assert tr.slug_for("y-2mate.com-spot-goes-to-the-farm-1993-uk-vhs-144p", {}) == "y-2matecom-spot-goes-to-the-farm-1993-uk-vhs-144"
    assert tr.slug_for("\u00dcnknown tape", {}) == "unknown"


def test_catalog_clips_resolve_files_and_skip_missing(tmp_path):
    (tmp_path / "europe").mkdir()
    (tmp_path / "europe" / "a_home_15s.mp4").write_bytes(b"x")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps([{"identifier": "tape-a", "file": "europe\\a_home_15s.mp4"}, {"identifier": "b", "file": "europe\\missing.mp4"}]),
        encoding="utf-8",
    )
    clips = tr._catalog_clips(tmp_path, catalog)
    assert [(c.name, slug) for c, slug in clips] == [("a_home_15s.mp4", "tape-a")]


def test_catalog_clips_accept_the_region_dict_shape(tmp_path):
    (tmp_path / "c.mp4").write_bytes(b"x")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"europe": [{"identifier": "c", "file": "c.mp4"}]}), encoding="utf-8")
    assert [slug for _c, slug in tr._catalog_clips(tmp_path, catalog)] == ["c"]


def test_build_manifest_takes_whole_clips_without_cutting(tmp_path):
    clips_dir = tmp_path / "corpus"
    clips_dir.mkdir()
    (clips_dir / "one_15s.mp4").write_bytes(b"clip one")
    (clips_dir / "two_15s.mp4").write_bytes(b"clip two")
    grid = {"excerpt_s": 15, "margin_s": 0}
    with (
        patch.object(tr, "_extract", side_effect=lambda src, dst: dst.write_bytes(b"wav") or dst),
        patch.object(tr, "probe_duration", return_value=15.0),
        patch.object(tr, "cut_excerpt") as cut,
    ):
        manifest = tr.build_manifest(clips_dir, grid, tmp_path / "out", limit=1, whole=True)
    cut.assert_not_called()
    assert manifest["whole"] is True
    assert [(r["slug"], r["part"], r["start_s"]) for r in manifest["excerpts"]] == [("one15s", "whole", 0.0)]
    assert (tmp_path / "out" / "excerpts" / "one15s.mp4").read_bytes() == b"clip one"


def test_prepare_notes_the_probe_on_whole_clips_instead_of_refusing(tmp_path, capsys):
    grid = {"excerpt_s": 15, "margin_s": 0, "engines": {"cathar": {"process_mode": "cathar"}}, "variants": {"cathar": {"baseline": {}}}}
    manifest = {"whole": True, "excerpts": [{"probed_s": 15.0}]}
    args = type(
        "A", (), {"name": "t", "grid": tmp_path / "g.yaml", "tapes_dir": str(tmp_path), "limit": 0, "catalog": None, "whole": True}
    )()
    with (
        patch.object(tr, "load_grid", return_value=grid),
        patch.object(tr, "build_manifest", return_value=manifest),
        patch.object(tr, "_out_dir", return_value=tmp_path / "out"),
    ):
        tr.cmd_prepare(args)
    out = capsys.readouterr().out
    assert "note: cathar__baseline" in out
    assert "1 excerpts" in out
