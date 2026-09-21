"""The tuning driver's pure logic: excerpt planning, override validation, config materialisation, ranking, gates."""

from unittest.mock import patch

import pytest
import yaml

from scripts import tune_restoration as tr


def test_plan_excerpts_by_tape_length():
    assert [p for p, _s in tr.plan_excerpts(1320.0, 125.0, 5.0)] == ["start", "middle", "end"]
    assert [p for p, _s in tr.plan_excerpts(300.0, 125.0, 5.0)] == ["start", "end"]
    assert tr.plan_excerpts(132.0, 125.0, 5.0) == [("whole", 3.5)]
    assert tr.plan_excerpts(100.0, 125.0, 5.0) == []


def test_plan_excerpts_keeps_the_margins():
    parts = dict(tr.plan_excerpts(540.0, 125.0, 5.0))
    assert parts["start"] == 5.0
    assert parts["end"] == 540.0 - 5.0 - 125.0
    assert parts["middle"] == (540.0 - 125.0) / 2.0


def test_slug_for_maps_the_tata_names_and_folds_diacritics():
    assert tr.slug_for("Interviu Tele7abc Dr. Alexandrescu Cristian Feb 1997_stab_starlight_mini") == "tele7abc"
    assert tr.slug_for("\u021ai-ai vaccinat feti\u021ba \u00eempotriva cancerului 2010_starlight_mini") == "vaccin"
    assert tr.slug_for("Campanie Electorala, Mai 1992") == "campanie"
    assert tr.slug_for("\u00dcnknown tape") == "unknown"


def test_validate_overrides_against_the_apps_own_tables():
    fields = tr.known_config_fields()
    assert tr.validate_overrides({"cathar_alpha": 2.0, "cathar_enable_enhance": False, "cathar_denoise_method": "wiener"}, fields) == []
    problems = tr.validate_overrides(
        {"no_such_key": 1, "cathar_alpha": True, "cathar_denoise_method": "magic", "apl_neural_model": "C:/x.ckpt"}, fields
    )
    assert len(problems) == 4


def test_validate_overrides_checks_bounds():
    fields = {"x": (float, 0.0, 1.0)}
    assert tr.validate_overrides({"x": 2.0}, fields) == ["x: 2.0 is over the maximum 1.0"]
    assert tr.validate_overrides({"x": -1.0}, fields) == ["x: -1.0 is under the minimum 0.0"]
    assert tr.validate_overrides({"x": "no"}, fields)[0].startswith("x: expected a number")


def test_check_probe_material_refuses_a_probe_the_excerpt_cannot_feed():
    grid = {
        "engines": {"cathar": {"process_mode": "cathar"}, "apl": {"process_mode": "auto_pure_linear"}},
        "variants": {
            "cathar": {"baseline": {}, "long": {"cathar_noiseprint_duration_s": 9.0}},
            "apl": {"long": {"cathar_noiseprint_duration_s": 9.0}},
        },
    }
    manifest = {"excerpts": [{"probed_s": 124.98}]}
    problems = tr.check_probe_material(grid, manifest)
    assert len(problems) == 1
    assert problems[0].startswith("cathar__long")


def test_check_probe_material_covers_every_cathar_engine():
    grid = {
        "engines": {"cathar": {"process_mode": "cathar"}, "cathar075": {"process_mode": "cathar", "env": {"AI_RESTORE_CATHAR_BIN": "x"}}},
        "variants": {"cathar": {"baseline": {}}, "cathar075": {"long": {"cathar_noiseprint_duration_s": 9.0}}},
    }
    problems = tr.check_probe_material(grid, {"excerpts": [{"probed_s": 124.98}]})
    assert [problem.split(":")[0] for problem in problems] == ["cathar075__long"]


def test_engine_env_makes_repo_paths_absolute_and_keeps_other_values():
    env = tr.engine_env({"env": {"AI_RESTORE_CATHAR_BIN": "scripts/tune_restoration.py", "OTHER": "plain"}})
    assert env["AI_RESTORE_CATHAR_BIN"] == str((tr.REPO / "scripts" / "tune_restoration.py").resolve())
    assert env["OTHER"] == "plain"
    assert tr.engine_env({"process_mode": "cathar"}) == {}


def test_materialise_config_sets_mode_applies_overrides_and_keeps_unknown_keys():
    text = "process_mode: auto\ncathar_alpha: 2.5\nsomething_else: 7\n"
    out = yaml.safe_load(tr.materialise_config(text, "cathar", {"cathar_alpha": 3.5}))
    assert out["process_mode"] == "cathar"
    assert out["cathar_alpha"] == 3.5
    assert out["something_else"] == 7


def test_materialised_config_is_what_the_app_resolves(tmp_path, monkeypatch):
    text = (tr.REPO / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(tr.materialise_config(text, "cathar", {"cathar_alpha": 3.25}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    resolved, _source = tr.config_mod.load_config()
    assert resolved["cathar_alpha"] == 3.25
    assert resolved["process_mode"] == "cathar"


def test_assert_overrides_resolved_flags_a_reverted_value():
    assert (
        tr.assert_overrides_resolved(
            {"cathar_alpha": 3.5, "cathar_enable_enhance": False}, {"cathar_alpha": 3.5, "cathar_enable_enhance": False}
        )
        == []
    )
    assert tr.assert_overrides_resolved({"cathar_alpha": 3.5}, {"cathar_alpha": 2.5})[0].startswith("cathar_alpha")


def test_variant_ids_filter_by_engine_and_id():
    grid = {"variants": {"cathar": {"baseline": {}, "a": {"cathar_alpha": 2.0}}, "apl": {"baseline": {}}}}
    assert [row[0] for row in tr.variant_ids(grid)] == ["cathar__baseline", "cathar__a", "apl__baseline"]
    assert [row[0] for row in tr.variant_ids(grid, engines=["apl"])] == ["apl__baseline"]
    assert tr.variant_ids(grid, only=["cathar__a"])[0][3] == {"cathar_alpha": 2.0}


def test_run_variant_launches_a_fresh_interpreter_in_the_variant_dir(tmp_path):
    variant_dir, excerpts_dir = tmp_path / "runs" / "cathar__x", tmp_path / "excerpts"
    excerpts_dir.mkdir(parents=True)
    (excerpts_dir / "soti_start.mov").write_bytes(b"clip")
    manifest = {"excerpts": [{"excerpt": "soti_start.mov", "slug": "soti"}]}
    with (
        patch.object(tr, "resolved_config", return_value={"cathar_alpha": 3.5}),
        patch.object(tr, "is_valid_video", return_value=False),
        patch.object(tr.subprocess, "run") as run,
    ):
        timing = tr.run_variant(
            variant_dir, {"process_mode": "cathar", "suffix": "_Cathar_Cleaned"}, {"cathar_alpha": 3.5}, manifest, excerpts_dir, "py"
        )
    assert run.call_args.kwargs["cwd"] == str(variant_dir)
    assert run.call_args.kwargs["stdin"] is tr.subprocess.DEVNULL
    assert str(variant_dir / "excerpts" / "soti_start.mov") in run.call_args[0][0]
    assert timing["n"] == 1


def test_run_variant_hands_the_engine_env_to_the_child(tmp_path):
    variant_dir, excerpts_dir = tmp_path / "runs" / "cathar075__x", tmp_path / "excerpts"
    excerpts_dir.mkdir(parents=True)
    (excerpts_dir / "soti_start.mov").write_bytes(b"clip")
    manifest = {"excerpts": [{"excerpt": "soti_start.mov", "slug": "soti"}]}
    spec = {"process_mode": "cathar", "suffix": "_Cathar_Cleaned", "env": {"AI_RESTORE_CATHAR_BIN": "other-cathar"}}
    with (
        patch.object(tr, "resolved_config", return_value={}) as resolved,
        patch.object(tr, "is_valid_video", return_value=False),
        patch.object(tr.subprocess, "run") as run,
    ):
        tr.run_variant(variant_dir, spec, {}, manifest, excerpts_dir, "py")
    assert run.call_args.kwargs["env"]["AI_RESTORE_CATHAR_BIN"] == "other-cathar"
    assert resolved.call_args[0][2] == {"AI_RESTORE_CATHAR_BIN": "other-cathar"}


def test_run_variant_skips_a_complete_variant_and_refuses_reverted_overrides(tmp_path):
    variant_dir, excerpts_dir = tmp_path / "runs" / "cathar__x", tmp_path / "excerpts"
    variant_dir.mkdir(parents=True)
    excerpts_dir.mkdir()
    manifest = {"excerpts": [{"excerpt": "soti_start.mov", "slug": "soti"}]}
    with patch.object(tr, "is_valid_video", return_value=True), patch.object(tr.subprocess, "run") as run:
        assert tr.run_variant(variant_dir, {"process_mode": "cathar", "suffix": "_x"}, {}, manifest, excerpts_dir, "py") == {
            "skipped": True
        }
    run.assert_not_called()
    (excerpts_dir / "soti_start.mov").write_bytes(b"clip")
    with patch.object(tr, "resolved_config", return_value={"cathar_alpha": 2.5}), patch.object(tr, "is_valid_video", return_value=False):
        with pytest.raises(SystemExit, match="did not honour"):
            tr.run_variant(variant_dir, {"process_mode": "cathar", "suffix": "_x"}, {"cathar_alpha": 3.5}, manifest, excerpts_dir, "py")


def test_link_excerpts_falls_back_to_a_copy(tmp_path):
    excerpts_dir, target_dir = tmp_path / "e", tmp_path / "t"
    excerpts_dir.mkdir()
    (excerpts_dir / "a.mov").write_bytes(b"x")
    with patch.object(tr.os, "link", side_effect=OSError("cross-device")):
        paths = tr.link_excerpts(excerpts_dir, target_dir, {"excerpts": [{"excerpt": "a.mov"}]})
    assert paths[0].read_bytes() == b"x"
