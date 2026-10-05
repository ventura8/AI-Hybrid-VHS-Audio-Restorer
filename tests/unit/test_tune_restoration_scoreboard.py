"""The tuning driver's scoring side: aggregation across excerpts, gates, ranks, the scoreboard, family merges."""

import json
from pathlib import Path

from scripts import tune_restoration as tr


def _manifest():
    return {
        "excerpts": [
            {"excerpt": "soti_start.mov", "slug": "soti"},
            {"excerpt": "soti_end.mov", "slug": "soti"},
            {"excerpt": "vaccin_whole.mov", "slug": "vaccin"},
        ]
    }


def test_aggregate_variant_and_gates():
    pairs = {
        "soti_start": {"m.delta.median": 1.0, "hard_failures": []},
        "soti_end": {"m.delta.median": 3.0, "hard_failures": ["dsp.lkr"]},
        "vaccin_whole": {"m.delta.median": 2.0, "hard_failures": []},
    }
    summary = tr.aggregate_variant(pairs, _manifest())
    assert summary["metrics"]["m.delta.median"]["median"] == 2.0
    assert summary["per_tape"]["soti"]["m.delta.median"] == 2.0
    vetoed, reasons, per_tape = tr.apply_gates(summary, 1)
    assert (vetoed, reasons, per_tape) == (False, ["dsp.lkr"], {"soti": True, "vaccin": False})
    assert tr.apply_gates(summary, 0)[0] is True


def test_apply_gates_tolerates_what_the_baseline_fails_too():
    summary = tr.aggregate_variant({"soti_start": {"m": 1.0, "hard_failures": ["dsp.lkr"]}}, _manifest())
    assert tr.apply_gates(summary, 0)[0] is True
    assert tr.apply_gates(summary, 0, baseline_failed=1)[0] is False


def test_ranks_average_ties_and_put_missing_last():
    assert tr._ranks([3.0, 1.0, 3.0, None], "up") == [1.5, 3.0, 1.5, 4.0]
    assert tr._ranks([3.0, 1.0], "down") == [2.0, 1.0]


def _summary(median, vetoed=False, per_tape=None):
    return {"metrics": {"k": {"median": median, "p10": median}}, "vetoed": vetoed, "per_tape": per_tape or {}, "per_tape_vetoed": {}}


def test_rank_variants_ranks_every_variant_by_direction():
    summaries = {"a": _summary(1.0), "b": _summary(3.0), "c": _summary(9.0, vetoed=True)}
    ranks = tr.rank_variants(summaries, {"k": "up"})
    assert ranks == {"c": 1.0, "b": 2.0, "a": 3.0}
    assert tr.rank_variants(summaries, {"missing": "up"}) == {"a": None, "b": None, "c": None}


def test_best_engine_per_tape_ranks_across_engines():
    summaries = {
        "cathar__baseline": _summary(0.0, per_tape={"soti": {"k": 2.0}, "vaccin": {"k": 5.0}}),
        "apl__baseline": _summary(0.0, per_tape={"soti": {"k": 4.0}, "vaccin": {"k": 1.0}}),
    }
    best = tr.best_engine_per_tape(summaries, {"k": "up"})
    assert best["soti"]["engine"] == "apl"
    assert best["vaccin"]["engine"] == "cathar"


def _board():
    grid = {"engines": {"cathar": {}, "apl": {}}, "ranking": {"k": "up"}}
    manifest = {"config_sha256": "abc123def456", "excerpts": [{"excerpt": "x"}], "tapes_dir": "D:/tapes"}
    summaries = {
        "cathar__baseline": {**_summary(1.0, per_tape={"soti": {"k": 1.0}}), "veto_reasons": [], "overrides": {}},
        "cathar__alpha": {**_summary(2.0, per_tape={"soti": {"k": 2.0}}), "veto_reasons": [], "overrides": {"cathar_alpha": 3.5}},
        "apl__baseline": {**_summary(0.0, vetoed=True, per_tape={"soti": {"k": 0.0}}), "veto_reasons": ["dsp.lkr"], "overrides": {}},
    }
    return tr.build_scoreboard("t", grid, manifest, summaries)


def test_scoreboard_recommends_the_best_unvetoed_variant_per_engine():
    board = _board()
    assert board["recommendation"]["cathar"]["best"] == "cathar__alpha"
    assert board["recommendation"]["apl"]["best"] is None


def test_scoreboard_renders_warnings_and_survives_json():
    board = _board()
    text = tr.render_scoreboard_md(board)
    assert "## Warnings" in text
    assert "dsp.lkr" in text
    assert json.loads(json.dumps(board, default=str))["per_tape_engine"]["soti"]["variant"] == "cathar__alpha"


def test_flatten_pair_carries_hard_failures():
    result = {"variants": {"v": {"aggregate": {"m": {"delta": {"median": 1.0, "tail": 0.5}}}, "hard_failures": ["g"], "passed": False}}}
    flat = tr.flatten_pair(result, "v")
    assert flat["m.delta.median"] == 1.0
    assert flat["hard_failures"] == ["g"]
    assert flat["gates.passed"] == 0.0


def test_output_path_uses_the_engine_suffix():
    assert tr.output_path(Path("x/soti_start.mov"), "_Cathar_Cleaned") == Path("x/soti_start_Cathar_Cleaned.mov")


def _pair(clicks, disc):
    agg = {"dsp.clicks_per_s": {"delta": {"median": clicks, "tail": clicks}}, "mos.sigmos_disc": {"delta": {"median": disc, "tail": disc}}}
    row = {
        "window": 0,
        "source": {"dsp.clicks_per_s": 0.0, "mos.sigmos_disc": 3.0},
        "output": {"dsp.clicks_per_s": clicks},
        "delta": dict(agg),
    }
    row["delta"] = {"dsp.clicks_per_s": clicks, "mos.sigmos_disc": disc}
    variant = {"rows": [row], "aggregate": agg, "file": {"file.lufs": {"delta": clicks}}, "families": {"dsp": "ok"}, "speaker_floor": None}
    return {"variants": {"v": variant}, "metrics": {"dsp.clicks_per_s": {"family": "dsp"}}}


def _merged_variant():
    stored, fresh = _pair(clicks=2.0, disc=-0.9), _pair(clicks=0.0, disc=-0.1)
    stored["variants"]["v"]["families"]["mos"] = "ok"
    gates = {"clicks": tr.gates_mod.Gate("dsp.clicks_per_s", "delta.median", "<=", 0.5)}
    return tr.merge_families(stored, fresh, ("dsp",), gates)["variants"]["v"]


def test_merge_families_replaces_only_the_rescored_family():
    variant = _merged_variant()
    assert variant["rows"][0]["delta"] == {"dsp.clicks_per_s": 0.0, "mos.sigmos_disc": -0.9}
    assert variant["aggregate"]["mos.sigmos_disc"]["delta"]["median"] == -0.9
    assert variant["families"] == {"dsp": "ok", "mos": "ok"}


def test_merge_families_rereads_the_verdicts():
    variant = _merged_variant()
    assert variant["passed"] is True
    assert variant["hard_failures"] == []


def test_merge_families_takes_the_fresh_pair_when_the_windows_differ():
    stored, fresh = _pair(clicks=2.0, disc=-0.9), _pair(clicks=0.0, disc=-0.1)
    fresh["variants"]["v"]["rows"].append(dict(fresh["variants"]["v"]["rows"][0]))
    merged = tr.merge_families(stored, fresh, ("dsp",), {})
    assert merged["variants"]["v"] is fresh["variants"]["v"]
