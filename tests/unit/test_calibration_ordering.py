"""The known-ordering set: the v3 ranking, the by-ear rules, the ledger's verdicts per tape, and the per-route view."""

import json

import numpy as np
import pytest

from scripts import calibration_ordering as co

AIR = "dsp.balance_air_db"


def _summary(value):
    return {"median": value, "tail": value, "n": 1}


def _variant(failures=(), readings=None, verdicts=None):
    aggregate = {metric: {"source": _summary(0.0), "output": _summary(v), "delta": _summary(v)} for metric, v in (readings or {}).items()}
    flagged = [{"gate": g, "metric": "m", "stat": "median", "value": 1.0, "status": "failed", "severity": "hard"} for g in failures]
    return {"hard_failures": list(failures), "verdicts": flagged + list(verdicts or []), "aggregate": aggregate}


def test_the_v3_score_is_the_two_sided_distance_lower_is_better():
    """The v3 score is the two sided distance lower is better."""
    result = {
        "variants": {"far": _variant(readings={AIR: 2.0}), "near": _variant(readings={AIR: -0.5}), "flat": _variant(readings={AIR: 0.0})}
    }
    scores = co.harness_scores(result, {})
    assert scores == {"far": 2.0, "near": 0.5, "flat": 0.0}
    assert co.harness_ranking(result, {}) == ["flat", "near", "far"]


def test_hard_failures_rank_before_the_score():
    """Hard failures rank before the score."""
    result = {"variants": {"broken": _variant(["dsp.clicks"], {AIR: 0.0}), "bright": _variant(readings={AIR: 1.0})}}
    assert co.harness_ranking(result, {}) == ["bright", "broken"]


def test_a_reading_one_variant_lacks_counts_against_it():
    """A reading one variant lacks counts against it."""
    result = {"variants": {"unread": _variant(readings={"dsp.lkr": 0.0}), "read": _variant(readings={AIR: 1.0})}}
    scores = co.harness_scores(result, {})
    assert scores["unread"] > scores["read"]


def test_without_any_grid_reading_nothing_is_ranked():
    """The v2 failure: no ranked reading was scored, so the order was the listing order and the rules tested it."""
    result = {"variants": {"deesser_bug": _variant(["dsp.hf_4k8k"]), "apl": _variant(), "stitched": _variant()}}
    rules = co.ordering_rules(result, {})
    assert rules["ranked_on"] == 0
    assert rules["bug_in_bottom_two"] is None
    assert rules["good_in_top"] is None
    assert rules["bug_flagged_muffled"] is True


def test_the_default_grid_holds_two_sided_dsp_readings_with_a_floor_dead_zone():
    """The default grid holds two sided dsp readings with a floor dead zone."""
    grid = co.default_grid({"dsp.balance_air_db": {"centre": 0.0, "floor": 0.1}})
    assert grid["dsp.balance_air_db.delta.median"] == {"target": 0.0, "family": "timbre", "dead_zone": pytest.approx(0.3)}
    assert "dsp.gap_atten_db.delta.median" not in grid
    assert "dsp.balance_top_db.delta.median" not in grid
    assert "dsp.hf_4k8k.delta.median" not in grid


def test_the_default_grid_leaves_r11_out_until_a_verdict_ranks_it():
    """R11 is a two-sided dsp reading, but on speech it reads the low end's noise removed: the stand-in grid does not rank it."""
    assert co.METRICS["dsp.lf_programme_db"].better == co.TWO_SIDED
    assert not any(key.startswith("dsp.lf_programme_db") for key in co.default_grid({}))
    assert co.FAMILY_BY_PREFIX["dsp.lf_"] == "timbre"


def test_flat_readings_carry_the_tuning_grid_names():
    """Flat readings carry the tuning grid names."""
    flat = co.flat_readings(_variant(readings={AIR: 1.5}))
    assert flat["dsp.balance_air_db.delta.median"] == 1.5
    assert flat["dsp.balance_air_db.output.tail"] == 1.5
    assert "dsp.balance_air_db.source.median" not in flat


def _by_ear():
    result = {
        "variants": {
            "deesser_bug": _variant(["dsp.hf_4k8k"], {"dsp.hf_4k8k": -30.0, AIR: -3.0}),
            "single4s": _variant([], {"dsp.hf_4k8k": -7.0, AIR: -1.0}),
            "stitched": _variant([], {"dsp.hf_4k8k": -4.0, AIR: -0.2}),
            "apl": _variant([], {AIR: 0.1}),
            "cathar075": _variant([], {AIR: 0.0}),
        }
    }
    return co.ordering_rules(result, {})


def test_the_rules_flag_the_deesser_bug_and_sink_it():
    """The rules flag the deesser bug and sink it."""
    rules = _by_ear()
    assert rules["bug_flagged_muffled"]
    assert rules["bug_in_bottom_two"]
    assert rules["ranked_on"] == 5


def test_the_rules_place_the_good_variants_where_the_ear_did():
    """The rules place the good variants where the ear did."""
    rules = _by_ear()
    assert rules["single4s_duller_than_stitched"]
    assert rules["apl_not_altered"]
    assert rules["good_in_top"]


def _listened(statuses, gate="listener.hiss"):
    verdict = {"metric": "dsp.gap_air_db", "stat": "median", "value": 0.0, "severity": "flag"}
    return {
        label: {"hard_failures": [], "verdicts": [{**verdict, "gate": gate, "status": s}], "aggregate": {}} for label, s in statuses.items()
    }


def test_flags_reproduced_needs_every_flagged_label_failing_and_every_clean_one_passing():
    """Flags reproduced needs every flagged label failing and every clean one passing."""
    flags, clean = {"listener.hiss": ["hissy"]}, {"listener.hiss": ["quiet"]}
    assert co.flags_reproduced(_listened({"hissy": "failed", "quiet": "passed"}), flags, clean) is True
    assert co.flags_reproduced(_listened({"hissy": "failed", "quiet": "failed"}), flags, clean) is False
    assert co.flags_reproduced(_listened({"hissy": "passed", "quiet": "passed"}), flags, clean) is False


def test_an_output_accepted_with_no_complaint_must_pass_every_listener_flag():
    """An output accepted with no complaint must pass every listener flag."""
    assert co.flags_reproduced(_listened({"fine": "passed"}), {}, {"*": ["fine"]}) is True
    assert co.flags_reproduced(_listened({"fine": "failed"}, "listener.dead_air"), {}, {"*": ["fine"]}) is False


def test_flags_reproduced_is_unread_without_lists_or_with_labels_the_tape_lacks():
    """Flags reproduced is unread without lists or with labels the tape lacks."""
    assert co.flags_reproduced(_listened({"hissy": "passed"}), None, None) is None
    assert co.flags_reproduced(_listened({"hissy": "passed"}), {"listener.hiss": ["absent"]}, {}) is None


def _record(tape, round_id, flagged, clean, kind="flag"):
    """A ledger record holding only what the calibration reads; its stimuli are the labels its lists name."""
    stimuli = sorted({label for lists in (flagged, clean) for labels in lists.values() for label in labels})
    return {"tape": tape, "round": round_id, "question": {"type": kind, "stimuli": stimuli}, "answer": {"flagged": flagged, "clean": clean}}


def test_a_listening_tape_reads_the_ledger_tape_its_name_starts_with():
    """A listening tape reads the ledger tape its name starts with."""
    assert co.ledger_tape("tele7abc_listen", {}, {"tele7abc", "tele"}) == "tele7abc"
    assert co.ledger_tape("soti", {}, {"soti"}) == "soti"
    assert co.ledger_tape("x", {"ledger_tape": "soti"}, set()) == "soti"
    assert co.ledger_tape("vaccin", {}, {"soti"}) is None


def test_the_ledger_lists_replace_the_manifests_narrowed_to_the_scored_labels():
    """The ledger lists replace the manifests narrowed to the scored labels."""
    records = [
        _record("tele7abc", "1", {"listener.dull": ["deesser_bug", "other"]}, {"*": ["apl"]}),
        _record("tele7abc", "3", {}, {}, "preference"),
    ]
    entry = {"variants": {"deesser_bug": "a.wav", "apl": "b.wav"}, "flags": {"listener.hiss": ["apl"]}}
    verdicts = co.tape_verdicts("tele7abc", entry, records)
    assert verdicts == {
        "flags": {"listener.dull": ["deesser_bug"]},
        "clean": {"*": ["apl"]},
        "verdict_source": "ledger",
        "rounds": ["1"],
        "gate_rounds": {"listener.dull": ["1"], "*": ["1"]},
    }


def test_a_tape_the_ledger_does_not_flag_keeps_the_manifest_lists():
    """A tape the ledger does not flag keeps the manifest lists."""
    entry = {"variants": {"apl": "b.wav"}, "flags": {"listener.hiss": ["apl"]}}
    assert co.tape_verdicts("soti", entry, [_record("soti", "3", {}, {}, "preference")])["verdict_source"] == "manifest"
    assert co.tape_verdicts("soti", entry, None)["flags"] == {"listener.hiss": ["apl"]}


def test_without_a_ledger_an_entry_naming_its_ledger_tape_reads_the_manifest():
    """`--no-ledger` (records None) never reaches the ledger, even for an entry that names its ledger tape."""
    entry = {"variants": {"a": "a.wav"}, "ledger_tape": "tele7abc", "flags": {"listener.hiss": ["a"]}}
    verdicts = co.tape_verdicts("x", entry, None)
    assert (verdicts["verdict_source"], verdicts["flags"], verdicts["gate_rounds"]) == ("manifest", {"listener.hiss": ["a"]}, {})


def test_a_flag_takes_the_rounds_of_the_records_that_name_it_on_a_scored_label():
    """A round-2 record that names hiss on another file only says nothing about hiss on the scored set."""
    records = [
        _record("tele7abc", "1", {"listener.dull": ["deesser_bug"]}, {}),
        _record("tele7abc", "2", {"listener.hiss": ["final2"]}, {"listener.dull": ["deesser_bug"]}),
    ]
    assert co.gate_rounds(records, {"deesser_bug"}) == {"listener.dull": ["1", "2"]}


def test_flags_no_gate_reads_are_reported():
    """Flags no gate reads are reported."""
    ordering = {"t": {"flags": {"listener.dull": ["a"]}, "clean": {"*": ["b"], "listener.hiss": ["b"]}}}
    assert co.unmapped_flags(ordering, {"listener.hiss": None}) == ["listener.dull"]


def test_the_ledger_reader_returns_the_records_or_none(tmp_path, monkeypatch):
    """The ledger reader returns the records or none."""
    path = tmp_path / "verdicts.jsonl"
    path.write_text("", "utf-8")
    assert co.ledger_records(path) == []
    monkeypatch.setattr(co, "import_module", _missing)
    assert co.ledger_records(path) is None


def _missing(_name):
    raise ImportError("no ledger")


def test_a_stored_result_covers_the_families_it_was_asked_for():
    """A stored result covers the families it was asked for."""
    v2 = {"variants": {"a": {"families": {"dsp": "ok"}}}}
    assert co.covers(v2, ["dsp"])
    assert not co.covers(v2, ["dsp", "mos"])
    assert co.covers({"requested": ["dsp", "mos"], "variants": {}}, ["mos"])


def test_numpy_values_serialise():
    """Numpy values serialise."""
    assert json.loads(json.dumps({"a": np.float32(1.5), "b": np.arange(2), "c": np.bool_(True)}, default=co.json_default)) == {
        "a": 1.5,
        "b": [0, 1],
        "c": True,
    }


def test_a_flag_whose_gate_read_nothing_is_counted_unread_not_failed():
    """A ledger flag named before its gate existed, or a reading not scored, is neither reproduced nor broken."""
    variants = _listened({"dull": "skipped", "hissy": "failed"})
    flags = {"listener.dull": ["dull"], "listener.hiss": ["hissy"]}
    assert co.flags_reproduced(variants, flags, {}) is True
    assert co.flags_unread(variants, flags, {}) == 1


def test_only_the_ledger_records_that_judged_a_scored_label_count():
    """A round-2 flag on other files of the same tape says nothing about the round-1 set."""
    records = [
        _record("tele7abc", "1", {"listener.dull": ["deesser_bug"]}, {}),
        _record("tele7abc", "2", {"listener.hiss": ["final2"]}, {}),
    ]
    verdicts = co.tape_verdicts("tele7abc", {"variants": {"deesser_bug": "a.wav"}}, records)
    assert verdicts["flags"] == {"listener.dull": ["deesser_bug"]}
    assert verdicts["rounds"] == ["1"]
