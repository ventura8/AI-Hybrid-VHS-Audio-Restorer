"""The verdict ledger: records round-trip, broken records name their problem, a label means one file per tape."""

import hashlib
import json

import pytest

from scripts.restoration_quality import ledger

# The sets experiments/tata_listen/check_verdicts.py hard-coded before it read the ledger.
OLD_EXPECTED = {
    "listener.pause_collapse": {"apl__roformer", "apl__roformer_no_expander_no_air"},
    "listener.dead_air": {"apl__baseline", "apl__no_air", "apl__no_expander", "apl__no_expander_no_air"},
    "listener.hiss": {"cathar__alpha_2_0", "cathar__alpha_2_0_beta_0_005"},
    "listener.sibilance_thin": {"apl__baseline", "apl__no_air", "apl__roformer"},
}
OLD_CLEAN = {"cathar075__alpha_2_0", "cathar__baseline"}


def _base(rid, kind, answer, labels=("a", "b")):
    """A valid record of question type `kind` over `labels`, by-ear unless the caller sets trials."""
    return {
        "id": rid,
        "date": "2026-10-09",
        "round": "1",
        "tape": "tele7abc",
        "windows": [[1.0, 11.0]],
        "files": [ledger.file_entry(label, f"{label}.wav") for label in labels],
        "question": {"type": kind, "stimuli": list(labels), "blind": True, "text": "which?"},
        "answer": answer,
        "n_trials": None,
        "n_correct": None,
        "confidence": None,
        "playback": {"device": "headphones", "volume": None},
        "context": {},
    }


def _flag(rid="flag-1"):
    return _base(rid, "flag", {"flagged": {"listener.hiss": ["a"]}, "clean": {ledger.ANY_FLAG: ["b"]}})


def _preference(rid="pref-1"):
    return _base(rid, "preference", {"order": [["a"], ["b"]]})


def _abx(rid="abx-1"):
    record = _base(rid, "abx", {"heard": True, "p_value": ledger.binomial_p(17, 24)})
    record.update(n_trials=24, n_correct=17, confidence=0.97)
    return record


def _pair(rid="pair-1"):
    record = _base(rid, "pair", {"counts": {"a": 2, "b": 1, ledger.SAME: 1}})
    record.update(n_trials=4)
    return record


def _blend(rid="blend-1"):
    record = _base(rid, "blend", {"threshold_x": 0.3, "band": [0.2, 0.5], "responses": [[1.0, True], [0.3, False]]})
    record.update(n_trials=2, n_correct=1)
    return record


BUILDERS = {"flag": _flag, "preference": _preference, "abx": _abx, "pair": _pair, "blend": _blend}


def _set(record, path, value):
    """`record` with the value at the key path replaced."""
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return record


BROKEN = [
    ("flag", ("id",), "has space", "id must"),
    ("flag", ("date",), "2026-13-01", "date must"),
    ("flag", ("round",), " ", "round must"),
    ("flag", ("tape",), "a/b", "tape must"),
    ("flag", ("windows",), [[5.0, 1.0]], "windows must"),
    ("flag", ("files", 0, "sha256"), "xyz", "files must"),
    ("flag", ("question", "stimuli"), ["a", "c"], "entry in files"),
    ("flag", ("question", "type"), "rank", "question.type"),
    ("flag", ("question", "blind"), "yes", "question.blind"),
    ("flag", ("confidence",), 2.0, "confidence must"),
    ("flag", ("playback",), {"device": 3}, "playback must"),
    ("flag", ("context",), [], "context must"),
    ("flag", ("context",), {"heard": "apl"}, "context.heard"),
    ("flag", ("context",), {"heard": ["c", "c"]}, "context.heard"),
    ("flag", ("context",), {"verdict_group": "two words"}, "context.verdict_group"),
    ("flag", ("answer", "flagged"), {"hiss": ["a"]}, "flagged must map"),
    ("flag", ("answer", "clean"), {"listener.hiss": ["c"]}, "clean must map"),
    ("flag", ("answer", "clean"), {"listener.hiss": ["a"]}, "cannot be flagged"),
    ("flag", ("n_trials",), 3, "holds no trials"),
    ("abx", ("answer", "p_value"), 0.5, "binomial"),
    ("abx", ("n_correct",), 25, "n_trials >= 1"),
    ("abx", ("answer", "heard"), "yes", "answer.heard"),
    ("pair", ("answer", "counts", "a"), 5, "add up"),
    ("pair", ("answer", "counts"), {"a": 1, "b": 3}, "count both"),
    ("pair", ("n_correct",), 2, "no right answer"),
    ("blend", ("answer", "band"), [0.5, 0.2], "band must"),
    ("blend", ("answer", "threshold_x"), -1.0, "threshold_x"),
    ("blend", ("answer", "responses"), [[1.0, True]], "responses must"),
    ("preference", ("answer", "order"), [["a", "b"]], "two or more"),
    ("preference", ("answer", "order"), [["a"], ["a"]], "two or more"),
    ("preference", ("n_correct",), 1, "holds no trials"),
]


def test_binomial_p_matches_the_abx_table():
    """17/24 is p 0.032 and 12/16 p 0.038 (the design's power table); nothing right is p 1."""
    assert ledger.binomial_p(17, 24) == pytest.approx(0.0320, abs=5e-4)
    assert ledger.binomial_p(12, 16) == pytest.approx(0.0384, abs=5e-4)
    assert ledger.binomial_p(0, 5) == 1.0


def test_every_question_type_round_trips(tmp_path):
    """What is appended reads back unchanged, in order."""
    path = tmp_path / "verdicts.jsonl"
    records = [build() for build in BUILDERS.values()]
    for record in records:
        ledger.append(record, path)
    assert ledger.read(path) == records


def test_appended_lines_are_lf_terminated_json(tmp_path):
    """One JSON object per LF-terminated line, never CRLF, whatever the platform."""
    path = tmp_path / "verdicts.jsonl"
    ledger.append(_flag("x-1"), path)
    ledger.append_many([_abx("x-2"), _pair("x-3")], path)
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert [json.loads(line)["id"] for line in raw.decode("utf-8").splitlines()] == ["x-1", "x-2", "x-3"]


def test_appending_after_a_last_line_without_a_line_end_keeps_both_records(tmp_path):
    """A hand-added record with no final LF reads whole; the next append ends its line first instead of joining it."""
    path = tmp_path / "verdicts.jsonl"
    path.write_text(json.dumps(_flag("hand-1")), encoding="utf-8")
    assert [record["id"] for record in ledger.read(path)] == ["hand-1"]
    ledger.append_many([_abx("x-2"), _pair("x-3")], path)
    assert [record["id"] for record in ledger.read(path)] == ["hand-1", "x-2", "x-3"]
    assert path.read_bytes().count(b"\n") == 3
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    ledger.append(_flag("x-1"), empty)
    assert empty.read_bytes() == (json.dumps(_flag("x-1")) + "\n").encode("utf-8")


def test_append_refuses_a_duplicate_id_and_an_invalid_record(tmp_path):
    """The ledger is append-only by id, and nothing invalid gets in."""
    path = tmp_path / "verdicts.jsonl"
    ledger.append(_flag(), path)
    duplicate, invalid = _flag(), _set(_flag("flag-2"), ("date",), "today")
    with pytest.raises(ledger.LedgerError, match="already in the ledger"):
        ledger.append(duplicate, path)
    with pytest.raises(ledger.LedgerError, match="new record 1: date must"):
        ledger.append(invalid, path)


def test_append_many_writes_everything_or_nothing(tmp_path):
    """One bad record, or two new records sharing an id, keeps the whole batch out of the ledger."""
    path = tmp_path / "verdicts.jsonl"
    one_bad = [_flag("ok-1"), _set(_flag("bad-2"), ("date",), "today")]
    twins = [_flag("twin"), _preference("twin")]
    with pytest.raises(ledger.LedgerError, match="new record 2: date must"):
        ledger.append_many(one_bad, path)
    with pytest.raises(ledger.LedgerError, match="new records: duplicate id 'twin'"):
        ledger.append_many(twins, path)
    assert not path.exists()


def test_read_names_the_line_of_a_broken_record(tmp_path):
    """A line that is not JSON, or not a valid record, is reported with its line number."""
    path = tmp_path / "verdicts.jsonl"
    path.write_text(json.dumps(_flag()) + "\n\nnot json\n", encoding="utf-8")
    with pytest.raises(ledger.LedgerError, match=r":3: not JSON"):
        ledger.read(path)
    path.write_text(json.dumps(_set(_flag(), ("tape",), "")) + "\n", encoding="utf-8")
    with pytest.raises(ledger.LedgerError, match=r":1: tape must"):
        ledger.read(path)


def test_read_refuses_duplicate_ids_and_reads_a_missing_ledger_as_empty(tmp_path):
    """Two lines with one id are an error; a ledger not written yet holds nothing."""
    path = tmp_path / "verdicts.jsonl"
    assert not ledger.read(path)
    path.write_text((json.dumps(_flag()) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ledger.LedgerError, match="duplicate id"):
        ledger.read(path)


@pytest.mark.parametrize("kind, path, value, fragment", BROKEN)
def test_validation_names_the_broken_field(kind, path, value, fragment):
    """Each schema rule produces its own message."""
    problems = ledger.validate_record(_set(BUILDERS[kind](), path, value))
    assert any(fragment in problem for problem in problems), problems


def test_validation_of_a_non_object_and_of_missing_fields():
    """A non-object and a record missing fields stop at that, before the field rules."""
    record = _flag()
    del record["answer"]
    assert ledger.validate_record([]) == ["a record must be a JSON object"]
    assert ledger.validate_record(record) == ["missing field 'answer'"]


def test_a_pair_stimulus_cannot_be_named_same():
    """'same' is the pair answer's third option, so it cannot also be a stimulus."""
    record = _base("p", "pair", {"counts": {"a": 1, ledger.SAME: 1}}, labels=("a", ledger.SAME))
    record.update(n_trials=2)
    assert any("cannot be named" in problem for problem in ledger.validate_record(record))


def test_sha256_of_hashes_small_files_only(tmp_path):
    """A small file gets its digest; a missing file or one over the cap stays unhashed."""
    path = tmp_path / "cut.wav"
    path.write_bytes(b"abc")
    assert ledger.sha256_of(path) == hashlib.sha256(b"abc").hexdigest()
    assert ledger.sha256_of(path, max_bytes=2) is None
    assert ledger.sha256_of(tmp_path / "missing.wav") is None


def _relabelled(rid, path, sha256=None, tape="tele7abc"):
    """A preference record whose stimulus 'a' is the file at `path`."""
    record = _preference(rid)
    record["files"][0] = ledger.file_entry("a", path, sha256)
    record["tape"] = tape
    return record


def test_a_label_names_one_file_per_tape(tmp_path):
    """Reusing stimulus 'a' for another file on the same tape is refused by append and by read."""
    path = tmp_path / "verdicts.jsonl"
    ledger.append(_flag(), path)
    relabelled = _relabelled("pref-2", "other.wav")
    with pytest.raises(ledger.LedgerError, match="give it a label of its own"):
        ledger.append(relabelled, path)
    path.write_text(json.dumps(_flag()) + "\n" + json.dumps(_relabelled("pref-2", "other.wav")) + "\n", encoding="utf-8")
    with pytest.raises(ledger.LedgerError, match="record 'pref-2'"):
        ledger.read(path)


def test_label_reuse_for_the_same_file_or_on_another_tape_is_fine(tmp_path):
    """The same path, the same digest at another path, another tape and a differing reference entry all pass."""
    path = tmp_path / "verdicts.jsonl"
    first = _relabelled("one", "a.wav", "1" * 64)
    first["files"].append(ledger.file_entry("source", "src1.wav"))
    second = _relabelled("two", "copy/a.wav", "1" * 64)
    second["files"].append(ledger.file_entry("source", "src2.wav"))
    ledger.append_many([first, second, _relabelled("three", "elsewhere.wav", tape="soti"), _relabelled("four", "a.wav")], path)
    assert [record["id"] for record in ledger.read(path)] == ["one", "two", "three", "four"]


def test_label_conflicts_trusts_the_digests_over_the_path():
    """One path holding other audio (a rebuilt file) conflicts; a relative and an absolute spelling of one file do not."""
    rebuilt = ledger.label_conflicts([_relabelled("one", "a.wav", "1" * 64)], _relabelled("two", "a.wav", "2" * 64))
    respelled = ledger.label_conflicts([_relabelled("one", "a.wav")], _relabelled("two", str(ledger.REPO_ROOT / "a.wav")))
    assert len(rebuilt) == 1
    assert "'a' on tape 'tele7abc'" in rebuilt[0]
    assert not respelled


def test_flag_verdicts_and_heard_labels_merge_records():
    """Flag records merge per flag; heard labels add the session's unjudged variants."""
    first, second = _flag("f-1"), _flag("f-2")
    second["answer"] = {"flagged": {"listener.hiss": ["b"]}, "clean": {"listener.dead_air": ["a"]}}
    second["context"] = {"heard": ["c"]}
    flagged, clean = ledger.flag_verdicts([first, second, _preference()])
    assert flagged == {"listener.hiss": {"a", "b"}}
    assert clean == {ledger.ANY_FLAG: {"b"}, "listener.dead_air": {"a"}}
    assert ledger.heard_labels([first, second]) == {"a", "b", "c"}


def test_preference_pairs_cross_every_tier():
    """Every label is preferred to every label of every later tier, and to none of its own."""
    record = _base("p", "preference", {"order": [["a"], ["b", "c"], ["d"]]}, labels=("a", "b", "c", "d"))
    pairs = ledger.preference_pairs(record)
    assert pairs == [("a", "b"), ("a", "c"), ("a", "d"), ("b", "d"), ("c", "d")]
    assert ledger.for_tape([record], "soti") == []


def _shipped():
    return {record["id"]: record for record in ledger.read()}


def test_shipped_ledger_holds_rounds_one_to_three():
    """The backfill: round one, the v2 plateaus (round two), the air shelf (round three)."""
    records = _shipped()
    assert {"1", "2", "3"} <= {record["round"] for record in records.values()}
    assert records["r3-vaccin-air-preference"]["answer"]["order"][0] == ["B_air_1dB"]
    assert records["r2-soti-apl-over-cathar"]["answer"]["order"] == [["final2_apl"], ["final2_cathar"]]


def test_shipped_ledger_encodes_the_sets_check_verdicts_hard_coded():
    """Round one's Tele7abc flags and its accepted outputs are exactly what the check used to hard-code."""
    records = [record for record in ledger.for_tape(ledger.read(), "tele7abc") if record["id"].startswith("r1-tele7abc-listen")]
    flagged, clean = ledger.flag_verdicts(records)
    assert flagged == OLD_EXPECTED
    assert clean[ledger.ANY_FLAG] == OLD_CLEAN


def test_shipped_ledger_holds_only_what_was_said():
    """No flag inferred from "b is better" in round three; no clean list the known set's words never gave."""
    records = _shipped()
    assert not [record for record in records.values() if record["round"] == "3" and record["question"]["type"] == "flag"]
    assert not any(records[f"r1-{tape}-known-flags"]["answer"]["clean"] for tape in ("tele7abc", "soti", "vaccin"))
    assert records["r1-tele7abc-known-flags"]["answer"]["flagged"] == {
        "listener.dull": ["deesser_bug", "single4s"],
        "listener.hiss": ["single4s"],
    }


def test_shipped_alpha2_verdict_includes_the_baseline_it_replaced():
    """SOTI and Vaccin "alpha 2 sounds better" moved the alpha default off cathar__baseline, so the record says so."""
    records = _shipped()
    expected = [["cathar__alpha_2_0"], ["apl__baseline", "cathar__baseline"]]
    assert [records[f"r1-{tape}-alpha2-better"]["answer"]["order"] for tape in ("soti", "vaccin")] == [expected, expected]
    assert "cathar075__alpha_2_0" in records["r1-soti-alpha2-better"]["context"]["heard"]


def test_shipped_ledger_hashes_every_file_but_the_full_tape_captures():
    """Every stimulus and reference up to 1 GiB (the WAV excerpts) carries its digest; only the .mov tapes stay null."""
    entries = [entry for record in ledger.read() for entry in record["files"]]
    assert all(entry["sha256"] for entry in entries if not entry["path"].endswith(".mov"))
    assert not [entry for entry in entries if entry["path"].endswith(".mov") and entry["sha256"]]
