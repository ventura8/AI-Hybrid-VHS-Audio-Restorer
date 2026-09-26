"""Pure logic of the self-driving tuner: candidate ids, neighbours, combinations, the verdict."""

from scripts import autotune_restoration as at


def test_candidate_id_ignores_none_values_and_key_order():
    assert at.candidate_id({"b": 1, "a": None}) == at.candidate_id({"b": 1})
    assert at.candidate_id({"a": 1, "b": 2}) == at.candidate_id({"b": 2, "a": 1})
    assert at.candidate_id({"a": 1}) != at.candidate_id({"a": 1.5})


def test_split_overrides_routes_env_keys_and_drops_defaults():
    config, env = at.split_overrides({"cathar_alpha": 1.5, "env:AI_RESTORE_CATHAR_BIN": "x", "apl_neural_model": None})
    assert config == {"cathar_alpha": 1.5}
    assert env == {"AI_RESTORE_CATHAR_BIN": "x"}


def test_neighbour_moves_step_to_adjacent_values_only():
    incumbent = {"cathar_alpha": 2.0, "cathar_beta": 0.005}
    moves = dict(at.neighbour_moves("cathar", incumbent))
    assert moves["cathar_alpha"] in (1.5, 2.5)
    assert [v for k, v in at.neighbour_moves("cathar", incumbent) if k == "cathar_alpha"] == [1.5, 2.5]
    assert [v for k, v in at.neighbour_moves("cathar", incumbent) if k == "cathar_beta"] == [0.01]


def test_propose_never_returns_the_incumbent_itself():
    incumbent = {"apl_spectral_alpha_tonal": 2.0}
    proposals = at.propose("apl", incumbent)
    assert at.candidate_id(incumbent) not in proposals
    assert all(p != incumbent for p in proposals.values())


def test_combine_merges_two_winning_moves_and_ignores_a_single_one():
    incumbent = {"cathar_alpha": 2.0, "cathar_beta": 0.01}
    winners = [{"cathar_alpha": 1.5, "cathar_beta": 0.01}, {"cathar_alpha": 2.0, "cathar_beta": 0.005}]
    combo = at.combine(incumbent, winners)
    assert list(combo.values()) == [{"cathar_alpha": 1.5, "cathar_beta": 0.005}]
    assert at.combine(incumbent, winners[:1]) == {}


def _scores():
    base = {"mos.sigmos_col.delta.median": 0.1, "speech.cer.output.median": 0.08, "gates.hard_failures": 1.0}
    better = {"mos.sigmos_col.delta.median": 0.3, "speech.cer.output.median": 0.05, "gates.hard_failures": 1.0}
    worse = {"mos.sigmos_col.delta.median": 0.0, "speech.cer.output.median": 0.09, "gates.hard_failures": 0.0}
    failing = {"mos.sigmos_col.delta.median": 0.5, "speech.cer.output.median": 0.01, "gates.hard_failures": 3.0}
    return {
        "inc": {"t1": base, "t2": base},
        "good": {"t1": better, "t2": better},
        "bad": {"t1": worse, "t2": worse},
        "fail": {"t1": failing, "t2": failing},
    }


RANKING = {"mos.sigmos_col.delta.median": "up", "speech.cer.output.median": "down"}


def test_judge_accepts_the_candidate_that_wins_every_tape():
    verdicts = at.judge(_scores(), "inc", RANKING)
    assert verdicts["good"]["qualifies"]
    assert verdicts["good"]["wins"] == 2
    assert not verdicts["inc"]["qualifies"]


def test_judge_rejects_a_worse_candidate_and_one_with_new_hard_failures():
    verdicts = at.judge(_scores(), "inc", RANKING)
    assert not verdicts["bad"]["qualifies"]
    assert not verdicts["fail"]["qualifies"]
    assert verdicts["fail"]["failures"] == 6.0


def test_matching_value_reads_booleans_and_numbers_the_way_the_app_reports_them():
    assert at._matching_value("True", [True, False]) is True
    assert at._matching_value(2.0, [1.5, 2.0, 2.5]) == 2.0
    assert at._matching_value(7.0, [1.5, 2.0]) is None


def test_describe_shows_only_what_changed_from_the_incumbent():
    assert at.describe({"a": 1, "b": 2}, {"a": 1, "b": 3}) == '{"b": 2}'
    assert at.describe({}, {}) == "defaults"


def test_score_round_launches_at_most_parallel_scorers_at_once(tmp_path):
    from unittest.mock import patch

    launched, alive = [], []

    def fake_launch(slug, tape, cids, out_dir, families, gates, language):
        launched.append(slug)
        alive.append(slug)
        return None, None

    def fake_harvest(slug, cids, log, proc, out_dir):
        alive.remove(slug)
        for cid in cids:
            at._score_path(out_dir, cid, slug).parent.mkdir(parents=True, exist_ok=True)
            at._score_path(out_dir, cid, slug).write_text('{"x": 1.0}', encoding="utf-8")

    peak = []
    original = fake_launch

    def counting_launch(*args):
        result = original(*args)
        peak.append(len(alive))
        return result

    tapes = {"a": "a.mov", "b": "b.mov", "c": "c.mov"}
    with patch.object(at, "_launch_scorer", counting_launch), patch.object(at, "_harvest", fake_harvest):
        scores = at.score_round({"c1": {}, "c2": {}}, tapes, tmp_path, ("dsp",), None, "ro", parallel=2)
    assert launched == ["a", "b", "c"]
    assert max(peak) <= 2
    assert scores["c1"]["c"] == {"x": 1.0}


def test_drop_cache_entries_removes_the_scored_outputs_entries_and_keeps_the_sources(tmp_path):
    from scripts.restoration_quality import audio_io

    wav = tmp_path / "cands" / "abc" / "soti.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"RIFF")
    cache = tmp_path / "cache"
    cache.mkdir()
    output_key, source_key = audio_io.file_key(wav), "0123456789abcdef"
    for name in (f"{output_key}_219_16000.npy", f"{output_key}_dcfree.wav", f"{source_key}_219_16000.npy", f"{source_key}_soti.wav"):
        (cache / name).write_bytes(b"x")
    at._drop_cache_entries(tmp_path, "abc", "soti")
    at._drop_cache_entries(tmp_path, "missing", "soti")
    assert sorted(p.name for p in cache.iterdir()) == [f"{source_key}_219_16000.npy", f"{source_key}_soti.wav"]
