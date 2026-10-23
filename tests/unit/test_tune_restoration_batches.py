"""The tuning driver launches the app in command-line-sized batches, and only on excerpts not restored yet."""

from unittest.mock import patch

from scripts import tune_restoration as tr

PREFIX = ["py", "app.py"]


def _cost(batch):
    return sum(len(part) + 3 for part in [*PREFIX, *batch])


def test_paths_are_split_into_batches_under_the_budget_in_order():
    """Ten 100-character paths under a 450-character budget: every path kept, in order, no batch over the budget."""
    paths = [f"{i}".rjust(100, "x") for i in range(10)]
    batches = tr.command_batches(PREFIX, paths, 450)
    assert [path for batch in batches for path in batch] == paths
    assert all(_cost(batch) <= 450 for batch in batches)
    assert [len(batch) for batch in batches] == [4, 4, 2]


def test_a_path_longer_than_the_budget_still_gets_its_own_batch_and_no_paths_give_none():
    """A path that cannot fit runs alone rather than being dropped; nothing to run is no batch at all."""
    assert tr.command_batches(PREFIX, ["y" * 600, "z"], 450) == [["y" * 600], ["z"]]
    assert tr.command_batches(PREFIX, [], 450) == []


def test_only_excerpts_without_a_restored_output_are_pending(tmp_path):
    """An excerpt whose output already decodes is not restored again."""
    done, todo = tmp_path / "a.mov", tmp_path / "b.mov"
    with patch.object(tr, "is_valid_video", side_effect=lambda path: "a_x" in str(path)):
        assert tr.pending_excerpts([done, todo], "_x") == [todo]


def _launch_three(tmp_path, monkeypatch):
    """Three excerpts under a budget that fits one path per call; returns the subprocess.run stand-in and the timing."""
    excerpts = [tmp_path / f"clip{i}.mov" for i in range(3)]
    budget = sum(len(part) + 3 for part in ["py", str(tr.REPO / "restore_audio_hybrid.py"), str(excerpts[0].resolve())]) + 2
    monkeypatch.setattr(tr, "COMMAND_LINE_BUDGET", budget)
    with patch.object(tr.subprocess, "run") as run:
        timing = tr._launch(tmp_path, excerpts, "py", keep_work=False)
    return run, timing, excerpts


def test_the_app_runs_once_per_batch_in_order(tmp_path, monkeypatch):
    """One app run per excerpt here, in the excerpts' order, into the variant's one log."""
    run, timing, excerpts = _launch_three(tmp_path, monkeypatch)
    assert [call.args[0][-1] for call in run.call_args_list] == [str(e.resolve()) for e in excerpts]
    assert (timing["n"], (tmp_path / "run.log").exists()) == (3, True)


def test_each_batch_is_timed_by_its_own_size(tmp_path, monkeypatch):
    """A one-excerpt batch gets one excerpt's timeout, not the whole variant's."""
    run, _timing, _excerpts = _launch_three(tmp_path, monkeypatch)
    assert {call.kwargs["timeout"] for call in run.call_args_list} == {tr.RUN_TIMEOUT_PER_EXCERPT_S}
