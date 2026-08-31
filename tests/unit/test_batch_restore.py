"""Unit tests for multi-folder batch restoration runner."""

from unittest.mock import patch

import scripts.batch_restore as br


def test_parse_target_directories_default():
    """Test resolving default directories when no args provided."""
    dirs = br._parse_target_directories(["batch_restore.py"])
    assert len(dirs) == 1
    assert dirs[0] == br.Path(".").resolve()


def test_parse_target_directories_cli(tmp_path):
    """Test resolving custom target directories from CLI args."""
    target1 = tmp_path / "target1"
    target2 = tmp_path / "target2"
    dirs = br._parse_target_directories(["batch_restore.py", str(target1), str(target2)])
    assert len(dirs) == 2
    assert dirs[0] == target1.resolve()
    assert dirs[1] == target2.resolve()


def test_scan_folder_videos_inaccessible(tmp_path):
    """Test scanning a non-existent directory."""
    non_existent = tmp_path / "does_not_exist"
    assert br._scan_folder_videos(non_existent) == []


def test_scan_folder_videos_filters_cleaned(tmp_path):
    """Test folder scanning skips cleaned output files."""
    v1 = tmp_path / "video1.mp4"
    v1.write_text("raw")
    v1_clean = tmp_path / "video1_Pure_Cleaned.mp4"
    v1_clean.write_text("clean")
    v2 = tmp_path / "video2.avi"
    v2.write_text("raw")
    txt = tmp_path / "notes.txt"
    txt.write_text("notes")

    videos = br._scan_folder_videos(tmp_path)
    video_names = {v.name for v in videos}
    assert video_names == {"video1.mp4", "video2.avi"}


def test_collect_video_queue(tmp_path):
    """Test queue aggregation across multiple folders."""
    d1 = tmp_path / "d1"
    d1.mkdir()
    (d1 / "a.mp4").write_text("a")
    d2 = tmp_path / "d2"
    d2.mkdir()
    (d2 / "b.mkv").write_text("b")

    queue = br._collect_video_queue([d1, d2])
    assert len(queue) == 2


@patch("scripts.batch_restore.is_verified_video", return_value=True)
def test_is_already_restored_true(mock_valid, tmp_path):
    """Test detection of already processed video."""
    video = tmp_path / "clip.mp4"
    assert br._is_already_restored(video) is True


@patch("scripts.batch_restore._is_already_restored", return_value=True)
def test_process_video_item_skip(mock_restored, tmp_path):
    """Test skipping already restored video."""
    video = tmp_path / "clip.mp4"
    status = br._process_video_item(video, "NVIDIA", 1, 1)
    assert status == "SKIPPED"


@patch("scripts.batch_restore.process_hybrid_audio", return_value=True)
@patch("scripts.batch_restore._is_already_restored", return_value=False)
def test_process_video_item_success(mock_restored, mock_process, tmp_path):
    """Test successful restoration of a single video."""
    video = tmp_path / "clip.mp4"
    status = br._process_video_item(video, "NVIDIA", 1, 1)
    assert status == "SUCCESS"
    mock_process.assert_called_once()


@patch("scripts.batch_restore.process_hybrid_audio", return_value=False)
@patch("scripts.batch_restore._is_already_restored", return_value=False)
def test_process_video_item_failure(mock_restored, mock_process, tmp_path):
    """Test failed restoration of a single video."""
    video = tmp_path / "clip.mp4"
    status = br._process_video_item(video, "NVIDIA", 1, 1)
    assert status == "FAILED"


@patch("scripts.batch_restore.process_hybrid_audio", side_effect=RuntimeError("Process crash"))
@patch("scripts.batch_restore._is_already_restored", return_value=False)
def test_process_video_item_exception(mock_restored, mock_process, tmp_path):
    """Test exception handling during restoration of a single video."""
    video = tmp_path / "clip.mp4"
    status = br._process_video_item(video, "NVIDIA", 1, 1)
    assert status == "FAILED"


def test_print_batch_summary(capsys, tmp_path):
    """Test batch summary printer."""
    v1 = tmp_path / "v1.mp4"
    v2 = tmp_path / "v2.mp4"
    results = {v1: "SUCCESS", v2: "SKIPPED"}
    br._print_batch_summary(results, 42.5)
    captured = capsys.readouterr()
    assert "BATCH RESTORATION SUMMARY REPORT" in captured.out
    assert "Successfully Restored : 1" in captured.out
    assert "Skipped Videos        : 1" in captured.out


@patch("scripts.batch_restore.check_dependencies", return_value=False)
def test_run_batch_restoration_missing_deps(mock_check):
    """Test early return on missing dependencies."""
    assert br.run_batch_restoration() is False


@patch("scripts.batch_restore._collect_video_queue", return_value=[])
@patch("scripts.batch_restore._show_banner")
@patch("scripts.batch_restore.get_gpu_name", return_value="GPU")
@patch("scripts.batch_restore.check_dependencies", return_value=True)
def test_run_batch_restoration_empty_queue(mock_check, mock_gpu, mock_banner, mock_queue):
    """Test running batch with empty video queue."""
    assert br.run_batch_restoration() is True


@patch("scripts.batch_restore._process_video_item", return_value="SUCCESS")
@patch("scripts.batch_restore._collect_video_queue")
@patch("scripts.batch_restore._show_banner")
@patch("scripts.batch_restore.get_gpu_name", return_value="GPU")
@patch("scripts.batch_restore.check_dependencies", return_value=True)
def test_run_batch_restoration_full_flow(mock_check, mock_gpu, mock_banner, mock_queue, mock_proc, tmp_path):
    """Test full batch restoration execution flow."""
    v = tmp_path / "movie.mp4"
    mock_queue.return_value = [v]
    assert br.run_batch_restoration([tmp_path]) is True


@patch("scripts.batch_restore.run_batch_restoration", return_value=True)
@patch("sys.exit")
def test_main_success(mock_exit, mock_run):
    """Test main entry point success exit."""
    br.main()
    mock_exit.assert_called_once_with(0)
