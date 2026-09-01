"""Unit tests for Internet Archive corpus curation and benchmark tooling."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import scripts.benchmark_ia_corpus as bic
import scripts.curate_ia_corpus as cic


def test_sanitize_slug():
    """Slug generation removes illegal characters and repeated underscores."""
    raw = "Jackie Chan: My Story (1999) / UK VHS"
    slug = cic._sanitize_slug(raw)
    assert ":" not in slug
    assert "/" not in slug
    assert "Jackie_Chan_My_Story_1999_UK_VHS" in slug


def test_build_copy_cmd():
    """Stream copy command constructs correct arguments."""
    target = Path("test.mp4")
    cmd = cic._build_copy_cmd("http://example.com/stream.mp4", target, 30, 20)
    for expected in ("-ss", "30", "-t", "20", "-c", "copy"):
        assert expected in cmd


def test_build_transcode_cmd():
    """Transcode fallback command constructs correct arguments."""
    target = Path("test.mp4")
    cmd = cic._build_transcode_cmd("http://example.com/stream.mp4", target, 15, 10)
    for expected in ("-ss", "15", "-t", "10", "libx264", "pcm_f32le"):
        assert expected in cmd


def test_execute_ffmpeg_success():
    """Successful subprocess returns True."""
    with patch("scripts.curate_ia_corpus.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        assert cic._execute_ffmpeg(["ffmpeg", "-version"]) is True


def test_execute_ffmpeg_failure():
    """Failed subprocess returns False."""
    with patch("scripts.curate_ia_corpus.subprocess.run", side_effect=OSError("FFmpeg error")):
        assert cic._execute_ffmpeg(["ffmpeg", "-version"]) is False


def test_download_clip_copy_success(tmp_path):
    """Downloads clip via copy without needing fallback."""
    target = tmp_path / "out.mp4"

    def fake_exec(cmd, **kwargs):
        Path(cmd[-1]).touch()
        return True

    with (
        patch("scripts.curate_ia_corpus._execute_ffmpeg", side_effect=fake_exec),
        patch("scripts.curate_ia_corpus._get_clip_duration", return_value=20.0),
        patch("scripts.curate_ia_corpus.is_valid_video", return_value=True),
    ):
        assert cic._download_clip("http://url", target, 0, 10) is True


def test_download_clip_fallback_transcode(tmp_path):
    """Downloads clip via transcode when stream copy fails."""
    target = tmp_path / "out.mp4"
    calls = 0

    def fake_exec(cmd, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return False
        Path(cmd[-1]).touch()
        return True

    with (
        patch("scripts.curate_ia_corpus._execute_ffmpeg", side_effect=fake_exec),
        patch("scripts.curate_ia_corpus._get_clip_duration", return_value=20.0),
        patch("scripts.curate_ia_corpus.is_valid_video", return_value=True),
    ):
        assert cic._download_clip("http://url", target, 0, 10) is True


def test_process_item_skips_existing(tmp_path):
    """Existing valid clip skips download."""
    item = {"identifier": "tape1", "genre": "home", "stream_url": "http://url"}
    with (
        patch("scripts.curate_ia_corpus.is_valid_video", return_value=True),
        patch("scripts.curate_ia_corpus._get_clip_duration", return_value=20.0),
        patch("scripts.curate_ia_corpus._download_clip") as mock_dl,
    ):
        assert cic._process_item(item, tmp_path, 0, 20, force=False) is True
        mock_dl.assert_not_called()


def test_curate_region_processes_items(tmp_path):
    """Curate region processes items up to limit."""
    items = [
        {"identifier": "t1", "genre": "home", "stream_url": "http://u1"},
        {"identifier": "t2", "genre": "tv", "stream_url": "http://u2"},
    ]
    with patch("scripts.curate_ia_corpus._process_item", return_value=True):
        count = cic._curate_region(items, "europe", tmp_path, 0, 20, limit=1, force=False)
        assert count == 1


def test_curate_corpus(tmp_path):
    """Curate corpus coordinates curation across regions."""
    cat = tmp_path / "catalog.json"
    cat.write_text(json.dumps({"europe": [{"identifier": "e1", "stream_url": "http://u"}]}))
    with patch("scripts.curate_ia_corpus._curate_region", return_value=1):
        stats = cic.curate_corpus(cat, tmp_path)
        assert stats.get("europe") == 1


def test_benchmark_noise_floor_db():
    """Estimates noise floor via windowed RMS."""
    sr = 44100
    data = np.ones(sr * 2, dtype=np.float32) * 0.05
    nf = bic._compute_noise_floor_db(data, sr, fallback_rms_db=-20.0)
    assert -30.0 < nf < -20.0


def test_benchmark_noise_floor_short_audio():
    """Short audio falls back to overall RMS."""
    sr = 44100
    data = np.ones(sr // 10, dtype=np.float32) * 0.05
    assert bic._compute_noise_floor_db(data, sr, fallback_rms_db=-25.0) == -25.0


def test_benchmark_spectral_ratio():
    """Computes spectral peak ratio vs background."""
    f = np.linspace(0, 20000, 4000)
    psd = np.ones_like(f) * 1.0
    psd[np.argmin(np.abs(f - 15625))] = 10.0
    ratio = bic._compute_spectral_ratio(psd, f, 15625.0)
    assert ratio >= 9.0


def test_benchmark_rumble_pct():
    """Computes sub-100Hz rumble percentage."""
    f = np.linspace(0, 1000, 1000)
    psd = np.zeros_like(f)
    psd[:100] = 1.0
    rumble = bic._compute_rumble_pct(psd, f, cutoff_hz=100.0)
    assert 90.0 <= rumble <= 100.0


def test_benchmark_split_channels():
    """Splits stereo and mono audio into channels."""
    stereo = np.zeros((100, 2), dtype=np.float32)
    stereo[:, 0] = 1.0
    stereo[:, 1] = 0.5
    mono, left, right = bic._split_channels(stereo)
    assert np.mean(mono) == 0.75
    assert np.mean(left) == 1.0
    assert np.mean(right) == 0.5


def test_benchmark_calculate_deltas():
    """Calculates deltas between original and restored metrics."""
    orig = {
        "noise_floor_db": -40.0,
        "snr_db": 20.0,
        "crt_whistle_ratio": 8.0,
        "mains_hum_ratio": 6.0,
        "rumble_energy_pct": 25.0,
        "stereo_balance_diff_db": 4.0,
    }
    rest = {
        "noise_floor_db": -55.0,
        "snr_db": 35.0,
        "crt_whistle_ratio": 1.0,
        "mains_hum_ratio": 1.0,
        "rumble_energy_pct": 5.0,
        "stereo_balance_diff_db": 0.5,
    }
    deltas = bic._calculate_deltas(orig, rest)
    expected_pairs = (
        ("noise_reduction_db", 15.0),
        ("snr_gain_db", 15.0),
        ("crt_attenuation_ratio", 8.0),
        ("rumble_reduction_pct", 20.0),
        ("balance_improvement_db", 3.5),
    )
    for key, val in expected_pairs:
        assert deltas[key] == val


def test_benchmark_find_clip_meta(tmp_path):
    """Finds clip meta by matching identifier or default."""
    index = {"": {"region": "europe"}, "jackie_chan": {"region": "europe", "genre": "home", "crt_hz": 15625.0}}
    p1 = tmp_path / "jackie_chan_home_20s.mp4"
    meta1 = bic._find_clip_meta(p1, index)
    assert meta1["genre"] == "home"

    p2 = tmp_path / "unknown_american_clip.mp4"
    meta2 = bic._find_clip_meta(p2, index)
    assert meta2["region"] == "america"
    assert meta2["crt_hz"] == 15734.0


def test_benchmark_load_meta_index_skips_empty(tmp_path):
    """Verifies that empty identifiers are excluded from loaded catalog index."""
    catalog_file = tmp_path / "cat.json"
    catalog_file.write_text(json.dumps({"Europe": [{"identifier": "  "}, {"identifier": "tape1", "title": "Tape 1"}]}))
    loaded = bic._load_meta_index(catalog_file)
    assert "" not in loaded
    assert "tape1" in loaded


def test_benchmark_format_markdown_report():
    """Formats markdown report with headers and tables."""
    summary = {
        "overall": {
            "cathar": {
                "noise_reduction_db": 18.5,
                "snr_gain_db": 17.2,
                "crt_attenuation_ratio": 9.4,
                "mains_attenuation_ratio": 7.1,
                "rumble_reduction_pct": 22.0,
                "balance_improvement_db": 3.8,
            }
        },
        "by_region": {
            "cathar": {
                "europe": {"noise_reduction_db": 19.0, "snr_gain_db": 18.0},
                "america": {"noise_reduction_db": 18.0, "snr_gain_db": 16.5},
            }
        },
        "by_genre": {
            "cathar": {
                "home": {"noise_reduction_db": 18.0},
                "tv": {"noise_reduction_db": 19.0},
                "music": {"noise_reduction_db": 18.5},
            }
        },
    }
    report = bic._format_markdown_report(summary, ["cathar"], total_clips=40)
    assert "# Internet Archive VHS Audio Restoration Benchmark Report" in report
    assert "| `cathar` | +18.50 dB" in report
    assert "Europe (PAL 50Hz)" in report
    assert "America (NTSC 60Hz)" in report


def test_benchmark_eval_single_clip_cleans_wavs(tmp_path):
    """Verifies that original and restored WAV files are cleaned up after evaluation."""
    clip = tmp_path / "test_clip.mp4"
    clip.touch()
    meta = {"identifier": "test_clip", "crt_hz": 15625.0, "notch_hz": 50.0}
    out_dir = tmp_path / "out"
    eval_dir = tmp_path / "eval"
    out_dir.mkdir()
    eval_dir.mkdir()

    rest_vid = out_dir / "test_clip_Cathar_Cleaned.mp4"

    def mock_extract(src, dst):
        dst.touch()
        return True

    sample_metrics = {
        "noise_floor_db": -40.0,
        "snr_db": 20.0,
        "crt_whistle_ratio": 5.0,
        "mains_hum_ratio": 3.0,
        "rumble_energy_pct": 10.0,
        "stereo_balance_diff_db": 1.0,
    }

    with (
        patch("scripts.benchmark_ia_corpus._extract_pcm", side_effect=mock_extract),
        patch("scripts.benchmark_ia_corpus._run_mode_restoration", return_value=rest_vid),
        patch("scripts.benchmark_ia_corpus.analyze_audio", return_value=sample_metrics),
    ):
        res = bic._eval_single_clip(clip, meta, ["cathar"], out_dir, eval_dir, "cpu")
        assert res is not None
        assert "cathar" in res["restored"]

    assert not list(eval_dir.glob("*.wav"))
