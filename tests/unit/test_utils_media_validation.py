"""Unit tests for strict ffprobe-backed media container validation."""

from pathlib import Path
from unittest.mock import patch

import pytest

import modules.utils


def test_probe_stream_types_parses_codec_types():
    """_probe_stream_types should return the codec types reported by ffprobe."""
    with patch("modules.utils.subprocess.check_output", return_value=b"video\naudio\n\n") as mock_out:
        assert modules.utils._probe_stream_types("clip.mp4") == {"video", "audio"}
    assert mock_out.call_args.kwargs["timeout"] == 30


def test_probe_stream_types_returns_empty_on_failure():
    """Unreadable or corrupt containers should probe as no streams at all."""
    with patch("modules.utils.subprocess.check_output", side_effect=Exception("corrupt")):
        assert modules.utils._probe_stream_types("broken.mp4") == set()


def test_is_verified_video_rejects_undersized_file(tmp_path):
    """Strict verification should reject files that fail the basic size check."""
    small = tmp_path / "small.mp4"
    small.write_bytes(b"x" * 100)
    assert modules.utils.is_verified_video(small) is False


@pytest.mark.parametrize(
    ("stream_types", "expected"),
    [
        ({"video", "audio"}, True),
        ({"video"}, False),
        ({"audio"}, False),
        (set(), False),
    ],
)
def test_is_verified_video_requires_video_and_audio(tmp_path, stream_types, expected):
    """Only containers carrying both a video and an audio stream are verified."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 11000)
    with patch("modules.utils._probe_stream_types", return_value=stream_types):
        assert modules.utils.is_verified_video(clip) is expected


def test_models_dir_is_anchored_to_the_package(tmp_path, monkeypatch):
    """The model store must not follow the working directory."""
    import modules.utils

    expected = Path(modules.utils.__file__).resolve().parent.parent / "models"
    monkeypatch.chdir(tmp_path)
    assert modules.utils.MODELS_DIR == expected
    assert modules.utils.MODELS_DIR.is_absolute()
