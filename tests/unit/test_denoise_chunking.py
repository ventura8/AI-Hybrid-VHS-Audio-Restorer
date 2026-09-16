"""Duration-based chunking of the neural denoiser: the plan, the streaming split and join, and the seams."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

import modules.processing as processing
from modules import denoise_chunking as chunking

RATE = 100  # Hz: small files, exact frame arithmetic
OVERLAP = int(chunking.CHUNK_OVERLAP_SECONDS * RATE)
STRIDE = 50  # the models' patch stride is a 44.1 kHz sample count; scaled down for these files


FADE = 4  # the junction fade, scaled down with the stride


@pytest.fixture(autouse=True)
def _small_patch_stride():
    with patch("modules.denoise_chunking.PATCH_FRAMES", STRIDE), patch("modules.denoise_chunking.JUNCTION_FADE_FRAMES", FADE):
        yield


def _wav(path, seconds, peak=0.5, channels=2, seed=0):
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal((int(seconds * RATE), channels)) * 0.2).astype(np.float32)
    data *= peak / np.abs(data).max()
    sf.write(str(path), data, RATE, subtype="FLOAT")
    return path


def _read(path):
    return sf.read(str(path), dtype="float32", always_2d=True)[0]


def _identity_denoiser(calls):
    """A stand-in separator that copies the chunk to its clean-stem name and counts its calls."""

    def denoise(chunk_wav, out_dir):
        calls.append(chunk_wav.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        produced = out_dir / f"{chunk_wav.stem}_(No Noise)_UVR-DeNoise.wav"
        sf.write(str(produced), _read(chunk_wav), RATE, subtype="FLOAT")
        return produced

    return denoise


# --------------------------------------------------------------------------- plan


def test_plan_is_empty_when_the_file_fits_or_the_chunk_would_not_clear_the_fade():
    assert chunking.plan(7 * RATE, RATE, chunk_seconds=7) == []
    assert chunking.plan(10**6, RATE, chunk_seconds=2 * chunking.CHUNK_OVERLAP_SECONDS) == []


@pytest.mark.parametrize(
    ("total_gb", "minutes"),
    [(61.6, 57), (30.0, 28), (16.0, 15), (12.0, 11), (8.0, 7), (2.0, 5), (512.0, 120)],
)
def test_chunk_length_is_sized_from_the_machine_and_clamped(total_gb, minutes):
    assert chunking.chunk_seconds_for_host(total_gb) == minutes * 60


def test_chunk_length_reads_the_machine_or_falls_back():
    with patch("modules.denoise_chunking.host_memory_gb", return_value=30.0):
        assert chunking.chunk_seconds_for_host() == 28 * 60
    with patch.dict("sys.modules", {"psutil": None}), patch("modules.denoise_chunking.cgroup_memory_gb", return_value=None):
        assert chunking.host_memory_gb() == chunking.FALLBACK_HOST_GB


def test_a_container_memory_limit_caps_the_machine_memory(tmp_path):
    """psutil reports the host inside a container; the cgroup limit is what the process really has."""
    limit = tmp_path / "memory.max"
    limit.write_text(str(8 * 2**30))
    # psutil is not installed in the light test environment, so a stand-in is injected
    fake_psutil = MagicMock(virtual_memory=MagicMock(return_value=MagicMock(total=64 * 2**30)))
    with patch("modules.denoise_chunking.CGROUP_LIMIT_FILES", (str(limit),)):
        assert chunking.cgroup_memory_gb() == 8.0
        with patch.dict("sys.modules", {"psutil": fake_psutil}):
            assert chunking.host_memory_gb() == 8.0
    limit.write_text("max")
    with patch("modules.denoise_chunking.CGROUP_LIMIT_FILES", (str(limit), str(tmp_path / "absent"))):
        assert chunking.cgroup_memory_gb() is None


def test_a_configured_chunk_length_overrides_the_automatic_one():
    with patch("modules.denoise_chunking.NEURAL_CHUNK_SECONDS", 1234.0):
        assert chunking.resolved_chunk_seconds() == 1234.0
    with patch("modules.denoise_chunking.NEURAL_CHUNK_SECONDS", 0.0), patch("modules.denoise_chunking.host_memory_gb", return_value=61.6):
        assert chunking.resolved_chunk_seconds() == 57 * 60


def _pairs(ranges):
    return list(zip(ranges, ranges[1:]))


def test_plan_cuts_equal_chunks_within_the_limit_that_overlap_by_the_fade():
    frames = 30 * RATE
    ranges = chunking.plan(frames, RATE, chunk_seconds=7)
    assert (ranges[0][0], ranges[-1][1]) == (0, frames)
    assert max(stop - start for start, stop in ranges) <= 7 * RATE
    assert {prev[1] - cur[0] for prev, cur in _pairs(ranges)} == {OVERLAP}
    assert len({cur[0] - prev[0] for prev, cur in _pairs(ranges)}) == 1


def test_plan_starts_every_chunk_on_the_patch_stride():
    ranges = chunking.plan(30 * RATE, RATE, chunk_seconds=7)
    assert all(start % STRIDE == 0 for start, _stop in ranges)


def test_plan_at_the_real_rate_lands_the_three_hour_capture_on_the_models_grid():
    with patch("modules.denoise_chunking.PATCH_FRAMES", 192 * 15360):
        ranges = chunking.plan(487_267_200, 44100, chunk_seconds=7200)
    assert len(ranges) == 2 and all(start % (192 * 15360) == 0 for start, _stop in ranges)
    assert all(stop - start <= 7200 * 44100 for start, stop in ranges)


def test_anchor_range_is_the_stride_aligned_block_holding_the_loudest_moment(tmp_path):
    data = np.full((30 * RATE, 2), 0.01, dtype=np.float32)
    data[1234] = 0.9  # inside block 24 (frames 1200-1249)
    sf.write(str(tmp_path / "in.wav"), data, RATE, subtype="FLOAT")
    with sf.SoundFile(str(tmp_path / "in.wav")) as source:
        assert chunking.anchor_range(source) == (1200, 1250)


def test_split_gives_every_chunk_its_anchor_and_a_stride_of_true_context(tmp_path):
    data = np.full((30 * RATE, 2), 0.01, dtype=np.float32)
    data[1234] = 0.9
    source = tmp_path / "in.wav"
    sf.write(str(source), data, RATE, subtype="FLOAT")
    frames = 30 * RATE
    ranges = chunking.plan(frames, RATE, chunk_seconds=7)
    paths, margins = chunking.split(source, tmp_path / "chunks", ranges)
    for path, (start, stop), (lead, tail) in zip(paths, ranges, margins):
        _check_chunk_layout(_read(path), data, (start, stop), (lead, tail), frames)


def _check_chunk_layout(chunk, data, span, margins, frames):
    """One chunk file: [anchor, faded into][context before][the range][context after]."""
    (start, stop), (lead, tail) = span, margins
    has_anchor = start <= 1200 and 1250 <= stop
    before, after = min(STRIDE, start), min(STRIDE, frames - stop)
    assert (lead, tail) == ((0 if has_anchor else STRIDE) + before, after)
    assert len(chunk) == lead + stop - start + tail
    body_end, after_end = lead + stop - start, stop + after
    np.testing.assert_array_equal(chunk[lead:body_end], data[start:stop])
    np.testing.assert_array_equal(chunk[body_end:], data[stop:after_end])
    _check_context(chunk, data, start, lead, before, has_anchor)


def _check_context(chunk, data, start, lead, before, has_anchor):
    """The context before the range is the source's, past the junction fade when an anchor precedes it."""
    faded_in = not has_anchor and before > 0
    context_from = lead - before + (FADE if faded_in else 0)
    source_from = start - lead + context_from
    np.testing.assert_array_equal(chunk[context_from:lead], data[source_from:start])
    if not has_anchor:
        _check_anchor_junction(chunk, data, faded_in)


def _check_anchor_junction(chunk, data, faded_in):
    """The anchor fades out, and the context fades in when there is one: a junction, not a cut."""
    anchor_end = STRIDE - FADE
    np.testing.assert_array_equal(chunk[:anchor_end], data[1200:1250][:anchor_end])
    assert chunk[STRIDE - 1, 0] == 0.0
    assert (chunk[STRIDE, 0] == 0.0) == faded_in


def test_plan_defaults_to_the_configured_chunk_length():
    with patch("modules.denoise_chunking.NEURAL_CHUNK_SECONDS", 7.0):
        assert chunking.plan(30 * RATE, RATE)


def test_duration_needs_chunking_reads_the_file(tmp_path):
    assert chunking.duration_needs_chunking(_wav(tmp_path / "long.wav", 30), chunk_seconds=7) is True
    assert chunking.duration_needs_chunking(_wav(tmp_path / "short.wav", 5), chunk_seconds=7) is False


def test_duration_needs_chunking_leaves_an_unreadable_file_whole(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_text("audio")
    assert chunking.duration_needs_chunking(bad, chunk_seconds=7) is False


# --------------------------------------------------------------------------- split and join


def test_split_writes_the_planned_ranges_as_float32(tmp_path):
    source = _wav(tmp_path / "in.wav", 30)
    ranges = chunking.plan(30 * RATE, RATE, chunk_seconds=7)
    paths, margins = chunking.split(source, tmp_path / "chunks", ranges)
    data = _read(source)
    for path, (start, stop), (lead, tail) in zip(paths, ranges, margins):
        assert sf.info(str(path)).subtype == "FLOAT"
        body_end = lead + stop - start
        np.testing.assert_array_equal(_read(path)[lead:body_end], data[start:stop])


def test_run_with_an_identity_denoiser_reproduces_the_source(tmp_path):
    """The crossfade of two identical signals sums to unity, so the join is the source."""
    source = _wav(tmp_path / "in.wav", 30)
    joined = chunking.run(source, tmp_path / "out", _identity_denoiser([]), chunk_seconds=7)
    assert joined == tmp_path / "out" / "in_(No Noise)_UVR-DeNoise.wav"
    np.testing.assert_allclose(_read(joined), _read(source), atol=1e-6)


def test_run_applies_the_whole_file_peak_rule_once_over_the_join(tmp_path):
    source = _wav(tmp_path / "hot.wav", 30, peak=1.2)
    joined = chunking.run(source, tmp_path / "out", _identity_denoiser([]), chunk_seconds=7)
    out, src = _read(joined), _read(source)
    assert np.abs(out).max() == pytest.approx(chunking.WHOLE_FILE_PEAK, abs=1e-6)
    np.testing.assert_allclose(out, src * (chunking.WHOLE_FILE_PEAK / np.abs(src).max()), atol=1e-6)


def test_run_resumes_from_finished_chunks(tmp_path):
    source = _wav(tmp_path / "in.wav", 30)
    first, second = [], []
    chunking.run(source, tmp_path / "out", _identity_denoiser(first), chunk_seconds=7)
    chunking.run(source, tmp_path / "out", _identity_denoiser(second), chunk_seconds=7)
    assert len(first) == len(chunking.plan(30 * RATE, RATE, chunk_seconds=7)) and second == []


def test_run_leaves_no_partial_and_keeps_chunks_under_the_output_directory(tmp_path):
    source = _wav(tmp_path / "in.wav", 30)
    out_dir = tmp_path / "out"
    chunking.run(source, out_dir, _identity_denoiser([]), chunk_seconds=7)
    assert not [p for p in out_dir.rglob("*") if ".tmp." in p.name]
    assert [p.name for p in out_dir.glob("*.wav")] == ["in_(No Noise)_UVR-DeNoise.wav"]


def _with_chunk_resized(tmp_path, paths, index, delta):
    """The chunk outputs with chunk `index` trimmed (delta < 0) or extended (delta > 0)."""
    data = _read(paths[index])
    data = data[:delta] if delta < 0 else np.concatenate([data, np.zeros((delta, data.shape[1]), np.float32)])
    resized = tmp_path / f"resized_{index}.wav"
    sf.write(str(resized), data, RATE, subtype="FLOAT")
    rest = index + 1
    return paths[:index] + [resized] + paths[rest:]


def test_join_pads_the_separator_trim_and_keeps_the_source_length(tmp_path):
    """The separator returns a few hundred frames fewer than it was given; the join must absorb that."""
    source = _wav(tmp_path / "in.wav", 30)
    ranges = chunking.plan(30 * RATE, RATE, chunk_seconds=7)
    paths, margins = chunking.split(source, tmp_path / "chunks", ranges)
    outputs = _with_chunk_resized(tmp_path, paths, 1, -10)
    outputs = _with_chunk_resized(tmp_path, outputs, len(paths) - 1, -10)
    joined = chunking.join(outputs, ranges, tmp_path / "joined.wav", margins)
    out, src = _read(joined), _read(source)
    assert len(out) == len(src)
    np.testing.assert_array_equal(out[-10:], 0.0)
    np.testing.assert_allclose(out[:-10], src[:-10], atol=0.02)


@pytest.mark.parametrize("delta", [-2 * RATE, 10])
def test_join_refuses_a_chunk_short_by_more_than_the_trim_or_longer_than_its_range(tmp_path, delta):
    source = _wav(tmp_path / "in.wav", 30)
    ranges = chunking.plan(30 * RATE, RATE, chunk_seconds=7)
    paths, margins = chunking.split(source, tmp_path / "chunks", ranges)
    with pytest.raises(RuntimeError, match="chunk 1 came back with"):
        chunking.join(_with_chunk_resized(tmp_path, paths, 1, delta), ranges, tmp_path / "joined.wav", margins)
    assert not (tmp_path / "joined.wav").exists()


def _limiting_denoiser(calls, limit_chunk="chunk_0000.wav"):
    """An identity denoiser that, like the separator, clips one chunk to full scale on its first attempt."""

    def denoise(chunk_wav, out_dir):
        calls.append(chunk_wav.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        data = _read(chunk_wav)
        if chunk_wav.name == limit_chunk:
            data = np.clip(data * 3.0, -chunking.CHUNK_PEAK, chunking.CHUNK_PEAK)
        produced = out_dir / f"{chunk_wav.stem}_(No Noise)_UVR-DeNoise.wav"
        sf.write(str(produced), data, RATE, subtype="FLOAT")
        return produced

    return denoise


def test_a_peak_limited_chunk_is_denoised_again_at_half_scale_and_the_join_undoes_it(tmp_path):
    """The retry input is the chunk at half scale, which the clipping denoiser then leaves alone."""
    source = _wav(tmp_path / "in.wav", 30)
    calls = []
    joined = chunking.run(source, tmp_path / "out", _limiting_denoiser(calls), chunk_seconds=7)
    assert calls[:2] == ["chunk_0000.wav", "chunk_0000_retry.wav"]
    np.testing.assert_allclose(_read(joined), _read(source), atol=1e-6)


def test_a_finished_retry_is_reused_on_resume(tmp_path):
    source = _wav(tmp_path / "in.wav", 30)
    chunking.run(source, tmp_path / "out", _limiting_denoiser([]), chunk_seconds=7)
    again = []
    chunking.run(source, tmp_path / "out", _limiting_denoiser(again), chunk_seconds=7)
    assert again == []


def test_joined_name_keeps_the_separator_suffix():
    assert chunking.joined_name("a/b/track.wav", "c/chunk_0003.wav", "c/chunk_0003_(No Noise)_M.wav") == "track_(No Noise)_M.wav"


# --------------------------------------------------------------------------- processing integration


def test_run_denoise_separator_takes_the_chunked_path_for_a_long_track(tmp_path):
    joined = tmp_path / "joined.wav"
    _wav(joined, 3)
    with (
        patch("modules.processing._denoise_chunking.duration_needs_chunking", return_value=True),
        patch("modules.processing._denoise_chunking.run", return_value=joined) as run,
        patch("modules.processing._build_separator") as build,
    ):
        result = processing._run_denoise_separator(tmp_path / "in.wav", tmp_path / "out", "Selected", "warn", "err")
    assert result == joined and run.call_count == 1 and build.call_count == 0


def test_run_denoise_separator_takes_the_whole_path_for_a_short_track(tmp_path):
    with (
        patch("modules.processing._denoise_chunking.duration_needs_chunking", return_value=False),
        patch("modules.processing._denoise_chunking.run") as run,
        patch("modules.processing._denoise_whole", return_value=_wav(tmp_path / "clean.wav", 3)) as whole,
    ):
        processing._run_denoise_separator(tmp_path / "in.wav", tmp_path / "out", "Selected", "warn", "err")
    assert whole.call_count == 1 and run.call_count == 0


def test_denoise_one_chunk_writes_at_full_scale_and_returns_the_clean_stem(tmp_path):
    out_dir = tmp_path / "out_0000"
    separator = MagicMock()

    def separate(path):
        _wav(out_dir / "chunk_0000_(No Noise)_M.wav", 3)

    separator.separate.side_effect = separate
    with (
        patch("modules.processing._build_separator", return_value=separator) as build,
        patch("modules.processing._load_separator_model"),
    ):
        produced = processing._denoise_one_chunk("M.pth")(tmp_path / "chunk_0000.wav", out_dir)
    assert produced.name == "chunk_0000_(No Noise)_M.wav"
    assert build.call_args.kwargs["normalization_threshold"] == chunking.CHUNK_PEAK


def test_denoise_one_chunk_raises_when_the_separator_wrote_no_clean_stem(tmp_path):
    with (
        patch("modules.processing._build_separator", return_value=MagicMock()),
        patch("modules.processing._load_separator_model"),
        pytest.raises(RuntimeError, match="no clean stem"),
    ):
        processing._denoise_one_chunk("M.pth")(Path("chunk_0000.wav"), tmp_path / "out_0000")
