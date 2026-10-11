"""The chunked denoiser with an MDXC model: its grid, its input level, its clean stem, and a join equal to the whole-file pass.

The stand-in separator is audio-separator 0.47.0's RoFormer path in miniature: the input peak
rule, windows overlap-added on the step grid with a Hamming weight and a tail window, a
level- and framing-dependent mask in place of the network, the residual as the mix less the
target, the output peak rule, and '(other)' written before '(dry)'. A chunk that saw what the
whole file's windows saw comes back as the whole file's samples; one cut to the VR grid does not.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

import modules.processing as processing
from modules import denoise_chunking as chunking
from modules import denoise_model
from tests.unit.test_denoise_model import SHIPPED_YAML

RATE = 100  # Hz: small files, exact frame arithmetic
MODEL = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
MODEL_STEM = "denoise_mel_band_roformer_aufr33_sdr_27"  # the separator's model name: the file name up to its first dot
UVR_LITE = "UVR-DeNoise-Lite.pth"
WINDOW, STEP = 40, 10  # the shipped 352,800 / 88,200, scaled down with the rate
# The shipped yaml with the window scaled down: hop 1 x (41 - 1) samples, four windows deep.
SMALL_YAML = (
    SHIPPED_YAML.replace("stft_hop_length: 441", "stft_hop_length: 1")
    .replace("hop_length: 441\n  n_fft", "hop_length: 1\n  n_fft")
    .replace("  dim_t: 801\n  num_overlap", "  dim_t: 41\n  num_overlap")
)
# A VR patch stride that, like the real one (2,949,120 = 33.4 steps of 88,200), is off the step grid.
OFF_GRID_STRIDE = 55
CHUNK_SECONDS = 7


@pytest.fixture(autouse=True)
def _small_store(tmp_path, monkeypatch):
    store = tmp_path / "models"
    store.mkdir()
    (store / f"{Path(MODEL).stem}_config.yaml").write_text(SMALL_YAML, encoding="utf-8")
    monkeypatch.setenv(denoise_model.MODEL_DIR_ENV, str(store))
    with patch("modules.denoise_chunking.PATCH_FRAMES", OFF_GRID_STRIDE), patch("modules.denoise_chunking.JUNCTION_FADE_FRAMES", 4):
        yield store


def _wav(path, seconds, peak=0.5, seed=0):
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal((int(seconds * RATE), 2)) * 0.2).astype(np.float32)
    data *= np.float32(peak) / np.abs(data).max()
    sf.write(str(path), data, RATE, subtype="FLOAT")
    return path


def _read(path):
    return sf.read(str(path), dtype="float32", always_2d=True)[0]


def _normalize(wave, max_peak):
    """spec_utils.normalize: scaled down to `max_peak` when its peak is above it, in place."""
    maxv = np.abs(wave).max()
    if maxv > max_peak:
        wave *= max_peak / maxv
    return wave


def _window_starts(length):
    """MDXCSeparator._roformer_chunk_starts: the step grid, then one window ending at the input's end."""
    starts = []
    for offset in range(0, length, STEP):
        if offset + WINDOW >= length:
            return starts + [max(length - WINDOW, 0)]
        starts.append(offset)
    return starts


def _network(part):
    """The network's stand-in: a mask from the window's own power spectrum, so both framing and level matter."""
    spectrum = np.fft.rfft(part, axis=-1)
    power = np.abs(spectrum) ** 2
    return np.fft.irfft(spectrum * (power / (power + 0.5)), n=part.shape[-1], axis=-1).astype(np.float32)


def _demix(mix):
    """The RoFormer branch of MDXCSeparator.demix: Hamming-weighted overlap-add of independently denoised windows."""
    result, counter = np.zeros_like(mix), np.zeros_like(mix)
    weight = scipy.signal.windows.hamming(WINDOW).astype(np.float32)
    for start in _window_starts(mix.shape[1]):
        end = start + WINDOW
        part = mix[:, start:end]
        length = part.shape[-1]
        end = start + length
        result[:, start:end] += _network(part)[:, :length] * weight[:length]
        counter[:, start:end] += weight[:length]
    return result / np.maximum(counter, np.float32(1e-10))


class _FakeMdxc:
    """The separator, as `_build_separator` would configure it, running the stand-in RoFormer."""

    def __init__(self, output_dir, threshold, calls):
        self.output_dir, self.threshold, self.calls = Path(output_dir), threshold, calls

    def separate(self, path):
        self.calls.append(Path(path).name)
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        mix = _normalize(np.ascontiguousarray(data.T), self.threshold)
        dry = _demix(mix)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for stem, wave in (("other", mix - dry), ("dry", dry)):
            wave = _normalize(_normalize(wave, self.threshold), self.threshold)
            sf.write(str(self.output_dir / f"{Path(path).stem}_({stem})_{MODEL_STEM}.wav"), wave.T, rate, subtype="FLOAT")

    def download_model_files(self, model):
        """Separator.download_model_files: the checkpoint's yaml lands in the store, as on a first load."""
        self.calls.append(f"download {model}")
        (denoise_model.model_store() / f"{Path(model).stem}_config.yaml").write_text(SMALL_YAML, encoding="utf-8")


class _FakeVr(_FakeMdxc):
    """A VR denoiser in the same shape: three quarters of the input as '(No Noise)', the rest as '(Noise)'."""

    def separate(self, path):
        self.calls.append(Path(path).name)
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for stem, wave in (("Noise", data * np.float32(0.25)), ("No Noise", data * np.float32(0.75))):
            sf.write(str(self.output_dir / f"{Path(path).stem}_({stem})_UVR-DeNoise-Lite.wav"), wave, rate, subtype="FLOAT")


def _separators(fake, calls):
    """Patches the processing module's separator factory and model load with the stand-in."""

    def build(output_dir, normalization_threshold=chunking.WHOLE_FILE_PEAK):
        return fake(output_dir, normalization_threshold, calls)

    return (
        patch("modules.processing._build_separator", side_effect=build),
        patch("modules.processing._load_separator_model"),
    )


def _whole_and_chunked(tmp_path, source, model, fake=_FakeMdxc, chunk_model=None):
    """(whole-file clean stem, chunked join) from the processing module's own two paths, and the separator calls."""
    calls = []
    build, load = _separators(fake, calls)
    with build, load:
        whole = processing._denoise_whole(source, tmp_path / "whole", model, "warn", False)
        joined = chunking.run(source, tmp_path / "chunked", processing._denoise_one_chunk(model), CHUNK_SECONDS, chunk_model)
    return whole, joined, calls


# --------------------------------------------------------------------------- the grid


def test_grid_for_an_mdxc_model_is_its_step_with_a_window_of_context():
    assert chunking.grid_for(MODEL) == chunking.Grid(STEP, WINDOW, anchored=False, peak_matched=True, tag=f"_mdxc{STEP}")


def test_grid_for_the_shipped_roformer_at_44_1_khz(_small_store):
    (_small_store / f"{Path(MODEL).stem}_config.yaml").write_text(SHIPPED_YAML, encoding="utf-8")
    assert chunking.grid_for(MODEL) == chunking.Grid(88_200, 352_800, anchored=False, peak_matched=True, tag="_mdxc88200")


@pytest.mark.parametrize("model", [None, "", UVR_LITE, "UVR-DeNoise.pth"])
def test_a_vr_model_or_none_keeps_the_patch_grid_and_its_anchor(model):
    assert chunking.grid_for(model) == chunking.Grid(OFF_GRID_STRIDE, OFF_GRID_STRIDE)
    assert chunking.grid_for(model).tag == ""


def test_a_checkpoint_the_fetch_left_without_a_config_takes_the_vr_grid_and_says_so(_small_store):
    (_small_store / f"{Path(MODEL).stem}_config.yaml").unlink()
    fetched = []
    with patch("modules.denoise_chunking.log_msg") as log:
        assert chunking._grid_after_fetch(MODEL, lambda: fetched.append(MODEL)) == chunking.vr_grid()
    assert fetched == [MODEL]
    assert "No usable config" in log.call_args.args[0]
    assert log.call_args.kwargs == {"level": "WARNING"}


@pytest.mark.parametrize("model", [MODEL, UVR_LITE, None])
def test_nothing_is_fetched_or_warned_when_the_config_is_there_or_none_is_due(model):
    with patch("modules.denoise_chunking.log_msg") as log:
        chunking._grid_after_fetch(model, lambda: pytest.fail("fetched"))
    assert log.call_count == 0


def test_the_plan_of_a_three_hour_capture_lands_on_the_roformer_step():
    """57-minute chunks (this host's automatic length) of a 3h04m capture, every start on the 2 s step."""
    ranges = chunking.plan(487_267_200, 44100, chunk_seconds=3420, stride=88_200)
    assert len(ranges) == 4
    assert all(start % 88_200 == 0 for start, _stop in ranges)
    assert all(stop - start <= 3420 * 44100 + 88_200 for start, stop in ranges)
    assert ranges[-1][1] == 487_267_200


def test_whether_a_file_is_chunked_does_not_depend_on_the_grid():
    assert chunking.plan(30 * RATE, RATE, CHUNK_SECONDS, stride=STEP)
    assert chunking.plan(7 * RATE, RATE, CHUNK_SECONDS, stride=STEP) == []


# --------------------------------------------------------------------------- the chunk files


def _separator_input(source):
    """The whole file as the separator feeds it to the network: read as float32 and peak-normalised at 0.9."""
    return _normalize(np.ascontiguousarray(_read(source).T), chunking.WHOLE_FILE_PEAK).T


@pytest.mark.parametrize("peak", [0.5, 0.9, 1.3])
def test_mdxc_chunks_hold_the_whole_files_input_samples_bit_for_bit(tmp_path, peak):
    """No anchor, a window of true context either side, and the whole file's input gain."""
    source = _wav(tmp_path / "in.wav", 30, peak=peak)
    ranges = chunking.plan(30 * RATE, RATE, CHUNK_SECONDS, stride=STEP)
    paths, margins = chunking.split(source, tmp_path / "chunks", ranges, chunking.grid_for(MODEL))
    fed = _separator_input(source)
    for path, (start, stop), (lead, tail) in zip(paths, ranges, margins):
        assert (lead, tail) == (min(WINDOW, start), min(WINDOW, 30 * RATE - stop))
        first, last = start - lead, stop + tail
        np.testing.assert_array_equal(_read(path), fed[first:last])
        assert np.abs(_read(path)).max() <= chunking.CHUNK_PEAK


def test_the_input_gain_is_the_separators_peak_rule(tmp_path):
    with sf.SoundFile(str(_wav(tmp_path / "quiet.wav", 3, peak=0.9))) as quiet:
        assert chunking.whole_file_input_gain(quiet) == 1.0
    with sf.SoundFile(str(_wav(tmp_path / "hot.wav", 3, peak=1.5))) as hot:
        gain = chunking.whole_file_input_gain(hot)
    assert gain == chunking.WHOLE_FILE_PEAK / np.float32(1.5)
    assert gain.dtype == np.float32


# --------------------------------------------------------------------------- the join against the whole file


@pytest.mark.parametrize("peak", [0.5, 1.3])
def test_the_mdxc_join_is_the_whole_file_result(tmp_path, peak):
    """Body and seams alike: every window touching a chunk's range is one the whole file ran."""
    source = _wav(tmp_path / "in.wav", 30, peak=peak)
    whole, joined, calls = _whole_and_chunked(tmp_path, source, MODEL, chunk_model=MODEL)
    assert whole.name == f"in_(dry)_{MODEL_STEM}.wav"
    assert joined.name == f"in_(dry)_{MODEL_STEM}.wav"
    assert len(calls) == 1 + len(chunking.plan(30 * RATE, RATE, CHUNK_SECONDS, stride=STEP))
    np.testing.assert_allclose(_read(joined), _read(whole), rtol=0, atol=1e-6)


def test_cut_to_the_vr_grid_the_same_model_drifts_from_the_whole_file(tmp_path):
    """The control: before this grid, chunks sat off the step and were normalised by their own peak."""
    source = _wav(tmp_path / "in.wav", 30, peak=1.3)
    whole, joined, _calls = _whole_and_chunked(tmp_path, source, MODEL, chunk_model=None)
    assert np.abs(_read(joined) - _read(whole)).max() > 1e-3


def _through_the_runner(tmp_path, calls):
    """(chunked join from processing's own runner, as the APL step calls it; the whole-file clean stem) of a hot 30 s file."""
    source = _wav(tmp_path / "in.wav", 30, peak=1.3)
    build, load = _separators(_FakeMdxc, calls)
    with build, load, patch("modules.denoise_chunking.NEURAL_CHUNK_SECONDS", CHUNK_SECONDS):
        joined = processing._run_denoise_separator(source, tmp_path / "chunked", "Selected", "warn", "err", False, MODEL)
        whole = processing._denoise_whole(source, tmp_path / "whole", MODEL, "warn", False)
    return joined, whole


def _assert_cut_to_the_mdxc_grid(tmp_path, joined, whole):
    ranges = chunking.plan(30 * RATE, RATE, CHUNK_SECONDS, stride=STEP)
    assert (tmp_path / "chunked" / f"chunks_{len(ranges)}x{ranges[0][1]}_mdxc{STEP}").is_dir()
    assert joined.name == whole.name == f"in_(dry)_{MODEL_STEM}.wav"
    np.testing.assert_allclose(_read(joined), _read(whole), rtol=0, atol=1e-6)


def test_the_denoise_runner_hands_its_model_to_the_chunked_pass(tmp_path):
    """A track over the chunk length is cut to the model's grid; a store holding its yaml fetches nothing."""
    calls = []
    joined, whole = _through_the_runner(tmp_path, calls)
    _assert_cut_to_the_mdxc_grid(tmp_path, joined, whole)
    assert not [call for call in calls if call.startswith("download")]


def test_a_fresh_store_fetches_the_model_before_its_chunks_are_cut(tmp_path, _small_store):
    """The yaml arrives with the model's first load; without the fetch a first long tape would be cut to the VR grid."""
    config = _small_store / f"{Path(MODEL).stem}_config.yaml"
    config.unlink()
    calls = []
    joined, whole = _through_the_runner(tmp_path, calls)
    assert calls[0] == f"download {MODEL}"
    assert config.is_file()
    _assert_cut_to_the_mdxc_grid(tmp_path, joined, whole)


def test_the_fetcher_downloads_the_model_files_without_loading_the_model(tmp_path):
    with patch("modules.processing._build_separator") as build:
        processing._model_fetcher(MODEL, tmp_path)()
    build.assert_called_once_with(tmp_path)
    build.return_value.download_model_files.assert_called_once_with(MODEL)
    assert build.return_value.load_model.call_count == 0


def test_the_chunk_directory_names_the_grid(tmp_path):
    source = _wav(tmp_path / "in.wav", 30)
    _whole, _joined, _calls = _whole_and_chunked(tmp_path, source, MODEL, chunk_model=MODEL)
    ranges = chunking.plan(30 * RATE, RATE, CHUNK_SECONDS, stride=STEP)
    assert (tmp_path / "chunked" / f"chunks_{len(ranges)}x{ranges[0][1]}_mdxc{STEP}").is_dir()


# --------------------------------------------------------------------------- the clean stem on both paths


def test_a_vr_model_still_takes_no_noise_on_both_paths(tmp_path):
    source = _wav(tmp_path / "in.wav", 30)
    whole, joined, _calls = _whole_and_chunked(tmp_path, source, UVR_LITE, fake=_FakeVr, chunk_model=UVR_LITE)
    assert whole.name == joined.name == "in_(No Noise)_UVR-DeNoise-Lite.wav"
    np.testing.assert_allclose(_read(joined), _read(source) * np.float32(0.75), atol=1e-6)


def test_a_chunk_left_with_only_its_residual_is_denoised_again(tmp_path):
    """A run stopped between the separator's two writes leaves '(other)' alone: that is no finished chunk."""
    source = _wav(tmp_path / "in.wav", 30)
    ranges = chunking.plan(30 * RATE, RATE, CHUNK_SECONDS, stride=STEP)
    stale = tmp_path / "chunked" / f"chunks_{len(ranges)}x{ranges[0][1]}_mdxc{STEP}" / "out_0000"
    stale.mkdir(parents=True)
    _wav(stale / f"chunk_0000_(other)_{MODEL_STEM}.wav", 9)
    _whole, _joined, calls = _whole_and_chunked(tmp_path, source, MODEL, chunk_model=MODEL)
    assert "chunk_0000.wav" in calls


def test_the_chunk_step_raises_when_only_residuals_came_back(tmp_path):
    out_dir = tmp_path / "out_0000"

    class _ResidualOnly(_FakeMdxc):
        def separate(self, path):
            _wav(out_dir / f"chunk_0000_(other)_{MODEL_STEM}.wav", 3)

    build, load = _separators(_ResidualOnly, [])
    with build, load, pytest.raises(RuntimeError, match="no clean stem"):
        processing._denoise_one_chunk(MODEL)(_wav(tmp_path / "chunk_0000.wav", 3), out_dir)


# --------------------------------------------------------------------------- what a previous run left


def _left_behind(directory, stems):
    """A denoise directory holding the stems a previous run wrote."""
    directory.mkdir()
    for stem in stems:
        _wav(directory / f"in_({stem})_{MODEL_STEM}.wav", 3)
    return directory


def test_a_finished_dry_output_is_reused(tmp_path):
    done = _left_behind(tmp_path / "done", ("other", "dry"))
    with patch("modules.processing._run_denoise_separator") as run:
        reused = processing._denoise_full_audio_step(_wav(tmp_path / "in.wav", 3), done, denoise_model=MODEL)
    assert reused.name == f"in_(dry)_{MODEL_STEM}.wav"
    assert run.call_count == 0


def test_a_residual_alone_is_denoised_again(tmp_path):
    source = _wav(tmp_path / "in.wav", 3)
    stale = _left_behind(tmp_path / "stale", ("other",))
    with patch("modules.processing._run_denoise_separator", return_value=source) as run:
        assert processing._denoise_full_audio_step(source, stale, denoise_model=MODEL) == source
    assert run.call_count == 1


def test_the_whole_file_pass_refuses_an_output_that_is_only_a_residual(tmp_path):
    out_dir = tmp_path / "whole"
    out_dir.mkdir()
    _wav(out_dir / f"in_(other)_{MODEL_STEM}.wav", 3)
    with pytest.raises(RuntimeError, match="no denoised output"):
        processing._select_denoised_candidate(list(out_dir.glob("*.wav")), "no denoised output", tmp_path / "in.wav", False, MODEL)
    assert processing._select_denoised_candidate([], "warn", tmp_path / "in.wav", True, MODEL) == tmp_path / "in.wav"
