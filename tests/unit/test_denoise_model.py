"""The denoise model's declared output: the clean stem every selection site takes, and the MDXC window grid."""

from pathlib import Path
from unittest.mock import patch

import pytest

from modules import denoise_model

ROFORMER = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
ROFORMER_MODEL_STEM = "denoise_mel_band_roformer_aufr33_sdr_27"
UVR_LITE = "UVR-DeNoise-Lite.pth"
UVR = "UVR-DeNoise.pth"

# The sections of models/denoise_mel_band_roformer_aufr33_sdr_27.9959_config.yaml the app
# reads, as the model store holds them, the Python tuple tag a safe loader refuses included.
SHIPPED_YAML = """audio:
  chunk_size: 352800
  dim_f: 1024
  dim_t: 801
  hop_length: 441
  n_fft: 2048
  num_channels: 2
  sample_rate: 44100
  min_mean_abs: 000

model:
  dim: 384
  num_bands: 60
  stft_n_fft: 2048
  stft_hop_length: 441
  stft_win_length: 2048
  multi_stft_resolutions_window_sizes: !!python/tuple
  - 4096
  - 2048
  - 1024
  multi_stft_hop_size: 147

training:
  instruments:
  - dry
  - other
  target_instrument: dry

inference:
  batch_size: 2
  dim_t: 801
  num_overlap: 4
"""


def _store(tmp_path, name=f"{Path(ROFORMER).stem}_config.yaml", text=SHIPPED_YAML):
    store = tmp_path / "models"
    store.mkdir(parents=True, exist_ok=True)
    (store / name).write_text(text, encoding="utf-8")
    return store


def _names(paths):
    return [Path(path).name for path in paths]


# --------------------------------------------------------------------------- the model store


def test_model_store_follows_the_separator_override_or_the_app_store(tmp_path):
    assert denoise_model.model_store({denoise_model.MODEL_DIR_ENV: str(tmp_path)}) == tmp_path
    assert denoise_model.model_store({}) == Path(denoise_model.utils.MODELS_DIR)


def test_the_environment_override_is_read_at_call_time(tmp_path, monkeypatch):
    store = _store(tmp_path)
    monkeypatch.setenv(denoise_model.MODEL_DIR_ENV, str(store))
    assert denoise_model.target_tag(ROFORMER) == "(dry)"


def test_config_path_finds_the_config_beside_the_checkpoint_in_either_naming(tmp_path):
    store = _store(tmp_path)
    (store / "model_bs_roformer_ep_317_sdr_12.9755.yaml").write_text(SHIPPED_YAML, encoding="utf-8")
    assert denoise_model.config_path(ROFORMER, store) == store / f"{Path(ROFORMER).stem}_config.yaml"
    assert denoise_model.config_path("model_bs_roformer_ep_317_sdr_12.9755.ckpt", store).name.endswith("12.9755.yaml")


@pytest.mark.parametrize("model", [UVR_LITE, None, ""])
def test_a_vr_model_or_no_model_has_no_config(tmp_path, model):
    assert denoise_model.config_path(model, _store(tmp_path)) is None


def test_the_shipped_yaml_parses_despite_its_python_tuple_tag(tmp_path):
    data = denoise_model.model_config(ROFORMER, _store(tmp_path))
    assert data["model"]["multi_stft_resolutions_window_sizes"] == [4096, 2048, 1024]
    assert data["training"]["target_instrument"] == "dry"


def test_an_unreadable_or_non_mapping_yaml_is_no_config(tmp_path):
    store = _store(tmp_path, text="audio: [unclosed\n")
    with patch("modules.denoise_model.utils.log_msg") as log:
        assert denoise_model.model_config(ROFORMER, store) is None
    assert "Could not read the model config" in log.call_args.args[0]
    listed = _store(tmp_path / "list", text="- dry\n- other\n")
    assert denoise_model.model_config(ROFORMER, listed) is None


def test_a_rewritten_yaml_is_read_again(tmp_path):
    store = _store(tmp_path)
    assert denoise_model.target_tag(ROFORMER, store) == "(dry)"
    _store(tmp_path, text=SHIPPED_YAML.replace("target_instrument: dry", "target_instrument: speech_clean"))
    assert denoise_model.target_tag(ROFORMER, store) == "(speech_clean)"


# --------------------------------------------------------------------------- targets and residuals


def test_target_tag_reads_the_training_target_and_never_a_residual(tmp_path):
    assert denoise_model.target_tag(ROFORMER, _store(tmp_path)) == "(dry)"
    assert denoise_model.target_tag(UVR_LITE, _store(tmp_path)) is None
    noisy = _store(tmp_path / "noisy", text=SHIPPED_YAML.replace("target_instrument: dry", "target_instrument: Noise"))
    assert denoise_model.target_tag(ROFORMER, noisy) is None
    untargeted = _store(tmp_path / "untargeted", text=SHIPPED_YAML.replace("target_instrument: dry", "target_instrument: null"))
    assert denoise_model.target_tag(ROFORMER, untargeted) is None


def test_residual_names_add_the_instruments_beside_a_clean_target(tmp_path):
    assert denoise_model.residual_names(ROFORMER, _store(tmp_path)) == ("noise", "other", "other")
    assert denoise_model.residual_names(UVR_LITE, _store(tmp_path)) == denoise_model.RESIDUAL_NAMES
    noise_target = SHIPPED_YAML.replace("- other", "- noise").replace("target_instrument: dry", "target_instrument: noise")
    assert denoise_model.residual_names(ROFORMER, _store(tmp_path / "n", text=noise_target)) == denoise_model.RESIDUAL_NAMES


# --------------------------------------------------------------------------- the clean stem


def _roformer_outputs(base):
    return [Path(f"{base}_(other)_{ROFORMER_MODEL_STEM}.wav"), Path(f"{base}_(dry)_{ROFORMER_MODEL_STEM}.wav")]


def _uvr_outputs(base, model="UVR-DeNoise-Lite"):
    return [Path(f"{base}_(Noise)_{model}.wav"), Path(f"{base}_(No Noise)_{model}.wav")]


@pytest.mark.parametrize("base", ["surgical", "chunk_0003", "Zed tape (1)", "a"])
def test_the_roformer_clean_stem_is_its_dry_target(tmp_path, base):
    store = _store(tmp_path)
    assert denoise_model.clean_stem(_roformer_outputs(base), ROFORMER, store).name == f"{base}_(dry)_{ROFORMER_MODEL_STEM}.wav"


@pytest.mark.parametrize("model", [UVR_LITE, UVR, None])
def test_the_uvr_clean_stem_is_no_noise_and_never_noise(tmp_path, model):
    chosen = denoise_model.clean_stem(_uvr_outputs("track"), model, _store(tmp_path))
    assert chosen.name == "track_(No Noise)_UVR-DeNoise-Lite.wav"


def test_without_its_yaml_the_roformer_still_gets_dry_since_other_is_a_residual(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert denoise_model.clean_stem(_roformer_outputs("in"), ROFORMER, empty).name.startswith("in_(dry)_")


@pytest.mark.parametrize(
    "names",
    [
        ["in_(other)_m.wav"],
        ["in_(Noise)_UVR-DeNoise.wav"],
        ["in_(noise)_m.wav", "in_(Other)_m.wav"],
        [],
    ],
)
def test_a_residual_alone_is_no_clean_stem(tmp_path, names):
    assert denoise_model.clean_stem([Path(name) for name in names], ROFORMER, _store(tmp_path)) is None


def test_only_the_stem_the_separator_appended_makes_a_residual(tmp_path):
    """A background stem from a vocal split is '..._(other)_...' before any denoiser touches it."""
    outputs = [Path("audio_(other)_kim_(Noise)_X.wav"), Path("audio_(other)_kim_(denoised)_X.wav")]
    assert denoise_model.clean_stem(outputs, "Some-Model.onnx", tmp_path).name == "audio_(other)_kim_(denoised)_X.wav"
    assert denoise_model.clean_stem([Path("plain.wav")], None, tmp_path).name == "plain.wav"


def test_an_unknown_model_keeps_the_first_file_in_name_order(tmp_path):
    names = [Path("background_(Background)_denoised.wav"), Path("Background_(Vocals)_x.wav")]
    chosen = denoise_model.clean_stem(names, "Some-Model.onnx", tmp_path)
    assert chosen.name == "background_(Background)_denoised.wav"


def test_a_noise_target_model_gets_its_clean_instrument(tmp_path):
    text = SHIPPED_YAML.replace("- other", "- noise").replace("target_instrument: dry", "target_instrument: noise")
    store = _store(tmp_path, text=text)
    outputs = [Path("in_(noise)_m.wav"), Path("in_(dry)_m.wav")]
    assert denoise_model.clean_stem(outputs, ROFORMER, store).name == "in_(dry)_m.wav"


def _head_choice(candidates):
    """The whole-file pass's selection at bb5fa2a: '(No Noise)' first, else the first file in name order."""
    ordered = sorted(candidates, key=lambda path: path.name.lower())
    clean = [path for path in ordered if "(No Noise)" in path.name]
    return clean[0] if clean else ordered[0]


@pytest.mark.parametrize("base", ["surgical_notched", "x", "Tape 3 (side B)", "zz_last", "In-Put.v2"])
def test_whole_file_runs_keep_the_file_they_chose_before(tmp_path, base):
    """Hashes of files under the chunk length cannot move: the resolver picks what name order picked."""
    store = _store(tmp_path)
    for outputs, model in ((_roformer_outputs(base), ROFORMER), (_uvr_outputs(base), UVR_LITE), (_uvr_outputs(base, "UVR-DeNoise"), UVR)):
        assert denoise_model.clean_stem(outputs, model, store) == _head_choice(outputs)
        assert denoise_model.clean_stem(list(reversed(outputs)), model, store) == _head_choice(outputs)


# --------------------------------------------------------------------------- the MDXC window grid


def test_the_shipped_roformer_overlap_adds_8_s_windows_every_2_s(tmp_path):
    assert denoise_model.mdxc_window(ROFORMER, _store(tmp_path)) == (352_800, 88_200)


def test_a_vr_model_or_a_missing_yaml_has_no_mdxc_window(tmp_path):
    assert denoise_model.mdxc_window(UVR_LITE, _store(tmp_path)) is None
    assert denoise_model.mdxc_window(ROFORMER, tmp_path) is None


def test_a_non_roformer_mdxc_model_sizes_its_window_from_the_audio_hop(tmp_path):
    text = SHIPPED_YAML.replace("  num_bands: 60\n", "").replace("hop_length: 441\n  n_fft", "hop_length: 1024\n  n_fft")
    store = _store(tmp_path, name="MDX23C-8KFFT-InstVoc_HQ.yaml", text=text)
    assert denoise_model.mdxc_window("MDX23C-8KFFT-InstVoc_HQ.ckpt", store) == (1024 * 800, 1024 * 800 // 4)


def test_a_roformer_marker_takes_the_stft_hop_whatever_the_file_is_called(tmp_path):
    text = SHIPPED_YAML.replace("hop_length: 441\n  n_fft", "hop_length: 1024\n  n_fft")
    store = _store(tmp_path, name="plain_name_config.yaml", text=text)
    assert denoise_model.mdxc_window("plain_name.ckpt", store) == (352_800, 88_200)


def test_the_overlap_defaults_to_the_separators_eight(tmp_path):
    store = _store(tmp_path, text=SHIPPED_YAML.replace("  num_overlap: 4\n", ""))
    assert denoise_model.mdxc_window(ROFORMER, store) == (352_800, 44_100)


@pytest.mark.parametrize(
    "text",
    [
        SHIPPED_YAML.replace("inference:", "unused:"),
        SHIPPED_YAML.replace("num_overlap: 4", "num_overlap: 0"),
        SHIPPED_YAML.replace("num_overlap: 4", "num_overlap: -4"),
        SHIPPED_YAML.replace("  dim_t: 801\n  num_overlap", "  dim_t: eight\n  num_overlap"),
        SHIPPED_YAML.replace("num_overlap: 4", "num_overlap: 400000000"),
    ],
)
def test_an_unusable_inference_section_has_no_window(tmp_path, text):
    assert denoise_model.mdxc_window(ROFORMER, _store(tmp_path, text=text)) is None
