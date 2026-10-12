"""`sync_dtw_enable`: true keeps every DTW decision as it was; false sends the length-preserving chains to shift sync."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import modules.config
from modules import apl_stems, processing, sync

REPO = Path(__file__).resolve().parents[2]
REF, PROC, OUT = Path("ref.wav"), Path("proc.wav"), Path("out.wav")


def _loaded(user_config):
    """The configuration load_config resolves from a config.yaml holding `user_config`."""
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = user_config
    with patch("modules.config.Path.exists", return_value=True), patch.object(modules.config, "yaml", mock_yaml):
        with patch("builtins.open", MagicMock()), patch("builtins.print") as warn:
            conf, _ = modules.config.load_config()
    return conf, warn


def _aligned(sync_method, length_preserving, enabled=True, default_method="shift"):
    """Which aligner `_align_stems` calls: "dtw" or "shift"."""
    with (
        patch("modules.sync.SYNC_DTW_ENABLE", enabled),
        patch("modules.sync.SYNC_METHOD", default_method),
        patch("modules.sync._align_stems_dtw", return_value=OUT) as dtw,
        patch("modules.sync._align_stems_shift", return_value=OUT) as shift,
        patch("modules.sync.log_msg") as log,
    ):
        assert sync._align_stems(REF, PROC, OUT, sync_method=sync_method, length_preserving=length_preserving) == OUT
    called = {"dtw": dtw, "shift": shift}
    assert sum(mock.call_count for mock in called.values()) == 1
    (name,) = [name for name, mock in called.items() if mock.called]
    called[name].assert_called_once_with(REF, PROC, OUT)
    return name, log


def test_the_key_defaults_to_dtw_in_the_code_and_in_the_shipped_config():
    """No config entry and the repository's config.yaml both keep DTW on, as every release has run."""
    conf, warn = _loaded({"process_mode": "cathar"})
    assert conf["sync_dtw_enable"] is True
    warn.assert_not_called()
    shipped = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    assert shipped["sync_dtw_enable"] is True
    assert modules.config.SYNC_DTW_ENABLE is True
    assert ("sync_dtw_enable", True) in modules.config._BOOL_CONFIG_FIELDS


@pytest.mark.parametrize(
    "raw, expected, warned", [("false", False, False), ("0", False, False), ("on", True, False), ("maybe", True, True)]
)
def test_the_key_parses_like_every_boolean(raw, expected, warned):
    """A quoted 'false' or '0' is off and 'on' is on; anything else warns and keeps DTW on."""
    conf, warn = _loaded({"sync_dtw_enable": raw})
    assert conf["sync_dtw_enable"] is expected
    assert warn.called is warned


@pytest.mark.parametrize("length_preserving", [True, False])
def test_on_by_default_a_dtw_decision_still_warps(length_preserving):
    """True: the drift clip takes DTW on every chain, as before, and nothing is logged about the key."""
    name, log = _aligned("dtw", length_preserving)
    assert name == "dtw"
    log.assert_not_called()


def test_off_a_length_preserving_chain_takes_shift_and_says_so():
    """False: the scanner's DTW decision becomes a shift sync for a chain that keeps the source's time base."""
    name, log = _aligned("dtw", True, enabled=False)
    assert name == "shift"
    assert "sync_dtw_enable" in log.call_args.args[0]


def test_off_the_stem_modes_keep_dtw():
    """False leaves a chain not marked length-preserving (hybrid, multipass, auto_pure) on DTW."""
    assert _aligned("dtw", False, enabled=False)[0] == "dtw"


def test_off_the_configured_method_is_overridden_too():
    """With no strategy override, `sync_method: dtw` from config.yaml also becomes shift on such a chain."""
    assert _aligned(None, True, enabled=False, default_method="dtw")[0] == "shift"
    assert _aligned(None, True, enabled=True, default_method="dtw")[0] == "dtw"


@pytest.mark.parametrize("enabled", [True, False])
def test_a_shift_decision_is_never_touched(enabled):
    """A clip the scanner sends to shift takes shift whatever the key says."""
    name, log = _aligned("shift", True, enabled=enabled)
    assert name == "shift"
    log.assert_not_called()


def test_the_single_track_pipeline_marks_its_chain_length_preserving(tmp_path):
    """Every single-track mode (auto_pure_linear, cathar, arnndn, denoise_only, the native chains) syncs as length-preserving."""
    filtered = tmp_path / "filtered.wav"
    with patch("modules.processing._align_stems") as align, patch("modules.processing._final_mux_single_audio_step"):
        processing._process_single_track_pipeline(
            tmp_path, tmp_path / "in.wav", "v.mp4", "o.mp4", 10.0, lambda *_a, **_k: filtered, "step", "Label", sync_method="dtw"
        )
    assert align.call_args.kwargs == {"sync_method": "dtw", "length_preserving": True}


def test_the_stem_mix_keeps_dtw_unless_its_caller_says_otherwise(tmp_path):
    """`_align_and_mix_stems` defaults to not length-preserving (the stem modes) and passes the flag to both stems."""
    strategy = {"sync_method": "dtw"}
    for flag, expected in ((None, False), (True, True)):
        kwargs = {} if flag is None else {"length_preserving": flag}
        with patch("modules.processing._align_stems") as align, patch("modules.processing._final_mix_step"):
            processing._align_and_mix_stems(
                tmp_path, REF, tmp_path / "v.wav", tmp_path / "b.wav", "v.mp4", "o.mp4", 9.0, strategy, **kwargs
            )
        assert [call.kwargs for call in align.call_args_list] == [{"sync_method": "dtw", "length_preserving": expected}] * 2


def test_auto_pure_linear_stem_path_syncs_as_length_preserving(tmp_path):
    """The APL stem path's vocal chain and background pass keep the source's time base."""
    strategy = {"profile": {"tonal_persistence": 0.2}, "sync_method": "dtw"}
    with (
        patch("modules.processing._separate_stems_step", return_value=(tmp_path / "v.wav", tmp_path / "b.wav")),
        patch("modules.apl_stems._background_pass", return_value=tmp_path / "bg.wav"),
        patch("modules.processing._align_and_mix_stems") as mix,
    ):
        assert apl_stems.execute(tmp_path, REF, REF, "v.mp4", "o.mp4", 5.0, strategy, lambda wav, out_dir, total_duration=None: wav)
    assert mix.call_args.kwargs == {"length_preserving": True}
    assert mix.call_args.args[7] is strategy
