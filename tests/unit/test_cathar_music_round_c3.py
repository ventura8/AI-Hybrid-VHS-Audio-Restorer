"""cathar's music profile, round C3: its own subtraction floor, spike-repair switch and dewind cutoff.

Until 2026-10-09 a music tape ran `cathar_beta`, `cathar_enable_repair` and `cathar_dewind_cutoff`. The
command lines below were recorded from HEAD 54048a9 (before the split) with the repository's config.yaml:
a speech tape and a music tape, the noise print path replaced by `<NP>`. cathar is deterministic, so the same
command lines on the same input are the same audio; the new keys' defaults must rebuild them exactly.
"""

from unittest.mock import MagicMock, patch

import pytest

import modules.cathar as cathar
from modules import config as cfg
from modules import processing
from modules.filters import _precondition_filter_from_config

NP = "<NP>"
SHARED_HEAD = [
    ["dewind", "--cutoff", "80"],
    ["azimuth", "--max-ms", "5.0", "--method", "gcc-phat"],
    ["stereo", "--mono-below", "100"],
    ["declick", "--method", "ar", "--threshold", "8.0"],
    ["decrackle", "--sensitivity", "6"],
    ["inpaint", "--max-gap-ms", "50", "--iterations", "3"],
]
REPAIR_HEAD = [
    ["declip", "--method", "spade", "--threshold", "0.95"],
    ["dehum", "--freq", "50", "--harmonics", "8", "--adaptive"],
    ["repair", "-s", "2"],
]
HEAD_COMMANDS = {
    "speech": SHARED_HEAD
    + [["deplosive", "-s", "4"]]
    + REPAIR_HEAD
    + [
        ["denoise", "--alpha", "1.0", "--beta", "0.02", "--noiseprint", NP],
        ["deesser", "--bands", "3", "-f", "4000", "--threshold=12.0"],
    ],
    "music": SHARED_HEAD + REPAIR_HEAD + [["denoise", "--alpha", "0.5", "--beta", "0.02", "--coherent"]],
}
PRECONDITION_HEAD = {
    "speech": "highpass=f=60,adeclick,bandreject=f=50.0:width_type=q:w=15,"
    "bandreject=f=100.0:width_type=q:w=15,bandreject=f=15625.0:width_type=q:w=30",
    "music": "highpass=f=60,adeclick,bandreject=f=50.0:width_type=q:w=15,"
    "bandreject=f=100.0:width_type=q:w=15,bandreject=f=15625.0:width_type=q:w=60",
}
PERSISTENCE = {"speech": 0.001, "music": 0.2}
NEW_KEYS = ("cathar_music_beta", "cathar_music_enable_repair", "cathar_music_dewind_cutoff")
SPEECH_KEYS = ("cathar_beta", "cathar_enable_repair", "cathar_dewind_cutoff")


def _strategy(material):
    return {
        "precondition_filters": {"notch_hz": 50.0, "highpass_hz": 60, "crt_notch_hz": 15625.0},
        "profile": {"tonal_persistence": PERSISTENCE[material]},
    }


def _commands(tmp_path, material):
    """Every cathar command line the pipeline issues for this material, the print's path as `<NP>`."""
    print_json = tmp_path / "noise.np.json"
    print_json.write_text("{}", encoding="utf-8")
    calls = []

    def record(cmd, _input, output, *_args, **_kwargs):
        calls.append([str(part).replace(str(print_json), NP) for part in cmd])
        return output

    with (
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._run_cathar_step", side_effect=record),
        patch("modules.cathar._cathar_noiseprint_step", return_value=print_json),
        patch("modules.cathar.log_msg"),
    ):
        cathar.filter_cathar_vhs_pipeline(tmp_path / "orig.wav", tmp_path, total_duration=600.0, strategy=_strategy(material))
    return calls


@pytest.mark.parametrize("material", ["speech", "music"])
def test_the_default_config_issues_heads_command_lines(tmp_path, material):
    """The repository's config.yaml builds HEAD 54048a9's cathar commands on both materials, byte for byte."""
    assert _commands(tmp_path, material) == HEAD_COMMANDS[material]


@pytest.mark.parametrize("material", ["speech", "music"])
def test_the_default_precondition_graph_is_heads(material):
    """The music profile's pre-conditioning graph (its CRT notch width) is untouched by round C3."""
    precond = processing._precondition_config_for("cathar", _strategy(material))
    assert _precondition_filter_from_config(precond) == PRECONDITION_HEAD[material]


def test_the_new_keys_default_to_what_music_ran_before():
    """0.02, on and 80 Hz: the speech keys' values, unset and as the repository's config.yaml sets them."""
    defaults = cfg._typed_config_defaults()
    assert tuple(defaults[key] for key in NEW_KEYS) == (0.02, True, 80)
    assert tuple(defaults[key] for key in NEW_KEYS) == tuple(defaults[key] for key in SPEECH_KEYS)
    assert (cfg.CATHAR_MUSIC_BETA, cfg.CATHAR_MUSIC_ENABLE_REPAIR, cfg.CATHAR_MUSIC_DEWIND_CUTOFF) == (0.02, True, 80)
    assert (cfg.CATHAR_BETA, cfg.CATHAR_ENABLE_REPAIR, cfg.CATHAR_DEWIND_CUTOFF) == (0.02, True, 80)


def _resolved(user):
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = user
    with patch.object(cfg, "yaml", mock_yaml), patch("builtins.open", MagicMock()), patch("builtins.print"):
        conf, _source = cfg.load_config()
    return tuple(conf[key] for key in NEW_KEYS)


def test_the_new_keys_coerce_and_are_bounded_like_their_speech_twins():
    """Quoted values are read; a negative floor or cutoff, a word and a None fall back to the defaults."""
    assert _resolved({"cathar_music_beta": "0.04", "cathar_music_enable_repair": "off", "cathar_music_dewind_cutoff": "40"}) == (
        0.04,
        False,
        40,
    )
    assert _resolved({"cathar_music_beta": -0.1, "cathar_music_enable_repair": "maybe", "cathar_music_dewind_cutoff": -60}) == (
        0.02,
        True,
        80,
    )
    assert _resolved({"cathar_music_beta": None, "cathar_music_enable_repair": None, "cathar_music_dewind_cutoff": True}) == (
        0.02,
        True,
        80,
    )


def _patched_profile(beta=0.04, repair=False, cutoff=40):
    return (
        patch("modules.cathar.CATHAR_MUSIC_BETA", beta),
        patch("modules.cathar.CATHAR_MUSIC_ENABLE_REPAIR", repair),
        patch("modules.cathar.CATHAR_MUSIC_DEWIND_CUTOFF", cutoff),
    )


def test_the_music_keys_move_the_music_commands_only(tmp_path):
    """Floor 0.04, no spike repair and a 40 Hz dewind on music; speech keeps HEAD's commands."""
    beta, repair, cutoff = _patched_profile()
    with beta, repair, cutoff:
        music = _commands(tmp_path, "music")
        speech = _commands(tmp_path, "speech")
    assert music[0] == ["dewind", "--cutoff", "40"]
    assert ["repair", "-s", "2"] not in music
    assert music[-1] == ["denoise", "--alpha", "0.5", "--beta", "0.04", "--coherent"]
    assert speech == HEAD_COMMANDS["speech"]


def test_the_speech_keys_no_longer_reach_music(tmp_path):
    """cathar_beta, cathar_enable_repair and cathar_dewind_cutoff move speech and leave music at its own keys."""
    with (
        patch("modules.cathar.CATHAR_BETA", 0.005),
        patch("modules.cathar.CATHAR_ENABLE_REPAIR", False),
        patch("modules.cathar.CATHAR_DEWIND_CUTOFF", 60),
    ):
        music = _commands(tmp_path, "music")
        speech = _commands(tmp_path, "speech")
    assert music == HEAD_COMMANDS["music"]
    assert speech[0] == ["dewind", "--cutoff", "60"]
    assert ["repair", "-s", "2"] not in speech
    assert ["denoise", "--alpha", "1.0", "--beta", "0.005", "--noiseprint", NP] in speech


def test_the_material_settings_carry_the_floor_the_repair_switch_and_the_cutoff():
    beta, repair, cutoff = _patched_profile()
    with beta, repair, cutoff, patch("modules.cathar.log_msg"):
        music = cathar._material_settings(_strategy("music"))
        speech = cathar._material_settings(_strategy("speech"))
    assert (music["beta"], music["repair"], music["dewind_cutoff"]) == (0.04, False, 40)
    assert (speech["beta"], speech["repair"], speech["dewind_cutoff"]) == (cathar.CATHAR_BETA, cathar.CATHAR_ENABLE_REPAIR, 80)


def test_the_music_log_names_the_floor_the_cutoff_and_the_repair_switch():
    beta, repair, cutoff = _patched_profile()
    with beta, repair, cutoff, patch("modules.cathar.log_msg") as log:
        cathar._material_settings(_strategy("music"))
    message = log.call_args.args[0]
    assert "subtracting at 0.5 (floor 0.04), dewind at 40 Hz" in message
    assert message.endswith("spike repair off.")


def test_the_split_band_passes_the_materials_floor_to_both_bands(tmp_path):
    """Each band's subtraction keeps the music floor, as the single pass does."""
    beta, repair, cutoff = _patched_profile(beta=0.01)
    with (
        beta,
        repair,
        cutoff,
        patch("modules.cathar.CATHAR_SPLIT_BAND_HZ", 6000),
        patch("modules.cathar.CATHAR_DENOISE_METHOD", "spectral"),
        patch("modules.split_band.recombine", side_effect=lambda low, high, target, hz: target),
    ):
        commands = _commands(tmp_path, "music")
    denoises = [cmd for cmd in commands if cmd[0] == "denoise"]
    assert [cmd[cmd.index("--beta") + 1] for cmd in denoises] == ["0.01", "0.01"]


def test_an_unset_material_value_reads_the_configured_default():
    """Called without the material's values (any other caller), the helpers give the configured speech defaults."""
    assert (cathar._dewind_cutoff(None), cathar._dewind_cutoff(60.0)) == (cathar.CATHAR_DEWIND_CUTOFF, 60)
    assert (cathar._repair_wanted(None), cathar._repair_wanted(0)) == (cathar.CATHAR_ENABLE_REPAIR, False)


def test_the_repair_pass_takes_the_materials_switch_over_the_default(tmp_path):
    in_wav = tmp_path / "in.wav"
    with (
        patch("modules.cathar._run_cathar_step", side_effect=lambda cmd, i, o, *a, **k: o),
        patch("modules.cathar.CATHAR_ENABLE_DECLIP", False),
        patch("modules.cathar.CATHAR_ENABLE_DEHUM", False),
        patch("modules.cathar.CATHAR_ENABLE_DEWOW", False),
        patch("modules.cathar.CATHAR_ENABLE_DEREVERB", False),
        patch("modules.cathar.CATHAR_ENABLE_REPAIR", True),
    ):
        assert cathar._cathar_repair_pass(in_wav, tmp_path).name == "repaired_in.wav"
        assert cathar._cathar_repair_pass(in_wav, tmp_path, repair=False) == in_wav
