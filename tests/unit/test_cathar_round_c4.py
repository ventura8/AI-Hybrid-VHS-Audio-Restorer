"""cathar round C4: declip on the scanner's clipping verdict, dehum at cathar's own mains detection.

Until 2026-10-10 the chain ran `declip` on every tape and gave `dehum --freq` the scanner's mains family (none
when the scan found no hum, `notch_freq` when the strategy carried no reading). The command lines below were
recorded from HEAD 549705a with the repository's config.yaml, the noise print path replaced by `<NP>`. cathar
is deterministic, so the same command lines on the same input are the same audio; the new keys' defaults must
rebuild them exactly. auto_pure_linear calls the same two steps (physical repair, tonal cleanup), so their
command lines are pinned too, and the new keys must not reach them.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

import modules.auto_scanner as auto_scanner
import modules.cathar as cathar
from modules import config as cfg
from modules import physical_repair, spectral_denoise
from modules.filters import DEAD_CHANNEL_DB

NP = "<NP>"
SHARED_HEAD = [
    ["dewind", "--cutoff", "80"],
    ["azimuth", "--max-ms", "5.0", "--method", "gcc-phat"],
    ["stereo", "--mono-below", "100"],
    ["declick", "--method", "ar", "--threshold", "8.0"],
    ["decrackle", "--sensitivity", "6"],
    ["inpaint", "--max-gap-ms", "50", "--iterations", "3"],
]
DECLIP = ["declip", "--method", "spade", "--threshold", "0.95"]
REPAIR = ["repair", "-s", "2"]
TAIL = {
    "speech": [
        ["denoise", "--alpha", "1.0", "--beta", "0.02", "--noiseprint", NP],
        ["deesser", "--bands", "3", "-f", "4000", "--threshold=12.0"],
    ],
    "music": [["denoise", "--alpha", "0.5", "--beta", "0.02", "--coherent"]],
}
PERSISTENCE = {"speech": 0.001, "music": 0.2}
UNSET = None


def _dehum(freq, adaptive=True):
    return ["dehum", "--freq", freq, "--harmonics", "8"] + (["--adaptive"] if adaptive else [])


def _chain(material, freq, declip=True):
    """A chain's command lines: the dehum family it is given (None: no dehum) and whether declip runs."""
    deplosive = [["deplosive", "-s", "4"]] if material == "speech" else []
    repair = ([DECLIP] if declip else []) + ([_dehum(freq)] if freq else []) + [REPAIR]
    return SHARED_HEAD + deplosive + repair + TAIL[material]


def _strategy(material, notch, clipping):
    """A scanned strategy; the clip (or none) shows on both readings, as a clip on both channels does."""
    filters = {"highpass_hz": 60, "crt_notch_hz": 15625.0}
    if notch is not UNSET:
        filters["notch_hz"] = notch
    profile = {"tonal_persistence": PERSISTENCE[material], "has_clipping": clipping, "has_clipping_any_channel": clipping}
    return {"precondition_filters": filters, "profile": profile}


def _recorder(calls, print_json=None):
    def record(cmd, _input, output, *_args, **_kwargs):
        calls.append([str(part).replace(str(print_json), NP) if print_json else str(part) for part in cmd])
        return output

    return record


def _commands(tmp_path, strategy, log=None):
    """Every cathar command line the pipeline issues for this strategy, the print's path as `<NP>`."""
    print_json = tmp_path / "noise.np.json"
    print_json.write_text("{}", encoding="utf-8")
    calls = []
    with (
        patch("modules.cathar._require_cathar_binary"),
        patch("modules.cathar._run_cathar_step", side_effect=_recorder(calls, print_json)),
        patch("modules.cathar._cathar_noiseprint_step", return_value=print_json),
        patch("modules.cathar.log_msg", log or MagicMock()),
    ):
        cathar.filter_cathar_vhs_pipeline(tmp_path / "orig.wav", tmp_path, total_duration=600.0, strategy=strategy)
    return calls


# (material, scanned notch_hz, scanner's clipping verdict, the dehum family HEAD 549705a issued)
HEAD_CASES = [
    ("speech", 50.0, False, "50"),
    ("speech", 50.0, True, "50"),
    ("speech", 60.0, True, "60"),
    ("speech", 100.0, False, "50"),
    ("speech", 120.0, False, "60"),
    ("speech", 0.0, False, None),
    ("speech", 0.0, True, None),
    ("speech", UNSET, False, "50"),
    ("music", 50.0, False, "50"),
    ("music", 0.0, True, None),
]


@pytest.mark.parametrize(("material", "notch", "clipping", "freq"), HEAD_CASES)
def test_the_default_config_issues_heads_command_lines(tmp_path, material, notch, clipping, freq):
    """The repository's config.yaml builds HEAD 549705a's commands: declip always, dehum at the scanned family or none."""
    assert _commands(tmp_path, _strategy(material, notch, clipping)) == _chain(material, freq)


def test_without_a_strategy_the_default_config_issues_heads_command_lines(tmp_path):
    """No strategy: the speech settings, declip, and dehum at the configured notch_freq (50 Hz)."""
    assert _commands(tmp_path, None) == _chain("speech", "50")


def test_the_new_keys_default_to_todays_behaviour():
    """The resolved config, its module constants and cathar's copies all hold every release's behaviour."""
    assert (cfg.CATHAR_DECLIP_MODE, cfg.CATHAR_DEHUM_FREQ) == ("always", "scanner")
    assert (cfg.CONFIG["cathar_declip_mode"], cfg.CONFIG["cathar_dehum_freq"]) == ("always", "scanner")
    assert (cathar.CATHAR_DECLIP_MODE, cathar.CATHAR_DEHUM_FREQ) == ("always", "scanner")


def _loaded(user):
    mock_yaml = MagicMock()
    mock_yaml.safe_load.return_value = user
    with (
        patch("modules.config.Path.exists", return_value=True),
        patch.object(cfg, "yaml", mock_yaml),
        patch("builtins.open", MagicMock()),
        patch("builtins.print") as warn,
    ):
        conf, _source = cfg.load_config()
    return conf, warn


def test_a_config_without_the_keys_resolves_todays_values_quietly():
    """An older config.yaml, without the C4 keys, gets every release's behaviour and no warning."""
    conf, warn = _loaded({"process_mode": "cathar"})
    assert (conf["cathar_declip_mode"], conf["cathar_dehum_freq"]) == ("always", "scanner")
    warn.assert_not_called()


@pytest.mark.parametrize(
    ("raw", "expected", "warned"),
    [
        (" Detected ", "detected", False),
        ("always", "always", False),
        ("sometimes", "always", True),
        (1, "always", True),
        (None, "always", True),
    ],
)
def test_the_declip_mode_is_a_choice(raw, expected, warned):
    """Case and spaces are forgiven, like cathar_denoise_method; anything else warns and keeps 'always'."""
    conf, warn = _loaded({"cathar_declip_mode": raw})
    assert conf["cathar_declip_mode"] == expected
    assert warn.called is warned


@pytest.mark.parametrize(
    ("raw", "expected", "warned"),
    [
        ("scanner", "scanner", False),
        (" Scanner ", "scanner", False),
        (0, 0, False),
        (0.0, 0, False),
        ("0", 0, False),
        (" 0.0 ", 0, False),
        (-0.0, 0, False),
        (1e-12, "scanner", True),
        ("nan", "scanner", True),
        (50, "scanner", True),
        (False, "scanner", True),
        (None, "scanner", True),
        ("auto", "scanner", True),
        ([0], "scanner", True),
    ],
)
def test_the_dehum_frequency_is_the_scanner_or_zero(raw, expected, warned):
    """0, quoted or not, is cathar's own detection; a bool is never 0, and anything else warns and keeps the scanner."""
    conf, warn = _loaded({"cathar_dehum_freq": raw})
    assert conf["cathar_dehum_freq"] == expected
    assert warn.called is warned


@pytest.mark.parametrize(
    ("strategy", "verdict"),
    [
        (_strategy("speech", 50.0, True), True),
        (_strategy("speech", 50.0, False), False),
        ({"profile": {"has_clipping": False, "has_clipping_any_channel": True}}, True),
        ({"profile": {"has_clipping": True}}, None),
        ({"profile": {"has_clipping": False}}, None),
        ({"profile": {"has_clipping_any_channel": 1}}, None),
        ({"profile": {"has_clipping_any_channel": "yes"}}, None),
        ({"profile": {"has_clipping_any_channel": None}}, None),
        ({"profile": None}, None),
        ({"precondition_filters": {}}, None),
        (None, None),
    ],
)
def test_the_clipping_verdict_is_the_channel_count_or_none(strategy, verdict):
    """The reading over every channel when it is a bool; no profile, no key or any other value is no verdict."""
    assert cathar._clipping_verdict(strategy) is verdict


@pytest.mark.parametrize(("material", "clipping"), [("speech", True), ("speech", False), ("music", True), ("music", False)])
def test_detected_runs_declip_only_on_a_tape_the_scanner_found_clipped(tmp_path, material, clipping):
    """Every other command line stays HEAD's."""
    with patch("modules.cathar.CATHAR_DECLIP_MODE", "detected"):
        commands = _commands(tmp_path, _strategy(material, 50.0, clipping))
    assert commands == _chain(material, "50", declip=clipping)


def test_detected_logs_the_skipped_declip():
    """A verdict of no clipping skips the stage with an INFO line naming why."""
    with patch("modules.cathar.CATHAR_DECLIP_MODE", "detected"), patch("modules.cathar.log_msg") as log:
        assert cathar._declip_wanted(_strategy("speech", 50.0, False)) is False
    assert "the scan found no clipping on any channel" in log.call_args.args[0]
    assert log.call_args.kwargs.get("level", "INFO") == "INFO"


def test_detected_declips_a_one_channel_clip_the_mono_mix_missed(tmp_path):
    """The live side of a dead-channel tape clips; the mono mix (has_clipping) halves it, the channel count does not."""
    strategy = _strategy("speech", 50.0, False)
    strategy["profile"]["has_clipping_any_channel"] = True
    with patch("modules.cathar.CATHAR_DECLIP_MODE", "detected"):
        assert _commands(tmp_path, strategy) == _chain("speech", "50")


# The strategy the scanner returns when it cannot read the audio: its profile carries no clipping reading.
FALLBACK_STRATEGY = auto_scanner._build_default_strategy()


def _logged(log, level):
    """The messages a mocked log_msg received at this level (log_msg's default is INFO)."""
    return [call.args[0] for call in log.call_args_list if call.kwargs.get("level", "INFO") == level]


@pytest.mark.parametrize(("strategy", "freq"), [(FALLBACK_STRATEGY, None), (None, "50")])
def test_detected_without_a_verdict_declips_as_always_and_warns(tmp_path, strategy, freq):
    """A failed scan, or a caller with no strategy, gets 'always''s declip and a WARNING, never a silent skip."""
    log = MagicMock()
    with patch("modules.cathar.CATHAR_DECLIP_MODE", "detected"):
        commands = _commands(tmp_path, strategy, log=log)
    assert commands == _chain("speech", freq)
    warnings = _logged(log, "WARNING")
    assert len(warnings) == 1
    assert "no clipping verdict in the scan; running Peak Saturation Declip as on 'always'" in warnings[0]
    assert "Skipping Peak Saturation Declip" not in " ".join(_logged(log, "INFO"))


def test_always_never_reads_the_verdict():
    """The default mode neither reads the scan nor logs: a strategy without a verdict declips quietly."""
    with patch("modules.cathar.log_msg") as log:
        assert cathar._declip_wanted(None) is True
        assert cathar._declip_wanted(_strategy("speech", 50.0, False)) is True
    log.assert_not_called()


@pytest.mark.parametrize(("mode", "clipping"), [("always", True), ("always", False), ("detected", True), ("detected", False)])
def test_the_switch_off_runs_no_declip_whatever_the_mode(mode, clipping):
    with patch("modules.cathar.CATHAR_DECLIP_MODE", mode), patch("modules.cathar.CATHAR_ENABLE_DECLIP", False):
        assert cathar._declip_wanted(_strategy("speech", 50.0, clipping)) is False


def test_the_repair_pass_keeps_the_configured_switch_without_the_chains_decision(tmp_path):
    """A direct caller passes no decision and gets cathar_enable_declip; the chain's decision wins when given."""
    in_wav = tmp_path / "in.wav"
    calls = []
    with patch("modules.cathar._run_cathar_step", side_effect=_recorder(calls)), patch("modules.cathar.CATHAR_DECLIP_MODE", "detected"):
        cathar._cathar_analog_repair_pass(in_wav, tmp_path, notch_freq=50.0)
        cathar._cathar_analog_repair_pass(in_wav, tmp_path, notch_freq=50.0, declip=False)
    assert calls == [DECLIP, _dehum("50"), REPAIR, _dehum("50"), REPAIR]


@pytest.mark.parametrize(("notch", "clipping"), [(50.0, False), (60.0, True), (0.0, False), (UNSET, False)])
def test_zero_lets_cathar_detect_the_mains_on_every_tape(tmp_path, notch, clipping):
    """`dehum --freq 0` whatever the scanner read, a tape it found no hum on included; every other line stays HEAD's."""
    with patch("modules.cathar.CATHAR_DEHUM_FREQ", 0):
        commands = _commands(tmp_path, _strategy("speech", notch, clipping))
    assert commands == _chain("speech", "0")


def test_zero_keeps_the_adaptive_switch(tmp_path):
    calls = []
    with patch("modules.cathar._run_cathar_step", side_effect=_recorder(calls)):
        cathar._cathar_dehum_step(tmp_path / "in.wav", tmp_path, freq=0.0, adaptive=False, autodetect=True)
    assert calls == [_dehum("0", adaptive=False)]


def test_the_switch_off_runs_no_dehum_at_zero(tmp_path):
    with patch("modules.cathar.CATHAR_DEHUM_FREQ", 0), patch("modules.cathar.CATHAR_ENABLE_DEHUM", False):
        commands = _commands(tmp_path, _strategy("speech", 50.0, False))
    assert commands == _chain("speech", None)


# freq -> the family HEAD 549705a's _cathar_dehum_step issued (None: the step returned its input).
HEAD_DEHUM_FAMILY = [
    (49.9, "50"),
    (50.0, "50"),
    (54.9, "50"),
    (55.0, "60"),
    (59.94, "60"),
    (60.0, "60"),
    (100.0, "50"),
    (120.0, "60"),
    (0.0, None),
    (None, None),
    (-1.0, None),
]


@pytest.mark.parametrize(("freq", "family"), HEAD_DEHUM_FAMILY)
@pytest.mark.parametrize("adaptive", [True, False])
def test_the_dehum_step_issues_heads_command_lines(tmp_path, freq, family, adaptive):
    in_wav = tmp_path / "in.wav"
    calls = []
    with patch("modules.cathar._run_cathar_step", side_effect=_recorder(calls)):
        result = cathar._cathar_dehum_step(in_wav, tmp_path, freq=freq, adaptive=adaptive, harmonics=8)
    assert calls == ([_dehum(family, adaptive)] if family else [])
    assert (result == in_wav) is (family is None)


@pytest.mark.parametrize("freq", [None, 0.0, 50.0, 120.0])
def test_autodetect_ignores_the_frequency_it_is_given(tmp_path, freq):
    calls = []
    with patch("modules.cathar._run_cathar_step", side_effect=_recorder(calls)):
        cathar._cathar_dehum_step(tmp_path / "in.wav", tmp_path, freq=freq, autodetect=True)
    assert calls == [_dehum("0")]


def test_auto_pure_linears_declip_and_dehum_ignore_the_c4_keys(tmp_path):
    """The physical repair's declip (its own has_clipping plan) and the tonal cleanup's dehum issue HEAD's command lines."""
    in_wav = tmp_path / "in.wav"
    calls = []
    with (
        patch("modules.cathar._run_cathar_step", side_effect=_recorder(calls)),
        patch("modules.cathar.CATHAR_DECLIP_MODE", "detected"),
        patch("modules.cathar.CATHAR_DEHUM_FREQ", 0),
    ):
        physical_repair._run_stages(in_wav, tmp_path, {"declip": True}, None)
        spectral_denoise._dehum(in_wav, tmp_path, 50.0, None)
        spectral_denoise._dehum(in_wav, tmp_path, 120.0, None)
    assert calls == [DECLIP, _dehum("50"), _dehum("60")]


RATE = 48000


def _capture(left, right):
    """Three seconds of 440 Hz at the two channel gains, hard-clipped at full scale (0 gain: a dead channel)."""
    tone = np.sin(2.0 * np.pi * 440.0 * np.arange(RATE * 3) / RATE)
    return np.column_stack([np.clip(left * tone, -1.0, 1.0), np.clip(right * tone, -1.0, 1.0)])


def test_the_scan_flags_the_clipped_live_side_of_a_dead_channel_tape(tmp_path):
    """The mono mix halves the live side (has_clipping False); the channel count flags it, and 'detected' declips.

    The pre-conditioning mirrors the live side to full scale (balance past DEAD_CHANNEL_DB) and keeps its
    adeclip off, as before: has_clipping and every pre-conditioning setting are unchanged.
    """
    wav = tmp_path / "dead_right.wav"
    stereo = _capture(1.5, 0.0)
    stereo[:, 1] = 1e-5 * np.random.default_rng(3).standard_normal(len(stereo))
    sf.write(str(wav), stereo, RATE, subtype="FLOAT")
    with patch("modules.auto_scanner.log_msg"):
        strategy = auto_scanner.scan_and_decide_restoration_strategy(wav, executed_mode="cathar")
    profile, precondition = strategy["profile"], strategy["precondition_filters"]
    assert (profile["has_clipping"], profile["has_clipping_any_channel"]) == (False, True)
    assert precondition["balance_db"] > DEAD_CHANNEL_DB
    assert precondition["enable_adeclip"] is False
    with patch("modules.cathar.CATHAR_DECLIP_MODE", "detected"), patch("modules.cathar.log_msg"):
        assert cathar._declip_wanted(strategy) is True


@pytest.mark.parametrize(
    ("left", "right", "mono", "any_channel"),
    [(1.5, 1.5, True, True), (1.5, 0.0, False, True), (0.9, 0.9, False, False), (0.9, 0.0, False, False)],
)
def test_the_channel_count_never_misses_what_the_mono_mix_flags(left, right, mono, any_channel):
    """Both channels clipped: both readings flag it; only the live side clipped: only the channel count does."""
    capture = _capture(left, right)
    mono_mix = capture.mean(axis=1)
    profile = auto_scanner._extract_profile_from_signal(mono_mix, RATE, stereo_signal=capture)
    assert (profile["has_clipping"], profile["has_clipping_any_channel"]) == (mono, any_channel)
    assert isinstance(profile["has_clipping_any_channel"], bool)


@pytest.mark.parametrize("clipped", [True, False])
def test_without_the_channels_the_mono_reading_stands(clipped):
    """A mono capture reads the same samples both ways; a caller passing no channels gets the mono verdict."""
    mono = _capture(1.5 if clipped else 0.9, 0.0)[:, 0]
    assert auto_scanner._detect_clipping_any_channel(mono, clipped) is clipped
    assert auto_scanner._detect_clipping_any_channel(None, clipped) is clipped


def test_auto_pure_linears_declip_plan_keeps_the_mono_mix_reading(tmp_path):
    """physical_repair still reads has_clipping: the channel count reaches only cathar's 'detected' gate."""
    profile = {"has_clipping": False, "has_clipping_any_channel": True}
    with patch("modules.physical_repair.detect_mute_spans", return_value=0):
        assert physical_repair._stage_plan(tmp_path / "in.wav", {"profile": profile})["declip"] is False
