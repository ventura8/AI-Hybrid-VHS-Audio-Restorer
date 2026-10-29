"""The listening command line since 2026-10-12: playback state, one speech level, the protocol, the raw catch, the threshold run."""

import argparse

import numpy as np
import pytest
import soundfile as sf

from scripts import listen_ab
from scripts.restoration_quality import ledger, listen_cuts, listen_protocol, playback_calibration
from tests.unit.test_listen_ab_cli import PLAYBACK, RATE, _answering_serve, _args, _recorded_block, _voice_file


def _protocol_counts(tmp_path, mode, *extra):
    """`(practice, catch)` presentations the protocol builds for `mode` from the command line."""
    rng = np.random.default_rng(6)
    args = _args(tmp_path, *extra, mode=mode)
    protocol = listen_ab.build_protocol(args, listen_ab.build_mode(args, ["a", "b"], rng), rng, ("catch:air",))
    return protocol.order.count("familiar"), protocol.order.count("catch")


def test_the_protocol_takes_each_modes_defaults_or_the_command_line(tmp_path):
    """abx: 2 practice trials and one catch per 8; pair: none to practise on; counts on the command line win."""
    counts = [
        _protocol_counts(tmp_path, "abx"),
        _protocol_counts(tmp_path, "pair"),
        _protocol_counts(tmp_path, "abx", "--catch", "0", "--familiar", "1"),
    ]
    assert counts == [(2, 3), (0, 1), (1, 0)]


@pytest.mark.parametrize("argv", [["threshold", "--volume", "40"], ["threshold", "--device", "hp"]])
def test_device_and_volume_are_required(argv):
    """Without --device or --volume the command line is refused before anything runs."""
    with pytest.raises(SystemExit):
        listen_ab.parse_args(argv)


def test_the_playback_record_keeps_the_whole_chain_as_text(tmp_path):
    """Device, volume, connection, EQ, noise cancelling and surround, as given (null when not given)."""
    args = _args(tmp_path, "--connection", "USB 24/48", "--eq", "off", "--anc", "off")
    assert listen_ab.playback_of(args) == {
        "device": "SteelSeries Arctis Nova Pro Wireless",
        "volume": "40",
        "connection": "USB 24/48",
        "eq": "off",
        "anc": "off",
        "surround": None,
    }


@pytest.mark.parametrize(
    "extra, fragment",
    [
        (["--stim", "PLACEHOLDER"], "threshold takes no --stim, --anchor or --catch-raw"),
        (["--start", "3"], "threshold takes no --start"),
        (["--trials", "10"], "--trials must be at least 15 for threshold"),
    ],
)
def test_threshold_argument_problems(tmp_path, extra, fragment):
    """The threshold run takes no stimuli, no start, and at least three trials per frequency."""
    sf.write(str(tmp_path / "a.wav"), np.zeros(RATE, dtype=np.float32), RATE)
    extra = [f"a={tmp_path / 'a.wav'}" if item == "PLACEHOLDER" else item for item in extra]
    args = listen_ab.parse_args(["threshold", *PLAYBACK, *extra])
    assert fragment in "; ".join(listen_ab.argument_problems(args))


def test_a_comparison_needs_a_tape_and_the_threshold_run_does_not(tmp_path):
    """abx without --tape is refused; the threshold run needs none."""
    sf.write(str(tmp_path / "a.wav"), np.zeros(RATE, dtype=np.float32), RATE)
    stims = ["--stim", f"a={tmp_path / 'a.wav'}", "--stim", f"b={tmp_path / 'a.wav'}"]
    assert "abx needs --tape" in listen_ab.argument_problems(listen_ab.parse_args(["abx", *stims, *PLAYBACK]))
    assert not listen_ab.argument_problems(listen_ab.parse_args(["threshold", *PLAYBACK]))


def test_a_listener_is_a_slug():
    """The threshold run's listener is a slug, as the calibration file keys it."""
    assert listen_ab.listener_arg("user") == "user"
    with pytest.raises(argparse.ArgumentTypeError):
        listen_ab.listener_arg("two words")


def test_main_records_the_cut_window_and_the_level(tmp_path, monkeypatch):
    """The recorded window spans the asked 8 s; both voices were brought to -23 dBFS and the record says how."""
    _code, record, _first = _recorded_block(tmp_path, monkeypatch)
    context = record["context"]
    span = record["windows"][0][1] - record["windows"][0][0]
    assert (span, context["target_dbfs"]) == (pytest.approx(8.0), listen_cuts.TARGET_ACTIVE_DBFS)
    assert context["gains_db"]["a"] == pytest.approx(context["target_dbfs"] - context["levels_dbfs"]["a"], abs=0.02)
    assert (max(context["peaks_dbfs"].values()) < listen_cuts.PEAK_MAX_DBFS, set(context["catch_gains_db"])) == (True, {"catch:air"})
    assert len(context["loops"]["pause"]) == 2


def test_main_records_the_playback_and_the_protocol(tmp_path, monkeypatch):
    """The record keeps the device and volume, two practice trials, two catch trials caught, no flag, and the power."""
    _code, record, _first = _recorded_block(tmp_path, monkeypatch)
    context = record["context"]
    assert (record["playback"]["device"], record["playback"]["volume"]) == ("SteelSeries Arctis Nova Pro Wireless", "40")
    assert (context["familiarisation"], context["catch"]["caught"], context["catch"]["missed"]) == ({"trials": 2, "right": 2}, 2, 0)
    assert (context["block_flags"], context["power"]["critical_correct"]) == ([], 12)


def test_levelling_problem_names_silence_then_clipping():
    """Silent cuts are named first; cuts that would clip at the target name their peak and where every stimulus fits."""
    clean = {"silent": [], "unlevelled": [], "target_dbfs": -23.0, "peaks_dbfs": {"a": -3.0}, "fits_at_dbfs": -20.2}
    assert listen_ab.levelling_problem(clean, 80.0) == ""
    assert listen_ab.levelling_problem({**clean, "silent": ["a"]}, 80.0).startswith("silent at 80.0 s")
    clipping = {**clean, "unlevelled": ["tape"], "peaks_dbfs": {"tape": 1.5}, "fits_at_dbfs": -24.7}
    message = listen_ab.levelling_problem(clipping, 80.0)
    assert (
        message
        == "cannot be levelled at -23 dBFS: tape peaks at +1.5 dBFS; every stimulus fits at --level-dbfs -24.7 or below, or cut elsewhere"
    )


def test_main_refuses_a_stimulus_that_would_clip_at_the_level(tmp_path, monkeypatch):
    """A quiet voice with a loud click cannot be brought to -23 dBFS without clipping: refused before it is served."""
    monkeypatch.setattr(listen_ab, "serve", lambda *_args, **_kwargs: pytest.fail("an unlevelled block was served"))
    first = _voice_file(tmp_path / "a.wav")
    audio, _rate = sf.read(str(first), dtype="float32")
    clicked = (0.05 * audio).astype(np.float32)
    clicked[3 * RATE] = 0.9
    sf.write(str(tmp_path / "b.wav"), clicked, RATE, subtype="FLOAT")
    argv = ["abx", "--tape", "vaccin", "--stim", f"a={first}", "--stim", f"b={tmp_path / 'b.wav'}", "--start", "1", *PLAYBACK]
    with pytest.raises(SystemExit, match="cannot be levelled at -23 dBFS: b peaks at"):
        listen_ab.main([*argv, "--ledger", str(tmp_path / "v.jsonl"), "--no-browser"])


def test_main_with_a_raw_tape_adds_the_half_raw_catch(tmp_path, monkeypatch):
    """--catch-raw: the raw tape is a file of the record but not a stimulus, and its half-raw catch plays among the catch trials."""
    first, second, raw = _voice_file(tmp_path / "a.wav"), _voice_file(tmp_path / "b.wav"), _voice_file(tmp_path / "raw.wav")
    monkeypatch.setattr(listen_ab, "serve", _answering_serve)
    argv = ["abx", "--tape", "vaccin", "--stim", f"a={first}", "--stim", f"b={second}", "--catch-raw", str(raw), "--start", "1"]
    argv += ["--trials", "16", "--catch", "4", "--port", "9100", "--no-browser", "--ledger", str(tmp_path / "v.jsonl"), *PLAYBACK]
    assert listen_ab.main(argv) == 0
    record = ledger.read(tmp_path / "v.jsonl")[0]
    labels = (list(map(_label, record["files"])), record["question"]["stimuli"], set(map(_label, record["context"]["catch"]["trials"])))
    assert labels == (["a", "b", "catch_raw"], ["a", "b"], {"catch:air", "catch:raw"})


def _label(entry):
    """A file entry's label, or a catch trial's control (its first element)."""
    return entry["label"] if isinstance(entry, dict) else entry[0]


def test_main_runs_a_threshold_block_into_the_calibration_file(tmp_path, monkeypatch, capsys):
    """The threshold run end to end with the stand-in server: one calibration record under the listener's name."""
    monkeypatch.setattr(listen_ab, "serve", _answering_serve)
    argv = ["threshold", "--listener", "user", "--trials", "15", "--calibration", str(tmp_path / "cal.jsonl"), *PLAYBACK]
    assert listen_ab.main([*argv, "--port", "9100", "--no-browser", "--seed", "3", "--round", "session1"]) == 0
    record = playback_calibration.read(tmp_path / "cal.jsonl")[0]
    practice = record["context"]["familiarisation"]["trials"]
    assert (record["listener"], record["round"], "-threshold-user-" in record["id"]) == ("user", "session1", True)
    assert (practice, "thresholds: 1 kHz" in capsys.readouterr().out) == (listen_protocol.FAMILIAR_TRIALS["threshold"], True)
