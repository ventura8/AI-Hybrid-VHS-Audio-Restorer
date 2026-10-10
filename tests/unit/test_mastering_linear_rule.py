"""The mux's loudness stage: ffmpeg's linear rule, the gain_limiter fallback, and the record of what loudnorm did.

The defaults must leave every graph the mux runs as it was, string for string; the
fallback and the record exist only behind `loudnorm_linear_fallback` and the event log.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import modules.config as cfg
from modules import mastering

TARGET = "I=-16.0:TP=-1.0:LRA=20.0"


def _measured(i, tp, lra, thresh="-30.00"):
    """An analysis block as loudnorm prints it: decimal strings."""
    return {"input_i": i, "input_tp": tp, "input_lra": lra, "input_thresh": thresh, "target_offset": "0.10"}


# Readings ffmpeg 8.0.1 printed for synthetic programmes, with the normalization_type its
# applied pass then reported (scratch check of the rule before this module used it).
FFMPEG_DECISIONS = [
    ("-29.08", "-19.95", "0.20", 20.0, "linear"),
    ("-29.09", "-0.92", "0.20", 20.0, "dynamic"),
    ("-41.12", "-10.46", "0.20", 20.0, "dynamic"),
    ("-21.11", "-12.38", "0.20", 20.0, "linear"),
    ("-29.06", "-20.04", "0.30", 1.0, "linear"),
    ("-41.11", "-26.02", "0.20", 20.0, "dynamic"),
    ("-21.72", "-11.97", "10.50", 20.0, "linear"),
    ("-21.70", "-11.44", "10.50", 5.0, "dynamic"),
    ("-20.75", "-20.00", "0.00", 20.0, "dynamic"),
]


@pytest.mark.parametrize("i, tp, lra, target_lra, ffmpeg_mode", FFMPEG_DECISIONS)
def test_the_rule_reproduces_ffmpegs_own_decisions(i, tp, lra, target_lra, ffmpeg_mode):
    """Each programme's mode by the rule is the one ffmpeg reported."""
    blockers = mastering.linear_mode_blockers(_measured(i, tp, lra), target_lra)
    assert ("dynamic" if blockers else "linear") == ffmpeg_mode


@pytest.mark.parametrize(
    "measurements, expected",
    [
        (_measured("-20.00", "-5.00", "6.00"), ()),
        (_measured("-20.00", "-4.99", "6.00"), (mastering.BLOCKER_TRUE_PEAK,)),
        (_measured("-20.00", "-3.00", "25.00"), (mastering.BLOCKER_LRA, mastering.BLOCKER_TRUE_PEAK)),
        (_measured("-20.00", "-9.00", "25.00"), (mastering.BLOCKER_LRA,)),
        (_measured("-20.00", "-9.00", "6.00", thresh="-70.00"), (mastering.BLOCKER_UNREADABLE,)),
        (_measured("0.00", "-9.00", "6.00"), (mastering.BLOCKER_UNREADABLE,)),
        (_measured("-20.00", "99.00", "6.00"), (mastering.BLOCKER_UNREADABLE,)),
        ({"input_i": "-20.00", "input_tp": "-9.00", "input_lra": "nan"}, (mastering.BLOCKER_UNREADABLE,)),
    ],
)
def test_each_half_of_the_rule_and_the_unset_values(measurements, expected):
    """The true peak after a +4 dB gain sits on the -1 dBTP target at -5.00 (linear) and over it at -4.99."""
    assert mastering.linear_mode_blockers(measurements, 20.0) == expected


# A programme under loudnorm's 3 s frame buffer measures LRA 0 (the unset value) and runs linear anyway.
SHORT_CLIP = _measured("-27.66", "-12.88", "0.00", thresh="-38.26")


@pytest.mark.parametrize(
    "measurements, duration, expected",
    [
        (SHORT_CLIP, 2.9, ()),
        (SHORT_CLIP, 2.99, ()),
        (_measured("-20.00", "-3.00", "25.00"), 2.5, ()),
        (SHORT_CLIP, 3.0, (mastering.BLOCKER_UNREADABLE,)),
        (SHORT_CLIP, 3.1, (mastering.BLOCKER_UNREADABLE,)),
        (_measured("-20.00", "-3.00", "6.00"), 3.0, (mastering.BLOCKER_TRUE_PEAK,)),
        (SHORT_CLIP, None, (mastering.BLOCKER_UNREADABLE,)),
        (SHORT_CLIP, 0.0, (mastering.BLOCKER_UNREADABLE,)),
        (SHORT_CLIP, -1.0, (mastering.BLOCKER_UNREADABLE,)),
    ],
)
def test_a_programme_under_three_seconds_runs_linear_whatever_was_measured(measurements, duration, expected):
    """ffmpeg forced linear at 2.9 s and ran dynamic from 3.0 s; an unknown or non-positive duration keeps the measured rule."""
    assert mastering.linear_mode_blockers(measurements, 20.0, duration) == expected


def _logged_mode(measurements, duration=None):
    with patch("modules.mastering.log_msg") as log, patch.object(cfg, "LOUDNORM_TARGET_LRA", 20.0):
        blockers = mastering._log_loudness_range(measurements, duration)
    return blockers, log.call_args[0][0]


def test_the_log_says_why_a_short_programme_is_linear():
    """The unreadable LRA of a 2.5 s clip does not make the log call it dynamic."""
    blockers, message = _logged_mode(SHORT_CLIP, 2.5)
    assert blockers == ()
    assert message.endswith("-> linear (programme under 3 s)")


def test_a_long_programme_logs_plain_linear():
    """The note is for the short case only."""
    _blockers, message = _logged_mode(_measured("-20.00", "-9.00", "6.00"), 120.0)
    assert message.endswith("-> linear")


def test_the_log_names_the_true_peak_when_it_alone_rules_out_linear():
    """The case the range-only log used to call linear."""
    blockers, message = _logged_mode(_measured("-20.00", "-3.00", "6.00"))
    assert blockers == (mastering.BLOCKER_TRUE_PEAK,)
    assert message.endswith("-> dynamic (true peak after the gain above the target)")


def test_the_log_names_both_halves_when_both_fail():
    """Both reasons, in the rule's order."""
    _blockers, message = _logged_mode(_measured("-20.00", "-3.00", "25.00"))
    assert "measured LRA above the target; true peak after the gain above the target" in message


def _applied(measurements, fallback):
    with patch.object(cfg, "LOUDNORM_LINEAR_FALLBACK", fallback), patch("modules.mastering.log_msg"):
        return mastering._applied_loudness(measurements, mastering.linear_mode_blockers(measurements, 20.0))


@pytest.mark.parametrize(
    "measurements, fallback",
    [
        (_measured("-20.00", "-3.00", "6.00"), "ffmpeg"),
        (_measured("-20.00", "-3.00", "25.00"), "gain_limiter"),
        (_measured("-20.00", "-9.00", "6.00"), "gain_limiter"),
        (_measured("-20.00", "-9.00", "6.00", thresh="-70.00"), "gain_limiter"),
    ],
)
def test_loudnorm_keeps_every_case_but_the_true_peak_alone_under_the_fallback(measurements, fallback):
    """The default, a range over target, a linear programme and an unreadable one all get today's arguments."""
    assert _applied(measurements, fallback) == mastering._measured_loudnorm_args(measurements)


def test_the_fallback_takes_one_gain_and_the_limiter_for_the_true_peak_case():
    """volume to the target, then the shared limiter."""
    applied = _applied(_measured("-22.15", "-3.00", "6.00"), "gain_limiter")
    assert applied == mastering.LinearGain(6.15)
    chain = mastering._mastering_chain(applied, "mixed")
    assert chain == f"volume=6.15dB,aresample=44100,{mastering.LOUDNORM_TRUE_PEAK_LIMITER}[mixed]"


def _resolved(resolver, reports):
    """Runs a resolver with the analysis pass mocked; returns (result, the graphs it was asked to run)."""
    with (
        patch("modules.mastering._run_loudness_analysis", side_effect=reports) as run,
        patch.object(cfg, "ENABLE_LOUDNORM", True),
        patch.object(cfg, "LOUDNORM_TARGET_LRA", 20.0),
        patch("modules.mastering.log_msg"),
    ):
        result = resolver()
    return result, [call.args[1] for call in run.call_args_list]


def test_the_single_track_measurement_runs_todays_graph_once_by_default(monkeypatch):
    """Log off: one analysis pass, the same graph and arguments as before."""
    monkeypatch.delenv("AI_RESTORE_EVENT_LOG", raising=False)
    measurements = _measured("-20.00", "-3.00", "6.00")
    result, graphs = _resolved(lambda: mastering._resolve_single_track_loudnorm_args("v.mp4", "a.wav"), [measurements])
    assert graphs == [f"[1:a]loudnorm={TARGET}:print_format=json[mastered]"]
    assert result == mastering._measured_loudnorm_args(measurements)


def test_the_mix_measurement_runs_todays_graph_once_by_default(monkeypatch):
    """Log off: the mix is measured with the same graph as before."""
    monkeypatch.delenv("AI_RESTORE_EVENT_LOG", raising=False)
    _result, graphs = _resolved(
        lambda: mastering._resolve_loudnorm_args("v.mp4", "voc.wav", "bg.wav", None, None), [_measured("-20.00", "-9.00", "6.00")]
    )
    assert graphs == [f"{mastering._build_mix_base_expression(None, None)},loudnorm={TARGET}:print_format=json[mastered]"]


def _work_track(tmp_path):
    """A processed track inside a tape's work directory."""
    track = tmp_path / ".temp_work_Tape" / "apl" / "processed.wav"
    track.parent.mkdir(parents=True, exist_ok=True)
    return track


def _read(path):
    """A written record, parsed."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _recorded(tmp_path, monkeypatch, reports, fallback="ffmpeg", mode="auto_pure_linear", duration=None):
    """Resolves a single track with the event log on; returns (result, graphs run, sidecar, event record)."""
    monkeypatch.setenv("AI_RESTORE_EVENT_LOG", str(tmp_path / "events"))
    monkeypatch.setattr(cfg, "PROCESS_MODE", mode)
    track = _work_track(tmp_path)
    with patch.object(cfg, "LOUDNORM_LINEAR_FALLBACK", fallback):
        result, graphs = _resolved(lambda: mastering._resolve_single_track_loudnorm_args("v.mp4", track, total_duration=duration), reports)
    sidecar = _read(track.parent / f"loudnorm__Tape__{mode}__processed.json")
    record = _read(tmp_path / "events" / f"loudnorm__Tape__{mode}__processed.json")
    return result, graphs, sidecar, record


def test_the_record_carries_ffmpegs_own_type_from_the_applied_pass(tmp_path, monkeypatch):
    """The second pass runs the applied arguments; the record holds its report."""
    measurements = _measured("-20.00", "-3.00", "6.00")
    report = {**measurements, "normalization_type": "dynamic", "output_i": "-16.01", "output_tp": "-1.00"}
    result, graphs, sidecar, record = _recorded(tmp_path, monkeypatch, [measurements, report])
    assert graphs[1] == f"[1:a]loudnorm={result}:print_format=json[mastered]"
    assert (sidecar["normalization_type"], sidecar["predicted"], sidecar["blockers"]) == ("dynamic", "dynamic", ["true_peak"])
    assert (record["recording"], record["output_i"], record["measured"]["input_tp"]) == ("Tape", "-16.01", "-3.00")


def test_the_mux_duration_reaches_the_rule_so_a_short_programme_stays_with_loudnorm(tmp_path, monkeypatch):
    """A 2.5 s track the true peak alone would send to the fallback: ffmpeg runs it linear, so the record agrees and no warning."""
    measurements = _measured("-20.00", "-3.00", "6.00")
    with patch("modules.mastering._check_rule") as check:
        result, _graphs, sidecar, _record = _recorded(
            tmp_path, monkeypatch, [measurements, {"normalization_type": "linear"}], "gain_limiter", duration=2.5
        )
    assert result == mastering._measured_loudnorm_args(measurements)
    assert (sidecar["normalization_type"], sidecar["predicted"], sidecar["blockers"]) == ("linear", "linear", [])
    check.assert_called_once_with("linear", "linear")


def test_each_modes_run_on_one_tape_keeps_its_own_record(tmp_path, monkeypatch):
    """Two modes share the tape's work folder; the sidecar and the event record carry the mode in their names and headers."""
    for mode in ("auto_pure_linear", "cathar"):
        _recorded(tmp_path, monkeypatch, [_measured("-20.00", "-9.00", "6.00"), {"normalization_type": "linear"}], mode=mode)
    sidecars = [_read(path) for path in sorted(_work_track(tmp_path).parent.glob("loudnorm__Tape__*__processed.json"))]
    assert [sidecar["mode"] for sidecar in sidecars] == ["auto_pure_linear", "cathar"]
    assert len(list((tmp_path / "events").glob("loudnorm__Tape__*__processed.json"))) == 2


def _render_report(tp):
    """loudnorm's analysis of the fallback's render: its input readings are the render's output."""
    return {"input_i": "-16.03", "input_tp": tp, "input_lra": "1.10", "input_thresh": "-26.20", "target_offset": "0.03"}


def test_the_fallback_records_what_its_render_peaks_at(tmp_path, monkeypatch):
    """The second pass reads the gain, the resample and the limiter through loudnorm's analysis; the record holds the output."""
    reports = [_measured("-20.00", "-3.00", "6.00"), _render_report("0.24")]
    result, graphs, sidecar, _record = _recorded(tmp_path, monkeypatch, reports, "gain_limiter")
    assert result == mastering.LinearGain(4.0)
    render = f"volume=4.00dB,aresample=44100,{mastering.LOUDNORM_TRUE_PEAK_LIMITER}"
    assert graphs[1] == f"[1:a]{render},loudnorm={TARGET}:print_format=json[mastered]"
    assert (sidecar["normalization_type"], sidecar["gain_db"], sidecar["fallback"]) == ("gain_limiter", 4.0, "gain_limiter")
    assert (sidecar["output_i"], sidecar["output_tp"], sidecar["output_lra"]) == ("-16.03", "0.24", "1.10")


def test_an_unreadable_fallback_render_records_its_gain_alone(tmp_path, monkeypatch):
    """No report: the type and the gain, the readings empty."""
    _result, _graphs, sidecar, _record = _recorded(tmp_path, monkeypatch, [_measured("-20.00", "-3.00", "6.00"), None], "gain_limiter")
    assert (sidecar["normalization_type"], sidecar["gain_db"]) == ("gain_limiter", 4.0)
    assert sidecar["output_tp"] is None


@pytest.mark.parametrize("output_tp, warned", [("0.24", True), ("-0.99", True), ("-1.00", False), ("-1.90", False), (None, False)])
def test_a_render_over_the_true_peak_target_is_warned(output_tp, warned):
    """The sample-peak limiter can leave intersample overs; the log says so, and is quiet at or under -1 dBTP."""
    with patch("modules.mastering.log_msg") as log:
        mastering._check_true_peak(output_tp)
    assert log.called is warned


def test_an_unreadable_applied_pass_records_no_type(tmp_path, monkeypatch):
    """A failed report leaves the type empty and keeps the prediction."""
    _result, _graphs, sidecar, _record = _recorded(tmp_path, monkeypatch, [_measured("-20.00", "-9.00", "6.00"), None])
    assert sidecar["normalization_type"] is None
    assert sidecar["predicted"] == "linear"


@pytest.mark.parametrize("actual, warned", [("dynamic", True), ("linear", False), (None, False), ("gain_limiter", False)])
def test_a_decision_the_rule_did_not_predict_is_logged(actual, warned):
    """Only a real disagreement warns."""
    with patch("modules.mastering.log_msg") as log:
        mastering._check_rule("linear", actual)
    assert log.called is warned


def test_the_loudness_stage_reads_either_kind():
    """A string is loudnorm's arguments; a LinearGain is a volume filter."""
    assert mastering._loudness_stage("I=-16.0") == "loudnorm=I=-16.0"
    assert mastering._loudness_stage(mastering.LinearGain(-2.5)) == "volume=-2.50dB"


def test_both_mux_graphs_render_the_fallbacks_gain_before_the_limiter():
    """The single-track and the mix graphs carry the LinearGain as a volume stage, ahead of the shared tail."""
    tail = f"aresample=44100,{mastering.LOUDNORM_TRUE_PEAK_LIMITER}"
    with patch.object(cfg, "ENABLE_LOUDNORM", True):
        single = mastering._build_single_audio_filter_expression(mastering.LinearGain(3.0))
        mix = mastering._build_mix_filter_expression(loudnorm_args=mastering.LinearGain(-2.5))
    assert single == f"[1:a]volume=3.00dB,{tail}[mastered]"
    assert mix == f"{mastering._build_mix_base_expression(None, None)},volume=-2.50dB,{tail}[mixed]"
