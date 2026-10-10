"""The audibility CLI and the file-level check behind it: the PCM hash, the null test and the NMR, every path confined."""

import builtins
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts import audibility_check, cli_paths
from scripts.restoration_quality import auditory

RATE = 44100
SCRIPT = Path(audibility_check.__file__)


def _tone_bursts(seconds=3.0, seed=5):
    """A 440 Hz tone in 0.25 s bursts over a faint hiss."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    gate = ((t % 0.5) < 0.25).astype(np.float64)
    return (0.1 * np.sin(2 * np.pi * 440.0 * t) * gate + 1e-4 * rng.standard_normal(len(t))).astype(np.float32)


def _write(path, audio, rate=RATE, subtype="FLOAT"):
    sf.write(str(path), audio, rate, subtype=subtype)
    return path


def _pair(tmp_path, candidate=None, candidate_rate=RATE):
    """Incumbent and candidate WAVs under `tmp_path`; the candidate defaults to the incumbent's audio."""
    voice = _tone_bursts()
    incumbent = _write(tmp_path / "incumbent.wav", voice)
    return incumbent, _write(tmp_path / "candidate.wav", voice if candidate is None else candidate, candidate_rate)


def test_the_audio_hash_reads_the_decoded_pcm_not_the_container(tmp_path):
    """The same samples in a 32-bit and a 64-bit float WAV hash alike; one nudged sample or another rate does not."""
    voice = _tone_bursts(seconds=1.0)
    as_float = _write(tmp_path / "a.wav", voice)
    nudged = voice.copy()
    nudged[100] += np.float32(1e-3)
    assert auditory.audio_sha256(as_float) == auditory.audio_sha256(_write(tmp_path / "b.wav", voice, subtype="DOUBLE"))
    assert auditory.audio_sha256(as_float) != auditory.audio_sha256(_write(tmp_path / "c.wav", nudged))
    assert auditory.pcm_sha256(voice, RATE) != auditory.pcm_sha256(voice, 48000)


def test_compare_files_marks_audio_identical_across_containers(tmp_path):
    """Equal PCM is `identical` and skips the analysis: no lag, no difference, nothing audible."""
    incumbent, _candidate = _pair(tmp_path)
    same = auditory.compare_files(incumbent, _write(tmp_path / "double.wav", _tone_bursts(), subtype="DOUBLE"))
    assert (same["identical"], same["audible"], same["lag"]) == (True, False, 0)
    assert same["incumbent_sha256"] == same["candidate_sha256"]


def test_identical_files_read_identical_and_the_json_is_written(tmp_path, capsys):
    """The readings are printed and, with --json, written into a directory created on the way."""
    incumbent, candidate = _pair(tmp_path)
    target = tmp_path / "reports" / "audibility.json"
    result = audibility_check.main([str(incumbent), str(candidate), "--json", str(target)])
    assert result["identical"] is True
    assert json.loads(target.read_text(encoding="utf-8")) == result
    assert '"identical": true' in capsys.readouterr().out


def test_a_louder_candidate_reads_audible(tmp_path):
    """+3.5 dB on every sample is no tie, and the alignment finds no lag or flip."""
    incumbent, candidate = _pair(tmp_path, _tone_bursts() * np.float32(1.5))
    result = audibility_check.main([str(incumbent), str(candidate)])
    assert (result["identical"], result["audible"]) == (False, True)
    assert (result["lag"], result["polarity"]) == (0, 1)


def test_files_at_two_sample_rates_stop_with_a_message(tmp_path):
    """Two rates cannot be compared sample by sample: the CLI exits naming both."""
    incumbent, candidate = _pair(tmp_path, _tone_bursts() * np.float32(0.5), candidate_rate=48000)
    with pytest.raises(SystemExit, match="sample rates differ: 44100 Hz against 48000 Hz"):
        audibility_check.main([str(incumbent), str(candidate)])


def _usage_error(argv, capsys):
    """The exit code and stderr of a `main(argv)` that argparse refuses."""
    with pytest.raises(SystemExit) as excinfo:
        audibility_check.main(argv)
    return excinfo.value.code, capsys.readouterr().err


def test_an_option_shaped_incumbent_is_a_usage_error(tmp_path, capsys):
    """argparse refuses a value shaped like an option (exit 2) and names the argument and the reason."""
    incumbent, _candidate = _pair(tmp_path)
    code, err = _usage_error(["--", "-incumbent.wav", str(incumbent)], capsys)
    assert code == 2
    assert "argument incumbent: looks like an option, not a path: '-incumbent.wav'" in err


def test_a_missing_candidate_is_a_usage_error(tmp_path, capsys):
    """argparse refuses an input that does not exist (exit 2), naming the candidate argument and the path."""
    incumbent, _candidate = _pair(tmp_path)
    code, err = _usage_error([str(incumbent), str(tmp_path / "missing.wav")], capsys)
    assert code == 2
    assert "argument candidate: does not exist:" in err
    assert "missing.wav" in err


def test_paths_are_confined_where_they_are_used(tmp_path, monkeypatch):
    """A path outside the allowed roots is refused right before it is read."""
    incumbent, candidate = _pair(tmp_path)
    monkeypatch.setattr(cli_paths, "allowed_roots", lambda: (tmp_path / "elsewhere",))
    with pytest.raises(SystemExit, match="must lie inside"):
        audibility_check.main([str(incumbent), str(candidate)])


def test_the_script_runs_standalone_when_scripts_is_not_on_the_path(tmp_path, monkeypatch, capsys):
    """Run as a file, the script puts the repository on sys.path itself when `scripts` does not import."""
    incumbent, candidate = _pair(tmp_path)
    real_import = builtins.__import__
    refused = []

    def first_import_fails(name, *args, **kwargs):
        if name == "scripts.cli_paths" and not refused:
            refused.append(name)
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", first_import_fails)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(incumbent), str(candidate)])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    assert refused == ["scripts.cli_paths"]
    assert sys.path[0] == str(SCRIPT.resolve().parent.parent)
    assert '"identical": true' in capsys.readouterr().out


def test_a_nan_in_a_file_reads_audible_and_is_counted(tmp_path):
    """A broken render is no tie: the DC removal spreads one NaN over its channel, and every such sample is counted."""
    holed = _tone_bursts()
    holed[1000] = np.nan
    incumbent, candidate = _pair(tmp_path, holed)
    result = auditory.compare_files(incumbent, candidate)
    assert (result["identical"], result["audible"]) == (False, True)
    assert result["nonfinite"] == len(holed)


def test_two_different_broken_renders_are_never_read_identical(tmp_path):
    """One NaN in each file decodes both to an all-NaN channel that hashes alike: the shortcut is skipped, the pair reads audible."""
    t = np.arange(3 * RATE) / RATE
    first = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    second = (0.1 * np.sin(2 * np.pi * 660.0 * t) + 0.01 * np.random.default_rng(9).standard_normal(len(t))).astype(np.float32)
    first[1000], second[50000] = np.nan, np.nan
    result = auditory.compare_files(_write(tmp_path / "first.wav", first), _write(tmp_path / "second.wav", second))
    assert result["incumbent_sha256"] == result["candidate_sha256"]
    assert (result["identical"], result["audible"], result["nonfinite"]) == (False, True, 2 * len(t))


def test_a_candidate_cut_short_reads_audible_with_the_missing_span(tmp_path):
    """The first 2 s match the incumbent but for each file's own DC removal (NMR about -80 dB), yet the missing second is no tie."""
    incumbent, candidate = _pair(tmp_path, _tone_bursts()[: 2 * RATE])
    result = audibility_check.main([str(incumbent), str(candidate)])
    assert (result["identical"], result["audible_frac"], result["length_mismatch_s"]) == (False, 0.0, 1.0)
    assert result["nmr_max"] < -60.0
    assert result["audible"] is True
