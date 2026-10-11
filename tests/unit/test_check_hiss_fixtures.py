"""The ear v4 fixture probe: the gapped programmes, the noise shapes, the cases, the rows and the acceptance summary."""

import json

import numpy as np
import pytest
import soundfile as sf

from scripts import check_hiss_fixtures as chf
from tests.unit.test_restoration_quality_hiss_floor import RATE, hiss, voice

SECONDS = 12.0


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures(tmp_path_factory):
    """A one-fixture realistic-v2 tree: target, clean and a degraded copy carrying hiss; a second, music-only record is skipped."""
    root = tmp_path_factory.mktemp("rv2")
    folder = root / "xx"
    folder.mkdir()
    target = voice(SECONDS)
    noise = hiss(len(target), seed=8, level=3e-4)
    for part, audio in (("target", target), ("clean", target), ("vhs", target + noise)):
        sf.write(str(folder / f"mid00_speech_m12_{part}.wav"), audio.astype(np.float32), RATE, subtype="FLOAT")
    records = [
        {"name": "mid00_speech_m12", "defects": ["tape_noise"], "clean": "mid00_speech_m12_clean.wav"},
        {"name": "mid00_speech_m09", "defects": ["tape_noise"]},
        {"name": "mid00_hum_m12", "defects": ["hum"]},
    ]
    records[0].update({"target": "mid00_speech_m12_target.wav", "degraded": "mid00_speech_m12_vhs.wav"})
    (root / "manifest.json").write_text(json.dumps({"languages": {"xx": records}}), encoding="utf-8")
    return root


def _run(fixtures, tmp_path, *extra):
    out = tmp_path / "rows.jsonl"
    assert chf.main(["--fixtures", str(fixtures), "--languages", "xx", "--out", str(out), *extra]) == 0
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def test_the_probe_scores_every_case_of_each_m12_tape_noise_fixture(fixtures, tmp_path, capsys):
    """One row for the one m12 tape-noise record, with every case read and the summary printed."""
    rows = _run(fixtures, tmp_path, "--shape", "hiss")
    assert [row["name"] for row in rows] == ["mid00_speech_m12"]
    assert set(rows[0]["cases"]) == {"identity", "prog_only", "fresh", *chf.LADDER, "delay037", "gain_up", "gain_down"} | {
        "soxr",
        "requantise",
        "hf_left",
        "lp4k_all",
        "gated_m0",
        "islands_15",
    }
    assert "oracle ladder monotone" in capsys.readouterr().out


def test_a_shape_written_from_a_capture_drives_the_noise(fixtures, tmp_path):
    """`--shape-from` writes the three-row shape file and the probe then reads it as its shape."""
    shape = tmp_path / "shape.npy"
    rows = _run(fixtures, tmp_path, "--shape-from", str(fixtures / "xx" / "mid00_speech_m12_vhs.wav"), "--shape-out", str(shape))
    assert np.load(shape).shape[0] == 3
    assert rows[0]["shape"] == str(shape)


@pytest.mark.parametrize("shape", ["corpus", "white"])
def test_the_corpus_and_white_shapes_give_a_hiss_and_a_fresh_realisation(shape):
    """Each named shape returns two noises as long as the programme, different draws."""
    programme = chf.programme_of(voice(SECONDS), RATE, "full")
    corpus = hiss(len(voice(SECONDS)), seed=8, level=3e-4)
    noise, fresh = chf.make_noise(shape, corpus, programme, RATE)
    assert (len(noise), len(fresh)) == (len(programme), len(programme))
    assert not np.array_equal(noise, fresh)


def test_the_linear_programme_is_low_passed_and_gapped():
    """`linear` takes the highs off; both kinds splice in the two 1.2 s gaps."""
    linear, full = (chf.programme_of(voice(SECONDS), RATE, kind) for kind in ("linear", "full"))
    cut = int(6000 * len(full) / RATE)
    assert len(linear) == len(full) == int(SECONDS * RATE) + 2 * int(1.2 * RATE)
    assert np.sum(np.abs(np.fft.rfft(linear)[cut:])) < np.sum(np.abs(np.fft.rfft(full)[cut:]))


LADDER_CASES = ("identity", "oracle_m6", "oracle_m12", "oracle_m20", "oracle_m30", "prog_only")
OTHER_CASES = ("fresh", "delay037", "gain_up", "gain_down", "soxr", "requantise", "hf_left", "lp4k_all", "gated_m0", "islands_15")


def _readings(value):
    return dict.fromkeys(chf.READINGS, value)


def _row(identity, ladder=None, validated=True, **cases):
    """A fabricated row whose margins read `identity` and the given ladder; every other case reads the identity."""
    values = {**dict(zip(LADDER_CASES, ladder or [identity] * 6)), **{name: cases.get(name, identity) for name in OTHER_CASES}}
    return {"validated": validated, "cases": {name: _readings(value) for name, value in values.items()}}


def test_the_summary_reads_the_acceptance_rows():
    """A monotone ladder whose perfect output is inaudible, exact gains and a fresh realisation 0.5 dB off."""
    rows = [_row(5.0, [5.0, 0.0, -4.0, -8.0, -9.0, -9.5], fresh=5.5, gated_m0=5.2), _row(1.0, validated=False)]
    lines = chf.summary_lines(rows)
    assert lines[0] == "validated 1/2"
    assert "(0.5, 0.5)" in lines[5]
    assert lines[6].startswith("oracle ladder monotone 1/1; perfect <= 0 1/1")


def test_a_ladder_that_rises_or_was_not_read_is_not_monotone():
    """A rise of more than 0.05 dB, or a step read None, fails the ladder."""
    assert not chf.monotone(_row(5.0, [5.0, 5.1, 0.0, -1.0, -2.0, -3.0]))
    assert not chf.monotone(_row(5.0, [5.0, None, 0.0, -1.0, -2.0, -3.0]))
    assert chf.p95_max([]) == (None, None)


def test_an_anchor_without_a_level_has_no_error():
    """`no_floor` gives no anchor error."""
    anchor = chf.hiss_floor.Anchor(None, np.zeros(0), 0.0, chf.hiss_floor.NO_FLOOR, False)
    assert chf.anchor_error(anchor, np.zeros(10), RATE, None) is None
