"""The opt-in digital-mute fixture class (`make_realistic_fixtures_v2.py --digital-mute`).

An edited upload carries digital silence that a tape never did: exact zeros or a dither far
under any tape floor, cut hard into the file. What has to hold is that the gaps are that and
nothing quieter-but-real, that the reference and target are silent where the upload is, that
the class keeps to its own random stream, and that the default set does not write it.
"""

import json
import sys
from unittest.mock import patch

import numpy as np
import soundfile as sf

from scripts import make_realistic_fixtures_v2 as gen

RATE = 44100
SECONDS = 15


def _signals(seed=1):
    """A reference, target and degraded take standing in for a speech-led fixture."""
    rng = np.random.default_rng(seed)
    voice = (0.2 * np.sin(2 * np.pi * 220 * np.arange(SECONDS * RATE) / RATE)).astype(np.float32)
    return voice, voice * 0.9, voice + rng.normal(0.0, 0.01, len(voice)).astype(np.float32)


def _samples(gap):
    """A gap's (start, stop) in samples, as `with_digital_mutes` cut it."""
    return round(gap["start_s"] * RATE), round(gap["stop_s"] * RATE)


def _gap_reading(degraded, gap):
    """(length in seconds, level in dBFS, drawn level or None) of one gap; exact zeros read about -600 dBFS."""
    start, stop = _samples(gap)
    level = 20 * np.log10(np.sqrt(np.mean(degraded[start:stop].astype(np.float64) ** 2)) + 1e-30)
    return gap["stop_s"] - gap["start_s"], level, gap["level_dbfs"]


def _readings(seeds):
    readings = []
    for seed in seeds:
        (_reference, _target, degraded), gaps = gen.with_digital_mutes(_signals(), RATE, np.random.default_rng(seed))
        readings.extend(_gap_reading(degraded, gap) for gap in gaps)
    return readings


def test_every_gap_is_digital_and_between_three_quarters_of_a_second_and_three():
    """Over 40 draws: every gap is 0.75-3 s of exact zeros or of a dither at -120 to -100 dBFS."""
    readings = _readings(range(40))
    assert all(0.75 <= length <= 3.0 for length, _level, _drawn in readings)
    assert max(level for _length, level, _drawn in readings) <= -99.0


def _by_kind(readings):
    """The exact-zero gaps' levels, and the dither gaps' distance from the level they were drawn at."""
    zeros, dithers = [], []
    for _length, level, drawn in readings:
        if drawn is None:
            zeros.append(level)
        else:
            dithers.append(abs(level - drawn))
    return zeros, dithers


def test_exact_zero_gaps_hold_zeros_and_dither_gaps_their_drawn_level():
    """A gap with no drawn level is exact zeros; a gap with one reads that level within a decibel; both kinds occur."""
    zeros, dithers = _by_kind(_readings(range(20)))
    assert (bool(zeros), bool(dithers)) == (True, True)
    assert (max(zeros) < -300.0, max(dithers) < 1.0) == (True, True)


def _muted_signals():
    """The (reference, target, degraded) take before and after `with_digital_mutes`, and the mask of its gap samples."""
    reference, target, degraded = _signals()
    muted, gaps = gen.with_digital_mutes((reference, target, degraded), RATE, np.random.default_rng(3))
    inside = np.zeros(len(reference), dtype=bool)
    for start, stop in map(_samples, gaps):
        inside[start:stop] = True
    return (reference, target, degraded), muted, inside


def test_the_reference_and_target_are_silent_where_the_upload_is():
    """Inside a gap the reference and target are silent too."""
    _originals, (new_reference, new_target, _new_degraded), inside = _muted_signals()
    assert not new_reference[inside].any()
    assert not new_target[inside].any()


def test_the_upload_and_reference_are_untouched_outside_the_gaps():
    """Outside every gap, nothing moved."""
    (reference, _target, degraded), (new_reference, _new_target, new_degraded), inside = _muted_signals()
    assert np.array_equal(new_degraded[~inside], degraded[~inside])
    assert np.array_equal(new_reference[~inside], reference[~inside])


def test_gaps_never_overlap_and_half_the_fixtures_open_on_a_blank_lead_in():
    """Sorted, apart, inside the take; the first gap sits at the head on about half the draws."""
    heads = 0
    for seed in range(60):
        spans = gen.mute_spans(SECONDS * RATE, RATE, np.random.default_rng(seed))
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        assert spans[-1][1] <= SECONDS * RATE
        heads += int(spans[0][0] == 0)
    assert 15 <= heads <= 45


def _clean_segment(*_args):
    yield "en_voice", 0, (0.2 * np.sin(2 * np.pi * 180 * np.arange(4 * RATE) / RATE)).astype(np.float32)


def test_the_class_writes_a_triple_and_records_its_gaps(tmp_path):
    """One fixture per voice, named by the class, with the gaps in its manifest record."""
    noise = [np.random.default_rng(5).normal(0.0, 0.01, RATE).astype(np.float32)]
    with patch.object(gen, "_clean_segments", _clean_segment):
        (record,) = gen._build_digital_mute([], tmp_path, 4 * RATE, noise, [], np.random.default_rng(7))
    assert (record["name"], record["defects"]) == ("en_voice00_digitalmute_m15", ["tape_noise", "digital_mute"])
    degraded, rate = sf.read(str(tmp_path / record["degraded"]), dtype="float32")
    assert (rate, len(degraded), len(record["mute_gaps"]) >= 1) == (RATE, 4 * RATE, True)
    assert json.loads(json.dumps(record))["mute_gaps"] == record["mute_gaps"]


def _recorder(seen, name):
    def record(*args):
        seen.setdefault(name, []).append(int(args[-1].integers(1 << 30)))
        return []

    return record


def _record_builders(monkeypatch):
    """Replaces the four fixture-class builders with recorders; returns what they fill: builder -> one draw per call."""
    seen = {}
    for name in ("_build_speech_led", "_build_music_led", "_build_defect_classes", "_build_digital_mute"):
        monkeypatch.setattr(gen, name, _recorder(seen, name))
    return seen


def test_the_class_is_opt_in_and_draws_on_a_stream_of_its_own(tmp_path, monkeypatch):
    """Off by default; on, its stream is seeded by the voice and shared with none of the other helpers."""
    seen = _record_builders(monkeypatch)
    gen._build([], tmp_path, RATE, 1, [np.zeros(RATE)], [], 4242, 0)
    assert "_build_digital_mute" not in seen
    for voice in (0, 1):
        gen._build([], tmp_path, RATE, 1, [np.zeros(RATE)], [], 4242, voice, True)
    mute = seen["_build_digital_mute"]
    assert mute[0] != mute[1]
    assert mute[0] not in {draws[1] for name, draws in seen.items() if name != "_build_digital_mute"}


def test_the_flag_parses_off_by_default():
    with patch.object(sys, "argv", ["make_realistic_fixtures_v2.py", "--digital-mute"]):
        assert gen._parse_args().digital_mute is True
    with patch.object(sys, "argv", ["make_realistic_fixtures_v2.py"]):
        assert gen._parse_args().digital_mute is False


def _main_build_flags(tmp_path, extra):
    """Runs `main` on one fake language and returns the (voice, digital_mute) each `_build` call received."""
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    calls = []

    def build(*args):
        calls.append(args[7:])
        return [{"name": "en_voice00_speech_m15"}]

    argv = ["make_realistic_fixtures_v2.py", "--output-dir", str(tmp_path / "out"), "--catalog", str(catalog), *extra]
    with (
        patch.object(sys, "argv", argv),
        patch.object(gen, "build_noise_bank", return_value=[np.zeros(RATE, dtype=np.float32)]),
        patch.object(gen, "_discover_languages", return_value={"en": [tmp_path / "en_voice_clean.wav"]}),
        patch.object(gen, "_first_segment", return_value=np.zeros(RATE, dtype=np.float32)),
        patch.object(gen, "_build", side_effect=build),
    ):
        (tmp_path / "out").mkdir(exist_ok=True)
        gen.main()
    return calls


def test_main_hands_the_flag_to_the_builder_and_writes_the_manifest(tmp_path):
    """`--digital-mute` reaches `_build` as its digital_mute argument; without it `_build` gets False."""
    assert _main_build_flags(tmp_path, ["--digital-mute"]) == [(0, True)]
    assert _main_build_flags(tmp_path, []) == [(0, False)]
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["languages"]["en"] == [{"name": "en_voice00_speech_m15"}]
