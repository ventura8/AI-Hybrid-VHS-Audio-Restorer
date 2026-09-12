"""Tests for the calibrated fixture generator and the realism check that judges it.

Every real-tape verdict on this branch is tested against these fixtures, so what has to hold
is that each fault is actually in the file the class says carries it, at a level the scanner
reads, and that the reference and target stay what the docstrings promise: the reference is
everything the tape carried, the target is the programme wanted back. The checks are CPU-only
and run on a second of synthetic speech, since the generator's own run over the corpus takes
the corpus.
"""

import numpy as np
import pytest

from scripts import make_realistic_fixtures_v2 as gen
from scripts import realistic_defects as defects
from scripts import validate_fixture_realism as check

RATE = 44100


def _voice(seconds=2.0, seed=3):
    """Voiced, breathy, level-varying audio standing in for Piper speech."""
    rng = np.random.default_rng(seed)
    time = np.arange(int(RATE * seconds)) / RATE
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 1.3 * time) ** 2
    voiced = sum(np.sin(2 * np.pi * 150.0 * harmonic * time) / harmonic for harmonic in range(1, 12))
    return (0.3 * envelope * (voiced + 0.02 * rng.normal(0.0, 1.0, len(time)))).astype(np.float32)


def _noise(seconds=1.0, seed=5):
    return np.random.default_rng(seed).normal(0.0, 0.01, int(RATE * seconds)).astype(np.float32)


def _rng():
    return np.random.default_rng(11)


# --- the noise bank ---------------------------------------------------------------------------


def test_the_bank_takes_hiss_and_refuses_tones_fades_and_silence():
    """A window is noise when it is steady, noise-like and quiet; level alone admitted all three."""
    rng = np.random.default_rng(1)
    hiss = rng.normal(0.0, 0.01, RATE * 4).astype(np.float32)
    tone = (0.5 * np.sin(2 * np.pi * 1000.0 * np.arange(RATE * 4) / RATE)).astype(np.float32)
    fade = hiss * np.linspace(1.0, 0.01, RATE * 4).astype(np.float32)
    silence = np.zeros(RATE * 4, dtype=np.float32)
    assert defects._is_stationary_noise(hiss)
    assert not defects._is_stationary_noise(tone)
    assert not defects._is_stationary_noise(fade)
    assert not defects._is_stationary_noise(silence)


def test_the_bank_strips_the_capture_tones_out_of_its_windows():
    """The line whine and the mains series are faults with classes of their own, not noise."""
    time = np.arange(RATE * 4) / RATE
    tones = 0.1 * np.sin(2 * np.pi * 15625.0 * time) + 0.1 * np.sin(2 * np.pi * 50.0 * time)
    window = (np.random.default_rng(2).normal(0.0, 0.001, len(time)) + tones).astype(np.float32)
    cleaned = defects._without_tones(window, RATE)
    spectrum = np.abs(np.fft.rfft(cleaned))
    freqs = np.fft.rfftfreq(len(cleaned), 1.0 / RATE)
    assert spectrum[np.argmin(np.abs(freqs - 15625.0))] < 0.01 * spectrum.max()
    assert spectrum[np.argmin(np.abs(freqs - 50.0))] < 0.01 * spectrum.max()


# --- the injectors ----------------------------------------------------------------------------


def test_crackle_is_impulsive_and_read_as_clicks():
    """The click detector the chain gates decrackle on must see the crackle class."""
    from modules.filters import _detect_click_density

    voice = _voice()
    assert not _detect_click_density(voice)
    assert _detect_click_density(defects.inject_crackle(voice, RATE, _rng()))


def test_dropouts_are_true_silences_inside_the_audio():
    """Head-contact loss is signal absence, with an edge, never in the opening half second."""
    out = defects.inject_dropouts(_voice(), RATE, _rng())
    quiet = np.abs(out) < 1e-6
    assert quiet.sum() > int(0.02 * RATE)
    assert not quiet[: RATE // 2 - 1].any()


def test_clipping_sits_at_the_ceiling_the_scanner_looks_for():
    """Flat tops at 0.985 in numbers, or the declip stage never runs."""
    clipped = defects.inject_clipping(_voice())
    assert np.isclose(np.max(np.abs(clipped)), defects.CLIP_CEILING)
    assert int(np.sum(np.abs(clipped) >= defects.CLIP_CEILING)) >= 20


def test_the_whistle_lands_on_a_line_rate():
    voice = _voice()
    out = defects.inject_line_whistle(voice, RATE, _rng(), line_rate_hz=15625.0)
    spectrum = np.abs(np.fft.rfft(out - voice))
    freqs = np.fft.rfftfreq(len(out), 1.0 / RATE)
    assert abs(freqs[np.argmax(spectrum)] - 15625.0) < 5.0


def test_wow_and_flutter_keeps_the_length_and_moves_the_pitch():
    voice = _voice()
    warped = defects.inject_wow_and_flutter(voice, RATE)
    assert warped.shape == voice.shape
    assert not np.allclose(warped, voice)


def test_azimuth_delays_the_right_channel_and_imbalance_lowers_it():
    voice = _voice()
    skewed = defects.assemble_stereo(voice, RATE, _rng(), azimuth=True)
    lag = np.argmax(np.correlate(skewed[:20000, 1], skewed[:20000, 0], mode="full")) - 19999
    assert 0 < lag <= 20
    unbalanced = defects.assemble_stereo(voice, RATE, _rng(), imbalance=True)
    assert np.sqrt(np.mean(unbalanced[:, 1] ** 2)) < 0.8 * np.sqrt(np.mean(unbalanced[:, 0] ** 2))


def test_head_switching_buzz_repeats_at_the_field_rate():
    voice = np.zeros(RATE, dtype=np.float32) + 0.01
    out = defects.inject_head_switching(voice, RATE, _rng(), field_hz=50.0) - voice
    bursts = np.flatnonzero(np.abs(out) > 0)
    assert len(bursts) > 0
    starts = bursts[np.diff(bursts, prepend=-10) > 5]
    assert 45 <= len(starts) <= 55


def test_modulation_noise_is_absent_without_signal():
    silence = np.zeros(RATE, dtype=np.float32)
    assert np.allclose(defects.inject_modulation_noise(silence, RATE, _rng()), 0.0)
    voice = _voice(1.0)
    assert not np.allclose(defects.inject_modulation_noise(voice, RATE, _rng()), voice)


def test_print_through_is_a_pre_echo():
    """The layer above prints ahead, so an event late in the take appears faintly one wrap earlier."""
    audio = np.zeros(RATE * 5, dtype=np.float32)
    audio[RATE * 4] = 1.0
    out = defects.inject_print_through(audio, RATE)
    wrap = int(defects.PRINT_THROUGH_WRAP_S * RATE)
    assert np.isclose(out[RATE * 4 - wrap], 10.0 ** (defects.PRINT_THROUGH_DB / 20.0))


def test_print_through_on_a_take_shorter_than_a_wrap_still_prints():
    """A short take gets its echo half a take ahead rather than no echo at all."""
    audio = np.zeros(RATE * 2, dtype=np.float32)
    audio[int(RATE * 1.5)] = 1.0
    out = defects.inject_print_through(audio, RATE)
    assert np.isclose(out[int(RATE * 1.5) - RATE], 10.0 ** (defects.PRINT_THROUGH_DB / 20.0))


def test_the_track_switch_dulls_one_stretch_and_leaves_the_rest():
    voice = _voice(4.0)
    out = defects.inject_track_switch(voice, RATE, _noise(), _rng())
    changed = np.flatnonzero(~np.isclose(out, voice, atol=1e-6))
    assert 0 < len(changed) < len(voice)


def test_the_ghost_is_faint_and_band_limited():
    voice = _voice()
    other = _voice(seed=9)
    out = defects.inject_ghost_recording(voice, RATE, other)
    added = out - voice
    assert 0 < np.sqrt(np.mean(added**2)) < 0.2 * np.sqrt(np.mean(voice**2))


# --- the generator's chain ---------------------------------------------------------------------


def test_the_reference_carries_the_room_and_the_target_does_not():
    """The reference is what the tape carried, the target the programme wanted back."""
    reference, target = gen.condition_programme(_voice(), RATE, _rng(), music=False, other=_voice(seed=9))
    assert reference.shape == target.shape
    assert np.max(np.abs(reference)) <= 0.7 + 1e-6
    assert not np.allclose(reference, target)


def test_degrade_keeps_the_gain_trajectory_on_reference_and_target():
    """Recorder gain control is part of the recording, so both carry it and the noise sits on top."""
    programme = gen.condition_programme(_voice(), RATE, _rng(), music=False)
    reference, target, degraded = gen.degrade(programme, RATE, _noise(), 15.0, False, _rng())
    assert reference.shape == target.shape == degraded.shape
    assert np.max(np.abs(reference)) <= 0.7 + 1e-6
    residual = degraded - reference
    margin = 20 * np.log10(np.sqrt(np.mean(reference**2)) / (np.sqrt(np.mean(residual**2)) + 1e-12))
    assert 13.0 < margin < 17.0


@pytest.mark.parametrize("name", sorted(gen.DEFECT_CLASSES))
def test_every_defect_class_builds_a_matching_triple(name, tmp_path):
    """Each class comes out with a reference, a target and a degraded file of one length, and changes the audio."""
    if name == "codec":
        pytest.skip("runs the codec through FFmpeg; covered by the generator run")
    programme = gen.condition_programme(_voice(), RATE, _rng(), music=False)
    faults = gen.DEFECT_CLASSES[name]
    reference, target, degraded = gen.degrade_with(programme, RATE, _noise(), 15.0, faults, _rng(), _voice(seed=9), tmp_path)
    assert len(reference) == len(target) == len(degraded)
    assert np.isfinite(degraded).all()
    assert not np.allclose(degraded, reference)


def test_every_class_stream_is_seeded_by_the_voice(tmp_path, monkeypatch):
    """Five voices under one noise window would be one draw: each helper's stream must differ by voice."""
    draws = {}

    def recorder(name):
        def record(*args):
            draws.setdefault(name, []).append(int(args[-1].integers(1 << 30)))
            return []

        return record

    monkeypatch.setattr(gen, "_build_speech_led", recorder("speech"))
    monkeypatch.setattr(gen, "_build_music_led", recorder("music"))
    monkeypatch.setattr(gen, "_build_defect_classes", recorder("defects"))
    for voice in (0, 1, 1):
        gen._build([], tmp_path, RATE, 1, [np.zeros(RATE)], [], 4242, voice)
    assert {name: seen[1] == seen[2] for name, seen in draws.items()} == {"speech": True, "music": True, "defects": True}
    assert {name: seen[0] != seen[1] for name, seen in draws.items()} == {"speech": True, "music": True, "defects": True}
    assert len({seen[0] for seen in draws.values()}) == 3, "the three helpers share a stream"


def test_stereo_classes_come_out_stereo_and_the_rest_mono():
    programme = gen.condition_programme(_voice(), RATE, _rng(), music=False)
    _r, _t, stereo = gen.degrade_with(programme, RATE, _noise(), 15.0, ("azimuth",), _rng())
    _r, _t, mono = gen.degrade_with(programme, RATE, _noise(), 15.0, ("crackle",), _rng())
    assert stereo.ndim == 2 and stereo.shape[1] == 2
    assert mono.ndim == 1


# --- the realism check ---------------------------------------------------------------------------


def test_the_class_of_a_fixture_is_read_from_its_name():
    assert check._class_of("mid00_speech_m15_hum") == "speech"
    assert check._class_of("mid01_musiconly_m24") == "musiconly"
    assert check._class_of("mid00_low_level_m15") == "low_level"
    assert check._class_of("mid00_worn_m09") == "worn"


@pytest.mark.parametrize(
    ("rule", "removal", "deviation", "expected"),
    [
        (check.REMOVAL_DOMINATES, 2.0, 0.2, True),
        (check.REMOVAL_DOMINATES, 2.0, 0.8, False),
        (check.REMOVAL_DOMINATES, -1.0, -0.5, False),
        (check.DEVIATION_DECIDES, -3.0, -0.3, True),
        (check.DEVIATION_DECIDES, 3.0, 0.1, False),
    ],
)
def test_agreement_rules_follow_the_real_tape_verdicts(rule, removal, deviation, expected):
    """Removal dominates for a stronger subtraction; deviation decides for a stage that damages programme."""
    assert check._agrees(rule, removal=removal, deviation=deviation) is expected


def test_every_defect_class_has_a_variant_in_the_check():
    """A class the generator writes and the check never runs is coverage on paper."""
    checked = {variant.rsplit("_m", 1)[0] for variant in check.DEFECT_VARIANTS}
    assert checked == set(gen.DEFECT_CLASSES)


def test_the_blend_split_keeps_every_cut_of_one_voice_together():
    """The naming rule lives with the dataset builder, torch-free, so the split is checked on every CI job."""
    from scripts.build_blend_dataset import _recording_of

    for name in ("en_mid00_speech_m15_hum", "en_mid01_musiconly_m24", "en_mid00_worn_m09", "en_mid00_hiss_only"):
        assert _recording_of(name) == "en_mid"
    assert _recording_of("de_mid00_speech_m15") != _recording_of("en_mid00_speech_m15")


def test_the_source_id_counts_only_fixtures_that_produced_data(tmp_path, monkeypatch):
    """A skipped record must not shift the ids that index the names on the training side."""
    from scripts import build_blend_dataset as build

    records = [
        {"name": "en_mid00_speech_m15", "clean": "missing.wav", "degraded": "missing.wav"},
        {"name": "en_mid00_flutter_m15", "clean": "a.wav", "degraded": "a.wav", "defects": ["flutter"]},
        {"name": "en_mid01_speech_m15", "clean": "a.wav", "degraded": "a.wav"},
    ]
    (tmp_path / "a.wav").write_bytes(b"")
    monkeypatch.setattr(build, "_denoise", lambda degraded, work: tmp_path / "a.wav")
    monkeypatch.setattr(build, "_pair_arrays", lambda *paths: tuple(np.ones((3, 2)) for _ in range(5)))
    *_arrays, sources, names = build._collect(records, tmp_path, tmp_path / "work", limit=None)
    assert names == ["en_mid01_speech_m15"]
    assert sources.tolist() == [0, 0, 0]


def _matrix(**tripped):
    """A coverage matrix over three classes, five fixtures each, with the given own-detector hit counts."""
    hits = {name: dict.fromkeys(check.DETECTORS, 0) for name in ("crackle", "dropout", "resonance")}
    for name, (detector, count) in tripped.items():
        assert detector in check.DETECTORS, f"{detector!r} is not a detector the check knows"
        hits[name][detector] = count
    return hits, dict.fromkeys(hits, 5)


def test_a_class_blind_to_its_own_detector_fails_the_run_and_a_documented_one_does_not():
    """The realism check exits nonzero on what would leave a stage unexercised, not on the scanner's known limits."""
    hits, counts = _matrix(crackle=("clicks", 4), dropout=("dropouts", 1), resonance=("resonance", 0))
    assert check._blind_classes(hits, counts) == ["dropout"]


def test_a_documented_blind_pair_still_requires_the_class_to_trip_its_other_detectors():
    """The worn tape's hum is excused; its clicks, dropouts, drift and DC bias are not."""
    hits = {"worn": dict.fromkeys(check.DETECTORS, 5)}
    hits["worn"]["hum"] = 0
    assert check._blind_classes(hits, {"worn": 5}) == []
    hits["worn"]["dc"] = 0
    assert check._blind_classes(hits, {"worn": 5}) == ["worn"]


@pytest.mark.parametrize(
    ("summary", "reasons"),
    [
        ({"agreement": 4, "comparisons": 4, "blind_classes": [], "repair_not_free": 0}, 0),
        ({"agreement": 4, "comparisons": 4}, 0),
        ({"agreement": 3, "comparisons": 4, "blind_classes": ["dropout"], "repair_not_free": 1}, 3),
    ],
)
def test_every_failed_check_is_a_reason_to_exit_nonzero(summary, reasons):
    assert len(check._failures(summary)) == reasons


def test_each_fault_entry_runs_once_and_the_whine_goes_on_before_the_wear():
    """A class's faults are one injection each, and the recorded whine precedes the dropouts that lose it."""
    order = [name for name, _inject in gen.TAPE_FAULTS]
    assert len(set(order)) == len(order)
    assert order.index("recordedwhine") < order.index("dropout") < order.index("flutter")


@pytest.mark.parametrize("name", ["flutter", "ep", "worn"])
def test_the_transport_classes_carry_a_recorded_whine_rather_than_a_second_whistle(name):
    """The whine the drift detector needs is a named fault of the transport classes, not the whistle class's."""
    assert "recordedwhine" in gen.DEFECT_CLASSES[name]
    assert "whistle" not in gen.DEFECT_CLASSES[name]
    assert gen.DEFECT_CLASSES["whistle"] == ("whistle",)
