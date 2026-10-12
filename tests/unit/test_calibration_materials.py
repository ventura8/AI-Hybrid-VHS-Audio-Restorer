"""The calibration's base materials, read from a realistic-v2 layout: the speech fixture's noise and the music bed's own."""

import numpy as np
import pytest
import soundfile as sf

from scripts import calibrate_quality_metrics as cal

RATE = 16000


def _write(folder, stem, part, audio):
    sf.write(str(folder / f"{stem}_{part}.wav"), np.asarray(audio, dtype=np.float32), RATE, subtype="FLOAT")


def _fixtures(root, music_vhs_extra=0):
    """One language's triplets: the speech fixture, the donor voice and the music bed, each capture its clean + a known noise."""
    folder = root / "en"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(5)
    speech, speech_noise = 0.1 * rng.standard_normal(RATE), 0.01 * rng.standard_normal(RATE)
    bed, bed_noise = 0.2 * rng.standard_normal(RATE), 0.003 * rng.standard_normal(RATE + music_vhs_extra)
    for part, audio in (("target", speech), ("clean", speech), ("vhs", speech + speech_noise)):
        _write(folder, cal.SPEECH_FIXTURE, part, audio)
    _write(folder, cal.DONOR_FIXTURE, "target", np.roll(speech, 100))
    for part, audio in (("target", 0.9 * bed), ("clean", bed), ("vhs", np.pad(bed, (0, music_vhs_extra)) + bed_noise)):
        _write(folder, cal.MUSIC_FIXTURE, part, audio)
    return root, speech_noise, bed, bed_noise


def test_the_stems_cases_get_the_music_beds_clean_programme_and_its_own_noise(tmp_path):
    """`music_bed` is the music fixture's clean bed and `music_noise` its capture minus it."""
    root, _speech_noise, bed, bed_noise = _fixtures(tmp_path)
    materials = cal._materials(root, "en")
    assert np.allclose(materials["music_bed"], bed, atol=1e-6)
    assert np.allclose(materials["music_noise"], bed_noise, atol=1e-6)


def test_the_speech_noise_and_the_music_target_stay_as_they_were(tmp_path):
    """The speech fixture's `noise` (vhs - clean) and the music `target` the other cases use are unchanged."""
    root, speech_noise, bed, _bed_noise = _fixtures(tmp_path)
    materials = cal._materials(root, "en")
    assert np.allclose(materials["noise"], speech_noise, atol=1e-6)
    assert np.allclose(materials["music"], 0.9 * bed, atol=1e-6)
    assert materials["rate"] == RATE


def test_a_capture_longer_than_its_bed_is_cut_to_the_bed(tmp_path):
    """The bed and its noise share one length, the shorter of the clean and the capture."""
    root, _speech_noise, bed, bed_noise = _fixtures(tmp_path, music_vhs_extra=300)
    capture = cal._music_capture(root, "en")
    assert len(capture["music_bed"]) == len(capture["music_noise"]) == len(bed)
    assert np.allclose(capture["music_noise"], bed_noise[: len(bed)], atol=1e-6)


def test_a_missing_music_capture_stops_the_run(tmp_path):
    """Without the music fixture's capture there is no recorded noise to build the stems' cases on."""
    root, *_rest = _fixtures(tmp_path)
    (root / "en" / f"{cal.MUSIC_FIXTURE}_vhs.wav").unlink()
    with pytest.raises(SystemExit, match="fixture missing"):
        cal._music_capture(root, "en")
