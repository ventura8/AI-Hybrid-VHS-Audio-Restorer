"""The pause floor's hiss shelf: the shelf's cut taken from the source's hiss only, its programme above the corner kept.

The fixture is the shaped-fill tests' tape voice (tone bursts over room tone and white hiss) with
a soft 6 kHz partial held through its first five seconds: soft programme above the corner that
the restored copy, its pauses pushed 40 dB down, has lost, the way the Mel-RoFormer empties a
music bed. Held through the whole file, the partial would be the print's 5th percentile itself,
and the shelf would take it for hiss, as the reference suppressor would.
"""

import functools
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import soundfile as sf

import modules.pause_fill_shape as pause_fill_shape
import modules.pause_floor as pause_floor
from tests.unit.test_pause_floor import RATE, _gated, _on
from tests.unit.test_pause_floor_hissfree import LOW_BAND, _added, _band_db, _pair, _pause_mask, _settings, _tape_voice

PARTIAL_HZ = 6000.0
PARTIAL_BAND, HISS_BAND = (5800.0, 6200.0), (8000.0, 12000.0)
PARTIAL_S = 5.0


@functools.lru_cache(maxsize=None)
def _voice_with_partial():
    """The tape voice plus a 6 kHz partial at 2e-3 through PARTIAL_S (it stands far over the white hiss's bins), faded
    out over 50 ms. Cached: read only."""
    t = np.arange(len(_tape_voice())) / RATE
    partial = 2e-3 * np.sin(2 * np.pi * PARTIAL_HZ * t) * np.clip((PARTIAL_S - t) / 0.05, 0.0, 1.0)
    return (_tape_voice() + partial[:, None]).astype(np.float32)


def _keep(reference, restored, work, **settings):
    """The keeper on (12 dB, p15) on the 'frame' target at the given shape; returns the path it hands on."""
    with ExitStack() as stack:
        for context in (*_on(), *_settings(target="frame", **settings)):
            stack.enter_context(context)
        return pause_floor.apply_when_needed(reference, restored, work)


def _fill_db(tmp_path, shape):
    """The fill's (low band, partial, hiss band) levels in the deep pauses, in dB against the source's own."""
    source = _voice_with_partial()
    reference, restored = _pair(tmp_path, _gated(source), source)
    added = _added(_keep(reference, restored, tmp_path, shape=shape), restored)
    mask = _pause_mask(source) & (np.arange(len(source)) < (PARTIAL_S - 0.1) * RATE)
    return tuple(_band_db(added, band, mask) - _band_db(source, band, mask) for band in (LOW_BAND, PARTIAL_BAND, HISS_BAND))


def test_the_hiss_shelf_fills_the_programme_the_shelf_cuts_and_cuts_the_hiss_as_deep(tmp_path):
    low, partial, hiss = _fill_db(tmp_path / "hiss", "hiss_shelf")
    shelf_low, shelf_partial, shelf_hiss = _fill_db(tmp_path / "shelf", "shelf")
    assert (abs(low + 12.0) < 1.5, abs(shelf_low + 12.0) < 1.5) == (True, True)
    assert (abs(partial + 12.0) < 2.0, abs(shelf_partial + 30.0) < 2.5) == (True, True)
    assert (abs(hiss + 30.0) < 2.5, abs(shelf_hiss + 30.0) < 2.5) == (True, True)


def test_the_shelf_response_is_the_gain_the_shaper_gives_a_steady_tone():
    shaper = pause_floor.shaper_for("hiss_shelf", RATE, 3000.0, 18.0)
    response = pause_fill_shape.shelf_response(shaper, RATE)
    freqs = np.fft.rfftfreq(pause_fill_shape.FRAME, 1.0 / RATE)
    t = np.arange(RATE) / RATE
    for hz in (500.0, 3000.0, 9000.0):
        hz = freqs[int(np.argmin(np.abs(freqs - hz)))]
        tone = np.sin(2 * np.pi * hz * t)[:, None]
        inner = slice(RATE // 4, -RATE // 4)
        out = pause_floor.shaped(tone, shaper)[inner]
        assert abs(np.sqrt(2.0 * np.mean(out**2)) - response[int(np.argmin(np.abs(freqs - hz)))]) < 0.01
    assert (abs(response[0] - 1.0) < 1e-3, abs(response[-1] - 10.0 ** (-18.0 / 20.0)) < 1e-3) == (True, True)


def test_without_a_print_the_hiss_shelf_fills_as_the_plain_shelf(tmp_path):
    source = _voice_with_partial()
    reference, restored = _pair(tmp_path, _gated(source), source)
    with patch.object(pause_fill_shape, "source_print", return_value=None), patch("modules.pause_floor.log_msg") as log:
        fallback = _keep(reference, restored, tmp_path / "hiss", shape="hiss_shelf")
    shelf = _keep(reference, restored, tmp_path / "shelf", shape="shelf")
    assert np.array_equal(sf.read(str(fallback))[0], sf.read(str(shelf))[0])
    assert any("No print to shelve the hiss by" in call.args[0] for call in log.call_args_list)


def test_the_hiss_shelved_source_is_written_once_and_reused(tmp_path):
    source = _voice_with_partial()
    reference, restored = _pair(tmp_path, _gated(source), source)
    first = _keep(reference, restored, tmp_path / "work", shape="hiss_shelf")
    shelved = sorted((tmp_path / "work" / "pause_floor").glob("hiss_shelved_*"))
    assert [path.name for path in shelved] == ["hiss_shelved_3000c18.00_ref.wav"]
    first.unlink()
    with patch.object(pause_fill_shape, "write_hiss_shelved", side_effect=AssertionError("written twice")):
        again = _keep(reference, restored, tmp_path / "work", shape="hiss_shelf")
    assert again == first


def test_the_hiss_shelved_source_passes_the_band_under_the_corner_whole(tmp_path):
    source = _voice_with_partial()
    reference = tmp_path / "ref.wav"
    sf.write(str(reference), source, RATE, subtype="FLOAT")
    shaper = pause_floor.shaper_for("hiss_shelf", RATE, 3000.0, 18.0)
    shelved = pause_fill_shape.hiss_shelved(reference, tmp_path, shaper, pause_fill_shape.source_print(reference), (3000.0, 18.0))
    out = sf.read(str(shelved), dtype="float64", always_2d=True)[0]
    whole = np.arange(len(source)) < (PARTIAL_S - 0.1) * RATE
    assert abs(_band_db(out, LOW_BAND, whole) - _band_db(source, LOW_BAND, whole)) < 0.2
    assert abs(_band_db(out, PARTIAL_BAND, whole) - _band_db(source, PARTIAL_BAND, whole)) < 1.0
    assert _band_db(out, HISS_BAND, whole) - _band_db(source, HISS_BAND, whole) < -15.0
