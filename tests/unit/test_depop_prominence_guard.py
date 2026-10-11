"""The pop repair's prominence guard: a lone click is still refilled, a cluster of outliers is not.

On real tape the residual threshold alone refilled 6-12 spans a second per channel of clean
speech, single-sample outliers that come in clusters. The guard (`apl_depop_min_prominence_db`)
refills a span only when its peak residual stands that far over every other residual within
5 ms; off (0, the default) the stage refills exactly what it always did.
"""

from unittest.mock import patch

import numpy as np
import soundfile as sf

from modules import config, impulse_repair
from tests.unit.test_impulse_repair import RATE, _voiced

CLUSTER = [RATE // 2 + offset for offset in range(0, 360, 90)]
CLICK = RATE // 4


def _cluster_and_click():
    """A voice with four small outliers 2 ms apart (each one isolated by the density rule) and one lone click."""
    audio = _voiced()
    damaged = audio.copy()
    for position in CLUSTER:
        damaged[position] += 0.06
    click = slice(CLICK, CLICK + 3)
    damaged[click] += 0.4
    return audio, damaged


def _touched(repaired, damaged, position):
    window = slice(position - 4, position + 5)
    return bool(np.any(repaired[window] != damaged[window]))


def test_the_guard_is_off_by_default():
    assert config.CONFIG["apl_depop_min_prominence_db"] == 0.0
    assert config.APL_DEPOP_MIN_PROMINENCE_DB == 0.0


def test_off_the_stage_refills_what_it_always_did():
    """No guard argument, a zero and a negative value are the same stage, sample for sample."""
    _audio, damaged = _cluster_and_click()
    plain, plain_count = impulse_repair.remove_pops(damaged)
    for setting in (0.0, -3.0):
        guarded, count = impulse_repair.remove_pops(damaged, impulse_repair.DEFAULT_THRESHOLD, setting)
        assert np.array_equal(guarded, plain)
        assert count == plain_count
    assert all(_touched(plain, damaged, position) for position in CLUSTER)


def test_on_a_cluster_of_outliers_is_left_alone_and_the_lone_click_refilled():
    audio, damaged = _cluster_and_click()
    repaired, count = impulse_repair.remove_pops(damaged, impulse_repair.DEFAULT_THRESHOLD, 6.0)
    assert count == 1
    assert not any(_touched(repaired, damaged, position) for position in CLUSTER)
    click = slice(CLICK, CLICK + 3)
    assert np.max(np.abs(repaired[click] - audio[click])) < 0.05


def test_prominence_is_read_against_the_largest_residual_around_the_span():
    residual = np.zeros(1000)
    residual[500] = 1.0
    residual[600] = 0.4
    assert impulse_repair._prominent(residual, 499, 502, 6.0)
    assert not impulse_repair._prominent(residual, 499, 502, 9.0)
    assert impulse_repair._prominent(residual, 499, 502, 0.0)


def test_a_span_with_nothing_around_it_is_prominent():
    residual = np.ones(10)
    assert impulse_repair._prominent(residual, 0, 10, 6.0)


def test_a_span_too_long_for_a_pop_is_refused_whatever_the_guard():
    residual = np.zeros(1000)
    residual[400] = 1.0
    long_stop = 400 + impulse_repair.MAX_SPAN_SAMPLES + 1
    assert not impulse_repair._refillable(residual, 400, long_stop, 0.0)
    assert impulse_repair._refillable(residual, 398, 404, 6.0)


def test_the_detection_mask_is_the_residual_test(tmp_path):
    """detect_pops reads the same residual and scale the repair does."""
    _audio, damaged = _cluster_and_click()
    residual, scale = impulse_repair._residual(damaged.astype(np.float64))
    mask = impulse_repair.detect_pops(damaged)
    assert np.array_equal(mask, np.abs(residual) > impulse_repair.DEFAULT_THRESHOLD * scale)
    assert np.isinf(impulse_repair._residual(np.zeros(100))[1]).all()


def _log_line(tmp_path, setting):
    _audio, damaged = _cluster_and_click()
    source = tmp_path / "in.wav"
    sf.write(str(source), damaged, RATE, subtype="FLOAT")
    with patch("modules.impulse_repair.log_msg") as log:
        assert impulse_repair.depop(source, tmp_path / f"out{setting:g}", min_prominence_db=setting) is not None
    return log.call_args[0][0]


def test_the_log_line_reads_as_it_always_did_with_the_guard_off(tmp_path):
    assert _log_line(tmp_path, 0.0) == "    [Repair] Refilled 5 pop spans."


def test_the_log_line_names_the_guard_when_it_is_on(tmp_path):
    assert _log_line(tmp_path, 6.0) == "    [Repair] Refilled 1 pop spans (prominence guard 6 dB)."
