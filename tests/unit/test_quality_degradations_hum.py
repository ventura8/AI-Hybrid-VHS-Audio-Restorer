"""The hum case puts its hum on the mains R0 names on its base, the series `dsp.hum_excess_db` reads there."""

import numpy as np
import scipy.signal

from scripts import quality_degradations as deg
from scripts import realistic_defects as defects
from scripts.restoration_quality import dsp_metrics, runner, source_profile

RATE = 16000
LEVELS = deg.DEGRADATIONS["hum"].levels


def _sixty_hertz_base():
    """15 s of noise carrying a weak 60 Hz series: R0 names 60 Hz on it."""
    noise = (3e-3 * np.random.default_rng(3).standard_normal(15 * RATE)).astype(np.float32)
    return defects.inject_mains_hum(noise, RATE, base_hz=60.0, level=0.003, rng=np.random.default_rng(1))


def _fundamental_hz(added):
    """The strongest frequency under 80 Hz of the hum a case added."""
    freqs, psd = scipy.signal.welch(added, RATE, nperseg=16384)
    low = (freqs > 30.0) & (freqs < 80.0)
    return float(freqs[low][np.argmax(psd[low])])


def _outputs():
    base = _sixty_hertz_base()
    return base, [deg.apply("hum", level, base, RATE, {}, np.random.default_rng(2))[1] for level in LEVELS]


def test_the_hum_lands_on_the_mains_r0_names():
    """On a 60 Hz capture R0 names 60 Hz and the injected series is 60 Hz."""
    base, outputs = _outputs()
    assert deg.capture_mains_hz(base, RATE) == 60.0
    assert abs(_fundamental_hz(outputs[-1] - base) - 60.0) < 1.0


def test_the_reading_at_r0s_mains_rises_with_the_level():
    """The reading at R0's mains orders the three levels, the severest over 2 dB."""
    base, outputs = _outputs()
    rises = [dsp_metrics.hum_excess_db(out, RATE, 60.0) - dsp_metrics.hum_excess_db(base, RATE, 60.0) for out in outputs]
    assert rises[0] < rises[1] < rises[2]
    assert rises[-1] > 2.0


def test_without_a_mains_the_hum_falls_back_to_the_runners_fifty_hertz(monkeypatch):
    """A base on which R0 names no mains takes 50 Hz hum, the frequency the runner reads such a source at."""
    monkeypatch.setattr(source_profile, "capture_profile", lambda _audio, _rate: {"mains_hz": None})
    base = _sixty_hertz_base()
    assert deg.capture_mains_hz(base, RATE) == runner.DEFAULT_MAINS_HZ == deg.DEFAULT_MAINS_HZ
    _source, output = deg.apply("hum", LEVELS[-1], base, RATE, {}, np.random.default_rng(2))
    assert abs(_fundamental_hz(output - base) - 50.0) < 1.0


def test_hum_takes_its_mains_as_an_argument():
    """`hum` passes its mains to the injector; its default is the runner's fallback."""
    silence = np.zeros(4 * RATE, dtype=np.float32)
    assert abs(_fundamental_hz(deg.hum(silence, RATE, 0.01, np.random.default_rng(0), 60.0)) - 60.0) < 1.0
    assert abs(_fundamental_hz(deg.hum(silence, RATE, 0.01, np.random.default_rng(0))) - 50.0) < 1.0
