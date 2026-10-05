"""Pause readings that hear what the listener heard: dead air, hiss left, and pauses collapsing.

The Tata listening set showed the median level of the source's quiet frames cannot tell a
dead pause from a hissy one: every restoration drives the deepest pauses to -96..-104 dBFS,
so a mean over them is noise. The ear reacts to two other things, both read on 20 ms
frames classed by the SOURCE's level:

- the air left in the inter-word gaps (source frames between p15 and p40): the 3-10 kHz
  power there relative to the loud frames (>= p70). Source -6..-4.5 dB on the three speech
  tapes; APL baseline -27 / -33 / -33 ("silent in pauses"), cathar alpha 2 -18.5 / -17 / -15
  ("has hiss"), the cathar plateau -16.5 / -11.5 / -10, the APL plateau -11 / -13 / -7.
- the pause depth: the loud frames' median level minus the deep frames' (<= p15), output
  minus source. The roformer variant reads +41 / +42 / +54 (the long pauses collapse),
  APL baseline +36 / +39 / +45, cathar baseline +32 / +33 / +40, the cathar plateau
  +23 / +23 / +30, the APL plateau +9 / +4 / +17.

Read relative to the loud frames on each side, both survive the gain match (cathar
attenuates the speech itself by ~4 dB).
"""

import numpy as np

from scripts.measure_tradeoff import LOUD_PERCENTILE
from scripts.restoration_quality.dsp_metrics import framed_psd

FRAME_S = 0.02
DEEP_PERCENTILE = 15.0
GAP_PERCENTILES = (15.0, 40.0)
AIR_BAND_HZ = (3000.0, 10000.0)
MIN_CLASS_FRAMES = 20
NAMES = ("gap_air_db", "pause_depth_db")


def level_classes(level_db):
    """`(deep, gap, loud)` masks over the source's frame levels in dB."""
    deep_cut = np.percentile(level_db, DEEP_PERCENTILE)
    gap_cut = np.percentile(level_db, GAP_PERCENTILES[1])
    loud_cut = np.percentile(level_db, LOUD_PERCENTILE)
    return level_db <= deep_cut, (level_db > deep_cut) & (level_db <= gap_cut), level_db >= loud_cut


def gap_air_db(power, freqs, gap, loud):
    """Median 3-10 kHz power on the gap frames over the loud frames, in dB (one side)."""
    band = power[:, (freqs >= AIR_BAND_HZ[0]) & (freqs < AIR_BAND_HZ[1])].sum(axis=1)
    return float(10.0 * np.log10((np.median(band[gap]) + 1e-20) / (np.median(band[loud]) + 1e-20)))


def pause_depth_db(level_db, deep, loud):
    """How far under the loud frames the deep pauses sit, in dB (one side)."""
    return float(np.median(level_db[loud]) - np.median(level_db[deep]))


def pause_readings(source, output, rate):
    """`{"gap_air_db": (source, output), "pause_depth_db": (source, output)}`, or Nones when a class is too thin."""
    frame = max(1, int(FRAME_S * rate))
    count = min(len(source), len(output)) // frame
    if count < 3 * MIN_CLASS_FRAMES:
        return {name: (None, None) for name in NAMES}
    freqs, src_power, src_level = framed_psd(source[: count * frame], rate, frame)
    _freqs, out_power, out_level = framed_psd(output[: count * frame], rate, frame)
    src_db, out_db = _db(src_level), _db(out_level)
    deep, gap, loud = level_classes(src_db)
    if min(deep.sum(), gap.sum(), loud.sum()) < MIN_CLASS_FRAMES:
        return {name: (None, None) for name in NAMES}
    return {
        "gap_air_db": (gap_air_db(src_power, freqs, gap, loud), gap_air_db(out_power, freqs, gap, loud)),
        "pause_depth_db": (pause_depth_db(src_db, deep, loud), pause_depth_db(out_db, deep, loud)),
    }


def _db(level):
    return 20.0 * np.log10(np.asarray(level, dtype=np.float64) + 1e-9)
