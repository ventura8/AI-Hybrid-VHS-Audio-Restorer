"""Orchestration: load a pair, window it, run each metric family, aggregate, gate.

Families run in the order dsp -> stems -> speech -> mos with one model resident at a
time. A family whose model or library is missing is recorded as unavailable with the
reason; it never aborts the run, so `--metrics dsp` needs neither torch nor a download.

Ear v3 wiring (the dsp family):

- R0, the capture profile (`source_profile.capture_profile`), is read once per source on
  the raw multichannel file (its channel state needs both channels) and cached beside the
  source's other features as `<cache>/profile/<source key>_<code hash>.json`; the code hash
  covers `source_profile.py` and the modules it reads with (`measure_hum`, `dsp_metrics`,
  `pause_metrics`), so a change there re-reads every profile (about a second per source)
  instead of serving a stale one. It lands in the card as `meta.*` (source side, displayed) and in
  `card.meta["source_profile"]`, and it replaces the 50 Hz / 15625 Hz assumptions:
  `dsp.hum_excess_db` reads the source's `mains_hz` and `dsp.whistle_db` its `line_hz`,
  falling back to 50 / 15625 Hz when R0 finds none.
- R1 (`balance_metrics`) reads every window on the aligned pair (its programme gain match
  cancels `align_pair`'s gain exactly), clipped to R0's programme bandwidth as designed.
  `dsp.balance_top_db` is presence or air, whichever moved further: the bright / dull flags.
  A window whose source holds an analog mute reads R1 None (`holds_mute`): there R1's quiet
  frames measure the mute, not the hiss under the programme, and read hiss removal as
  dulling. A mute is 5 % of the window's live 20 ms frames 15 dB or more under the source's
  floor (`mute_floor_db`), decided on the source so every variant skips the same windows.
  The floor is the median over the programme windows of each one's p10 from seven windows up
  (a mute up to 7.5 s touches three windows at most, so it cannot own it), and below that
  the higher of the median and the p25 of every live frame in them: a mute sets the floor
  only when it fills more than half the windows (the higher middle p10 of an even count)
  and a quarter of the span too. Each alone missed one.
  A 15 s pair has two windows, both over its last 7.5 s, so a 2 s mute there owned the
  median (floor -90 dBFS, both windows read air -19 dB unguarded and guarded alike); a 4 s
  mute at 13 s owned the median of five windows on 37.5-40 s pairs and sat half-way in six
  on 45 s (-52.5 dBFS over a -65 dBFS mute), and windows 0-2 read air -19 dB and tilt -6.4
  to -7.4 dB/oct. A blank lead-in counts in the span of the window it opens, so 4 s of blank
  before 11 s of speech (8 + 15 s, 10 + 20 s and 60 + 15 s alike) owned the p25 (floor
  -90 dBFS, the windows over it read air -14 to -19 dB). On the runner tests' speech (15 s,
  hiss at -50 or -40 dBFS removed by an oracle) a stretch 25-40 dB under the hiss reads air
  -19 dB and tilt -7 dB/oct once it fills 10.7 % of the window, where the p10 R1 cuts its
  quiet frames from falls into it, and 0.0 at 10.0 %; 20 dB under reads the same at 4 s and
  nothing at 2 s, 15 dB under under 0.4 dB at both. Over 1-4 s mutes at -90, -70 and
  -65 dBFS, at every 0.5 s position on 15, 20, 30, 37.5, 40, 45 and 50 s pairs, the guard
  covers every window a mute fills 10.7 % of, except a 4 s mute in 15 s (over a quarter of
  the span and in both windows: 24 % at every depth). The median from five windows with
  the p25 alone below covered 88-91 % of the 4 s cases on 37.5 and 40 s pairs (89 % on 45 s
  at -65 dBFS) and none in 15 s; the median alone covered 17-32 % of the 2-4 s cases on 15
  and 20 s pairs at -90 dBFS (none on 15 s at -65 dBFS) and 11-20 % on 30 s pairs at
  -65 dBFS. Skipped windows on material without a planted mute (2026-10-09; the median alone
  in brackets): `artifacts/realistic-v2` `_vhs` 0 of 920 (0), `_clean` 23 (0), `_target` 293
  (95), 249 of them window 0 over the Piper lead-in (a median 0.76 s run under the floor);
  IA corpus 20 of 109 (6), three of them over two -86 dBFS stretches the median missed (3.5 s
  at 16.5-20 s in windows 1-2, the 1.06 s opening in window 0),
  two (jakki-brambles windows 0-1, skipped by the median too) over the 8.2 s at -62 dBFS
  that open the clip, 41 % of it; the three Tata sources 1 of 137 (1). The cost falls on the
  calibration's clean speech targets: R1 reads its spectral tilt cases on window 1 only on
  en, es and fr and not at all on it (the median alone: both windows on all four; de's
  1122 Hz band leaves no tilt either way). The p20 keeps every one of those windows (IA 14
  skipped, targets 245) but misses those -86 dBFS stretches, 22 % of the clip together: a wrong reading
  costs more than a lost one.
- R2 (`sibilance`, abs level and texture) and R4 (`pause_metrics.gap_residual_readings`,
  speech and mixed windows) are clipped to R0's brickwall, NOT to its programme bandwidth.
  The programme bandwidth is read on the loud frames, which are voiced: no 's' frame is a
  loud frame, and the pauses hold the residual the listener hears above the voiced band.
  Measured read-only on 2026-10-09 (medians over the speech windows, band = R0's
  programme bandwidth, Tele7abc 4490 Hz, SOTI 5040 Hz, Vaccin 7127 Hz):
    * the v4_air set (APL, air shelf at 7.5 kHz +2 / +1 / off; the user picked +1, round 3,
      and heard +2 thin, round 2): R2 abs level with the brickwall reads +0.53 / +0.25 /
      -0.04 (Tele7abc), +0.27 / +0.14 / -0.01 (SOTI), +0.63 / +0.32 / +0.03 (Vaccin, 5 of
      61 windows); clipped to the band +0.13 / +0.06 / -0.02 on Tele7abc: a quarter of it;
    * the round-one Tele7abc cache: R4's HF excess reads cathar alpha 2 on 0.7.3 (heard:
      hiss) +3.11 dB and on 0.7.5 (clean) +0.49 with the brickwall, -0.32 for both clipped
      to the band; R2's texture reads APL baseline / no_air / roformer ("distortion of
      spoken 's'") +2.18..+2.26 and the cathar renders +0.83..+1.34 with the brickwall,
      +0.06..+0.08 against +0.21..+0.33 clipped to the band, the verdict reversed. APL
      no_air reads abs level +0.05 against baseline's +0.72: round one's "thin" on no_air
      was the texture, not a level loss.
  The texture's benign floor (hiss removed above the band moves it -0.1..-0.2 on synthetic
  speech, `sibilance.py`) sits inside the grids' dead zone around the accepted readings.
- R7 and the sync (`file_metrics.file_entries`) run once per pair on the raw, un-matched
  mono pair; the pause VAD that ran (`pause_metrics.resolved_vad_name`) goes into
  `card.meta["pause_vad"]`.
"""

import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from scripts import measure_hum
from scripts.restoration_quality import (
    audio_io,
    balance_metrics,
    dsp_metrics,
    file_metrics,
    pause_metrics,
    sibilance,
    source_profile,
    transient_metrics,
)
from scripts.restoration_quality.gates import GATES, evaluate_gates, flag_failures, hard_failures
from scripts.restoration_quality.scorecard import METRICS, ScoreCard, WindowRow, aggregate

ALL_FAMILIES = ("dsp", "stems", "speech", "mos")
SCHEMA = 1
MOS_TARGET_LUFS = -23.0
PEAK_GUARD = 0.99
UNAVAILABLE_ERRORS = (ImportError, SystemExit, RuntimeError, OSError, ValueError, AttributeError, TypeError, KeyError)
DEFAULT_MAINS_HZ = 50.0
PROFILE_DIR = "profile"
# The capture-profile fields shown as `meta.*` readings (numbers only; the channel state stays in card.meta).
PROFILE_READINGS = ("prog_bandwidth_hz", "brickwall_hz", "mains_hz", "mains_evidence_db", "line_hz", "line_ppm", "line_sd_ppm")
SPEECH_ROUTES = ("speech", "mixed")
# R1's mute guard (module docstring): 20 ms frames, each window's p10 as its floor, the file's
# floor the median of those over seven programme windows or more (a mute up to 7.5 s touches
# three windows at most, so it cannot own a median of seven; five let a 4 s mute own it on a
# 37.5-40 s pair, half of six's on 45 s), else the higher of that median and the p25 of every
# live frame in them; a window with 5 % of its live frames 15 dB under that floor skips R1.
MUTE_FRAME_S = pause_metrics.FRAME_S
MUTE_FLOOR_PERCENTILE = 10.0
MUTE_MEDIAN_WINDOWS = 7
MUTE_SPAN_PERCENTILE = 25.0
MUTE_MARGIN_DB = 15.0
MUTE_MAX_SHARE = 0.05
LEVEL_EPS = 1e-9


class ModelRegistry:
    """Lazy model store: `get(name)` loads on first use, `release()` frees the GPU between families."""

    def __init__(self, device="cuda", models_dir=None):
        self.device = device
        self.models_dir = Path(models_dir) if models_dir else None
        self._loaded = {}

    def get(self, name, loader):
        if name not in self._loaded:
            self._loaded[name] = loader(self.device, self.models_dir)
        return self._loaded[name]

    def release(self):
        self._loaded.clear()
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass


class Pair:
    """The aligned, gain-matched source/output pair and its windows, shared by every family."""

    def __init__(self, source_wav, output_wav, cache_dir, seconds, hop):
        self.cache_dir = Path(cache_dir)
        # Path-based consumers (the stem separator, the trade metric) must see DC-free audio too.
        self.source_wav = audio_io.dc_free_copy(source_wav, self.cache_dir)
        self.output_wav = audio_io.dc_free_copy(output_wav, self.cache_dir)
        source_audio, self.rate = audio_io.load_audio(self.source_wav)
        output_audio, output_rate = audio_io.load_audio(self.output_wav)
        if output_rate != self.rate:
            output_audio = audio_io._resample(audio_io.to_mono(output_audio), output_rate, self.rate)[:, None]
        self.raw_source, self.raw_output = audio_io.to_mono(source_audio), audio_io.to_mono(output_audio)
        self.source, self.output, self.lag = audio_io.align_pair(self.raw_source, self.raw_output)
        self.windows = audio_io.windows(len(self.source), self.rate, seconds, hop)
        self.routes = [audio_io.route_window(self.source[w.slice_of(self.rate)], self.rate) for w in self.windows]
        # The output is routed too: a window with programme whose output reads as silence is dead air.
        self.output_routes = [audio_io.route_window(self.output[w.slice_of(self.rate)], self.rate) for w in self.windows]
        self.source_key, self.output_key = audio_io.file_key(self.source_wav), audio_io.file_key(self.output_wav)
        self._resampled = {}
        # R0, read by the dsp family on first use (`load_profile`).
        self.profile = None

    def at(self, side, rate):
        """`side` ("source" | "output") resampled to `rate` once and cached on disk and in memory."""
        if (side, rate) not in self._resampled:
            mono, key = (self.source, self.source_key) if side == "source" else (self.output, self.output_key)
            self._resampled[(side, rate)] = audio_io.resample_cached(mono, self.rate, rate, self.cache_dir, f"{key}_{self.lag}")
        return self._resampled[(side, rate)]

    def mos_gain(self):
        """One gain that brings the source to -23 LUFS, applied to both sides before the MOS models."""
        lufs, _lra = dsp_metrics.loudness(self.source, self.rate)
        gain = 10.0 ** ((MOS_TARGET_LUFS - lufs) / 20.0) if np.isfinite(lufs) else 1.0
        peak = max(float(np.abs(self.source).max()), float(np.abs(self.output).max()), 1e-9)
        return min(gain, PEAK_GUARD / peak)


# ----------------------------------------------------------------------------- R0, the capture profile


PROFILE_CODE = (source_profile, measure_hum, dsp_metrics, pause_metrics)


def profile_code_hash():
    """A short hash of the code a capture profile comes from: a cached profile from other code is never served."""
    digest = hashlib.sha256()
    for module in PROFILE_CODE:
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()[:12]


def profile_path(cache_dir, source_key):
    """Where the capture profile of the source with `source_key` is cached."""
    return Path(cache_dir) / PROFILE_DIR / f"{source_key}_{profile_code_hash()}.json"


def read_profile(path):
    """The cached profile at `path`, or None when it is missing or unreadable."""
    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return profile if isinstance(profile, dict) else None


def write_profile(path, profile):
    """Writes `profile` to `path` through a temporary file, so a parallel scorer never reads half of it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(profile, default=_json_default))
    os.replace(temporary, path)


def load_profile(pair):
    """R0 for the pair's source: from the cache, else read on the raw multichannel file and cached."""
    path = profile_path(pair.cache_dir, pair.source_key)
    profile = read_profile(path)
    if profile is None:
        audio, rate = audio_io.load_audio(pair.source_wav)
        profile = source_profile.capture_profile(audio, rate)
        write_profile(path, profile)
    pair.profile = profile
    return profile


def _profile(pair):
    """The pair's capture profile, `{}` before the dsp family reads it (or on a pair that has none)."""
    return getattr(pair, "profile", None) or {}


def _field(pair, name, default=None):
    value = _profile(pair).get(name)
    return default if value is None else value


# ----------------------------------------------------------------------------- windows


def _rows_for(pair):
    output_routes = getattr(pair, "output_routes", None) or pair.routes
    return [
        WindowRow(w.index, w.start_s, w.end_s, route, output_route=out) for w, route, out in zip(pair.windows, pair.routes, output_routes)
    ]


def _dsp_window(pair, row, mute_floor=None):
    """Every DSP guardrail on one window; paired readings go on the output side with a zero source.

    `mute_floor` is the source's floor (`mute_floor_db`): R1 reads None on a window that holds
    an analog mute under it (`holds_mute`); None reads R1 on every window.
    """
    sl = row_slice(pair, row)
    src, out = pair.source[sl], pair.output[sl]
    row.source["dsp.residual_noise_db"], row.output["dsp.residual_noise_db"] = 0.0, dsp_metrics.residual_noise_db(src, out)
    row.source["dsp.lkr"], row.output["dsp.lkr"] = 0.0, dsp_metrics.lkr_musical_noise(src, out, pair.rate)
    row.source["dsp.dropouts"], row.output["dsp.dropouts"] = 0.0, float(dsp_metrics.dropout_count(src, out, pair.rate))
    for name, (src_ratio, out_ratio) in dsp_metrics.loud_band_ratios_db(src, out, pair.rate).items():
        row.source[f"dsp.{name}"], row.output[f"dsp.{name}"] = src_ratio, out_ratio
    for name, (src_value, out_value) in dsp_metrics.pause_dynamics(src, out, pair.rate).items():
        row.source[f"dsp.{name}"], row.output[f"dsp.{name}"] = src_value, out_value
    tilt = row.output.get("dsp.pause_tilt_db")
    row.source["dsp.pause_tilt_abs_db"], row.output["dsp.pause_tilt_abs_db"] = 0.0, (None if tilt is None else abs(tilt))
    _both(row, "dsp.clicks_per_s", dsp_metrics.click_density, src, out, pair.rate)
    _both(row, "dsp.whistle_db", dsp_metrics.whistle_line_db, src, out, pair.rate, _field(pair, "line_hz", dsp_metrics.WHISTLE_HZ))
    _both(row, "dsp.hum_excess_db", dsp_metrics.hum_excess_db, src, out, pair.rate, _field(pair, "mains_hz", DEFAULT_MAINS_HZ))
    _balance(row, balance_window(src, out, pair.rate, _field(pair, "prog_bandwidth_hz"), mute_floor))


def _both(row, name, function, src, out, rate, *args):
    row.source[name], row.output[name] = function(src, rate, *args), function(out, rate, *args)


def balance_window(src, out, rate, bandwidth_hz, mute_floor=None):
    """R1's readings on one window, every one None when the source holds an analog mute under `mute_floor`."""
    if holds_mute(src, rate, mute_floor):
        return dict.fromkeys(balance_metrics.READINGS)
    return balance_metrics.balance_readings(src, out, rate, bandwidth_hz)


def _balance(row, readings):
    """R1's `readings` on one window (paired: 0 on the source side) and the presence-or-air reading the bright / dull flags read."""
    for name, value in readings.items():
        row.source[f"dsp.{name}"], row.output[f"dsp.{name}"] = 0.0, value
    top = top_db(readings["balance_presence_db"], readings["balance_air_db"])
    row.source["dsp.balance_top_db"], row.output["dsp.balance_top_db"] = 0.0, top


def top_db(presence, air):
    """Presence or air, whichever moved further from the source; None when neither was read."""
    read = [value for value in (presence, air) if value is not None]
    return max(read, key=abs) if read else None


def mute_floor_db(pair):
    """The source's floor, from the programme windows' 20 ms levels; None with no live frame.

    Seven programme windows or more: the median of each window's p10, where its pauses sit (the
    hiss on a linear track, the floor on a Hi-Fi one). A mute up to 7.5 s touches at most three
    windows, so it cannot own the median of seven. Fewer: the higher of that median and the p25
    of every live frame in them (`span_floor_db`), the median taking the higher middle p10 of an
    even count, so a mute sets the floor only when it fills more than half the windows and a
    quarter of the span too. Either alone missed one: a 15 s pair has two windows, both over its
    last 7.5 s, so a 2 s mute there was the median and the guard never
    fired; a blank lead-in counts in the span of the window it opens, so 4 s of blank before
    11 s of speech was the p25. Windows routed as silence (a blank stretch whole) are left out
    unless every window is one.
    """
    windows = programme_windows(pair)
    if len(windows) >= MUTE_MEDIAN_WINDOWS:
        return median_floor_db(pair, windows)
    return _higher(median_floor_db(pair, windows, upper=True), span_floor_db(pair, windows))


def _higher(*floors):
    """The highest of the floors that were read, None when none was."""
    read = [floor for floor in floors if floor is not None]
    return max(read) if read else None


def median_floor_db(pair, windows, upper=False):
    """The median over `windows` of each one's p10 (`quiet_level_db`), None when none has a live frame.

    `upper` takes the higher of the two middle p10s of an even count instead of their mean: a mute
    over exactly half the windows of a short pair otherwise sets the floor half-way between itself
    and the hiss in dB, which caught a -90 dBFS lead-in but not a -65 dBFS one under -40 dBFS hiss.
    """
    floors = [quiet_level_db(pair.source[w.slice_of(pair.rate)], pair.rate) for w in windows]
    floors = [floor for floor in floors if floor is not None]
    if not floors:
        return None
    return float(np.percentile(floors, 50, method="higher" if upper else "linear"))


def span_floor_db(pair, windows):
    """The p25 of the live 20 ms levels over the source samples `windows` cover (each sample once), None with none.

    Over 10.7 % of a window a mute owns R1's quiet frames; the p25 stays off the mute while it
    fills under a quarter of the covered span (3.75 s of 15 s, 5 s of 20 s). The covered span
    holds the blank part of a window a lead-in opens, so `mute_floor_db` takes the median too.
    """
    covered = np.zeros(len(pair.source), dtype=bool)
    for window in windows:
        covered[window.slice_of(pair.rate)] = True
    levels = live_levels_db(pair.source[covered], pair.rate)
    return float(np.percentile(levels, MUTE_SPAN_PERCENTILE)) if levels.size else None


def programme_windows(pair):
    """The windows not routed as silence, or every window when all of them are."""
    kept = [w for w, route in zip(pair.windows, pair.routes) if route != "silence"]
    return kept or list(pair.windows)


def quiet_level_db(mono, rate):
    """The p10 of `live_levels_db`, or None when no frame is live."""
    levels = live_levels_db(mono, rate)
    return float(np.percentile(levels, MUTE_FLOOR_PERCENTILE)) if levels.size else None


def live_levels_db(mono, rate):
    """The 20 ms frame levels in dB within R1's 90 dB of the loudest one: digital silence and padding are no mute."""
    levels = 20.0 * np.log10(dsp_metrics.frame_levels(mono, max(1, int(MUTE_FRAME_S * rate))) + LEVEL_EPS)
    return levels[levels > levels.max() - balance_metrics.DYNAMIC_RANGE_DB] if levels.size else levels


def holds_mute(mono, rate, floor_db):
    """Whether a source window holds an analog mute: 5 % of its live 20 ms frames 15 dB or more under `floor_db`."""
    if floor_db is None:
        return False
    levels = live_levels_db(mono, rate)
    return bool(levels.size) and float(np.mean(levels <= floor_db - MUTE_MARGIN_DB)) >= MUTE_MAX_SHARE


def _listener_window(pair, row):
    """The listener readings on one window: the pauses as the ear hears them, the 's', and a window that fell silent."""
    sl = row_slice(pair, row)
    src, out = pair.source[sl], pair.output[sl]
    brickwall = _field(pair, "brickwall_hz")
    readings = {**pause_metrics.pause_readings(src, out, pair.rate), **sibilance.sib_readings(src, out, pair.rate, brickwall)}
    readings.update(_route_readings(row.route, src, out, pair.rate, brickwall))
    for name, (src_value, out_value) in readings.items():
        row.source[f"dsp.{name}"], row.output[f"dsp.{name}"] = src_value, out_value
    _abs_delta(row, "dsp.sib_centroid_hz", "dsp.sib_centroid_abs_hz")
    silent = row.output_route == "silence" and row.route != "silence"
    row.source["dsp.output_silent"], row.output["dsp.output_silent"] = 0.0, 1.0 if silent else 0.0


def _route_readings(route, src, out, rate, brickwall):
    """The readings only some routes carry: the attacks on music, the residual in the true pauses on speech."""
    readings = {}
    if route in ("music", "mixed"):
        readings.update(transient_metrics.transient_readings(src, out, rate))
    if route in SPEECH_ROUTES:
        readings.update(pause_metrics.gap_residual_readings(src, out, rate, bandwidth_hz=brickwall))
    return readings


def _abs_delta(row, name, abs_name):
    """The rankable form of a signed paired reading: |output - source| on the output side, zero on the source side."""
    src_value, out_value = row.source.get(name), row.output.get(name)
    missing = src_value is None or out_value is None
    row.source[abs_name], row.output[abs_name] = 0.0, (None if missing else abs(out_value - src_value))


def row_slice(pair, row):
    return slice(int(round(row.start_s * pair.rate)), int(round(row.end_s * pair.rate)))


# ----------------------------------------------------------------------------- families


def _dsp_family(pair, card, _registry):
    profile = load_profile(pair)
    mute_floor = mute_floor_db(pair)
    for row in card.rows:
        _dsp_window(pair, row, mute_floor)
        _listener_window(pair, row)
    _zimtohrli(pair, card)
    src_lufs, src_lra = dsp_metrics.loudness(pair.raw_source, pair.rate)
    out_lufs, out_lra = dsp_metrics.loudness(pair.raw_output, pair.rate)
    card.file["file.lufs"] = {"source": src_lufs, "output": out_lufs, "delta": out_lufs - src_lufs}
    card.file["file.lra"] = {"source": src_lra, "output": out_lra, "delta": out_lra - src_lra}
    trade = dsp_metrics.trade(pair.source_wav, pair.output_wav) or {}
    for key, value in trade.items():
        card.file[f"file.{key}"] = {"source": 0.0, "output": value, "delta": value}
    card.file.update(file_metrics.file_entries(pair.raw_source, pair.raw_output, pair.rate))
    card.file.update(dropout_entry(pair))
    card.file.update(profile_entries(profile))
    card.meta.update({"source_profile": profile, "pause_vad": pause_metrics.resolved_vad_name()})


def dropout_entry(pair):
    """`file.dropouts`: the holes over the whole aligned pair, where the window median of `dsp.dropouts` goes blind.

    One hole sits in at most two of the 15 s windows (2 of 40 on a 300 s cut), so a few holes
    in a long file leave every window median at 0 (calibration v3, `dsp_metrics.dropout_count`).
    """
    holes = float(dsp_metrics.dropout_count(pair.source, pair.output, pair.rate))
    return {"file.dropouts": {"source": 0.0, "output": holes, "delta": holes}}


def profile_entries(profile):
    """The capture profile's numbers as `meta.*` file entries, on the source side only (a profile is not a change)."""
    return {f"meta.{name}": {"source": profile[name]} for name in PROFILE_READINGS if profile.get(name) is not None}


_UNAVAILABLE_SAID = set()


def _zimtohrli(pair, card):
    """The 48 kHz psychoacoustic distance on the loud frames; a missing binding costs only this reading, said once."""
    from scripts.restoration_quality import judges

    try:
        judges.score_zimtohrli(pair, card)
    except UNAVAILABLE_ERRORS as exc:
        if "zimtohrli" not in _UNAVAILABLE_SAID:
            _UNAVAILABLE_SAID.add("zimtohrli")
            print(f"dsp.zimtohrli_loud unavailable: {type(exc).__name__}: {exc}")


def _stems_family(pair, card, registry):
    from scripts.restoration_quality import stem_metrics

    stem_metrics.score(pair, card, registry)


def _speech_family(pair, card, registry):
    from scripts.restoration_quality import speech_models

    speech_models.score(pair, card, registry)


def _mos_family(pair, card, registry):
    from scripts.restoration_quality import mos_models

    mos_models.score(pair, card, registry)


FAMILIES = {"dsp": _dsp_family, "stems": _stems_family, "speech": _speech_family, "mos": _mos_family}


def _run_family(name, pair, card, registry):
    """Runs one family, recording success or the reason it could not run."""
    try:
        FAMILIES[name](pair, card, registry)
        card.families[name] = "ok"
    except UNAVAILABLE_ERRORS as exc:
        card.families[name] = f"unavailable: {type(exc).__name__}: {exc}"
    finally:
        registry.release()


def _file_aggregate(card):
    """File-level readings in the same shape as window aggregates so gates can read them."""
    for name, entry in card.file.items():
        card.aggregate[name] = {
            side: {"median": float(entry[side]), "tail": float(entry[side]), "n": 1}
            for side in ("source", "output", "delta")
            if entry.get(side) is not None
        }


def score_pair(
    source_wav,
    output_wav,
    *,
    windows=15.0,
    hop=7.5,
    families=ALL_FAMILIES,
    registry=None,
    cache_dir="experiments/quality_cache",
    gates=GATES,
    language="ro",
):
    """Scores `output_wav` against `source_wav`; returns a ScoreCard with rows, aggregates and verdicts."""
    registry = registry or ModelRegistry()
    registry.language = language
    pair = Pair(source_wav, output_wav, cache_dir, windows, hop)
    card = ScoreCard(rows=_rows_for(pair))
    card.lag = pair.lag
    for name in (family for family in ALL_FAMILIES if family in families):
        _run_family(name, pair, card, registry)
    card.aggregate = aggregate(card.rows, METRICS)
    _file_aggregate(card)
    card.verdicts = evaluate_gates(card.aggregate, gates, card.speaker_floor)
    card.passed = not hard_failures(card.verdicts)
    return card, pair


def card_to_dict(card, path):
    """A JSON-ready view of one card."""
    return {
        "path": str(path),
        "lag_samples": getattr(card, "lag", 0),
        "families": card.families,
        "rows": [
            {
                "window": r.window,
                "start_s": r.start_s,
                "end_s": r.end_s,
                "route": r.route,
                "output_route": r.output_route,
                "source": r.source,
                "output": r.output,
                "delta": r.delta(),
            }
            for r in card.rows
        ],
        "file": card.file,
        "aggregate": card.aggregate,
        "speaker_floor": card.speaker_floor,
        "verdicts": card.verdicts,
        "hard_failures": hard_failures(card.verdicts),
        "listener_flags": flag_failures(card.verdicts),
        "passed": getattr(card, "passed", None),
        "meta": getattr(card, "meta", {}),
    }


def score_variants(source, outputs, *, cache_dir="experiments/quality_cache", gates=GATES, **options):
    """The JSON document for one source and several labelled outputs; cards and pairs are returned for the listening set."""
    cache_dir = Path(cache_dir)
    source_wav = audio_io.extract_wav(source, cache_dir)
    result = {
        "schema": SCHEMA,
        "source": {"path": str(source), "wav": str(source_wav), "key": audio_io.file_key(source_wav)},
        "variants": {},
    }
    cards = {}
    for label, path in outputs.items():
        output_wav = audio_io.extract_wav(path, cache_dir)
        card, pair = score_pair(source_wav, output_wav, cache_dir=cache_dir, gates=gates, **options)
        result["variants"][label] = card_to_dict(card, path)
        cards[label] = (card, pair)
    _describe(result, next(iter(cards.values()), None), gates, options)
    return result, cards


def _describe(result, first, gates, options):
    """The document's windows, the source's capture profile, and the metric and gate registries it was scored with."""
    card, pair = first or (None, None)
    result["windows"] = {
        "seconds": options.get("windows", 15.0),
        "hop": options.get("hop", 7.5),
        "count": len(card.rows) if card else 0,
    }
    if _profile(pair):
        result["source"]["profile"] = _profile(pair)
    result.update(_registries(gates))


def _registries(gates):
    """`{"metrics": ..., "gates": ...}`: what every reading means and every gate checked."""
    return {
        "metrics": {name: _metric_entry(spec) for name, spec in METRICS.items()},
        "gates": {name: _gate_entry(gate) for name, gate in gates.items()},
    }


def _gate_entry(gate):
    """A gate as the JSON document records it; an uncalibrated starting threshold says so."""
    return {
        "metric": gate.metric,
        "stat": gate.stat,
        "op": gate.op,
        "threshold": gate.threshold,
        "severity": gate.severity,
        "calibrated": getattr(gate, "calibrated", True),
    }


def _metric_entry(spec):
    """A metric's registry entry for the JSON document; a two-sided one carries its neutral target."""
    entry = {"family": spec.family, "better": spec.better, "unit": spec.unit}
    if spec.better == "two-sided":
        entry["target"] = spec.target
    return entry


def aggregates_for_tuning(result, label):
    """Flat `{metric.side.stat: value}` for one variant, the shape a sweep ranks on."""
    variant = result["variants"][label]
    flat = {}
    for name, entry in variant["aggregate"].items():
        flat.update(_flat_entry(name, entry))
    flat["gates.hard_failures"] = float(len(variant["hard_failures"]))
    flat["gates.listener_flags"] = float(len(variant.get("listener_flags", [])))
    flat["gates.passed"] = 1.0 if variant["passed"] else 0.0
    return flat


FLAT_STATS = ("median", "tail", "max", "p95")


def _flat_entry(name, entry):
    return {
        f"{name}.{side}.{stat}": entry[side][stat]
        for side in ("output", "delta")
        if entry.get(side)
        for stat in FLAT_STATS
        if stat in entry[side]
    }


def write_result(result, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
