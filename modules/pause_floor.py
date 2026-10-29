"""The pause floor keeper: a restored pause keeps its own, quieter texture instead of collapsing.

The listener called the `auto_pure_linear` outputs "silent in pauses". The harness read it as
the inter-word gaps' air falling 20 dB under the source's (`dsp.gap_air_db` -27 against -6)
and the deep pauses sinking 36-41 dB further under the speech than they were: the mask-based
denoiser leaves near-digital silence, and the polish expander pushes whatever is left a
further 7-10 dB down. Nothing in either engine put a floor back.

This stage does. In the frames the SOURCE calls quiet (at or under its `quiet_percentile`
level), an attenuated copy of the source is added under the restored audio so the pause sits
`fill_db` under the source's own pause level -- the tape's real texture, quieter, never
louder than the source, and never touched where the source carries programme. Only the
deficit is filled: where the restored audio already sits at or above the target, the gain is
zero and the samples pass through to the bit. The reference is the pre-conditioned audio the
engine started from, so the fill carries no hum or line whistle the notches already took.

Two settings choose how the fill is levelled and what it carries; their defaults keep every
output's bytes. `pause_floor_target` 'cluster' (the default) aims every pause at one level,
the median of the quiet frames minus `fill_db`. The diagnosis of 2026-10-11 found three ways
it misses: on dense speech or music that median sits 9-15 dB over the true pauses; where the
neural stage passed the floor, the fill adds on top of it coherently (+5.6 dB over the source
on a music clip); and the separator's peak normalisation drops the programme up to 10.6 dB
across the neural stage, so a target read on the source alone sits that much too loud.
'frame' aims each quiet frame at `fill_db` under the source frame itself, scaled by the
restored-over-source programme gain on the source's loud frames (`programme_gain`), and fills
no frame whose restored level already meets that target.

`pause_floor_shape` 'flat' (the default) fills with the source as it is, its tape hiss
included: the fill put the hiss back into pauses the Mel-RoFormer had taken to -93 dB.
'shelf' takes the fill from a copy of the source cut `pause_floor_hf_cut_db` above
`pause_floor_shelf_hz`, 'lowpass' from a copy low-passed there (`shaper_for`), so a pause
keeps its low room tone and loses the hiss. With a shaped fill the deficit is read on the
restored audio through the same shaping: a hiss burst the neural stage left in a pause holds
no low band, so it never stands in for the body the fill keeps.

`pause_floor_deficit` 'broadband' (the default) reads that deficit on the shaped audio as a
whole, where the low band carries most of the power: a pause whose room tone the chain left
at its target is not filled, and the highs the mask emptied stay at digital silence (the IA
hunt's Pavarotti documentary at fill 24: low band -81 dB against a -82 dB target, highs at
-127 dB, so the pump read +46 dB). 'bands' splits the shelf at its corner and lifts each band
to its own target, the low band `fill_db` under the source's and the highs `fill_db` plus the
cut under the source's highs. It reaches only the shelves; a flat or low-passed fill has no
second band to fill.

With `AI_RESTORE_EVENT_LOG=<dir>` set, the stage writes a record into that folder
(`modules/event_log.py`), in seconds on the restored file's timeline, with the fill depth,
the quiet percentile, the target and the shape, and how many frames were filled or why none
were. Two sets of spans:
`events_s` is the pauses found, the runs of 20 ms frames the stage calls quiet on the
source (the smoothed quiet weight over 0.5); `filled_s` is the frames the fill lifts (a
gain above zero), which can also take the frame just outside each pause and the frames
around a lone quiet one, where the three-frame smoothing leaves a weight of 1/3. The
per-sample gain reaches up to one frame (20 ms) past each end of a `filled_s` run (the
interpolation between frame centres and the 20 ms smoothing) and is exactly zero beyond.
Switched off, the record says so (`skipped`), and a stage that raised records
`failed: <error>`. Unset, nothing is written and the spans are not even computed.
"""

from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import scipy.signal
import soundfile as sf

from . import event_log, pause_fill_shape
from .config import (
    ENABLE_PAUSE_FLOOR,
    PAUSE_FLOOR_DEFICIT,
    PAUSE_FLOOR_FILL_DB,
    PAUSE_FLOOR_HF_CUT_DB,
    PAUSE_FLOOR_QUIET_PERCENTILE,
    PAUSE_FLOOR_SHAPE,
    PAUSE_FLOOR_SHELF_HZ,
    PAUSE_FLOOR_SUPPRESSED_ONLY,
    PAUSE_FLOOR_TARGET,
)
from .hygiene import atomic_target
from .impulse_repair import _spans
from .stage_names import stage_output
from .utils import is_valid_audio, log_msg

FRAME_S = 0.02
# A pause's frames sit within this many dB of the quietest ones; the speech 20 dB and more above.
QUIET_MARGIN_DB = 10.0
MAX_LAG_FRAMES = 10
MIN_QUIET_FRAMES = 50
MAX_LENGTH_DIFF_S = 0.5
MIN_DEFICIT_DB = 0.5
DEFICIT_SHARE = 0.95
SMOOTH_FRAMES = 3
BLOCK_SAMPLES = 1 << 20
STAGE_FAILURES = (OSError, RuntimeError, ValueError, MemoryError)
STAGE = "pause_floor"
# The 'frame' target: the restored programme's level is read on the source's frames at or over this percentile.
LOUD_PERCENTILE = 70.0
FRAME_TARGET = "frame"
# The shaped fills: Butterworth low-passes run forward and backward (zero phase, the magnitude
# squared: -6 dB at the corner). The shelf's 4th order gives 48 dB/octave under the cut, the
# low-pass's 2nd order a Linkwitz-Riley 4th order (24 dB/octave), as prototyped (P1b, P1c).
SHELF_ORDER = 4
LOWPASS_ORDER = 2
HISS_SHELF = "hiss_shelf"
SHAPED = ("shelf", "lowpass", HISS_SHELF)
# The shapes that cut above the corner rather than drop it.
SHELVES = ("shelf", HISS_SHELF)
# pause_floor_deficit: the deficit read per band (low band, highs) instead of on the shaped audio as a whole.
BANDS = "bands"
# The corner never reaches the Nyquist frequency, where a Butterworth design is undefined.
MAX_CORNER_SHARE = 0.45
# Samples of real audio read either side of a shaped block: the forward-backward filter's
# start-up dies out well inside them (about 500 samples at a 3 kHz corner, 7500 at 200 Hz), so
# block after block matches the whole file shaped at once to rounding.
SHAPE_CONTEXT_SAMPLES = 1 << 14
# The output's name carries a non-default fill, so a render at other settings is never reused.
SHAPE_TAGS = {"shelf": "_shelf{hz:g}c{cut:g}", "lowpass": "_lowpass{hz:g}", HISS_SHELF: "_hshelf{hz:g}c{cut:g}"}
SHAPE_NOTES = {
    "shelf": ", the hiss cut {cut:g} dB above {hz:g} Hz",
    "lowpass": ", low-passed at {hz:g} Hz",
    HISS_SHELF: ", the hiss (not the programme) cut {cut:g} dB above {hz:g} Hz",
}


class Profile(NamedTuple):
    """The fill's settings for one file (`profile_for`): `target` and `shape` as `pause_floor_target` and
    `pause_floor_shape`; `fill_db`, `shelf_hz` and `cut_db` as `pause_floor_fill_db`, `pause_floor_shelf_hz` and
    `pause_floor_hf_cut_db`; `deficit` as `pause_floor_deficit`."""

    target: str
    shape: str
    fill_db: float
    shelf_hz: float
    cut_db: float
    deficit: str = "broadband"


def release_profile():
    """The fill every release has run, the keys' defaults: what a file the reference suppressor did not run on takes
    while `pause_floor_suppressed_only` is on. A function, so importing this post-cache module calls nothing."""
    return Profile("cluster", "flat", 12.0, 3000.0, 18.0)


def configured():
    """The configured fill (`Profile`)."""
    return Profile(
        PAUSE_FLOOR_TARGET, PAUSE_FLOOR_SHAPE, PAUSE_FLOOR_FILL_DB, PAUSE_FLOOR_SHELF_HZ, PAUSE_FLOOR_HF_CUT_DB, PAUSE_FLOOR_DEFICIT
    )


def profile_for(suppressed, cut_db=None, fill_db=None):
    """The fill for one file: the configured one, its shelf cut `cut_db` and its depth `fill_db` when given (cathar's
    `cathar_pause_floor_hf_cut_db` and `cathar_pause_floor_fill_db`), or the release fill while
    `pause_floor_suppressed_only` is on and the reference suppressor did not run on the file (`suppressed` False)."""
    if PAUSE_FLOOR_SUPPRESSED_ONLY and not suppressed:
        return release_profile()
    own = {key: float(value) for key, value in (("cut_db", cut_db), ("fill_db", fill_db)) if value is not None}
    return configured()._replace(**own)


class Shaper(NamedTuple):
    """A shaped fill's filter: `wet` of a zero-phase low band (`sos`) plus `dry` of everything above it."""

    sos: np.ndarray
    dry: float
    wet: float = 1.0


class Envelopes(NamedTuple):
    """The frame levels the gains are read on, on one timeline (the lag taken out, trimmed to the overlap).

    `reference` and `restored` are broadband; `fill` is the audio the fill adds (the shaped
    source, or the source itself when the fill is flat) and `band` the restored audio through
    the same shaping. `weight` is the smoothed quiet weight on the source.
    """

    reference: np.ndarray
    restored: np.ndarray
    fill: np.ndarray
    band: np.ndarray
    weight: np.ndarray
    lag: int


class Shaping(NamedTuple):
    """What the fill is taken from and how it and the deficit are shaped.

    `source` is the file the fill copies (the source, or its hiss shelved on 'hiss_shelf');
    `fill` the shaper it goes through on the way (None when flat or already shaped); `band` the
    shaper the restored audio's deficit is read through (None when flat).
    """

    source: Path
    fill: Optional[Shaper]
    band: Optional[Shaper]


class Fill(NamedTuple):
    """What the stage found.

    `gain` is the per-sample gain (None when it fills nothing), `outcome` the frames it fills
    or why it fills none, `found` the spans for the event log, and `programme_gain_db` the
    restored programme's gain the 'frame' target read (None on 'cluster').
    """

    gain: Optional[np.ndarray]
    outcome: object
    found: tuple
    programme_gain_db: Optional[float]


def _frame_rms(blocks, frame):
    """RMS of every whole `frame`-sample frame of a stream of mono float64 blocks, carried across block edges."""
    levels, carry = [], np.zeros(0)
    for block in blocks:
        mono = np.concatenate((carry, block))
        usable = (len(mono) // frame) * frame
        levels.append(np.sqrt(np.mean(mono[:usable].reshape(-1, frame) ** 2, axis=1)))
        carry = mono[usable:]
    return np.concatenate(levels)


def frame_levels(wav_path, frame):
    """RMS of every whole `frame`-sample frame of the channel-mean, streamed; and the file's rate, channels, length."""
    with sf.SoundFile(str(wav_path)) as handle:
        blocks = handle.blocks(blocksize=BLOCK_SAMPLES, dtype="float32", always_2d=True)
        levels = _frame_rms((block.mean(axis=1).astype(np.float64) for block in blocks), frame)
        return levels, handle.samplerate, handle.channels, handle.frames


def shaper_for(shape, rate, corner_hz, cut_db):
    """The filter of a `shape` fill at `rate`, or None for 'flat' (the source as it is).

    'shelf' is a 4th-order Butterworth low-pass at `corner_hz`, run forward and backward, plus
    the rest of the signal `cut_db` down: the low band whole, everything above the corner
    `cut_db` under it ('hiss_shelf' reads the restored deficit through the same filter).
    'lowpass' is a 2nd-order one run the same way with nothing above it (`cut_db` unused). The
    corner stays under 45 % of the rate.
    """
    if shape not in SHAPED:
        return None
    order, dry = (SHELF_ORDER, 10.0 ** (-cut_db / 20.0)) if shape in SHELVES else (LOWPASS_ORDER, 0.0)
    corner = min(float(corner_hz), MAX_CORNER_SHARE * rate)
    return Shaper(scipy.signal.butter(order, corner, btype="low", fs=rate, output="sos"), dry)


def shaped(data, shaper):
    """`data` (samples x channels, float64) through the shaper, padded at its ends as `sosfiltfilt` pads by default."""
    edge = 3 * (2 * len(shaper.sos) + 1)
    low = scipy.signal.sosfiltfilt(shaper.sos, data, axis=0, padlen=min(edge, len(data) - 1))
    return shaper.wet * low + shaper.dry * (data - low)


def shaped_block(handle, start, stop, shaper):
    """Samples [start, stop) of an open file through the shaper, read with SHAPE_CONTEXT_SAMPLES of context either side.

    Inside the file the context carries the filter's start-up, so the block matches the same
    span of the whole file shaped at once; at the file's two ends the filter pads as it pads
    the whole file.
    """
    low, high = max(start - SHAPE_CONTEXT_SAMPLES, 0), min(stop + SHAPE_CONTEXT_SAMPLES, handle.frames)
    handle.seek(low)
    data = handle.read(high - low, dtype="float32", always_2d=True).astype(np.float64)
    inner = slice(start - low, stop - low)
    return shaped(data, shaper)[inner]


def shaped_frame_levels(wav_path, frame, shaper):
    """`frame_levels` of the shaped audio: the RMS of every whole frame of its channel-mean, streamed."""
    with sf.SoundFile(str(wav_path)) as handle:
        spans = ((start, min(start + BLOCK_SAMPLES, handle.frames)) for start in range(0, handle.frames, BLOCK_SAMPLES))
        return _frame_rms((shaped_block(handle, start, stop, shaper).mean(axis=1) for start, stop in spans), frame)


def envelope_lag(reference, restored):
    """Frames by which `restored` trails `reference`, from the envelopes' cross-correlation within +-MAX_LAG_FRAMES."""
    count = min(len(reference), len(restored))
    a = np.log10(reference[:count] + 1e-9)
    b = np.log10(restored[:count] + 1e-9)
    a, b = a - a.mean(), b - b.mean()
    lags = list(range(-MAX_LAG_FRAMES, MAX_LAG_FRAMES + 1))
    return lags[int(np.argmax([_overlap_score(a, b, lag, count) for lag in lags]))]


def _overlap_score(a, b, lag, count):
    """Dot product of the two envelopes with `b` shifted by `lag` frames, over their overlap."""
    a_start, b_start = max(0, -lag), max(0, lag)
    a_stop, b_stop = count - max(0, lag), count - max(0, -lag)
    return float(np.dot(a[a_start:a_stop], b[b_start:b_stop]))


def quiet_weight(reference, percentile):
    """1 on the frames within QUIET_MARGIN_DB of the source's quiet level, 0 elsewhere, smoothed over three frames.

    The quiet level is the `percentile` frame level; the margin takes the whole pause cluster
    with it (a pause's frames sit within a few dB of each other, the speech 20 dB above), so
    the fill covers pauses evenly instead of the scattered quietest frames alone.
    """
    quiet = (reference <= np.percentile(reference, percentile) * 10.0 ** (QUIET_MARGIN_DB / 20.0)).astype(np.float64)
    kernel = np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES
    return np.convolve(quiet, kernel, mode="same")


def cluster_target(fill, weight, fill_db):
    """The 'cluster' target: one level for every pause, `fill_db` under the median quiet frame of `fill`."""
    quiet = weight > 0.5
    floor = np.percentile(fill[quiet], 50.0) if quiet.any() else 0.0
    return floor * 10.0 ** (-fill_db / 20.0)


def programme_gain(reference, restored):
    """The restored programme's amplitude over the source's: the root of their energy ratio on the source's loud frames.

    The separator normalises a peak over 0.9 down to it, so on a hot or declipped source the
    programme leaves the neural stage up to 10.6 dB under its input; the 'frame' target
    follows it. A tape whose loud frames are mostly noise reads low here, and its fill comes
    out quieter.
    """
    loud = reference >= np.percentile(reference, LOUD_PERCENTILE)
    return float(np.sqrt(np.sum(restored[loud] ** 2) / (np.sum(reference[loud] ** 2) + 1e-20)))


def frame_target(fill, gain, fill_db):
    """The 'frame' target: `fill_db` under each frame of `fill`, at the restored programme's level `gain`."""
    return gain * 10.0 ** (-fill_db / 20.0) * fill


def deficit_gains(fill, restored, weight, target):
    """Per-frame gain on `fill` that lifts the restored level to `target`: zero where it already meets it, never above 1."""
    deficit = np.sqrt(np.maximum(target**2 - restored**2, 0.0))
    return weight * np.clip(deficit / (fill + 1e-12), 0.0, 1.0)


def fill_gains(reference, restored, weight, fill_db):
    """Per-frame gain on the source that lifts the restored pause to the target, never above the source itself."""
    return deficit_gains(reference, restored, weight, cluster_target(reference, weight, fill_db))


def gain_curve(frame_gain, length, frame, rate):
    """Per-sample gain from the per-frame gains, 20 ms Hann-smoothed, snapped to exactly zero away from the fill."""
    centres = (np.arange(len(frame_gain)) + 0.5) * frame
    gain = np.interp(np.arange(length), centres, frame_gain)
    ramp = max(int(FRAME_S * rate), 1)
    kernel = np.hanning(ramp + 2)[1:-1]
    padded = np.pad(gain, ramp, mode="edge")
    inner = slice(ramp, ramp + length)
    smoothed = np.convolve(padded, kernel / kernel.sum(), mode="same")[inner]
    smoothed[np.abs(smoothed) < 1e-9] = 0.0
    return smoothed


def fill_file(reference_wav, restored_wav, target_wav, gain, shaper=None):
    """Writes `restored + gain * reference` over the overlap, the restored tail beyond it copied through.

    With a `shaper` the reference is shaped first, block by block.
    """
    return fill_file_parts(reference_wav, restored_wav, target_wav, [(gain, shaper)])


def fill_file_parts(reference_wav, restored_wav, target_wav, parts):
    """`fill_file` with one gained, shaped copy of the reference per (gain, shaper) of `parts`, summed in order."""
    with (
        atomic_target(target_wav) as partial,
        sf.SoundFile(str(reference_wav)) as reference,
        sf.SoundFile(str(restored_wav)) as restored,
        sf.SoundFile(str(partial), "w", samplerate=restored.samplerate, channels=restored.channels, subtype="FLOAT") as out,
    ):
        for start in range(0, restored.frames, BLOCK_SAMPLES):
            block = restored.read(BLOCK_SAMPLES, dtype="float32", always_2d=True)
            stop = start + len(block)
            out.write((block.astype(np.float64) + _parts_block(reference, start, stop, parts)).astype(np.float32))
    return Path(target_wav)


def _parts_block(reference, start, stop, parts):
    """The fill of every part for [start, stop): the first part's samples, each further part added on."""
    fill = _reference_block(reference, start, stop, *parts[0])
    for gain, shaper in parts[1:]:
        fill = fill + _reference_block(reference, start, stop, gain, shaper)
    return fill


def _reference_block(reference, start, stop, gain, shaper=None):
    """The gained reference samples for [start, stop), zero beyond the gain curve or the reference."""
    fill = np.zeros((stop - start, reference.channels), dtype=np.float64)
    usable = min(stop, len(gain), reference.frames) - start
    if usable <= 0:
        return fill
    lo, hi = start, start + usable
    fill[:usable] = _reference_samples(reference, lo, hi, shaper) * gain[lo:hi][:, None]
    return fill


def _reference_samples(reference, start, stop, shaper):
    """The reference's samples [start, stop) as float64, through the shaper when there is one."""
    if shaper is not None:
        return shaped_block(reference, start, stop, shaper)
    reference.seek(start)
    return reference.read(stop - start, dtype="float32", always_2d=True).astype(np.float64)


def _deficit_db(target, restored, weight):
    """How far the restored pauses sit under the target, in dB, on the quiet frames."""
    quiet = weight > 0.5
    if not quiet.any():
        return np.zeros(0)
    return 20.0 * np.log10((np.broadcast_to(target, restored.shape)[quiet] + 1e-12) / (restored[quiet] + 1e-12))


def _shape_mismatch(reference_meta, restored_meta):
    """Why the two files cannot be paired, or None."""
    (rate, channels, frames), (rate2, channels2, frames2) = reference_meta, restored_meta
    if rate != rate2 or channels != channels2:
        return "reference and restored audio differ in rate or channels"
    if abs(frames - frames2) > MAX_LENGTH_DIFF_S * rate:
        return "reference and restored audio differ in length"
    return None


def _nothing_to_fill(env, target, kind):
    """Why the envelopes give the stage nothing to do, or None; `kind` is the target's ('cluster' or 'frame')."""
    if int((env.weight > 0.5).sum()) < MIN_QUIET_FRAMES:
        return "too few quiet frames"
    if _targets_met(_deficit_db(target, env.band, env.weight), kind):
        return "nothing to fill"
    return None


def _targets_met(deficit, kind):
    """Whether the quiet frames' deficits (dB) leave nothing worth filling on a `kind` target.

    On 'cluster', 95 % of the quiet frames within MIN_DEFICIT_DB of the one level. On 'frame',
    no quiet frame MIN_DEFICIT_DB or more short of its own target: most quiet frames of a tape
    whose restored floor survived meet theirs, and a share rule would then leave its few
    emptied pauses dead (abc-primetime-1991: 0.5 s of true pause at -77.6 dB under the source,
    skipped at the 95 % share; five hunt clips kept a 0.1-0.2 s hole under a 10-frame minimum).
    """
    if kind == FRAME_TARGET:
        return not bool((deficit >= MIN_DEFICIT_DB).any())
    return float(np.mean(deficit < MIN_DEFICIT_DB)) >= DEFICIT_SHARE


def _overlap(ref_count, res_count, lag):
    """The slices of the reference's and the restored envelopes that overlap once the lag is taken out."""
    ref_start, res_start = max(-lag, 0), max(lag, 0)
    count = min(ref_count - ref_start, res_count - res_start)
    return slice(ref_start, ref_start + count), slice(res_start, res_start + count)


def _envelopes(levels):
    """`Envelopes` from the (reference, restored, fill, band) frame levels, each on its own file's timeline.

    The lag comes from the broadband pair; the fill and the band follow the file they were read from.
    """
    reference, restored, fill, band = levels
    lag = envelope_lag(reference, restored)
    ref_cut, res_cut = _overlap(len(reference), len(restored), lag)
    weight = quiet_weight(reference[ref_cut], PAUSE_FLOOR_QUIET_PERCENTILE)
    return Envelopes(reference[ref_cut], restored[res_cut], fill[ref_cut], band[res_cut], weight, lag)


def _paired_levels(reference, restored, frame, shaping):
    """The (reference, restored, fill, band) frame levels; `reference` and `restored` are (path, broadband levels).

    Flat, the fill and the band are the broadband pair itself; shaped, the fill's file and the
    restored file are read again through their shapers (`Shaping`).
    """
    (_, reference_levels), (restored_wav, restored_levels) = reference, restored
    if shaping.band is None:
        return reference_levels, restored_levels, reference_levels, restored_levels
    fill = frame_levels(shaping.source, frame)[0] if shaping.fill is None else shaped_frame_levels(shaping.source, frame, shaping.fill)
    return reference_levels, restored_levels, fill, shaped_frame_levels(restored_wav, frame, shaping.band)


def _target(env, profile):
    """The fill's target and the programme gain in dB: per frame on 'frame', one level for every pause (no gain) on 'cluster'."""
    if profile.target != FRAME_TARGET:
        return cluster_target(env.fill, env.weight, profile.fill_db), None
    gain = programme_gain(env.reference, env.restored)
    return frame_target(env.fill, gain, profile.fill_db), float(20.0 * np.log10(gain + 1e-12))


def _target_for(restored_wav, audio_dir, profile):
    output_dir = Path(audio_dir) / "pause_floor"
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{profile.fill_db:g}db_p{PAUSE_FLOOR_QUIET_PERCENTILE:g}{_variant_tag(profile)}"
    return stage_output(output_dir, f"paused_{tag}", restored_wav)


def _variant_tag(profile):
    """'' for the default fill (the cluster target, flat); else the target and the shape with its settings, and
    '_bands' for a shelf whose deficit is read per band."""
    target = "" if profile.target != FRAME_TARGET else f"_{FRAME_TARGET}"
    bands = "_bands" if _banded(profile) else ""
    return target + SHAPE_TAGS.get(profile.shape, "").format(hz=profile.shelf_hz, cut=profile.cut_db) + bands


def _banded(profile):
    """Whether `profile` reads its deficit per band: `pause_floor_deficit` 'bands' on a shelf."""
    return profile.deficit == BANDS and profile.shape in SHELVES


def _note(programme_gain_db, profile):
    """What the log line says the fill aims at."""
    shape = SHAPE_NOTES.get(profile.shape, "").format(hz=profile.shelf_hz, cut=profile.cut_db)
    shape += ", each band to its own target" if _banded(profile) else ""
    if programme_gain_db is None:
        return f"the source's pauses{shape}"
    return f"each source frame at the restored programme's level ({programme_gain_db:+.1f} dB){shape}"


def pause_spans(mask, lag):
    """The runs of a per-frame mask on the aligned envelopes, as frame spans on the restored file's timeline."""
    offset = max(lag, 0)
    return [(start + offset, stop + offset) for start, stop in _spans(mask)]


def _found(weight, gains, lag):
    """The frame spans the event log records: the pauses found and the frames filled; both empty while the log is off."""
    if not event_log.enabled():
        return [], []
    filled = [] if gains is None else pause_spans(gains > 0.0, lag)
    return pause_spans(weight > 0.5, lag), filled


def _restored_gain(levels, timeline, profile):
    """The fill for the restored file's timeline, from the (reference, restored, fill, band) frame levels.

    Without a gain the outcome is why there is none. The spans are the pauses found and the
    frames filled, as frame spans on the restored timeline, read only while the event log is
    on. `timeline` is the restored file's (frame, rate, length).
    """
    env = _envelopes(levels)
    target, programme_gain_db = _target(env, profile)
    lifted, reason = _lift(env, target, profile.target, timeline)
    if lifted is None:
        return Fill(None, reason, _found(env.weight, None, env.lag), programme_gain_db)
    curve, gains = lifted
    return Fill(curve, int((gains > 0).sum()), _found(env.weight, gains, env.lag), programme_gain_db)


def _lift(env, target, kind, timeline):
    """`((per-sample gain, per-frame gains), None)` lifting `env.band` to `target`, or `(None, why there is none)`.

    `kind` is the target's ('cluster' or 'frame'); `timeline` the restored file's (frame, rate, length).
    """
    reason = _nothing_to_fill(env, target, kind)
    if reason is not None:
        return None, reason
    frame, rate, frames = timeline
    gains = deficit_gains(env.fill, env.band, env.weight, target)
    frame_gain = np.concatenate((np.zeros(max(env.lag, 0)), gains))
    return (gain_curve(frame_gain, frames, frame, rate), gains), None


def band_shapers(shaping):
    """`(fills, bands)`: the shapers that split the fill and the restored audio at the shelf's corner, low band first.

    The fill's highs keep the shelf's cut (a 'hiss_shelf' source carries its own, so they pass whole); the
    restored highs are read as they are, against a target `fill_db` plus the cut under the source's highs.
    """
    sos = shaping.band.sos
    fill_dry = 1.0 if shaping.fill is None else shaping.fill.dry
    return (Shaper(sos, 0.0), Shaper(sos, fill_dry, 0.0)), (Shaper(sos, 0.0), Shaper(sos, 1.0, 0.0))


def _band_levels(pair, frame, shaping):
    """Per band (low first), the (reference, restored, fill, band) frame levels of a per-band deficit."""
    (_, reference), (restored_wav, restored) = pair
    fills, bands = band_shapers(shaping)
    return [
        (reference, restored, shaped_frame_levels(shaping.source, frame, fill), shaped_frame_levels(restored_wav, frame, band))
        for fill, band in zip(fills, bands)
    ]


def _band_fill(band_levels, profile, timeline):
    """The `Fill` of a per-band deficit: `gain` holds one per-sample curve per band (zero where a band meets its
    target), the outcome the frames either band lifts."""
    env, programme_gain_db, lifted = _lift_bands(band_levels, profile, timeline)
    live = [found for found, _reason in lifted if found is not None]
    if not live:
        return Fill(None, lifted[0][1], _found(env.weight, None, env.lag), programme_gain_db)
    filled = _either_lifted(live)
    return Fill(_band_curves(lifted, timeline[2]), int(filled.sum()), _found(env.weight, filled, env.lag), programme_gain_db)


def _lift_bands(band_levels, profile, timeline):
    """`(the low band's envelopes, the programme gain in dB, [(lifted, why not)] per band)`."""
    envs = [_envelopes(levels) for levels in band_levels]
    targets = [_target(env, profile) for env in envs]
    return envs[0], targets[0][1], [_lift(env, target, profile.target, timeline) for env, (target, _db) in zip(envs, targets)]


def _band_curves(lifted, length):
    """One per-sample curve per band, all zeros for a band that meets its target."""
    return tuple(np.zeros(length) if found is None else found[0] for found, _reason in lifted)


def _either_lifted(live):
    """1.0 on the frames either band lifts, 0.0 elsewhere."""
    return np.logical_or.reduce([gains > 0.0 for _curve, gains in live]).astype(np.float64)


def _thresholds(profile):
    """The settings every record of the stage carries; the shape's corner and cut only when the fill is shaped, the
    release fill named when `pause_floor_suppressed_only` gave it to a file the suppressor did not run on."""
    shape = {"shelf_hz": profile.shelf_hz, "hf_cut_db": profile.cut_db} if profile.shape in SHAPED else {}
    if _banded(profile):
        shape["deficit"] = BANDS
    if PAUSE_FLOOR_SUPPRESSED_ONLY:
        shape["release_fill"] = profile == release_profile() != configured()
    target = {"target": profile.target, "shape": profile.shape}
    return {"fill_db": profile.fill_db, "quiet_percentile": PAUSE_FLOOR_QUIET_PERCENTILE, **target, **shape}


def _log_pauses(restored_wav, fill, timeline, profile):
    """Writes the pauses found and the frames filled, in seconds, into the event log when it is on.

    The record's `outcome` is what the stage did: the number of frames filled, or the reason
    it filled none; `programme_gain_db` is the restored programme's gain the 'frame' target
    read (null on 'cluster'). `timeline` is the restored file's (frame, rate).
    """
    frame, rate = timeline
    pauses, filled = fill.found
    details = {**_thresholds(profile), "outcome": fill.outcome, "programme_gain_db": fill.programme_gain_db}
    details["filled_s"] = event_log.seconds(_in_samples(filled, frame), rate)
    event_log.write_spans(STAGE, restored_wav, _in_samples(pauses, frame), rate, **details)


def _in_samples(spans, frame):
    """Frame spans as sample spans."""
    return [(start * frame, stop * frame) for start, stop in spans]


def keep_floor(reference_wav, restored_wav, audio_dir, profile):
    """Runs the stage at `profile`; returns the new path, or `restored_wav` with the reason logged."""
    frame = max(1, int(FRAME_S * sf.info(str(restored_wav)).samplerate))
    pair, reason, (rate, frames) = _levels_pair(reference_wav, restored_wav, frame)
    shaping = None if reason else _shaping(pair, (frame, rate), audio_dir, profile)
    fill = Fill(None, reason, ([], []), None) if reason else _fill_for(pair, (shaping, profile), (frame, rate, frames))
    _log_pauses(restored_wav, fill, (frame, rate), profile)
    if fill.gain is None:
        log_msg(f"    [Pause Floor] Skipped: {fill.outcome}.")
        return restored_wav
    target = _target_for(restored_wav, audio_dir, profile)
    if is_valid_audio(target):
        log_msg(f"    [Pause Floor] Reusing {target.name}.")
        return target
    log_msg(
        f"    [Pause Floor] Filling {fill.outcome} quiet frames to {profile.fill_db:g} dB under {_note(fill.programme_gain_db, profile)}."
    )
    return _write_fill((shaping, profile), restored_wav, target, fill.gain)


def _write_fill(plan, restored_wav, target, gain):
    """Writes the filled file: the shaped source at `gain`, or on a per-band deficit its two bands at theirs.

    `plan` is (`Shaping`, `Profile`).
    """
    shaping, profile = plan
    if not _banded(profile):
        return fill_file(shaping.source, restored_wav, target, gain, shaping.fill)
    return fill_file_parts(shaping.source, restored_wav, target, list(zip(gain, band_shapers(shaping)[0])))


def _levels_pair(reference_wav, restored_wav, frame):
    """`((path, broadband frame levels) of the reference and the restored file, why they cannot be paired or None,
    (the restored file's rate, length))`."""
    reference, *reference_meta = frame_levels(reference_wav, frame)
    restored, rate, channels, frames = frame_levels(restored_wav, frame)
    reason = _shape_mismatch(tuple(reference_meta), (rate, channels, frames))
    return ((reference_wav, reference), (restored_wav, restored)), reason, (rate, frames)


def _fill_for(pair, plan, timeline):
    """The `Fill` for a matched pair.

    `pair` is the reference's and the restored file's (path, broadband frame levels); their
    shaped levels are read here when the fill is shaped. `plan` is (`Shaping`, `Profile`);
    `timeline` is the restored file's (frame, rate, length).
    """
    shaping, profile = plan
    if _banded(profile):
        return _band_fill(_band_levels(pair, timeline[0], shaping), profile, timeline)
    return _restored_gain(_paired_levels(*pair, timeline[0], shaping), timeline, profile)


def _shaping(pair, timeline, audio_dir, profile):
    """The `Shaping` of the fill at `profile`; `timeline` is the restored file's (frame, rate).

    'hiss_shelf' takes its fill from the source with its hiss shelved
    (`pause_fill_shape.hiss_shelved`), and falls back to the plain shelf when the source holds
    too little live audio to learn its print from.
    """
    reference_wav = Path(pair[0][0])
    shaper = shaper_for(profile.shape, timeline[1], profile.shelf_hz, profile.cut_db)
    if profile.shape != HISS_SHELF:
        return Shaping(reference_wav, shaper, shaper)
    noise = pause_fill_shape.source_print(reference_wav)
    if noise is None:
        log_msg("    [Pause Floor] No print to shelve the hiss by: the plain shelf fills.")
        return Shaping(reference_wav, shaper, shaper)
    shelved = pause_fill_shape.hiss_shelved(reference_wav, audio_dir, shaper, noise, (profile.shelf_hz, profile.cut_db))
    return Shaping(shelved, None, shaper)


def apply_when_needed(reference_wav, restored_wav, audio_dir, strategy=None, suppressed=False, cut_db=None, fill_db=None):
    """Keeps a floor under the restored pauses; returns the new path, or the restored audio untouched.

    `strategy` is accepted for the chain's uniform stage signature and unused. `suppressed` says
    whether the reference suppressor ran on this file (`reference_suppress.is_output`): with
    `pause_floor_suppressed_only` on, a file it did not run on takes the release fill
    (`profile_for`). `cut_db` and `fill_db` are the configured fill's shelf cut and depth for this
    engine (cathar's own); None takes `pause_floor_hf_cut_db` and `pause_floor_fill_db`.
    """
    del strategy
    profile = profile_for(suppressed, cut_db, fill_db)
    if not ENABLE_PAUSE_FLOOR or profile.fill_db <= 0.0:
        event_log.write_skip(STAGE, restored_wav, "switched off", **_thresholds(profile))
        return restored_wav
    try:
        return keep_floor(reference_wav, restored_wav, audio_dir, profile)
    except STAGE_FAILURES as exc:
        log_msg(f"    [Pause Floor] Skipped after failure: {exc}")
        event_log.write_skip(STAGE, restored_wav, f"failed: {exc}", **_thresholds(profile))
        return restored_wav
