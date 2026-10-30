"""cathar's noise-print mute guard (`cathar_noiseprint_mute_guard`): a digital mute is not tape noise.

The quietest window of an edited upload can be a digital mute (a fade to black, a blank
lead-in, a DVD's silence), and a print learned from it removes nothing. Measured on the 136 IA
clips of experiments/auto_engines_v140/ia_features.json (cathar 0.7.3 outputs, 2026-09-18): 7
picked a 0.75 s window under -70 dBFS (-80.0 to -128.4), and there APL removed a median 9.7 dB
more than cathar against 3.38 dB on the other 114.

A candidate window is a mute when most of its samples hold the value before them (exact zeros,
a held DC level), or when it sits under NOISEPRINT_MUTE_FLOOR_DBFS and at least
NOISEPRINT_MUTE_UNDER_P20_DB under the p20 of the material's 23 ms frame levels. The p20 leaves
out frames under NOISEPRINT_DIGITAL_FLOOR_DBFS, a level no VHS capture's analog floor reaches
(the quietest analog window on the IA corpus, bbc-two-continuity's, sits at -82.7 dBFS; every
local 60 s excerpt's above -65.4), so a mute filling a fifth of a clip cannot become its own
reference: on superstition_202101's print input the mute pulls the p20 of all frames to
-61.9 dBFS against -52.0 without it. An analog floor stays a candidate: bbc-two-continuity's
chosen window sits 0.6 dB under its p20 there.

The guard only removes candidates. The single-window pick keeps cathar's 40-position grid and
order, and the stitched pick its spread over the quietest fifth (taken over the windows kept),
so with nothing rejected either pick is the unguarded one; with everything rejected there is no
print, and the denoise estimates the noise itself. cathar mode alone passes the key
(`cathar.filter_cathar_vhs_pipeline`); every other caller of the noise-print step is unguarded.
"""

import numpy as np
import soundfile as sf

from .utils import log_msg

NOISEPRINT_MUTE_FLOOR_DBFS = -70.0
NOISEPRINT_MUTE_UNDER_P20_DB = 30.0
NOISEPRINT_MUTE_HELD_SHARE = 0.5
NOISEPRINT_MUTE_PERCENTILE = 20.0
NOISEPRINT_DIGITAL_FLOOR_DBFS = -90.0
# 1024 samples at 44.1 kHz, the frame the IA evidence read its p20 on.
NOISEPRINT_MUTE_FRAME_S = 1024 / 44100
# The single-window search's grid (cathar._evaluate_quiet_probes): this many windows at most.
PROBE_POSITIONS = 40
# The stitched pick's blocks (cathar._window_levels): this many hops per read.
BLOCK_HOPS = 240
LOG_PREFIX = "    [Cathar] Noise-print mute guard:"


def db(rms):
    """An RMS level (a number or an array) in dBFS; digital silence reads -240 dBFS, not minus infinity."""
    return 20.0 * np.log10(np.maximum(rms, 1e-12))


def mute_frame(rate):
    """The guard's frame in samples at this rate (1024 at 44.1 kHz)."""
    return max(1, int(round(NOISEPRINT_MUTE_FRAME_S * rate)))


def frame_db(mono, frame):
    """Level in dBFS of every whole frame of `mono`."""
    count = len(mono) // frame
    energies = np.mean(np.asarray(mono[: count * frame], dtype=np.float64).reshape(count, frame) ** 2, axis=1)
    return db(np.sqrt(energies))


def tape_p20_db(levels_db):
    """The p20 of the frame levels above the digital floor, or None when every frame is digital silence."""
    tape = levels_db[levels_db > NOISEPRINT_DIGITAL_FLOOR_DBFS]
    return float(np.percentile(tape, NOISEPRINT_MUTE_PERCENTILE)) if len(tape) else None


def held_share(samples):
    """Share of samples equal to the one before: exact zeros and a held value read 1, tape hiss about 0."""
    return float(np.mean(np.diff(samples) == 0)) if len(samples) > 1 else 1.0


def mute_reason(level_db, held, p20_db):
    """Why a candidate window is a digital mute, or None when it is tape; material all under the digital floor has no p20."""
    if held >= NOISEPRINT_MUTE_HELD_SHARE:
        return f"{held:.0%} of its samples hold one value"
    if level_db >= NOISEPRINT_MUTE_FLOOR_DBFS:
        return None
    if p20_db is None:
        return "the whole material sits under the digital floor"
    return f"{p20_db - level_db:.1f} dB under the frame p20" if level_db <= p20_db - NOISEPRINT_MUTE_UNDER_P20_DB else None


def p20_text(p20_db):
    """The p20 as the log prints it."""
    return "no frame above the digital floor" if p20_db is None else f"frame p20 {p20_db:.1f} dBFS"


def guarded_quiet_probe(mono, win, rate, offset=0):
    """The single-window pick behind the guard: the quietest window of cathar's grid that is not a mute, or None.

    The grid, the RMS and the order are `cathar._evaluate_quiet_probes`'s, so with nothing
    rejected the pick is the same window. `offset` (the searched block's first sample in the
    file) only places the log's times. Needs `len(mono) > win`.
    """
    p20_db = tape_p20_db(frame_db(mono, mute_frame(rate)))
    max_start = len(mono) - win
    kept = []
    for pos in np.linspace(0, max_start, min(PROBE_POSITIONS, max_start + 1), dtype=int):
        start = int(pos)
        stop = start + win
        chunk = mono[start:stop]
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
        reason = mute_reason(float(db(rms)), held_share(chunk), p20_db)
        if reason is None:
            kept.append((start, rms))
            continue
        log_msg(f"{LOG_PREFIX} rejected the window at {(offset + pos) / rate:.2f} s ({db(rms):.1f} dBFS): {reason}")
    return _log_guarded_probe(min(kept, key=lambda item: item[1]) if kept else None, rate, offset, p20_db)


def _log_guarded_probe(best, rate, offset, p20_db):
    """Logs the guarded single-window pick (or that none qualified) and returns its start sample, or None."""
    if best is None:
        log_msg(f"{LOG_PREFIX} every candidate window is a digital mute ({p20_text(p20_db)}); denoising without a learned print")
        return None
    log_msg(f"{LOG_PREFIX} learning from the window at {(offset + best[0]) / rate:.2f} s ({db(best[1]):.1f} dBFS, {p20_text(p20_db)})")
    return best[0]


def mute_evidence(wav_path, hop_frames, window_hops, rate):
    """Per window, laid out as `cathar._window_levels` lays them out: the share of held samples; and the frame p20.

    Streamed in the same blocks as the levels, so window k here is window k there.
    """
    held, levels_db = [], [np.zeros(0)]
    frame = mute_frame(rate)
    with sf.SoundFile(str(wav_path)) as handle:
        for block in handle.blocks(blocksize=hop_frames * BLOCK_HOPS, dtype="float32", always_2d=True):
            mono = block.mean(axis=1)
            hops = len(mono) // hop_frames
            held.extend(np.mean(np.diff(mono[: hops * hop_frames].reshape(hops, hop_frames), axis=1) == 0, axis=1).tolist())
            levels_db.append(frame_db(mono, frame))
    p20_db = tape_p20_db(np.concatenate(levels_db))
    if len(held) < window_hops:
        return np.zeros(0), p20_db
    return np.convolve(np.asarray(held, dtype=np.float64), np.ones(window_hops), mode="valid") / window_hops, p20_db


def rejected_runs(rejected):
    """(first, last) index of every run of consecutive rejected windows."""
    index = np.flatnonzero(rejected)
    if len(index) == 0:
        return []
    breaks = np.flatnonzero(np.diff(index) > 1)
    firsts = np.concatenate(([index[0]], index[breaks + 1]))
    lasts = np.concatenate((index[breaks], [index[-1]]))
    return list(zip(firsts.tolist(), lasts.tolist()))


def _log_rejected_runs(reasons, levels_db, starts, window_frames, rate):
    """One line per run of rejected windows (they overlap at a 0.25 s hop): its span, its quietest window and why."""
    for first, last in rejected_runs(np.array([reason is not None for reason in reasons], dtype=bool)):
        stop = last + 1
        quietest = first + int(np.argmin(levels_db[first:stop]))
        log_msg(
            f"{LOG_PREFIX} rejected {last - first + 1} window(s) from {starts[first] / rate:.2f} "
            f"to {(starts[last] + window_frames) / rate:.2f} s (quietest {levels_db[quietest]:.1f} dBFS: {reasons[quietest]})"
        )


def pick_guarded_pauses(input_wav, windows, count, rate, pick):
    """The stitched pick behind the guard: `pick` (cathar's spread over the quietest fifth) over the windows that are not mutes.

    `windows` is `(levels, starts, hop, window_hops)` as cathar reads them. The fifth is taken
    over the windows the guard keeps, so a long mute cannot fill it; with nothing rejected the
    pick is the unguarded one. An empty result drops the learned print.
    """
    levels, starts, hop, window_hops = windows
    held, p20_db = mute_evidence(input_wav, hop, window_hops, rate)
    levels_db = db(levels)
    reasons = [mute_reason(float(level), float(share), p20_db) for level, share in zip(levels_db, held)]
    keep = np.array([reason is None for reason in reasons], dtype=bool)
    _log_rejected_runs(reasons, levels_db, starts, hop * window_hops, rate)
    picked = pick(levels[keep], starts[keep], hop * window_hops, count)
    _log_guarded_pauses(picked, levels_db[keep], starts[keep], p20_db)
    return picked


def _log_guarded_pauses(picked, kept_db, kept_starts, p20_db):
    """Logs the levels of the pauses the stitched print is learned from and the p20, or that none qualified."""
    if not picked:
        log_msg(f"{LOG_PREFIX} every window is a digital mute ({p20_text(p20_db)}); denoising without a learned print")
        return
    picked_db = kept_db[np.searchsorted(kept_starts, picked)]
    log_msg(f"{LOG_PREFIX} pauses at {picked_db.min():.1f} to {picked_db.max():.1f} dBFS, {p20_text(p20_db)}")
