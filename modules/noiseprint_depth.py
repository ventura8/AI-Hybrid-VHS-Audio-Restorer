"""cathar's noise-print depth guard (`cathar_noiseprint_min_depth_db`): a print learned from programme is not tape noise.

cathar learns its noise print from the quietest stretch of the material: the single quietest
0.75 s on a clip, six stitched 0.75 s pauses from 90 s of material (`cathar._probe_duration_s`).
A 15 s clip with no pause has no quiet stretch, so the window is programme. On the IA hunt
(tune_ia_hunt_v3, 235 clips, 2026-10-12) cathar failed `stems.octave_programme` on 64 of the
176 clips that read, 63 of them on the speech profile; the learned-print subtraction took the
octave on 56, and on the failing speech-profile clips the print's window sat a median 8.3 dB
under the loud frames against 13.0 dB on the passing ones (41 of the 76 within 10 dB fail, 3 of
the 28 at 25 dB or deeper). At alpha 1.0 the subtraction then empties every background cell at
or under the print (5 to 30 dB) while the voice-led cells keep their level.

The guard measures the print's depth: the RMS of the window or windows the print was learned
from, picked again exactly as the extractor picked them, under the material's loud frames (the
p90 of the mono mix's 1024-sample frame RMS, streamed). Under `cathar_noiseprint_min_depth_db`
the print is programme, and `cathar_noiseprint_shallow_action` decides what the denoise gets:

- 'scale': the print's magnitudes (cathar 0.8.0 writes the print as a JSON list of 1025 mean
  STFT magnitudes, 2048 Hann / hop 512, of the probe's left channel) times
  10^(-(D - depth)/20), so the print sits exactly D dB under the loud frames; written beside
  the learned print as `<print>_depth<D>db.np.json`;
- 'drop': no learned print, so `denoise` runs without `--noiseprint`, as the music profile does.

Run through this module at the denoise stage (2026-10-12, the cathar 0.8.0 CLI on CPU on each
clip's own stage input, the 159 speech-profile hunt clips): at D = 20, 'scale' leaves 25 of the
62 failing clips failing and 'drop' 17, neither fails a passing clip, and quiet-frame removal
falls from 5.49 / 3.23 dB (failing / passing) to 1.70 / 1.66 ('scale') and 0.11 / 0.30
('drop'); at D = 13, 38 and 22 still fail, none new. A print deep enough is handed on
untouched, so its command lines and its bytes are the chain's without the guard. The four
local Tata tapes' stitched pauses sit only 13.7-16.9 dB under their loud frames: at 20 the
guard fires on all four and the denoise removes about half the noise ('scale') or none
('drop'); at 13 all four keep their bytes. Off by default (0); cathar mode alone reads the keys.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .utils import log_msg

# The loud frames: the p90 of the mono mix's frame RMS over 1024-sample frames.
LOUD_FRAME = 1024
LOUD_PERCENTILE = 90.0
# Frames per streamed read (`cathar._window_levels` reads 240 hops at a time).
BLOCK_FRAMES = 240
SHALLOW_ACTIONS = ("scale", "drop")
PRINT_SUFFIX = ".np.json"
LOG_PREFIX = "    [Cathar] Noise-print depth guard:"


def db(rms):
    """An RMS level in dBFS; digital silence reads -240 dBFS, not minus infinity."""
    return 20.0 * float(np.log10(max(float(rms), 1e-12)))


def _frame_energies(mono):
    """Mean square of every whole LOUD_FRAME frame of a mono block."""
    count = len(mono) // LOUD_FRAME
    frames = np.asarray(mono[: count * LOUD_FRAME], dtype=np.float64).reshape(count, LOUD_FRAME)
    return np.mean(frames**2, axis=1)


def loud_level(wav_path):
    """The p90 of the material's 1024-sample frame RMS (mono mix), streamed block by block; 0 when no frame fits."""
    energies = [np.zeros(0)]
    with sf.SoundFile(str(wav_path)) as handle:
        for block in handle.blocks(blocksize=LOUD_FRAME * BLOCK_FRAMES, dtype="float32", always_2d=True):
            energies.append(_frame_energies(block.mean(axis=1)))
    levels = np.sqrt(np.concatenate(energies))
    return float(np.percentile(levels, LOUD_PERCENTILE)) if len(levels) else 0.0


def windows_level(wav_path, starts, frames):
    """RMS of the mono mix over the windows (`frames` samples from each start), read one window at a time."""
    if not starts:
        raise ValueError("no probe window to measure")
    total, count = 0.0, 0
    for start in starts:
        block, _rate = sf.read(str(wav_path), dtype="float32", start=int(start), frames=int(frames), always_2d=True)
        mono = block.mean(axis=1).astype(np.float64)
        total += float(np.sum(mono**2))
        count += len(mono)
    return float(np.sqrt(total / count)) if count else 0.0


def print_depth_db(wav_path, windows):
    """`(depth, loud dBFS, print dBFS)`: how far the print's windows sit under the loud frames. `windows` is `(starts, frames)`."""
    starts, frames = windows
    probe_db = db(windows_level(wav_path, starts, frames))
    loud_db = db(loud_level(wav_path))
    return loud_db - probe_db, loud_db, probe_db


def scaled_print_path(np_path, min_depth_db):
    """The scaled print's path: beside the learned one, its name carrying the depth it was scaled to."""
    np_path = Path(np_path)
    stem = np_path.name[: -len(PRINT_SUFFIX)] if np_path.name.endswith(PRINT_SUFFIX) else np_path.stem
    return np_path.with_name(f"{stem}_depth{float(min_depth_db):g}db{PRINT_SUFFIX}")


def scale_print(np_path, gain_db, out_path):
    """Writes the print's magnitudes times 10^(gain_db/20) to out_path; refuses anything but a list of numbers."""
    magnitudes = json.loads(Path(np_path).read_text(encoding="utf-8"))
    if not isinstance(magnitudes, list) or not all(isinstance(value, (int, float)) for value in magnitudes):
        raise ValueError(f"{Path(np_path).name} is not a list of magnitudes")
    scaled = np.asarray(magnitudes, dtype=np.float64) * 10.0 ** (gain_db / 20.0)
    tmp = Path(out_path).with_suffix(".tmp")
    tmp.write_text(json.dumps(scaled.tolist()), encoding="utf-8")
    tmp.replace(out_path)
    return Path(out_path)


def guard_print(np_path, wav_path, pick, min_depth_db, action):
    """The print the denoise subtracts: the learned one when deep enough, else scaled to `min_depth_db` or None ('drop').

    `pick()` returns the `(starts, frames)` the print was learned from. A pick or a depth that
    cannot be read, or a print that cannot be scaled, keeps the learned print: the guard only
    ever falls back to what the chain did without it.
    """
    try:
        windows = pick()
        depth, loud_db, probe_db = print_depth_db(wav_path, windows)
    except (OSError, RuntimeError, ValueError) as exc:
        log_msg(f"{LOG_PREFIX} could not measure the print's depth ({exc}); the learned print is kept", level="WARNING")
        return np_path
    reading = f"the print's {len(windows[0])} window(s) sit {depth:.1f} dB under the loud frames ({probe_db:.1f} / {loud_db:.1f} dBFS)"
    if depth >= min_depth_db:
        log_msg(f"{LOG_PREFIX} {reading}, at least {min_depth_db:g} dB; the learned print is kept")
        return np_path
    if action == "drop":
        log_msg(f"{LOG_PREFIX} {reading}, under {min_depth_db:g} dB: the print is programme; denoising without a learned print")
        return None
    return _scaled(np_path, depth - min_depth_db, scaled_print_path(np_path, min_depth_db), reading)


def _scaled(np_path, gain_db, out_path, reading):
    """The scaled print, logged; the learned print (with a WARNING) when it cannot be scaled."""
    try:
        scale_print(np_path, gain_db, out_path)
    except (OSError, ValueError) as exc:
        log_msg(f"{LOG_PREFIX} {reading}, but the print could not be scaled ({exc}); the learned print is kept", level="WARNING")
        return np_path
    log_msg(f"{LOG_PREFIX} {reading}: the print is programme; scaled by {gain_db:.1f} dB into {Path(out_path).name}")
    return Path(out_path)
