"""The programme level across the neural stage, put back where it entered.

The separator peak-normalises what it is given (`WHOLE_FILE_PEAK`, 0.9, in
`modules/denoise_chunking.py`; a chunked run applies the same rule at the join) and writes
the denoised audio at that scale. On a hot or declipped source the programme therefore leaves
the neural stage several decibels under where it entered: on the diagnosis's hunt
(2026-10-11) the loud-frame level fell 3 dB or more on 11 of 235 clips and 6 dB or more on 5
(fourstarrevue: declip peak 3.08, neural peak 0.90, -10.7 dB; fun-song -10.1 dB). Every
absolute threshold after it then reads the wrong level: the polish expander's knee (the
scanner's noise floor +8 dB, in the source's terms) and the pause floor keeper's target, whose
fill landed +4.4 to +8.9 dB over the source's own pauses on those two clips.

With `apl_neural_level_restore` on, the neural output is scaled by the least-squares
projection gain of its input onto it on the input's loud frames in 100-4000 Hz
(`hiss_floor.programme_match`), so the programme leaves the stage at the level it entered and
the noise the stage removed does not count. A gain within `TOLERANCE_DB` of unity is left
alone, and so is a pair whose programmes do not follow each other. The intermediates are
32-bit float, so a peak over full scale is carried to the mux's limiter, not clipped.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from . import hiss_floor
from .config import APL_NEURAL_LEVEL_RESTORE
from .hygiene import atomic_target
from .utils import is_valid_audio, log_msg

# Under half a decibel the stage leaves the output alone. The separator rescales only an
# input that peaks over 0.9; on the 218 hunt and diagnosis clips whose input did not, the gain
# read +0.004 dB at the median (p95 +0.15 dB), and 7 passed 0.5 dB, the model's own level
# change (up to 2.7 dB on two quiet broadcast captures). On the 26 that peaked over 0.9 it
# matched the separator's peak ratio within 0.1 dB (fourstarrevue +10.68 against +10.69).
TOLERANCE_DB = 0.5
BLOCK_SAMPLES = 1 << 20
STAGE = "neural_level"
STAGE_FAILURES = (OSError, RuntimeError, ValueError, MemoryError)


def programme_gain(neural_input_wav, neural_output_wav):
    """`(gain, why)`: the amplitude gain that puts the output's programme at the input's level, or None and the reason."""
    stats = hiss_floor.frame_pair(neural_input_wav, neural_output_wav)
    if isinstance(stats, str):
        return None, stats
    gain, correlation = hiss_floor.programme_match(stats)
    if correlation < hiss_floor.MIN_CORRELATION or gain <= 0.0:
        return None, f"the output's programme does not follow the input's (r {correlation:.2f})"
    return gain, None


def _scaled(source_wav, target_wav, gain):
    """Writes `source_wav` times `gain` as 32-bit float, streamed and atomically."""
    with (
        atomic_target(target_wav) as partial,
        sf.SoundFile(str(source_wav)) as source,
        sf.SoundFile(str(partial), "w", samplerate=source.samplerate, channels=source.channels, subtype="FLOAT") as out,
    ):
        for block in source.blocks(blocksize=BLOCK_SAMPLES, dtype="float64", always_2d=True):
            out.write((block * gain).astype(np.float32))
    return Path(target_wav)


def _target_for(neural_output_wav, audio_dir, gain_db):
    output_dir = Path(audio_dir) / STAGE
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"levelled_{gain_db:+.2f}db_{Path(neural_output_wav).name}"


def restore(neural_input_wav, neural_output_wav, audio_dir):
    """Runs the stage; returns the levelled output, or the neural output with the reason logged."""
    gain, why = programme_gain(neural_input_wav, neural_output_wav)
    if gain is None:
        log_msg(f"    [Neural Level] Skipped: {why}.")
        return neural_output_wav
    gain_db = float(20.0 * np.log10(gain))
    if abs(gain_db) < TOLERANCE_DB:
        log_msg(f"    [Neural Level] Skipped: the programme left the neural stage {-gain_db:+.2f} dB from where it entered.")
        return neural_output_wav
    target = _target_for(neural_output_wav, audio_dir, gain_db)
    if is_valid_audio(target):
        log_msg(f"    [Neural Level] Reusing {target.name}.")
        return target
    log_msg(f"    [Neural Level] The programme left the neural stage {-gain_db:+.2f} dB from its input; restoring {gain_db:+.2f} dB.")
    return _scaled(neural_output_wav, target, gain)


def restore_when_needed(neural_input_wav, neural_output_wav, audio_dir, stages):
    """The neural output at its input's programme level; else the output untouched.

    `stages` holds the caller's post-neural switches: the stage runs when the caller opted
    in (`neural_level_restore`) and `apl_neural_level_restore` is on, and the neural stage
    ran (its output is not its input).
    """
    wanted = stages.get("neural_level_restore") and APL_NEURAL_LEVEL_RESTORE
    if not wanted or Path(neural_input_wav) == Path(neural_output_wav):
        return neural_output_wav
    try:
        return restore(neural_input_wav, neural_output_wav, audio_dir)
    except STAGE_FAILURES as exc:
        log_msg(f"    [Neural Level] Skipped after failure: {exc}")
        return neural_output_wav
